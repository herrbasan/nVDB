//! `nvdb serve` — the resident HTTP/1.1 front end of the v2 lexical engine.
//!
//! The transport is [`nhttp`] (tokio + hyper): one router, connection and body
//! caps, idle/read/write timeouts, a bounded blocking pool and a graceful
//! drain. Every engine call — Tantivy, the dense lane, the ONNX reranker — is
//! synchronous Rust and runs on [`nhttp::spawn_blocking`], so a reactor thread
//! never stalls behind an index lock. The process owns the [`V2Index`] handles;
//! clients only send documents and queries and receive results — a client
//! cannot reach the filesystem.
//!
//! `GET /metrics` is nhttp's built-in counter snapshot; everything below is an
//! nVDB route.
//!
//! # Routes
//!
//! | Method | Path | Body | 200 response |
//! |---|---|---|---|
//! | `GET` | `/health` | — | `{"ok":true,"engine":"nvdb-v2","collections":[names],"reranker_loaded":bool}` |
//! | `GET` | `/v2/collections` | — | `{"collections":[{name,doc_count,acl_generation,watermark}]}` |
//! | `POST` | `/v2/collections` | `{"name","config"?}` | `{"ok":true,"collection":{...}}` |
//! | `POST` | `/v2/collections/{name}/upsert` | `{"docs":[V2Document],"commit"?}` | `{"upserted":N}` |
//! | `POST` | `/v2/collections/{name}/delete` | `{"ids":[...],"commit"?}` | `{"deleted":N}` |
//! | `POST` | `/v2/collections/{name}/acl` | `{"id","acl":[...],"commit"?}` | `{"ok":true,"acl_generation":N}` |
//! | `POST` | `/v2/collections/{name}/commit` | — | `{"ok":true,"doc_count":N}` |
//! | `POST` | `/v2/collections/{name}/search` | `V2Query` | `{"hits":[V2SearchHit],"took_ms":f64}` |
//! | `POST` | `/v2/collections/{name}/hybrid_search` | `V2HybridQuery` | `{"hits":[V2HybridHit],"took_ms":f64}` |
//! | `GET` | `/v2/collections/{name}/pending_embeddings?limit=N` | — | `{"pending":[PendingEmbedding],"count":N}` |
//! | `POST` | `/v2/collections/{name}/attach_vectors` | `{"attachments":[VectorAttachment]}` | `{"attached":N,"vec_generation":N}` |
//! | `GET` | `/v2/collections/{name}/fingerprints` | — | `{"doc_count":N,"watermark":Option<SyncWatermark>,"entries":[FingerprintEntry]}` |
//! | `POST` | `/v2/collections/{name}/integrity_check` | `{"expected":[ExpectedFingerprint],"check_vectors"?}` | `IntegrityDiff` |
//! | `POST` | `/v2/collections/{name}/watermark` | `SyncWatermark` | `{"ok":true,"watermark":SyncWatermark}` |
//! | `POST` | `/v2/collections/{name}/level2` | `{"query_text","principals","hits",...}` | `V2Level2Result` |
//!
//! `commit` defaults to `true` on every write route except `attach_vectors`,
//! which commits its own link generation atomically because the caller has
//! already paid for the embeddings.
//!
//! # Failures
//!
//! Every non-2xx answer of an nVDB route carries `{"error": "...", "code": "..."}`.
//! Rejections made by the transport itself — bearer-token auth (`401`), the body
//! cap (`413`) and the connection cap (`503`) — carry nhttp's own envelope and
//! happen before any route runs.
//!
//! | Status | Meaning |
//! |---|---|
//! | `400` | malformed body, invalid collection name (traversal included), invalid query, no principals, no vector lane, an invalid watermark |
//! | `403` | a directory the server refuses to take over (non-empty, not a v2 collection) |
//! | `404` | unknown route, unknown collection, no manifest |
//! | `409` | persisted collection conflicts with the requested config or build versions; a stale vector attachment |
//! | `500` | filesystem, Tantivy or state corruption; a reranker that broke its contract |
//! | `503` | index degraded (a degraded boot-heal watermark), or `level2` on a server without a reranker |
//!
//! Collection names are restricted to `[a-zA-Z0-9._-]+` and `.`/`..` are
//! rejected, so a name can never escape `root_dir`.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nhttp::{Method, Request, Response, Router, ServerConfig, StatusCode};
use parking_lot::RwLock;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::rerank::Reranker;
use crate::sync::{SyncWatermark, VectorAttachment};
use crate::v2::level2::{
    DEFAULT_RELEVANCE_FLOOR, DEFAULT_RERANK_WEIGHT, DEFAULT_SHORTLIST, LEVEL2_RRF_K,
};
use crate::v2::{
    ExpectedFingerprint, V2CollectionConfig, V2Document, V2Error, V2HybridHit, V2HybridQuery,
    V2Index, V2Level2Hit, V2Level2Result, V2Manifest, V2Query, V2_MANIFEST_FILE,
};

/// Hard cap on a request body. Larger requests are answered with `413`.
const MAX_BODY: usize = 64 * 1024 * 1024;

/// Concurrent connection cap. Excess connections are answered with `503`.
///
/// The hand-rolled front end capped this at 64 because every connection owned a
/// thread. Under nhttp a connection is a task, so the cap is the transport's
/// default: high enough not to throttle concurrent clients, still bounded so a
/// flood is refused rather than swallowing the process.
const MAX_CONNECTIONS: usize = 1024;

/// Idle read/write timeout, so a silent client cannot pin a thread forever.
const IO_TIMEOUT: Duration = Duration::from_secs(30);

/// Default page size of `pending_embeddings`.
const DEFAULT_PENDING_LIMIT: usize = 100;

/// Deepest path the catch-all 404 routes claim. nVDB's own routes stop at four
/// segments, so this leaves headroom without registering an unbounded table.
const CATCH_ALL_SEGMENTS: usize = 8;

/// Subsystem tag every nVDB log line carries.
const SUBSYSTEM: &str = "serve";

// ─── Shared state ───────────────────────────────────────────────────

/// The open engine state behind the HTTP surface.
///
/// Each collection is its own lock, so a search never waits for an unrelated
/// collection's write. Everything here is synchronous: the async layer only
/// hands work over through [`nhttp::spawn_blocking`].
pub struct V2Server {
    root_dir: PathBuf,
    collections: RwLock<HashMap<String, Arc<RwLock<V2Index>>>>,
    /// Optional Level-2 reranker. `None` makes `/level2` answer `503`: a server
    /// without a model must not silently degrade into Level-1 ordering.
    reranker: Option<Arc<dyn Reranker>>,
    /// nhttp's live counters, installed at bind time. The catch-all route claims
    /// every path shape — including `/metrics` — so nVDB answers `/metrics` from
    /// this handle rather than leaving the path to nhttp's own fallback.
    metrics: OnceLock<Arc<nhttp::Metrics>>,
    /// The process logger, installed at bind time. `None` in library/test
    /// instances, so nothing writes into `./logs` unless the caller asked.
    logger: OnceLock<nlogger::Logger>,
}

impl V2Server {
    /// Open `root_dir` and adopt every v2 collection found inside it.
    ///
    /// A collection is a direct subdirectory holding a `v2_manifest.json`. One
    /// that cannot be opened aborts startup — a server that silently hides a
    /// broken collection serves answers that look complete and are not.
    fn open(root_dir: &Path, reranker: Option<Arc<dyn Reranker>>) -> Result<Self, V2Error> {
        fs::create_dir_all(root_dir)?;
        let mut collections = HashMap::new();
        for entry in fs::read_dir(root_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let dir = entry.path();
            if !dir.join(V2_MANIFEST_FILE).is_file() {
                continue;
            }
            let name = entry.file_name().into_string().map_err(|_| {
                V2Error::Corrupt(format!(
                    "collection directory {} is not valid UTF-8",
                    dir.display()
                ))
            })?;
            let index = open_collection_dir(&name, &dir)?;
            collections.insert(name, Arc::new(RwLock::new(index)));
        }
        Ok(Self {
            root_dir: root_dir.to_path_buf(),
            collections: RwLock::new(collections),
            reranker,
            metrics: OnceLock::new(),
            logger: OnceLock::new(),
        })
    }

    /// Hand the bound server's counters to the `/metrics` route.
    fn install_metrics(&self, metrics: Arc<nhttp::Metrics>) {
        self.metrics
            .set(metrics)
            .expect("the metrics handle is installed once, at bind time");
    }

    fn metrics_snapshot(&self) -> nhttp::MetricsSnapshot {
        self.metrics
            .get()
            .expect("bind installs the metrics handle before the socket serves")
            .snapshot()
    }

    /// Hand the process logger to the failure path. No logger, no log lines.
    fn install_logger(&self, logger: Option<&nlogger::Logger>) {
        if let Some(logger) = logger {
            let _ = self.logger.set(logger.clone());
        }
    }

    /// A failed request is an operational signal, not noise: only 5xx is
    /// recorded, and it carries enough detail to find the failing route.
    fn log_failure(&self, error: &ApiError) {
        let Some(logger) = self.logger.get() else {
            return;
        };
        if error.status < 500 {
            return;
        }
        logger.error_meta(
            SUBSYSTEM,
            &format!("{}: {}", error.code, error.message),
            nlogger::Meta::new()
                .field("status", i64::from(error.status))
                .field("code", error.code),
        );
    }

    /// Open `root_dir` with a Level-2 reranker attached.
    pub fn with_reranker(
        root_dir: impl AsRef<Path>,
        reranker: Arc<dyn Reranker>,
    ) -> Result<Self, V2Error> {
        Self::open(root_dir.as_ref(), Some(reranker))
    }

    fn collection(&self, name: &str) -> Result<Arc<RwLock<V2Index>>, ApiError> {
        self.collections.read().get(name).cloned().ok_or_else(|| {
            ApiError::new(
                404,
                "unknown_collection",
                format!("collection '{name}' is not open"),
            )
        })
    }
}

fn open_collection_dir(name: &str, dir: &Path) -> Result<V2Index, V2Error> {
    let manifest = V2Manifest::load(dir)?;
    V2Index::create_or_open(dir, manifest.config).map_err(|error| {
        V2Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "cannot open collection '{name}' at {}: {error}",
                dir.display()
            ),
        ))
    })
}

// ─── Entry points ───────────────────────────────────────────────────

/// A running server: its bound address plus the means to stop it.
///
/// Dropping the handle shuts the server down, so a test cannot leak a listener
/// or an open index writer by returning early.
pub struct V2ServerHandle {
    addr: SocketAddr,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl V2ServerHandle {
    /// Address the server actually bound (`127.0.0.1:0` returns the chosen port).
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stop accepting, drain in-flight requests, and release every index.
    ///
    /// Releasing matters: a Tantivy writer lock is held until the last
    /// [`V2Index`] is dropped, so a restarted server on the same directory
    /// would land on a stale lock otherwise. The join waits for the serving
    /// thread to drop its runtime *and* its state, so the lock is gone by the
    /// time this returns. Idempotent.
    pub fn shutdown(&mut self) {
        if let Some(signal) = self.shutdown.take() {
            let _ = signal.send(());
        }
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                eprintln!("nvdb serve: the serving thread panicked during shutdown");
            }
        }
    }
}

impl Drop for V2ServerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Open `root_dir`, bind `addr`, and serve on a background thread.
///
/// `addr` is a `host:port` pair; `"127.0.0.1:0"` picks a free port and
/// [`V2ServerHandle::addr`] reports the one chosen. No logger is attached, so a
/// test instance never writes into `./logs`.
pub fn start_in_background(
    root_dir: impl AsRef<Path>,
    addr: &str,
) -> Result<V2ServerHandle, V2Error> {
    start(root_dir.as_ref(), addr, None, None, None)
}

/// Like [`start_in_background`], with a Level-2 reranker attached.
pub fn start_in_background_with_reranker(
    root_dir: impl AsRef<Path>,
    addr: &str,
    reranker: Arc<dyn Reranker>,
) -> Result<V2ServerHandle, V2Error> {
    start(root_dir.as_ref(), addr, Some(reranker), None, None)
}

/// Like [`start_in_background`], guarded by a bearer token.
pub fn start_in_background_with_token(
    root_dir: impl AsRef<Path>,
    addr: &str,
    token: impl Into<String>,
) -> Result<V2ServerHandle, V2Error> {
    start(root_dir.as_ref(), addr, None, None, Some(token.into()))
}

fn start(
    root_dir: &Path,
    addr: &str,
    reranker: Option<Arc<dyn Reranker>>,
    logger: Option<nlogger::Logger>,
    token: Option<String>,
) -> Result<V2ServerHandle, V2Error> {
    let server = Arc::new(V2Server::open(root_dir, reranker)?);
    let bound = bind(Arc::clone(&server), addr, logger.clone(), token)?;
    let local_addr = bound.local_addr();

    // Built here, not on the serving thread: a runtime that cannot start is a
    // startup failure, not a server that silently answers nothing.
    let runtime = build_runtime()?;
    let (shutdown, signal) = tokio::sync::oneshot::channel();
    let thread = std::thread::Builder::new()
        .name("nvdb-serve".to_string())
        .spawn(move || {
            // A dropped sender means the handle went away without shutdown();
            // treating that as a stop is what keeps Drop honest.
            let outcome = runtime.block_on(bound.serve_until(async move {
                let _ = signal.await;
            }));
            if let Err(error) = outcome {
                eprintln!("nvdb serve: {error}");
            }
        })?;
    // Announced only once the serving thread exists: a listener that reported
    // readiness and then failed to start would be a lie the client acts on.
    announce(&server, local_addr, logger.as_ref());
    Ok(V2ServerHandle {
        addr: local_addr,
        shutdown: Some(shutdown),
        thread: Some(thread),
    })
}

/// Open `root_dir` and serve on `addr` until SIGINT. Blocks.
pub fn serve_blocking(root_dir: impl AsRef<Path>, addr: &str) -> Result<(), V2Error> {
    serve_blocking_configured(root_dir.as_ref(), addr, None, None, None)
}

/// Like [`serve_blocking`], with a Level-2 reranker attached.
pub fn serve_blocking_with_reranker(
    root_dir: impl AsRef<Path>,
    addr: &str,
    reranker: Arc<dyn Reranker>,
) -> Result<(), V2Error> {
    serve_blocking_configured(root_dir.as_ref(), addr, Some(reranker), None, None)
}

/// The full CLI path: open `root_dir`, attach a reranker and a logger when the
/// caller has one, and serve until SIGINT.
///
/// `token` switches on bearer auth: every request except `/health` and
/// `/metrics` then needs `Authorization: Bearer <token>`. `None` keeps the
/// historical open server.
pub fn serve_blocking_configured(
    root_dir: impl AsRef<Path>,
    addr: &str,
    reranker: Option<Arc<dyn Reranker>>,
    logger: Option<nlogger::Logger>,
    token: Option<String>,
) -> Result<(), V2Error> {
    let server = Arc::new(V2Server::open(root_dir.as_ref(), reranker)?);
    let bound = bind(Arc::clone(&server), addr, logger.clone(), token)?;
    announce(&server, bound.local_addr(), logger.as_ref());
    let runtime = build_runtime()?;
    let outcome = runtime.block_on(bound.serve());
    if let Some(logger) = logger.as_ref() {
        logger.info(SUBSYSTEM, "nvdb serve stopped");
    }
    outcome.map_err(|error| serve_error(addr, error))
}

/// Build the `nvdb` logger writing JSONL into `logs_dir`.
///
/// `LOG_LEVEL` and `LOG_RETENTION_DAYS` are honoured through
/// [`nlogger::Config::with_env_overrides`].
pub fn build_logger(logs_dir: &Path) -> Result<nlogger::Logger, V2Error> {
    let config = nlogger::Config::new("nvdb", logs_dir)
        .with_env_overrides()
        .map_err(|error| V2Error::InvalidConfig(format!("logger: {error}")))?;
    nlogger::Logger::new(config).map_err(|error| V2Error::InvalidConfig(format!("logger: {error}")))
}

fn build_runtime() -> Result<tokio::runtime::Runtime, V2Error> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("nvdb-serve-worker")
        .build()
        .map_err(|error| {
            V2Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("build the serve runtime: {error}"),
            ))
        })
}

fn bind(
    server: Arc<V2Server>,
    addr: &str,
    logger: Option<nlogger::Logger>,
    token: Option<String>,
) -> Result<nhttp::BoundServer<Arc<V2Server>>, V2Error> {
    let (bind, port) = split_addr(addr)?;
    // Monitoring stays reachable when auth is on: a probe that must hold the
    // secret cannot tell "down" from "unauthorized".
    let auth_exempt_paths = match token {
        Some(_) => vec!["/health".to_string(), "/metrics".to_string()],
        None => Vec::new(),
    };
    let failure_logger = logger.clone();
    let config = ServerConfig {
        bind,
        port,
        token,
        auth_exempt_paths,
        max_connections: MAX_CONNECTIONS,
        max_body_bytes: MAX_BODY,
        read_timeout: IO_TIMEOUT,
        idle_timeout: IO_TIMEOUT,
        write_timeout: IO_TIMEOUT,
        // Long enough for a real upsert batch to finish, bounded so a wedged
        // handler cannot keep the process alive forever.
        shutdown_drain_timeout: Duration::from_secs(30),
        logger,
        // `/health` is nVDB's own route: it reports collections and the
        // reranker, which nhttp's built-in liveness line cannot. `/metrics` is
        // answered by the catch-all from this server's own counter handle, so
        // nhttp's fallback would never be reached.
        built_in_health: false,
        built_in_metrics: false,
    };
    nhttp::Server::bind(config, router(), Arc::clone(&server))
        .map(|bound| {
            server.install_logger(failure_logger.as_ref());
            server.install_metrics(bound.metrics());
            bound
        })
        .map_err(|error| serve_error(addr, error))
}

/// `host:port` → the pair nhttp binds. IPv6 literals keep their brackets off.
fn split_addr(addr: &str) -> Result<(String, u16), V2Error> {
    let invalid = || {
        V2Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("'{addr}' is not a host:port address"),
        ))
    };
    let (host, port) = if let Some(rest) = addr.strip_prefix('[') {
        rest.split_once("]:")
            .map(|(host, port)| (host.to_string(), port))
            .ok_or_else(invalid)?
    } else {
        addr.rsplit_once(':')
            .map(|(host, port)| (host.to_string(), port))
            .ok_or_else(invalid)?
    };
    let port = port
        .parse::<u16>()
        .map_err(|error| V2Error::InvalidConfig(format!("'{addr}': invalid port: {error}")))?;
    Ok((host, port))
}

fn serve_error(addr: &str, error: nhttp::ServerError) -> V2Error {
    V2Error::Io(std::io::Error::new(
        std::io::ErrorKind::Other,
        format!("serve {addr}: {error}"),
    ))
}

fn announce(server: &V2Server, addr: SocketAddr, logger: Option<&nlogger::Logger>) {
    let message = format!(
        "nvdb serve listening on http://{addr}/ ({} collections, reranker: {})",
        server.collections.read().len(),
        if server.reranker.is_some() {
            "loaded"
        } else {
            "none"
        }
    );
    match logger {
        Some(logger) => logger.info(SUBSYSTEM, &message),
        None => eprintln!("{message}"),
    }
}

// ─── Router ─────────────────────────────────────────────────────────

/// The route table. Patterns are nhttp's: `:name` is one path segment.
///
/// Every handler is `async` only at the edge: it owns the request, hands the
/// synchronous engine call to [`offload`], and encodes the answer.
fn router() -> Router<Arc<V2Server>> {
    let router = Router::new()
        .get("/health", handle_health)
        .get("/v2/collections", handle_list_collections)
        .post("/v2/collections", handle_create_collection)
        .post("/v2/collections/:name/upsert", handle_upsert)
        .post("/v2/collections/:name/delete", handle_delete)
        .post("/v2/collections/:name/acl", handle_update_acl)
        .post("/v2/collections/:name/commit", handle_commit)
        .post("/v2/collections/:name/search", handle_search)
        .post("/v2/collections/:name/hybrid_search", handle_hybrid_search)
        .get(
            "/v2/collections/:name/pending_embeddings",
            handle_pending_embeddings,
        )
        .post(
            "/v2/collections/:name/attach_vectors",
            handle_attach_vectors,
        )
        .get("/v2/collections/:name/fingerprints", handle_fingerprints)
        .post(
            "/v2/collections/:name/integrity_check",
            handle_integrity_check,
        )
        .post("/v2/collections/:name/watermark", handle_set_watermark)
        .post("/v2/collections/:name/level2", handle_level2);
    claim_unmatched(router)
}

/// nhttp answers an unmatched path with its own envelope; nVDB's contract is
/// `404 {"error": ..., "code": "no_route"}`. Claiming every method/length shape
/// keeps that contract. A static segment always outranks a parameter at the same
/// position, so no real route can be stolen by a catch-all.
fn claim_unmatched(mut router: Router<Arc<V2Server>>) -> Router<Arc<V2Server>> {
    for method in [
        Method::GET,
        Method::POST,
        Method::PUT,
        Method::PATCH,
        Method::DELETE,
        Method::HEAD,
        Method::OPTIONS,
    ] {
        for segments in 0..=CATCH_ALL_SEGMENTS {
            let pattern = catch_all_pattern(segments);
            router = router.route(method.clone(), &pattern, handle_no_route);
        }
    }
    router
}

fn catch_all_pattern(segments: usize) -> String {
    let mut pattern = String::new();
    for _ in 0..segments {
        pattern.push_str("/:rest");
    }
    if pattern.is_empty() {
        pattern.push('/');
    }
    pattern
}

// ─── Edge handlers ──────────────────────────────────────────────────

async fn handle_health(server: Arc<V2Server>, _request: Request) -> Response {
    finish(offload(Arc::clone(&server), move || health(&server)).await)
}

async fn handle_list_collections(server: Arc<V2Server>, _request: Request) -> Response {
    finish(offload(Arc::clone(&server), move || list_collections(&server)).await)
}

async fn handle_create_collection(server: Arc<V2Server>, request: Request) -> Response {
    finish(offload(Arc::clone(&server), move || create_collection(&server, &request)).await)
}

async fn handle_upsert(server: Arc<V2Server>, request: Request) -> Response {
    let name = name_param(&request);
    finish(offload(Arc::clone(&server), move || upsert(&server, &name, &request)).await)
}

async fn handle_delete(server: Arc<V2Server>, request: Request) -> Response {
    let name = name_param(&request);
    finish(offload(Arc::clone(&server), move || delete(&server, &name, &request)).await)
}

async fn handle_update_acl(server: Arc<V2Server>, request: Request) -> Response {
    let name = name_param(&request);
    finish(offload(Arc::clone(&server), move || update_acl(&server, &name, &request)).await)
}

async fn handle_commit(server: Arc<V2Server>, request: Request) -> Response {
    let name = name_param(&request);
    finish(offload(Arc::clone(&server), move || commit(&server, &name)).await)
}

async fn handle_search(server: Arc<V2Server>, request: Request) -> Response {
    let name = name_param(&request);
    finish(offload(Arc::clone(&server), move || search(&server, &name, &request)).await)
}

async fn handle_hybrid_search(server: Arc<V2Server>, request: Request) -> Response {
    let name = name_param(&request);
    finish(offload(Arc::clone(&server), move || hybrid_search(&server, &name, &request)).await)
}

async fn handle_pending_embeddings(server: Arc<V2Server>, request: Request) -> Response {
    let name = name_param(&request);
    finish(offload(Arc::clone(&server), move || pending_embeddings(&server, &name, &request)).await)
}

async fn handle_attach_vectors(server: Arc<V2Server>, request: Request) -> Response {
    let name = name_param(&request);
    finish(offload(Arc::clone(&server), move || attach_vectors(&server, &name, &request)).await)
}

async fn handle_fingerprints(server: Arc<V2Server>, request: Request) -> Response {
    let name = name_param(&request);
    finish(offload(Arc::clone(&server), move || fingerprints(&server, &name)).await)
}

async fn handle_integrity_check(server: Arc<V2Server>, request: Request) -> Response {
    let name = name_param(&request);
    finish(offload(Arc::clone(&server), move || integrity_check(&server, &name, &request)).await)
}

async fn handle_set_watermark(server: Arc<V2Server>, request: Request) -> Response {
    let name = name_param(&request);
    finish(offload(Arc::clone(&server), move || set_watermark(&server, &name, &request)).await)
}

async fn handle_level2(server: Arc<V2Server>, request: Request) -> Response {
    let name = name_param(&request);
    finish(offload(Arc::clone(&server), move || level2(&server, &name, &request)).await)
}

async fn handle_no_route(server: Arc<V2Server>, request: Request) -> Response {
    // `/metrics` is the one path the catch-all answers instead of refusing: the
    // body is nhttp's own counter snapshot, byte for byte.
    if request.method == Method::GET && request.path == "/metrics" {
        return Response::json_raw(StatusCode::OK, server.metrics_snapshot().to_json());
    }
    ApiError::new(
        404,
        "no_route",
        format!("no route: {} {}", request.method, request.path),
    )
    .into()
}

// ─── Edge plumbing ──────────────────────────────────────────────────

/// Run a synchronous engine call on nhttp's blocking pool. The reactor thread
/// keeps serving other connections while Tantivy, the dense lane or ONNX works.
/// A 5xx outcome is recorded on the process logger before it becomes a response.
async fn offload<F>(server: Arc<V2Server>, work: F) -> ApiResult
where
    F: FnOnce() -> ApiResult + Send + 'static,
{
    let outcome = match nhttp::spawn_blocking(work).await {
        Ok(outcome) => outcome,
        Err(error) => Err(ApiError::new(
            500,
            "engine_failure",
            format!("blocking worker: {error}"),
        )),
    };
    if let Err(error) = &outcome {
        server.log_failure(error);
    }
    outcome
}

fn finish(outcome: ApiResult) -> Response {
    match outcome {
        Ok(body) => Response::json_raw(StatusCode::OK, encode(&body)),
        Err(error) => error.into(),
    }
}

fn encode(body: &Value) -> Vec<u8> {
    serde_json::to_vec(body).expect("a serde_json::Value always serializes")
}

/// The one path parameter every collection route carries.
fn name_param(request: &Request) -> String {
    request
        .param("name")
        .expect("the router only reaches a :name route with a name")
        .to_string()
}

/// One `key=value` pair of the request target's query string.
///
/// Only the shapes this server issues are supported: a flat list of
/// `key=value` pairs separated by `&`. Percent-decoding is not applied because
/// no route accepts free text through the query string, and a bare `?limit`
/// (no `=`) still means "absent", exactly as before the nhttp move.
fn query_value<'a>(query: Option<&'a str>, key: &str) -> Option<&'a str> {
    query?
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(name, _)| *name == key)
        .map(|(_, value)| value)
}

// ─── Errors ─────────────────────────────────────────────────────────

/// A failed request: the status and code are decided where the failure is
/// known, never inferred later from a message string.
struct ApiError {
    status: u16,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: u16, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(400, code, message)
    }

    fn from_engine(error: V2Error) -> Self {
        let (status, code) = match &error {
            V2Error::MissingPrincipals => (400, "missing_principals"),
            V2Error::InvalidQuery(_) => (400, "invalid_query"),
            V2Error::InvalidConfig(_) => (400, "invalid_config"),
            V2Error::WrongDimension { .. } => (400, "wrong_dimension"),
            V2Error::VectorLaneNotConfigured => (400, "vector_lane_not_configured"),
            V2Error::PromotedFieldType { .. } => (400, "promoted_field_type"),
            V2Error::NotOpen { .. } => (404, "not_open"),
            V2Error::SchemaMismatch(_) => (409, "schema_mismatch"),
            V2Error::VersionMismatch { .. } => (409, "version_mismatch"),
            // A stale completion is a race the client must resolve by re-reading
            // the pending list, not a malformed request.
            V2Error::StaleVectorCas { .. } => (409, "stale_vector_cas"),
            // A degraded watermark is the boot-heal-incomplete signal: the
            // index is readable in principle but refuses to answer until healed.
            V2Error::IndexDegraded { .. } => (503, "index_degraded"),
            V2Error::Corrupt(_) => (500, "corrupt_state"),
            V2Error::Rerank(_)
            | V2Error::Sync(_)
            | V2Error::Tantivy(_)
            | V2Error::Io(_)
            | V2Error::Json(_) => (500, "engine_failure"),
        };
        Self::new(status, code, error.to_string())
    }
}

impl From<ApiError> for Response {
    fn from(error: ApiError) -> Self {
        let status = StatusCode::from_u16(error.status)
            .expect("every ApiError status is a valid HTTP status code");
        Response::json_raw(
            status,
            encode(&json!({ "error": error.message, "code": error.code })),
        )
    }
}

type ApiResult = Result<Value, ApiError>;

// ─── Routes ─────────────────────────────────────────────────────────

fn health(server: &V2Server) -> ApiResult {
    Ok(json!({
        "ok": true,
        "engine": "nvdb-v2",
        "collections": sorted_names(server),
        "reranker_loaded": server.reranker.is_some(),
    }))
}

fn sorted_names(server: &V2Server) -> Vec<String> {
    let collections = server.collections.read();
    let mut names: Vec<String> = collections.keys().cloned().collect();
    names.sort();
    names
}

fn describe(name: &str, index: &V2Index) -> Value {
    json!({
        "name": name,
        "doc_count": index.doc_count(),
        "acl_generation": index.manifest().acl_generation,
        "watermark": index.manifest().watermark,
    })
}

fn list_collections(server: &V2Server) -> ApiResult {
    let collections = server.collections.read();
    let mut names: Vec<&String> = collections.keys().collect();
    names.sort();
    let described: Vec<Value> = names
        .into_iter()
        .map(|name| describe(name, &collections[name].read()))
        .collect();
    Ok(json!({ "collections": described }))
}

/// Collection names are one path segment: `[a-zA-Z0-9._-]+`, never `.`/`..`.
fn validate_collection_name(name: &str) -> Result<(), ApiError> {
    if name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_collection_name",
            "collection name must not be empty",
        ));
    }
    if name == "." || name == ".." {
        return Err(ApiError::bad_request(
            "invalid_collection_name",
            format!("'{name}' is not a collection name"),
        ));
    }
    if name.contains('/') || name.contains('\\') {
        return Err(ApiError::bad_request(
            "invalid_collection_name",
            format!("collection name '{name}' must not contain a path separator"),
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return Err(ApiError::bad_request(
            "invalid_collection_name",
            format!("collection name '{name}' may only contain [a-zA-Z0-9._-]"),
        ));
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateCollectionBody {
    name: String,
    #[serde(default)]
    config: Option<V2CollectionConfig>,
}

fn create_collection(server: &V2Server, request: &Request) -> ApiResult {
    let body: CreateCollectionBody = parse_body(request)?;
    validate_collection_name(&body.name)?;

    let dir = server.root_dir.join(&body.name);
    if dir.exists() && !dir.join(V2_MANIFEST_FILE).is_file() {
        let occupied = fs::read_dir(&dir)
            .map_err(|error| {
                ApiError::new(
                    500,
                    "engine_failure",
                    format!("inspect {}: {error}", dir.display()),
                )
            })?
            .next()
            .is_some();
        if occupied {
            return Err(ApiError::new(
                403,
                "foreign_directory",
                format!(
                    "{} exists and is not a v2 collection; refusing to take it over",
                    dir.display()
                ),
            ));
        }
    }

    // Held across the open so two concurrent creates cannot both build a writer
    // for the same directory.
    let mut collections = server.collections.write();
    if let Some(existing) = collections.get(&body.name) {
        let described = describe(&body.name, &existing.read());
        return Ok(json!({ "ok": true, "collection": described }));
    }
    let index = V2Index::create_or_open(&dir, body.config.unwrap_or_default())
        .map_err(ApiError::from_engine)?;
    let described = describe(&body.name, &index);
    collections.insert(body.name.clone(), Arc::new(RwLock::new(index)));
    Ok(json!({ "ok": true, "collection": described }))
}

fn parse_body<T: for<'de> Deserialize<'de>>(request: &Request) -> Result<T, ApiError> {
    if request.body.is_empty() {
        return Err(ApiError::bad_request(
            "empty_body",
            "this route requires a JSON body",
        ));
    }
    serde_json::from_slice(&request.body)
        .map_err(|error| ApiError::bad_request("invalid_body", error.to_string()))
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpsertBody {
    docs: Vec<V2Document>,
    #[serde(default = "default_true")]
    commit: bool,
}

fn upsert(server: &V2Server, name: &str, request: &Request) -> ApiResult {
    let body: UpsertBody = parse_body(request)?;
    let collection = server.collection(name)?;
    let mut index = collection.write();
    let upserted = body.docs.len();
    index
        .upsert_batch(body.docs)
        .map_err(ApiError::from_engine)?;
    if body.commit {
        index.commit().map_err(ApiError::from_engine)?;
    }
    Ok(json!({ "upserted": upserted }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeleteBody {
    ids: Vec<String>,
    #[serde(default = "default_true")]
    commit: bool,
}

fn delete(server: &V2Server, name: &str, request: &Request) -> ApiResult {
    let body: DeleteBody = parse_body(request)?;
    let collection = server.collection(name)?;
    let mut index = collection.write();
    let mut deleted = 0usize;
    for id in &body.ids {
        if index.fingerprint(id).is_some() {
            deleted += 1;
        }
        index.delete(id).map_err(ApiError::from_engine)?;
    }
    if body.commit {
        index.commit().map_err(ApiError::from_engine)?;
    }
    Ok(json!({ "deleted": deleted }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AclBody {
    id: String,
    acl: Vec<String>,
    #[serde(default = "default_true")]
    commit: bool,
}

fn update_acl(server: &V2Server, name: &str, request: &Request) -> ApiResult {
    let body: AclBody = parse_body(request)?;
    let collection = server.collection(name)?;
    let mut index = collection.write();
    // The fast path: overlay + manifest only, no Tantivy segment is touched.
    index
        .update_acl_only(&body.id, body.acl)
        .map_err(ApiError::from_engine)?;
    if body.commit {
        index.commit().map_err(ApiError::from_engine)?;
    }
    Ok(json!({
        "ok": true,
        "acl_generation": index.acl_overlay().generation(),
    }))
}

fn commit(server: &V2Server, name: &str) -> ApiResult {
    let collection = server.collection(name)?;
    let mut index = collection.write();
    index.commit().map_err(ApiError::from_engine)?;
    Ok(json!({ "ok": true, "doc_count": index.doc_count() }))
}

fn search(server: &V2Server, name: &str, request: &Request) -> ApiResult {
    let query: V2Query = parse_body(request)?;
    let collection = server.collection(name)?;
    let index = collection.read();
    let started = Instant::now();
    let hits = index.search(&query).map_err(ApiError::from_engine)?;
    let took_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(json!({ "hits": hits, "took_ms": took_ms }))
}

fn hybrid_search(server: &V2Server, name: &str, request: &Request) -> ApiResult {
    let query: V2HybridQuery = parse_body(request)?;
    let collection = server.collection(name)?;
    let index = collection.read();
    let started = Instant::now();
    let hits = index.hybrid_search(&query).map_err(ApiError::from_engine)?;
    let took_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(json!({ "hits": hits, "took_ms": took_ms }))
}

fn pending_embeddings(server: &V2Server, name: &str, request: &Request) -> ApiResult {
    let limit = match query_value(request.query.as_deref(), "limit") {
        Some(raw) => raw.parse::<usize>().map_err(|error| {
            ApiError::bad_request(
                "invalid_limit",
                format!("limit must be a positive integer: {error}"),
            )
        })?,
        None => DEFAULT_PENDING_LIMIT,
    };
    if limit == 0 {
        return Err(ApiError::bad_request(
            "invalid_limit",
            "limit must be greater than zero",
        ));
    }
    let collection = server.collection(name)?;
    let index = collection.read();
    let pending = index.next_pending_embeddings(limit);
    Ok(json!({ "count": pending.len(), "pending": pending }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AttachVectorsBody {
    attachments: Vec<VectorAttachment>,
}

fn attach_vectors(server: &V2Server, name: &str, request: &Request) -> ApiResult {
    let body: AttachVectorsBody = parse_body(request)?;
    let collection = server.collection(name)?;
    let mut index = collection.write();
    // Commits its own link generation: a `commit` flag would suggest the
    // embeddings could be lost until someone calls it.
    let attached = index
        .attach_vectors(body.attachments)
        .map_err(ApiError::from_engine)?;
    Ok(json!({
        "attached": attached,
        "vec_generation": index.vec_generation(),
    }))
}

fn fingerprints(server: &V2Server, name: &str) -> ApiResult {
    let collection = server.collection(name)?;
    let index = collection.read();
    let entries = index.fingerprint_entries();
    Ok(json!({
        "doc_count": index.doc_count(),
        "watermark": index.manifest().watermark,
        "entries": entries,
    }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegrityCheckBody {
    expected: Vec<ExpectedFingerprint>,
    #[serde(default)]
    check_vectors: bool,
}

fn integrity_check(server: &V2Server, name: &str, request: &Request) -> ApiResult {
    let body: IntegrityCheckBody = parse_body(request)?;
    let collection = server.collection(name)?;
    let index = collection.read();
    let diff = index.integrity_check_entries(&body.expected, body.check_vectors);
    Ok(json!(diff))
}

/// Wire shape of [`SyncWatermark`]: the degradation pair is optional on the
/// wire so an ordinary (healthy) watermark is three short fields.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WatermarkBody {
    source_epoch_uuid: String,
    synced_seq: u64,
    #[serde(default)]
    degraded: bool,
    #[serde(default)]
    degraded_reason: Option<String>,
}

fn set_watermark(server: &V2Server, name: &str, request: &Request) -> ApiResult {
    let body: WatermarkBody = parse_body(request)?;
    let watermark = SyncWatermark {
        source_epoch_uuid: body.source_epoch_uuid,
        synced_seq: body.synced_seq,
        degraded: body.degraded,
        degraded_reason: body.degraded_reason,
    };
    // The invariants are the caller's to satisfy, not to discover later at open.
    watermark
        .validate()
        .map_err(|error| ApiError::bad_request("invalid_watermark", error.to_string()))?;
    let collection = server.collection(name)?;
    let mut index = collection.write();
    index
        .set_watermark(watermark.clone())
        .map_err(ApiError::from_engine)?;
    Ok(json!({ "ok": true, "watermark": watermark }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Level2Body {
    query_text: String,
    principals: Vec<String>,
    hits: Vec<V2HybridHit>,
    /// Authoritative text per document id, when the caller already has it.
    #[serde(default)]
    texts: Option<HashMap<String, String>>,
    #[serde(default)]
    shortlist: Option<usize>,
    #[serde(default)]
    rerank_weight: Option<f32>,
    #[serde(default)]
    relevance_floor: Option<f32>,
}

fn level2(server: &V2Server, name: &str, request: &Request) -> ApiResult {
    let body: Level2Body = parse_body(request)?;
    let collection = server.collection(name)?;
    let index = collection.read();

    // Fail closed, in this order: principals, then degradation, then a server
    // that cannot rerank at all.
    index
        .ensure_readable(&body.principals)
        .map_err(ApiError::from_engine)?;
    let Some(reranker) = server.reranker.as_ref() else {
        return Err(ApiError::new(
            503,
            "reranker_not_configured",
            "this server runs without a reranker; start it with --reranker-model",
        ));
    };

    let shortlist = body.shortlist.unwrap_or(DEFAULT_SHORTLIST);
    if shortlist == 0 {
        return Err(ApiError::bad_request(
            "invalid_shortlist",
            "shortlist must be greater than zero",
        ));
    }
    let rerank_weight = body.rerank_weight.unwrap_or(DEFAULT_RERANK_WEIGHT);
    if !rerank_weight.is_finite() || rerank_weight < 0.0 {
        return Err(ApiError::bad_request(
            "invalid_rerank_weight",
            "rerank_weight must be finite and non-negative",
        ));
    }
    let floor = body.relevance_floor.unwrap_or(DEFAULT_RELEVANCE_FLOOR);
    if !floor.is_finite() {
        return Err(ApiError::bad_request(
            "invalid_relevance_floor",
            "relevance_floor must be finite",
        ));
    }

    // Level 1 is evidence, not permission: the ACL overlay decides who may see
    // a candidate, and a revoked document never reaches the reranker.
    let principals: HashSet<&str> = body.principals.iter().map(String::as_str).collect();
    let survivors: Vec<V2HybridHit> = body
        .hits
        .into_iter()
        .filter(|hit| index.acl_overlay().is_visible(&hit.hit.id, &principals))
        .take(shortlist)
        .collect();

    let candidate_texts: Vec<String> = survivors
        .iter()
        .map(|hit| {
            body.texts
                .as_ref()
                .and_then(|texts| texts.get(&hit.hit.id))
                .cloned()
                .or_else(|| hit.hit.excerpt.clone())
                .unwrap_or_else(|| {
                    format!("{} {}", hit.hit.title, hit.hit.path)
                        .trim()
                        .to_string()
                })
        })
        .collect();

    let scores: Vec<f32> = if survivors.is_empty() {
        Vec::new()
    } else {
        let documents: Vec<&str> = candidate_texts.iter().map(String::as_str).collect();
        let scores = reranker
            .score(&body.query_text, &documents)
            .map_err(|error| ApiError::new(500, "rerank_failed", error.to_string()))?;
        if scores.len() != documents.len() {
            return Err(ApiError::new(
                500,
                "rerank_failed",
                format!(
                    "reranker returned {} scores for {} documents",
                    scores.len(),
                    documents.len()
                ),
            ));
        }
        if scores.iter().any(|score| !score.is_finite()) {
            return Err(ApiError::new(
                500,
                "rerank_failed",
                "reranker returned a non-finite score",
            ));
        }
        scores
    };

    // Weighted RRF over the Level-1 order (weight 1.0) and the reranker order.
    let mut order: Vec<f32> = (0..survivors.len())
        .map(|index| 1.0 / (LEVEL2_RRF_K + (index + 1) as f32))
        .collect();
    if !scores.is_empty() {
        let mut rerank_order: Vec<usize> = (0..survivors.len()).collect();
        rerank_order.sort_by(|left, right| {
            scores[*right]
                .partial_cmp(&scores[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        for (rank, index) in rerank_order.into_iter().enumerate() {
            order[index] += rerank_weight / (LEVEL2_RRF_K + (rank + 1) as f32);
        }
    }
    let mut positions: Vec<usize> = (0..survivors.len()).collect();
    positions.sort_by(|left, right| {
        order[*right]
            .partial_cmp(&order[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });

    let best_rerank_score = scores.iter().copied().reduce(f32::max);
    let rejected = best_rerank_score.map(|best| best < floor).unwrap_or(false);

    let mut hits: Vec<V2Level2Hit> = Vec::with_capacity(survivors.len());
    for index in positions {
        hits.push(V2Level2Hit {
            hybrid: survivors[index].clone(),
            rerank_score: scores.get(index).copied(),
            hydrated_text: Some(candidate_texts[index].clone()),
        });
    }
    Ok(json!(V2Level2Result {
        rejected,
        best_rerank_score,
        hits,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_names_reject_traversal_and_separators() {
        for name in ["../evil", "a/b", "a\\b", "..", ".", "", "a b", "ü"] {
            assert!(
                validate_collection_name(name).is_err(),
                "'{name}' must be rejected"
            );
        }
        for name in ["docs", "docs-v2", "a.b_c-1", "0"] {
            assert!(
                validate_collection_name(name).is_ok(),
                "'{name}' must be accepted"
            );
        }
    }

    #[test]
    fn engine_errors_map_to_explicit_statuses() {
        let missing = ApiError::from_engine(V2Error::MissingPrincipals);
        assert_eq!(missing.status, 400);
        assert_eq!(missing.code, "missing_principals");

        let degraded = ApiError::from_engine(V2Error::IndexDegraded {
            reason: "apply failed".to_string(),
        });
        assert_eq!(degraded.status, 503);
        assert_eq!(degraded.code, "index_degraded");

        let conflict = ApiError::from_engine(V2Error::SchemaMismatch("promoted".to_string()));
        assert_eq!(conflict.status, 409);
    }
}
