//! `nvdb serve` — the resident HTTP/1.1 front end of the v2 lexical engine.
//!
//! Hand-rolled on `std::net`, the same shape as `nDB`'s daemon: one thread per
//! connection behind a cap, keep-alive, read/write timeouts and a body cap. The
//! process owns the [`V2Index`] handles; clients only send documents and
//! queries and receive results — a client cannot reach the filesystem.
//!
//! # Routes
//!
//! | Method | Path | Body | 200 response |
//! |---|---|---|---|
//! | `GET` | `/health` | — | `{"ok":true,"engine":"nvdb-v2","collections":[names]}` |
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
//!
//! `commit` defaults to `true` on every write route except `attach_vectors`,
//! which commits its own link generation atomically because the caller has
//! already paid for the embeddings.
//!
//! # Failures
//!
//! Every non-2xx answer carries `{"error": "...", "code": "..."}`:
//!
//! | Status | Meaning |
//! |---|---|
//! | `400` | malformed body, invalid collection name (traversal included), invalid query, no principals, no vector lane |
//! | `403` | a directory the server refuses to take over (non-empty, not a v2 collection) |
//! | `404` | unknown route, unknown collection, no manifest |
//! | `409` | persisted collection conflicts with the requested config or build versions; a stale vector attachment |
//! | `500` | filesystem, Tantivy or state corruption |
//! | `503` | index degraded (a degraded boot-heal watermark surfaces here) |
//!
//! Collection names are restricted to `[a-zA-Z0-9._-]+` and `.`/`..` are
//! rejected, so a name can never escape `root_dir`.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex, RwLock};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::sync::VectorAttachment;
use crate::v2::{
    V2CollectionConfig, V2Document, V2Error, V2HybridQuery, V2Index, V2Manifest, V2Query,
    V2_MANIFEST_FILE,
};

/// Hard cap on a request body. Larger requests are answered with `413`.
const MAX_BODY: usize = 64 * 1024 * 1024;

/// Concurrent connection cap. Excess connections are answered with `503`.
const MAX_CONNECTIONS: usize = 64;

/// Idle read/write timeout, so a silent client cannot pin a thread forever.
const IO_TIMEOUT: Duration = Duration::from_secs(30);

/// Default page size of `pending_embeddings`.
const DEFAULT_PENDING_LIMIT: usize = 100;

// ─── Shared state ───────────────────────────────────────────────────

/// The open engine state behind the HTTP surface.
///
/// Each collection is its own lock, so a search never waits for an unrelated
/// collection's write.
pub struct V2Server {
    root_dir: PathBuf,
    collections: RwLock<HashMap<String, Arc<RwLock<V2Index>>>>,
    running: AtomicBool,
    active: Mutex<usize>,
    drained: Condvar,
}

impl V2Server {
    /// Open `root_dir` and adopt every v2 collection found inside it.
    ///
    /// A collection is a direct subdirectory holding a `v2_manifest.json`. One
    /// that cannot be opened aborts startup — a server that silently hides a
    /// broken collection serves answers that look complete and are not.
    fn open(root_dir: &Path) -> Result<Self, V2Error> {
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
            running: AtomicBool::new(true),
            active: Mutex::new(0),
            drained: Condvar::new(),
        })
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
    server: Option<Arc<V2Server>>,
    accept: Option<JoinHandle<()>>,
}

impl V2ServerHandle {
    /// Address the server actually bound (`127.0.0.1:0` returns the chosen port).
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stop accepting, wait for in-flight requests, and release every index.
    ///
    /// Releasing matters: a Tantivy writer lock is held until the last
    /// [`V2Index`] is dropped, so a restarted server on the same directory
    /// would land on a stale lock otherwise. Idempotent.
    pub fn shutdown(&mut self) {
        let Some(server) = self.server.take() else {
            return;
        };
        server.running.store(false, Ordering::SeqCst);
        // A blocking accept() only returns when a connection arrives; this one
        // is dropped unread because the running flag is already false.
        let wake = if self.addr.ip().is_unspecified() {
            SocketAddr::from(([127, 0, 0, 1], self.addr.port()))
        } else {
            self.addr
        };
        if let Ok(stream) = TcpStream::connect(wake) {
            drop(stream);
        } else {
            // Unreachable while the listener is alive (bound port, backlog far
            // above the connection cap). Reported instead of swallowed: the
            // accept thread would otherwise stay blocked and the join below
            // would hang forever.
            eprintln!("nvdb serve: shutdown could not wake the accept loop on {wake}");
            return;
        }
        if let Some(accept) = self.accept.take() {
            let _ = accept.join();
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
/// `addr` is anything [`TcpListener::bind`] accepts; `"127.0.0.1:0"` picks a
/// free port and [`V2ServerHandle::addr`] reports the one chosen.
pub fn start_in_background(
    root_dir: impl AsRef<Path>,
    addr: &str,
) -> Result<V2ServerHandle, V2Error> {
    let server = Arc::new(V2Server::open(root_dir.as_ref())?);
    let listener = bind(addr)?;
    let local_addr = listener.local_addr()?;
    eprintln!(
        "nvdb serve listening on http://{local_addr}/ ({} collections)",
        server.collections.read().len()
    );
    let accept_server = Arc::clone(&server);
    let accept = std::thread::Builder::new()
        .name("nvdb-serve-accept".to_string())
        .spawn(move || accept_loop(&listener, &accept_server))?;
    Ok(V2ServerHandle {
        addr: local_addr,
        server: Some(server),
        accept: Some(accept),
    })
}

/// Open `root_dir` and serve on `addr` until the listener fails. Blocks.
pub fn serve_blocking(root_dir: impl AsRef<Path>, addr: &str) -> Result<(), V2Error> {
    let server = Arc::new(V2Server::open(root_dir.as_ref())?);
    let listener = bind(addr)?;
    eprintln!(
        "nvdb serve listening on http://{}/ ({} collections)",
        listener.local_addr()?,
        server.collections.read().len()
    );
    accept_loop(&listener, &server);
    Ok(())
}

fn bind(addr: &str) -> Result<TcpListener, V2Error> {
    TcpListener::bind(addr).map_err(|error| {
        V2Error::Io(std::io::Error::new(
            error.kind(),
            format!("bind {addr}: {error}"),
        ))
    })
}

// ─── Accept loop ────────────────────────────────────────────────────

fn accept_loop(listener: &TcpListener, server: &Arc<V2Server>) {
    for stream in listener.incoming() {
        if !server.running.load(Ordering::SeqCst) {
            break;
        }
        let stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("nvdb serve: accept error: {error}");
                continue;
            }
        };
        {
            let mut active = server.active.lock();
            if *active >= MAX_CONNECTIONS {
                drop(active);
                let mut stream = stream;
                let mut response =
                    Response::error(503, "connection_cap", "connection cap exceeded");
                // The client is refused either way; a failed notice to one that
                // already vanished changes nothing.
                let _ = response.write(&mut stream, false);
                continue;
            }
            *active += 1;
        }
        let handler_server = Arc::clone(server);
        std::thread::spawn(move || {
            handle_connection(stream, &handler_server);
            let mut active = handler_server.active.lock();
            *active -= 1;
            if *active == 0 {
                handler_server.drained.notify_all();
            }
        });
    }
    // In-flight handlers still hold index locks; shutdown() must not return
    // before they drop them.
    let mut active = server.active.lock();
    while *active > 0 {
        server.drained.wait(&mut active);
    }
}

fn handle_connection(stream: TcpStream, server: &Arc<V2Server>) {
    if stream.set_read_timeout(Some(IO_TIMEOUT)).is_err() {
        return;
    }
    if let Err(error) = stream.set_write_timeout(Some(IO_TIMEOUT)) {
        eprintln!("nvdb serve: cannot set write timeout: {error}");
    }
    let mut stream = stream;
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(clone) => clone,
        Err(error) => {
            eprintln!("nvdb serve: cannot clone connection: {error}");
            return;
        }
    });
    loop {
        let request = match read_request(&mut reader, &mut stream) {
            Ok(Some(request)) => request,
            Ok(None) => return,
            Err(rejection) => {
                let mut response =
                    Response::error(rejection.status, rejection.code, rejection.message);
                let _ = response.write(&mut stream, false);
                return;
            }
        };
        let keep_alive = request.keep_alive;
        let mut response = route(server, &request);
        if response.write(&mut stream, keep_alive).is_err() || !keep_alive {
            return;
        }
    }
}

// ─── HTTP plumbing ──────────────────────────────────────────────────

struct Request {
    method: String,
    path: String,
    query: Option<String>,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    keep_alive: bool,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    /// One `key=value` pair of the request target's query string.
    ///
    /// Only the shapes this server issues are supported: a flat list of
    /// `key=value` pairs separated by `&`. Percent-decoding is not applied
    /// because no route accepts free text through the query string.
    fn query_value(&self, key: &str) -> Option<&str> {
        self.query
            .as_deref()?
            .split('&')
            .filter_map(|pair| pair.split_once('='))
            .find(|(name, _)| name == &key)
            .map(|(_, value)| value)
    }
}

/// Why the connection is answered and closed instead of routed.
struct Rejection {
    status: u16,
    code: &'static str,
    message: String,
}

impl Rejection {
    fn new(status: u16, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }
}

fn read_request(
    reader: &mut BufReader<TcpStream>,
    stream: &mut TcpStream,
) -> Result<Option<Request>, Rejection> {
    let mut line = String::new();
    match reader.read_line(&mut line) {
        Ok(0) => return Ok(None),
        Ok(_) => {}
        Err(_) => return Ok(None),
    }
    let line = line.trim_end();
    if line.is_empty() {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| Rejection::new(400, "malformed_request", "missing method"))?
        .to_string();
    let target = parts
        .next()
        .ok_or_else(|| Rejection::new(400, "malformed_request", "missing path"))?
        .to_string();
    let version = parts.next().unwrap_or("HTTP/1.1");
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path.to_string(), Some(query.to_string())),
        None => (target, None),
    };

    let mut headers = Vec::new();
    loop {
        let mut header_line = String::new();
        reader.read_line(&mut header_line).map_err(|error| {
            Rejection::new(400, "malformed_request", format!("header read: {error}"))
        })?;
        let header_line = header_line.trim_end();
        if header_line.is_empty() {
            break;
        }
        if let Some((key, value)) = header_line.split_once(':') {
            headers.push((key.trim().to_string(), value.trim().to_string()));
        }
    }

    let mut request = Request {
        method,
        path,
        query,
        headers,
        body: Vec::new(),
        keep_alive: !version.eq_ignore_ascii_case("HTTP/1.0"),
    };
    for (key, value) in &request.headers {
        if key.eq_ignore_ascii_case("connection") {
            if value.eq_ignore_ascii_case("close") {
                request.keep_alive = false;
            } else if value.eq_ignore_ascii_case("keep-alive") {
                request.keep_alive = true;
            }
        }
    }

    let content_length = request
        .header("content-length")
        .map(|value| value.trim().parse::<usize>())
        .transpose()
        .map_err(|error| {
            Rejection::new(
                400,
                "malformed_request",
                format!("invalid content-length: {error}"),
            )
        })?
        .unwrap_or(0);
    if content_length > MAX_BODY {
        return Err(Rejection::new(
            413,
            "payload_too_large",
            format!("body of {content_length} bytes exceeds {MAX_BODY}"),
        ));
    }

    // curl sends this for bodies above ~1KB; not answering stalls the client.
    if let Some(expect) = request.header("expect") {
        if expect.eq_ignore_ascii_case("100-continue") {
            stream
                .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
                .map_err(|error| {
                    Rejection::new(400, "malformed_request", format!("continue write: {error}"))
                })?;
        }
    }

    if content_length > 0 {
        request.body.resize(content_length, 0);
        reader.read_exact(&mut request.body).map_err(|error| {
            Rejection::new(400, "malformed_request", format!("body read: {error}"))
        })?;
    }
    Ok(Some(request))
}

struct Response {
    status: u16,
    body: Value,
}

impl Response {
    fn ok(body: Value) -> Self {
        Self { status: 200, body }
    }

    fn error(status: u16, code: &str, message: impl Into<String>) -> Self {
        Self {
            status,
            body: json!({ "error": message.into(), "code": code }),
        }
    }

    fn write(&mut self, stream: &mut TcpStream, keep_alive: bool) -> std::io::Result<()> {
        let payload = serde_json::to_vec(&self.body)?;
        let connection = if keep_alive { "keep-alive" } else { "close" };
        let head = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: {}\r\n\r\n",
            self.status,
            reason(self.status),
            payload.len(),
            connection
        );
        stream.write_all(head.as_bytes())?;
        stream.write_all(&payload)?;
        stream.flush()
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
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
        Response::error(error.status, error.code, error.message)
    }
}

type ApiResult = Result<Value, ApiError>;

// ─── Routing ────────────────────────────────────────────────────────

fn route(server: &V2Server, request: &Request) -> Response {
    match dispatch(server, request) {
        Ok(body) => Response::ok(body),
        Err(error) => error.into(),
    }
}

fn dispatch(server: &V2Server, request: &Request) -> ApiResult {
    let segments: Vec<&str> = request.path.split('/').filter(|s| !s.is_empty()).collect();
    match (request.method.as_str(), segments.as_slice()) {
        ("GET", ["health"]) => health(server),
        ("GET", ["v2", "collections"]) => list_collections(server),
        ("POST", ["v2", "collections"]) => create_collection(server, request),
        ("POST", ["v2", "collections", name, "upsert"]) => upsert(server, name, request),
        ("POST", ["v2", "collections", name, "delete"]) => delete(server, name, request),
        ("POST", ["v2", "collections", name, "acl"]) => update_acl(server, name, request),
        ("POST", ["v2", "collections", name, "commit"]) => commit(server, name),
        ("POST", ["v2", "collections", name, "search"]) => search(server, name, request),
        ("POST", ["v2", "collections", name, "hybrid_search"]) => {
            hybrid_search(server, name, request)
        }
        ("GET", ["v2", "collections", name, "pending_embeddings"]) => {
            pending_embeddings(server, name, request)
        }
        ("POST", ["v2", "collections", name, "attach_vectors"]) => {
            attach_vectors(server, name, request)
        }
        _ => Err(ApiError::new(
            404,
            "no_route",
            format!("no route: {} {}", request.method, request.path),
        )),
    }
}

fn health(server: &V2Server) -> ApiResult {
    Ok(json!({
        "ok": true,
        "engine": "nvdb-v2",
        "collections": sorted_names(server),
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
    let limit = match request.query_value("limit") {
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
