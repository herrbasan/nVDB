//! Phase 2: `nvdb serve` integration tests over real TCP.
//!
//! Every assertion goes through an actual HTTP/1.1 round trip against a server
//! bound to `127.0.0.1:0`, so the route table, the JSON contract and the
//! fail-closed ACL path are exercised the way a client sees them.
//!
//! Coverage: `/health`, collection create (including a rejected path
//! traversal and a refused foreign directory), upsert with and without
//! `commit`, the missing-principals rejection, a successful search with
//! `took_ms`, the ACL-only fast path hiding a document immediately, delete, and
//! a restart that rediscovers the collection from disk.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nvdb::v2::server::{start_in_background, start_in_background_with_reranker};
use serde_json::{json, Value};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Raw HTTP client
// ---------------------------------------------------------------------------

fn call(addr: SocketAddr, method: &str, path: &str, body: Option<&Value>) -> (u16, Value) {
    let payload = match body {
        Some(value) => serde_json::to_vec(value).expect("serialize request body"),
        None => Vec::new(),
    };
    let mut stream = TcpStream::connect(addr).expect("connect to serve");
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .expect("read timeout");
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream.write_all(head.as_bytes()).expect("write request head");
    stream.write_all(&payload).expect("write request body");
    stream.flush().expect("flush request");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");
    let text = String::from_utf8(raw).expect("response is UTF-8");
    let (head, body) = text
        .split_once("\r\n\r\n")
        .expect("response has a header separator");
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .expect("status line");
    let value = if body.trim().is_empty() {
        Value::Null
    } else {
        serde_json::from_str(body).expect("response body is JSON")
    };
    (status, value)
}

fn health(addr: SocketAddr) -> (u16, Value) {
    call(addr, "GET", "/health", None)
}

fn create_collection(addr: SocketAddr, name: &str) -> (u16, Value) {
    call(addr, "POST", "/v2/collections", Some(&json!({ "name": name })))
}

fn doc(id: &str, title: &str, body: &str, acl: &[&str]) -> Value {
    json!({
        "id": id,
        "kind": "doc",
        "title": title,
        "body": body,
        "acl": acl,
    })
}

fn create_vector_collection(addr: SocketAddr, name: &str) -> (u16, Value) {
    call(
        addr,
        "POST",
        "/v2/collections",
        Some(&json!({
            "name": name,
            "config": { "vector_lane": { "dim": 4, "model_version": "test-v1" } },
        })),
    )
}

fn upsert_docs(addr: SocketAddr, name: &str, docs: Vec<Value>) -> (u16, Value) {
    call(
        addr,
        "POST",
        &format!("/v2/collections/{name}/upsert"),
        Some(&json!({ "docs": docs })),
    )
}

fn fingerprints(addr: SocketAddr, name: &str) -> (u16, Value) {
    call(
        addr,
        "GET",
        &format!("/v2/collections/{name}/fingerprints"),
        None,
    )
}

/// One Level-1 `V2HybridHit` as a client would send it back for refinement.
fn level1_hit(id: &str, title: &str, path: &str, rank: usize) -> Value {
    let score = 1.0 / rank as f64;
    json!({
        "hit": {
            "id": id,
            "parent_id": null,
            "chunk_index": null,
            "kind": "doc",
            "title": title,
            "path": path,
            "excerpt": null,
            "score": score,
            "bm25_score": score,
            "explain": null,
        },
        "lexical_score": score,
        "lexical_rank": rank,
        "vector_score": null,
        "vector_rank": null,
        "rrf_score": score,
    })
}

/// A deterministic reranker: documents containing its keyword score `0.9`,
/// everything else `0.01`. It records the exact texts it was shown, so a test
/// can prove a revoked document never reached the model.
struct KeywordReranker {
    keyword: String,
    seen: Mutex<Vec<String>>,
}

impl KeywordReranker {
    fn new(keyword: &str) -> Arc<Self> {
        Arc::new(Self {
            keyword: keyword.to_string(),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn seen(&self) -> Vec<String> {
        self.seen.lock().expect("reranker input lock").clone()
    }
}

impl nvdb::Reranker for KeywordReranker {
    fn score(&self, query: &str, documents: &[&str]) -> nvdb::Result<Vec<f32>> {
        let mut seen = self.seen.lock().expect("reranker input lock");
        seen.clear();
        // Query-conditioned on purpose: a nonsense query must score nothing high,
        // which is what makes the relevance floor observable.
        let query_is_relevant = query.to_lowercase().contains(&self.keyword);
        let mut scores = Vec::with_capacity(documents.len());
        for document in documents {
            seen.push((*document).to_string());
            scores.push(
                if query_is_relevant && document.to_lowercase().contains(&self.keyword) {
                    0.9
                } else {
                    0.01
                },
            );
        }
        Ok(scores)
    }
}

// ---------------------------------------------------------------------------
// Health and collection creation
// ---------------------------------------------------------------------------

#[test]
fn health_reports_engine_and_rejects_path_traversal() {
    let temp = TempDir::new().expect("temp dir");
    let mut server = start_in_background(temp.path(), "127.0.0.1:0").expect("start server");
    let addr = server.addr();

    let (status, body) = health(addr);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["ok"], json!(true));
    assert_eq!(body["engine"], json!("nvdb-v2"));
    assert_eq!(body["collections"], json!([]));

    let (status, body) = create_collection(addr, "docs");
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["ok"], json!(true));
    assert_eq!(body["collection"]["name"], json!("docs"));
    assert_eq!(body["collection"]["doc_count"], json!(0));
    assert!(body["collection"]["watermark"].is_null());

    let (status, body) = create_collection(addr, "../evil");
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("invalid_collection_name"));
    assert!(body["error"].is_string());
    assert!(
        !temp.path().parent().unwrap().join("evil").exists(),
        "a traversal attempt must not create a directory"
    );

    // The collection is open, and the directory is now a real v2 collection.
    let (status, body) = health(addr);
    assert_eq!(status, 200);
    assert_eq!(body["collections"], json!(["docs"]));
    assert!(temp.path().join("docs").join("v2_manifest.json").is_file());

    server.shutdown();
}

#[test]
fn foreign_directory_is_refused_instead_of_hijacked() {
    let temp = TempDir::new().expect("temp dir");
    let foreign = temp.path().join("foreign");
    std::fs::create_dir_all(&foreign).expect("create foreign dir");
    std::fs::write(foreign.join("notes.txt"), b"not a collection").expect("write foreign file");

    let mut server = start_in_background(temp.path(), "127.0.0.1:0").expect("start server");
    let (status, body) = create_collection(server.addr(), "foreign");
    assert_eq!(status, 403, "{body}");
    assert_eq!(body["code"], json!("foreign_directory"));
    assert!(foreign.join("notes.txt").is_file(), "the file is untouched");
    server.shutdown();
}

// ---------------------------------------------------------------------------
// Write path, search, ACL fast path, delete
// ---------------------------------------------------------------------------

#[test]
fn upsert_commit_search_acl_and_delete_round_trip() {
    let temp = TempDir::new().expect("temp dir");
    let mut server = start_in_background(temp.path(), "127.0.0.1:0").expect("start server");
    let addr = server.addr();
    assert_eq!(create_collection(addr, "docs").0, 200);

    let upsert_path = "/v2/collections/docs/upsert";
    let (status, body) = call(
        addr,
        "POST",
        upsert_path,
        Some(&json!({
            "docs": [
                doc("doc-1", "Digital funding programme", "The guideline describes the funding programme.", &["public"]),
                doc("doc-2", "Archive index", "The archive keeps historic records.", &["public"]),
            ],
            "commit": false,
        })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["upserted"], json!(2));

    // Uncommitted writes are not readable yet — commit is a real gate.
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/search",
        Some(&json!({ "text": "funding", "principals": ["public"] })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["hits"], json!([]));

    let (status, body) = call(addr, "POST", "/v2/collections/docs/commit", None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["ok"], json!(true));
    assert_eq!(body["doc_count"], json!(2));

    // Fail closed: a query without principals is refused outright.
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/search",
        Some(&json!({ "text": "funding" })),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("missing_principals"));

    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/search",
        Some(&json!({ "text": "funding", "principals": ["public"], "top_k": 5 })),
    );
    assert_eq!(status, 200, "{body}");
    let hits = body["hits"].as_array().expect("hits array").clone();
    assert_eq!(hits.len(), 1, "{body}");
    assert_eq!(hits[0]["id"], json!("doc-1"));
    assert!(hits[0]["score"].is_number());
    let took_ms = body["took_ms"].as_f64().expect("took_ms is a number");
    assert!(took_ms >= 0.0);

    // ACL-only fast path: no reindex, and the next search already agrees.
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/acl",
        Some(&json!({ "id": "doc-1", "acl": ["private"] })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["ok"], json!(true));
    let generation = body["acl_generation"].as_u64().expect("acl_generation");
    assert!(generation >= 1);

    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/search",
        Some(&json!({ "text": "funding", "principals": ["public"] })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["hits"], json!([]), "revoked group must see nothing");

    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/search",
        Some(&json!({ "text": "funding", "principals": ["private"] })),
    );
    assert_eq!(status, 200, "{body}");
    let hits = body["hits"].as_array().expect("hits array");
    assert_eq!(hits.len(), 1, "{body}");
    assert_eq!(hits[0]["id"], json!("doc-1"));

    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/delete",
        Some(&json!({ "ids": ["doc-1"] })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["deleted"], json!(1));

    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/search",
        Some(&json!({ "text": "funding", "principals": ["private"] })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["hits"], json!([]));

    let (status, body) = call(addr, "GET", "/v2/collections", None);
    assert_eq!(status, 200, "{body}");
    let described = body["collections"].as_array().expect("collections array");
    assert_eq!(described.len(), 1);
    assert_eq!(described[0]["name"], json!("docs"));
    assert_eq!(described[0]["doc_count"], json!(1));
    assert!(
        described[0]["acl_generation"].as_u64().expect("acl_generation") >= generation,
        "the overlay generation must not move backwards"
    );

    // Unknown routes and collections fail with their own codes.
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/nope/search",
        Some(&json!({ "text": "funding", "principals": ["public"] })),
    );
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["code"], json!("unknown_collection"));

    let (status, body) = call(addr, "GET", "/nope", None);
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["code"], json!("no_route"));

    server.shutdown();
}

// ---------------------------------------------------------------------------
// Restart: the collection is rediscovered from disk
// ---------------------------------------------------------------------------

#[test]
fn restart_rediscovers_the_collection_and_its_documents() {
    let temp = TempDir::new().expect("temp dir");
    let root: &Path = temp.path();

    {
        let server = start_in_background(root, "127.0.0.1:0").expect("start first server");
        let addr = server.addr();
        assert_eq!(create_collection(addr, "docs").0, 200);
        let (status, body) = call(
            addr,
            "POST",
            "/v2/collections/docs/upsert",
            Some(&json!({
                "docs": [doc("doc-1", "Digital funding programme", "The guideline describes the funding programme.", &["public"])],
            })),
        );
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["upserted"], json!(1));
        // Dropping the handle must release the index writer.
    }

    let mut server = start_in_background(root, "127.0.0.1:0").expect("start second server");
    let addr = server.addr();

    let (status, body) = health(addr);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["collections"], json!(["docs"]));

    let (status, body) = call(addr, "GET", "/v2/collections", None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["collections"][0]["name"], json!("docs"));
    assert_eq!(body["collections"][0]["doc_count"], json!(1));

    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/search",
        Some(&json!({ "text": "funding", "principals": ["public"] })),
    );
    assert_eq!(status, 200, "{body}");
    let hits = body["hits"].as_array().expect("hits array");
    assert_eq!(hits.len(), 1, "{body}");
    assert_eq!(hits[0]["id"], json!("doc-1"));

    server.shutdown();
}

// ---------------------------------------------------------------------------
// Degraded index
// ---------------------------------------------------------------------------

#[test]
fn degraded_index_refuses_search_with_503() {
    use nvdb::v2::{V2CollectionConfig, V2Index};

    let temp = TempDir::new().expect("temp dir");
    {
        let mut index = V2Index::create_or_open(
            temp.path().join("docs"),
            V2CollectionConfig::default(),
        )
        .expect("create index");
        index
            .mark_degraded("apply failed during the last sync")
            .expect("mark degraded");
    }

    let mut server = start_in_background(temp.path(), "127.0.0.1:0").expect("start server");
    let addr = server.addr();

    let (status, body) = health(addr);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["collections"], json!(["docs"]));

    let (status, body) = call(addr, "GET", "/v2/collections", None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["collections"][0]["watermark"]["degraded"], json!(true));

    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/search",
        Some(&json!({ "text": "anything", "principals": ["public"] })),
    );
    assert_eq!(status, 503, "{body}");
    assert_eq!(body["code"], json!("index_degraded"));

    server.shutdown();
}

// ---------------------------------------------------------------------------
// Malformed requests
// ---------------------------------------------------------------------------
#[test]
fn malformed_bodies_are_rejected_with_400() {
    let temp = TempDir::new().expect("temp dir");
    let mut server = start_in_background(temp.path(), "127.0.0.1:0").expect("start server");
    let addr = server.addr();
    assert_eq!(create_collection(addr, "docs").0, 200);

    let (status, body) = call(addr, "POST", "/v2/collections", None);
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("empty_body"));

    let (status, body) = call(addr, "POST", "/v2/collections", Some(&json!({ "nam": "docs" })));
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("invalid_body"));

    // An unknown query key is a client bug, not a silently ignored field.
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/search",
        Some(&json!({ "text": "funding", "principal": ["public"] })),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("invalid_body"));

    server.shutdown();
}

// ---------------------------------------------------------------------------
// Fingerprint exchange
// ---------------------------------------------------------------------------

#[test]
fn fingerprints_report_hashes_vectors_and_acl_in_id_order() {
    let temp = TempDir::new().expect("temp dir");
    let mut server = start_in_background(temp.path(), "127.0.0.1:0").expect("start server");
    let addr = server.addr();

    let (status, body) = create_vector_collection(addr, "docs");
    assert_eq!(status, 200, "{body}");

    let (status, body) = upsert_docs(
        addr,
        "docs",
        vec![
            doc(
                "doc-b",
                "Archive index",
                "The archive keeps historic records.",
                &["public"],
            ),
            doc(
                "doc-a",
                "Digital funding programme",
                "The guideline describes the funding programme.",
                &["public", "hr"],
            ),
        ],
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["upserted"], json!(2));

    let (status, body) = fingerprints(addr, "docs");
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["doc_count"], json!(2));
    assert!(body["watermark"].is_null());
    let entries = body["entries"].as_array().expect("entries array");
    assert_eq!(entries.len(), 2, "{body}");
    assert_eq!(entries[0]["id"], json!("doc-a"), "sorted by id");
    assert_eq!(entries[1]["id"], json!("doc-b"));
    assert_eq!(entries[0]["acl"], json!(["hr", "public"]));
    assert_eq!(entries[0]["has_vector"], json!(false));
    assert_eq!(entries[1]["has_vector"], json!(false));
    let content_hash = entries[0]["content_hash"]
        .as_str()
        .expect("content_hash is a string")
        .to_string();
    assert_eq!(content_hash.len(), 64, "{body}");
    assert_eq!(
        entries[0]["envelope_hash"]
            .as_str()
            .expect("envelope_hash")
            .len(),
        64
    );

    // Attaching the embedding the entry names flips `has_vector` on the next read.
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/attach_vectors",
        Some(&json!({
            "attachments": [{
                "id": "doc-a",
                "content_hash": content_hash,
                "model_version": "test-v1",
                "vector": [1.0, 0.0, 0.0, 0.0],
            }],
        })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["attached"], json!(1));

    let (status, body) = fingerprints(addr, "docs");
    assert_eq!(status, 200, "{body}");
    let entries = body["entries"].as_array().expect("entries array");
    assert_eq!(entries[0]["has_vector"], json!(true), "{body}");
    assert_eq!(entries[1]["has_vector"], json!(false));

    server.shutdown();
}

#[test]
fn integrity_check_reports_every_kind_of_difference() {
    let temp = TempDir::new().expect("temp dir");
    let mut server = start_in_background(temp.path(), "127.0.0.1:0").expect("start server");
    let addr = server.addr();
    assert_eq!(create_collection(addr, "docs").0, 200);

    let (status, body) = upsert_docs(
        addr,
        "docs",
        vec![
            doc("doc-a", "Funding", "The funding programme.", &["public"]),
            doc("doc-b", "Archive", "Historic records.", &["public"]),
        ],
    );
    assert_eq!(status, 200, "{body}");

    let (status, body) = fingerprints(addr, "docs");
    assert_eq!(status, 200, "{body}");
    let entries = body["entries"].as_array().expect("entries array").clone();
    let expected: Vec<Value> = entries
        .iter()
        .map(|entry| {
            json!({
                "id": entry["id"],
                "content_hash": entry["content_hash"],
                "envelope_hash": entry["envelope_hash"],
            })
        })
        .collect();

    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/integrity_check",
        Some(&json!({ "expected": expected })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["clean"], json!(true), "{body}");
    for key in [
        "missing",
        "extra",
        "content_mismatched",
        "envelope_mismatched",
        "missing_vectors",
    ] {
        assert_eq!(body[key], json!([]), "{key} must be empty: {body}");
    }

    // `check_vectors` defaults to false; when requested, a lane-less collection
    // has no embedding for anyone.
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/integrity_check",
        Some(&json!({ "expected": expected, "check_vectors": true })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["clean"], json!(false), "{body}");
    assert_eq!(body["missing_vectors"], json!(["doc-a", "doc-b"]));

    // A wrong text hash, a wrong envelope hash, an unknown document and an
    // unexpected document all land in their own vector.
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/integrity_check",
        Some(&json!({
            "expected": [
                {
                    "id": "doc-a",
                    "content_hash": "0".repeat(64),
                    "envelope_hash": "1".repeat(64),
                },
                {
                    "id": "doc-c",
                    "content_hash": "2".repeat(64),
                    "envelope_hash": "3".repeat(64),
                },
            ],
        })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["clean"], json!(false));
    assert_eq!(body["missing"], json!(["doc-c"]));
    assert_eq!(body["extra"], json!(["doc-b"]));
    assert_eq!(body["content_mismatched"], json!(["doc-a"]));
    assert_eq!(body["envelope_mismatched"], json!(["doc-a"]));
    assert_eq!(body["missing_vectors"], json!([]));

    server.shutdown();
}

// ---------------------------------------------------------------------------
// Watermark route
// ---------------------------------------------------------------------------

#[test]
fn watermark_is_validated_persisted_and_survives_reopen() {
    let temp = TempDir::new().expect("temp dir");
    let root: &Path = temp.path();

    {
        let server = start_in_background(root, "127.0.0.1:0").expect("start server");
        let addr = server.addr();
        assert_eq!(create_collection(addr, "docs").0, 200);

        let watermark_path = "/v2/collections/docs/watermark";
        for invalid in [
            json!({ "source_epoch_uuid": "", "synced_seq": 3 }),
            json!({ "source_epoch_uuid": "epoch-1", "synced_seq": 3, "degraded": true }),
            json!({
                "source_epoch_uuid": "epoch-1",
                "synced_seq": 3,
                "degraded": false,
                "degraded_reason": "not degraded",
            }),
        ] {
            let (status, body) = call(addr, "POST", watermark_path, Some(&invalid));
            assert_eq!(status, 400, "{body}");
            assert_eq!(body["code"], json!("invalid_watermark"), "{body}");
        }

        // Nothing invalid was persisted.
        let (status, body) = fingerprints(addr, "docs");
        assert_eq!(status, 200, "{body}");
        assert!(body["watermark"].is_null(), "{body}");

        let (status, body) = call(
            addr,
            "POST",
            watermark_path,
            Some(&json!({ "source_epoch_uuid": "epoch-1", "synced_seq": 3 })),
        );
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["ok"], json!(true));
        assert_eq!(body["watermark"]["source_epoch_uuid"], json!("epoch-1"));
        assert_eq!(body["watermark"]["synced_seq"], json!(3));
        assert_eq!(body["watermark"]["degraded"], json!(false));
        assert!(body["watermark"]["degraded_reason"].is_null(), "{body}");

        let (status, body) = fingerprints(addr, "docs");
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["watermark"]["synced_seq"], json!(3));
        // Dropping the handle releases the index writer.
    }

    let mut server = start_in_background(root, "127.0.0.1:0").expect("start second server");
    let addr = server.addr();
    let (status, body) = fingerprints(addr, "docs");
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["watermark"]["source_epoch_uuid"], json!("epoch-1"));
    assert_eq!(body["watermark"]["synced_seq"], json!(3));
    assert_eq!(body["watermark"]["degraded"], json!(false));

    // A healthy watermark keeps the collection readable.
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/search",
        Some(&json!({ "text": "anything", "principals": ["public"] })),
    );
    assert_eq!(status, 200, "{body}");

    server.shutdown();
}

// ---------------------------------------------------------------------------
// Level-2 refinement route
// ---------------------------------------------------------------------------

#[test]
fn level2_fails_closed_without_principals_or_a_reranker() {
    let temp = TempDir::new().expect("temp dir");
    let mut server = start_in_background(temp.path(), "127.0.0.1:0").expect("start server");
    let addr = server.addr();
    assert_eq!(create_collection(addr, "docs").0, 200);

    let (status, body) = health(addr);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["reranker_loaded"], json!(false), "{body}");

    let level2_path = "/v2/collections/docs/level2";
    let (status, body) = call(
        addr,
        "POST",
        level2_path,
        Some(&json!({ "query_text": "funding", "principals": [], "hits": [] })),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("missing_principals"));

    let (status, body) = call(
        addr,
        "POST",
        level2_path,
        Some(&json!({ "query_text": "funding", "principals": ["public"], "hits": [] })),
    );
    assert_eq!(status, 503, "{body}");
    assert_eq!(body["code"], json!("reranker_not_configured"));

    server.shutdown();
}

#[test]
fn level2_reranks_filters_revoked_documents_and_enforces_the_floor() {
    let temp = TempDir::new().expect("temp dir");
    let reranker = KeywordReranker::new("historic");
    let mut server =
        start_in_background_with_reranker(temp.path(), "127.0.0.1:0", reranker.clone())
            .expect("start server");
    let addr = server.addr();

    let (status, body) = health(addr);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["reranker_loaded"], json!(true), "{body}");

    assert_eq!(create_collection(addr, "docs").0, 200);
    let (status, body) = upsert_docs(
        addr,
        "docs",
        vec![
            doc(
                "doc-a",
                "Digital funding programme",
                "The guideline describes the funding programme.",
                &["public"],
            ),
            doc(
                "doc-b",
                "Historic archive",
                "The archive keeps historic records.",
                &["public"],
            ),
            doc(
                "doc-c",
                "Secret memo",
                "The memo is not for the public.",
                &["public"],
            ),
        ],
    );
    assert_eq!(status, 200, "{body}");

    // Revoke doc-c after the index saw it: Level 1 still believes it is public.
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/acl",
        Some(&json!({ "id": "doc-c", "acl": ["private"] })),
    );
    assert_eq!(status, 200, "{body}");

    let level2_path = "/v2/collections/docs/level2";
    let level1 = json!([
        level1_hit("doc-c", "Secret memo", "memo.md", 1),
        level1_hit("doc-a", "Digital funding programme", "funding.md", 2),
        level1_hit("doc-b", "Historic archive", "archive.md", 3),
    ]);

    // The reranker prefers doc-b, and its weight (50) dominates the Level-1 order.
    let (status, body) = call(
        addr,
        "POST",
        level2_path,
        Some(&json!({
            "query_text": "historic records",
            "principals": ["public"],
            "hits": level1,
            "rerank_weight": 50.0,
            "relevance_floor": 0.5,
        })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["rejected"], json!(false), "{body}");
    let best = body["best_rerank_score"].as_f64().expect("best score");
    assert!((best - 0.9).abs() < 1e-6, "{body}");

    let hits = body["hits"].as_array().expect("hits array");
    let ids: Vec<&str> = hits
        .iter()
        .map(|hit| hit["hybrid"]["hit"]["id"].as_str().expect("hit id"))
        .collect();
    assert_eq!(ids, vec!["doc-b", "doc-a"], "reranker order wins: {body}");
    let hit_scores: Vec<f64> = hits
        .iter()
        .map(|hit| hit["rerank_score"].as_f64().expect("rerank score"))
        .collect();
    assert!((hit_scores[0] - 0.9).abs() < 1e-6, "{body}");
    assert!((hit_scores[1] - 0.01).abs() < 1e-6, "{body}");
    assert_eq!(
        hits[0]["hydrated_text"],
        json!("Historic archive archive.md"),
        "{body}"
    );

    // The revoked document was dropped before the model saw anything, and it is
    // absent from the answer.
    assert!(!ids.contains(&"doc-c"), "{body}");
    let seen = reranker.seen();
    assert_eq!(
        seen.len(),
        2,
        "only authorised candidates are scored: {seen:?}"
    );
    assert!(
        seen.iter()
            .all(|text| !text.to_lowercase().contains("memo")),
        "revoked text must never reach the reranker: {seen:?}"
    );

    // Caller-supplied authoritative text overrides title+path. Both documents
    // now look the same to the reranker, so the Level-1 order decides.
    let (status, body) = call(
        addr,
        "POST",
        level2_path,
        Some(&json!({
            "query_text": "historic records",
            "principals": ["public"],
            "hits": json!([
                level1_hit("doc-a", "Digital funding programme", "funding.md", 1),
                level1_hit("doc-b", "Historic archive", "archive.md", 2),
            ]),
            "texts": {
                "doc-a": "historic override text",
                "doc-b": "historic archive text",
            },
            "relevance_floor": 0.5,
        })),
    );
    assert_eq!(status, 200, "{body}");
    let hits = body["hits"].as_array().expect("hits array");
    assert_eq!(hits[0]["hybrid"]["hit"]["id"], json!("doc-a"), "{body}");
    assert_eq!(hits[0]["hydrated_text"], json!("historic override text"));

    // Nothing matches: the floor turns confident-looking noise into a rejection.
    let (status, body) = call(
        addr,
        "POST",
        level2_path,
        Some(&json!({
            "query_text": "zorblat",
            "principals": ["public"],
            "hits": json!([level1_hit("doc-b", "Historic archive", "archive.md", 1)]),
            "relevance_floor": 0.5,
        })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["rejected"], json!(true), "{body}");
    let best = body["best_rerank_score"].as_f64().expect("best score");
    assert!((best - 0.01).abs() < 1e-6, "{body}");
    assert_eq!(
        body["hits"].as_array().expect("hits array").len(),
        1,
        "a rejection keeps its evidence: {body}"
    );

    // Without an explicit floor the default (0.02) still rejects a nonsense match.
    let (status, body) = call(
        addr,
        "POST",
        level2_path,
        Some(&json!({
            "query_text": "zorblat",
            "principals": ["public"],
            "hits": json!([level1_hit("doc-b", "Historic archive", "archive.md", 1)]),
        })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["rejected"], json!(true), "{body}");

    server.shutdown();
}
