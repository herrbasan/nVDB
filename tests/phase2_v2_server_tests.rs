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
use std::time::Duration;

use nvdb::v2::server::start_in_background;
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
