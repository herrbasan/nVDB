//! Phase 3: unified hybrid search, `content_hash`-keyed vectors and Level-2
//! refinement.
//!
//! Coverage: content-hash embedding deduplication and reuse (ACL-only,
//! metadata-only, text-edit invalidation, persistence across reopen), the
//! derived pending list and the atomic compare-and-set attach, ACL and columnar
//! filter enforcement in the dense lane, scale-free RRF fusion with a single
//! diversity cap, Level-2 authoritative ACL re-check with a pluggable reranker
//! and a relevance floor, and the HTTP round trip through `nvdb serve`.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use nvdb::rerank::Reranker;
use nvdb::sync::{
    IndexedEnvelope, SourceChange, SourceConnector, SourceLineage, SourceRecord, SourceSnapshot,
    SyncError, VectorAttachment,
};
use nvdb::v2::server::start_in_background;
use nvdb::v2::{
    V2CollectionConfig, V2Document, V2Error, V2Filters, V2HybridHit, V2HybridQuery, V2Index,
    V2Level2Options, V2Query, VectorLaneConfig,
};
use serde_json::{json, Value};
use tempfile::TempDir;

const DIM: usize = 4;
const MODEL: &str = "test-embed-v1";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn groups(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| value.to_string()).collect()
}

fn lane_config() -> V2CollectionConfig {
    V2CollectionConfig::with_vector_lane(VectorLaneConfig::new(DIM, MODEL))
}

fn open(dir: &Path) -> V2Index {
    V2Index::create_or_open(dir, lane_config()).expect("open vector-lane collection")
}

fn doc(id: &str, title: &str, body: &str, acl: &[&str]) -> V2Document {
    let mut document = V2Document::new(id, "doc");
    document.title = title.to_string();
    document.body = body.to_string();
    document.acl = groups(acl);
    document
}

fn attachment(id: &str, content_hash: &str, vector: Vec<f32>) -> VectorAttachment {
    VectorAttachment {
        id: id.to_string(),
        content_hash: content_hash.to_string(),
        model_version: MODEL.to_string(),
        vector,
    }
}

fn attach(index: &mut V2Index, id: &str, vector: Vec<f32>) {
    let content_hash = index
        .next_pending_embeddings(usize::MAX)
        .into_iter()
        .find(|pending| pending.id == id)
        .map(|pending| pending.content_hash)
        .expect("document must be pending before it can be attached");
    index
        .attach_vectors(vec![attachment(id, &content_hash, vector)])
        .expect("attach vector");
}

fn hybrid_ids(hits: &[V2HybridHit]) -> Vec<String> {
    hits.iter().map(|hit| hit.hit.id.clone()).collect()
}

fn hybrid_query(text: &str, principals: &[&str], top_k: usize) -> V2Query {
    V2Query::new(text)
        .principals(principals.iter().copied())
        .top_k(top_k)
}

// ---------------------------------------------------------------------------
// 1. Content-hash vectors: dedup, reuse, invalidation, persistence
// ---------------------------------------------------------------------------

#[test]
fn content_hash_vectors_dedup_reuse_and_survive_reopen() {
    let temp = TempDir::new().expect("temp dir");
    let dir = temp.path().join("index");
    let mut index = open(&dir);

    let mut report = doc(
        "a",
        "Quarterly Report",
        "revenue grew twelve percent against plan",
        &["public"],
    );
    report.folder = Some("finance".to_string());
    index.upsert(report.clone()).expect("upsert");
    index.commit().expect("commit");

    let pending = index.next_pending_embeddings(10);
    assert_eq!(
        pending.len(),
        1,
        "a live document without a vector is pending"
    );
    assert_eq!(pending[0].id, "a");
    assert_eq!(
        pending[0].content_hash,
        report.content_hash(),
        "the pending hash is the wire (hex) form of the envelope content hash"
    );
    assert_eq!(pending[0].title, "Quarterly Report");

    assert_eq!(
        index
            .attach_vectors(vec![attachment(
                "a",
                &report.content_hash(),
                vec![1.0, 0.0, 0.0, 0.0]
            )])
            .expect("attach"),
        1
    );
    assert!(index.next_pending_embeddings(10).is_empty());
    assert_eq!(index.vec_generation(), 1, "an attach writes generation 1");

    // Identical text is a dedup hit: one embedding, two documents, zero extra
    // embeddings requested.
    let twin = doc(
        "b",
        "Quarterly Report",
        "revenue grew twelve percent against plan",
        &["public"],
    );
    assert_eq!(twin.content_hash(), report.content_hash());
    index.upsert(twin.clone()).expect("upsert twin");
    index.commit().expect("commit");
    assert_eq!(
        index.vectors().len(),
        1,
        "identical content hashes share exactly one embedding"
    );
    assert!(
        index.next_pending_embeddings(10).is_empty(),
        "the twin is auto-linked by content hash"
    );
    assert!(index.vector_for("b").is_some());

    // ACL-only upsert: no re-embedding, the link survives untouched.
    let generation_before = index.vec_generation();
    let mut revoked = report.clone();
    revoked.acl = groups(&["finance"]);
    index.upsert(revoked.clone()).expect("acl-only upsert");
    index.commit().expect("commit");
    assert!(
        index.vector_for("a").is_some(),
        "ACL-only change keeps the vector"
    );
    assert_eq!(
        index.vec_generation(),
        generation_before,
        "an unchanged content hash does not rewrite the link map"
    );

    // Metadata-only upsert: same story.
    let mut annotated = revoked.clone();
    annotated.metadata = json!({ "reviewed": true });
    index
        .upsert(annotated.clone())
        .expect("metadata-only upsert");
    index.commit().expect("commit");
    assert!(index.vector_for("a").is_some());
    assert!(index.next_pending_embeddings(10).is_empty());

    // Text edit invalidates the link, so the document becomes pending again.
    let mut edited = annotated.clone();
    edited.body = "revenue fell three percent against plan".to_string();
    assert_ne!(edited.content_hash(), report.content_hash());
    index.upsert(edited.clone()).expect("text edit");
    index.commit().expect("commit");
    assert!(
        index.vector_for("a").is_none(),
        "edited text must not reuse the old embedding"
    );
    let pending = index.next_pending_embeddings(10);
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].id, "a");
    assert_eq!(pending[0].content_hash, edited.content_hash());

    index
        .attach_vectors(vec![attachment(
            "a",
            &edited.content_hash(),
            vec![0.0, 1.0, 0.0, 0.0],
        )])
        .expect("attach edited text");
    drop(index);

    let reopened = open(&dir);
    assert_eq!(reopened.vectors().len(), 2, "the old embedding is kept");
    assert_eq!(
        reopened.vector_for("a").expect("link for a"),
        &[0.0f32, 1.0, 0.0, 0.0]
    );
    assert_eq!(
        reopened.vector_for("b").expect("link for b"),
        &[1.0f32, 0.0, 0.0, 0.0]
    );
    assert!(reopened.next_pending_embeddings(10).is_empty());
    assert!(reopened.vec_generation() >= 2);

    // Deleting drops the link; the shared embedding survives because "b" still
    // references it.
    let mut reopened = reopened;
    reopened.delete("b").expect("delete");
    reopened.commit().expect("commit");
    assert!(reopened.vector_for("b").is_none());
    assert_eq!(reopened.vectors().len(), 2);
    assert_eq!(
        reopened.gc_vectors(),
        1,
        "the rebuild-time GC drops b's embedding"
    );
    reopened.commit().expect("commit");
    assert_eq!(reopened.vectors().len(), 1);
}

// ---------------------------------------------------------------------------
// 2. Derived pending list + atomic compare-and-set attach
// ---------------------------------------------------------------------------

#[test]
fn pending_embeddings_are_derived_and_attach_is_atomic() {
    let temp = TempDir::new().expect("temp dir");
    let mut index = open(&temp.path().join("index"));

    let docs = [
        doc("a", "Alpha", "first document", &["public"]),
        doc("b", "Beta", "second document", &["public"]),
        doc("c", "Gamma", "third document", &["public"]),
    ];
    for document in &docs {
        index.upsert(document.clone()).expect("upsert");
    }
    index.commit().expect("commit");

    // Deterministic order and a respected limit.
    let pending = index.next_pending_embeddings(2);
    assert_eq!(pending.len(), 2);
    assert_eq!(
        pending.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(index.next_pending_embeddings(10).len(), 3);
    assert!(index.next_pending_embeddings(0).is_empty());

    let hash_a = docs[0].content_hash();
    let hash_b = docs[1].content_hash();
    let hash_c = docs[2].content_hash();

    // A stale completion rejects the whole batch: the valid entry must not land.
    let error = index
        .attach_vectors(vec![
            attachment("a", &hash_a, vec![1.0, 0.0, 0.0, 0.0]),
            attachment("b", "deadbeef", vec![1.0, 0.0, 0.0, 0.0]),
        ])
        .expect_err("a malformed content hash is not silently skipped");
    assert!(matches!(error, V2Error::InvalidQuery(_)), "{error}");
    let error = index
        .attach_vectors(vec![
            attachment("a", &hash_a, vec![1.0, 0.0, 0.0, 0.0]),
            attachment("b", &hash_c, vec![1.0, 0.0, 0.0, 0.0]),
        ])
        .expect_err("c's completion offered for b");
    assert!(matches!(error, V2Error::StaleVectorCas { .. }), "{error}");
    assert_eq!(
        index.next_pending_embeddings(10).len(),
        3,
        "a rejected batch must not half-apply"
    );
    assert_eq!(index.vec_generation(), 0);
    assert_eq!(index.vectors().len(), 0);

    // Wrong dimension.
    let error = index
        .attach_vectors(vec![attachment("a", &hash_a, vec![1.0, 0.0, 0.0])])
        .expect_err("wrong dimension");
    assert!(
        matches!(error, V2Error::WrongDimension { expected, got } if expected == DIM && got == 3),
        "{error}"
    );

    // Wrong model.
    let mut wrong_model = attachment("a", &hash_a, vec![1.0, 0.0, 0.0, 0.0]);
    wrong_model.model_version = "other-model".to_string();
    let error = index
        .attach_vectors(vec![wrong_model])
        .expect_err("wrong model");
    assert!(matches!(error, V2Error::InvalidQuery(_)), "{error}");

    // Non-finite values.
    let error = index
        .attach_vectors(vec![attachment(
            "a",
            &hash_a,
            vec![f32::NAN, 0.0, 0.0, 0.0],
        )])
        .expect_err("NaN embedding");
    assert!(matches!(error, V2Error::InvalidQuery(_)), "{error}");

    // Deleted document.
    index.delete("c").expect("delete");
    index.commit().expect("commit");
    let error = index
        .attach_vectors(vec![attachment("c", &hash_c, vec![1.0, 0.0, 0.0, 0.0])])
        .expect_err("deleted document");
    assert!(matches!(error, V2Error::StaleVectorCas { .. }), "{error}");

    // Now a fully valid batch lands in one atomically committed generation.
    let attached = index
        .attach_vectors(vec![
            attachment("a", &hash_a, vec![1.0, 0.0, 0.0, 0.0]),
            attachment("b", &hash_b, vec![0.0, 1.0, 0.0, 0.0]),
        ])
        .expect("valid batch");
    assert_eq!(attached, 2);
    assert!(index.next_pending_embeddings(10).is_empty());
    assert!(
        index.vec_generation() >= 1,
        "each link change bumps the generation"
    );
}

// ---------------------------------------------------------------------------
// 3. The dense lane obeys the ACL overlay and the columnar filters
// ---------------------------------------------------------------------------

#[test]
fn vector_and_hybrid_lanes_never_leak_past_acl_or_filters() {
    let temp = TempDir::new().expect("temp dir");
    let mut index = open(&temp.path().join("index"));

    let mut secret = doc(
        "secret",
        "Salary bands",
        "confidential compensation spreadsheet",
        &["hr-secret"],
    );
    secret.folder = Some("hr".to_string());
    secret.ext = Some("pdf".to_string());

    let mut public_hr = doc(
        "public-hr",
        "Holiday policy",
        "how many days of leave",
        &["public"],
    );
    public_hr.folder = Some("hr".to_string());
    public_hr.ext = Some("md".to_string());

    let mut public_docs = doc(
        "public-docs",
        "Onboarding",
        "first week checklist",
        &["public"],
    );
    public_docs.folder = Some("docs".to_string());
    public_docs.ext = Some("md".to_string());

    for document in [&secret, &public_hr, &public_docs] {
        index.upsert(document.clone()).expect("upsert");
    }
    index.commit().expect("commit");

    // Every document gets a vector pointing at the same query direction, so the
    // only thing separating them is ACL and the filters.
    attach(&mut index, "secret", vec![1.0, 0.0, 0.0, 0.0]);
    attach(&mut index, "public-hr", vec![1.0, 0.0, 0.0, 0.0]);
    attach(&mut index, "public-docs", vec![1.0, 0.0, 0.0, 0.0]);
    index.commit().expect("commit");

    let vector_only = |principals: &[&str], filters: V2Filters| {
        let mut query = V2Query::new("")
            .principals(principals.iter().copied())
            .top_k(10);
        query.filters = filters;
        V2HybridQuery::new(query)
            .lexical_enabled(false)
            .vector(vec![1.0, 0.0, 0.0, 0.0])
    };

    let hits = index
        .hybrid_search(&vector_only(&["public"], V2Filters::none()))
        .expect("vector-only search");
    let ids = hybrid_ids(&hits);
    assert!(
        !ids.contains(&"secret".to_string()),
        "the dense lane must obey the ACL overlay: {ids:?}"
    );
    assert_eq!(ids.len(), 2);
    for hit in &hits {
        assert!(hit.lexical_score.is_none());
        assert!(hit.lexical_rank.is_none());
        assert!(hit.vector_score.is_some());
    }

    let hits = index
        .hybrid_search(&vector_only(&["hr-secret"], V2Filters::none()))
        .expect("secret vector search");
    assert_eq!(hybrid_ids(&hits), vec!["secret".to_string()]);

    let hits = index
        .hybrid_search(&vector_only(&["public"], V2Filters::none().folder("hr")))
        .expect("folder-filtered vector search");
    assert_eq!(
        hybrid_ids(&hits),
        vec!["public-hr".to_string()],
        "the folder filter must intersect the visible set"
    );

    let hits = index
        .hybrid_search(&vector_only(&["public"], V2Filters::none().ext("pdf")))
        .expect("ext-filtered vector search");
    assert!(
        hits.is_empty(),
        "the only pdf is invisible to 'public': {hits:?}"
    );

    // The same authority applies to the fused path.
    let mut query = V2Query::new("leave days").principals(["public"]).top_k(10);
    query.filters = V2Filters::none().folder("hr");
    let hits = index
        .hybrid_search(&V2HybridQuery::new(query).vector(vec![1.0, 0.0, 0.0, 0.0]))
        .expect("hybrid search");
    assert_eq!(hybrid_ids(&hits), vec!["public-hr".to_string()]);

    // Filter-only scans still respect the overlay.
    let mut query = V2Query::new("").principals(["public"]).top_k(10);
    query.filters = V2Filters::none().folder("hr");
    let hits = index
        .hybrid_search(&V2HybridQuery::new(query))
        .expect("lexical filter-only search");
    assert_eq!(hybrid_ids(&hits), vec!["public-hr".to_string()]);
}

// ---------------------------------------------------------------------------
// 4. Unified RRF: paraphrases lift, exact hits stay, one diversity cap
// ---------------------------------------------------------------------------

#[test]
fn hybrid_rrf_lifts_paraphrases_and_caps_parents_once() {
    let temp = TempDir::new().expect("temp dir");
    let mut index = open(&temp.path().join("index"));

    let exact = doc(
        "exact",
        "Digitalisation funding programme",
        "the digitalisation funding programme pays for software",
        &["public"],
    );
    // No shared term with the query: only the dense lane can find it.
    let paraphrase = doc(
        "paraphrase",
        "Zuschüsse für moderne Bürosoftware",
        "Beihilfen für die Umstellung auf digitale Abläufe",
        &["public"],
    );
    let unrelated = doc(
        "unrelated",
        "Cafeteria opening hours",
        "the canteen opens at eleven",
        &["public"],
    );
    for document in [&exact, &paraphrase, &unrelated] {
        index.upsert(document.clone()).expect("upsert");
    }
    index.commit().expect("commit");

    attach(&mut index, "exact", vec![1.0, 0.05, 0.0, 0.0]);
    attach(&mut index, "paraphrase", vec![1.0, 0.0, 0.0, 0.0]);
    attach(&mut index, "unrelated", vec![0.0, 0.0, 1.0, 0.0]);
    index.commit().expect("commit");

    let lexical = index
        .search(&hybrid_query(
            "digitalisation funding programme",
            &["public"],
            10,
        ))
        .expect("lexical search");
    let lexical_ids: Vec<String> = lexical.iter().map(|hit| hit.id.clone()).collect();
    assert_eq!(lexical_ids.first().map(String::as_str), Some("exact"));
    assert!(
        !lexical_ids.contains(&"paraphrase".to_string()),
        "the paraphrase carries no query term: {lexical_ids:?}"
    );

    let query = V2HybridQuery::new(hybrid_query(
        "digitalisation funding programme",
        &["public"],
        10,
    ))
    .vector(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = index.hybrid_search(&query).expect("hybrid search");
    let ids = hybrid_ids(&hits);
    assert_eq!(
        ids.first().map(String::as_str),
        Some("exact"),
        "the exact lexical match keeps rank 1: {ids:?}"
    );
    assert!(
        ids.contains(&"paraphrase".to_string()),
        "the dense lane lifts the paraphrase: {ids:?}"
    );
    // An orthogonal document is still a (score-zero) dense candidate; it must
    // never outrank a real match.
    let paraphrase_position = ids.iter().position(|id| id == "paraphrase");
    let unrelated_position = ids.iter().position(|id| id == "unrelated");
    assert!(
        unrelated_position.is_none() || paraphrase_position < unrelated_position,
        "a zero-similarity document must rank below the paraphrase: {ids:?}"
    );

    let paraphrase_hit = hits
        .iter()
        .find(|hit| hit.hit.id == "paraphrase")
        .expect("paraphrase hit");
    assert!(paraphrase_hit.lexical_rank.is_none());
    assert_eq!(paraphrase_hit.vector_rank, Some(1));
    assert!(paraphrase_hit.vector_score.expect("vector score") > 0.99);
    assert_eq!(paraphrase_hit.hit.score, paraphrase_hit.rrf_score);
    let exact_hit = hits
        .iter()
        .find(|hit| hit.hit.id == "exact")
        .expect("exact hit");
    assert_eq!(exact_hit.lexical_rank, Some(1));
    assert!(exact_hit.lexical_score.is_some());

    // Lane weights are honoured.
    let vector_heavy = query.clone().weights(0.0, 1.0);
    let hits = index.hybrid_search(&vector_heavy).expect("vector-heavy");
    assert_eq!(
        hits.first().map(|hit| hit.hit.id.as_str()),
        Some("paraphrase")
    );

    // Diversity is applied once, on the fused order.
    let mut chunked = Vec::new();
    for index_in_parent in 0..3 {
        let mut chunk = doc(
            &format!("p1-{index_in_parent}"),
            "Digitalisation funding programme",
            &format!(
                "the digitalisation funding programme pays for software, part {index_in_parent}"
            ),
            &["public"],
        );
        chunk.parent_id = Some("p1".to_string());
        chunk.chunk_index = Some(index_in_parent);
        chunked.push(chunk);
    }
    let mut solo = doc(
        "solo",
        "Digitalisation funding programme summary",
        "the digitalisation funding programme in one page",
        &["public"],
    );
    solo.parent_id = Some("p2".to_string());
    chunked.push(solo);
    for document in &chunked {
        index.upsert(document.clone()).expect("upsert chunk");
    }
    index.commit().expect("commit");
    for chunk in &chunked {
        attach(&mut index, &chunk.id, vec![0.95, 0.05, 0.0, 0.0]);
    }
    index.commit().expect("commit");

    let mut lex = hybrid_query("digitalisation funding programme", &["public"], 10);
    lex.group_by_parent = Some(nvdb::v2::GroupByParent::new(1));
    let hits = index
        .hybrid_search(&V2HybridQuery::new(lex).vector(vec![1.0, 0.0, 0.0, 0.0]))
        .expect("diverse hybrid search");
    let ids = hybrid_ids(&hits);
    let p1_count = ids.iter().filter(|id| id.starts_with("p1-")).count();
    assert_eq!(p1_count, 1, "one hit per parent, applied once: {ids:?}");
    assert!(ids.contains(&"solo".to_string()), "{ids:?}");
    assert!(ids.len() <= 10);

    // Fail-closed and fail-fast guards.
    assert!(matches!(
        index.hybrid_search(&V2HybridQuery::new(V2Query::new("anything").top_k(5))),
        Err(V2Error::MissingPrincipals)
    ));
    assert!(matches!(
        index.hybrid_search(
            &V2HybridQuery::new(hybrid_query("x", &["public"], 5)).lexical_enabled(false)
        ),
        Err(V2Error::InvalidQuery(_))
    ));
    assert!(matches!(
        index.hybrid_search(&V2HybridQuery::new(hybrid_query("x", &["public"], 5)).dense(true)),
        Err(V2Error::InvalidQuery(_))
    ));
    assert!(matches!(
        index.hybrid_search(
            &V2HybridQuery::new(hybrid_query("x", &["public"], 5)).vector(vec![1.0, 0.0])
        ),
        Err(V2Error::WrongDimension {
            expected: DIM,
            got: 2
        })
    ));
}

// ---------------------------------------------------------------------------
// 5. Level-2 refinement
// ---------------------------------------------------------------------------

/// A source whose ACLs can change *after* the index synced from it.
struct MutableSource {
    records: Mutex<BTreeMap<String, SourceRecord>>,
}

impl MutableSource {
    fn new(documents: &[V2Document]) -> Self {
        let records = documents
            .iter()
            .map(|document| {
                let envelope = IndexedEnvelope {
                    id: document.id.clone(),
                    parent_id: document.parent_id.clone(),
                    chunk_index: document.chunk_index,
                    kind: document.kind.clone(),
                    indexable_text: document.indexable_text(),
                    acl: document.acl.clone(),
                    metadata: document.metadata.clone(),
                    schema_version: 2,
                    analyzer_version: 2,
                };
                (document.id.clone(), SourceRecord::new(envelope))
            })
            .collect();
        Self {
            records: Mutex::new(records),
        }
    }

    fn set_acl(&self, id: &str, acl: &[&str]) {
        let mut records = self.records.lock().expect("source lock");
        let record = records.get_mut(id).expect("record exists");
        record.envelope.acl = groups(acl);
    }
}

impl SourceConnector for MutableSource {
    fn lineage(&self) -> Result<SourceLineage, SyncError> {
        Ok(SourceLineage {
            source_epoch_uuid: "epoch-1".to_string(),
            last_seq: self.records.lock().expect("source lock").len() as u64,
            min_available_seq: 0,
        })
    }

    fn snapshot_at(&self) -> Result<SourceSnapshot, SyncError> {
        Ok(SourceSnapshot {
            lineage: self.lineage()?,
            records: self
                .records
                .lock()
                .expect("source lock")
                .values()
                .cloned()
                .collect(),
        })
    }

    fn changes_since(&self, _seq: u64) -> Result<Vec<SourceChange>, SyncError> {
        Ok(Vec::new())
    }

    fn get_doc(&self, id: &str) -> Result<Option<SourceRecord>, SyncError> {
        Ok(self.records.lock().expect("source lock").get(id).cloned())
    }

    fn resolve_principals(&self, _user_id: &str) -> Result<Vec<String>, SyncError> {
        Ok(groups(&["public"]))
    }
}

/// A reranker with either keyword scoring or fixed scores; records its input so
/// a test can prove revoked text never reached it.
struct RecordingReranker {
    fixed: Option<Vec<f32>>,
    seen: Mutex<Vec<String>>,
}

impl RecordingReranker {
    fn keyword() -> Self {
        Self {
            fixed: None,
            seen: Mutex::new(Vec::new()),
        }
    }

    fn fixed(scores: Vec<f32>) -> Self {
        Self {
            fixed: Some(scores),
            seen: Mutex::new(Vec::new()),
        }
    }

    fn seen(&self) -> Vec<String> {
        self.seen.lock().expect("reranker lock").clone()
    }
}

impl Reranker for RecordingReranker {
    fn score(&self, query: &str, documents: &[&str]) -> nvdb::Result<Vec<f32>> {
        let mut seen = self.seen.lock().expect("reranker lock");
        seen.clear();
        seen.extend(documents.iter().map(|document| document.to_string()));
        drop(seen);
        if let Some(scores) = &self.fixed {
            return Ok(scores.clone());
        }
        let query = query.to_lowercase();
        let terms: Vec<&str> = query.split_whitespace().collect();
        Ok(documents
            .iter()
            .map(|document| {
                let document = document.to_lowercase();
                terms
                    .iter()
                    .filter(|term| document.contains(**term))
                    .count() as f32
            })
            .collect())
    }
}

#[test]
fn level2_rechecks_source_acl_reranks_and_respects_the_floor() {
    let temp = TempDir::new().expect("temp dir");
    let mut index = open(&temp.path().join("index"));

    let keep = doc(
        "keep",
        "Revenue review",
        "revenue grew twelve percent against plan",
        &["public"],
    );
    let revoked = doc(
        "revoked",
        "Revenue secrets",
        "revenue grew and the secret plan leaked",
        &["public"],
    );
    let weak = doc(
        "weak",
        "Cafeteria",
        "the canteen opens at eleven",
        &["public"],
    );
    for document in [&keep, &revoked, &weak] {
        index.upsert(document.clone()).expect("upsert");
    }
    index.commit().expect("commit");

    let level1 = index
        .hybrid_search(
            &V2HybridQuery::new(hybrid_query("revenue grew", &["public"], 10))
                .vector(vec![0.0, 0.0, 1.0, 0.0]),
        )
        .expect("level-1 search");
    let level1_ids = hybrid_ids(&level1);
    assert!(
        level1_ids.contains(&"revoked".to_string()),
        "{level1_ids:?}"
    );
    assert!(level1_ids.contains(&"keep".to_string()), "{level1_ids:?}");

    // The source revokes `revoked` after the last sync: the derived index still
    // believes the old ACL, and Level 2 must not.
    let source = MutableSource::new(&[keep.clone(), revoked.clone(), weak.clone()]);
    source.set_acl("revoked", &["secret-team"]);

    let reranker = RecordingReranker::keyword();
    let result = index
        .refine_level2(
            "revenue grew",
            &groups(&["public"]),
            level1,
            Some(&source),
            Some(&reranker),
            &V2Level2Options::new(),
        )
        .expect("level-2 refinement");

    let ids: Vec<&str> = result
        .hits
        .iter()
        .map(|hit| hit.hybrid.hit.id.as_str())
        .collect();
    assert!(
        !ids.contains(&"revoked"),
        "a document revoked after the last sync must be dropped: {ids:?}"
    );
    assert!(ids.contains(&"keep"), "{ids:?}");
    for hit in &result.hits {
        assert!(
            !hit.hydrated_text
                .as_deref()
                .unwrap_or_default()
                .contains("secret plan"),
            "revoked text must never surface"
        );
        assert!(hit.rerank_score.is_some());
    }
    let seen = reranker.seen();
    assert!(
        seen.iter().all(|text| !text.contains("secret plan")),
        "revoked text must never reach the reranker: {seen:?}"
    );
    assert_eq!(
        seen.len(),
        1,
        "only the surviving document is scored: {seen:?}"
    );
    assert_eq!(
        result.best_rerank_score,
        Some(2.0),
        "the surviving revenue document matched both query terms"
    );
    assert!(!result.rejected);
    assert_eq!(
        result.hits.first().map(|hit| hit.hybrid.hit.id.as_str()),
        Some("keep")
    );

    // A reranker that prefers a lower-ranked survivor reorders it.
    let level1 = index
        .hybrid_search(&V2HybridQuery::new(hybrid_query(
            "revenue grew",
            &["public"],
            10,
        )))
        .expect("level-1 search");
    let reranker = RecordingReranker::fixed(vec![0.0, 7.0]);
    let result = index
        .refine_level2(
            "revenue grew",
            &groups(&["public"]),
            level1,
            None,
            Some(&reranker),
            &V2Level2Options::new()
                .rerank_weight(5.0)
                .relevance_floor(None),
        )
        .expect("level-2 refinement");
    assert_eq!(
        result.hits.first().map(|hit| hit.hybrid.hit.id.as_str()),
        Some("revoked"),
        "a heavy reranker weight lifts the second survivor"
    );
    assert_eq!(result.best_rerank_score, Some(7.0));
    assert!(!result.rejected, "no floor means no rejection");

    // A relevance floor rejects a nonsense query while keeping the evidence.
    let level1 = index
        .hybrid_search(&V2HybridQuery::new(hybrid_query(
            "revenue grew",
            &["public"],
            10,
        )))
        .expect("level-1 search");
    let reranker = RecordingReranker::keyword();
    let result = index
        .refine_level2(
            "zzz qqq",
            &groups(&["public"]),
            level1.clone(),
            None,
            Some(&reranker),
            &V2Level2Options::new(),
        )
        .expect("level-2 refinement");
    assert!(result.rejected, "a nonsense query must be rejected");
    assert_eq!(result.best_rerank_score, Some(0.0));
    assert!(!result.hits.is_empty(), "rejected keeps the evidence");

    let result = index
        .refine_level2(
            "revenue grew",
            &groups(&["public"]),
            level1,
            None,
            Some(&reranker),
            &V2Level2Options::new(),
        )
        .expect("level-2 refinement");
    assert!(!result.rejected, "a real match passes the floor");

    // A missing document and an absent source cooperate instead of failing.
    let level1 = index
        .hybrid_search(&V2HybridQuery::new(hybrid_query(
            "revenue grew",
            &["public"],
            10,
        )))
        .expect("level-1 search");
    let mut shortlist = V2Level2Options::new();
    shortlist.shortlist = 1;
    let result = index
        .refine_level2(
            "revenue grew",
            &groups(&["public"]),
            level1,
            None,
            None,
            &shortlist,
        )
        .expect("level-2 without a reranker");
    assert_eq!(result.hits.len(), 1);
    assert!(result.best_rerank_score.is_none());
    assert!(!result.rejected);
    assert!(result.hits[0].hydrated_text.is_some());

    assert!(matches!(
        index.refine_level2("q", &[], Vec::new(), None, None, &V2Level2Options::new()),
        Err(V2Error::MissingPrincipals)
    ));

    // A reranker that breaks its contract fails loudly instead of guessing.
    let level1 = index
        .hybrid_search(&V2HybridQuery::new(hybrid_query(
            "revenue grew",
            &["public"],
            10,
        )))
        .expect("level-1 search");
    let broken = RecordingReranker::fixed(vec![1.0]);
    assert!(matches!(
        index.refine_level2(
            "revenue grew",
            &groups(&["public"]),
            level1,
            None,
            Some(&broken),
            &V2Level2Options::new(),
        ),
        Err(V2Error::Rerank(_))
    ));
}

#[test]
fn long_exact_phrase_and_hybrid_lexical_spike_survive_vector_heavy_rrf() {
    // Regression for herrbasan/nVDB#5:
    // 1. A 12-word exact phrase (> old PHRASE_MAX_TERMS=8) must keep PhraseQuery
    //    and all-terms boosts and rank #1 lexically.
    // 2. Under semantic-weighted hybrid RRF (w_L=0.3, w_V=1.0, rrf_k=20), a
    //    >= 2.5x BM25 exact-phrase outlier must not be buried behind 25 dense-only
    //    neighbors, while a flat topical query keeps dense in the lead.
    let dir = TempDir::new().expect("temp dir");
    let mut index = open(dir.path());

    let phrase = "being aware of ones own thinking and relation to other thinking things";
    let mut target = doc(
        "target",
        "DeepSeek Overview",
        &format!("I am convinced consciousness means {phrase} in practice."),
        &["public"],
    );
    target.parent_id = Some("target-sess".to_string());
    index.upsert(target).expect("upsert target");

    // 25 competing documents: dense vectors point straight at the query axis,
    // and their bodies repeat common terms from the phrase without matching it verbatim.
    for i in 0..25 {
        let mut distractor = doc(
            &format!("dist-{i}"),
            &format!("Other thinking things {i}"),
            &format!(
                "thinking thinking other things and other things to relation of thinking aware {i}"
            ),
            &["public"],
        );
        distractor.parent_id = Some(format!("dist-sess-{i}"));
        index.upsert(distractor).expect("upsert distractor");
    }
    index.commit().expect("commit");

    // Target sits orthogonal to the query vector (misses dense top-20); distractors align with it.
    attach(&mut index, "target", vec![0.0, 1.0, 0.0, 0.0]);
    for i in 0..25 {
        let eps = (i as f32) * 0.001;
        attach(&mut index, &format!("dist-{i}"), vec![1.0 - eps, eps, 0.0, 0.0]);
    }
    index.commit().expect("commit");

    // 1. Lexical 12-word exact phrase ranks #1.
    let lex_hits = index
        .search(&hybrid_query(phrase, &["public"], 10))
        .expect("lexical search");
    assert_eq!(
        lex_hits.first().map(|h| h.id.as_str()),
        Some("target"),
        "12-word exact phrase must rank #1 lexically: {:?}",
        lex_hits.iter().take(5).map(|h| (&h.id, h.score)).collect::<Vec<_>>()
    );

    // 2. Hybrid with semantic weights (w_L=0.3, w_V=1.0, rrf_k=20) promotes the
    //    exact-phrase lexical spike to #1 instead of dropping it behind vec#1..vec#25.
    let hyb = V2HybridQuery::new(hybrid_query(phrase, &["public"], 10))
        .vector(vec![1.0, 0.0, 0.0, 0.0])
        .candidate_k(200)
        .rrf_k(20)
        .weights(0.3, 1.0);
    let hyb_hits = index.hybrid_search(&hyb).expect("hybrid search");
    assert_eq!(
        hyb_hits.first().map(|h| h.hit.id.as_str()),
        Some("target"),
        "exact-phrase lexical spike must survive semantic-weighted hybrid RRF: {:?}",
        hyb_hits.iter().take(5).map(|h| (&h.hit.id, h.rrf_score, h.lexical_rank, h.vector_rank)).collect::<Vec<_>>()
    );

    // 3. Flat topical query (no >= 2.5x BM25 spike) still lets dense lead.
    let topical = V2HybridQuery::new(hybrid_query("thinking things", &["public"], 10))
        .vector(vec![1.0, 0.0, 0.0, 0.0])
        .candidate_k(200)
        .rrf_k(20)
        .weights(0.3, 1.0);
    let topical_hits = index.hybrid_search(&topical).expect("topical hybrid");
    assert!(
        topical_hits
            .first()
            .is_some_and(|h| h.hit.id.starts_with("dist-")),
        "flat topical query must keep dense-aligned hits ahead: {:?}",
        topical_hits.iter().take(3).map(|h| (&h.hit.id, h.rrf_score)).collect::<Vec<_>>()
    );
}

#[test]
fn query_that_is_a_filename_stem_ranks_the_file_above_a_body_mention() {
    // Regression for herrbasan/nVDB#4: the query IS the filename stem, so the
    // file must outrank a document that merely mentions `md-blocks-spec.md`.
    let dir = TempDir::new().expect("temp dir");
    let mut index = open(dir.path());

    let mut spec = doc(
        "spec",
        "MD-Blocks Specification",
        "Sections, blocks, columns and named data ride in invisible HTML comments.",
        &["public"],
    );
    spec.name = "md-blocks-spec.md".to_string();
    spec.path = "documentation/md-blocks/md-blocks-spec.md".to_string();
    index.upsert(spec).expect("upsert spec");

    let mut brief = doc(
        "brief",
        "Authoring brief",
        "Keep the rules concise; see md-blocks-spec.md for the full rules and examples.",
        &["public"],
    );
    brief.name = "BRIEF.md".to_string();
    brief.path = "documentation/md-blocks/BRIEF.md".to_string();
    index.upsert(brief).expect("upsert brief");
    index.commit().expect("commit");

    let hits = index
        .search(&hybrid_query("md-blocks-spec", &["public"], 10))
        .expect("lexical search");
    assert_eq!(
        hits.first().map(|h| h.id.as_str()),
        Some("spec"),
        "the file itself must outrank a body mention of its name: {:?}",
        hits.iter().map(|h| (&h.id, h.score)).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// 6. HTTP round trip
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
    stream
        .write_all(head.as_bytes())
        .expect("write request head");
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

#[test]
fn http_round_trip_pending_attach_and_hybrid_search() {
    let temp = TempDir::new().expect("temp dir");
    let mut server = start_in_background(temp.path(), "127.0.0.1:0").expect("start server");
    let addr = server.addr();

    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections",
        Some(&json!({
            "name": "docs",
            "config": { "vector_lane": { "dim": DIM, "model_version": MODEL } }
        })),
    );
    assert_eq!(status, 200, "{body}");

    let exact = doc(
        "exact",
        "Digitalisation funding programme",
        "the digitalisation funding programme pays for software",
        &["public"],
    );
    let paraphrase = doc(
        "paraphrase",
        "Zuschüsse für moderne Bürosoftware",
        "Beihilfen für die Umstellung auf digitale Abläufe",
        &["public"],
    );
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/upsert",
        Some(&json!({
            "docs": [
                { "id": "exact", "kind": "doc", "title": exact.title, "body": exact.body, "acl": ["public"] },
                { "id": "paraphrase", "kind": "doc", "title": paraphrase.title, "body": paraphrase.body, "acl": ["public"] },
            ]
        })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["upserted"], json!(2));

    let (status, body) = call(
        addr,
        "GET",
        "/v2/collections/docs/pending_embeddings?limit=10",
        None,
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["count"], json!(2), "{body}");
    let pending = body["pending"].as_array().expect("pending array").clone();
    let first = &pending[0];
    assert_eq!(first["id"], json!("exact"));
    assert_eq!(first["content_hash"], json!(exact.content_hash()));
    assert_eq!(first["title"], json!("Digitalisation funding programme"));

    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/attach_vectors",
        Some(&json!({
            "attachments": [
                {
                    "id": "exact",
                    "content_hash": exact.content_hash(),
                    "model_version": MODEL,
                    "vector": [1.0, 0.05, 0.0, 0.0],
                },
                {
                    "id": "paraphrase",
                    "content_hash": paraphrase.content_hash(),
                    "model_version": MODEL,
                    "vector": [1.0, 0.0, 0.0, 0.0],
                },
            ]
        })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["attached"], json!(2));
    assert!(body["vec_generation"].as_u64().expect("generation") >= 1);

    let (status, body) = call(addr, "GET", "/v2/collections/docs/pending_embeddings", None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["count"], json!(0), "{body}");

    // A stale completion is a 409, not a silent no-op.
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/attach_vectors",
        Some(&json!({
            "attachments": [{
                "id": "exact",
                "content_hash": "0".repeat(64),
                "model_version": MODEL,
                "vector": [1.0, 0.0, 0.0, 0.0],
            }]
        })),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(body["code"], json!("stale_vector_cas"));

    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/hybrid_search",
        Some(&json!({
            "lexical": {
                "text": "digitalisation funding programme",
                "principals": ["public"],
                "top_k": 10,
            },
            "query_vector": [1.0, 0.0, 0.0, 0.0],
        })),
    );
    assert_eq!(status, 200, "{body}");
    let hits = body["hits"].as_array().expect("hits array").clone();
    let ids: Vec<&str> = hits
        .iter()
        .map(|hit| hit["hit"]["id"].as_str().expect("hit id"))
        .collect();
    assert_eq!(ids.first().copied(), Some("exact"), "{ids:?}");
    assert!(ids.contains(&"paraphrase"), "{ids:?}");
    assert!(body["took_ms"].as_f64().expect("took_ms") >= 0.0);
    let paraphrase_hit = hits
        .iter()
        .find(|hit| hit["hit"]["id"] == json!("paraphrase"))
        .expect("paraphrase hit");
    assert!(paraphrase_hit["lexical_rank"].is_null());
    assert_eq!(paraphrase_hit["vector_rank"], json!(1));
    assert!(paraphrase_hit["rrf_score"].as_f64().expect("rrf") > 0.0);

    // Fail-closed over HTTP too.
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/docs/hybrid_search",
        Some(&json!({
            "lexical": { "text": "anything", "top_k": 5 },
            "query_vector": [1.0, 0.0, 0.0, 0.0],
        })),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("missing_principals"));

    let (status, body) = call(
        addr,
        "GET",
        "/v2/collections/docs/pending_embeddings?limit=0",
        None,
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("invalid_limit"));

    // A collection without a lane keeps the cheap answer instead of guessing.
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections",
        Some(&json!({ "name": "lexical" })),
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = call(
        addr,
        "GET",
        "/v2/collections/lexical/pending_embeddings",
        None,
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["count"], json!(0));
    let (status, body) = call(
        addr,
        "POST",
        "/v2/collections/lexical/hybrid_search",
        Some(&json!({
            "lexical": { "text": "anything", "principals": ["public"], "top_k": 5 },
            "query_vector": [1.0, 0.0, 0.0, 0.0],
        })),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("vector_lane_not_configured"));

    server.shutdown();
}
