//! Phase 2: `V2Index` integration tests (Tantivy-backed lexical retrieval).
//!
//! Coverage: German compound decompounding (both directions, rebuild vs
//! incremental), single-edit typo tolerance, exact-title and identifier
//! matching, columnar filters (including promoted metadata columns), the
//! authoritative ACL overlay (fail-closed and ACL-only fast path), native
//! group-by-parent diversity with explanations, manifest version gating, and
//! the full Phase 0 sync contract (incremental, no-op skip, ACL-only, snapshot
//! rebuild on compaction and backup restore) against a real `V2Index`.

use std::collections::{BTreeMap, HashSet};
use std::sync::Mutex;

use nvdb::sync::{
    SourceChange, SourceChangeOp, SourceConnector, SourceLineage, SourceRecord, SourceSnapshot,
    SyncError, SyncOutcome,
};
use nvdb::v2::{
    body_part_terms, decompose_german_compound, PromotedField, PromotedFieldType, PromotedFilter,
    TimeRange, V2CollectionConfig, V2Document, V2Error, V2Filters, V2Index, V2Query, V2SearchHit,
    CURRENT_ANALYZER_VERSION, CURRENT_SCHEMA_VERSION, GERMAN_LEXICON_VERSION,
};
use serde_json::{json, Value};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn groups(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| value.to_string()).collect()
}

fn doc(id: &str, title: &str, body: &str) -> V2Document {
    let mut document = V2Document::new(id, "doc");
    document.title = title.to_string();
    document.body = body.to_string();
    document.acl = groups(&["public"]);
    document
}

fn open_index(dir: &TempDir) -> V2Index {
    V2Index::create_or_open(dir.path().join("index"), V2CollectionConfig::default())
        .expect("create v2 index")
}

fn search(index: &V2Index, text: &str) -> Vec<V2SearchHit> {
    index
        .search(&V2Query::new(text).principals(["public"]).top_k(20))
        .expect("search")
}

fn ids(hits: &[V2SearchHit]) -> Vec<String> {
    hits.iter().map(|hit| hit.id.clone()).collect()
}

fn sorted_ids(hits: &[V2SearchHit]) -> Vec<String> {
    let mut ids = ids(hits);
    ids.sort();
    ids
}

fn has(hits: &[V2SearchHit], id: &str) -> bool {
    hits.iter().any(|hit| hit.id == id)
}

// ---------------------------------------------------------------------------
// Journal-backed source that carries real V2Documents.
// ---------------------------------------------------------------------------

struct SourceState {
    epoch: String,
    last_seq: u64,
    min_available_seq: u64,
    docs: BTreeMap<String, V2Document>,
    journal: Vec<SourceChange>,
}

struct FakeSource {
    state: Mutex<SourceState>,
}

impl FakeSource {
    fn new(epoch: &str) -> Self {
        Self {
            state: Mutex::new(SourceState {
                epoch: epoch.to_string(),
                last_seq: 0,
                min_available_seq: 0,
                docs: BTreeMap::new(),
                journal: Vec::new(),
            }),
        }
    }

    fn upsert(&self, document: V2Document) -> u64 {
        let mut state = self.state.lock().unwrap();
        apply_upsert(&mut state, document)
    }

    fn delete(&self, id: &str) -> u64 {
        let mut state = self.state.lock().unwrap();
        let seq = state.last_seq + 1;
        state.docs.remove(id);
        state.journal.push(SourceChange {
            seq,
            id: id.to_string(),
            op: SourceChangeOp::Delete,
        });
        state.last_seq = seq;
        seq
    }

    fn set_acl(&self, id: &str, acl: &[&str]) -> u64 {
        let mut state = self.state.lock().unwrap();
        let mut document = state.docs.get(id).cloned().expect("document exists");
        document.acl = groups(acl);
        apply_upsert(&mut state, document)
    }

    /// Re-emit the current record at a new sequence (an unchanged write).
    fn reemit(&self, id: &str) -> u64 {
        let mut state = self.state.lock().unwrap();
        let document = state.docs.get(id).cloned().expect("document exists");
        let seq = state.last_seq + 1;
        state.journal.push(SourceChange {
            seq,
            id: id.to_string(),
            op: SourceChangeOp::Upsert(record_for(&document)),
        });
        state.last_seq = seq;
        seq
    }

    /// Simulate journal compaction: nothing before `min_available_seq` survives.
    fn compact(&self, min_available_seq: u64) {
        let mut state = self.state.lock().unwrap();
        state.min_available_seq = min_available_seq;
        state
            .journal
            .retain(|change| change.seq >= min_available_seq);
    }

    /// Simulate a restored backup taken at `target_seq`.
    fn restore_to(&self, target_seq: u64) {
        let mut state = self.state.lock().unwrap();
        let known_documents = state.docs.clone();
        let mut docs = BTreeMap::new();
        let mut journal = Vec::new();
        for change in state
            .journal
            .iter()
            .filter(|change| change.seq <= target_seq)
        {
            journal.push(change.clone());
            match &change.op {
                SourceChangeOp::Upsert(record) => {
                    if let Some(document) = known_documents.get(&record.id) {
                        docs.insert(record.id.clone(), document.clone());
                    }
                }
                SourceChangeOp::Delete => {
                    docs.remove(&change.id);
                }
            }
        }
        state.docs = docs;
        state.journal = journal;
        state.last_seq = target_seq;
        state.min_available_seq = 0;
    }

    fn document(&self, id: &str) -> Option<V2Document> {
        self.state.lock().unwrap().docs.get(id).cloned()
    }

    fn head(&self) -> u64 {
        self.state.lock().unwrap().last_seq
    }

    fn mapper(&self) -> impl Fn(&SourceRecord) -> V2Document + '_ {
        move |record| {
            self.document(&record.id)
                .unwrap_or_else(|| panic!("source has no document '{}'", record.id))
        }
    }
}

fn record_for(document: &V2Document) -> SourceRecord {
    SourceRecord::new(
        document.to_indexed_envelope(CURRENT_SCHEMA_VERSION, CURRENT_ANALYZER_VERSION),
    )
}

fn apply_upsert(state: &mut SourceState, document: V2Document) -> u64 {
    let seq = state.last_seq + 1;
    let record = record_for(&document);
    state.docs.insert(document.id.clone(), document);
    state.journal.push(SourceChange {
        seq,
        id: record.id.clone(),
        op: SourceChangeOp::Upsert(record),
    });
    state.last_seq = seq;
    seq
}

impl SourceConnector for FakeSource {
    fn lineage(&self) -> Result<SourceLineage, SyncError> {
        let state = self.state.lock().unwrap();
        Ok(SourceLineage {
            source_epoch_uuid: state.epoch.clone(),
            last_seq: state.last_seq,
            min_available_seq: state.min_available_seq,
        })
    }

    fn snapshot_at(&self) -> Result<SourceSnapshot, SyncError> {
        let state = self.state.lock().unwrap();
        Ok(SourceSnapshot {
            lineage: SourceLineage {
                source_epoch_uuid: state.epoch.clone(),
                last_seq: state.last_seq,
                min_available_seq: state.min_available_seq,
            },
            records: state.docs.values().map(record_for).collect(),
        })
    }

    fn changes_since(&self, seq: u64) -> Result<Vec<SourceChange>, SyncError> {
        let state = self.state.lock().unwrap();
        if seq + 1 < state.min_available_seq {
            return Err(SyncError::CursorExpired {
                requested_seq: seq,
                min_available_seq: state.min_available_seq,
                last_seq: state.last_seq,
            });
        }
        if seq > state.last_seq {
            return Err(SyncError::CursorAhead {
                synced_seq: seq,
                source_last_seq: state.last_seq,
            });
        }
        Ok(state
            .journal
            .iter()
            .filter(|change| change.seq > seq)
            .cloned()
            .collect())
    }

    fn get_doc(&self, id: &str) -> Result<Option<SourceRecord>, SyncError> {
        let state = self.state.lock().unwrap();
        Ok(state.docs.get(id).map(record_for))
    }

    fn resolve_principals(&self, _user_id: &str) -> Result<Vec<String>, SyncError> {
        Ok(Vec::new())
    }
}

// ---------------------------------------------------------------------------
// 1. German compounding, both directions, rebuild vs incremental.
// ---------------------------------------------------------------------------

fn compound_corpus() -> Vec<V2Document> {
    vec![
        doc(
            "foerder_compound",
            "Förderprogramm Digitalisierung",
            "Das Förderprogramm fördert digitale Projekte.",
        ),
        doc(
            "foerder_parts",
            "Förderung und Programm",
            "Förderung für Programme und Projekte.",
        ),
        doc(
            "datenschutz_compound",
            "Datenschutzrichtlinie",
            "Die Datenschutzrichtlinie regelt den Umgang mit Daten.",
        ),
        doc(
            "datenschutz_parts",
            "Daten und Schutz",
            "Daten und Schutz werden getrennt behandelt.",
        ),
        doc(
            "forschung_compound",
            "Forschungszulage",
            "Die Forschungszulage gilt für Institute.",
        ),
        doc(
            "forschung_parts",
            "Forschung und Zulage",
            "Forschung und Zulage getrennt betrachtet.",
        ),
        doc(
            "handbuch_compound",
            "Anwendungshandbuch",
            "Das Anwendungshandbuch beschreibt die Anwendung.",
        ),
        doc(
            "handbuch_parts",
            "Anwendung und Handbuch",
            "Anwendung und Handbuch getrennt.",
        ),
    ]
}

#[test]
fn german_compounds_match_in_both_directions() {
    // Structural guarantee first: the frozen splitter produces the morphemes
    // the retrieval assertions rely on.
    let expected: [(&str, &[&str]); 4] = [
        ("Förderprogramm", &["foerder", "programm"]),
        ("Datenschutzrichtlinie", &["datenschutz", "richtlinie"]),
        ("Forschungszulage", &["forschung", "zulage"]),
        ("Anwendungshandbuch", &["anwendung", "handbuch"]),
    ];
    for (word, parts) in expected {
        let decomposed = decompose_german_compound(word);
        for part in parts {
            assert!(
                decomposed.iter().any(|candidate| candidate == part),
                "{word} decomposed to {decomposed:?}, missing {part}"
            );
        }
    }

    let temp = TempDir::new().unwrap();
    let mut index = open_index(&temp);
    index.upsert_batch(compound_corpus()).unwrap();
    index.commit().unwrap();

    // compound query -> component document
    for (query, target) in [
        ("Förderprogramm", "foerder_parts"),
        ("Datenschutz", "datenschutz_parts"),
        ("Forschungszulage", "forschung_parts"),
        ("Anwendungshandbuch", "handbuch_parts"),
    ] {
        let hits = search(&index, query);
        assert!(
            has(&hits, target),
            "compound query '{query}' must reach component document '{target}': {:?}",
            ids(&hits)
        );
    }

    // component query -> compound document
    for (query, target) in [
        ("Förderung", "foerder_compound"),
        ("Schutz", "datenschutz_compound"),
        ("Zulage", "forschung_compound"),
        ("Handbuch", "handbuch_compound"),
    ] {
        let hits = search(&index, query);
        assert!(
            has(&hits, target),
            "component query '{query}' must reach compound document '{target}': {:?}",
            ids(&hits)
        );
    }

    // The full compound title still outranks the split document.
    let hits = search(&index, "Datenschutzrichtlinie");
    assert_eq!(hits[0].id, "datenschutz_compound", "{:?}", ids(&hits));

    // Every part the indexer writes is a real indexed term.
    let parts = body_part_terms("Datenschutzrichtlinie");
    assert!(parts.iter().any(|part| part == "datenschutz"));
    assert!(parts.iter().any(|part| part == "richtlinie"));
}

#[test]
fn compound_ranking_is_deterministic_across_rebuild_and_incremental() {
    let queries = [
        "Förderprogramm",
        "Förderung",
        "Datenschutzrichtlinie",
        "Datenschutz",
        "Forschungszulage",
        "Anwendungshandbuch",
    ];

    // Incremental path: direct upserts.
    let incremental_dir = TempDir::new().unwrap();
    let mut incremental = open_index(&incremental_dir);
    incremental.upsert_batch(compound_corpus()).unwrap();
    incremental.commit().unwrap();

    // Rebuild path: a clean snapshot rebuild through Phase 0 sync.
    let rebuild_dir = TempDir::new().unwrap();
    let mut rebuilt = open_index(&rebuild_dir);
    let source = FakeSource::new("epoch-compounds");
    for document in compound_corpus() {
        source.upsert(document);
    }
    let outcome = rebuilt.sync_from(&source, source.mapper()).unwrap();
    assert_eq!(
        outcome,
        SyncOutcome::Rebuilt {
            watermark_seq: source.head(),
            docs_indexed: 8,
        }
    );

    for query in queries {
        assert_eq!(
            ids(&search(&incremental, query)),
            ids(&search(&rebuilt, query)),
            "ranking for '{query}' must not depend on the write path"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. One-edit typo tolerance without breaking exact-title ranking.
// ---------------------------------------------------------------------------

#[test]
fn one_edit_typos_are_tolerated_without_breaking_exact_title_ranking() {
    let temp = TempDir::new().unwrap();
    let mut index = open_index(&temp);

    let exact = doc(
        "exact_title",
        "Consciousness Record",
        "AI consciousness record of the session",
    );
    let typo = doc(
        "typo_title",
        "conciousness record",
        "AI conciousness record written by hand",
    );
    index.upsert_batch(vec![exact, typo]).unwrap();
    index.commit().unwrap();

    // Correctly spelled query: the exact-title document ranks first.
    let hits = search(&index, "AI Consciousness");
    assert!(has(&hits, "exact_title"), "{:?}", ids(&hits));
    assert_eq!(hits[0].id, "exact_title", "{:?}", ids(&hits));

    // Misspelled query: fuzzy expansion recovers the correctly spelled doc.
    let fuzzy = index
        .search(
            &V2Query::new("AI Conciousness")
                .principals(["public"])
                .fuzzy_edits(1)
                .top_k(20),
        )
        .unwrap();
    assert!(has(&fuzzy, "exact_title"), "{:?}", ids(&fuzzy));
    assert!(has(&fuzzy, "typo_title"), "{:?}", ids(&fuzzy));

    // Multi-term misspelling keeps working.
    let fuzzy_multi = index
        .search(
            &V2Query::new("conciousness record")
                .principals(["public"])
                .fuzzy_edits(1)
                .top_k(20),
        )
        .unwrap();
    assert!(has(&fuzzy_multi, "exact_title"), "{:?}", ids(&fuzzy_multi));

    // Without fuzzy expansion the misspelling must not reach the exact doc.
    let strict = index
        .search(
            &V2Query::new("Conciousness")
                .principals(["public"])
                .top_k(20),
        )
        .unwrap();
    assert_eq!(
        ids(&strict),
        vec!["typo_title".to_string()],
        "a misspelled single term must not match the correctly spelled document"
    );

    // The same single term with one edit recovers the exact-title document.
    let fuzzy_single = index
        .search(
            &V2Query::new("Conciousness")
                .principals(["public"])
                .fuzzy_edits(1)
                .top_k(20),
        )
        .unwrap();
    assert!(
        has(&fuzzy_single, "exact_title"),
        "{:?}",
        ids(&fuzzy_single)
    );
}

// ---------------------------------------------------------------------------
// 3. Exact-title and identifier (path) matching.
// ---------------------------------------------------------------------------

#[test]
fn exact_title_and_path_identifiers_both_match() {
    let temp = TempDir::new().unwrap();
    let mut index = open_index(&temp);

    let mut by_title = doc("by_title", "Quarterly Report", "some body text");
    by_title.path = "records/2024/q1.md".to_string();
    by_title.ext = Some(".md".to_string());

    let mut by_path = doc("by_path", "Notes", "unrelated body");
    by_path.path = "docs/q1/quarterly-report.pdf".to_string();
    by_path.name = "quarterly-report.pdf".to_string();
    by_path.ext = Some("pdf".to_string());

    let mut neither = doc("neither", "Something Else", "nothing relevant here");
    neither.path = "docs/other.md".to_string();

    index
        .upsert_batch(vec![by_title, by_path, neither])
        .unwrap();
    index.commit().unwrap();

    let hits = search(&index, "Quarterly Report");
    assert!(has(&hits, "by_title"), "{:?}", ids(&hits));
    assert!(has(&hits, "by_path"), "{:?}", ids(&hits));
    assert!(!has(&hits, "neither"), "{:?}", ids(&hits));
    assert_eq!(
        hits[0].id,
        "by_title",
        "exact title must win: {:?}",
        ids(&hits)
    );

    // Identifier matching works on its own (no title help).
    let identifier = search(&index, "quarterly-report.pdf");
    assert!(has(&identifier, "by_path"), "{:?}", ids(&identifier));

    // Exact-title search through a path-independent columnar filter.
    let filtered = index
        .search(
            &V2Query::new("Quarterly Report")
                .principals(["public"])
                .filters(V2Filters::none().ext("pdf")),
        )
        .unwrap();
    assert_eq!(ids(&filtered), vec!["by_path".to_string()]);
}

#[test]
fn exact_body_directive_outranks_partial_title_and_path_matches() {
    let temp = TempDir::new().unwrap();
    let mut index = open_index(&temp);

    let mut author_bio = doc(
        "author_bio",
        "David A. Renelt",
        "Author profile for David A. Renelt, essays and biography.",
    );
    author_bio.path = "blog/authors/david-a-renelt.md".to_string();
    author_bio.name = "david-a-renelt.md".to_string();

    let mut handover = doc(
        "handover",
        "nSpeech Dialogue Mode Handover",
        "Example speaker block:\n<!-- mb:block preset=narration:david-a-renelt -->\nSpoken text.\n<!-- mb:/block -->",
    );
    handover.path = "scratch/nspeech-dialogue-mode-handover.md".to_string();
    handover.name = "nspeech-dialogue-mode-handover.md".to_string();

    index.upsert_batch(vec![author_bio, handover]).unwrap();
    index.commit().unwrap();

    let hits = index
        .search(
            &V2Query::new("preset=narration:david-a-renelt")
                .principals(["public"])
                .fuzzy_edits(1)
                .top_k(5),
        )
        .unwrap();
    assert_eq!(
        hits.first().map(|h| h.id.as_str()),
        Some("handover"),
        "verbatim directive match in body must outrank partial title/path hits: {:?}",
        ids(&hits)
    );
}

// ---------------------------------------------------------------------------
// 4. Columnar filters.
// ---------------------------------------------------------------------------

fn promoted_config() -> V2CollectionConfig {
    V2CollectionConfig::with_promoted(vec![
        PromotedField::new("priority", PromotedFieldType::I64),
        PromotedField::new("lang", PromotedFieldType::String),
        PromotedField::new("archived", PromotedFieldType::Bool),
        PromotedField::new("score", PromotedFieldType::F64),
    ])
}

fn promoted_doc(
    id: &str,
    folder: &str,
    ext: &str,
    kind: &str,
    created: i64,
    metadata: Value,
) -> V2Document {
    let mut document = doc(id, "Monthly report", "the report body");
    document.folder = Some(folder.to_string());
    document.ext = Some(ext.to_string());
    document.kind = kind.to_string();
    document.created_at = Some(created);
    document.updated_at = Some(created + 10);
    document.metadata = metadata;
    document
}

#[test]
fn columnar_filters_narrow_the_candidate_set() {
    let temp = TempDir::new().unwrap();
    let mut index = V2Index::create_or_open(temp.path().join("index"), promoted_config()).unwrap();

    index
        .upsert_batch(vec![
            promoted_doc(
                "finance_md",
                "finance",
                ".md",
                "doc",
                100,
                json!({"priority": 3, "lang": "de", "archived": false, "score": 0.5}),
            ),
            promoted_doc(
                "finance_chunk",
                "finance",
                "md",
                "chunk",
                200,
                json!({"priority": 1, "lang": "en", "archived": false, "score": 1.5}),
            ),
            promoted_doc(
                "legal_pdf",
                "legal",
                ".pdf",
                "doc",
                300,
                json!({"priority": 5, "lang": "de", "archived": true, "score": 2.5}),
            ),
        ])
        .unwrap();
    index.commit().unwrap();

    let run = |filters: V2Filters| -> Vec<String> {
        let hits = index
            .search(
                &V2Query::new("report")
                    .principals(["public"])
                    .filters(filters),
            )
            .expect("filtered search");
        sorted_ids(&hits)
    };

    // ext is normalised on both sides: `.md`, `md` and `MD` are the same.
    assert_eq!(
        run(V2Filters::none().ext(".md")),
        vec!["finance_chunk".to_string(), "finance_md".to_string()]
    );
    assert_eq!(run(V2Filters::none().ext("MD")).len(), 2);
    assert_eq!(
        run(V2Filters::none().ext("pdf")),
        vec!["legal_pdf".to_string()]
    );

    assert_eq!(
        run(V2Filters::none().kind("doc")),
        vec!["finance_md".to_string(), "legal_pdf".to_string()]
    );
    assert_eq!(
        run(V2Filters::none().folder("finance")),
        vec!["finance_chunk".to_string(), "finance_md".to_string()]
    );
    assert_eq!(
        run(V2Filters::none().created_at(TimeRange::between(150, 250))),
        vec!["finance_chunk".to_string()]
    );
    assert_eq!(
        run(V2Filters::none().updated_at(TimeRange::from(305))),
        vec!["legal_pdf".to_string()]
    );

    // Promoted metadata columns.
    assert_eq!(
        run(V2Filters::none().promoted(PromotedFilter::gte("priority", 3i64))),
        vec!["finance_md".to_string(), "legal_pdf".to_string()]
    );
    assert_eq!(
        run(V2Filters::none().promoted(PromotedFilter::eq("lang", "de"))),
        vec!["finance_md".to_string(), "legal_pdf".to_string()]
    );
    assert_eq!(
        run(V2Filters::none().promoted(PromotedFilter::eq("archived", true))),
        vec!["legal_pdf".to_string()]
    );
    assert_eq!(
        run(V2Filters::none().promoted(PromotedFilter::lte("score", 1.5f64))),
        vec!["finance_chunk".to_string(), "finance_md".to_string()]
    );
    assert_eq!(
        run(V2Filters::none()
            .kind("doc")
            .promoted(PromotedFilter::eq("lang", "de"))),
        vec!["finance_md".to_string(), "legal_pdf".to_string()]
    );

    // Fail-loud misuse.
    let unknown = index
        .search(
            &V2Query::new("report")
                .principals(["public"])
                .filters(V2Filters::none().promoted(PromotedFilter::eq("nope", "x"))),
        )
        .unwrap_err();
    assert!(
        matches!(unknown, V2Error::InvalidQuery(_)),
        "expected InvalidQuery, got {unknown:?}"
    );

    let mismatched = index
        .search(
            &V2Query::new("report")
                .principals(["public"])
                .filters(V2Filters::none().promoted(PromotedFilter::eq("lang", 5i64))),
        )
        .unwrap_err();
    assert!(
        matches!(mismatched, V2Error::PromotedFieldType { .. }),
        "expected PromotedFieldType, got {mismatched:?}"
    );

    let unsupported = index
        .search(
            &V2Query::new("report")
                .principals(["public"])
                .filters(V2Filters::none().promoted(PromotedFilter::gte("lang", "de"))),
        )
        .unwrap_err();
    assert!(
        matches!(unsupported, V2Error::InvalidQuery(_)),
        "expected InvalidQuery, got {unsupported:?}"
    );
}

// ---------------------------------------------------------------------------
// 5. ACL overlay: fail-closed, ACL-only fast path, immediate revocation.
// ---------------------------------------------------------------------------

#[test]
fn acl_overlay_is_authoritative_and_updates_without_reindexing() {
    let temp = TempDir::new().unwrap();
    let mut index = open_index(&temp);

    let mut secret = doc("secret", "Salary report", "engineering salary report");
    secret.acl = groups(&["eng"]);
    let open = doc("open", "Expense report", "public expense report");
    index.upsert_batch(vec![secret, open]).unwrap();
    index.commit().unwrap();

    let visible = |index: &V2Index, principals: &[&str]| -> Vec<String> {
        let hits = index
            .search(
                &V2Query::new("report")
                    .principals(groups(principals))
                    .top_k(20),
            )
            .expect("search");
        sorted_ids(&hits)
    };

    assert_eq!(visible(&index, &["public"]), vec!["open".to_string()]);
    assert_eq!(
        visible(&index, &["eng"]),
        vec!["secret".to_string()],
        "the eng group must see the secret document"
    );

    // Empty principals fail closed, never "match everything".
    let error = index.search(&V2Query::new("report")).unwrap_err();
    assert!(
        matches!(error, V2Error::MissingPrincipals),
        "expected MissingPrincipals, got {error:?}"
    );

    let before_docs = index.reader_doc_count();
    let before_generation = index.acl_overlay().generation();

    // Grant access without touching Tantivy and without a commit.
    let mut granted = doc("secret", "Salary report", "engineering salary report");
    granted.acl = groups(&["public"]);
    index
        .update_acl_only("secret", granted.acl.clone())
        .unwrap();

    assert!(index.acl_overlay().generation() > before_generation);
    assert_eq!(
        index.reader_doc_count(),
        before_docs,
        "no re-index may happen"
    );
    assert_eq!(
        visible(&index, &["public"]),
        vec!["open".to_string(), "secret".to_string()],
        "the ACL-only fast path must be visible immediately"
    );

    // Revoke again: the eng group loses the document immediately.
    index.update_acl_only("secret", groups(&["hr"])).unwrap();
    assert_eq!(visible(&index, &["public"]), vec!["open".to_string()]);
    assert!(visible(&index, &["eng"]).is_empty());
    assert_eq!(visible(&index, &["hr"]), vec!["secret".to_string()]);

    // An empty ACL hides the document from everyone.
    index.update_acl_only("secret", Vec::new()).unwrap();
    assert!(visible(&index, &["hr"]).is_empty());
    assert_eq!(index.acl_overlay().doc_count(), 2);

    // The generation and the overlay survive a reopen.
    let generation = index.acl_overlay().generation();
    drop(index);
    let reopened =
        V2Index::create_or_open(temp.path().join("index"), V2CollectionConfig::default()).unwrap();
    assert_eq!(reopened.acl_overlay().generation(), generation);
    assert!(!reopened
        .acl_overlay()
        .is_visible("secret", &HashSet::from(["hr"])));
    assert!(reopened
        .acl_overlay()
        .is_visible("open", &HashSet::from(["public"])));
}

// ---------------------------------------------------------------------------
// 6. Native group-by-parent diversity and explanations.
// ---------------------------------------------------------------------------

#[test]
fn group_by_parent_caps_hits_per_parent_and_explains_them() {
    let temp = TempDir::new().unwrap();
    let mut index = open_index(&temp);

    let mut documents = Vec::new();
    for chunk in 0..3 {
        let mut document = doc(
            &format!("alpha-{chunk}"),
            "Alpha retrieval",
            "alpha alpha alpha body",
        );
        document.parent_id = Some("parent-alpha".to_string());
        document.chunk_index = Some(chunk);
        documents.push(document);
    }
    for chunk in 0..2 {
        let mut document = doc(&format!("beta-{chunk}"), "Alpha notes", "alpha beta body");
        document.parent_id = Some("parent-beta".to_string());
        document.chunk_index = Some(chunk);
        documents.push(document);
    }
    let orphan = doc("orphan", "Alpha orphan", "alpha alone");
    documents.push(orphan);

    index.upsert_batch(documents).unwrap();
    index.commit().unwrap();

    let hits = index
        .search(
            &V2Query::new("alpha")
                .principals(["public"])
                .top_k(10)
                .group_by_parent(2)
                .explain(true),
        )
        .unwrap();

    let per_parent = |parent: &str| {
        hits.iter()
            .filter(|hit| hit.parent_id.as_deref() == Some(parent))
            .count()
    };
    assert_eq!(per_parent("parent-alpha"), 2, "{:?}", ids(&hits));
    assert_eq!(per_parent("parent-beta"), 2, "{:?}", ids(&hits));
    assert_eq!(hits.len(), 5, "2 + 2 + orphan: {:?}", ids(&hits));
    assert!(has(&hits, "orphan"), "unparented docs are never capped");

    let explain = hits[0].explain.as_ref().expect("explain requested");
    assert_eq!(explain.total_terms, 1);
    assert!(explain.coverage > 0.0);
    assert_eq!(explain.final_score, hits[0].score);
    assert_eq!(explain.bm25_score, hits[0].bm25_score);
    assert_eq!(explain.candidates_considered, 6);
    assert_eq!(explain.diversity_filtered, 1, "one alpha chunk was capped");

    // A zero cap is a config error, not "unlimited".
    let invalid = index
        .search(
            &V2Query::new("alpha")
                .principals(["public"])
                .group_by_parent(0),
        )
        .unwrap_err();
    assert!(matches!(invalid, V2Error::InvalidQuery(_)));
}

// ---------------------------------------------------------------------------
// 7. Manifest version gating and reopen durability.
// ---------------------------------------------------------------------------

#[test]
fn manifest_version_mismatch_fails_fast_on_reopen() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("index");
    {
        let mut index = V2Index::create_or_open(&path, V2CollectionConfig::default()).unwrap();
        index.upsert(doc("doc", "Persisted", "body")).unwrap();
        index.commit().unwrap();
    }

    let analyzer_bumped = V2CollectionConfig {
        analyzer_version: CURRENT_ANALYZER_VERSION + 1,
        ..V2CollectionConfig::default()
    };
    match V2Index::create_or_open(&path, analyzer_bumped) {
        Err(V2Error::VersionMismatch {
            field,
            expected,
            found,
        }) => {
            assert_eq!(field, "analyzer_version");
            assert_eq!(expected, CURRENT_ANALYZER_VERSION + 1);
            assert_eq!(found, CURRENT_ANALYZER_VERSION);
        }
        other => panic!("expected VersionMismatch, got {:?}", other.err()),
    }

    let schema_bumped = V2CollectionConfig {
        schema_version: CURRENT_SCHEMA_VERSION + 1,
        ..V2CollectionConfig::default()
    };
    assert!(matches!(
        V2Index::create_or_open(&path, schema_bumped),
        Err(V2Error::VersionMismatch {
            field: "schema_version",
            ..
        })
    ));

    let lexicon_bumped = V2CollectionConfig {
        lexicon_version: GERMAN_LEXICON_VERSION + 1,
        ..V2CollectionConfig::default()
    };
    assert!(matches!(
        V2Index::create_or_open(&path, lexicon_bumped),
        Err(V2Error::VersionMismatch {
            field: "lexicon_version",
            ..
        })
    ));

    // A different promoted-field contract is a schema mismatch, not a crash.
    let promoted = V2CollectionConfig {
        promoted_fields: vec![PromotedField::new("priority", PromotedFieldType::I64)],
        ..V2CollectionConfig::default()
    };
    assert!(matches!(
        V2Index::create_or_open(&path, promoted),
        Err(V2Error::SchemaMismatch(_))
    ));

    // The exact config reopens and still serves the committed document.
    let index = V2Index::create_or_open(&path, V2CollectionConfig::default()).unwrap();
    assert_eq!(index.doc_count(), 1);
    assert_eq!(ids(&search(&index, "Persisted")), vec!["doc".to_string()]);
}

// ---------------------------------------------------------------------------
// 8. Full Phase 0 sync integration against the real index.
// ---------------------------------------------------------------------------

#[test]
fn sync_applies_incremental_batches_and_rebuilds_after_compaction() {
    let temp = TempDir::new().unwrap();
    let mut index = open_index(&temp);

    let source = FakeSource::new("epoch-sync");
    let mut first = doc("d1", "Alpha report", "alpha body");
    first.acl = groups(&["eng"]);
    source.upsert(first);
    let mut second = doc("d2", "Beta report", "beta body");
    second.acl = groups(&["eng"]);
    source.upsert(second);

    // First sync: no watermark -> clean rebuild.
    assert_eq!(
        index.sync_from(&source, source.mapper()).unwrap(),
        SyncOutcome::Rebuilt {
            watermark_seq: 2,
            docs_indexed: 2,
        }
    );
    assert_eq!(index.manifest().watermark.as_ref().unwrap().synced_seq, 2);
    assert!(index.integrity_check(&source).unwrap().is_clean());

    // Incremental batch: upsert + delete + ACL-only change.
    let mut third = doc("d3", "Gamma report", "gamma body");
    third.acl = groups(&["eng"]);
    source.upsert(third);
    source.delete("d2");
    source.set_acl("d1", &["hr"]);
    let outcome = index.sync_from(&source, source.mapper()).unwrap();
    assert_eq!(
        outcome,
        SyncOutcome::Incremental {
            from_seq: 2,
            to_seq: 5,
            applied: 3,
            skipped: 0,
        }
    );
    assert_eq!(index.doc_count(), 2, "d1 and d3 are live, d2 is gone");
    assert_eq!(index.reader_doc_count(), 2);

    // The ACL-only change is already authoritative.
    let hr = index
        .search(&V2Query::new("report").principals(["hr"]).top_k(10))
        .unwrap();
    assert_eq!(ids(&hr), vec!["d1".to_string()]);
    let eng = index
        .search(&V2Query::new("report").principals(["eng"]).top_k(10))
        .unwrap();
    assert_eq!(ids(&eng), vec!["d3".to_string()]);
    assert!(index.integrity_check(&source).unwrap().is_clean());

    // A replayed no-op record is skipped by envelope hash.
    source.reemit("d1");
    assert_eq!(
        index.sync_from(&source, source.mapper()).unwrap(),
        SyncOutcome::Incremental {
            from_seq: 5,
            to_seq: 6,
            applied: 0,
            skipped: 1,
        }
    );
    assert_eq!(
        index.sync_from(&source, source.mapper()).unwrap(),
        SyncOutcome::UpToDate { watermark_seq: 6 }
    );

    // Journal compaction: the cursor expires, the index rebuilds cleanly.
    source.upsert(doc("d4", "Delta report", "delta body"));
    source.upsert(doc("d5", "Epsilon report", "epsilon body"));
    let head = source.head();
    source.compact(head);
    assert_eq!(
        index.sync_from(&source, source.mapper()).unwrap(),
        SyncOutcome::Rebuilt {
            watermark_seq: head,
            docs_indexed: 4,
        },
        "d1, d3, d4 and d5 are live after the compaction rebuild"
    );
    assert!(index.integrity_check(&source).unwrap().is_clean());

    // A degraded index refuses reads and heals on the next rebuild.
    index.mark_degraded("simulated apply failure").unwrap();
    let degraded = index
        .search(&V2Query::new("report").principals(["eng"]).top_k(10))
        .unwrap_err();
    assert!(
        matches!(degraded, V2Error::IndexDegraded { .. }),
        "expected IndexDegraded, got {degraded:?}"
    );
    let outcome = index.sync_from(&source, source.mapper()).unwrap();
    assert!(matches!(outcome, SyncOutcome::Rebuilt { .. }));
    assert!(!index
        .search(&V2Query::new("report").principals(["eng"]).top_k(10))
        .unwrap()
        .is_empty());

    // The watermark, the overlay and the documents survive a reopen.
    let watermark = index.manifest().watermark.clone();
    drop(index);
    let reopened =
        V2Index::create_or_open(temp.path().join("index"), V2CollectionConfig::default()).unwrap();
    assert_eq!(reopened.manifest().watermark, watermark);
    assert!(reopened.integrity_check(&source).unwrap().is_clean());
    assert!(!reopened
        .search(&V2Query::new("report").principals(["eng", "hr"]).top_k(10))
        .unwrap()
        .is_empty());
}

#[test]
fn sync_rebuilds_after_a_restored_backup() {
    let temp = TempDir::new().unwrap();
    let mut index = open_index(&temp);

    let source = FakeSource::new("epoch-restore");
    source.upsert(doc("d1", "Alpha report", "alpha body"));
    source.upsert(doc("d2", "Beta report", "beta body"));
    source.upsert(doc("d3", "Gamma report", "gamma body"));
    assert_eq!(
        index.sync_from(&source, source.mapper()).unwrap(),
        SyncOutcome::Rebuilt {
            watermark_seq: 3,
            docs_indexed: 3,
        }
    );

    // Restore a backup taken at seq 1: d2 and d3 never existed.
    source.restore_to(1);
    let outcome = index.sync_from(&source, source.mapper()).unwrap();
    assert_eq!(
        outcome,
        SyncOutcome::Rebuilt {
            watermark_seq: 1,
            docs_indexed: 1,
        },
        "watermark ahead of the source head must force a snapshot rebuild"
    );

    let hits = index
        .search(&V2Query::new("report").principals(["public"]).top_k(10))
        .unwrap();
    assert_eq!(ids(&hits), vec!["d1".to_string()]);
    assert!(index.integrity_check(&source).unwrap().is_clean());
}

#[test]
fn source_lineage_is_reported_from_the_watermark() {
    let temp = TempDir::new().unwrap();
    let mut index = open_index(&temp);
    assert!(index.source_lineage().is_none());

    let source = FakeSource::new("epoch-lineage");
    source.upsert(doc("d1", "Alpha", "alpha"));
    index.sync_from(&source, source.mapper()).unwrap();

    let lineage = index.source_lineage().expect("lineage after sync");
    assert_eq!(lineage.source_epoch_uuid, "epoch-lineage");
    assert_eq!(lineage.last_seq, 1);
}
