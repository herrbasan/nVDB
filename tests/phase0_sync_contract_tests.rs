//! Phase 0 sync, snapshot, ACL & revision contract tests.
//!
//! Exit gate: every crash, backup-restore, compaction-cursor-expiry,
//! ACL-revocation and concurrent-rebuild scenario must pass before the index
//! engine is written on top of this contract.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use nvdb::sync::{
    compute_content_hash, compute_envelope_hash, FaultInjection, IndexedEnvelope, SourceChange,
    SourceChangeOp, SourceConnector, SourceLineage, SourceRecord, SourceSnapshot, SyncCoordinator,
    SyncError, SyncOutcome, VectorAttachment,
};
use serde_json::{json, Value};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Test source: a journal-backed system of record with a lineage epoch.
// ---------------------------------------------------------------------------

struct SourceState {
    epoch: String,
    last_seq: u64,
    min_available_seq: u64,
    docs: BTreeMap<String, SourceRecord>,
    journal: Vec<SourceChange>,
    principals: BTreeMap<String, Vec<String>>,
}

struct FakeSource {
    state: Mutex<SourceState>,
    snapshot_calls: AtomicUsize,
    after_snapshot: Mutex<Option<SnapshotHook>>,
}

type SnapshotHook = Box<dyn FnOnce(&mut SourceState) + Send>;

impl FakeSource {
    fn new(epoch: &str) -> Self {
        Self {
            state: Mutex::new(SourceState {
                epoch: epoch.to_string(),
                last_seq: 0,
                min_available_seq: 0,
                docs: BTreeMap::new(),
                journal: Vec::new(),
                principals: BTreeMap::new(),
            }),
            snapshot_calls: AtomicUsize::new(0),
            after_snapshot: Mutex::new(None),
        }
    }

    fn upsert(&self, id: &str, text: &str, acl: &[&str], metadata: Value) -> u64 {
        let mut state = self.state.lock().unwrap();
        apply_upsert(&mut state, id, text, acl, metadata)
    }

    fn delete(&self, id: &str) -> u64 {
        let mut state = self.state.lock().unwrap();
        apply_delete(&mut state, id)
    }

    fn set_acl(&self, id: &str, acl: &[&str]) -> u64 {
        let mut state = self.state.lock().unwrap();
        apply_set_acl(&mut state, id, acl)
    }

    fn set_metadata(&self, id: &str, metadata: Value) -> u64 {
        let mut state = self.state.lock().unwrap();
        apply_set_metadata(&mut state, id, metadata)
    }

    /// Re-emit the current record at a new sequence (an unchanged write).
    fn reemit(&self, id: &str) -> u64 {
        let mut state = self.state.lock().unwrap();
        let record = state.docs.get(id).cloned().expect("record exists");
        let seq = state.last_seq + 1;
        state.journal.push(SourceChange {
            seq,
            id: id.to_string(),
            op: SourceChangeOp::Upsert(record),
        });
        state.last_seq = seq;
        seq
    }

    fn set_principal(&self, user: &str, groups: &[&str]) {
        let mut state = self.state.lock().unwrap();
        state.principals.insert(
            user.to_string(),
            groups.iter().map(|group| group.to_string()).collect(),
        );
    }

    /// Simulate journal compaction: everything before `min_available_seq` is gone.
    fn compact(&self, min_available_seq: u64) {
        let mut state = self.state.lock().unwrap();
        state.min_available_seq = min_available_seq;
        state.journal.retain(|change| change.seq >= min_available_seq);
    }

    /// Simulate restoring a backup taken at `target_seq`: the journal tail and
    /// every later document disappear.
    fn restore_to(&self, target_seq: u64) {
        let mut state = self.state.lock().unwrap();
        let mut docs = BTreeMap::new();
        let mut journal = Vec::new();
        for change in state.journal.iter().filter(|change| change.seq <= target_seq) {
            journal.push(change.clone());
            match &change.op {
                SourceChangeOp::Upsert(record) => {
                    docs.insert(record.id.clone(), record.clone());
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

    /// Simulate the source being replaced by a different lineage.
    fn replace_epoch(&self, epoch: &str) {
        let mut state = self.state.lock().unwrap();
        state.epoch = epoch.to_string();
    }

    /// Run `hook` after the next `snapshot_at()` has captured its view.
    fn on_next_snapshot(&self, hook: impl FnOnce(&mut SourceState) + Send + 'static) {
        *self.after_snapshot.lock().unwrap() = Some(Box::new(hook));
    }

    fn snapshot_calls(&self) -> usize {
        self.snapshot_calls.load(Ordering::SeqCst)
    }

    fn live_ids(&self) -> Vec<String> {
        let state = self.state.lock().unwrap();
        state.docs.keys().cloned().collect()
    }
}

fn apply_upsert(
    state: &mut SourceState,
    id: &str,
    text: &str,
    acl: &[&str],
    metadata: Value,
) -> u64 {
    let seq = state.last_seq + 1;
    let envelope = IndexedEnvelope {
        id: id.to_string(),
        parent_id: None,
        chunk_index: None,
        kind: "doc".to_string(),
        indexable_text: text.to_string(),
        acl: acl.iter().map(|group| group.to_string()).collect(),
        metadata,
        schema_version: 1,
        analyzer_version: 1,
    };
    let record = SourceRecord::new(envelope);
    state.docs.insert(id.to_string(), record.clone());
    state.journal.push(SourceChange {
        seq,
        id: id.to_string(),
        op: SourceChangeOp::Upsert(record),
    });
    state.last_seq = seq;
    seq
}

fn apply_delete(state: &mut SourceState, id: &str) -> u64 {
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

fn apply_set_acl(state: &mut SourceState, id: &str, acl: &[&str]) -> u64 {
    let existing = state.docs.get(id).cloned().expect("record exists");
    let text = existing.envelope.indexable_text.clone();
    let metadata = existing.envelope.metadata.clone();
    apply_upsert(state, id, &text, acl, metadata)
}

fn apply_set_metadata(state: &mut SourceState, id: &str, metadata: Value) -> u64 {
    let existing = state.docs.get(id).cloned().expect("record exists");
    let text = existing.envelope.indexable_text.clone();
    let acl: Vec<String> = existing.envelope.acl.clone();
    let acl: Vec<&str> = acl.iter().map(String::as_str).collect();
    apply_upsert(state, id, &text, &acl, metadata)
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
        let snapshot = {
            let state = self.state.lock().unwrap();
            SourceSnapshot {
                lineage: SourceLineage {
                    source_epoch_uuid: state.epoch.clone(),
                    last_seq: state.last_seq,
                    min_available_seq: state.min_available_seq,
                },
                records: state.docs.values().cloned().collect(),
            }
        };
        self.snapshot_calls.fetch_add(1, Ordering::SeqCst);
        let hook = self.after_snapshot.lock().unwrap().take();
        if let Some(hook) = hook {
            let mut state = self.state.lock().unwrap();
            hook(&mut state);
        }
        Ok(snapshot)
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
        Ok(state.docs.get(id).cloned())
    }

    fn resolve_principals(&self, user_id: &str) -> Result<Vec<String>, SyncError> {
        let state = self.state.lock().unwrap();
        Ok(state.principals.get(user_id).cloned().unwrap_or_default())
    }
}

fn principals(groups: &[&str]) -> Vec<String> {
    groups.iter().map(|group| group.to_string()).collect()
}

// ---------------------------------------------------------------------------
// 1. Crash recovery: apply failure degradation + restart healing.
// ---------------------------------------------------------------------------

#[test]
fn apply_failure_degrades_index_and_heals_after_restart() {
    let temp = TempDir::new().unwrap();
    let watermark_path = temp.path().join("index").join("sync_watermark.json");

    let source = FakeSource::new("epoch-a");
    source.upsert("d1", "alpha", &["eng"], json!({"n": 1}));
    source.upsert("d2", "beta", &["eng"], json!({"n": 2}));

    let mut coordinator = SyncCoordinator::with_watermark_path(&watermark_path);
    assert!(matches!(
        coordinator.sync(&source).unwrap(),
        SyncOutcome::Rebuilt {
            watermark_seq: 2,
            docs_indexed: 2
        }
    ));

    // The source commits seq 3; the dual write into the index fails.
    source.upsert("d3", "gamma", &["eng"], json!({"n": 3}));
    coordinator.set_fault_injection(FaultInjection::FailApplyAtSeq(3));

    match coordinator.sync(&source).expect_err("apply must fail at seq 3") {
        SyncError::ApplyFailed { seq, .. } => assert_eq!(seq, 3),
        other => panic!("expected ApplyFailed, got {other:?}"),
    }

    let watermark = coordinator.watermark().expect("watermark committed");
    assert_eq!(watermark.synced_seq, 2, "must stop at the last contiguous seq");
    assert!(watermark.degraded);
    assert!(watermark.degraded_reason.is_some());
    assert!(coordinator.live_doc("d3").is_none(), "failed event not applied");

    // Queries fail loudly while degraded.
    let eng = principals(&["eng"]);
    match coordinator
        .query_visible_ids(&source, &eng)
        .expect_err("degraded index must refuse queries")
    {
        SyncError::IndexDegraded { .. } => {}
        other => panic!("expected IndexDegraded, got {other:?}"),
    }

    // The degradation is durable: a fresh process sees it.
    assert!(watermark_path.exists());
    let mut reopened = SyncCoordinator::open(&watermark_path).unwrap();
    let persisted = reopened.watermark().expect("persisted watermark");
    assert_eq!(persisted.synced_seq, 2);
    assert!(persisted.degraded);
    match reopened.query_visible_ids(&source, &eng).unwrap_err() {
        SyncError::IndexDegraded { .. } => {}
        other => panic!("expected IndexDegraded after restart, got {other:?}"),
    }

    // A later sync heals: degraded watermarks are rebuilt from the source.
    match reopened.sync(&source).unwrap() {
        SyncOutcome::Rebuilt {
            watermark_seq,
            docs_indexed,
        } => {
            assert_eq!(watermark_seq, 3);
            assert_eq!(docs_indexed, 3);
        }
        other => panic!("expected a clean rebuild, got {other:?}"),
    }
    assert!(!reopened.watermark().unwrap().degraded);
    assert_eq!(
        reopened.query_visible_ids(&source, &eng).unwrap(),
        vec!["d1", "d2", "d3"]
    );

    // The original coordinator heals the same way.
    reopened.set_fault_injection(FaultInjection::None);
    let outcome = coordinator.sync(&source).unwrap();
    assert!(matches!(outcome, SyncOutcome::Rebuilt { .. }));
    assert!(!coordinator.watermark().unwrap().degraded);
    assert_eq!(coordinator.query_visible_ids(&source, &eng).unwrap().len(), 3);
}

// ---------------------------------------------------------------------------
// 2. Backup restore / epoch divergence.
// ---------------------------------------------------------------------------

#[test]
fn backup_restore_behind_watermark_rebuilds_and_drops_phantoms() {
    let source = FakeSource::new("epoch-a");
    source.upsert("d1", "one", &["eng"], json!({}));
    source.upsert("d2", "two", &["eng"], json!({}));
    source.upsert("d3", "three", &["eng"], json!({}));

    let mut coordinator = SyncCoordinator::new();
    match coordinator.sync(&source).unwrap() {
        SyncOutcome::Rebuilt {
            watermark_seq,
            docs_indexed,
        } => {
            assert_eq!(watermark_seq, 3);
            assert_eq!(docs_indexed, 3);
        }
        other => panic!("expected rebuild, got {other:?}"),
    }

    // The source is restored from a backup taken after seq 2.
    source.restore_to(2);

    match coordinator.sync(&source).unwrap() {
        SyncOutcome::Rebuilt {
            watermark_seq,
            docs_indexed,
        } => {
            assert_eq!(watermark_seq, 2);
            assert_eq!(docs_indexed, 2);
        }
        other => panic!("expected rebuild after restore, got {other:?}"),
    }
    assert!(coordinator.live_doc("d3").is_none(), "phantom doc dropped");
    assert_eq!(
        coordinator
            .query_visible_ids(&source, &principals(&["eng"]))
            .unwrap(),
        vec!["d1", "d2"]
    );
}

#[test]
fn new_source_epoch_rebuilds_even_with_higher_head() {
    let epoch_a = FakeSource::new("epoch-a");
    epoch_a.upsert("old-1", "old", &["eng"], json!({}));

    let mut coordinator = SyncCoordinator::new();
    assert!(matches!(
        coordinator.sync(&epoch_a).unwrap(),
        SyncOutcome::Rebuilt {
            watermark_seq: 1,
            docs_indexed: 1
        }
    ));

    // A different lineage, ahead of the watermark.
    let epoch_b = FakeSource::new("epoch-b");
    epoch_b.upsert("new-1", "new one", &["eng"], json!({}));
    epoch_b.upsert("new-2", "new two", &["eng"], json!({}));

    match coordinator.sync(&epoch_b).unwrap() {
        SyncOutcome::Rebuilt {
            watermark_seq,
            docs_indexed,
        } => {
            assert_eq!(watermark_seq, 2);
            assert_eq!(docs_indexed, 2);
        }
        other => panic!("expected rebuild on epoch change, got {other:?}"),
    }
    assert_eq!(
        coordinator
            .query_visible_ids(&epoch_b, &principals(&["eng"]))
            .unwrap(),
        vec!["new-1", "new-2"]
    );
    assert!(coordinator.live_doc("old-1").is_none());
    assert_eq!(
        coordinator.watermark().unwrap().source_epoch_uuid,
        "epoch-b".to_string()
    );
}

// ---------------------------------------------------------------------------
// 3. Compaction cursor expiry.
// ---------------------------------------------------------------------------

#[test]
fn compacted_cursor_rebuilds_but_caught_up_index_does_not() {
    let source = FakeSource::new("epoch-a");
    for index in 1..=5 {
        source.upsert(&format!("d{index}"), &format!("text {index}"), &["eng"], json!({}));
    }

    let mut coordinator = SyncCoordinator::new();
    assert!(matches!(
        coordinator.sync(&source).unwrap(),
        SyncOutcome::Rebuilt {
            watermark_seq: 5,
            docs_indexed: 5
        }
    ));

    for index in 6..=8 {
        source.upsert(&format!("d{index}"), &format!("text {index}"), &["eng"], json!({}));
    }
    // Everything up to and including seq 6 is compacted away: seq 5 + 1 is gone.
    source.compact(7);

    match coordinator.sync(&source).unwrap() {
        SyncOutcome::Rebuilt {
            watermark_seq,
            docs_indexed,
        } => {
            assert_eq!(watermark_seq, 8);
            assert_eq!(docs_indexed, 8);
        }
        other => panic!("expected rebuild after cursor expiry, got {other:?}"),
    }

    // An already-caught-up coordinator must not rebuild needlessly.
    let calls_before = source.snapshot_calls();
    assert_eq!(
        coordinator.sync(&source).unwrap(),
        SyncOutcome::UpToDate { watermark_seq: 8 }
    );
    assert_eq!(source.snapshot_calls(), calls_before);
}

// ---------------------------------------------------------------------------
// 4. ACL revocation, fail-closed at every layer.
// ---------------------------------------------------------------------------

#[test]
fn acl_revocation_fails_closed_before_and_after_sync() {
    let source = FakeSource::new("epoch-a");
    source.upsert("hr-doc", "salary bands", &["hr"], json!({}));
    source.upsert("eng-doc", "source code", &["eng"], json!({}));
    source.set_principal("alice", &["hr"]);
    source.set_principal("bob", &["eng"]);

    let mut coordinator = SyncCoordinator::new();
    coordinator.sync(&source).unwrap();

    // No principals: never widen to "match everything".
    match coordinator.query_visible_ids(&source, &[]).unwrap_err() {
        SyncError::MissingPrincipals => {}
        other => panic!("expected MissingPrincipals, got {other:?}"),
    }
    match coordinator
        .hydrate_level2(&source, "eng-doc", &[])
        .unwrap_err()
    {
        SyncError::MissingPrincipals => {}
        other => panic!("expected MissingPrincipals, got {other:?}"),
    }

    let alice = source.resolve_principals("alice").unwrap();
    let bob = source.resolve_principals("bob").unwrap();
    assert_eq!(coordinator.query_visible_ids(&source, &alice).unwrap(), vec!["hr-doc"]);
    assert_eq!(coordinator.query_visible_ids(&source, &bob).unwrap(), vec!["eng-doc"]);

    // Revoke in the source without syncing: the index is now behind, and every
    // query refuses rather than serving a revoked document.
    source.set_acl("hr-doc", &["eng"]);
    match coordinator.query_visible_ids(&source, &alice).unwrap_err() {
        SyncError::BootHealIncomplete {
            synced_seq,
            source_last_seq,
        } => {
            assert_eq!(synced_seq, 2);
            assert_eq!(source_last_seq, 3);
        }
        other => panic!("expected BootHealIncomplete, got {other:?}"),
    }

    // Level-2 hydration re-checks the authoritative ACL even while the index
    // still holds the stale (visible) hit.
    assert!(coordinator.live_doc("hr-doc").is_some());
    match coordinator.hydrate_level2(&source, "hr-doc", &alice).unwrap_err() {
        SyncError::AclDenied { id } => assert_eq!(id, "hr-doc"),
        other => panic!("expected AclDenied, got {other:?}"),
    }
    assert!(coordinator
        .hydrate_level2(&source, "hr-doc", &bob)
        .is_ok());
    match coordinator
        .hydrate_level2(&source, "ghost", &alice)
        .unwrap_err()
    {
        SyncError::AclDenied { id } => assert_eq!(id, "ghost"),
        other => panic!("expected AclDenied for a missing doc, got {other:?}"),
    }

    // After sync the revocation is applied to the index.
    coordinator.sync(&source).unwrap();
    assert!(coordinator.query_visible_ids(&source, &alice).unwrap().is_empty());
    assert_eq!(
        coordinator.query_visible_ids(&source, &bob).unwrap(),
        vec!["eng-doc", "hr-doc"]
    );
}

// ---------------------------------------------------------------------------
// 5. envelope_hash vs content_hash separation, vector reuse and CAS attach.
// ---------------------------------------------------------------------------

#[test]
fn envelope_hash_gates_updates_content_hash_gates_vectors() {
    let source = FakeSource::new("epoch-a");
    source.upsert(
        "doc",
        "hello world",
        &["eng"],
        json!({"category": "a", "nested": {"x": 1}}),
    );

    let mut coordinator = SyncCoordinator::new();
    coordinator.sync(&source).unwrap();

    let first = coordinator.live_doc("doc").unwrap().clone();
    assert_eq!(first.content_hash, compute_content_hash("hello world"));
    assert_eq!(first.envelope_hash, compute_envelope_hash(&first.envelope));
    assert!(first.vector.is_none());

    assert_eq!(
        coordinator
            .attach_vectors(vec![VectorAttachment {
                id: "doc".to_string(),
                content_hash: first.content_hash.clone(),
                model_version: "model-v1".to_string(),
                vector: vec![0.25, 0.5],
            }])
            .unwrap(),
        1
    );
    assert!(coordinator.next_pending_embeddings(16).is_empty());

    // ACL-only update: envelope hash changes (not skipped), content hash does
    // not (the cached vector is reused).
    source.set_acl("doc", &["eng", "ops"]);
    coordinator.sync(&source).unwrap();
    let acl_entry = coordinator.live_doc("doc").unwrap().clone();
    assert_eq!(acl_entry.content_hash, first.content_hash);
    assert_ne!(acl_entry.envelope_hash, first.envelope_hash);
    assert!(acl_entry.acl.contains("eng") && acl_entry.acl.contains("ops"));
    assert!(
        acl_entry.vector.is_some(),
        "content-identical update must reuse the attached vector"
    );
    assert!(coordinator.next_pending_embeddings(16).is_empty());

    // Metadata-only update: same contract.
    source.set_metadata("doc", json!({"category": "b"}));
    coordinator.sync(&source).unwrap();
    let metadata_entry = coordinator.live_doc("doc").unwrap().clone();
    assert_eq!(metadata_entry.content_hash, first.content_hash);
    assert_ne!(metadata_entry.envelope_hash, acl_entry.envelope_hash);
    assert!(metadata_entry.vector.is_some());
    assert!(coordinator.next_pending_embeddings(16).is_empty());

    // Re-emitting an identical envelope is a true no-op: skipped, not applied.
    let seq_before = source.lineage().unwrap().last_seq;
    source.reemit("doc");
    match coordinator.sync(&source).unwrap() {
        SyncOutcome::Incremental {
            from_seq,
            to_seq,
            applied,
            skipped,
        } => {
            assert_eq!(from_seq, seq_before);
            assert_eq!(to_seq, seq_before + 1);
            assert_eq!(applied, 0);
            assert_eq!(skipped, 1);
        }
        other => panic!("expected an incremental no-op, got {other:?}"),
    }
    assert!(coordinator.live_doc("doc").unwrap().vector.is_some());

    // Text edit: both hashes change, the vector is invalidated.
    source.upsert(
        "doc",
        "hello world v2",
        &["eng", "ops"],
        json!({"category": "b"}),
    );
    coordinator.sync(&source).unwrap();
    let edited = coordinator.live_doc("doc").unwrap().clone();
    assert_ne!(edited.content_hash, first.content_hash);
    assert_ne!(edited.envelope_hash, first.envelope_hash);
    assert!(edited.vector.is_none());
    let pending = coordinator.next_pending_embeddings(16);
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].0, "doc");
    assert_eq!(pending[0].1, edited.content_hash);
    assert_eq!(pending[0].2, "hello world v2");

    // A completion for the old content hash is rejected loudly.
    match coordinator
        .attach_vectors(vec![VectorAttachment {
            id: "doc".to_string(),
            content_hash: first.content_hash.clone(),
            model_version: "model-v1".to_string(),
            vector: vec![0.1],
        }])
        .unwrap_err()
    {
        SyncError::StaleVectorCas {
            id,
            expected_content_hash,
            actual_content_hash,
            expected_model,
        } => {
            assert_eq!(id, "doc");
            assert_eq!(expected_content_hash, first.content_hash);
            assert_eq!(actual_content_hash, Some(edited.content_hash.clone()));
            assert_eq!(expected_model, "model-v1");
        }
        other => panic!("expected StaleVectorCas, got {other:?}"),
    }

    // A valid completion for the new content hash is accepted, and a stale one
    // in the same batch must reject the whole batch.
    assert_eq!(
        coordinator
            .attach_vectors(vec![VectorAttachment {
                id: "doc".to_string(),
                content_hash: edited.content_hash.clone(),
                model_version: "model-v1".to_string(),
                vector: vec![0.9, 0.1],
            }])
            .unwrap(),
        1
    );
    source.upsert("other", "other text", &["eng"], json!({}));
    coordinator.sync(&source).unwrap();
    let other = coordinator.live_doc("other").unwrap().content_hash.clone();
    match coordinator
        .attach_vectors(vec![
            VectorAttachment {
                id: "other".to_string(),
                content_hash: other.clone(),
                model_version: "model-v1".to_string(),
                vector: vec![0.3],
            },
            VectorAttachment {
                id: "doc".to_string(),
                content_hash: first.content_hash.clone(),
                model_version: "model-v1".to_string(),
                vector: vec![0.4],
            },
        ])
        .unwrap_err()
    {
        SyncError::StaleVectorCas { .. } => {}
        other_error => panic!("expected StaleVectorCas, got {other_error:?}"),
    }
    assert!(
        coordinator.live_doc("other").unwrap().vector.is_none(),
        "batch validation must reject before attaching anything"
    );

    // Completions for deleted documents are rejected as stale.
    source.delete("doc");
    coordinator.sync(&source).unwrap();
    match coordinator
        .attach_vectors(vec![VectorAttachment {
            id: "doc".to_string(),
            content_hash: edited.content_hash.clone(),
            model_version: "model-v1".to_string(),
            vector: vec![0.5],
        }])
        .unwrap_err()
    {
        SyncError::StaleVectorCas {
            actual_content_hash,
            ..
        } => assert!(actual_content_hash.is_none()),
        other => panic!("expected StaleVectorCas for a deleted doc, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 6. Concurrent rebuild: mutations that land after the snapshot is taken.
// ---------------------------------------------------------------------------

#[test]
fn rebuild_catches_up_mutations_landing_after_the_snapshot() {
    let source = FakeSource::new("epoch-a");
    source.upsert("doc1", "alpha", &["eng"], json!({"v": 1}));
    source.upsert("doc2", "beta", &["eng"], json!({"v": 1}));
    source.upsert("doc3", "gamma", &["eng"], json!({"v": 1}));

    let mut coordinator = SyncCoordinator::new();
    coordinator.sync(&source).unwrap();
    let doc1 = coordinator.live_doc("doc1").unwrap().clone();
    coordinator
        .attach_vectors(vec![VectorAttachment {
            id: "doc1".to_string(),
            content_hash: doc1.content_hash.clone(),
            model_version: "model-v1".to_string(),
            vector: vec![0.1, 0.2],
        }])
        .unwrap();

    // These land after `snapshot_at()` captured its view (seqs 4..7).
    source.on_next_snapshot(|state| {
        apply_upsert(state, "doc4", "delta", &["ops"], json!({"v": 1}));
        apply_delete(state, "doc2");
        apply_set_acl(state, "doc1", &["eng", "ops"]);
        apply_upsert(state, "doc3", "gamma v2", &["eng"], json!({"v": 2}));
    });

    // A new lineage epoch forces the rebuild path through `sync`.
    source.replace_epoch("epoch-a-restored");

    match coordinator.sync(&source).unwrap() {
        SyncOutcome::Rebuilt {
            watermark_seq,
            docs_indexed,
        } => {
            assert_eq!(watermark_seq, 7, "rebuild must replay the concurrent tail");
            assert_eq!(docs_indexed, 3);
        }
        other => panic!("expected rebuild, got {other:?}"),
    }

    // Exact authoritative state, record by record.
    let authoritative = source.live_ids();
    let mut live: Vec<String> = coordinator
        .live_docs()
        .map(|(id, _)| id.clone())
        .collect();
    live.sort();
    assert_eq!(live, authoritative);
    for id in &authoritative {
        let record = source.get_doc(id).unwrap().expect("authoritative record");
        let entry = coordinator.live_doc(id).unwrap();
        assert_eq!(entry.envelope, record.envelope);
        assert_eq!(entry.content_hash, record.content_hash);
        assert_eq!(entry.envelope_hash, record.envelope_hash);
    }

    // ACL changes survived the rebuild; the deleted doc is gone.
    assert_eq!(
        coordinator
            .query_visible_ids(&source, &principals(&["eng"]))
            .unwrap(),
        vec!["doc1", "doc3"]
    );
    assert_eq!(
        coordinator
            .query_visible_ids(&source, &principals(&["ops"]))
            .unwrap(),
        vec!["doc1", "doc4"]
    );
    assert!(coordinator.live_doc("doc2").is_none());

    // The content-identical ACL update preserved doc1's vector; the text edit
    // on doc3 invalidated whatever it had.
    assert!(coordinator.live_doc("doc1").unwrap().vector.is_some());
    assert!(coordinator.live_doc("doc3").unwrap().vector.is_none());
    assert_eq!(
        coordinator.watermark().unwrap().source_epoch_uuid,
        "epoch-a-restored".to_string()
    );
    assert_eq!(source.snapshot_calls(), 2);
}
