//! Authoritative sync, snapshot and revision contract (Phase 0).
//!
//! This module is the contract layer between a system of record (`nDB`, a
//! filesystem walker, any adapter) and the derived search index. It does not
//! store documents for query: it owns the *correctness* rules that make a
//! derived index safe to serve:
//!
//! - **Lineage & watermarks:** every source exposes `(source_epoch_uuid,
//!   last_seq, min_available_seq)`; the index commits the highest *contiguous*
//!   `synced_seq` it applied.
//! - **Two hashes:** [`compute_content_hash`] covers searchable text only and
//!   gates embedding reuse; [`compute_envelope_hash`] covers the full envelope
//!   (ACL, metadata, versions, ids) and gates no-op skipping. An ACL-only
//!   update therefore wakes the index *and* reuses the cached vector.
//! - **Divergence detection:** a changed epoch, a restored backup
//!   (`synced_seq > last_seq`) or an expired cursor refuses incremental
//!   catch-up and forces a clean snapshot rebuild.
//! - **Fail-closed reads:** degraded indexes and un-caught-up indexes refuse
//!   queries; principals are mandatory; Level-2 hydration re-checks ACL
//!   against the authoritative source.
//!
//! [`SyncCoordinator`] carries an in-memory reference store so the contract is
//! verifiable end-to-end before any index engine exists (Phase 0 exit gate).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Lineage of a source instance.
///
/// `source_epoch_uuid` changes whenever the source is a *different* lineage
/// (re-created database, restored foreign backup). `last_seq` is the highest
/// committed mutation sequence. `min_available_seq` is the oldest sequence
/// still retrievable through [`SourceConnector::changes_since`]; it advances
/// when the source compacts its journal — compaction never resets `last_seq`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLineage {
    /// Identity of this source lineage.
    pub source_epoch_uuid: String,
    /// Highest committed mutation sequence number.
    pub last_seq: u64,
    /// Oldest sequence number still available for incremental sync.
    pub min_available_seq: u64,
}

/// The full indexed envelope of one chunk/document.
///
/// This is the unit that both hashes are computed over. `indexable_text` is the
/// concatenation/structure of every searchable text field; `metadata` is the
/// open bag stored alongside, and `acl` holds **group IDs** — never user lists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexedEnvelope {
    /// Document/chunk identifier, unique inside the source.
    pub id: String,
    /// Owning document for a chunk, if this envelope is a chunk.
    pub parent_id: Option<String>,
    /// Chunk position inside the parent, if applicable.
    pub chunk_index: Option<u32>,
    /// Document kind (`doc`, `memory`, `chat`, `image`, `record`, ...).
    pub kind: String,
    /// Concatenated/structured searchable text. Input to [`compute_content_hash`].
    pub indexable_text: String,
    /// Group IDs allowed to see this document (sorted, deduplicated when hashed).
    pub acl: Vec<String>,
    /// Open metadata bag, stored and filtered on, not part of `content_hash`.
    pub metadata: Value,
    /// Envelope schema version; bump invalidates skip decisions.
    pub schema_version: u32,
    /// Analyzer version; bump invalidates skip decisions.
    pub analyzer_version: u32,
}

/// A source record: the envelope plus the two hashes the source computed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceRecord {
    /// Record identifier (must equal `envelope.id`).
    pub id: String,
    /// The full envelope.
    pub envelope: IndexedEnvelope,
    /// Hash of `envelope.indexable_text` only — gates embedding reuse.
    pub content_hash: String,
    /// Hash of the canonicalised full envelope — gates no-op skipping.
    pub envelope_hash: String,
}

impl SourceRecord {
    /// Build a record, computing both hashes from the envelope.
    pub fn new(envelope: IndexedEnvelope) -> Self {
        let content_hash = compute_content_hash(&envelope.indexable_text);
        let envelope_hash = compute_envelope_hash(&envelope);
        Self {
            id: envelope.id.clone(),
            envelope,
            content_hash,
            envelope_hash,
        }
    }
}

/// Mutation operation carried by a [`SourceChange`].
// The Phase 0 connector contract fixes this shape (`Upsert(SourceRecord)`); an
// indirection would change the public type for a few hundred bytes saved.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SourceChangeOp {
    /// Insert or replace the record.
    Upsert(SourceRecord),
    /// Remove the record identified by the enclosing `id`.
    Delete,
}

/// One ordered mutation in the source journal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceChange {
    /// Strictly increasing sequence number, dense within a lineage.
    pub seq: u64,
    /// Record identifier the change applies to.
    pub id: String,
    /// What happened.
    pub op: SourceChangeOp,
}

/// A consistent point-in-time view of the source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceSnapshot {
    /// Lineage as of the snapshot instant: `last_seq` is the replay cursor.
    pub lineage: SourceLineage,
    /// Every live record at that instant.
    pub records: Vec<SourceRecord>,
}

/// Errors of the sync contract.
#[derive(Debug, Error)]
pub enum SyncError {
    /// The requested cursor has been compacted away.
    #[error(
        "cursor expired: requested seq {requested_seq}, oldest available {min_available_seq}, source at {last_seq}"
    )]
    CursorExpired {
        /// Sequence the caller asked from.
        requested_seq: u64,
        /// Oldest sequence still available.
        min_available_seq: u64,
        /// Current source head.
        last_seq: u64,
    },

    /// The source lineage changed underneath the caller.
    #[error("source epoch mismatch: expected '{expected}', found '{actual}'")]
    EpochMismatch {
        /// Epoch the caller built against.
        expected: String,
        /// Epoch the source reports now.
        actual: String,
    },

    /// The index watermark is ahead of the source (restored backup / lost tail).
    #[error("cursor ahead: index synced to {synced_seq}, source head is {source_last_seq}")]
    CursorAhead {
        /// Watermark sequence.
        synced_seq: u64,
        /// Current source head.
        source_last_seq: u64,
    },

    /// The index is degraded after an apply failure and refuses reads.
    #[error("index degraded: {reason}")]
    IndexDegraded {
        /// Why the index is degraded.
        reason: String,
    },

    /// The index has not caught up with the source; reads are refused.
    #[error(
        "boot-heal incomplete: synced {synced_seq}, source at {source_last_seq} — queries blocked until caught up"
    )]
    BootHealIncomplete {
        /// Watermark sequence.
        synced_seq: u64,
        /// Current source head.
        source_last_seq: u64,
    },

    /// A query carried no principals. Never widen to "match everything".
    #[error("missing principals: queries must carry at least one principal")]
    MissingPrincipals,

    /// Access denied (or document gone) against the authoritative source.
    #[error("access denied for document '{id}'")]
    AclDenied {
        /// Requested document.
        id: String,
    },

    /// A vector completion no longer matches the document's current state.
    #[error(
        "stale vector for '{id}': expected content hash '{expected_content_hash}' (model '{expected_model}'), document has {actual_content_hash:?}"
    )]
    StaleVectorCas {
        /// Document identifier.
        id: String,
        /// Content hash the embedding was produced for.
        expected_content_hash: String,
        /// Current content hash, or `None` when the document is gone.
        actual_content_hash: Option<String>,
        /// Model version the embedding was produced with.
        expected_model: String,
    },

    /// Applying a change failed mid-batch; the index is now degraded.
    #[error("apply failed at seq {seq}: {reason}")]
    ApplyFailed {
        /// Sequence that failed.
        seq: u64,
        /// Failure detail.
        reason: String,
    },

    /// The source violated the connector contract.
    #[error("source error: {0}")]
    SourceIo(String),

    /// Local filesystem failure (watermark persistence).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Watermark (de)serialisation failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Read-side contract every indexable source must implement.
///
/// Implementations own durability and ordering; the index owns correctness of
/// what it derives from these calls.
pub trait SourceConnector {
    /// Current lineage: epoch, head sequence and oldest available sequence.
    fn lineage(&self) -> Result<SourceLineage, SyncError>;

    /// Consistent point-in-time snapshot of every live record.
    fn snapshot_at(&self) -> Result<SourceSnapshot, SyncError>;

    /// All mutations with `seq > since`, strictly increasing.
    ///
    /// Returns [`SyncError::CursorExpired`] when at least one required event
    /// has been compacted away (`since + 1 < min_available_seq`), and
    /// [`SyncError::CursorAhead`] when `since > last_seq`.
    fn changes_since(&self, seq: u64) -> Result<Vec<SourceChange>, SyncError>;

    /// Authoritative lookup, used for Level-2 hydration.
    fn get_doc(&self, id: &str) -> Result<Option<SourceRecord>, SyncError>;

    /// Map a user to the group IDs they hold *now*. The index stores groups;
    /// it never expands groups into user lists.
    fn resolve_principals(&self, user_id: &str) -> Result<Vec<String>, SyncError>;
}

/// Durable sync watermark, committed atomically after every applied batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncWatermark {
    /// Source lineage the sequence belongs to.
    pub source_epoch_uuid: String,
    /// Highest contiguous sequence successfully applied.
    pub synced_seq: u64,
    /// Set when an apply failed after the source committed.
    pub degraded: bool,
    /// Human-readable degradation cause, required when `degraded` is set.
    pub degraded_reason: Option<String>,
}

impl SyncWatermark {
    /// Persist atomically: write `<path>.tmp`, `sync_all()`, rename onto `path`.
    ///
    /// On Windows a held destination handle can make the replace fail; the
    /// fallback removes the destination and retries the rename.
    pub fn save_atomic(&self, path: impl AsRef<Path>) -> Result<(), SyncError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut tmp = path.as_os_str().to_os_string();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);

        let bytes = serde_json::to_vec_pretty(self)?;
        {
            let mut file = fs::File::create(&tmp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }

        if fs::rename(&tmp, path).is_ok() {
            return Ok(());
        }
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                return Err(SyncError::Io(e));
            }
        }
        if let Err(e) = fs::rename(&tmp, path) {
            let _ = fs::remove_file(&tmp);
            return Err(SyncError::Io(e));
        }
        Ok(())
    }

    /// Load and validate a watermark written by [`SyncWatermark::save_atomic`].
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SyncError> {
        let bytes = fs::read(path)?;
        let watermark: SyncWatermark = serde_json::from_slice(&bytes)?;
        watermark.validate()?;
        Ok(watermark)
    }

    fn validate(&self) -> Result<(), SyncError> {
        if self.source_epoch_uuid.is_empty() {
            return Err(invalid_data("watermark has an empty source_epoch_uuid"));
        }
        match (&self.degraded_reason, self.degraded) {
            (Some(_), true) | (None, false) => Ok(()),
            (None, true) => Err(invalid_data("watermark is degraded without a reason")),
            (Some(_), false) => Err(invalid_data(
                "watermark carries a degraded reason while not degraded",
            )),
        }
    }
}

fn invalid_data(message: &str) -> SyncError {
    SyncError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message,
    ))
}

/// A vector attached to a live document, keyed by the content hash it embeds.
#[derive(Debug, Clone, PartialEq)]
pub struct AttachedVector {
    /// Embedding values.
    pub vector: Vec<f32>,
    /// Content hash the embedding was produced for.
    pub content_hash: String,
    /// Model/version that produced the embedding.
    pub model_version: String,
}

/// One live document as held by the index.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexedDocEntry {
    /// Full envelope.
    pub envelope: IndexedEnvelope,
    /// Hash of the searchable text.
    pub content_hash: String,
    /// Hash of the canonicalised envelope.
    pub envelope_hash: String,
    /// Group IDs allowed to see the document.
    pub acl: HashSet<String>,
    /// Cached embedding, if one has been attached for this content hash.
    pub vector: Option<AttachedVector>,
}

/// A pending embedding completion offered by the host's embed loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorAttachment {
    /// Document the embedding belongs to.
    pub id: String,
    /// Content hash the host embedded.
    pub content_hash: String,
    /// Model/version the host used.
    pub model_version: String,
    /// Embedding values.
    pub vector: Vec<f32>,
}

/// Result of one [`SyncCoordinator::sync`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncOutcome {
    /// The index rebuilt from a fresh snapshot.
    Rebuilt {
        /// Sequence the watermark now holds.
        watermark_seq: u64,
        /// Live documents after the rebuild.
        docs_indexed: usize,
    },
    /// The index applied an incremental batch.
    Incremental {
        /// Sequence the batch started after.
        from_seq: u64,
        /// Sequence the watermark now holds.
        to_seq: u64,
        /// Events applied.
        applied: usize,
        /// Events skipped because `envelope_hash` matched (true no-ops).
        skipped: usize,
    },
    /// The index was already caught up; nothing was applied.
    UpToDate {
        /// Sequence the watermark holds.
        watermark_seq: u64,
    },
}

/// Test hook for simulating a dual-write apply failure.
///
/// Armed faults fire **once**, then disarm, so a subsequent heal is
/// deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FaultInjection {
    /// No fault.
    #[default]
    None,
    /// Fail while applying the change with this sequence number.
    FailApplyAtSeq(u64),
}

/// Phase 0 contract engine plus in-memory reference store.
///
/// The coordinator is not the query engine: it verifies the sync contract
/// end-to-end (crash, restore, compaction, ACL, CAS vector attach) before the
/// index core is built on top of it.
pub struct SyncCoordinator {
    watermark: Option<SyncWatermark>,
    watermark_path: Option<PathBuf>,
    live_docs: HashMap<String, IndexedDocEntry>,
    fault: FaultInjection,
    /// Set when the process has a watermark but no in-memory document state
    /// (restart): the next sync must rebuild before reads are served.
    needs_rebuild: bool,
}

impl Default for SyncCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl SyncCoordinator {
    /// Create an empty coordinator with no persistent watermark.
    pub fn new() -> Self {
        Self {
            watermark: None,
            watermark_path: None,
            live_docs: HashMap::new(),
            fault: FaultInjection::None,
            needs_rebuild: false,
        }
    }

    /// Create an empty coordinator that persists its watermark to `path` on
    /// every commit.
    pub fn with_watermark_path(path: impl Into<PathBuf>) -> Self {
        Self {
            watermark_path: Some(path.into()),
            ..Self::new()
        }
    }

    /// Re-open a coordinator from a persisted watermark.
    ///
    /// The reference store keeps documents in memory only, so an existing
    /// watermark means the next [`SyncCoordinator::sync`] rebuilds before
    /// reads are served (reported as [`SyncError::BootHealIncomplete`]).
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, SyncError> {
        let path = path.into();
        let watermark = match SyncWatermark::load(&path) {
            Ok(watermark) => Some(watermark),
            Err(SyncError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };
        Ok(Self {
            // The reference store holds documents in memory only, so a loaded
            // watermark means the document state must be rebuilt before reads.
            needs_rebuild: watermark.is_some(),
            watermark,
            watermark_path: Some(path),
            live_docs: HashMap::new(),
            fault: FaultInjection::None,
        })
    }

    /// Current watermark, if any commit has happened.
    pub fn watermark(&self) -> Option<&SyncWatermark> {
        self.watermark.as_ref()
    }

    /// Look up one live index entry.
    pub fn live_doc(&self, id: &str) -> Option<&IndexedDocEntry> {
        self.live_docs.get(id)
    }

    /// Number of live documents in the reference store.
    pub fn live_doc_count(&self) -> usize {
        self.live_docs.len()
    }

    /// Iterate over every live index entry.
    pub fn live_docs(&self) -> impl Iterator<Item = (&String, &IndexedDocEntry)> + '_ {
        self.live_docs.iter()
    }

    /// Install or clear the apply-failure test hook.
    pub fn set_fault_injection(&mut self, fault: FaultInjection) {
        self.fault = fault;
    }

    /// Bring the index up to date with the source.
    ///
    /// Rebuilds from a snapshot when there is no watermark, the watermark is
    /// degraded, the epoch changed, the watermark is ahead of the source
    /// (restored backup), the journal cursor expired, or a feed gap is seen.
    /// Otherwise applies the batch in strictly increasing order and commits the
    /// highest contiguous sequence.
    pub fn sync(&mut self, source: &dyn SourceConnector) -> Result<SyncOutcome, SyncError> {
        let lineage = source.lineage()?;

        let must_rebuild = self.needs_rebuild
            || match &self.watermark {
                None => true,
                Some(watermark) => {
                    watermark.degraded
                        || watermark.source_epoch_uuid != lineage.source_epoch_uuid
                        || watermark.synced_seq > lineage.last_seq
                }
            };
        if must_rebuild {
            return self.rebuild_from_snapshot(source);
        }

        let start_seq = match &self.watermark {
            Some(watermark) => watermark.synced_seq,
            // `must_rebuild` covers the missing-watermark case; rebuilding is
            // the only correct response if that ever changes.
            None => return self.rebuild_from_snapshot(source),
        };
        if start_seq == lineage.last_seq {
            return Ok(SyncOutcome::UpToDate {
                watermark_seq: start_seq,
            });
        }

        let changes = match source.changes_since(start_seq) {
            Ok(changes) => changes,
            Err(SyncError::CursorExpired { .. }) => return self.rebuild_from_snapshot(source),
            Err(e) => return Err(e),
        };

        let mut ordered: Vec<SourceChange> = changes
            .into_iter()
            .filter(|change| change.seq > start_seq)
            .collect();
        ordered.sort_by_key(|change| change.seq);

        let mut synced_seq = start_seq;
        let mut applied = 0usize;
        let mut skipped = 0usize;
        for change in &ordered {
            if change.seq != synced_seq + 1 {
                // A feed gap means the connector cannot be trusted: recover
                // from the authoritative snapshot instead of diverging.
                return self.rebuild_from_snapshot(source);
            }
            if let Some(reason) = self.take_apply_fault(change.seq) {
                self.mark_degraded(&lineage, synced_seq, reason.clone())?;
                return Err(SyncError::ApplyFailed {
                    seq: change.seq,
                    reason,
                });
            }
            match apply_change(&mut self.live_docs, change) {
                Ok(ApplyDelta::Applied) => applied += 1,
                Ok(ApplyDelta::Skipped) => skipped += 1,
                Err(reason) => {
                    self.mark_degraded(&lineage, synced_seq, reason.clone())?;
                    return Err(SyncError::ApplyFailed {
                        seq: change.seq,
                        reason,
                    });
                }
            }
            synced_seq = change.seq;
        }

        if synced_seq < lineage.last_seq {
            return Err(SyncError::SourceIo(format!(
                "source reported last_seq {} but produced no changes after seq {synced_seq}",
                lineage.last_seq
            )));
        }

        self.watermark = Some(SyncWatermark {
            source_epoch_uuid: lineage.source_epoch_uuid,
            synced_seq,
            degraded: false,
            degraded_reason: None,
        });
        self.persist_watermark()?;
        Ok(SyncOutcome::Incremental {
            from_seq: start_seq,
            to_seq: synced_seq,
            applied,
            skipped,
        })
    }

    /// Rebuild the whole index from a consistent snapshot, then replay any
    /// mutations that landed while the snapshot was being built.
    ///
    /// Cached vectors are preserved for documents whose `content_hash` did not
    /// change. The swap is atomic: on any failure the previous live state is
    /// left untouched.
    pub fn rebuild_from_snapshot(
        &mut self,
        source: &dyn SourceConnector,
    ) -> Result<SyncOutcome, SyncError> {
        for _attempt in 0..4 {
            let snapshot = source.snapshot_at()?;
            let epoch = snapshot.lineage.source_epoch_uuid.clone();

            let mut staging: HashMap<String, IndexedDocEntry> =
                HashMap::with_capacity(snapshot.records.len());
            for record in &snapshot.records {
                let entry = entry_from_record(record, self.live_docs.get(&record.id))
                    .map_err(SyncError::SourceIo)?;
                staging.insert(record.id.clone(), entry);
            }

            let mut cursor = snapshot.lineage.last_seq;
            let mut converged = true;
            loop {
                let current = source.lineage()?;
                if current.source_epoch_uuid != epoch {
                    converged = false;
                    break;
                }
                if current.last_seq <= cursor {
                    break;
                }
                let changes = match source.changes_since(cursor) {
                    Ok(changes) => changes,
                    Err(SyncError::CursorExpired { .. }) => {
                        converged = false;
                        break;
                    }
                    Err(e) => return Err(e),
                };
                let mut ordered: Vec<&SourceChange> = changes
                    .iter()
                    .filter(|change| change.seq > cursor)
                    .collect();
                ordered.sort_by_key(|change| change.seq);
                if ordered.is_empty() {
                    converged = false;
                    break;
                }
                for change in ordered {
                    if change.seq != cursor + 1 {
                        converged = false;
                        break;
                    }
                    if let Some(reason) = self.take_apply_fault(change.seq) {
                        // The staged state is discarded: the previous live
                        // state is still the only consistent one.
                        self.mark_degraded_preserving_state(&epoch, reason.clone())?;
                        return Err(SyncError::ApplyFailed {
                            seq: change.seq,
                            reason,
                        });
                    }
                    apply_change(&mut staging, change).map_err(|reason| {
                        SyncError::SourceIo(format!(
                            "invalid change at seq {}: {reason}",
                            change.seq
                        ))
                    })?;
                    cursor = change.seq;
                }
                if !converged {
                    break;
                }
            }
            if !converged {
                continue;
            }

            self.live_docs = staging;
            self.needs_rebuild = false;
            self.watermark = Some(SyncWatermark {
                source_epoch_uuid: epoch,
                synced_seq: cursor,
                degraded: false,
                degraded_reason: None,
            });
            self.persist_watermark()?;
            return Ok(SyncOutcome::Rebuilt {
                watermark_seq: cursor,
                docs_indexed: self.live_docs.len(),
            });
        }
        Err(SyncError::SourceIo(
            "snapshot rebuild did not converge after 4 attempts".to_string(),
        ))
    }

    /// Return the live document IDs visible to `principals`.
    ///
    /// Fail-closed on every axis: a degraded index, an empty principal list, a
    /// missing watermark, an epoch mismatch, or an index that has not caught up
    /// with the source head all return an error instead of a (possibly wrong)
    /// result set.
    pub fn query_visible_ids(
        &self,
        source: &dyn SourceConnector,
        principals: &[String],
    ) -> Result<Vec<String>, SyncError> {
        let watermark = match &self.watermark {
            Some(watermark) => watermark,
            None => {
                let lineage = source.lineage()?;
                return Err(SyncError::BootHealIncomplete {
                    synced_seq: 0,
                    source_last_seq: lineage.last_seq,
                });
            }
        };
        if watermark.degraded {
            return Err(SyncError::IndexDegraded {
                reason: watermark
                    .degraded_reason
                    .clone()
                    .unwrap_or_else(|| "index marked degraded without a reason".to_string()),
            });
        }
        if principals.is_empty() {
            return Err(SyncError::MissingPrincipals);
        }

        let lineage = source.lineage()?;
        if self.needs_rebuild
            || watermark.source_epoch_uuid != lineage.source_epoch_uuid
            || watermark.synced_seq != lineage.last_seq
        {
            return Err(SyncError::BootHealIncomplete {
                synced_seq: watermark.synced_seq,
                source_last_seq: lineage.last_seq,
            });
        }

        let principals: HashSet<&str> = principals.iter().map(String::as_str).collect();
        let mut visible: Vec<String> = self
            .live_docs
            .iter()
            .filter(|(_, entry)| entry.acl.iter().any(|group| principals.contains(group.as_str())))
            .map(|(id, _)| id.clone())
            .collect();
        visible.sort();
        Ok(visible)
    }

    /// Fetch the authoritative record for Level-2 disclosure.
    ///
    /// The index is never trusted for access decisions: the record is read from
    /// the source and its ACL re-checked there. Missing documents are reported
    /// as [`SyncError::AclDenied`] so existence is not leaked.
    pub fn hydrate_level2(
        &self,
        source: &dyn SourceConnector,
        id: &str,
        principals: &[String],
    ) -> Result<SourceRecord, SyncError> {
        if principals.is_empty() {
            return Err(SyncError::MissingPrincipals);
        }
        if let Some(watermark) = &self.watermark {
            if watermark.degraded {
                return Err(SyncError::IndexDegraded {
                    reason: watermark
                        .degraded_reason
                        .clone()
                        .unwrap_or_else(|| "index marked degraded without a reason".to_string()),
                });
            }
        }
        let record = source
            .get_doc(id)?
            .ok_or_else(|| SyncError::AclDenied { id: id.to_string() })?;
        let principals: HashSet<&str> = principals.iter().map(String::as_str).collect();
        if !record
            .envelope
            .acl
            .iter()
            .any(|group| principals.contains(group.as_str()))
        {
            return Err(SyncError::AclDenied { id: id.to_string() });
        }
        Ok(record)
    }

    /// Documents that still need an embedding: `live ∧ ¬has_vector`.
    ///
    /// Purely derived from index state — there is no separate queue that can
    /// drift. Returns `(id, content_hash, indexable_text)` ordered by id.
    pub fn next_pending_embeddings(&self, limit: usize) -> Vec<(String, String, String)> {
        let mut pending: Vec<(String, String, String)> = self
            .live_docs
            .iter()
            .filter(|(_, entry)| entry.vector.is_none())
            .map(|(id, entry)| {
                (
                    id.clone(),
                    entry.content_hash.clone(),
                    entry.envelope.indexable_text.clone(),
                )
            })
            .collect();
        pending.sort();
        pending.truncate(limit);
        pending
    }

    /// Attach embedding completions with compare-and-set on
    /// `(id, content_hash, model_version)`.
    ///
    /// The batch is validated before anything is stored: one stale completion
    /// rejects the whole batch, so a partial attach cannot hide a race.
    pub fn attach_vectors(
        &mut self,
        entries: Vec<VectorAttachment>,
    ) -> Result<usize, SyncError> {
        for attachment in &entries {
            match self.live_docs.get(&attachment.id) {
                Some(entry) if entry.content_hash == attachment.content_hash => {}
                Some(entry) => {
                    return Err(SyncError::StaleVectorCas {
                        id: attachment.id.clone(),
                        expected_content_hash: attachment.content_hash.clone(),
                        actual_content_hash: Some(entry.content_hash.clone()),
                        expected_model: attachment.model_version.clone(),
                    })
                }
                None => {
                    return Err(SyncError::StaleVectorCas {
                        id: attachment.id.clone(),
                        expected_content_hash: attachment.content_hash.clone(),
                        actual_content_hash: None,
                        expected_model: attachment.model_version.clone(),
                    })
                }
            }
        }

        let mut attached = 0usize;
        for attachment in entries {
            if let Some(entry) = self.live_docs.get_mut(&attachment.id) {
                entry.vector = Some(AttachedVector {
                    vector: attachment.vector,
                    content_hash: attachment.content_hash,
                    model_version: attachment.model_version,
                });
                attached += 1;
            }
        }
        Ok(attached)
    }

    fn take_apply_fault(&mut self, seq: u64) -> Option<String> {
        if let FaultInjection::FailApplyAtSeq(fault_seq) = self.fault {
            if fault_seq == seq {
                self.fault = FaultInjection::None;
                return Some(format!("injected dual-write apply failure at seq {seq}"));
            }
        }
        None
    }

    fn mark_degraded(
        &mut self,
        lineage: &SourceLineage,
        synced_seq: u64,
        reason: String,
    ) -> Result<(), SyncError> {
        self.watermark = Some(SyncWatermark {
            source_epoch_uuid: lineage.source_epoch_uuid.clone(),
            synced_seq,
            degraded: true,
            degraded_reason: Some(reason),
        });
        self.persist_watermark()
    }

    /// Degrade without touching the live state: the watermark keeps the epoch
    /// and sequence of the state that is still being served (nothing), so the
    /// next sync rebuilds.
    fn mark_degraded_preserving_state(
        &mut self,
        fallback_epoch: &str,
        reason: String,
    ) -> Result<(), SyncError> {
        let (source_epoch_uuid, synced_seq) = match &self.watermark {
            Some(watermark) => (watermark.source_epoch_uuid.clone(), watermark.synced_seq),
            None => (fallback_epoch.to_string(), 0),
        };
        self.watermark = Some(SyncWatermark {
            source_epoch_uuid,
            synced_seq,
            degraded: true,
            degraded_reason: Some(reason),
        });
        self.persist_watermark()
    }

    fn persist_watermark(&self) -> Result<(), SyncError> {
        if let (Some(path), Some(watermark)) = (&self.watermark_path, &self.watermark) {
            watermark.save_atomic(path)?;
        }
        Ok(())
    }
}

enum ApplyDelta {
    Applied,
    Skipped,
}

fn apply_change(
    live: &mut HashMap<String, IndexedDocEntry>,
    change: &SourceChange,
) -> Result<ApplyDelta, String> {
    match &change.op {
        SourceChangeOp::Upsert(record) => {
            let entry = entry_from_record(record, live.get(&record.id))?;
            if let Some(existing) = live.get(&record.id) {
                if existing.envelope_hash == entry.envelope_hash {
                    return Ok(ApplyDelta::Skipped);
                }
            }
            live.insert(record.id.clone(), entry);
            Ok(ApplyDelta::Applied)
        }
        SourceChangeOp::Delete => {
            live.remove(&change.id);
            Ok(ApplyDelta::Applied)
        }
    }
}

fn entry_from_record(
    record: &SourceRecord,
    previous: Option<&IndexedDocEntry>,
) -> Result<IndexedDocEntry, String> {
    if record.id != record.envelope.id {
        return Err(format!(
            "record id '{}' does not match envelope id '{}'",
            record.id, record.envelope.id
        ));
    }
    let content_hash = compute_content_hash(&record.envelope.indexable_text);
    if content_hash != record.content_hash {
        return Err(format!(
            "content_hash mismatch for '{}': record says '{}', text hashes to '{}'",
            record.id, record.content_hash, content_hash
        ));
    }
    let envelope_hash = compute_envelope_hash(&record.envelope);
    if envelope_hash != record.envelope_hash {
        return Err(format!(
            "envelope_hash mismatch for '{}': record says '{}', envelope hashes to '{}'",
            record.id, record.envelope_hash, envelope_hash
        ));
    }
    let vector = previous
        .filter(|entry| entry.content_hash == content_hash)
        .and_then(|entry| entry.vector.clone());
    Ok(IndexedDocEntry {
        acl: record.envelope.acl.iter().cloned().collect(),
        envelope: record.envelope.clone(),
        content_hash,
        envelope_hash,
        vector,
    })
}

/// Deterministic content hash: BLAKE3 over `indexable_text` only.
///
/// This gates embedding reuse. ACL, metadata and chunk position deliberately do
/// not participate — an ACL-only update must keep its vector.
pub fn compute_content_hash(indexable_text: &str) -> String {
    to_hex(&compute_content_digest(indexable_text))
}

/// Raw BLAKE3 digest behind [`compute_content_hash`].
///
/// Callers that store the hash (the v2 fingerprint sidecar) keep the 32 bytes
/// instead of the 64-character hex form.
pub fn compute_content_digest(indexable_text: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hash_field(&mut hasher, b"domain", b"nvdb.content.v1");
    hash_field(&mut hasher, b"text", indexable_text.as_bytes());
    finalize_digest(hasher)
}

/// Deterministic envelope hash: BLAKE3 over the canonicalised full envelope.
///
/// Covers `id`, `parent_id`, `chunk_index`, `kind`, `content_hash`, the sorted
/// and deduplicated `acl`, canonical JSON `metadata` (object keys sorted), and
/// both schema/analyzer versions. This gates no-op skipping — an ACL- or
/// metadata-only change always changes this hash and is therefore applied.
pub fn compute_envelope_hash(envelope: &IndexedEnvelope) -> String {
    to_hex(&compute_envelope_digest(envelope))
}

/// Raw BLAKE3 digest behind [`compute_envelope_hash`].
pub fn compute_envelope_digest(envelope: &IndexedEnvelope) -> [u8; 32] {
    let content_hash = compute_content_hash(&envelope.indexable_text);
    let mut hasher = blake3::Hasher::new();
    hash_field(&mut hasher, b"domain", b"nvdb.envelope.v1");
    hash_field(&mut hasher, b"id", envelope.id.as_bytes());

    match &envelope.parent_id {
        Some(parent) => {
            hash_field(&mut hasher, b"parent_id.present", b"1");
            hash_field(&mut hasher, b"parent_id", parent.as_bytes());
        }
        None => hash_field(&mut hasher, b"parent_id.present", b"0"),
    }
    match envelope.chunk_index {
        Some(index) => {
            hash_field(&mut hasher, b"chunk_index.present", b"1");
            hash_field(&mut hasher, b"chunk_index", &index.to_le_bytes());
        }
        None => hash_field(&mut hasher, b"chunk_index.present", b"0"),
    }

    hash_field(&mut hasher, b"kind", envelope.kind.as_bytes());
    hash_field(&mut hasher, b"content_hash", content_hash.as_bytes());

    let mut acl: Vec<&str> = envelope.acl.iter().map(String::as_str).collect();
    acl.sort_unstable();
    acl.dedup();
    hash_field(&mut hasher, b"acl.len", &(acl.len() as u64).to_le_bytes());
    for group in acl {
        hash_field(&mut hasher, b"acl", group.as_bytes());
    }

    let mut metadata = String::new();
    write_canonical_json(&envelope.metadata, &mut metadata);
    hash_field(&mut hasher, b"metadata", metadata.as_bytes());

    hash_field(
        &mut hasher,
        b"schema_version",
        &envelope.schema_version.to_le_bytes(),
    );
    hash_field(
        &mut hasher,
        b"analyzer_version",
        &envelope.analyzer_version.to_le_bytes(),
    );

    finalize_digest(hasher)
}

fn finalize_digest(hasher: blake3::Hasher) -> [u8; 32] {
    *hasher.finalize().as_bytes()
}

fn to_hex(digest: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

fn hash_field(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn write_canonical_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => write_json_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            // serde_json maps are BTree-backed today, but sorting explicitly
            // keeps the hash stable if `preserve_order` is ever enabled by a
            // dependency in the same build.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            out.push('{');
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_json_string(key, out);
                out.push(':');
                write_canonical_json(&map[*key], out);
            }
            out.push('}');
        }
    }
}

fn write_json_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            control if (control as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn envelope(acl: &[&str], metadata: Value) -> IndexedEnvelope {
        IndexedEnvelope {
            id: "doc".to_string(),
            parent_id: Some("parent".to_string()),
            chunk_index: Some(0),
            kind: "doc".to_string(),
            indexable_text: "hello world".to_string(),
            acl: acl.iter().map(|group| group.to_string()).collect(),
            metadata,
            schema_version: 1,
            analyzer_version: 1,
        }
    }

    #[test]
    fn content_hash_ignores_acl_and_metadata() {
        let plain = envelope(&["eng"], json!({}));
        let locked = envelope(&["hr", "legal"], json!({"secret": true}));
        assert_eq!(
            compute_content_hash(&plain.indexable_text),
            compute_content_hash(&locked.indexable_text)
        );
    }

    #[test]
    fn envelope_hash_is_acl_order_insensitive_and_dedups() {
        let first = envelope(&["eng", "ops"], json!({"a": 1}));
        let second = envelope(&["ops", "eng", "ops"], json!({"a": 1}));
        assert_eq!(compute_envelope_hash(&first), compute_envelope_hash(&second));
        assert_ne!(
            compute_envelope_hash(&first),
            compute_envelope_hash(&envelope(&["eng"], json!({"a": 1})))
        );
    }

    #[test]
    fn envelope_hash_is_metadata_key_order_insensitive() {
        let first = envelope(&["eng"], json!({"b": 2, "a": 1}));
        let second = envelope(&["eng"], json!({"a": 1, "b": 2}));
        assert_eq!(compute_envelope_hash(&first), compute_envelope_hash(&second));
    }

    #[test]
    fn envelope_hash_separates_content_and_metadata() {
        let base = envelope(&["eng"], json!({"a": 1}));
        let mut metadata_changed = base.clone();
        metadata_changed.metadata = json!({"a": 2});
        let mut text_changed = base.clone();
        text_changed.indexable_text = "hello world v2".to_string();

        assert_ne!(
            compute_envelope_hash(&base),
            compute_envelope_hash(&metadata_changed)
        );
        assert_eq!(
            compute_content_hash(&base.indexable_text),
            compute_content_hash(&metadata_changed.indexable_text)
        );
        assert_ne!(
            compute_content_hash(&base.indexable_text),
            compute_content_hash(&text_changed.indexable_text)
        );
    }

    #[test]
    fn watermark_round_trips_and_validates() {
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join("nested").join("watermark.json");

        let watermark = SyncWatermark {
            source_epoch_uuid: "epoch".to_string(),
            synced_seq: 42,
            degraded: true,
            degraded_reason: Some("apply failed".to_string()),
        };
        watermark.save_atomic(&path).unwrap();
        assert_eq!(SyncWatermark::load(&path).unwrap(), watermark);

        let inconsistent = SyncWatermark {
            source_epoch_uuid: "epoch".to_string(),
            synced_seq: 42,
            degraded: false,
            degraded_reason: Some("stale reason".to_string()),
        };
        inconsistent.save_atomic(&path).unwrap();
        assert!(matches!(
            SyncWatermark::load(&path),
            Err(SyncError::Io(_))
        ));
    }
}
