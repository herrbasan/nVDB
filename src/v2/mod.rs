//! `nVDB` v2: lexical retrieval embedded on Tantivy, behind the Phase 0
//! sync/snapshot/ACL contract.
//!
//! The v1 surface ([`crate::Collection`], [`crate::lexical`], [`crate::sync`])
//! is untouched. The v2 surface is additive and owns one responsibility:
//! serving a *derived* lexical index that is always rebuildable from an
//! authoritative source.
//!
//! # Modules
//!
//! - [`schema`] — collection config, indexed document, hashes.
//! - [`query`] — query, filter, hit and explanation types.
//! - [`analyzer`] — the three built-in analyzers and German part extraction.
//! - [`german_lexicon`] — the frozen, versioned compound morpheme lexicon.
//! - [`acl_overlay`] — generation-stamped mutable ACL sidecar.
//! - [`vectors`] — `content_hash`-keyed vector store and generation-stamped
//!   `vec_map` sidecar.
//! - [`level2`] — Level-2 refinement: authoritative hydration, ACL re-check and
//!   pluggable reranking over Level-1 hybrid hits.
//! - [`engine`] — [`V2Index`]: manifest, Phase A retrieval, Phase B rescoring,
//!   hybrid fusion, Phase 0 sync integration and integrity checks.
//! - [`server`] — the resident localhost HTTP surface (`nvdb serve`).
//!
//! # Example
//!
//! ```no_run
//! use nvdb::v2::{V2CollectionConfig, V2Document, V2Index, V2Query};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut index = V2Index::create_or_open("data/lexical", V2CollectionConfig::default())?;
//!
//! let mut doc = V2Document::new("doc-1", "doc");
//! doc.title = "Förderprogramm Digitalisierung".to_string();
//! doc.body = "Die Richtlinie beschreibt das Förderprogramm.".to_string();
//! doc.acl = vec!["public".to_string()];
//! index.upsert(doc)?;
//! index.commit()?;
//!
//! let hits = index.search(&V2Query::new("foerder").principals(["public"]))?;
//! assert_eq!(hits.len(), 1);
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;

use thiserror::Error;

use crate::sync::SyncError;

pub mod acl_overlay;
pub mod analyzer;
pub mod engine;
pub mod german_lexicon;
pub mod level2;
pub mod query;
pub mod schema;
pub mod server;
pub mod vectors;

pub use acl_overlay::AclOverlay;
pub use analyzer::{body_part_terms, extract_body_parts, MAX_BODY_PARTS};
pub use engine::{DocFingerprint, IntegrityReport, V2Index, V2Manifest, V2_MANIFEST_FILE};
pub use german_lexicon::{decompose_german_compound, GERMAN_LEXICON_VERSION};
pub use level2::{V2Level2Hit, V2Level2Options, V2Level2Result};
pub use query::{
    GroupByParent, PromotedFilter, PromotedOp, PromotedValue, TimeRange, V2Explain, V2Filters,
    V2HybridHit, V2HybridQuery, V2Query, V2SearchHit, DEFAULT_CANDIDATE_K, DEFAULT_RRF_K,
};
pub use schema::{
    FieldWeights, PromotedField, PromotedFieldType, V2CollectionConfig, V2Document,
    VectorLaneConfig, CURRENT_ANALYZER_VERSION, CURRENT_SCHEMA_VERSION, DEFAULT_FLAT_SEARCH_CUTOFF,
    MAX_EXCERPT_BYTES,
};
pub use vectors::{
    digest_to_hex, hex_to_digest, vec_map_path, ContentDigest, PendingEmbedding, VecMap,
    VectorStore, VECTORS_FILE, VEC_MAP_PREFIX, VEC_MAP_SUFFIX,
};

/// Errors of the v2 engine.
#[derive(Debug, Error)]
pub enum V2Error {
    /// The persisted collection was built with different versions.
    #[error("version mismatch for {field}: manifest has {found}, expected {expected}")]
    VersionMismatch {
        /// Which version differs.
        field: &'static str,
        /// Version the caller asked for.
        expected: u32,
        /// Version found in the manifest.
        found: u32,
    },

    /// The persisted schema does not match the requested config.
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),

    /// The collection could not be opened because no manifest exists.
    #[error("no v2 collection manifest at {}", path.display())]
    NotOpen {
        /// Expected manifest path.
        path: PathBuf,
    },

    /// The collection config is not usable.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// The persisted state is unreadable (missing overlay generation, ...).
    #[error("corrupt collection state: {0}")]
    Corrupt(String),

    /// The query is not usable.
    #[error("invalid query: {0}")]
    InvalidQuery(String),

    /// A query carried no principals. Never widen to "match everything".
    #[error("missing principals: queries must carry at least one principal")]
    MissingPrincipals,

    /// A query vector does not match the configured lane dimension.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    WrongDimension {
        /// Lane dimension.
        expected: usize,
        /// Dimension the caller supplied.
        got: usize,
    },

    /// The dense lane was requested on a collection without one.
    #[error("this collection has no vector lane: set V2CollectionConfig::vector_lane")]
    VectorLaneNotConfigured,

    /// A vector completion no longer applies to the live document it names.
    #[error("stale vector attachment for '{id}': {reason}")]
    StaleVectorCas {
        /// Document the attachment named.
        id: String,
        /// Why the compare-and-set failed.
        reason: String,
    },

    /// The reranker failed or broke its contract.
    #[error("reranker failure: {0}")]
    Rerank(String),

    /// The index is degraded after an apply failure and refuses reads.
    #[error("index degraded: {reason}")]
    IndexDegraded {
        /// Why the index is degraded.
        reason: String,
    },

    /// A promoted column was given a value of the wrong type.
    #[error("promoted field '{field}' expects {expected} but the input is {found}")]
    PromotedFieldType {
        /// Promoted field name.
        field: String,
        /// Declared column type.
        expected: &'static str,
        /// Type actually supplied.
        found: String,
    },

    /// Phase 0 sync contract failure.
    #[error(transparent)]
    Sync(#[from] SyncError),

    /// Tantivy failure.
    #[error(transparent)]
    Tantivy(#[from] tantivy::TantivyError),

    /// Local filesystem failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// (De)serialisation failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}
