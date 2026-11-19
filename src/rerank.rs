//! Optional query-document reranking.
//!
//! nVDB owns the retrieval funnel: callers provide a reranker implementation
//! and the payload field containing candidate text. The backend may be CPU or
//! GPU; the database only depends on this trait.

use crate::Result;

/// Scores candidate documents against the original query.
///
/// Implementations must return exactly one finite score per document. Higher
/// means more relevant.
pub trait Reranker {
    /// Score documents in input order.
    fn score(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>>;
}

/// Default blend weight for the reranker score.
pub const DEFAULT_RERANK_WEIGHT: f32 = 0.2;
/// Default number of fused candidates shown to the reranker.
pub const DEFAULT_RERANK_SHORTLIST: usize = 10;
