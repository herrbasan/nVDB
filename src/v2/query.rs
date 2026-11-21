//! Query, filter and result types of the v2 engine.
//!
//! [`V2Query`] is deliberately fail-closed: [`V2Query::principals`] must carry
//! at least one group ID, otherwise
//! [`crate::v2::V2Index::search`] returns
//! [`crate::v2::V2Error::MissingPrincipals`] instead of widening to "match
//! everything".

use serde::{Deserialize, Serialize};

use crate::v2::schema::PromotedFieldType;

/// Cap on how many chunks may be returned per `parent_id`.
///
/// Documents without a `parent_id` are never grouped and never capped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupByParent {
    /// Maximum hits sharing one `parent_id`.
    pub max_per_parent: usize,
}

impl GroupByParent {
    /// Create a diversity cap.
    pub fn new(max_per_parent: usize) -> Self {
        Self { max_per_parent }
    }
}

/// Inclusive range over a timestamp column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TimeRange {
    /// Inclusive lower bound.
    pub min: Option<i64>,
    /// Inclusive upper bound.
    pub max: Option<i64>,
}

impl TimeRange {
    /// Inclusive `[min, max]` range.
    pub fn between(min: i64, max: i64) -> Self {
        Self {
            min: Some(min),
            max: Some(max),
        }
    }

    /// Open-ended lower bound.
    pub fn from(min: i64) -> Self {
        Self {
            min: Some(min),
            max: None,
        }
    }

    /// Open-ended upper bound.
    pub fn until(max: i64) -> Self {
        Self {
            min: None,
            max: Some(max),
        }
    }
}

/// Comparison operator of a promoted-field filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromotedOp {
    /// Equality (the only operator for `String` and `Bool`).
    Eq,
    /// Greater than or equal (numeric only).
    Gte,
    /// Less than or equal (numeric only).
    Lte,
}

/// Typed value of a promoted-field filter.
///
/// The JSON form is untagged: `"text"`, `42`, `4.2` and `true` deserialize to
/// the matching variant. `I64` is tried before `F64`, so an integral literal
/// stays an integer instead of silently widening.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromotedValue {
    /// Text value.
    String(String),
    /// Signed integer value.
    I64(i64),
    /// Float value.
    F64(f64),
    /// Boolean value.
    Bool(bool),
}

impl PromotedValue {
    /// Column type this value is compatible with.
    pub fn field_type(&self) -> PromotedFieldType {
        match self {
            PromotedValue::String(_) => PromotedFieldType::String,
            PromotedValue::I64(_) => PromotedFieldType::I64,
            PromotedValue::F64(_) => PromotedFieldType::F64,
            PromotedValue::Bool(_) => PromotedFieldType::Bool,
        }
    }

    /// True when this value can be used with `operator`.
    pub fn supports(&self, operator: PromotedOp) -> bool {
        match operator {
            PromotedOp::Eq => true,
            PromotedOp::Gte | PromotedOp::Lte => {
                !matches!(self, PromotedValue::String(_) | PromotedValue::Bool(_))
            }
        }
    }
}

/// One predicate over a promoted metadata column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotedFilter {
    /// Promoted field name, as declared in the collection config.
    pub field: String,
    /// Comparison operator.
    pub op: PromotedOp,
    /// Value to compare against.
    pub value: PromotedValue,
}

impl PromotedFilter {
    /// `field == value`.
    pub fn eq(field: impl Into<String>, value: impl Into<PromotedValue>) -> Self {
        Self {
            field: field.into(),
            op: PromotedOp::Eq,
            value: value.into(),
        }
    }

    /// `field >= value`.
    pub fn gte(field: impl Into<String>, value: impl Into<PromotedValue>) -> Self {
        Self {
            field: field.into(),
            op: PromotedOp::Gte,
            value: value.into(),
        }
    }

    /// `field <= value`.
    pub fn lte(field: impl Into<String>, value: impl Into<PromotedValue>) -> Self {
        Self {
            field: field.into(),
            op: PromotedOp::Lte,
            value: value.into(),
        }
    }
}

impl From<String> for PromotedValue {
    fn from(value: String) -> Self {
        PromotedValue::String(value)
    }
}

impl From<&str> for PromotedValue {
    fn from(value: &str) -> Self {
        PromotedValue::String(value.to_string())
    }
}

impl From<i64> for PromotedValue {
    fn from(value: i64) -> Self {
        PromotedValue::I64(value)
    }
}

impl From<i32> for PromotedValue {
    fn from(value: i32) -> Self {
        PromotedValue::I64(i64::from(value))
    }
}

impl From<f64> for PromotedValue {
    fn from(value: f64) -> Self {
        PromotedValue::F64(value)
    }
}

impl From<bool> for PromotedValue {
    fn from(value: bool) -> Self {
        PromotedValue::Bool(value)
    }
}

/// Columnar pre-filters applied as `Occur::Must` clauses in Phase A.
///
/// Every `Some` value narrows the candidate set inside Tantivy, so Phase B only
/// sees documents that already satisfy the filters.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct V2Filters {
    /// Exact containing folder (case-insensitive).
    pub folder: Option<String>,
    /// File extension; `.md` and `MD` both mean `md`.
    pub ext: Option<String>,
    /// Document kind.
    pub kind: Option<String>,
    /// Lifecycle status.
    pub status: Option<String>,
    /// MIME type.
    pub mime: Option<String>,
    /// Creation-time range.
    pub created_at: Option<TimeRange>,
    /// Modification-time range.
    pub updated_at: Option<TimeRange>,
    /// Promoted-column predicates (all must match).
    pub promoted: Vec<PromotedFilter>,
}

impl V2Filters {
    /// Empty filter set.
    pub fn none() -> Self {
        Self::default()
    }

    /// Restrict to one folder.
    pub fn folder(mut self, folder: impl Into<String>) -> Self {
        self.folder = Some(folder.into());
        self
    }

    /// Restrict to one extension.
    pub fn ext(mut self, ext: impl Into<String>) -> Self {
        self.ext = Some(ext.into());
        self
    }

    /// Restrict to one document kind.
    pub fn kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = Some(kind.into());
        self
    }

    /// Restrict to one status.
    pub fn status(mut self, status: impl Into<String>) -> Self {
        self.status = Some(status.into());
        self
    }

    /// Restrict to one MIME type.
    pub fn mime(mut self, mime: impl Into<String>) -> Self {
        self.mime = Some(mime.into());
        self
    }

    /// Restrict by creation time.
    pub fn created_at(mut self, range: TimeRange) -> Self {
        self.created_at = Some(range);
        self
    }

    /// Restrict by modification time.
    pub fn updated_at(mut self, range: TimeRange) -> Self {
        self.updated_at = Some(range);
        self
    }

    /// Add a promoted-column predicate.
    pub fn promoted(mut self, filter: PromotedFilter) -> Self {
        self.promoted.push(filter);
        self
    }

    /// True when at least one predicate narrows the candidate set.
    ///
    /// The vector lane uses this to decide between a filter-only posting walk
    /// and a plain ACL check over the content-hash map.
    pub fn is_active(&self) -> bool {
        self.folder.is_some()
            || self.ext.is_some()
            || self.kind.is_some()
            || self.status.is_some()
            || self.mime.is_some()
            || self.created_at.is_some()
            || self.updated_at.is_some()
            || !self.promoted.is_empty()
    }
}

/// Text query with filters, principals and Phase B controls.
///
/// The JSON form is the `serve` request body: every field is optional except
/// what the engine insists on, and an absent principal list stays empty so the
/// search fails closed with [`crate::v2::V2Error::MissingPrincipals`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct V2Query {
    /// Query text. Empty means "filter only".
    pub text: String,
    /// Maximum hits returned.
    pub top_k: usize,
    /// Group IDs the caller holds. Must not be empty.
    pub principals: Vec<String>,
    /// Levenshtein distance for the fuzzy expansion (`0` disables it).
    pub fuzzy_edits: u8,
    /// Attach a Phase B explanation to every hit.
    pub explain: bool,
    /// Diversity cap per `parent_id`.
    pub group_by_parent: Option<GroupByParent>,
    /// Columnar pre-filters.
    pub filters: V2Filters,
}

impl Default for V2Query {
    fn default() -> Self {
        Self::new("")
    }
}

impl V2Query {
    /// A query for `text` with `top_k = 10` and no principals.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            top_k: 10,
            principals: Vec::new(),
            fuzzy_edits: 0,
            explain: false,
            group_by_parent: None,
            filters: V2Filters::default(),
        }
    }

    /// Set the result limit.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the principals the caller holds.
    pub fn principals<I, S>(mut self, principals: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.principals = principals.into_iter().map(Into::into).collect();
        self
    }

    /// Set the fuzzy edit distance (`0` disables fuzzy expansion).
    ///
    /// At most the first six non-numeric terms of four characters or more are
    /// expanded, at a reduced weight, on `title` and `body`.
    pub fn fuzzy_edits(mut self, edits: u8) -> Self {
        self.fuzzy_edits = edits;
        self
    }

    /// Request a Phase B explanation per hit.
    pub fn explain(mut self, explain: bool) -> Self {
        self.explain = explain;
        self
    }

    /// Cap hits per `parent_id`.
    pub fn group_by_parent(mut self, max_per_parent: usize) -> Self {
        self.group_by_parent = Some(GroupByParent::new(max_per_parent));
        self
    }

    /// Set the columnar filters.
    pub fn filters(mut self, filters: V2Filters) -> Self {
        self.filters = filters;
        self
    }
}

/// One ranked search hit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2SearchHit {
    /// Document identifier.
    pub id: String,
    /// Owning document for a chunk.
    pub parent_id: Option<String>,
    /// Chunk position inside the parent.
    pub chunk_index: Option<u32>,
    /// Document kind.
    pub kind: String,
    /// Title as indexed.
    pub title: String,
    /// Path as indexed.
    pub path: String,
    /// Stored excerpt, when the collection stores excerpts.
    pub excerpt: Option<String>,
    /// Ranked score: the Phase A score the hit was ordered by.
    ///
    /// Phase B only enforces ACL and the diversity cap, so this is the Tantivy
    /// score of the weighted query (field weights, phrase boost and fuzzy
    /// expansion included).
    pub score: f32,
    /// Same value as [`V2SearchHit::score`], named for the explain block.
    pub bm25_score: f32,
    /// Phase B explanation, present when requested.
    pub explain: Option<V2Explain>,
}

/// Phase B explanation of one hit.
///
/// Computing it costs a tokenizer pass over the stored title and path, which is
/// why it is populated only when [`V2Query::explain`] is set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2Explain {
    /// Distinct query terms matched by the fields a hit exposes (title, path).
    pub matched_terms: Vec<String>,
    /// Distinct query terms the query produced.
    pub total_terms: usize,
    /// Fraction of query terms matched (`0.0..=1.0`).
    pub coverage: f32,
    /// Whether the verbatim multi-term query appears in the hit title.
    pub phrase_bonus: bool,
    /// Ranked score (see [`V2SearchHit::bm25_score`]).
    pub bm25_score: f32,
    /// Final score of the hit, equal to its ordered score.
    pub final_score: f32,
    /// Candidates removed by the ACL overlay before `top_k` winners existed.
    pub acl_filtered: usize,
    /// Candidates removed by the group-by-parent cap before `top_k` winners existed.
    pub diversity_filtered: usize,
    /// Candidate documents examined from the Phase A order until `top_k` won.
    pub candidates_considered: usize,
}

/// Default number of candidates each lane contributes to the fusion.
pub const DEFAULT_CANDIDATE_K: usize = 200;

/// Default RRF damping constant.
pub const DEFAULT_RRF_K: u32 = 60;

/// One unified lexical + dense query.
///
/// [`Self::lexical`] carries the text, principals, filters and the diversity
/// cap; the dense lane only adds a query vector. `dense` defaults to
/// `query_vector.is_some()` at deserialization time, so a client that sends a
/// vector gets the vector lane without having to repeat the flag, and a client
/// that sends neither stays lexical.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "V2HybridQueryWire")]
pub struct V2HybridQuery {
    /// Lexical query: text, principals, filters, `top_k`, diversity, fuzzy.
    pub lexical: V2Query,
    /// Query embedding; required when [`Self::dense`] is set.
    pub query_vector: Option<Vec<f32>>,
    /// Whether the dense lane runs.
    pub dense: bool,
    /// Whether the lexical lane runs.
    pub lexical_enabled: bool,
    /// Candidates each lane contributes before fusion.
    pub candidate_k: usize,
    /// RRF damping constant; higher flattens the rank contribution.
    pub rrf_k: u32,
    /// Weight of the lexical lane in the fused score.
    pub lexical_weight: f32,
    /// Weight of the dense lane in the fused score.
    pub vector_weight: f32,
}

/// Wire shape of [`V2HybridQuery`]: `dense` is optional so its default can
/// depend on whether a query vector was supplied.
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct V2HybridQueryWire {
    lexical: V2Query,
    query_vector: Option<Vec<f32>>,
    dense: Option<bool>,
    lexical_enabled: bool,
    candidate_k: usize,
    rrf_k: u32,
    lexical_weight: f32,
    vector_weight: f32,
}

impl Default for V2HybridQueryWire {
    fn default() -> Self {
        Self {
            lexical: V2Query::default(),
            query_vector: None,
            dense: None,
            lexical_enabled: true,
            candidate_k: DEFAULT_CANDIDATE_K,
            rrf_k: DEFAULT_RRF_K,
            lexical_weight: 1.0,
            vector_weight: 0.3,
        }
    }
}

impl From<V2HybridQueryWire> for V2HybridQuery {
    fn from(wire: V2HybridQueryWire) -> Self {
        let dense = wire.dense.unwrap_or(wire.query_vector.is_some());
        Self {
            lexical: wire.lexical,
            query_vector: wire.query_vector,
            dense,
            lexical_enabled: wire.lexical_enabled,
            candidate_k: wire.candidate_k,
            rrf_k: wire.rrf_k,
            lexical_weight: wire.lexical_weight,
            vector_weight: wire.vector_weight,
        }
    }
}

impl V2HybridQuery {
    /// A lexical-only hybrid query for `lexical`.
    pub fn new(lexical: V2Query) -> Self {
        Self {
            lexical,
            query_vector: None,
            dense: false,
            lexical_enabled: true,
            candidate_k: DEFAULT_CANDIDATE_K,
            rrf_k: DEFAULT_RRF_K,
            lexical_weight: 1.0,
            vector_weight: 0.3,
        }
    }

    /// Attach a query embedding and enable the dense lane.
    pub fn vector(mut self, vector: Vec<f32>) -> Self {
        self.query_vector = Some(vector);
        self.dense = true;
        self
    }

    /// Enable or disable the dense lane.
    pub fn dense(mut self, dense: bool) -> Self {
        self.dense = dense;
        self
    }

    /// Enable or disable the lexical lane.
    pub fn lexical_enabled(mut self, enabled: bool) -> Self {
        self.lexical_enabled = enabled;
        self
    }

    /// Candidates per lane before fusion.
    pub fn candidate_k(mut self, candidate_k: usize) -> Self {
        self.candidate_k = candidate_k;
        self
    }

    /// RRF damping constant.
    pub fn rrf_k(mut self, rrf_k: u32) -> Self {
        self.rrf_k = rrf_k;
        self
    }

    /// Lane weights of the fused score.
    pub fn weights(mut self, lexical_weight: f32, vector_weight: f32) -> Self {
        self.lexical_weight = lexical_weight;
        self.vector_weight = vector_weight;
        self
    }
}

/// One fused hybrid hit.
///
/// `hit.score` is the fused RRF score and `hit.bm25_score` mirrors the lexical
/// score (`0.0` when only the dense lane found the document). The per-lane
/// fields keep the evidence: which lanes found the document, at what rank and
/// with what raw score. A document found by one lane only is normal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2HybridHit {
    /// Hydrated envelope; `hit.score` carries the fused RRF score.
    pub hit: V2SearchHit,
    /// Raw lexical score (the Phase A BM25-derived score), when the lexical
    /// lane returned the document.
    pub lexical_score: Option<f32>,
    /// 1-based lexical rank, when the lexical lane returned the document.
    pub lexical_rank: Option<usize>,
    /// Higher-is-better dense similarity, when the dense lane returned it.
    pub vector_score: Option<f32>,
    /// 1-based dense rank, when the dense lane returned the document.
    pub vector_rank: Option<usize>,
    /// Reciprocal-rank-fusion score the fused order was built from.
    pub rrf_score: f32,
}

impl V2HybridHit {
    /// Document identifier (shorthand for `hit.hit.id`).
    pub fn id(&self) -> &str {
        &self.hit.id
    }
}
