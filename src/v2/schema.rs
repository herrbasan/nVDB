//! `nVDB` v2 collection configuration and indexable document.
//!
//! [`V2CollectionConfig`] is persisted verbatim in the collection manifest and
//! is part of the open-time contract: a schema, analyzer or lexicon version
//! mismatch refuses to open instead of silently serving differently-analysed
//! results.
//!
//! [`V2Document`] is the write-side shape. It carries the core envelope
//! (`id`, `parent_id`, `chunk_index`, `kind`), the searchable text, the
//! columnar filter/sort columns, an open metadata bag, and an optional excerpt.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::sync::{compute_content_hash, compute_envelope_hash, IndexedEnvelope};
use crate::v2::german_lexicon::GERMAN_LEXICON_VERSION;

/// Schema version understood by this build.
///
/// Bumped to 2 when `body`/`headings`/`body_parts` stopped being stored and the
/// sidecars moved to a compact binary format: a v1 directory is refused at open
/// instead of being read with v2 assumptions.
pub const CURRENT_SCHEMA_VERSION: u32 = 2;

/// Analyzer version understood by this build.
///
/// Bumped to 2 when `nvdb_text` moved from "folded form *plus* lowered form at
/// the same position" to one canonical folded token per word.
pub const CURRENT_ANALYZER_VERSION: u32 = 2;

/// Weighted contribution of every searchable field to the Phase A query.
///
/// Title exactness dominates; the open body is the baseline. `body_parts`
/// carries its own smaller weight because it holds derived German morphemes,
/// not original text — it must stay strictly below `body` so an exact compound
/// match always outranks a document that merely contains the parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FieldWeights {
    /// Whole-title exact match weight.
    pub title_exact: f32,
    /// Analyzed title weight.
    pub title: f32,
    /// Path/identifier weight.
    pub path: f32,
    /// Display-name weight.
    pub name: f32,
    /// Heading text weight.
    pub headings: f32,
    /// Category weight.
    pub categories: f32,
    /// Tag weight.
    pub tags: f32,
    /// Body text weight.
    pub body: f32,
    /// Derived German compound-part weight.
    pub body_parts: f32,
}

impl Default for FieldWeights {
    fn default() -> Self {
        Self {
            title_exact: 8.0,
            title: 4.0,
            path: 3.0,
            name: 3.0,
            headings: 2.5,
            categories: 2.0,
            tags: 2.0,
            body: 1.0,
            body_parts: 0.35,
        }
    }
}

impl FieldWeights {
    /// Weight of a searchable field by its schema name, if it is weighted.
    pub fn weight_for(&self, field: &str) -> Option<f32> {
        match field {
            "title_exact" => Some(self.title_exact),
            "title" => Some(self.title),
            "path" => Some(self.path),
            "name" => Some(self.name),
            "headings" => Some(self.headings),
            "categories" => Some(self.categories),
            "tags" => Some(self.tags),
            "body" => Some(self.body),
            "body_parts" => Some(self.body_parts),
            _ => None,
        }
    }
}

/// Type of a promoted metadata key.
///
/// A promoted key is indexed as a typed column so it can be filtered without
/// scanning the stored metadata bag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromotedFieldType {
    /// UTF-8 text, matched as a lowercased whole value.
    String,
    /// Signed 64-bit integer.
    I64,
    /// 64-bit float.
    F64,
    /// Boolean.
    Bool,
}

/// A metadata key promoted to a typed, filterable column.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotedField {
    /// Metadata key to promote.
    pub name: String,
    /// Column type the key is indexed as.
    pub field_type: PromotedFieldType,
}

impl PromotedField {
    /// Create a promoted field declaration.
    pub fn new(name: impl Into<String>, field_type: PromotedFieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
        }
    }
}

/// Default ceiling below which the vector lane scans every embedding.
pub const DEFAULT_FLAT_SEARCH_CUTOFF: usize = 50_000;

/// An optional dense vector lane.
///
/// The lane is keyed by content hash, so it is invisible to lexical-only work:
/// a collection with no lane never allocates a vector file, never scans for
/// pending embeddings and answers `VectorLaneNotConfigured` instead of guessing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VectorLaneConfig {
    /// Embedding dimension; every attached vector must match exactly.
    pub dim: usize,
    /// Metric used to score a query vector against a stored embedding.
    pub distance: crate::distance::Distance,
    /// Model/version that produced the embeddings; part of the store key.
    pub model_version: String,
    /// Embedding count above which an ANN lane would be built.
    ///
    /// v2 has no ANN lane yet, so a collection above the cutoff still scans
    /// flat — the field records the intended policy rather than silently
    /// claiming an index that does not exist.
    pub flat_search_cutoff: usize,
}

impl Default for VectorLaneConfig {
    fn default() -> Self {
        Self {
            dim: 0,
            distance: crate::distance::Distance::Cosine,
            model_version: String::new(),
            flat_search_cutoff: DEFAULT_FLAT_SEARCH_CUTOFF,
        }
    }
}

impl VectorLaneConfig {
    /// Cosine lane for `dim`-dimensional embeddings from `model_version`.
    pub fn new(dim: usize, model_version: impl Into<String>) -> Self {
        Self {
            dim,
            model_version: model_version.into(),
            ..Self::default()
        }
    }

    /// Replace the distance metric.
    pub fn distance(mut self, distance: crate::distance::Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Replace the flat-search cutoff.
    pub fn flat_search_cutoff(mut self, cutoff: usize) -> Self {
        self.flat_search_cutoff = cutoff;
        self
    }

    /// Reject a lane that could never score a vector.
    ///
    /// Runs at create and open, so a zero dimension or an empty model version
    /// fails at the boundary instead of at the first attach.
    pub fn validate(&self) -> Result<(), String> {
        if self.dim == 0 {
            return Err("vector lane dim must be greater than zero".to_string());
        }
        if self.model_version.trim().is_empty() {
            return Err("vector lane model_version must not be empty".to_string());
        }
        if self.flat_search_cutoff == 0 {
            return Err("vector lane flat_search_cutoff must be greater than zero".to_string());
        }
        Ok(())
    }
}

/// Persistent configuration of a v2 collection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct V2CollectionConfig {
    /// Envelope schema version; a bump invalidates skip decisions.
    pub schema_version: u32,
    /// Analyzer version; a bump invalidates every indexed token.
    pub analyzer_version: u32,
    /// German lexicon version; see [`GERMAN_LEXICON_VERSION`].
    pub lexicon_version: u32,
    /// Phase A field weights.
    pub field_weights: FieldWeights,
    /// Metadata keys indexed as typed columns.
    pub promoted_fields: Vec<PromotedField>,
    /// When true, a document's excerpt (up to 1024 bytes) is stored.
    pub store_excerpt: bool,
    /// Optional dense vector lane; `None` makes the collection lexical-only.
    pub vector_lane: Option<VectorLaneConfig>,
}

impl Default for V2CollectionConfig {
    fn default() -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            analyzer_version: CURRENT_ANALYZER_VERSION,
            lexicon_version: GERMAN_LEXICON_VERSION,
            field_weights: FieldWeights::default(),
            promoted_fields: Vec::new(),
            store_excerpt: false,
            vector_lane: None,
        }
    }
}

impl V2CollectionConfig {
    /// Config with the given promoted fields and every other default.
    pub fn with_promoted(promoted_fields: Vec<PromotedField>) -> Self {
        Self {
            promoted_fields,
            ..Self::default()
        }
    }

    /// Config with a dense vector lane and every other default.
    pub fn with_vector_lane(vector_lane: VectorLaneConfig) -> Self {
        Self {
            vector_lane: Some(vector_lane),
            ..Self::default()
        }
    }
}

/// Maximum stored excerpt length, in bytes.
pub const MAX_EXCERPT_BYTES: usize = 1024;

/// Normalise a file-extension column value: lowercase, leading dot stripped.
///
/// Both the write path and the query path use this, so `.MD`, `md` and ` .md `
/// are the same column value.
pub fn normalize_extension(ext: &str) -> String {
    ext.trim().trim_start_matches('.').to_lowercase()
}

/// One indexable document (or chunk) of a v2 collection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2Document {
    /// Document/chunk identifier, unique inside the collection.
    pub id: String,
    /// Owning document for a chunk, if this document is a chunk.
    #[serde(default)]
    pub parent_id: Option<String>,
    /// Chunk position inside the parent, if applicable.
    #[serde(default)]
    pub chunk_index: Option<u32>,
    /// Document kind (`doc`, `memory`, `chat`, `image`, `record`, ...).
    pub kind: String,
    /// Title text, also indexed verbatim for exact-title matching.
    #[serde(default)]
    pub title: String,
    /// Display name (file name, entity name, ...).
    #[serde(default)]
    pub name: String,
    /// Path or identifier.
    #[serde(default)]
    pub path: String,
    /// Flattened heading text.
    #[serde(default)]
    pub headings: String,
    /// Categories.
    #[serde(default)]
    pub categories: Vec<String>,
    /// Tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Body text.
    #[serde(default)]
    pub body: String,
    /// Containing folder, as a columnar filter value.
    #[serde(default)]
    pub folder: Option<String>,
    /// File extension. Normalised on write/filter: lowercase, no leading dot.
    #[serde(default)]
    pub ext: Option<String>,
    /// MIME type, as a columnar filter value.
    #[serde(default)]
    pub mime: Option<String>,
    /// Lifecycle status, as a columnar filter value.
    #[serde(default)]
    pub status: Option<String>,
    /// Creation timestamp (unix seconds or milliseconds — the caller decides).
    #[serde(default)]
    pub created_at: Option<i64>,
    /// Modification timestamp.
    #[serde(default)]
    pub updated_at: Option<i64>,
    /// Group IDs allowed to read this document. Never user IDs.
    #[serde(default)]
    pub acl: Vec<String>,
    /// Open metadata bag: stored, not searched; promoted keys are also columns.
    #[serde(default = "empty_metadata")]
    pub metadata: Value,
    /// Optional short excerpt.
    #[serde(default)]
    pub excerpt: Option<String>,
}

fn empty_metadata() -> Value {
    Value::Object(serde_json::Map::new())
}

impl V2Document {
    /// Create a document with only `id` and `kind` populated.
    pub fn new(id: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            parent_id: None,
            chunk_index: None,
            kind: kind.into(),
            title: String::new(),
            name: String::new(),
            path: String::new(),
            headings: String::new(),
            categories: Vec::new(),
            tags: Vec::new(),
            body: String::new(),
            folder: None,
            ext: None,
            mime: None,
            status: None,
            created_at: None,
            updated_at: None,
            acl: Vec::new(),
            metadata: empty_metadata(),
            excerpt: None,
        }
    }

    /// Deterministic concatenation of every searchable text field.
    ///
    /// This is the input of [`V2Document::content_hash`]: it gates embedding
    /// reuse, so ACL, metadata and column-only changes deliberately leave it
    /// untouched.
    pub fn indexable_text(&self) -> String {
        let mut text = String::with_capacity(
            self.title.len()
                + self.name.len()
                + self.path.len()
                + self.headings.len()
                + self.body.len()
                + 32,
        );
        for part in [&self.title, &self.name, &self.path, &self.headings] {
            text.push_str(part);
            text.push('\n');
        }
        text.push_str(&self.categories.join(" "));
        text.push('\n');
        text.push_str(&self.tags.join(" "));
        text.push('\n');
        text.push_str(&self.body);
        text
    }

    /// Hash of the searchable text only ([`crate::sync::compute_content_hash`]).
    pub fn content_hash(&self) -> String {
        compute_content_hash(&self.indexable_text())
    }

    /// Hash of the canonicalised envelope
    /// ([`crate::sync::compute_envelope_hash`]).
    ///
    /// Uses [`CURRENT_SCHEMA_VERSION`] and [`CURRENT_ANALYZER_VERSION`]; the
    /// engine hashes with the *configured* versions through
    /// [`V2Document::to_indexed_envelope`] instead.
    pub fn envelope_hash(&self) -> String {
        compute_envelope_hash(
            &self.to_indexed_envelope(CURRENT_SCHEMA_VERSION, CURRENT_ANALYZER_VERSION),
        )
    }

    /// Normalised extension column: lowercase, leading dot stripped.
    pub fn normalized_ext(&self) -> Option<String> {
        let ext = self.ext.as_deref()?;
        let normalized = normalize_extension(ext);
        if normalized.is_empty() {
            None
        } else {
            Some(normalized)
        }
    }

    /// Excerpt truncated to [`MAX_EXCERPT_BYTES`] at a character boundary, if
    /// one is present.
    pub fn bounded_excerpt(&self) -> Option<String> {
        let excerpt = self.excerpt.as_deref()?;
        if excerpt.len() <= MAX_EXCERPT_BYTES {
            return Some(excerpt.to_string());
        }
        let mut end = MAX_EXCERPT_BYTES;
        while end > 0 && !excerpt.is_char_boundary(end) {
            end -= 1;
        }
        Some(excerpt[..end].to_string())
    }

    /// Map onto the Phase 0 [`IndexedEnvelope`] used for hashing and sync.
    ///
    /// `indexable_text` holds every searchable field; `metadata` stays the open
    /// bag; `acl` is carried verbatim (the envelope hash sorts and dedups it).
    pub fn to_indexed_envelope(
        &self,
        schema_version: u32,
        analyzer_version: u32,
    ) -> IndexedEnvelope {
        IndexedEnvelope {
            id: self.id.clone(),
            parent_id: self.parent_id.clone(),
            chunk_index: self.chunk_index,
            kind: self.kind.clone(),
            indexable_text: self.indexable_text(),
            acl: self.acl.clone(),
            metadata: self.metadata.clone(),
            schema_version,
            analyzer_version,
        }
    }
}

/// Hash of an envelope with its ACL removed.
///
/// Two envelopes that differ only in `acl` share this hash, which is exactly
/// what the ACL-only sync fast path tests for. `compare_without_acl` exists so
/// both [`crate::sync::IndexedEnvelope`] and [`V2Document`] can use it.
pub fn envelope_hash_without_acl(envelope: &IndexedEnvelope) -> String {
    let mut stripped = envelope.clone();
    stripped.acl.clear();
    compute_envelope_hash(&stripped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_documented_weights() {
        let weights = FieldWeights::default();
        assert_eq!(weights.title_exact, 8.0);
        assert_eq!(weights.body_parts, 0.35);
        assert!(
            weights.body_parts < weights.body && weights.body < weights.title,
            "derived parts must rank below the original text"
        );
        assert_eq!(weights.weight_for("title"), Some(4.0));
        assert_eq!(weights.weight_for("unknown"), None);
    }

    #[test]
    fn config_defaults_carry_the_frozen_versions() {
        let config = V2CollectionConfig::default();
        assert_eq!(config.schema_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(config.analyzer_version, CURRENT_ANALYZER_VERSION);
        assert_eq!(config.lexicon_version, GERMAN_LEXICON_VERSION);
        assert!(config.promoted_fields.is_empty());
        assert!(!config.store_excerpt);
    }

    #[test]
    fn ext_is_normalised() {
        let mut doc = V2Document::new("doc", "file");
        doc.ext = Some(".MD".to_string());
        assert_eq!(doc.normalized_ext(), Some("md".to_string()));
        doc.ext = Some(" .md ".to_string());
        assert_eq!(doc.normalized_ext(), Some("md".to_string()));
        doc.ext = Some("".to_string());
        assert_eq!(doc.normalized_ext(), None);
    }

    #[test]
    fn content_hash_ignores_acl_and_columns_but_not_text() {
        let mut base = V2Document::new("doc", "file");
        base.title = "Quarterly Report".to_string();
        base.body = "revenue grew".to_string();

        let mut acl_changed = base.clone();
        acl_changed.acl = vec!["hr".to_string()];
        acl_changed.folder = Some("finance".to_string());
        assert_eq!(base.content_hash(), acl_changed.content_hash());
        assert_ne!(base.envelope_hash(), acl_changed.envelope_hash());

        let mut text_changed = base.clone();
        text_changed.body = "revenue fell".to_string();
        assert_ne!(base.content_hash(), text_changed.content_hash());
    }

    #[test]
    fn acl_only_changes_share_the_no_acl_hash() {
        let mut base = V2Document::new("doc", "file");
        base.body = "text".to_string();
        let mut acl_changed = base.clone();
        acl_changed.acl = vec!["group".to_string()];

        let first = base.to_indexed_envelope(1, 1);
        let second = acl_changed.to_indexed_envelope(1, 1);
        assert_eq!(
            envelope_hash_without_acl(&first),
            envelope_hash_without_acl(&second)
        );
        assert_ne!(
            compute_envelope_hash(&first),
            compute_envelope_hash(&second)
        );
    }

    #[test]
    fn excerpts_are_bounded_on_char_boundaries() {
        let mut doc = V2Document::new("doc", "file");
        doc.excerpt = Some("ä".repeat(MAX_EXCERPT_BYTES));
        let bounded = doc.bounded_excerpt().expect("excerpt");
        assert!(bounded.len() <= MAX_EXCERPT_BYTES);
        assert!(bounded.chars().all(|character| character == 'ä'));
    }
}
