//! The v2 index engine.
//!
//! One [`V2Index`] owns
//!
//! - a Tantivy directory (immutable segments, BM25 Phase A retrieval),
//! - a mutable [`AclOverlay`] sidecar (authoritative, generation-stamped
//!   access decisions),
//! - a [`V2Manifest`] (`v2_manifest.json`, written atomically),
//! - per-document fingerprints (`v2_fingerprints.bin`) that gate the Phase 0
//!   no-op skip and the ACL-only fast path, and
//! - the optional vector lane: content-addressed embeddings (`vectors.bin`) plus
//!   the generation-stamped `doc_id -> content_hash` links
//!   (`vec_map.gen_<N>.bin`, named by `V2Manifest::vec_generation`).
//!
//! The index is *derived*: full content lives in `nDB`, so Tantivy stores only
//! the envelope needed to render a hit (`id`, `parent_id`, `chunk_index`,
//! `kind`, `title`, `path`, `name`, `folder`, `ext`, `mime`, `status`,
//! timestamps, non-empty metadata, and an opt-in bounded `excerpt`). `body`,
//! `headings` and `body_parts` are indexed but **not** stored.
//!
//! Retrieval is two-phase by design:
//!
//! **Phase A** is one weighted Tantivy `BooleanQuery` over `title_exact`,
//! `title`, `path`, `name`, `headings`, `categories`, `tags`, `body` and
//! `body_parts`, with `Occur::Must` columnar pre-filters, a postings-only
//! `PhraseQuery` boost for multi-term queries, and a capped single-edit fuzzy
//! expansion. It fetches up to `max(top_k * 20, 200)` candidates, widening
//! until enough survive.
//!
//! **Phase B** touches only what correctness needs: `id` and `parent_id` come
//! from fast fields, so the authoritative ACL overlay and the per-`parent_id`
//! cap are enforced on the postings order without decompressing a single stored
//! document. It stops as soon as `top_k` winners exist and hydrates the stored
//! envelope for those winners only — no re-tokenization, no body-part
//! extraction, no doc-store reads for rejected candidates.
//!
//! Reads are fail-closed: no principals, or a degraded watermark, is an error —
//! never a widened result set.
//!
//! Retrieval above Phase B is [`V2Index::hybrid_search`]: both lanes produce
//! `candidate_k` candidates, RRF fuses their ranks, `group_by_parent` is applied
//! once on the fused order, and only the winners are hydrated. The dense lane
//! obeys the same authority as the lexical one — the ACL overlay plus the
//! query's columnar filters — so no fusion weight can widen visibility.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::ops::Bound;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tantivy::collector::{DocSetCollector, TopDocs};
use tantivy::columnar::StrColumn;
use tantivy::query::{
    AllQuery, BooleanQuery, BoostQuery, FuzzyTermQuery, Occur, PhraseQuery, Query, RangeQuery,
    TermQuery,
};
use tantivy::schema::{
    Field, IndexRecordOption, NumericOptions, Schema, TextFieldIndexing, TextOptions,
    Value as SchemaValue,
};
use tantivy::{
    DocAddress, Index, IndexReader, IndexWriter, ReloadPolicy, Searcher, TantivyDocument, Term,
};

use crate::distance::Distance;
use crate::sync::{
    compute_content_digest, compute_envelope_digest, IndexedEnvelope, SourceChange, SourceChangeOp,
    SourceConnector, SourceLineage, SourceRecord, SyncError, SyncOutcome, SyncWatermark,
    VectorAttachment,
};
use crate::v2::acl_overlay::AclOverlay;
use crate::v2::analyzer::{
    body_part_terms, exact_token, extract_body_parts, identifier_tokens, register_tokenizers,
    text_tokens, TOKENIZER_EXACT, TOKENIZER_IDENTIFIER, TOKENIZER_TEXT,
};
use crate::v2::german_lexicon::fold_umlauts;
use crate::v2::query::{
    PromotedOp, PromotedValue, V2Explain, V2Filters, V2HybridHit, V2HybridQuery, V2Query,
    V2SearchHit,
};
use crate::v2::schema::{normalize_extension, PromotedFieldType, V2CollectionConfig, V2Document};
use crate::v2::vectors::{digest_to_hex, hex_to_digest, PendingEmbedding, VecMap, VectorStore};
use crate::v2::V2Error;

/// Manifest file name inside the collection directory.
pub const V2_MANIFEST_FILE: &str = "v2_manifest.json";

/// Fingerprint sidecar file name inside the collection directory.
pub const V2_FINGERPRINTS_FILE: &str = "v2_fingerprints.bin";

/// Magic bytes of the fingerprint sidecar.
const FINGERPRINT_MAGIC: &[u8; 4] = b"NVFP";

/// Format version of the fingerprint sidecar.
const FINGERPRINT_FORMAT_VERSION: u8 = 1;

/// Tantivy index writer heap budget.
const INDEX_WRITER_HEAP_BYTES: usize = 50_000_000;

/// Phase A candidate multiplier over `top_k`.
const CANDIDATE_MULTIPLIER: usize = 20;

/// Phase A minimum candidate count.
const CANDIDATE_MINIMUM: usize = 200;

/// Multiplier applied when too few candidates survive Phase B.
const WIDEN_FACTOR: usize = 5;

/// Maximum Phase A fetch rounds.
const WIDEN_ROUNDS: usize = 4;

/// Weight factor of the fuzzy expansion relative to the field weight.
const FUZZY_WEIGHT_FACTOR: f32 = 0.25;

/// Shortest query term eligible for fuzzy expansion.
const FUZZY_MIN_TERM_CHARS: usize = 4;

/// Maximum number of distinct query terms expanded into fuzzy clauses.
///
/// Each expansion is one FST walk; an unbounded long query would multiply that
/// cost for no ranking benefit.
const FUZZY_MAX_TERMS: usize = 6;

/// Score multiplier of the multi-term title phrase clause, relative to the title weight.
const TITLE_PHRASE_BOOST: f32 = 2.0;

/// Score multiplier of the multi-term body phrase clause, relative to the body weight.
/// High enough that an exact verbatim multi-token match in `body` (e.g. a code/directive
/// literal like `preset=narration:david-a-renelt`) outranks partial-subset matches in
/// `title` + `path` + `name` (`herrbasan/nVDB#4`, `herrbasan/mcp_server#88`).
const BODY_PHRASE_BOOST: f32 = 10.0;

/// Bonus multiplier for documents that contain every distinct query term (all-terms
/// conjunction across title/headings/path/name/body), preventing partial title/path hits
/// from drowning out full-coverage matches.
const ALL_TERMS_BOOST: f32 = 4.0;

/// Shortest query (in distinct terms) that gets a phrase or all-terms conjunction clause.
const PHRASE_MIN_TERMS: usize = 2;

/// Longest query (in tokens/distinct terms) that gets a phrase or all-terms conjunction clause.
/// Raised from 8 to 32 (herrbasan/nVDB#5): longer verbatim phrases (e.g. 12-word quotes)
/// silently lost both PhraseQuery and ALL_TERMS_BOOST at 8; postings-only PhraseQuery
/// short-circuits faster as term count grows.
const PHRASE_MAX_TERMS: usize = 32;

/// Minimum BM25 ratio over the 5th distinct-parent lexical hit for a candidate to count
/// as an exact lexical outlier (PhraseQuery / title_exact / all-terms spike) in hybrid RRF
/// when `lexical_weight < vector_weight` (herrbasan/nVDB#5). Flat topical queries sit around
/// 1.1x-1.4x; verbatim phrase/title hits land at 6x-45x.
const LEXICAL_SPIKE_RATIO: f32 = 2.5;
const LEXICAL_SPIKE_BASELINE_PARENTS: usize = 5;

/// Names that cannot be promoted, because the engine already uses them.
const RESERVED_FIELD_NAMES: &[&str] = &[
    "id",
    "parent_id",
    "chunk_index",
    "kind",
    "title_exact",
    "title",
    "name",
    "path",
    "headings",
    "categories",
    "tags",
    "body",
    "body_parts",
    "folder",
    "ext",
    "mime",
    "status",
    "created_at",
    "updated_at",
    "excerpt",
    "metadata_json",
];

/// Persisted collection state, written atomically at `<dir>/v2_manifest.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2Manifest {
    /// Collection configuration this index was built with.
    pub config: V2CollectionConfig,
    /// Generation of the live ACL overlay file.
    pub acl_generation: u64,
    /// Generation of the live `doc_id -> content_hash` vector map file.
    ///
    /// `0` means no generation was ever written; the field is defaulted so a
    /// Phase 2 manifest still opens.
    #[serde(default)]
    pub vec_generation: u64,
    /// Sync watermark; `None` until the first successful sync or rebuild.
    pub watermark: Option<SyncWatermark>,
    /// Number of live documents.
    pub doc_count: u64,
}

impl V2Manifest {
    /// Load the manifest of `dir`.
    pub fn load(dir: &Path) -> Result<Self, V2Error> {
        let path = manifest_path(dir);
        let bytes = fs::read(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                V2Error::NotOpen { path }
            } else {
                V2Error::Io(error)
            }
        })?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Persist the manifest atomically.
    pub fn save_atomic(&self, dir: &Path) -> Result<(), V2Error> {
        write_atomic(&manifest_path(dir), &serde_json::to_vec_pretty(self)?)
    }
}

/// Per-document hashes that gate skip decisions.
///
/// The three values are raw BLAKE3-256 digests; the hex form is what
/// [`compute_content_hash`] / [`compute_envelope_hash`] return. Keeping the raw
/// bytes means loading the sidecar performs no encoding work at all — it is on
/// the read-only open path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocFingerprint {
    /// Digest of the searchable text only.
    pub content_hash: [u8; 32],
    /// Digest of the canonicalised envelope.
    pub envelope_hash: [u8; 32],
    /// Digest of the envelope with its ACL removed.
    pub no_acl_hash: [u8; 32],
}

/// Result of [`V2Index::integrity_check`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IntegrityReport {
    /// Lineage epoch of the source at check time.
    pub source_epoch_uuid: String,
    /// Source head sequence at check time.
    pub source_last_seq: u64,
    /// Sequence the index watermark holds.
    pub watermark_seq: Option<u64>,
    /// Live documents in the source snapshot.
    pub source_doc_count: usize,
    /// Documents the index tracks.
    pub indexed_doc_count: usize,
    /// Documents Tantivy reports for the live reader.
    pub reader_doc_count: u64,
    /// Documents the ACL overlay holds.
    pub acl_doc_count: usize,
    /// Source documents the index does not know.
    pub missing: Vec<String>,
    /// Indexed documents the source no longer has.
    pub extra: Vec<String>,
    /// Documents whose stored hashes differ from the source.
    pub mismatched: Vec<String>,
    /// Documents whose overlay entry does not match the source ACL.
    pub acl_misaligned: Vec<String>,
}

impl IntegrityReport {
    /// True when the index and the source agree on every checked axis.
    pub fn is_clean(&self) -> bool {
        self.missing.is_empty()
            && self.extra.is_empty()
            && self.mismatched.is_empty()
            && self.acl_misaligned.is_empty()
            && self.reader_doc_count == self.indexed_doc_count as u64
            && self.acl_doc_count == self.indexed_doc_count
    }
}

/// One document's fingerprint as exposed to a synchronising caller.
///
/// The hex hashes are the same values [`crate::sync::compute_content_hash`] and
/// [`crate::sync::compute_envelope_hash`] produce, so a caller can compare them
/// against its own source-side hashes without decoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FingerprintEntry {
    /// Document identifier.
    pub id: String,
    /// Hex digest of the searchable text.
    pub content_hash: String,
    /// Hex digest of the canonicalised envelope (ACL included).
    pub envelope_hash: String,
    /// Whether the index holds an embedding for the live text.
    pub has_vector: bool,
    /// Group IDs the ACL overlay holds for the document.
    pub acl: Vec<String>,
}

/// One fingerprint a caller expects the index to hold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedFingerprint {
    /// Document identifier.
    pub id: String,
    /// Hex digest of the source's searchable text.
    pub content_hash: String,
    /// Hex digest of the source's canonicalised envelope.
    pub envelope_hash: String,
}

/// Difference between the caller's expected fingerprints and the live ones.
///
/// Every vector is sorted and deduplicated. `missing_vectors` names documents
/// the index *tracks* but holds no embedding for; a document that is missing
/// altogether is reported once, in [`Self::missing`], because it must be
/// upserted before an embedding can apply to it at all.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct IntegrityDiff {
    /// True when every difference vector is empty.
    pub clean: bool,
    /// Expected documents the index does not track.
    pub missing: Vec<String>,
    /// Tracked documents the caller did not expect.
    pub extra: Vec<String>,
    /// Documents whose searchable text hash differs.
    pub content_mismatched: Vec<String>,
    /// Documents whose envelope hash differs.
    pub envelope_mismatched: Vec<String>,
    /// Tracked documents without an embedding, when vectors were checked.
    pub missing_vectors: Vec<String>,
}

/// One promoted metadata column.
#[derive(Debug, Clone)]
struct PromotedColumn {
    name: String,
    field: Field,
    field_type: PromotedFieldType,
}

/// Field handles of the v2 schema.
#[derive(Debug, Clone)]
struct V2Fields {
    id: Field,
    parent_id: Field,
    chunk_index: Field,
    kind: Field,
    title_exact: Field,
    title: Field,
    name: Field,
    path: Field,
    headings: Field,
    categories: Field,
    tags: Field,
    body: Field,
    body_parts: Field,
    folder: Field,
    ext: Field,
    mime: Field,
    status: Field,
    created_at: Field,
    updated_at: Field,
    excerpt: Field,
    metadata_json: Field,
    promoted: Vec<PromotedColumn>,
}

impl V2Fields {
    fn promoted(&self, name: &str) -> Option<&PromotedColumn> {
        self.promoted.iter().find(|column| column.name == name)
    }
}

/// Lightweight per-segment fast-field handles used by Phase B.
///
/// Holding the columns per segment turns the ACL and diversity checks into one
/// dictionary lookup per candidate — no stored-document decompression.
struct SegmentLight {
    id: StrColumn,
    parent_id: Option<StrColumn>,
}

impl SegmentLight {
    fn open(searcher: &Searcher, segment_ord: u32) -> Result<Self, V2Error> {
        let segment = searcher.segment_reader(segment_ord);
        let fast_fields = segment.fast_fields();
        let id = fast_fields.str("id")?.ok_or_else(|| {
            V2Error::Corrupt(format!(
                "segment {segment_ord} has no fast 'id' column; the index was built by an incompatible schema"
            ))
        })?;
        Ok(Self {
            id,
            parent_id: fast_fields.str("parent_id")?,
        })
    }
}

/// One Phase B winner, before its stored envelope is hydrated.
struct Winner {
    score: f32,
    address: DocAddress,
    id: String,
    parent_id: Option<String>,
}

/// One accumulated fusion entry during Level-1 hybrid search.
#[derive(Default)]
struct FusedHit {
    lexical_score: Option<f32>,
    lexical_rank: Option<usize>,
    vector_score: Option<f32>,
    vector_rank: Option<usize>,
    rrf: f32,
    explain: Option<V2Explain>,
}

/// The hit envelope as stored in the Tantivy doc store.
struct StoredEnvelope {
    parent_id: Option<String>,
    chunk_index: Option<u32>,
    kind: String,
    title: String,
    path: String,
    excerpt: Option<String>,
}

impl StoredEnvelope {
    fn to_search_hit(&self, id: &str, score: f32) -> V2SearchHit {
        V2SearchHit {
            id: id.to_string(),
            parent_id: self.parent_id.clone(),
            chunk_index: self.chunk_index,
            kind: self.kind.clone(),
            title: self.title.clone(),
            path: self.path.clone(),
            excerpt: self.excerpt.clone(),
            score,
            bm25_score: score,
            explain: None,
        }
    }
}

/// BM25 score of the `LEXICAL_SPIKE_BASELINE_PARENTS`-th distinct parent in the
/// lexical lane, used to detect exact phrase/title outliers in hybrid RRF.
fn lexical_spike_baseline(lexical_hits: &[V2SearchHit]) -> Option<f32> {
    let mut seen_parents: HashSet<&str> = HashSet::with_capacity(LEXICAL_SPIKE_BASELINE_PARENTS);
    for hit in lexical_hits {
        let key = hit.parent_id.as_deref().unwrap_or(hit.id.as_str());
        if seen_parents.insert(key) && seen_parents.len() == LEXICAL_SPIKE_BASELINE_PARENTS {
            return (hit.score > 0.0 && hit.score.is_finite()).then_some(hit.score);
        }
    }
    None
}

/// Convert a distance metric into a higher-is-better similarity.
///
/// Consistent with `src/search.rs::compute_score`: `crate::distance` returns
/// cosine *similarity* (not a distance), dot product as-is, and Euclidean
/// distance, which is squashed into `(0, 1]` so the rank evidence stays
/// comparable across metrics.
fn vector_similarity(distance: Distance, query: &[f32], document: &[f32]) -> Result<f32, V2Error> {
    let value = distance
        .compute(query, document)
        .map_err(|error| V2Error::InvalidQuery(error.to_string()))?;
    if !value.is_finite() {
        return Err(V2Error::Corrupt(
            "a stored embedding produced a non-finite similarity".to_string(),
        ));
    }
    Ok(match distance {
        Distance::DotProduct => value,
        Distance::Cosine => value,
        Distance::Euclidean => 1.0 / (1.0 + value.max(0.0)),
    })
}

/// Tantivy-backed, ACL-fail-closed lexical index.
pub struct V2Index {
    dir: PathBuf,
    config: V2CollectionConfig,
    index: Index,
    /// Created on the first write, never on open: a read-only process must not
    /// pay for a writer arena or a writer lock.
    writer: Option<IndexWriter<TantivyDocument>>,
    reader: IndexReader,
    fields: V2Fields,
    acl: AclOverlay,
    manifest: V2Manifest,
    fingerprints: HashMap<String, DocFingerprint>,
    /// Content-addressed embeddings (`vectors.bin`).
    vectors: VectorStore,
    /// `doc_id -> content_hash` link map (`vec_map.gen_<N>.bin`).
    vec_map: VecMap,
    /// Set when the link map changed and must be persisted with a new
    /// generation; an untouched map is never rewritten.
    vec_map_dirty: bool,
    /// Set when `vectors.bin` gained or lost an entry.
    vectors_dirty: bool,
}

impl V2Index {
    /// Create the collection in `dir`, or open it when a manifest exists.
    ///
    /// Opening validates `config` against the persisted manifest and fails with
    /// [`V2Error::VersionMismatch`] when schema, analyzer or lexicon versions
    /// differ.
    pub fn create_or_open(
        dir: impl AsRef<Path>,
        config: V2CollectionConfig,
    ) -> Result<Self, V2Error> {
        let dir = dir.as_ref().to_path_buf();
        validate_vector_lane(&config)?;
        fs::create_dir_all(&dir)?;
        if manifest_path(&dir).exists() {
            return Self::open(dir, config);
        }

        let (schema, fields) = build_schema(&config)?;
        let index = if tantivy_meta_path(&dir).exists() {
            let existing = Index::open_in_dir(&dir)?;
            if existing.schema() != schema {
                return Err(V2Error::SchemaMismatch(
                    "a Tantivy index exists whose schema does not match the requested config"
                        .to_string(),
                ));
            }
            existing
        } else {
            Index::create_in_dir(&dir, schema)?
        };
        register_tokenizers(&index);

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        let acl = AclOverlay::new();
        let manifest = V2Manifest {
            config: config.clone(),
            acl_generation: acl.generation(),
            vec_generation: 0,
            watermark: None,
            doc_count: 0,
        };

        let mut index = Self {
            dir,
            config,
            index,
            writer: None,
            reader,
            fields,
            acl,
            manifest,
            fingerprints: HashMap::new(),
            vectors: VectorStore::new(),
            vec_map: VecMap::new(),
            vec_map_dirty: false,
            vectors_dirty: false,
        };
        index.persist_state()?;
        Ok(index)
    }

    /// Open an existing collection, validating `expected` against the manifest.
    ///
    /// `schema_version`, `analyzer_version`, `lexicon_version` and
    /// `promoted_fields` must match the manifest exactly; the caller's
    /// `field_weights` and `store_excerpt` are adopted for the live process
    /// because they only affect querying and newly written documents.
    pub fn open(dir: impl AsRef<Path>, expected: V2CollectionConfig) -> Result<Self, V2Error> {
        let dir = dir.as_ref().to_path_buf();
        validate_vector_lane(&expected)?;
        let manifest = V2Manifest::load(&dir)?;
        validate_config(&manifest.config, &expected)?;

        let (schema, fields) = build_schema(&expected)?;
        let index = Index::open_in_dir(&dir)?;
        if index.schema() != schema {
            return Err(V2Error::SchemaMismatch(
                "the persisted Tantivy schema does not match the persisted config".to_string(),
            ));
        }
        register_tokenizers(&index);

        let acl = AclOverlay::load(&dir, manifest.acl_generation)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let fingerprints = load_fingerprints(&dir)?;
        let vectors = VectorStore::load(&dir)?;
        let vec_map = VecMap::load(&dir, manifest.vec_generation)?;

        Ok(Self {
            dir,
            config: expected,
            index,
            writer: None,
            reader,
            fields,
            acl,
            manifest,
            fingerprints,
            vectors,
            vec_map,
            vec_map_dirty: false,
            vectors_dirty: false,
        })
    }

    /// Collection directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Live configuration (validated against the persisted manifest).
    pub fn config(&self) -> &V2CollectionConfig {
        &self.config
    }

    /// Current manifest.
    pub fn manifest(&self) -> &V2Manifest {
        &self.manifest
    }

    /// Live ACL overlay.
    pub fn acl_overlay(&self) -> &AclOverlay {
        &self.acl
    }

    /// Documents tracked by the index.
    pub fn doc_count(&self) -> usize {
        self.fingerprints.len()
    }

    /// Documents visible through the live Tantivy reader.
    pub fn reader_doc_count(&self) -> u64 {
        self.reader.searcher().num_docs()
    }

    /// Fingerprint of one document, if tracked.
    pub fn fingerprint(&self, id: &str) -> Option<&DocFingerprint> {
        self.fingerprints.get(id)
    }

    /// Every tracked document's fingerprint and ACL, ordered by `id`.
    ///
    /// This is the read side of the fingerprint exchange: a caller compares the
    /// returned hashes against its own source hashes and reports only what
    /// differs, instead of shipping the whole index.
    pub fn fingerprint_entries(&self) -> Vec<FingerprintEntry> {
        let mut entries: Vec<FingerprintEntry> = self
            .fingerprints
            .iter()
            .map(|(id, fingerprint)| FingerprintEntry {
                id: id.clone(),
                content_hash: digest_to_hex(&fingerprint.content_hash),
                envelope_hash: digest_to_hex(&fingerprint.envelope_hash),
                has_vector: self.vector_for(id).is_some(),
                acl: self.acl.groups(id).unwrap_or(&[]).to_vec(),
            })
            .collect();
        entries.sort_unstable_by(|left, right| left.id.cmp(&right.id));
        entries
    }

    /// Compare the caller's expected fingerprints against the live ones.
    ///
    /// Hex comparison is case-insensitive so an uppercased hash from another
    /// implementation is not reported as a mismatch. With `check_vectors`,
    /// every expected document the index tracks must also hold an embedding for
    /// its live text.
    pub fn integrity_check_entries(
        &self,
        expected: &[ExpectedFingerprint],
        check_vectors: bool,
    ) -> IntegrityDiff {
        let mut diff = IntegrityDiff::default();
        let mut seen: HashSet<&str> = HashSet::with_capacity(expected.len());
        for entry in expected {
            seen.insert(entry.id.as_str());
            match self.fingerprints.get(&entry.id) {
                None => diff.missing.push(entry.id.clone()),
                Some(stored) => {
                    if !digest_to_hex(&stored.content_hash)
                        .eq_ignore_ascii_case(&entry.content_hash)
                    {
                        diff.content_mismatched.push(entry.id.clone());
                    }
                    if !digest_to_hex(&stored.envelope_hash)
                        .eq_ignore_ascii_case(&entry.envelope_hash)
                    {
                        diff.envelope_mismatched.push(entry.id.clone());
                    }
                    if check_vectors && self.vector_for(&entry.id).is_none() {
                        diff.missing_vectors.push(entry.id.clone());
                    }
                }
            }
        }
        for id in self.fingerprints.keys() {
            if !seen.contains(id.as_str()) {
                diff.extra.push(id.clone());
            }
        }
        diff.missing.sort_unstable();
        diff.extra.sort_unstable();
        diff.content_mismatched.sort_unstable();
        diff.envelope_mismatched.sort_unstable();
        diff.missing_vectors.sort_unstable();
        diff.clean = diff.missing.is_empty()
            && diff.extra.is_empty()
            && diff.content_mismatched.is_empty()
            && diff.envelope_mismatched.is_empty()
            && diff.missing_vectors.is_empty();
        diff
    }

    /// Insert or replace one document.
    ///
    /// The replace is a `delete_term(id)` + `add_document` pair inside the same
    /// writer, so a re-index of the same id can never duplicate a hit. Changes
    /// become visible after [`V2Index::commit`].
    ///
    /// The vector link follows `content_hash`: an ACK-only, metadata-only or
    /// no-op re-upsert keeps an existing embedding (zero re-embedding), while an
    /// edit that lands on unseen text drops the link and the document reappears
    /// through [`V2Index::next_pending_embeddings`].
    pub fn upsert(&mut self, doc: V2Document) -> Result<(), V2Error> {
        let fingerprint = fingerprint_from_document(&doc, &self.config);
        self.write_document(&doc)?;
        self.acl.update_acl(&doc.id, doc.acl.clone());
        self.fingerprints.insert(doc.id.clone(), fingerprint);
        self.reconcile_vector_link(&doc.id);
        Ok(())
    }

    /// Insert or replace a batch of documents.
    pub fn upsert_batch(&mut self, docs: Vec<V2Document>) -> Result<(), V2Error> {
        for doc in docs {
            self.upsert(doc)?;
        }
        Ok(())
    }

    /// Change only the ACL of an already indexed document.
    ///
    /// This is the fast path the Phase 0 ACL-only signal unlocks: the overlay
    /// and the manifest are rewritten atomically, **no Tantivy segment is
    /// touched**, and the next search already sees the new visibility.
    pub fn update_acl_only(&mut self, id: &str, acl: Vec<String>) -> Result<(), V2Error> {
        self.acl.update_acl(id, acl);
        self.persist_state()
    }

    /// Delete one document.
    ///
    /// The vector link is dropped; the embedding itself stays behind until a
    /// rebuild collects it, because another document may share the same
    /// content hash.
    pub fn delete(&mut self, id: &str) -> Result<(), V2Error> {
        let id_term = Term::from_field_text(self.fields.id, id);
        self.writer()?.delete_term(id_term);
        self.acl.remove(id);
        self.fingerprints.remove(id);
        if self.vec_map.remove(id) {
            self.vec_map_dirty = true;
        }
        Ok(())
    }

    /// Commit pending Tantivy changes, then persist overlay, manifest and
    /// fingerprints.
    ///
    /// Order matters: the ACL generation file is written first, the manifest
    /// that points at it is swapped next, and only then are superseded
    /// generations deleted.
    pub fn commit(&mut self) -> Result<(), V2Error> {
        if let Some(writer) = self.writer.as_mut() {
            writer.commit()?;
        }
        self.reader.reload()?;
        self.persist_state()
    }

    /// Mark the index degraded: reads fail until a rebuild heals it.
    pub fn mark_degraded(&mut self, reason: impl Into<String>) -> Result<(), V2Error> {
        let reason = reason.into();
        let (epoch, synced_seq) = match &self.manifest.watermark {
            Some(watermark) => (watermark.source_epoch_uuid.clone(), watermark.synced_seq),
            None => (String::new(), 0),
        };
        self.manifest.watermark = Some(SyncWatermark {
            source_epoch_uuid: epoch,
            synced_seq,
            degraded: true,
            degraded_reason: Some(reason),
        });
        V2Manifest::save_atomic(&self.manifest, &self.dir)
    }

    /// Atomically replace the sync watermark and persist the manifest.
    ///
    /// The invariants are checked at the engine boundary as well as at the HTTP
    /// boundary: a degraded-without-reason or epoch-less watermark would make
    /// the next open read a state the index can never have produced.
    pub fn set_watermark(&mut self, watermark: SyncWatermark) -> Result<(), V2Error> {
        watermark
            .validate()
            .map_err(|error| V2Error::InvalidConfig(error.to_string()))?;
        self.manifest.watermark = Some(watermark);
        self.manifest.doc_count = self.fingerprints.len() as u64;
        V2Manifest::save_atomic(&self.manifest, &self.dir)
    }

    /// Two-phase search.
    ///
    /// Fails closed with [`V2Error::MissingPrincipals`] for an empty principal
    /// list and with [`V2Error::IndexDegraded`] while the watermark is
    /// degraded.
    pub fn search(&self, query: &V2Query) -> Result<Vec<V2SearchHit>, V2Error> {
        self.ensure_readable(&query.principals)?;
        if let Some(group) = query.group_by_parent {
            if group.max_per_parent == 0 {
                return Err(V2Error::InvalidQuery(
                    "group_by_parent.max_per_parent must be greater than zero".to_string(),
                ));
            }
        }

        let top_k = query.top_k.max(1);
        let principals: HashSet<&str> = query.principals.iter().map(String::as_str).collect();
        let root = self.build_query(query)?;
        let searcher = self.reader.searcher();
        let total_docs = searcher.num_docs() as usize;
        if total_docs == 0 {
            return Ok(Vec::new());
        }

        let query_terms = distinct_terms(&query.text);
        let mut limit = (top_k * CANDIDATE_MULTIPLIER).max(CANDIDATE_MINIMUM);
        let mut hits: Vec<V2SearchHit> = Vec::new();
        for round in 0..WIDEN_ROUNDS {
            let candidates = searcher.search(root.as_ref(), &TopDocs::with_limit(limit))?;
            let examined = candidates.len();
            hits = self.phase_b(
                &searcher,
                &candidates,
                query,
                &principals,
                &query_terms,
                top_k,
            )?;
            let enough = hits.len() >= top_k;
            let exhausted = examined < limit || limit >= total_docs;
            if enough || exhausted || round + 1 == WIDEN_ROUNDS {
                break;
            }
            limit = limit
                .saturating_mul(WIDEN_FACTOR)
                .min(total_docs.max(limit + 1));
        }
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Read-side gate: principals must be present and the index must not be
    /// degraded.
    ///
    /// Shared by `search`, `hybrid_search` and Level-2 refinement so the
    /// fail-closed prelude cannot drift between entry points.
    pub(crate) fn ensure_readable(&self, principals: &[String]) -> Result<(), V2Error> {
        if principals.is_empty() {
            return Err(V2Error::MissingPrincipals);
        }
        if let Some(watermark) = &self.manifest.watermark {
            if watermark.degraded {
                return Err(V2Error::IndexDegraded {
                    reason: watermark
                        .degraded_reason
                        .clone()
                        .unwrap_or_else(|| "index marked degraded without a reason".to_string()),
                });
            }
        }
        Ok(())
    }

    /// The vector store (`vectors.bin`).
    pub fn vectors(&self) -> &VectorStore {
        &self.vectors
    }

    /// Generation of the live `doc_id -> content_hash` map.
    pub fn vec_generation(&self) -> u64 {
        self.manifest.vec_generation
    }

    /// Embedding linked to `id`, when the lane holds one for its live text.
    pub fn vector_for(&self, id: &str) -> Option<&[f32]> {
        let lane = self.config.vector_lane.as_ref()?;
        let digest = self.vec_map.get(id)?;
        self.vectors.get(digest, &lane.model_version)
    }

    /// Live documents that still need an embedding for the configured lane.
    ///
    /// Pending state is *derived* — `live ∧ ¬has_vector` — so there is no
    /// durable queue that could disagree with the index. Returns an empty list
    /// when the collection has no vector lane. Ordered by id so a host can
    /// resume a batch job deterministically.
    pub fn next_pending_embeddings(&self, limit: usize) -> Vec<PendingEmbedding> {
        if limit == 0 {
            return Vec::new();
        }
        let Some(lane) = self.config.vector_lane.as_ref() else {
            return Vec::new();
        };
        let mut pending: Vec<(&String, &DocFingerprint)> = self
            .fingerprints
            .iter()
            .filter(|(id, fingerprint)| {
                self.vec_map.get(id.as_str()) != Some(&fingerprint.content_hash)
                    || !self
                        .vectors
                        .contains(&fingerprint.content_hash, &lane.model_version)
            })
            .collect();
        pending.sort_unstable_by(|left, right| left.0.cmp(right.0));
        pending.truncate(limit);

        let ids: Vec<String> = pending.iter().map(|(id, _)| (*id).clone()).collect();
        let titles = match self.load_envelopes(&ids) {
            Ok(envelopes) => envelopes,
            Err(error) => {
                // A display title must never make an embedding batch fail; the
                // loss is reported so it stays visible instead of silent.
                eprintln!("nvdb v2: pending-embedding titles unavailable: {error}");
                HashMap::new()
            }
        };
        pending
            .into_iter()
            .map(|(id, fingerprint)| PendingEmbedding {
                id: id.clone(),
                content_hash: digest_to_hex(&fingerprint.content_hash),
                title: titles
                    .get(id)
                    .map(|envelope| envelope.title.clone())
                    .unwrap_or_default(),
            })
            .collect()
    }

    /// Attach embeddings produced by the host's embed loop.
    ///
    /// Compare-and-set on `(id, content_hash, model_version)`: the entire batch
    /// is validated before anything is written, so a single stale completion
    /// rejects the batch instead of half-applying it. A re-attach of an
    /// already-known `(content_hash, model_version)` is a dedup hit and costs no
    /// store growth.
    ///
    /// The link map and the manifest are committed atomically here — an attach
    /// is not deferred to [`V2Index::commit`] because the caller has already
    /// paid for the embedding.
    pub fn attach_vectors(&mut self, entries: Vec<VectorAttachment>) -> Result<usize, V2Error> {
        let lane = self
            .config
            .vector_lane
            .as_ref()
            .ok_or(V2Error::VectorLaneNotConfigured)?;
        if entries.is_empty() {
            return Ok(0);
        }

        let mut validated: Vec<(String, [u8; 32], usize)> = Vec::with_capacity(entries.len());
        for (index, entry) in entries.iter().enumerate() {
            let digest = hex_to_digest(&entry.content_hash).ok_or_else(|| {
                V2Error::InvalidQuery(format!(
                    "vector attachment for '{}' carries a malformed content_hash (expected 64 hex characters)",
                    entry.id
                ))
            })?;
            if entry.vector.len() != lane.dim {
                return Err(V2Error::WrongDimension {
                    expected: lane.dim,
                    got: entry.vector.len(),
                });
            }
            if !entry.vector.iter().all(|value| value.is_finite()) {
                return Err(V2Error::InvalidQuery(format!(
                    "vector attachment for '{}' contains a non-finite value",
                    entry.id
                )));
            }
            if entry.model_version != lane.model_version {
                return Err(V2Error::InvalidQuery(format!(
                    "vector attachment for '{}' was produced by model '{}' but the lane expects '{}'",
                    entry.id, entry.model_version, lane.model_version
                )));
            }
            let Some(fingerprint) = self.fingerprints.get(&entry.id) else {
                return Err(V2Error::StaleVectorCas {
                    id: entry.id.clone(),
                    reason: "document is not live".to_string(),
                });
            };
            if fingerprint.content_hash != digest {
                return Err(V2Error::StaleVectorCas {
                    id: entry.id.clone(),
                    reason: format!(
                        "content_hash {} does not match the live document's {}",
                        entry.content_hash,
                        digest_to_hex(&fingerprint.content_hash)
                    ),
                });
            }
            validated.push((entry.id.clone(), digest, index));
        }

        let mut newly_added: Vec<([u8; 32], String)> = Vec::new();
        for (_, digest, index) in &validated {
            if !self.vectors.contains(digest, &lane.model_version) {
                self.vectors
                    .insert(*digest, &lane.model_version, entries[*index].vector.clone());
                newly_added.push((*digest, lane.model_version.clone()));
            }
        }
        for (id, digest, _) in &validated {
            if self.vec_map.insert(id, *digest) {
                self.vec_map_dirty = true;
            }
        }

        // Persist only vector state: append new vectors in O(batch) I/O, then
        // atomically swap vec_map + manifest without rewriting unchanged ACL or
        // fingerprint sidecars (`herrbasan/nVDB#3`).
        if !newly_added.is_empty() {
            self.vectors.append_records(&self.dir, &newly_added)?;
        }
        if self.vec_map_dirty {
            let vec_generation = self.vec_map.generation();
            self.vec_map.save_atomic(&self.dir, vec_generation)?;
            self.manifest.vec_generation = vec_generation;
            V2Manifest::save_atomic(&self.manifest, &self.dir)?;
            VecMap::cleanup_superseded(&self.dir, vec_generation)?;
            self.vec_map_dirty = false;
        }
        Ok(validated.len())
    }

    /// Drop embeddings no live document references and persist when anything
    /// went.
    ///
    /// Runs on rebuild; a write never pays for it.
    pub fn gc_vectors(&mut self) -> usize {
        let Some(lane) = self.config.vector_lane.as_ref() else {
            return 0;
        };
        let referenced: HashSet<([u8; 32], String)> = self
            .vec_map
            .iter()
            .map(|(_, digest)| (*digest, lane.model_version.clone()))
            .collect();
        let dropped = self.vectors.retain_referenced(&referenced);
        if dropped > 0 {
            self.vectors_dirty = true;
        }
        dropped
    }

    /// Keep, drop or refresh the vector link of one document after a write.
    fn reconcile_vector_link(&mut self, id: &str) {
        let Some(lane) = self.config.vector_lane.as_ref() else {
            return;
        };
        let Some(fingerprint) = self.fingerprints.get(id) else {
            if self.vec_map.remove(id) {
                self.vec_map_dirty = true;
            }
            return;
        };
        let digest = fingerprint.content_hash;
        if self.vec_map.get(id) == Some(&digest)
            && self.vectors.contains(&digest, &lane.model_version)
        {
            return;
        }
        // Identical text already embedded (another document, or an edit that
        // returned to earlier wording) is a dedup hit: link it, do not embed it
        // again.
        if self.vectors.contains(&digest, &lane.model_version) {
            if self.vec_map.insert(id, digest) {
                self.vec_map_dirty = true;
            }
            return;
        }
        if self.vec_map.remove(id) {
            self.vec_map_dirty = true;
        }
    }

    fn reconcile_all_vector_links(&mut self) {
        if self.config.vector_lane.is_none() {
            return;
        }
        let fingerprints = &self.fingerprints;
        let changed = self
            .vec_map
            .retain(|id, digest| fingerprints.get(id).map(|f| &f.content_hash) == Some(digest));
        if changed {
            self.vec_map_dirty = true;
        }
    }

    /// Unified lexical + dense retrieval with scale-free RRF fusion.
    ///
    /// Both lanes produce `candidate_k` candidates, RRF merges them, and
    /// `group_by_parent` is applied **once** on the fused order — a per-lane cap
    /// would let one lane's second chunk hide the other lane's first. Only the
    /// final `top_k` winners are hydrated.
    ///
    /// The dense lane honours the same authority as the lexical lane: the ACL
    /// overlay, plus the query's columnar filters (evaluated with a filter-only
    /// posting walk and intersected with the overlay).
    pub fn hybrid_search(&self, query: &V2HybridQuery) -> Result<Vec<V2HybridHit>, V2Error> {
        self.ensure_readable(&query.lexical.principals)?;
        if !query.dense && !query.lexical_enabled {
            return Err(V2Error::InvalidQuery(
                "hybrid_search needs at least one enabled lane".to_string(),
            ));
        }
        if query.candidate_k == 0 {
            return Err(V2Error::InvalidQuery(
                "candidate_k must be greater than zero".to_string(),
            ));
        }
        if !query.lexical_weight.is_finite()
            || !query.vector_weight.is_finite()
            || query.lexical_weight < 0.0
            || query.vector_weight < 0.0
        {
            return Err(V2Error::InvalidQuery(
                "lane weights must be finite and non-negative".to_string(),
            ));
        }
        if let Some(group) = query.lexical.group_by_parent {
            if group.max_per_parent == 0 {
                return Err(V2Error::InvalidQuery(
                    "group_by_parent.max_per_parent must be greater than zero".to_string(),
                ));
            }
        }

        let top_k = query.lexical.top_k.max(1);
        // Both lanes must be able to fill `top_k` even when the caller set a
        // smaller candidate budget.
        let lane_k = query.candidate_k.max(top_k);
        let principals: HashSet<&str> = query
            .lexical
            .principals
            .iter()
            .map(String::as_str)
            .collect();

        let mut lexical_hits: Vec<V2SearchHit> = Vec::new();
        if query.lexical_enabled {
            let mut lane_query = query.lexical.clone();
            lane_query.top_k = lane_k;
            // Diversity belongs to the fused order, never to one lane.
            lane_query.group_by_parent = None;
            lexical_hits = self.search(&lane_query)?;
        }

        let mut vector_hits: Vec<(String, f32)> = Vec::new();
        if query.dense {
            let lane = self
                .config
                .vector_lane
                .as_ref()
                .ok_or(V2Error::VectorLaneNotConfigured)?;
            let query_vector = query.query_vector.as_deref().ok_or_else(|| {
                V2Error::InvalidQuery(
                    "the dense lane is enabled but the query carries no query_vector".to_string(),
                )
            })?;
            if query_vector.len() != lane.dim {
                return Err(V2Error::WrongDimension {
                    expected: lane.dim,
                    got: query_vector.len(),
                });
            }
            if !query_vector.iter().all(|value| value.is_finite()) {
                return Err(V2Error::InvalidQuery(
                    "query_vector contains a non-finite value".to_string(),
                ));
            }
            let allowed = self.filtered_allowed_ids(&query.lexical.filters, &principals)?;
            for (id, digest) in self.vec_map.iter() {
                let visible = match &allowed {
                    Some(allowed) => allowed.contains(id),
                    None => self.acl.is_visible(id, &principals),
                };
                if !visible {
                    continue;
                }
                let Some(vector) = self.vectors.get(digest, &lane.model_version) else {
                    continue;
                };
                vector_hits.push((
                    id.clone(),
                    vector_similarity(lane.distance, query_vector, vector)?,
                ));
            }
            vector_hits.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.0.cmp(&right.0))
            });
            vector_hits.truncate(lane_k);
        }

        let mut fused: HashMap<String, FusedHit> = HashMap::new();
        let damping = query.rrf_k as f32;
        let spike_threshold = if query.dense
            && query.lexical_enabled
            && query.lexical_weight > 0.0
            && query.vector_weight > query.lexical_weight
        {
            lexical_spike_baseline(&lexical_hits).map(|base| base * LEXICAL_SPIKE_RATIO)
        } else {
            None
        };
        let spike_weight = query.lexical_weight.max(query.vector_weight * 1.1);
        let mut lexical_by_id: HashMap<&str, &V2SearchHit> = HashMap::new();
        for (index, hit) in lexical_hits.iter().enumerate() {
            let rank = index + 1;
            lexical_by_id.insert(hit.id.as_str(), hit);
            let entry = fused.entry(hit.id.clone()).or_default();
            entry.lexical_score = Some(hit.score);
            entry.lexical_rank = Some(rank);
            entry.explain = hit.explain.clone();
            let weight = if spike_threshold.is_some_and(|t| hit.score >= t) {
                spike_weight
            } else {
                query.lexical_weight
            };
            entry.rrf += weight / (damping + rank as f32);
        }
        for (index, (id, score)) in vector_hits.iter().enumerate() {
            let rank = index + 1;
            let entry = fused.entry(id.clone()).or_default();
            entry.vector_score = Some(*score);
            entry.vector_rank = Some(rank);
            entry.rrf += query.vector_weight / (damping + rank as f32);
        }

        let mut ordered: Vec<(String, FusedHit)> = fused.into_iter().collect();
        ordered.sort_by(|left, right| {
            right
                .1
                .rrf
                .partial_cmp(&left.1.rrf)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });

        // Diversity needs `parent_id` before hydration; the lexical lane already
        // carries it, so only vector-only candidates are looked up.
        let mut parents: HashMap<String, Option<String>> = HashMap::new();
        for (id, _) in &ordered {
            if let Some(hit) = lexical_by_id.get(id.as_str()) {
                parents.insert(id.clone(), hit.parent_id.clone());
            }
        }
        let missing: Vec<String> = ordered
            .iter()
            .filter(|(id, _)| !parents.contains_key(id))
            .map(|(id, _)| id.clone())
            .collect();
        parents.extend(self.resolve_parents(&missing)?);

        let mut winners: Vec<(String, FusedHit)> = Vec::with_capacity(top_k);
        let mut per_parent: HashMap<&str, usize> = HashMap::new();
        for (id, entry) in ordered {
            if winners.len() >= top_k {
                break;
            }
            if let Some(group) = query.lexical.group_by_parent {
                if let Some(parent) = parents.get(&id).and_then(|parent| parent.as_deref()) {
                    let count = per_parent.entry(parent).or_insert(0);
                    if *count >= group.max_per_parent {
                        continue;
                    }
                    *count += 1;
                }
            }
            winners.push((id, entry));
        }

        let unhydrated: Vec<String> = winners
            .iter()
            .filter(|(id, _)| !lexical_by_id.contains_key(id.as_str()))
            .map(|(id, _)| id.clone())
            .collect();
        let envelopes = self.load_envelopes(&unhydrated)?;

        let mut hits: Vec<V2HybridHit> = Vec::with_capacity(winners.len());
        for (id, entry) in winners {
            let mut hit = match lexical_by_id.get(id.as_str()) {
                Some(hit) => (*hit).clone(),
                None => {
                    let Some(envelope) = envelopes.get(&id) else {
                        // The document left the index between the scan and the
                        // hydration: it is no longer live, so it is not evidence.
                        continue;
                    };
                    envelope.to_search_hit(&id, entry.vector_score.unwrap_or(0.0))
                }
            };
            hit.score = entry.rrf;
            hit.bm25_score = entry.lexical_score.unwrap_or(0.0);
            hit.explain = entry.explain;
            hits.push(V2HybridHit {
                hit,
                lexical_score: entry.lexical_score,
                lexical_rank: entry.lexical_rank,
                vector_score: entry.vector_score,
                vector_rank: entry.vector_rank,
                rrf_score: entry.rrf,
            });
        }
        Ok(hits)
    }

    /// Ids that pass the query's columnar filters *and* the ACL overlay.
    ///
    /// `Ok(None)` means the query carries no columnar filter, so the caller can
    /// check the overlay alone while walking the content-hash map.
    fn filtered_allowed_ids(
        &self,
        filters: &V2Filters,
        principals: &HashSet<&str>,
    ) -> Result<Option<HashSet<String>>, V2Error> {
        if !filters.is_active() {
            return Ok(None);
        }
        let clauses: Vec<(Occur, Box<dyn Query>)> = self
            .filter_queries(filters)?
            .into_iter()
            .map(|query| (Occur::Must, query))
            .collect();
        let query = BooleanQuery::new(clauses);
        let searcher = self.reader.searcher();
        let addresses = searcher.search(&query, &DocSetCollector)?;

        let mut allowed: HashSet<String> = HashSet::with_capacity(addresses.len());
        let mut segments: HashMap<u32, SegmentLight> = HashMap::new();
        let mut scratch = String::new();
        for address in addresses {
            let light = match segments.entry(address.segment_ord) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    entry.insert(SegmentLight::open(&searcher, address.segment_ord)?)
                }
            };
            let Some(id) = fast_field_string(&light.id, address.doc_id, &mut scratch)? else {
                continue;
            };
            if self.acl.is_visible(&id, principals) {
                allowed.insert(id);
            }
        }
        Ok(Some(allowed))
    }

    /// `parent_id` of every listed document that the reader can still resolve.
    fn resolve_parents(&self, ids: &[String]) -> Result<HashMap<String, Option<String>>, V2Error> {
        let mut parents: HashMap<String, Option<String>> = HashMap::with_capacity(ids.len());
        if ids.is_empty() {
            return Ok(parents);
        }
        let searcher = self.reader.searcher();
        let mut segments: HashMap<u32, SegmentLight> = HashMap::new();
        let mut scratch = String::new();
        for (id, address) in self.resolve_addresses(&searcher, ids)? {
            let light = match segments.entry(address.segment_ord) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    entry.insert(SegmentLight::open(&searcher, address.segment_ord)?)
                }
            };
            let parent_id = match &light.parent_id {
                Some(column) => fast_field_string(column, address.doc_id, &mut scratch)?,
                None => None,
            };
            parents.insert(id, parent_id);
        }
        Ok(parents)
    }

    /// Stored envelopes of the listed documents.
    fn load_envelopes(&self, ids: &[String]) -> Result<HashMap<String, StoredEnvelope>, V2Error> {
        let mut envelopes: HashMap<String, StoredEnvelope> = HashMap::with_capacity(ids.len());
        if ids.is_empty() {
            return Ok(envelopes);
        }
        let searcher = self.reader.searcher();
        for (id, address) in self.resolve_addresses(&searcher, ids)? {
            let stored: TantivyDocument = searcher.doc(address)?;
            envelopes.insert(id, self.stored_envelope(&stored));
        }
        Ok(envelopes)
    }

    /// Doc address per id, in one posting-list walk over the `id` terms.
    fn resolve_addresses(
        &self,
        searcher: &Searcher,
        ids: &[String],
    ) -> Result<Vec<(String, DocAddress)>, V2Error> {
        let clauses: Vec<(Occur, Box<dyn Query>)> = ids
            .iter()
            .map(|id| (Occur::Should, term_query(self.fields.id, id)))
            .collect();
        let query = BooleanQuery::new(clauses);
        let candidates = searcher.search(&query, &TopDocs::with_limit(ids.len().max(1)))?;

        let mut resolved: Vec<(String, DocAddress)> = Vec::with_capacity(candidates.len());
        let mut segments: HashMap<u32, SegmentLight> = HashMap::new();
        let mut scratch = String::new();
        for (_, address) in candidates {
            let light = match segments.entry(address.segment_ord) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    entry.insert(SegmentLight::open(searcher, address.segment_ord)?)
                }
            };
            let Some(id) = fast_field_string(&light.id, address.doc_id, &mut scratch)? else {
                continue;
            };
            resolved.push((id, address));
        }
        Ok(resolved)
    }

    fn stored_envelope(&self, stored: &TantivyDocument) -> StoredEnvelope {
        StoredEnvelope {
            parent_id: text_value(stored, self.fields.parent_id),
            chunk_index: stored
                .get_first(self.fields.chunk_index)
                .and_then(|value| value.as_u64())
                .map(|value| value as u32),
            kind: text_value(stored, self.fields.kind).unwrap_or_default(),
            title: text_value(stored, self.fields.title).unwrap_or_default(),
            path: text_value(stored, self.fields.path).unwrap_or_default(),
            excerpt: text_value(stored, self.fields.excerpt),
        }
    }

    /// Compare the live index against the authoritative source snapshot.
    /// reader document count, and the ACL overlay alignment. Hashes are the
    /// ones the source supplied on the last sync; documents written directly
    /// through [`V2Index::upsert`] are compared against their own envelope, so a
    /// source/mapper mismatch is reported instead of hidden.
    pub fn integrity_check(
        &self,
        source: &dyn SourceConnector,
    ) -> Result<IntegrityReport, V2Error> {
        let snapshot = source.snapshot_at()?;
        let mut report = IntegrityReport {
            source_epoch_uuid: snapshot.lineage.source_epoch_uuid.clone(),
            source_last_seq: snapshot.lineage.last_seq,
            watermark_seq: self.manifest.watermark.as_ref().map(|w| w.synced_seq),
            source_doc_count: snapshot.records.len(),
            indexed_doc_count: self.fingerprints.len(),
            reader_doc_count: self.reader_doc_count(),
            acl_doc_count: self.acl.doc_count(),
            ..IntegrityReport::default()
        };

        let mut seen: HashSet<&str> = HashSet::with_capacity(snapshot.records.len());
        for record in &snapshot.records {
            seen.insert(record.id.as_str());
            let expected = fingerprint_from_envelope(&record.envelope);
            match self.fingerprints.get(&record.id) {
                None => report.missing.push(record.id.clone()),
                Some(stored) => {
                    if stored.content_hash != expected.content_hash
                        || stored.envelope_hash != expected.envelope_hash
                    {
                        report.mismatched.push(record.id.clone());
                    }
                }
            }
            let expected_acl = normalize_groups(&record.envelope.acl);
            match self.acl.groups(&record.id) {
                Some(groups) if groups == expected_acl.as_slice() => {}
                _ => report.acl_misaligned.push(record.id.clone()),
            }
        }

        for id in self.fingerprints.keys() {
            if !seen.contains(id.as_str()) {
                report.extra.push(id.clone());
            }
        }
        for id in self.acl.doc_ids() {
            if !self.fingerprints.contains_key(id) {
                report.acl_misaligned.push(id.clone());
            }
        }
        report.missing.sort_unstable();
        report.extra.sort_unstable();
        report.mismatched.sort_unstable();
        report.acl_misaligned.sort_unstable();
        report.acl_misaligned.dedup();
        Ok(report)
    }

    /// Bring the index up to date with `source`, mapping each record to a
    /// [`V2Document`].
    ///
    /// Full Phase 0 contract: epoch and cursor-ahead divergence force a clean
    /// snapshot rebuild; an expired journal cursor forces a rebuild; an equal
    /// `envelope_hash` is skipped; a changed `content_hash` with an unchanged
    /// no-ACL hash is applied through [`V2Index::update_acl_only`] (zero
    /// re-indexing); the watermark is committed atomically at the end.
    pub fn sync_from<M>(
        &mut self,
        source: &dyn SourceConnector,
        mapper: M,
    ) -> Result<SyncOutcome, V2Error>
    where
        M: Fn(&SourceRecord) -> V2Document,
    {
        let lineage = source.lineage()?;
        let must_rebuild = match &self.manifest.watermark {
            None => true,
            Some(watermark) => {
                watermark.degraded
                    || watermark.source_epoch_uuid != lineage.source_epoch_uuid
                    || watermark.synced_seq > lineage.last_seq
            }
        };
        if must_rebuild {
            return self.rebuild_inner(source, &mapper);
        }

        let start_seq = match &self.manifest.watermark {
            Some(watermark) => watermark.synced_seq,
            None => return self.rebuild_inner(source, &mapper),
        };
        if start_seq == lineage.last_seq {
            return Ok(SyncOutcome::UpToDate {
                watermark_seq: start_seq,
            });
        }

        let changes = match source.changes_since(start_seq) {
            Ok(changes) => changes,
            Err(SyncError::CursorExpired { .. }) => return self.rebuild_inner(source, &mapper),
            Err(error) => return Err(V2Error::Sync(error)),
        };

        let mut ordered: Vec<SourceChange> = changes
            .into_iter()
            .filter(|change| change.seq > start_seq)
            .collect();
        ordered.sort_by_key(|change| change.seq);

        let mut cursor = start_seq;
        let mut applied = 0usize;
        let mut skipped = 0usize;
        for change in &ordered {
            if change.seq != cursor + 1 {
                // A feed gap means the connector cannot be trusted; recover
                // from the authoritative snapshot instead of diverging.
                return self.rebuild_inner(source, &mapper);
            }
            match self.apply_source_change(change, &mapper) {
                Ok(ApplyOutcome::Applied) => applied += 1,
                Ok(ApplyOutcome::AclOnly) => applied += 1,
                Ok(ApplyOutcome::Skipped) => skipped += 1,
                Err(error) => {
                    self.mark_degraded(error.to_string())?;
                    return Err(error);
                }
            }
            cursor = change.seq;
        }

        if cursor < lineage.last_seq {
            let error = SyncError::SourceIo(format!(
                "source reported last_seq {} but produced no changes after seq {cursor}",
                lineage.last_seq
            ));
            self.mark_degraded(error.to_string())?;
            return Err(V2Error::Sync(error));
        }

        self.manifest.watermark = Some(SyncWatermark {
            source_epoch_uuid: lineage.source_epoch_uuid,
            synced_seq: cursor,
            degraded: false,
            degraded_reason: None,
        });
        self.commit()?;
        Ok(SyncOutcome::Incremental {
            from_seq: start_seq,
            to_seq: cursor,
            applied,
            skipped,
        })
    }

    /// Rebuild the whole index from a consistent snapshot and replay any
    /// mutations that landed while the snapshot was being applied.
    pub fn rebuild_from_snapshot<M>(
        &mut self,
        source: &dyn SourceConnector,
        mapper: M,
    ) -> Result<SyncOutcome, V2Error>
    where
        M: Fn(&SourceRecord) -> V2Document,
    {
        self.rebuild_inner(source, &mapper)
    }

    fn rebuild_inner<M>(
        &mut self,
        source: &dyn SourceConnector,
        mapper: &M,
    ) -> Result<SyncOutcome, V2Error>
    where
        M: Fn(&SourceRecord) -> V2Document,
    {
        for _attempt in 0..4 {
            let snapshot = source.snapshot_at()?;
            let epoch = snapshot.lineage.source_epoch_uuid.clone();

            self.writer()?.delete_all_documents()?;
            let mut overlay_docs: HashMap<String, Vec<String>> =
                HashMap::with_capacity(snapshot.records.len());
            let mut fingerprints: HashMap<String, DocFingerprint> =
                HashMap::with_capacity(snapshot.records.len());
            for record in &snapshot.records {
                let doc = mapper(record);
                if doc.id != record.id {
                    return Err(V2Error::Sync(SyncError::SourceIo(format!(
                        "mapper produced id '{}' for source record '{}'",
                        doc.id, record.id
                    ))));
                }
                self.write_document(&doc)?;
                overlay_docs.insert(record.id.clone(), record.envelope.acl.clone());
                fingerprints.insert(
                    record.id.clone(),
                    fingerprint_from_envelope(&record.envelope),
                );
            }
            self.writer()?.commit()?;
            self.acl.replace_all(overlay_docs);
            self.fingerprints = fingerprints;
            self.reconcile_all_vector_links();
            self.gc_vectors();

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
                    Err(error) => return Err(V2Error::Sync(error)),
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
                    self.apply_source_change(change, mapper)?;
                    cursor = change.seq;
                }
                if !converged {
                    break;
                }
            }
            if !converged {
                continue;
            }

            self.manifest.watermark = Some(SyncWatermark {
                source_epoch_uuid: epoch,
                synced_seq: cursor,
                degraded: false,
                degraded_reason: None,
            });
            self.commit()?;
            return Ok(SyncOutcome::Rebuilt {
                watermark_seq: cursor,
                docs_indexed: self.fingerprints.len(),
            });
        }
        Err(V2Error::Sync(SyncError::SourceIo(
            "snapshot rebuild did not converge after 4 attempts".to_string(),
        )))
    }

    fn apply_source_change<M>(
        &mut self,
        change: &SourceChange,
        mapper: &M,
    ) -> Result<ApplyOutcome, V2Error>
    where
        M: Fn(&SourceRecord) -> V2Document,
    {
        match &change.op {
            SourceChangeOp::Upsert(record) => {
                if record.id != record.envelope.id {
                    return Err(V2Error::Sync(SyncError::SourceIo(format!(
                        "record id '{}' does not match envelope id '{}'",
                        record.id, record.envelope.id
                    ))));
                }
                let fingerprint = fingerprint_from_envelope(&record.envelope);
                if let Some(stored) = self.fingerprints.get(&record.id) {
                    if stored.envelope_hash == fingerprint.envelope_hash {
                        return Ok(ApplyOutcome::Skipped);
                    }
                    if stored.content_hash == fingerprint.content_hash
                        && stored.no_acl_hash == fingerprint.no_acl_hash
                    {
                        self.acl.update_acl(&record.id, record.envelope.acl.clone());
                        self.fingerprints.insert(record.id.clone(), fingerprint);
                        return Ok(ApplyOutcome::AclOnly);
                    }
                }
                let doc = mapper(record);
                if doc.id != record.id {
                    return Err(V2Error::Sync(SyncError::SourceIo(format!(
                        "mapper produced id '{}' for source record '{}'",
                        doc.id, record.id
                    ))));
                }
                self.write_document(&doc)?;
                self.acl.update_acl(&record.id, record.envelope.acl.clone());
                self.fingerprints.insert(record.id.clone(), fingerprint);
                self.reconcile_vector_link(&record.id);
                Ok(ApplyOutcome::Applied)
            }
            SourceChangeOp::Delete => {
                let id_term = Term::from_field_text(self.fields.id, &change.id);
                self.writer()?.delete_term(id_term);
                self.acl.remove(&change.id);
                self.fingerprints.remove(&change.id);
                if self.vec_map.remove(&change.id) {
                    self.vec_map_dirty = true;
                }
                Ok(ApplyOutcome::Applied)
            }
        }
    }

    fn persist_state(&mut self) -> Result<(), V2Error> {
        let generation = self.acl.generation();
        self.acl.save_atomic(&self.dir, generation)?;
        self.manifest.acl_generation = generation;
        self.manifest.doc_count = self.fingerprints.len() as u64;
        self.manifest.config = self.config.clone();

        // Embeddings are content-addressed and additive, so they are written
        // before the manifest that makes the new link generation reachable.
        if self.vectors_dirty {
            self.vectors.save_atomic(&self.dir)?;
            self.vectors_dirty = false;
        }
        if self.vec_map_dirty {
            let vec_generation = self.vec_map.generation();
            self.vec_map.save_atomic(&self.dir, vec_generation)?;
            self.manifest.vec_generation = vec_generation;
            V2Manifest::save_atomic(&self.manifest, &self.dir)?;
            VecMap::cleanup_superseded(&self.dir, vec_generation)?;
            self.vec_map_dirty = false;
        } else {
            V2Manifest::save_atomic(&self.manifest, &self.dir)?;
        }
        AclOverlay::cleanup_superseded(&self.dir, generation)?;
        save_fingerprints(&self.dir, &self.fingerprints)?;
        Ok(())
    }

    /// The Tantivy writer, created on first use.
    ///
    /// `open()` is a read path: most processes never write, and building a
    /// writer allocates the indexing arena and takes the writer lock.
    fn writer(&mut self) -> Result<&mut IndexWriter<TantivyDocument>, V2Error> {
        if self.writer.is_none() {
            let writer = self
                .index
                .writer_with_num_threads::<TantivyDocument>(1, INDEX_WRITER_HEAP_BYTES)?;
            self.writer = Some(writer);
        }
        self.writer
            .as_mut()
            .ok_or_else(|| V2Error::Corrupt("index writer was not created".to_string()))
    }

    fn write_document(&mut self, doc: &V2Document) -> Result<(), V2Error> {
        let id_term = Term::from_field_text(self.fields.id, &doc.id);
        let tantivy_doc = self.build_tantivy_document(doc)?;
        let writer = self.writer()?;
        writer.delete_term(id_term);
        writer.add_document(tantivy_doc)?;
        Ok(())
    }

    fn build_tantivy_document(&self, doc: &V2Document) -> Result<TantivyDocument, V2Error> {
        let mut tantivy_doc = TantivyDocument::default();
        tantivy_doc.add_text(self.fields.id, &doc.id);
        if let Some(parent_id) = &doc.parent_id {
            tantivy_doc.add_text(self.fields.parent_id, parent_id);
        }
        if let Some(chunk_index) = doc.chunk_index {
            tantivy_doc.add_u64(self.fields.chunk_index, u64::from(chunk_index));
        }
        tantivy_doc.add_text(self.fields.kind, exact_token(&doc.kind));
        tantivy_doc.add_text(self.fields.title_exact, &doc.title);
        tantivy_doc.add_text(self.fields.title, &doc.title);
        tantivy_doc.add_text(self.fields.name, &doc.name);
        tantivy_doc.add_text(self.fields.path, &doc.path);
        tantivy_doc.add_text(self.fields.headings, &doc.headings);
        for category in &doc.categories {
            tantivy_doc.add_text(self.fields.categories, category);
        }
        for tag in &doc.tags {
            tantivy_doc.add_text(self.fields.tags, tag);
        }
        tantivy_doc.add_text(self.fields.body, &doc.body);
        tantivy_doc.add_text(self.fields.body_parts, self.body_parts(doc));
        if let Some(folder) = &doc.folder {
            tantivy_doc.add_text(self.fields.folder, exact_token(folder));
        }
        if let Some(ext) = doc.normalized_ext() {
            tantivy_doc.add_text(self.fields.ext, ext);
        }
        if let Some(mime) = &doc.mime {
            tantivy_doc.add_text(self.fields.mime, exact_token(mime));
        }
        if let Some(status) = &doc.status {
            tantivy_doc.add_text(self.fields.status, exact_token(status));
        }
        if let Some(created_at) = doc.created_at {
            tantivy_doc.add_i64(self.fields.created_at, created_at);
        }
        if let Some(updated_at) = doc.updated_at {
            tantivy_doc.add_i64(self.fields.updated_at, updated_at);
        }
        if self.config.store_excerpt {
            if let Some(excerpt) = doc.bounded_excerpt() {
                tantivy_doc.add_text(self.fields.excerpt, excerpt);
            }
        }
        if !is_empty_metadata(&doc.metadata) {
            tantivy_doc.add_text(
                self.fields.metadata_json,
                serde_json::to_string(&doc.metadata)?,
            );
        }

        for column in &self.fields.promoted {
            let value = match doc.metadata.get(column.name.as_str()) {
                Some(value) => value,
                None => continue,
            };
            match (column.field_type, value) {
                (PromotedFieldType::String, Value::String(text)) => {
                    tantivy_doc.add_text(column.field, exact_token(text));
                }
                (PromotedFieldType::I64, Value::Number(number)) => {
                    let number = number.as_i64().ok_or_else(|| {
                        promoted_error(&column.name, PromotedFieldType::I64, "a non-integer number")
                    })?;
                    tantivy_doc.add_i64(column.field, number);
                }
                (PromotedFieldType::F64, Value::Number(number)) => {
                    let number = number.as_f64().ok_or_else(|| {
                        promoted_error(&column.name, PromotedFieldType::F64, "a non-numeric number")
                    })?;
                    tantivy_doc.add_f64(column.field, number);
                }
                (PromotedFieldType::Bool, Value::Bool(flag)) => {
                    tantivy_doc.add_bool(column.field, *flag);
                }
                (field_type, other) => {
                    return Err(promoted_error(
                        &column.name,
                        field_type,
                        json_type_name(other),
                    ))
                }
            }
        }
        Ok(tantivy_doc)
    }

    fn body_parts(&self, doc: &V2Document) -> String {
        extract_body_parts(&format!("{} {} {}", doc.title, doc.headings, doc.body))
    }

    fn build_query(&self, query: &V2Query) -> Result<Box<dyn Query>, V2Error> {
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        if let Some(text_query) = self.text_query(&query.text, query.fuzzy_edits) {
            clauses.push((Occur::Must, text_query));
        }
        for filter in self.filter_queries(&query.filters)? {
            clauses.push((Occur::Must, filter));
        }
        if clauses.is_empty() {
            clauses.push((Occur::Must, Box::new(AllQuery)));
        }
        Ok(Box::new(BooleanQuery::new(clauses)))
    }

    fn text_query(&self, text: &str, fuzzy_edits: u8) -> Option<Box<dyn Query>> {
        let weights = &self.config.field_weights;
        let exact = exact_token(text);
        let ordered = text_tokens(text);
        let words = dedup(ordered.clone());
        let identifiers = dedup(identifier_tokens(text));
        let parts = dedup(body_part_terms(text));

        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        if !exact.is_empty() && weights.title_exact > 0.0 {
            clauses.push((
                Occur::Should,
                boost(
                    term_query(self.fields.title_exact, &exact),
                    weights.title_exact,
                ),
            ));
        }
        for word in &words {
            for (field, weight) in [
                (self.fields.title, weights.title),
                (self.fields.headings, weights.headings),
                (self.fields.body, weights.body),
            ] {
                if weight > 0.0 {
                    clauses.push((Occur::Should, boost(term_query(field, word), weight)));
                }
            }
        }
        for identifier in &identifiers {
            for (field, weight) in [
                (self.fields.path, weights.path),
                (self.fields.name, weights.name),
                (self.fields.categories, weights.categories),
                (self.fields.tags, weights.tags),
            ] {
                if weight > 0.0 {
                    clauses.push((Occur::Should, boost(term_query(field, identifier), weight)));
                }
            }
        }
        if weights.body_parts > 0.0 {
            for part in &parts {
                clauses.push((
                    Occur::Should,
                    boost(term_query(self.fields.body_parts, part), weights.body_parts),
                ));
            }
        }
        if (PHRASE_MIN_TERMS..=PHRASE_MAX_TERMS).contains(&ordered.len()) {
            for (field, boost_weight) in [
                (self.fields.title, weights.title * TITLE_PHRASE_BOOST),
                (self.fields.body, weights.body * BODY_PHRASE_BOOST),
            ] {
                if boost_weight <= 0.0 {
                    continue;
                }
                let terms: Vec<Term> = ordered
                    .iter()
                    .map(|word| Term::from_field_text(field, word))
                    .collect();
                clauses.push((
                    Occur::Should,
                    boost(Box::new(PhraseQuery::new(terms)), boost_weight),
                ));
            }
        }
        if (PHRASE_MIN_TERMS..=PHRASE_MAX_TERMS).contains(&words.len()) {
            let coverage_fields: Vec<Field> = [
                (self.fields.title, weights.title),
                (self.fields.headings, weights.headings),
                (self.fields.path, weights.path),
                (self.fields.name, weights.name),
                (self.fields.body, weights.body),
            ]
            .into_iter()
            .filter(|(_, w)| *w > 0.0)
            .map(|(f, _)| f)
            .collect();
            if !coverage_fields.is_empty() {
                let must_terms: Vec<(Occur, Box<dyn Query>)> = words
                    .iter()
                    .map(|word| {
                        let any_field: Vec<(Occur, Box<dyn Query>)> = coverage_fields
                            .iter()
                            .map(|field| (Occur::Should, term_query(*field, word)))
                            .collect();
                        (Occur::Must, Box::new(BooleanQuery::new(any_field)) as Box<dyn Query>)
                    })
                    .collect();
                clauses.push((
                    Occur::Should,
                    boost(Box::new(BooleanQuery::new(must_terms)), ALL_TERMS_BOOST),
                ));
            }
        }
        if fuzzy_edits > 0 {
            let distance = fuzzy_edits.min(2);
            let expandable = words
                .iter()
                .filter(|word| word.chars().count() >= FUZZY_MIN_TERM_CHARS && !is_numeric(word))
                .take(FUZZY_MAX_TERMS);
            for word in expandable {
                for (field, weight) in [
                    (self.fields.title, weights.title),
                    (self.fields.body, weights.body),
                ] {
                    if weight <= 0.0 {
                        continue;
                    }
                    let fuzzy: Box<dyn Query> = Box::new(FuzzyTermQuery::new(
                        Term::from_field_text(field, word),
                        distance,
                        true,
                    ));
                    clauses.push((Occur::Should, boost(fuzzy, weight * FUZZY_WEIGHT_FACTOR)));
                }
            }
        }

        if clauses.is_empty() {
            None
        } else {
            Some(Box::new(BooleanQuery::new(clauses)))
        }
    }

    fn filter_queries(&self, filters: &V2Filters) -> Result<Vec<Box<dyn Query>>, V2Error> {
        let mut queries: Vec<Box<dyn Query>> = Vec::new();
        if let Some(folder) = &filters.folder {
            queries.push(term_query(self.fields.folder, &exact_token(folder)));
        }
        if let Some(ext) = &filters.ext {
            let normalized = normalize_extension(ext);
            if !normalized.is_empty() {
                queries.push(term_query(self.fields.ext, &normalized));
            }
        }
        if let Some(kind) = &filters.kind {
            queries.push(term_query(self.fields.kind, &exact_token(kind)));
        }
        if let Some(status) = &filters.status {
            queries.push(term_query(self.fields.status, &exact_token(status)));
        }
        if let Some(mime) = &filters.mime {
            queries.push(term_query(self.fields.mime, &exact_token(mime)));
        }
        if let Some(range) = &filters.created_at {
            queries.push(self.i64_range(self.fields.created_at, *range));
        }
        if let Some(range) = &filters.updated_at {
            queries.push(self.i64_range(self.fields.updated_at, *range));
        }
        for promoted in &filters.promoted {
            queries.push(self.promoted_query(promoted)?);
        }
        Ok(queries)
    }

    fn i64_range(&self, field: Field, range: crate::v2::query::TimeRange) -> Box<dyn Query> {
        let lower = match range.min {
            Some(min) => Bound::Included(min),
            None => Bound::Unbounded,
        };
        let upper = match range.max {
            Some(max) => Bound::Included(max),
            None => Bound::Unbounded,
        };
        let name = self.index.schema().get_field_name(field).to_string();
        Box::new(RangeQuery::new_i64_bounds(name, lower, upper))
    }

    fn promoted_query(
        &self,
        filter: &crate::v2::query::PromotedFilter,
    ) -> Result<Box<dyn Query>, V2Error> {
        let column = self.fields.promoted(&filter.field).ok_or_else(|| {
            V2Error::InvalidQuery(format!(
                "promoted field '{}' is not declared in the collection config",
                filter.field
            ))
        })?;
        if !filter.value.supports(filter.op) {
            return Err(V2Error::InvalidQuery(format!(
                "promoted field '{}' does not support {:?} for value {:?}",
                filter.field, filter.op, filter.value
            )));
        }
        let expected = column.field_type;
        let found = filter.value.field_type();
        if expected != found {
            return Err(promoted_error(
                &column.name,
                expected,
                promoted_type_name(found),
            ));
        }
        let name = self.index.schema().get_field_name(column.field).to_string();
        let query: Box<dyn Query> =
            match (&filter.value, filter.op) {
                (PromotedValue::String(text), _) => term_query(column.field, &exact_token(text)),
                (PromotedValue::Bool(flag), _) => Box::new(TermQuery::new(
                    Term::from_field_bool(column.field, *flag),
                    IndexRecordOption::Basic,
                )),
                (PromotedValue::I64(value), PromotedOp::Eq) => Box::new(TermQuery::new(
                    Term::from_field_i64(column.field, *value),
                    IndexRecordOption::Basic,
                )),
                (PromotedValue::F64(value), PromotedOp::Eq) => Box::new(TermQuery::new(
                    Term::from_field_f64(column.field, *value),
                    IndexRecordOption::Basic,
                )),
                (PromotedValue::I64(value), PromotedOp::Gte) => Box::new(
                    RangeQuery::new_i64_bounds(name, Bound::Included(*value), Bound::Unbounded),
                ),
                (PromotedValue::I64(value), PromotedOp::Lte) => Box::new(
                    RangeQuery::new_i64_bounds(name, Bound::Unbounded, Bound::Included(*value)),
                ),
                (PromotedValue::F64(value), PromotedOp::Gte) => Box::new(
                    RangeQuery::new_f64_bounds(name, Bound::Included(*value), Bound::Unbounded),
                ),
                (PromotedValue::F64(value), PromotedOp::Lte) => Box::new(
                    RangeQuery::new_f64_bounds(name, Bound::Unbounded, Bound::Included(*value)),
                ),
            };
        Ok(query)
    }

    fn phase_b(
        &self,
        searcher: &Searcher,
        candidates: &[(f32, DocAddress)],
        query: &V2Query,
        principals: &HashSet<&str>,
        query_terms: &[String],
        top_k: usize,
    ) -> Result<Vec<V2SearchHit>, V2Error> {
        let mut segments: HashMap<u32, SegmentLight> = HashMap::new();
        let mut parents: HashMap<String, usize> = HashMap::new();
        let mut winners: Vec<Winner> = Vec::with_capacity(top_k.min(candidates.len()));
        let mut acl_filtered = 0usize;
        let mut diversity_filtered = 0usize;
        let mut examined = 0usize;
        let mut scratch = String::new();

        for (score, address) in candidates {
            if winners.len() >= top_k {
                break;
            }
            examined += 1;
            let light = match segments.entry(address.segment_ord) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    entry.insert(SegmentLight::open(searcher, address.segment_ord)?)
                }
            };
            let id =
                fast_field_string(&light.id, address.doc_id, &mut scratch)?.ok_or_else(|| {
                    V2Error::Corrupt(format!(
                        "document {address:?} has no fast 'id' value; the segment is corrupt"
                    ))
                })?;
            if !self.acl.is_visible(&id, principals) {
                acl_filtered += 1;
                continue;
            }
            let parent_id = match &light.parent_id {
                Some(column) => fast_field_string(column, address.doc_id, &mut scratch)?,
                None => None,
            };
            if let (Some(group), Some(parent)) = (query.group_by_parent, parent_id.as_deref()) {
                let count = parents.entry(parent.to_string()).or_insert(0);
                if *count >= group.max_per_parent {
                    diversity_filtered += 1;
                    continue;
                }
                *count += 1;
            }
            winners.push(Winner {
                score: *score,
                address: *address,
                id,
                parent_id,
            });
        }

        let phrase = phrase_text(&query.text);
        let multi_term = query_terms.len() >= 2;
        let mut hits: Vec<V2SearchHit> = Vec::with_capacity(winners.len());
        for winner in winners {
            let stored: TantivyDocument = searcher.doc(winner.address)?;
            let title = text_value(&stored, self.fields.title).unwrap_or_default();
            let path = text_value(&stored, self.fields.path).unwrap_or_default();
            let hit = V2SearchHit {
                id: winner.id,
                parent_id: winner.parent_id,
                chunk_index: stored
                    .get_first(self.fields.chunk_index)
                    .and_then(|value| value.as_u64())
                    .map(|value| value as u32),
                kind: text_value(&stored, self.fields.kind).unwrap_or_default(),
                title,
                path,
                excerpt: text_value(&stored, self.fields.excerpt),
                score: winner.score,
                bm25_score: winner.score,
                explain: None,
            };
            let explain = if query.explain {
                let (matched_terms, coverage) = coverage(query_terms, &hit.title, &hit.path);
                let phrase_bonus =
                    multi_term && !phrase.is_empty() && contains_phrase(&hit.title, &phrase);
                Some(V2Explain {
                    matched_terms,
                    total_terms: query_terms.len(),
                    coverage,
                    phrase_bonus,
                    bm25_score: hit.bm25_score,
                    final_score: hit.score,
                    acl_filtered,
                    diversity_filtered,
                    candidates_considered: examined,
                })
            } else {
                None
            };
            hits.push(V2SearchHit { explain, ..hit });
        }
        Ok(hits)
    }
}

/// Lexical coverage of `query_terms` over the fields a hit can show without the
/// (unstored) body — only computed when the caller asked for an explanation.
fn coverage(query_terms: &[String], title: &str, path: &str) -> (Vec<String>, f32) {
    if query_terms.is_empty() {
        return (Vec::new(), 0.0);
    }
    let mut present: HashSet<String> = HashSet::new();
    present.extend(text_tokens(title));
    present.extend(identifier_tokens(path));

    let mut matched: Vec<String> = Vec::new();
    for term in query_terms {
        if present.contains(term) && !matched.iter().any(|seen| seen == term) {
            matched.push(term.clone());
        }
    }
    let coverage = matched.len() as f32 / query_terms.len() as f32;
    (matched, coverage)
}

/// Resolve one fast-field string value for `doc`.
fn fast_field_string(
    column: &StrColumn,
    doc: u32,
    scratch: &mut String,
) -> Result<Option<String>, V2Error> {
    let Some(ord) = column.term_ords(doc).next() else {
        return Ok(None);
    };
    scratch.clear();
    if !column
        .ord_to_str(ord, scratch)
        .map_err(|error| V2Error::Corrupt(format!("fast field string decode failed: {error}")))?
    {
        return Ok(None);
    }
    Ok(Some(scratch.clone()))
}

enum ApplyOutcome {
    Applied,
    AclOnly,
    Skipped,
}

fn manifest_path(dir: &Path) -> PathBuf {
    dir.join(V2_MANIFEST_FILE)
}

fn tantivy_meta_path(dir: &Path) -> PathBuf {
    dir.join("meta.json")
}

fn fingerprints_path(dir: &Path) -> PathBuf {
    dir.join(V2_FINGERPRINTS_FILE)
}

fn load_fingerprints(dir: &Path) -> Result<HashMap<String, DocFingerprint>, V2Error> {
    let path = fingerprints_path(dir);
    match fs::read(&path) {
        Ok(bytes) => parse_fingerprints(&bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(HashMap::new()),
        Err(error) => Err(V2Error::Io(error)),
    }
}

/// Compact binary form: `magic | version | count | (id_len, id, 3 × 32-byte hash)`.
///
/// The hashes are stored raw instead of hex, which keeps the sidecar small and
/// makes loading a read plus a hash-map insert instead of a JSON parse — this
/// file sits on the read-only open path.
fn save_fingerprints(
    dir: &Path,
    fingerprints: &HashMap<String, DocFingerprint>,
) -> Result<(), V2Error> {
    let mut bytes: Vec<u8> = Vec::with_capacity(9 + fingerprints.len() * 140);
    bytes.extend_from_slice(FINGERPRINT_MAGIC);
    bytes.push(FINGERPRINT_FORMAT_VERSION);
    bytes.extend_from_slice(&(fingerprints.len() as u32).to_le_bytes());
    for (id, fingerprint) in fingerprints {
        push_sized(&mut bytes, id.as_bytes());
        bytes.extend_from_slice(&fingerprint.content_hash);
        bytes.extend_from_slice(&fingerprint.envelope_hash);
        bytes.extend_from_slice(&fingerprint.no_acl_hash);
    }
    write_atomic(&fingerprints_path(dir), &bytes)
}

fn parse_fingerprints(bytes: &[u8]) -> Result<HashMap<String, DocFingerprint>, V2Error> {
    let mut reader = ByteReader::new(bytes);
    let magic = reader
        .take(FINGERPRINT_MAGIC.len())
        .ok_or_else(|| V2Error::Corrupt("fingerprint sidecar is truncated".to_string()))?;
    if magic != FINGERPRINT_MAGIC {
        return Err(V2Error::Corrupt(
            "fingerprint sidecar has bad magic bytes".to_string(),
        ));
    }
    let version = reader
        .u8()
        .ok_or_else(|| V2Error::Corrupt("fingerprint sidecar has no version".to_string()))?;
    if version != FINGERPRINT_FORMAT_VERSION {
        return Err(V2Error::Corrupt(format!(
            "fingerprint sidecar uses unsupported version {version}"
        )));
    }
    let count = reader
        .u32()
        .ok_or_else(|| V2Error::Corrupt("fingerprint sidecar has no count".to_string()))?
        as usize;
    let mut fingerprints: HashMap<String, DocFingerprint> = HashMap::with_capacity(count);
    for _ in 0..count {
        let id = reader.string().ok_or_else(|| {
            V2Error::Corrupt("fingerprint sidecar has a truncated id".to_string())
        })?;
        let content_hash = reader.digest()?;
        let envelope_hash = reader.digest()?;
        let no_acl_hash = reader.digest()?;
        fingerprints.insert(
            id,
            DocFingerprint {
                content_hash,
                envelope_hash,
                no_acl_hash,
            },
        );
    }
    Ok(fingerprints)
}

fn push_sized(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
    bytes.extend_from_slice(value);
}

/// Bounds-checked cursor over a persisted sidecar.
struct ByteReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.position.checked_add(count)?;
        let slice = self.bytes.get(self.position..end)?;
        self.position = end;
        Some(slice)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u32(&mut self) -> Option<u32> {
        let raw: [u8; 4] = self.take(4)?.try_into().ok()?;
        Some(u32::from_le_bytes(raw))
    }

    fn string(&mut self) -> Option<String> {
        let len = self.u32()? as usize;
        let raw = self.take(len)?;
        String::from_utf8(raw.to_vec()).ok()
    }

    fn digest(&mut self) -> Result<[u8; 32], V2Error> {
        let raw = self.take(32).ok_or_else(|| {
            V2Error::Corrupt("fingerprint sidecar has a truncated hash".to_string())
        })?;
        raw.try_into()
            .map_err(|_| V2Error::Corrupt("fingerprint hash has the wrong length".to_string()))
    }
}

/// Write `bytes` to `path` through `<path>.tmp` + `sync_all` + rename.
///
/// Shared by every v2 sidecar: the ACL overlay, the fingerprints, `vectors.bin`
/// and each vec-map generation. On Windows a held destination handle can make
/// the replace fail; the fallback removes the destination and retries.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), V2Error> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    if fs::rename(&tmp, path).is_ok() {
        return Ok(());
    }
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            let _ = fs::remove_file(&tmp);
            return Err(V2Error::Io(error));
        }
    }
    if let Err(error) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(V2Error::Io(error));
    }
    Ok(())
}

fn validate_config(
    persisted: &V2CollectionConfig,
    expected: &V2CollectionConfig,
) -> Result<(), V2Error> {
    let versions = [
        (
            "schema_version",
            persisted.schema_version,
            expected.schema_version,
        ),
        (
            "analyzer_version",
            persisted.analyzer_version,
            expected.analyzer_version,
        ),
        (
            "lexicon_version",
            persisted.lexicon_version,
            expected.lexicon_version,
        ),
    ];
    for (field, found, expected_value) in versions {
        if found != expected_value {
            return Err(V2Error::VersionMismatch {
                field,
                expected: expected_value,
                found,
            });
        }
    }
    if persisted.promoted_fields != expected.promoted_fields {
        return Err(V2Error::SchemaMismatch(format!(
            "promoted fields differ: manifest has {:?}, expected {:?}",
            persisted.promoted_fields, expected.promoted_fields
        )));
    }
    // The vector lane is part of the schema contract: a different dimension or
    // model version would make every stored embedding unreadable, so it is
    // refused instead of silently ignored.
    if persisted.vector_lane != expected.vector_lane {
        return Err(V2Error::SchemaMismatch(format!(
            "vector lane differs: manifest has {:?}, expected {:?}",
            persisted.vector_lane, expected.vector_lane
        )));
    }
    Ok(())
}

/// Reject a lane that could never score a vector, at create and at open.
fn validate_vector_lane(config: &V2CollectionConfig) -> Result<(), V2Error> {
    match &config.vector_lane {
        Some(lane) => lane.validate().map_err(V2Error::InvalidConfig),
        None => Ok(()),
    }
}

fn build_schema(config: &V2CollectionConfig) -> Result<(Schema, V2Fields), V2Error> {
    let mut builder = Schema::builder();

    // `id`/`parent_id` are the only fields Phase B reads before it knows a
    // candidate survived, so they are STRING | STORED | FAST.
    let id = builder.add_text_field("id", raw_fast_text());
    let parent_id = builder.add_text_field("parent_id", raw_fast_text());
    let chunk_index = builder.add_u64_field("chunk_index", numeric_options());
    let kind = builder.add_text_field("kind", exact_stored_text());
    let title_exact = builder.add_text_field("title_exact", exact_indexed_text());
    let title = builder.add_text_field("title", analyzed_stored_text(TOKENIZER_TEXT));
    let name = builder.add_text_field("name", analyzed_stored_text(TOKENIZER_IDENTIFIER));
    let path = builder.add_text_field("path", analyzed_stored_text(TOKENIZER_IDENTIFIER));
    let headings = builder.add_text_field("headings", analyzed_indexed_text(TOKENIZER_TEXT));
    let categories =
        builder.add_text_field("categories", analyzed_indexed_text(TOKENIZER_IDENTIFIER));
    let tags = builder.add_text_field("tags", analyzed_indexed_text(TOKENIZER_IDENTIFIER));
    let body = builder.add_text_field("body", analyzed_indexed_text(TOKENIZER_TEXT));
    let body_parts = builder.add_text_field("body_parts", analyzed_indexed_text(TOKENIZER_TEXT));
    let folder = builder.add_text_field("folder", exact_stored_text());
    let ext = builder.add_text_field("ext", exact_stored_text());
    let mime = builder.add_text_field("mime", exact_stored_text());
    let status = builder.add_text_field("status", exact_stored_text());
    let created_at = builder.add_i64_field("created_at", numeric_options());
    let updated_at = builder.add_i64_field("updated_at", numeric_options());
    let excerpt = builder.add_text_field("excerpt", stored_only_text());
    let metadata_json = builder.add_text_field("metadata_json", stored_only_text());

    let mut promoted: Vec<PromotedColumn> = Vec::with_capacity(config.promoted_fields.len());
    let mut seen: HashSet<&str> = HashSet::new();
    for promoted_field in &config.promoted_fields {
        let name = promoted_field.name.trim();
        if name.is_empty() {
            return Err(V2Error::InvalidConfig(
                "a promoted field name cannot be empty".to_string(),
            ));
        }
        if RESERVED_FIELD_NAMES.contains(&name) {
            return Err(V2Error::InvalidConfig(format!(
                "promoted field '{name}' collides with a reserved field"
            )));
        }
        if !seen.insert(name) {
            return Err(V2Error::InvalidConfig(format!(
                "promoted field '{name}' is declared twice"
            )));
        }
        let field = match promoted_field.field_type {
            PromotedFieldType::String => builder.add_text_field(name, exact_indexed_text()),
            PromotedFieldType::I64 => builder.add_i64_field(name, numeric_options()),
            PromotedFieldType::F64 => builder.add_f64_field(name, numeric_options()),
            PromotedFieldType::Bool => builder.add_bool_field(name, numeric_options()),
        };
        promoted.push(PromotedColumn {
            name: name.to_string(),
            field,
            field_type: promoted_field.field_type,
        });
    }

    let schema = builder.build();
    Ok((
        schema,
        V2Fields {
            id,
            parent_id,
            chunk_index,
            kind,
            title_exact,
            title,
            name,
            path,
            headings,
            categories,
            tags,
            body,
            body_parts,
            folder,
            ext,
            mime,
            status,
            created_at,
            updated_at,
            excerpt,
            metadata_json,
            promoted,
        },
    ))
}

/// `STRING | STORED | FAST` over the raw tokenizer: one exact value per
/// document, readable from the fast-field column without touching the store.
fn raw_fast_text() -> TextOptions {
    TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("raw")
                .set_index_option(IndexRecordOption::Basic),
        )
        .set_fast(Some("raw"))
        .set_stored()
}

fn exact_stored_text() -> TextOptions {
    TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(TOKENIZER_EXACT)
                .set_index_option(IndexRecordOption::Basic),
        )
        .set_stored()
}

fn exact_indexed_text() -> TextOptions {
    TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer(TOKENIZER_EXACT)
            .set_index_option(IndexRecordOption::Basic),
    )
}

fn analyzed_stored_text(tokenizer: &str) -> TextOptions {
    analyzed_indexed_text(tokenizer).set_stored()
}

fn analyzed_indexed_text(tokenizer: &str) -> TextOptions {
    TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer(tokenizer)
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    )
}

fn stored_only_text() -> TextOptions {
    TextOptions::default().set_stored()
}

fn numeric_options() -> NumericOptions {
    NumericOptions::default()
        .set_indexed()
        .set_fast()
        .set_stored()
}

fn term_query(field: Field, text: &str) -> Box<dyn Query> {
    Box::new(TermQuery::new(
        Term::from_field_text(field, text),
        IndexRecordOption::Basic,
    ))
}

fn boost(query: Box<dyn Query>, weight: f32) -> Box<dyn Query> {
    Box::new(BoostQuery::new(query, weight))
}

fn text_value(document: &TantivyDocument, field: Field) -> Option<String> {
    document
        .get_first(field)
        .and_then(|value| value.as_str())
        .map(|text| text.to_string())
}

fn fingerprint_from_envelope(envelope: &IndexedEnvelope) -> DocFingerprint {
    DocFingerprint {
        content_hash: compute_content_digest(&envelope.indexable_text),
        envelope_hash: compute_envelope_digest(envelope),
        no_acl_hash: envelope_digest_without_acl(envelope),
    }
}

/// Digest of `envelope` with its ACL removed, without re-serialising to hex.
fn envelope_digest_without_acl(envelope: &IndexedEnvelope) -> [u8; 32] {
    let mut stripped = envelope.clone();
    stripped.acl.clear();
    compute_envelope_digest(&stripped)
}

fn fingerprint_from_document(doc: &V2Document, config: &V2CollectionConfig) -> DocFingerprint {
    fingerprint_from_envelope(
        &doc.to_indexed_envelope(config.schema_version, config.analyzer_version),
    )
}

fn normalize_groups(groups: &[String]) -> Vec<String> {
    let mut normalized: Vec<String> = groups
        .iter()
        .map(|group| group.trim().to_string())
        .filter(|group| !group.is_empty())
        .collect();
    normalized.sort_unstable();
    normalized.dedup();
    normalized
}

fn distinct_terms(text: &str) -> Vec<String> {
    dedup(text_tokens(text))
}

fn dedup(values: Vec<String>) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::with_capacity(values.len());
    let mut out: Vec<String> = Vec::with_capacity(values.len());
    for value in values {
        if seen.insert(value.clone()) {
            out.push(value);
        }
    }
    out
}

fn is_numeric(token: &str) -> bool {
    token.chars().all(|character| character.is_ascii_digit()) || token.parse::<f64>().is_ok()
}

/// True when a document's metadata bag carries nothing worth a stored field.
fn is_empty_metadata(metadata: &Value) -> bool {
    match metadata {
        Value::Null => true,
        Value::Object(map) => map.is_empty(),
        _ => false,
    }
}

fn phrase_text(text: &str) -> String {
    fold_umlauts(text.trim())
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn contains_phrase(text: &str, phrase: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    let haystack = fold_umlauts(text);
    haystack.contains(phrase)
}

fn promoted_error(name: &str, expected: PromotedFieldType, found: &str) -> V2Error {
    V2Error::PromotedFieldType {
        field: name.to_string(),
        expected: promoted_type_name(expected),
        found: found.to_string(),
    }
}

fn promoted_type_name(field_type: PromotedFieldType) -> &'static str {
    match field_type {
        PromotedFieldType::String => "String",
        PromotedFieldType::I64 => "I64",
        PromotedFieldType::F64 => "F64",
        PromotedFieldType::Bool => "Bool",
    }
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "Null",
        Value::Bool(_) => "Bool",
        Value::Number(_) => "Number",
        Value::String(_) => "String",
        Value::Array(_) => "Array",
        Value::Object(_) => "Object",
    }
}

impl V2Index {
    /// Lineage recorded in the watermark, if any.
    ///
    /// `min_available_seq` is not tracked by the index (it is a source
    /// property); it is reported as `0`.
    pub fn source_lineage(&self) -> Option<SourceLineage> {
        self.manifest
            .watermark
            .as_ref()
            .map(|watermark| SourceLineage {
                source_epoch_uuid: watermark.source_epoch_uuid.clone(),
                last_seq: watermark.synced_seq,
                min_available_seq: 0,
            })
    }
}
