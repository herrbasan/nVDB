# nVDB Project - Agent Guide

## Project Overview

nVDB is a **hybrid retrieval engine** for document corpora, written in Rust. Two surfaces live in one crate:

- **v2 (current, production)** — `src/v2/`: Tantivy BM25F lexical retrieval, a `content_hash`-keyed dense vector lane, RRF hybrid fusion, ONNX cross-encoder Level-2 reranking, generation-stamped ACLs and the Phase 0 sync/integrity contract. Exposed as a resident HTTP service via `nvdb serve` (`src/bin/nvdb.rs` → `src/v2/server.rs`) and as the `nvdb::v2` Rust API.
- **v1 (legacy library)** — `Database`/`Collection`, LSM-Lite segments + WAL, HNSW, payload filter DSL (`src/lib.rs`, `src/search.rs`, `src/hnsw.rs`, `src/filter.rs`). Still compiled, tested and reachable through the CLI administration commands, but it is not what production consumers use.

nVDB is a **derived index**: never the source of truth, always rebuildable. Full text lives in the authoritative store (`nDB`); nVDB stores only the hit envelope plus per-document content/envelope hashes.

### Production consumers

Both talk to `nvdb serve` over localhost HTTP — nothing links the crate directly.

| Consumer | Integration |
|---|---|
| `mcp_server` | `src/agents/vdb/` — `v2-client.js` (HTTP client), `storage-connector.js`, `search-options.js` (per-request query-shape RRF overrides), `memory` agent (`memory` collection) |
| `LLM-Gateway-Chat` | retrieval over its document corpus through the same HTTP API |

Default deployment: `nvdb serve --dir <path> --host 127.0.0.1 --port 3510 --reranker-model <dir>` from a `--features rerank-onnx` build.

### Key Characteristics

| Attribute | Decision | Rationale |
|-----------|----------|-----------|
| **Language** | Rust (Edition 2021, MSRV 1.75) | Memory safety, SIMD, no GC pauses |
| **Deployment** | v2: standalone HTTP service (`nvdb serve`); v1: embedded library | Single node, one process per collection root |
| **Lexical index** | Tantivy (`tantivy` 0.22) | Pure Rust, MIT, proven BM25F |
| **Storage (v2)** | Tantivy segments + binary sidecars (`v2_manifest.json`, `acl_overlay.gen_N.bin`, `v2_fingerprints.bin`, `vectors.bin`, `vec_map.gen_N.bin`) | Cheap open, no JSON parsing on the read path |
| **Storage (v1)** | LSM-Lite (memtable + mmap segments) + WAL | Instant writes, zero-copy reads, durability |
| **Serialization** | `rkyv` (v1), length-prefixed binary + JSON manifests (v2) | Zero-copy where it matters |
| **Validation** | Checksum-once, trust-after | BLAKE3/CRC64 in header; validate on open only |
| **Index** | HNSW (v1 only); v2 dense lane scans flat | No ANN for v2 today — stated, not implied |
| **SIMD** | `wide` crate | Stable Rust, portable (AVX2/AVX-512/NEON) |
| **Reranking** | Optional ONNX cross-encoder, `rerank-onnx` feature | No model artifacts in a default build |
| **Durability** | Configurable (v1) | `Buffered` vs `FdatasyncEachBatch` |
| **Safety** | Multi-process locks | `flock` prevents dual-writer corruption |

### Project Structure

```
.
├── Cargo.toml              # Package manifest
├── Cargo.lock              # Dependency lock file
├── src/
│   ├── lib.rs              # Main library, Database and Collection types
│   ├── error.rs            # Error types and Result alias
│   ├── distance.rs         # SIMD distance functions (dot, cosine, euclidean)
│   ├── search.rs           # Search API and exact search implementation
│   ├── sync.rs             # Phase 0 sync/snapshot/ACL/revision contract
│   ├── filter.rs           # Mongo-like Filter DSL
│   ├── hnsw.rs             # HNSW index for approximate search
│   ├── segment.rs          # Memory-mapped segment files
│   ├── memtable.rs         # In-memory write buffer (HashMap + SoA)
│   ├── wal.rs              # Write-ahead log for crash recovery
│   ├── manifest.rs         # Collection manifest (segments, WAL seq)
│   ├── compaction.rs       # Segment compaction and index rebuild
│   ├── lock.rs             # Multi-process locking (flock)
│   ├── id.rs               # Internal/external ID mapping
│   ├── rerank.rs           # Reranker trait (v1 + v2 Level-2 use it)
│   ├── rerank_onnx.rs      # OnnxReranker, `rerank-onnx` feature only
│   ├── bin/nvdb.rs         # CLI (v1 admin commands + `serve` for the v2 HTTP service)
│   └── v2/                 # Phase 2/3: Tantivy lexical + hybrid retrieval (additive)
│       ├── mod.rs          # Public surface + V2Error
│       ├── schema.rs       # V2CollectionConfig, VectorLaneConfig, V2Document, hashes
│       ├── query.rs        # V2Query, V2HybridQuery, filters, hits, explains
│       ├── analyzer.rs     # nvdb_exact / nvdb_identifier / nvdb_text, body parts
│       ├── german_lexicon.rs  # Frozen German morpheme lexicon + splitter
│       ├── acl_overlay.rs  # Generation-stamped ACL sidecar
│       ├── vectors.rs      # vectors.bin + vec_map.gen_<N>.bin, pending list
│       ├── level2.rs       # Level-2: source ACL re-check, reranker, floor
│       ├── engine.rs       # V2Index: manifest, Phase A + B, hybrid, sync
│       └── server.rs       # `nvdb serve`: std-only HTTP/1.1 front end
├── tests/                  # Integration tests by phase
│   ├── phase0_sync_contract_tests.rs
│   ├── phase2_v2_engine_tests.rs
│   ├── phase2_v2_server_tests.rs
│   ├── phase3_v2_hybrid_tests.rs
│   ├── phase1a_segment_tests.rs
│   ├── phase1a_locking_tests.rs
│   ├── phase2_search_tests.rs
│   ├── phase3_hnsw_tests.rs
│   ├── phase4_filter_tests.rs
│   └── phase5_compaction_tests.rs
├── benches/                # Criterion benchmarks (v1 paths)
│   ├── bench_search.rs
│   ├── bench_insert.rs
│   └── bench_recovery.rs
├── documentation/          # Reference docs: v2-server.md (current) plus
│                           # architecture/rust-api/filter-dsl (v1 crate, clearly scoped)
├── docs/                   # Working notes and history
│   ├── README.md           # Docs index
│   ├── user-guide.md       # v1 API guide
│   ├── integration-guide.md
│   ├── scope-and-boundaries.md
│   ├── test-documentation.md
│   ├── adr/                # v1 architecture decision records
│   └── archive/            # historical plans and phase handovers
└── examples/               # v1 usage examples (web_service.rs is commented out)
```

There are no Node.js bindings. The `napi/` addon and its `cdylib` workspace member were removed on 2026-11-22; nVDB is consumed as a Rust crate or through the `nvdb serve` HTTP API. If a task asks for Node bindings, treat that as a new project.

### Core Dependencies

```toml
memmap2 = "0.9"          # Memory-mapped files
rkyv = "0.7"             # Zero-copy serialization
serde = "1.0"            # Serialization framework
serde_json = "1.0"       # JSON support
arc-swap = "1.6"         # Atomic data structure updates
wide = "0.7"             # Portable SIMD
thiserror = "1.0"        # Error derive macros
crc32fast = "1.3"        # CRC32 checksums
parking_lot = "0.12"     # Synchronization primitives
blake3 = "1.5"           # Cryptographic hashing
byteorder = "1.5"        # Byte order handling
bincode = "1.3"          # Binary serialization
fastrand = "2.0"         # Fast random number generation
tantivy = "0.22"         # Phase 2 lexical engine (MIT, pure Rust)
ort = "2.0.0-rc.15"      # Optional, `rerank-onnx` only
tokenizers = "0.22.2"    # Optional, `rerank-onnx` only
```

Phase 3 adds **no** crate: the vector lane reuses `crate::distance` (SIMD),
`crate::rerank::Reranker`, `crate::sync`, `blake3`, `parking_lot` and `serde`.

---

## Build and Test Commands

### Building

```bash
# Build debug version
cargo build

# Build release version with the Level-2 ONNX reranker
cargo build --release --features rerank-onnx

# Build and open documentation
cargo doc --open
```

`rerank-onnx` is the only feature (`Cargo.toml`: `rerank-onnx = ["dep:ort", "dep:tokenizers"]`). Without it, `nvdb::rerank_onnx` and `--reranker-model` do not exist; with it, a model that fails to load aborts startup.

### Running the HTTP service

```bash
# default: 127.0.0.1:3510, no reranker
nvdb serve --dir <collection-root>

# with Level-2 reranking (needs a rerank-onnx build and <dir>/model.onnx + <dir>/tokenizer.json)
nvdb serve --dir <collection-root> --host 0.0.0.0 --port 3510 --reranker-model <model-dir>
```

Every direct subdirectory of `--dir` holding a `v2_manifest.json` is adopted at startup; one that fails to open aborts the process. `GET /health` reports `reranker_loaded`, which is how you tell "not configured" apart from "configured but rejected a request".

### Testing

```bash
# Run all tests (lib + every integration suite)
cargo test

# Run specific test suite
cargo test --test phase0_sync_contract_tests
cargo test --test phase2_v2_engine_tests
cargo test --test phase1a_segment_tests
cargo test --test phase1a_locking_tests
cargo test --test phase2_search_tests
cargo test --test phase2_v2_server_tests
cargo test --test phase3_v2_hybrid_tests
cargo test --test phase3_hnsw_tests
cargo test --test phase4_filter_tests
cargo test --test phase5_compaction_tests

# Run with output visible
cargo test -- --nocapture

# Run in release mode (for performance validation)
cargo test --release

# Run property-based tests only
cargo test proptest
```

### Benchmarking

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench bench_search
cargo bench --bench bench_insert
cargo bench --bench bench_recovery
```

### Running Examples

```bash
# Basic usage example (v1 API)
cargo run --example basic_usage

# RAG system example (v1 API)
cargo run --example rag_system
```

`examples/web_service.rs` is fully commented out and references an `axum` feature that does not exist — do not advertise it. The v2 service is built in, not an example.

---

## Code Style Guidelines

### Rust Style

- **Edition**: 2021
- **MSRV**: 1.75
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting (warnings enabled for missing docs and rust_2018_idioms)

### Documentation

- All public items must have doc comments (`#![warn(missing_docs)]`)
- Use `//!` for module-level documentation
- Include examples in doc comments where appropriate
- Mark unsafe code with `SAFETY:` comments

### Error Handling

- Use `thiserror` for error enums
- All errors are explicit (`Result<T, Error>`)
- No panics in library code
- Provide context for I/O errors using `Error::io_err()`

```rust
// Good: Explicit error with context
pub fn io_err(path: impl Into<PathBuf>, context: impl Into<String>) -> impl FnOnce(std::io::Error) -> Self

// Good: Specific error variants
#[error("collection '{name}' is locked by another process")]
CollectionLocked { name: String }
```

### Naming Conventions

- Types: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Modules: `snake_case`

### Module Organization

```rust
// lib.rs pattern: re-export public API
pub mod error;
pub mod distance;
// ...

// Re-export commonly used items
pub use error::{Error, Result};
pub use distance::Distance;
// ...
```

---

## Testing Instructions

### Test Organization

| Location | Purpose |
|----------|---------|
| `src/*/tests` | Unit tests within modules (using `#[cfg(test)]`) |
| `tests/` | Integration tests by development phase |
| `benches/` | Criterion benchmarks |

### Writing Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_description() {
        // Arrange
        let input = ...;
        
        // Act
        let result = function_under_test(input);
        
        // Assert
        assert_eq!(result, expected);
    }

    // Use proptest for property-based testing
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_invariant_holds(input in strategy()) {
            prop_assert!(check_invariant(input));
        }
    }
}
```

### Writing Integration Tests

```rust
// tests/phaseX_feature_tests.rs
use nVDB::{Database, CollectionConfig, Document, /* ... */};
use tempfile::TempDir;

#[test]
fn test_scenario_description() {
    let temp_dir = TempDir::new().unwrap();
    let db = Database::open(temp_dir.path()).unwrap();
    
    // Test code...
}
```

### Test Data Guidelines

- Use `tempfile::TempDir` for test isolation
- Use sequential values (0.0, 1.0, 2.0...) for deterministic vectors
- Test dimensions: 4 (unit tests), 384, 768, 1536 (integration)
- Clean up is automatic via `TempDir` drop

### Property-Based Testing

Use `proptest` for verifying invariants:

```rust
proptest! {
    #[test]
    fn prop_simd_scalar_match(
        a in prop::collection::vec(-10.0f32..10.0, 1..100),
        b in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let simd = simd_function(&a, &b);
        let scalar = scalar_function(&a, &b);
        prop_assert!((simd - scalar).abs() < 1e-3);
    }
}
```

---

## Security Considerations

### Input Validation

- Vector dimensions are validated on insert
- Document IDs must be valid strings
- JSON payloads are validated during deserialization

```rust
// Dimension validation
if doc.vector.len() != self.config.dim {
    return Err(Error::WrongDimension {
        expected: self.config.dim,
        got: doc.vector.len(),
    });
}
```

### File Security

- Checksum verification on segment open (BLAKE3)
- CRC32 on WAL records
- Atomic file updates (write-to-temp + rename)
- `flock` for single-writer enforcement across processes

### Memory Safety

- All `unsafe` code must have `// SAFETY:` comments
- Prefer safe Rust; unsafe only for SIMD or mmap access
- mmap is read-only; no mutable aliasing

### Potential Risks

| Risk | Mitigation |
|------|------------|
| SIGBUS on storage error | File size check before mmap; `madvise(MADV_POPULATE_READ)` |
| Data corruption | Checksum verification; WAL for recovery |
| Concurrent write | `flock` exclusive lock |
| Unbounded memory | WAL flush threshold (64MB); compaction |

### Unsafe Code Guidelines

When using `unsafe`:

```rust
// SAFETY: Vector is 64-byte aligned, dimension matches
unsafe {
    let ptr = vector_data.as_ptr() as *const f32x8;
    // ...
}
```

---

## Architecture Overview

### Storage Model (LSM-Lite — v1 crate)

The v1 surface below is still compiled and tested but is not used by production consumers; the v2 engine stores Tantivy segments plus binary sidecars (see "Lexical Retrieval Engine" below).

| Component | Purpose | Mutability |
|-----------|---------|------------|
| **Memtable** | Recent writes in RAM | Mutable (WAL-backed) |
| **Segments** | Immutable historical data | Read-only, mmap'd |
| **WAL** | Durability log | Append-only |
| **HNSW** | ANN index | Rebuilt on compaction |
| **Manifest** | Atomic state transitions | Atomic rename only |

### Write Path

1. Append to WAL (fdatasync if `Durability::FdatasyncEachBatch`)
2. Insert into memtable (`HashMap` + SoA buffer)
3. Return immediately
4. When WAL exceeds threshold (64MB) or `flush()` called:
   - Freeze memtable → write to new segment
   - Update manifest → reset WAL

### Read Path

1. Search memtable (HashMap lookup or SoA scan)
2. Search all segments (mmap'd, zero-copy)
3. Merge results

### Concurrency Model

| Component | Access Pattern | Implementation |
|-----------|---------------|----------------|
| Segments | Immutable, lock-free | mmap + Arc |
| Segment list | Atomic updates | `ArcSwap<Vec<Arc<Segment>>>` |
| Memtable | Read-heavy, write-exclusive | `RwLock<HashMap>` + SoA buffer |
| WAL | Append-only, single writer | `Mutex<File>` |
| Manifest | Atomic replace only | Write-temp + rename |

**Invariants:**
- Single writer per collection (enforced by `flock`)
- Readers never block (immutable segments)
- Writers don't block readers (copy memtable, swap Arc)

### Sync & Indexing Contract (`src/sync.rs`, Phase 0)

nVDB is a *derived* index: never the source of truth, always rebuildable.

- **Lineage:** sources expose `(source_epoch_uuid, last_seq, min_available_seq)`; the index commits the highest *contiguous* `synced_seq`.
- **Divergence:** a changed epoch, `synced_seq > last_seq` (restored backup), or `CursorExpired` refuses incremental catch-up and forces a clean snapshot rebuild — never silent divergence.
- **Dual-write failure:** a failed apply sets `degraded = true` in the persisted watermark; reads fail with `IndexDegraded` until a rebuild heals.
- **Two hashes:** `content_hash` = searchable text only (gates embedding reuse); `envelope_hash` = canonical full envelope incl. sorted ACL + canonical JSON metadata (gates no-op skip, so ACL/metadata-only updates are never dropped).
- **Fail-closed reads:** empty principals → `MissingPrincipals`; any catch-up lag → `BootHealIncomplete`; Level-2 hydration re-checks ACL against the authoritative source → `AclDenied`.
- **Vectors:** pending = `live ∧ ¬has_vector` (no separate queue); `attach_vectors` is compare-and-set on `(id, content_hash, model_version)`; stale completions fail loudly with `StaleVectorCas`.

### Lexical Retrieval Engine (`src/v2/`, Phase 2)

Phase 1's kill-switch verdict: **embed Tantivy (MIT, pure Rust)** behind our own API rather than writing an inverted index. `src/v2/` is additive — v1 (`Collection`, `src/lexical.rs`) is untouched.

**Files on disk (one collection directory):**

| Artifact | Written | Meaning |
|----------|---------|---------|
| `meta.json`, `*.term`, `*.idx`, `*.store` | Tantivy | immutable segments; only the hit envelope is stored (`body`, `headings`, `body_parts` are indexed but **not** stored) |
| `v2_manifest.json` | atomic rename | `config`, `acl_generation`, `watermark`, `doc_count` |
| `acl_overlay.gen_<N>.bin` | tmp + `sync_all` + rename | authoritative ACL map per generation; compact binary, group sets interned (`doc_id -> set index`) |
| `v2_fingerprints.bin` | atomic rename | per-document `content_hash` / `envelope_hash` / `no_acl_hash` as raw 32-byte BLAKE3 digests |
| `vectors.bin` | append (v1 files: one-time tmp + rename upgrade) | content-addressed embeddings, key `(content_hash, model_version)` — identical text is embedded once |
| `vec_map.gen_<N>.bin` | tmp + `sync_all` + rename | `doc_id -> content_hash` links per generation; the manifest names the live one (`vec_generation`) and superseded generations are deleted after the swap |

Both sidecars are length-prefixed binary, not JSON: `open()` is a read path, and parsing megabytes of JSON (26k entries) cost ~20 ms for nothing. An `IndexWriter` is created on the **first write**, never on open.

**Open-time contract:** `schema_version`, `analyzer_version` and `lexicon_version` must match the manifest, and `promoted_fields` must be identical — otherwise `V2Error::VersionMismatch` / `SchemaMismatch`. A differing analyzer is never silently tolerated. (`schema_version` 2 = unstored `body`, fast `id`/`parent_id`, binary sidecars; `analyzer_version` 2 = one canonical token per word.)

**Analyzers** (registered on every open): `nvdb_exact` (trim + lowercase, whole string), `nvdb_identifier` (whole token + sub-tokens on `/ \ . - _ :` **plus** the extension-stripped stem of the last path segment when that segment ends in a 1–5 character ASCII-alphanumeric extension, so `md-blocks-spec` matches `md-blocks-spec.md` as one token), `nvdb_text` (Unicode word segmentation + lowercase, then `ä->ae, ö->oe, ü->ue, ß->ss` folded into **one** canonical token per word, so `Förderprogramm` and `Foerderprogramm` are the same term and BM25 length norms stay exact).

**Stored vs indexed fields:** `id`, `parent_id` (both `STRING | STORED | FAST`), `chunk_index`, `kind`, `title`, `path`, `name`, `folder`, `ext`, `mime`, `status`, `created_at`, `updated_at`, non-empty `metadata`, and the opt-in `excerpt` (≤ 1 KB) are stored. `title_exact`, `headings`, `categories`, `tags`, `body`, `body_parts` are indexed only — `nVDB` is a derived indexer, full content lives in `nDB`.

**German compounds:** `decompose_german_compound` is a deterministic fewest-parts DP over a frozen, versioned lexicon (500+ morphemes, umlauts folded, Fugenlaute `s/es/n/en/er`, parts ≥ 4 chars). Index time writes `body_parts` (lexicon prefixes of every word + nested decomposition); query time searches the *same* term set against `body_parts` only, at `FieldWeights::body_parts = 0.35` — strictly below `body` (1.0) and `title` (4.0), so a document holding the exact compound always outranks one holding only its parts. Both directions therefore work: compound query → component document, component query → compound document. `GERMAN_LEXICON_VERSION` gates the index.

**Two-phase search:**
- *Phase A* — one weighted `BooleanQuery` over `title_exact`, `title`, `path`, `name`, `headings`, `categories`, `tags`, `body`, `body_parts` (`FieldWeights`; defaults `8.0 / 4.0 / 3.0 / 3.0 / 2.5 / 2.0 / 2.0 / 1.0 / 0.35`), with `Occur::Must` columnar filters (`folder`, `ext`, `kind`, `status`, `mime`, `created_at`/`updated_at` ranges, typed promoted columns), a postings-only `PhraseQuery` boost for queries of 2..=32 terms — `TITLE_PHRASE_BOOST` ×2 on `title`, `BODY_PHRASE_BOOST` ×10 on `body` (applied to the field weight), an `ALL_TERMS_BOOST` ×4 conjunction clause, and, when `fuzzy_edits > 0`, a distance-1 `FuzzyTermQuery` (transposition cost 1) at `×0.25` per non-numeric term ≥ 4 chars on `title` and `body`, capped at the first 6 terms. A single-token query that is exactly a filename stem also gets `NAME_EXACT_BOOST` ×6 against `name` and `path`; split parts stay unboosted and multi-word queries are unaffected (herrbasan/nVDB#4). Fetch `max(top_k * 20, 200)`, widening ×5 up to 4 rounds. Zero document-store reads.
- *Phase B* — walk the Phase A order and read **only** `id`/`parent_id` from fast fields to enforce the authoritative ACL overlay (fail-closed) and the native `group_by_parent` cap; stop as soon as `top_k` winners exist; hydrate the stored envelope for those winners alone. `explain` (coverage, matched terms, phrase flag, counters) is computed only when `query.explain == true`, from the stored `title`/`path`.

**ACL:** never read from segments. `update_acl_only` bumps the overlay generation and rewrites overlay + manifest atomically — no re-index, immediately visible. An absent document, an empty group list, or empty principals is not visible.

**Sync:** `sync_from` implements the full Phase 0 contract — epoch/cursor-ahead divergence and `CursorExpired` force `rebuild_from_snapshot`; an unchanged `envelope_hash` is skipped; an unchanged `content_hash` with an unchanged no-ACL hash is applied through `update_acl_only`; the watermark commits atomically. `integrity_check` compares live hashes against the source snapshot and verifies reader/overlay alignment.

**Diversity/typo rule of thumb:** `fuzzy_edits` is opt-in; exact field weights keep a correct title ahead of a fuzzy match, so recall is added without reordering what the user typed correctly.

**HTTP front end (`src/v2/server.rs`, `nvdb serve`):** hand-rolled HTTP/1.1 on `std::net` — no new crate, same shape as `nDB`'s daemon (thread per connection behind a cap, keep-alive, 30 s I/O timeouts, 64 MB body cap). `V2Server` owns `root_dir: PathBuf` and `collections: RwLock<HashMap<String, Arc<RwLock<V2Index>>>>`; startup adopts every direct subdirectory holding a `v2_manifest.json` and **fails loud** if one will not open.

```
GET  /health                            -> {"ok":true,"engine":"nvdb-v2","collections":[...],"reranker_loaded":bool}
GET  /v2/collections                    -> {"collections":[{name,doc_count,acl_generation,watermark}]}
POST /v2/collections                    <- {"name","config"?}
POST /v2/collections/{name}/upsert      <- {"docs":[V2Document],"commit"?=true} -> {"upserted":N}
POST /v2/collections/{name}/delete      <- {"ids":[...],"commit"?=true}         -> {"deleted":N}
POST /v2/collections/{name}/acl         <- {"id","acl":[...],"commit"?=true}    -> {"ok":true,"acl_generation":N}
POST /v2/collections/{name}/commit      -> {"ok":true,"doc_count":N}
POST /v2/collections/{name}/search      <- V2Query JSON -> {"hits":[V2SearchHit],"took_ms":f64}
POST /v2/collections/{name}/hybrid_search <- V2HybridQuery JSON -> {"hits":[V2HybridHit],"took_ms":f64}
GET  /v2/collections/{name}/pending_embeddings?limit=100 -> {"count":N,"pending":[PendingEmbedding]}
POST /v2/collections/{name}/attach_vectors <- {"attachments":[VectorAttachment]} -> {"attached":N,"vec_generation":N}
GET  /v2/collections/{name}/fingerprints -> {"doc_count":N,"watermark":SyncWatermark?,"entries":[FingerprintEntry]}
POST /v2/collections/{name}/integrity_check <- {"expected":[ExpectedFingerprint],"check_vectors"?=false} -> IntegrityDiff
POST /v2/collections/{name}/watermark    <- SyncWatermark -> {"ok":true,"watermark":SyncWatermark}
POST /v2/collections/{name}/level2       <- {"query_text","principals","hits":[...],"texts"?,"shortlist"?,"rerank_weight"?,"relevance_floor"?} -> V2Level2Result
```

Errors are `{"error": "...", "code": "..."}` with an explicit status: `400` malformed body / invalid name (traversal included) / `MissingPrincipals` / `WrongDimension` / `vector_lane_not_configured` / `invalid_watermark`, `403` a non-v2 directory the server refuses to take over, `404` unknown route or collection, `409` config/version conflict or a stale vector attachment (`stale_vector_cas`), `500` IO, corruption or a reranker that broke its contract (`rerank_failed`), `503` `IndexDegraded` (the degraded boot-heal watermark) or `reranker_not_configured` (`level2` on a server started without `--reranker-model`). Collection names are `[a-zA-Z0-9._-]+` and `.`/`..` are rejected, so a name can never escape `root_dir`. `V2Query` is the wire type (`Deserialize` with `deny_unknown_fields`); `V2SearchHit`/`V2Explain` are `Serialize`.

`start_in_background(root, addr)` (with `127.0.0.1:0`) returns a handle carrying the bound `SocketAddr`; `shutdown()` stops the accept loop, waits for in-flight requests and drops every index so the Tantivy writer lock is released and a restart on the same directory works. `serve_blocking` is the CLI path: `nvdb serve --dir <path> [--host 127.0.0.1] [--port 3510] [--reranker-model <dir>]`. `start_in_background_with_reranker` / `serve_blocking_with_reranker` (and `V2Server::with_reranker`) attach an `Arc<dyn Reranker>`; without one `/level2` answers `503 reranker_not_configured` rather than silently dropping to Level-1 ordering. Under `--features rerank-onnx`, `--reranker-model <dir>` loads `<dir>/model.onnx` + `<dir>/tokenizer.json` and **fails startup** if either will not load; without the feature the flag is a startup error rather than a silent downgrade.

**Fingerprint exchange.** `V2Index::fingerprint_entries()` returns `{id, content_hash, envelope_hash, has_vector, acl}` per tracked document, sorted by id, with the hashes in the same hex form `compute_content_hash`/`compute_envelope_hash` produce. `integrity_check_entries(&expected, check_vectors)` compares caller-supplied `{id, content_hash, envelope_hash}` triples against the live fingerprints (hex compared case-insensitively) and returns `IntegrityDiff { clean, missing, extra, content_mismatched, envelope_mismatched, missing_vectors }`; `missing_vectors` names documents the index *tracks* without an embedding and is only populated when `check_vectors` is set. `set_watermark(SyncWatermark)` validates the watermark invariants and persists `v2_manifest.json` atomically — the route validates the wire form (optional `degraded`/`degraded_reason`) and answers `400 invalid_watermark`.

**HTTP Level 2.** The route re-runs the Level-2 contract against the **ACL overlay** instead of a `SourceConnector`: Level-1 hits whose id the overlay does not grant to the caller are dropped before anything else, so a revoked document is neither scored nor echoed (Level 1 is evidence, not permission). Survivors are capped at `shortlist` (20) *after* the ACL filter; text is `texts[id]`, else `hit.excerpt`, else `title + " " + path`; the reranker is verified (exactly one finite score per document, else `500 rerank_failed`) and fused with the Level-1 order by the same weighted RRF (`LEVEL2_RRF_K = 60`, `1.0` vs `rerank_weight`, default `0.2`). `best_rerank_score < relevance_floor` (default `0.02`) sets `rejected = true` while keeping the hits as low-relevance evidence.

### Vector Lane & Unified Hybrid Search (`src/v2/vectors.rs`, `level2.rs`, Phase 3)

**Opt-in and cheap.** `V2CollectionConfig::vector_lane: Option<VectorLaneConfig>` defaults to
`None`; a lexical-only collection therefore writes no vector file, answers
`next_pending_embeddings` with an empty list and rejects `attach_vectors` /
dense search with `VectorLaneNotConfigured` instead of guessing. The lane is
`{ dim, distance (default Cosine), model_version, flat_search_cutoff }` and is
part of the open-time schema contract — a changed dimension or model version is a
`SchemaMismatch`, never a silent reinterpretation of stored embeddings.

**Keyed by content, never by id.** `vectors.bin` maps
`(content_hash, model_version) -> Vec<f32>`, so two documents with identical
searchable text share one embedding. The file is **append-only**: `attach_vectors`
appends new records instead of rewriting the store, and a legacy v1 fixed-count
file is upgraded once, atomically, on the first append
(`VECTORS_FORMAT_VERSION_V1`). `vec_map.gen_<N>.bin` holds the mutable
`doc_id -> content_hash` links, generation-stamped like the ACL overlay:
new generation file → `manifest.vec_generation` swap → cleanup of superseded
generations. **Immutable segments are never touched by a vector attach.**

| Event | Effect on the link |
|---|---|
| ACL-only or metadata-only upsert (same `content_hash`) | kept — zero re-embedding |
| text edit onto unseen text | dropped → the doc reappears in `next_pending_embeddings` |
| text edit onto text another document already embedded | re-linked (dedup hit), no embedding request |
| `delete(id)` | dropped; the embedding stays until a rebuild GCs unreferenced hashes |

**Pending is derived:** `next_pending_embeddings(limit)` = `live ∧ ¬has_vector`,
ordered by id. There is no durable queue that could drift. `attach_vectors` is
compare-and-set on `(id, content_hash, model_version)`: the **whole batch** is
validated (dimension, finiteness, model, liveness, hash equality) before
anything is written, so one stale completion rejects the batch with
`StaleVectorCas` instead of half-applying it. Attach commits its own link
generation atomically.

**Fusion (Level 1).** `V2Index::hybrid_search(&V2HybridQuery)` runs the lexical
lane with `top_k = candidate_k` and **no** per-lane diversity cap, and the dense
lane over the `vec_map` with `crate::distance` (SIMD). The dense lane honours the
same authority as the lexical one: with any columnar filter active it collects
candidates through a filter-only posting walk (`DocSetCollector`) and intersects
with the ACL overlay; otherwise it checks `acl_overlay.is_visible` while scanning
the link map. Similarity is higher-is-better and consistent with
`src/search.rs::compute_score` (cosine is already a similarity; Euclidean is
`1/(1+d)`). `rrf = lw/(rrf_k + lex_rank) + vw/(rrf_k + vec_rank)` with 1-based
ranks, `rrf_k = 60`, `candidate_k = 200`, `lexical_weight = 1.0`,
`vector_weight = 0.3` by default (`DEFAULT_RRF_K`, `DEFAULT_CANDIDATE_K` in
`query.rs`). All are per-request wire fields, so a client may override them
(`mcp_server` does: adaptive query-shape weights with `rrfK: 20`).
`group_by_parent` is applied **once** on the fused order, then `top_k` winners are
hydrated — vector-only winners included. `dense` defaults to
`query_vector.is_some()` in the wire form.

**Lexical spike preservation.** When `vector_weight > lexical_weight` and both
lanes run, the engine computes a BM25 baseline: the score of the
`LEXICAL_SPIKE_BASELINE_PARENTS` (5) distinct parent in the lexical lane. A
candidate scoring `>= 2.5 ×` that baseline (`LEXICAL_SPIKE_RATIO`) is an exact
outlier — phrase, title-exact or all-terms hit — and gets effective lexical
weight `max(lexical_weight, vector_weight × 1.1)`. Flat topical queries sit at
1.1–1.4× the baseline; verbatim hits land at 6–45×. This keeps a correct exact
match on top without raising lexical weight for every query (herrbasan/nVDB#5).

**Refinement (Level 2).** `refine_level2(query_text, principals, hits, source,
reranker, options)` takes the first `shortlist` (20) survivors. With a
`SourceConnector` every survivor is re-read and its **authoritative** ACL
re-checked; a document revoked or deleted since the last sync is dropped
silently — announcing it would disclose that it exists — and its text never
reaches the reranker. Reranker text is `envelope.indexable_text` when a source is
present, otherwise `excerpt` or `title + " " + path`. A `Reranker` (the trait
requires `Send + Sync` because the HTTP server shares one behind an `Arc`)
result is verified (one finite score per document, else `V2Error::Rerank`) and
fused with
the Level-1 order by weighted RRF (`1.0` vs `rerank_weight`, default `0.2`).
`best_rerank_score < relevance_floor` (default `0.02`) sets `rejected = true`
while keeping the hits as low-relevance evidence.

**No ANN lane yet.** `flat_search_cutoff` records the intended policy; v2 always
scans the content-hash map flat, which the field's documentation states instead
of implying an index that does not exist.

---

## API Quick Reference

### v2 (production surface)

```rust
use nvdb::v2::{
    V2CollectionConfig, V2Document, V2HybridQuery, V2Index, V2Query, VectorLaneConfig,
};

let config = V2CollectionConfig::with_vector_lane(VectorLaneConfig::new(4, "demo-4d"));
let mut index = V2Index::create_or_open("data/docs", config)?;
let mut doc = V2Document::new("doc-1", "doc");
doc.title = "Förderprogramm Digitalisierung".into();
doc.path = "programs/foerderprogramm-digitalisierung.md".into();
doc.body = "Die Richtlinie beschreibt das Förderprogramm.".into();
doc.acl = vec!["public".into()];
index.upsert(doc)?;
index.commit()?;

// Lexical / filter-only. Principals are mandatory.
let hits = index.search(&V2Query::new("foerder").principals(["public"]).top_k(5))?;

// Hybrid (needs the vector_lane above).
let hits = index.hybrid_search(
    &V2HybridQuery::new(V2Query::new("foerderprogramm").principals(["public"]))
        .vector(vec![0.10, 0.20, 0.30, 0.40])
        .candidate_k(200),
)?;

let pending = index.next_pending_embeddings(100);  // live ∧ ¬has_vector
// index.attach_vectors(...) — CAS on (id, content_hash, model_version)
```

The HTTP service is the same surface with JSON bodies; see the route table above.

### v1 (legacy crate surface)

Still compiled and tested; used by the CLI administration commands, not by production consumers.

```rust
let db = Database::open(path)?;                  // Open or create
let coll = db.create_collection(name, config)?;  // New collection
let coll = db.get_collection(name)?;             // Existing collection
let names = db.list_collections();               // List collections
```

### Document Operations

```rust
coll.insert(document)?;                      // Add or replace by ID
coll.insert_batch(documents)?;               // Bulk insert
coll.get(id)?;                               // Retrieve by ID
coll.delete(id)?;                            // Soft delete
```

### Search

```rust
use nVDB::{Search, Distance, Filter};

// Exact search
let results = coll.search(
    Search::new(&query).top_k(10).distance(Distance::Cosine)
)?;

// Approximate search with HNSW
let results = coll.search(
    Search::new(&query).top_k(10).approximate(true).ef(100)
)?;

// With filter
let results = coll.search(
    Search::new(&query)
        .top_k(10)
        .filter(Filter::and([
            Filter::eq("category", "books"),
            Filter::gt("year", 2020),
        ]))
)?;
```

### Maintenance

```rust
coll.flush()?;                               // Freeze memtable → segment
coll.compact()?;                             // Remove deleted docs, rebuild index
coll.sync()?;                                // Ensure WAL durability
coll.rebuild_index()?;                       // Build HNSW index
coll.delete_index()?;                        // Remove HNSW index
```

---

## Documentation Maintenance

**When modifying code, you MUST update corresponding documentation:**

1. `documentation/v2-server.md` — the current reference for the v2 engine, collection layout, retrieval pipeline and HTTP API. Update when routes, wire types, defaults, fusion weights, reranking behaviour or on-disk artifacts change.

2. `README.md` — human-facing overview. Keep it accurate about what nVDB is today; it must never claim Node.js bindings or features that do not exist.

3. `AGENTS.md` (this file) — Update when:
   - Changing architectural decisions
   - Modifying file formats or protocols
   - Adding new constraints or non-goals
   - Changing build/test commands

4. `docs/test-documentation.md` — Update when adding or removing test suites (note: much of it describes v1 phases).

Historical working notes live in `docs/archive/`; leave them unless they are actively misleading.

**Documentation that does not match code is a bug.**

---

## Non-Goals

- **Embedding generation** (see `docs/scope-and-boundaries.md`)
- Distributed operation (use application-layer sharding if needed)
- ACID transactions across multiple documents
- Complex query language (no joins, aggregations)
- ANN/approximate search on the v2 dense lane (flat scan only; v1 has HNSW)
- Schema migrations within a collection (fixed dimension at creation)
- Background/async compaction (synchronous only)
- Multi-writer per collection root (per-collection file locks, no coordination)
- Node.js / NAPI bindings (removed 2026-11-22; use the `nvdb serve` HTTP API)

---

*See `documentation/v2-server.md` for the current engine and HTTP reference.*
*See `docs/test-documentation.md` for test suite details (largely v1-era).*
*See `docs/user-guide.md` for v1 API usage examples.*
