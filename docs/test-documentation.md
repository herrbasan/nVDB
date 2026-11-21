# nVDB Test Documentation

> **Last Updated:** 2026-11-21  
> **Phase:** 3 — Unified hybrid search, `content_hash`-keyed vectors and Level-2 refinement (`src/v2/`)  
> **Commit:** Phase 3 (`src/v2/vectors.rs`, `src/v2/level2.rs`, hybrid search) + `tests/phase3_v2_hybrid_tests.rs`

## Overview

This document describes the test suite for nVDB. Tests are organized by component and verify both functional correctness and architectural invariants.

---

## Test Organization

```
tests/
├── phase0_sync_contract_tests.rs # Sync/snapshot/ACL/revision contract tests
├── phase2_v2_engine_tests.rs    # v2 lexical engine (Tantivy) integration tests
├── phase3_v2_hybrid_tests.rs    # v2 hybrid search, vectors, Level-2 refinement
├── phase1a_segment_tests.rs    # Segment file format tests
├── phase1a_locking_tests.rs    # Multi-process locking tests
├── phase2_search_tests.rs      # Exact similarity search tests
├── phase3_hnsw_tests.rs        # HNSW approximate search tests
├── phase4_filter_tests.rs      # Filter DSL integration tests
├── phase5_compaction_tests.rs  # Compaction integration tests
├── lexical_tests.rs            # Lexical index tests
├── lexical_collection_tests.rs # Collection-level lexical tests
├── hybrid_search_tests.rs      # Hybrid (lexical + vector) retrieval tests
└── (unit tests in src/*.rs)    # Module-level unit tests
```

---

## Running Tests

```bash
# All tests
cargo test  # lib + every integration suite

# Specific test suite
cargo test --test phase0_sync_contract_tests
cargo test --test phase2_v2_engine_tests
cargo test --test phase2_v2_server_tests
cargo test --test phase3_v2_hybrid_tests
cargo test --test phase1a_segment_tests
cargo test --test phase1a_locking_tests
cargo test --test phase2_search_tests
cargo test --test phase3_hnsw_tests
cargo test --test phase4_filter_tests
cargo test --test phase5_compaction_tests

# With output
cargo test -- --nocapture

# Release mode (for performance validation)
cargo test --release
```

---

## Phase 0: Sync, Snapshot, ACL & Revision Contract Tests

### Contract Tests (`tests/phase0_sync_contract_tests.rs`)

Phase 0 has its own exit gate: the sync/crash/revocation contract must be green
before any index-engine code is written on top of it. The suite drives
`nvdb::sync::SyncCoordinator` through a journal-backed `SourceConnector` fake.

| Test | Purpose | Success Criteria |
|------|---------|------------------|
| `apply_failure_degrades_index_and_heals_after_restart` | Dual-write apply failure | Injection at seq N leaves `synced_seq = N-1`, `degraded = true` (persisted); queries fail `IndexDegraded`; later sync rebuilds and serves |
| `backup_restore_behind_watermark_rebuilds_and_drops_phantoms` | Restored backup (`synced_seq > last_seq`) | Refuses incremental catch-up, rebuilds from snapshot, post-backup docs disappear |
| `new_source_epoch_rebuilds_even_with_higher_head` | Epoch divergence | A new `source_epoch_uuid` with a higher head still forces a clean rebuild |
| `compacted_cursor_rebuilds_but_caught_up_index_does_not` | Compaction cursor expiry | `CursorExpired` triggers a rebuild; an already-caught-up index returns `UpToDate` without taking a snapshot |
| `acl_revocation_fails_closed_before_and_after_sync` | Fail-closed ACL at every layer | Empty principals → `MissingPrincipals`; behind-source index → `BootHealIncomplete` (revoked doc never served); Level-2 hydration re-checks source ACL → `AclDenied`; after sync the doc is invisible |
| `envelope_hash_gates_updates_content_hash_gates_vectors` | Hash separation + vector CAS | ACL/metadata-only updates change `envelope_hash` but reuse the vector; text edits invalidate it; no-op replays are skipped; stale attach (old content hash or deleted doc) → `StaleVectorCas`; a stale batch attaches nothing |
| `rebuild_catches_up_mutations_landing_after_the_snapshot` | Concurrent rebuild | Insert/delete/update/ACL mutations after `snapshot_at()` are replayed via `changes_since`, ending in exact authoritative state |

**Key Invariants Verified:**
- Watermark commits are contiguous; failures degrade instead of diverging
- Two hashes stay separated: ACL/metadata updates are never skipped, vectors are never re-paid for
- Reads fail closed on degraded state, empty principals, and any catch-up lag
- Level-2 hydration trusts only the authoritative source for access decisions

---

## Phase 2 (General Indexer): Lexical Retrieval Engine (`tests/phase2_v2_engine_tests.rs`)

Phase 2 embeds Tantivy (MIT, pure Rust) behind `nvdb::v2`. The suite runs against
a real `V2Index` on disk, with a journal-backed `SourceConnector` fake that
carries real `V2Document`s (so hashes and ACLs are the same ones the engine
computes).

| Test | Purpose | Success Criteria |
|------|---------|------------------|
| `german_compounds_match_in_both_directions` | Frozen lexicon + DP splitter wired into `body_parts` | `Förderprogramm`, `Datenschutzrichtlinie`, `Forschungszulage`, `Anwendungshandbuch` decompound as expected; compound query reaches the component document **and** component query reaches the compound document; the full compound title still ranks first |
| `compound_ranking_is_deterministic_across_rebuild_and_incremental` | Write-path independence | Six compound queries return identical id order whether built by `upsert` or by a Phase 0 snapshot rebuild |
| `one_edit_typos_are_tolerated_without_breaking_exact_title_ranking` | `FuzzyTermQuery` distance 1, transposition cost 1 | `Conciousness` (no fuzzy) matches only the misspelled doc; with `fuzzy_edits(1)` it also finds the correct one; the correctly spelled query keeps the exact-title doc at rank 1 |
| `exact_title_and_path_identifiers_both_match` | `title_exact` + `nvdb_identifier` | `Quarterly Report` finds the exact-title doc and the `docs/q1/quarterly-report.pdf` doc, never the unrelated one; the exact title ranks first; `ext("pdf")` isolates the path match |
| `columnar_filters_narrow_the_candidate_set` | `Occur::Must` columnar filters | `ext(".md")` == `ext("md")`; `kind`, `folder`, `created_at`/`updated_at` ranges, and promoted `I64`/`String`/`Bool`/`F64` columns all narrow correctly; unknown column → `InvalidQuery`; wrong type → `PromotedFieldType`; range on a string → `InvalidQuery` |
| `acl_overlay_is_authoritative_and_updates_without_reindexing` | Authoritative ACL + ACL-only fast path | Empty principals → `MissingPrincipals`; `update_acl_only` is visible immediately with **no** reader growth (no re-index); revocation hides instantly; empty ACL hides from everyone; generation and overlay survive reopen |
| `group_by_parent_caps_hits_per_parent_and_explains_them` | Native diversity cap + `explain` | At most `max_per_parent` hits per `parent_id`, unparented docs never capped, `explain` reports coverage/BM25/final score and the diversity counter; `max_per_parent = 0` is `InvalidQuery` |
| `manifest_version_mismatch_fails_fast_on_reopen` | Open-time contract | Bumped `analyzer_version`/`schema_version`/`lexicon_version` → `VersionMismatch` with the offending field; changed `promoted_fields` → `SchemaMismatch`; the exact config reopens and still serves committed docs |
| `sync_applies_incremental_batches_and_rebuilds_after_compaction` | Full Phase 0 sync on the real index | First sync rebuilds; incremental applies upsert + delete + ACL-only; replaying an identical record is skipped by envelope hash; `UpToDate` when caught up; journal compaction → `CursorExpired` rebuild; degraded index refuses reads and heals on rebuild; watermark/overlay/documents survive reopen; `integrity_check` clean at every step |
| `sync_rebuilds_after_a_restored_backup` | `synced_seq > last_seq` divergence | Restoring a backup forces a snapshot rebuild; documents added after the backup disappear |
| `source_lineage_is_reported_from_the_watermark` | Watermark accessor | `None` before the first sync, epoch + sequence after |

**Key Invariants Verified:**
- ACL decisions never come from a Tantivy segment; an ACL-only change costs no re-index
- Reads fail closed on empty principals and on a degraded watermark
- A version bump can never serve differently-analysed results silently
- German compound recall is symmetric and independent of the write path
- Sync never diverges: epoch change, restored backup, expired cursor and feed gaps all rebuild

**Storage contract exercised by the suite:** only the hit envelope is stored (`id`,
`parent_id`, `chunk_index`, `kind`, `title`, `path`, `name`, `folder`, `ext`,
`mime`, `status`, timestamps, non-empty metadata, opt-in `excerpt`).
`body`, `headings`, `body_parts`, `categories` and `tags` are indexed but not
stored, and `id`/`parent_id` are fast fields — Phase B enforces ACL and the
per-parent cap without reading a stored document. The unit tests in
`src/v2/analyzer.rs` pin the canonical single-token analyzer
(`text_tokens_emit_one_canonical_token_per_word`), and the ACL overlay unit
tests pin the interned group-set encoding round trip.

---

## Phase 2 (General Indexer): HTTP Surface (`tests/phase2_v2_server_tests.rs`)

`nvdb serve` (`src/v2/server.rs`) is the resident localhost front end of the v2
engine — hand-rolled HTTP/1.1 on `std::net` (no new crate), one thread per
connection behind a cap, JSON in and out. The suite binds the server to
`127.0.0.1:0`, talks to it over a real TCP socket, and asserts on the raw status
line and JSON body, so the route table, the error codes and the fail-closed ACL
path are tested exactly as a client sees them.

| Test | Purpose | Success Criteria |
|------|---------|------------------|
| `health_reports_engine_and_rejects_path_traversal` | `/health`, collection create, name validation | `{"ok":true,"engine":"nvdb-v2","collections":[]}` before any create; `POST /v2/collections {"name":"docs"}` opens the collection and writes `v2_manifest.json`; `../evil` is `400 invalid_collection_name` and creates nothing outside the root |
| `foreign_directory_is_refused_instead_of_hijacked` | Startup-adjacent safety | A pre-existing non-empty directory that is not a v2 collection answers `403 foreign_directory` and its files are untouched |
| `upsert_commit_search_acl_and_delete_round_trip` | Whole write/read surface | `commit:false` upserts are invisible until `POST .../commit` reports `doc_count`; a query without principals is `400 missing_principals`; a valid query returns hits with a numeric `took_ms`; `POST .../acl` bumps `acl_generation` and the revoked group sees zero hits on the very next search while the new group sees the doc; delete reports `deleted:1`, empties the search, and `/v2/collections` reports the remaining `doc_count`; unknown collection and unknown route are `404` with their own `code` |
| `restart_rediscovers_the_collection_and_its_documents` | Persistence across processes | After a shutdown, a second server on the same root lists the collection in `/health` and `/v2/collections` and finds the committed document by search — proving the index writer was released |
| `malformed_bodies_are_rejected_with_400` | Fail-loud input handling | An absent body is `400 empty_body`; a misspelled key (`nam`, `principal`) is `400 invalid_body` instead of being silently ignored |
| `degraded_index_refuses_search_with_503` | Degraded boot-heal watermark | A collection marked degraded at open lists its `watermark.degraded = true` and search answers `503 index_degraded` instead of a partial result |

**Key Invariants Verified:**
- A search without principals can never widen to "match everything" over HTTP either
- The ACL-only route is the fast path: visibility flips with no re-index, and the new generation is reported
- A collection name can never escape the root directory, and a foreign directory is never taken over
- Shutdown releases the index, so restarting on the same directory works

**Unit tests (`src/v2/server.rs`):**

| Test | Purpose |
|------|---------|
| `collection_names_reject_traversal_and_separators` | `../evil`, `a/b`, `a\b`, `.`, `..`, empty, spaces and non-ASCII are rejected; `docs`, `docs-v2`, `a.b_c-1`, `0` are accepted |
| `engine_errors_map_to_explicit_statuses` | `MissingPrincipals` → `400`, `IndexDegraded` → `503`, `SchemaMismatch` → `409` |

---

## Phase 3 (General Indexer): Hybrid Search, Vectors & Level-2 (`tests/phase3_v2_hybrid_tests.rs`)

Phase 3 adds the dense lane (`src/v2/vectors.rs`), scale-free fusion
(`V2Index::hybrid_search`) and authoritative refinement (`src/v2/level2.rs`).
The suite uses 4-dimensional embeddings so the arithmetic is inspectable, and a
`MutableSource` whose ACLs change *after* the index synced — the only way to
prove Level 2 does not trust derived state.

| Test | Purpose | Success Criteria |
|------|---------|------------------|
| `content_hash_vectors_dedup_reuse_and_survive_reopen` | Content-keyed embeddings | A live document without a vector is pending (hex `content_hash`, stored `title`); attach clears it and writes generation 1; a second document with identical text is auto-linked and shares **one** `vectors.bin` entry; ACL-only and metadata-only upserts keep the link and do **not** rewrite the map; a text edit drops the link and re-lists the document as pending; after reopen both links and both embeddings survive and nothing is pending; deleting one twin keeps the shared embedding, and `gc_vectors()` drops exactly the now-unreferenced one |
| `pending_embeddings_are_derived_and_attach_is_atomic` | Derived pending list + CAS attach | Pending order is by id and `limit` is respected (`limit = 0` → empty); a malformed hash is `InvalidQuery`; a completion carrying another document's hash is `StaleVectorCas` and **nothing** in the batch lands (pending count and generation stay at their pre-batch values, store still empty); wrong dimension → `WrongDimension`; wrong model and NaN → `InvalidQuery`; a deleted document → `StaleVectorCas`; a fully valid batch attaches 2 and empties the pending list |
| `vector_and_hybrid_lanes_never_leak_past_acl_or_filters` | Dense-lane authority | Three documents with identical vectors but different ACL/folder/ext: a vector-only search for `public` never returns the `hr-secret` document (0 leaks) and its hits carry no lexical evidence; `hr-secret` sees only its own document; `folder("hr")` intersects the visible set; `ext("pdf")` yields nothing because the only pdf is invisible; the fused path and a filter-only scan return the same single visible document |
| `hybrid_rrf_lifts_paraphrases_and_caps_parents_once` | RRF fusion + single diversity cap | The lexical lane alone ranks the exact-title document first and never returns the term-free paraphrase; hybrid returns both with the exact match still rank 1 (lexical rank 1, RRF numerator evidence present); the paraphrase carries `vector_rank = 1`, no lexical rank and `hit.score == rrf_score`; a zero-similarity document never outranks it; `weights(0, 1)` puts the paraphrase first; `group_by_parent(1)` over three chunks of one parent plus a second parent returns one hit for that parent and keeps the other parent, applied once on the fused list; empty principals → `MissingPrincipals`, both lanes off → `InvalidQuery`, `dense` without a vector → `InvalidQuery`, wrong dimension → `WrongDimension` |
| `level2_rechecks_source_acl_reranks_and_respects_the_floor` | Authoritative re-check, reranking, floor | A document revoked in the source after the last sync is dropped although Level 1 returned it — its text appears neither in `hydrated_text` nor in the reranker's recorded input, and only the survivors are scored (1 document, not 2); `best_rerank_score` is the maximum; a reranker that prefers the lower-ranked survivor reorders it at `rerank_weight = 5.0`; a nonsense query with floor `0.02` sets `rejected = true` while keeping the hits, a real match does not; `shortlist = 1` keeps one hit; no reranker → `best_rerank_score = None` and `rejected = false`; empty principals → `MissingPrincipals`; a reranker returning the wrong number of scores → `V2Error::Rerank` |
| `http_round_trip_pending_attach_and_hybrid_search` | `nvdb serve` end to end | `POST /v2/collections` with a vector lane, `upsert`, then `GET .../pending_embeddings?limit=10` reports both documents with hex hashes and titles; `POST .../attach_vectors` reports `attached: 2` and a bumped `vec_generation`; the pending list empties; a stale attach is `409 stale_vector_cas`; `POST .../hybrid_search` returns the exact match first and the paraphrase with `vector_rank = 1` and a numeric `took_ms`; a bodyless query is `400 missing_principals`; `limit=0` is `400 invalid_limit`; a lane-less collection reports `count: 0` and answers dense search with `400 vector_lane_not_configured` |

**Key Invariants Verified:**
- An embedding is keyed by content + model, so identical text costs one embedding
  and ACL/metadata-only writes cost none
- Pending embeddings are derived (`live ∧ ¬has_vector`), never a second queue
- A vector batch is all-or-nothing: one stale completion attaches nothing
- The dense lane is exactly as authoritative as the lexical lane (ACL overlay +
  the same columnar filters), so a hybrid query cannot widen visibility
- Diversity is applied once, after fusion — no lane can hide the other's winners
- Level 2 never serves, and never reranks, text the source no longer authorises
- The relevance floor turns a nonsense query into `rejected = true` instead of noise

**Unit tests (`src/v2/vectors.rs`):**

| Test | Purpose |
|------|---------|
| `store_round_trips_and_deduplicates_by_content_and_model` | Same key overwrites (dedup); a new model version is a new entry; the binary form round trips |
| `store_rejects_truncated_and_non_finite_input` | Truncated bytes and a stored NaN fail at parse time, not at query time |
| `gc_drops_every_unreferenced_embedding` | `retain_referenced` drops exactly the unreferenced keys |
| `vec_map_bumps_generation_only_on_real_change` | A repeated link is a no-op; a change bumps once; generations round trip |
| `vec_map_missing_generation_zero_is_empty_but_later_generations_are_corrupt` | Generation 0 without a file is normal, a missing generation above 0 is corruption |
| `hex_digest_round_trips` | Hex form is 64 chars, case-insensitive on input, rejects malformed input |

---

## Phase 2: Search Tests

### Search Tests (`tests/phase2_search_tests.rs`)

| Test | Purpose | Success Criteria |
|------|---------|------------------|
| `test_search_basic_cosine` | Cosine similarity search | Query [1,0,0] returns "a" (exact match) first, then "d" (45°) |
| `test_search_top_k_limit` | Top-k limiting | Different k values return correct number of results |
| `test_search_different_distances` | All distance metrics | Dot, Cosine, Euclidean all return expected ordering |
| `test_search_over_segments` | Cross-storage search | Documents found in both memtable and flushed segments |
| `test_search_dimension_mismatch` | Input validation | Wrong dimension returns Error::WrongDimension |
| `test_search_deterministic_tie_breaking` | Stable ordering | Identical scores tie-break by internal ID |
| `test_search_empty_collection` | Edge case | Empty collection returns empty results |
| `test_search_deleted_documents_excluded` | Soft delete | Deleted documents not in search results |
| `test_search_large_dimension` | Real-world dims | 768-dimension vectors work correctly |
| `test_search_persistence` | Recovery | Search works after close/reopen |
| `test_search_result_with_payload` | Payload handling | Documents found (payload retrieval documented) |

**Key Invariants Verified:**
- 100% recall (exact search)
- SIMD vs scalar parity within FP tolerance
- Deterministic tie-breaking: score descending, then internal_id ascending
- Search spans memtable + all segments

---

## Phase 3: HNSW Index Tests

### HNSW Tests (`tests/phase3_hnsw_tests.rs`)

| Test | Purpose | Success Criteria |
|------|---------|------------------|
| `test_approximate_search_api` | New builder methods work | `.approximate(true).ef(64)` returns results |
| `test_exact_fallback_when_no_index` | Graceful fallback | Falls back to exact search when no HNSW index |
| `test_hnsw_persistence` | Index save/load | Index persists across close/reopen |
| `test_delete_index` | Index removal | `delete_index()` removes index, search still works |
| `test_recall_vs_exact` | Approximate vs exact | Some overlap between exact and approximate results |
| `test_ef_parameter_effect` | EF parameter | Different ef values produce results |
| `test_search_builder_methods` | API verification | `is_approximate()`, `ef_value()` work correctly |

**Key Invariants Verified:**
- HNSW index builds without errors
- Approximate search returns results
- Fallback to exact search when index missing
- Index persists across database close/reopen
- `ef` parameter affects search scope

### HNSW Unit Tests (`src/hnsw.rs`)

| Test | Purpose |
|------|---------|
| `test_hnsw_params_default` | Default parameters (M=16, ef=32) |
| `test_hnsw_params_with_m` | Custom M parameter |
| `test_hnsw_build_and_search` | Index builds and returns results |
| `test_hnsw_search_dimension_mismatch` | Wrong dimension returns error |
| `test_hnsw_empty_search` | Empty index returns error |
| `test_hnsw_recall` | Recall > 30% on synthetic data |
| `test_candidate_ordering` | Max-heap ordering correct |
| `test_csr_layout` | CSR format structure valid |
| `test_hnsw_graph_has_edges` | Graph construction creates edges |

**Implementation Notes:**
- CSR (Compressed Sparse Row) layout for cache efficiency
- Multi-layer graph with probabilistic level assignment
- Greedy search with dynamic candidate list (ef parameter)
- Basic implementation prioritizes correctness over optimal recall

---

## Phase 4: Filter DSL Tests

### Filter Tests (`tests/phase4_filter_tests.rs`)

| Test | Purpose | Success Criteria |
|------|---------|------------------|
| `test_filter_with_exact_search` | Basic filter integration | Filter applied to exact search results |
| `test_filter_comparison_operators` | Gt, Gte, Lt, Lte | All comparison operators work correctly |
| `test_filter_and_or_operators` | Logical operators | And/Or combine filters correctly |
| `test_filter_in_operator` | Array membership | `In` filter matches values in array |
| `test_filter_nested_fields` | Dot notation | `"user.name"` accesses nested JSON |
| `test_filter_selective_returns_fewer_results` | Post-filtering behavior | Filtered search may return < top_k results |
| `test_filter_no_matches` | Empty results | Filter matching nothing returns empty vec |
| `test_filter_over_segments` | Cross-storage filtering | Filter works across memtable + segments |
| `test_filter_with_approximate_search` | HNSW + filtering | Filter works with approximate search |
| `test_filter_complex_nested` | Complex queries | Nested And/Or with dot notation |
| `test_filter_numeric_type_coercion` | Int/float comparison | `5` equals `5.0` in filters |
| `test_filter_documents_without_payload` | Missing payload handling | Documents without payload excluded when filter present |
| `test_filter_combined_with_distance_metric` | Filter + distance | Filter works with all distance metrics |

**Key Invariants Verified:**
- Post-filtering: Filter applied after vector search
- Missing field = document excluded (filter fails)
- Numeric coercion: integers and floats comparable
- MongoDB-compatible dot notation for nested access
- Works with both exact and approximate search

### Filter Unit Tests (`src/filter.rs`)

| Test | Purpose |
|------|---------|
| `test_filter_eq_string` | Equality with strings |
| `test_filter_eq_number` | Equality with numbers (int/float coercion) |
| `test_filter_eq_bool` | Equality with booleans |
| `test_filter_eq_missing_field` | Missing field returns false |
| `test_filter_gt` | Greater than comparison |
| `test_filter_gte` | Greater than or equal |
| `test_filter_lt` | Less than comparison |
| `test_filter_lte` | Less than or equal |
| `test_filter_numeric_coercion` | Int vs float comparison |
| `test_filter_in` | Array membership |
| `test_filter_in_numeric` | In with numeric coercion |
| `test_filter_and` | Logical AND |
| `test_filter_or` | Logical OR |
| `test_filter_nested_and_or` | Complex boolean logic |
| `test_filter_nested_field` | Dot notation access |
| `test_filter_deeply_nested` | Multi-level nesting |
| `test_filter_nested_missing` | Missing nested field |
| `test_filter_empty_and` | Empty And returns true |
| `test_filter_empty_or` | Empty Or returns false |
| `test_get_field_simple` | Direct field access |
| `test_get_field_nested` | Dot notation traversal |

**Filter DSL API:**
```rust
// Basic predicates
Filter::eq("field", value)
Filter::gt("field", value)
Filter::gte("field", value)
Filter::lt("field", value)
Filter::lte("field", value)
Filter::in_("field", [v1, v2, v3])

// Logical operators
Filter::and([filter1, filter2])
Filter::or([filter1, filter2])

// Nested fields
Filter::eq("user.name", "alice")
```

---

### Distance Unit Tests (`src/distance.rs`)

| Test | Purpose |
|------|---------|
| `test_dot_product_basic` | Correctness: 4D dot product |
| `test_dot_product_aligned` | SIMD alignment: 8D (one chunk) |
| `test_dot_product_with_remainder` | Remainder handling: 10D (1 chunk + 2) |
| `test_cosine_same_vector` | Self-similarity = 1.0 |
| `test_cosine_orthogonal` | Orthogonal vectors = 0.0 |
| `test_cosine_opposite` | Opposite vectors = -1.0 |
| `test_cosine_zero_vector` | Zero vector handling |
| `test_euclidean_basic` | L2 distance correctness |
| `test_euclidean_same_vector` | Self-distance = 0.0 |
| `test_euclidean_aligned` | SIMD alignment for L2 |
| `test_distance_enum` | Distance::compute() API |
| `test_distance_dimension_mismatch` | Error handling |
| `test_higher_is_better` | Distance metric semantics |
| `test_large_dimension` | 384, 768, 1536 dims with SIMD vs scalar |

**Scalar Fallback:**
All SIMD functions have scalar implementations in `distance::scalar` module for verification.

---

## Phase 1B: WAL, Memtable & Recovery Tests

### Unit Tests

#### `src/wal.rs` — Write-Ahead Log

| Test | Purpose |
|------|---------|
| `test_wal_append_and_replay` | Basic append/replay cycle |
| `test_wal_idempotent_replay` | Same record skipped on replay |
| `test_wal_corruption_truncation` | Corrupt tail truncated gracefully |
| `test_wal_reset` | WAL clear/reset functionality |
| `test_delete_record` | Delete operation serialization |

#### `src/manifest.rs` — Collection Manifest

| Test | Purpose |
|------|---------|
| `test_manifest_roundtrip` | Save/load preserves data |
| `test_manifest_atomic_update` | Write-temp-rename atomicity |
| `test_manifest_manager` | Manager wrapper functionality |
| `test_remove_segments` | Segment removal from manifest |
| `test_manifest_not_found` | Missing file handling |

#### `src/memtable.rs` — In-Memory Storage

| Test | Purpose |
|------|---------|
| `test_memtable_insert_and_get` | Basic insert/retrieve |
| `test_memtable_dimension_mismatch` | Validation |
| `test_memtable_delete` | Soft delete |
| `test_memtable_iter` | SoA iterator skips deleted |
| `test_memtable_replace` | Update existing document |
| `test_memtable_soa_layout` | Contiguous vector storage |
| `test_frozen_memtable` | Flush preparation |
| `test_active_count` | Deleted vs total count |

#### `src/lib.rs` — Collection API

| Test | Purpose |
|------|---------|
| `test_collection_insert_and_get` | Basic document operations |
| `test_collection_insert_batch` | Batch insert efficiency |
| `test_collection_delete` | Soft delete via API |
| `test_collection_flush` | Memtable → segment |
| `test_collection_persistence` | Close/reopen data integrity |

---

## Phase 1A: Storage Primitives Tests

### Segment Tests (`tests/phase1a_segment_tests.rs`)

| Test | Purpose | Success Criteria |
|------|---------|------------------|
| `test_segment_create_and_reopen` | End-to-end segment lifecycle | Create → write 100 docs → reopen → all data intact |
| `test_segment_header_format` | Binary format verification | Magic bytes, version, dimension, doc_count at correct offsets |
| `test_segment_corruption_detection` | Checksum validation | Corrupted file fails checksum verification |
| `test_segment_with_empty_payloads` | Optional payload handling | Documents without payloads deserialize correctly |
| `test_segment_iterators` | Iterator correctness | Vector iterator and full iterator return expected data |
| `test_large_dimension_vectors` | Common embedding dimensions | 384, 768, 1536 dimensions all work correctly |
| `test_segment_sharing_across_threads` | Thread safety | 4 threads concurrently read from shared Arc<Segment> |
| `test_empty_segment_rejected` | Validation | Building empty segment returns error |
| `test_dimension_mismatch_detected` | Input validation | Wrong vector dimension detected at insert time |

**Key Invariants Verified:**
- 64-byte header alignment
- Little-endian byte order
- BLAKE3 checksum computation
- Zero-copy vector access via raw pointers
- Thread-safe read-only access

### Locking Tests (`tests/phase1a_locking_tests.rs`)

| Test | Purpose | Success Criteria |
|------|---------|------------------|
| `test_lock_basic_acquire_and_release` | Lock lifecycle | Lock acquired, `is_locked()` returns true, released on drop |
| `test_lock_exclusivity` | Single-writer guarantee | Second lock attempt fails with `CollectionLocked` |
| `test_lock_multiple_collections` | Isolation | Different collections can be locked simultaneously |
| `test_lock_thread_safety` | Cross-thread behavior | Lock status checkable from other threads |
| `test_lock_file_persistence` | Lock file handling | LOCK file created and persists after unlock |
| `test_lock_drop_releases` | RAII correctness | 10 consecutive acquire/drop cycles all succeed |
| `test_lock_recovery_after_abandoned_lock` | Crash recovery | Can acquire lock even if LOCK file exists (process died) |

**Platform-Specific Behavior:**
- **Unix**: `flock(2)` with `LOCK_EX | LOCK_NB`
- **Windows**: `LockFileEx` with `EXCLUSIVE | FAIL_IMMEDIATELY`

**Known Limitations:**
- Same-process lock exclusivity on Windows may vary by Windows version (LockFileEx behavior)
- Cross-process locking is the primary guarantee

### Unit Tests (in `src/` modules)

#### `src/id.rs` — ID Mapping

| Test | Purpose |
|------|---------|
| `test_basic_mapping` | String ↔ u32 bidirectional mapping |
| `test_duplicate_insert` | Duplicate IDs return existing internal ID |
| `test_remove` | Removal clears both directions |
| `test_roundtrip_vec` | Serialization round-trip preserves all mappings |

#### `src/segment.rs` — Segment Builder

| Test | Purpose |
|------|---------|
| `test_segment_builder_and_open` | Build segment, verify all fields |
| `test_dimension_mismatch` | Wrong dimension rejected at insert |
| `test_empty_segment_fails` | Empty segment build returns error |
| `test_checksum_verification` | Corruption detected via checksum mismatch |

#### `src/lib.rs` — Database API

| Test | Purpose |
|------|---------|
| `test_database_open` | Database directory creation |
| `test_collection_config` | Config builder pattern |

---

## Test Data Characteristics

### Vector Data
- **Dimensions tested:** 4 (unit tests), 384, 768, 1536 (integration tests)
- **Document counts:** 1-100 (scalable to millions)
- **Values:** Sequential floats (0.0, 1.0, 2.0...) for deterministic verification

### Payload Data
- JSON format: `{"id": "docN", "value": N}`
- Optional payloads (None) also tested

### File Locations
All tests use `tempfile::TempDir` for isolation:
- Each test gets unique temp directory
- Automatically cleaned up on test completion
- No interference between tests

---

## Success Criteria (Phase 1A)

From `docs/development-plan.md`:

> - Create segment, write documents, close, reopen — data intact via mmap
> - Checksum verification passes; corruption detected and rejected
> - Two processes cannot open same collection as writers simultaneously

**Verification:**
- ✅ `test_segment_create_and_reopen` — Data integrity
- ✅ `test_segment_corruption_detection` — Checksum validation
- ✅ `test_lock_exclusivity` — Single-writer enforcement

---

## Success Criteria (Phase 4)

From `docs/development-plan.md`:

> - Filter correctness: 100% — Unit tests for all predicate types
> - Post-filter recall: Documented — May return <k results if filter is selective
> - Filter evaluation <50% of search time for typical predicates

**Verification:**
- ✅ 15 unit tests for Filter DSL (all predicates, nesting, type coercion)
- ✅ 13 integration tests for filtered search
- ✅ Post-filtering behavior verified: selective filters return fewer results
- ✅ Works with both exact and approximate search

## Performance Baselines

Current test execution time (debug build, Windows 11, NVMe SSD):

| Test Suite | Time |
|------------|------|
| Unit tests (59) | ~60ms |
| Locking tests (7) | ~5ms |
| Segment tests (9) | ~40ms |
| Search tests (11) | ~20ms |
| Phase 4 tests (13) | ~60ms |
| **Total** | **~185ms** |

---

## Phase 5: Compaction Tests

### Unit Tests (`src/compaction.rs`)

| Test | Purpose |
|------|---------|
| `test_merge_segments_basic` | Merge multiple segments |
| `test_merge_segments_with_deletes` | Filter out deleted docs during merge |
| `test_merge_segments_newer_wins` | Document updates: newer version takes precedence |
| `test_cleanup_temp_files` | Orphan temp file cleanup |
| `test_compact_empty_collection` | Empty collection compaction |

### Integration Tests (`tests/phase5_compaction_tests.rs`)

| Test | Purpose | Success Criteria |
|------|---------|------------------|
| `test_compaction_reduces_size` | Space reclamation | 50% deletes → ~50% size reduction |
| `test_compaction_with_multiple_segments` | Multi-segment merge | All segments merged into one |
| `test_compaction_preserves_data_integrity` | Data preservation | All non-deleted docs intact after compaction |
| `test_compaction_query_after` | Search correctness | Same results before/after compaction |
| `test_compaction_empty_collection` | Empty collection handling | No error, no change |
| `test_compaction_no_deletes` | No-op compaction | Docs preserved when no deletes exist |
| `test_compaction_all_deleted` | All docs deleted | Collection becomes empty |
| `test_compaction_rebuilds_index` | HNSW index rebuild | Index rebuilt and functional after compaction |
| `test_compaction_persistence` | Data survives reopen | Compacted data valid after close/reopen |
| `test_compaction_orphan_cleanup` | Temp file cleanup | Orphan files removed on startup |
| `test_compaction_with_document_updates` | Update handling | Newer document versions preserved |
| `test_compaction_idempotent` | Idempotent compaction | Multiple compactions produce same result |

### Integration Tests (in `src/lib.rs`)

| Test | Purpose |
|------|---------|
| `test_collection_delete` | Soft delete via API |

## Phase 6: Hardening Tests

### Property-Based Tests (`proptest`)

Added property-based tests to verify key invariants:

#### WAL Properties (`src/wal.rs`)

| Test | Property Verified |
|------|-------------------|
| `prop_wal_replay_idempotent` | Replaying WAL produces identical results |
| `prop_wal_sequence_monotonic` | Sequence numbers are strictly increasing |
| `prop_record_count_matches_appends` | Record count equals successful appends |
| `prop_delete_records_preserved` | Delete records survive WAL round-trip |

#### Filter Properties (`src/filter.rs`)

| Test | Property Verified |
|------|-------------------|
| `prop_eq_reflexive` | Eq filter matches identical values |
| `prop_eq_deterministic` | Filter evaluation is deterministic |
| `prop_gt_lt_exclusive` | Gt and Lt are mutually exclusive |
| `prop_gte_equivalent` | Gte ≡ Gt OR Eq |
| `prop_lte_equivalent` | Lte ≡ Lt OR Eq |
| `prop_and_empty_always_true` | Empty And returns true |
| `prop_or_empty_always_false` | Empty Or returns false |
| `prop_numeric_coercion_eq` | Integers equal their float equivalents |

#### Distance Properties (`src/distance.rs`)

| Test | Property Verified |
|------|-------------------|
| `prop_dot_product_simd_scalar_match` | SIMD == Scalar (dot product) |
| `prop_cosine_similarity_simd_scalar_match` | SIMD == Scalar (cosine) |
| `prop_euclidean_distance_simd_scalar_match` | SIMD == Scalar (euclidean) |
| `prop_cosine_bounded` | Cosine ∈ [-1, 1] |
| `prop_cosine_symmetric` | cos(a,b) = cos(b,a) |
| `prop_euclidean_symmetric` | dist(a,b) = dist(b,a) |
| `prop_euclidean_non_negative` | Distance ≥ 0 |
| `prop_distance_enum_consistent` | Distance::compute matches functions |

### Benchmarks (`criterion`)

Benchmark suite in `benches/`:

| Benchmark | Variables | Metrics |
|-----------|-----------|---------|
| `bench_search` | Dataset size, ef parameter, dimension | Latency p50/p95/p99 |
| `bench_insert` | Batch size, dimension | Throughput (docs/sec) |
| `bench_recovery` | WAL size, segment count | Recovery time |

Run benchmarks:
```bash
cargo bench
```

---

## Test Count Summary

| Phase | Unit Tests | Integration | Property | Benchmarks | Total |
|-------|------------|-------------|----------|------------|-------|
| Core | 103 | - | - | - | 103 |
| Phase 0 (sync contract) | 5 | 7 | - | - | 12 |
| Phase 2 (v2 lexical engine) | 19 | 11 | - | - | 30 |
| Phase 2 (v2 HTTP surface) | 2 | 6 | - | - | 8 |
| Phase 3 (v2 hybrid + vectors) | 6 | 6 | - | - | 12 |
| Phase 1A | - | 16 | - | - | 16 |
| Phase 2 | - | 11 | - | - | 11 |
| Phase 3 | - | 7 | - | - | 7 |
| Phase 4 | - | 13 | - | - | 13 |
| Phase 5 | - | 12 | - | - | 12 |
| Phase 6 | - | - | 20 | 3 suites | 20+ |
| **Total** | **111** | **71** | **20** | **3** | **205+** |

---

## Document History

| Date | Change |
|------|--------|
| 2026-02-14 | Initial test documentation for Phase 1A |
| 2026-02-14 | Updated for Phase 2: Search tests, SIMD tests, 70 total tests |
| 2026-02-14 | Updated for Phase 3: HNSW tests, 84 total tests |
| 2026-02-14 | Updated for Phase 4: Filter DSL tests, 140 total tests |
| 2026-02-14 | Updated for Phase 5: Compaction tests, 157 total tests |
| 2026-02-14 | Updated for Phase 6: Property tests, benchmarks, 185+ total tests |
| 2026-11-21 | Added Phase 0 sync/snapshot/ACL/revision contract suite (`src/sync.rs`, 7 tests + 5 unit tests) |
| 2026-11-21 | Added Phase 2 v2 lexical engine (`src/v2/`, Tantivy 0.22): 11 integration tests + 19 unit tests (lexicon, analyzers, ACL overlay, schema) |
| 2026-11-21 | Phase 2 exit-gate fixes: `body`/`headings`/`body_parts` no longer stored, `id`/`parent_id` FAST, Phase B reads no stored document before a candidate wins, `nvdb_text` emits one canonical token per word (`schema_version` 2, `analyzer_version` 2), `body_parts` weight 0.35, ACL overlay/fingerprints in compact binary |
| 2026-11-21 | Added `nvdb serve` (`src/v2/server.rs`): 6 HTTP integration tests over real TCP + 2 unit tests (name validation, error→status mapping) |
| 2026-11-21 | Added Phase 3 (`src/v2/vectors.rs`, `src/v2/level2.rs`, `V2Index::hybrid_search`): `content_hash`-keyed vectors + generation-stamped `vec_map`, derived pending list with atomic CAS attach, unified RRF hybrid search with a single diversity cap, Level-2 source-ACL re-check + reranker + relevance floor, and three new `nvdb serve` routes — 6 integration tests (`tests/phase3_v2_hybrid_tests.rs`) + 6 unit tests (`src/v2/vectors.rs`). Measured: `cargo test --lib` 150 passed, `phase3_v2_hybrid_tests` 6 passed |

---

## Notes

- **File metadata vs explicit dates:** This file includes explicit timestamps because:
  1. Git history may not be available in all contexts (e.g., source tarballs)
  2. File modification times vary by filesystem and copy operations
  3. Explicit dates make the document self-contained

- **Test determinism:** All tests use fixed seeds or sequential values where applicable. No random data without explicit seeding.

- **Platform coverage:** Currently tested on Windows 11. Unix (Linux/macOS) coverage planned for CI/CD.
