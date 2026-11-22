# nVDB v2 — Hybrid Retrieval Engine & HTTP Service

> Reference for `src/v2/` and `nvdb serve`. This is the current engine. The v1 crate (`Database`, LSM-Lite, HNSW) is documented separately in `architecture.md` and `rust-api.md`.

---

## What v2 is

A **derived** retrieval index over a document corpus. The authoritative content lives in a source store (`nDB`); nVDB keeps only the hit envelope, per-document hashes, the ACL overlay and optional embeddings. It is always rebuildable.

Two access surfaces, one implementation:

- **Rust API** — `nvdb::v2` (`V2Index`, `V2CollectionConfig`, `V2Document`, `V2Query`, `V2HybridQuery`).
- **HTTP service** — `nvdb serve`, backed by `V2Server` in `src/v2/server.rs`.

## Build & run

```bash
# reranker support is a compile-time feature
cargo build --release --features rerank-onnx

nvdb serve --dir <collection-root> [--host 127.0.0.1] [--port 3510] [--reranker-model <dir>]
```

| Flag | Default | Meaning |
|---|---|---|
| `--dir` | required | Root directory; each collection is a direct subdirectory |
| `--host` | `127.0.0.1` | Bind address (`0.0.0.0` for LAN) |
| `--port` | `3510` | Port, `1..=65535` |
| `--reranker-model` | none | Directory with `model.onnx` + `tokenizer.json` |

Startup behaviour:

- Creates `--dir` if missing.
- Adopts every direct subdirectory containing `v2_manifest.json`.
- **Aborts** if any collection fails to open — a server that hides a broken collection returns answers that look complete and are not.
- `--reranker-model` without a `rerank-onnx` build is a startup error, not a silent downgrade.
- With the feature, a model that fails to load aborts startup.

In-process: `server::start_in_background(root, addr)` returns a handle with the bound `SocketAddr`; `shutdown()` stops the accept loop, drains in-flight requests and drops every index so the Tantivy writer lock is released and a restart on the same directory works. `serve_blocking` / `serve_blocking_with_reranker` are the CLI paths.

## Collection directory

```
<root>/<collection>/
  v2_manifest.json            config, acl_generation, watermark, doc_count, vec_generation
  meta.json, *.term, *.idx, *.store     Tantivy segments (immutable)
  acl_overlay.gen_<N>.bin     authoritative ACL map for generation N
  v2_fingerprints.bin         per-doc content_hash / envelope_hash / no_acl_hash
  vectors.bin                 (content_hash, model_version) -> vector, append-only
  vec_map.gen_<N>.bin         doc_id -> content_hash for generation N
```

Binary sidecars are length-prefixed, not JSON: `open()` is a read path, and parsing megabytes of JSON on every open cost ~20 ms for nothing. An `IndexWriter` is created on the first write, never on open. Superseded `acl_overlay` / `vec_map` generations are deleted after the atomic swap.

**Open-time contract.** `schema_version`, `analyzer_version`, `lexicon_version` and `promoted_fields` must match the manifest exactly, otherwise `VersionMismatch` / `SchemaMismatch`. A changed analyzer is never tolerated silently. A changed vector dimension or model version is also a schema mismatch — stored embeddings are never reinterpreted.

## Collection config

`V2CollectionConfig` (all fields defaulted):

| Field | Default | Meaning |
|---|---|---|
| `schema_version` | 2 | Envelope schema; a bump invalidates skip decisions |
| `analyzer_version` | 2 | Analyzer contract |
| `lexicon_version` | `GERMAN_LEXICON_VERSION` (1) | German morpheme lexicon |
| `field_weights` | see below | Phase A weights |
| `promoted_fields` | `[]` | Metadata keys indexed as typed columns |
| `store_excerpt` | `false` | Store a document excerpt (≤ 1024 bytes) |
| `vector_lane` | `None` | Dense lane; `None` = lexical only |

`FieldWeights` defaults:

```
title_exact 8.0   title 4.0   path 3.0   name 3.0   headings 2.5
categories  2.0   tags  2.0   body 1.0   body_parts 0.35
```

`VectorLaneConfig`: `dim`, `distance` (default `Cosine`), `model_version`, `flat_search_cutoff` (default 50 000). The cutoff records the intended ANN policy; v2 always scans flat.

Promoted field types: `string` (lowercased whole-value match), `i64`, `f64`, `bool`. Engine-reserved names (`id`, `title`, `body`, `path`, …) cannot be promoted.

## Document envelope

`V2Document` — only `id` and `kind` are required:

`id`, `parent_id?`, `chunk_index?`, `kind`, `title`, `name`, `path`, `headings`, `categories[]`, `tags[]`, `body`, `folder?`, `ext?` (normalised lowercase, no dot), `mime?`, `status?`, `created_at?`, `updated_at?` (caller chooses seconds or ms), `acl[]` (group ids, never user ids), `metadata` (open bag; stored, not searched; promoted keys are also columns), `excerpt?`.

Stored vs indexed:

- **Stored:** `id`, `parent_id`, `chunk_index`, `kind`, `title`, `path`, `name`, `folder`, `ext`, `mime`, `status`, `created_at`, `updated_at`, non-empty `metadata`, optional `excerpt`.
- **Indexed only:** `title_exact`, `headings`, `categories`, `tags`, `body`, `body_parts`. Full content lives in the source store.

## Analyzers

- `nvdb_exact` — trim + lowercase, whole string. Feeds `title_exact`.
- `nvdb_identifier` — the whole string plus sub-tokens split on `/ \ . - _ :`, **plus** the extension-stripped stem of the last path segment when that segment ends in a 1–5 character ASCII-alphanumeric extension (`md-blocks-spec` for `md-blocks-spec.md`). Feeds `name`/`path`.
- `nvdb_text` — Unicode word segmentation + lowercase, then `ä→ae, ö→oe, ü→ue, ß→ss` folded into **one** canonical token per word. `Förderprogramm` and `Foerderprogramm` are the same term, and BM25 length norms stay exact.

## Retrieval pipeline

### Phase A — postings only

One weighted `BooleanQuery` across the nine weighted fields, plus:

- **Columnar `Must` filters** — `folder`, `ext`, `kind`, `status`, `mime`, `created_at`/`updated_at` ranges, promoted typed columns. Filtering happens inside Tantivy, before scoring.
- **Phrase boost** — a postings-only `PhraseQuery` for 2..=32 distinct terms: `title` weight ×2, `body` weight ×10. Up to 32 terms so long verbatim quotes keep their boost; they used to lose it past 8.
- **All-terms conjunction** — every query term present, ×4.
- **Exact filename stem** — a single-token query that is exactly a path stem gets ×6 against `name` and `path`. Separator-split parts stay unboosted; multi-word queries are unaffected.
- **Fuzzy (opt-in)** — `fuzzy_edits > 0` adds a distance-1 `FuzzyTermQuery` (transposition cost 1) at ×0.25 for non-numeric terms of ≥ 4 chars, first six terms only, on `title` and `body`. Exact weights keep a correct title ahead of a fuzzy match.
- **Widening** — fetch `max(top_k × 20, 200)` hits, widen ×5 up to 4 rounds when filters or ACL remove too many.

Phase A performs **zero** document-store reads.

### Phase B — authority and hydration

Walks the Phase A order reading **only** `id`/`parent_id` from fast fields to:

1. Enforce the ACL overlay, fail-closed.
2. Apply `group_by_parent` (`{max_per_parent}`).

Stops as soon as `top_k` winners exist, then hydrates the stored envelope for those winners alone. `explain: true` adds matched terms, coverage, phrase flag and counters (`acl_filtered`, `diversity_filtered`, `candidates_considered`), computed from the stored `title`/`path`.

### German compounds

`decompose_german_compound` is a deterministic fewest-parts DP over a frozen, versioned lexicon (500+ morphemes, umlaut folding, Fugenlaute `s/es/n/en/er`, parts ≥ 4 chars). Index time writes `body_parts` (lexicon prefixes of each word plus nested decomposition). Query time searches the same derived term set against `body_parts` only, at weight 0.35. Because that is strictly below `body`, a document holding the exact compound always outranks one holding only the parts. Compound → component and component → compound both resolve.

## Dense lane

- **Keyed by content** — `(content_hash, model_version) → Vec<f32>`. Identical searchable text shares one embedding.
- **Append-only `vectors.bin`** — attach appends records. A legacy v1 fixed-count file is upgraded once, atomically, on first append.
- **Generation-stamped link map** — `vec_map.gen_<N>.bin` maps `doc_id → content_hash`; the manifest names the live generation and superseded files are deleted after the swap.
- **Pending is derived** — `next_pending_embeddings(limit)` = `live ∧ ¬has_vector`, ordered by id. No queue to drift.
- **Compare-and-set attach** — `attach_vectors(attachments)` validates the **whole batch** (dimension, finiteness, model version, liveness, hash equality) before writing. Any stale completion rejects the batch (`StaleVectorCas` / `409 stale_vector_cas`); nothing is half-applied.
- **Link transitions** — ACL/metadata-only upsert keeps the link; text edit to unseen text drops it (the document reappears as pending); text edit onto text another document already embedded re-links with no embedding request; delete drops the link and leaves the embedding until a rebuild GCs unreferenced hashes.

## Hybrid fusion

```
score = lw / (rrf_k + lexical_rank) + vw / (rrf_k + vector_rank)
```

1-based ranks. Defaults (`DEFAULT_RRF_K` 60, `DEFAULT_CANDIDATE_K` 200, `lexical_weight` 1.0, `vector_weight` 0.3) are per-request wire fields.

- `dense` on the wire defaults to `query_vector.is_some()`; `lexical_enabled` defaults to `true`.
- **Lexical spike preservation** — when `vector_weight > lexical_weight`, take the BM25 score of the 5th distinct-parent lexical hit as a baseline. A candidate at ≥ 2.5× that baseline (phrase, title-exact, all-terms hit) keeps effective lexical weight `max(lw, vw × 1.1)`. Typical topical queries sit at 1.1–1.4×; verbatim hits at 6–45×.
- **Diversity after fusion** — `group_by_parent` runs once on the fused order, then `top_k` winners are hydrated, vector-only winners included.
- **Authority parity** — the dense lane applies the same columnar filters (via a filter-only posting walk intersected with the ACL overlay) and the same visibility check while scanning.
- Similarity is higher-is-better and matches `src/search.rs::compute_score` (cosine as similarity; Euclidean `1/(1+d)`).

## Level 2 — ACL re-check + cross-encoder

`refine_level2(...)` / `POST /v2/collections/{name}/level2`:

1. **Principals first, then degradation, then reranker availability** — fail closed in that order.
2. Level-1 hits lacking ACL coverage are **dropped before scoring**. Level 1 is evidence, not permission; a revoked document is neither scored nor echoed.
3. Survivors are capped at `shortlist` (default 20) *after* the ACL filter.
4. Text is `texts[id]`, else the hit `excerpt`, else `title + " " + path`.
5. The reranker must return exactly one finite score per document (else `Rerank` / `500 rerank_failed`).
6. Scores fuse with the Level-1 order by weighted RRF: `LEVEL2_RRF_K` 60, Level-1 weight 1.0, `rerank_weight` default 0.2.
7. `best_rerank_score < relevance_floor` (default 0.02) sets `rejected: true`; the hits are still returned as low-relevance evidence.

Without a configured reranker, `/level2` answers `503 reranker_not_configured`. There is no silent fallback to Level-1 ordering.

The in-process variant takes a `SourceConnector` and re-reads every survivor, so a document revoked or deleted since the last sync is dropped silently (announcing it would disclose that it exists) and never reaches the reranker.

## ACL, sync and integrity

- **ACL overlay** — never read from segments. `update_acl_only` bumps the generation and rewrites overlay + manifest atomically. No re-index, immediately visible. Absent documents, empty group lists and empty principal lists are invisible; a query with no principals fails with `MissingPrincipals`.
- **Two hashes** — `content_hash` covers searchable text and gates embedding reuse. `envelope_hash` covers the canonical envelope including sorted ACL and canonical JSON metadata, so ACL-only and metadata-only updates are never skipped as no-ops.
- **Sync watermark** — the index commits the highest *contiguous* synced sequence. A changed source epoch, a cursor ahead of the source, or `CursorExpired` forces a full snapshot rebuild. A failed apply persists `degraded = true`; reads fail with `IndexDegraded` until a rebuild heals.
- **Fingerprints** — `fingerprint_entries()` exports `{id, content_hash, envelope_hash, has_vector, acl}` sorted by id, hex-encoded like the hash functions.
- **Integrity check** — compares caller-supplied `{id, content_hash, envelope_hash}` triples against live fingerprints and returns `{clean, missing, extra, content_mismatched, envelope_mismatched, missing_vectors}`. `missing_vectors` (tracked documents without an embedding) is populated only with `check_vectors: true`.

## HTTP API

All routes are JSON. Writes accept `commit` (default `true`) except `attach_vectors`, which commits its own link generation. `V2Query` and the other wire bodies deserialize with `deny_unknown_fields`.

| Method | Path | Request | Response |
|---|---|---|---|
| `GET` | `/health` | — | `{"ok":true,"engine":"nvdb-v2","collections":[...],"reranker_loaded":bool}` |
| `GET` | `/v2/collections` | — | `{"collections":[{name,doc_count,acl_generation,watermark}]}` |
| `POST` | `/v2/collections` | `{"name","config"?}` | `{"ok":true,"collection":{...}}` |
| `POST` | `/v2/collections/{name}/upsert` | `{"docs":[V2Document],"commit"?}` | `{"upserted":N}` |
| `POST` | `/v2/collections/{name}/delete` | `{"ids":[...],"commit"?}` | `{"deleted":N}` |
| `POST` | `/v2/collections/{name}/acl` | `{"id","acl":[...],"commit"?}` | `{"ok":true,"acl_generation":N}` |
| `POST` | `/v2/collections/{name}/commit` | — | `{"ok":true,"doc_count":N}` |
| `POST` | `/v2/collections/{name}/search` | `V2Query` | `{"hits":[V2SearchHit],"took_ms":f64}` |
| `POST` | `/v2/collections/{name}/hybrid_search` | `V2HybridQuery` | `{"hits":[V2HybridHit],"took_ms":f64}` |
| `GET` | `/v2/collections/{name}/pending_embeddings?limit=N` | — | `{"pending":[PendingEmbedding],"count":N}` |
| `POST` | `/v2/collections/{name}/attach_vectors` | `{"attachments":[VectorAttachment]}` | `{"attached":N,"vec_generation":N}` |
| `GET` | `/v2/collections/{name}/fingerprints` | — | `{"doc_count":N,"watermark":...,"entries":[FingerprintEntry]}` |
| `POST` | `/v2/collections/{name}/integrity_check` | `{"expected":[...],"check_vectors"?}` | `IntegrityDiff` |
| `POST` | `/v2/collections/{name}/watermark` | `SyncWatermark` | `{"ok":true,"watermark":...}` |
| `POST` | `/v2/collections/{name}/level2` | `{"query_text","principals","hits","texts"?,"shortlist"?,"rerank_weight"?,"relevance_floor"?}` | `V2Level2Result` |

### `V2Query` wire shape

```
{ "text": "...", "principals": ["group"], "top_k": 10,
  "fuzzy_edits": 0, "explain": false,
  "group_by_parent": {"max_per_parent": 3},
  "filters": {
    "folder": "docs", "ext": "md", "kind": "doc", "status": "active", "mime": "text/markdown",
    "created_at": {"min": 1700000000}, "updated_at": {"min": 1700000000},
    "promoted": [{"field": "year", "op": "gte", "value": 2023}]
  } }
```

`text: ""` means filter-only. A missing or empty `principals` list fails with `MissingPrincipals`.

### `V2HybridQuery` wire shape

```
{ "lexical": V2Query, "query_vector": [f32], "dense": true,
  "lexical_enabled": true, "candidate_k": 200, "rrf_k": 60,
  "lexical_weight": 1.0, "vector_weight": 0.3 }
```

### Hits

`V2SearchHit`: `id`, `parent_id`, `chunk_index`, `kind`, `title`, `path`, `excerpt`, `score`, `bm25_score`, `explain`.

`V2HybridHit`: `hit` (a `V2SearchHit` whose `score` is the fused RRF value), `lexical_score`, `vector_score`, `lexical_rank`, `vector_rank`.

### Errors

Body: `{"error": "...", "code": "..."}`.

| Status | Codes |
|---|---|
| `400` | malformed body, invalid name (traversal included), `MissingPrincipals`, `WrongDimension`, `vector_lane_not_configured`, `invalid_watermark` |
| `403` | non-v2 directory the server refuses to take over |
| `404` | unknown route or collection |
| `409` | config/version conflict, `stale_vector_cas` |
| `413` | body over 64 MB |
| `500` | IO, corrupt state, `rerank_failed` |
| `503` | `index_degraded`, `reranker_not_configured`, connection cap |

Collection names match `[a-zA-Z0-9._-]+`; `.` and `..` are rejected, so a name can never escape the collection root.

### Limits

Hand-rolled HTTP/1.1 on `std::net`: thread per connection behind a 64-connection cap, keep-alive, 30 s read/write timeouts, 64 MB body cap.

## Tests

| Suite | Covers |
|---|---|
| `phase0_sync_contract_tests` | sync/snapshot/ACL/revision contract |
| `phase2_v2_engine_tests` | engine, phases, ACL overlay, fingerprints |
| `phase2_v2_server_tests` | HTTP routing, errors, limits, lifecycle |
| `phase3_v2_hybrid_tests` | vectors, pending/attach CAS, fusion, spikes, stem matching |

```bash
cargo test --test phase3_v2_hybrid_tests
```
