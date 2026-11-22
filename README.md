# nVDB

> Hybrid retrieval engine for document corpora — Tantivy BM25F + dense vectors, fused by rank, reranked by an ONNX cross-encoder, served over a local HTTP API.

nVDB is a **derived retrieval index**: it is rebuilt from an authoritative source (a document store such as `nDB`) and never owns the data. One Rust crate (`nvdb`) plus one executable (`nvdb serve`) give you:

- **Lexical retrieval on Tantivy** — weighted BM25F across title, path, name, headings, categories, tags and body, with exact-identifier, filename-stem, phrase and German-compound matching.
- **Dense retrieval** keyed by content hash, so identical text is embedded once and metadata/ACL edits never trigger re-embedding.
- **Hybrid fusion** of both lanes by Reciprocal Rank Fusion, with per-parent diversity caps.
- **Level-2 reranking** with an ONNX cross-encoder over the fused shortlist, re-verified against the ACL before scoring.
- **ACL-gated results** — a query must carry principals; a revoked document is never scored, echoed or counted.

Everything runs in one process against one directory. There is no cluster, no external service and no model bundled into the binary.

## Quick start

Build with the reranker feature (required for `--reranker-model`):

```bash
cargo build --release --features rerank-onnx
```

Start the service. Each collection is a direct subdirectory of `--dir`; every direct subdirectory containing a `v2_manifest.json` is adopted at startup, and a collection that will not open aborts startup rather than disappearing quietly.

```bash
./target/release/nvdb serve --dir ./nvdb-data --host 127.0.0.1 --port 3510 \
  --reranker-model ./models/bge-reranker
```

`--host` defaults to `127.0.0.1`, `--port` to `3510`. Pass `--host 0.0.0.0` for LAN access. `--reranker-model` expects a directory holding `model.onnx` and `tokenizer.json`; it is only available in a `rerank-onnx` build, and a model that fails to load aborts startup. Without the flag, `/level2` answers `503 reranker_not_configured` — it never silently degrades to Level-1 ordering.

Create a collection, index a document, query it:

```bash
# Collection with a dense lane (needed for hybrid_search).
# dim/model_version must match the embeddings you attach later.
# Omit "vector_lane" for a lexical-only collection.
curl -s -X POST http://127.0.0.1:3510/v2/collections \
  -d '{"name":"docs","config":{"vector_lane":{"dim":4,"model_version":"demo-4d"}}}'

curl -s -X POST http://127.0.0.1:3510/v2/collections/docs/upsert \
  -d '{"docs":[{
        "id":"doc-1",
        "kind":"doc",
        "title":"Förderprogramm Digitalisierung",
        "path":"programs/foerderprogramm-digitalisierung.md",
        "body":"Die Richtlinie beschreibt das Förderprogramm.",
        "acl":["public"]
      }]}'

# Lexical / filter-only search (principals are mandatory)
curl -s -X POST http://127.0.0.1:3510/v2/collections/docs/search \
  -d '{"text":"foerder","principals":["public"],"top_k":5}'

# Hybrid: lexical + dense lanes fused into one ranking
curl -s -X POST http://127.0.0.1:3510/v2/collections/docs/hybrid_search \
  -d '{"lexical":{"text":"foerderprogramm","principals":["public"],"top_k":10},
       "query_vector":[0.10,0.20,0.30,0.40],"candidate_k":200}'
```

Vectors are attached out of band. The index exposes what it still needs and accepts completions with a compare-and-set:

```bash
curl -s 'http://127.0.0.1:3510/v2/collections/docs/pending_embeddings?limit=100'
# content_hash comes from that pending list; model_version must equal the lane config.
curl -s -X POST http://127.0.0.1:3510/v2/collections/docs/attach_vectors \
  -d '{"attachments":[{"id":"doc-1","content_hash":"<hex from pending_embeddings>","model_version":"demo-4d","vector":[0.10,0.20,0.30,0.40]}]}'
```

## Features

### Lexical lane (Tantivy BM25F)

- **Field weights** (defaults): `title_exact` 8.0, `title` 4.0, `path` 3.0, `name` 3.0, `headings` 2.5, `categories` 2.0, `tags` 2.0, `body` 1.0, `body_parts` 0.35. Override per collection via `config.field_weights`.
- **Exact identifiers and filename stems** — a query that is a filename stem (`md-blocks-spec`) matches `name`/`path` as a whole token; separator-split parts remain available but unboosted. Only extension-stripped stems with a 1–5 character ASCII extension qualify.
- **German compounds** — a frozen, versioned lexicon decomposes compounds at index time into `body_parts`; queries search the same derived terms, so `Förderprogramm` ↔ `Foerderprogramm` and compound ↔ component both resolve. An exact compound outranks a document that merely contains the parts.
- **Phrase and conjunction boosts** — postings-only phrase boost on `title` (×2) and `body` (×10) for 2–32 term queries, plus an all-terms conjunction boost. Longer verbatim quotes keep their boost; they used to silently lose it past 8 terms.
- **Opt-in fuzzy** — `fuzzy_edits` (Levenshtein distance) expands up to the first six non-numeric terms of four or more characters at ×0.25, so typo tolerance is added without displacing correct exact matches.
- **Two-phase search** — Phase A is pure postings work with columnar `Must` filters and no document-store reads. Phase B reads only `id`/`parent_id` from fast fields to enforce the ACL overlay and `group_by_parent`, then hydrates just the winners.
- **Columnar filters** — `folder`, `ext`, `kind`, `status`, `mime`, `created_at`/`updated_at` ranges, plus promoted metadata fields typed as `string`, `i64`, `f64` or `bool`.
- **Explanations** — `explain: true` returns coverage, matched terms, a phrase flag and counters per hit.

### Dense lane

- **Keyed by content, not by id** — `(content_hash, model_version) → vector`. Two documents with the same searchable text share one embedding; an ACL or metadata edit changes the envelope hash but not the content hash, so nothing is re-embedded.
- **Append-only `vectors.bin`** — attachments append records; legacy v1 fixed-count files are upgraded once, atomically, on first attach.
- **Derived pending list** — `GET /pending_embeddings` is computed from `live ∧ ¬has_vector`; there is no durable queue that can drift.
- **Compare-and-set attachment** — the whole batch is validated (dimension, finiteness, model version, liveness, hash equality) before anything is written. One stale completion rejects the batch with `409 stale_vector_cas` instead of half-applying it.
- **Flat scan today** — `flat_search_cutoff` (default 50 000) records the intended ANN policy; v2 always scans the content-hash map. No HNSW index is built for the v2 lane.

### Hybrid fusion

- **RRF** — `score = lw/(rrf_k + lex_rank) + vw/(rrf_k + vec_rank)`, 1-based ranks. Defaults: `rrf_k` 60, `candidate_k` 200, `lexical_weight` 1.0, `vector_weight` 0.3. All are per-request fields on `hybrid_search`.
- **Lexical spikes survive vector-heavy weighting** — when `vector_weight > lexical_weight`, a candidate whose BM25 score is at least 2.5× the fifth distinct-parent lexical hit is treated as an exact outlier (phrase, title-exact, all-terms) and keeps effective lexical weight. Flat topical queries sit around 1.1–1.4×; verbatim hits land at 6–45×.
- **Diversity after fusion** — `group_by_parent` (with `max_per_parent`) is applied once on the fused order, then `top_k` winners are hydrated, vector-only winners included.
- **Lane authority** — the dense lane honours the same filters and ACL overlay as the lexical lane.

### Level-2 reranking

`POST /v2/collections/{name}/level2` refines Level-1 hits:

- Hits the ACL overlay does not grant to the caller are dropped **before** anything else. Level 1 is evidence, not permission.
- Survivors are capped at `shortlist` (default 20) *after* the ACL filter.
- Reranker text is the caller's `texts[id]`, else the hit `excerpt`, else `title + " " + path`.
- The reranker must return exactly one finite score per document (else `500 rerank_failed`). Scores fuse with the Level-1 order by weighted RRF (`LEVEL2_RRF_K` 60, Level-1 weight 1.0 vs `rerank_weight`, default 0.2).
- If the best rerank score is below `relevance_floor` (default 0.02), the result is marked `rejected: true` while the hits are still returned as low-relevance evidence.

### Sync, ACL and integrity

- **Two hashes** — `content_hash` covers searchable text and gates embedding reuse. `envelope_hash` covers the canonical full envelope including sorted ACL and canonical metadata, so ACL-only and metadata-only changes are never treated as no-ops.
- **Generation-stamped ACL overlay** — `update_acl_only` rewrites `acl_overlay.gen_<N>.bin` and the manifest atomically. No re-index, immediately visible. Absent documents, empty ACLs and empty principal lists are all invisible; a query without principals fails with `MissingPrincipals`.
- **Watermark contract** — the index commits the highest contiguous synced sequence. A changed source epoch, a cursor ahead of the source, or an expired cursor forces a clean snapshot rebuild rather than incremental drift. A failed apply persists `degraded` and reads fail until a rebuild heals.
- **Integrity checks** — `GET /fingerprints` exports per-document hashes; `POST /integrity_check` compares them against the source snapshot and can report tracked documents that still lack a vector.

## HTTP API

All routes are JSON. Writes accept `commit` (default `true`) except `attach_vectors`, which commits its own link generation.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Engine name, collection names, `reranker_loaded` |
| `GET` | `/v2/collections` | Collection names with `doc_count`, `acl_generation`, `watermark` |
| `POST` | `/v2/collections` | Create a collection (`name`, optional `config`) |
| `POST` | `/v2/collections/{name}/upsert` | Upsert documents |
| `POST` | `/v2/collections/{name}/delete` | Delete ids |
| `POST` | `/v2/collections/{name}/acl` | Update one document's ACL |
| `POST` | `/v2/collections/{name}/commit` | Commit pending writes |
| `POST` | `/v2/collections/{name}/search` | Lexical / filter-only query |
| `POST` | `/v2/collections/{name}/hybrid_search` | Lexical + dense fusion |
| `GET` | `/v2/collections/{name}/pending_embeddings` | Documents awaiting vectors |
| `POST` | `/v2/collections/{name}/attach_vectors` | CAS vector attachment |
| `GET` | `/v2/collections/{name}/fingerprints` | Per-document content/envelope hashes |
| `POST` | `/v2/collections/{name}/integrity_check` | Compare against expected fingerprints |
| `POST` | `/v2/collections/{name}/watermark` | Persist the sync watermark |
| `POST` | `/v2/collections/{name}/level2` | ACL re-check + cross-encoder rerank |

Errors are `{"error": "...", "code": "..."}`. Notable codes: `400` malformed body, `MissingPrincipals`, `WrongDimension`, `vector_lane_not_configured`, `invalid_watermark`; `403` a non-v2 directory the server refuses to take over; `404` unknown route or collection; `409` schema/config conflict or `stale_vector_cas`; `500` IO/corruption/`rerank_failed`; `503` `index_degraded` or `reranker_not_configured`. Collection names match `[a-zA-Z0-9._-]+`, and `.`/`..` are rejected, so a name cannot escape the collection root.

The server is hand-rolled HTTP/1.1 on `std::net`: one thread per connection behind a 64-connection cap, keep-alive, 30 s I/O timeouts, 64 MB request-body cap (`413` above it).

## CLI

`nvdb serve` runs the service. The remaining commands are the v1 single-collection administration surface:

```
nvdb init <path> --dim N        initialize a collection
nvdb info|verify|compact <path> inspect, check, compact in place
nvdb export|import|merge ...    move data between databases
nvdb dump <path>                JSON Lines to stdout
nvdb config get|set ...         metadata/config
nvdb search <path> <vector>     vector search (v1)
nvdb rebuild-index <path>       rebuild the v1 HNSW graph
```

## Build and test

```bash
# debug build
cargo build

# release build with the Level-2 reranker
cargo build --release --features rerank-onnx

# all test suites (v1 + v2 engine, server, hybrid, sync contract)
cargo test

# one suite
cargo test --test phase3_v2_hybrid_tests

# benchmarks
cargo bench

# API docs
cargo doc --open
```

MSRV is Rust 1.75. `rerank-onnx` is the only optional feature; it pulls in `ort` and `tokenizers`, and without it both the ONNX reranker module and `--reranker-model` are unavailable.

## Design boundaries

- **No embedding generation.** The caller produces vectors; nVDB stores, keys and searches them.
- **Derived, rebuildable state.** The source of truth lives elsewhere; the index is always reconstructible.
- **Single node, single process per collection root.** Multi-writer safety comes from per-collection file locks, not coordination.
- **No transactions across documents, no joins, no aggregations.**
- **Flat dense scan.** Approximate search is not implemented for the v2 lane.

## Documentation

| Document | Description |
|---|---|
| [v2 engine & HTTP API](documentation/v2-server.md) | Collection layout, retrieval pipeline, fusion, reranking, sync contract |
| [Architecture](documentation/architecture.md) | v1 crate internals: LSM-Lite storage, segments, WAL, HNSW |
| [Rust API](documentation/rust-api.md) | v1 crate API reference |
| [Filter DSL](documentation/filter-dsl.md) | v1 payload filter DSL |

## License

MIT OR Apache-2.0
