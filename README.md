# nVDB

> High-performance embedded vector database for AI/ML workflows.

nVDB is an **embedded vector database** with SIMD-accelerated similarity search, optional HNSW approximate search, and MongoDB-like metadata filtering. Standalone embeddable database for Node.js and Rust applications.

> **⚠️ BREAKING CHANGE (v3 Architecture)**
> nVDB has transitioned to a **Database-as-a-Folder** architecture natively managed in Rust. Collections are now isolated folders containing `meta.json` (dimensionality and schemas) and data structures. Native delta operations like `patchPayload` replace total re-writes, drastically reducing I/O when only updating metadata on vectors.

## Features

- **SIMD-accelerated search** — AVX2/AVX-512/NEON via the `wide` crate (8-wide `f32x8`)
- **HNSW approximate search** — Sub-linear search with configurable recall/latency tradeoffs
- **LSM-Lite storage** — Memtable + memory-mapped segments + WAL for crash recovery
- **Zero-copy reads** — Memory-mapped segments mean instant startup, no loading
- **MongoDB-like filters** — Query by metadata with a familiar DSL
- **Multiple collections** — Different embedding models in one database
- **Node.js native bindings** — napi-rs powered, zero-copy where possible
- **Crash-safe** — WAL ensures no data loss on power failure

## Quick Start

### Rust

```rust
use nvdb::{Database, CollectionConfig, Document, Distance, Search};
use serde_json::json;

// Open a database
let db = Database::open("./data")?;

// Create a collection with 768-dimensional vectors
let coll = db.create_collection("embeddings", CollectionConfig::new(768))?;

// Insert a document
coll.insert(Document {
    id: "doc1".to_string(),
    vector: vec![0.1; 768],
    payload: Some(json!({"title": "Hello world", "category": "greeting"})),
})?;

// Flush to segment
coll.flush()?;

// Search
let query = vec![0.1; 768];
let results = coll.search(
    Search::new(&query)
        .top_k(10)
        .distance(Distance::Cosine)
)?;

for m in &results {
    println!("{}: score={:.4}", m.id, m.score);
}
```

### Node.js

```js
const { Database } = require('nvdb');

const db = new Database('./data');
const coll = db.createCollection('embeddings', 768);

// Insert
coll.insert('doc1', vector768, JSON.stringify({ title: 'Hello world' }));

// Search
const results = coll.search({
    vector: queryVector,
    topK: 10,
    distance: 'cosine'
});

for (const match of results) {
    console.log(`${match.id}: score=${match.score.toFixed(4)}`);
}
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](documentation/architecture.md) | Internal design, storage format, concurrency model |
| [Rust API](documentation/rust-api.md) | Complete Rust API reference |
| [Node.js API](documentation/nodejs-api.md) | Complete Node.js API reference |
| [Filter DSL](documentation/filter-dsl.md) | Metadata filtering with MongoDB-like syntax |

## API Overview

### Database Operations

```rust
let db = Database::open("./data")?;
let coll = db.create_collection("embeddings", CollectionConfig::new(768))?;
let names = db.list_collections();
db.drop_collection("old_data")?;
```

### Document Operations

```rust
// Insert
coll.insert(Document {
    id: "doc1".to_string(),
    vector: vec![0.1; 768],
    payload: Some(json!({"title": "Example"})),
})?;

// Batch insert
coll.insert_batch(docs)?;

// Get
let doc = coll.get("doc1")?;

// Delete (soft delete)
coll.delete("doc1")?;

// Flush memtable to segment
coll.flush()?;
```

### Search

```rust
use nvdb::{Search, Distance, Filter};

// Exact search (100% recall)
let results = coll.search(Search::new(&query).top_k(10))?;

// Approximate search with HNSW
let results = coll.search(
    Search::new(&query)
        .top_k(10)
        .approximate(true)
        .ef(128)
)?;

// Filtered search
let results = coll.search(
    Search::new(&query)
        .top_k(10)
        .filter(Filter::and([
            Filter::eq("category", "tutorial"),
            Filter::gte("year", 2023),
        ]))
)?;
```

### Index Management

```rust
// Build HNSW index
coll.rebuild_index(None, None)?;

// Build with custom parameters
coll.rebuild_index(
    Some(HnswParams::with_m(32)),
    Some(Distance::Cosine),
)?;

// Compact segments + rebuild index
coll.compact()?;

// Remove index
coll.delete_index()?;
```

## Architecture

nVDB uses an **LSM-Lite** architecture:

```
┌─────────────────────────────────────────────┐
│  Memtable (HashMap + SoA)    ← Read+Write  │
├─────────────────────────────────────────────┤
│  Segments (mmap, immutable)  ← Read-only   │
├─────────────────────────────────────────────┤
│  WAL (append-only)           ← Crash safe  │
├─────────────────────────────────────────────┤
│  HNSW Index (CSR graph)      ← On-demand   │
└─────────────────────────────────────────────┘
```

- **Writes** go to WAL + memtable
- **Reads** check memtable first, then segments
- **Flush** freezes memtable → new segment
- **Compaction** merges segments, removes deleted docs

## Performance

| Dataset Size | Exact Search | Approximate (HNSW) |
|--------------|--------------|-------------------|
| 10K docs | <1ms | <1ms |
| 100K docs | 5-10ms | 1-2ms |
| 1M docs | 50-100ms | 2-5ms |
| 10M docs | 500ms-1s | 5-10ms |

*Benchmarks with 768-dimensional vectors, cosine similarity, single thread.*

## Distance Metrics

| Metric | Score Direction | Best For |
|--------|----------------|----------|
| **Cosine** (default) | Higher = more similar | Normalized embeddings |
| **Dot Product** | Higher = more similar | Pre-normalized vectors |
| **Euclidean** | Lower = more similar | Spatial data |

All metrics use SIMD acceleration (AVX2/AVX-512/NEON).

## Build & Test

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --open
```

## Examples

```bash
cargo run --example basic_usage
cargo run --example rag_system
cargo run --example web_service
```

## When to Use nVDB

- Storing embeddings from LLMs (768-1536 dimensions)
- Similarity search with optional metadata filtering
- Read-heavy workloads with occasional bulk writes
- Single-node deployments where embedded databases are appropriate
- Electron and Node.js applications needing local vector search

## When NOT to Use nVDB

- Client-server architecture (use Qdrant, Milvus, etc.)
- Distributed/multi-node requirements
- Multiple independent applications writing to the same data
- Complex multi-document transactions

## License

MIT OR Apache-2.0
