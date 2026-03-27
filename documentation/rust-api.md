# nVDB Rust API Reference

> Complete API documentation for the `nvdb` Rust crate.

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
nvdb = { path = "../nvdb" }
serde_json = "1"
```

---

## Quick Start

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

// Flush to segment (or let auto-flush handle it)
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

---

## `Database`

The top-level database instance. Contains multiple collections, each with its own vector dimension.

### Opening

#### `Database::open(path) -> Result<Arc<Database>>`

Open or create a database at the given directory path.

- Creates the directory if it doesn't exist.
- Loads the database manifest (collection list).
- Returns `Arc<Database>` for shared ownership.

```rust
let db = Database::open("./mydata")?;
```

### Collection Management

#### `db.create_collection(name, config) -> Result<Collection>`

Create a new collection with the given name and configuration.

- Fails if a collection with the same name already exists.
- Creates the collection directory, manifest, and empty WAL.
- Acquires an exclusive lock on the collection.

```rust
let coll = db.create_collection("embeddings", CollectionConfig::new(768))?;
```

#### `db.get_collection(name) -> Result<Collection>`

Open an existing collection by name.

- Fails if the collection doesn't exist.
- Loads segments, replays WAL, loads HNSW index if present.

```rust
let coll = db.get_collection("embeddings")?;
```

#### `db.list_collections() -> Vec<String>`

List all collection names in the database.

```rust
for name in db.list_collections() {
    println!("Collection: {}", name);
}
```

#### `db.drop_collection(name) -> Result<()>`

Permanently delete a collection and all its data.

- Removes the collection directory and all contents.
- The collection must not be open elsewhere.

```rust
db.drop_collection("old_embeddings")?;
```

---

## `CollectionConfig`

Configuration for creating a collection.

### `CollectionConfig::new(dim) -> CollectionConfig`

Create a config with the given vector dimension and default durability (`Buffered`).

```rust
let config = CollectionConfig::new(1536);
```

### `config.with_durability(durability) -> CollectionConfig`

Set the durability level.

```rust
use nvdb::Durability;

let config = CollectionConfig::new(768)
    .with_durability(Durability::FdatasyncEachBatch);
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dim` | `usize` | required | Vector dimension (e.g. 768, 1536) |
| `durability` | `Durability` | `Buffered` | Write durability level |

---

## `Durability`

```rust
pub enum Durability {
    Buffered,              // OS page cache only (fastest)
    FdatasyncEachBatch,    // fdatasync after each batch (safest)
}
```

---

## `Document`

A vector document with optional metadata payload.

```rust
pub struct Document {
    pub id: String,              // User-provided unique ID
    pub vector: Vec<f32>,        // Embedding vector
    pub payload: Option<Value>,  // Optional JSON metadata
}
```

### Creating Documents

```rust
let doc = Document {
    id: "doc1".to_string(),
    vector: vec![0.1, 0.2, 0.3, 0.4],
    payload: Some(json!({
        "title": "Example",
        "category": "test",
        "year": 2024
    })),
};
```

---

## `Collection`

The main interface for vector operations. Each collection has a fixed dimension.

### Write Operations

#### `coll.insert(doc) -> Result<()>`

Insert or replace a document.

- Validates vector dimension matches collection config.
- Appends to WAL for durability.
- Inserts into memtable.
- Auto-flushes if WAL exceeds 64MB.

```rust
coll.insert(Document {
    id: "doc1".to_string(),
    vector: vec![0.1; 768],
    payload: Some(json!({"title": "Example"})),
})?;
```

#### `coll.insert_batch(docs) -> Result<()>`

Insert multiple documents in a single batch.

- More efficient than N individual inserts (single WAL sync).
- Validates all documents before writing.
- Maximum batch size: 64MB.

```rust
let docs: Vec<Document> = (0..100)
    .map(|i| Document {
        id: format!("doc_{}", i),
        vector: vec![0.1; 768],
        payload: Some(json!({"index": i})),
    })
    .collect();

coll.insert_batch(docs)?;
```

#### `coll.delete(id) -> Result<bool>`

Delete a document by ID (soft delete).

- Returns `true` if the document existed.
- Document is physically removed during compaction.

```rust
let existed = coll.delete("doc1")?;
```

### Read Operations

#### `coll.get(id) -> Result<Option<Document>>`

Get a document by ID.

- Searches memtable first, then segments (newest to oldest).
- Returns `None` if not found or deleted.

```rust
if let Some(doc) = coll.get("doc1")? {
    println!("Found: {} (dim={})", doc.id, doc.vector.len());
}
```

### Search

#### `coll.search(search: &Search) -> Result<Vec<Match>>`

Search for similar vectors. See [Search Builder](#search-builder) for configuration.

```rust
// Exact search (100% recall)
let results = coll.search(
    Search::new(&query_vector)
        .top_k(10)
        .distance(Distance::Cosine)
)?;

// Approximate search with HNSW
let results = coll.search(
    Search::new(&query_vector)
        .top_k(10)
        .distance(Distance::Cosine)
        .approximate(true)
        .ef(128)
)?;

// With metadata filter
let results = coll.search(
    Search::new(&query_vector)
        .top_k(10)
        .filter(Filter::eq("category", "tutorial"))
)?;
```

### Maintenance

#### `coll.flush() -> Result<()>`

Flush the memtable to a new segment.

1. Freeze current memtable
2. Write to new segment file
3. Update manifest
4. Reset WAL

```rust
coll.flush()?;
```

#### `coll.sync() -> Result<()>`

Force-sync the WAL to disk.

```rust
coll.sync()?;
```

#### `coll.compact() -> Result<CompactionResult>`

Merge all segments and remove deleted documents.

- Rebuilds HNSW index if one exists.
- Crash-safe: old segments preserved until atomic manifest update.

```rust
let result = coll.compact()?;
println!("{} → {} documents", result.docs_before, result.docs_after);
```

#### `coll.rebuild_index(params, distance) -> Result<()>`

Build or rebuild the HNSW index.

- Scans all segments and memtable.
- Can be expensive for large collections.

```rust
use nvdb::{HnswParams, Distance};

coll.rebuild_index(
    Some(HnswParams::with_m(32)),
    Some(Distance::Cosine),
)?;
```

#### `coll.delete_index() -> Result<()>`

Remove the HNSW index. Subsequent approximate searches fall back to exact search.

```rust
coll.delete_index()?;
```

#### `coll.has_index() -> bool`

Check if an HNSW index exists.

```rust
if !coll.has_index() {
    coll.rebuild_index(None, None)?;
}
```

#### `coll.stats() -> CollectionStats`

Get collection statistics.

```rust
let stats = coll.stats();
println!("Memtable: {} docs", stats.memtable_docs);
println!("Segments: {} ({} docs)", stats.segment_count, stats.total_segment_docs);
```

---

## `Search` Builder

Configure search queries using the builder pattern.

### `Search::new(vector) -> Search`

Create a new search with the given query vector.

```rust
let search = Search::new(&query_vector);
```

### Methods

| Method | Default | Description |
|--------|---------|-------------|
| `.top_k(n)` | 10 | Number of results to return |
| `.distance(metric)` | `Cosine` | Distance metric |
| `.approximate(bool)` | `false` | Use HNSW approximate search |
| `.ef(n)` | index default | HNSW search quality parameter |
| `.filter(f)` | `None` | Metadata filter |

### Examples

```rust
// Simple top-10 cosine search
let results = coll.search(Search::new(&query).top_k(10))?;

// Dot product, top 5
let results = coll.search(
    Search::new(&query)
        .top_k(5)
        .distance(Distance::DotProduct)
)?;

// HNSW approximate with high recall
let results = coll.search(
    Search::new(&query)
        .top_k(20)
        .approximate(true)
        .ef(256)
)?;

// Filtered search
let results = coll.search(
    Search::new(&query)
        .top_k(10)
        .filter(Filter::and([
            Filter::eq("status", "active"),
            Filter::gt("year", 2020),
        ]))
)?;
```

---

## `Distance`

```rust
pub enum Distance {
    Cosine,        // cosine similarity: [-1, 1], higher = more similar
    DotProduct,    // dot product: unbounded, higher = more similar
    Euclidean,     // L2 distance: ≥0, lower = more similar
}
```

---

## `Match`

A search result.

```rust
pub struct Match {
    pub id: String,              // Document ID
    pub score: f32,              // Similarity score
    pub payload: Option<Value>,  // Document metadata
}
```

---

## `Filter`

MongoDB-like filter DSL for metadata filtering. See [Filter DSL Reference](filter-dsl.md) for full documentation.

```rust
use nvdb::Filter;

// Equality
Filter::eq("category", "books")

// Comparison
Filter::gt("year", 2020)

// Combined
Filter::and([
    Filter::eq("status", "active"),
    Filter::gte("score", 4.5),
])
```

---

## `HnswParams`

HNSW index parameters.

```rust
pub struct HnswParams {
    pub m: usize,               // Max neighbors per node (default: 16)
    pub ef_construction: usize,  // Build candidate pool (default: 64)
    pub ef_search: usize,        // Search candidate pool (default: 32)
    pub level_factor: f32,       // Layer probability (default: 1/ln(M))
}
```

### `HnswParams::default()`

Sensible defaults for most use cases (M=16).

### `HnswParams::with_m(m) -> HnswParams`

Create params with a specific M value. Automatically sets `ef_construction = 4*M`, `ef_search = 2*M`.

```rust
let params = HnswParams::with_m(32)
    .with_ef_construction(200)
    .with_ef_search(100);
```

---

## `CompactionResult`

```rust
pub struct CompactionResult {
    pub docs_before: usize,
    pub docs_after: usize,
    pub segments_merged: usize,
    pub new_segment: PathBuf,
    pub index_rebuilt: bool,
}
```

---

## `CollectionStats`

```rust
pub struct CollectionStats {
    pub memtable_docs: usize,
    pub segment_count: usize,
    pub total_segment_docs: usize,
}
```

---

## Error Handling

All operations return `nvdb::Result<T>` which is `Result<T, nvdb::Error>`.

```rust
use nvdb::{Database, Error};

match Database::open("./data") {
    Ok(db) => { /* ... */ }
    Err(Error::Io { path, message }) => {
        eprintln!("IO error at {}: {}", path.display(), message);
    }
    Err(Error::WrongDimension { expected, got }) => {
        eprintln!("Expected {}D vectors, got {}D", expected, got);
    }
    Err(Error::CollectionExists { name }) => {
        eprintln!("Collection '{}' already exists", name);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

### Common Errors

| Error | When |
|-------|------|
| `WrongDimension` | Vector length ≠ collection dimension |
| `CollectionExists` | Creating a collection that already exists |
| `CollectionNotFound` | Getting a collection that doesn't exist |
| `Corruption` | Invalid file format or checksum |
| `Io` | Filesystem errors |
