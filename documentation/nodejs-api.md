# nVDB Node.js API Reference

> Complete API documentation for the `nvdb` Node.js native addon.

---

## Installation

```bash
npm install nvdb
```

The package includes prebuilt native binaries for:
- Windows x64
- macOS x64 / arm64
- Linux x64

### Building from Source

```bash
cd nvdb/napi && node setup.js
```

Or manually:

```bash
cargo build --release -p nvdb-node
```

---

## Quick Start

```js
const { Database } = require('nvdb');

// Open a database
const db = new Database('./data');

// Create a collection with 768-dimensional vectors
const coll = db.createCollection('embeddings', 768);

// Insert documents
coll.insert('doc1', [0.1, 0.2, 0.3, /* ...768 floats */], JSON.stringify({
    title: 'Hello world',
    category: 'greeting'
}));

// Batch insert
coll.insertBatch([
    { id: 'doc2', vector: [0.2, 0.3, 0.4], payload: JSON.stringify({ title: 'World' }) },
    { id: 'doc3', vector: [0.3, 0.4, 0.5], payload: JSON.stringify({ title: 'Test' }) },
]);

// Flush to disk
coll.flush();

// Search
const results = coll.search({
    vector: [0.1, 0.2, 0.3],
    topK: 10,
    distance: 'cosine'
});

for (const match of results) {
    console.log(`${match.id}: score=${match.score.toFixed(4)}`);
    if (match.payload) {
        console.log('  payload:', JSON.parse(match.payload));
    }
}

// Cleanup
db.dropCollection('embeddings');
```

---

## `Database`

### Constructor

#### `new Database(path)`

Open or create a database at the given directory path.

```js
const db = new Database('./data');
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `string` | Directory path for the database |

---

### Methods

#### `db.createCollection(name, dimension, options?) -> Collection`

Create a new collection with the specified vector dimension.

```js
const coll = db.createCollection('embeddings', 768);
```

```js
// With sync durability
const coll = db.createCollection('embeddings', 768, {
    durability: 'sync'
});
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Collection name |
| `dimension` | `number` | Vector dimension (e.g. 768, 1536) |
| `options` | `CollectionOptions?` | Optional configuration |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `durability` | `'buffered' \| 'sync'` | `'buffered'` | Write durability level |

---

#### `db.getCollection(name) -> Collection`

Open an existing collection.

```js
const coll = db.getCollection('embeddings');
```

---

#### `db.listCollections() -> string[]`

List all collection names.

```js
const names = db.listCollections();
console.log('Collections:', names);
```

---

#### `db.dropCollection(name) -> void`

Permanently delete a collection and all its data.

```js
db.dropCollection('old_embeddings');
```

---

## `Collection`

A collection of vectors with the same dimension.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `string` | Collection name (read-only) |
| `config` | `CollectionConfig` | Collection configuration (read-only) |
| `stats` | `CollectionStats` | Collection statistics (read-only) |

---

### Insert Operations

#### `coll.insert(id, vector, payload?) -> void`

Insert or replace a single document.

```js
coll.insert('doc1', [0.1, 0.2, 0.3, 0.4], JSON.stringify({
    title: 'Example',
    category: 'test'
}));
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `string` | Unique document ID |
| `vector` | `number[]` | Embedding vector (float64, converted to float32 internally) |
| `payload` | `string?` | Optional JSON string metadata |

---

#### `coll.insertBatch(docs) -> void`

Insert multiple documents efficiently.

```js
coll.insertBatch([
    { id: 'doc1', vector: [0.1, 0.2, 0.3], payload: JSON.stringify({ tag: 'a' }) },
    { id: 'doc2', vector: [0.4, 0.5, 0.6], payload: JSON.stringify({ tag: 'b' }) },
    { id: 'doc3', vector: [0.7, 0.8, 0.9] },
]);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `docs` | `InsertDoc[]` | Array of documents to insert |

**InsertDoc:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique document ID |
| `vector` | `number[]` | Embedding vector |
| `payload` | `string?` | Optional JSON string metadata |

---

### Read Operations

#### `coll.get(id) -> Document | null`

Get a document by ID.

```js
const doc = coll.get('doc1');
if (doc) {
    console.log('ID:', doc.id);
    console.log('Vector length:', doc.vector.length);
    console.log('Payload:', doc.payload ? JSON.parse(doc.payload) : null);
}
```

**Returns:** `Document` or `null` if not found.

**Document:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Document ID |
| `vector` | `number[]` | Embedding vector (float64) |
| `payload` | `string?` | JSON string metadata |

---

#### `coll.delete(id) -> boolean`

Delete a document by ID (soft delete).

```js
const existed = coll.delete('doc1');
console.log('Existed:', existed);
```

**Returns:** `true` if the document existed and was deleted.

---

### Search

#### `coll.search(options) -> Match[]`

Search for similar vectors.

```js
// Basic search
const results = coll.search({
    vector: queryVector,
    topK: 10,
    distance: 'cosine'
});

// Approximate search with HNSW
const results = coll.search({
    vector: queryVector,
    topK: 10,
    distance: 'cosine',
    approximate: true,
    ef: 128
});

// Filtered search
const results = coll.search({
    vector: queryVector,
    topK: 10,
    filter: JSON.stringify({ Eq: { field: 'category', value: 'books' } })
});
```

**SearchOptions:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `vector` | `number[]` | required | Query vector |
| `topK` | `number?` | `10` | Number of results |
| `distance` | `'cosine' \| 'dot' \| 'euclidean'` | `'cosine'` | Distance metric |
| `approximate` | `boolean?` | `false` | Use HNSW index |
| `ef` | `number?` | index default | HNSW search quality |
| `filter` | `string?` | `null` | JSON filter string |

**Match:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Document ID |
| `score` | `number` | Similarity score (higher = better for cosine/dot) |
| `payload` | `string?` | JSON string metadata |

---

### Filtered Search with FilterBuilder

Use the `FilterBuilder` to construct filter JSON:

```js
const { Database, FilterBuilder } = require('nvdb');

const db = new Database('./data');
const coll = db.getCollection('embeddings');

// Simple equality filter
const results = coll.search({
    vector: queryVector,
    topK: 10,
    filter: FilterBuilder.eq('category', 'books')
});

// Combined filters
const filter = JSON.stringify({
    And: [
        { Eq: { field: 'category', value: 'books' } },
        { Gt: { field: 'year', value: 2020 } }
    ]
});

const results = coll.search({
    vector: queryVector,
    topK: 10,
    filter: filter
});
```

See [Filter DSL Reference](filter-dsl.md) for all filter operators.

---

### Maintenance

#### `coll.flush() -> void`

Flush the memtable to a new segment file on disk.

```js
coll.flush();
```

#### `coll.sync() -> void`

Force-sync the WAL to disk for maximum durability.

```js
coll.sync();
```

#### `coll.compact() -> CompactionResult`

Merge all segments and remove deleted documents.

```js
const result = coll.compact();
console.log(`Compacted: ${result.docsBefore} → ${result.docsAfter} docs`);
console.log(`Segments merged: ${result.segmentsMerged}`);
console.log(`Index rebuilt: ${result.indexRebuilt}`);
```

**CompactionResult:**

| Field | Type | Description |
|-------|------|-------------|
| `docsBefore` | `number` | Documents before compaction |
| `docsAfter` | `number` | Documents after compaction |
| `segmentsMerged` | `number` | Number of segments merged |
| `indexRebuilt` | `boolean` | Whether HNSW index was rebuilt |

#### `coll.rebuildIndex() -> void`

Build or rebuild the HNSW index for approximate search.

```js
coll.rebuildIndex();
```

#### `coll.deleteIndex() -> void`

Remove the HNSW index. Subsequent approximate searches fall back to exact search.

```js
coll.deleteIndex();
```

#### `coll.hasIndex() -> boolean`

Check if an HNSW index exists.

```js
if (!coll.hasIndex()) {
    coll.rebuildIndex();
}
```

---

## `CollectionStats`

| Field | Type | Description |
|-------|------|-------------|
| `memtableDocs` | `number` | Documents in memtable (not yet flushed) |
| `segmentCount` | `number` | Number of segment files |
| `totalSegmentDocs` | `number` | Total documents across all segments |

```js
const stats = coll.stats;
console.log(`Memtable: ${stats.memtableDocs} docs`);
console.log(`Segments: ${stats.segmentCount} (${stats.totalSegmentDocs} docs)`);
```

---

## `FilterBuilder`

Static helper for constructing filter JSON strings.

| Method | Description |
|--------|-------------|
| `FilterBuilder.eq(field, value)` | Equality: field == value |
| `FilterBuilder.gt(field, value)` | Greater than: field > value |
| `FilterBuilder.gte(field, value)` | Greater than or equal: field >= value |
| `FilterBuilder.lt(field, value)` | Less than: field < value |
| `FilterBuilder.lte(field, value)` | Less than or equal: field <= value |
| `FilterBuilder.ne(field, value)` | Not equal: field != value |
| `FilterBuilder.in(field, values)` | In array: field IN values |
| `FilterBuilder.and(filters)` | Logical AND: all must match |
| `FilterBuilder.or(filters)` | Logical OR: any must match |

---

## Type Definitions

Full TypeScript definitions are available in `nvdb/napi/index.d.ts`.

```typescript
import {
    Database,
    Collection,
    FilterBuilder,
    CollectionOptions,
    InsertDoc,
    Document,
    SearchOptions,
    Match,
    CompactionResult,
    CollectionStats
} from 'nvdb';
```

---

## Error Handling

All operations throw JavaScript errors on failure:

```js
try {
    const coll = db.createCollection('embeddings', 768);
    coll.insert('doc1', [0.1, 0.2]); // Wrong dimension!
} catch (err) {
    console.error('Error:', err.message);
    // "Failed to insert document: WrongDimension { expected: 768, got: 2 }"
}
```

### Common Errors

| Error Message | Cause |
|---------------|-------|
| `Failed to open database: ...` | Invalid path or permissions |
| `Failed to create collection: CollectionExists` | Collection already exists |
| `Failed to get collection: CollectionNotFound` | Collection doesn't exist |
| `Failed to insert document: WrongDimension` | Vector length ≠ collection dimension |
| `Invalid JSON payload: ...` | Malformed JSON in payload string |
| `Search failed: ...` | Invalid search parameters |
