# nVDB Filter DSL Reference

> Complete guide to nVDB's MongoDB-like filter DSL for metadata filtering.

---

## Overview

nVDB filters are **JSON objects** that express conditions on document metadata (payload). They are used with the `search()` method to narrow results by payload fields.

Filters support:
- **Comparison operators**: `eq`, `gt`, `gte`, `lt`, `lte`, `ne`, `in`
- **Logical operators**: `and`, `or`
- **Nested field access**: dot notation (e.g. `"user.name"`)

---

## Rust API

### Comparison Filters

```rust
use nvdb::Filter;
use serde_json::json;

// Equality: field == value
Filter::eq("category", "books")

// Greater than: field > value
Filter::gt("year", 2020)

// Greater than or equal: field >= value
Filter::gte("score", 4.5)

// Less than: field < value
Filter::lt("price", 100)

// Less than or equal: field <= value
Filter::lte("age", 65)

// Not equal: field != value
Filter::ne("status", "deleted")

// In array: field IN [values]
Filter::in_("category", vec!["books", "movies", "music"])
```

### Logical Operators

```rust
// AND: all filters must match
Filter::and([
    Filter::eq("status", "active"),
    Filter::gt("year", 2020),
])

// OR: any filter must match
Filter::or([
    Filter::eq("category", "books"),
    Filter::eq("category", "movies"),
])
```

### Nested Field Access

Use dot notation to access nested JSON fields:

```rust
Filter::eq("user.name", "alice")
Filter::gt("metadata.rating", 4.0)
```

This works with payloads like:

```json
{
    "user": { "name": "alice", "id": 42 },
    "metadata": { "rating": 4.5, "views": 100 }
}
```

### Using Filters in Search

```rust
use nvdb::{Search, Filter};

let results = coll.search(
    Search::new(&query)
        .top_k(10)
        .filter(Filter::and([
            Filter::eq("category", "tutorial"),
            Filter::gte("year", 2023),
        ]))
)?;
```

---

## Node.js API

### FilterBuilder

The `FilterBuilder` class provides static methods to construct filter JSON strings:

```js
const { FilterBuilder } = require('nvdb');

// Equality
const filter = FilterBuilder.eq('category', 'books');

// Comparison
const filter = FilterBuilder.gt('year', 2020);

// Combined
const filter = FilterBuilder.and([
    FilterBuilder.eq('status', 'active'),
    FilterBuilder.gte('score', 4.5)
]);
```

### Using Filters in Search

```js
const results = coll.search({
    vector: queryVector,
    topK: 10,
    filter: FilterBuilder.eq('category', 'books')
});
```

### Raw JSON Filters

You can also construct filter JSON directly:

```js
const results = coll.search({
    vector: queryVector,
    topK: 10,
    filter: JSON.stringify({
        And: [
            { Eq: { field: 'category', value: 'books' } },
            { Gt: { field: 'year', value: 2020 } }
        ]
    })
});
```

---

## Filter Operators Reference

### Comparison Operators

| Operator | Rust | JSON | Description |
|----------|------|------|-------------|
| Equal | `Filter::eq(field, value)` | `{"Eq": {"field": "...", "value": ...}}` | Field equals value |
| Not Equal | `Filter::ne(field, value)` | `{"Ne": {"field": "...", "value": ...}}` | Field does not equal value |
| Greater Than | `Filter::gt(field, value)` | `{"Gt": {"field": "...", "value": ...}}` | Field > value |
| Greater or Equal | `Filter::gte(field, value)` | `{"Gte": {"field": "...", "value": ...}}` | Field >= value |
| Less Than | `Filter::lt(field, value)` | `{"Lt": {"field": "...", "value": ...}}` | Field < value |
| Less or Equal | `Filter::lte(field, value)` | `{"Lte": {"field": "...", "value": ...}}` | Field <= value |
| In | `Filter::in_(field, values)` | `{"In": {"field": "...", "values": [...]}}` | Field in array |

### Logical Operators

| Operator | Rust | JSON | Description |
|----------|------|------|-------------|
| AND | `Filter::and(filters)` | `{"And": [...]}` | All filters must match |
| OR | `Filter::or(filters)` | `{"Or": [...]}` | Any filter must match |

---

## Type Coercion

Numeric comparisons automatically coerce between integers and floats:

```rust
// This matches documents with {"count": 5.5}
Filter::gt("count", 5)

// This matches documents with {"count": 5}
Filter::gt("count", 4.5)
```

---

## Missing Fields

If a filter references a field that doesn't exist in a document's payload, the filter evaluates to `false` and the document is excluded from results.

```rust
// Only matches documents that HAVE a "rating" field with value >= 4.0
Filter::gte("rating", 4.0)
```

---

## Examples

### Simple Equality

Find documents where category is "books":

```rust
// Rust
Filter::eq("category", "books")
```

```js
// Node.js
FilterBuilder.eq('category', 'books')
```

### Range Query

Find documents with score between 4.0 and 5.0:

```rust
Filter::and([
    Filter::gte("score", 4.0),
    Filter::lte("score", 5.0),
])
```

### Multi-Value Match

Find documents in specific categories:

```rust
Filter::in_("category", vec!["books", "movies", "music"])
```

### Complex Nested Logic

Find active tutorials from 2023 or later with high ratings:

```rust
Filter::and([
    Filter::eq("status", "active"),
    Filter::eq("type", "tutorial"),
    Filter::gte("year", 2023),
    Filter::or([
        Filter::gte("rating", 4.5),
        Filter::gte("views", 10000),
    ]),
])
```

### Nested Field Access

Filter by nested user information:

```rust
Filter::eq("author.name", "Alice")
```

This matches documents with payload:

```json
{
    "author": {
        "name": "Alice",
        "id": 42
    }
}
```

---

## Performance Notes

- Filters are applied **post-search** for HNSW approximate search (the index returns more candidates, then filters narrow them)
- Filters are applied **during scan** for exact search (documents that don't match are skipped)
- For best HNSW performance with filters, the system fetches `2 × top_k` candidates to account for filtered-out documents
- Complex nested filters have minimal overhead — evaluation is a simple recursive descent
