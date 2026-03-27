# nVDB Architecture

> Internal design and data flow of the nVDB vector database.

---

## Overview

nVDB is an **embedded vector database** with LSM-Lite storage, SIMD-accelerated distance computation, and optional HNSW approximate search. It uses a memtable + immutable segments + WAL architecture inspired by LSM trees, but simplified for vector workloads.

```
┌──────────────────────────────────────────────────────────┐
│                       Application                         │
├──────────────┬───────────────────┬────────────────────────┤
│   Insert /   │   Exact Search    │   Approximate Search   │
│   Delete     │   (Brute-force)   │   (HNSW Index)         │
│   WAL + MT   │   SIMD Scan       │   Graph Traversal      │
├──────────────┴───────────────────┴────────────────────────┤
│                    Collection                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Memtable   │  │  Segments    │  │  HNSW Index      │ │
│  │  (HashMap)  │  │  (mmap)      │  │  (CSR Graph)     │ │
│  │  Read+Write │  │  Read-only   │  │  On-demand load  │ │
│  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘ │
│         │                │                    │           │
├─────────┴────────────────┴────────────────────┴───────────┤
│                      WAL (Append-Only)                     │
│              Crash Recovery + Durability                    │
├───────────────────────────────────────────────────────────┤
│                    Filesystem                              │
│     MANIFEST │ wal.log │ segments/*.nvdb │ index.hnsw      │
└───────────────────────────────────────────────────────────┘
```

---

## Storage Architecture

### LSM-Lite

nVDB uses a simplified LSM architecture:

| Layer | Structure | Access | Purpose |
|-------|-----------|--------|---------|
| **Memtable** | `HashMap` + SoA arrays | Read + Write | Recent writes, O(1) lookup |
| **Segments** | Memory-mapped files | Read-only | Immutable, zero-copy reads |
| **WAL** | Append-only file | Write + Replay | Crash recovery |
| **HNSW** | CSR graph file | Read-only | Approximate search |

### Write Path

```
insert(doc)
  │
  ├─ 1. Validate dimension
  ├─ 2. Append to WAL (durability)
  ├─ 3. Insert into memtable (HashMap + SoA)
  └─ 4. Check WAL size → auto-flush if ≥ 64MB
```

### Read Path

```
get(id)
  │
  ├─ 1. Check memtable (O(1) HashMap lookup)
  └─ 2. Search segments newest→oldest (mmap, zero-copy)
```

### Search Path

```
search(query_vector)
  │
  ├─ Exact (default):
  │   ├─ Scan memtable vectors (SIMD)
  │   ├─ Scan all segment vectors (SIMD, mmap)
  │   └─ Merge + sort by score → top_k
  │
  └─ Approximate (HNSW):
      ├─ Traverse graph from entry point
      ├─ Greedy descent through layers
      ├─ Post-filter if filter provided
      └─ Sort + truncate → top_k
```

---

## Concurrency Model

**Single-writer, multi-reader** using `RwLock` and `ArcSwap`:

| Operation | Lock | Behavior |
|-----------|------|----------|
| `insert()` | WAL `Mutex` + memtable `RwLock` (write) | Exclusive write |
| `insert_batch()` | WAL `Mutex` + memtable `RwLock` (write) | Exclusive write |
| `delete()` | WAL `Mutex` + memtable `RwLock` (write) | Exclusive write |
| `get()` | memtable `RwLock` (read) + `ArcSwap` segments | Concurrent reads |
| `search()` | memtable `RwLock` (read) + `ArcSwap` segments | Concurrent reads |
| `flush()` | memtable `RwLock` (write) + manifest `Mutex` | Blocks writes |
| `compact()` | Full synchronization | Blocks all operations |

Segments use `ArcSwap` for lock-free reads — when a flush or compaction creates new segments, the atomic pointer is swapped without blocking readers.

---

## Segment File Format

Each segment is a binary file with memory-mapped access:

```
┌──────────────────────────────────────────────┐
│  Header (64 bytes, aligned)                   │
│  ┌──────────────────────────────────────────┐ │
│  │ [4]   magic: "nvdb"                      │ │
│  │ [2]   version: u16                       │ │
│  │ [2]   reserved                           │ │
│  │ [4]   dimension: u32                     │ │
│  │ [8]   doc_count: u64                     │ │
│  │ [8]   vector_offset: u64                 │ │
│  │ [8]   id_mapping_offset: u64             │ │
│  │ [8]   payload_offset: u64                │ │
│  │ [8]   checksum: u64                      │ │
│  │ [8]   reserved                           │ │
│  └──────────────────────────────────────────┘ │
├──────────────────────────────────────────────┤
│  Vector Data (64-byte aligned)                │
│  ┌──────────────────────────────────────────┐ │
│  │  [dim × 4 bytes] vector[0]               │ │
│  │  [dim × 4 bytes] vector[1]               │ │
│  │  ...                                     │ │
│  └──────────────────────────────────────────┘ │
├──────────────────────────────────────────────┤
│  ID Mapping (internal ↔ external)             │
│  ┌──────────────────────────────────────────┐ │
│  │  [4] count: u32                          │ │
│  │  [count × (4 + len + 1)] entries         │ │
│  └──────────────────────────────────────────┘ │
├──────────────────────────────────────────────┤
│  Payloads (JSON, optional)                    │
│  ┌──────────────────────────────────────────┐ │
│  │  Length-prefixed JSON per document        │ │
│  └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

### Key Design Decisions

- **64-byte alignment** for vector data enables AVX-512 SIMD loads
- **Memory-mapped** reads mean zero-copy access to vectors
- **Internal IDs** (u32) used for compact graph storage; external IDs (String) mapped via ID mapping table
- **Immutable** — segments are never modified after creation

---

## WAL Format

The Write-Ahead Log ensures durability across crashes:

```
Record: [seq:u64][len:u32][crc32:u32][opcode:u8][body]

opcode 1 = Insert: [id_len:u32][id_bytes][dim:u32][vector_bytes][payload_len:u32][payload_bytes]
opcode 2 = Delete: [id_len:u32][id_bytes]
```

### Recovery

On startup, the WAL is replayed from `last_applied_seq + 1`:

1. Read each record, verify CRC32
2. Apply inserts/deletes to fresh memtable
3. Corrupt/partial records at tail are truncated
4. Update manifest with new `last_wal_seq`

### Auto-Flush

When the WAL reaches **64MB**, the memtable is automatically flushed to a new segment and the WAL is reset. This prevents unbounded WAL growth.

---

## HNSW Index

### Algorithm

Hierarchical Navigable Small World graphs provide sub-linear search complexity:

- **Layer 0**: All nodes, M neighbors each (dense)
- **Layer k>0**: Subset of nodes, M neighbors (sparse)
- **Search**: Start at top layer, greedily descend to layer 0
- **Construction**: Insert node at layers 0..l where l ~ Geometric(1/ln(M))

### CSR Storage

The graph uses Compressed Sparse Row format for cache efficiency:

```
neighbors: [n0_0, n0_1, ..., n1_0, n1_1, ..., nN_0, ...]
offsets:   [0, M, 2M, ..., N×M]

Node i's neighbors = neighbors[offsets[i]..offsets[i+1]]
```

This is 20-40% faster than pointer-based layouts due to better cache locality.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 16 | Max neighbors per node |
| `ef_construction` | 64 | Candidate pool during build |
| `ef_search` | 32 | Candidate pool during search |
| `level_factor` | 1/ln(M) | Probability of higher layers |

---

## Distance Metrics

All distance functions use **SIMD acceleration** via the `wide` crate:

| Metric | Formula | SIMD | Score Direction |
|--------|---------|------|-----------------|
| **Cosine** | dot(a,b) / (‖a‖·‖b‖) | `f32x8` (8-wide) | Higher = more similar |
| **Dot Product** | Σ a[i]·b[i] | `f32x8` (8-wide) | Higher = more similar |
| **Euclidean** | √Σ(a[i]-b[i])² | `f32x8` (8-wide) | Lower = more similar |

SIMD processes 8 floats per cycle, providing near-linear speedup on AVX2/AVX-512/NEON hardware.

---

## Compaction

Over time, segments accumulate deleted (tombstoned) documents. Compaction reclaims space:

1. Flush memtable to segment
2. Merge all segments into one
3. Remove deleted documents
4. Rebuild HNSW index (if exists)
5. Atomic manifest update
6. Reset WAL

Compaction is **crash-safe**: if interrupted, old segments remain valid via the previous manifest.

---

## File Layout

```
mydb/
├── MANIFEST                    # Database-level manifest (collection list)
├── embeddings/                 # Collection "embeddings"
│   ├── MANIFEST                # Collection manifest (segments, config, index)
│   ├── wal.log                 # Write-Ahead Log
│   ├── segments/
│   │   ├── 0001.nvdb           # Segment 1 (immutable, mmap)
│   │   ├── 0002.nvdb           # Segment 2
│   │   └── 0003.nvdb           # Segment 3
│   └── index.hnsw              # HNSW index (optional)
└── documents/                  # Collection "documents"
    ├── MANIFEST
    ├── wal.log
    ├── segments/
    │   └── 0001.nvdb
    └── index.hnsw
```

---

## Durability Levels

| Level | Behavior | Use Case |
|-------|----------|----------|
| `Buffered` (default) | Append to OS page cache, return | Fastest, caches/temp data |
| `FdatasyncEachBatch` | `fdatasync()` after each batch | Maximum safety, critical data |

```rust
let config = CollectionConfig::new(768)
    .with_durability(Durability::FdatasyncEachBatch);
```

---

## Document IDs

### Internal vs External

nVDB uses a dual-ID system:

- **External ID** (`String`): User-provided, e.g. `"doc1"`, `"user:alice"`
- **Internal ID** (`u32`): Auto-assigned, compact, used in segments and HNSW graph

The ID mapping table in each segment translates between them. Internal IDs enable:
- Compact HNSW graph storage (4 bytes vs 16+ bytes per reference)
- Aligned vector access by index
- Efficient SoA (Structure of Arrays) layout in memtable
