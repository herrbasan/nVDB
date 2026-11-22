# nVDB Scope and Boundaries

> **Scope note.** This page states the product boundary (no embedding generation) and is still current. The integration examples were written for the v1 embedded crate; they have been retargeted at the v2 engine served by `nvdb serve` (`HTTP/JSON` on `127.0.0.1:3510`). See [v2-server.md](../documentation/v2-server.md).
>
> **What nVDB is, what it isn't, and why**

---

## Core Scope

nVDB is a **derived retrieval index** — it is rebuilt from an authoritative document store and never owns the data. It does **one thing well**: **high-performance retrieval and ranking**. Production retrieval is the v2 engine (`src/v2/`) served over local HTTP by `nvdb serve`; the v1 crate (`Database`/`Collection`, LSM-Lite + HNSW) still exists and is tested, but is not the production path.

```
Your App                          nVDB
┌──────────────┐    ┌───────────────────────┐
│ Text/Data    │───▶│ Embedding Service     │──┐
│ (your job)   │    │ (your job)            │  │
└──────────────┘    └───────────────────────┘  │
                                                 ▼
                                         ┌───────────────────────┐
                                         │ nvdb serve            │
                                         │ BM25F + vectors + RRF │
                                         │ + cross-encoder rerank│
                                         └───────────────────────┘
```

### nVDB Handles:
- ✅ Lexical retrieval on Tantivy (BM25F, columnar filters)
- ✅ Dense vector storage keyed by content hash and CAS attachment
- ✅ Hybrid fusion (RRF) and Level-2 ONNX cross-encoder reranking
- ✅ ACL overlay enforcement and sync watermark/integrity checks
- ✅ Multi-collection management under one `--dir`
- ✅ Persistence and crash recovery

### nVDB Does NOT Handle:
- ❌ Text embedding generation
- ❌ Model management
- ❌ Tokenization
- ❌ External API calls (OpenAI, etc.)
- ❌ Image/audio processing

---

## Why This Boundary?

### 1. Separation of Concerns

Embedding is a **different problem** with different constraints:

| Aspect | Embedding | Vector Search |
|--------|-----------|---------------|
| **Dependencies** | Heavy (models, APIs) | None (self-contained) |
| **Latency** | 100ms-10s (API call) | <1ms (local) |
| **Failure modes** | Network, rate limits, costs | Local disk, memory |
| **Updates** | Model versions, API changes | Stable file format |

### 2. Existing Solutions Are Excellent

Don't reinvent what already works:

| Use Case | Recommended Solution |
|----------|---------------------|
| Cloud API embeddings | OpenAI, Cohere, Voyage AI |
| Local embeddings | Ollama, sentence-transformers |
| Self-hosted models | HuggingFace, text-embeddings-inference |
| Multi-modal | CLIP (OpenAI), ImageBind (Meta) |

### 3. Zero-Cost Abstractions

Per nVDB philosophy: **"Pay only for what you use"**.

Embedding integration would add:
- 100MB-2GB of model files (optional but confusing)
- Network dependency for cloud APIs
- Configuration complexity (API keys, rate limits)
- Version management (model updates)

**Current nVDB binary: ~5MB**  
**With embedding models: 500MB-2GB**

---

## Integration Patterns

All examples call the v2 HTTP API of a running `nvdb serve` (`http://127.0.0.1:3510`). The caller generates vectors; nVDB stores and ranks them.

### Pattern 1: Cloud API (OpenAI, etc.)

```javascript
const BASE = 'http://127.0.0.1:3510';
const OpenAI = require('openai');

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Create the collection once (dim/model_version must match the embeddings).
await fetch(`${BASE}/v2/collections`, {
  method: 'POST',
  body: JSON.stringify({
    name: 'docs',
    config: { vector_lane: { dim: 1536, model_version: 'text-embedding-3-small' } },
  }),
});

async function ingestDocument(id, text, metadata) {
  // 1. Generate embedding (your responsibility)
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  });
  const vector = response.data[0].embedding;

  // 2. Store text + metadata; the dense lane stays pending until the vector is attached
  await fetch(`${BASE}/v2/collections/docs/upsert`, {
    method: 'POST',
    body: JSON.stringify({
      docs: [{ id, kind: 'doc', title: metadata.title ?? id, body: text, acl: metadata.acl ?? ['public'] }],
    }),
  });

  // 3. Attach the vector out of band with a compare-and-set
  const pending = await (await fetch(`${BASE}/v2/collections/docs/pending_embeddings?limit=100`)).json();
  const item = pending.items.find((p) => p.id === id);
  await fetch(`${BASE}/v2/collections/docs/attach_vectors`, {
    method: 'POST',
    body: JSON.stringify({
      attachments: [{ id, content_hash: item.content_hash, model_version: 'text-embedding-3-small', vector }],
    }),
  });
}
```

### Pattern 2: Local Embeddings (Ollama)

Same flow with a local embedder — replace the first half of `ingestDocument`:

```javascript
const ollama = require('ollama');

// Local embedding - no network, no API keys
const response = await ollama.embeddings({ model: 'nomic-embed-text', prompt: text });
const vector = response.embedding;
```

### Pattern 3: Batch Processing

```javascript
async function ingestBatch(documents) {
  // Batch embedding for efficiency
  const vectors = await embedBatch(documents.map((d) => d.text), 100);

  // One upsert for the whole batch; vectors attach afterwards against content_hash
  await fetch(`${BASE}/v2/collections/docs/upsert`, {
    method: 'POST',
    body: JSON.stringify({
      docs: documents.map((d) => ({ id: d.id, kind: 'doc', title: d.id, body: d.text, acl: d.metadata?.acl ?? ['public'] })),
    }),
  });

  const pending = await (await fetch(`${BASE}/v2/collections/docs/pending_embeddings?limit=1000`)).json();
  const byId = new Map(pending.items.map((p) => [p.id, p.content_hash]));
  await fetch(`${BASE}/v2/collections/docs/attach_vectors`, {
    method: 'POST',
    body: JSON.stringify({
      attachments: documents.map((d, i) => ({
        id: d.id,
        content_hash: byId.get(d.id),
        model_version: 'nomic-embed-text',
        vector: vectors[i],
      })),
    }),
  });
}
```

---

## Optional Helpers (Future)

Helper packages for common embedding-provider patterns are **not part of this repository** and none is published today. Any such helper would be a separate project, community-maintained or example-only, and not required for nVDB functionality.

---

## FAQ

### "But Pinecone/Weaviate/Chroma handle embedding..."

Those are **managed services** with different constraints:
- They control the infrastructure
- They charge per embedding + storage + search
- They can update models centrally

nVDB is **software you run yourself**:
- You control the infrastructure
- You choose your embedding provider
- You pay only for what you use (no nVDB fees)

### "I want a one-line ingest experience"

Use a managed service if that's your priority. nVDB trades convenience for:
- **Performance** (local process, loopback HTTP — no cloud hop)
- **Control** (choose any embedding model)
- **Cost** (no per-request fees)
- **Privacy** (data stays on your machine)

### "What about local models?"

If you need offline embeddings, run Ollama or similar alongside nVDB and point your ingestion code at both: Ollama produces the vectors, `nvdb serve` takes them via `attach_vectors`.

```bash
# Ollama on :11434, nVDB on :3510 — two local processes, no cloud dependency
nvdb serve --dir ./nvdb-data --host 127.0.0.1 --port 3510 \
  --reranker-model ./models/bge-reranker
```

---

## Summary

| Question | Answer |
|----------|--------|
| Does nVDB generate embeddings? | **No** |
| Will it ever? | **Not in core** (optional helpers maybe) |
| Why? | Separation of concerns, existing solutions are excellent |
| What should I use? | OpenAI, Ollama, HuggingFace, or any embedding service |
| How do I integrate? | Generate embeddings → Pass vectors to nVDB |

**nVDB is a vector database, not an embedding service.**

Use the best tool for each job.
