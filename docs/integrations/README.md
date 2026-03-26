# nVDB Integrations

This directory contains integration guides for using nVDB in different environments and languages.

## Core Library vs Integrations

**nVDB core** is an **embedded library** - it runs in-process with your Rust application. The integrations below are optional ways to use nVDB from other languages or environments:

| Integration | Best For | Performance | File System |
|-------------|----------|-------------|-------------|
| **[Rust](./user-guide.md)** | Embedded in your app | Native speed | Full (mmap) |
| **[N-API](./napi.md)** | Node.js apps | Native speed | Full (mmap) |
| **[WebAssembly](./wasm.md)** | Browser/Edge | ~80-90% native | In-memory only |
| **[gRPC](./grpc.md)** | Multi-language, network access | Network overhead | Full (service) |

## Quick Selection Guide

```
What is your deployment environment?

  ┌─────────────┐
  │  Rust app?  │───yes──▶ nVDB crate (see User Guide)
  └──────┬──────┘
         │no
  ┌──────▼──────┐
  │Node.js srv? │───yes──▶ N-API
  └──────┬──────┘
         │no
  ┌──────▼──────┐
  │Browser/Edge?│───yes──▶ WebAssembly
  └──────┬──────┘
         │no
  ┌──────▼──────┐
  │   Other?     │───yes──▶ gRPC service
  └─────────────┘
```

## Installation

### N-API (Node.js)

```bash
npm install nvdb-node
```

### WebAssembly

```bash
npm install nVDB-wasm
```

### gRPC Client (Node.js)

```bash
npm install @grpc/grpc-js @grpc/proto-loader nVDB-grpc-client
```

## Feature Comparison

| Feature | N-API | WASM | gRPC |
|---------|-------|------|------|
| **Performance** | Native | ~80-90% | Network overhead |
| **Persistence** | Memory-mapped files | In-memory / manual | Memory-mapped files |
| **SIMD** | AVX2/AVX-512 | SIMD128 | AVX2/AVX-512 (server) |
| **Threading** | Multi-thread | Single-thread | Multi-thread |
| **Bundle Size** | ~5MB (platform-specific) | ~2MB (universal) | ~500KB (client) |
| **TypeScript** | Optional definitions | Optional definitions | Generated from proto |
| **Browser** | ❌ | ✅ | ✅ (via HTTP/2) |
| **Edge Functions** | ❌ | ✅ | ✅ |
| **Multi-language** | ❌ (Node.js only) | ✅ (any WASM runtime) | ✅ (any gRPC client) |

## Code Examples

### N-API

```javascript
const { Database } = require('nvdb-node');

const db = new Database('./data');
const coll = db.createCollection('docs', 768);

coll.insert('id', vector, JSON.stringify(payload));
const results = coll.search({ vector: query, topK: 10 });
```

### WebAssembly

```javascript
import init, { WasmDB } from 'nVDB-wasm';

await init();
const db = new WasmDB();
const coll = db.createCollection('docs', 768);

coll.insert('id', new Float32Array(vector), JSON.stringify(payload));
const results = coll.search(new Float32Array(query), 10);
```

### gRPC

```javascript
const { VectorServiceClient } = require('nVDB-grpc-client');

const client = new VectorServiceClient('localhost:50051');

await client.insert({ collection: 'docs', document: { id, vector, payload } });
const { results } = await client.search({ collection: 'docs', vector: query, topK: 10 });
```

## TypeScript Support

All three integrations provide TypeScript definitions:

- **N-API**: JSDoc annotations + optional `@types/nvdb-node`
- **WASM**: JSDoc annotations + optional type definitions
- **gRPC**: Auto-generated from `.proto` files

All examples use **plain JavaScript** - TypeScript is completely optional.

## Getting Help

- **N-API issues**: See [N-API troubleshooting](./napi.md#troubleshooting)
- **WASM issues**: See [WASM troubleshooting](./wasm.md#troubleshooting)
- **gRPC issues**: See [gRPC troubleshooting](./grpc.md#troubleshooting)
- **General issues**: https://github.com/nvdb/nvdb/issues
