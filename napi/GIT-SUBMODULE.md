# Git Submodule Integration Guide

This guide explains how to integrate nVDB into your Node.js project using git submodules (without NPM).

## Quick Start

### 1. Add nVDB as a Submodule

```bash
cd your-project
git submodule add https://github.com/nvdb/nvdb.git nVDB
git submodule update --init --recursive
```

### 2. Build the Native Module

**Option A: Use the setup script (recommended)**

```bash
cd nVDB/napi
node setup.js
```

**Option B: Manual build**

```bash
cd nVDB

# Build the native module
cargo build --release -p nvdb-node

# Copy/link the binary (platform-specific)

# Windows:
copy target\release\nvdb_node.dll napi\nvdb-node.win32-x64-msvc.node

# macOS (Intel):
ln -s target/release/libnvdb_node.dylib napi/nvdb-node.darwin-x64.node

# macOS (Apple Silicon):
ln -s target/release/libnvdb_node.dylib napi/nvdb-node.darwin-arm64.node

# Linux (x64):
ln -s target/release/libnvdb_node.so napi/nvdb-node.linux-x64-gnu.node
```

### 3. Use in Your Project

```javascript
// In your project's JavaScript files
const { Database, FilterBuilder } = require('./nVDB/napi');

const db = new Database('./data');
const collection = db.createCollection('docs', 1536);

collection.insert('doc1', vector, JSON.stringify({ title: 'Hello' }));
const results = collection.search({ vector: query, topK: 10 });
```

## Project Structure

After setup, your project should look like:

```
your-project/
├── nVDB/                           # git submodule
│   ├── napi/
│   │   ├── index.js              # JS loader
│   │   ├── index.d.ts            # TypeScript types
│   │   ├── nvdb-node.XXX.node     # native binary (created by setup)
│   │   └── setup.js              # build helper
│   ├── src/                       # Rust source
│   └── target/release/            # Rust build output
├── your-app.js                    # your code
├── .gitmodules                    # git tracks submodules here
└── package.json                   # your project's package.json
```

## Environment Variable Override

If you want to use a different native binary location:

```bash
# Windows
set NODE_NVDB_NATIVE_PATH=D:\custom\path\nvdb_node.dll
node your-app.js

# macOS/Linux
export NODE_NVDB_NATIVE_PATH=/custom/path/libnvdb_node.so
node your-app.js
```

## Updating nVDB

To update to the latest version:

```bash
cd nVDB
git pull origin main
cd napi
node setup.js  # Rebuild native module
```

## CI/CD Integration

For automated builds, add this to your CI pipeline:

```yaml
# Example GitHub Actions step
- name: Build nVDB native module
  run: |
    git submodule update --init --recursive
    cd nVDB/napi
    node setup.js
```

## Troubleshooting

### "Native binary not found"

Run `node setup.js` in the `nVDB/napi` directory, or set `NODE_NVDB_NATIVE_PATH`.

### "Failed to load native module"

- Ensure you're using Node.js >= 16
- Check that the binary matches your platform/architecture
- On Windows, you may need Visual C++ Redistributables

### "Collection locked by another process"

Only one process can open a collection at a time. Close other Node.js processes using the database.

## Advanced: Multiple Native Versions

If you need to support multiple platforms, you can build all variants and commit them:

```bash
# In nVDB/napi - after building on each platform
git add nvdb-node.*.node
git commit -m "Add native binaries"
```

Then in your main project, the loader will automatically pick the correct one based on `process.platform` and `process.arch`.
