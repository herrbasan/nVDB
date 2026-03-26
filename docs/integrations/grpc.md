# nVDB gRPC Integration

> **Optional standalone service** - wraps the embedded library with a network interface
> **Version:** 0.1.0
> **Last Updated:** 2026-02-15

---

## Overview

nVDB is designed as an **embedded library** - the core library runs in-process with your application. The gRPC integration is an **optional wrapper** that exposes nVDB over a network when you need a client-server architecture.

| Use Case | Recommended Approach |
|----------|---------------------|
| Embedded in your app | Use nVDB directly (`nVDB` crate) |
| Python/Go/other languages | gRPC service wrapper |
| Multi-language team | gRPC service wrapper |
| Microservices | gRPC service wrapper |

### When to Use This

| Scenario | Solution |
|----------|----------|
| **Python app needs nVDB** | Run `nVDB-grpc.exe`, connect via gRPC client |
| **Multi-language team** | One binary, many clients |
| **Separate database server** | Run on dedicated machine/container |
| **Microservices** | Vector service independent of app logic |

### Architecture

```
┌─ Machine / Container ─────────────────────┐
│  ┌─────────────────────────────────────┐  │
│  │  nVDB-grpc (standalone executable)   │  │
│  │                                      │  │
│  │  gRPC Server :50051 ──▶ nVDB Engine  │  │
│  │                    (Collections,     │  │
│  │                     HNSW, mmap)      │  │
│  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
     ┌────┐        ┌────┐        ┌────┐
     │Node│        │Py  │        │Go  │
     └────┘        └────┘        └────┘
     Clients
```

---

## Quick Start

### 1. Download the Binary

```bash
# Linux x64
curl -LO https://github.com/nvdb/nvdb/releases/latest/download/nVDB-grpc-linux-x64.tar.gz
tar xzf nVDB-grpc-linux-x64.tar.gz

# Windows x64 (PowerShell)
Invoke-WebRequest https://github.com/nvdb/nvdb/releases/latest/download/nVDB-grpc-windows-x64.zip -OutFile nVDB-grpc.zip
Expand-Archive nVDB-grpc.zip -DestinationPath C:\nVDB

# macOS (Apple Silicon)
curl -LO https://github.com/nvdb/nvdb/releases/latest/download/nVDB-grpc-macos-arm64.tar.gz
tar xzf nVDB-grpc-macos-arm64.tar.gz
```

### 2. Run the Server

```bash
# Start with defaults (port 50051, data in ./data)
./nVDB-grpc

# Or specify options
./nVDB-grpc --data-dir /var/lib/nVDB --port 50051
```

### 3. Connect from Your Language

```javascript
// Node.js example
const client = new VectorService('localhost:50051', grpc.credentials.createInsecure());
const results = await client.search({ vector: query, topK: 10 });
```

---

## Installation

### Option 1: Pre-built Binary (Recommended)

Download from GitHub releases:

| Platform | Download |
|----------|----------|
| Linux x64 | `nVDB-grpc-linux-x64.tar.gz` |
| Linux ARM64 | `nVDB-grpc-linux-arm64.tar.gz` |
| Windows x64 | `nVDB-grpc-windows-x64.zip` |
| macOS x64 | `nVDB-grpc-macos-x64.tar.gz` |
| macOS ARM64 | `nVDB-grpc-macos-arm64.tar.gz` |

```bash
# Example: Linux install
wget https://github.com/nvdb/nvdb/releases/latest/download/nVDB-grpc-linux-x64.tar.gz
tar xzf nVDB-grpc-linux-x64.tar.gz
sudo mv nVDB-grpc /usr/local/bin/
sudo chmod +x /usr/local/bin/nVDB-grpc
```

### Option 2: Cargo Install

```bash
cargo install nVDB-grpc
```

### Option 3: Docker

```bash
docker run -p 50051:50051 -v $(pwd)/data:/data nVDB/nVDB-grpc:latest
```

---

## CLI Reference

### Basic Usage

```bash
nVDB-grpc [OPTIONS]
```

### Flags

| Flag | Environment Variable | Default | Description |
|------|---------------------|---------|-------------|
| `--data-dir` | `NDB_DATA_DIR` | `./data` | Database storage directory |
| `--port` | `NDB_PORT` | `50051` | gRPC server port |
| `--host` | `NDB_HOST` | `0.0.0.0` | Bind address (`127.0.0.1` for local only) |
| `--tls-cert` | `NDB_TLS_CERT` | - | TLS certificate file |
| `--tls-key` | `NDB_TLS_KEY` | - | TLS private key file |
| `--read-only` | `NDB_READ_ONLY` | `false` | Run in read-only mode |
| `--max-msg-size` | `NDB_MAX_MSG_SIZE` | `64MB` | Max gRPC message size |

### Examples

```bash
# Basic - local development
nVDB-grpc

# Production - specific directories
nVDB-grpc --data-dir /var/lib/nVDB --port 50051

# Local only (no external access)
nVDB-grpc --host 127.0.0.1

# With TLS
nVDB-grpc --tls-cert server.crt --tls-key server.key

# Read replica (read-only access)
nVDB-grpc --data-dir /shared/nVDB --read-only
```

---

## Running as a Service

### Windows Service

```powershell
# Download and extract to C:\nVDB
# Create data directory
New-Item -ItemType Directory -Force -Path C:\nVDB\data

# Install using nssm (https://nssm.cc/)
nssm install nVDB-GRPC C:\nVDB\nVDB-grpc.exe
nssm set nVDB-GRPC AppDirectory C:\nVDB
nssm set nVDB-GRPC AppParameters --data-dir C:\nVDB\data --host 127.0.0.1
nssm start nVDB-GRPC

# Check status
nssm status nVDB-GRPC
```

### Linux systemd

```ini
# /etc/systemd/system/nVDB-grpc.service
[Unit]
Description=nVDB gRPC Vector Database
After=network.target

[Service]
Type=simple
User=nVDB
Group=nVDB
WorkingDirectory=/var/lib/nVDB
ExecStart=/usr/local/bin/nVDB-grpc --data-dir /var/lib/nVDB --port 50051
Restart=always
RestartSec=5
Environment="RUST_LOG=info"

[Install]
WantedBy=multi-user.target
```

```bash
# Setup
sudo useradd -r -s /bin/false nVDB
sudo mkdir -p /var/lib/nVDB
sudo chown nVDB:nVDB /var/lib/nVDB
sudo cp nVDB-grpc /usr/local/bin/
sudo cp nVDB-grpc.service /etc/systemd/system/

# Start
sudo systemctl daemon-reload
sudo systemctl enable nVDB-grpc
sudo systemctl start nVDB-grpc
sudo systemctl status nVDB-grpc

# Logs
sudo journalctl -u nVDB-grpc -f
```

### macOS LaunchAgent

```xml
<!-- ~/Library/LaunchAgents/com.nVDB.grpc.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.nVDB.grpc</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/nVDB-grpc</string>
        <string>--data-dir</string>
        <string>/Users/yourname/nVDB-data</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.nVDB.grpc.plist
launchctl start com.nVDB.grpc
```

---

## Client Usage

### Protocol Buffer Definition

Save as `nVDB.proto`:

```protobuf
syntax = "proto3";
package nVDB;

service VectorService {
  rpc CreateCollection(CreateCollectionRequest) returns (Collection);
  rpc GetCollection(GetCollectionRequest) returns (Collection);
  rpc ListCollections(Empty) returns (CollectionList);
  
  rpc Insert(InsertRequest) returns (Document);
  rpc Get(GetRequest) returns (Document);
  rpc Delete(DeleteRequest) returns (DeleteResponse);
  
  rpc Search(SearchRequest) returns (SearchResponse);
  
  rpc Flush(CollectionRequest) returns (Empty);
  rpc Compact(CollectionRequest) returns (CompactionResult);
  rpc BuildIndex(BuildIndexRequest) returns (Empty);
}

message Empty {}

message Collection {
  string name = 1;
  uint32 dimension = 2;
  string durability = 3;
}

message CreateCollectionRequest {
  string name = 1;
  uint32 dimension = 2;
  string durability = 3;  // "buffered" or "sync"
}

message GetCollectionRequest {
  string name = 1;
}

message CollectionList {
  repeated Collection collections = 1;
}

message Document {
  string id = 1;
  repeated float vector = 2;
  string payload = 3;  // JSON string
}

message InsertRequest {
  string collection = 1;
  Document document = 2;
}

message GetRequest {
  string collection = 1;
  string id = 2;
}

message DeleteRequest {
  string collection = 1;
  string id = 2;
}

message DeleteResponse {
  bool deleted = 1;
}

message SearchRequest {
  string collection = 1;
  repeated float vector = 2;
  uint32 top_k = 3;
  string distance = 4;      // "cosine", "dot", "euclidean"
  bool approximate = 5;     // Use HNSW index
  uint32 ef = 6;            // HNSW quality parameter
  string filter = 7;        // JSON filter expression
}

message Match {
  string id = 1;
  float score = 2;
  string payload = 3;
}

message SearchResponse {
  repeated Match results = 1;
  uint32 total_candidates = 2;
}

message CollectionRequest {
  string name = 1;
}

message CompactionResult {
  uint32 docs_before = 1;
  uint32 docs_after = 2;
}

message BuildIndexRequest {
  string collection = 1;
}
```

### Node.js Client

```bash
npm install @grpc/grpc-js @grpc/proto-loader
```

```javascript
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const path = require('path');

// Load proto
const PROTO_PATH = path.join(__dirname, 'nVDB.proto');
const packageDef = protoLoader.loadSync(PROTO_PATH);
const ndbProto = grpc.loadPackageDefinition(packageDef).nVDB;

// Create client
const client = new ndbProto.VectorService(
  'localhost:50051',
  grpc.credentials.createInsecure()
);

// Promise wrapper
const call = (method) => (req) => new Promise((resolve, reject) => {
  client[method](req, (err, res) => err ? reject(err) : resolve(res));
});

// API
const api = {
  createCollection: call('createCollection'),
  insert: call('insert'),
  search: call('search'),
};

// Usage
async function main() {
  // Create collection
  await api.createCollection({
    name: 'documents',
    dimension: 768,
    durability: 'sync'
  });

  // Insert
  await api.insert({
    collection: 'documents',
    document: {
      id: 'doc1',
      vector: new Array(768).fill(0.1),
      payload: JSON.stringify({ title: 'Hello' })
    }
  });

  // Search
  const { results } = await api.search({
    collection: 'documents',
    vector: new Array(768).fill(0.15),
    topK: 10,
    approximate: true
  });

  console.log(results);
}

main().catch(console.error);
```

### Python Client

```bash
pip install grpcio grpcio-tools
```

```python
import grpc
import ndb_pb2
import ndb_pb2_grpc
import json

# Generate code from proto
# python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. nVDB.proto

# Connect
channel = grpc.insecure_channel('localhost:50051')
stub = ndb_pb2_grpc.VectorServiceStub(channel)

# Create collection
stub.CreateCollection(ndb_pb2.CreateCollectionRequest(
    name='documents',
    dimension=768,
    durability='sync'
))

# Insert
stub.Insert(ndb_pb2.InsertRequest(
    collection='documents',
    document=ndb_pb2.Document(
        id='doc1',
        vector=[0.1] * 768,
        payload=json.dumps({'title': 'Hello'})
    )
))

# Search
response = stub.Search(ndb_pb2.SearchRequest(
    collection='documents',
    vector=[0.15] * 768,
    top_k=10,
    approximate=True
))

for match in response.results:
    print(f"{match.id}: {match.score}")
```

### Go Client

```bash
go get google.golang.org/grpc
```

```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "google.golang.org/grpc"
    pb "your-module/nVDB"
)

func main() {
    conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()
    
    client := pb.NewVectorServiceClient(conn)
    ctx := context.Background()
    
    // Create collection
    client.CreateCollection(ctx, &pb.CreateCollectionRequest{
        Name:       "documents",
        Dimension:  768,
        Durability: "sync",
    })
    
    // Search
    res, err := client.Search(ctx, &pb.SearchRequest{
        Collection:  "documents",
        Vector:      make([]float32, 768),
        TopK:        10,
        Approximate: true,
    })
    if err != nil {
        log.Fatal(err)
    }
    
    for _, m := range res.Results {
        fmt.Printf("%s: %f\n", m.Id, m.Score)
    }
}
```

---

## Docker Deployment

### Quick Run

```bash
docker run -d \
  --name nVDB-grpc \
  -p 50051:50051 \
  -v $(pwd)/data:/data \
  -e NDB_DATA_DIR=/data \
  nVDB/nVDB-grpc:latest
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  nVDB:
    image: nVDB/nVDB-grpc:latest
    ports:
      - "50051:50051"
    volumes:
      - nVDB-data:/data
    environment:
      - NDB_DATA_DIR=/data
      - NDB_HOST=0.0.0.0
      - RUST_LOG=info
    restart: unless-stopped

volumes:
  nVDB-data:
```

```bash
docker-compose up -d
docker-compose logs -f
```

### Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nVDB-grpc
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nVDB-grpc
  template:
    metadata:
      labels:
        app: nVDB-grpc
    spec:
      containers:
      - name: nVDB
        image: nVDB/nVDB-grpc:latest
        ports:
        - containerPort: 50051
        volumeMounts:
        - name: data
          mountPath: /data
        env:
        - name: NDB_DATA_DIR
          value: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: nVDB-data
---
apiVersion: v1
kind: Service
metadata:
  name: nVDB-grpc
spec:
  selector:
    app: nVDB-grpc
  ports:
  - port: 50051
    targetPort: 50051
```

---

## Read Replica Setup

For scaling reads, run multiple instances with `--read-only`:

```
┌──────────────────────────────────────────┐
│           Writer (Primary)               │
│     nVDB-grpc --data-dir /shared          │
└──────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │    Shared Volume  │
         └─────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌────────┐    ┌────────┐    ┌────────┐
│ Reader │    │ Reader │    │ Reader │
│  :50051│    │  :50052│    │  :50053│
│ --read │    │ --read │    │ --read │
│ -only  │    │ -only  │    │ -only  │
└────────┘    └────────┘    └────────┘
```

```bash
# Primary (writes)
nVDB-grpc --data-dir /shared/nVDB --port 50051

# Readers (read-only)
nVDB-grpc --data-dir /shared/nVDB --port 50052 --read-only
nVDB-grpc --data-dir /shared/nVDB --port 50053 --read-only
```

**Note:** Only one writer allowed. Writers use file locking to prevent corruption.

---

## TLS / Security

### Generate Self-Signed Certificates

```bash
# Generate CA
openssl req -new -x509 -days 365 -keyout ca.key -out ca.crt

# Generate server cert
openssl req -new -keyout server.key -out server.csr
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -out server.crt -days 365
```

### Server with TLS

```bash
nVDB-grpc --tls-cert server.crt --tls-key server.key
```

### Client with TLS

```javascript
const fs = require('fs');
const credentials = grpc.credentials.createSsl(
  fs.readFileSync('ca.crt')
);
const client = new ndbProto.VectorService('server:50051', credentials);
```

---

## Troubleshooting

### Connection Refused

```bash
# Check server is running
grpcurl -plaintext localhost:50051 list

# Check firewall
nc -zv localhost 50051

# Check logs
journalctl -u nVDB-grpc -f
```

### Permission Denied

```bash
# Fix data directory permissions
sudo chown -R $(whoami):$(whoami) /var/lib/nVDB
```

### High Memory Usage

```bash
# Check collection stats
grpcurl -plaintext localhost:50051 nVDB.VectorService/GetStats

# Compact to reclaim space
grpcurl -plaintext localhost:50051 nVDB.VectorService/Compact
```

### Collection Locked

```bash
# Another instance has the database open
# Check for running processes
lsof /var/lib/nVDB/*/LOCK
```

---

## Building from Source

If you need to compile the binary yourself:

```bash
# Clone
git clone https://github.com/nvdb/nvdb
cd nVDB/nVDB-grpc

# Build release binary
cargo build --release

# Output: target/release/nVDB-grpc
```

### Cross-Compile

```bash
# Windows from Linux
cargo build --release --target x86_64-pc-windows-gnu

# ARM64
cargo build --release --target aarch64-unknown-linux-gnu
```

---

## License

MIT OR Apache-2.0 (same as nVDB)
