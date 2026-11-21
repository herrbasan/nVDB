//! Content-hash-keyed vector lane (`vectors.bin` + `vec_map.gen_<N>.bin`).
//!
//! The vector lane is keyed by **content**, never by document id. Two documents
//! with identical searchable text share one embedding, and an ACL-only or
//! metadata-only upsert keeps an existing embedding because
//! [`crate::sync::compute_content_hash`] covers the searchable text alone.
//!
//! Two files, two very different lifecycles:
//!
//! - [`VectorStore`] (`vectors.bin`) is content-addressed and *append-mostly*:
//!   `(content_hash, model_version) -> Vec<f32>`, packed little-endian with a
//!   magic/version header, rewritten atomically through `.tmp` + `sync_all` +
//!   rename. Unreferenced entries are dropped by
//!   [`VectorStore::retain_referenced`] during a rebuild, never during a write.
//! - [`VecMap`] (`vec_map.gen_<N>.bin`) is the mutable `doc_id -> content_hash`
//!   link. It is generation-stamped exactly like the ACL overlay: a new
//!   generation file is written first, `V2Manifest::vec_generation` is swapped
//!   next, and only then are superseded generations deleted. Immutable segments
//!   are never mutated by a vector attach.
//!
//! Pending embeddings are *derived* — `live ∧ ¬has_vector` — so there is no
//! durable work queue that could drift from the index.

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::v2::engine::write_atomic;
use crate::v2::V2Error;

/// Vector store file name inside the collection directory.
pub const VECTORS_FILE: &str = "vectors.bin";

/// Prefix of every persisted `doc_id -> content_hash` generation.
pub const VEC_MAP_PREFIX: &str = "vec_map.gen_";

/// Suffix of every persisted `doc_id -> content_hash` generation.
pub const VEC_MAP_SUFFIX: &str = ".bin";

/// Magic bytes of `vectors.bin`.
const VECTORS_MAGIC: &[u8; 4] = b"NVVC";

/// Legacy fixed-count format version (read-compatible with existing files).
const VECTORS_FORMAT_VERSION_V1: u8 = 1;

/// Append-only format version (`magic | 2 | [record]*` until EOF).
const VECTORS_FORMAT_VERSION_V2: u8 = 2;

/// Magic bytes of a persisted vec-map generation.
const VEC_MAP_MAGIC: &[u8; 4] = b"NVMV";

/// Format version of a persisted vec-map generation.
const VEC_MAP_FORMAT_VERSION: u8 = 1;

/// A raw BLAKE3-256 content digest.
pub type ContentDigest = [u8; 32];

/// Key of one stored embedding: the content digest plus the model that made it.
///
/// The model version is part of the key so a model upgrade can attach new
/// embeddings before the old ones are garbage-collected.
type VectorKey = (ContentDigest, String);

/// Content-addressed vector store.
#[derive(Debug, Clone, Default)]
pub struct VectorStore {
    entries: HashMap<VectorKey, Vec<f32>>,
}

impl VectorStore {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored embeddings.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no embedding is stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Stored embedding for `(digest, model_version)`.
    pub fn get(&self, digest: &ContentDigest, model_version: &str) -> Option<&[f32]> {
        self.entries
            .get(&(*digest, model_version.to_string()))
            .map(Vec::as_slice)
    }

    /// True when `(digest, model_version)` holds an embedding.
    pub fn contains(&self, digest: &ContentDigest, model_version: &str) -> bool {
        self.entries
            .contains_key(&(*digest, model_version.to_string()))
    }

    /// Store an embedding. Identical `(digest, model_version)` keys overwrite,
    /// which is the deduplication path: two documents with the same text cost
    /// exactly one embedding.
    pub fn insert(&mut self, digest: ContentDigest, model_version: &str, vector: Vec<f32>) {
        self.entries
            .insert((digest, model_version.to_string()), vector);
    }

    /// Every stored key, for garbage collection.
    pub fn keys(&self) -> impl Iterator<Item = &VectorKey> + '_ {
        self.entries.keys()
    }

    /// Drop every embedding that no `doc_id -> content_hash` link references.
    ///
    /// Returns the number of dropped embeddings. Runs on rebuild/compaction, so
    /// a write never pays for it.
    pub fn retain_referenced(&mut self, referenced: &HashSet<VectorKey>) -> usize {
        let before = self.entries.len();
        self.entries.retain(|key, _| referenced.contains(key));
        before - self.entries.len()
    }

    /// Load `vectors.bin` from `dir`; a missing file is an empty store.
    pub fn load(dir: &Path) -> Result<Self, V2Error> {
        let path = dir.join(VECTORS_FILE);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Self::new()),
            Err(error) => return Err(V2Error::Io(error)),
        };
        Self::from_bytes(&bytes)
    }

    /// Persist the entire store atomically as a compacted v2 `vectors.bin`.
    pub fn save_atomic(&self, dir: &Path) -> Result<(), V2Error> {
        write_atomic(&dir.join(VECTORS_FILE), &self.to_bytes())
    }

    /// Append newly inserted `(digest, model_version)` records to `vectors.bin`
    /// in `O(batch)` I/O instead of rewriting the full file (`herrbasan/nVDB#3`).
    ///
    /// - Missing file: creates a fresh v2 file with header + records + `sync_all`.
    /// - Existing v2 file: appends only the new records + `sync_all`.
    /// - Legacy v1 file: upgrades once via `save_atomic` (v2); subsequent calls append.
    pub fn append_records(
        &self,
        dir: &Path,
        keys: &[(ContentDigest, String)],
    ) -> Result<(), V2Error> {
        if keys.is_empty() {
            return Ok(());
        }
        let path = dir.join(VECTORS_FILE);
        let existing_version = match std::fs::File::open(&path) {
            Ok(mut file) => {
                let mut header = [0u8; 5];
                match file.read_exact(&mut header) {
                    Ok(()) => {
                        if &header[..4] != VECTORS_MAGIC {
                            return Err(corrupt("vectors.bin has bad magic bytes"));
                        }
                        Some(header[4])
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                        return Err(corrupt("vectors.bin is truncated before its header"));
                    }
                    Err(error) => return Err(V2Error::Io(error)),
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(V2Error::Io(error)),
        };

        match existing_version {
            Some(VECTORS_FORMAT_VERSION_V1) => {
                // One-time atomic upgrade from v1 (fixed count) to v2 (append-only).
                return self.save_atomic(dir);
            }
            Some(VECTORS_FORMAT_VERSION_V2) => {
                let mut buf = Vec::with_capacity(keys.len() * 128);
                for (digest, model_version) in keys {
                    let vector = self
                        .entries
                        .get(&(*digest, model_version.clone()))
                        .ok_or_else(|| corrupt("append_records called for key missing from VectorStore"))?;
                    encode_record(&mut buf, digest, model_version, vector);
                }
                let mut file = OpenOptions::new().append(true).open(&path)?;
                file.write_all(&buf)?;
                file.sync_all()?;
                Ok(())
            }
            None => {
                let mut buf = Vec::with_capacity(5 + keys.len() * 128);
                buf.extend_from_slice(VECTORS_MAGIC);
                buf.push(VECTORS_FORMAT_VERSION_V2);
                for (digest, model_version) in keys {
                    let vector = self
                        .entries
                        .get(&(*digest, model_version.clone()))
                        .ok_or_else(|| corrupt("append_records called for key missing from VectorStore"))?;
                    encode_record(&mut buf, digest, model_version, vector);
                }
                write_atomic(&path, &buf)
            }
            Some(other) => Err(corrupt(&format!(
                "vectors.bin uses unsupported version {other}"
            ))),
        }
    }

    /// Serialise in v2 append-only format: `magic | 2 | (digest, model, dim, f32 × dim)*`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::with_capacity(5 + self.entries.len() * 128);
        bytes.extend_from_slice(VECTORS_MAGIC);
        bytes.push(VECTORS_FORMAT_VERSION_V2);
        for ((digest, model_version), vector) in &self.entries {
            encode_record(&mut bytes, digest, model_version, vector);
        }
        bytes
    }

    /// Parse bytes written in v1 (fixed-count) or v2 (append-only) format.
    ///
    /// Every invariant the read path relies on is enforced here — a truncated
    /// file, an unknown version, a zero dimension, a duplicate key or a
    /// non-finite value is corruption and fails at open instead of producing a
    /// silently wrong ranking later.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, V2Error> {
        let mut reader = ByteReader::new(bytes);
        let magic = reader
            .take(VECTORS_MAGIC.len())
            .ok_or_else(|| corrupt("vectors.bin is truncated before its magic"))?;
        if magic != VECTORS_MAGIC {
            return Err(corrupt("vectors.bin has bad magic bytes"));
        }
        let version = reader
            .u8()
            .ok_or_else(|| corrupt("vectors.bin has no format version"))?;

        match version {
            VECTORS_FORMAT_VERSION_V1 => {
                let count = reader
                    .u32()
                    .ok_or_else(|| corrupt("vectors.bin has no count"))?
                    as usize;
                let mut entries: HashMap<VectorKey, Vec<f32>> = HashMap::with_capacity(count);
                for _ in 0..count {
                    let (key, vector) = decode_record(&mut reader)?;
                    if entries.insert(key, vector).is_some() {
                        return Err(corrupt(
                            "vectors.bin holds the same (content_hash, model_version) twice",
                        ));
                    }
                }
                Ok(Self { entries })
            }
            VECTORS_FORMAT_VERSION_V2 => {
                let mut entries: HashMap<VectorKey, Vec<f32>> = HashMap::new();
                while reader.remaining() > 0 {
                    let (key, vector) = decode_record(&mut reader)?;
                    if entries.insert(key, vector).is_some() {
                        return Err(corrupt(
                            "vectors.bin holds the same (content_hash, model_version) twice",
                        ));
                    }
                }
                Ok(Self { entries })
            }
            other => Err(corrupt(&format!(
                "vectors.bin uses unsupported version {other}"
            ))),
        }
    }
}

fn encode_record(bytes: &mut Vec<u8>, digest: &ContentDigest, model_version: &str, vector: &[f32]) {
    bytes.extend_from_slice(digest);
    push_str(bytes, model_version);
    bytes.extend_from_slice(&(vector.len() as u32).to_le_bytes());
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn decode_record(reader: &mut ByteReader<'_>) -> Result<(VectorKey, Vec<f32>), V2Error> {
    let digest = reader.digest()?;
    let model_version = reader
        .string()
        .ok_or_else(|| corrupt("vectors.bin has a truncated model version"))?;
    let dim = reader
        .u32()
        .ok_or_else(|| corrupt("vectors.bin has a truncated dimension"))? as usize;
    if dim == 0 {
        return Err(corrupt("vectors.bin holds a zero-dimension embedding"));
    }
    let mut vector: Vec<f32> = Vec::with_capacity(dim);
    for _ in 0..dim {
        let value = reader
            .f32()
            .ok_or_else(|| corrupt("vectors.bin has a truncated embedding"))?;
        if !value.is_finite() {
            return Err(corrupt("vectors.bin holds a non-finite value"));
        }
        vector.push(value);
    }
    Ok(((digest, model_version), vector))
}

/// Generation-stamped `doc_id -> content_hash` link map.
#[derive(Debug, Clone, Default)]
pub struct VecMap {
    generation: u64,
    docs: HashMap<String, ContentDigest>,
}

impl VecMap {
    /// Empty map at generation `0`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current generation.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Number of linked documents.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// True when no document is linked.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Content digest linked to `doc_id`, if any.
    pub fn get(&self, doc_id: &str) -> Option<&ContentDigest> {
        self.docs.get(doc_id)
    }

    /// Every `(doc_id, content digest)` link.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ContentDigest)> + '_ {
        self.docs.iter()
    }

    /// Link `doc_id` to `digest`, bumping the generation only on a real change.
    ///
    /// Returns whether anything changed, so the caller can skip a rewrite.
    pub fn insert(&mut self, doc_id: &str, digest: ContentDigest) -> bool {
        if self.docs.get(doc_id) == Some(&digest) {
            return false;
        }
        self.docs.insert(doc_id.to_string(), digest);
        self.generation += 1;
        true
    }

    /// Drop one link, bumping the generation when it was present.
    pub fn remove(&mut self, doc_id: &str) -> bool {
        if self.docs.remove(doc_id).is_some() {
            self.generation += 1;
            true
        } else {
            false
        }
    }

    /// Drop every link for which `keep` returns false.
    ///
    /// The generation bumps once when anything was dropped, not once per link.
    pub fn retain(&mut self, keep: impl Fn(&str, &ContentDigest) -> bool) -> bool {
        let before = self.docs.len();
        self.docs
            .retain(|doc_id, digest| keep(doc_id.as_str(), digest));
        if self.docs.len() == before {
            return false;
        }
        self.generation = self.generation.saturating_add(1);
        true
    }

    /// Replace every link (used by a clean rebuild).
    pub fn replace_all(&mut self, docs: HashMap<String, ContentDigest>) -> bool {
        if self.docs == docs {
            return false;
        }
        self.docs = docs;
        self.generation = self.generation.saturating_add(1);
        true
    }

    /// Serialise as generation `generation`.
    pub fn to_bytes(&self, generation: u64) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::with_capacity(21 + self.docs.len() * 108);
        bytes.extend_from_slice(VEC_MAP_MAGIC);
        bytes.push(VEC_MAP_FORMAT_VERSION);
        bytes.extend_from_slice(&generation.to_le_bytes());
        bytes.extend_from_slice(&(self.docs.len() as u32).to_le_bytes());
        for (doc_id, digest) in &self.docs {
            push_str(&mut bytes, doc_id);
            bytes.extend_from_slice(digest);
        }
        bytes
    }

    /// Parse a generation written by [`VecMap::to_bytes`].
    pub fn from_bytes(bytes: &[u8], generation: u64) -> Result<Self, V2Error> {
        let mut reader = ByteReader::new(bytes);
        let magic = reader
            .take(VEC_MAP_MAGIC.len())
            .ok_or_else(|| vec_map_corrupt(generation, "truncated magic"))?;
        if magic != VEC_MAP_MAGIC {
            return Err(vec_map_corrupt(generation, "bad magic bytes"));
        }
        let version = reader
            .u8()
            .ok_or_else(|| vec_map_corrupt(generation, "truncated format version"))?;
        if version != VEC_MAP_FORMAT_VERSION {
            return Err(vec_map_corrupt(
                generation,
                &format!("unsupported format version {version}"),
            ));
        }
        let written_generation = reader
            .u64()
            .ok_or_else(|| vec_map_corrupt(generation, "truncated generation"))?;
        let count = reader
            .u32()
            .ok_or_else(|| vec_map_corrupt(generation, "truncated document count"))?
            as usize;
        let mut docs: HashMap<String, ContentDigest> = HashMap::with_capacity(count);
        for _ in 0..count {
            let doc_id = reader
                .string()
                .ok_or_else(|| vec_map_corrupt(generation, "truncated document id"))?;
            let digest = reader
                .digest_at(generation)
                .map_err(|detail| vec_map_corrupt(generation, &detail))?;
            docs.insert(doc_id, digest);
        }
        Ok(Self {
            generation: written_generation,
            docs,
        })
    }

    /// Write generation `generation` atomically into `dir`.
    ///
    /// Superseded generations are **not** deleted here; call
    /// [`VecMap::cleanup_superseded`] after the manifest swap.
    pub fn save_atomic(&mut self, dir: &Path, generation: u64) -> Result<(), V2Error> {
        self.generation = generation;
        write_atomic(&vec_map_path(dir, generation), &self.to_bytes(generation))
    }

    /// Load the generation the manifest points at.
    ///
    /// Generation `0` with no file is the normal state of a collection that has
    /// never had a vector attached; a generation above `0` without its file is
    /// corruption.
    pub fn load(dir: &Path, generation: u64) -> Result<Self, V2Error> {
        let path = vec_map_path(dir, generation);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if generation == 0 {
                    return Ok(Self::new());
                }
                return Err(V2Error::Corrupt(format!(
                    "vector map generation {generation} is missing at {}",
                    path.display()
                )));
            }
            Err(error) => return Err(V2Error::Io(error)),
        };
        Self::from_bytes(&bytes, generation)
    }

    /// Delete every persisted generation except `keep_generation`.
    ///
    /// Runs after the manifest has been swapped, so the live generation is
    /// never the one being removed.
    pub fn cleanup_superseded(dir: &Path, keep_generation: u64) -> Result<(), V2Error> {
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(V2Error::Io(error)),
        };
        let keep = vec_map_path(dir, keep_generation);
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path == keep {
                continue;
            }
            let name = match path.file_name().and_then(|name| name.to_str()) {
                Some(name) => name,
                None => continue,
            };
            if name.starts_with(VEC_MAP_PREFIX) && name.ends_with(VEC_MAP_SUFFIX) {
                std::fs::remove_file(&path)?;
            }
        }
        Ok(())
    }
}

/// Path of one persisted vec-map generation.
pub fn vec_map_path(dir: &Path, generation: u64) -> PathBuf {
    dir.join(format!("{VEC_MAP_PREFIX}{generation}{VEC_MAP_SUFFIX}"))
}

/// Lowercase hex form of a content digest.
///
/// [`crate::sync::compute_content_hash`] returns exactly this, so the wire form
/// of a content hash is hex even though the sidecars keep raw bytes.
pub fn digest_to_hex(digest: &ContentDigest) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Parse a 64-character hex content hash into its raw digest.
pub fn hex_to_digest(hex: &str) -> Option<ContentDigest> {
    let hex = hex.trim();
    if hex.len() != 64 {
        return None;
    }
    let bytes = hex.as_bytes();
    let mut digest: ContentDigest = [0u8; 32];
    for (index, chunk) in bytes.chunks_exact(2).enumerate() {
        let high = hex_nibble(chunk[0])?;
        let low = hex_nibble(chunk[1])?;
        digest[index] = (high << 4) | low;
    }
    Some(digest)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn corrupt(detail: &str) -> V2Error {
    V2Error::Corrupt(detail.to_string())
}

fn vec_map_corrupt(generation: u64, detail: &str) -> V2Error {
    V2Error::Corrupt(format!("vector map generation {generation}: {detail}"))
}

fn push_str(bytes: &mut Vec<u8>, text: &str) {
    bytes.extend_from_slice(&(text.len() as u32).to_le_bytes());
    bytes.extend_from_slice(text.as_bytes());
}

/// Bounds-checked cursor over a persisted sidecar.
struct ByteReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }

    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.position.checked_add(count)?;
        let slice = self.bytes.get(self.position..end)?;
        self.position = end;
        Some(slice)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u32(&mut self) -> Option<u32> {
        let raw: [u8; 4] = self.take(4)?.try_into().ok()?;
        Some(u32::from_le_bytes(raw))
    }

    fn u64(&mut self) -> Option<u64> {
        let raw: [u8; 8] = self.take(8)?.try_into().ok()?;
        Some(u64::from_le_bytes(raw))
    }

    fn f32(&mut self) -> Option<f32> {
        let raw: [u8; 4] = self.take(4)?.try_into().ok()?;
        Some(f32::from_le_bytes(raw))
    }

    fn string(&mut self) -> Option<String> {
        let len = self.u32()? as usize;
        let raw = self.take(len)?;
        String::from_utf8(raw.to_vec()).ok()
    }

    fn digest(&mut self) -> Result<ContentDigest, V2Error> {
        let raw = self
            .take(32)
            .ok_or_else(|| corrupt("vectors.bin has a truncated content hash"))?;
        raw.try_into()
            .map_err(|_| corrupt("vectors.bin has a malformed content hash"))
    }

    fn digest_at(&mut self, generation: u64) -> Result<ContentDigest, String> {
        let raw = self
            .take(32)
            .ok_or_else(|| "truncated content hash".to_string())?;
        raw.try_into()
            .map_err(|_| format!("generation {generation} has a malformed content hash"))
    }
}

/// A live document that still needs an embedding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingEmbedding {
    /// Document identifier.
    pub id: String,
    /// Hex content hash of the searchable text the embedding must cover.
    pub content_hash: String,
    /// Stored title, so the embedding host does not need a second lookup.
    pub title: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_round_trips_and_deduplicates_by_content_and_model() {
        let digest = [7u8; 32];
        let mut store = VectorStore::new();
        store.insert(digest, "m1", vec![1.0, 0.0, 0.5]);
        assert_eq!(store.len(), 1);
        store.insert(digest, "m1", vec![1.0, 0.0, 0.5]);
        assert_eq!(store.len(), 1, "the same key must not grow the store");
        store.insert(digest, "m2", vec![0.0, 1.0, 0.0]);
        assert_eq!(store.len(), 2, "a model upgrade is a new entry");

        let parsed = VectorStore::from_bytes(&store.to_bytes()).expect("round trip");
        assert_eq!(parsed.get(&digest, "m1"), Some(&[1.0f32, 0.0, 0.5][..]));
        assert_eq!(parsed.get(&digest, "m2"), Some(&[0.0f32, 1.0, 0.0][..]));
        assert_eq!(parsed.get(&digest, "absent"), None);
    }

    #[test]
    fn store_rejects_truncated_and_non_finite_input() {
        let mut store = VectorStore::new();
        store.insert([1u8; 32], "m", vec![1.0, 2.0]);
        let bytes = store.to_bytes();
        assert!(VectorStore::from_bytes(&bytes[..bytes.len() - 3]).is_err());

        let mut poisoned = VectorStore::new();
        poisoned.insert([2u8; 32], "m", vec![f32::NAN]);
        assert!(VectorStore::from_bytes(&poisoned.to_bytes()).is_err());
    }

    #[test]
    fn gc_drops_every_unreferenced_embedding() {
        let mut store = VectorStore::new();
        store.insert([1u8; 32], "m", vec![1.0]);
        store.insert([2u8; 32], "m", vec![2.0]);
        let mut referenced: HashSet<VectorKey> = HashSet::new();
        referenced.insert(([1u8; 32], "m".to_string()));
        assert_eq!(store.retain_referenced(&referenced), 1);
        assert!(store.contains(&[1u8; 32], "m"));
        assert!(!store.contains(&[2u8; 32], "m"));
    }

    #[test]
    fn vec_map_bumps_generation_only_on_real_change() {
        let mut map = VecMap::new();
        assert!(map.insert("a", [1u8; 32]), "the first link is a change");
        let first = map.generation();
        assert!(!map.insert("a", [1u8; 32]), "the same link is a no-op");
        assert_eq!(map.generation(), first);
        assert!(map.insert("a", [2u8; 32]));
        assert!(map.generation() > first);
        assert!(map.remove("a"));
        assert!(!map.remove("a"));

        map.insert("b", [3u8; 32]);
        let generation = map.generation();
        let parsed = VecMap::from_bytes(&map.to_bytes(generation), generation).expect("round trip");
        assert_eq!(parsed.get("b"), Some(&[3u8; 32]));
        assert_eq!(parsed.generation(), generation);
    }

    #[test]
    fn vec_map_missing_generation_zero_is_empty_but_later_generations_are_corrupt() {
        let temp = tempfile::TempDir::new().unwrap();
        let dir = temp.path();
        assert!(VecMap::load(dir, 0)
            .expect("generation 0 is empty")
            .is_empty());
        assert!(VecMap::load(dir, 3).is_err());
    }

    #[test]
    fn hex_digest_round_trips() {
        let digest = [0xab_u8; 32];
        let hex = digest_to_hex(&digest);
        assert_eq!(hex.len(), 64);
        assert_eq!(hex_to_digest(&hex), Some(digest));
        assert_eq!(hex_to_digest(&hex.to_uppercase()), Some(digest));
        assert_eq!(hex_to_digest("not-a-hash"), None);
        assert_eq!(hex_to_digest(&hex[..62]), None);
    }

    #[test]
    fn append_records_grows_v2_incrementally() {
        let temp = tempfile::TempDir::new().unwrap();
        let dir = temp.path();
        let mut store = VectorStore::new();
        store.insert([1u8; 32], "m1", vec![1.0, 2.0]);
        store
            .append_records(dir, &[([1u8; 32], "m1".to_string())])
            .expect("initial v2 write");
        let len_after_one = std::fs::metadata(dir.join(VECTORS_FILE)).unwrap().len();

        store.insert([2u8; 32], "m1", vec![3.0, 4.0]);
        store
            .append_records(dir, &[([2u8; 32], "m1".to_string())])
            .expect("second v2 append");
        let len_after_two = std::fs::metadata(dir.join(VECTORS_FILE)).unwrap().len();
        assert!(len_after_two > len_after_one);

        let reloaded = VectorStore::load(dir).expect("reload v2");
        assert_eq!(reloaded.len(), 2);
        assert_eq!(reloaded.get(&[1u8; 32], "m1"), Some(&[1.0f32, 2.0][..]));
        assert_eq!(reloaded.get(&[2u8; 32], "m1"), Some(&[3.0f32, 4.0][..]));
    }

    #[test]
    fn append_records_upgrades_legacy_v1_in_place() {
        let temp = tempfile::TempDir::new().unwrap();
        let dir = temp.path();
        // Hand-craft a legacy v1 file with 1 record
        let mut v1 = Vec::new();
        v1.extend_from_slice(VECTORS_MAGIC);
        v1.push(VECTORS_FORMAT_VERSION_V1);
        v1.extend_from_slice(&1u32.to_le_bytes());
        encode_record(&mut v1, &[9u8; 32], "m1", &[0.5, 0.25]);
        std::fs::write(dir.join(VECTORS_FILE), &v1).unwrap();

        let mut store = VectorStore::load(dir).expect("load legacy v1");
        assert_eq!(store.len(), 1);
        store.insert([10u8; 32], "m1", vec![0.75, 1.25]);
        store
            .append_records(dir, &[([10u8; 32], "m1".to_string())])
            .expect("upgrade v1 to v2");

        let raw = std::fs::read(dir.join(VECTORS_FILE)).unwrap();
        assert_eq!(raw[4], VECTORS_FORMAT_VERSION_V2);
        let reloaded = VectorStore::load(dir).expect("reload upgraded v2");
        assert_eq!(reloaded.len(), 2);
        assert_eq!(reloaded.get(&[9u8; 32], "m1"), Some(&[0.5f32, 0.25][..]));
        assert_eq!(reloaded.get(&[10u8; 32], "m1"), Some(&[0.75f32, 1.25][..]));
    }
}
