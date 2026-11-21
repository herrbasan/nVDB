//! Generation-stamped mutable ACL overlay.
//!
//! Access decisions never come from a Tantivy segment: segments are immutable,
//! and re-indexing a whole document to change its visibility would be wasteful
//! and slow. The overlay is a small sidecar map `doc_id -> allowed group IDs`
//! that is replaced atomically per generation:
//!
//! - [`AclOverlay::update_acl`] bumps the generation and changes an entry
//!   **without touching any segment**.
//! - [`AclOverlay::save_atomic`] writes `acl_overlay.gen_<N>.bin.tmp`, calls
//!   `sync_all()`, then renames onto `acl_overlay.gen_<N>.bin`. It does not
//!   delete anything: superseded generations are removed only after the
//!   manifest points at the new one
//!   ([`AclOverlay::cleanup_superseded`]), so a crash at any point leaves a
//!   readable generation behind.
//! - [`AclOverlay::is_visible`] is fail-closed: an absent document, an entry
//!   without groups, or an empty principal set is not visible.
//!
//! Group sets are interned — the on-disk file carries a small table of distinct
//! ACLs and a `doc_id -> table index` map. Real collections use a handful of
//! distinct ACLs across tens of thousands of documents, so this keeps both the
//! file and the map small; together with the length-prefixed binary encoding it
//! is what keeps a read-only `open()` cheap (a JSON sidecar with 26k entries
//! cost ~20 ms of pure parsing).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::v2::V2Error;

/// Prefix of every persisted overlay generation.
pub const ACL_OVERLAY_PREFIX: &str = "acl_overlay.gen_";

/// Suffix of every persisted overlay generation.
pub const ACL_OVERLAY_SUFFIX: &str = ".bin";

/// Magic bytes of a persisted overlay generation.
const OVERLAY_MAGIC: &[u8; 4] = b"NVAC";

/// Format version of a persisted overlay generation.
const OVERLAY_FORMAT_VERSION: u8 = 1;

/// Mutable generation-stamped access overlay.
#[derive(Debug, Clone, Default)]
pub struct AclOverlay {
    generation: u64,
    /// Distinct ACLs, referenced by index from `docs`.
    group_sets: Vec<Vec<String>>,
    docs: HashMap<String, u32>,
}

impl AclOverlay {
    /// Empty overlay at generation `0`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current generation.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Number of documents carrying an ACL entry.
    pub fn doc_count(&self) -> usize {
        self.docs.len()
    }

    /// Group IDs allowed to read `doc_id`, if the document is known.
    pub fn groups(&self, doc_id: &str) -> Option<&[String]> {
        let index = *self.docs.get(doc_id)? as usize;
        self.group_sets.get(index).map(Vec::as_slice)
    }

    /// All document IDs with an ACL entry.
    pub fn doc_ids(&self) -> impl Iterator<Item = &String> + '_ {
        self.docs.keys()
    }

    /// Replace the ACL of one document and bump the generation.
    ///
    /// Group IDs are trimmed, empty entries dropped, then sorted and
    /// deduplicated so the overlay is canonical and diff-friendly.
    pub fn update_acl(&mut self, doc_id: &str, groups: Vec<String>) {
        let index = self.intern(normalize_groups(groups));
        self.docs.insert(doc_id.to_string(), index);
        self.generation += 1;
    }

    /// Remove one document and bump the generation when it was present.
    pub fn remove(&mut self, doc_id: &str) {
        if self.docs.remove(doc_id).is_some() {
            self.generation += 1;
        }
    }

    /// Replace every entry (used by a clean rebuild) and bump the generation.
    pub fn replace_all(&mut self, docs: HashMap<String, Vec<String>>) {
        self.group_sets = Vec::new();
        self.docs = HashMap::with_capacity(docs.len());
        for (doc_id, groups) in docs {
            let index = self.intern(normalize_groups(groups));
            self.docs.insert(doc_id, index);
        }
        self.generation += 1;
    }

    /// Fail-closed visibility test.
    ///
    /// Returns `false` when the document has no entry, its entry is empty, or
    /// the caller holds none of its groups.
    pub fn is_visible(&self, doc_id: &str, principals: &HashSet<&str>) -> bool {
        if principals.is_empty() {
            return false;
        }
        let Some(index) = self.docs.get(doc_id) else {
            return false;
        };
        match self.group_sets.get(*index as usize) {
            Some(groups) => groups
                .iter()
                .any(|group| principals.contains(group.as_str())),
            None => false,
        }
    }

    /// Index of `groups` in the distinct-set table, inserting it when new.
    fn intern(&mut self, groups: Vec<String>) -> u32 {
        if let Some(index) = self
            .group_sets
            .iter()
            .position(|existing| *existing == groups)
        {
            return index as u32;
        }
        self.group_sets.push(groups);
        (self.group_sets.len() - 1) as u32
    }

    /// Serialise this overlay as generation `generation`.
    pub fn to_bytes(&self, generation: u64) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::with_capacity(16 + self.docs.len() * 64);
        bytes.extend_from_slice(OVERLAY_MAGIC);
        bytes.push(OVERLAY_FORMAT_VERSION);
        bytes.extend_from_slice(&generation.to_le_bytes());
        push_len(&mut bytes, self.group_sets.len());
        for groups in &self.group_sets {
            push_len(&mut bytes, groups.len());
            for group in groups {
                push_str(&mut bytes, group);
            }
        }
        push_len(&mut bytes, self.docs.len());
        for (doc_id, index) in &self.docs {
            push_str(&mut bytes, doc_id);
            bytes.extend_from_slice(&index.to_le_bytes());
        }
        bytes
    }

    /// Parse a generation written by [`AclOverlay::to_bytes`].
    pub fn from_bytes(bytes: &[u8], generation: u64) -> Result<Self, V2Error> {
        let mut reader = ByteReader::new(bytes);
        let magic = reader
            .take(OVERLAY_MAGIC.len())
            .ok_or_else(|| overlay_corrupt(generation, "truncated magic"))?;
        if magic != OVERLAY_MAGIC {
            return Err(overlay_corrupt(generation, "bad magic bytes"));
        }
        let version = reader
            .u8()
            .ok_or_else(|| overlay_corrupt(generation, "truncated format version"))?;
        if version != OVERLAY_FORMAT_VERSION {
            return Err(overlay_corrupt(
                generation,
                &format!("unsupported format version {version}"),
            ));
        }
        let written_generation = reader
            .u64()
            .ok_or_else(|| overlay_corrupt(generation, "truncated generation"))?;
        let set_count = reader
            .len()
            .ok_or_else(|| overlay_corrupt(generation, "truncated group-set count"))?;
        let mut group_sets: Vec<Vec<String>> = Vec::with_capacity(set_count);
        for _ in 0..set_count {
            let group_count = reader
                .len()
                .ok_or_else(|| overlay_corrupt(generation, "truncated group count"))?;
            let mut groups: Vec<String> = Vec::with_capacity(group_count);
            for _ in 0..group_count {
                let group = reader
                    .string()
                    .ok_or_else(|| overlay_corrupt(generation, "truncated group id"))?;
                groups.push(group);
            }
            group_sets.push(groups);
        }
        let count = reader
            .len()
            .ok_or_else(|| overlay_corrupt(generation, "truncated document count"))?;
        let mut docs: HashMap<String, u32> = HashMap::with_capacity(count);
        for _ in 0..count {
            let doc_id = reader
                .string()
                .ok_or_else(|| overlay_corrupt(generation, "truncated document id"))?;
            let index = reader
                .u32()
                .ok_or_else(|| overlay_corrupt(generation, "truncated group-set index"))?;
            if index as usize >= group_sets.len() {
                return Err(overlay_corrupt(
                    generation,
                    "document references a missing group set",
                ));
            }
            docs.insert(doc_id, index);
        }
        Ok(Self {
            generation: written_generation,
            group_sets,
            docs,
        })
    }

    /// Write generation `generation` atomically into `dir`.
    ///
    /// `sync_all()` runs before the rename, and a held destination handle on
    /// Windows triggers the remove-and-retry fallback. Superseded generations
    /// are **not** deleted here; call [`AclOverlay::cleanup_superseded`] after
    /// the manifest swap.
    pub fn save_atomic(&self, dir: &Path, generation: u64) -> Result<(), V2Error> {
        fs::create_dir_all(dir)?;
        let path = overlay_path(dir, generation);
        let mut tmp = path.as_os_str().to_os_string();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);

        let bytes = self.to_bytes(generation);
        {
            let mut file = fs::File::create(&tmp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }

        if fs::rename(&tmp, &path).is_ok() {
            return Ok(());
        }
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                let _ = fs::remove_file(&tmp);
                return Err(V2Error::Io(error));
            }
        }
        if let Err(error) = fs::rename(&tmp, &path) {
            let _ = fs::remove_file(&tmp);
            return Err(V2Error::Io(error));
        }
        Ok(())
    }

    /// Load generation `generation` from `dir`.
    pub fn load(dir: &Path, generation: u64) -> Result<Self, V2Error> {
        let path = overlay_path(dir, generation);
        let bytes = fs::read(&path).map_err(|error| {
            V2Error::Corrupt(format!(
                "ACL overlay generation {generation} is missing at {}: {error}",
                path.display()
            ))
        })?;
        Self::from_bytes(&bytes, generation)
    }

    /// Delete every persisted generation except `keep_generation`.
    ///
    /// Runs after the manifest has been swapped onto `keep_generation`, so the
    /// live generation is never the one being removed.
    pub fn cleanup_superseded(dir: &Path, keep_generation: u64) -> Result<(), V2Error> {
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(V2Error::Io(error)),
        };
        let keep = overlay_path(dir, keep_generation);
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
            if name.starts_with(ACL_OVERLAY_PREFIX) && name.ends_with(ACL_OVERLAY_SUFFIX) {
                fs::remove_file(&path)?;
            }
        }
        Ok(())
    }
}

/// Path of one persisted overlay generation.
pub fn overlay_path(dir: &Path, generation: u64) -> PathBuf {
    dir.join(format!(
        "{ACL_OVERLAY_PREFIX}{generation}{ACL_OVERLAY_SUFFIX}"
    ))
}

fn overlay_corrupt(generation: u64, detail: &str) -> V2Error {
    V2Error::Corrupt(format!("ACL overlay generation {generation}: {detail}"))
}

fn push_len(bytes: &mut Vec<u8>, len: usize) {
    bytes.extend_from_slice(&(len as u32).to_le_bytes());
}

fn push_str(bytes: &mut Vec<u8>, text: &str) {
    push_len(bytes, text.len());
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
        let bytes: [u8; 4] = self.take(4)?.try_into().ok()?;
        Some(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Option<u64> {
        let bytes: [u8; 8] = self.take(8)?.try_into().ok()?;
        Some(u64::from_le_bytes(bytes))
    }

    fn len(&mut self) -> Option<usize> {
        Some(self.u32()? as usize)
    }

    fn string(&mut self) -> Option<String> {
        let len = self.len()?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).ok()
    }
}

fn normalize_groups(groups: Vec<String>) -> Vec<String> {
    let mut normalized: Vec<String> = groups
        .into_iter()
        .map(|group| group.trim().to_string())
        .filter(|group| !group.is_empty())
        .collect();
    normalized.sort_unstable();
    normalized.dedup();
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    fn principals<'a>(values: &'a [&'a str]) -> HashSet<&'a str> {
        values.iter().copied().collect()
    }

    #[test]
    fn update_bumps_generation_and_normalises_groups() {
        let mut overlay = AclOverlay::new();
        assert_eq!(overlay.generation(), 0);
        overlay.update_acl(
            "doc",
            vec![" ops ".to_string(), "eng".to_string(), "ops".to_string()],
        );
        assert_eq!(overlay.generation(), 1);
        assert_eq!(
            overlay.groups("doc").expect("entry"),
            &["eng".to_string(), "ops".to_string()]
        );
    }

    #[test]
    fn visibility_is_fail_closed() {
        let mut overlay = AclOverlay::new();
        overlay.update_acl("doc", vec!["eng".to_string()]);
        assert!(overlay.is_visible("doc", &principals(&["eng"])));
        assert!(!overlay.is_visible("doc", &principals(&["hr"])));
        assert!(!overlay.is_visible("doc", &principals(&[])));
        assert!(!overlay.is_visible("absent", &principals(&["eng"])));

        overlay.update_acl("empty", Vec::new());
        assert!(!overlay.is_visible("empty", &principals(&["eng"])));
    }

    #[test]
    fn generations_round_trip_and_cleanup_keeps_only_the_live_one() {
        let temp = tempfile::TempDir::new().unwrap();
        let dir = temp.path();

        let mut overlay = AclOverlay::new();
        overlay.update_acl("a", vec!["eng".to_string()]);
        overlay.save_atomic(dir, overlay.generation()).unwrap();
        let first = overlay.generation();

        overlay.update_acl("b", vec!["hr".to_string()]);
        overlay.save_atomic(dir, overlay.generation()).unwrap();
        let second = overlay.generation();

        assert!(overlay_path(dir, first).exists());
        assert!(overlay_path(dir, second).exists());

        AclOverlay::cleanup_superseded(dir, second).unwrap();
        assert!(!overlay_path(dir, first).exists());

        let loaded = AclOverlay::load(dir, second).unwrap();
        assert_eq!(loaded.generation(), second);
        assert!(loaded.is_visible("b", &principals(&["hr"])));
        assert!(loaded.is_visible("a", &principals(&["eng"])));
    }
}
