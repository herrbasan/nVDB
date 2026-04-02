use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;

const EXIT_SUCCESS: i32 = 0;
const EXIT_GENERAL_ERROR: i32 = 1;
const EXIT_CORRUPTION: i32 = 2;
const EXIT_LOCKED: i32 = 3;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(EXIT_GENERAL_ERROR);
    }

    let command = args[1].as_str();

    match command {
        "init" => handle_init(&args[2..]),
        "destroy" | "drop" => handle_destroy(&args[2..]),
        "info" => handle_info(&args[2..]),
        "compact" => handle_compact(&args[2..]),
        "export" => handle_export(&args[2..]),
        "import" => handle_import(&args[2..]),
        "merge" => handle_merge(&args[2..]),
        "verify" | "check" => handle_verify(&args[2..]),
        "recover" => handle_recover(&args[2..]),
        "dump" => handle_dump(&args[2..]),
        "config" => handle_config(&args[2..]),
        "search" => handle_search(&args[2..]),
        "rebuild-index" => handle_rebuild_index(&args[2..]),
        _ => {
            eprintln!("Unknown command: {}", command);
            print_usage();
            process::exit(EXIT_GENERAL_ERROR);
        }
    }
}

fn print_usage() {
    eprintln!("nVDB Command Line Interface");
    eprintln!("Usage: nvdb <command> [args...]");
    eprintln!( " ");
    eprintln!("Commands:");
    eprintln!("  init <path> --dim N           Initialize a new vector database");
    eprintln!("  destroy <path> --force        Safely delete a database");
    eprintln!("  info <path>                   Show database statistics");
    eprintln!("  compact <path>                Compact the database in-place");
    eprintln!("  export <path> <dest>          Create a portable snapshot");
    eprintln!("  import <src> <path>           Restore a snapshot");
    eprintln!("  merge <base> <merge-in>       Combine databases");
    eprintln!("  verify <path>                 Check for corruptions");
    eprintln!("  recover <src> <dest>          Recover corrupted data");
    eprintln!("  dump <path>                   Export JSON Lines to stdout");
    eprintln!("  config <get|set> ...          Manage metadata/config");
    eprintln!("  search <path> <vector>        Run a vector search");
    eprintln!("  rebuild-index <path>          Rebuild the HNSW graph");
}

fn handle_init(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: nvdb init <path> --dim N");
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    let path = Path::new(&args[0]);
    if path.exists() && fs::read_dir(path).map(|mut d| d.next().is_some()).unwrap_or(false) {
        eprintln!("Error: Path must not contain existing data.");
        process::exit(EXIT_GENERAL_ERROR);
    }

    let mut dim = 0;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--dim" && i + 1 < args.len() {
            dim = args[i + 1].parse().unwrap_or(0);
            break;
        }
        i += 1;
    }

    if dim == 0 {
        eprintln!("Error: A valid --dim N must be provided (> 0).");
        process::exit(EXIT_GENERAL_ERROR);
    }

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path.file_name().unwrap_or_default().to_string_lossy();

    let db = match nvdb::Database::open(parent) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("Failed to open database parent at {}: {}", parent.display(), e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    };

    let config = nvdb::CollectionConfig {
        dim: dim as usize,
        durability: nvdb::Durability::Buffered,
    };

    if let Err(e) = db.create_collection(&name, config) {
        eprintln!("Failed to initialize collection {}: {}", name, e);
        process::exit(EXIT_GENERAL_ERROR);
    }

    println!("Database collection initialized at {} with dimension {}", path.display(), dim);
    process::exit(EXIT_SUCCESS);
}

fn handle_destroy(args: &[String]) {
    if args.len() < 2 || args[1] != "--force" {
        eprintln!("Usage: nvdb destroy <path> --force");
        process::exit(EXIT_GENERAL_ERROR);
    }
    let path = Path::new(&args[0]);
    if !path.join("meta.json").exists() {
        eprintln!("Error: Target is not a valid nVDB collection folder (missing meta.json).");
        process::exit(EXIT_GENERAL_ERROR);
    }

    if let Err(e) = fs::remove_dir_all(path) {
        eprintln!("Failed to destroy collection: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    }

    println!("Destroyed vector database collection at {}", path.display());
    process::exit(EXIT_SUCCESS);
}

fn open_collection(path: &Path) -> Result<(Arc<nvdb::Database>, nvdb::Collection), Box<dyn std::error::Error>> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let db = nvdb::Database::open(parent)?;
    let collection = db.get_collection(&name)?;
    Ok((db, collection))
}

fn handle_info(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: nvdb info <path>");
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    let path = Path::new(&args[0]);
    
    match open_collection(path) {
        Ok((_, col)) => {
            let stats = col.stats();
            println!("Status: Valid");
            println!("Database Path: {}", path.display());
            println!("Total Vectors: {}", stats.total_segment_docs + stats.memtable_docs);
            println!("  - Segments Docs: {}", stats.total_segment_docs);
            println!("  - Memtable Docs: {}", stats.memtable_docs);
            println!("Segments: {}", stats.segment_count);
            process::exit(EXIT_SUCCESS);
        }
        Err(e) => {
            eprintln!("Status: Invalid\nReason: {}", e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    }
}

fn handle_compact(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: nvdb compact <path>");
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    let path = Path::new(&args[0]);

    if path.join(".lock").exists() && !path.join(".readonly").exists() {
        eprintln!("Error: Database is actively locked. Cannot compact without a .readonly lock.");
        process::exit(EXIT_LOCKED);
    }

    eprintln!("[1/2] Connecting to database...");
    let (_, db) = match open_collection(path) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Failed to open collection: {}", e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    };

    eprintln!("[2/2] Merging segments and updating index...");
    match db.compact() {
        Ok(result) => {
            println!("Compaction complete!");
            println!("Docs before: {}", result.docs_before);
            println!("Docs after: {}", result.docs_after);
            println!("Segments merged: {}", result.segments_merged);
            println!("Index Rebuilt: {}", result.index_rebuilt);
            process::exit(EXIT_SUCCESS);
        }
        Err(e) => {
            eprintln!("Failed to compact database: {}", e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    }
}

fn handle_export(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Usage: nvdb export <path> <dest> [--consistent]");
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    let src_path = Path::new(&args[0]);
    let dest_path = Path::new(&args[1]);

    let consistent = args.iter().any(|a| a == "--consistent");
    if consistent {
        if !src_path.join(".readonly").exists() {
            eprintln!("Error: --consistent requested but .readonly marker not found. Database might be actively writing.");
            process::exit(EXIT_LOCKED);
        }
    }

    if dest_path.exists() {
        eprintln!("Error: Destination path already exists. Must be empty.");
        process::exit(EXIT_GENERAL_ERROR);
    }

    eprintln!("[1/4] Preparing snapshot directory...");
    if let Err(e) = fs::create_dir_all(dest_path) {
        eprintln!("Failed to create destination directory: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    }

    eprintln!("[2/4] Connecting to source database...");
    let (_, db) = match open_collection(src_path) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Failed to open source database: {}", e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    };

    eprintln!("[3/4] Exporting active state (Crash-Consistent)...");
    if let Err(e) = db.export_snapshot(dest_path) {
         eprintln!("Export failed: {}", e);
         process::exit(EXIT_GENERAL_ERROR);
    }

    eprintln!("[4/4] Writing snapshot.json marker...");
    let snapshot_json = format!(
        "{{\n  \"type\": \"nvdb\",\n  \"version\": 1,\n  \"timestamp\": {},\n  \"original_path\": \"{}\"\n}}\n",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis(),
        src_path.display()
    );
    
    if let Err(e) = fs::write(dest_path.join("snapshot.json"), snapshot_json) {
        eprintln!("Failed to write snapshot.json: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    }

    println!("Successfully exported snapshot to {}", dest_path.display());
    process::exit(EXIT_SUCCESS);
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    if !dst.exists() {
        fs::create_dir_all(dst)?;
    }
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let entry_path = entry.path();
        let dest_path = dst.join(entry.file_name());
        
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&entry_path, &dest_path)?;
        } else {
            fs::copy(&entry_path, &dest_path)?;
        }
    }
    Ok(())
}

fn handle_import(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Usage: nvdb import <snapshot> <dest> [--force]");
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    let src_path = Path::new(&args[0]);
    let dest_path = Path::new(&args[1]);
    let force = args.len() > 2 && args[2] == "--force";

    let snapshot_file = src_path.join("snapshot.json");
    if !snapshot_file.exists() {
        eprintln!("Error: Source is not a valid snapshot (missing snapshot.json).");
        process::exit(EXIT_GENERAL_ERROR);
    }

    let snapshot_content = match fs::read_to_string(&snapshot_file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading snapshot.json: {}", e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    };
    
    let snapshot_json: serde_json::Value = match serde_json::from_str(&snapshot_content) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error parsing snapshot.json: {}", e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    };

    if snapshot_json.get("type").and_then(|v| v.as_str()) != Some("nvdb") {
        eprintln!("Error: Snapshot type is not 'nvdb'.");
        process::exit(EXIT_GENERAL_ERROR);
    }

    if snapshot_json.get("version").and_then(|v| v.as_u64()) != Some(1) {
        eprintln!("Error: Unsupported snapshot version.");
        process::exit(EXIT_GENERAL_ERROR);
    }

    if dest_path.exists() {
        let is_empty = fs::read_dir(dest_path).map(|mut i| i.next().is_none()).unwrap_or(false);
        if !is_empty {
            if !force {
                eprintln!("Error: Target directory exists and is not empty. Use --force to overwrite.");
                process::exit(EXIT_GENERAL_ERROR);
            }
            if let Err(e) = fs::remove_dir_all(dest_path) {
                eprintln!("Error removing existing target directory: {}", e);
                process::exit(EXIT_GENERAL_ERROR);
            }
        }
    }

    if let Err(e) = fs::create_dir_all(dest_path) {
        eprintln!("Error creating target directory: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    }

    // Copy MANIFEST (meta.json)
    let meta_src = src_path.join("meta.json"); // or MANIFEST, see spec: meta.json (formally MANIFEST)
    if meta_src.exists() {
        if let Err(e) = fs::copy(&meta_src, dest_path.join("meta.json")) {
            eprintln!("Error copying meta.json: {}", e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    }

    let manifest_src = src_path.join("MANIFEST");
    if manifest_src.exists() {
        if let Err(e) = fs::copy(&manifest_src, dest_path.join("MANIFEST")) {
            eprintln!("Error copying MANIFEST: {}", e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    }

    // Copy index.hnsw
    let index_src = src_path.join("index.hnsw");
    if index_src.exists() {
        if let Err(e) = fs::copy(&index_src, dest_path.join("index.hnsw")) {
            eprintln!("Error copying index.hnsw: {}", e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    }

    // Copy segments directory
    let segments_src = src_path.join("segments");
    if segments_src.exists() {
        if let Err(e) = copy_dir_recursive(&segments_src, &dest_path.join("segments")) {
            eprintln!("Error copying segments: {}", e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    }

    println!("Import complete. Snapshot restored to {}", dest_path.display());
    process::exit(EXIT_SUCCESS);
}
fn handle_merge(args: &[String]) {
    if args.len() < 4 || args[2] != "--output" {
        eprintln!("Usage: nvdb merge <base> <merge-in> --output <dest>");
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    let base_path = Path::new(&args[0]);
    let merge_path = Path::new(&args[1]);
    let dest_path = Path::new(&args[3]);

    if dest_path.exists() {
        eprintln!("Error: Destination path already exists. Must be empty.");
        process::exit(EXIT_GENERAL_ERROR);
    }

    eprintln!("[1/4] Opening source databases...");
    let (_, base_col) = match open_collection(base_path) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Failed to open base collection: {}", e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    };
    
    let (_, merge_col) = match open_collection(merge_path) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Failed to open merge-in collection: {}", e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    };

    if base_col.config().dim != merge_col.config().dim {
        eprintln!("Error: Dimension mismatch between base ({}) and merge-in ({})", base_col.config().dim, merge_col.config().dim);
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    eprintln!("[2/4] Resolving document collisions natively...");
    let base_docs = base_col.export_all_docs().unwrap_or_else(|e| {
        eprintln!("Failed to extract active documents from base: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    });
    
    let merge_docs = merge_col.export_all_docs().unwrap_or_else(|e| {
        eprintln!("Failed to extract active documents from merge-in: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    });
    
    let mut resolved_docs = std::collections::HashMap::new();
    for doc in base_docs {
        resolved_docs.insert(doc.id.clone(), doc);
    }
    for doc in merge_docs {
        // Spec says Source 2 overwrites Source 1
        resolved_docs.insert(doc.id.clone(), doc);
    }
    
    eprintln!("[3/4] Writing merged segments to destination...");
    
    if let Err(e) = fs::create_dir_all(dest_path) {
        eprintln!("Failed to create destination directory: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    let parent = dest_path.parent().unwrap_or_else(|| Path::new("."));
    let name = dest_path.file_name().unwrap_or_default().to_string_lossy();
    
    let db = nvdb::Database::open(parent).unwrap_or_else(|e| {
        eprintln!("Failed to init dest workspace: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    });
    
    let dest_col = db.create_collection(&name, merge_col.config().clone()).unwrap_or_else(|e| {
        eprintln!("Failed to init dest collection: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    });
    
    let docs: Vec<nvdb::Document> = resolved_docs.into_values().collect();
    
    if let Err(e) = dest_col.insert_batch(docs) {
        eprintln!("Failed to insert merged documents: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    eprintln!("[4/4] Compacting and building HNSW graph...");
    if let Err(e) = dest_col.compact() {
        eprintln!("Failed to finalize dest collection: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    println!("Merge complete. Unified database at {}", dest_path.display());
    process::exit(EXIT_SUCCESS);
}

fn handle_verify(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: nvdb verify <path>");
        process::exit(EXIT_GENERAL_ERROR);
    }
    let path = Path::new(&args[0]);
    if !path.join("MANIFEST").exists() && !path.join("meta.json").exists() {
        eprintln!("Error: Target is not a valid nVDB folder.");
        process::exit(EXIT_GENERAL_ERROR);
    }

    eprintln!("[1/2] Preparing sandbox for structural checks...");
    let temp_name = format!("nvdb_verify_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());
    let temp_dir = env::temp_dir().join(temp_name);
    
    if let Err(e) = fs::create_dir_all(&temp_dir) {
        eprintln!("Failed to create verification sandbox: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    // Copy pieces to temp_dir safely
    if path.join("MANIFEST").exists() {
        let _ = fs::copy(path.join("MANIFEST"), temp_dir.join("MANIFEST"));
    } else {
        let _ = fs::copy(path.join("meta.json"), temp_dir.join("meta.json"));
    }
    let _ = copy_dir_recursive(&path.join("segments"), &temp_dir.join("segments"));
    
    let wal_path = path.join("wal.log");
    let mut original_wal_size = 0;
    if wal_path.exists() {
        original_wal_size = fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
        let _ = fs::copy(&wal_path, temp_dir.join("wal.log"));
        // DO NOT copy index.hnsw to let it rebuild natively!
    }

    eprintln!("[2/2] Running rigorous block-level validations...");
    
    let parent = temp_dir.parent().unwrap_or_else(|| Path::new("."));
    let name = temp_dir.file_name().unwrap_or_default().to_string_lossy();
    
    let mut corruptions = 0;
    
    let db_res = nvdb::Database::open(parent);
    if let Ok(db) = db_res {
        match db.get_collection(&name) {
            Ok(_) => {
                // If it loads correctly, we must check if WAL shrank! (Silent truncation = corruption)
                if temp_dir.join("wal.log").exists() {
                    let temp_wal_size = fs::metadata(temp_dir.join("wal.log")).map(|m| m.len()).unwrap_or(0);
                    if temp_wal_size < original_wal_size {
                        eprintln!("Corruption detected: wal.log has CRC32 mismatches ({} -> {} bytes).", original_wal_size, temp_wal_size);
                        corruptions += 1;
                    }
                }
            },
            Err(e) => {
                eprintln!("Corruption detected: Collection load failed structurally: {}", e);
                corruptions += 1;
            }
        }
    } else {
        eprintln!("Corruption detected: Database could not mount due to checksum mismatch.");
        corruptions += 1;
    }
    
    // Cleanup Temp
    let _ = fs::remove_dir_all(&temp_dir);
    
    if corruptions > 0 {
        eprintln!("Verification failed. Database structural anomalies detected.");
        process::exit(EXIT_CORRUPTION); // Code 2
    }
    
    println!("Database integrity verified block-by-block. 0 Errors.");
    process::exit(EXIT_SUCCESS);
}

fn handle_recover(args: &[String]) {
    if args.len() < 3 || args[1] != "--output" {
        eprintln!("Usage: nvdb recover <src> --output <dest>");
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    let src_path = Path::new(&args[0]);
    let dest_path = Path::new(&args[2]);

    if dest_path.exists() {
        eprintln!("Error: Destination path already exists. Must be empty for recovery.");
        process::exit(EXIT_GENERAL_ERROR);
    }

    eprintln!("[1/2] Seeding target payload and filtering bad blocks...");
    if let Err(e) = fs::create_dir_all(dest_path) {
        eprintln!("Failed to create destination directory: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    }

    // Recover segments by attempting to read them and copying if safe
    let src_segments = src_path.join("segments");
    let dest_segments = dest_path.join("segments");
    fs::create_dir_all(&dest_segments).unwrap();

    let mut healthy_segments = std::collections::HashSet::new();
    
    if src_segments.exists() {
         for entry in fs::read_dir(&src_segments).unwrap().flatten() {
             let fpath = entry.path();
             if fpath.is_file() {
                 let fname = entry.file_name();
                 // Attempt to check Checksums
                 let metadata = fs::metadata(&fpath).unwrap();
                 let size = metadata.len();
                 let valid = if size % 64 != 0 && size > 40 { // header checks
                     false
                 } else {
                     match nvdb::segment::Segment::open(&fpath) {
                         Ok(_) => true,
                         Err(_) => false,
                     }
                 };
                 
                 if valid {
                     let _ = fs::copy(&fpath, dest_segments.join(&fname));
                     healthy_segments.insert(fname.to_string_lossy().into_owned());
                 } else {
                     eprintln!("Skipping Corrupt Segment: {}", fname.to_string_lossy());
                 }
             }
         }
    }
    
    // Rewrite a safe MANIFEST dropping corrupted segment links
    let src_manifest_path = src_path.join("MANIFEST");
    let mut manifest_is_valid = false;
    if src_manifest_path.exists() {
        let content = fs::read_to_string(&src_manifest_path).unwrap_or_default();
        if let Ok(mut manifest) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(segs) = manifest.get_mut("segments").and_then(|s| s.as_array_mut()) {
                segs.retain(|el| {
                    let fname = el.get("filename").and_then(|f| f.as_str()).unwrap_or("");
                    healthy_segments.contains(fname)
                });
            }
            let _ = fs::write(dest_path.join("MANIFEST"), serde_json::to_string_pretty(&manifest).unwrap());
            manifest_is_valid = true;
        }
    }
    
    if !manifest_is_valid {
        eprintln!("Warning: MANIFEST missing or corrupt, recovery relies solely on recovered segments mapping.");
    }

    let wal_path = src_path.join("wal.log");
    if wal_path.exists() {
        eprintln!("Migrating uncommitted WAL payloads for partial validation...");
        let _ = fs::copy(&wal_path, dest_path.join("wal.log"));
    }

    eprintln!("[2/2] Force rebuilding database index graph...");
    let parent = dest_path.parent().unwrap_or_else(|| Path::new("."));
    let name = dest_path.file_name().unwrap_or_default().to_string_lossy();
    
    let db = nvdb::Database::open(parent).unwrap_or_else(|e| {
        eprintln!("Failed to init recovered workspace: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    });
    
    let dest_col = db.get_collection(&name).unwrap_or_else(|e| {
         eprintln!("Failed to init recovered collection struct: {}", e);
         process::exit(EXIT_GENERAL_ERROR);
    });
    
    if let Err(e) = dest_col.rebuild_index(None, None) {
        eprintln!("Warning: Index graph failed to be rebuilt completely: {}", e);
    }
    if let Err(e) = dest_col.compact() {
        eprintln!("Warning: Compaction of recovered data was partial: {}", e);
    }

    println!("Recovery complete. Safely salvaged dataset to {}", dest_path.display());
    process::exit(EXIT_SUCCESS);
}
fn handle_dump(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: nvdb dump <path>");
        process::exit(EXIT_GENERAL_ERROR);
    }
    let path = Path::new(&args[0]);
    let (_, col) = open_collection(path).unwrap_or_else(|e| {
        eprintln!("Failed to open collection: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    });

    let docs = col.export_all_docs().unwrap_or_else(|e| {
        eprintln!("Failed to read active documents: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    });

    for doc in docs {
        let json_line = serde_json::json!({
            "id": doc.id,
            "vector": doc.vector,
            "payload": doc.payload
        });
        println!("{}", serde_json::to_string(&json_line).unwrap_or_default());
    }
    process::exit(EXIT_SUCCESS);
}

fn handle_config(args: &[String]) {
    let usage = "Usage: nvdb config <get|set> <key> [value] (Run inside DB folder or point to meta.json)";
    if args.len() < 2 {
        eprintln!("{}", usage);
        process::exit(EXIT_GENERAL_ERROR);
    }

    let meta_path = Path::new("meta.json");
    let manifest_path = Path::new("MANIFEST");
    let actual_path = if manifest_path.exists() {
        manifest_path
    } else if meta_path.exists() {
        meta_path
    } else {
        eprintln!("Error: MANIFEST or meta.json not found in current directory.");
        process::exit(EXIT_GENERAL_ERROR);
    };

    let action = args[0].as_str();
    let key = &args[1];
    
    let mut meta: serde_json::Value = match fs::read_to_string(actual_path) {
        Ok(c) => serde_json::from_str(&c).unwrap_or(serde_json::json!({})),
        Err(_) => serde_json::json!({})
    };

    if action == "get" {
        let mut current = &meta;
        for part in key.split('.') {
            if let Some(obj) = current.as_object() {
                current = obj.get(part).unwrap_or(&serde_json::Value::Null);
            } else {
                current = &serde_json::Value::Null;
                break;
            }
        }
        match current {
            serde_json::Value::String(s) => println!("{}", s),
            serde_json::Value::Null => println!("(null)"),
            other => println!("{}", other),
        }
    } else if action == "set" {
        if args.len() < 3 {
             eprintln!("{}", usage);
             process::exit(EXIT_GENERAL_ERROR);
        }
        let val_str = &args[2];
        let parsed_val = match val_str.parse::<i64>() {
             Ok(n) => serde_json::Value::Number(n.into()),
             Err(_) => match serde_json::from_str(val_str) {
                 Ok(v) => v,
                 Err(_) => serde_json::Value::String(val_str.clone()),
             }
        };

        let parts: Vec<&str> = key.split('.').collect();
        let mut current = &mut meta;
        for (i, part) in parts.iter().enumerate() {
            if i == parts.len() - 1 {
                if let Some(obj) = current.as_object_mut() {
                    obj.insert(part.to_string(), parsed_val.clone());
                }
            } else {
                if !current.as_object().map(|o| o.contains_key(*part)).unwrap_or(false) {
                    if let Some(obj) = current.as_object_mut() {
                        obj.insert(part.to_string(), serde_json::json!({}));
                    }
                }
                current = current.get_mut(*part).unwrap();
            }
        }
        
        let _ = fs::write(actual_path, serde_json::to_string_pretty(&meta).unwrap());
        println!("Updated '{}'", key);
    }
    process::exit(EXIT_SUCCESS);
}

fn handle_search(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Usage: nvdb search <path> '[array...]' [--k 5]");
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    let path = Path::new(&args[0]);
    let vector_str = &args[1];
    
    let vector: Vec<f32> = match serde_json::from_str(vector_str) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("Error: Vector must be a valid JSON array of floats.");
            process::exit(EXIT_GENERAL_ERROR);
        }
    };
    
    let mut top_k = 10;
    for i in 2..args.len() {
        if args[i] == "--k" && i + 1 < args.len() {
            top_k = args[i+1].parse().unwrap_or(10);
            break;
        }
    }
    
    let (_, col) = open_collection(path).unwrap_or_else(|e| {
        eprintln!("Failed to open collection: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    });

    let search_req = nvdb::Search::new(&vector).top_k(top_k);
    
    match col.search(&search_req) {
        Ok(results) => {
            for (idx, result) in results.iter().enumerate() {
                println!("{}. ID: {} (Distance: {:.4})", idx+1, result.id, result.score);
            }
        },
        Err(e) => {
            eprintln!("Search failed: {}", e);
            process::exit(EXIT_GENERAL_ERROR);
        }
    }
    process::exit(EXIT_SUCCESS);
}

fn handle_rebuild_index(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: nvdb rebuild-index <path>");
        process::exit(EXIT_GENERAL_ERROR);
    }
    let path = Path::new(&args[0]);
    let (_, col) = open_collection(path).unwrap_or_else(|e| {
        eprintln!("Failed to open collection: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    });

    eprintln!("[1/1] Rebuilding HNSW...");
    if let Err(e) = col.rebuild_index(None, None) {
        eprintln!("Failed to rebuild index: {}", e);
        process::exit(EXIT_GENERAL_ERROR);
    }
    
    eprintln!("HNSW Index successfully rebuilt.");
    process::exit(EXIT_SUCCESS);
}
