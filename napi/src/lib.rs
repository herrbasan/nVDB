//! nvdb Node.js Native Bindings
//!
//! This crate provides N-API bindings for the nvdb vector database,
//! enabling native-speed vector operations from Node.js.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;

use nvdb::{Database as RustDatabase, Collection as RustCollection, CollectionConfig, Document};
use nvdb::{Durability, Distance, Search, Filter};

// ─── Async Tasks ───────────────────────────────────────────────

pub struct SearchTask {
    coll: Arc<RustCollection>,
    options: SearchOptions,
}

#[napi]
impl Task for SearchTask {
    type Output = Vec<nvdb::Match>;
    type JsValue = Vec<MatchJs>;
    
    fn compute(&mut self) -> Result<Self::Output> {
         let query_f32: Vec<f32> = self.options.vector.iter().map(|v| *v as f32).collect();
         let mut search = Search::new(&query_f32);
         if let Some(top_k) = self.options.top_k {
             search = search.top_k(top_k as usize);
         }
         if let Some(distance) = &self.options.distance {
             let metric = match distance.as_str() {
                 "dot" => Distance::DotProduct,
                 "euclidean" => Distance::Euclidean,
                 _ => Distance::Cosine,
             };
             search = search.distance(metric);
         }
         if let Some(approximate) = self.options.approximate {
             search = search.approximate(approximate);
         }
         if let Some(ef) = self.options.ef {
             search = search.ef(ef as usize);
         }
         if let Some(filter_json) = &self.options.filter {
             let filter: Filter = serde_json::from_str(filter_json)
                 .map_err(|e| Error::from_reason(format!("Invalid filter JSON: {}", e)))?;
             search = search.filter(filter);
         }
         self.coll.search(&search).map_err(|e| Error::from_reason(format!("Search failed: {}", e)))
    }
    
    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        let results = output.into_iter().map(|m| MatchJs {
            id: m.id,
            score: m.score as f64,
            payload: m.payload.map(|p| serde_json::to_string(&p).unwrap_or_default()),
        }).collect();
        Ok(results)
    }
}

pub struct CompactTask {
    coll: Arc<RustCollection>,
}

#[napi]
impl Task for CompactTask {
    type Output = nvdb::CompactionResult;
    type JsValue = CompactionResultJs;
    
    fn compute(&mut self) -> Result<Self::Output> {
         self.coll.compact().map_err(|e| Error::from_reason(format!("Compact failed: {}", e)))
    }
    
    fn resolve(&mut self, env: Env, result: Self::Output) -> Result<Self::JsValue> {
        Ok(CompactionResultJs {
            docs_before: result.docs_before as u32,
            docs_after: result.docs_after as u32,
            segments_merged: result.segments_merged as u32,
            index_rebuilt: result.index_rebuilt,
        })
    }
}

pub struct RebuildIndexTask {
    coll: Arc<RustCollection>,
    options: Option<RebuildIndexOptions>,
}

#[napi]
impl Task for RebuildIndexTask {
    type Output = ();
    type JsValue = ();
    
    fn compute(&mut self) -> Result<Self::Output> {
         let (params, distance) = convert_rebuild_options(self.options.take())?;
         self.coll.rebuild_index(params, distance).map_err(|e| Error::from_reason(format!("Failed to rebuild index: {}", e)))
    }
    
    fn resolve(&mut self, env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(())
    }
}

pub struct ExportTask {
    coll: Arc<RustCollection>,
    dest: std::path::PathBuf,
}

#[napi]
impl Task for ExportTask {
    type Output = ();
    type JsValue = ();
    
    fn compute(&mut self) -> Result<Self::Output> {
         self.coll.export_snapshot(&self.dest).map_err(|e| Error::from_reason(format!("Export failed: {}", e)))
    }
    
    fn resolve(&mut self, env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(())
    }
}

// ─── Helper for FilterBuilder ─────────────────────────────────

/// Database class for Node.js
///
/// Wraps the Rust Database type and provides JS-friendly methods.
#[napi]
pub struct Database {
    inner: std::sync::RwLock<Option<Arc<RustDatabase>>>,
}

impl Database {
    fn inner(&self) -> Result<Arc<RustDatabase>> {
        self.inner.read().unwrap().clone().ok_or_else(|| Error::from_reason("Database closed"))
    }
}

#[napi]
impl Database {
    /// Open or create a database at the given path
    #[napi(constructor)]
    pub fn new(path: String) -> Result<Self> {
        let inner = RustDatabase::open(&path)
            .map_err(|e| Error::from_reason(format!("Failed to open database: {}", e)))?;
        Ok(Self { inner: std::sync::RwLock::new(Some(inner)) })
    }

    /// Create a new collection
    #[napi]
    pub fn create_collection(
        &self,
        name: String,
        dimension: u32,
        options: Option<CollectionOptions>,
    ) -> Result<Collection> {
        let durability = options
            .and_then(|o| o.durability)
            .map(|d| match d.as_str() {
                "sync" => Durability::FdatasyncEachBatch,
                _ => Durability::Buffered,
            })
            .unwrap_or_default();

        let config = CollectionConfig {
            dim: dimension as usize,
            durability,
        };

        let coll = self
            .inner()?.create_collection(&name, config)
            .map_err(|e| Error::from_reason(format!("Failed to create collection: {}", e)))?;

        Ok(Collection {
            inner: std::sync::RwLock::new(Some(Arc::new(coll))),
            _db: std::sync::RwLock::new(Some(self.inner()?.clone())),
        })
    }

    /// Get an existing collection
    /// Release the database lock
    #[napi]
    pub fn close(&self) -> Result<()> {
        *self.inner.write().unwrap() = None;
        Ok(())
    }

    #[napi]
    pub fn get_collection(&self, name: String) -> Result<Collection> {
        let coll = self
            .inner()?.get_collection(&name)
            .map_err(|e| Error::from_reason(format!("Failed to get collection: {}", e)))?;

        Ok(Collection {
            inner: std::sync::RwLock::new(Some(Arc::new(coll))),
            _db: std::sync::RwLock::new(Some(self.inner()?.clone())),
        })
    }

    /// List all collection names
    #[napi]
    pub fn list_collections(&self) -> Result<Vec<String>> {
        Ok(self.inner()?.list_collections())
    }

    /// Drop (delete) a collection
    #[napi]
    pub fn drop_collection(&self, name: String) -> Result<()> {
        self.inner()?.drop_collection(&name)
            .map_err(|e| Error::from_reason(format!("Failed to drop collection: {}", e)))
    }
}

/// Collection options for creation
#[napi(object)]
pub struct CollectionOptions {
    pub durability: Option<String>,
}

/// HNSW index parameters
#[napi(object)]
pub struct HnswParamsJs {
    pub m: Option<u32>,
    pub ef_construction: Option<u32>,
    pub ef_search: Option<u32>,
}

/// Rebuild index options
#[napi(object)]
pub struct RebuildIndexOptions {
    pub params: Option<HnswParamsJs>,
    pub distance: Option<String>,
}

/// Collection class for Node.js
///
/// Wraps the Rust Collection type and provides JS-friendly methods.
#[napi]
pub struct Collection {
    inner: std::sync::RwLock<Option<Arc<RustCollection>>>,
    _db: std::sync::RwLock<Option<Arc<RustDatabase>>>,
}

impl Collection {
    fn inner(&self) -> Result<Arc<RustCollection>> {
        self.inner.read().unwrap().clone().ok_or_else(|| Error::from_reason("Collection closed"))
    }
}

#[napi]
impl Collection {
    /// Get the collection name
    #[napi(getter)]
    pub fn name(&self) -> Result<String> {
        Ok(self.inner()?.name().to_string())
    }

    /// Release the collection lock
    #[napi]
    pub fn close(&self) -> Result<()> {
        *self.inner.write().unwrap() = None;
        *self._db.write().unwrap() = None;
        Ok(())
    }

    /// Get the collection configuration
    #[napi(getter)]
    pub fn config(&self) -> Result<CollectionConfigJs> {
        let config = self.inner()?.config().clone();
        Ok(CollectionConfigJs {
            dim: config.dim as u32,
            durability: match config.durability {
                Durability::FdatasyncEachBatch => "sync".to_string(),
                Durability::Buffered => "buffered".to_string(),
            },
        })
    }

    /// Insert a single document
    #[napi]
    pub fn insert(&self, id: String, vector: Vec<f64>, payload: Option<String>) -> Result<()> {
        let vector_f32: Vec<f32> = vector.iter().map(|v| *v as f32).collect();
        
        let payload_json = payload
            .map(|p| serde_json::from_str(&p))
            .transpose()
            .map_err(|e| Error::from_reason(format!("Invalid JSON payload: {}", e)))?;

        let doc = Document {
            id,
            vector: vector_f32,
            payload: payload_json,
        };

        self.inner()?.insert(doc)
            .map_err(|e| Error::from_reason(format!("Failed to insert document: {}", e)))?;

        Ok(())
    }

    /// Insert multiple documents in a batch
    #[napi]
    pub fn insert_batch(&self, docs: Vec<InsertDoc>) -> Result<()> {
        let documents: Vec<Document> = docs
            .into_iter()
            .map(|d| {
                let payload_json = d
                    .payload
                    .map(|p| serde_json::from_str(&p))
                    .transpose()
                    .map_err(|e| Error::from_reason(format!("Invalid JSON payload: {}", e)))?;

                let vector_f32: Vec<f32> = d.vector.iter().map(|v| *v as f32).collect();

                Ok(Document {
                    id: d.id,
                    vector: vector_f32,
                    payload: payload_json,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        self.inner()?.insert_batch(documents)
            .map_err(|e| Error::from_reason(format!("Failed to insert batch: {}", e)))?;

        Ok(())
    }

    /// Get a document by ID
    #[napi]
    pub fn get(&self, id: String) -> Result<Option<DocumentJs>> {
        let doc = self.inner()?.get(&id)
            .map_err(|e| Error::from_reason(format!("Failed to get document: {}", e)))?;

        Ok(doc.map(|d| DocumentJs {
            id: d.id,
            vector: d.vector.iter().map(|v| *v as f64).collect(),
            payload: d.payload.map(|p| p.to_string()),
        }))
    }

    /// Delete a document by ID
    #[napi]
    pub fn delete(&self, id: String) -> Result<bool> {
        self.inner()?.delete(&id)
            .map_err(|e| Error::from_reason(format!("Failed to delete document: {}", e)))
    }

    /// Search for similar vectors
    #[napi]
    pub fn search(&self, options: SearchOptions) -> Result<AsyncTask<SearchTask>> {
        Ok(AsyncTask::new(SearchTask {
            coll: self.inner()?,
            options,
        }))
    }

    /// Flush memtable to disk
    #[napi]
    pub fn flush(&self) -> Result<()> {
        self.inner()?.flush()
            .map_err(|e| Error::from_reason(format!("Flush failed: {}", e)))
    }

    /// Sync WAL to disk
    #[napi]
    pub fn sync(&self) -> Result<()> {
        self.inner()?.sync()
            .map_err(|e| Error::from_reason(format!("Sync failed: {}", e)))
    }

    /// Compact the collection
    #[napi]
    pub fn compact(&self) -> Result<AsyncTask<CompactTask>> {
        Ok(AsyncTask::new(CompactTask {
            coll: self.inner()?,
        }))
    }

    /// Rebuild the HNSW index with optional parameters
    #[napi]
    pub fn rebuild_index(&self, options: Option<RebuildIndexOptions>) -> Result<AsyncTask<RebuildIndexTask>> {
        Ok(AsyncTask::new(RebuildIndexTask {
            coll: self.inner()?,
            options,
        }))
    }

    /// Delete the HNSW index
    #[napi]
    pub fn delete_index(&self) -> Result<()> {
        self.inner()?.delete_index()
            .map_err(|e| Error::from_reason(format!("Failed to delete index: {}", e)))
    }

    /// Check if HNSW index exists
    #[napi]
    pub fn has_index(&self) -> Result<bool> {
        Ok(self.inner()?.has_index())
    }

    /// Export a consistent snapshot of the collection
    #[napi]
    pub fn export_snapshot(&self, dest: String) -> Result<AsyncTask<ExportTask>> {
        Ok(AsyncTask::new(ExportTask {
            coll: self.inner()?,
            dest: std::path::PathBuf::from(dest),
        }))
    }

    /// Get collection statistics
    #[napi(getter)]
    pub fn stats(&self) -> Result<CollectionStatsJs> {
        let stats = self.inner()?.stats();
        Ok(CollectionStatsJs {
            memtable_docs: stats.memtable_docs as u32,
            segment_count: stats.segment_count as u32,
            total_segment_docs: stats.total_segment_docs as u32,
        })
    }
}

/// Collection configuration (JS representation)
#[napi(object)]
pub struct CollectionConfigJs {
    pub dim: u32,
    pub durability: String,
}

/// Document for insertion
#[napi(object)]
pub struct InsertDoc {
    pub id: String,
    pub vector: Vec<f64>,
    pub payload: Option<String>,
}

/// Document (JS representation)
#[napi(object)]
pub struct DocumentJs {
    pub id: String,
    pub vector: Vec<f64>,
    pub payload: Option<String>,
}

/// Search options
#[napi(object)]
pub struct SearchOptions {
    pub vector: Vec<f64>,
    pub top_k: Option<u32>,
    pub distance: Option<String>,
    pub approximate: Option<bool>,
    pub ef: Option<u32>,
    pub filter: Option<String>,
}

/// Search match result (JS representation)
#[napi(object)]
pub struct MatchJs {
    pub id: String,
    pub score: f64,
    pub payload: Option<String>,
}

/// Compaction result (JS representation)
#[napi(object)]
pub struct CompactionResultJs {
    pub docs_before: u32,
    pub docs_after: u32,
    pub segments_merged: u32,
    pub index_rebuilt: bool,
}

/// Collection statistics (JS representation)
#[napi(object)]
pub struct CollectionStatsJs {
    pub memtable_docs: u32,
    pub segment_count: u32,
    pub total_segment_docs: u32,
}

/// FilterBuilder for constructing query filters
/// 
/// Produces JSON in the format expected by the Rust Filter enum:
/// {"Eq": {"field": "name", "value": "value"}}
#[napi]
pub struct FilterBuilder;

#[napi]
impl FilterBuilder {
    /// Create an equality filter: field == value
    #[napi]
    pub fn eq(field: String, value: serde_json::Value) -> String {
        let filter = serde_json::json!({
            "Eq": { "field": field, "value": value }
        });
        filter.to_string()
    }

    /// Create a greater-than filter: field > value
    #[napi]
    pub fn gt(field: String, value: serde_json::Value) -> String {
        let filter = serde_json::json!({
            "Gt": { "field": field, "value": value }
        });
        filter.to_string()
    }

    /// Create a greater-than-or-equal filter: field >= value
    #[napi]
    pub fn gte(field: String, value: serde_json::Value) -> String {
        let filter = serde_json::json!({
            "Gte": { "field": field, "value": value }
        });
        filter.to_string()
    }

    /// Create a less-than filter: field < value
    #[napi]
    pub fn lt(field: String, value: serde_json::Value) -> String {
        let filter = serde_json::json!({
            "Lt": { "field": field, "value": value }
        });
        filter.to_string()
    }

    /// Create a less-than-or-equal filter: field <= value
    #[napi]
    pub fn lte(field: String, value: serde_json::Value) -> String {
        let filter = serde_json::json!({
            "Lte": { "field": field, "value": value }
        });
        filter.to_string()
    }

    /// Create a not-equal filter: field != value
    #[napi]
    pub fn ne(field: String, value: serde_json::Value) -> String {
        let filter = serde_json::json!({
            "Ne": { "field": field, "value": value }
        });
        filter.to_string()
    }

    /// Create an "in" filter: field IN values
    #[napi]
    pub fn in_(field: String, values: Vec<serde_json::Value>) -> String {
        let filter = serde_json::json!({
            "In": { "field": field, "values": values }
        });
        filter.to_string()
    }

    /// Create a logical AND filter: all filters must match
    #[napi]
    pub fn and(filters: Vec<String>) -> Result<String> {
        let mut filter_objs = Vec::with_capacity(filters.len());
        for f in filters {
            let obj: Filter = serde_json::from_str(&f)
                .map_err(|e| Error::from_reason(format!("Invalid filter JSON: {}", e)))?;
            filter_objs.push(obj);
        }

        let filter = Filter::And(filter_objs);
        Ok(serde_json::to_string(&filter).unwrap())
    }

    /// Create a logical OR filter: any filter must match
    #[napi]
    pub fn or(filters: Vec<String>) -> Result<String> {
        let mut filter_objs = Vec::with_capacity(filters.len());
        for f in filters {
            let obj: Filter = serde_json::from_str(&f)
                .map_err(|e| Error::from_reason(format!("Invalid filter JSON: {}", e)))?;
            filter_objs.push(obj);
        }

        let filter = Filter::Or(filter_objs);
        Ok(serde_json::to_string(&filter).unwrap())
    }
}

/// Helper function to convert JS rebuild options to Rust types
fn convert_rebuild_options(options: Option<RebuildIndexOptions>) -> Result<(Option<nvdb::HnswParams>, Option<nvdb::Distance>)> {
    let params = options.as_ref().and_then(|o| o.params.as_ref()).map(|p| {
        let mut params = nvdb::HnswParams::default();
        if let Some(m) = p.m {
            params = nvdb::HnswParams::with_m(m as usize);
        }
        if let Some(ef_construction) = p.ef_construction {
            params = params.with_ef_construction(ef_construction as usize);
        }
        if let Some(ef_search) = p.ef_search {
            params = params.with_ef_search(ef_search as usize);
        }
        params
    });

    let distance = options.as_ref().and_then(|o| o.distance.as_ref()).map(|d| {
        match d.as_str() {
            "dot" => nvdb::Distance::DotProduct,
            "euclidean" => nvdb::Distance::Euclidean,
            _ => nvdb::Distance::Cosine,
        }
    });

    Ok((params, distance))

}
