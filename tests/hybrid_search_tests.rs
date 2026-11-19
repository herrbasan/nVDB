use nvdb::{CollectionConfig, Database, Document, Filter, HybridSearch, LexicalField, Reranker};
use serde_json::json;
use tempfile::TempDir;

struct FixedReranker(Vec<f32>);

impl Reranker for FixedReranker {
    fn score(&self, _query: &str, documents: &[&str]) -> nvdb::Result<Vec<f32>> {
        assert_eq!(documents.len(), self.0.len());
        Ok(self.0.clone())
    }
}

fn doc(id: &str, vector: [f32; 3], title: &str, folder: &str) -> Document {
    Document {
        id: id.to_string(),
        vector: vector.to_vec(),
        payload: Some(json!({ "title": title, "folder": folder })),
    }
}

fn collection(temp: &TempDir) -> nvdb::Collection {
    let db = Database::open(temp.path()).unwrap();
    let config = CollectionConfig::new(3).with_lexical_fields(vec![
        LexicalField::new("title").with_weight(3.0),
        LexicalField::new("folder"),
    ]);
    db.create_collection("docs", config).unwrap()
}

#[test]
fn lexical_match_rescues_a_dense_rank_three_result() {
    let temp = TempDir::new().unwrap();
    let coll = collection(&temp);
    coll.insert(doc("dense", [1.0, 0.0, 0.0], "machine notes", "notes"))
        .unwrap();
    coll.insert(doc("second", [0.9, 0.1, 0.0], "more machine notes", "notes"))
        .unwrap();
    coll.insert(doc(
        "article",
        [0.0, 1.0, 0.0],
        "Consciousness Is the Record",
        "blog",
    ))
    .unwrap();

    let results = coll
        .hybrid_search(&HybridSearch::new(&[1.0, 0.0, 0.0], "Consciousness Is the Record").top_k(3))
        .unwrap();
    assert_eq!(results[0].id, "article");
    assert_eq!(results[0].lexical_rank, Some(1));
    assert_eq!(results[0].vector_rank, Some(3));
}

#[test]
fn filter_applies_to_both_lanes() {
    let temp = TempDir::new().unwrap();
    let coll = collection(&temp);
    coll.insert(doc("dense", [1.0, 0.0, 0.0], "machine notes", "notes"))
        .unwrap();
    coll.insert(doc(
        "article",
        [0.0, 1.0, 0.0],
        "Consciousness Is the Record",
        "blog",
    ))
    .unwrap();

    let filter = Filter::Eq {
        field: "folder".to_string(),
        value: json!("notes"),
    };
    let results = coll
        .hybrid_search(
            &HybridSearch::new(&[1.0, 0.0, 0.0], "Consciousness Is the Record")
                .filter(filter),
        )
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "dense");
}

#[test]
fn empty_query_text_returns_dense_ranking() {
    let temp = TempDir::new().unwrap();
    let coll = collection(&temp);
    coll.insert(doc("dense", [1.0, 0.0, 0.0], "machine notes", "notes"))
        .unwrap();
    coll.insert(doc("other", [0.0, 1.0, 0.0], "other", "notes"))
        .unwrap();

    let results = coll
        .hybrid_search(&HybridSearch::new(&[1.0, 0.0, 0.0], "").top_k(2))
        .unwrap();
    assert_eq!(results[0].id, "dense");
    assert_eq!(results[0].vector_rank, Some(1));
    assert_eq!(results[0].lexical_rank, None);
}

#[test]
fn diversity_caps_chunks_per_payload_group() {
    let temp = TempDir::new().unwrap();
    let coll = collection(&temp);
    for index in 0..5 {
        let mut chunk = doc(
            &format!("chunk{index}"),
            [1.0 - index as f32 * 0.01, 0.0, 0.0],
            "Consciousness Is the Record",
            "blog",
        );
        chunk.payload = Some(json!({
            "title": "Consciousness Is the Record",
            "folder": "blog",
            "path": "blog/attic/consciousness-is-the-record.md"
        }));
        coll.insert(chunk).unwrap();
    }
    coll.insert(doc("another", [0.0, 1.0, 0.0], "other article", "notes"))
        .unwrap();

    let results = coll
        .hybrid_search(
            &HybridSearch::new(&[1.0, 0.0, 0.0], "Consciousness Is the Record")
                .top_k(10)
                .diversity("path", 2),
        )
        .unwrap();
    let target_chunks = results
        .iter()
        .filter(|result| {
            result.payload.as_ref().unwrap()["path"]
                == "blog/attic/consciousness-is-the-record.md"
        })
        .count();
    assert_eq!(target_chunks, 2);
    assert!(results.iter().any(|result| result.id == "another"));
}

#[test]
fn reranker_blend_can_promote_and_preserves_rrf() {
    let temp = TempDir::new().unwrap();
    let coll = collection(&temp);
    coll.insert(doc("dense", [1.0, 0.0, 0.0], "dense", "notes"))
        .unwrap();
    coll.insert(doc("article", [0.8, 0.2, 0.0], "article", "blog"))
        .unwrap();

    let search = HybridSearch::new(&[1.0, 0.0, 0.0], "").top_k(2);
    let baseline = coll.hybrid_search(&search).unwrap();
    assert_eq!(baseline[0].id, "dense");

    let reranked = coll
        .hybrid_search_reranked(&search, &FixedReranker(vec![0.0, 1.0]), "title", 0.8, 2)
        .unwrap();
    assert_eq!(reranked[0].id, "article");
    assert_eq!(reranked[0].rerank_score, Some(1.0));
    assert!(reranked[0].rrf_score > 0.0);
}

#[test]
fn reranker_rejects_missing_text_fields() {
    let temp = TempDir::new().unwrap();
    let coll = collection(&temp);
    coll.insert(doc("article", [1.0, 0.0, 0.0], "article", "blog"))
        .unwrap();
    let search = HybridSearch::new(&[1.0, 0.0, 0.0], "article").top_k(2);
    let error = coll
        .hybrid_search_reranked(&search, &FixedReranker(vec![1.0]), "missing", 0.2, 2)
        .unwrap_err();
    assert!(error.to_string().contains("no payload field missing"));
}
