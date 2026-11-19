use nvdb::{CollectionConfig, Database, Document, Filter, LexicalField};
use serde_json::json;
use tempfile::TempDir;

fn doc(id: &str, title: &str, folder: &str) -> Document {
    Document {
        id: id.to_string(),
        vector: vec![0.0, 0.0, 1.0],
        payload: Some(json!({ "title": title, "folder": folder })),
    }
}

fn config() -> CollectionConfig {
    CollectionConfig::new(3).with_lexical_fields(vec![
        LexicalField::new("title").with_weight(3.0),
        LexicalField::new("folder"),
    ])
}

#[test]
fn collection_lexical_search_covers_insert_delete_and_filter() {
    let temp = TempDir::new().unwrap();
    let db = Database::open(temp.path()).unwrap();
    let coll = db.create_collection("docs", config()).unwrap();

    coll.insert(doc("article", "Consciousness Is the Record", "blog"))
        .unwrap();
    coll.insert(doc("other", "Machine consciousness", "notes"))
        .unwrap();

    let hits = coll.lexical_search("Consciousness Is the Record", 10, None).unwrap();
    assert_eq!(hits[0].id, "article");

    let filter = Filter::Eq {
        field: "folder".to_string(),
        value: json!("blog"),
    };
    let filtered = coll.lexical_search("consciousness record", 10, Some(&filter)).unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].id, "article");

    coll.delete("article").unwrap();
    assert!(coll
        .lexical_search("Consciousness Is the Record", 10, None)
        .unwrap()
        .iter()
        .all(|hit| hit.id != "article"));
}

#[test]
fn lexical_index_survives_flush_and_reopen() {
    let temp = TempDir::new().unwrap();
    {
        let db = Database::open(temp.path()).unwrap();
        let coll = db.create_collection("docs", config()).unwrap();
        coll.insert(doc("article", "Consciousness Is the Record", "blog"))
            .unwrap();
        coll.flush().unwrap();
    }

    let db = Database::open(temp.path()).unwrap();
    let coll = db.get_collection("docs").unwrap();
    let hits = coll.lexical_search("Consciousness Is the Record", 10, None).unwrap();
    assert_eq!(hits[0].id, "article");
}

#[test]
fn replacing_a_document_replaces_its_lexical_terms() {
    let temp = TempDir::new().unwrap();
    let db = Database::open(temp.path()).unwrap();
    let coll = db.create_collection("docs", config()).unwrap();

    coll.insert(doc("article", "Old title", "blog")).unwrap();
    coll.insert(doc("article", "New title", "blog")).unwrap();

    assert!(coll.lexical_search("old", 10, None).unwrap().is_empty());
    assert_eq!(coll.lexical_search("new", 10, None).unwrap()[0].id, "article");
}

#[test]
fn collections_without_lexical_fields_have_no_lexical_index() {
    let temp = TempDir::new().unwrap();
    let db = Database::open(temp.path()).unwrap();
    let coll = db.create_collection("docs", CollectionConfig::new(3)).unwrap();
    coll.insert(doc("article", "Consciousness Is the Record", "blog"))
        .unwrap();
    assert!(coll.lexical_search("consciousness", 10, None).unwrap().is_empty());
}
