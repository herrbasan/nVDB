use nvdb::lexical::{tokenize, LexicalIndex};

#[test]
fn tokenize_lowercases_and_keeps_identifiers() {
    assert_eq!(
        tokenize("Consciousness Is the Record — 47.5k, AI_Club"),
        vec!["consciousness", "is", "the", "record", "47", "5k", "ai", "club"]
    );
    assert_eq!(tokenize("a I x"), Vec::<String>::new());
}

#[test]
fn exact_title_outranks_body_mentions() {
    let mut index = LexicalIndex::new();
    index.insert("article", &[("Consciousness Is the Record", 3.0), ("body text", 1.0)]);
    index.insert("other", &[("machine consciousness notes", 1.0)]);
    let hits = index.search("Consciousness Is the Record", 10);
    assert_eq!(hits[0].id, "article");
}

#[test]
fn funding_query_finds_term_dense_retrieval_missed() {
    let mut index = LexicalIndex::new();
    index.insert(
        "mem_6830",
        &[(
            "Funding conclusion Prototype Fund 47.5k application November 30 funding",
            1.0,
        )],
    );
    index.insert("mem_other", &[("Prototype design funding notes", 1.0)]);
    let hits = index.search("Prototype Fund 47.5k application November 30 funding", 10);
    assert_eq!(hits[0].id, "mem_6830");
}

#[test]
fn update_removes_old_terms_and_delete_removes_document() {
    let mut index = LexicalIndex::new();
    index.insert("doc", &[("old vocabulary", 1.0)]);
    assert_eq!(index.search("vocabulary", 10).len(), 1);

    index.insert("doc", &[("new vocabulary", 1.0)]);
    assert_eq!(index.search("old", 10).len(), 0);
    assert_eq!(index.search("new", 10)[0].id, "doc");

    assert!(index.remove("doc"));
    assert!(index.is_empty());
    assert!(index.search("new", 10).is_empty());
    assert!(!index.remove("doc"));
}

#[test]
fn empty_text_is_not_indexed() {
    let mut index = LexicalIndex::new();
    index.insert("doc", &[("", 1.0)]);
    assert_eq!(index.len(), 0);
    assert!(index.search("anything", 10).is_empty());
}

#[test]
fn ties_are_deterministic_by_id() {
    let mut index = LexicalIndex::new();
    index.insert("b", &[("same words", 1.0)]);
    index.insert("a", &[("same words", 1.0)]);
    let hits = index.search("same words", 10);
    assert_eq!(hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
}

#[test]
fn top_k_is_respected() {
    let mut index = LexicalIndex::new();
    for i in 0..10 {
        index.insert(&format!("doc{i}"), &[("shared term", 1.0)]);
    }
    assert_eq!(index.search("shared term", 3).len(), 3);
    assert!(index.search("shared term", 0).is_empty());
}
