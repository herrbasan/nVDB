//! Lexical (BM25) candidate retrieval.
//!
//! This is the exact-term lane for hybrid retrieval: titles, paths, identifiers,
//! error strings and quoted vocabulary that dense embeddings blur. It is
//! deliberately independent of vector search and carries no persistence of its
//! own; callers rebuild it from stored documents.

use std::collections::{HashMap, HashSet};

const K1: f32 = 1.2;
const B: f32 = 0.75;

/// One lexical result. Scores are BM25 relevance and comparable only within the
/// same query/index state.
#[derive(Debug, Clone, PartialEq)]
pub struct LexicalHit {
    /// External document ID.
    pub id: String,
    /// BM25 score.
    pub score: f32,
}

#[derive(Debug, Clone)]
struct DocStats {
    terms: HashMap<String, f32>,
    len: u32,
}

/// In-memory inverted index with BM25 ranking.
#[derive(Debug, Default)]
pub struct LexicalIndex {
    postings: HashMap<String, HashMap<String, f32>>,
    docs: HashMap<String, DocStats>,
    total_len: u64,
}

impl LexicalIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of documents with indexable text.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Whether the index holds no documents.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Insert or replace a document.
    ///
    /// Each field is `(text, weight)`. Weights increase term frequency without
    /// changing document length, so a title can outrank the same word in body
    /// text without corrupting the BM25 length normalisation.
    pub fn insert(&mut self, id: &str, fields: &[(&str, f32)]) {
        assert!(!id.is_empty(), "lexical insert needs a non-empty document id");
        self.remove(id);

        let mut terms = HashMap::new();
        let mut len = 0u32;
        for (text, weight) in fields {
            assert!(
                weight.is_finite() && *weight > 0.0,
                "lexical field weight must be finite and positive (got {weight})"
            );
            for token in tokenize(text) {
                len = len.saturating_add(1);
                *terms.entry(token).or_insert(0.0) += *weight;
            }
        }
        if terms.is_empty() {
            return;
        }

        for (term, tf) in &terms {
            self.postings
                .entry(term.clone())
                .or_default()
                .insert(id.to_string(), *tf);
        }
        self.total_len += len as u64;
        self.docs.insert(id.to_string(), DocStats { terms, len });
    }

    /// Remove a document. Returns whether it was present.
    pub fn remove(&mut self, id: &str) -> bool {
        let Some(stats) = self.docs.remove(id) else {
            return false;
        };
        self.total_len = self.total_len.saturating_sub(stats.len as u64);
        for term in stats.terms.keys() {
            if let Some(postings) = self.postings.get_mut(term) {
                postings.remove(id);
                if postings.is_empty() {
                    self.postings.remove(term);
                }
            }
        }
        true
    }

    /// Return at most `top_k` documents for `query`, ordered by BM25 score.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<LexicalHit> {
        if top_k == 0 || self.docs.is_empty() {
            return Vec::new();
        }

        let mut query_terms: HashMap<String, f32> = HashMap::new();
        for token in tokenize(query) {
            *query_terms.entry(token).or_insert(0.0) += 1.0;
        }
        if query_terms.is_empty() {
            return Vec::new();
        }

        let n = self.docs.len() as f32;
        let avg_len = (self.total_len as f32 / n).max(f32::EPSILON);
        let mut scores: HashMap<&str, f32> = HashMap::new();
        let mut matched = HashSet::new();

        for (term, query_tf) in &query_terms {
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let df = postings.len() as f32;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for (id, tf) in postings {
                let doc_len = self.docs[id].len as f32;
                let norm = K1 * (1.0 - B + B * doc_len / avg_len);
                let term_score = idf * (tf * (K1 + 1.0)) / (tf + norm);
                *scores.entry(id.as_str()).or_insert(0.0) += term_score * query_tf;
                matched.insert(id.as_str());
            }
        }

        let mut hits: Vec<LexicalHit> = matched
            .into_iter()
            .map(|id| LexicalHit {
                id: id.to_string(),
                score: scores[id],
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        hits.truncate(top_k);
        hits
    }
}

/// Lowercase Unicode alphanumeric tokens of length >= 2.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lower in ch.to_lowercase() {
                current.push(lower);
            }
        } else if !current.is_empty() {
            if current.chars().count() >= 2 {
                tokens.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    if current.chars().count() >= 2 {
        tokens.push(current);
    }
    tokens
}
