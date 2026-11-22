//! Index/query analyzers for the `nVDB` v2 engine.
//!
//! Three built-in Tantivy tokenizers are registered on every
//! [`crate::v2::V2Index`]:
//!
//! | name | behaviour | used for |
//! |------|-----------|----------|
//! | [`TOKENIZER_EXACT`] | trim + Unicode lowercase, one token | `title_exact`, columnar equality fields |
//! | [`TOKENIZER_IDENTIFIER`] | lowercase, whole token **and** sub-tokens split on `/ \ . - _ :` plus whitespace | `path`, `name`, `tags`, `categories` |
//! | [`TOKENIZER_TEXT`] | Unicode word segmentation + lowercase, then `ä->ae, ö->oe, ü->ue, ß->ss` folded into **one** canonical token per word | `title`, `headings`, `body`, `body_parts` |
//!
//! Query construction uses the same helpers as the tokenizers ([`exact_token`],
//! [`identifier_tokens`], [`text_tokens`]) so a term can never be spelled one
//! way at index time and another at query time.

use std::collections::HashSet;

use tantivy::tokenizer::{Token, TokenStream, Tokenizer};
use tantivy::Index;

use crate::v2::german_lexicon::{
    decompose_german_compound, fold_umlauts, is_lexicon_word, MIN_PART_CHARS,
};

/// Analyzer name for whole-string, lowercased matching.
pub const TOKENIZER_EXACT: &str = "nvdb_exact";

/// Analyzer name for path/identifier matching.
pub const TOKENIZER_IDENTIFIER: &str = "nvdb_identifier";

/// Analyzer name for natural-language text.
pub const TOKENIZER_TEXT: &str = "nvdb_text";

/// Characters that separate sub-tokens inside a path or identifier.
const IDENTIFIER_SEPARATORS: [char; 6] = ['/', '\\', '.', '-', '_', ':'];

/// Longest trailing file extension treated as such (`md`, `json`, `xlsx`).
/// A longer dotted suffix belongs to the stem, not the extension.
const MAX_EXTENSION_CHARS: usize = 5;

/// Hard cap on indexed `body_parts` tokens per document.
///
/// Keeps a pathological body from exploding the index; truncation follows
/// source order, so it is deterministic for a given input.
pub const MAX_BODY_PARTS: usize = 8192;

/// Register the three `nVDB` analyzers on `index`.
///
/// Must run on every open: Tantivy persists the schema but not the tokenizer
/// registry.
pub fn register_tokenizers(index: &Index) {
    let tokenizers = index.tokenizers();
    tokenizers.register(TOKENIZER_EXACT, ExactTokenizer);
    tokenizers.register(TOKENIZER_IDENTIFIER, IdentifierTokenizer);
    tokenizers.register(TOKENIZER_TEXT, TextTokenizer);
}

/// Canonical whole-string form: trimmed and Unicode-lowercased.
pub fn exact_token(text: &str) -> String {
    text.trim().to_lowercase()
}

/// Tokens emitted for an identifier-shaped value.
///
/// The first token is the whole (trimmed, lowercased) string, followed by the
/// sub-tokens split on [`IDENTIFIER_SEPARATORS`] and whitespace. Duplicates are
/// dropped so a term is never scored twice for the same document.
pub fn identifier_tokens(text: &str) -> Vec<String> {
    identifier_token_spans(text)
        .into_iter()
        .map(|(_, _, token)| token)
        .collect()
}

/// Tokens emitted for natural-language text.
///
/// Every word becomes exactly **one** token: lowercased and umlaut-folded
/// (`Förderung` -> `foerderung`), so `Förderprogramm` and `Foerderprogramm`
/// produce the identical term. Emitting the folded form only keeps BM25
/// document-length norms exact — a second token per umlaut word would inflate
/// the field length of every German document.
pub fn text_tokens(text: &str) -> Vec<String> {
    text_token_spans(text)
        .into_iter()
        .map(|(_, _, token)| token)
        .collect()
}

/// Extracted German compound parts of `text`, space-joined for indexing into
/// the `body_parts` subfield.
///
/// For every word this emits
///
/// - every lexicon morpheme that is a prefix of the word (so `Förderung`
///   contributes `foerder` and a `Förderprogramm` query can reach it), and
/// - the deterministic morpheme split of the word itself (so
///   `Datenschutzrichtlinie` contributes `datenschutz` and `richtlinie`).
///
/// Both directions are required: compound query → component document *and*
/// component query → compound document.
pub fn extract_body_parts(text: &str) -> String {
    body_part_terms(text).join(" ")
}

/// The individual compound-part terms of `text`.
///
/// Identical to the terms [`extract_body_parts`] joins, exposed so the query
/// path can search for exactly what the index path wrote — this is what makes
/// `Förderprogramm` reach a document that only says `Förderung`, and vice
/// versa.
pub fn body_part_terms(text: &str) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut visited: HashSet<String> = HashSet::new();
    let mut parts: Vec<String> = Vec::new();
    for word in word_runs(text) {
        collect_word_parts(word, MAX_NESTING_DEPTH, &mut parts, &mut seen, &mut visited);
        if parts.len() >= MAX_BODY_PARTS {
            break;
        }
    }
    parts
}

/// How deep nested compounds are followed (`Datenschutzrichtlinie` ->
/// `datenschutz` -> `daten`, `schutz`).
const MAX_NESTING_DEPTH: usize = 3;

fn collect_word_parts(
    word: &str,
    depth: usize,
    parts: &mut Vec<String>,
    seen: &mut HashSet<String>,
    visited: &mut HashSet<String>,
) {
    if parts.len() >= MAX_BODY_PARTS {
        return;
    }
    let folded = fold_umlauts(word);
    if !visited.insert(folded.clone()) {
        return;
    }
    let chars: Vec<char> = folded.chars().collect();
    if chars.len() >= MIN_PART_CHARS {
        for cut in MIN_PART_CHARS..=chars.len() {
            let prefix: String = chars[..cut].iter().collect();
            if is_lexicon_word(&prefix) {
                push_part(parts, seen, prefix);
            }
        }
    }
    if depth == 0 {
        return;
    }
    for part in decompose_german_compound(word) {
        collect_word_parts(&part, depth - 1, parts, seen, visited);
        push_part(parts, seen, part);
    }
}

fn push_part(parts: &mut Vec<String>, seen: &mut HashSet<String>, part: String) {
    if seen.insert(part.clone()) {
        parts.push(part);
    }
}

fn identifier_token_spans(text: &str) -> Vec<(usize, usize, String)> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let base = text.len() - text.trim_start().len();
    let mut tokens: Vec<(usize, usize, String)> = Vec::with_capacity(5);
    let whole = trimmed.to_lowercase();
    tokens.push((base, base + trimmed.len(), whole.clone()));
    if let Some(stem) = identifier_stem_span(trimmed, base) {
        if tokens.iter().all(|(_, _, existing)| *existing != stem.2) {
            tokens.push(stem);
        }
    }

    let mut run_start: Option<usize> = None;
    for (offset, character) in trimmed.char_indices() {
        let is_separator = IDENTIFIER_SEPARATORS.contains(&character) || character.is_whitespace();
        if is_separator {
            if let Some(start) = run_start.take() {
                push_identifier_run(&mut tokens, &whole, base, start, &trimmed[start..offset]);
            }
        } else if run_start.is_none() {
            run_start = Some(offset);
        }
    }
    if let Some(start) = run_start {
        push_identifier_run(&mut tokens, &whole, base, start, &trimmed[start..]);
    }
    tokens
}

/// Extension-stripped stem of the last path segment, if that segment ends in a
/// short ASCII-alphanumeric extension.
///
/// A query that *is* the filename stem (`md-blocks-spec`) must match the file
/// whose name carries the extension (`md-blocks-spec.md`) as a whole token, not
/// only through its separator-split parts (`herrbasan/nVDB#4`).
fn identifier_stem_span(trimmed: &str, base: usize) -> Option<(usize, usize, String)> {
    let segment_start = trimmed.rfind(|character| character == '/' || character == '\\');
    let segment_start = segment_start.map_or(0, |index| index + 1);
    let segment = &trimmed[segment_start..];
    let dot = segment.rfind('.')?;
    if dot == 0 {
        return None;
    }
    let extension = &segment[dot + 1..];
    if extension.is_empty()
        || extension.len() > MAX_EXTENSION_CHARS
        || !extension
            .chars()
            .all(|character| character.is_ascii_alphanumeric())
    {
        return None;
    }
    Some((
        base + segment_start,
        base + segment_start + dot,
        segment[..dot].to_lowercase(),
    ))
}

fn push_identifier_run(
    tokens: &mut Vec<(usize, usize, String)>,
    whole: &str,
    base: usize,
    start: usize,
    run: &str,
) {
    let lowered = run.to_lowercase();
    if lowered.is_empty() || lowered == whole {
        return;
    }
    if tokens.iter().any(|(_, _, existing)| *existing == lowered) {
        return;
    }
    tokens.push((base + start, base + start + run.len(), lowered));
}

fn text_token_spans(text: &str) -> Vec<(usize, usize, String)> {
    word_spans(text)
        .into_iter()
        .map(|(start, end)| (start, end, canonical_token(&text[start..end])))
        .filter(|(_, _, token)| !token.is_empty())
        .collect()
}

/// The canonical analyzed form of one word: lowercase, umlauts folded.
fn canonical_token(word: &str) -> String {
    fold_umlauts(word)
}

fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(8);
    let mut run_start: Option<usize> = None;
    for (offset, character) in text.char_indices() {
        if character.is_alphanumeric() {
            if run_start.is_none() {
                run_start = Some(offset);
            }
            continue;
        }
        if let Some(start) = run_start.take() {
            spans.push((start, offset));
        }
    }
    if let Some(start) = run_start {
        spans.push((start, text.len()));
    }
    spans
}

fn word_runs(text: &str) -> Vec<&str> {
    word_spans(text)
        .into_iter()
        .map(|(start, end)| &text[start..end])
        .collect()
}

/// Tantivy token stream over a pre-computed vector of tokens.
#[derive(Clone, Debug, Default)]
struct VectorTokenStream {
    pending: Vec<Token>,
    next: usize,
    current: Token,
}

impl VectorTokenStream {
    fn new(pending: Vec<Token>) -> Self {
        Self {
            pending,
            next: 0,
            current: Token::default(),
        }
    }
}

impl TokenStream for VectorTokenStream {
    fn advance(&mut self) -> bool {
        if self.next < self.pending.len() {
            self.current = std::mem::take(&mut self.pending[self.next]);
            self.next += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.current
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.current
    }
}

fn push_text_token(tokens: &mut Vec<Token>, start: usize, end: usize, run: &str, position: usize) {
    let text = canonical_token(run);
    if text.is_empty() {
        return;
    }
    tokens.push(Token {
        offset_from: start,
        offset_to: end,
        position,
        text,
        position_length: 1,
    });
}

/// Whole-string, lowercased tokenizer.
#[derive(Clone, Debug, Default)]
struct ExactTokenizer;

impl Tokenizer for ExactTokenizer {
    type TokenStream<'a> = VectorTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> VectorTokenStream {
        if text.trim().is_empty() {
            return VectorTokenStream::new(Vec::new());
        }
        VectorTokenStream::new(vec![Token {
            offset_from: 0,
            offset_to: text.len(),
            position: 0,
            text: exact_token(text),
            position_length: 1,
        }])
    }
}

/// Path/identifier tokenizer: whole token plus separator-split sub-tokens.
#[derive(Clone, Debug, Default)]
struct IdentifierTokenizer;

impl Tokenizer for IdentifierTokenizer {
    type TokenStream<'a> = VectorTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> VectorTokenStream {
        let tokens: Vec<Token> = identifier_token_spans(text)
            .into_iter()
            .enumerate()
            .map(|(position, (from, to, value))| Token {
                offset_from: from,
                offset_to: to,
                position,
                text: value,
                position_length: 1,
            })
            .collect();
        VectorTokenStream::new(tokens)
    }
}

/// Natural-language tokenizer: word segmentation, lowercase, umlaut folding.
#[derive(Clone, Debug, Default)]
struct TextTokenizer;

impl Tokenizer for TextTokenizer {
    type TokenStream<'a> = VectorTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> VectorTokenStream {
        let mut tokens: Vec<Token> = Vec::new();
        for (position, (start, end)) in word_spans(text).into_iter().enumerate() {
            push_text_token(&mut tokens, start, end, &text[start..end], position);
        }
        VectorTokenStream::new(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_token_trims_and_lowercases() {
        assert_eq!(exact_token("  Quarterly Report "), "quarterly report");
        assert_eq!(exact_token(""), "");
    }

    #[test]
    fn identifier_tokens_emit_whole_and_parts() {
        assert_eq!(
            identifier_tokens("docs/Q1/quarterly-report.pdf"),
            vec![
                "docs/q1/quarterly-report.pdf".to_string(),
                "quarterly-report".to_string(),
                "docs".to_string(),
                "q1".to_string(),
                "quarterly".to_string(),
                "report".to_string(),
                "pdf".to_string(),
            ]
        );
    }

    #[test]
    fn identifier_tokens_emit_extension_stripped_stem() {
        assert_eq!(
            identifier_tokens("md-blocks-spec.md"),
            vec![
                "md-blocks-spec.md".to_string(),
                "md-blocks-spec".to_string(),
                "md".to_string(),
                "blocks".to_string(),
                "spec".to_string(),
            ]
        );
        assert_eq!(
            identifier_tokens("documentation/md-blocks/md-blocks-spec.md"),
            vec![
                "documentation/md-blocks/md-blocks-spec.md".to_string(),
                "md-blocks-spec".to_string(),
                "documentation".to_string(),
                "md".to_string(),
                "blocks".to_string(),
                "spec".to_string(),
            ]
        );
        assert_eq!(
            identifier_tokens("md-blocks-spec"),
            vec![
                "md-blocks-spec".to_string(),
                "md".to_string(),
                "blocks".to_string(),
                "spec".to_string(),
            ],
            "a value without a file extension emits no extra token"
        );
    }

    #[test]
    fn text_tokens_emit_one_canonical_token_per_word() {
        let tokens = text_tokens("Förderung und Programme");
        assert_eq!(
            tokens,
            vec![
                "foerderung".to_string(),
                "und".to_string(),
                "programme".to_string(),
            ],
            "umlauts fold into the single canonical token, not a second term"
        );
        assert_eq!(
            text_tokens("Foerderprogramm"),
            text_tokens("Förderprogramm")
        );
        assert_eq!(text_tokens("STRASSE"), text_tokens("Straße"));
    }

    #[test]
    fn body_parts_cover_both_directions() {
        let compound = extract_body_parts("Datenschutzrichtlinie");
        assert!(compound.contains("datenschutz"), "parts: {compound}");
        assert!(compound.contains("daten"), "parts: {compound}");
        assert!(
            compound.split(' ').any(|part| part == "schutz"),
            "nested part missing: {compound}"
        );

        let component = extract_body_parts("Daten Schutz");
        assert!(component.contains("daten"), "parts: {component}");
        assert!(component.contains("schutz"), "parts: {component}");

        let foerderung = extract_body_parts("Förderung für Programme");
        assert!(foerderung.contains("foerder"), "parts: {foerderung}");
        assert!(foerderung.contains("programm"), "parts: {foerderung}");
    }

    #[test]
    fn vector_token_stream_is_panic_free_before_advance() {
        let mut stream = VectorTokenStream::new(Vec::new());
        assert!(!stream.advance());
        assert!(stream.token().text.is_empty());
    }
}
