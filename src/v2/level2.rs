//! Level-2 refinement of Level-1 hybrid hits.
//!
//! Level 1 is cheap and derived: it fuses lexical and dense ranks and hydrates
//! only the winners. Level 2 is the step that is allowed to touch the
//! **authoritative** source, and it exists for two reasons:
//!
//! - **Correctness.** A document can be revoked in the source after the last
//!   sync. A derived index still believes the old ACL, so a Level-1 hit is
//!   evidence, not permission: every survivor is re-read from the source and
//!   re-checked against the caller's principals. A revoked or missing document
//!   is dropped *silently* — announcing it would disclose that it exists.
//! - **Quality.** An optional [`crate::rerank::Reranker`] scores the survivor
//!   texts against the original query, and its order is fused with the Level-1
//!   order by weighted RRF. A relevance floor turns "nothing here actually
//!   matches" into an explicit [`V2Level2Result::rejected`] instead of a list of
//!   confident-looking noise.
//!
//! Candidates dropped by the source re-check never reach the reranker, so no
//! revoked text is sent to a model and no revoked text can leak into a result.

use std::collections::HashSet;

use crate::rerank::Reranker;
use crate::sync::SourceConnector;
use crate::v2::query::{V2HybridHit, V2SearchHit};
use crate::v2::{V2Error, V2Index};

/// RRF damping constant of the Level-1 / reranker fusion.
const LEVEL2_RRF_K: f32 = 60.0;

/// Default number of survivors handed to the reranker.
pub const DEFAULT_SHORTLIST: usize = 20;

/// Default weight of the reranker order relative to the Level-1 order.
pub const DEFAULT_RERANK_WEIGHT: f32 = 0.2;

/// Default relevance floor below which a result set is rejected.
pub const DEFAULT_RELEVANCE_FLOOR: f32 = 0.02;

/// Controls of one Level-2 refinement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct V2Level2Options {
    /// How many Level-1 survivors may be refined.
    pub shortlist: usize,
    /// Weight of the reranker order in the fused order (`1.0` is the Level-1
    /// weight).
    pub rerank_weight: f32,
    /// Minimum reranker score for a result set to count as a real match.
    pub relevance_floor: Option<f32>,
}

impl Default for V2Level2Options {
    fn default() -> Self {
        Self {
            shortlist: DEFAULT_SHORTLIST,
            rerank_weight: DEFAULT_RERANK_WEIGHT,
            relevance_floor: Some(DEFAULT_RELEVANCE_FLOOR),
        }
    }
}

impl V2Level2Options {
    /// Options with every default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the survivor shortlist.
    pub fn shortlist(mut self, shortlist: usize) -> Self {
        self.shortlist = shortlist;
        self
    }

    /// Set the reranker weight.
    pub fn rerank_weight(mut self, weight: f32) -> Self {
        self.rerank_weight = weight;
        self
    }

    /// Set (or clear) the relevance floor.
    pub fn relevance_floor(mut self, floor: Option<f32>) -> Self {
        self.relevance_floor = floor;
        self
    }
}

/// One refined hit.
#[derive(Debug, Clone, PartialEq)]
pub struct V2Level2Hit {
    /// The Level-1 hit, re-ordered; `hybrid.hit.score` still holds the Level-1
    /// fused score so the two stages stay distinguishable.
    pub hybrid: V2HybridHit,
    /// Reranker score, when a reranker ran.
    pub rerank_score: Option<f32>,
    /// Authoritative text the reranker saw: the source's `indexable_text` when a
    /// connector was supplied, otherwise the hit's own excerpt/title+path.
    pub hydrated_text: Option<String>,
}

/// Result of one Level-2 refinement.
#[derive(Debug, Clone, PartialEq)]
pub struct V2Level2Result {
    /// True when the best reranker score fell below the relevance floor: the
    /// query found nothing relevant and `hits` is low-relevance evidence, not an
    /// answer.
    pub rejected: bool,
    /// Highest reranker score, when a reranker ran.
    pub best_rerank_score: Option<f32>,
    /// Survivors in refined order.
    pub hits: Vec<V2Level2Hit>,
}

impl V2Index {
    /// Refine Level-1 hits against the authoritative source and an optional
    /// reranker.
    ///
    /// Fails closed on empty principals and on a degraded watermark. With a
    /// connector, every survivor is re-read and its ACL re-checked; a document
    /// that was revoked or deleted after the last sync is dropped without a
    /// trace.
    pub fn refine_level2(
        &self,
        query_text: &str,
        principals: &[String],
        level1_hits: Vec<V2HybridHit>,
        source: Option<&dyn SourceConnector>,
        reranker: Option<&dyn Reranker>,
        options: &V2Level2Options,
    ) -> Result<V2Level2Result, V2Error> {
        self.ensure_readable(principals)?;
        if options.shortlist == 0 {
            return Err(V2Error::InvalidQuery(
                "level-2 shortlist must be greater than zero".to_string(),
            ));
        }
        if !options.rerank_weight.is_finite() || options.rerank_weight < 0.0 {
            return Err(V2Error::InvalidQuery(
                "level-2 rerank_weight must be finite and non-negative".to_string(),
            ));
        }

        let mut survivors: Vec<Survivor> =
            Vec::with_capacity(options.shortlist.min(level1_hits.len()));
        for hybrid in level1_hits.into_iter().take(options.shortlist) {
            let text = match source {
                Some(connector) => match connector.get_doc(&hybrid.hit.id)? {
                    // Gone from the source since the last sync: not evidence.
                    None => continue,
                    Some(record) => {
                        if record.id != hybrid.hit.id {
                            return Err(V2Error::Sync(crate::sync::SyncError::SourceIo(format!(
                                "connector returned record '{}' for '{}'",
                                record.id, hybrid.hit.id
                            ))));
                        }
                        if !is_visible(&record.envelope.acl, principals) {
                            // Revoked after the last sync: drop, never disclose.
                            continue;
                        }
                        record.envelope.indexable_text
                    }
                },
                None => candidate_text(&hybrid.hit),
            };
            survivors.push(Survivor { hybrid, text });
        }

        let rerank_scores = match reranker {
            Some(reranker) if !survivors.is_empty() => {
                let documents: Vec<&str> = survivors.iter().map(|s| s.text.as_str()).collect();
                let scores = reranker
                    .score(query_text, &documents)
                    .map_err(|error| V2Error::Rerank(error.to_string()))?;
                if scores.len() != documents.len() {
                    return Err(V2Error::Rerank(format!(
                        "reranker returned {} scores for {} documents",
                        scores.len(),
                        documents.len()
                    )));
                }
                if scores.iter().any(|score| !score.is_finite()) {
                    return Err(V2Error::Rerank(
                        "reranker returned a non-finite score".to_string(),
                    ));
                }
                Some(scores)
            }
            _ => None,
        };

        let mut order: Vec<(usize, f32)> = (0..survivors.len()).map(|index| (index, 0.0)).collect();
        for (index, entry) in order.iter_mut().enumerate() {
            entry.1 += 1.0 / (LEVEL2_RRF_K + (index + 1) as f32);
        }
        if let Some(scores) = &rerank_scores {
            let mut rerank_order: Vec<usize> = (0..survivors.len()).collect();
            rerank_order.sort_by(|left, right| {
                scores[*right]
                    .partial_cmp(&scores[*left])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.cmp(right))
            });
            for (rank, index) in rerank_order.into_iter().enumerate() {
                order[index].1 += options.rerank_weight / (LEVEL2_RRF_K + (rank + 1) as f32);
            }
        }
        order.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });

        let best_rerank_score = rerank_scores
            .as_ref()
            .and_then(|scores| scores.iter().copied().reduce(f32::max));
        let rejected = match (options.relevance_floor, best_rerank_score) {
            (Some(floor), Some(best)) => best < floor,
            _ => false,
        };

        let mut position: Vec<usize> = vec![0; survivors.len()];
        for (slot, (index, _)) in order.iter().enumerate() {
            position[*index] = slot;
        }
        let mut ranked: Vec<(usize, Survivor)> = survivors.into_iter().enumerate().collect();
        ranked.sort_by_key(|(index, _)| position[*index]);

        let mut hits: Vec<V2Level2Hit> = Vec::with_capacity(ranked.len());
        for (index, survivor) in ranked {
            hits.push(V2Level2Hit {
                hydrated_text: Some(survivor.text),
                hybrid: survivor.hybrid,
                rerank_score: rerank_scores.as_ref().map(|scores| scores[index]),
            });
        }
        Ok(V2Level2Result {
            rejected,
            best_rerank_score,
            hits,
        })
    }
}

/// One Level-1 hit plus the text the reranker is allowed to see.
struct Survivor {
    hybrid: V2HybridHit,
    text: String,
}

/// Fail-closed ACL test against authoritative group IDs.
fn is_visible(acl: &[String], principals: &[String]) -> bool {
    if principals.is_empty() {
        return false;
    }
    let held: HashSet<&str> = principals.iter().map(String::as_str).collect();
    acl.iter().any(|group| held.contains(group.trim()))
}

/// Reranker input text when no source can answer: the hit's own excerpt,
/// otherwise `title path`.
fn candidate_text(hit: &V2SearchHit) -> String {
    match &hit.excerpt {
        Some(excerpt) => excerpt.clone(),
        None => format!("{} {}", hit.title, hit.path).trim().to_string(),
    }
}
