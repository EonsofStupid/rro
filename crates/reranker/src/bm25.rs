//! The lexical (Okapi BM25) default reranker.
//!
//! Weightless and deterministic. It re-scores recall candidates by BM25 of the
//! query against the candidate set as an ad-hoc corpus, which sharpens ordering
//! beyond raw vector cosine. It is the honest floor the DevPULSE (Nemotron)
//! reranker must beat.

use std::collections::HashMap;

use async_trait::async_trait;
use rro_core::text::content_tokens;
use rro_core::{Candidate, Reranker, Result};

/// Okapi BM25 reranker.
#[derive(Debug, Clone)]
pub struct LexicalReranker {
    k1: f32,
    b: f32,
}

impl Default for LexicalReranker {
    fn default() -> Self {
        LexicalReranker { k1: 1.2, b: 0.75 }
    }
}

impl LexicalReranker {
    /// BM25 with standard parameters (k1 = 1.2, b = 0.75).
    pub fn new() -> Self {
        Self::default()
    }

    /// BM25 with custom parameters.
    pub fn with_params(k1: f32, b: f32) -> Self {
        LexicalReranker { k1, b }
    }
}

#[async_trait]
impl Reranker for LexicalReranker {
    async fn rerank(
        &self,
        query: &str,
        mut candidates: Vec<Candidate>,
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        let q_terms = content_tokens(query);
        if candidates.is_empty() || q_terms.is_empty() {
            candidates.truncate(top_k);
            return Ok(candidates);
        }

        // Per-candidate tokenization + document frequencies across the set.
        let docs: Vec<Vec<String>> = candidates.iter().map(|c| content_tokens(&c.text)).collect();
        let n = docs.len() as f32;
        let avgdl = (docs.iter().map(|d| d.len()).sum::<usize>() as f32 / n).max(1.0);

        let mut df: HashMap<&str, u32> = HashMap::new();
        for term in q_terms.iter().map(String::as_str) {
            let count = docs.iter().filter(|d| d.iter().any(|t| t == term)).count() as u32;
            df.insert(term, count);
        }

        for (cand, doc) in candidates.iter_mut().zip(&docs) {
            let dl = doc.len() as f32;
            let mut tf: HashMap<&str, f32> = HashMap::new();
            for t in doc {
                *tf.entry(t.as_str()).or_insert(0.0) += 1.0;
            }
            let mut score = 0.0f32;
            for term in q_terms.iter().map(String::as_str) {
                let f = *tf.get(term).unwrap_or(&0.0);
                if f == 0.0 {
                    continue;
                }
                let n_q = *df.get(term).unwrap_or(&0) as f32;
                // BM25 idf with the +0.5 smoothing; clamp to >= 0.
                let idf = (((n - n_q + 0.5) / (n_q + 0.5)) + 1.0).ln().max(0.0);
                let denom = f + self.k1 * (1.0 - self.b + self.b * dl / avgdl);
                score += idf * (f * (self.k1 + 1.0)) / denom;
            }
            cand.score = score;
        }

        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        candidates.truncate(top_k);
        Ok(candidates)
    }

    fn model_name(&self) -> &str {
        "bm25-lexical"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ranks_by_term_overlap() {
        let rr = LexicalReranker::new();
        let cands = vec![
            Candidate::new("1", "the migration guide for postgres upgrades", 0.5),
            Candidate::new("2", "a recipe for banana bread", 0.5),
            Candidate::new("3", "postgres upgrade migration steps and rollback", 0.5),
        ];
        let out = rr
            .rerank("postgres migration upgrade", cands, 3)
            .await
            .unwrap();
        assert_eq!(out[0].id.as_str(), "3");
        assert_eq!(out.last().unwrap().id.as_str(), "2");
    }
}
