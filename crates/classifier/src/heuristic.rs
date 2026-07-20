//! The weightless heuristic readiness judge.
//!
//! It answers one question: *is the retrieved context enough to reason on?* It
//! does so from query-term coverage across the top candidates — no weights, no
//! network. It is the honest default the DevPULSE classifier will replace with
//! a learned judgment.

use std::collections::HashSet;

use async_trait::async_trait;
use rro_core::text::content_tokens;
use rro_core::{Candidate, Classifier, Readiness, Result};

/// Coverage-based readiness classifier.
#[derive(Debug, Clone)]
pub struct HeuristicClassifier {
    /// How many top candidates to consider as the reasoning context.
    top_context: usize,
    /// Union coverage at/above this is `ready`.
    ready_at: f32,
    /// Union coverage at/above this (but below `ready_at`) is `partial`.
    partial_at: f32,
}

impl Default for HeuristicClassifier {
    fn default() -> Self {
        HeuristicClassifier {
            top_context: 5,
            ready_at: 0.5,
            partial_at: 0.25,
        }
    }
}

impl HeuristicClassifier {
    /// Construct with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the coverage thresholds.
    pub fn with_thresholds(ready_at: f32, partial_at: f32) -> Self {
        HeuristicClassifier {
            top_context: 5,
            ready_at,
            partial_at,
        }
    }
}

#[async_trait]
impl Classifier for HeuristicClassifier {
    async fn classify(&self, query: &str, context: &[Candidate]) -> Result<Readiness> {
        let q_terms: HashSet<String> = content_tokens(query).into_iter().collect();
        if context.is_empty() {
            return Ok(Readiness::not_ready(
                0.0,
                "no_context",
                "recall returned no candidates",
            ));
        }
        if q_terms.is_empty() {
            return Ok(Readiness::not_ready(
                0.0,
                "empty_query",
                "query has no content terms to satisfy",
            ));
        }

        let k = self.top_context.min(context.len());
        let mut covered: HashSet<&String> = HashSet::new();
        for cand in &context[..k] {
            let toks: HashSet<String> = content_tokens(&cand.text).into_iter().collect();
            for q in &q_terms {
                if toks.contains(q) {
                    covered.insert(q);
                }
            }
        }

        let coverage = covered.len() as f32 / q_terms.len() as f32;
        let rationale = format!(
            "{}/{} query terms covered across top {}",
            covered.len(),
            q_terms.len(),
            k
        );

        let readiness = if coverage >= self.ready_at {
            Readiness::ready(coverage, rationale)
        } else if coverage >= self.partial_at {
            Readiness::not_ready(coverage, "partial", rationale)
        } else {
            Readiness::not_ready(coverage, "insufficient", rationale)
        };
        Ok(readiness)
    }

    fn model_name(&self) -> &str {
        "heuristic-coverage"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn full_coverage_is_ready() {
        let c = HeuristicClassifier::new();
        let ctx = vec![Candidate::new(
            "1",
            "postgres migration upgrade rollback steps",
            1.0,
        )];
        let r = c
            .classify("postgres migration upgrade", &ctx)
            .await
            .unwrap();
        assert!(r.ready);
    }

    #[tokio::test]
    async fn no_context_is_not_ready() {
        let c = HeuristicClassifier::new();
        let r = c.classify("anything", &[]).await.unwrap();
        assert!(!r.ready);
        assert_eq!(r.label, "no_context");
    }
}
