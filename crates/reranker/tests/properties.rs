//! Property-based invariants for the lexical reranker.

use proptest::prelude::*;
use reranker::LexicalReranker;
use rro_core::{Candidate, Reranker};
use std::collections::HashSet;

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(f)
}

proptest! {
    /// Rerank returns a score-descending sub-multiset of its input, truncated
    /// to top_k, and never fabricates a candidate.
    #[test]
    fn rerank_is_sorted_subset(
        texts in prop::collection::vec("[a-z ]{0,40}", 0..15),
        query in "[a-z ]{0,24}",
        k in 0usize..20,
    ) {
        let cands: Vec<Candidate> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| Candidate::new(format!("c{i}"), t.clone(), 0.0))
            .collect();
        let in_ids: HashSet<String> = cands.iter().map(|c| c.id.as_str().to_string()).collect();
        let n = cands.len();

        let out = block_on(LexicalReranker::new().rerank(&query, cands, k)).unwrap();

        prop_assert!(out.len() <= k.min(n));
        for c in &out {
            prop_assert!(in_ids.contains(c.id.as_str()), "fabricated candidate id");
        }
        for w in out.windows(2) {
            prop_assert!(w[0].score >= w[1].score, "results not sorted descending");
        }
    }
}
