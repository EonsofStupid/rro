//! Property-based invariants for the deterministic embedder.

use embedder::DeterministicEmbedder;
use proptest::prelude::*;
use rro_core::Embedder;

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(f)
}

proptest! {
    /// Same input, same embedding — always. Determinism is the contract.
    #[test]
    fn deterministic(s in ".{0,120}") {
        let e = DeterministicEmbedder::new();
        let a = block_on(e.embed_one(&s)).unwrap();
        let b = block_on(e.embed_one(&s)).unwrap();
        prop_assert_eq!(&a, &b);
        prop_assert_eq!(a.dim(), DeterministicEmbedder::DEFAULT_DIM);
    }

    /// Embeddings are unit-norm (or zero for content-free text).
    #[test]
    fn unit_or_zero_norm(s in ".{0,120}") {
        let n = block_on(DeterministicEmbedder::new().embed_one(&s)).unwrap().norm();
        prop_assert!(n < 1e-3 || (n - 1.0).abs() < 1e-3, "norm not 0 or 1: {n}");
    }

    /// Batch embedding preserves order and matches one-at-a-time embedding.
    #[test]
    fn batch_matches_single(texts in prop::collection::vec(".{0,60}", 0..8)) {
        let e = DeterministicEmbedder::new();
        let batch = block_on(e.embed(&texts)).unwrap();
        prop_assert_eq!(batch.len(), texts.len());
        for (i, t) in texts.iter().enumerate() {
            let one = block_on(e.embed_one(t)).unwrap();
            prop_assert_eq!(&batch[i], &one);
        }
    }
}
