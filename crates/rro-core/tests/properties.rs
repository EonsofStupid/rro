//! Property-based invariants for the core vector math.

use proptest::prelude::*;
use rro_core::Embedding;

proptest! {
    /// Cosine similarity is always within [-1, 1] (allowing float slack).
    #[test]
    fn cosine_within_bounds(
        a in prop::collection::vec(-100.0f32..100.0, 1..48),
        b in prop::collection::vec(-100.0f32..100.0, 1..48),
    ) {
        let n = a.len().min(b.len());
        let ea = Embedding(a[..n].to_vec());
        let eb = Embedding(b[..n].to_vec());
        let c = ea.cosine(&eb);
        prop_assert!((-1.001..=1.001).contains(&c), "cosine out of bounds: {c}");
    }

    /// A normalized vector has norm 0 (only if it was all-zero) or ~1.
    #[test]
    fn normalized_is_unit_or_zero(v in prop::collection::vec(-100.0f32..100.0, 1..48)) {
        let e = Embedding(v).normalized();
        let n = e.norm();
        prop_assert!(n < 1e-3 || (n - 1.0).abs() < 1e-3, "norm not 0 or 1: {n}");
    }

    /// Dot product is symmetric.
    #[test]
    fn dot_is_symmetric(v in prop::collection::vec(-10.0f32..10.0, 1..24)) {
        let e = Embedding(v);
        let f = Embedding(e.0.iter().rev().copied().collect());
        prop_assert!((e.dot(&f) - f.dot(&e)).abs() < 1e-3);
    }
}
