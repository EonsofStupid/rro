//! Property-based invariants for the recall store.

use proptest::prelude::*;
use recall::FlatRecall;
use rro_core::{Embedding, Recall, VectorRecord};

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(f)
}

proptest! {
    /// After upserting N distinct ids, `len` is N; search returns a
    /// score-descending list no longer than min(k, N).
    #[test]
    fn search_is_sorted_and_bounded(
        vecs in prop::collection::vec(prop::collection::vec(-1.0f32..1.0, 4..=4), 1..24),
        k in 1usize..30,
    ) {
        let store = FlatRecall::new();
        let records: Vec<VectorRecord> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| VectorRecord::new(format!("id{i}"), Embedding(v.clone()), format!("t{i}")))
            .collect();
        let n = records.len();
        block_on(store.upsert(records)).unwrap();

        prop_assert_eq!(block_on(store.len()).unwrap(), n);

        let res = block_on(store.search(&Embedding(vecs[0].clone()), k)).unwrap();
        prop_assert!(res.len() <= k.min(n));
        for w in res.windows(2) {
            prop_assert!(w[0].score >= w[1].score, "results not sorted descending");
        }
    }

    /// Re-upserting the same id overwrites rather than duplicating.
    #[test]
    fn upsert_is_idempotent_by_id(v in prop::collection::vec(-1.0f32..1.0, 4..=4)) {
        let store = FlatRecall::new();
        block_on(store.upsert(vec![VectorRecord::new("x", Embedding(v.clone()), "a")])).unwrap();
        block_on(store.upsert(vec![VectorRecord::new("x", Embedding(v), "b")])).unwrap();
        prop_assert_eq!(block_on(store.len()).unwrap(), 1);
    }
}
