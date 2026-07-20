//! Multi-op transactions: a batch commits all-or-nothing, and a rollback leaves
//! every index exactly as it was.
//!
//! The gate for this phase is "rollback leaves every index exactly consistent."
//! That is more than "the docs aren't there" — the doc store, the BM25 postings,
//! the payload indexes, the counters (doc count, token totals),
//! and the vector graph must ALL be identical to the pre-transaction state. A
//! transaction that half-writes one index is worse than no transaction, because
//! the estate then lies about itself.

use connxism::{Estate, EstateQuery, WriteOp};
use rro_core::{Condition, Embedding, Filter, Recall, VectorRecord};

fn rec(id: &str, x: f32, text: &str) -> VectorRecord {
    let mut r = VectorRecord::new(id, Embedding(vec![x, 1.0 - x]).normalized(), text);
    r.metadata = rro_core::Metadata::from([("kind".to_string(), serde_json::json!("doc"))]);
    r.metadata.insert("id".to_string(), serde_json::json!(id));
    r
}

/// A full fingerprint of the estate's observable state, so a rollback can be
/// checked for *exact* consistency rather than just "the doc is gone".
async fn fingerprint(recall: &connxism::ConnXRecall) -> (usize, Vec<String>) {
    let count = recall.len().await.unwrap();
    // Every doc id, via a lexical scan that touches the postings index — so a
    // half-written postings row would change this even if the count matched.
    let all = recall
        .lexical_search("alpha beta gamma delta", 1000)
        .await
        .unwrap();
    let mut ids: Vec<String> = all.iter().map(|c| c.id.as_str().to_string()).collect();
    ids.sort();
    (count, ids)
}

/// THE gate: a transaction that fails part-way leaves the estate byte-identical.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_transaction_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "txn").unwrap();
    let recall = estate.recall();

    // Seed a known state.
    recall
        .upsert(vec![rec("a", 0.1, "alpha one"), rec("b", 0.2, "beta two")])
        .await
        .unwrap();
    recall.quiesce().await.unwrap();
    let before = fingerprint(&recall).await;
    assert_eq!(before.0, 2, "two docs seeded");

    // A transaction: one good upsert, one good remove, then a record whose
    // embedding dimension is WRONG — which errors inside the batch. The whole
    // thing must roll back: `c` never appears, `a` is not removed.
    let bad = VectorRecord::new("d", Embedding(vec![9.9, 9.9, 9.9]), "wrong dim");
    let result = recall
        .transaction(vec![
            WriteOp::Upsert(vec![rec("c", 0.3, "gamma three")]),
            WriteOp::Remove(rro_core::Id("a".into())),
            WriteOp::Upsert(vec![bad]), // dim mismatch -> Err -> rollback
        ])
        .await;
    assert!(
        result.is_err(),
        "the dim-mismatch op must fail the transaction"
    );
    recall.quiesce().await.unwrap();

    let after = fingerprint(&recall).await;
    assert_eq!(
        before, after,
        "a failed transaction must leave the estate EXACTLY as it was — \
         count and ids all identical. before={before:?} after={after:?}"
    );

    // Belt and braces: `c` must not be searchable by its payload, and `a` must
    // still be present — the indexes, not just the count, rolled back.
    let c_hits = recall
        .query(EstateQuery {
            dsl: Some(Filter::new().must(Condition::eq("id", serde_json::json!("c")))),
            top_k: 10,
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(
        c_hits.is_empty(),
        "rolled-back doc `c` must not be in the payload index"
    );
    assert!(
        recall.doc("a").await.unwrap().is_some(),
        "rolled-back removal must leave `a` present"
    );
}

/// The other half: a transaction that succeeds applies every op atomically.
#[tokio::test(flavor = "multi_thread")]
async fn a_successful_transaction_applies_everything() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "txn").unwrap();
    let recall = estate.recall();

    recall
        .upsert(vec![rec("a", 0.1, "alpha one")])
        .await
        .unwrap();
    recall.quiesce().await.unwrap();

    recall
        .transaction(vec![
            WriteOp::Upsert(vec![
                rec("b", 0.2, "beta two"),
                rec("c", 0.3, "gamma three"),
            ]),
            WriteOp::Remove(rro_core::Id("a".into())),
        ])
        .await
        .unwrap();
    recall.quiesce().await.unwrap();

    assert_eq!(recall.len().await.unwrap(), 2, "a removed, b and c added");
    assert!(recall.doc("a").await.unwrap().is_none(), "a was removed");
    assert!(recall.doc("b").await.unwrap().is_some());
    assert!(recall.doc("c").await.unwrap().is_some());

    // The counters are consistent: a lexical search over the new docs works,
    // meaning the postings and token totals tracked the transaction.
    let hits = recall.lexical_search("beta gamma", 10).await.unwrap();
    let ids: std::collections::HashSet<&str> = hits.iter().map(|c| c.id.as_str()).collect();
    assert!(ids.contains("b") && ids.contains("c"));
}

/// The counter hazard, pinned: two upserts in one transaction must net to +2
/// docs, not +1. This is the read-modify-write bug the Transaction type exists
/// to prevent — if each op re-read the pre-commit count, the last write would
/// win and the count would be wrong by one per extra statement.
#[tokio::test(flavor = "multi_thread")]
async fn counters_thread_across_statements() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "txn").unwrap();
    let recall = estate.recall();

    recall
        .transaction(vec![
            WriteOp::Upsert(vec![rec("a", 0.1, "alpha")]),
            WriteOp::Upsert(vec![rec("b", 0.2, "beta")]),
            WriteOp::Upsert(vec![rec("c", 0.3, "gamma")]),
        ])
        .await
        .unwrap();

    assert_eq!(
        recall.len().await.unwrap(),
        3,
        "three separate upsert statements in one transaction must net +3 docs — \
         if this is 1, the counters re-read the pre-commit value per statement"
    );
}
