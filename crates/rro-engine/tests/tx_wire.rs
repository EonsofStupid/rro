//! The `tx` verb: atomic transactions over the a2a wire.
//!
//! Phase 5b built `ConnXRecall::transaction` (multi-op, verified rollback). This
//! is the same primitive made network-accessible: a client sends a sequence of
//! upserts and removes and the node commits them all or none. The upserts are
//! embedded server-side *before* the transaction opens, so a model failure aborts
//! before any durable write.

use std::sync::Arc;

use rro_client::Client;
use rro_core::Document;
use rro_engine::{FlowNode, ReasonReadyObject};
use rro_net::tcp;

async fn node(estate: Arc<connxism::Estate>) -> Client {
    let flow = Arc::new(
        ReasonReadyObject::builder()
            .recall(Arc::new(estate.recall()))
            .build(),
    );
    let node = FlowNode::new(flow, "tx-node").with_estate(estate);
    let (addr, task) = tcp::serve("127.0.0.1:0", Arc::new(node)).await.unwrap();
    std::mem::forget(task);
    Client::new(addr.to_string())
}

fn doc(id: &str, text: &str) -> Document {
    Document::new(text).with_id(id)
}

/// A successful transaction commits every op, atomically, over the wire.
#[tokio::test(flavor = "multi_thread")]
async fn tx_commits_a_mixed_batch() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "txw").unwrap());
    let client = node(estate.clone()).await;

    // Seed one doc via the same verb.
    client
        .transaction(serde_json::json!([{ "upsert": [doc("a", "alpha one")] }]))
        .await
        .unwrap();

    // A batch: add two, remove the first — as one unit.
    let committed = client
        .transaction(serde_json::json!([
            { "upsert": [doc("b", "beta two"), doc("c", "gamma three")] },
            { "remove": "a" },
        ]))
        .await
        .unwrap();
    assert_eq!(committed, 2, "two ops committed");

    rro_core::Recall::quiesce(&estate.recall()).await.unwrap();
    let recall = estate.recall();
    assert!(
        rro_core::Recall::len(&recall).await.unwrap() == 2,
        "a removed, b+c added"
    );
    assert!(recall.doc("a").await.unwrap().is_none());
    assert!(recall.doc("b").await.unwrap().is_some());
    assert!(recall.doc("c").await.unwrap().is_some());
}

/// A malformed op rolls the whole wire transaction back — nothing lands.
#[tokio::test(flavor = "multi_thread")]
async fn tx_rolls_back_on_a_bad_op() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "txw").unwrap());
    let client = node(estate.clone()).await;

    client
        .transaction(serde_json::json!([{ "upsert": [doc("keep", "keep me")] }]))
        .await
        .unwrap();
    rro_core::Recall::quiesce(&estate.recall()).await.unwrap();
    let before = {
        let r = estate.recall();
        rro_core::Recall::len(&r).await.unwrap()
    };

    // Second op is neither upsert nor remove -> the verb rejects the whole tx.
    let result = client
        .transaction(serde_json::json!([
            { "upsert": [doc("ghost", "should not persist")] },
            { "bogus": true },
        ]))
        .await;
    assert!(result.is_err(), "a malformed op must fail the transaction");

    rro_core::Recall::quiesce(&estate.recall()).await.unwrap();
    let recall = estate.recall();
    assert_eq!(
        rro_core::Recall::len(&recall).await.unwrap(),
        before,
        "a rejected transaction must leave the estate unchanged"
    );
    assert!(
        recall.doc("ghost").await.unwrap().is_none(),
        "the ghost upsert from a rolled-back tx must not persist"
    );
}
