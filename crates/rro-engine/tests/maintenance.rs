//! Sprint 25 gates: flush and compact answer over live TCP, fsync estates
//! stay exact, per-CF sizes surface in health, and every query path is
//! exact after a full compaction pass.

use std::sync::Arc;

use rro_client::Client;
use rro_core::{Condition, Embedding, EstateQuery, Filter, Recall, VectorRecord};
use rro_engine::{FlowNode, ReasonReadyObject};
use rro_net::tcp;

fn rec(id: &str, seed: f32, team: &str) -> VectorRecord {
    let mut r = VectorRecord::new(
        id,
        Embedding(vec![seed, 1.0 - seed, 0.5, 0.25]),
        format!("maintenance corpus entry {id}"),
    );
    r.metadata.insert("team".into(), serde_json::json!(team));
    r
}

#[tokio::test(flavor = "multi_thread")]
async fn flush_and_compact_over_the_wire_with_exact_queries_after() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "mt").unwrap());
    estate.create_payload_index("team").unwrap();
    let recall = estate.recall();

    // Seed, then churn: overwrite half, remove a quarter — garbage for
    // compaction to chew on.
    let seed_batch: Vec<VectorRecord> = (0..200)
        .map(|i| {
            rec(
                &format!("d{i:03}"),
                i as f32 / 200.0,
                if i % 2 == 0 { "red" } else { "blue" },
            )
        })
        .collect();
    recall.upsert(seed_batch).await.unwrap();
    for i in (0..200).step_by(2) {
        recall
            .upsert(vec![rec(&format!("d{i:03}"), 0.5, "green")])
            .await
            .unwrap();
    }
    for i in (0..200).step_by(4) {
        recall
            .remove(&format!("d{i:03}").as_str().into())
            .await
            .unwrap();
    }
    recall.quiesce().await.unwrap();

    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = FlowNode::new(flow, "mt-node").with_estate(estate.clone());
    let (addr, _task) = tcp::serve("127.0.0.1:0", Arc::new(node)).await.unwrap();
    let client = Client::new(addr.to_string());

    // Ground truth before maintenance.
    let truth_green = estate
        .ids_where(&Filter::default().must(Condition::eq("team", serde_json::json!("green"))))
        .unwrap()
        .unwrap();
    let truth_df = estate.term_df("maintenance").unwrap();
    let truth_count = recall.len().await.unwrap();
    assert_eq!(truth_count, 150);

    // Unknown verbs reply with an error instead of hanging the client.
    let err = client.sample(1, 1).await; // known verb sanity first
    assert!(err.is_ok());
    let unknown = rro_net::tcp::request(
        addr,
        &rro_net::Message::request("t", "mt-node", "no_such_verb", serde_json::json!({})),
    )
    .await
    .unwrap();
    assert!(unknown.body["error"]
        .as_str()
        .unwrap()
        .contains("unknown verb"));

    // Flush over the wire.
    client.flush().await.unwrap();

    // Compact over the wire; sizes come back for every CF.
    let sizes = client.compact().await.unwrap();
    assert_eq!(sizes.len(), connxism::COLUMN_FAMILY_COUNT);
    let docs_bytes: u64 = sizes
        .iter()
        .find(|(n, _)| n == "docs")
        .map(|(_, b)| *b)
        .unwrap();
    assert!(docs_bytes > 0, "flushed+compacted docs CF has SST bytes");

    // Health surfaces the same sizes.
    let health = client.health().await.unwrap();
    assert!(
        health["estate"]["cf_bytes"]
            .as_array()
            .map(|a| a.len() == sizes.len())
            .unwrap_or(false),
        "cf_bytes in health"
    );

    // Every query path exact after compaction.
    assert_eq!(recall.len().await.unwrap(), truth_count);
    assert_eq!(
        estate
            .ids_where(&Filter::default().must(Condition::eq("team", serde_json::json!("green"))))
            .unwrap()
            .unwrap(),
        truth_green
    );
    assert_eq!(estate.term_df("maintenance").unwrap(), truth_df);
    let hits = recall
        .query(EstateQuery::hybrid(
            "maintenance corpus entry",
            Embedding(vec![0.3, 0.7, 0.5, 0.25]),
            10,
        ))
        .await
        .unwrap();
    assert_eq!(hits.len(), 10);
    assert!(
        recall.doc("d000").await.unwrap().is_none(),
        "removed stays removed"
    );
    assert!(recall.doc("d001").await.unwrap().is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn fsync_estates_accept_writes_and_stay_exact() {
    let dir = tempfile::tempdir().unwrap();
    let estate = connxism::Estate::open_with(
        dir.path(),
        "fs",
        connxism::EstateConfig {
            fsync_writes: true,
            ..connxism::EstateConfig::default()
        },
    )
    .unwrap();
    let recall = estate.recall();
    recall
        .upsert(vec![rec("a", 0.2, "x"), rec("b", 0.8, "y")])
        .await
        .unwrap();
    recall.remove(&"b".into()).await.unwrap();
    assert_eq!(recall.len().await.unwrap(), 1);
    let hits = recall
        .query(EstateQuery::hybrid(
            "maintenance corpus entry",
            Embedding(vec![0.2, 0.8, 0.5, 0.25]),
            5,
        ))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.as_str(), "a");
}
