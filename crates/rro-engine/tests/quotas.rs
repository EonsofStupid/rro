//! Sprint 26 gates: every quota rejects exactly at its boundary with the
//! typed error (one-under passes, one-over fails), health carries the
//! configured limits, and the wire surfaces refusals cleanly.

use std::sync::Arc;

use rro_client::Client;
use rro_core::{Embedding, EstateQuery, Recall, RroError, VectorRecord};
use rro_engine::{FlowNode, ReasonReadyObject};
use rro_net::tcp;

fn rec(id: &str, seed: f32) -> VectorRecord {
    VectorRecord::new(
        id,
        Embedding(vec![seed, 0.5, 0.25, 0.125]),
        format!("quota corpus {id}"),
    )
}

fn quota_estate(dir: &std::path::Path) -> connxism::Estate {
    connxism::Estate::open_with(
        dir,
        "q",
        connxism::EstateConfig {
            quotas: connxism::Quotas {
                max_docs: Some(10),
                max_payload_bytes: Some(256),
                max_top_k: Some(50),
                max_batch: Some(5),
            },
            ..connxism::EstateConfig::default()
        },
    )
    .unwrap()
}

fn is_quota(e: &RroError) -> bool {
    matches!(e, RroError::Quota(_))
}

#[tokio::test(flavor = "multi_thread")]
async fn each_quota_rejects_exactly_at_the_boundary() {
    let dir = tempfile::tempdir().unwrap();
    let estate = quota_estate(dir.path());
    let recall = estate.recall();

    // max_batch: 5 passes, 6 rejects (typed).
    let five: Vec<VectorRecord> = (0..5).map(|i| rec(&format!("b{i}"), 0.1)).collect();
    recall.upsert(five).await.unwrap();
    let six: Vec<VectorRecord> = (0..6).map(|i| rec(&format!("c{i}"), 0.1)).collect();
    let err = recall.upsert(six).await.unwrap_err();
    assert!(is_quota(&err), "{err}");

    // max_payload_bytes: under passes, over rejects.
    let mut small = rec("p_ok", 0.2);
    small
        .metadata
        .insert("note".into(), serde_json::json!("x".repeat(100)));
    recall.upsert(vec![small]).await.unwrap();
    let mut big = rec("p_no", 0.2);
    big.metadata
        .insert("note".into(), serde_json::json!("x".repeat(300)));
    let err = recall.upsert(vec![big]).await.unwrap_err();
    assert!(is_quota(&err), "{err}");

    // max_docs = 10: we hold 6; four more pass (exactly at cap)…
    let four: Vec<VectorRecord> = (0..4).map(|i| rec(&format!("d{i}"), 0.3)).collect();
    recall.upsert(four).await.unwrap();
    assert_eq!(recall.len().await.unwrap(), 10);
    // …the eleventh rejects; overwrites (not net-new) still pass.
    let err = recall.upsert(vec![rec("d_new", 0.4)]).await.unwrap_err();
    assert!(is_quota(&err), "{err}");
    recall.upsert(vec![rec("d0", 0.9)]).await.unwrap(); // overwrite ok
    assert_eq!(recall.len().await.unwrap(), 10);

    // max_top_k: 50 passes, 51 rejects.
    let q =
        |k: usize| EstateQuery::hybrid("quota corpus", Embedding(vec![0.1, 0.5, 0.25, 0.125]), k);
    assert!(recall.query(q(50)).await.is_ok());
    let err = recall.query(q(51)).await.unwrap_err();
    assert!(is_quota(&err), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn health_reports_limits_and_wire_refuses_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(quota_estate(dir.path()));
    estate.recall().upsert(vec![rec("a", 0.2)]).await.unwrap();

    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = FlowNode::new(flow, "q-node").with_estate(estate);
    let (addr, _task) = tcp::serve("127.0.0.1:0", Arc::new(node)).await.unwrap();
    let client = Client::new(addr.to_string());

    // Health carries the configured limits.
    let health = client.health().await.unwrap();
    assert_eq!(health["estate"]["quotas"]["max_top_k"], 50);
    assert_eq!(health["estate"]["quotas"]["max_docs"], 10);

    // An over-limit wire query returns a clean typed refusal — the
    // connection is answered, not dropped.
    let err = client
        .query(&EstateQuery::text("quota corpus", 51))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("quota"), "{err}");
    // The node is still healthy afterward (same client, new request).
    assert!(client.ping().await.unwrap());
}
