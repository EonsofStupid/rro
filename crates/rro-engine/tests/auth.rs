//! P5 gate: a2a capability tokens — a token-bearing node refuses non-bearers
//! for every verb except the `ping` liveness probe.

use std::sync::Arc;

use rro_engine::{FlowNode, ReasonReadyObject};
use rro_net::{tcp, Message};

#[tokio::test(flavor = "multi_thread")]
async fn token_gated_node_refuses_non_bearers() {
    let flow = Arc::new(ReasonReadyObject::default_engine());
    flow.index(rro_engine::sample_corpus()).await.unwrap();
    let node = Arc::new(FlowNode::new(flow, "guarded").with_token("s3cret"));
    let (addr, _task) = tcp::serve("127.0.0.1:0", node).await.unwrap();

    // ping stays open: the liveness probe needs no capability.
    let pong = tcp::request(
        addr,
        &Message::request("c", "guarded", "ping", serde_json::json!({})),
    )
    .await
    .unwrap();
    assert_eq!(pong.body["pong"], serde_json::json!(true));

    // ask without the token: refused.
    let refused = tcp::request(
        addr,
        &Message::request(
            "c",
            "guarded",
            "ask",
            serde_json::json!({"query": "postgres"}),
        ),
    )
    .await
    .unwrap();
    assert_eq!(refused.body["error"], serde_json::json!("unauthorized"));

    // wrong token: refused.
    let wrong = tcp::request(
        addr,
        &Message::request(
            "c",
            "guarded",
            "ask",
            serde_json::json!({"query": "postgres"}),
        )
        .with_token("nope"),
    )
    .await
    .unwrap();
    assert_eq!(wrong.body["error"], serde_json::json!("unauthorized"));

    // bearer: answered, full pipeline.
    let ok = tcp::request(
        addr,
        &Message::request(
            "c",
            "guarded",
            "ask",
            serde_json::json!({"query": "postgres upgrade"}),
        )
        .with_token("s3cret"),
    )
    .await
    .unwrap();
    assert!(
        ok.body["candidates"]
            .as_array()
            .map(|c| !c.is_empty())
            .unwrap_or(false),
        "bearer gets real answers: {}",
        ok.body
    );
}
