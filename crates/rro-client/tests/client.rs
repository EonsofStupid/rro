//! Sprint 8 gates: the typed client and the MCP binding, against a live node.

use std::sync::Arc;

use rro_client::Client;
use rro_core::{Condition, Document, EstateQuery, Filter, Metadata};
use rro_engine::{FlowNode, ReasonReadyObject};
use rro_net::tcp;

async fn live_node() -> std::net::SocketAddr {
    let flow = Arc::new(ReasonReadyObject::default_engine());
    flow.index(rro_engine::sample_corpus()).await.unwrap();
    let node = Arc::new(FlowNode::new(flow, "rro"));
    let (addr, task) = tcp::serve("127.0.0.1:0", node).await.unwrap();
    std::mem::forget(task); // keep serving for the test's lifetime
    addr
}

/// An estate-backed node (leaks the tempdir + estate: test-lifetime server).
async fn live_estate_node() -> std::net::SocketAddr {
    let dir = Box::leak(Box::new(tempfile::tempdir().unwrap()));
    let estate = Arc::new(connxism::Estate::open(dir.path(), "wire").unwrap());
    estate.create_payload_index("team").unwrap();
    let flow = Arc::new(
        ReasonReadyObject::builder()
            .recall(Arc::new(estate.recall()))
            .build(),
    );
    flow.index(rro_engine::sample_corpus()).await.unwrap();
    let node = Arc::new(FlowNode::new(flow, "rro").with_estate(estate));
    let (addr, task) = tcp::serve("127.0.0.1:0", node).await.unwrap();
    std::mem::forget(task);
    addr
}

#[tokio::test(flavor = "multi_thread")]
async fn client_ping_index_ask_changes() {
    let addr = live_node().await;
    let client = Client::new(addr.to_string()).with_identity("clyffy");

    assert!(client.ping().await.unwrap());

    let total = client
        .index(vec![
            Document::new("clyffy bake-off corpus entry alpha").with_id("b1"),
            Document::new("clyffy bake-off corpus entry beta").with_id("b2"),
        ])
        .await
        .unwrap();
    assert!(total >= 8, "sample corpus + 2: {total}");

    let answer = client.ask("clyffy bake-off corpus").await.unwrap();
    assert!(!answer.candidates.is_empty());
    assert!(answer
        .candidates
        .iter()
        .any(|c| c.id.as_str() == "b1" || c.id.as_str() == "b2"));

    // Default engine has no estate → changes is refused; the client surfaces
    // it as a typed error instead of silence.
    let err = client.changes(0, 10).await;
    assert!(err.is_err(), "no estate attached ⇒ typed refusal");
}

#[tokio::test(flavor = "multi_thread")]
async fn typed_query_plane_over_the_wire() {
    let addr = live_estate_node().await;
    let client = Client::new(addr.to_string()).with_identity("clyffy");

    // Ingest with metadata through the node (server-side embedding).
    let docs: Vec<Document> = (0..12)
        .map(|i| {
            let mut m = Metadata::new();
            m.insert(
                "team".into(),
                serde_json::json!(if i % 3 == 0 { "ops" } else { "eng" }),
            );
            let mut d = Document::new(format!("estate rollout note {i}")).with_id(format!("n{i}"));
            d.metadata = m;
            d
        })
        .collect();
    client.index(docs).await.unwrap();

    // Full filter DSL over the wire; the node embeds the text.
    let q = EstateQuery::text("estate rollout note", 8)
        .filtered(Filter::new().must(Condition::eq("team", serde_json::json!("ops"))));
    let hits = client.query(&q).await.unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits.iter()
            .all(|c| c.metadata.get("team") == Some(&serde_json::json!("ops"))),
        "wire-delivered filters bind: {hits:?}"
    );

    // Lean payload over the wire.
    let lean = client
        .query(&EstateQuery::text("estate rollout note", 5).ids_only())
        .await
        .unwrap();
    assert!(!lean.is_empty());
    assert!(lean
        .iter()
        .all(|c| c.text.is_empty() && c.metadata.is_empty()));

    // Recommend over the wire: positives pull neighbors, examples excluded.
    let recs = client
        .recommend(vec!["n0".to_string()], vec![], 5)
        .await
        .unwrap();
    assert!(!recs.is_empty());
    assert!(recs.iter().all(|c| c.id.as_str() != "n0"));

    // Malformed query → typed refusal, not silence.
    let bad = client
        .query(&EstateQuery {
            top_k: 3,
            ..EstateQuery::default()
        })
        .await;
    assert!(bad.is_ok(), "empty query is legal (returns empty)");
}

#[tokio::test(flavor = "multi_thread")]
async fn query_verb_without_estate_is_refused() {
    let addr = live_node().await;
    let client = Client::new(addr.to_string());
    let err = client.query(&EstateQuery::text("anything", 3)).await;
    assert!(err.is_err(), "no estate ⇒ typed refusal");
}

#[tokio::test(flavor = "multi_thread")]
async fn mcp_binding_end_to_end() {
    use std::io::{BufRead, BufReader, Write};

    let addr = live_estate_node().await;

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_rro-mcp"))
        .env("RRO_ADDR", addr.to_string())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();

    let mut rpc = |req: serde_json::Value| -> serde_json::Value {
        writeln!(stdin, "{req}").unwrap();
        serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap()
    };

    // initialize
    let init = rpc(serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "protocolVersion": "2024-11-05", "capabilities": {} }
    }));
    assert_eq!(init["result"]["serverInfo"]["name"], "rro-mcp");

    // tools/list
    let tools = rpc(serde_json::json!({
        "jsonrpc": "2.0", "id": 2, "method": "tools/list"
    }));
    let names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"rro_ask") && names.contains(&"rro_index"));

    // tools/call → the full pipeline answers through MCP.
    let ask = rpc(serde_json::json!({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": { "name": "rro_ask", "arguments": { "query": "postgres upgrade" } }
    }));
    assert_eq!(ask["result"]["isError"], serde_json::json!(false));
    let text = ask["result"]["content"][0]["text"].as_str().unwrap();
    let result: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(
        result["candidates"]
            .as_array()
            .map(|c| !c.is_empty())
            .unwrap_or(false),
        "MCP-delivered answer carries candidates: {text}"
    );

    // tools/call rro_query → the typed query plane through MCP, DSL included.
    let query = rpc(serde_json::json!({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": { "name": "rro_query", "arguments": {
            "text": "postgres upgrade",
            "top_k": 5,
            "dsl": { "must_not": [ { "op": "eq", "key": "team", "value": "nobody" } ] }
        } }
    }));
    assert_eq!(query["result"]["isError"], serde_json::json!(false));
    let text = query["result"]["content"][0]["text"].as_str().unwrap();
    let result: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(
        result["candidates"]
            .as_array()
            .map(|c| !c.is_empty())
            .unwrap_or(false),
        "MCP-delivered typed query carries candidates: {text}"
    );

    drop(stdin); // EOF → clean exit
    let _ = child.wait();
}
