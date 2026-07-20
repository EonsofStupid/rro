//! Phase 12 gate: RBAC over the a2a wire and the HTTP doorway.
//!
//! A signed HS256 token carries a role and a namespace scope. This proves, on
//! the real transport (not just the unit level), that:
//!   - role allow/deny is enforced per verb (a reader cannot write),
//!   - an expired or wrong-signature token is refused,
//!   - a namespace-scoped token cannot act on a node serving another namespace,
//!   - a reader's `sql` write is refused (the text surface is covered too),
//!   - the HTTP doorway carries the bearer token into the same gate.

use std::sync::Arc;

use rro_engine::{AuthPolicy, Claims, FlowNode, ReasonReadyObject, Role};
use rro_net::{tcp, Message};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn guarded_node(policy: AuthPolicy) -> (std::net::SocketAddr, Arc<FlowNode>) {
    let dir = tempfile::tempdir().unwrap();
    // Leak the tempdir: the node outlives this helper and the estate must stay open.
    let path = dir.keep();
    let estate = Arc::new(connxism::Estate::open(&path, "rbac").unwrap());
    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = Arc::new(
        FlowNode::new(flow, "guarded")
            .with_estate(estate)
            .with_auth(policy),
    );
    let (addr, _task) = tcp::serve("127.0.0.1:0", node.clone()).await.unwrap();
    (addr, node)
}

async fn call(addr: std::net::SocketAddr, verb: &str, token: Option<&str>) -> serde_json::Value {
    let mut msg = Message::request("c", "guarded", verb, serde_json::json!({}));
    if let Some(t) = token {
        msg = msg.with_token(t);
    }
    tcp::request(addr, &msg).await.unwrap().body
}

#[tokio::test(flavor = "multi_thread")]
async fn roles_gate_verbs_over_the_wire() {
    let policy = AuthPolicy::new(b"local-key".to_vec());
    let (addr, _node) = guarded_node(policy.clone()).await;

    let reader = policy.issue_for("r", Role::Reader, None, 3600);
    let writer = policy.issue_for("w", Role::Writer, None, 3600);
    let admin = policy.issue_for("a", Role::Admin, None, 3600);

    // ping is always open — no token needed.
    assert_eq!(
        call(addr, "ping", None).await["pong"],
        serde_json::json!(true)
    );

    // A reader may read (health) but not write (index) or administer (compact).
    assert!(call(addr, "health", Some(&reader))
        .await
        .get("node")
        .is_some());
    assert_eq!(
        call(addr, "index", Some(&reader)).await["error"],
        "unauthorized"
    );
    assert_eq!(
        call(addr, "compact", Some(&reader)).await["error"],
        "unauthorized"
    );

    // A writer may index but not compact.
    assert!(call(addr, "index", Some(&writer))
        .await
        .get("error")
        .is_none());
    assert_eq!(
        call(addr, "compact", Some(&writer)).await["error"],
        "unauthorized"
    );

    // An admin may compact.
    assert!(call(addr, "compact", Some(&admin))
        .await
        .get("error")
        .is_none());

    // No token, or a token signed by a different key → refused.
    assert_eq!(call(addr, "health", None).await["error"], "unauthorized");
    let forged = AuthPolicy::new(b"attacker-key".to_vec()).issue_for("x", Role::Admin, None, 3600);
    assert_eq!(
        call(addr, "health", Some(&forged)).await["error"],
        "unauthorized"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn expired_token_is_refused() {
    let policy = AuthPolicy::new(b"k".to_vec());
    let (addr, _node) = guarded_node(policy.clone()).await;

    let expired = policy.issue(&Claims {
        sub: "x".into(),
        role: Role::Admin,
        ns: None,
        iat: 0,
        exp: 1, // 1970 — long past
    });
    assert_eq!(
        call(addr, "health", Some(&expired)).await["error"],
        "unauthorized"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_namespace_scoped_token_cannot_cross_namespaces() {
    // The node serves namespace "acme".
    let policy = AuthPolicy::new(b"k".to_vec()).for_namespace("acme");
    let (addr, _node) = guarded_node(policy.clone()).await;

    // A token scoped to "acme" works here.
    let acme = policy.issue_for("a", Role::Reader, Some("acme"), 3600);
    assert!(call(addr, "health", Some(&acme))
        .await
        .get("node")
        .is_some());

    // A token scoped to "globex" is refused on the acme node, even though its
    // signature and role are valid — isolation the same key cannot bypass.
    let globex = policy.issue_for("g", Role::Admin, Some("globex"), 3600);
    assert_eq!(
        call(addr, "health", Some(&globex)).await["error"],
        "unauthorized"
    );

    // An unscoped (global) token still works on the scoped node.
    let global = policy.issue_for("root", Role::Reader, None, 3600);
    assert!(call(addr, "health", Some(&global))
        .await
        .get("node")
        .is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_readers_sql_write_is_refused() {
    let policy = AuthPolicy::new(b"k".to_vec());
    let (addr, _node) = guarded_node(policy.clone()).await;
    let reader = policy.issue_for("r", Role::Reader, None, 3600);

    // A write statement from a reader → unauthorized (not merely read_only).
    let mut msg = Message::request(
        "c",
        "guarded",
        "sql",
        serde_json::json!({ "sql": "DELETE doc1" }),
    )
    .with_token(reader.clone());
    let reply = tcp::request(addr, &msg).await.unwrap().body;
    assert_eq!(
        reply["error"], "unauthorized",
        "reader write refused: {reply}"
    );

    // The same reader may DEFINE? No — DEFINE is a write too. But a read-side
    // sql (INFO) is allowed.
    msg = Message::request("c", "guarded", "sql", serde_json::json!({ "sql": "INFO" }))
        .with_token(reader);
    let reply = tcp::request(addr, &msg).await.unwrap().body;
    assert_ne!(
        reply["error"], "unauthorized",
        "reader INFO allowed: {reply}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn http_doorway_enforces_rbac() {
    let policy = AuthPolicy::new(b"k".to_vec());
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "rbac").unwrap());
    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = Arc::new(
        FlowNode::new(flow, "guarded")
            .with_estate(estate)
            .with_auth(policy.clone()),
    );
    let (addr, _http) = rro_engine::serve_http("127.0.0.1:0", node).await.unwrap();

    let reader = policy.issue_for("r", Role::Reader, None, 3600);

    // POST /v/index with a reader token → 401 unauthorized over HTTP.
    let (status, _) = http(addr, "POST", "/v/index", Some(&reader), Some("{}")).await;
    assert!(status.contains("401"), "reader index over http: {status}");

    // GET /health with the reader token → 200.
    let (status, body) = http(addr, "GET", "/health", Some(&reader), None).await;
    assert!(status.contains("200"), "reader health: {status}");
    assert!(body.contains("\"node\""), "health body: {body}");

    // No token → 401.
    let (status, _) = http(addr, "GET", "/health", None, None).await;
    assert!(status.contains("401"), "no token: {status}");
}

async fn http(
    addr: std::net::SocketAddr,
    method: &str,
    path: &str,
    token: Option<&str>,
    body: Option<&str>,
) -> (String, String) {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: x\r\n");
    if let Some(t) = token {
        req.push_str(&format!("Authorization: Bearer {t}\r\n"));
    }
    if let Some(b) = body {
        req.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}",
            b.len()
        ));
    } else {
        req.push_str("\r\n");
    }
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = String::new();
    stream.read_to_string(&mut buf).await.unwrap();
    let (head, body) = buf.split_once("\r\n\r\n").unwrap();
    (
        head.lines().next().unwrap_or("").to_string(),
        body.to_string(),
    )
}
