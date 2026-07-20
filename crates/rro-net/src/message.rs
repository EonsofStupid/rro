//! The a2a wire vocabulary: nodes and messages.

use serde::{Deserialize, Serialize};

/// Identity of a node (agent) on the network.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    /// Wrap a node name.
    pub fn new(s: impl Into<String>) -> Self {
        NodeId(s.into())
    }
    /// Borrow as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for NodeId {
    fn from(s: &str) -> Self {
        NodeId(s.to_string())
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One agent-to-agent message. Transport-agnostic; serialized as a single JSON
/// object (no embedded newlines) so it frames cleanly over a stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Correlation id; a reply echoes the request's id.
    pub id: String,
    /// Sender.
    pub from: NodeId,
    /// Intended recipient.
    pub to: NodeId,
    /// The action, e.g. `recall`, `classify`, `ping`.
    pub verb: String,
    /// Capability token (v1: shared secret). Nodes configured with a token
    /// reject messages that don't bear it. Absent on open nodes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// Free-form payload.
    pub body: serde_json::Value,
}

impl Message {
    /// Build a request.
    pub fn request(
        from: impl Into<NodeId>,
        to: impl Into<NodeId>,
        verb: impl Into<String>,
        body: serde_json::Value,
    ) -> Self {
        Message {
            id: uuid_like(),
            from: from.into(),
            to: to.into(),
            verb: verb.into(),
            token: None,
            body,
        }
    }

    /// Attach a capability token.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Build a reply to this message, swapping from/to and keeping the id.
    /// Replies never echo the bearer token.
    pub fn reply(&self, body: serde_json::Value) -> Self {
        Message {
            id: self.id.clone(),
            from: self.to.clone(),
            to: self.from.clone(),
            verb: format!("{}.reply", self.verb),
            token: None,
            body,
        }
    }
}

// A tiny, dependency-light correlation id (not a real UUID; ids need only be
// unique enough to correlate a reply within a session).
fn uuid_like() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("m-{n:016x}")
}
