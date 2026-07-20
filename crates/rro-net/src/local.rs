//! In-process a2a: many agents, one process, zero sockets.
//!
//! The [`LocalBus`] routes messages between handlers registered under node ids.
//! This is the embedded case — agents co-located in one binary talking a2a
//! without a network hop — and it shares the exact [`Handler`] contract with
//! the TCP transport, so a node does not know or care which it is on.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rro_core::{Result, RroError};

use crate::handler::Handler;
use crate::message::{Message, NodeId};

/// A process-local message bus.
#[derive(Clone, Default)]
pub struct LocalBus {
    nodes: Arc<RwLock<HashMap<NodeId, Arc<dyn Handler>>>>,
}

impl LocalBus {
    /// A new, empty bus.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `handler` under node id `id`.
    pub fn register(&self, id: impl Into<NodeId>, handler: Arc<dyn Handler>) -> Result<()> {
        self.nodes
            .write()
            .map_err(|_| RroError::Net("bus lock poisoned".into()))?
            .insert(id.into(), handler);
        Ok(())
    }

    /// Deliver `msg` to its target node, returning that node's reply (if any).
    pub async fn dispatch(&self, msg: Message) -> Result<Option<Message>> {
        let handler = {
            let map = self
                .nodes
                .read()
                .map_err(|_| RroError::Net("bus lock poisoned".into()))?;
            map.get(&msg.to).cloned()
        };
        match handler {
            Some(h) => h.handle(msg).await,
            None => Err(RroError::Net(format!("no such node: {}", msg.to))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handler::PingHandler;

    #[tokio::test]
    async fn local_ping_pong() {
        let bus = LocalBus::new();
        bus.register(
            "b",
            Arc::new(PingHandler {
                me: NodeId::new("b"),
            }),
        )
        .unwrap();
        let reply = bus
            .dispatch(Message::request("a", "b", "ping", serde_json::json!({})))
            .await
            .unwrap()
            .expect("reply");
        assert_eq!(reply.body["pong"], serde_json::json!(true));
    }
}
