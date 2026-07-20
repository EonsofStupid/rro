//! # rro-net
//!
//! The agent-to-agent (a2a) / node networking surface for Reason Ready.
//! Embedding the engine in-process does **not** cut it off: nodes speak the
//! same [`Handler`] contract whether they are co-located ([`LocalBus`]) or
//! remote ([`tcp`]).
//!
//! - [`Message`] / [`NodeId`] — the a2a vocabulary.
//! - [`Handler`] — what a node does with a message.
//! - [`LocalBus`] — in-process routing between many agents.
//! - [`tcp::serve`] / [`tcp::request`] — the same contract over the wire.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod handler;
mod local;
mod message;
pub mod tcp;

pub use handler::{Handler, PingHandler};
pub use local::LocalBus;
pub use message::{Message, NodeId};
