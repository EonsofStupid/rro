//! The one error type the whole engine speaks.

use thiserror::Error;

/// Everything fallible in Reason Ready returns [`Result`].
pub type Result<T> = std::result::Result<T, RroError>;

/// A single, faithful error type spanning every layer of the flow.
#[derive(Debug, Error)]
pub enum RroError {
    /// The embedder (perception) failed.
    #[error("embed: {0}")]
    Embed(String),

    /// The recall / vector-memory layer failed.
    #[error("recall: {0}")]
    Recall(String),

    /// The reranker failed.
    #[error("rerank: {0}")]
    Rerank(String),

    /// The reason-ready classifier failed.
    #[error("classify: {0}")]
    Classify(String),

    /// The connectome (visual map) failed to build or render.
    #[error("connectome: {0}")]
    Connectome(String),

    /// The a2a / node networking surface failed.
    #[error("net: {0}")]
    Net(String),

    /// A configured resource limit refused the operation.
    #[error("quota: {0}")]
    Quota(String),

    /// A vector was offered at the wrong dimension.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch {
        /// The dimension the store/model expects.
        expected: usize,
        /// The dimension actually provided.
        got: usize,
    },

    /// A configuration value was missing or invalid.
    #[error("config: {0}")]
    Config(String),

    /// Wrapped I/O failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Wrapped (de)serialization failure.
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),

    /// A catch-all with a human message.
    #[error("{0}")]
    Msg(String),
}

impl RroError {
    /// Build a free-form error.
    pub fn msg(s: impl Into<String>) -> Self {
        RroError::Msg(s.into())
    }
}
