//! Signal-emitted embedding + reranking.
//!
//! The embedder and reranker do not call a model directly — they **emit a
//! signal** (`verb: "embed"` / `verb: "rerank"`) onto the a2a bus and await the
//! reply. A [`ModelNode`] handler fulfils that signal by calling the actual
//! model backend (a vLLM quadlet over its localhost HTTP edge, reusing
//! [`embedder::OpenAiEmbedder`] / [`reranker::HttpReranker`]).
//!
//! Why the indirection: it is the same [`Handler`] contract whether the model is
//! co-located ([`LocalBus`]) or on a remote GPU node (`tcp`), so the fulfiller is
//! transparent — the flow never learns *where* the model runs. It also puts
//! every embed/rerank on the signal spine, where throughput, backpressure, and
//! observability live, instead of a blocking client buried in the flow.
//!
//! The model never loads in this process: the signal is fulfilled by whatever
//! backend the [`ModelNode`] wraps.

use std::sync::Arc;

use async_trait::async_trait;
use rro_core::{Candidate, Embedder, Embedding, Reranker, Result, RroError};
use rro_net::{Handler, LocalBus, Message, NodeId};
use serde_json::{json, Value};

/// How a batch of text is being embedded — routed to the instruction-aware
/// path on the fulfilling model (documents vs queries embed differently).
fn embed_mode(msg_mode: Option<&str>) -> EmbedMode {
    match msg_mode {
        Some("documents") => EmbedMode::Documents,
        Some("queries") => EmbedMode::Queries,
        _ => EmbedMode::Plain,
    }
}

#[derive(Clone, Copy)]
enum EmbedMode {
    Plain,
    Documents,
    Queries,
}

impl EmbedMode {
    fn as_str(self) -> &'static str {
        match self {
            EmbedMode::Plain => "plain",
            EmbedMode::Documents => "documents",
            EmbedMode::Queries => "queries",
        }
    }
}

/// The fulfiller: a node that answers `embed`/`rerank` signals by running the
/// wrapped model backend. Construct it around a vLLM-backed
/// [`embedder::OpenAiEmbedder`] + [`reranker::HttpReranker`] (see
/// [`connect_vllm_signals`]) — or, in tests, any [`Embedder`]/[`Reranker`].
pub struct ModelNode {
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
}

impl ModelNode {
    /// Wrap a model backend as a signal fulfiller.
    pub fn new(embedder: Arc<dyn Embedder>, reranker: Arc<dyn Reranker>) -> Self {
        ModelNode { embedder, reranker }
    }

    async fn do_embed(&self, body: &Value) -> Result<Message> {
        let texts: Vec<String> = serde_json::from_value(
            body.get("texts")
                .cloned()
                .ok_or_else(|| RroError::Embed("embed signal missing `texts`".into()))?,
        )?;
        let mode = embed_mode(body.get("mode").and_then(Value::as_str));
        let vectors = match mode {
            EmbedMode::Documents => self.embedder.embed_documents(&texts).await?,
            EmbedMode::Queries => self.embedder.embed_queries(&texts).await?,
            EmbedMode::Plain => self.embedder.embed(&texts).await?,
        };
        Ok(reply_body(json!({ "vectors": vectors })))
    }

    async fn do_rerank(&self, body: &Value) -> Result<Message> {
        let query = body
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| RroError::Rerank("rerank signal missing `query`".into()))?;
        let candidates: Vec<Candidate> = serde_json::from_value(
            body.get("candidates")
                .cloned()
                .ok_or_else(|| RroError::Rerank("rerank signal missing `candidates`".into()))?,
        )?;
        let top_k = body
            .get("top_k")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(candidates.len());
        let ranked = self.reranker.rerank(query, candidates, top_k).await?;
        Ok(reply_body(json!({ "candidates": ranked })))
    }
}

/// Build a reply message body. The bus swaps from/to and echoes the id via
/// [`Message::reply`]; here we only carry the payload, so we stamp a throwaway
/// envelope the caller's `reply` overwrites.
fn reply_body(body: Value) -> Message {
    // A placeholder envelope; `Handler::handle` returns this via `msg.reply`,
    // which is what actually sets from/to/id — see the impl below.
    Message::request("model", "engine", "reply", body)
}

#[async_trait]
impl Handler for ModelNode {
    async fn handle(&self, msg: Message) -> Result<Option<Message>> {
        let out = match msg.verb.as_str() {
            "embed" => self.do_embed(&msg.body).await?,
            "rerank" => self.do_rerank(&msg.body).await?,
            _ => return Ok(None),
        };
        Ok(Some(msg.reply(out.body)))
    }
}

/// An [`Embedder`] that emits an `embed` signal to a [`ModelNode`] and returns
/// the fulfilled vectors. Holds no model — the compute happens in the fulfiller.
pub struct SignalEmbedder {
    bus: LocalBus,
    from: NodeId,
    to: NodeId,
    token: Option<String>,
    dim: usize,
}

impl SignalEmbedder {
    /// A signal embedder that reaches the model node `to` over `bus`. `dim` is
    /// the fulfilling model's output dimension (probed once at assembly).
    pub fn new(bus: LocalBus, to: impl Into<NodeId>, dim: usize) -> Self {
        SignalEmbedder {
            bus,
            from: NodeId::new("engine"),
            to: to.into(),
            token: None,
            dim,
        }
    }

    /// Attach a capability token carried on every emitted signal.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    async fn dispatch_embed(&self, texts: &[String], mode: EmbedMode) -> Result<Vec<Embedding>> {
        let mut msg = Message::request(
            self.from.clone(),
            self.to.clone(),
            "embed",
            json!({ "texts": texts, "mode": mode.as_str() }),
        );
        if let Some(t) = &self.token {
            msg = msg.with_token(t.clone());
        }
        let reply = self
            .bus
            .dispatch(msg)
            .await?
            .ok_or_else(|| RroError::Embed("model node gave no reply to `embed`".into()))?;
        let vectors = reply
            .body
            .get("vectors")
            .cloned()
            .ok_or_else(|| RroError::Embed("embed reply missing `vectors`".into()))?;
        Ok(serde_json::from_value(vectors)?)
    }
}

#[async_trait]
impl Embedder for SignalEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.dispatch_embed(texts, EmbedMode::Plain).await
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.dispatch_embed(texts, EmbedMode::Documents).await
    }

    async fn embed_queries(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.dispatch_embed(texts, EmbedMode::Queries).await
    }

    fn model_name(&self) -> &str {
        "signal-embedder"
    }
}

/// A [`Reranker`] that emits a `rerank` signal to a [`ModelNode`].
pub struct SignalReranker {
    bus: LocalBus,
    from: NodeId,
    to: NodeId,
    token: Option<String>,
}

impl SignalReranker {
    /// A signal reranker that reaches the model node `to` over `bus`.
    pub fn new(bus: LocalBus, to: impl Into<NodeId>) -> Self {
        SignalReranker {
            bus,
            from: NodeId::new("engine"),
            to: to.into(),
            token: None,
        }
    }

    /// Attach a capability token carried on every emitted signal.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }
}

#[async_trait]
impl Reranker for SignalReranker {
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<Candidate>,
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        let mut msg = Message::request(
            self.from.clone(),
            self.to.clone(),
            "rerank",
            json!({ "query": query, "candidates": candidates, "top_k": top_k }),
        );
        if let Some(t) = &self.token {
            msg = msg.with_token(t.clone());
        }
        let reply = self
            .bus
            .dispatch(msg)
            .await?
            .ok_or_else(|| RroError::Rerank("model node gave no reply to `rerank`".into()))?;
        let ranked = reply
            .body
            .get("candidates")
            .cloned()
            .ok_or_else(|| RroError::Rerank("rerank reply missing `candidates`".into()))?;
        Ok(serde_json::from_value(ranked)?)
    }

    fn model_name(&self) -> &str {
        "signal-reranker"
    }
}

/// The signal-emitting model pair plus the bus that carries their signals to the
/// fulfilling [`ModelNode`].
pub struct SignalModels {
    /// The signal-emitting embedder for the flow.
    pub embedder: Arc<SignalEmbedder>,
    /// The signal-emitting reranker for the flow.
    pub reranker: Arc<SignalReranker>,
    /// The bus the model node is registered on (kept alive by the caller).
    pub bus: LocalBus,
}

/// Wire the signal path to a **vLLM** backend: connect the localhost embed +
/// rerank endpoints, register them as a [`ModelNode`] fulfiller on a fresh
/// [`LocalBus`], and hand back signal-emitting [`SignalEmbedder`]/[`SignalReranker`]
/// for the flow. The model runs in the vLLM quadlet, never in this process.
///
/// `embed_endpoint` / `rerank_endpoint` are the vLLM URLs (e.g.
/// `http://127.0.0.1:8092/v1/embeddings` and `http://127.0.0.1:8092/rerank`).
pub async fn connect_vllm_signals(
    embed_endpoint: impl Into<String>,
    rerank_endpoint: impl Into<String>,
) -> Result<SignalModels> {
    use embedder::{OpenAiEmbedConfig, OpenAiEmbedder, OpenAiKind};
    use reranker::{HttpRerankConfig, HttpRerankKind, HttpReranker};

    let embedder =
        OpenAiEmbedder::connect(OpenAiEmbedConfig::new(embed_endpoint, OpenAiKind::Vllm)).await?;
    let dim = embedder.dim();
    let reranker =
        HttpReranker::connect(HttpRerankConfig::new(rerank_endpoint, HttpRerankKind::Vllm)).await?;

    let bus = LocalBus::new();
    bus.register(
        "model",
        Arc::new(ModelNode::new(Arc::new(embedder), Arc::new(reranker))),
    )?;

    Ok(SignalModels {
        embedder: Arc::new(SignalEmbedder::new(bus.clone(), "model", dim)),
        reranker: Arc::new(SignalReranker::new(bus.clone(), "model")),
        bus,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedder::DeterministicEmbedder;
    use reranker::LexicalReranker;

    /// Build a bus with a weightless model node — proves the signal round-trip
    /// without any live server.
    fn wired() -> (SignalEmbedder, SignalReranker, LocalBus, usize) {
        let embedder = DeterministicEmbedder::new();
        let dim = embedder.dim();
        let bus = LocalBus::new();
        bus.register(
            "model",
            Arc::new(ModelNode::new(
                Arc::new(embedder),
                Arc::new(LexicalReranker::new()),
            )),
        )
        .unwrap();
        (
            SignalEmbedder::new(bus.clone(), "model", dim),
            SignalReranker::new(bus.clone(), "model"),
            bus,
            dim,
        )
    }

    #[tokio::test]
    async fn embed_signal_round_trips() {
        let (emb, _rr, _bus, dim) = wired();
        let out = emb
            .embed(&["hello".to_string(), "world".to_string()])
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].dim(), dim);
        assert_eq!(emb.dim(), dim);
    }

    #[tokio::test]
    async fn embed_matches_the_backend_directly() {
        let (emb, _rr, _bus, _dim) = wired();
        let direct = DeterministicEmbedder::new()
            .embed(&["same text".to_string()])
            .await
            .unwrap();
        let viasignal = emb.embed(&["same text".to_string()]).await.unwrap();
        assert_eq!(direct, viasignal, "signal path must equal the backend");
    }

    #[tokio::test]
    async fn rerank_signal_round_trips() {
        let (_emb, rr, _bus, _dim) = wired();
        let cands = vec![
            Candidate::new("a", "the quick brown fox", 0.1),
            Candidate::new("b", "unrelated content here", 0.2),
        ];
        let out = rr.rerank("quick fox", cands, 2).await.unwrap();
        assert_eq!(out.len(), 2);
    }

    #[tokio::test]
    async fn unknown_verb_gets_no_reply() {
        let (_emb, _rr, bus, _dim) = wired();
        let reply = bus
            .dispatch(Message::request("engine", "model", "nonsense", json!({})))
            .await
            .unwrap();
        assert!(reply.is_none());
    }
}
