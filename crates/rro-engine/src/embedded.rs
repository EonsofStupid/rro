//! Turnkey in-process engine: open an estate, point at HTTP model servers, get a
//! ready [`EmbeddedEngine`]. The assembly the `rro` daemon does, as one call — so
//! a consumer embeds RRO without naming connxism/embedder/reranker/rrd itself.
//!
//! Two ways in:
//!   - [`EmbeddedEngine::deterministic`] — **import-and-go, no servers.** A
//!     weightless embedder + lexical reranker over a real persistent estate, for
//!     a first run, tests, and CI. Synchronous; nothing to provision.
//!   - [`EmbeddedEngine::embed_http`] — the real semantic engine over vLLM/llama.cpp
//!     embedder + reranker HTTP servers (both must be up; the constructor probes
//!     them and cross-checks the embedder's dimension against the estate).

use std::path::Path;
use std::sync::Arc;

use connectome::ConnectomeGraph;
use rro_core::{Document, Embedder, Metadata, RecallResult, Result, RroError};

use crate::flow::{ObjectBuilder, ReasonReadyObject};

/// How an [`EmbeddedEngine`] was assembled — reported by [`EmbeddedEngine::health`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineMode {
    /// Weightless: deterministic embedder + lexical reranker, no external servers.
    Deterministic,
    /// Real semantic engine backed by HTTP embedder/reranker servers.
    Http,
    /// Real semantic engine over a vLLM quadlet, reached via **signal-emitted**
    /// embed/rerank (a2a bus → [`ModelNode`](crate::signal_model::ModelNode)
    /// fulfiller). The model runs in the quadlet, never in this process.
    Signal,
}

/// A liveness snapshot of an [`EmbeddedEngine`] — cheap to obtain, safe to poll.
#[derive(Debug, Clone)]
pub struct Health {
    /// Whether a probe embed succeeded (the model path answers).
    pub ready: bool,
    /// The estate's fixed vector dimension, once the first document is indexed.
    pub dim: Option<usize>,
    /// How the engine was assembled.
    pub mode: EngineMode,
    /// Human-readable detail (the error, when not ready).
    pub note: String,
}

/// An estate plus its assembled flow, held together — the estate stays alive so
/// its out-of-band graph applier keeps running.
pub struct EmbeddedEngine {
    estate: Arc<connxism::Estate>,
    flow: ReasonReadyObject,
    mode: EngineMode,
}

impl EmbeddedEngine {
    /// Import-and-go: a fully in-process engine over a real persistent estate at
    /// `path` (named `name`), using the deterministic embedder + lexical reranker.
    /// **No external servers, no async, no GPU** — for a first run, tests, and CI.
    /// Recall is lexical + structural; swap to [`Self::embed_http`] for semantic
    /// embeddings. Errors only if the estate can't be opened or its fixed
    /// dimension disagrees with the deterministic embedder.
    pub fn deterministic(path: impl AsRef<Path>, name: &str) -> Result<Self> {
        let estate = Arc::new(connxism::Estate::open(path, name)?);
        let embedder = embedder::DeterministicEmbedder::new();
        preflight_dim(&estate, embedder.dim())?;
        let flow = ObjectBuilder::new()
            .rrd(Arc::new(rrd::Rrd::new()))
            .recall(Arc::new(estate.recall()))
            .embedder(Arc::new(embedder))
            .reranker(Arc::new(reranker::LexicalReranker::new()))
            .build();
        Ok(Self {
            estate,
            flow,
            mode: EngineMode::Deterministic,
        })
    }

    /// Assemble over vLLM model servers + a connxism estate at `path` (named
    /// `name`): RRD front door, HTTP embedder/reranker, the estate's hybrid recall.
    /// Async because the embedder/reranker connect probes the servers (the
    /// embedder reads its dimension) — they must be up.
    ///
    /// Preflight: the embedder's dimension is cross-checked against the estate's
    /// fixed dimension (if it already has one), so a mismatched model fails HERE
    /// with a guiding message rather than late, on the first `index`.
    pub async fn embed_http(
        path: impl AsRef<Path>,
        name: &str,
        embed_url: &str,
        rerank_url: &str,
    ) -> Result<Self> {
        let estate = Arc::new(connxism::Estate::open(path, name)?);
        let embedder = embedder::OpenAiEmbedder::connect(embedder::OpenAiEmbedConfig::new(
            embed_url,
            embedder::OpenAiKind::Vllm,
        ))
        .await?;
        // Fail fast on a dimension mismatch before the reranker connect / any index.
        preflight_dim(&estate, embedder.dim())?;
        let reranker = reranker::HttpReranker::connect(reranker::HttpRerankConfig::new(
            rerank_url,
            reranker::HttpRerankKind::Vllm,
        ))
        .await?;
        let flow = ObjectBuilder::new()
            .rrd(Arc::new(rrd::Rrd::new()))
            .recall(Arc::new(estate.recall()))
            .embedder(Arc::new(embedder))
            .reranker(Arc::new(reranker))
            .build();
        Ok(Self {
            estate,
            flow,
            mode: EngineMode::Http,
        })
    }

    /// Assemble the **signal-emitted** real engine over a vLLM quadlet: the
    /// embedder and reranker EMIT `embed`/`rerank` signals onto an a2a bus, and a
    /// [`ModelNode`](crate::signal_model::ModelNode) fulfils them by calling the
    /// vLLM localhost endpoints. The model runs in the quadlet, never in this
    /// process, and the flow never learns where it lives — same contract whether
    /// the fulfiller is co-located or a remote GPU node.
    ///
    /// Async because the fulfiller connects and probes the vLLM servers (the
    /// embedder reads its dimension), which is cross-checked against the estate's
    /// fixed dimension. This is the path the fabric runs on — every embed/rerank
    /// rides the signal spine.
    pub async fn vllm_signals(
        path: impl AsRef<Path>,
        name: &str,
        embed_url: &str,
        rerank_url: &str,
    ) -> Result<Self> {
        let estate = Arc::new(connxism::Estate::open(path, name)?);
        let models = crate::signal_model::connect_vllm_signals(embed_url, rerank_url).await?;
        // Fail fast on a dimension mismatch before any index.
        preflight_dim(&estate, models.embedder.dim())?;
        let flow = ObjectBuilder::new()
            .rrd(Arc::new(rrd::Rrd::new()))
            .recall(Arc::new(estate.recall()))
            .embedder(models.embedder)
            .reranker(models.reranker)
            .build();
        Ok(Self {
            estate,
            flow,
            mode: EngineMode::Signal,
        })
    }

    /// Ground a query — the full RRO pass (RRD gate → embed → hybrid recall →
    /// rerank → classify).
    pub async fn ask(&self, query: &str) -> Result<RecallResult> {
        self.flow.ask(query).await
    }

    /// Ground a query WITH shaping metadata — this is the path that makes RRD's
    /// shape/intent real (plain [`ask`](Self::ask) passes no fields, so the shape
    /// fingerprint is inert). Pass request context (source, tags, security hints)
    /// as `fields` to drive the classifier's shape census and routing.
    pub async fn ask_with(&self, query: &str, fields: &Metadata) -> Result<RecallResult> {
        self.flow.ask_with(query, fields).await
    }

    /// Ground a query and also return the connectome graph (the knowledge map over
    /// the recalled set) — the "intelligence, not just RAG" surface.
    pub async fn ask_with_map(&self, query: &str) -> Result<(RecallResult, ConnectomeGraph)> {
        self.flow.ask_with_map(query).await
    }

    /// The connectome graph over an already-obtained recall result.
    pub fn connectome(&self, result: &RecallResult) -> ConnectomeGraph {
        self.flow.connectome(result)
    }

    /// Index documents into the estate.
    pub async fn index(&self, docs: Vec<Document>) -> Result<usize> {
        self.flow.index(docs).await
    }

    /// The estate's fixed vector dimension, once the first document is indexed
    /// (`None` on a fresh estate). Read live from the estate's health snapshot,
    /// since `dim` is written by the first upsert *after* open (the boot-time
    /// `info()` copy goes stale).
    pub fn dim(&self) -> Option<usize> {
        self.estate.health().ok().and_then(|h| h.dim)
    }

    /// Probe liveness after construction — servers can die after startup, and
    /// `ask` would then fail per-query with no signal. Runs one probe embed
    /// through the model path and reports the outcome. The deterministic engine
    /// is always ready.
    pub async fn health(&self) -> Health {
        let (ready, note) = match self.flow.embed_query("health probe").await {
            Ok(_) => (true, "ok".to_string()),
            Err(e) => (false, format!("embed probe failed: {e}")),
        };
        Health {
            ready,
            dim: self.dim(),
            mode: self.mode,
            note,
        }
    }

    /// The underlying estate — direct recall + admin (flush/compact/snapshot).
    pub fn estate(&self) -> &connxism::Estate {
        &self.estate
    }
}

/// Cross-check an embedder's dimension against the estate's fixed dimension.
///
/// The estate learns its `dim` from the first upsert and then rejects any vector
/// of a different width. Opening an existing estate with a differently-sized
/// embedder therefore *succeeds* and only fails later on the first `index` with a
/// bare `DimMismatch`. Catch it at construction with a message that names both
/// sides and what to do.
fn preflight_dim(estate: &connxism::Estate, embedder_dim: usize) -> Result<()> {
    // Live read: `dim` is persisted by the first upsert, so read it from the
    // estate's health snapshot rather than the boot-time `info()` copy.
    if let Some(existing) = estate.health().ok().and_then(|h| h.dim) {
        if existing != embedder_dim {
            return Err(RroError::Recall(format!(
                "embedder dimension {embedder_dim} does not match estate '{}', which is fixed at \
                 {existing}-d (set by its first index). Use a {existing}-d embedder, or index into \
                 a fresh estate path.",
                estate.info().name
            )));
        }
    }
    Ok(())
}
