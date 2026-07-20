//! The engine contract: four traits, one per swappable component.
//!
//! Everything that can be model-backed and tuned lives behind a trait here so
//! the flow depends on capabilities, not implementations. The DevPULSE models
//! (Qwen embedder, Nemotron reranker) plug in as alternate [`Embedder`] /
//! [`Reranker`] impls without touching a single line of the flow.

use async_trait::async_trait;

use crate::error::Result;
use crate::types::{Candidate, Embedding, Id, Metadata, Readiness, SparseVector};

/// A record as it lives in the recall store.
#[derive(Debug, Clone)]
pub struct VectorRecord {
    /// Stable id.
    pub id: Id,
    /// The dense vector.
    pub embedding: Embedding,
    /// Optional weighted sparse vector (learned sparse / custom weights);
    /// stores that maintain a sparse index use it for sparse retrieval.
    pub sparse: Option<SparseVector>,
    /// Named auxiliary vectors (each name is its own space with its own
    /// dimensionality — e.g. `title` vs `body` embeddings per point).
    pub named: std::collections::BTreeMap<String, Embedding>,
    /// Late-interaction token vectors (ColBERT-style), scored by MaxSim.
    pub multi: Vec<Embedding>,
    /// The named collection this record belongs to (stores that support
    /// collections scope queries by it).
    pub collection: Option<String>,
    /// The text this vector represents (returned in candidates).
    pub text: String,
    /// Structured metadata.
    pub metadata: Metadata,
}

impl VectorRecord {
    /// Construct a record with empty metadata.
    pub fn new(id: impl Into<Id>, embedding: Embedding, text: impl Into<String>) -> Self {
        VectorRecord {
            id: id.into(),
            embedding,
            sparse: None,
            named: std::collections::BTreeMap::new(),
            multi: Vec::new(),
            collection: None,
            text: text.into(),
            metadata: Metadata::new(),
        }
    }

    /// Builder-style sparse vector attachment.
    pub fn with_sparse(mut self, sparse: SparseVector) -> Self {
        self.sparse = Some(sparse);
        self
    }

    /// Attach (or replace) a named vector — its own space, its own dims.
    pub fn with_named(mut self, name: impl Into<String>, v: Embedding) -> Self {
        self.named.insert(name.into(), v);
        self
    }

    /// Attach late-interaction token vectors (scored by MaxSim).
    pub fn with_multi(mut self, vectors: Vec<Embedding>) -> Self {
        self.multi = vectors;
        self
    }

    /// Place this record in a named collection.
    pub fn in_collection(mut self, name: impl Into<String>) -> Self {
        self.collection = Some(name.into());
        self
    }
}

/// Perception: turns text into dense vectors. DevPULSE backbone: Qwen.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Output dimensionality of this embedder.
    fn dim(&self) -> usize;

    /// Embed a batch of texts, preserving order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>>;

    /// Embed a single text. Default routes through [`Embedder::embed`].
    async fn embed_one(&self, text: &str) -> Result<Embedding> {
        let batch = [text.to_string()];
        let mut out = self.embed(&batch).await?;
        out.pop()
            .ok_or_else(|| crate::error::RroError::Embed("empty embedding batch".into()))
    }

    /// Embed texts that are being STORED — documents/passages.
    ///
    /// Split from [`Embedder::embed_queries`] because instruction-aware models
    /// embed the two sides differently: Qwen3-Embedding prefixes a query with
    /// `Instruct: <task>\nQuery:` and leaves documents bare. Sending a document
    /// down the query path costs retrieval quality *silently* — nothing errors,
    /// the vectors are just worse — so the distinction is part of the contract
    /// rather than a convention call sites are trusted to remember.
    ///
    /// Symmetric embedders ignore the split; the default routes to
    /// [`Embedder::embed`], so existing impls stay correct untouched.
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.embed(texts).await
    }

    /// Embed texts that are being SEARCHED WITH — queries. See
    /// [`Embedder::embed_documents`] for why this is a separate method.
    async fn embed_queries(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.embed(texts).await
    }

    /// Embed a single document. Default routes through
    /// [`Embedder::embed_documents`].
    async fn embed_document_one(&self, text: &str) -> Result<Embedding> {
        let batch = [text.to_string()];
        let mut out = self.embed_documents(&batch).await?;
        out.pop()
            .ok_or_else(|| crate::error::RroError::Embed("empty embedding batch".into()))
    }

    /// Embed a single query. Default routes through [`Embedder::embed_queries`].
    async fn embed_query_one(&self, text: &str) -> Result<Embedding> {
        let batch = [text.to_string()];
        let mut out = self.embed_queries(&batch).await?;
        out.pop()
            .ok_or_else(|| crate::error::RroError::Embed("empty embedding batch".into()))
    }

    /// Name of the active model, for telemetry and the connectome.
    fn model_name(&self) -> &str {
        "embedder"
    }
}

/// Dense vector memory: the Recall engine.
#[async_trait]
pub trait Recall: Send + Sync {
    /// Insert or overwrite records by id.
    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<()>;

    /// Nearest-neighbour search; returns up to `top_k` candidates, best first.
    async fn search(&self, query: &Embedding, top_k: usize) -> Result<Vec<Candidate>>;

    /// Hybrid search: dense vector similarity fused with lexical relevance.
    ///
    /// Stores that maintain a lexical index (e.g. BM25 postings) override this
    /// and fuse the two rankings (typically by reciprocal rank fusion). The
    /// default falls back to pure vector [`Recall::search`], so every store is
    /// hybrid-callable.
    async fn hybrid_search(
        &self,
        query_text: &str,
        query: &Embedding,
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        let _ = query_text;
        self.search(query, top_k).await
    }

    /// Number of records currently held.
    async fn len(&self) -> Result<usize>;

    /// Whether the store is empty.
    async fn is_empty(&self) -> Result<bool> {
        Ok(self.len().await? == 0)
    }

    /// Remove a record by id. Default is a no-op for append-only stores.
    async fn remove(&self, id: &Id) -> Result<()> {
        let _ = id;
        Ok(())
    }

    /// Wait until background index maintenance (if any) has caught up with
    /// every accepted write. Stores with out-of-band index apply override
    /// this; fully-synchronous stores are always caught up (the default).
    async fn quiesce(&self) -> Result<()> {
        Ok(())
    }
}

/// True-relevance ordering over recall candidates. DevPULSE backbone: Nemotron.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Re-score and re-order `candidates` against `query`, returning up to
    /// `top_k`, best first.
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<Candidate>,
        top_k: usize,
    ) -> Result<Vec<Candidate>>;

    /// Name of the active model, for telemetry and the connectome.
    fn model_name(&self) -> &str {
        "reranker"
    }
}

/// The Reason Ready daemon: decides whether context is sufficient to reason on.
#[async_trait]
pub trait Classifier: Send + Sync {
    /// Judge readiness of `context` for answering `query`.
    async fn classify(&self, query: &str, context: &[Candidate]) -> Result<Readiness>;

    /// Name of the active model, for telemetry and the connectome.
    fn model_name(&self) -> &str {
        "classifier"
    }
}
