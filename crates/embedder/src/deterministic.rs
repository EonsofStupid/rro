//! The deterministic, weightless default embedder.
//!
//! It maps text into a fixed-dimension unit vector via feature hashing over
//! content tokens. It is *not* semantic — it captures lexical overlap — but it
//! is deterministic, fast, dependency-free, and makes cosine similarity
//! meaningful, so the whole flow runs end-to-end before any DevPULSE weights
//! exist. Swap it for [`crate::DevPulseEmbedder`] when the tuned model lands.

use async_trait::async_trait;
use rro_core::text::content_tokens;
use rro_core::{Embedder, Embedding, Result};

use crate::tokenize::bucket;

/// A deterministic feature-hashing embedder.
#[derive(Debug, Clone)]
pub struct DeterministicEmbedder {
    dim: usize,
}

impl DeterministicEmbedder {
    /// Default dimensionality of the deterministic embedder.
    pub const DEFAULT_DIM: usize = 384;

    /// Construct with the default dimension.
    pub fn new() -> Self {
        Self {
            dim: Self::DEFAULT_DIM,
        }
    }

    /// Construct with a chosen dimension (must be > 0).
    pub fn with_dim(dim: usize) -> Self {
        Self { dim: dim.max(1) }
    }

    fn embed_text(&self, text: &str) -> Embedding {
        let mut v = vec![0.0f32; self.dim];
        for tok in content_tokens(text) {
            let (idx, sign) = bucket(&tok, self.dim);
            v[idx] += sign;
        }
        Embedding(v).normalized()
    }
}

impl Default for DeterministicEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Embedder for DeterministicEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        Ok(texts.iter().map(|t| self.embed_text(t)).collect())
    }

    fn model_name(&self) -> &str {
        "deterministic-hash"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn similar_text_scores_higher() {
        let e = DeterministicEmbedder::new();
        let q = e.embed_one("the cat sat on the mat").await.unwrap();
        let near = e.embed_one("a cat is sitting on a mat").await.unwrap();
        let far = e
            .embed_one("quantum chromodynamics lagrangian")
            .await
            .unwrap();
        assert!(q.cosine(&near) > q.cosine(&far));
        assert_eq!(q.dim(), DeterministicEmbedder::DEFAULT_DIM);
    }
}
