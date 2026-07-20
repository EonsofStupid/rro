//! The semantic router: tag classification as routing in embedding space.
//!
//! A **route** is a tag with exemplar embeddings; its centroid is the tag's
//! location in the *same* embedding space recall uses. Classification is
//! cosine against centroids with a per-route threshold — no extra model, no
//! extra forward pass: at ingest the document embedding already exists, so
//! routing costs `K` dot products per document.
//!
//! The embedder behind the space is whatever implements `rro_core::Embedder`
//! — the deterministic default today, the DevPULSE (Qwen) sentence encoder
//! when weights land. Routes are data, not code: add exemplars, the router
//! gets smarter. Zero-shot / NLI classification deliberately does **not**
//! run here — if it earns a place, it is at plan-compile time (once per new
//! sliver), never per document.

use serde::{Deserialize, Serialize};

use rro_core::Embedding;

/// A tag's location in embedding space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    /// The tag this route assigns.
    pub tag: String,
    /// Centroid of the exemplar embeddings (unit length).
    pub centroid: Embedding,
    /// Minimum cosine for the tag to fire.
    pub threshold: f32,
}

/// A scored tag assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutedTag {
    /// The tag.
    pub tag: String,
    /// Cosine similarity to the route centroid.
    pub score: f32,
}

/// Routes documents to tags by embedding-space proximity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticRouter {
    routes: Vec<Route>,
}

impl SemanticRouter {
    /// An empty router (no tags fire).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a route from already-embedded exemplars.
    ///
    /// The caller embeds exemplar phrases with the engine's `Embedder` (the
    /// same one indexing documents) and hands the vectors in; the route
    /// centroid is their normalized mean. Empty exemplar lists are ignored.
    pub fn add_route(
        &mut self,
        tag: impl Into<String>,
        exemplars: &[Embedding],
        threshold: f32,
    ) -> &mut Self {
        if exemplars.is_empty() {
            return self;
        }
        let dim = exemplars[0].dim();
        let mut sum = vec![0.0f32; dim];
        for e in exemplars {
            for (s, v) in sum.iter_mut().zip(e.as_slice()) {
                *s += v;
            }
        }
        let centroid = Embedding(sum).normalized();
        self.routes.push(Route {
            tag: tag.into(),
            centroid,
            threshold,
        });
        self
    }

    /// Route one document embedding: every tag whose cosine clears its
    /// threshold, best first.
    pub fn route(&self, doc: &Embedding) -> Vec<RoutedTag> {
        let mut out: Vec<RoutedTag> = self
            .routes
            .iter()
            .filter_map(|r| {
                let score = doc.cosine(&r.centroid);
                (score >= r.threshold).then(|| RoutedTag {
                    tag: r.tag.clone(),
                    score,
                })
            })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out
    }

    /// Number of registered routes.
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// Whether the router has no routes.
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: &[f32]) -> Embedding {
        Embedding(v.to_vec()).normalized()
    }

    #[test]
    fn routes_to_nearest_tag_above_threshold() {
        let mut r = SemanticRouter::new();
        r.add_route(
            "ops",
            &[unit(&[1.0, 0.0, 0.0]), unit(&[0.9, 0.1, 0.0])],
            0.5,
        );
        r.add_route("cooking", &[unit(&[0.0, 1.0, 0.0])], 0.5);

        let hits = r.route(&unit(&[0.95, 0.05, 0.0]));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tag, "ops");

        let none = r.route(&unit(&[0.0, 0.0, 1.0]));
        assert!(none.is_empty(), "orthogonal doc must route nowhere");
    }

    #[test]
    fn multiple_tags_sorted_by_score() {
        let mut r = SemanticRouter::new();
        r.add_route("a", &[unit(&[1.0, 0.0])], 0.1);
        r.add_route("b", &[unit(&[0.7, 0.7])], 0.1);
        let hits = r.route(&unit(&[1.0, 0.2]));
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].tag, "a");
        assert!(hits[0].score >= hits[1].score);
    }
}
