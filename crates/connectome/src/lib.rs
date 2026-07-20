//! # connectome
//!
//! The visual/relational map of Reason Ready — the engine's sensory surface.
//! It turns a flow pass into a graph a UI can render, so a non-technical viewer
//! can *see* the recall happen: the query, the pipeline it flowed through, the
//! candidates it surfaced (sized by score), and the reason-ready verdict.
//!
//! The map is pure data ([`ConnectomeGraph`], serde-serializable) plus a
//! Graphviz `to_dot()` for quick inspection — no rendering runtime baked in, so
//! any front end can consume it.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod graph;

pub use graph::{ConnectomeGraph, Edge, EdgeKind, Node, NodeKind};

use rro_core::{Candidate, Readiness};

/// Which pipeline stages to draw, in flow order.
const STAGES: &[(&str, &str)] = &[
    ("stage:embedder", "embedder · perceive"),
    ("stage:recall", "recall · vector memory"),
    ("stage:reranker", "reranker · relevance"),
    ("stage:classifier", "classifier · reason-ready"),
];

/// Builds [`ConnectomeGraph`]s from flow passes.
///
/// Stateless today; kept as a type so a future map can accumulate across passes
/// (session memory, cross-query links) without changing callers.
#[derive(Debug, Default, Clone)]
pub struct Connectome;

impl Connectome {
    /// A new builder.
    pub fn new() -> Self {
        Connectome
    }

    /// Build the map for a single pass: the query, the pipeline, the ranked
    /// candidates, and the readiness verdict.
    pub fn map(
        &self,
        query: &str,
        candidates: &[Candidate],
        readiness: &Readiness,
    ) -> ConnectomeGraph {
        let mut g = ConnectomeGraph::new();

        // Query root.
        g.node("query", NodeKind::Query, query, None);

        // Pipeline spine.
        for (id, label) in STAGES {
            g.node(*id, NodeKind::Stage, *label, None);
        }
        g.edge("query", STAGES[0].0, EdgeKind::Flow, 1.0);
        for pair in STAGES.windows(2) {
            g.edge(pair[0].0, pair[1].0, EdgeKind::Flow, 1.0);
        }

        // Candidates hang off the reranker, sized by score.
        for cand in candidates {
            let node_id = format!("cand:{}", cand.id);
            let label = truncate(&cand.text, 60);
            g.node(&node_id, NodeKind::Candidate, label, Some(cand.score));
            g.edge("stage:reranker", &node_id, EdgeKind::Ranked, cand.score);
        }

        // Verdict.
        let verdict_label = format!(
            "{} ({:.0}%)",
            if readiness.ready {
                "READY"
            } else {
                &readiness.label
            },
            readiness.confidence * 100.0
        );
        g.node(
            "readiness",
            NodeKind::Readiness,
            verdict_label,
            Some(readiness.confidence),
        );
        g.edge(
            "stage:classifier",
            "readiness",
            EdgeKind::Verdict,
            readiness.confidence,
        );

        g
    }
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_has_query_pipeline_and_verdict() {
        let cands = vec![Candidate::new("1", "hello world", 0.9)];
        let r = Readiness::ready(0.8, "ok");
        let g = Connectome::new().map("hi", &cands, &r);
        assert!(g.nodes.iter().any(|n| n.kind == NodeKind::Query));
        assert!(g.nodes.iter().any(|n| n.id == "cand:1"));
        assert!(g.nodes.iter().any(|n| n.kind == NodeKind::Readiness));
        // JSON + DOT both render.
        assert!(g.to_json().unwrap().contains("\"nodes\""));
        assert!(g.to_dot().contains("digraph connectome"));
    }
}
