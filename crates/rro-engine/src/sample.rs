//! A tiny built-in corpus so a fresh daemon / demo is queryable out of the box.

use rro_core::Document;

/// A handful of documents spanning a few topics.
pub fn sample_corpus() -> Vec<Document> {
    [
        ("d1", "Postgres major version upgrades require a dump and restore or pg_upgrade; always take a backup and test the rollback path first."),
        ("d2", "Vector search finds nearest neighbours by cosine or dot product over dense embeddings; recall quality depends on the embedding model."),
        ("d3", "A reranker is a cross-encoder that re-scores retrieved candidates against the query for sharper ordering than first-stage retrieval."),
        ("d4", "Banana bread is best with overripe bananas, a little cinnamon, and melted butter folded into the batter before baking."),
        ("d5", "Agent-to-agent (a2a) protocols let independent agents exchange requests and results over a shared message contract."),
        ("d6", "Tokio is an asynchronous runtime for Rust; signal handling lets a daemon shut down cleanly on Ctrl-C or SIGTERM."),
    ]
    .into_iter()
    .map(|(id, text)| Document::new(text).with_id(id))
    .collect()
}
