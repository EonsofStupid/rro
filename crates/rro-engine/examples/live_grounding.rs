//! Live end-to-end grounding proof against the running nemotron vLLM servers.
//!
//! Exercises the FULL RRO signal path with real models:
//!   index → SignalEmbedder emits `embed` → ModelNode → vLLM :8090 (nemotron)
//!   ask   → embed query + hybrid recall → SignalReranker emits `rerank` → vLLM :8092
//!
//! Run: `cargo run --example live_grounding` (with the embed/rerank servers up).
use rro_engine::{Document, EmbeddedEngine};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let embed = std::env::var("CLYFFY_EMBED_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8090/v1/embeddings".to_string());
    let rerank = std::env::var("CLYFFY_RERANK_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8092/rerank".to_string());

    println!("opening EmbeddedEngine::vllm_signals\n  embed={embed}\n  rerank={rerank}");
    let engine = EmbeddedEngine::vllm_signals(dir.path(), "clyffy", &embed, &rerank).await?;
    println!("engine up (real nemotron via the signal seam). dim={:?}\n", engine.dim());

    let docs = vec![
        Document::new("clyffy is Jesse's version of Claude — a generator and orchestrator, the forward-facing brand for all AI."),
        Document::new("RRO (Reason Ready Objects) is the Fjall-native recall + intelligence engine: signal-driven, hybrid search, RRD classifier."),
        Document::new("The GB10 trio serves the 120B brain (nemo-super-120b-nvfp4) over vLLM on port 8091."),
        Document::new("Banana bread is best with overripe bananas, a little cinnamon, and melted butter folded into the batter."),
        Document::new("The nemotron embedder and reranker are served on :8090 and :8092; RRO reaches them by emitting signals."),
    ];
    let n = engine.index(docs).await?;
    println!("indexed {n} reason-ready objects (embedded through the signal seam).\n");

    for q in ["what is RRO?", "how does clyffy reach its models?", "what serves the 120B brain?"] {
        let result = engine.ask(q).await?;
        println!("Q: {q}");
        for c in result.candidates.iter().take(3) {
            println!("   {:.4}  {}", c.score, c.text);
        }
        println!("   readiness: ready={} ({})\n", result.readiness.ready, result.readiness.label);
    }

    println!("END-TO-END OK: real nemotron embeddings + reranking through the RRO signal path.");
    Ok(())
}
