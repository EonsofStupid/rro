//! End-to-end Reason Ready demo.
//!
//! Run: `cargo run --example demo -p rro-engine`
//!
//! Indexes a small corpus with the default (weightless) engine, then runs a few
//! queries through the full flow and prints the ranked context, the reason-ready
//! verdict, and the connectome map.

use rro_engine::{sample_corpus, ReasonReadyObject};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let flow = ReasonReadyObject::default_engine();

    let indexed = flow.index(sample_corpus()).await?;
    println!("== Reason Ready — RRO ==");
    print!("components:");
    for (stage, model) in flow.model_names() {
        print!("  {stage}={model}");
    }
    println!("\nindexed {indexed} documents\n");

    let queries = [
        "how do I upgrade postgres safely?",
        "what does a reranker do?",
        "recipe with ripe bananas",
        "tell me about undersea volcanoes",
    ];

    for q in queries {
        let (result, map) = flow.ask_with_map(q).await?;
        let r = &result.readiness;
        println!("query: {q}");
        println!(
            "  reason-ready: {} [{}] confidence={:.2} — {}",
            if r.ready { "YES" } else { "no" },
            r.label,
            r.confidence,
            r.rationale
        );
        for (i, c) in result.candidates.iter().enumerate() {
            println!("  {}. ({:.3}) {}", i + 1, c.score, truncate(&c.text, 72));
        }
        println!(
            "  connectome: {} nodes / {} edges\n",
            map.nodes.len(),
            map.edges.len()
        );
    }

    // Show one full connectome map as DOT — paste into any Graphviz viewer.
    let (_r, map) = flow.ask_with_map("what does a reranker do?").await?;
    println!("---- connectome (Graphviz DOT) for 'what does a reranker do?' ----");
    println!("{}", map.to_dot());

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max - 1).collect();
        out.push('…');
        out
    }
}
