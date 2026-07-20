//! The P3 gate, measured: **route → recall beats flat hybrid on a linked
//! corpus.**
//!
//! Construction: every query has a golden doc AND a decoy doc carrying the
//! *same anchor term* — lexically and semantically near-identical, so flat
//! hybrid cannot reliably tell them apart. But the golden is RELATEd to the
//! query's project and the decoy to a different one. The map resolves the
//! route (project → contained docs), the treasure answers inside it: routed
//! accuracy must be perfect while flat sits near a coin flip.

use connxism::{Estate, TraversalSpec};
use embedder::DeterministicEmbedder;
use rro_core::{Embedder, Recall, VectorRecord};

const QUERIES: usize = 40;
const NOISE: usize = 1500; // past the ANN threshold — realistic dense path

#[tokio::test(flavor = "multi_thread")]
async fn routed_recall_beats_flat_hybrid_on_ambiguity() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "routing-gate").unwrap();
    let recall = estate.recall();
    let embed = DeterministicEmbedder::new();

    let mut records = Vec::new();
    let mut queries = Vec::new();

    for i in 0..NOISE {
        let text = format!(
            "background noise document number {i} about topic {}",
            i % 13
        );
        records.push(rec(&embed, &format!("noise{i}"), &text).await);
    }

    for q in 0..QUERIES {
        let anchor = format!("anchorterm{q}");
        // Golden and decoy share the anchor; filler barely differs.
        let gold_text = format!("{anchor} rollout checklist steps alpha section");
        let decoy_text = format!("{anchor} rollout checklist steps omega section");
        records.push(rec(&embed, &format!("gold{q}"), &gold_text).await);
        records.push(rec(&embed, &format!("decoy{q}"), &decoy_text).await);

        // The map: golden belongs to this query's project, decoy elsewhere.
        estate
            .relate(&format!("proj{q}"), "contains", &format!("gold{q}"))
            .unwrap();
        estate
            .relate("proj-other", "contains", &format!("decoy{q}"))
            .unwrap();

        queries.push((format!("{anchor} rollout checklist"), q));
    }

    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();

    let mut flat_hits = 0usize;
    let mut routed_hits = 0usize;

    for (query_text, q) in &queries {
        let qemb = embed.embed_one(query_text).await.unwrap();
        let gold_id = format!("gold{q}");

        // Flat hybrid: whole estate, no map.
        let flat = recall.hybrid_search(query_text, &qemb, 1).await.unwrap();
        flat_hits += usize::from(flat.first().map(|c| c.id.as_str()) == Some(gold_id.as_str()));

        // Routed: the map resolves the project's neighborhood, the treasure
        // answers inside it.
        let scope = estate
            .traverse(&[&format!("proj{q}")], &TraversalSpec::default())
            .unwrap();
        let routed = recall
            .scoped_search(query_text, &qemb, 1, scope)
            .await
            .unwrap();
        routed_hits += usize::from(routed.first().map(|c| c.id.as_str()) == Some(gold_id.as_str()));
    }

    let flat_acc = flat_hits as f64 / QUERIES as f64;
    let routed_acc = routed_hits as f64 / QUERIES as f64;
    println!("P3 GATE — flat hybrid accuracy@1: {flat_acc:.3}; routed accuracy@1: {routed_acc:.3}");

    assert_eq!(
        routed_acc, 1.0,
        "routed recall must disambiguate perfectly on the linked corpus"
    );
    assert!(
        flat_acc <= 0.8,
        "the corpus must be genuinely ambiguous for the gate to mean anything (flat = {flat_acc})"
    );
    assert!(
        routed_acc > flat_acc,
        "route→recall must beat flat hybrid (routed {routed_acc} vs flat {flat_acc})"
    );
}

async fn rec(embed: &DeterministicEmbedder, id: &str, text: &str) -> VectorRecord {
    VectorRecord::new(id, embed.embed_one(text).await.unwrap(), text)
}
