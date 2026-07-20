//! Sprint 27 gates: highlights ride candidates over the wire (analyzer-
//! aware, offset-exact), the `info` verb reports the live catalog, and
//! feed stats track writes.

use std::sync::Arc;

use rro_client::Client;
use rro_core::{Embedding, EstateQuery, Recall, VectorRecord};
use rro_engine::{FlowNode, ReasonReadyObject};
use rro_net::tcp;

fn rec(id: &str, text: &str) -> VectorRecord {
    VectorRecord::new(id, Embedding(vec![0.3, 0.4, 0.2, 0.1]), text).in_collection("docs")
}

#[tokio::test(flavor = "multi_thread")]
async fn highlights_ride_candidates_over_the_wire() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(
        connxism::Estate::open_with(
            dir.path(),
            "hl",
            connxism::EstateConfig {
                analyzer: rro_core::text::Analyzer::stemming(),
                ..connxism::EstateConfig::default()
            },
        )
        .unwrap(),
    );
    let recall = estate.recall();
    recall
        .upsert(vec![
            rec(
                "a",
                "The runner was running quickly through connected estates",
            ),
            rec("b", "entirely unrelated words in this document"),
        ])
        .await
        .unwrap();

    // Local: stemmed query highlights inflected surface forms, offset-exact.
    let hits = recall
        .query(
            EstateQuery {
                text: Some("run connection".into()),
                vector: Some(Embedding(vec![0.3, 0.4, 0.2, 0.1])),
                top_k: 2,
                ..EstateQuery::default()
            }
            .highlighted(),
        )
        .await
        .unwrap();
    let top = hits.iter().find(|c| c.id.as_str() == "a").expect("doc a");
    let words: Vec<&str> = top
        .highlights
        .iter()
        .map(|&(s, e)| &top.text[s..e])
        .collect();
    assert_eq!(words, vec!["running", "connected"]);

    // Default: no highlights.
    let plain = recall
        .query(EstateQuery::hybrid(
            "run connection",
            Embedding(vec![0.3, 0.4, 0.2, 0.1]),
            2,
        ))
        .await
        .unwrap();
    assert!(plain.iter().all(|c| c.highlights.is_empty()));

    // Over the wire: spans survive serde on the query verb.
    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = FlowNode::new(flow, "hl-node").with_estate(estate.clone());
    let (addr, _task) = tcp::serve("127.0.0.1:0", Arc::new(node)).await.unwrap();
    let wire = Client::new(addr.to_string())
        .query(
            &EstateQuery {
                text: Some("run connection".into()),
                vector: Some(Embedding(vec![0.3, 0.4, 0.2, 0.1])),
                top_k: 2,
                ..EstateQuery::default()
            }
            .highlighted(),
        )
        .await
        .unwrap();
    let top = wire.iter().find(|c| c.id.as_str() == "a").expect("doc a");
    let words: Vec<&str> = top
        .highlights
        .iter()
        .map(|&(s, e)| &top.text[s..e])
        .collect();
    assert_eq!(words, vec!["running", "connected"]);

    // Old candidate payloads (no highlights field) still parse.
    let old: rro_core::Candidate =
        serde_json::from_str(r#"{"id":"x","text":"t","score":0.5}"#).unwrap();
    assert!(old.highlights.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn info_verb_reports_the_live_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "cat").unwrap());
    estate.create_payload_index("team").unwrap();
    estate.create_alias("prod", "docs").unwrap();
    let recall = estate.recall();
    recall
        .upsert(vec![rec("a", "first entry"), rec("b", "second entry")])
        .await
        .unwrap();
    recall.remove(&"b".into()).await.unwrap();

    // Feed stats track writes: 3 rows (2 upserts + 1 remove), 0-based.
    let stats = estate.feed_stats().unwrap();
    assert_eq!(stats.first_seq, Some(0));
    assert_eq!(stats.next_seq, 3);
    assert_eq!(stats.retained, 3);

    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = FlowNode::new(flow, "cat-node").with_estate(estate.clone());
    let (addr, _task) = tcp::serve("127.0.0.1:0", Arc::new(node)).await.unwrap();
    let info = Client::new(addr.to_string()).info().await.unwrap();

    assert_eq!(info["estate"]["name"], "cat");
    assert_eq!(info["estate"]["analyzer"]["tokenizer"], "word");
    assert_eq!(info["payload_indexes"][0], "team");
    assert_eq!(info["collections"][0][0], "docs");
    assert_eq!(info["collections"][0][1], 1);
    assert_eq!(info["aliases"]["prod"], "docs");
    assert_eq!(info["feed"]["next_seq"], 3);
    assert_eq!(info["feed"]["retained"], 3);
    assert_eq!(info["health"]["docs"], 1);
}
