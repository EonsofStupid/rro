//! Integration: the full estate roundtrip — registry, persistence, hybrid.

use connxism::{
    ConnectorInfo, ConnectorKind, Estate, NodeInfo, SyncState, SyncStatus, Transport, WarpPoint,
};
use rro_core::{Embedding, Recall, VectorRecord};

fn rec(id: &str, v: &[f32], text: &str) -> VectorRecord {
    VectorRecord::new(id, Embedding(v.to_vec()), text)
}

#[tokio::test(flavor = "multi_thread")]
async fn estate_roundtrip_and_hybrid() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "test-estate").unwrap();

    // ---- registry: node + warp points ----
    estate
        .register_node(NodeInfo {
            id: "n1".into(),
            name: "edge-agent".into(),
            warp_points: vec![WarpPoint {
                transport: Transport::Tcp,
                address: "127.0.0.1:7878".into(),
                capabilities: vec!["ask".into(), "map".into()],
            }],
            last_seen: 0,
        })
        .unwrap();
    estate
        .add_warp_point(
            "n1",
            WarpPoint {
                transport: Transport::Mcp,
                address: "mesh://n1".into(),
                capabilities: vec!["ingest".into()],
            },
        )
        .unwrap();
    let node = estate.node("n1").unwrap().unwrap();
    assert_eq!(node.warp_points.len(), 2);

    // ---- registry: connector + sync state ----
    estate
        .register_connector(ConnectorInfo {
            id: "c1".into(),
            name: "team drive".into(),
            kind: ConnectorKind::Drive,
            provider: "drive".into(),
            uri: "drive://team".into(),
            sync: SyncState::default(),
            registered_at: 0,
        })
        .unwrap();
    estate
        .update_sync(
            "c1",
            SyncState {
                cursor: Some("page-2".into()),
                docs_synced: 42,
                last_sync: Some(connxism::now_ms()),
                status: SyncStatus::Idle,
            },
        )
        .unwrap();
    let conn = estate.connector("c1").unwrap().unwrap();
    assert_eq!(conn.sync.docs_synced, 42);
    assert_eq!(conn.sync.status, SyncStatus::Idle);

    // ---- documents: dense + lexical + hybrid ----
    let recall = estate.recall();
    recall
        .upsert(vec![
            rec(
                "d1",
                &[1.0, 0.0, 0.0],
                "postgres upgrade guide with rollback steps",
            ),
            rec("d2", &[0.0, 1.0, 0.0], "banana bread recipe with cinnamon"),
            rec(
                "d3",
                &[0.9, 0.1, 0.0],
                "database migration checklist for postgres",
            ),
        ])
        .await
        .unwrap();
    assert_eq!(recall.len().await.unwrap(), 3);

    // Dense: nearest to d1's vector.
    let dense = recall
        .search(&Embedding(vec![1.0, 0.0, 0.0]), 2)
        .await
        .unwrap();
    assert_eq!(dense[0].id.as_str(), "d1");

    // Lexical: BM25 finds the postgres docs, not the recipe.
    let lex = recall
        .lexical_search("postgres migration", 3)
        .await
        .unwrap();
    assert!(lex.iter().any(|c| c.id.as_str() == "d3"));
    assert!(lex.iter().all(|c| c.id.as_str() != "d2"));

    // Hybrid: vector says d1, lexical says d3 — fusion surfaces both on top.
    let hybrid = recall
        .hybrid_search("postgres migration", &Embedding(vec![1.0, 0.0, 0.0]), 2)
        .await
        .unwrap();
    let top2: Vec<&str> = hybrid.iter().map(|c| c.id.as_str()).collect();
    assert!(top2.contains(&"d1") && top2.contains(&"d3"), "got {top2:?}");
    assert!(
        !hybrid[0].text.is_empty(),
        "hybrid candidates carry payloads"
    );

    // ---- overwrite semantics ----
    recall
        .upsert(vec![rec(
            "d2",
            &[0.0, 1.0, 0.0],
            "sourdough starter instructions",
        )])
        .await
        .unwrap();
    assert_eq!(
        recall.len().await.unwrap(),
        3,
        "overwrite must not duplicate"
    );
    let lex = recall.lexical_search("banana cinnamon", 3).await.unwrap();
    assert!(
        lex.is_empty(),
        "old postings must be retracted, got {lex:?}"
    );

    // ---- remove ----
    recall.remove(&"d2".into()).await.unwrap();
    assert_eq!(recall.len().await.unwrap(), 2);

    // ---- tags ----
    estate.tag("d1", &["ops".into(), "db".into()]).unwrap();
    estate.tag("d3", &["db".into()]).unwrap();
    let db_docs = estate.docs_by_tag("db").unwrap();
    let all_tags = estate.tags().unwrap();
    let ops_docs = estate.docs_by_tag("ops").unwrap();
    assert_eq!(
        db_docs.len(),
        2,
        "docs_by_tag(db)={db_docs:?} tags={all_tags:?} ops={ops_docs:?}"
    );

    // ---- shapes census exists; trends record + read back ----
    estate.record_trend("ingest.docs_per_sec", 1234.5).unwrap();
    estate.record_trend("ingest.docs_per_sec", 2345.6).unwrap();
    let series = estate.trend("ingest.docs_per_sec").unwrap();
    assert_eq!(series.len(), 2);
    assert!(series[0].at <= series[1].at);
}

#[tokio::test(flavor = "multi_thread")]
async fn estate_persists_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let estate = Estate::open(dir.path(), "persist").unwrap();
        estate
            .recall()
            .upsert(vec![rec(
                "p1",
                &[0.5, 0.5],
                "durable memory survives restarts",
            )])
            .await
            .unwrap();
    } // estate dropped, DB closed

    let estate = Estate::open(dir.path(), "persist").unwrap();
    let recall = estate.recall();
    assert_eq!(recall.len().await.unwrap(), 1);
    let hits = recall.lexical_search("durable memory", 1).await.unwrap();
    assert_eq!(hits[0].id.as_str(), "p1");
    // Dimension guard survived too.
    let err = recall
        .upsert(vec![rec("p2", &[1.0, 0.0, 0.0], "wrong dim")])
        .await;
    assert!(err.is_err(), "dim guard must persist across reopen");
}

#[tokio::test(flavor = "multi_thread")]
async fn read_your_writes_through_pending_overlay() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "ryw").unwrap();
    let recall = estate.recall();

    // Seed past the ANN threshold so dense search uses the graph + overlay.
    let seed: Vec<_> = (0..1500)
        .map(|i| {
            rec(
                &format!("seed{i}"),
                &[(i % 97) as f32, 1.0, 0.5],
                "background noise document",
            )
        })
        .collect();
    recall.upsert(seed).await.unwrap();

    // A fresh upsert must be findable IMMEDIATELY (before the applier runs).
    recall
        .upsert(vec![rec(
            "fresh",
            &[0.0, 0.0, 42.0],
            "fresh unique payload",
        )])
        .await
        .unwrap();
    let hits = recall
        .search(&Embedding(vec![0.0, 0.0, 1.0]), 3)
        .await
        .unwrap();
    assert_eq!(hits[0].id.as_str(), "fresh", "read-your-writes must hold");

    // A fresh remove must mask immediately, too.
    recall.remove(&"fresh".into()).await.unwrap();
    let hits = recall
        .search(&Embedding(vec![0.0, 0.0, 1.0]), 3)
        .await
        .unwrap();
    assert!(hits.iter().all(|c| c.id.as_str() != "fresh"));

    // And after quiesce, results are identical (graph caught up).
    recall.quiesce().await.unwrap();
    let hits = recall
        .search(&Embedding(vec![0.0, 0.0, 1.0]), 3)
        .await
        .unwrap();
    assert!(hits.iter().all(|c| c.id.as_str() != "fresh"));
}
