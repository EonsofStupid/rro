//! Sprint 10 gates: grouped search, recommend-by-example, context-steered
//! discovery, and batched queries.

use connxism::{Estate, EstateQuery};
use rro_core::{Embedding, Metadata, Recall, VectorRecord};

fn noise(seed: u64, dim: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (((x as f64 / u64::MAX as f64) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Two well-separated clusters (A around +e1, B around +e2) with per-doc noise.
async fn seed_clusters(estate: &Estate, per_cluster: usize, dim: usize) -> connxism::ConnXRecall {
    let recall = estate.recall();
    let mut records = Vec::new();
    for (cluster, base_axis) in [("a", 0usize), ("b", 1usize)] {
        for i in 0..per_cluster {
            let mut v = noise(
                (cluster.len() * 1000 + i) as u64 + base_axis as u64 * 77,
                dim,
                0.15,
            );
            v[base_axis] += 1.0;
            let mut r = VectorRecord::new(
                format!("{cluster}{i}"),
                Embedding(v),
                format!("cluster {cluster} member {i}"),
            );
            let mut m = Metadata::new();
            m.insert("cluster".into(), serde_json::json!(cluster));
            m.insert("slot".into(), serde_json::json!(i % 4));
            r.metadata = m;
            records.push(r);
        }
    }
    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();
    recall
}

#[tokio::test(flavor = "multi_thread")]
async fn grouped_search_invariants() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "grp").unwrap();
    let recall = seed_clusters(&estate, 40, 16).await;

    let mut qv = vec![0.0f32; 16];
    qv[0] = 1.0;
    qv[1] = 1.0; // between the clusters: both groups should surface
    let groups = recall
        .query_grouped(
            EstateQuery::hybrid("cluster member", Embedding(qv), 0).ids_only(),
            "slot",
            3,
            4,
        )
        .await
        .unwrap();

    assert!(!groups.is_empty() && groups.len() <= 3, "≤3 groups");
    let mut seen_keys = std::collections::HashSet::new();
    for g in &groups {
        assert!(seen_keys.insert(g.key.clone()), "group keys are distinct");
        assert!(!g.hits.is_empty() && g.hits.len() <= 4, "≤4 hits per group");
        for c in &g.hits {
            assert_eq!(
                c.metadata.get("slot").map(ToString::to_string),
                Some(g.key.clone()),
                "every hit belongs to its group"
            );
        }
    }
    // Groups are ordered by their best hit.
    for w in groups.windows(2) {
        assert!(w[0].hits[0].score >= w[1].hits[0].score);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recommend_steers_toward_positives_and_away_from_negatives() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "rec").unwrap();
    let recall = seed_clusters(&estate, 40, 16).await;

    let hits = recall
        .recommend(
            &["a0".to_string(), "a1".to_string()],
            &["b0".to_string()],
            10,
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 10);
    for c in &hits {
        assert!(
            c.id.as_str() != "a0" && c.id.as_str() != "a1" && c.id.as_str() != "b0",
            "examples never appear in results"
        );
    }
    let from_a = hits
        .iter()
        .filter(|c| c.id.as_str().starts_with('a'))
        .count();
    assert!(
        from_a == 10,
        "steering must put all top-10 in the positive cluster, got {from_a}/10"
    );

    // No known positive examples → typed error, not silence.
    assert!(recall
        .recommend(&["nope".to_string()], &[], 5)
        .await
        .is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn discover_ranks_by_context_pair_agreement() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "dis").unwrap();
    let recall = seed_clusters(&estate, 40, 16).await;

    // Query sits exactly between the clusters; the context pair points at A.
    let mut qv = vec![0.0f32; 16];
    qv[0] = 1.0;
    qv[1] = 1.0;
    let q = Embedding(qv);

    let neutral = recall.search(&q, 10).await.unwrap();
    let a_neutral = neutral
        .iter()
        .filter(|c| c.id.as_str().starts_with('a'))
        .count();

    let steered = recall
        .discover(&q, &[("a2".to_string(), "b2".to_string())], 10)
        .await
        .unwrap();
    let a_steered = steered
        .iter()
        .filter(|c| c.id.as_str().starts_with('a'))
        .count();

    println!("DISCOVER GATE — cluster-A hits: neutral {a_neutral}/10 → steered {a_steered}/10");
    assert!(
        a_steered > a_neutral || a_steered == 10,
        "context pairs must pull results toward the positive side: {a_neutral} → {a_steered}"
    );
    assert_eq!(steered.len(), 10);
}

#[tokio::test(flavor = "multi_thread")]
async fn batch_equals_sequential() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "bat").unwrap();
    let recall = seed_clusters(&estate, 20, 16).await;

    let mut qa = vec![0.0f32; 16];
    qa[0] = 1.0;
    let mut qb = vec![0.0f32; 16];
    qb[1] = 1.0;
    let queries = vec![
        EstateQuery::hybrid("cluster a member", Embedding(qa), 5),
        EstateQuery::hybrid("cluster b member", Embedding(qb), 5),
    ];

    let batched = recall.query_batch(queries.clone()).await.unwrap();
    assert_eq!(batched.len(), 2);
    for (q, expect) in queries.into_iter().zip(&batched) {
        let single = recall.query(q).await.unwrap();
        let ids: Vec<&str> = single.iter().map(|c| c.id.as_str()).collect();
        let bids: Vec<&str> = expect.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, bids, "batch results equal one-at-a-time results");
    }
}
