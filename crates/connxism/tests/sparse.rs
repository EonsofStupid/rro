//! Sprint 11 gates: weighted sparse vectors — exact ranking vs brute force,
//! planted-dimension retrieval, retraction on overwrite/remove, and fusion.

use connxism::{Estate, EstateQuery};
use rro_core::{Embedding, Recall, SparseVector, VectorRecord};

fn lcg(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed as f64 / u64::MAX as f64) as f32) * 2.0 - 1.0
}

/// Deterministic sparse vector: ~8 non-zeros drawn from dims [0, 50).
fn sparse(seed: u64) -> SparseVector {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let pairs: Vec<(u32, f32)> = (0..8)
        .map(|_| {
            let w = lcg(&mut s);
            let dim = ((s >> 17) % 50) as u32;
            (dim, w.abs() + 0.05)
        })
        .collect();
    SparseVector::new(pairs)
}

async fn seed_corpus(estate: &Estate, n: usize) -> (connxism::ConnXRecall, Vec<SparseVector>) {
    let recall = estate.recall();
    let mut sparses = Vec::with_capacity(n);
    let mut records = Vec::with_capacity(n);
    for i in 0..n {
        let sv = sparse(i as u64);
        sparses.push(sv.clone());
        let mut s = (i as u64).wrapping_add(7);
        let dense = Embedding((0..16).map(|_| lcg(&mut s)).collect());
        records.push(
            VectorRecord::new(format!("doc{i}"), dense, format!("sparse corpus entry {i}"))
                .with_sparse(sv),
        );
    }
    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();
    (recall, sparses)
}

#[tokio::test(flavor = "multi_thread")]
async fn sparse_ranking_matches_brute_force() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "sp").unwrap();
    let (recall, sparses) = seed_corpus(&estate, 200).await;

    for qseed in [1000u64, 2000, 3000] {
        let q = sparse(qseed);
        let hits = recall.sparse_search(&q, 10).await.unwrap();

        // Brute force over the in-test copies.
        let mut truth: Vec<(usize, f32)> = sparses
            .iter()
            .map(|sv| q.dot(sv))
            .enumerate()
            .filter(|(_, s)| *s > 0.0)
            .collect();
        truth.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| format!("doc{}", a.0).cmp(&format!("doc{}", b.0)))
        });
        truth.truncate(10);

        assert_eq!(hits.len(), truth.len().min(10));
        for (hit, (ti, ts)) in hits.iter().zip(&truth) {
            assert_eq!(hit.id.as_str(), format!("doc{ti}"), "rank order matches");
            assert!(
                (hit.score - ts).abs() < 1e-5,
                "scores are exact dots: {} vs {ts}",
                hit.score
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn planted_dimension_and_retraction() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "plant").unwrap();
    let (recall, _) = seed_corpus(&estate, 50).await;

    // Plant a unique dimension on one doc.
    let golden = VectorRecord::new(
        "golden",
        Embedding(vec![0.5; 16]),
        "the planted golden entry",
    )
    .with_sparse(SparseVector::new([(9999u32, 2.5f32)]));
    recall.upsert(vec![golden]).await.unwrap();

    let probe = SparseVector::new([(9999u32, 1.0f32)]);
    let hits = recall.sparse_search(&probe, 5).await.unwrap();
    assert_eq!(hits.len(), 1, "df=1 dimension hits exactly one doc");
    assert_eq!(hits[0].id.as_str(), "golden");
    assert!((hits[0].score - 2.5).abs() < 1e-6);

    // Overwrite without the planted dim → the old row must retract.
    let rewritten = VectorRecord::new("golden", Embedding(vec![0.5; 16]), "rewritten")
        .with_sparse(SparseVector::new([(7u32, 1.0f32)]));
    recall.upsert(vec![rewritten]).await.unwrap();
    assert!(
        recall.sparse_search(&probe, 5).await.unwrap().is_empty(),
        "overwrite retracts old sparse rows"
    );

    // Remove → all rows gone.
    recall.remove(&"golden".into()).await.unwrap();
    let probe7 = SparseVector::new([(7u32, 1.0f32)]);
    assert!(recall
        .sparse_search(&probe7, 5)
        .await
        .unwrap()
        .iter()
        .all(|c| c.id.as_str() != "golden"));
}

#[tokio::test(flavor = "multi_thread")]
async fn sparse_fuses_into_the_query_plane() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "fuse").unwrap();
    let (recall, _) = seed_corpus(&estate, 50).await;

    // A doc that is dense-invisible (orthogonal-ish) but sparse-strong.
    let target = VectorRecord::new(
        "target",
        Embedding({
            let mut v = vec![0.0f32; 16];
            v[15] = 1.0;
            v
        }),
        "unrelated wording entirely",
    )
    .with_sparse(SparseVector::new([(4242u32, 3.0f32)]));
    recall.upsert(vec![target]).await.unwrap();
    recall.quiesce().await.unwrap();

    let mut qs = 999u64;
    let qv = Embedding((0..16).map(|_| lcg(&mut qs)).collect());

    // Without sparse: the target does not surface for this query.
    let plain = recall
        .query(EstateQuery::hybrid("sparse corpus entry", qv.clone(), 5))
        .await
        .unwrap();
    assert!(plain.iter().all(|c| c.id.as_str() != "target"));

    // With the sparse half: fusion pulls it in.
    let fused = recall
        .query(
            EstateQuery::hybrid("sparse corpus entry", qv, 5)
                .sparse_vector(SparseVector::new([(4242u32, 1.0f32)])),
        )
        .await
        .unwrap();
    assert!(
        fused.iter().any(|c| c.id.as_str() == "target"),
        "sparse ranking must fuse into the results: {:?}",
        fused.iter().map(|c| c.id.as_str()).collect::<Vec<_>>()
    );
    // Sparse-only query works too (no text, no dense vector).
    let sparse_only = recall
        .query(EstateQuery {
            sparse: Some(SparseVector::new([(4242u32, 1.0f32)])),
            top_k: 3,
            ..EstateQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(sparse_only.len(), 1);
    assert_eq!(sparse_only[0].id.as_str(), "target");
}
