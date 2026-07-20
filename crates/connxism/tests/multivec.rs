//! Sprint 12 gates: multi-vector per point — named spaces vs brute force,
//! cross-space isolation, per-point update/retraction, per-name dim guard,
//! MaxSim late-interaction rescore, and the query-plane wiring.

use connxism::{Estate, EstateQuery};
use rro_core::{maxsim, Embedding, Recall, VectorRecord};

fn lcg(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed as f64 / u64::MAX as f64) as f32) * 2.0 - 1.0
}

fn vec_of(seed: u64, dim: usize) -> Embedding {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    Embedding((0..dim).map(|_| lcg(&mut s)).collect())
}

/// 40 docs, each with a default 16-dim vector plus named `title` (8-dim)
/// and `body` (24-dim) vectors.
async fn seed(estate: &Estate) -> connxism::ConnXRecall {
    let recall = estate.recall();
    let records: Vec<VectorRecord> = (0..40)
        .map(|i| {
            VectorRecord::new(
                format!("doc{i}"),
                vec_of(i as u64, 16),
                format!("named corpus entry {i}"),
            )
            .with_named("title", vec_of(1000 + i as u64, 8))
            .with_named("body", vec_of(2000 + i as u64, 24))
        })
        .collect();
    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();
    recall
}

#[tokio::test(flavor = "multi_thread")]
async fn named_search_matches_brute_force_and_isolates_spaces() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "nv").unwrap();
    let recall = seed(&estate).await;

    let q = vec_of(999, 8);
    let hits = recall.named_search("title", &q, 5).await.unwrap();
    assert_eq!(hits.len(), 5);

    // Brute force over the same title vectors.
    let mut truth: Vec<(String, f32)> = (0..40)
        .map(|i| (format!("doc{i}"), q.cosine(&vec_of(1000 + i as u64, 8))))
        .collect();
    truth.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (hit, (tid, ts)) in hits.iter().zip(&truth) {
        assert_eq!(hit.id.as_str(), tid);
        assert!((hit.score - ts).abs() < 1e-5);
    }

    // Cross-space isolation: rank the same corpus in body-space; the top
    // title hit must not be the top body hit by construction of the seeds.
    let qb = vec_of(999, 24);
    let body_hits = recall.named_search("body", &qb, 5).await.unwrap();
    let mut body_truth: Vec<(String, f32)> = (0..40)
        .map(|i| (format!("doc{i}"), qb.cosine(&vec_of(2000 + i as u64, 24))))
        .collect();
    body_truth.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    assert_eq!(body_hits[0].id.as_str(), body_truth[0].0);
    assert_ne!(
        hits[0].id.as_str(),
        body_hits[0].id.as_str(),
        "title and body spaces rank independently"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn per_point_update_retraction_and_dim_guard() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "upd").unwrap();
    let recall = seed(&estate).await;

    // Make doc7's title vector exactly the probe → it must rank first.
    let probe = vec_of(31337, 8);
    let updated = VectorRecord::new("doc7", vec_of(7, 16), "named corpus entry 7")
        .with_named("title", probe.clone())
        .with_named("body", vec_of(2007, 24));
    recall.upsert(vec![updated]).await.unwrap();
    let hits = recall.named_search("title", &probe, 1).await.unwrap();
    assert_eq!(hits[0].id.as_str(), "doc7");
    assert!((hits[0].score - 1.0).abs() < 1e-5, "exact match cosine = 1");

    // Overwrite doc7 WITHOUT a title vector → its title row must retract.
    let stripped = VectorRecord::new("doc7", vec_of(7, 16), "named corpus entry 7")
        .with_named("body", vec_of(2007, 24));
    recall.upsert(vec![stripped]).await.unwrap();
    let hits = recall.named_search("title", &probe, 40).await.unwrap();
    assert!(
        hits.iter().all(|c| c.id.as_str() != "doc7"),
        "dropped name retracts its row"
    );
    // Body space is untouched by the title retraction.
    let body_hits = recall
        .named_search("body", &vec_of(2007, 24), 1)
        .await
        .unwrap();
    assert_eq!(body_hits[0].id.as_str(), "doc7");

    // Remove retracts every named row.
    recall.remove(&"doc7".into()).await.unwrap();
    let body_hits = recall
        .named_search("body", &vec_of(2007, 24), 40)
        .await
        .unwrap();
    assert!(body_hits.iter().all(|c| c.id.as_str() != "doc7"));

    // Per-name dim guard: `title` was fixed at 8 dims.
    let bad = VectorRecord::new("bad", vec_of(1, 16), "bad").with_named("title", vec_of(2, 9));
    assert!(recall.upsert(vec![bad]).await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn maxsim_rescore_beats_plain_dense_on_planted_token() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "ms").unwrap();
    let recall = estate.recall();

    // Corpus: docs whose pooled (default) vectors all sit near the query,
    // each carrying bland token vectors.
    let mut records: Vec<VectorRecord> = Vec::new();
    let qdense = vec_of(555, 16);
    for i in 0..30 {
        let mut dense = qdense.clone();
        // Small per-doc jitter: all docs are dense-plausible.
        let mut s = i as u64 + 1;
        for x in dense.0.iter_mut() {
            *x += lcg(&mut s) * 0.15;
        }
        records.push(
            VectorRecord::new(format!("doc{i}"), dense, format!("token corpus entry {i}"))
                .with_multi(vec![
                    vec_of(3000 + i as u64, 12),
                    vec_of(4000 + i as u64, 12),
                ]),
        );
    }
    // The target: dense-mediocre (extra jitter) but one token vector exactly
    // matches a query token.
    let planted = vec_of(77777, 12);
    let mut far = qdense.clone();
    let mut s = 424242u64;
    for x in far.0.iter_mut() {
        *x += lcg(&mut s) * 0.6;
    }
    records.push(
        VectorRecord::new("target", far, "the planted token entry")
            .with_multi(vec![planted.clone(), vec_of(5000, 12)]),
    );
    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();

    // A query token aligned with the planted token dominates MaxSim: scale
    // it up so Σ max dots is decisive.
    let qtokens = vec![Embedding(planted.0.iter().map(|x| x * 4.0).collect())];

    // Plain dense: target is NOT first (it is the dense-farthest doc).
    let plain = recall
        .query(EstateQuery::hybrid("token corpus entry", qdense.clone(), 5))
        .await
        .unwrap();
    assert_ne!(plain[0].id.as_str(), "target");

    // MaxSim rescore: target must be first, score exactly brute force.
    let rescored = recall
        .query(
            EstateQuery::hybrid("token corpus entry", qdense.clone(), 5)
                .multi_query(qtokens.clone()),
        )
        .await
        .unwrap();
    assert_eq!(
        rescored[0].id.as_str(),
        "target",
        "late interaction surfaces the planted-token doc: {:?}",
        rescored.iter().map(|c| c.id.as_str()).collect::<Vec<_>>()
    );
    let brute = maxsim(&qtokens, &[planted, vec_of(5000, 12)]);
    assert!((rescored[0].score - brute).abs() < 1e-3);
    assert!(!rescored[0].text.is_empty(), "winners are hydrated");
}

#[tokio::test(flavor = "multi_thread")]
async fn query_plane_using_and_wire_shape() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "qp").unwrap();
    let recall = seed(&estate).await;

    // `using` routes the dense half to the named space.
    let probe = vec_of(1000 + 13, 8); // doc13's exact title vector
    let hits = recall
        .query(
            EstateQuery {
                vector: Some(probe.clone()),
                top_k: 3,
                ..EstateQuery::default()
            }
            .using("title"),
        )
        .await
        .unwrap();
    assert_eq!(hits[0].id.as_str(), "doc13");
    assert!(!hits[0].text.is_empty());

    // The new fields ride the wire: serde roundtrip, and old payloads
    // (without them) still parse.
    let q = EstateQuery::hybrid("t", vec_of(1, 16), 5)
        .using("title")
        .multi_query(vec![vec_of(2, 12)]);
    let json = serde_json::to_string(&q).unwrap();
    let back: EstateQuery = serde_json::from_str(&json).unwrap();
    assert_eq!(back.using.as_deref(), Some("title"));
    assert_eq!(back.multi.as_ref().unwrap().len(), 1);
    let old: EstateQuery = serde_json::from_str(r#"{"text":"x","vector":null,"top_k":3}"#).unwrap();
    assert!(old.using.is_none() && old.multi.is_none());
}
