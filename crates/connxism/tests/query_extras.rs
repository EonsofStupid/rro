//! Sprint 17 gates: pagination (offset), vector selectors, deterministic
//! random sampling, and the pairwise similarity matrix.

use connxism::{Estate, EstateQuery};
use rro_core::{Embedding, Recall, VectorRecord};

fn lcg(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed as f64 / u64::MAX as f64) as f32) * 2.0 - 1.0
}

fn vec_of(seed: u64) -> Embedding {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    Embedding((0..8).map(|_| lcg(&mut s)).collect())
}

async fn seed(estate: &Estate, n: u64) -> connxism::ConnXRecall {
    let recall = estate.recall();
    let records: Vec<VectorRecord> = (0..n)
        .map(|i| {
            VectorRecord::new(
                format!("doc{i:02}"),
                vec_of(i),
                format!("paged corpus entry {i}"),
            )
        })
        .collect();
    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();
    recall
}

#[tokio::test(flavor = "multi_thread")]
async fn offset_paginates_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "pg").unwrap();
    let recall = seed(&estate, 30).await;

    // The full ranking at depth 15…
    let full = recall
        .query(EstateQuery::hybrid("paged corpus entry", vec_of(7), 15))
        .await
        .unwrap();
    assert_eq!(full.len(), 15);

    // …pages [0..5), [5..10), [10..15) reproduce it slice by slice.
    for page in 0..3usize {
        let hits = recall
            .query(EstateQuery::hybrid("paged corpus entry", vec_of(7), 5).offset(page * 5))
            .await
            .unwrap();
        let want: Vec<&str> = full[page * 5..page * 5 + 5]
            .iter()
            .map(|c| c.id.as_str())
            .collect();
        let got: Vec<&str> = hits.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(got, want, "page {page} equals the full ranking's slice");
    }

    // Offset past the corpus → empty.
    let hits = recall
        .query(EstateQuery::hybrid("paged corpus entry", vec_of(7), 5).offset(1000))
        .await
        .unwrap();
    assert!(hits.is_empty());

    // Wire shape: serde default keeps old payloads parsing.
    let old: EstateQuery = serde_json::from_str(r#"{"text":"x","vector":null,"top_k":3}"#).unwrap();
    assert_eq!(old.offset, 0);
    assert!(!old.with_vectors);
}

#[tokio::test(flavor = "multi_thread")]
async fn with_vectors_returns_stored_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "wv").unwrap();
    let recall = seed(&estate, 10).await;

    // Default: no vectors on candidates.
    let plain = recall
        .query(EstateQuery::hybrid("paged corpus entry", vec_of(3), 5))
        .await
        .unwrap();
    assert!(plain.iter().all(|c| c.vector.is_none()));

    // with_vectors: each winner carries exactly its upserted vector.
    let hits = recall
        .query(EstateQuery::hybrid("paged corpus entry", vec_of(3), 5).with_vectors())
        .await
        .unwrap();
    assert_eq!(hits.len(), 5);
    for c in &hits {
        let i: u64 = c.id.as_str()[3..].parse().unwrap();
        let stored = c.vector.as_ref().expect("vector requested");
        for (a, b) in stored.0.iter().zip(&vec_of(i).0) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // Candidate serde: vector rides when present, absent field parses.
    let json = serde_json::to_string(&hits[0]).unwrap();
    assert!(json.contains("\"vector\""));
    let old: rro_core::Candidate =
        serde_json::from_str(r#"{"id":"x","text":"t","score":0.5}"#).unwrap();
    assert!(old.vector.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn sampling_is_deterministic_and_distinct() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "sm").unwrap();
    let _ = seed(&estate, 40).await;

    let a = estate.sample(10, 42).unwrap();
    let b = estate.sample(10, 42).unwrap();
    assert_eq!(a.len(), 10);
    let ids_a: Vec<&str> = a.iter().map(|d| d.id.as_str()).collect();
    let ids_b: Vec<&str> = b.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(ids_a, ids_b, "same seed, same corpus → same sample");

    // Distinct members, all real.
    let set: std::collections::HashSet<&str> = ids_a.iter().copied().collect();
    assert_eq!(set.len(), 10);
    assert!(ids_a.iter().all(|id| id.starts_with("doc")));

    // A different seed draws a different sample (40 choose 10 — collision
    // would be astonishing).
    let c = estate.sample(10, 43).unwrap();
    let ids_c: Vec<&str> = c.iter().map(|d| d.id.as_str()).collect();
    assert_ne!(ids_a, ids_c);

    // n beyond the corpus returns the whole corpus.
    assert_eq!(estate.sample(1000, 1).unwrap().len(), 40);
    assert!(estate.sample(0, 1).unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn similarity_matrix_equals_direct_cosine() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "mx").unwrap();
    let recall = seed(&estate, 6).await;

    let ids: Vec<String> = (0..4).map(|i| format!("doc{i:02}")).collect();
    let matrix = recall.similarity_matrix(&ids).await.unwrap();
    assert_eq!(matrix.len(), 6, "4 choose 2 pairs");
    for (a, b, s) in &matrix {
        let ia: u64 = a[3..].parse().unwrap();
        let ib: u64 = b[3..].parse().unwrap();
        let truth = vec_of(ia).cosine(&vec_of(ib));
        assert!((s - truth).abs() < 1e-6, "{a}~{b}: {s} vs {truth}");
    }

    // Unknown ids are skipped, not errors.
    let with_ghost = recall
        .similarity_matrix(&["doc00".into(), "ghost".into(), "doc01".into()])
        .await
        .unwrap();
    assert_eq!(with_ghost.len(), 1);
    assert_eq!(with_ghost[0].0, "doc00");
    assert_eq!(with_ghost[0].1, "doc01");
}
