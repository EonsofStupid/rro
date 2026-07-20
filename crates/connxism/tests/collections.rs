//! Sprint 16 gates: named collections in one estate — leak-proof scoping,
//! membership retraction on move/remove, drop removes exactly its members,
//! and the collection field rides the query contract's wire shape.

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

fn rec(id: &str, seed: u64, text: &str, coll: Option<&str>) -> VectorRecord {
    let mut r = VectorRecord::new(id, vec_of(seed), text);
    if let Some(c) = coll {
        r = r.in_collection(c);
    }
    r
}

/// One estate: 10 docs in `alpha`, 10 in `beta`, 5 uncollected floaters.
/// All share vocabulary so lexical ranking alone would happily leak.
async fn seed(estate: &Estate) -> connxism::ConnXRecall {
    let recall = estate.recall();
    let mut records = Vec::new();
    for i in 0..10u64 {
        records.push(rec(
            &format!("a{i}"),
            i,
            &format!("shared corpus entry number {i}"),
            Some("alpha"),
        ));
        records.push(rec(
            &format!("b{i}"),
            100 + i,
            &format!("shared corpus entry number {i}"),
            Some("beta"),
        ));
    }
    for i in 0..5u64 {
        records.push(rec(
            &format!("f{i}"),
            200 + i,
            &format!("shared corpus entry number {i}"),
            None,
        ));
    }
    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();
    recall
}

#[tokio::test(flavor = "multi_thread")]
async fn collections_never_leak() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "c").unwrap();
    let recall = seed(&estate).await;

    // Registry lists both with exact counts.
    let mut colls = estate.collections().unwrap();
    colls.sort();
    assert_eq!(
        colls,
        vec![("alpha".to_string(), 10), ("beta".to_string(), 10)]
    );

    // Scoped query returns ONLY alpha members, at full depth.
    let hits = recall
        .query(EstateQuery::hybrid("shared corpus entry", vec_of(3), 25).in_collection("alpha"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 10, "exactly the collection, nothing else");
    assert!(hits.iter().all(|c| c.id.as_str().starts_with('a')));

    // Beta likewise; floaters appear in neither.
    let hits = recall
        .query(EstateQuery::hybrid("shared corpus entry", vec_of(3), 25).in_collection("beta"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 10);
    assert!(hits.iter().all(|c| c.id.as_str().starts_with('b')));

    // Unscoped queries still see everything (25 docs).
    let hits = recall
        .query(EstateQuery::hybrid("shared corpus entry", vec_of(3), 30))
        .await
        .unwrap();
    assert_eq!(hits.len(), 25);

    // Collection + explicit scope = intersection.
    let hits = recall
        .query(
            EstateQuery::hybrid("shared corpus entry", vec_of(3), 25)
                .in_collection("alpha")
                .within(vec!["a3".into(), "b3".into(), "f0".into()]),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.as_str(), "a3");

    // Unknown collection → empty, not an error.
    let hits = recall
        .query(EstateQuery::hybrid("shared", vec_of(3), 5).in_collection("nope"))
        .await
        .unwrap();
    assert!(hits.is_empty());

    // Wire shape: rides serde; old payloads parse with None.
    let q = EstateQuery::hybrid("t", vec_of(1), 5).in_collection("alpha");
    let back: EstateQuery = serde_json::from_str(&serde_json::to_string(&q).unwrap()).unwrap();
    assert_eq!(back.collection.as_deref(), Some("alpha"));
    let old: EstateQuery = serde_json::from_str(r#"{"text":"x","vector":null,"top_k":3}"#).unwrap();
    assert!(old.collection.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn membership_moves_and_drop_retracts_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "d").unwrap();
    let recall = seed(&estate).await;

    // Move a3 from alpha to beta: alpha loses it, beta gains it.
    recall
        .upsert(vec![rec(
            "a3",
            3,
            "shared corpus entry number 3",
            Some("beta"),
        )])
        .await
        .unwrap();
    assert!(!estate
        .collection_members("alpha")
        .unwrap()
        .contains(&"a3".to_string()));
    assert!(estate
        .collection_members("beta")
        .unwrap()
        .contains(&"a3".to_string()));

    // Remove b5 entirely: beta shrinks.
    recall.remove(&"b5".into()).await.unwrap();
    assert!(!estate
        .collection_members("beta")
        .unwrap()
        .contains(&"b5".to_string()));

    // Drop beta (now a3 + 9 b-docs = 10 members after move & remove).
    let feed_before = estate.changes(0, 10_000).unwrap().len();
    let total_before = recall.len().await.unwrap();
    let dropped = estate.drop_collection("beta").unwrap();
    assert_eq!(dropped, 10);
    assert_eq!(recall.len().await.unwrap(), total_before - 10);

    // Exactly its members died: alpha (9 left) + floaters intact.
    assert_eq!(estate.collection_members("alpha").unwrap().len(), 9);
    assert!(recall.doc("f0").await.unwrap().is_some());
    assert!(recall.doc("a0").await.unwrap().is_some());
    assert!(recall.doc("b0").await.unwrap().is_none());
    assert!(
        recall.doc("a3").await.unwrap().is_none(),
        "moved doc died with beta"
    );

    // Registry deregistered beta; changefeed recorded the removes.
    let names: Vec<String> = estate
        .collections()
        .unwrap()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(names, vec!["alpha"]);
    let feed_after = estate.changes(0, 10_000).unwrap();
    assert_eq!(feed_after.len(), feed_before + 10);

    // Dropped docs are gone from search too (no ghost hits).
    let hits = recall
        .query(EstateQuery::hybrid("shared corpus entry", vec_of(3), 30))
        .await
        .unwrap();
    assert!(hits.iter().all(|c| !c.id.as_str().starts_with('b')));
    assert_eq!(hits.len(), 14); // 9 alpha + 5 floaters
}
