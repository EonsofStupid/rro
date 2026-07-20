//! Sprint 15 gates: datetime range filters answered index-first (exact
//! id-set from ordered `PIDX_DT` scans, equal to brute-force truth), UUID
//! typed keys serving equality, and REBUILD INDEX re-typing legacy rows.

use connxism::{Estate, EstateQuery};
use rro_core::{Condition, Embedding, Filter, Recall, VectorRecord};

fn rec(id: &str, created: &str, owner: &str) -> VectorRecord {
    let mut r = VectorRecord::new(
        id,
        Embedding(vec![0.1, 0.5, 0.3, 0.2]),
        format!("dated entry {id}"),
    );
    r.metadata
        .insert("created".into(), serde_json::json!(created));
    r.metadata.insert("owner".into(), serde_json::json!(owner));
    r
}

/// 60 docs at hourly timestamps across 2026-07-13..15, mixed offsets.
async fn seed(estate: &Estate) -> connxism::ConnXRecall {
    estate.create_payload_index("created").unwrap();
    estate.create_payload_index("owner").unwrap();
    let recall = estate.recall();
    let mut records = Vec::new();
    for i in 0..60u32 {
        let (day, hour) = (13 + i / 24, i % 24);
        // Every third doc uses a +02:00 offset spelling of the same clock
        // grid — instant comparison must not care about the spelling.
        let created = if i % 3 == 0 {
            format!("2026-07-{day:02}T{:02}:00:00+02:00", (hour + 2) % 24)
        } else {
            format!("2026-07-{day:02}T{hour:02}:00:00Z")
        };
        // The +02:00 spelling wraps past midnight for hours 22/23; skip the
        // wrap ambiguity by keeping those on Z.
        let created = if i % 3 == 0 && hour >= 22 {
            format!("2026-07-{day:02}T{hour:02}:00:00Z")
        } else {
            created
        };
        records.push(rec(
            &format!("doc{i:02}"),
            &created,
            &format!("owner{}", i % 4),
        ));
    }
    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();
    recall
}

#[tokio::test(flavor = "multi_thread")]
async fn date_range_is_index_first_and_exact() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "dt").unwrap();
    let recall = seed(&estate).await;

    let filter = Filter::default().must(Condition::date_range(
        "created",
        Some("2026-07-13T20:00:00Z"),
        Some("2026-07-14T04:00:00Z"),
    ));

    // Index-first: the filter resolves to an exact id-set from PIDX_DT.
    let ids = estate
        .ids_where(&filter)
        .unwrap()
        .expect("fully indexed filter must resolve from the index");
    // Brute-force truth: hours 20..=28 inclusive → 9 docs.
    let truth: Vec<String> = (20..=28).map(|i| format!("doc{i:02}")).collect();
    assert_eq!(ids, truth, "ordered id-set equals brute force");

    // The query plane returns exactly the same set (scored inside it).
    let hits = recall
        .query(
            EstateQuery::hybrid("dated entry", Embedding(vec![0.1, 0.5, 0.3, 0.2]), 20)
                .filtered(filter.clone()),
        )
        .await
        .unwrap();
    let mut got: Vec<&str> = hits.iter().map(|c| c.id.as_str()).collect();
    got.sort();
    assert_eq!(got, truth.iter().map(String::as_str).collect::<Vec<_>>());

    // Offset spellings compare by instant: a bound written in +05:00 picks
    // the same docs as its Z equivalent.
    let alt = Filter::default().must(Condition::date_range(
        "created",
        Some("2026-07-14T01:00:00+05:00"), // == 2026-07-13T20:00:00Z
        Some("2026-07-14T04:00:00Z"),
    ));
    assert_eq!(estate.ids_where(&alt).unwrap().unwrap(), truth);

    // Half-open range + post-filter equivalence (belt and braces): the
    // DSL's own matches() agrees with the index on every doc.
    let open = Filter::default().must(Condition::date_range(
        "created",
        Some("2026-07-15T06:00:00Z"),
        None::<String>,
    ));
    let idx_ids = estate.ids_where(&open).unwrap().unwrap();
    let mut post = Vec::new();
    for i in 0..60u32 {
        let id = format!("doc{i:02}");
        let doc = recall.doc(&id).await.unwrap().unwrap();
        if open.matches(&doc.metadata) {
            post.push(id);
        }
    }
    assert_eq!(idx_ids, post, "index strategy equals post-filter truth");
    assert!(!idx_ids.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn uuid_typed_keys_serve_equality() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "uu").unwrap();
    estate.create_payload_index("ref").unwrap();
    let recall = estate.recall();

    let uuid = "550e8400-e29b-41d4-a716-446655440000";
    let mut a = rec("a", "2026-07-16T00:00:00Z", "o");
    a.metadata.insert("ref".into(), serde_json::json!(uuid));
    let mut b = rec("b", "2026-07-16T00:00:00Z", "o");
    b.metadata.insert(
        "ref".into(),
        serde_json::json!("123e4567-e89b-12d3-a456-426614174000"),
    );
    recall.upsert(vec![a, b]).await.unwrap();

    let filter = Filter::default().must(Condition::eq("ref", serde_json::json!(uuid)));
    assert_eq!(
        estate.ids_where(&filter).unwrap().unwrap(),
        vec!["a".to_string()],
        "uuid equality resolves from its typed index rows"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn rebuild_payload_index_retypes_and_survives() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "rb").unwrap();
    let recall = seed(&estate).await;

    // Rebuild is idempotent on a healthy index: same answers after.
    let filter = Filter::default().must(Condition::date_range(
        "created",
        Some("2026-07-14T00:00:00Z"),
        Some("2026-07-14T02:00:00Z"),
    ));
    let before = estate.ids_where(&filter).unwrap().unwrap();
    estate.rebuild_payload_index("created").unwrap();
    assert_eq!(estate.ids_where(&filter).unwrap().unwrap(), before);
    assert_eq!(before.len(), 3);

    // Rebuilding an unindexed field is an error, not a silent no-op.
    assert!(estate.rebuild_payload_index("nope").is_err());

    // And the estate still answers queries end-to-end.
    let hits = recall
        .query(
            EstateQuery::hybrid("dated entry", Embedding(vec![0.1, 0.5, 0.3, 0.2]), 5)
                .filtered(filter),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 3);
}
