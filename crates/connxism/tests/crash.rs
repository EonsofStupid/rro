//! P5 ops gates: snapshots open as working estates, and the estate survives
//! a hard process death (kill-9 class: `abort()` — no destructors, no
//! graceful flush) with counts, search, and the changefeed intact.

use connxism::Estate;
use rro_core::{Embedding, Recall, VectorRecord};

fn rec(id: &str, seed: u64, text: &str) -> VectorRecord {
    // Deterministic non-trivial vector.
    let mut x = seed.wrapping_add(0x9E37);
    let v: Vec<f32> = (0..16)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x as f64 / u64::MAX as f64) as f32) * 2.0 - 1.0
        })
        .collect();
    VectorRecord::new(id, Embedding(v), text)
}

const CRASH_DOCS: usize = 500;

/// Child role: ingest durably, then die with no destructors. Only active
/// when spawned by the parent test (CRASH_DIR set) — a plain test run
/// returns immediately.
#[tokio::test(flavor = "multi_thread")]
async fn crash_writer_role() {
    let Ok(dir) = std::env::var("CRASH_DIR") else {
        return;
    };
    let estate = Estate::open(&dir, "crash").unwrap();
    let recall = estate.recall();
    let records: Vec<VectorRecord> = (0..CRASH_DOCS)
        .map(|i| {
            rec(
                &format!("crash{i}"),
                i as u64,
                &format!("crash recovery payload {i} durable"),
            )
        })
        .collect();
    recall.upsert(records).await.unwrap();
    // Hard death. No Drop, no WAL-friendly shutdown, applier thread killed
    // mid-flight — exactly what kill -9 does to a process.
    std::process::abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn estate_survives_hard_process_death() {
    if std::env::var("CRASH_DIR").is_ok() {
        return; // never recurse inside the child
    }
    let dir = tempfile::tempdir().unwrap();

    for round in 1..=3u32 {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "crash_writer_role", "--test-threads=1"])
            .env("CRASH_DIR", dir.path())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(!status.success(), "round {round}: the child must die hard");

        // Recovery: reopen (Fjall journal replay + ANN rebuild) and verify.
        let estate = Estate::open(dir.path(), "crash").unwrap();
        let recall = estate.recall();
        assert_eq!(
            recall.len().await.unwrap(),
            CRASH_DOCS,
            "round {round}: every durably-acked doc survives the crash"
        );
        // Search works end-to-end after recovery (payloads + vectors + BM25).
        let hits = recall
            .lexical_search("crash recovery durable", 5)
            .await
            .unwrap();
        assert!(!hits.is_empty(), "round {round}: lexical index consistent");
        let dense = recall.search(&rec("q", 7, "q").embedding, 5).await.unwrap();
        assert_eq!(dense.len(), 5, "round {round}: dense path consistent");
        // Changefeed replayed intact and ordered.
        let changes = estate.changes(0, 10_000).unwrap();
        assert!(changes.len() >= CRASH_DOCS);
        assert!(changes.windows(2).all(|w| w[0].seq < w[1].seq));
        drop(recall);
        drop(estate); // clean close before the next crashing round
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn snapshot_opens_as_a_working_estate() {
    if std::env::var("CRASH_DIR").is_ok() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let snap_dir = tempfile::tempdir().unwrap();
    let snap_path = snap_dir.path().join("snap");

    {
        let estate = Estate::open(dir.path(), "live").unwrap();
        let recall = estate.recall();
        recall
            .upsert(vec![
                rec("s1", 1, "snapshot subject alpha"),
                rec("s2", 2, "snapshot subject beta"),
            ])
            .await
            .unwrap();
        recall.quiesce().await.unwrap();
        estate.relate("proj", "contains", "s1").unwrap();
        estate.snapshot_to(&snap_path).unwrap();

        // The live estate keeps moving after the snapshot…
        recall
            .upsert(vec![rec("s3", 3, "post snapshot gamma")])
            .await
            .unwrap();
    }

    // …while the snapshot opens as a complete, point-in-time estate.
    let snap = Estate::open(&snap_path, "snap").unwrap();
    let recall = snap.recall();
    assert_eq!(recall.len().await.unwrap(), 2, "point-in-time: no s3");
    let hits = recall.lexical_search("snapshot subject", 5).await.unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(
        snap.relations_out("proj", Some("contains")).unwrap().len(),
        1,
        "relations captured"
    );
}
