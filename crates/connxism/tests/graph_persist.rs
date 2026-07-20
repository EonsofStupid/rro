//! Phase 6a/6b — the ANN graph is persisted, so a clean reopen *loads* it instead
//! of rebuilding it from scratch, and its vectors are paged from disk so RAM does
//! not scale with the dataset.
//!
//! The graph is a cache derived from the durable `vecs` column family. A clean
//! shutdown captures its structure to `CF_GRAPH` (tagged with the changefeed seq
//! it was taken at) and its vectors to a `graph.vectors` sidecar; the next open
//! loads that structure iff the seq still matches the live `feed_seq`, and pages
//! the vectors from the sidecar. This turns an O(N log N) rebuild-on-open into an
//! O(read) load and keeps vector RAM bounded — "restart in read-time, RSS tracks
//! the working set."
//!
//! What must hold, and is gated here:
//!  1. **Loads, and is identical.** After a clean reopen the graph is loaded
//!     (`graph_was_loaded()`), and it returns the *same* search results as before
//!     the restart — a persisted graph that answered differently would be a
//!     silent corruption, worse than rebuilding.
//!  2. **RAM is bounded.** Reopened, the graph's vector heap stays within the
//!     cache budget, not the dataset size (6b).
//!  3. **Falls back safely.** If the persisted blob is stale (a crash left it
//!     behind newer writes), the open rebuilds from the durable vectors and every
//!     document — including the ones written after the stale capture — is present.
//!     The cache is never trusted over the source of truth.

use connxism::{Estate, EstateConfig, EstateQuery};
use rro_core::{Embedding, Recall, VectorRecord};

const DIM: usize = 32;

/// Deterministic pseudo-random unit vector, stable across process restarts (no
/// RNG state, no time) so "identical before and after" is a meaningful claim.
fn vec_for(seed: u64) -> Embedding {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let v: Vec<f32> = (0..DIM)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        })
        .collect();
    Embedding(v).normalized()
}

fn corpus(n: usize) -> Vec<VectorRecord> {
    (0..n)
        .map(|i| VectorRecord::new(format!("d{i}"), vec_for(i as u64), format!("document {i}")))
        .collect()
}

/// The ordered `(id, score)` list a dense query returns — the comparison key for
/// "identical results". Scores are quantized to avoid f32 jitter masking a real
/// match while still catching a genuinely different ranking.
async fn ranked(recall: &connxism::ConnXRecall, q: &Embedding, k: usize) -> Vec<(String, i64)> {
    recall
        .query(EstateQuery::hybrid("document", q.clone(), k))
        .await
        .unwrap()
        .into_iter()
        .map(|c| (c.id.as_str().to_string(), (c.score * 10_000.0) as i64))
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn reopen_loads_persisted_graph_and_results_are_identical() {
    let dir = tempfile::tempdir().unwrap();
    // Seed well past the ANN threshold so the dense path is the graph, not a
    // brute-force scan — the graph is the thing being persisted.
    let n = 2000;
    let query = vec_for(9_999_001);

    let before = {
        let estate = Estate::open(dir.path(), "gp").unwrap();
        let recall = estate.recall();
        recall.upsert(corpus(n)).await.unwrap();
        recall.quiesce().await.unwrap();
        let before = ranked(&recall, &query, 10).await;
        assert_eq!(before.len(), 10, "seed query must return a full page");
        // recall + estate drop here → clean shutdown persists the graph.
        before
    };

    let estate = Estate::open(dir.path(), "gp").unwrap();
    assert!(
        estate.graph_was_loaded(),
        "a clean reopen must LOAD the persisted graph, not rebuild it"
    );
    let recall = estate.recall();
    assert_eq!(recall.len().await.unwrap() as usize, n);

    let after = ranked(&recall, &query, 10).await;
    assert_eq!(
        before, after,
        "loaded graph must return byte-identical results to the graph it was saved from"
    );
}

/// A wider vector, so the dataset dwarfs a deliberately small page-cache budget
/// and "RAM is bounded" is a real claim rather than an artifact of a tiny corpus.
fn wide_vec(seed: u64, dim: usize) -> Embedding {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let v: Vec<f32> = (0..dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        })
        .collect();
    Embedding(v).normalized()
}

/// 6b: after a clean reopen the estate **pages** its vectors from the sidecar —
/// the graph loads, results are identical, the vector heap stays bounded by the
/// cache budget (not the dataset), and the sidecar file exists on disk.
#[tokio::test(flavor = "multi_thread")]
async fn reopen_pages_vectors_ram_bounded_and_identical() {
    let dir = tempfile::tempdir().unwrap();
    let dim = 128;
    let n = 3000;
    let dataset_bytes = n * dim * 4; // ≈ 1.5 MiB
    let cache_bytes = 256 * 1024; // ≈ 6× smaller than the dataset
    let cfg = || EstateConfig {
        graph_cache_bytes: cache_bytes,
        ..EstateConfig::default()
    };
    let query = wide_vec(9_100_001, dim);

    let before = {
        let estate = Estate::open_with(dir.path(), "pg6b", cfg()).unwrap();
        let recall = estate.recall();
        let records: Vec<_> = (0..n)
            .map(|i| {
                VectorRecord::new(
                    format!("d{i}"),
                    wide_vec(i as u64, dim),
                    format!("document {i}"),
                )
            })
            .collect();
        recall.upsert(records).await.unwrap();
        recall.quiesce().await.unwrap();
        let before = ranked(&recall, &query, 10).await;
        assert_eq!(before.len(), 10);
        before
    };

    // The clean shutdown wrote the vector sidecar.
    assert!(
        dir.path().join("graph.vectors").exists(),
        "persist must write the graph.vectors sidecar"
    );

    let estate = Estate::open_with(dir.path(), "pg6b", cfg()).unwrap();
    assert!(
        estate.graph_was_loaded(),
        "a clean reopen must LOAD the paged graph, not rebuild it"
    );
    let recall = estate.recall();

    let after = ranked(&recall, &query, 10).await;
    assert_eq!(before, after, "paged reopen must return identical results");

    // The vectors are paged: the graph's resident vector heap is bounded by the
    // cache budget and nowhere near the whole dataset.
    let resident = estate.graph_heap_vector_bytes();
    assert!(
        resident <= cache_bytes + 128 * 1024,
        "resident vector heap {resident} must stay within the cache budget {cache_bytes}"
    );
    assert!(
        resident < dataset_bytes / 2,
        "resident vector heap {resident} must be well under the dataset {dataset_bytes}"
    );
}

/// The measured payoff: loading the persisted graph is dramatically faster than
/// rebuilding it by re-inserting every vector. `#[ignore]` — it seeds 50k vectors
/// (~seconds) to make the gap unmistakable; run with
/// `cargo test -p connxism --release --test graph_persist -- --ignored`.
///
/// This is the 6a half of the scale story (startup in read-time); the 6b half
/// (vectors page from disk, RAM tracks the working set) is gated by
/// `scale_gate_paged_reopen_ram_bounded` below.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "50k-vector timing gate; run under --release --ignored"]
async fn load_is_faster_than_rebuild() {
    use std::time::Instant;

    let dir = tempfile::tempdir().unwrap();
    let n = 50_000;

    // Seed once, capturing the graph on clean drop.
    let rebuild_ms = {
        let estate = Estate::open(dir.path(), "gp").unwrap();
        let recall = estate.recall();
        recall.upsert(corpus(n)).await.unwrap();
        recall.quiesce().await.unwrap();
        // First reopen after a *rebuild-only* baseline: measure a from-scratch
        // rebuild by disabling persistence for this shutdown.
        estate.set_persist_graph_on_drop(false);
        drop(recall);
        drop(estate);

        let t = Instant::now();
        let estate = Estate::open(dir.path(), "gp").unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        assert!(!estate.graph_was_loaded(), "baseline must have rebuilt");
        // Now let this open persist the graph on drop for the load measurement.
        ms
    };

    let load_ms = {
        let t = Instant::now();
        let estate = Estate::open(dir.path(), "gp").unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        assert!(estate.graph_was_loaded(), "second reopen must have loaded");
        ms
    };

    println!(
        "6a — {n} vectors: rebuild {rebuild_ms:.0} ms → load {load_ms:.0} ms ({:.1}× faster)",
        rebuild_ms / load_ms
    );
    assert!(
        load_ms * 3.0 < rebuild_ms,
        "loading the graph must be at least 3× faster than rebuilding it \
         (load {load_ms:.0} ms vs rebuild {rebuild_ms:.0} ms)"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn stale_persisted_graph_is_rejected_and_rebuild_sees_all_writes() {
    let dir = tempfile::tempdir().unwrap();

    {
        let estate = Estate::open(dir.path(), "gp").unwrap();
        let recall = estate.recall();

        // Batch A, then capture the graph at seq_A.
        recall.upsert(corpus(1000)).await.unwrap();
        recall.quiesce().await.unwrap();
        estate.persist_graph().unwrap();

        // Batch B lands *after* the capture: the on-disk blob is now stale.
        let batch_b: Vec<_> = (1000..1500)
            .map(|i| VectorRecord::new(format!("d{i}"), vec_for(i as u64), format!("late {i}")))
            .collect();
        recall.upsert(batch_b).await.unwrap();
        recall.quiesce().await.unwrap();

        // Simulate a crash: do NOT re-persist on drop, so the stale blob survives
        // exactly as an unclean shutdown would leave it.
        estate.set_persist_graph_on_drop(false);
    } // dropped with the stale blob intact

    let estate = Estate::open(dir.path(), "gp").unwrap();
    assert!(
        !estate.graph_was_loaded(),
        "a blob tagged with an older feed_seq must be rejected → rebuild"
    );
    let recall = estate.recall();

    // The rebuild is from the durable vectors, so every document is present —
    // including batch B, which was never in the persisted graph.
    assert_eq!(recall.len().await.unwrap(), 1500);
    let late = vec_for(1_234);
    let hits = ranked(&recall, &late, 5).await;
    assert!(!hits.is_empty(), "rebuilt graph must be queryable");
    let hits_b = ranked(&recall, &vec_for(1_400), 5).await;
    assert!(
        hits_b.iter().any(|(id, _)| id == "d1400"),
        "a batch-B document must be findable after rebuild"
    );
}

/// Resident set size of this process, bytes (Linux `/proc/self/statm`). Noisy —
/// it includes the Fjall store, the allocator's retained pages, and the test's own
/// reference data — so it is *logged*, not asserted. The asserted RAM proof is
/// `graph_heap_vector_bytes`, which is exactly the graph's vector heap.
fn process_rss_bytes() -> usize {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let resident_pages: usize = statm
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    resident_pages * 4096
}

/// 6b at scale: a dataset far larger than the vector cache reopens **paged** —
/// the graph loads (not rebuilds), restart is fast, the graph's vector heap stays
/// bounded by the cache (not the dataset), and recall is preserved. `#[ignore]` —
/// it seeds 200k vectors (~minutes in release); run via `scripts/gates.sh` or
/// `cargo test -p connxism --release --test graph_persist -- --ignored`.
///
/// The RAM-bounded property is **scale-invariant** — it holds whenever the
/// dataset exceeds the cache — so 200k with an 8 MiB cache (≈9× smaller than the
/// dataset) proves what 10M would, without an hours-long HNSW build.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "200k-vector scale gate; run under --release --ignored"]
async fn scale_gate_paged_reopen_ram_bounded() {
    use std::time::Instant;

    let dir = tempfile::tempdir().unwrap();
    let dim = 96;
    let n = 200_000u64;
    let dataset_bytes = n as usize * dim * 4; // ≈ 73 MiB
    let cache_bytes = 8 * 1024 * 1024; // 8 MiB — ≈9× smaller than the dataset
    let cfg = || EstateConfig {
        graph_cache_bytes: cache_bytes,
        ..EstateConfig::default()
    };

    // The sample queries. Dense-only: a lexical token that matches NO document,
    // so RRF fusion reduces to the pure dense ranking (a common word would let the
    // lexical arm, which matches every doc, dominate — the "optimal lexical weight
    // ≈ 0" finding). We compare the paged reopen to the in-RAM build, an EXACT
    // identity check — the meaningful recall proof at scale, where "top-10 vs
    // brute force" is noise because uniform random vectors have no cluster
    // structure and the exact top-10 is arbitrary within a huge tie band.
    let queries: Vec<Embedding> = (0..50).map(|qi| wide_vec(9_000_000 + qi, dim)).collect();
    async fn top10(recall: &connxism::ConnXRecall, q: &Embedding) -> Vec<String> {
        recall
            .query(EstateQuery::hybrid(
                "zzq_no_such_lexical_token",
                q.clone(),
                10,
            ))
            .await
            .unwrap()
            .into_iter()
            .map(|c| c.id.as_str().to_string())
            .collect()
    }

    // Build + persist, capturing the in-RAM answers first.
    let before = {
        let estate = Estate::open_with(dir.path(), "scale", cfg()).unwrap();
        let recall = estate.recall();
        for start in (0..n).step_by(10_000) {
            let end = (start + 10_000).min(n);
            let recs: Vec<_> = (start..end)
                .map(|i| VectorRecord::new(format!("d{i}"), wide_vec(i, dim), format!("doc {i}")))
                .collect();
            recall.upsert(recs).await.unwrap();
        }
        recall.quiesce().await.unwrap();
        let mut before = Vec::with_capacity(queries.len());
        for q in &queries {
            before.push(top10(&recall, q).await);
        }
        before
    }; // drop streams the vector sidecar

    assert!(
        dir.path().join("graph.vectors").exists(),
        "persist must write the sidecar"
    );

    // Reopen, paged, and time it.
    let t = Instant::now();
    let estate = Estate::open_with(dir.path(), "scale", cfg()).unwrap();
    let reopen_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert!(estate.graph_was_loaded(), "must load paged, not rebuild");
    let recall = estate.recall();
    assert_eq!(recall.len().await.unwrap() as u64, n);

    // Paged answers must be identical to the in-RAM answers — the exact proof that
    // paging returns the right vectors at scale.
    let mut identical = 0usize;
    for (q, want) in queries.iter().zip(&before) {
        if &top10(&recall, q).await == want {
            identical += 1;
        }
    }
    let resident = estate.graph_heap_vector_bytes();
    let rss = process_rss_bytes();

    println!(
        "6b SCALE — n={n} dim={dim} dataset={}MiB | reopen={reopen_ms:.0}ms \
         graph_vector_heap={}KiB (cache budget {}KiB) paged==in-RAM {identical}/{} \
         process_rss={}MiB",
        dataset_bytes >> 20,
        resident >> 10,
        cache_bytes >> 10,
        queries.len(),
        rss >> 20,
    );

    assert!(
        resident <= cache_bytes + (1 << 20),
        "graph vector heap {resident} must stay within the cache budget {cache_bytes}"
    );
    assert!(
        resident * 4 < dataset_bytes,
        "graph vector heap {resident} must be far below the dataset {dataset_bytes}"
    );
    assert_eq!(
        identical,
        queries.len(),
        "every paged query must match the in-RAM answer exactly"
    );
}
