//! Feature-latency snapshot: one 50k-doc estate exercising every sprint
//! 11–21 retrieval path, p50/p95 per path, plus push-frame delivery
//! latency for `watch`. Run: `cargo run --release -p rro-engine --example featbench`

use std::sync::Arc;
use std::time::Instant;

use rro_core::{Condition, Embedding, EstateQuery, Filter, Recall, SparseVector, VectorRecord};
use rro_engine::{FlowNode, ReasonReadyObject};

const DOCS: usize = 50_000;
const QUERIES: usize = 200;
const DIM: usize = 64;

fn lcg(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed as f64 / u64::MAX as f64) as f32) * 2.0 - 1.0
}

fn vec_of(seed: u64) -> Embedding {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    Embedding((0..DIM).map(|_| lcg(&mut s)).collect())
}

fn pcts(mut ms: Vec<f64>) -> (f64, f64) {
    ms.sort_by(|a, b| a.total_cmp(b));
    let p = |q: f64| ms[((ms.len() as f64 - 1.0) * q) as usize];
    (p(0.5), p(0.95))
}

async fn bench<F, Fut>(name: &str, n: usize, mut f: F)
where
    F: FnMut(usize) -> Fut,
    Fut: std::future::Future<Output = usize>,
{
    let mut times = Vec::with_capacity(n);
    let mut hits = 0usize;
    for i in 0..n {
        let t = Instant::now();
        hits += f(i).await;
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let (p50, p95) = pcts(times);
    println!(
        "| {name} | {p50:.2} ms | {p95:.2} ms | {:.1} |",
        hits as f64 / n as f64
    );
}

#[tokio::main]
async fn main() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "featbench").unwrap());
    estate.create_payload_index("team").unwrap();
    estate.create_payload_index("loc").unwrap();
    let recall = estate.recall();

    // Seed: every doc gets team + geo; every 10th a sparse vector; every
    // 25th two MaxSim token vectors; docs 0..5000 in collection `hot`.
    let t = Instant::now();
    for chunk in (0..DOCS).collect::<Vec<_>>().chunks(512) {
        let records: Vec<VectorRecord> = chunk
            .iter()
            .map(|&i| {
                let mut r = VectorRecord::new(
                    format!("d{i:05}"),
                    vec_of(i as u64),
                    format!(
                        "feature corpus entry number {i} grp{} with shared vocabulary",
                        i % 500
                    ),
                );
                r.metadata
                    .insert("team".into(), serde_json::json!(format!("team{}", i % 40)));
                let lat = 40.0 + (i % 200) as f64 * 0.01;
                let lon = -74.0 + ((i / 200) % 200) as f64 * 0.01;
                r.metadata
                    .insert("loc".into(), serde_json::json!({"lat": lat, "lon": lon}));
                if i % 10 == 0 {
                    r = r.with_sparse(SparseVector::new([
                        ((i % 977) as u32, 1.0f32),
                        ((i % 499) as u32, 0.5f32),
                    ]));
                }
                if i % 25 == 0 {
                    r = r.with_multi(vec![vec_of(700_000 + i as u64), vec_of(800_000 + i as u64)]);
                }
                if i < 5_000 {
                    r = r.in_collection("hot");
                }
                r
            })
            .collect();
        recall.upsert(records).await.unwrap();
    }
    let ingest_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    recall.quiesce().await.unwrap();
    println!(
        "seeded {DOCS} docs in {ingest_s:.1}s ({:.0} docs/sec), catch-up {:.1}s\n",
        DOCS as f64 / ingest_s,
        t.elapsed().as_secs_f64()
    );
    println!("| path | p50 | p95 | avg hits |");
    println!("|---|---|---|---|");

    // Dense-only (ANN path, no lexical) — the fast reference point.
    bench("dense-only top-10", QUERIES, |i| {
        let r = recall.clone();
        async move {
            r.query(EstateQuery {
                vector: Some(vec_of(900_000 + i as u64)),
                top_k: 10,
                ..EstateQuery::default()
            })
            .await
            .unwrap()
            .len()
        }
    })
    .await;

    // Plain hybrid (the reference point).
    bench("hybrid top-10", QUERIES, |i| {
        let r = recall.clone();
        async move {
            r.query(EstateQuery::hybrid(
                "feature corpus entry",
                vec_of(1_000_000 + i as u64),
                10,
            ))
            .await
            .unwrap()
            .len()
        }
    })
    .await;

    // Selective + common: the group token hits ~100 docs (df=100), the
    // rest of the query is maximally common. Once the top-k floor arms
    // after the selective scan, max-score resolves the common terms by
    // point lookups instead of 50k-row scans.
    bench("hybrid selective+common (pruned)", QUERIES, |i| {
        let r = recall.clone();
        async move {
            r.query(EstateQuery::hybrid(
                format!("grp{} feature entry", (i * 7) % 500),
                vec_of(1_050_000 + i as u64),
                10,
            ))
            .await
            .unwrap()
            .len()
        }
    })
    .await;

    // Indexed equality filter (filter-first).
    bench("indexed filter (eq team)", QUERIES, |i| {
        let r = recall.clone();
        async move {
            r.query(
                EstateQuery::hybrid("feature corpus entry", vec_of(1_100_000 + i as u64), 10)
                    .filtered(Filter::default().must(Condition::eq(
                        "team",
                        serde_json::json!(format!("team{}", i % 40)),
                    ))),
            )
            .await
            .unwrap()
            .len()
        }
    })
    .await;

    // Geo radius 3 km (index-first Z-scan + exact re-check).
    bench("geo radius 3km", QUERIES, |i| {
        let r = recall.clone();
        async move {
            let lat = 40.3 + (i % 50) as f64 * 0.02;
            let lon = -73.7 + (i % 50) as f64 * 0.02;
            r.query(
                EstateQuery::hybrid("feature corpus entry", vec_of(1_200_000 + i as u64), 10)
                    .filtered(
                        Filter::default().must(Condition::geo_radius("loc", lat, lon, 3_000.0)),
                    ),
            )
            .await
            .unwrap()
            .len()
        }
    })
    .await;

    // Sparse-fused three-way.
    bench("sparse-fused hybrid", QUERIES, |i| {
        let r = recall.clone();
        async move {
            r.query(
                EstateQuery::hybrid("feature corpus entry", vec_of(1_300_000 + i as u64), 10)
                    .sparse_vector(SparseVector::new([((i % 977) as u32, 1.0f32)])),
            )
            .await
            .unwrap()
            .len()
        }
    })
    .await;

    // MaxSim rescore (fetch-deep + token-vector reads).
    bench("maxsim rescored", QUERIES, |i| {
        let r = recall.clone();
        async move {
            r.query(
                EstateQuery::hybrid("feature corpus entry", vec_of(1_400_000 + i as u64), 10)
                    .multi_query(vec![vec_of(700_000 + ((i * 25) % DOCS) as u64)]),
            )
            .await
            .unwrap()
            .len()
        }
    })
    .await;

    // Collection-scoped (exact scoring inside 5k members).
    bench("collection-scoped (5k)", QUERIES, |i| {
        let r = recall.clone();
        async move {
            r.query(
                EstateQuery::hybrid("feature corpus entry", vec_of(1_500_000 + i as u64), 10)
                    .in_collection("hot"),
            )
            .await
            .unwrap()
            .len()
        }
    })
    .await;

    // Watch: push-frame delivery latency (write commit → frame received).
    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = FlowNode::new(flow, "featbench").with_estate(estate.clone());
    let (addr, _task) = rro_net::tcp::serve("127.0.0.1:0", Arc::new(node))
        .await
        .unwrap();
    let client = rro_client::Client::new(addr.to_string());
    let health = client.health().await.unwrap();
    let since = health["estate"]["feed_seq"].as_u64().unwrap();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    tokio::spawn(async move {
        let _ = client
            .watch(since, move |_| {
                let _ = tx.send(());
                true
            })
            .await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(200)).await; // subscribe settles
    let mut lat = Vec::new();
    for i in 0..50 {
        let t = Instant::now();
        recall
            .upsert(vec![VectorRecord::new(
                format!("w{i}"),
                vec_of(2_000_000 + i as u64),
                "watch latency probe",
            )])
            .await
            .unwrap();
        rx.recv().await.unwrap();
        lat.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let (p50, p95) = pcts(lat);
    println!("| watch frame delivery | {p50:.2} ms | {p95:.2} ms | 1.0 |");
}
