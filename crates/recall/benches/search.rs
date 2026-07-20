//! Recall search-latency benchmarks over growing store sizes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use recall::FlatRecall;
use rro_core::{Embedding, Recall, VectorRecord};

fn pseudo_vec(seed: u64, dim: usize) -> Vec<f32> {
    // Cheap deterministic pseudo-random vector (xorshift), no rand dep.
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x as f64 / u64::MAX as f64) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bench_search(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let dim = 384;

    let mut g = c.benchmark_group("recall_search");
    for n in [1_000usize, 10_000, 50_000] {
        let store = FlatRecall::new();
        let records: Vec<VectorRecord> = (0..n)
            .map(|i| {
                VectorRecord::new(
                    format!("id{i}"),
                    Embedding(pseudo_vec(i as u64, dim)),
                    format!("doc {i}"),
                )
            })
            .collect();
        rt.block_on(store.upsert(records)).unwrap();
        let query = Embedding(pseudo_vec(u64::MAX / 2, dim));

        g.throughput(Throughput::Elements(1));
        g.bench_with_input(BenchmarkId::new("flat_top10", n), &store, |b, s| {
            b.iter(|| rt.block_on(s.search(&query, 10)).unwrap());
        });
    }
    g.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
