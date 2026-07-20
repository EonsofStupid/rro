//! Embedding throughput benchmarks.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use embedder::DeterministicEmbedder;
use rro_core::Embedder;

fn bench_embed(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let e = DeterministicEmbedder::new();
    let text = "the quick brown fox jumps over the lazy dog and files a bug report about it";

    let mut g = c.benchmark_group("embed");
    for batch in [1usize, 32, 256] {
        let texts: Vec<String> = (0..batch).map(|i| format!("{text} #{i}")).collect();
        g.throughput(Throughput::Elements(batch as u64));
        g.bench_with_input(BenchmarkId::new("deterministic", batch), &texts, |b, t| {
            b.iter(|| rt.block_on(e.embed(t)).unwrap());
        });
    }
    g.finish();
}

criterion_group!(benches, bench_embed);
criterion_main!(benches);
