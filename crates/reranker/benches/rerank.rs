//! Rerank latency benchmarks over candidate-set sizes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use reranker::LexicalReranker;
use rro_core::{Candidate, Reranker};

fn bench_rerank(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let rr = LexicalReranker::new();
    let vocab = [
        "postgres", "vector", "search", "tokio", "daemon", "banana", "recall", "index", "upgrade",
        "reranker", "graph", "memory", "agent", "network", "signal", "estate",
    ];

    let mut g = c.benchmark_group("rerank");
    for n in [20usize, 100, 500] {
        let cands: Vec<Candidate> = (0..n)
            .map(|i| {
                let text: Vec<&str> = (0..12)
                    .map(|j| vocab[(i * 7 + j * 3) % vocab.len()])
                    .collect();
                Candidate::new(format!("c{i}"), text.join(" "), 0.0)
            })
            .collect();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("bm25_top10", n), &cands, |b, cs| {
            b.iter(|| {
                rt.block_on(rr.rerank("postgres vector upgrade", cs.clone(), 10))
                    .unwrap()
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_rerank);
criterion_main!(benches);
