# Reason Ready — Testing & Quality

Rigor is a first-class feature. The engine is meant to be *proven*, not
asserted. This document is the testing contract.

## The pyramid

| Layer | Tool | What it guards |
|---|---|---|
| Unit | `cargo test` / `nextest` | component behaviour in isolation |
| Property | `proptest` | invariants over generated inputs |
| Integration | workspace tests | crates composed through `rro-core` traits |
| End-to-end | `rro-engine` example + tests | the full pass, incl. the daemon |
| Fuzz | `cargo-fuzz` | the a2a wire parser + every `deserialize` |
| Concurrency | `loom` (targeted) | shared-state locking under interleavings |
| Snapshot | `insta` | connectome output + API payloads |
| Bench | `criterion` | latency/throughput regressions |
| Load / soak | custom harness | the ingestion daemon under sustained load |

## Invariants under property test

- **Embedder** — deterministic (`embed(s) == embed(s)`); output dim is stable;
  vectors are unit-norm or zero.
- **Recall** — `upsert` then `search` returns the record; results are sorted by
  descending score; `len` equals the number of distinct ids; dimension mismatch
  always errors.
- **Reranker** — output is a score-sorted sub-multiset of the input truncated to
  `top_k`; no candidate is fabricated.
- **Cosine** — always within `[-1, 1]`.

## Quality: the bake-off harness

Retrieval quality is measured, not claimed. The harness runs a fixed eval set
(queries + relevance judgments) through the flow for each backend/config and
reports:

- **Quality:** recall@k, MRR, nDCG@k.
- **Performance:** p50/p95 latency, throughput (qps), peak RSS.

Results are emitted as a report + trend series so every change is measured and
the bake-off winner is data. Backends compared: candle (in-proc), llama.cpp,
vLLM, and (experimental) candle-vllm.

### The measurement harness

```sh
cargo run --release --bin rro-bench -- --docs 50000 --queries 500 --store estate
```

Measures the full ingestion machine (embed → index → persist) and hybrid query
latency (p50/p95/p99) against either store (`mem` | `estate`). External
baselines run *outside* this tree on the same corpus/queries and are compared
on the emitted numbers; measured results live in [BENCHMARKS](BENCHMARKS.md).

## The ingestion daemon (scale)

The tokio, signal-driven ingestion path is tested for:

- **State machine** — `Idle → Ingesting → Indexed` transitions are correct and
  observable; counts/errors/timestamps are accurate.
- **Backpressure** — bounded channels + a concurrency semaphore hold memory flat
  under a flood of upserts.
- **Graceful shutdown** — SIGTERM/Ctrl-C mid-ingest drains in-flight work and
  exits cleanly; no partial-index corruption.
- **Soak** — sustained concurrent upserts + queries for N minutes with bounded
  latency and no leak (watched via `tokio-console`).

## Running

```sh
cargo test --workspace            # unit + property + integration
cargo test --workspace --doc      # doctests
cargo bench --workspace           # criterion benches
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
# CI additionally runs: nextest, cargo-llvm-cov (coverage), cargo-deny (supply chain)
```

## CI gates (must pass to merge)

`fmt --check` · `clippy -D warnings` · `nextest` (stable + MSRV) · coverage
threshold · `cargo-deny` (licenses + advisories + bans) · `cargo doc` (no broken
intra-doc links).
