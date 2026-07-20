# ADR-0001 — Inference backends: Rust product, pluggable model serving

- **Status:** Accepted
- **Date:** 2026-07
- **Context tags:** architecture, inference, performance, deployment

## Context

Reason Ready needs model inference in three places: embedding (perception),
reranking (relevance), and — soon — generation. Candidate execution paths are
candle (Rust, in-process), llama.cpp (C++, GGUF), and vLLM (Python, GPU
serving). A recurring temptation is to port vLLM to Rust to keep the stack
single-language.

## Decision

**Rust is the engine, kernel, and product. Model inference is a pluggable
boundary behind a trait, served by the best tool per model. We do not port
vLLM to Rust.**

- **Rust (RRO):** memory, retrieval, state, routing, a2a, the deployable
  single binary — Clyffy's runtime spine.
- **candle (in-process, Rust):** small encoder models — the DevPULSE embedder
  (Qwen backbone), reranker (Nemotron backbone), and classifier. This is where
  a custom candle build genuinely pays off: no Python, no server hop, single
  binary, predictable latency. Enabled by the `candle` feature.
- **vLLM (external, Python, OpenAI-compatible API):** large-LLM generation at
  scale. Driven behind the `Generator` trait; never reimplemented.
- **llama.cpp (GGUF):** local / quantized / edge inference via server or FFI.
- **candle-vllm:** tracked as an *experimental* Rust-native generation backend
  for bake-offs, not a committed dependency.
- **Python:** owns DevPULSE training/tuning (HF/PEFT/TRL) and heavy GPU serving
  via vLLM. Headless Clyffy training stays Python.
- **Go:** not adopted; it introduces ops cost with no unique role here.

## Why not port vLLM to Rust

1. **The value is CUDA, not Python.** vLLM's throughput comes from PagedAttention,
   continuous batching, chunked prefill, prefix caching, speculative decoding,
   and fused kernels — already C++/CUDA running off-GIL. The Python is thin
   orchestration over GPU compute that dominates wall-clock. Rewriting the
   orchestration in Rust trades microseconds of scheduling against
   millisecond-scale GPU ops.
2. **Kernel coverage regression.** A Rust path would lean on candle, whose kernel
   coverage trails PyTorch/vLLM. For non-trivial models this is likely a **net
   performance loss**, not a gain.
3. **Feature-velocity loss.** vLLM ships new models, quant schemes, and kernels
   continuously. A reimplementation forfeits that and becomes a maintenance
   treadmill.

**Conclusion:** integrate vLLM as a backend; do not rebuild it.

## Consequences

- `rro-core` defines model traits (`Embedder`, `Reranker`, and a future
  `Generator`); backends are cargo features so builds stay lean.
- Config selects backends at runtime; a small provider registry resolves them.
- A bake-off harness compares backends on the same eval set (quality: recall@k,
  nDCG, MRR; performance: p50/p95 latency, throughput, RSS). Winner is data.
- The default build stays weightless and fully testable; backends are additive.

## Revisit if

- candle-vllm (or a Rust serving path) reaches vLLM-class throughput/feature
  parity on our target GPUs — then reassess the generation backend.
- A single-language deployment constraint outweighs the throughput cost.
