# MAP — the rro workspace

**RRO — Reason Ready Objects.** Pull it and you have one cohesive engine: a
dedicated AI memory with KV persistence, hybrid RAG, a graph, and a
classification spine — the well-organized structure meant to scale and grow with
your AI. One binary, one KV backend (Fjall), no deprecation.

## The four layers

```
              ┌───────────────────────────────────────────────┐
  ORCHESTRATE │ rro-engine   the daemon (`rro`), wires it all  │
              ├───────────────────────────────────────────────┤
  REASON      │ rrd · classifier   shape → intent → readiness  │
  (RRD spine) │ rro-ql             text → typed query plane    │
              ├───────────────────────────────────────────────┤
  RECALL      │ recall   dense ANN + exact store               │
  (RAG)       │ reranker embedder model-registry  perception   │
              │ connectome   the visual/relational graph       │
              ├───────────────────────────────────────────────┤
  PERSIST     │ connxism   Fjall estate: hybrid recall spine   │
  (KV + graph)│ connectors resumable source ingest + sync      │
              └───────────────────────────────────────────────┘
  everywhere  │ rro-core (contract) · rro-net (a2a) · rro-client│
```

## Crates (`crates/`)

| Crate | Role |
|---|---|
| `rro-core` | The contract: shared domain types + the engine traits every component implements. |
| `rrd` | The reason-ready object JIT — shape-lattice modes + slivers, semantic-router tagging, per-shape compiled plans, the RROs themselves. |
| `classifier` | The Reason Ready daemon — judges whether context is sufficient to reason on (readiness). |
| `recall` | Dense vector memory — the Recall engine (ANN graph + exact store). |
| `reranker` | True-relevance ordering — lexical default + Nemotron plug-point. |
| `embedder` | Perception — the `Embedder` trait: deterministic (CI/no-weights) default + Qwen/Nemotron plug-point. |
| `model-registry` | Selection is data, not code — turns embedder/reranker config into boxed trait objects. |
| `connectome` | The visual/relational map — renders how memories and reasoning connect. |
| `connxism` | The kvs-connectome. **Fjall-backed** estate: nodes, connectors, warp points, and the persistent hybrid (vector + BM25) recall spine. |
| `connectors` | Connector drivers + the sync engine — operators share sources; the estate ingests them, resumably. |
| `rro-net` | The a2a / node networking surface — embedded is not isolated. |
| `rro-ql` | RRQL — text → the typed query plane. Parsing only; no execution. |
| `rro-client` | The typed client — treat any rro node as local over the a2a layer-2 protocol. |
| `rro-engine` | The orchestrator — wires the components into one flow and runs the embedded `rro` daemon. |

## Storage

One KV backend: **Fjall 3.x** (pure-Rust LSM), named only inside `connxism`
behind the `kv` seam. No RocksDB, no dual-backend machinery. See
[README](README.md#storage-backend--fjall) and
[docs/PARITY.md](docs/PARITY.md).

## The two products, one engine

- **rro** — for devs with their own harness: pull the `rro-engine` git dep, follow
  the integration checklist.
- **clyffy** — the full turnkey system with rro already optimized + integrated.
