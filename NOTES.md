# What you actually wanted vs what was built — the honest ledger

_Authored 2026-07-16. Your requirements, where they were ignored, and what
must happen immediately. No spin._

## What you asked for (and where it was NOT delivered)

### 1. REAL models — Qwen embedder + Nemotron reranker. NOT synthetic/mock.
**Status: NOT DONE. This is the big one.**
- `crates/embedder/src/devpulse.rs` is a **stub** — `ModelSpec::qwen()` + a
  `candle` feature gate + `TODO(devpulse): load the Qwen backbone` that errors
  when called. Never wired.
- Everything ran on the **synthetic deterministic embedder**
  (`deterministic.rs`, hashed pseudo-vectors). So every accuracy number in the
  docs (1.000 accuracy@10, the bake-off, "1.000 vs 0.025") is **meaningless for
  real retrieval** — it measures fake vectors against fake vectors.
- **You still do not know if this engine is worth a damn on real text.** That is
  the core thing you wanted answered and it is unanswered.
- Blocker in THIS container: huggingface.co is 403 (egress policy), ~4.9 GB free
  disk. Qwen+Nemotron weights (~2.5 GB+) + candle build don't fit / can't fetch.
  **This box cannot finish it. It must move.**

### 2. Deployment = Podman QUADLETS. NOT Docker.
**Status: DONE WRONG.**
- `deploy/` ships a **Dockerfile** + a plain `rro.service`. No `.container`
  quadlet, no `.pod`, no podman anything. Podman isn't even installed here.
- What you wanted: a Podman **Quadlet** (`rro.container` → generated systemd
  unit via `podman-system-generator`), rootless, not Docker.

## What IS real and worth keeping (do NOT rebuild)

The engine itself is genuine, ~14,400 loc, 30 gated test files. It is
model-agnostic — it stores/indexes/queries whatever vectors it's handed:
- `connxism` — RocksDB estate: hybrid dense+BM25, payload indexes
  (datetime/uuid/geo), sparse, named vectors, collections/aliases, changefeed,
  quotas, flush/compact. Real.
- `recall` — the ANN graph. Algorithm real; its recall *quality* was only ever
  checked on synthetic vectors, so re-verify on real ones.
- **The trait seam** (`rro-core/src/traits.rs`: `Embedder`/`Reranker`/
  `Classifier`/`Recall`) — this is why you don't start over. Real Qwen+Nemotron
  drop in **behind these traits**; the flow, estate, and query plane don't change.

## What must happen IMMEDIATELY (in order)

1. **GET OUT OF THIS CONTAINER.** It cannot reach HF and has no disk/GPU budget.
   Everything is committed to `eonsofstupid/rro` — nothing is stranded.
   Move to an environment with: huggingface.co reachable (or weights mounted),
   real disk (10 GB+), and CPU/GPU inference budget.

2. **Wire the REAL models (the actual deliverable):**
   - `candle-core` + `candle-transformers` + `tokenizers` behind the `candle`
     feature; fill the `TODO` in `devpulse.rs` — load Qwen3-Embedding, forward
     pass, mean-pool. Same for Nemotron behind `Reranker`.
   - Swap the default embedder/reranker in `rro-engine`.
   - **Re-run the bake-off.** Only now does the accuracy number mean anything.
     Treat every current doc number as UNVERIFIED until this is done.

3. **Redo deployment as Podman Quadlets:**
   - Delete the Docker path. Author `deploy/rro.container` (Quadlet), rootless,
     `[Container]` + `[Service]` + `[Install]`; document `systemctl --user`
     install via the podman generator. `config.env` stays as the drop-in.

4. **Re-tune + re-gate on real vectors:** ANN `ef`/graph params were tuned on
   synthetic distributions; verify recall@10 on real Qwen embeddings.

## Bottom line
Engine: keep it. Models (Qwen+Nemotron): a stub — the thing you asked for, not
done, and impossible in this container. Deploy: wrong tech (Docker not Podman
Quadlets). Immediate move: **relocate to a capable environment, then wire the
real models and re-measure.** That is where the real answer to "is this engine
worth it" finally comes from.
