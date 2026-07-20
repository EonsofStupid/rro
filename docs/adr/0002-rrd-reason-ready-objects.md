# ADR-0002 — RRD: the Reason-Ready Object JIT

**Status:** accepted (design committed 2026-07-15; refine with the author's
review). This design predates this repo by years — it lived in research
sessions and was never committed anywhere. This document ends that. In the
author's words: *"In layman, it's shape and tags."* Reason-ready **objects**,
not reason-ready text.

## The idea

LLMs reason best over **structured, typed, provenance-carrying objects** —
schemas, fields, tags — not over undifferentiated text soup. (This is the same
direction Anthropic's interfaces push: typed content blocks, tool schemas,
structured outputs.) But real estates ingest arbitrary payloads from
connectors: mail, rows, files, events. Something must turn *arbitrary
payloads* into *reason-ready objects* — fast, consistently, at ingest scale.

That something is **RRD — the reason-ready JIT.**

The JIT analogy is exact, not decorative. Dynamic-language JITs (V8 et al.)
watch objects at runtime, group them by **hidden class — literally called a
"shape"** — compile a specialized fast path per shape, and cache it (inline
caches). Objects of a seen shape hit the compiled path; a new shape triggers
compilation once, then everything of that shape is fast.

RRD does this to data:

```
payload ──▶ shape inference ──▶ shape-cache lookup ──┬─ HIT ──▶ run compiled plan ──▶ RRO
            (Shape::of ✅)      (the inline cache)   └─ MISS ─▶ compile plan for shape,
                                                                cache it, run it ──▶ RRO
```

- **Shape** = the schema fingerprint of a payload (field → type). Already
  live in the estate (`connxism::Shape`, the shape census CF).
- **Plan** (the compiled artifact) = per-shape distillation: which fields are
  identity / content / salience / time; which tag rules apply; what the
  resulting object schema is. Compiled **once per shape**, cached, reused for
  every payload of that shape — that is the JIT.
- **Tags** = classification emitted by plans (and by operators), the second
  axis of specialization. Already live in the estate (tags CF).
- **RRO (Reason-Ready Object)** = the output: a typed object
  `{ id, shape_id, schema, fields (typed), tags, provenance (connector, doc,
  estate), salience, readiness_hints }`.

## Why this is load-bearing for the whole engine

- **The classifier stops guessing.** Today readiness is judged over raw
  candidate text. Over RROs it judges *structured evidence*: which schema
  fields are present, per-field coverage, provenance quality. The readiness
  gate becomes explainable — and trainable (the DevPULSE classifier's
  features are RRO fields, not tokens).
- **The reranker gets features** (field matches, tag agreement, provenance
  weight) beyond lexical/dense scores.
- **The connectome renders it natively** — shape and tag nodes are already
  first-class in the map. RRD makes them *the* organizing principle instead
  of passive census data.
- **JIT telemetry falls out of the event stream:** `rrd.compile` (shape
  miss), `rrd.hit`, cache hit-rate as a DuckDB trend — the estate literally
  reports how "warmed up" its understanding of an operator's data is.

## The gate ladder (canonical, 2026-07-15 — the operator's cascade)

Every payload climbs a staged ladder; each tier costs ~an order of magnitude
more than the last, so almost everything resolves cheap. Implemented in
`rrd::gates`:

| tier | budget | decides | status |
|---|---|---|---|
| source stamp | 1–10 µs | identity, session, project, mode, channel, source | ✅ `SourceStamp` on every RRO |
| L0 deterministic | 10–50 µs | schema (shape), cached plan, taint, size, routing | ✅ (`l0_deterministic` + shape/plan inline cache) |
| L1 lexical | 0.1–1 ms | unicode anomalies, secret signals, injection signals, operation/effect | ✅ (`l1_lexical`, one scan; flags never silently block) |
| L2 semantic | 2–20 ms | intent hierarchy, ambiguity, domain, risk, confidence | ✅ core (semantic router on precomputed embedding + ambiguity margin; hierarchy 🔨) |
| L3 action gate | at **every action** | fresh authorization, capability attenuation, confirmation | seam only (`ActionGate` trait) — lands with P5 auth |
| L4 deep evaluation | concurrent | larger model, output inspection, behavioral analysis | seam only (`DeepEvaluator` trait) — lands with P7 DevPULSE |

Session semantics (`rrd::trigger`): RRD fires on **conversation start** and
on **idle-resume** — the re-orientation moments — then routes fresh context
to intent: the operator *should* pick a mode (Dev / Creative / media), but
the engine also detects "we need to be in X mode", switches, and the expert
state absorbs the standing task list. Intent + tags are how RRD evolves.
(Griff — the operator-voice layer that keeps the host's language plain for
non-technical operators without rewarding underspecified work — consumes
RROs + readiness; it lives host-side, not in this engine.)

## The shape baseline: snapshot of normal, evolving forever (2026-07-15)

*"We start building a snapshot — a baseline, in layman — of shapes, and
improve predictability."* Implemented in `rrd::baseline`, grounded in three
proven bodies of practice:

1. **Compiler feedback vectors (V8-class JITs).** RRD is *the instant first
   thing*: every payload and every query hits the ladder at first touch,
   **before any embedding is paid for** — blocked payloads never reach the
   model (measured in `SyncReport::blocked`). Each context (connector,
   channel, session) accumulates type feedback exactly like a call site's
   feedback slot, and its **monomorphic → polymorphic → megamorphic** state
   is a first-class number: `predictability = 1 − normalized entropy` of its
   shape distribution.
2. **Reference profiles + PSI (ML-observability practice).** The baseline
   commits versioned **snapshots** — the durable "this is normal" — and
   measures each context's recent window against its snapshot with the
   population stability index. `PSI > 0.25` emits `rrd.drift`: the world
   changed at that source. Snapshots persist in the estate
   (`x:rrd:baseline`) and restore on session start — the baseline survives
   restarts and **grows across sessions** (gated by test).
3. **Speculative inline caching.** Before identifying a payload's shape, RRD
   *predicts* it from the context's distribution; the observation settles the
   ledger. The per-context **hit-rate curve is the measured "predictability
   is improving"** — it locks to ~1.0 on stable sources within one sync and
   is exported as an estate trend (`connector.<id>.predictability`).
   Recency-weighted decay (O(1), growing-unit) makes the baseline adapt on
   regime change: **drift alerts fast, identity changes slow** — both by
   design and both tested.

Ordering is enforced in code, not convention: the connector sync runs
distill (stamp→L0→L1→shape→predict) *before* embedding and routes L2 tags on
the survivor embeddings afterward; the query flow runs RRD as its first
stage (`flow.stage = "rrd"`), returns gated results for blocked queries
without ever invoking the embedder, and stamps routed **intent** tags onto
every `RecallResult`.

## Design (phase P4)

New crate **`rrd`** (component, depends only on `rro-core`; estate
integration via `connxism`):

- `ShapeRegistry` — shapes get stable ids + stats (promotes the existing
  census to a registry; persisted in `meta`/`shapes`). The registry is a
  **lattice, not a flat set** — this is the *sliver* scheme: **modes are the
  base shapes** (mail, record/row, document, event, location, media), and
  every observed shape attaches under its mode as a **sliver** — a thin
  specialization that inherits the mode plan and overrides only what its
  extra fields demand. Shapes *evolve*: drift produces a sibling sliver,
  recurring slivers get promoted with their own compiled plan, dead slivers
  age out. Tagging is hybrid with identification — tags emitted by mode-level
  rules refine sliver placement, and sliver placement scopes which tag rules
  fire.
- `Plan` — serializable per-shape distillation: field roles, tag rules,
  salience weights. Stored in a `plans` CF keyed by shape key. Versioned.
- `Distiller` (the JIT core) — `distill(payload) -> Rro`: shape-infer →
  cache lookup → compile-on-miss → execute. Compilation v1 is rule-derived
  (heuristics over field names/types + operator-supplied tag taxonomy);
  the seam allows a learned compiler later (DevPULSE).
- `Rro` — the typed object, serde-serializable, addressable in the estate
  (`rro` CF or materialized on demand), consumed by classifier/reranker/
  connectome/a2a.
- Ingestion machine gains an optional RRD stage: payload → RRO → index both.

## Invariants (testable)

1. Same shape ⇒ same plan (cache hit; compile exactly once per shape version).
2. `distill` is deterministic for a given (payload, plan version).
3. RROs always carry provenance; no orphan objects.
4. Shape drift (field added) ⇒ new shape id, new plan — never silent reuse.
5. Cache hit-rate is observable (`rrd.*` events) and trends upward on a
   stable corpus.

## Open for the author's review

- The exact expansion of "RRD" (the acronym) — the design above stands
  regardless.
- **Sliver** (recovered 2026-07-15, author's definition): the hybrid
  tagging/shape-identification scheme — modes as base shapes, observed
  shapes evolving beneath them. Captured in `ShapeRegistry` above; confirm
  the mode list.
- Tag taxonomy source: operator-defined, plan-derived, or both (assumed both).
- Whether RROs are always materialized (storage cost) or distilled on read
  (latency cost) — assumed: cache hot shapes, distill cold ones on read.
