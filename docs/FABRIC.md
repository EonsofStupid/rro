# FABRIC.md — the devstation fabric envelope (baseline)

The fabric is the always-on layer over the devstation: as content flows — a
`clyffy code` turn, later a file/git event — a **signal** is emitted carrying a
small, well-known set of metadata **at emit time**. That metadata is what lets
the rest of the engine (rrd classification, recall filtering, connectome growth,
security redaction) work precisely and cheaply instead of reconstructing context
after the fact.

This document is the **baseline** envelope — minimal on purpose, and designed to
**evolve on analytics**.

## The baseline envelope

Implemented in [`crates/rro-core/src/fabric.rs`] as typed accessors over the
ordinary [`Metadata`] bag. Every key is namespaced `fab.*` so it never collides
with a consumer's own metadata.

| Key | Type | Meaning | Default |
|---|---|---|---|
| `fab.source` | enum (open) | which tap emitted it — `clyffy_code` (v1), `file_event`, `git_event`, `shell`, `editor`, … | — (required) |
| `fab.session` | string | the session / locus the signal belongs to | — (required) |
| `fab.ts_ms` | u64 | emit time, epoch ms | — (required) |
| `fab.actor` | enum (open) | `operator` · `agent` · `system` | `operator` |
| `fab.domain` | string | rrd domain hint, refined by the classifier | unset |
| `fab.boundary` | string | tenant / project / scope — feeds connectome growth | unset |
| `fab.security` | enum (closed) | `public` · `operator` · `secret` — gates recall + redaction | `operator` |

The three **identity keys** (`source`, `session`, `ts_ms`) are required: a bag
without them is not a fabric signal, and `FabricMeta::read` returns `None`.

### Security is fail-safe

`fab.security` is the one **closed** enum. A *present but unrecognized* value
parses to `secret` (most restrictive) — we never widen access on a class we don't
understand. A *missing* key defaults to `operator` (the operator's own devstation
is the baseline scope).

## Why a bag, not a struct (the evolution policy)

The envelope lives as `fab.*` keys on `Metadata` rather than a rigid struct so
that **a new dimension is a plain key today, promoted to a typed field only once
DuckDB analytics show it matters.** `Source` and `Actor` are open enums
(`Other(String)`) for the same reason: a new tap or actor is a *value*, not a
breaking change. Keys already in the bag that are not part of the envelope are
never disturbed by `write_into`.

Promotion path: emit → DuckDB captures the `fab.*` distribution → a hot free-form
key earns a named enum variant / typed field in a later revision of this spec.

## Not yet in the baseline (deferred)

- A typed `FabricSignal` struct (promoted from the bag once analytics justify the
  shape).
- `content_ref` indirection (baseline carries content inline via `Document.text`).
- Per-key redaction transforms for `secret`.
