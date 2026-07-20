# connxism: RocksDB → Fjall 3.x port

Target: `crates/connxism` (~5.6k LOC). RocksDB usage is confined to 6 files and —
critically — already funneled through the `Db` wrapper in `estate.rs` (`cf()`,
`get_json`, `put_json`, `write(batch)`, `get_u64`). That wrapper is the seam. We do
**not** build a generic storage trait for a two-engine migration; we port the
wrapper directly and keep call sites stable.

Both backends coexist during the migration behind cargo features
(`kvs-fjall` pulls Fjall in; RocksDB stays the default) so the parity/differential
tests run against each. RocksDB stays the shipping default until Phase 7's gates
pass.

## Fjall 3.1.7 terminology

Fjall 3.1.7 renamed the classic pair. This plan uses the 3.1.7 names:

| concept | RocksDB | Fjall 3.1.7 | older Fjall |
|---|---|---|---|
| the whole store | `DB` | `Database` | `Keyspace` |
| one logical store | column family | `Keyspace` | `Partition` |
| atomic multi-store write | `WriteBatch` | `db.batch()` → `WriteBatch` | — |
| durability | `WriteOptions::set_sync` | `PersistMode::{SyncAll,Buffer}` | — |

Proven against the real API in `crates/connxism/tests/fjall_spike.rs` (4/4,
`--features kvs-fjall`): keyspaces↔CFs + restart durability, atomic cross-keyspace
batch, prefix scan, and the `tdf` RMW replacement.

---

## Phase 0 — Pin the contract (½ day) — invariants the port must preserve

1. **Atomicity unit**: one `WriteBatch` across multiple CFs commits atomically →
   Fjall cross-keyspace `db.batch()` through the shared journal. Like-for-like.
2. **Durability toggle**: `Db.1: bool` (fsync-on-write) → `db.persist(SyncAll)`
   after `batch.commit()` vs. leaving it buffered.
3. **Key ordering**: lexicographic, `SEP = 0x00` prefix encoding in `keys.rs`.
   Fjall is lexicographic; `keys.rs` ports byte-for-byte untouched. (Guard: Fjall
   keys ≤ 65,536 bytes, values ≤ 2^32 — term keys and vector blobs are inside
   both; verify no pathological tag/term input can exceed the key limit, add a
   guard in `keys.rs` if quotas don't already cover it.)
4. **Counter semantics**: `doc_count`, `total_tokens`, `feed_seq`, shape census are
   RMW owned by `Transaction` (read at begin, put at commit). Unchanged.
5. **`tdf` semantics**: associative i64 merge — the one thing Fjall has not. See
   Phase 3.

## Phase 1 — Db wrapper swap (1–2 days)

```rust
// estate.rs
pub(crate) struct Db {
    db: fjall::Database,
    parts: HashMap<&'static str, fjall::Keyspace>,  // opened once from COLUMN_FAMILIES
    fsync: bool,
}
```

- `cf(name)` → keyspace lookup (infallible after open; keep the `Result` to avoid
  churn at 60+ call sites).
- `get_json` / `put_json` / `get_u64` → `keyspace.get()` / `keyspace.insert()`.
  Single-op inserts still journal; prefer batches on hot paths, as today.
- `write(batch)` → the Fjall `WriteBatch` built in `txn.rs`, committed then
  `db.persist(SyncAll)` iff `fsync`.
- Open path: `Database::builder(path).open()` + `db.keyspace(name, opts)` per
  `COLUMN_FAMILIES`, per-keyspace `KeyspaceCreateOptions`.

## Phase 2 — Options translation (1 day). Delete more than you translate.

| RocksDB (current) | Fjall 3.x |
|---|---|
| `write_buffer_bytes` × CF budgeting | per-keyspace memtable size |
| `BlockBasedOptions` + block cache | database-level cache (unified block/blob) |
| LZ4 | LZ4 default; per-keyspace compression |
| Bloom filters | built-in per-keyspace filter policy |
| BlobDB on `vecs`/`nvecs`/`mvecs` | **KV separation** per-keyspace (value log) |
| compaction/flush thread tuning | Fjall background workers; mostly delete |
| `set_merge_operator_associative` on `tdf` | — (Phase 3) |

The BlobDB rationale (compaction must not rewrite vectors) maps directly onto
Fjall's value log — same intent, same knob shape.

## Phase 3 — Replace the `tdf` merge operator (1–2 days, the real work)

Today: blind `merge_cf(tdf, term, delta_i64_le)`, operands composed by
`merge_i64_add`. Fjall has no merge operators (3.1's compaction filters
transform/drop entries; they do not compose operands).

Replacement: **transaction-scoped delta accumulator**, mirroring what `txn.rs`
already does for counters:

- Add `df_deltas: BTreeMap<Vec<u8>, i64>` to `Transaction`.
- Write ops record `±1` per term into the map instead of emitting merge operands.
- At `commit()`: for each entry, `get` current i64 LE from `tdf`, add the delta,
  `insert` into the same batch. On-disk format (i64 LE) unchanged → **no `tdf`
  data migration**, read paths untouched.
- Correctness needs no concurrent writer between the reads and the batch commit.
  connxism is already effectively single-writer (counters read at begin); make it
  explicit with a `tokio::sync::Mutex<()>` writer gate held `begin`→`commit`.
  Recommendation: own the mutex + plain `Database` (keeps `txn.rs`'s design
  authority in our code; avoids coupling to Fjall's tx layer).
- Cost: each df touch is read+write instead of an appended operand. Batched per
  txn (one RMW per distinct term per txn), bounded by vocabulary-per-batch, not
  tokens — measure in Phase 7.

## Phase 4 — Iterators (½ day, mechanical)

- `iterator_cf(cf, From(&prefix, Forward))` + manual `starts_with` break →
  `keyspace.prefix(prefix)`; delete the manual prefix checks (~6 sites).
- `IteratorMode::Start` full scan (`vecs` rebuild) → `keyspace.iter()`.
- Fjall iterators are `DoubleEndedIterator` — free reverse scans if `query.rs`
  ever wants them. Iterators yield `Guard`; resolve with `.key()` / `.value()`.

## Phase 5 — Snapshot / flush / compaction (1 day)

- `snapshot_to` (RocksDB checkpoint) → `db.persist(SyncAll)` then a **directory
  copy** at the quiescent point (callers already snapshot at quiescence; the copy
  must include the journal). Longer-term: logical export (iterate keyspaces →
  archive), which doubles as the cross-engine migration tool.
- `flush()` / `flush_wal(true)` → `db.persist(SyncAll)`.
- Manual full-range compaction → verify Fjall 3's per-keyspace major-compaction
  surface at port time; if absent, drop the endpoint or make it advisory (LSM
  housekeeping is automatic).

## Phase 6 — Migration & rollout

1. No in-place conversion. Ship a `migrate` path: open old RocksDB read-only →
   iterate every CF → batched inserts into Fjall. Reuse `COLUMN_FAMILIES` as the
   manifest. The ANN graph rebuilds from `vecs` (the two-phase design pays off).
2. Keep RocksDB behind `kvs-rocks` for one release; `kvs-fjall` becomes default
   once gated. The `Db` seam makes this a small `cfg` surface, not a trait.
3. Parity gate: run connxism's existing suite against both engines; add a
   differential test that replays a recorded op log into both and diffs full CF
   dumps.

## Phase 7 — Bench gates (before deleting RocksDB)

- Bulk upsert throughput (firehose), realistic vector dims → validates KV
  separation.
- df-heavy ingest (high vocabulary churn) → validates the Phase 3 accumulator.
- Prefix-scan latency on `terms`/`sparse` under load.
- Snapshot time + recovery-from-copy correctness.

## Risks

- **Fjall 3 disk format is young** (Jan 2026); maintainer signalled feature work
  winding down into 2026 (reads as stabilization). We trade RocksDB's decade of
  scar tissue for a cleaner codebase; the op-log differential test is the
  insurance.
- Blocking-I/O discipline unchanged: Fjall is sync like RocksDB — keep the
  existing `spawn_blocking` boundaries.
- **Do not port and redesign simultaneously.** The `tdf` accumulator (Phase 3) is
  the only semantic change; everything else must be behavior-preserving or the
  differential test loses meaning.

**Estimated effort: ~6–9 focused days**, dominated by Phases 3 and 6.

---

## Backend parity matrix (authored 2026-07-18)

Both backends live behind the KV seam (`crate::kv`), one selected per build.
Every RocksDB capability that Fjall 3.1.7 **can express** has an implemented
Fjall equivalent in `kv/fjall.rs`; the connxism suite passes identically under
each. Open-path tuning is translated (not dropped) so Fjall is a first-class
peer, not a defaults build. **Two** RocksDB capabilities have no faithful Fjall
3.1.7 equivalent and are documented — not faked — in the *Capability conflicts*
section below.

| Capability | RocksDB (`kv/rocks.rs`) | Fjall (`kv/fjall.rs`) | Proof |
|---|---|---|---|
| open / CFs | `open_cf_descriptors` | `Database::builder` + `keyspace` per CF | suite opens |
| shared block/blob cache | `Cache::new_lru_cache(block_cache_bytes)` | `builder.cache_size(block_cache_bytes)` | opens |
| background workers | `increase_parallelism(background_jobs)` | `builder.worker_threads(background_jobs)` | opens |
| per-CF memtable | `set_write_buffer_size(write_buffer_bytes)` | `max_memtable_size(write_buffer_bytes)` | opens |
| point-lookup bloom | `set_bloom_filter(10.0)` on 9 CFs | `filter_policy(Bloom BitsPerKey 10.0)`, same 9 CFs | opens |
| scan CFs: no bloom | (extractor, no whole-key bloom) | `filter_policy(None)` + `expect_point_read_hits(false)` | postings scan |
| compression | Lz4; None on vec CFs | `CompressionType::Lz4`; `None` on vec CFs | suite |
| BlobDB (vec CFs) | `enable_blob_files` + `min_blob_size(4K)` | `with_kv_separation(threshold 4K)` | vector round-trip |
| get / put / get_json / put_json / get_u64 | `get_cf` / `put_cf` | `Keyspace::get` / `insert` | suite |
| atomic cross-CF batch | `WriteBatch` | `db.batch()` | transaction tests |
| durability toggle (fsync) | `WriteOptions::set_sync` | `persist(SyncAll)` iff fsync | reopen test |
| `tdf` document-freq merge | `merge_operator_associative(i64_add)` | RMW accumulator folded at `write()` (i64-LE unchanged) | df-counter tests |
| iterate from / all | `iterator_cf(From/Start)` | `Keyspace::range` / `iter` (Guard) | postings/scan tests |
| flush memtable | `flush_cf` | `rotate_memtable_and_wait()` | maintenance test |
| WAL sync | `flush_wal(sync)` | `persist(SyncData)` iff sync, else `Buffer` | flush test |
| compact | `compact_range_cf` | `major_compact()` (best-effort, logged) | compact test |
| cf size | `total-sst-files-size` property | `disk_space()` | cf_sizes |
| filter/index pinning | `pin_l0_filter_and_index_blocks` | `filter_block_pinning_policy` + `index_block_pinning_policy` on point-lookup CFs | opens |
| data block size | `set_block_size(16K)` | `data_block_size_policy(16K)` | opens |
| blob GC | `enable_blob_gc(true)` | `KvSeparationOptions` `staleness_threshold`/`age_cutoff` | vector round-trip |
| write-memory budget | global `db_write_buffer_size` | role-weighted per-keyspace `max_memtable_size` + `max_journaling_size` WAL cap | opens (see conflicts) |
| snapshot | `Checkpoint` (hard-link, atomic) | applier quiesced + directory copy under a held `Database::snapshot()` (MVCC pin) | crash/snapshot round-trip |

**Signals unchanged.** Estate-level telemetry (`rro_core::events::emit` for
`estate.flush`/`compact`/`snapshot`/`graph_persist`, the `SignalKind` stream) is
backend-agnostic and fires under both — the analytics/audit/logging baseline is
untouched by the backend choice. The Fjall backend additionally traces
compaction failures (`tracing::warn`).

**Known Fjall constraint (key length).** Fjall caps keys at 65 536 bytes;
RocksDB has no key-length limit. The only unbounded key path is `pidx`
(`keys::pidx_key(field, value, id)` from arbitrary metadata values) and
user-supplied `doc_id`. Guarded at the write boundary so **both** backends reject
an over-limit key identically (`keys::MAX_KEY_LEN`) — no silent cross-release
divergence.

## Capability conflicts (documented, not faked) — audited 2026-07-18

Two RocksDB capabilities have **no faithful equivalent in Fjall 3.1.7**. Rather
than fake parity, they are called out here and where they live in the code
(`kv/fjall.rs` module doc). Both were confirmed against the vendored `fjall`/
`lsm-tree` 3.1.7 source.

1. **`CF_TERMS` BM25 prefix bloom.** RocksDB accelerates posting-list *prefix*
   scans with a NUL-terminated `prefix_extractor` + `memtable_prefix_bloom_ratio`,
   skipping SSTs/memtables that hold no postings for a term. Fjall's filters are
   **whole-key, point-read only** (`FilterPolicyEntry` is `None | Bloom`; the bloom
   is consulted only in `Table::get`, never on range scans) — there is no prefix
   filter or key transform. `CF_TERMS` therefore runs unfiltered: **BM25 lookups
   stay correct** (a `range`/`prefix` seek still finds the postings), but the
   scan-skip optimization is absent; it is served by leveled locality + the shared
   block cache. Closing this would require forking `lsm-tree` to add a prefix
   filter — deferred until real-workload measurement (clyffy terminal → DuckDB)
   shows it matters.

2. **Global cross-keyspace memtable cap.** RocksDB's `db_write_buffer_size` is a
   single hard ceiling over all CFs. Fjall's equivalent (`max_write_buffer_size` /
   `WriteBufferManager`) is `#[deprecated = "todo"]` and off by default in 3.1.7.
   Replaced by a **role-weighted per-keyspace budget** (`memtable_size_for`: hot
   recall-write CFs get the full `write_buffer_bytes`, cold CFs a quarter) plus a
   `max_journaling_size` WAL cap, so aggregate write memory is still an intentional
   number — just enforced by sizing each keyspace rather than one global counter.

Everything else RocksDB tunes is implemented on Fjall (see the matrix above).
The `data_block_hash_ratio` in-block point-read index — a Fjall-native speedup
RocksDB has no analogue for — is available but **deferred** (it is `#[doc(hidden)]`
/ experimental in 3.1.7; not shipped unmeasured).
