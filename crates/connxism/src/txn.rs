//! Multi-statement transactions: BEGIN → writes → COMMIT (atomic) / CANCEL (discard).
//!
//! ## Why this is not just "accumulate a WriteBatch"
//!
//! Every write op already builds a [`crate::estate::Batch`], which is atomic on its
//! own — so single-op atomicity was never the gap. The gap is *multi-statement*
//! atomicity, and the naive version of it is silently wrong.
//!
//! The estate's counters — `doc_count`, `total_tokens`, `feed_seq`, the shape
//! census — are **read-modify-write**: an upsert reads `doc_count`, adds the net
//! new, and puts the result. Two upserts appended to one batch would each read
//! the same pre-commit `doc_count = N`, each put `N + 1`, and the last write
//! would win — the count ends at `N + 1` after adding two documents. The df
//! merge operands compose (they are associative merges), but the counters do not.
//!
//! So a transaction cannot be "call the existing helper twice into one batch". It
//! must own the counter state: read it once at [`Transaction::begin`], thread it
//! through every statement, and put it once at [`Transaction::commit`]. That is
//! what this type is.
//!
//! ## The unification
//!
//! Once the transaction owns the batch and the counters, a single write and a
//! multi-statement transaction are the same code path: a single write is just an
//! implicit one-statement transaction that commits immediately. Every write op is
//! written as `*_into(&mut Transaction, …)`; the public async method wraps one
//! op in `begin → op → commit`, and an explicit `BEGIN … COMMIT` wraps several.
//! There is no second, parallel "transactional" implementation to drift from the
//! real one.
//!
//! ## Rollback and the out-of-band graph
//!
//! The ANN graph is applied out-of-band from the durable `vecs` CF (the two-phase
//! design). A transaction never touches the graph directly: it collects the graph
//! ops as [`GraphOp`]s and pushes them to the pending applier **only at commit**,
//! *after* the durable batch has landed. On CANCEL the batch is dropped and the
//! graph ops are dropped with it — nothing durable was written, so the graph,
//! which derives from `vecs`, never sees the rolled-back records. Rollback needs
//! no 2PC precisely because the graph is downstream of the truth, not beside it.

use std::collections::BTreeMap;

use rro_core::{Embedding, Id, Result, RroError};

use crate::estate::{Batch, Db};
use crate::keys::{CF_META, META_DOC_COUNT, META_FEED_SEQ, META_SHAPES, META_TOTAL_TOKENS};
use crate::pending::Pending;

/// A deferred graph operation, applied to the out-of-band ANN index at commit.
pub(crate) enum GraphOp {
    /// The record's vector enters the graph.
    Upsert(Id, Embedding),
    /// The record leaves the graph (tombstone).
    Remove(Id),
}

/// One transaction: an accumulated durable batch, the threaded counter state, and
/// the graph ops to apply once the batch commits.
///
/// Borrows the `Db` and `Pending` for its lifetime — a transaction is a scoped
/// unit of work, not a stored handle. It is used inside a single blocking section
/// held under the estate's writer lock, so its counter reads are consistent and
/// no other writer interleaves.
pub(crate) struct Transaction<'a> {
    db: &'a Db,
    pending: &'a Pending,
    pub(crate) batch: Batch,

    // Counter state: read once here, mutated by each `*_into`, put once at commit.
    pub(crate) doc_count: u64,
    pub(crate) total_tokens: u64,
    pub(crate) feed_seq: u64,
    pub(crate) shapes: BTreeMap<String, u64>,
    counters_dirty: bool,

    graph_ops: Vec<GraphOp>,
}

impl<'a> Transaction<'a> {
    /// Open a transaction: snapshot the counters into memory.
    pub(crate) fn begin(db: &'a Db, pending: &'a Pending) -> Result<Self> {
        Ok(Transaction {
            batch: Batch::new(),
            doc_count: db.get_u64(META_DOC_COUNT)?,
            total_tokens: db.get_u64(META_TOTAL_TOKENS)?,
            feed_seq: db.get_u64(META_FEED_SEQ)?,
            shapes: db.get_json(CF_META, META_SHAPES)?.unwrap_or_default(),
            counters_dirty: false,
            graph_ops: Vec::new(),
            db,
            pending,
        })
    }

    /// Mark the counters as changed by this statement, so commit writes them back.
    /// A read-only transaction leaves them untouched and skips the meta writes.
    pub(crate) fn touch_counters(&mut self) {
        self.counters_dirty = true;
    }

    /// Queue a graph op — applied to the ANN index only if the transaction commits.
    pub(crate) fn push_graph(&mut self, op: GraphOp) {
        self.graph_ops.push(op);
    }

    /// Commit: put the counters, write the batch atomically, then push the graph
    /// ops. The order is load-bearing — the durable write lands first, so a graph
    /// op can never reference a record the estate does not have.
    pub(crate) fn commit(mut self) -> Result<()> {
        if self.counters_dirty {
            let meta = self.db.cf(CF_META)?;
            self.batch
                .put_cf(meta, META_DOC_COUNT, self.doc_count.to_le_bytes());
            self.batch
                .put_cf(meta, META_TOTAL_TOKENS, self.total_tokens.to_le_bytes());
            self.batch.put_cf(
                meta,
                META_SHAPES,
                serde_json::to_vec(&self.shapes).map_err(|e| RroError::Recall(e.to_string()))?,
            );
            self.batch
                .put_cf(meta, META_FEED_SEQ, self.feed_seq.to_le_bytes());
        }
        self.db.write(self.batch)?;
        for op in self.graph_ops.drain(..) {
            match op {
                GraphOp::Upsert(id, emb) => self.pending.push_upsert(id, emb),
                GraphOp::Remove(id) => self.pending.push_remove(id),
            }
        }
        Ok(())
    }

    // Rollback needs no method: a transaction that is dropped without `commit`
    // discards its batch unwritten and its graph ops with it. Nothing durable
    // landed, so the graph — which derives from the durable vectors — never sees
    // any of it. Any write op that returns `Err` therefore rolls the whole
    // transaction back for free, just by unwinding past the `commit` call.
}
