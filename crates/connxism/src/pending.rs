//! Out-of-band graph apply: the two-phase pattern's second phase.
//!
//! Upserts commit durably and **enqueue**; a dedicated applier thread drains
//! the queue into the ANN graph. Ingest is never blocked by graph
//! construction (until the backpressure cap), and searches stay correct via
//! the **pending overlay**: not-yet-applied vectors are scored exactly and
//! merged over the graph's results, and pending removals mask stale graph
//! hits — read-your-writes by construction. Crash-safe trivially: pendings
//! are already durable in the `vecs` column family, and reopening an estate
//! rebuilds the graph from it.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, RwLock as StdRwLock};

use recall::AnnIndex;
use rro_core::{Embedding, Id};

/// Backpressure cap: above this many queued entries, producers block.
const PENDING_CAP: usize = 200_000;
/// Applier batch size per graph write-lock acquisition.
const APPLY_BATCH: usize = 512;

#[derive(Default)]
struct State {
    /// Apply order (ids may repeat; the map holds the latest op).
    queue: VecDeque<Id>,
    /// Latest pending op per id: `Some` = upsert, `None` = remove.
    latest: HashMap<Id, Option<Embedding>>,
    /// The batch the applier is currently writing into the graph — moved out of
    /// `latest` when collected, and kept **visible to the overlay** until the
    /// graph apply completes. Without this, an op is momentarily in neither
    /// `latest` (removed on collect) nor the graph (not yet inserted), and a
    /// concurrent search loses read-your-writes. Cleared only after the apply.
    inflight: HashMap<Id, Option<Embedding>>,
    /// The applier is mid-batch (queue may be empty while work is in flight).
    applying: bool,
    /// Shutdown flag.
    stopped: bool,
}

/// Shared pending set + signaling.
pub(crate) struct Pending {
    state: Mutex<State>,
    /// Wakes the applier (work arrived / shutdown).
    work: Condvar,
    /// Wakes waiters (space available / drained).
    settled: Condvar,
}

impl Pending {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Pending {
            state: Mutex::new(State::default()),
            work: Condvar::new(),
            settled: Condvar::new(),
        })
    }

    /// Enqueue an upsert (blocks at the backpressure cap).
    pub(crate) fn push_upsert(&self, id: Id, embedding: Embedding) {
        let mut s = self.state.lock().expect("pending lock");
        while s.queue.len() >= PENDING_CAP && !s.stopped {
            s = self.settled.wait(s).expect("pending wait");
        }
        s.queue.push_back(id.clone());
        s.latest.insert(id, Some(embedding));
        drop(s);
        self.work.notify_one();
    }

    /// Enqueue a removal.
    pub(crate) fn push_remove(&self, id: Id) {
        let mut s = self.state.lock().expect("pending lock");
        s.queue.push_back(id.clone());
        s.latest.insert(id, None);
        drop(s);
        self.work.notify_one();
    }

    /// Snapshot the overlay for a search: pending upserts scored exactly by
    /// the caller, pending removals masked. Cheap when drained (the steady
    /// state); proportional to backlog when not.
    ///
    /// Reads both `latest` and the `inflight` batch so a write is visible for the
    /// whole handoff into the graph. `latest` wins over `inflight` for an id in
    /// both (it is the newer op), so a write that arrives while its predecessor is
    /// being applied still takes precedence.
    pub(crate) fn overlay(&self, query: &Embedding) -> (Vec<(Id, f32)>, Vec<Id>) {
        let s = self.state.lock().expect("pending lock");
        let q = query.normalized();
        let mut ups = Vec::new();
        let mut dels = Vec::new();
        let mut push = |id: &Id, op: &Option<Embedding>| match op {
            Some(emb) => ups.push((id.clone(), q.cosine(&emb.normalized()))),
            None => dels.push(id.clone()),
        };
        for (id, op) in &s.latest {
            push(id, op);
        }
        for (id, op) in &s.inflight {
            if !s.latest.contains_key(id) {
                push(id, op);
            }
        }
        (ups, dels)
    }

    /// How many graph ops are queued (plus one mid-apply, if any).
    pub(crate) fn backlog(&self) -> usize {
        let s = self.state.lock().expect("pending lock");
        s.queue.len() + usize::from(s.applying)
    }

    /// Block until every queued op has been applied to the graph — queue empty,
    /// nothing in flight, and no batch mid-apply.
    pub(crate) fn quiesce(&self) {
        let mut s = self.state.lock().expect("pending lock");
        while (!s.queue.is_empty() || s.applying || !s.inflight.is_empty()) && !s.stopped {
            s = self.settled.wait(s).expect("pending wait");
        }
    }

    /// Collect the next batch to apply: move up to [`APPLY_BATCH`] ops from
    /// `latest` into `inflight` (keeping them overlay-visible) and mark the
    /// applier busy. Blocks until work arrives; returns `None` on shutdown.
    fn collect_batch(&self) -> Option<Vec<(Id, Option<Embedding>)>> {
        let mut s = self.state.lock().expect("pending lock");
        while s.queue.is_empty() && !s.stopped {
            s = self.work.wait(s).expect("pending wait");
        }
        if s.stopped {
            return None;
        }
        let mut batch = Vec::with_capacity(APPLY_BATCH);
        while batch.len() < APPLY_BATCH {
            let Some(id) = s.queue.pop_front() else { break };
            // Duplicate queue entries: only the first pop finds the op in
            // `latest`; later pops skip. The op moves to `inflight` (a clone stays
            // visible to the overlay; the original goes into the batch to apply).
            if let Some(op) = s.latest.remove(&id) {
                s.inflight.insert(id.clone(), op.clone());
                batch.push((id, op));
            }
        }
        s.applying = true;
        Some(batch)
    }

    /// Mark the applied batch done: clear `inflight` (its ops are now in the
    /// graph) and wake quiesce waiters. Called **after** the graph write lock is
    /// released, so any search that saw the pre-apply graph still saw `inflight`.
    fn finish_batch(&self) {
        let mut s = self.state.lock().expect("pending lock");
        s.inflight.clear();
        s.applying = false;
        drop(s);
        self.settled.notify_all();
    }

    /// Signal shutdown and wake everyone.
    pub(crate) fn stop(&self) {
        let mut s = self.state.lock().expect("pending lock");
        s.stopped = true;
        drop(s);
        self.work.notify_all();
        self.settled.notify_all();
    }

    /// Spawn the applier thread for `ann`. Runs until [`Pending::stop`].
    pub(crate) fn spawn_applier(
        self: &Arc<Self>,
        ann: Arc<StdRwLock<AnnIndex>>,
    ) -> std::thread::JoinHandle<()> {
        let pending = Arc::clone(self);
        std::thread::Builder::new()
            .name("rro-ann-applier".into())
            .spawn(move || {
                while let Some(batch) = pending.collect_batch() {
                    // Apply outside the pending lock (graph lock only) — holding
                    // both would deadlock with `overlay`, which takes the pending
                    // lock while a search holds the graph read lock. `inflight`
                    // keeps the batch visible to the overlay for the whole window.
                    {
                        let mut graph = ann.write().expect("ann lock");
                        for (id, op) in batch {
                            match op {
                                Some(emb) => graph.insert(id, &emb),
                                None => graph.remove(&id),
                            }
                        }
                    }
                    // Only now clear inflight — after the graph write lock is
                    // released, so a search never sees the op in neither place.
                    pending.finish_batch();
                }
            })
            .expect("spawn applier")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use recall::AnnConfig;

    #[test]
    fn applier_drains_and_quiesce_waits() {
        let ann = Arc::new(StdRwLock::new(AnnIndex::new(AnnConfig::default())));
        let pending = Pending::new();
        let handle = pending.spawn_applier(ann.clone());

        for i in 0..1000 {
            pending.push_upsert(
                Id::new(format!("p{i}")),
                Embedding(vec![i as f32, 1.0, 0.5]),
            );
        }
        pending.quiesce();
        assert_eq!(ann.read().unwrap().len(), 1000);

        // Overlay is empty once drained.
        let (ups, dels) = pending.overlay(&Embedding(vec![1.0, 0.0, 0.0]));
        assert!(ups.is_empty() && dels.is_empty());

        pending.stop();
        handle.join().unwrap();
    }

    #[test]
    fn overlay_sees_unapplied_upserts_and_removes() {
        // No applier: everything stays pending.
        let pending = Pending::new();
        pending.push_upsert(Id::new("a"), Embedding(vec![1.0, 0.0]));
        pending.push_remove(Id::new("b"));

        let (ups, dels) = pending.overlay(&Embedding(vec![1.0, 0.0]));
        assert_eq!(ups.len(), 1);
        assert_eq!(ups[0].0.as_str(), "a");
        assert!(ups[0].1 > 0.99);
        assert_eq!(dels, vec![Id::new("b")]);
        pending.stop();
    }

    /// The race, deterministically: after a batch is collected (moved out of
    /// `latest`) but before it is applied to the graph, the op must STILL be
    /// visible via the overlay. This is exactly the window where a concurrent
    /// search previously lost read-your-writes. No applier thread runs — we drive
    /// the handoff by hand to sit in the window.
    #[test]
    fn overlay_stays_visible_across_the_collect_apply_window() {
        let pending = Pending::new();
        pending.push_upsert(Id::new("fresh"), Embedding(vec![0.0, 1.0]));

        // Collect the batch — this moves "fresh" out of `latest` and into
        // `inflight`, the exact point the old code went invisible.
        let batch = pending.collect_batch().expect("a batch");
        assert_eq!(batch.len(), 1);

        // WINDOW: not yet applied to any graph. Overlay must still see it.
        let (ups, _) = pending.overlay(&Embedding(vec![0.0, 1.0]));
        assert!(
            ups.iter().any(|(id, _)| id.as_str() == "fresh"),
            "a collected-but-not-applied op must stay visible in the overlay"
        );

        // Finishing the batch clears it (the graph would now hold it).
        pending.finish_batch();
        let (ups, _) = pending.overlay(&Embedding(vec![0.0, 1.0]));
        assert!(ups.is_empty(), "overlay is empty once the batch is applied");
        pending.stop();
    }

    /// A newer write that arrives while its predecessor is in flight must win:
    /// `latest` takes precedence over `inflight` for the same id.
    #[test]
    fn newer_write_overrides_an_inflight_predecessor() {
        let pending = Pending::new();
        pending.push_upsert(Id::new("x"), Embedding(vec![1.0, 0.0]));
        let _batch = pending.collect_batch().expect("a batch"); // x → inflight

        // A newer op for x lands in `latest` while the old one is in flight.
        pending.push_upsert(Id::new("x"), Embedding(vec![0.0, 1.0]));

        // Overlay reports x exactly once, scored by the NEWER vector.
        let (ups, _) = pending.overlay(&Embedding(vec![0.0, 1.0]));
        let xs: Vec<_> = ups.iter().filter(|(id, _)| id.as_str() == "x").collect();
        assert_eq!(xs.len(), 1, "x must appear once, not double-counted");
        assert!(xs[0].1 > 0.99, "the newer vector must win the score");
        pending.stop();
    }
}
