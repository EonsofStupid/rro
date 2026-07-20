//! The shape baseline: a snapshot of normal, evolving forever.
//!
//! Every distill observes `(context, sliver)` — the context being where the
//! payload came from (connector, channel, session). The baseline keeps a
//! **recency-weighted distribution** of slivers per context and globally,
//! and from it derives the three numbers that make RRD predictive instead of
//! merely reactive:
//!
//! - **prediction** — the expected next sliver for a context (an inline
//!   cache with speculation, V8-style). Hit-rate is tracked per context;
//!   the climbing curve *is* "improving predictability", measured.
//! - **predictability** — `1 − normalized entropy` of the context's shape
//!   distribution: 1.0 = monomorphic (one shape, fully predictable),
//!   → 0.0 = megamorphic (anything goes). The compiler-tiering ladder,
//!   applied to data.
//! - **drift** — PSI (population stability index) between the context's
//!   recent window and the last committed **snapshot** — the ML-observability
//!   reference-profile pattern. High PSI = the world changed; the baseline
//!   decays toward the new regime while the drift signal fires.
//!
//! Decay is O(1) via the growing-unit trick: instead of multiplying every
//! weight by λ per observation, the *increment unit* grows by 1/λ, so newer
//! observations simply weigh more; everything rescales when the unit gets
//! large. Snapshots are serializable and persist in the estate — the
//! baseline survives restarts and **grows across sessions**.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

/// Decay factor per observation (effective memory ≈ 1/(1−λ) ≈ 1000 obs).
const LAMBDA: f64 = 0.999;
/// Rescale threshold for the growing unit.
const RESCALE_AT: f64 = 1e12;
/// Recent-window length per context (drift comparison sample).
const WINDOW: usize = 128;
/// PSI above this is reported as drift.
pub const DRIFT_THRESHOLD: f64 = 0.25;

/// Confidence a context's prediction must reach before speculation is
/// **actionable** rather than merely observed.
///
/// The discipline this encodes: **track always, act rarely.** Observing a shape
/// costs a hashmap bump, so RRD watches everything from the first payload — but
/// a prediction is only worth acting on once the context has proven it is boring.
/// V8 does not optimise on one type observation, and neither should this.
///
/// 0.97 is deliberately high. The cost of a *wrong* speculation is a mis-routed
/// intent — a worse answer, arrived at faster, which is the worst trade the
/// engine can make. The cost of *not* speculating is that a query takes its
/// normal path. Those are not symmetric, so the bar is not 0.5.
///
/// It is a **ceiling on eagerness, not a promise**: a context can sit below it
/// forever, and that is a fact about the context (megamorphic input), not a
/// failure. Nothing is forced by reaching it either — see [`Speculation`], which
/// reports and lets the caller decide.
pub const SPECULATION_CONFIDENCE: f64 = 0.97;

/// Recency-weighted sliver distribution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Distribution {
    /// sliver id → decayed weight.
    weights: HashMap<u64, f64>,
    /// Sum of weights.
    total: f64,
}

impl Distribution {
    fn add(&mut self, sliver: u64, unit: f64) {
        *self.weights.entry(sliver).or_insert(0.0) += unit;
        self.total += unit;
    }

    fn rescale(&mut self, by: f64) {
        for w in self.weights.values_mut() {
            *w /= by;
        }
        self.total /= by;
    }

    /// Probability of a sliver under this distribution.
    pub fn p(&self, sliver: u64) -> f64 {
        if self.total <= 0.0 {
            0.0
        } else {
            self.weights.get(&sliver).copied().unwrap_or(0.0) / self.total
        }
    }

    /// Highest-probability sliver with its probability.
    pub fn argmax(&self) -> Option<(u64, f64)> {
        self.weights
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(&s, &w)| {
                (
                    s,
                    if self.total > 0.0 {
                        w / self.total
                    } else {
                        0.0
                    },
                )
            })
    }

    /// `1 − H/H_max`: 1.0 = monomorphic, → 0.0 = megamorphic.
    pub fn predictability(&self) -> f64 {
        let k = self.weights.len();
        if k <= 1 || self.total <= 0.0 {
            return 1.0;
        }
        let mut h = 0.0;
        for &w in self.weights.values() {
            let p = w / self.total;
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        1.0 - h / (k as f64).ln()
    }
}

/// Per-context state: distribution + recent window + speculation ledger.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextStats {
    /// Payloads observed in this context, ever.
    ///
    /// Not derivable from what was already here: `dist.total` is a sum of
    /// *decayed* weights, and `predictions` only counts passes that asked for a
    /// prediction. Speculation needs a raw count — 0.99 from two samples is not
    /// 0.99, it is two samples.
    pub observations: u64,
    /// The context's recency-weighted shape distribution.
    pub dist: Distribution,
    /// Last `WINDOW` observed slivers (drift sample).
    window: VecDeque<u64>,
    /// Speculative predictions issued.
    pub predictions: u64,
    /// Predictions that matched the observed sliver.
    pub hits: u64,
}

impl ContextStats {
    /// Prediction hit-rate so far (0 when nothing predicted).
    pub fn hit_rate(&self) -> f64 {
        if self.predictions == 0 {
            0.0
        } else {
            self.hits as f64 / self.predictions as f64
        }
    }
}

/// A committed, serializable snapshot: the reference profile drift is
/// measured against, and the unit of cross-session persistence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaselineSnapshot {
    /// Monotonic snapshot version.
    pub version: u64,
    /// Global shape distribution at snapshot time.
    pub global: Distribution,
    /// Per-context distributions at snapshot time.
    pub contexts: HashMap<String, Distribution>,
    /// Total observations ever folded in.
    pub observations: u64,
}

/// The living baseline.
#[derive(Debug, Default)]
pub struct ShapeBaseline {
    unit: f64,
    global: Distribution,
    contexts: HashMap<String, ContextStats>,
    observations: u64,
    reference: BaselineSnapshot,
}

impl ShapeBaseline {
    /// A fresh baseline (unit initialized on first observation).
    pub fn new() -> Self {
        ShapeBaseline {
            unit: 1.0,
            ..ShapeBaseline::default()
        }
    }

    /// Speculate the next sliver for `context` (id + confidence). The caller
    /// verifies against the observed fingerprint; [`ShapeBaseline::observe`]
    /// settles the ledger.
    pub fn predict(&self, context: &str) -> Option<(u64, f64)> {
        self.contexts.get(context)?.dist.argmax()
    }

    /// Fold one observation in; returns whether the pre-issued prediction
    /// (if any) hit. Call once per distill, after shape identification.
    pub fn observe(&mut self, context: &str, sliver: u64, predicted: Option<u64>) -> bool {
        if self.unit == 0.0 {
            self.unit = 1.0;
        }
        self.observations += 1;

        let unit = self.unit;
        self.global.add(sliver, unit);
        let ctx = self.contexts.entry(context.to_string()).or_default();
        ctx.observations += 1;
        ctx.dist.add(sliver, unit);
        ctx.window.push_back(sliver);
        if ctx.window.len() > WINDOW {
            ctx.window.pop_front();
        }

        let hit = match predicted {
            Some(p) => {
                ctx.predictions += 1;
                if p == sliver {
                    ctx.hits += 1;
                    true
                } else {
                    false
                }
            }
            None => false,
        };

        // O(1) decay: newer observations weigh more; rescale when large.
        self.unit /= LAMBDA;
        if self.unit > RESCALE_AT {
            let by = self.unit;
            self.global.rescale(by);
            for c in self.contexts.values_mut() {
                c.dist.rescale(by);
            }
            self.unit = 1.0;
        }
        hit
    }

    /// `1 − normalized entropy` for a context (1.0 when unseen — an empty
    /// context is trivially predictable until proven otherwise).
    pub fn predictability(&self, context: &str) -> f64 {
        self.contexts
            .get(context)
            .map(|c| c.dist.predictability())
            .unwrap_or(1.0)
    }

    /// Payloads observed in a context (0 if never seen).
    pub fn context_observations(&self, context: &str) -> u64 {
        self.contexts
            .get(context)
            .map(|c| c.observations)
            .unwrap_or(0)
    }

    /// Prediction hit-rate for a context.
    pub fn hit_rate(&self, context: &str) -> f64 {
        self.contexts
            .get(context)
            .map(|c| c.hit_rate())
            .unwrap_or(0.0)
    }

    /// PSI between the context's recent window and the committed snapshot's
    /// distribution for that context. > [`DRIFT_THRESHOLD`] = drift.
    pub fn drift(&self, context: &str) -> f64 {
        let Some(ctx) = self.contexts.get(context) else {
            return 0.0;
        };
        let Some(reference) = self.reference.contexts.get(context) else {
            return 0.0; // no committed reference yet: nothing to drift from
        };
        if ctx.window.is_empty() || reference.total <= 0.0 {
            return 0.0;
        }

        // Current window as a distribution.
        let mut current: HashMap<u64, f64> = HashMap::new();
        for &s in &ctx.window {
            *current.entry(s).or_insert(0.0) += 1.0;
        }
        let n = ctx.window.len() as f64;

        // PSI over the union of slivers, epsilon-smoothed.
        const EPS: f64 = 1e-4;
        let mut keys: Vec<u64> = current.keys().copied().collect();
        for k in reference.weights.keys() {
            if !current.contains_key(k) {
                keys.push(*k);
            }
        }
        let mut psi = 0.0;
        for k in keys {
            let p = (current.get(&k).copied().unwrap_or(0.0) / n).max(EPS);
            let q = reference.p(k).max(EPS);
            psi += (p - q) * (p / q).ln();
        }
        psi
    }

    /// Commit the current state as the new reference snapshot (and return it
    /// for persistence). Drift measures against this from now on.
    pub fn snapshot(&mut self) -> BaselineSnapshot {
        self.reference = BaselineSnapshot {
            version: self.reference.version + 1,
            global: self.global.clone(),
            contexts: self
                .contexts
                .iter()
                .map(|(k, v)| (k.clone(), v.dist.clone()))
                .collect(),
            observations: self.observations,
        };
        self.reference.clone()
    }

    /// Restore a persisted snapshot as both the live state and the reference
    /// — how the baseline survives restarts and grows across sessions.
    pub fn restore(snapshot: BaselineSnapshot) -> Self {
        let contexts = snapshot
            .contexts
            .iter()
            .map(|(k, dist)| {
                (
                    k.clone(),
                    ContextStats {
                        dist: dist.clone(),
                        ..ContextStats::default()
                    },
                )
            })
            .collect();
        ShapeBaseline {
            unit: 1.0,
            global: snapshot.global.clone(),
            contexts,
            observations: snapshot.observations,
            reference: snapshot,
        }
    }

    /// Total observations folded in (lifetime, across restores).
    pub fn observations(&self) -> u64 {
        self.observations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_rate_climbs_on_a_stable_stream() {
        let mut b = ShapeBaseline::new();
        let mut hits_late = 0;
        for i in 0..500 {
            let predicted = b.predict("conn-a").map(|(s, _)| s);
            let hit = b.observe("conn-a", 7, predicted);
            if i >= 400 {
                hits_late += u32::from(hit);
            }
        }
        assert_eq!(
            hits_late, 100,
            "a monomorphic stream becomes fully predictable"
        );
        assert!(b.hit_rate("conn-a") > 0.95);
        assert!(b.predictability("conn-a") > 0.999);
    }

    #[test]
    fn predictability_separates_mono_from_megamorphic() {
        let mut b = ShapeBaseline::new();
        for i in 0..400u64 {
            b.observe("mono", 1, None);
            b.observe("mega", i % 8, None);
        }
        assert!(b.predictability("mono") > 0.99);
        assert!(
            b.predictability("mega") < 0.05,
            "uniform over 8 ≈ maximum entropy"
        );
    }

    #[test]
    fn drift_fires_on_regime_change_and_decay_adapts() {
        let mut b = ShapeBaseline::new();
        for _ in 0..300 {
            b.observe("feed", 1, None);
        }
        b.snapshot(); // commit "normal"
        assert!(
            b.drift("feed") < 0.05,
            "stable stream: no drift vs its own snapshot"
        );

        // Regime change: a new shape takes over. Drift fires quickly (the
        // window outpaces the decayed distribution by design)…
        for _ in 0..200 {
            b.observe("feed", 2, None);
        }
        assert!(
            b.drift("feed") > DRIFT_THRESHOLD,
            "PSI must fire after the regime change: {}",
            b.drift("feed")
        );
        // …while the prediction flips only once the new regime has genuinely
        // out-weighed the old one under the ~1000-obs decay memory. That lag
        // is intentional: drift alerts fast, identity changes slow.
        assert_eq!(b.predict("feed").map(|(s, _)| s), Some(1), "not yet");
        for _ in 0..600 {
            b.observe("feed", 2, None);
        }
        assert_eq!(b.predict("feed").map(|(s, _)| s), Some(2), "decay adapted");
    }

    #[test]
    fn snapshot_restore_roundtrip_preserves_prediction() {
        let mut b = ShapeBaseline::new();
        for _ in 0..100 {
            b.observe("c", 42, None);
        }
        let snap = b.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        let restored: BaselineSnapshot = serde_json::from_str(&json).unwrap();
        let b2 = ShapeBaseline::restore(restored);
        assert_eq!(b2.predict("c").map(|(s, _)| s), Some(42));
        assert_eq!(b2.observations(), 100);
    }
}

/// What the baseline believes about a context right now, and whether that belief
/// has earned the right to change anything.
///
/// This is the read-out of "when something hits this context, is it 99% of the
/// time X?" — with the answer separated from the decision. Nothing here forces a
/// route. Shape is **one** signal among several triggers; this reports its
/// strength and lets the caller weigh it.
///
/// The split matters because the two failure modes are asymmetric. Acting on a
/// weak prediction mis-routes intent — a worse answer, delivered faster. Not
/// acting on a strong one costs a normal query path. So the default is: watch
/// everything, speculate on almost nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Speculation {
    /// The shape the cache expects next, if the context has ever been seen.
    pub sliver: Option<u64>,
    /// Its share of the context's recency-weighted mass (0..=1).
    pub confidence: f64,
    /// `1 − normalized entropy` over the whole distribution: 1.0 monomorphic,
    /// → 0.0 megamorphic. `confidence` is about one shape; this is about the
    /// context's overall boringness.
    pub predictability: f64,
    /// Observations folded into this context.
    pub observations: u64,
    /// PSI vs the committed snapshot. Above [`DRIFT_THRESHOLD`] the world has
    /// moved and the baseline is describing a context that no longer exists.
    pub drift: f64,
}

impl Speculation {
    /// Whether this belief is strong enough to act on.
    ///
    /// Three conditions, all necessary:
    ///
    /// * **confidence ≥ [`SPECULATION_CONFIDENCE`]** — the shape actually dominates.
    /// * **enough observations** — 0.99 from two samples is not 0.99, it is two
    ///   samples. `MIN_OBSERVATIONS` is the warm-up every inline cache has.
    /// * **not drifting** — a high-confidence prediction from a stale baseline is
    ///   the most dangerous state available: maximally certain about a world that
    ///   has changed. Drift *disables* speculation rather than lowering it,
    ///   because a drifting context has not become less predictable; it has
    ///   become predictable about the wrong thing.
    pub fn actionable(&self) -> bool {
        self.sliver.is_some()
            && self.confidence >= SPECULATION_CONFIDENCE
            && self.observations >= Self::MIN_OBSERVATIONS
            && self.drift <= DRIFT_THRESHOLD
    }

    /// Warm-up before any confidence is believed, however lopsided.
    pub const MIN_OBSERVATIONS: u64 = 30;

    /// Why speculation is or is not enabled — for signals and for the operator,
    /// because "we did not speculate" is only useful if it says which condition
    /// failed.
    pub fn why(&self) -> &'static str {
        if self.sliver.is_none() {
            "unseen: no prediction for this context yet"
        } else if self.observations < Self::MIN_OBSERVATIONS {
            "warming: too few observations to believe any confidence"
        } else if self.drift > DRIFT_THRESHOLD {
            "drifting: the baseline describes a context that has changed"
        } else if self.confidence < SPECULATION_CONFIDENCE {
            "megamorphic: no single shape dominates enough to bet on"
        } else {
            "actionable: one shape dominates a stable, well-observed context"
        }
    }
}
