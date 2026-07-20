//! # rrd — the reason-ready object JIT
//!
//! Compiles arbitrary payloads into **Reason-Ready Objects** the way a JIT
//! compiles dynamic code: payloads are grouped by **shape** (their hidden
//! class), a distillation **plan** is compiled once per shape and cached (the
//! inline cache), and every payload of a seen shape rides the compiled path.
//!
//! Classification is a cascade (see ADR-0002):
//! - **modes & slivers — structural** (field names/types), zero-model;
//! - **tags — semantic-router** over the engine's own embedding space,
//!   reusing the embedding recall already computed (zero marginal cost);
//! - **zero-shot/NLI — compile-time only** (the seam is `plan::compile`),
//!   never per document.
//!
//! Session semantics ([`SessionTrigger`]): RRD fires on conversation start
//! and on idle-resume — the re-orientation moments — routing fresh context
//! to intent (mode) and evolving the session's tags.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod baseline;
pub mod gates;
pub mod mode;
pub mod plan;
pub mod registry;
pub mod router;
pub mod rro;
pub mod shape;
pub mod trigger;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use rro_core::{Embedding, Metadata};

pub use baseline::{
    BaselineSnapshot, ShapeBaseline, Speculation, DRIFT_THRESHOLD, SPECULATION_CONFIDENCE,
};
pub use gates::{ActionGate, DeepEvaluator, GateVerdict, L0Config, LexicalSignals, SourceStamp};
pub use mode::Mode;
pub use plan::{FieldRole, Plan};
pub use registry::{ShapeRegistry, Sliver};
pub use router::{Route, RoutedTag, SemanticRouter};
pub use rro::{ReadinessHints, Rro};
pub use shape::ShapeFingerprint;
pub use trigger::{FireReason, SessionEvent, SessionTrigger};

/// The JIT: shape registry + plan cache + tag router, distilling payloads
/// into [`Rro`]s.
pub struct Rrd {
    registry: RwLock<ShapeRegistry>,
    plans: RwLock<HashMap<u64, Arc<Plan>>>,
    router: SemanticRouter,
    l0: gates::L0Config,
    baseline: RwLock<ShapeBaseline>,
    hits: AtomicU64,
    compiles: AtomicU64,
}

impl Rrd {
    /// A JIT with no tag routes (tags simply don't fire).
    pub fn new() -> Self {
        Self::with_router(SemanticRouter::new())
    }

    /// A JIT with a configured tag router.
    pub fn with_router(router: SemanticRouter) -> Self {
        Rrd {
            registry: RwLock::new(ShapeRegistry::new()),
            plans: RwLock::new(HashMap::new()),
            router,
            l0: gates::L0Config::default(),
            baseline: RwLock::new(ShapeBaseline::new()),
            hits: AtomicU64::new(0),
            compiles: AtomicU64::new(0),
        }
    }

    /// Route tags for an embedding directly (the post-embed half when the
    /// caller runs the RRD-first split: gate+shape before embedding, tags
    /// after). Same router, same space.
    pub fn route_tags(&self, embedding: &Embedding) -> Vec<RoutedTag> {
        self.router.route(embedding)
    }

    /// Commit and return a baseline snapshot for persistence (the estate
    /// stores it; restarts restore it — the baseline grows across sessions).
    pub fn baseline_snapshot(&self) -> BaselineSnapshot {
        let snap = self.baseline.write().expect("baseline lock").snapshot();
        rro_core::events::emit(
            "rrd.baseline.snapshot",
            serde_json::json!({
                "version": snap.version,
                "observations": snap.observations,
                "contexts": snap.contexts.len(),
            }),
        );
        snap
    }

    /// Restore a persisted baseline snapshot (live state + drift reference).
    pub fn restore_baseline(&self, snapshot: BaselineSnapshot) {
        *self.baseline.write().expect("baseline lock") = ShapeBaseline::restore(snapshot);
    }

    /// Predictability of a context: `1 − normalized entropy` of its shape
    /// distribution (1.0 = monomorphic, → 0.0 = megamorphic).
    pub fn predictability(&self, context: &str) -> f64 {
        self.baseline
            .read()
            .expect("baseline lock")
            .predictability(context)
    }

    /// What this context is *about to be*: the most likely next sliver and the
    /// share of the context's recency-weighted mass it holds.
    ///
    /// This is the inline cache's speculation, read out — "when something hits
    /// this context, 99% of the time it is shape X". `Some((sliver, 0.99))` says
    /// exactly that; `None` means the context has never been seen.
    ///
    /// It costs a hashmap lookup and an argmax, and it answers **before the
    /// embedder runs**. That is the whole point of shape as early intent: the
    /// engine can know what is probably coming while the query is still text.
    ///
    /// The number is only worth what the shapes are worth. A context whose
    /// payloads all carry empty metadata has exactly one sliver, so this returns
    /// it with confidence 1.0 — perfectly predictable and perfectly useless. See
    /// [`ShapeFingerprint`]: shape is fingerprinted from *fields*, so feed it
    /// fields.
    pub fn predict(&self, context: &str) -> Option<(u64, f64)> {
        self.baseline
            .read()
            .expect("baseline lock")
            .predict(context)
    }

    /// Everything the baseline believes about a context, and whether that belief
    /// is strong enough to act on — see [`Speculation::actionable`].
    ///
    /// This is the call for "track now, enable at 97%": read it on every pass,
    /// act on it only when it says so. It is a pre-model read — a hashmap lookup
    /// and an argmax — so watching costs nothing whether or not you ever
    /// speculate.
    pub fn speculation(&self, context: &str) -> Speculation {
        let b = self.baseline.read().expect("baseline lock");
        let predicted = b.predict(context);
        Speculation {
            sliver: predicted.map(|(s, _)| s),
            confidence: predicted.map(|(_, c)| c).unwrap_or(0.0),
            predictability: b.predictability(context),
            observations: b.context_observations(context),
            drift: b.drift(context),
        }
    }

    /// Speculative-prediction hit-rate for a context — the measured
    /// "predictability is improving" curve.
    pub fn hit_rate(&self, context: &str) -> f64 {
        self.baseline
            .read()
            .expect("baseline lock")
            .hit_rate(context)
    }

    /// PSI drift of a context's recent window vs the committed snapshot.
    pub fn drift(&self, context: &str) -> f64 {
        self.baseline.read().expect("baseline lock").drift(context)
    }

    /// Lifetime observations folded into the baseline (0 = fresh instance).
    pub fn baseline_observations(&self) -> u64 {
        self.baseline.read().expect("baseline lock").observations()
    }

    /// Distill one payload into an RRO with default (empty) source stamp and
    /// default L0 limits.
    pub fn distill(
        &self,
        doc_id: &str,
        text: &str,
        metadata: &Metadata,
        embedding: Option<&Embedding>,
    ) -> Rro {
        self.distill_stamped(doc_id, text, metadata, embedding, SourceStamp::default())
    }

    /// Distill one payload through the full gate ladder (ADR-0002):
    /// stamp → L0 deterministic → L1 lexical → structural (shape/mode/plan)
    /// → L2 semantic routing. A `Block` at L0 short-circuits (the RRO is
    /// returned un-distilled, carrying the verdict). L3 runs at action time
    /// via [`ActionGate`]; L4 runs concurrently via [`DeepEvaluator`].
    ///
    /// `embedding` is the document vector recall already computed at ingest —
    /// pass it in and tag routing costs only dot products. Without it, tags
    /// don't fire (structural distillation still runs fully).
    pub fn distill_stamped(
        &self,
        doc_id: &str,
        text: &str,
        metadata: &Metadata,
        embedding: Option<&Embedding>,
        stamp: SourceStamp,
    ) -> Rro {
        // L0 — deterministic gate: size/schema arithmetic, no scanning.
        let l0 = gates::l0_deterministic(&self.l0, text, metadata.len(), false);
        if l0 == GateVerdict::Block {
            rro_core::events::emit(
                "rrd.gate",
                serde_json::json!({ "tier": "l0", "verdict": "block", "doc": doc_id }),
            );
            return Rro {
                doc_id: doc_id.to_string(),
                sliver_id: u64::MAX,
                mode: Mode::Unshaped,
                fields: Metadata::new(),
                tags: Vec::new(),
                hints: ReadinessHints::default(),
                plan_version: 0,
                stamp,
                gate: GateVerdict::Block,
                signals: LexicalSignals::default(),
            };
        }

        // L1 — lexical signals: one scan, flags only (blocking on content is
        // an authorization decision, i.e. L3/L4).
        let signals = gates::l1_lexical(text);
        let gate = match (l0, signals.verdict()) {
            (GateVerdict::Flag, _) | (_, GateVerdict::Flag) => GateVerdict::Flag,
            _ => GateVerdict::Pass,
        };

        // Structural + L2 follow.
        // The baseline context: where this payload came from.
        let context = stamp
            .channel
            .clone()
            .or_else(|| stamp.source.clone())
            .or_else(|| stamp.project.clone())
            .unwrap_or_else(|| "global".to_string());

        // Speculate BEFORE identifying: the inline-cache prediction.
        let predicted = self
            .baseline
            .read()
            .expect("baseline lock")
            .predict(&context)
            .map(|(s, _)| s);

        // 1. Shape (the hidden class) + mode: structural, model-free.
        let shape = ShapeFingerprint::of(metadata);
        let mode = mode::identify(metadata);
        let (sliver_id, is_new) = self
            .registry
            .write()
            .expect("registry lock")
            .observe(&shape, mode);

        // Settle the speculation ledger and watch for drift. The baseline is
        // how RRD evolves: every observation sharpens the next prediction.
        {
            let mut baseline = self.baseline.write().expect("baseline lock");
            baseline.observe(&context, sliver_id, predicted);
            let drift = baseline.drift(&context);
            if drift > baseline::DRIFT_THRESHOLD {
                rro_core::events::emit(
                    "rrd.drift",
                    serde_json::json!({
                        "context": context,
                        "psi": drift,
                        "predictability": baseline.predictability(&context),
                    }),
                );
            }
        }

        // 2. Inline cache: compile once per sliver, reuse forever after.
        let plan = if is_new {
            let compiled = Arc::new(plan::compile(sliver_id, mode, &shape));
            self.plans
                .write()
                .expect("plans lock")
                .insert(sliver_id, compiled.clone());
            self.compiles.fetch_add(1, Ordering::Relaxed);
            rro_core::events::emit(
                "rrd.compile",
                serde_json::json!({
                    "sliver_id": sliver_id,
                    "mode": mode.name(),
                    "fields": shape.0.len(),
                }),
            );
            compiled
        } else {
            self.hits.fetch_add(1, Ordering::Relaxed);
            let cached = self
                .plans
                .read()
                .expect("plans lock")
                .get(&sliver_id)
                .cloned();
            match cached {
                Some(p) => p,
                // Registry knew the shape but the plan is missing (should not
                // happen); recompile rather than fail.
                None => {
                    let compiled = Arc::new(plan::compile(sliver_id, mode, &shape));
                    self.plans
                        .write()
                        .expect("plans lock")
                        .insert(sliver_id, compiled.clone());
                    compiled
                }
            }
        };

        // 3. Execute the plan: group fields by role.
        let mut by_role: Metadata = Metadata::new();
        let mut hints = ReadinessHints {
            has_content: !text.trim().is_empty(),
            ..ReadinessHints::default()
        };
        for (field, value) in metadata {
            let role = plan.roles.get(field).copied().unwrap_or(FieldRole::Other);
            let role_key = match role {
                FieldRole::Identity => "identity",
                FieldRole::Time => "time",
                FieldRole::Title => "title",
                FieldRole::Content => "content",
                FieldRole::Salience => "salience",
                FieldRole::Other => "other",
            };
            let nonempty = !value.is_null();
            match role {
                FieldRole::Identity if nonempty => hints.has_identity = true,
                FieldRole::Time if nonempty => hints.has_time = true,
                FieldRole::Title if nonempty => hints.has_title = true,
                FieldRole::Content if nonempty => hints.has_content = true,
                _ => {}
            }
            let slot = by_role
                .entry(role_key.to_string())
                .or_insert_with(|| serde_json::json!({}));
            if let Some(obj) = slot.as_object_mut() {
                obj.insert(field.clone(), value.clone());
            }
        }

        // 4. L2 — semantic routing on the already-computed embedding.
        let tags = match embedding {
            Some(e) => self.router.route(e),
            None => Vec::new(),
        };
        hints.tag_confidence = tags.first().map(|t| t.score).unwrap_or(0.0);
        let scores: Vec<f32> = tags.iter().map(|t| t.score).collect();
        hints.ambiguity_margin = gates::l2_ambiguity(&scores);

        Rro {
            doc_id: doc_id.to_string(),
            sliver_id,
            mode,
            fields: by_role,
            tags,
            hints,
            plan_version: plan.version,
            stamp,
            gate,
            signals,
        }
    }

    /// (cache hits, compiles) — JIT warm-up telemetry.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.compiles.load(Ordering::Relaxed),
        )
    }

    /// Distinct slivers observed.
    pub fn sliver_count(&self) -> usize {
        self.registry.read().expect("registry lock").len()
    }
}

impl Default for Rrd {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mail_meta(i: usize) -> Metadata {
        [
            ("from", serde_json::json!(format!("user{i}@x"))),
            ("subject", serde_json::json!("standup notes")),
            ("body", serde_json::json!("we shipped the estate")),
            ("sent_at", serde_json::json!(1_700_000_000 + i)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    #[test]
    fn compiles_once_per_shape_then_hits() {
        let rrd = Rrd::new();
        for i in 0..100 {
            let rro = rrd.distill(&format!("m{i}"), "text", &mail_meta(i), None);
            assert_eq!(rro.mode, Mode::Mail);
            assert!(rro.hints.has_identity && rro.hints.has_time && rro.hints.has_title);
        }
        let (hits, compiles) = rrd.stats();
        assert_eq!(compiles, 1, "one shape ⇒ exactly one compile");
        assert_eq!(hits, 99);
        assert_eq!(rrd.sliver_count(), 1);
    }

    #[test]
    fn distill_is_deterministic_and_drift_recompiles() {
        let rrd = Rrd::new();
        let a = rrd.distill("d", "t", &mail_meta(1), None);
        let b = rrd.distill("d", "t", &mail_meta(1), None);
        assert_eq!(
            serde_json::to_string(&a.fields).unwrap(),
            serde_json::to_string(&b.fields).unwrap()
        );
        assert_eq!(a.sliver_id, b.sliver_id);

        // Drift: extra field ⇒ new sliver ⇒ second compile.
        let mut drifted = mail_meta(1);
        drifted.insert("thread_id".into(), serde_json::json!("t-9"));
        let c = rrd.distill("d2", "t", &drifted, None);
        assert_ne!(c.sliver_id, a.sliver_id);
        assert_eq!(rrd.stats().1, 2);
    }

    #[test]
    fn tags_route_on_provided_embedding() {
        let mut router = SemanticRouter::new();
        router.add_route("ops", &[Embedding(vec![1.0, 0.0, 0.0]).normalized()], 0.5);
        let rrd = Rrd::with_router(router);
        let e = Embedding(vec![0.9, 0.1, 0.0]).normalized();
        let rro = rrd.distill("d", "deploy failed", &Metadata::new(), Some(&e));
        assert_eq!(rro.tags[0].tag, "ops");
        assert!(rro.hints.tag_confidence > 0.5);
        assert_eq!(rro.mode, Mode::Unshaped);
    }
}
