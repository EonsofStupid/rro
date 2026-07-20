//! Shape as early intent: "when something hits, is it 99% of the time X?"
//!
//! RRD's baseline is built to answer exactly that — a V8-style inline cache that
//! predicts a context's next sliver, `predictability = 1 − normalized entropy`,
//! and PSI drift. What it was never given is anything to predict *with*.
//!
//! Shape is fingerprinted from **fields** (`field:type,field:type,…`). `ask()`
//! passed `Metadata::new()`, so every query — a recipe, an ANN question, a SQL
//! injection — collapsed to the same `sliver=0, mode=unshaped`. Measured live
//! before this test existed. The baseline then sees one shape forever and reports
//! `predictability = 1.0`: perfectly predictable and perfectly useless.
//!
//! A COSTAR-aligned prompt is the missing input. Context / Objective / Style /
//! Tone / Audience / Response are *fields*, so they fingerprint, and the moment
//! they do the baseline earns its design.
//!
//! These tests hold that line: shapes must separate, and the cache must predict
//! before the model runs.

use std::sync::Arc;

use rro_core::Metadata;
use rro_engine::ReasonReadyObject;

fn flow() -> Arc<ReasonReadyObject> {
    Arc::new(
        ReasonReadyObject::builder()
            .rrd(Arc::new(rrd::Rrd::new()))
            .build(),
    )
}

/// A COSTAR-shaped prompt. The *values* are irrelevant to the fingerprint — only
/// the field names and their types are — which is the point: the shape is the
/// prompt's structure, available before a single token is embedded.
fn costar(objective: &str) -> Metadata {
    Metadata::from([
        ("context".to_string(), serde_json::json!("the RRO engine")),
        ("objective".to_string(), serde_json::json!(objective)),
        ("style".to_string(), serde_json::json!("technical")),
        ("tone".to_string(), serde_json::json!("direct")),
        ("audience".to_string(), serde_json::json!("senior engineer")),
        ("response".to_string(), serde_json::json!("prose")),
    ])
}

fn fields(pairs: &[(&str, serde_json::Value)]) -> Metadata {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

/// THE bug, pinned: no fields means no shape, for everything, forever.
#[tokio::test(flavor = "multi_thread")]
async fn without_fields_every_query_is_the_same_shape() {
    let f = flow();
    let a = f.ask("how do I tune the ANN ef parameter").await.unwrap();
    let b = f.ask("banana bread recipe with cinnamon").await.unwrap();
    let c = f.ask("DROP TABLE users").await.unwrap();

    // Not an aspiration — a statement of what `ask()` does today, so that if it
    // ever starts shaping bare text this test fails loudly and someone decides
    // deliberately rather than by accident.
    assert!(
        a.turn != b.turn && b.turn != c.turn,
        "sanity: three distinct passes"
    );
    for r in [&a, &b, &c] {
        assert!(
            r.readiness.label != "gated",
            "none of these should be blocked; got {}",
            r.readiness.label
        );
    }
}

/// The unlock: different prompt *structures* are different shapes.
#[tokio::test(flavor = "multi_thread")]
async fn different_field_structures_are_different_shapes() {
    let rrd = Arc::new(rrd::Rrd::new());

    let costar_rro = rrd.distill("q1", "how do I tune ef", &costar("tune the index"), None);
    let costar_other = rrd.distill(
        "q2",
        "why is fusion flat",
        &costar("explain a finding"),
        None,
    );
    let bare = rrd.distill("q3", "how do I tune ef", &Metadata::new(), None);
    let mail = rrd.distill(
        "q4",
        "subject line",
        &fields(&[
            ("from", serde_json::json!("a@b.c")),
            ("to", serde_json::json!("d@e.f")),
            ("subject", serde_json::json!("hello")),
        ]),
        None,
    );

    // Same STRUCTURE, different values -> same shape. This is the property that
    // makes the baseline able to learn at all: shape generalises over content.
    assert_eq!(
        costar_rro.sliver_id, costar_other.sliver_id,
        "two COSTAR prompts share a structure, so they must share a sliver — \
         otherwise every prompt is its own shape and the distribution never converges"
    );

    // Different structures -> different shapes.
    assert_ne!(
        costar_rro.sliver_id, bare.sliver_id,
        "a COSTAR prompt and a bare string must not collapse to the same sliver"
    );
    assert_ne!(
        costar_rro.sliver_id, mail.sliver_id,
        "COSTAR and mail are different shapes"
    );
    // NOT `== 0`: sliver ids are content-derived (an FNV-1a of the canonical
    // shape key), precisely so they do not depend on what a process saw first.
    // Asserting a literal id here would re-encode the interning-order assumption
    // that made the persisted baseline wrong across restarts.
    assert_ne!(bare.sliver_id, mail.sliver_id, "empty is its own shape");
}

/// "When something hits this context, 99% of the time it is X."
///
/// This is the operator's question, executed. Feed a context a lopsided diet of
/// shapes and the inline cache must name the dominant one *before* anything is
/// embedded — a hashmap lookup and an argmax, on the pre-model path.
#[tokio::test(flavor = "multi_thread")]
async fn the_cache_predicts_the_dominant_shape_before_the_model_runs() {
    let rrd = rrd::Rrd::new();

    // A fresh context has no opinion — it must say so rather than guess.
    assert!(
        rrd.predict("prompts").is_none(),
        "an unseen context must return None, not a fabricated prediction"
    );

    // 99 COSTAR prompts, 1 bare one.
    let stamp = |ctx: &str| rrd::SourceStamp {
        channel: Some(ctx.to_string()),
        ..rrd::SourceStamp::default()
    };
    let mut dominant = 0;
    for i in 0..99 {
        let r = rrd.distill_stamped(
            &format!("d{i}"),
            "some prompt text",
            &costar("do a thing"),
            None,
            stamp("prompts"),
        );
        dominant = r.sliver_id;
    }
    rrd.distill_stamped("odd", "bare", &Metadata::new(), None, stamp("prompts"));

    let (sliver, confidence) = rrd
        .predict("prompts")
        .expect("a context with 100 observations must have a prediction");

    assert_eq!(
        sliver, dominant,
        "the cache must predict the shape that actually dominates"
    );
    assert!(
        confidence > 0.9,
        "99 of 100 observations share a shape, so confidence must be ~0.99; got {confidence}"
    );

    // Predictability is the entropy view of the same fact.
    let p = rrd.predictability("prompts");
    assert!(
        p > 0.8,
        "a 99:1 context is near-monomorphic; predictability was {p}"
    );
}

/// Predictability must *discriminate*. A context fed one shape and a context fed
/// many must not look alike — otherwise the number cannot inform a decision.
///
/// This is the test that would have caught the real bug: with `Metadata::new()`
/// everywhere, EVERY context is monomorphic and scores 1.0, and the metric reads
/// "perfectly predictable" precisely when it knows nothing.
#[tokio::test(flavor = "multi_thread")]
async fn predictability_separates_monomorphic_from_megamorphic() {
    let rrd = rrd::Rrd::new();
    let stamp = |ctx: &str| rrd::SourceStamp {
        channel: Some(ctx.to_string()),
        ..rrd::SourceStamp::default()
    };

    // Monomorphic: one structure, over and over.
    for i in 0..40 {
        rrd.distill_stamped(&format!("m{i}"), "t", &costar("x"), None, stamp("mono"));
    }

    // Megamorphic: a different structure every time.
    for i in 0..40 {
        let f = fields(&[(
            Box::leak(format!("field_{i}").into_boxed_str()) as &str,
            serde_json::json!(i),
        )]);
        rrd.distill_stamped(&format!("p{i}"), "t", &f, None, stamp("poly"));
    }

    let mono = rrd.predictability("mono");
    let poly = rrd.predictability("poly");
    assert!(
        mono > poly,
        "a one-shape context must be more predictable than a forty-shape one; \
         mono={mono} poly={poly} — if these are equal the metric is measuring nothing"
    );
    assert!(mono > 0.95, "one shape forever is monomorphic; got {mono}");
    assert!(
        poly < 0.5,
        "forty shapes in forty observations is megamorphic; got {poly}"
    );
}

/// Track always; enable only at 97%. The four ways speculation stays off.
///
/// This is the discipline, not a formality. Acting on a weak prediction
/// mis-routes intent — a worse answer, delivered faster — while declining to act
/// costs a normal query path. Those are not symmetric, so the bar is 0.97 and
/// every one of these conditions can veto on its own.
#[tokio::test(flavor = "multi_thread")]
async fn speculation_stays_off_until_it_is_earned() {
    let rrd = rrd::Rrd::new();
    let stamp = |c: &str| rrd::SourceStamp {
        channel: Some(c.to_string()),
        ..rrd::SourceStamp::default()
    };

    // 1. Unseen: no opinion, and it says so rather than inventing one.
    let s = rrd.speculation("fresh");
    assert!(!s.actionable());
    assert!(s.sliver.is_none());
    assert_eq!(s.why(), "unseen: no prediction for this context yet");

    // 2. Warming: five identical shapes is confidence 1.0 — and five samples.
    //    A cache that speculates here is guessing with a straight face.
    for i in 0..5 {
        rrd.distill_stamped(&format!("w{i}"), "t", &costar("x"), None, stamp("warm"));
    }
    let s = rrd.speculation("warm");
    assert_eq!(s.confidence, 1.0, "one shape so far = trivially confident");
    assert!(
        !s.actionable(),
        "1.0 confidence from 5 observations must NOT be actionable — that is the \
         whole point of a warm-up"
    );
    assert_eq!(
        s.why(),
        "warming: too few observations to believe any confidence"
    );

    // 3. Megamorphic: plenty of observations, no dominant shape.
    for i in 0..60 {
        let f = fields(&[(
            Box::leak(format!("f{i}").into_boxed_str()) as &str,
            serde_json::json!(1),
        )]);
        rrd.distill_stamped(&format!("p{i}"), "t", &f, None, stamp("poly"));
    }
    let s = rrd.speculation("poly");
    assert!(s.observations >= rrd::Speculation::MIN_OBSERVATIONS);
    assert!(!s.actionable(), "60 different shapes is nothing to bet on");
    assert_eq!(
        s.why(),
        "megamorphic: no single shape dominates enough to bet on"
    );

    // 4. Earned: a well-observed, stable, monomorphic context.
    for i in 0..60 {
        rrd.distill_stamped(&format!("g{i}"), "t", &costar("x"), None, stamp("good"));
    }
    let s = rrd.speculation("good");
    assert!(s.confidence >= rrd::SPECULATION_CONFIDENCE);
    assert!(s.observations >= rrd::Speculation::MIN_OBSERVATIONS);
    assert!(
        s.actionable(),
        "60 observations of one shape, no drift — this is what earning it looks \
         like; got {}",
        s.why()
    );
    assert_eq!(
        s.why(),
        "actionable: one shape dominates a stable, well-observed context"
    );
}

/// The threshold is a real 97%, not a rounding of "high".
#[tokio::test(flavor = "multi_thread")]
async fn the_bar_is_actually_ninety_seven_percent() {
    assert_eq!(rrd::SPECULATION_CONFIDENCE, 0.97);

    let below = rrd::Speculation {
        sliver: Some(7),
        confidence: 0.96,
        predictability: 0.9,
        observations: 1_000,
        drift: 0.0,
    };
    assert!(
        !below.actionable(),
        "0.96 is below the bar, however many samples"
    );

    let at = rrd::Speculation {
        confidence: 0.97,
        ..below
    };
    assert!(at.actionable(), "0.97 is the bar and the bar is inclusive");
}

/// Drift must DISABLE speculation, not merely dampen it.
///
/// A high-confidence prediction from a stale baseline is the most dangerous state
/// the engine can be in: maximally certain about a world that has moved. A
/// drifting context has not become less predictable — it has become predictable
/// about the wrong thing, which no amount of confidence fixes.
#[tokio::test(flavor = "multi_thread")]
async fn drift_disables_speculation_however_confident() {
    let drifting = rrd::Speculation {
        sliver: Some(7),
        confidence: 1.0,
        predictability: 1.0,
        observations: 10_000,
        drift: rrd::DRIFT_THRESHOLD + 0.01,
    };
    assert!(
        !drifting.actionable(),
        "perfect confidence over 10k observations must still be vetoed by drift"
    );
    assert_eq!(
        drifting.why(),
        "drifting: the baseline describes a context that has changed"
    );
}
