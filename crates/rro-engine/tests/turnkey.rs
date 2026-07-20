//! The turnkey contract: a developer imports `rro-engine`, calls ONE constructor
//! with NO external servers, and gets a working recall+intelligence engine.
//!
//! This is the "frictionless import" the facade promises. If any of this needs a
//! vLLM server, a config file, or an async runtime the caller didn't ask for, the
//! promise is broken. `EmbeddedEngine::deterministic` is the import-and-go path;
//! these tests hold it honest end-to-end (index → ask → shape → health → dim).

use rro_core::Metadata;
use rro_engine::{sample_corpus, EmbeddedEngine, EngineMode};

/// Index-and-ask with zero setup: open a persistent estate, index the sample
/// corpus, ground a query — all in-process, no servers, no config.
#[tokio::test(flavor = "multi_thread")]
async fn deterministic_engine_indexes_and_answers_with_no_servers() {
    let dir = tempfile::tempdir().unwrap();
    let engine = EmbeddedEngine::deterministic(dir.path(), "turnkey").expect("open");

    // Fresh estate: no fixed dimension yet.
    assert_eq!(engine.dim(), None, "a fresh estate has no dim until first index");

    let n = engine.index(sample_corpus()).await.expect("index");
    assert!(n > 0, "the sample corpus indexed at least one document");

    // The estate's dimension is now fixed (the deterministic embedder's width).
    assert!(engine.dim().is_some(), "dim is fixed once documents are indexed");

    // Drain the out-of-band graph applier so the ANN index is live before we query
    // (otherwise recall races the async apply).
    engine.estate().quiesce();

    // The turnkey contract is that the full pass RUNS import-and-go and returns a
    // legible result for the query asked — not a claim about recall *quality*,
    // which is meaningless on the synthetic deterministic embedder and is proven
    // separately with real embeddings (clyffy + DuckDB).
    let query = "how do I upgrade postgres safely?";
    let result = engine.ask(query).await.expect("ask");
    assert_eq!(result.query, query, "the result is for the query asked");
    assert!(result.turn.get() > 0, "the turn ran and closed");
}

/// `ask_with` carries shaping metadata — the path that makes RRD's shape/intent
/// real (plain `ask` passes none). It must run end-to-end on the weightless engine.
#[tokio::test(flavor = "multi_thread")]
async fn ask_with_shaping_metadata_runs() {
    let dir = tempfile::tempdir().unwrap();
    let engine = EmbeddedEngine::deterministic(dir.path(), "shaped").expect("open");
    engine.index(sample_corpus()).await.expect("index");

    let mut fields = Metadata::new();
    fields.insert("source".into(), serde_json::json!("cli"));
    fields.insert("tags".into(), serde_json::json!(["ops", "postgres"]));

    let result = engine
        .ask_with("how do I upgrade postgres safely?", &fields)
        .await
        .expect("ask_with");
    assert!(result.turn.get() > 0, "a shaped turn still runs and closes");
}

/// The deterministic engine is always healthy (no servers to fall over) and
/// reports how it was assembled.
#[tokio::test(flavor = "multi_thread")]
async fn health_reports_ready_for_the_weightless_engine() {
    let dir = tempfile::tempdir().unwrap();
    let engine = EmbeddedEngine::deterministic(dir.path(), "health").expect("open");

    let h = engine.health().await;
    assert!(h.ready, "the weightless engine has no server to fail: {}", h.note);
    assert_eq!(h.mode, EngineMode::Deterministic);
}

/// Preflight: reopening an estate whose dimension already matches the embedder
/// succeeds (the check only fires on a real mismatch, which the deterministic
/// embedder's fixed width cannot produce against its own estate).
#[tokio::test(flavor = "multi_thread")]
async fn reopening_a_matching_estate_passes_preflight() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = EmbeddedEngine::deterministic(dir.path(), "reopen").expect("open");
        engine.index(sample_corpus()).await.expect("index");
    }
    // Reopen the now-dimensioned estate: preflight must NOT reject a matching dim.
    let engine = EmbeddedEngine::deterministic(dir.path(), "reopen").expect("reopen passes preflight");
    assert!(engine.dim().is_some(), "the reopened estate kept its fixed dim");
}
