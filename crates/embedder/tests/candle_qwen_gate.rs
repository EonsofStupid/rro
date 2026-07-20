//! P7.2's gate: does the vendored Qwen3 encoder actually produce real semantics?
//!
//! `#[ignore]` by default — these need weights on disk, which CI does not have.
//! Run on a box that has them:
//!
//! ```sh
//! RRO_TEST_QWEN_WEIGHTS=/path/to/qwen3-embedding-0-6b \
//!   cargo test -p embedder --features candle --test candle_qwen_gate -- --ignored --nocapture
//! ```
//!
//! The strong gate is `card_reference_scores`. MODELS.md only asks for
//! `king~queen > king~banana`, but that is weak: those are single tokens, so it
//! passes identically under last-token OR mean pooling, and cannot detect the
//! pooling bug this backend exists to avoid. The model card publishes exact
//! similarity scores for a specific 2-query × 2-doc example — an oracle. If our
//! Rust reproduces those numbers, then pooling, left-padding, the instruction
//! prefix, EOS handling, and normalization are all provably right at once.

#![cfg(feature = "candle")]

use embedder::{CandleQwenEmbedder, Embedder, QwenEmbedConfig};

fn weights() -> Option<String> {
    match std::env::var("RRO_TEST_QWEN_WEIGHTS") {
        Ok(p) if !p.trim().is_empty() && std::path::Path::new(&p).is_dir() => Some(p),
        _ => None,
    }
}

fn load(truncate: Option<usize>) -> CandleQwenEmbedder {
    let dir = weights().expect("set RRO_TEST_QWEN_WEIGHTS to the model dir");
    let mut cfg = QwenEmbedConfig::new(dir);
    cfg.truncate_dim = truncate;
    CandleQwenEmbedder::load(cfg).expect("load Qwen3 weights")
}

fn cosine(a: &rro_core::Embedding, b: &rro_core::Embedding) -> f32 {
    a.cosine(b)
}

/// THE gate: reproduce the model card's published scores.
///
/// Card (Qwen3-Embedding-0.6B, `Transformers Usage`):
///   `[[0.7645568251609802, 0.14142508804798126],
///     [0.13549736142158508, 0.5999549627304077]]`
#[tokio::test]
#[ignore]
async fn card_reference_scores() {
    let e = load(None);
    assert_eq!(e.dim(), 1024, "native dim");

    // Queries go through embed_queries (instruction-prefixed); documents go
    // through embed_documents (bare). Getting this backwards is the silent bug.
    let queries = vec![
        "What is the capital of China?".to_string(),
        "Explain gravity".to_string(),
    ];
    let docs = vec![
        "The capital of China is Beijing.".to_string(),
        "Gravity is a force that attracts two bodies towards each other. It gives weight to \
         physical objects and is responsible for the movement of planets around the sun."
            .to_string(),
    ];

    let qv = e.embed_queries(&queries).await.unwrap();
    let dv = e.embed_documents(&docs).await.unwrap();

    let got = [
        [cosine(&qv[0], &dv[0]), cosine(&qv[0], &dv[1])],
        [cosine(&qv[1], &dv[0]), cosine(&qv[1], &dv[1])],
    ];
    let want = [[0.7645568f32, 0.14142509], [0.13549736, 0.59995496]];

    println!("got:  {got:?}");
    println!("want: {want:?}");

    // Tolerance covers f32-vs-bf16 accumulation order, not a different
    // algorithm. A pooling/prefix/padding mistake moves these by >0.1, so this
    // is tight enough to catch every failure mode it is aimed at.
    for i in 0..2 {
        for j in 0..2 {
            let d = (got[i][j] - want[i][j]).abs();
            assert!(
                d < 0.02,
                "score[{i}][{j}] = {} but the card publishes {} (delta {d:.4}). \
                 A delta this large means the contract is wrong — check last-token pooling, \
                 left padding, the query instruction prefix, or the appended EOS.",
                got[i][j],
                want[i][j]
            );
        }
    }

    // The diagonal must dominate: each query matches its own document.
    assert!(got[0][0] > got[0][1], "capital query must prefer its doc");
    assert!(got[1][1] > got[1][0], "gravity query must prefer its doc");
}

/// MODELS.md's stated sanity gate. Weak on its own (see the module docs), kept
/// because the spec names it.
#[tokio::test]
#[ignore]
async fn semantic_sanity_king_queen_banana() {
    let e = load(None);
    let v = e
        .embed_documents(&[
            "king".to_string(),
            "queen".to_string(),
            "banana".to_string(),
        ])
        .await
        .unwrap();
    let kq = cosine(&v[0], &v[1]);
    let kb = cosine(&v[0], &v[2]);
    println!("cos(king,queen)={kq:.4}  cos(king,banana)={kb:.4}");
    assert!(kq > kb, "king~queen ({kq}) must exceed king~banana ({kb})");
}

/// Paraphrases score high, unrelated text scores low — multi-token, so unlike
/// the single-token gate this one genuinely exercises pooling.
#[tokio::test]
#[ignore]
async fn paraphrase_beats_unrelated() {
    let e = load(None);
    let v = e
        .embed_documents(&[
            "The cat sat on the mat.".to_string(),
            "A feline was resting upon the rug.".to_string(),
            "Quantum chromodynamics describes the strong interaction.".to_string(),
        ])
        .await
        .unwrap();
    let para = cosine(&v[0], &v[1]);
    let unrel = cosine(&v[0], &v[2]);
    println!("paraphrase={para:.4}  unrelated={unrel:.4}");
    assert!(
        para > unrel + 0.15,
        "paraphrase {para} vs unrelated {unrel}"
    );
}

/// The asymmetry is real and load-bearing: embedding a query through the
/// document path must produce a *different* vector. If these ever match, the
/// instruction prefix silently stopped being applied.
#[tokio::test]
#[ignore]
async fn query_and_document_paths_differ() {
    let e = load(None);
    let t = vec!["What is the capital of China?".to_string()];
    let as_query = e.embed_queries(&t).await.unwrap();
    let as_doc = e.embed_documents(&t).await.unwrap();
    let sim = cosine(&as_query[0], &as_doc[0]);
    println!("cos(same text as query vs as doc) = {sim:.4}");
    assert!(
        sim < 0.999,
        "query and document paths produced the same vector ({sim}) — the instruction \
         prefix is not being applied"
    );
}

/// Batched and single-text embedding must agree. This is the padding+mask proof:
/// a left-padded row is only equivalent to an unpadded one if the padding mask
/// blocks the pad run AND RoPE's relative-position invariance holds.
#[tokio::test]
#[ignore]
async fn batching_matches_single() {
    let e = load(None);
    let texts = vec![
        "short".to_string(),
        "a considerably longer sentence that will force the shorter rows in this batch to be \
         left-padded by a meaningful number of tokens"
            .to_string(),
    ];
    let batched = e.embed_documents(&texts).await.unwrap();
    let single_0 = e.embed_documents(&texts[0..1]).await.unwrap();
    let single_1 = e.embed_documents(&texts[1..2]).await.unwrap();

    let s0 = cosine(&batched[0], &single_0[0]);
    let s1 = cosine(&batched[1], &single_1[0]);
    println!("padded-vs-unpadded: short={s0:.6} long={s1:.6}");
    assert!(
        s0 > 0.999,
        "left-padded short row diverged from unpadded ({s0}) — the padding mask is wrong"
    );
    assert!(s1 > 0.999, "unpadded long row diverged ({s1})");
}

/// MRL: Qwen3-Embedding is matryoshka-trained, so a truncated vector must stay
/// unit-length and keep its semantics.
#[tokio::test]
#[ignore]
async fn matryoshka_truncation_stays_semantic() {
    let e = load(Some(256));
    assert_eq!(e.dim(), 256, "truncate_dim honored");
    let v = e
        .embed_documents(&[
            "king".to_string(),
            "queen".to_string(),
            "banana".to_string(),
        ])
        .await
        .unwrap();
    assert_eq!(v[0].dim(), 256);

    let norm: f32 = v[0].0.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "truncated vector must be re-normalized, got norm {norm}"
    );

    let kq = cosine(&v[0], &v[1]);
    let kb = cosine(&v[0], &v[2]);
    println!("MRL@256: cos(king,queen)={kq:.4} cos(king,banana)={kb:.4}");
    assert!(kq > kb, "semantics must survive MRL truncation");
}
