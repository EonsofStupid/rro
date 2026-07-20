//! Cross-engine agreement: candle, llama.cpp, and vLLM must agree.
//!
//! `#[ignore]` by default — needs weights on disk and live servers.
//!
//! ```sh
//! RRO_TEST_QWEN_WEIGHTS=/path/to/qwen3-embedding-0-6b \
//! RRO_TEST_LLAMACPP=http://127.0.0.1:8090/v1/embeddings \
//!   cargo test -p embedder --features candle --test engine_agreement -- --ignored --nocapture
//! ```
//!
//! Why this matters more than either engine's own gate: two independent
//! implementations of the same contract are a cross-check nothing else provides.
//! If candle's vendored encoder and llama.cpp's C++ one disagree on the same
//! text, at least one is wrong — and *which* numbers we later publish depends on
//! that answer. Agreement is evidence the contract (last-token pooling, left
//! padding, instruct prefix, EOS, L2 norm) is implemented right in both.
//!
//! Note these engines may serve **different model sizes** (0.6B locally vs 4B on
//! :8090), so vectors are not comparable elementwise. What must agree is the
//! *semantic structure*: the relative ordering of similarities. A retrieval
//! engine is judged on ranking, not on raw cosine values.

#![cfg(feature = "candle")]

use embedder::{
    CandleQwenEmbedder, Embedder, OpenAiEmbedConfig, OpenAiEmbedder, OpenAiKind, QwenEmbedConfig,
};

const QUERIES: [&str; 2] = ["What is the capital of China?", "Explain gravity"];
const DOCS: [&str; 2] = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical \
     objects and is responsible for the movement of planets around the sun.",
];

fn strings(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

/// The 2x2 query-vs-doc similarity matrix — the engine's semantic fingerprint.
async fn fingerprint(e: &dyn Embedder) -> [[f32; 2]; 2] {
    let q = e.embed_queries(&strings(&QUERIES)).await.unwrap();
    let d = e.embed_documents(&strings(&DOCS)).await.unwrap();
    [
        [q[0].cosine(&d[0]), q[0].cosine(&d[1])],
        [q[1].cosine(&d[0]), q[1].cosine(&d[1])],
    ]
}

/// Every engine must get the ranking right: each query prefers its own document.
fn assert_diagonal_dominates(name: &str, m: [[f32; 2]; 2]) {
    println!(
        "{name:10} [[{:.4}, {:.4}], [{:.4}, {:.4}]]",
        m[0][0], m[0][1], m[1][0], m[1][1]
    );
    assert!(
        m[0][0] > m[0][1],
        "{name}: capital query must prefer the capital doc ({} vs {})",
        m[0][0],
        m[0][1]
    );
    assert!(
        m[1][1] > m[1][0],
        "{name}: gravity query must prefer the gravity doc ({} vs {})",
        m[1][1],
        m[1][0]
    );
}

async fn candle() -> Option<CandleQwenEmbedder> {
    let dir = std::env::var("RRO_TEST_QWEN_WEIGHTS").ok()?;
    if !std::path::Path::new(&dir).is_dir() {
        return None;
    }
    CandleQwenEmbedder::load(QwenEmbedConfig::new(dir)).ok()
}

/// `None` only when the var is unset. If it IS set, a connect failure is a
/// **test failure**, not a skip.
///
/// The first version of this returned `.ok()`, which turned a real error (vLLM
/// rejecting the model name with a 404) into a silent green skip. A test that
/// hides a failure is worse than no test.
async fn http(var: &str, kind: OpenAiKind) -> Option<OpenAiEmbedder> {
    let ep = std::env::var(var).ok().filter(|s| !s.trim().is_empty())?;
    match OpenAiEmbedder::connect(OpenAiEmbedConfig::new(&ep, kind)).await {
        Ok(e) => Some(e),
        Err(e) => panic!("{var}={ep} is set but connecting failed: {e}"),
    }
}

#[tokio::test]
#[ignore]
async fn candle_and_llamacpp_agree_on_ranking() {
    let Some(c) = candle().await else {
        eprintln!("SKIP: set RRO_TEST_QWEN_WEIGHTS");
        return;
    };
    let Some(l) = http("RRO_TEST_LLAMACPP", OpenAiKind::LlamaCpp).await else {
        eprintln!("SKIP: set RRO_TEST_LLAMACPP");
        return;
    };

    println!("candle dim={} llamacpp dim={}", c.dim(), l.dim());
    let cm = fingerprint(&c).await;
    let lm = fingerprint(&l).await;

    assert_diagonal_dominates("candle", cm);
    assert_diagonal_dominates("llamacpp", lm);

    // Both must separate matched from mismatched by a wide margin. Absolute
    // cosines differ (different model sizes), so compare the *margin*, which is
    // what ranking actually depends on.
    for (name, m) in [("candle", cm), ("llamacpp", lm)] {
        let margin = (m[0][0] - m[0][1]).min(m[1][1] - m[1][0]);
        println!("{name} separation margin: {margin:.4}");
        assert!(
            margin > 0.3,
            "{name}: matched/mismatched margin {margin} is too small to be real semantics"
        );
    }
}

/// vLLM, when it is serving an embedding model.
#[tokio::test]
#[ignore]
async fn vllm_ranking_is_sane() {
    let Some(v) = http("RRO_TEST_VLLM", OpenAiKind::Vllm).await else {
        eprintln!("SKIP: set RRO_TEST_VLLM to a vLLM /v1/embeddings endpoint");
        return;
    };
    println!("vllm dim={}", v.dim());
    assert_diagonal_dominates("vllm", fingerprint(&v).await);
}

/// The strongest cross-engine check available: candle and vLLM running the
/// **same weights**.
///
/// Because the model is identical, this is not a structural comparison — the
/// vectors themselves must agree. Two independent implementations of Qwen3
/// (my vendored Rust encoder vs vLLM's CUDA kernels) landing on the same vector
/// means the whole contract is right in both: RoPE, GQA, the per-head q/k norms,
/// last-token pooling, left padding, EOS, normalization. Nothing else in the
/// suite can make that claim.
///
/// Requires vLLM serving the SAME dir as RRO_TEST_QWEN_WEIGHTS.
#[tokio::test]
#[ignore]
async fn candle_and_vllm_agree_elementwise_on_same_model() {
    let Some(c) = candle().await else {
        eprintln!("SKIP: set RRO_TEST_QWEN_WEIGHTS");
        return;
    };
    let Some(v) = http("RRO_TEST_VLLM", OpenAiKind::Vllm).await else {
        eprintln!("SKIP: set RRO_TEST_VLLM");
        return;
    };
    if c.dim() != v.dim() {
        eprintln!(
            "SKIP: vLLM serves dim {} but candle has {} — not the same model",
            v.dim(),
            c.dim()
        );
        return;
    }

    let docs = strings(&[
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
        "The cat sat on the mat.",
    ]);
    let cv = c.embed_documents(&docs).await.unwrap();
    let vv = v.embed_documents(&docs).await.unwrap();

    for (i, (a, b)) in cv.iter().zip(vv.iter()).enumerate() {
        let sim = a.cosine(b);
        println!("  doc {i}: candle vs vllm cosine = {sim:.6}");
        assert!(
            sim > 0.99,
            "same model, same text, but candle and vLLM disagree (cosine {sim}) on doc {i} — \
             one of the two implementations is wrong"
        );
    }

    // Queries too: the instruction prefix must land identically on both paths.
    let qs = strings(&QUERIES);
    let cq = c.embed_queries(&qs).await.unwrap();
    let vq = v.embed_queries(&qs).await.unwrap();
    for (i, (a, b)) in cq.iter().zip(vq.iter()).enumerate() {
        let sim = a.cosine(b);
        println!("  query {i}: candle vs vllm cosine = {sim:.6}");
        assert!(sim > 0.99, "query {i} diverged: {sim}");
    }
}

/// The asymmetry must hold over HTTP too: the endpoint embeds exactly the text
/// we send and applies no instruction of its own, so the prefix is ours to add.
#[tokio::test]
#[ignore]
async fn http_backend_applies_query_asymmetry() {
    let Some(l) = http("RRO_TEST_LLAMACPP", OpenAiKind::LlamaCpp).await else {
        eprintln!("SKIP: set RRO_TEST_LLAMACPP");
        return;
    };
    let t = strings(&["What is the capital of China?"]);
    let q = l.embed_queries(&t).await.unwrap();
    let d = l.embed_documents(&t).await.unwrap();
    let sim = q[0].cosine(&d[0]);
    println!("llamacpp cos(text as query, text as doc) = {sim:.4}");
    assert!(
        sim < 0.999,
        "the query instruction prefix is not being applied over HTTP ({sim})"
    );
}

/// Batching over the wire must not reorder vectors. The OpenAI spec allows
/// `data` in any order with an `index` field; a naive client that trusts arrival
/// order attaches vectors to the wrong text — silently, and only under load.
#[tokio::test]
#[ignore]
async fn http_batch_order_is_preserved() {
    let Some(l) = http("RRO_TEST_LLAMACPP", OpenAiKind::LlamaCpp).await else {
        eprintln!("SKIP: set RRO_TEST_LLAMACPP");
        return;
    };
    let texts = strings(&[
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
        "The cat sat on the mat.",
    ]);
    let batched = l.embed_documents(&texts).await.unwrap();
    for (i, t) in texts.iter().enumerate() {
        let single = l.embed_documents(std::slice::from_ref(t)).await.unwrap();
        let sim = batched[i].cosine(&single[0]);
        println!("  row {i}: batched vs single = {sim:.6}");
        assert!(
            sim > 0.999,
            "row {i} came back attached to the wrong text (sim {sim})"
        );
    }
}
