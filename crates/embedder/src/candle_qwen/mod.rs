//! The candle-backed Qwen3 embedder — the real forward pass.
//!
//! Everything here is dictated by the model's own files, not by convention.
//! Qwen3-Embedding gets five things wrong-quietly if you guess, so each is read
//! from the checkpoint and named explicitly:
//!
//! | contract | source of truth | wrong = |
//! |---|---|---|
//! | **last-token** pooling | `1_Pooling/config.json`: `pooling_mode_lasttoken: true` | mean-pooling looks fine, retrieves worse |
//! | **left** padding | the card's `last_token_pool` | pools a PAD token: garbage |
//! | **asymmetric** prompts | `config_sentence_transformers.json` | doc embedded as a query: 1–5% lost |
//! | **L2 normalize** | `modules.json` module `2_Normalize` | cosine is meaningless |
//! | **EOS appended** | `tokenizer.json` `TemplateProcessing` | pools the wrong token entirely |
//!
//! Note the docs in `ASSESSMENT.md`/`NOTES.md` say "mean-pool" for this model.
//! They are wrong; `1_Pooling/config.json` is authoritative. That mistake would
//! not fail a build or a test — it would just quietly make every number worse,
//! which is exactly the failure P7 exists to end.

mod encoder;

pub use encoder::Qwen3Encoder;

use std::path::{Path, PathBuf};

use crate::DEFAULT_QUERY_TASK;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use rro_core::{Embedder, Embedding, Result, RroError};
use tokenizers::Tokenizer;

/// Everything the Qwen3 forward pass needs that is *not* generic to embedders.
///
/// Per MODELS.md §1 rule 3 ("performance lives inside the backend"), this stays
/// here rather than leaking into the registry's generic `EmbedderConfig`.
#[derive(Debug, Clone)]
pub struct QwenEmbedConfig {
    /// Directory holding `model*.safetensors`, `config.json`, `tokenizer.json`.
    pub weights_dir: PathBuf,
    /// Where to run.
    pub device: Device,
    /// Texts per forward pass.
    pub batch: usize,
    /// Truncate + re-normalize to this dim (MRL). `None` = native.
    ///
    /// Only valid because Qwen3-Embedding is matryoshka-trained (32..=1024 for
    /// 0.6B). Truncating a non-MRL model's vector is just corruption.
    pub truncate_dim: Option<usize>,
    /// Instruction prefixed to queries. `None` = the card's default.
    pub query_task: Option<String>,
    /// Max tokens per text; longer is truncated.
    pub max_len: usize,
}

impl QwenEmbedConfig {
    /// A config with the card's defaults for `weights_dir`.
    pub fn new(weights_dir: impl Into<PathBuf>) -> Self {
        QwenEmbedConfig {
            weights_dir: weights_dir.into(),
            device: Device::Cpu,
            batch: 32,
            truncate_dim: None,
            query_task: None,
            max_len: 8192,
        }
    }
}

/// Qwen3-Embedding behind the [`Embedder`] trait.
///
/// Stateless: the encoder holds no KV cache, so this is `Send + Sync` with no
/// interior mutability and concurrent queries never serialize on a lock.
#[derive(Debug)]
pub struct CandleQwenEmbedder {
    encoder: Qwen3Encoder,
    tokenizer: Tokenizer,
    cfg: QwenEmbedConfig,
    /// `<|endoftext|>` — the pad token per `tokenizer_config.json`.
    pad_id: u32,
    dim: usize,
    name: String,
}

impl CandleQwenEmbedder {
    /// Load weights and warm the graph.
    pub fn load(cfg: QwenEmbedConfig) -> Result<Self> {
        let dir = cfg.weights_dir.clone();
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json"))
                .map_err(|e| embed_err(format!("read {}/config.json: {e}", dir.display())))?,
        )
        .map_err(|e| embed_err(format!("parse config.json: {e}")))?;

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| embed_err(format!("load tokenizer.json: {e}")))?;

        // bf16 on GPU (the checkpoint's native dtype); f32 on CPU, where bf16
        // kernels are slow and thinly covered.
        let dtype = if cfg.device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        };

        // MODELS.md §3 suggests mmap, but `VarBuilder::from_mmaped_safetensors`
        // is `unsafe` (an mmap is UB if the file is mutated underneath) and this
        // crate is `#![forbid(unsafe_code)]`. That guarantee is worth more than
        // the mmap: loading is a one-time startup cost, and `safetensors::load`
        // is a plain safe read. Revisit only if a huge checkpoint makes the
        // resident copy actually hurt.
        let shards = safetensors_shards(&dir)?;
        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        for shard in &shards {
            let part = candle_core::safetensors::load(shard, &cfg.device)
                .map_err(|e| embed_err(format!("load {}: {e}", shard.display())))?;
            tensors.extend(part);
        }
        let vb = VarBuilder::from_tensors(tensors, dtype, &cfg.device);

        // sentence-transformers exports store tensors at the root; a
        // Qwen3ForCausalLM export nests them under `model.`. Detect, don't guess.
        let prefix = Qwen3Encoder::detect_prefix(&vb).map_err(|e| embed_err(e.to_string()))?;
        let encoder =
            Qwen3Encoder::load(&config, vb, prefix).map_err(|e| embed_err(e.to_string()))?;

        let pad_id = tokenizer
            .token_to_id("<|endoftext|>")
            .unwrap_or(config.vocab_size as u32 - 1);

        let native = encoder.hidden_size();
        let dim = match cfg.truncate_dim {
            Some(d) if d == 0 || d > native => {
                return Err(embed_err(format!(
                    "truncate_dim {d} invalid: must be 1..={native} (the model's native dim)"
                )))
            }
            Some(d) => d,
            None => native,
        };

        let name = format!(
            "candle-qwen3-{}",
            dir.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
        );

        let me = CandleQwenEmbedder {
            encoder,
            tokenizer,
            cfg,
            pad_id,
            dim,
            name,
        };
        // Warm the graph so the first real query isn't paying compile/alloc cost.
        let _ = me.encode(&["warm".to_string()], None)?;
        Ok(me)
    }

    /// Tokenize, forward, pool, normalize, truncate. `task` is `Some` for
    /// queries (instruction-prefixed) and `None` for documents (bare).
    fn encode(&self, texts: &[String], task: Option<&str>) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let prepared: Vec<String> = match task {
            Some(t) => texts
                .iter()
                .map(|q| format!("Instruct: {t}\nQuery:{q}"))
                .collect(),
            None => texts.to_vec(),
        };

        let mut out = Vec::with_capacity(texts.len());
        for chunk in prepared.chunks(self.cfg.batch.max(1)) {
            out.extend(self.forward_chunk(chunk)?);
        }
        Ok(out)
    }

    fn forward_chunk(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        // `true` applies tokenizer.json's TemplateProcessing, which appends
        // <|endoftext|>. That EOS is the token last-token pooling returns, so
        // dropping it would silently change every embedding.
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| embed_err(format!("tokenize: {e}")))?;

        let mut ids: Vec<Vec<u32>> = encodings
            .iter()
            .map(|e| {
                let mut v = e.get_ids().to_vec();
                if v.len() > self.cfg.max_len {
                    // Keep the tail: it holds the EOS we pool.
                    v = v[v.len() - self.cfg.max_len..].to_vec();
                }
                v
            })
            .collect();

        let l = ids.iter().map(|v| v.len()).max().unwrap_or(1).max(1);
        let pad_lens: Vec<usize> = ids.iter().map(|v| l - v.len()).collect();
        // LEFT pad, so the last column is every row's real final token.
        for v in ids.iter_mut() {
            let pad = l - v.len();
            if pad > 0 {
                let mut padded = vec![self.pad_id; pad];
                padded.extend_from_slice(v);
                *v = padded;
            }
        }

        let flat: Vec<u32> = ids.concat();
        let b = ids.len();
        let input = Tensor::from_vec(flat, (b, l), self.encoder.device())
            .map_err(|e| embed_err(format!("input tensor: {e}")))?;

        let mask = self
            .encoder
            .left_pad_mask(&pad_lens, l)
            .map_err(|e| embed_err(format!("mask: {e}")))?;

        let hidden = self
            .encoder
            .forward(&input, Some(&mask))
            .map_err(|e| embed_err(format!("forward: {e}")))?;
        let pooled = self
            .encoder
            .pool_last(&hidden)
            .map_err(|e| embed_err(format!("pool: {e}")))?;
        let pooled = pooled
            .to_dtype(DType::F32)
            .map_err(|e| embed_err(format!("to f32: {e}")))?;

        let rows: Vec<Vec<f32>> = pooled
            .to_vec2()
            .map_err(|e| embed_err(format!("read vectors: {e}")))?;

        Ok(rows
            .into_iter()
            .map(|mut v| {
                // MRL: truncate first, then normalize — normalizing the full
                // vector and then slicing leaves a non-unit vector, and the
                // estate's cosine path assumes unit length.
                v.truncate(self.dim);
                Embedding(v).normalized()
            })
            .collect())
    }
}

fn embed_err(msg: impl Into<String>) -> RroError {
    RroError::Embed(msg.into())
}

/// Every `*.safetensors` in `dir`, sorted so sharded checkpoints load in order.
fn safetensors_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| embed_err(format!("read dir {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(embed_err(format!(
            "no *.safetensors in {} — is this a weights dir?",
            dir.display()
        )));
    }
    Ok(shards)
}

#[async_trait]
impl Embedder for CandleQwenEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    /// Bare text. `embed` is the DOCUMENT path deliberately: prefixing a
    /// document with the query instruction is the harmful direction, so the
    /// unspecified call defaults to the safe one.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.encode(texts, None)
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.encode(texts, None)
    }

    async fn embed_queries(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let task = self.cfg.query_task.as_deref().unwrap_or(DEFAULT_QUERY_TASK);
        self.encode(texts, Some(task))
    }

    fn model_name(&self) -> &str {
        &self.name
    }
}
