//! The candle-backed Qwen3 cross-encoder reranker.
//!
//! **MODELS.md §4.2 does not apply to this model, and following it literally
//! would produce garbage.** It says "forward, take the relevance logit", which
//! describes an `AutoModelForSequenceClassification` with a scalar relevance
//! head — true of `llama-nemotron-rerank-1b-v2`, but that model is
//! `llama_bidirec` custom_code and cannot be loaded by candle at all.
//!
//! Qwen3-Reranker is an `AutoModelForCausalLM`. It scores by *asking the model
//! a yes/no question* and reading the logits of the "yes" and "no" tokens at the
//! final position. From the model card:
//!
//! ```text
//! prefix = <|im_start|>system\nJudge whether the Document meets the requirements
//!          based on the Query and the Instruct provided. Note that the answer
//!          can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n
//! body   = <Instruct>: {task}\n<Query>: {query}\n<Document>: {doc}
//! suffix = <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
//!
//! logits = model(ids).logits[:, -1, :]
//! score  = softmax([logits[no_id], logits[yes_id]])[1]
//! ```
//!
//! Two details that are easy to get wrong and fail silently:
//! - The reranker's `tokenizer.json` has **no TemplateProcessing** (unlike the
//!   *embedder*'s, which appends `<|endoftext|>`). Encoding must add nothing.
//! - `tie_word_embeddings: true`, so there is no `lm_head` tensor — the head is
//!   the input embedding matrix.

use std::path::PathBuf;

use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use embedder::Qwen3Encoder;
use rro_core::{Candidate, Reranker, Result, RroError};
use tokenizers::Tokenizer;

/// The card's default task description.
pub const DEFAULT_RERANK_TASK: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

const PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based \
                      on the Query and the Instruct provided. Note that the answer can only be \
                      \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Config for the candle Qwen3 reranker.
#[derive(Debug, Clone)]
pub struct CandleRerankConfig {
    /// Dir with `model*.safetensors`, `config.json`, `tokenizer.json`.
    pub weights_dir: PathBuf,
    /// Where to run.
    pub device: Device,
    /// (query, doc) pairs per forward pass.
    pub batch: usize,
    /// Task description injected as `<Instruct>`. `None` = the card default.
    pub task: Option<String>,
    /// Max tokens per pair.
    pub max_len: usize,
}

impl CandleRerankConfig {
    /// Config for `weights_dir`.
    pub fn new(weights_dir: impl Into<PathBuf>) -> Self {
        CandleRerankConfig {
            weights_dir: weights_dir.into(),
            device: Device::Cpu,
            batch: 8,
            task: None,
            max_len: 8192,
        }
    }
}

/// Qwen3-Reranker behind the [`Reranker`] trait. Stateless (the encoder holds
/// no KV cache), so no mutex.
#[derive(Debug)]
pub struct CandleQwenReranker {
    encoder: Qwen3Encoder,
    tokenizer: Tokenizer,
    cfg: CandleRerankConfig,
    prefix_ids: Vec<u32>,
    suffix_ids: Vec<u32>,
    pad_id: u32,
    /// Tied-head rows for ["no", "yes"], in that order.
    no_yes: Tensor,
    name: String,
}

impl CandleQwenReranker {
    /// Load weights and warm the graph.
    pub fn load(cfg: CandleRerankConfig) -> Result<Self> {
        let dir = cfg.weights_dir.clone();
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json"))
                .map_err(|e| err(format!("read {}/config.json: {e}", dir.display())))?,
        )
        .map_err(|e| err(format!("parse config.json: {e}")))?;

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| err(format!("load tokenizer.json: {e}")))?;

        let dtype = if cfg.device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        };

        let mut shards: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| err(format!("read dir {}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(err(format!("no *.safetensors in {}", dir.display())));
        }
        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        for shard in &shards {
            let part = candle_core::safetensors::load(shard, &cfg.device)
                .map_err(|e| err(format!("load {}: {e}", shard.display())))?;
            tensors.extend(part);
        }
        let vb = VarBuilder::from_tensors(tensors, dtype, &cfg.device);
        let prefix = Qwen3Encoder::detect_prefix(&vb).map_err(|e| err(e.to_string()))?;
        let encoder = Qwen3Encoder::load(&config, vb, prefix).map_err(|e| err(e.to_string()))?;

        // add_special_tokens = false everywhere: this tokenizer has no
        // TemplateProcessing, and the chat scaffolding is explicit in
        // PREFIX/SUFFIX.
        let enc_no_special = |s: &str| -> Result<Vec<u32>> {
            tokenizer
                .encode(s, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| err(format!("tokenize: {e}")))
        };
        let prefix_ids = enc_no_special(PREFIX)?;
        let suffix_ids = enc_no_special(SUFFIX)?;

        let yes = tokenizer
            .token_to_id("yes")
            .ok_or_else(|| err("tokenizer has no `yes` token"))?;
        let no = tokenizer
            .token_to_id("no")
            .ok_or_else(|| err("tokenizer has no `no` token"))?;
        // [no, yes] — the card stacks false first, then takes index 1.
        let no_yes = encoder
            .tied_head_rows(&[no, yes])
            .map_err(|e| err(format!("tied head rows: {e}")))?;

        let pad_id = tokenizer.token_to_id("<|endoftext|>").unwrap_or(no);

        let name = format!(
            "candle-qwen3-rerank-{}",
            dir.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
        );

        let me = CandleQwenReranker {
            encoder,
            tokenizer,
            cfg,
            prefix_ids,
            suffix_ids,
            pad_id,
            no_yes,
            name,
        };
        let _ = me.score_pairs("warm", &["warm".to_string()])?;
        Ok(me)
    }

    /// P(yes) for each (query, doc) pair, in input order.
    fn score_pairs(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let task = self.cfg.task.as_deref().unwrap_or(DEFAULT_RERANK_TASK);
        let mut out = Vec::with_capacity(docs.len());
        for chunk in docs.chunks(self.cfg.batch.max(1)) {
            let bodies: Vec<String> = chunk
                .iter()
                .map(|d| format!("<Instruct>: {task}\n<Query>: {query}\n<Document>: {d}"))
                .collect();
            out.extend(self.forward_chunk(&bodies)?);
        }
        Ok(out)
    }

    fn forward_chunk(&self, bodies: &[String]) -> Result<Vec<f32>> {
        let room = self
            .cfg
            .max_len
            .saturating_sub(self.prefix_ids.len() + self.suffix_ids.len())
            .max(1);

        let mut ids: Vec<Vec<u32>> = Vec::with_capacity(bodies.len());
        for b in bodies {
            let mut body = self
                .tokenizer
                .encode(b.as_str(), false)
                .map_err(|e| err(format!("tokenize: {e}")))?
                .get_ids()
                .to_vec();
            body.truncate(room);
            let mut v =
                Vec::with_capacity(self.prefix_ids.len() + body.len() + self.suffix_ids.len());
            v.extend_from_slice(&self.prefix_ids);
            v.extend_from_slice(&body);
            v.extend_from_slice(&self.suffix_ids);
            ids.push(v);
        }

        // LEFT pad (card: padding_side='left'), so the last column is every
        // row's real final suffix token — the position whose logits we read.
        let l = ids.iter().map(|v| v.len()).max().unwrap_or(1).max(1);
        let pad_lens: Vec<usize> = ids.iter().map(|v| l - v.len()).collect();
        for v in ids.iter_mut() {
            let pad = l - v.len();
            if pad > 0 {
                let mut p = vec![self.pad_id; pad];
                p.extend_from_slice(v);
                *v = p;
            }
        }

        let b = ids.len();
        let input = Tensor::from_vec(ids.concat(), (b, l), self.encoder.device())
            .map_err(|e| err(format!("input tensor: {e}")))?;
        let mask = self
            .encoder
            .left_pad_mask(&pad_lens, l)
            .map_err(|e| err(format!("mask: {e}")))?;
        let hidden = self
            .encoder
            .forward(&input, Some(&mask))
            .map_err(|e| err(format!("forward: {e}")))?;
        // Last position only — the assistant's next token slot.
        let last = self
            .encoder
            .pool_last(&hidden)
            .map_err(|e| err(format!("pool: {e}")))?;

        // logits for exactly ["no","yes"]: (B,H) @ (H,2) -> (B,2)
        let two = self
            .no_yes
            .t()
            .and_then(|t| last.matmul(&t.contiguous()?))
            .map_err(|e| err(format!("head matmul: {e}")))?
            .to_dtype(DType::F32)
            .map_err(|e| err(format!("to f32: {e}")))?;

        // softmax over [no, yes]; score = P(yes). The card uses log_softmax then
        // exp of index 1, which is identical.
        let probs = candle_nn::ops::softmax_last_dim(&two)
            .map_err(|e| err(format!("softmax: {e}")))?
            .to_vec2::<f32>()
            .map_err(|e| err(format!("read scores: {e}")))?;

        Ok(probs.into_iter().map(|r| r[1]).collect())
    }
}

fn err(msg: impl Into<String>) -> RroError {
    RroError::Rerank(msg.into())
}

#[async_trait]
impl Reranker for CandleQwenReranker {
    async fn rerank(
        &self,
        query: &str,
        mut candidates: Vec<Candidate>,
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        if candidates.is_empty() {
            return Ok(candidates);
        }
        let docs: Vec<String> = candidates.iter().map(|c| c.text.clone()).collect();
        let scores = self.score_pairs(query, &docs)?;
        for (c, s) in candidates.iter_mut().zip(scores) {
            c.score = s;
        }
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        candidates.truncate(top_k);
        Ok(candidates)
    }

    fn model_name(&self) -> &str {
        &self.name
    }
}
