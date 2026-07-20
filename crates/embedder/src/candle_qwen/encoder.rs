//! A cache-free Qwen3 encoder.
//!
//! Adapted from `candle-transformers` `models/qwen3.rs` (Copyright the Candle
//! authors; MIT OR Apache-2.0). The upstream model is built for *generation*:
//! `forward` takes `&mut self` and appends to a `ConcatKvCache`, and the only
//! public reset lives on `ModelForCausalLM` — whose `base` is private and whose
//! `forward` returns logits. That makes the upstream `Model` unusable as an
//! encoder from outside the crate: a second `embed` call would attend over the
//! first call's keys, and there is no way to clear them.
//!
//! An embedder runs **one forward and never generates**, so the KV cache is
//! pure liability. Removing it is what this file is for, and it buys three
//! things beyond just unblocking us:
//!
//! 1. **`forward` takes `&self`.** With no cache there is no mutable state, so
//!    the embedder is `Send + Sync` with no mutex and no serialization between
//!    concurrent queries.
//! 2. **A real attention mask.** Upstream's flash paths only accept
//!    `causal_with_offset`, so left-padded batches cannot be expressed. Here the
//!    caller passes an additive mask, which is what left-padding requires — and
//!    left-padding is what the Qwen3-Embedding card specifies.
//! 3. **A configurable tensor prefix.** Upstream hardcodes `vb.pp("model.…")`,
//!    but a sentence-transformers export stores tensors at the root
//!    (`embed_tokens.weight`, not `model.embed_tokens.weight`).
//!
//! Deliberate trade-off: this uses the standard matmul attention path on every
//! device rather than the fused flash kernels, because those kernels cannot take
//! an arbitrary padding mask. Correctness over throughput; revisit only with a
//! measurement showing it matters.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Activation, Embedding, Linear, RmsNorm, VarBuilder};
use candle_transformers::models::qwen3::Config;
use candle_transformers::utils::repeat_kv;
use std::sync::Arc;

/// `linear_b` equivalent: bias presence is a config flag on Qwen3.
fn linear_maybe_bias(in_d: usize, out_d: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_d, out_d, vb)
    } else {
        candle_nn::linear_no_bias(in_d, out_d, vb)
    }
}

/// Precomputed RoPE tables.
#[derive(Debug)]
struct Rotary {
    sin: Tensor,
    cos: Tensor,
}

impl Rotary {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE to q,k of shape (B, H, L, D) at absolute positions `0..L`.
    ///
    /// Positions are the tensor indices, matching the HF reference: transformers
    /// defaults `position_ids` to `arange(seq_len)` for a plain forward, so a
    /// left-padded row's real tokens sit at `pad_len..L`. RoPE is relative — a
    /// uniform shift of every position in a row leaves `q_i · k_j` unchanged —
    /// so this is equivalent to embedding the row unpadded, which is exactly why
    /// batched and single-text embedding agree.
    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

#[derive(Debug)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: candle_nn::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

/// Qwen3 attention: GQA + per-head q/k RMSNorm, no KV cache.
#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary: Arc<Rotary>,
}

impl Attention {
    fn new(cfg: &Config, rotary: Arc<Rotary>, vb: VarBuilder) -> Result<Self> {
        if cfg.use_sliding_window {
            candle_core::bail!("sliding window is not supported by the RRO Qwen3 encoder")
        }
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        Ok(Self {
            q_proj: linear_maybe_bias(
                cfg.hidden_size,
                num_heads * head_dim,
                cfg.attention_bias,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_maybe_bias(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                cfg.attention_bias,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_maybe_bias(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                cfg.attention_bias,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_maybe_bias(
                num_heads * head_dim,
                cfg.hidden_size,
                cfg.attention_bias,
                vb.pp("o_proj"),
            )?,
            q_norm: candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            // The config's hidden_size is not always head_dim * num_heads.
            hidden_size: head_dim * num_heads,
            rotary,
        })
    }

    /// `mask` is additive, broadcastable to (B, H, L, L): 0 where attention is
    /// allowed, -inf where it is not.
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMSNorm on q and k — the Qwen3-specific bit.
        let q = self.q_norm.forward(&q.flatten(0, 2)?)?.reshape((
            b,
            self.num_heads,
            l,
            self.head_dim,
        ))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((
            b,
            self.num_kv_heads,
            l,
            self.head_dim,
        ))?;

        let (q, k) = self.rotary.apply(&q, &k)?;

        // GQA: broadcast kv heads up to query heads.
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, rotary: Arc<Rotary>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, rotary, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            ln1: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?.apply(&self.mlp)?;
        x + h2
    }
}

/// A Qwen3 backbone that returns hidden states. Stateless: `forward` is `&self`.
#[derive(Debug)]
pub struct Qwen3Encoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
    hidden_size: usize,
}

impl Qwen3Encoder {
    /// Load from `vb`, reading tensors under `prefix` (`""` for a
    /// sentence-transformers export whose tensors sit at the root; `"model"` for
    /// a `Qwen3ForCausalLM` export). Detect with [`Qwen3Encoder::detect_prefix`].
    pub fn load(cfg: &Config, vb: VarBuilder, prefix: &str) -> Result<Self> {
        if vb.dtype() == DType::F64 {
            candle_core::bail!("Qwen3 does not support f64; load as f32 (CPU) or bf16/f16 (GPU)")
        }
        let root = if prefix.is_empty() {
            vb.clone()
        } else {
            vb.pp(prefix)
        };
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, root.pp("embed_tokens"))?;
        let rotary = Arc::new(Rotary::new(vb.dtype(), cfg, vb.device())?);
        let vb_l = root.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_l.pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, root.pp("norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
        })
    }

    /// Which tensor layout this checkpoint uses. Returns `""` (root) or
    /// `"model"`, or an error naming what was looked for — a wrong guess here
    /// surfaces as a confusing missing-tensor error deep in the load.
    pub fn detect_prefix(vb: &VarBuilder) -> Result<&'static str> {
        if vb.contains_tensor("embed_tokens.weight") {
            Ok("")
        } else if vb.contains_tensor("model.embed_tokens.weight") {
            Ok("model")
        } else {
            candle_core::bail!(
                "not a Qwen3 checkpoint: found neither `embed_tokens.weight` nor \
                 `model.embed_tokens.weight` in the safetensors"
            )
        }
    }

    /// Native output dimensionality.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Model dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Encode `ids` (B, L) into hidden states (B, L, hidden_size).
    ///
    /// `mask` is additive and broadcastable to (B, 1, L, L). Pass `None` only
    /// for a single unpadded sequence.
    pub fn forward(&self, ids: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let mut h = self.embed_tokens.forward(ids)?;
        for layer in &self.layers {
            h = layer.forward(&h, mask)?;
        }
        self.norm.forward(&h)
    }

    /// Build the additive attention mask for a **left-padded** batch.
    ///
    /// Combines two constraints: causal (`j <= i`, this is a decoder backbone)
    /// and padding (key `j` must be a real token, i.e. `j >= pad_len[b]`).
    /// Without the padding half, every real token would attend to the pad run
    /// and the vectors would be quietly wrong — no error, just worse retrieval.
    ///
    /// The subtlety that bites: a **pad query row** (`i < pad`) satisfies no `j`
    /// under both rules at once, so its scores would be all `-inf`, softmax
    /// would yield NaN, and — because a masked weight is exactly `0` and
    /// `0 * NaN = NaN` — that NaN propagates into the *real* rows through the
    /// next layer's matmul. Masking cannot save you from a NaN key. So pad rows
    /// keep plain causal attention: they compute finite garbage that is never
    /// read (real rows mask them out as keys, and pooling only reads the last
    /// position, which is always real under left-padding).
    pub fn left_pad_mask(&self, pad_lens: &[usize], l: usize) -> Result<Tensor> {
        let b = pad_lens.len();
        let neg = f32::NEG_INFINITY;
        let mut data = Vec::with_capacity(b * l * l);
        for &pad in pad_lens {
            for i in 0..l {
                let query_is_pad = i < pad;
                for j in 0..l {
                    let causal_ok = j <= i;
                    // Real queries may only see real keys; pad queries fall back
                    // to causal-only so their row is never fully masked.
                    let key_ok = query_is_pad || j >= pad;
                    data.push(if causal_ok && key_ok { 0f32 } else { neg });
                }
            }
        }
        Tensor::from_vec(data, (b, 1, l, l), &self.device)?.to_dtype(self.dtype)
    }

    /// Rows of the tied LM head for `ids`.
    ///
    /// Qwen3-Reranker sets `tie_word_embeddings: true`, so its LM head *is* the
    /// input embedding matrix and no `lm_head` tensor exists in the checkpoint.
    /// A reranker only needs the logits of two tokens ("yes"/"no"), so instead
    /// of projecting the hidden state through all 151669 rows, take just those
    /// rows and do a 2-wide matmul. Same numbers, ~75000x less work.
    pub fn tied_head_rows(&self, ids: &[u32]) -> Result<Tensor> {
        let idx = Tensor::from_vec(ids.to_vec(), (ids.len(),), &self.device)?;
        self.embed_tokens.embeddings().index_select(&idx, 0)
    }

    /// Last-token pooling for a **left-padded** batch: the final column.
    ///
    /// Left-padding is what makes this a single slice — the last position is
    /// always a real token (the appended `<|endoftext|>`) for every row. This is
    /// the `left_padding` branch of the model card's `last_token_pool`.
    pub fn pool_last(&self, hidden: &Tensor) -> Result<Tensor> {
        let (_, l, _) = hidden.dims3()?;
        hidden.i((.., l - 1, ..))
    }
}
