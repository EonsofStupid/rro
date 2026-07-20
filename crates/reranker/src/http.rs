//! Cross-encoder reranking over an OpenAI-style `/rerank` endpoint — the
//! llama.cpp and vLLM reranker backends.
//!
//! Same shape as the embedder's HTTP client: hand-rolled HTTP/1.1 on tokio, no
//! reqwest/hyper. See `embedder::openai` for why RRO does not take an HTTP
//! framework dependency to POST one JSON body to localhost.
//!
//! **The two engines return different score scales**, verified live:
//!
//! ```text
//! vLLM     :8092 /rerank      relevance_score 1.0        vs 7.2e-06   (normalized)
//! llama.cpp:8093 /v1/rerank   relevance_score 18.68      vs -11.89    (raw logits)
//! ```
//!
//! Both rank identically, which is all a reranker owes: [`Reranker::rerank`] is
//! defined by the ORDER it returns. Nothing downstream may threshold on the raw
//! value without knowing the engine — so the scores are carried through as-is
//! and the contract stays "sorted, best first".

use std::time::Duration;

use async_trait::async_trait;
use rro_core::{Candidate, Reranker, Result, RroError};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Which reranker server is behind the endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpRerankKind {
    /// llama.cpp `--reranking` server (`/v1/rerank`), raw-logit scores.
    LlamaCpp,
    /// vLLM `/rerank`, normalized scores.
    Vllm,
}

impl HttpRerankKind {
    /// Telemetry name.
    pub fn as_str(&self) -> &'static str {
        match self {
            HttpRerankKind::LlamaCpp => "llamacpp",
            HttpRerankKind::Vllm => "vllm",
        }
    }

    /// The path each engine serves rerank on. They differ: llama.cpp mounts it
    /// under `/v1`, vLLM at the root.
    pub fn default_path(&self) -> &'static str {
        match self {
            HttpRerankKind::LlamaCpp => "/v1/rerank",
            HttpRerankKind::Vllm => "/rerank",
        }
    }
}

/// Config for an HTTP cross-encoder reranker.
#[derive(Debug, Clone)]
pub struct HttpRerankConfig {
    /// Full URL, e.g. `http://127.0.0.1:8093/v1/rerank`.
    pub endpoint: String,
    /// Which server.
    pub kind: HttpRerankKind,
    /// `model` field. vLLM enforces this; llama.cpp ignores it. `None` =
    /// discover from `/v1/models`.
    pub model: Option<String>,
    /// Candidates per request. Rerank cost is O(candidates), so this bounds the
    /// per-request work rather than the concurrency.
    pub batch: usize,
    /// Per-request timeout.
    pub timeout: Duration,
}

impl HttpRerankConfig {
    /// Config pointed at `endpoint`.
    pub fn new(endpoint: impl Into<String>, kind: HttpRerankKind) -> Self {
        HttpRerankConfig {
            endpoint: endpoint.into(),
            kind,
            model: None,
            batch: 64,
            timeout: Duration::from_secs(120),
        }
    }
}

/// A cross-encoder reranker backed by an HTTP endpoint.
#[derive(Debug)]
pub struct HttpReranker {
    cfg: HttpRerankConfig,
    host: String,
    port: u16,
    path: String,
    model: String,
    name: String,
}

impl HttpReranker {
    /// Resolve the model name and verify the endpoint answers.
    pub async fn connect(cfg: HttpRerankConfig) -> Result<Self> {
        let (host, port, path) = parse_url(&cfg.endpoint, cfg.kind.default_path())?;
        let model = match cfg.model.clone() {
            Some(m) => m,
            None => discover_model(&host, port, cfg.timeout).await?,
        };
        let name = format!(
            "{}-{}",
            cfg.kind.as_str(),
            model.rsplit('/').next().unwrap_or(&model)
        );
        let me = HttpReranker {
            cfg,
            host,
            port,
            path,
            model,
            name,
        };
        // Fail at startup, not at the first query.
        me.score("probe", &["probe document".to_string()]).await?;
        Ok(me)
    }

    /// Relevance for each document, in input order.
    async fn score(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": docs,
        })
        .to_string();
        let req = format!(
            "POST {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.path,
            self.host,
            self.port,
            body.len(),
            body
        );
        let raw = roundtrip(&self.host, self.port, &req, self.cfg.timeout).await?;
        let text = String::from_utf8_lossy(&raw);
        let (head, json) = text
            .split_once("\r\n\r\n")
            .ok_or_else(|| rerank_err("malformed HTTP response"))?;
        let status = head
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .unwrap_or("?");
        if status != "200" {
            return Err(rerank_err(format!(
                "{} returned HTTP {status}: {}",
                self.cfg.endpoint,
                json.chars().take(300).collect::<String>()
            )));
        }
        parse_rerank(json, docs.len(), &self.cfg.endpoint)
    }
}

/// Pull `results[].relevance_score` out, restoring input order via `index`.
fn parse_rerank(json: &str, want: usize, endpoint: &str) -> Result<Vec<f32>> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| rerank_err(format!("parse response: {e}")))?;
    let results = v
        .get("results")
        .and_then(|r| r.as_array())
        .ok_or_else(|| rerank_err(format!("{endpoint}: response has no `results` array")))?;

    // The endpoint may return results already sorted by relevance. Restore
    // INPUT order here: the caller pairs scores back to its own candidate list
    // by position, so trusting arrival order silently mis-assigns every score.
    let mut out = vec![f32::NAN; want];
    for (i, item) in results.iter().enumerate() {
        let idx = item
            .get("index")
            .and_then(|x| x.as_u64())
            .unwrap_or(i as u64) as usize;
        let score = item
            .get("relevance_score")
            .or_else(|| item.get("score"))
            .and_then(|x| x.as_f64())
            .ok_or_else(|| rerank_err(format!("{endpoint}: results[{i}] has no relevance_score")))?
            as f32;
        if idx >= want {
            return Err(rerank_err(format!(
                "{endpoint}: results[{i}].index {idx} out of range for {want} documents"
            )));
        }
        out[idx] = score;
    }
    if let Some(i) = out.iter().position(|s| s.is_nan()) {
        return Err(rerank_err(format!(
            "{endpoint}: no score returned for document {i}"
        )));
    }
    Ok(out)
}

async fn discover_model(host: &str, port: u16, timeout: Duration) -> Result<String> {
    let req =
        format!("GET /v1/models HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
    let raw = roundtrip(host, port, &req, timeout).await?;
    let text = String::from_utf8_lossy(&raw);
    let json = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b)
        .ok_or_else(|| rerank_err("malformed /v1/models response"))?;
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| rerank_err(format!("parse /v1/models: {e}")))?;
    v.get("data")
        .and_then(|d| d.as_array())
        .and_then(|a| a.first())
        .and_then(|m| m.get("id"))
        .and_then(|s| s.as_str())
        .or_else(|| {
            v.get("models")
                .and_then(|d| d.as_array())
                .and_then(|a| a.first())
                .and_then(|m| m.get("name").or_else(|| m.get("id")))
                .and_then(|s| s.as_str())
        })
        .map(|s| s.to_string())
        .ok_or_else(|| {
            rerank_err(format!(
                "could not discover a model from {host}:{port}/v1/models — set it explicitly"
            ))
        })
}

async fn roundtrip(host: &str, port: u16, req: &str, timeout: Duration) -> Result<Vec<u8>> {
    tokio::time::timeout(timeout, async {
        let mut stream = TcpStream::connect((host, port))
            .await
            .map_err(|e| rerank_err(format!("connect {host}:{port}: {e}")))?;
        stream
            .write_all(req.as_bytes())
            .await
            .map_err(|e| rerank_err(format!("write: {e}")))?;
        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .await
            .map_err(|e| rerank_err(format!("read: {e}")))?;
        Ok::<_, RroError>(buf)
    })
    .await
    .map_err(|_| rerank_err(format!("{host}:{port} timed out after {timeout:?}")))?
}

fn parse_url(url: &str, default_path: &str) -> Result<(String, u16, String)> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| rerank_err(format!("endpoint must start with http:// — got `{url}`")))?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, default_path),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (
            h.to_string(),
            p.parse::<u16>()
                .map_err(|_| rerank_err(format!("bad port in `{url}`")))?,
        ),
        None => (authority.to_string(), 80),
    };
    if host.is_empty() {
        return Err(rerank_err(format!("no host in `{url}`")));
    }
    Ok((host, port, path.to_string()))
}

fn rerank_err(msg: impl Into<String>) -> RroError {
    RroError::Rerank(msg.into())
}

#[async_trait]
impl Reranker for HttpReranker {
    async fn rerank(
        &self,
        query: &str,
        mut candidates: Vec<Candidate>,
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        if candidates.is_empty() {
            return Ok(candidates);
        }
        // Score in chunks, writing each score back onto its own candidate.
        for chunk in candidates.chunks_mut(self.cfg.batch.max(1)) {
            let docs: Vec<String> = chunk.iter().map(|c| c.text.clone()).collect();
            let scores = self.score(query, &docs).await?;
            for (c, s) in chunk.iter_mut().zip(scores) {
                c.score = s;
            }
        }
        // Descending relevance. total_cmp, not partial_cmp().unwrap(): a NaN
        // from a misbehaving endpoint should not panic the engine mid-query.
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        candidates.truncate(top_k);
        Ok(candidates)
    }

    fn model_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_restores_input_order() {
        // Endpoint returned best-first; index must restore input order.
        let json = r#"{"results":[
            {"index":1,"relevance_score":9.0},
            {"index":0,"relevance_score":1.0}
        ]}"#;
        let got = parse_rerank(json, 2, "t").unwrap();
        assert_eq!(got, vec![1.0, 9.0], "scores must land on their own inputs");
    }

    #[test]
    fn vllm_and_llamacpp_score_shapes_both_parse() {
        // vLLM: normalized. llama.cpp: raw logits, including negatives.
        let vllm = r#"{"results":[{"index":0,"relevance_score":1.0},{"index":1,"relevance_score":7.2e-06}]}"#;
        assert_eq!(parse_rerank(vllm, 2, "t").unwrap(), vec![1.0, 7.2e-06]);
        let llama = r#"{"results":[{"index":0,"relevance_score":18.68},{"index":1,"relevance_score":-11.89}]}"#;
        assert_eq!(parse_rerank(llama, 2, "t").unwrap(), vec![18.68, -11.89]);
    }

    #[test]
    fn missing_or_out_of_range_scores_are_errors() {
        assert!(
            parse_rerank(r#"{"results":[{"index":0,"relevance_score":1.0}]}"#, 2, "t").is_err()
        );
        assert!(
            parse_rerank(r#"{"results":[{"index":5,"relevance_score":1.0}]}"#, 1, "t").is_err()
        );
    }

    #[test]
    fn default_paths_differ_per_engine() {
        assert_eq!(HttpRerankKind::LlamaCpp.default_path(), "/v1/rerank");
        assert_eq!(HttpRerankKind::Vllm.default_path(), "/rerank");
        let (_, p, path) = parse_url("http://127.0.0.1:8092", "/rerank").unwrap();
        assert_eq!((p, path.as_str()), (8092, "/rerank"));
    }
}
