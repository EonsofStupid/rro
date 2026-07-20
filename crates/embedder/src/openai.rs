//! An OpenAI-compatible `/v1/embeddings` client — the llama.cpp and vLLM backends.
//!
//! Both engines speak the same wire format, so one implementation serves both;
//! the kind is carried only for telemetry and defaults. This is the engine-agnostic
//! seam: whichever server holds the weights, RRO sees an [`Embedder`].
//!
//! Hand-rolled HTTP/1.1 over `tokio::net::TcpStream`, matching `rro-engine`'s
//! ops responder ("a deliberately tiny, zero-dependency HTTP/1.1 responder").
//! RRO has no reqwest/hyper/axum anywhere and this does not add one — a
//! retrieval engine that pulls in a TLS stack and an async HTTP framework to
//! POST one JSON body to localhost has bought complexity, not capability.
//!
//! **This is a gateway, and that is a real trade-off.** RRO's whole pitch is
//! embedded and tokio-native with no model gateway; a node using this backend
//! depends on an external process being up. It earns its place two ways: it lets
//! a thin node borrow a GPU node's model, and it is an independent oracle — if
//! candle and llama.cpp disagree on the same text, one of them is wrong, and
//! that is worth knowing before trusting either.

use std::time::Duration;

use async_trait::async_trait;
use rro_core::{Embedder, Embedding, Result, RroError};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Which server is behind the endpoint. Same protocol; different defaults and
/// telemetry labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiKind {
    /// llama.cpp `--embedding` server.
    LlamaCpp,
    /// vLLM OpenAI server.
    Vllm,
}

impl OpenAiKind {
    /// Telemetry name.
    pub fn as_str(&self) -> &'static str {
        match self {
            OpenAiKind::LlamaCpp => "llamacpp",
            OpenAiKind::Vllm => "vllm",
        }
    }
}

/// Config for an OpenAI-compatible embedding endpoint.
#[derive(Debug, Clone)]
pub struct OpenAiEmbedConfig {
    /// Full URL, e.g. `http://127.0.0.1:8090/v1/embeddings`.
    pub endpoint: String,
    /// Which server (telemetry + defaults).
    pub kind: OpenAiKind,
    /// `model` field in the request body. `None` = discover it from
    /// `/v1/models`.
    ///
    /// Engines disagree here: vLLM enforces the name (`The model 'local' does
    /// not exist` → 404) while llama.cpp ignores it entirely. Any hardcoded
    /// default is therefore wrong for one of them, so the default is to ask the
    /// server what it serves.
    pub model: Option<String>,
    /// Texts per HTTP request.
    pub batch: usize,
    /// Truncate + re-normalize to this dim (MRL). `None` = whatever the server
    /// returns.
    pub truncate_dim: Option<usize>,
    /// Instruction prefixed to queries; documents stay bare. `None` = the
    /// Qwen3 default. The endpoint embeds exactly the text we send — it applies
    /// no instruction of its own — so the asymmetry is ours to enforce.
    pub query_task: Option<String>,
    /// Per-request timeout.
    pub timeout: Duration,
}

impl OpenAiEmbedConfig {
    /// Config pointed at `endpoint`.
    pub fn new(endpoint: impl Into<String>, kind: OpenAiKind) -> Self {
        OpenAiEmbedConfig {
            endpoint: endpoint.into(),
            kind,
            model: None,
            batch: 32,
            truncate_dim: None,
            query_task: None,
            timeout: Duration::from_secs(120),
        }
    }
}

/// An embedder backed by an OpenAI-compatible HTTP endpoint.
#[derive(Debug)]
pub struct OpenAiEmbedder {
    cfg: OpenAiEmbedConfig,
    host: String,
    port: u16,
    path: String,
    /// The resolved model name actually sent on the wire.
    model: String,
    dim: usize,
    name: String,
}

impl OpenAiEmbedder {
    /// Connect, probe the endpoint for its native dimension, and return a ready
    /// embedder.
    ///
    /// The probe is not ceremony: [`Embedder::dim`] is sync and the estate sizes
    /// its vector space from it, so the dimension must be known before the first
    /// real call. Probing also fails fast at startup if the server is down,
    /// rather than at the first query.
    pub async fn connect(cfg: OpenAiEmbedConfig) -> Result<Self> {
        let (host, port, path) = parse_url(&cfg.endpoint)?;
        let model = match cfg.model.clone() {
            Some(m) => m,
            None => discover_model(&host, port, &path, cfg.timeout).await?,
        };
        let name = format!("{}-{}", cfg.kind.as_str(), short_model(&model));
        let mut me = OpenAiEmbedder {
            cfg,
            host,
            port,
            path,
            model,
            dim: 0,
            name,
        };
        let probe = me.request(&["dimension probe".to_string()]).await?;
        let native = probe
            .first()
            .map(|v| v.len())
            .ok_or_else(|| embed_err("endpoint returned no vector for the dimension probe"))?;

        me.dim = match me.cfg.truncate_dim {
            Some(d) if d == 0 || d > native => {
                return Err(embed_err(format!(
                    "truncate_dim {d} invalid: endpoint's native dim is {native}"
                )))
            }
            Some(d) => d,
            None => native,
        };
        Ok(me)
    }

    async fn encode(&self, texts: &[String], task: Option<&str>) -> Result<Vec<Embedding>> {
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
            for v in self.request(chunk).await? {
                let mut v = v;
                // Truncate before normalizing: slicing a unit vector leaves it
                // non-unit, and the estate's cosine path assumes unit length.
                v.truncate(self.dim.max(1));
                out.push(Embedding(v).normalized());
            }
        }
        Ok(out)
    }

    /// One POST. Returns raw vectors in request order.
    async fn request(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let body = serde_json::json!({ "input": texts, "model": self.model }).to_string();
        let req = format!(
            "POST {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.path,
            self.host,
            self.port,
            body.len(),
            body
        );

        // Connection: close, so read to EOF — no chunked/keep-alive parsing.
        let raw = http_roundtrip(&self.host, self.port, &req, self.cfg.timeout).await?;

        let text = String::from_utf8_lossy(&raw);
        let (head, json) = text
            .split_once("\r\n\r\n")
            .ok_or_else(|| embed_err("malformed HTTP response: no header/body split"))?;
        let status = head
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .unwrap_or("?");
        if status != "200" {
            return Err(embed_err(format!(
                "{} returned HTTP {status}: {}",
                self.cfg.endpoint,
                json.chars().take(300).collect::<String>()
            )));
        }

        parse_embeddings(json, texts.len(), &self.cfg.endpoint)
    }
}

/// Ask the server what it serves and take the first model id.
///
/// llama.cpp answers with the full gguf path; vLLM with its
/// `--served-model-name`. Either is correct for its own engine, which is the
/// point: discovery works for both without the caller knowing which is behind
/// the endpoint.
async fn discover_model(host: &str, port: u16, path: &str, timeout: Duration) -> Result<String> {
    // /v1/embeddings -> /v1/models
    let models_path = match path.rfind('/') {
        Some(i) => format!("{}/models", &path[..i]),
        None => "/v1/models".to_string(),
    };
    let req =
        format!("GET {models_path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
    let raw = http_roundtrip(host, port, &req, timeout).await?;
    let text = String::from_utf8_lossy(&raw);
    let json = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b)
        .ok_or_else(|| embed_err("malformed /v1/models response"))?;
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| embed_err(format!("parse /v1/models: {e}")))?;
    // OpenAI shape: {"data":[{"id":...}]}. llama.cpp also exposes {"models":[{"name":...}]}.
    let id = v
        .get("data")
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
        .ok_or_else(|| {
            embed_err(format!(
                "could not discover a model from {host}:{port}{models_path} — set the model \
                 explicitly (RRO_EMBEDDER_MODEL)"
            ))
        })?;
    Ok(id.to_string())
}

/// Trim a long model id (llama.cpp returns a full path) for telemetry.
fn short_model(m: &str) -> String {
    m.rsplit('/').next().unwrap_or(m).to_string()
}

/// One request/response over a fresh connection (`Connection: close`).
async fn http_roundtrip(host: &str, port: u16, req: &str, timeout: Duration) -> Result<Vec<u8>> {
    tokio::time::timeout(timeout, async {
        let mut stream = TcpStream::connect((host, port))
            .await
            .map_err(|e| embed_err(format!("connect {host}:{port}: {e}")))?;
        stream
            .write_all(req.as_bytes())
            .await
            .map_err(|e| embed_err(format!("write: {e}")))?;
        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .await
            .map_err(|e| embed_err(format!("read: {e}")))?;
        Ok::<_, RroError>(buf)
    })
    .await
    .map_err(|_| embed_err(format!("{host}:{port} timed out after {timeout:?}")))?
}

/// Pull `data[].embedding` out, preserving request order.
fn parse_embeddings(json: &str, want: usize, endpoint: &str) -> Result<Vec<Vec<f32>>> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| embed_err(format!("parse response: {e}")))?;
    let data = v
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| embed_err(format!("{endpoint}: response has no `data` array")))?;

    // The spec allows out-of-order data with an `index` field; sort rather than
    // trust arrival order, or vectors silently attach to the wrong text.
    let mut rows: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
    for (i, item) in data.iter().enumerate() {
        let idx = item
            .get("index")
            .and_then(|x| x.as_u64())
            .unwrap_or(i as u64) as usize;
        let emb = item
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| embed_err(format!("{endpoint}: data[{i}] has no `embedding`")))?
            .iter()
            .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
            .collect::<Vec<f32>>();
        if emb.iter().any(|x| !x.is_finite()) {
            return Err(embed_err(format!(
                "{endpoint}: data[{i}] contains a non-finite value"
            )));
        }
        rows.push((idx, emb));
    }
    rows.sort_by_key(|(i, _)| *i);

    if rows.len() != want {
        return Err(embed_err(format!(
            "{endpoint}: asked for {want} embeddings, got {}",
            rows.len()
        )));
    }
    Ok(rows.into_iter().map(|(_, v)| v).collect())
}

/// Split `http://host:port/path` into parts. Deliberately minimal: this talks to
/// a local inference server, so no TLS and no redirects.
fn parse_url(url: &str) -> Result<(String, u16, String)> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| embed_err(format!("endpoint must start with http:// — got `{url}`")))?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/v1/embeddings"),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (
            h.to_string(),
            p.parse::<u16>()
                .map_err(|_| embed_err(format!("bad port in `{url}`")))?,
        ),
        None => (authority.to_string(), 80),
    };
    if host.is_empty() {
        return Err(embed_err(format!("no host in `{url}`")));
    }
    Ok((host, port, path.to_string()))
}

fn embed_err(msg: impl Into<String>) -> RroError {
    RroError::Embed(msg.into())
}

#[async_trait]
impl Embedder for OpenAiEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    /// Bare text — the document path, same rationale as the candle backend:
    /// prefixing a document is the harmful direction, so the unqualified call
    /// takes the safe one.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.encode(texts, None).await
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.encode(texts, None).await
    }

    async fn embed_queries(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let task = self
            .cfg
            .query_task
            .as_deref()
            .unwrap_or(crate::DEFAULT_QUERY_TASK);
        self.encode(texts, Some(task)).await
    }

    fn model_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_parsing() {
        let (h, p, path) = parse_url("http://127.0.0.1:8090/v1/embeddings").unwrap();
        assert_eq!(
            (h.as_str(), p, path.as_str()),
            ("127.0.0.1", 8090, "/v1/embeddings")
        );

        let (h, p, path) = parse_url("http://localhost:8092").unwrap();
        assert_eq!(
            (h.as_str(), p, path.as_str()),
            ("localhost", 8092, "/v1/embeddings")
        );

        assert!(parse_url("https://x/y").is_err(), "TLS is not supported");
        assert!(parse_url("127.0.0.1:8090").is_err(), "scheme required");
        assert!(parse_url("http://host:notaport/x").is_err());
    }

    #[test]
    fn response_parsing_respects_index_order() {
        // Deliberately out of order: index must win over arrival order.
        let json = r#"{"data":[
            {"index":1,"embedding":[0.0,1.0]},
            {"index":0,"embedding":[1.0,0.0]}
        ]}"#;
        let got = parse_embeddings(json, 2, "test").unwrap();
        assert_eq!(got[0], vec![1.0, 0.0], "index 0 must come first");
        assert_eq!(got[1], vec![0.0, 1.0]);
    }

    #[test]
    fn response_count_mismatch_is_an_error() {
        let json = r#"{"data":[{"index":0,"embedding":[1.0]}]}"#;
        assert!(parse_embeddings(json, 2, "test").is_err());
    }

    #[test]
    fn non_finite_values_are_rejected() {
        // A NaN reaching the estate poisons cosine silently; catch it at the wire.
        let json = r#"{"data":[{"index":0,"embedding":["not-a-number"]}]}"#;
        assert!(parse_embeddings(json, 1, "test").is_err());
    }
}
