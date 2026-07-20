//! The ops HTTP surface: `/metrics` (prometheus text), `/healthz`,
//! `/livez`, `/readyz` — a deliberately tiny, zero-dependency HTTP/1.1
//! responder (GET only, one request per connection). Scrapers and probes
//! need nothing fancier, and the a2a plane stays the real API.

use std::sync::Arc;

use rro_core::{Result, RroError};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

/// Serve the ops endpoints for `estate` until the task is dropped.
/// Returns the bound address (bind `127.0.0.1:0` for an OS port).
pub async fn serve_ops(
    addr: &str,
    estate: Arc<connxism::Estate>,
) -> Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|e| RroError::Net(format!("ops bind: {e}")))?;
    let local = listener
        .local_addr()
        .map_err(|e| RroError::Net(format!("ops local_addr: {e}")))?;
    let started = std::time::Instant::now();

    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let estate = estate.clone();
            tokio::spawn(async move {
                let _ = answer(stream, &estate, started).await;
            });
        }
    });
    Ok((local, task))
}

async fn answer(
    stream: TcpStream,
    estate: &connxism::Estate,
    started: std::time::Instant,
) -> std::io::Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut lines = BufReader::new(read_half).lines();

    // Request line, then drain headers to the blank line.
    let request = lines.next_line().await?.unwrap_or_default();
    while let Some(h) = lines.next_line().await? {
        if h.trim().is_empty() {
            break;
        }
    }
    let mut parts = request.split_whitespace();
    let (method, path) = (
        parts.next().unwrap_or(""),
        parts.next().unwrap_or("/").split('?').next().unwrap_or("/"),
    );

    let (status, content_type, body) = if method != "GET" {
        (
            "405 Method Not Allowed",
            "text/plain",
            "GET only\n".to_string(),
        )
    } else {
        match path {
            "/healthz" | "/livez" | "/readyz" => ("200 OK", "text/plain", "ok\n".to_string()),
            "/metrics" => match render_metrics(estate, started) {
                Ok(text) => ("200 OK", "text/plain; version=0.0.4", text),
                Err(e) => ("500 Internal Server Error", "text/plain", format!("{e}\n")),
            },
            _ => ("404 Not Found", "text/plain", "not found\n".to_string()),
        }
    };

    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    write_half.write_all(response.as_bytes()).await?;
    write_half.shutdown().await
}

/// Prometheus text exposition (format 0.0.4) from the estate's health
/// snapshot — gauges only, all prefixed `rro_`.
fn render_metrics(estate: &connxism::Estate, started: std::time::Instant) -> Result<String> {
    let h = estate.health()?;
    let mut out = String::with_capacity(512);
    fn gauge(out: &mut String, name: &str, help: &str, value: f64) {
        out.push_str(&format!(
            "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}\n"
        ));
    }
    gauge(
        &mut out,
        "rro_uptime_seconds",
        "Ops listener uptime.",
        started.elapsed().as_secs_f64(),
    );
    gauge(
        &mut out,
        "rro_docs_total",
        "Indexed documents.",
        h.docs as f64,
    );
    gauge(
        &mut out,
        "rro_feed_seq",
        "Next changefeed sequence.",
        h.feed_seq as f64,
    );
    gauge(
        &mut out,
        "rro_applier_backlog",
        "Graph ops awaiting the out-of-band applier.",
        h.applier_backlog as f64,
    );
    gauge(
        &mut out,
        "rro_collections_total",
        "Named collections.",
        h.collections.len() as f64,
    );
    for (name, count) in &h.collections {
        out.push_str(&format!(
            "rro_collection_docs{{collection=\"{name}\"}} {count}\n"
        ));
    }
    gauge(
        &mut out,
        "rro_issues_total",
        "Self-reported operational issues (default threshold).",
        estate.issues(10_000)?.len() as f64,
    );
    Ok(out)
}
