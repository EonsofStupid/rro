//! # connectors — drivers and the sync engine
//!
//! An operator shares a **connector** (a third-party source: files, feeds,
//! mailboxes, databases). A [`Driver`] pulls its content in **resumable
//! batches** behind a cursor; [`sync`] runs the full estate-side pipeline for
//! each batch:
//!
//! ```text
//! driver.pull(cursor) ─▶ RRD distill (mode + tags, gate ladder)
//!                      ─▶ recall.upsert (durable + indexed)
//!                      ─▶ RELATE connector ─contains→ doc
//!                      ─▶ estate.tag(...) from the RRO
//!                      ─▶ cursor advance (durable in SyncState)
//! ```
//!
//! The cursor advances only after the batch is durably ingested, so an
//! interrupted sync resumes exactly where it stopped — no duplicates, no
//! gaps. Every batch is evented (`connector.batch`, `connector.synced`).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod fs;
mod jsonl;

pub use fs::FsDriver;
pub use jsonl::JsonlDriver;

use async_trait::async_trait;
use connxism::{ConnXRecall, Estate, SyncState, SyncStatus};
use rrd::{GateVerdict, Rrd, SourceStamp};
use rro_core::{Document, Embedder, Recall, Result, RroError};

/// Estate key the RRD shape baseline persists under.
const BASELINE_KEY: &str = "rrd:baseline";

/// One resumable pull from a source.
pub struct Batch {
    /// Documents in this batch (empty = source drained at this cursor).
    pub docs: Vec<Document>,
    /// Cursor to persist once this batch is durably ingested; `None` when the
    /// source is fully drained.
    pub next_cursor: Option<String>,
}

/// A connector driver: pull content in resumable batches.
#[async_trait]
pub trait Driver: Send + Sync {
    /// Provider slug for the connector registry (e.g. `fs`, `jsonl`).
    fn provider(&self) -> &str;

    /// Pull the next batch after `cursor` (`None` = from the beginning).
    async fn pull(&self, cursor: Option<&str>) -> Result<Batch>;
}

/// Outcome of one [`sync`] run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncReport {
    /// Documents ingested by this run.
    pub ingested: u64,
    /// Documents blocked by the RRD gate ladder **before** any embedding
    /// cost was paid.
    pub blocked: u64,
    /// Batches pulled.
    pub batches: u64,
    /// Final cursor (persisted in the connector's [`SyncState`]).
    pub cursor: Option<String>,
}

/// Run one full sync pass for `connector_id`, resuming from its stored
/// cursor. Each document flows through the whole engine: RRD distills it
/// (mode + tags + gate verdict), recall ingests it, the estate RELATEs it to
/// its connector and tags it from the RRO. The cursor advances **after** the
/// batch is durable.
pub async fn sync(
    estate: &Estate,
    recall: &ConnXRecall,
    rrd: &Rrd,
    embedder: &dyn Embedder,
    driver: &dyn Driver,
    connector_id: &str,
) -> Result<SyncReport> {
    let conn = estate
        .connector(connector_id)?
        .ok_or_else(|| RroError::msg(format!("no such connector: {connector_id}")))?;

    // The baseline persists in the estate and grows across sessions: restore
    // it into a fresh Rrd before the first observation of this run.
    if rrd.baseline_observations() == 0 {
        if let Some(snap) = estate.get_component_json::<rrd::BaselineSnapshot>(BASELINE_KEY)? {
            rrd.restore_baseline(snap);
        }
    }

    let mut cursor = conn.sync.cursor.clone();
    let mut docs_synced = conn.sync.docs_synced;
    let mut ingested = 0u64;
    let mut blocked = 0u64;
    let mut batches = 0u64;

    estate.update_sync(
        connector_id,
        SyncState {
            cursor: cursor.clone(),
            docs_synced,
            last_sync: conn.sync.last_sync,
            status: SyncStatus::Syncing,
        },
    )?;

    loop {
        let batch = driver.pull(cursor.as_deref()).await?;
        if batch.docs.is_empty() {
            break;
        }
        batches += 1;

        // RRD IS FIRST — the instant a payload arrives: stamp, gate ladder,
        // shape/mode, baseline prediction. All of it BEFORE any embedding
        // cost is paid; blocked payloads never reach the model.
        let mut kept: Vec<(&Document, rrd::Rro)> = Vec::with_capacity(batch.docs.len());
        for doc in &batch.docs {
            let stamp = SourceStamp {
                channel: Some(connector_id.to_string()),
                source: Some(doc.id.as_str().to_string()),
                ..SourceStamp::default()
            };
            let rro = rrd.distill_stamped(doc.id.as_str(), &doc.text, &doc.metadata, None, stamp);
            if rro.gate == GateVerdict::Block {
                blocked += 1;
                continue;
            }
            kept.push((doc, rro));
        }
        if kept.is_empty() {
            cursor = batch.next_cursor.clone();
            if cursor.is_none() {
                break;
            }
            continue;
        }

        // Embed only the survivors; the same vectors serve recall AND the
        // L2 tag routing (post-embed half of the split distill).
        let texts: Vec<String> = kept.iter().map(|(d, _)| d.text.clone()).collect();
        let embeddings = embedder.embed_documents(&texts).await?;

        let mut records = Vec::with_capacity(kept.len());
        let mut post = Vec::with_capacity(kept.len()); // (doc_id, tags)
        for ((doc, rro), emb) in kept.iter().zip(&embeddings) {
            let mut tags: Vec<String> = rrd.route_tags(emb).into_iter().map(|t| t.tag).collect();
            tags.push(format!("mode:{}", rro.mode.name()));
            post.push((doc.id.as_str().to_string(), tags));

            let mut r = rro_core::VectorRecord::new(doc.id.clone(), emb.clone(), doc.text.clone());
            r.metadata = doc.metadata.clone();
            records.push(r);
        }

        // Durable ingest first…
        recall.upsert(records).await?;
        ingested += post.len() as u64;
        docs_synced += post.len() as u64;

        // …then the map: provenance edges + tags from the RROs.
        for (doc_id, tags) in &post {
            estate.relate(connector_id, "contains", doc_id)?;
            estate.tag(doc_id, tags)?;
        }

        // …and only now the cursor. A crash before this line replays the
        // batch (idempotent upserts), never skips it.
        cursor = batch.next_cursor.clone();
        estate.update_sync(
            connector_id,
            SyncState {
                cursor: cursor.clone(),
                docs_synced,
                last_sync: Some(connxism::now_ms()),
                status: SyncStatus::Syncing,
            },
        )?;
        rro_core::events::emit(
            "connector.batch",
            serde_json::json!({
                "connector": connector_id,
                "docs": post.len(),
                "cursor": cursor,
            }),
        );

        if cursor.is_none() {
            break;
        }
    }

    estate.update_sync(
        connector_id,
        SyncState {
            cursor: cursor.clone(),
            docs_synced,
            last_sync: Some(connxism::now_ms()),
            status: SyncStatus::Idle,
        },
    )?;
    estate.record_trend(
        &format!("connector.{connector_id}.docs_synced"),
        docs_synced as f64,
    )?;
    estate.record_trend(
        &format!("connector.{connector_id}.predictability"),
        rrd.predictability(connector_id),
    )?;

    // Commit the evolved baseline back to the estate: the snapshot IS the
    // durable "this is normal" the next session restores and grows.
    estate.put_component_json(BASELINE_KEY, &rrd.baseline_snapshot())?;

    rro_core::events::emit(
        "connector.synced",
        serde_json::json!({
            "connector": connector_id,
            "ingested": ingested,
            "blocked": blocked,
            "batches": batches,
            "predictability": rrd.predictability(connector_id),
            "hit_rate": rrd.hit_rate(connector_id),
        }),
    );

    Ok(SyncReport {
        ingested,
        blocked,
        batches,
        cursor,
    })
}
