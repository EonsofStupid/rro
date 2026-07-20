//! Retrieval strategies beyond one-query-one-list: grouped results,
//! recommendation by example, context-steered discovery, and batches.
//!
//! All of these compose the same primitives (the typed query, dense search,
//! the durable vector column family) — no parallel machinery to keep honest.

use std::collections::{HashMap, HashSet};

use rro_core::{Candidate, Embedding, EstateQuery, Recall as _, Result, RroError};

use crate::estate::Db;
use crate::keys::{self, CF_VECS};
use crate::store::ConnXRecall;

/// How hard grouped and steered searches over-fetch before regrouping.
const STRATEGY_OVERFETCH: usize = 4;

/// One group of results sharing a payload field value.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Group {
    /// The shared value of the group-by field (rendered as text).
    pub key: String,
    /// The group's hits, best first (at most `group_size`).
    pub hits: Vec<Candidate>,
}

/// Fetch one stored (full-precision) vector by document id.
fn stored_vec(db: &Db, id: &str) -> Result<Option<Embedding>> {
    let cf = db.cf(CF_VECS)?;
    Ok(db
        .get_cf(cf, id.as_bytes())?
        .map(|b| Embedding(keys::decode_vec(&b))))
}

impl ConnXRecall {
    /// Grouped search: run `q`, then keep at most `group_size` hits for each
    /// distinct value of `group_by`, returning up to `groups` groups ordered
    /// by their best hit. Documents without the field are skipped.
    pub async fn query_grouped(
        &self,
        mut q: EstateQuery,
        group_by: &str,
        groups: usize,
        group_size: usize,
    ) -> Result<Vec<Group>> {
        if groups == 0 || group_size == 0 {
            return Ok(Vec::new());
        }
        q.top_k = groups
            .saturating_mul(group_size)
            .saturating_mul(STRATEGY_OVERFETCH);
        q.with_payload = true;
        let mut hits = self.query(q).await?;

        // Lexical-only candidates may carry empty payloads; the group key
        // needs real metadata.
        for c in hits.iter_mut() {
            if c.metadata.is_empty() {
                if let Some(doc) = self.doc(c.id.as_str()).await? {
                    c.metadata = doc.metadata;
                    if c.text.is_empty() {
                        c.text = doc.text;
                    }
                }
            }
        }

        let mut order: Vec<String> = Vec::new();
        let mut buckets: HashMap<String, Vec<Candidate>> = HashMap::new();
        for c in hits {
            let Some(value) = c.metadata.get(group_by) else {
                continue;
            };
            let key = match value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let bucket = buckets.entry(key.clone()).or_default();
            if bucket.is_empty() {
                order.push(key);
            }
            if bucket.len() < group_size {
                bucket.push(c);
            }
        }

        // Hits arrive best-first, so first appearance == best hit: `order`
        // already ranks groups by their best member.
        Ok(order
            .into_iter()
            .take(groups)
            .map(|key| {
                let hits = buckets.remove(&key).unwrap_or_default();
                Group { key, hits }
            })
            .collect())
    }

    /// Recommend by example: steer toward the average of the `positive`
    /// documents' vectors and away from the `negative` ones, excluding the
    /// examples themselves from the results.
    pub async fn recommend(
        &self,
        positive: &[String],
        negative: &[String],
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let db = self.db.clone();
        let pos: Vec<String> = positive.to_vec();
        let neg: Vec<String> = negative.to_vec();
        let target = tokio::task::spawn_blocking(move || -> Result<Embedding> {
            let mut acc: Vec<f32> = Vec::new();
            let mut found_pos = 0usize;
            for id in &pos {
                if let Some(v) = stored_vec(&db, id)? {
                    let v = v.normalized();
                    if acc.is_empty() {
                        acc = vec![0.0; v.dim()];
                    }
                    for (a, x) in acc.iter_mut().zip(v.as_slice()) {
                        *a += x;
                    }
                    found_pos += 1;
                }
            }
            if found_pos == 0 {
                return Err(RroError::Recall(
                    "recommend: no positive example vectors found".into(),
                ));
            }
            for id in &neg {
                if let Some(v) = stored_vec(&db, id)? {
                    let v = v.normalized();
                    for (a, x) in acc.iter_mut().zip(v.as_slice()) {
                        *a -= x;
                    }
                }
            }
            Ok(Embedding(acc).normalized())
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))??;

        let exclude: HashSet<&str> = positive
            .iter()
            .chain(negative)
            .map(String::as_str)
            .collect();
        let mut hits = self.search(&target, top_k + exclude.len()).await?;
        hits.retain(|c| !exclude.contains(c.id.as_str()));
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Discover: rank candidates near `query` by how many context `pairs`
    /// (positive id, negative id) they agree with — a candidate "agrees"
    /// when it sits closer to the pair's positive than its negative.
    /// Ordering is (agreement, similarity); ties fall back to the dense score.
    pub async fn discover(
        &self,
        query: &Embedding,
        pairs: &[(String, String)],
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let mut hits = self
            .search(query, top_k.saturating_mul(STRATEGY_OVERFETCH))
            .await?;
        if pairs.is_empty() {
            hits.truncate(top_k);
            return Ok(hits);
        }

        let db = self.db.clone();
        let pairs: Vec<(String, String)> = pairs.to_vec();
        let ids: Vec<String> = hits.iter().map(|c| c.id.as_str().to_string()).collect();
        let agreement = tokio::task::spawn_blocking(move || -> Result<HashMap<String, i64>> {
            let mut ctx: Vec<(Embedding, Embedding)> = Vec::with_capacity(pairs.len());
            for (p, n) in &pairs {
                let (Some(pv), Some(nv)) = (stored_vec(&db, p)?, stored_vec(&db, n)?) else {
                    continue; // unknown example ids don't vote
                };
                ctx.push((pv, nv));
            }
            let mut out = HashMap::new();
            for id in &ids {
                let Some(v) = stored_vec(&db, id)? else {
                    continue;
                };
                let score: i64 = ctx
                    .iter()
                    .map(|(p, n)| if v.cosine(p) > v.cosine(n) { 1 } else { -1 })
                    .sum();
                out.insert(id.clone(), score);
            }
            Ok(out)
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))??;

        hits.sort_by(|a, b| {
            let aa = agreement.get(a.id.as_str()).copied().unwrap_or(i64::MIN);
            let bb = agreement.get(b.id.as_str()).copied().unwrap_or(i64::MIN);
            bb.cmp(&aa)
                .then_with(|| b.score.total_cmp(&a.score))
                .then_with(|| a.id.as_str().cmp(b.id.as_str()))
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Execute a batch of typed queries. v1 runs them sequentially — the
    /// win is one wire round-trip, not (yet) parallel execution.
    pub async fn query_batch(&self, queries: Vec<EstateQuery>) -> Result<Vec<Vec<Candidate>>> {
        let mut out = Vec::with_capacity(queries.len());
        for q in queries {
            out.push(self.query(q).await?);
        }
        Ok(out)
    }
}
