//! Relations: the map. RELATE-style edges and traversal over the estate.
//!
//! An edge is `(from) -[verb]-> (to)` between any two ids in the estate —
//! documents, nodes, connectors, or bare concept ids. Storage is LSM-native:
//! every RELATE blind-puts two rows (outbound anchored on `from`, inbound
//! anchored on `to`), so traversal in either direction is a sorted prefix
//! scan and writes never read.
//!
//! This is the first half of the fusion law: **the map resolves the route,
//! the treasure answers.** [`Estate::traverse`] produces the neighborhood;
//! `ConnXRecall::routed_search` (store.rs) runs exact hybrid recall inside it.

use std::collections::{HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use rro_core::Result;

use crate::estate::{Batch, Estate};
use crate::keys::{self, CF_RELS, REL_IN, REL_OUT};

/// One directed edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Relation {
    /// Source id.
    pub from: String,
    /// Edge verb (e.g. `contains`, `references`, `belongs_to`).
    pub verb: String,
    /// Target id.
    pub to: String,
}

/// How far and along what a traversal walks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalSpec {
    /// Verbs to follow; empty = all verbs.
    pub verbs: Vec<String>,
    /// Follow outbound edges.
    pub outbound: bool,
    /// Follow inbound edges.
    pub inbound: bool,
    /// Maximum hops from the start set.
    pub depth: usize,
    /// Hard cap on visited ids (breadth-first, nearest hops first).
    pub limit: usize,
}

impl Default for TraversalSpec {
    fn default() -> Self {
        TraversalSpec {
            verbs: Vec::new(),
            outbound: true,
            inbound: true,
            depth: 2,
            limit: 10_000,
        }
    }
}

impl Estate {
    /// RELATE: `(from) -[verb]-> (to)`. Idempotent; two blind puts.
    pub fn relate(&self, from: &str, verb: &str, to: &str) -> Result<()> {
        let handle = self.db.cf(CF_RELS)?;
        let mut batch = Batch::new();
        batch.put_cf(handle, keys::rel_key(REL_OUT, from, verb, to), []);
        batch.put_cf(handle, keys::rel_key(REL_IN, to, verb, from), []);
        self.db.write(batch)
    }

    /// Remove one edge (both direction rows).
    pub fn unrelate(&self, from: &str, verb: &str, to: &str) -> Result<()> {
        let handle = self.db.cf(CF_RELS)?;
        let mut batch = Batch::new();
        batch.delete_cf(handle, keys::rel_key(REL_OUT, from, verb, to));
        batch.delete_cf(handle, keys::rel_key(REL_IN, to, verb, from));
        self.db.write(batch)
    }

    /// Outbound edges of `from` (optionally restricted to one verb).
    pub fn relations_out(&self, from: &str, verb: Option<&str>) -> Result<Vec<Relation>> {
        self.scan_rels(REL_OUT, from, verb)
    }

    /// Inbound edges of `to` (optionally restricted to one verb).
    pub fn relations_in(&self, to: &str, verb: Option<&str>) -> Result<Vec<Relation>> {
        self.scan_rels(REL_IN, to, verb)
    }

    fn scan_rels(&self, dir: u8, anchor: &str, verb: Option<&str>) -> Result<Vec<Relation>> {
        let handle = self.db.cf(CF_RELS)?;
        let prefix = match verb {
            Some(v) => keys::rel_verb_prefix(dir, anchor, v),
            None => keys::rel_prefix(dir, anchor),
        };
        // Decode always needs the anchor-level prefix (its suffix is verb\0other).
        let anchor_prefix_len = keys::rel_prefix(dir, anchor).len();

        let mut out = Vec::new();
        for item in self.db.iter_from(handle, &prefix) {
            let (k, _) = item?;
            if !k.starts_with(&prefix) {
                break;
            }
            if let Some((verb, other)) = keys::rel_suffix(&k, anchor_prefix_len) {
                out.push(match dir {
                    REL_OUT => Relation {
                        from: anchor.to_string(),
                        verb,
                        to: other,
                    },
                    _ => Relation {
                        from: other,
                        verb,
                        to: anchor.to_string(),
                    },
                });
            }
        }
        Ok(out)
    }

    /// Breadth-first traversal from `start` ids under `spec`. Returns every
    /// visited id (including the starts), nearest hops first, capped by
    /// `spec.limit`. This is the route the recall scope comes from.
    pub fn traverse(&self, start: &[&str], spec: &TraversalSpec) -> Result<Vec<String>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut order: Vec<String> = Vec::new();
        let mut frontier: VecDeque<(String, usize)> = VecDeque::new();

        for s in start {
            if visited.insert((*s).to_string()) {
                order.push((*s).to_string());
                frontier.push_back(((*s).to_string(), 0));
            }
        }

        let verb_ok = |v: &str| spec.verbs.is_empty() || spec.verbs.iter().any(|s| s == v);

        while let Some((id, hops)) = frontier.pop_front() {
            if hops >= spec.depth || order.len() >= spec.limit {
                continue;
            }
            let mut neighbors: Vec<String> = Vec::new();
            if spec.outbound {
                for r in self.relations_out(&id, None)? {
                    if verb_ok(&r.verb) {
                        neighbors.push(r.to);
                    }
                }
            }
            if spec.inbound {
                for r in self.relations_in(&id, None)? {
                    if verb_ok(&r.verb) {
                        neighbors.push(r.from);
                    }
                }
            }
            for n in neighbors {
                if order.len() >= spec.limit {
                    break;
                }
                if visited.insert(n.clone()) {
                    order.push(n.clone());
                    frontier.push_back((n, hops + 1));
                }
            }
        }
        Ok(order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relate_scan_unrelate_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let estate = Estate::open(dir.path(), "rels").unwrap();

        estate.relate("proj:a", "contains", "doc:1").unwrap();
        estate.relate("proj:a", "contains", "doc:2").unwrap();
        estate.relate("doc:1", "references", "doc:2").unwrap();

        let out = estate.relations_out("proj:a", None).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|r| r.verb == "contains"));

        let inbound = estate.relations_in("doc:2", None).unwrap();
        assert_eq!(inbound.len(), 2); // from proj:a and doc:1

        let only_refs = estate.relations_in("doc:2", Some("references")).unwrap();
        assert_eq!(only_refs.len(), 1);
        assert_eq!(only_refs[0].from, "doc:1");

        estate.unrelate("proj:a", "contains", "doc:2").unwrap();
        assert_eq!(estate.relations_out("proj:a", None).unwrap().len(), 1);
        assert_eq!(estate.relations_in("doc:2", None).unwrap().len(), 1);
    }

    #[test]
    fn traversal_respects_depth_verbs_and_limit() {
        let dir = tempfile::tempdir().unwrap();
        let estate = Estate::open(dir.path(), "walk").unwrap();

        // proj -> d1 -> d2 -> d3 (chain via `refs`), plus an off-verb edge.
        estate.relate("proj", "contains", "d1").unwrap();
        estate.relate("d1", "refs", "d2").unwrap();
        estate.relate("d2", "refs", "d3").unwrap();
        estate.relate("d1", "ignores", "x1").unwrap();

        let all2 = estate
            .traverse(&["proj"], &TraversalSpec::default())
            .unwrap();
        assert!(all2.contains(&"d2".to_string()));
        assert!(!all2.contains(&"d3".to_string()), "depth 2 stops before d3");

        let refs_only = estate
            .traverse(
                &["proj"],
                &TraversalSpec {
                    verbs: vec!["contains".into(), "refs".into()],
                    depth: 3,
                    ..TraversalSpec::default()
                },
            )
            .unwrap();
        assert!(refs_only.contains(&"d3".to_string()));
        assert!(!refs_only.contains(&"x1".to_string()), "verb filter holds");

        let capped = estate
            .traverse(
                &["proj"],
                &TraversalSpec {
                    depth: 5,
                    limit: 2,
                    ..TraversalSpec::default()
                },
            )
            .unwrap();
        assert_eq!(capped.len(), 2, "limit caps visited set");
    }
}
