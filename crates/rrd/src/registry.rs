//! The shape registry: the sliver lattice.
//!
//! Modes are the base shapes; every observed shape attaches beneath its mode
//! as a **sliver** — a thin specialization. Shapes *evolve*: a drifted
//! payload (field added/removed) is a new sliver beside its sibling, never a
//! silent reuse of the old plan.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::mode::Mode;
use crate::shape::ShapeFingerprint;

/// A sliver id derived from the shape's canonical key — the same shape gets the
/// same id in every process, forever.
///
/// It used to be a dense counter (`next_id += 1`), assigned in first-seen order.
/// That is stable *within* one registry and meaningless across two, and the
/// baseline is persisted: `BaselineSnapshot` stores per-context distributions
/// **keyed by sliver id**, the estate writes it to `rrd:baseline`, and the daemon
/// restores it on boot. So the ids in a restored baseline referred to whatever
/// shapes happened to be interned in that order *last* run. Restart, see shapes
/// in a different order, and every weight in the restored distribution now points
/// at the wrong shape — silently, with no error and no way to notice.
///
/// The bug was masked by a second one: `ask()` passed empty metadata, so there
/// was only ever one shape and it was always id 0. It would have surfaced the
/// moment shapes started varying — which is exactly what feeding COSTAR fields
/// does. "The baseline grows across sessions" is the claim; this is what makes
/// it true.
///
/// FNV-1a, not `DefaultHasher`: std's hasher is `RandomState`-seeded per process,
/// which would reintroduce the identical bug in a subtler costume.
fn sliver_id(key: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

/// A registered sliver: one observed shape under a mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sliver {
    /// Content-derived id: the same shape yields the same id in any process.
    pub id: u64,
    /// Canonical shape key.
    pub key: String,
    /// The mode this sliver specializes.
    pub mode: Mode,
    /// Documents observed with this shape.
    pub count: u64,
}

/// The lattice: canonical shape key → sliver.
#[derive(Debug, Default)]
pub struct ShapeRegistry {
    slivers: HashMap<String, Sliver>,
}

impl ShapeRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe a shape: returns its sliver id and whether it is new (a JIT
    /// compile moment). Re-observation only bumps the count.
    pub fn observe(&mut self, shape: &ShapeFingerprint, mode: Mode) -> (u64, bool) {
        let key = shape.key();
        if let Some(s) = self.slivers.get_mut(&key) {
            s.count += 1;
            return (s.id, false);
        }
        let id = sliver_id(&key);
        self.slivers.insert(
            key.clone(),
            Sliver {
                id,
                key,
                mode,
                count: 1,
            },
        );
        (id, true)
    }

    /// All slivers under a mode (the mode's slice of the lattice).
    pub fn slivers_of(&self, mode: Mode) -> Vec<&Sliver> {
        let mut v: Vec<&Sliver> = self.slivers.values().filter(|s| s.mode == mode).collect();
        v.sort_by_key(|s| s.id);
        v
    }

    /// Number of distinct slivers observed.
    pub fn len(&self) -> usize {
        self.slivers.len()
    }

    /// Whether nothing has been observed yet.
    pub fn is_empty(&self) -> bool {
        self.slivers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rro_core::Metadata;

    fn shape(keys: &[&str]) -> ShapeFingerprint {
        let m: Metadata = keys
            .iter()
            .map(|k| (k.to_string(), serde_json::Value::from("x")))
            .collect();
        ShapeFingerprint::of(&m)
    }

    #[test]
    fn same_shape_registers_once_drift_registers_new() {
        let mut reg = ShapeRegistry::new();
        let (id1, new1) = reg.observe(&shape(&["from", "subject"]), Mode::Mail);
        let (id2, new2) = reg.observe(&shape(&["from", "subject"]), Mode::Mail);
        assert!(new1 && !new2);
        assert_eq!(id1, id2);

        // Drift: an extra field is a NEW sliver, never silent reuse.
        let (id3, new3) = reg.observe(&shape(&["from", "subject", "thread"]), Mode::Mail);
        assert!(new3);
        assert_ne!(id1, id3);
        assert_eq!(reg.slivers_of(Mode::Mail).len(), 2);
    }
}

#[cfg(test)]
mod id_stability_tests {
    use super::*;
    use rro_core::Metadata;

    fn shape(fields: &[(&str, serde_json::Value)]) -> ShapeFingerprint {
        let m: Metadata = fields
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        ShapeFingerprint::of(&m)
    }

    /// THE gate. Two registries that saw the world in **opposite orders** must
    /// still agree on every id.
    ///
    /// This is the bug that was there: ids came from `next_id += 1`, so registry
    /// A called COSTAR 0 and registry B called it 1. The baseline is persisted
    /// keyed by these ids and restored on boot, so a restart quietly repointed
    /// every learned weight at the wrong shape. Nothing would have failed; the
    /// numbers would just have been wrong.
    #[test]
    fn two_registries_agree_on_ids_regardless_of_what_they_saw_first() {
        let costar = shape(&[
            ("context", serde_json::json!("x")),
            ("objective", serde_json::json!("y")),
        ]);
        let mail = shape(&[
            ("from", serde_json::json!("a")),
            ("subject", serde_json::json!("b")),
        ]);
        let empty = shape(&[]);

        let mut a = ShapeRegistry::new();
        let (a_costar, _) = a.observe(&costar, Mode::Unshaped);
        let (a_mail, _) = a.observe(&mail, Mode::Mail);
        let (a_empty, _) = a.observe(&empty, Mode::Unshaped);

        // Same shapes, reverse order — as a restarted process would.
        let mut b = ShapeRegistry::new();
        let (b_empty, _) = b.observe(&empty, Mode::Unshaped);
        let (b_mail, _) = b.observe(&mail, Mode::Mail);
        let (b_costar, _) = b.observe(&costar, Mode::Unshaped);

        assert_eq!(a_costar, b_costar, "ids must not depend on insertion order");
        assert_eq!(a_mail, b_mail);
        assert_eq!(a_empty, b_empty);
    }

    /// Different shapes must not collide, or two intents share one baseline
    /// distribution and the prediction is an average of unrelated things.
    #[test]
    fn different_shapes_get_different_ids() {
        let mut r = ShapeRegistry::new();
        let (a, _) = r.observe(&shape(&[("context", serde_json::json!(1))]), Mode::Unshaped);
        let (b, _) = r.observe(
            &shape(&[("objective", serde_json::json!(1))]),
            Mode::Unshaped,
        );
        let (c, _) = r.observe(&shape(&[]), Mode::Unshaped);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    /// The same shape re-observed is the same sliver — it counts, it does not
    /// re-register. This is what lets the baseline's distribution converge.
    #[test]
    fn re_observation_is_the_same_sliver() {
        let mut r = ShapeRegistry::new();
        let s = shape(&[("context", serde_json::json!("v"))]);
        let (first, is_new) = r.observe(&s, Mode::Unshaped);
        assert!(is_new, "first sighting is a JIT compile moment");
        let (again, is_new) = r.observe(&s, Mode::Unshaped);
        assert!(!is_new, "second sighting is a cache hit");
        assert_eq!(first, again);
    }

    /// Field VALUES must not change the shape — only names and types. Otherwise
    /// every prompt is its own sliver, the distribution never converges, and
    /// predictability sits at 0 forever.
    #[test]
    fn values_do_not_change_the_shape_but_types_do() {
        let mut r = ShapeRegistry::new();
        let (a, _) = r.observe(
            &shape(&[("objective", serde_json::json!("tune the index"))]),
            Mode::Unshaped,
        );
        let (b, _) = r.observe(
            &shape(&[("objective", serde_json::json!("explain a finding"))]),
            Mode::Unshaped,
        );
        assert_eq!(a, b, "same field, same type, different value = same shape");

        let (c, _) = r.observe(
            &shape(&[("objective", serde_json::json!(42))]),
            Mode::Unshaped,
        );
        assert_ne!(a, c, "same field, different TYPE = different shape");
    }
}
