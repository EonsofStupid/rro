//! Feature hashing used by the deterministic default embedder.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Hash a token to a bucket index and a sign, the classic feature-hashing
/// (a.k.a. hashing-trick / signed random projection) construction.
///
/// Returns `(index in [0, dim), sign in {-1.0, +1.0})`.
pub fn bucket(token: &str, dim: usize) -> (usize, f32) {
    let mut h = DefaultHasher::new();
    token.hash(&mut h);
    let hv = h.finish();
    let index = (hv % dim as u64) as usize;
    // Use a different bit of the hash for the sign so index and sign decorrelate.
    let sign = if (hv >> 33) & 1 == 0 { 1.0 } else { -1.0 };
    (index, sign)
}
