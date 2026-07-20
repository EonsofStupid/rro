//! Distance kernels: unrolled inner loops on the engine's hottest path.
//!
//! Four independent accumulators break the sequential dependency chain so the
//! compiler auto-vectorizes and the CPU pipelines the multiplies. Portable
//! (no nightly, no intrinsics); explicit `std::simd`/intrinsic backends can
//! slot in behind these signatures later without touching callers.

/// Dot product with a 4-lane unrolled loop.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);

    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    let ca = a.chunks_exact(4);
    let cb = b.chunks_exact(4);
    let ra = ca.remainder();
    let rb = cb.remainder();
    for (x, y) in ca.zip(cb) {
        s0 += x[0] * y[0];
        s1 += x[1] * y[1];
        s2 += x[2] * y[2];
        s3 += x[3] * y[3];
    }
    let mut s = s0 + s1 + s2 + s3;
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// Squared L2 norm via [`dot`].
#[inline]
pub fn norm_sq(a: &[f32]) -> f32 {
    dot(a, a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_naive_dot() {
        let a: Vec<f32> = (0..131).map(|i| (i as f32) * 0.31 - 20.0).collect();
        let b: Vec<f32> = (0..131).map(|i| (i as f32) * -0.17 + 9.0).collect();
        let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - naive).abs() < 1e-2 * naive.abs().max(1.0));
        assert_eq!(dot(&[], &[]), 0.0);
    }
}
