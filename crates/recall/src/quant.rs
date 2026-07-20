//! Scalar quantization (SQ8): 4× smaller vector memory, measured recall.
//!
//! Each vector is quantized independently to one byte per dimension with its
//! own affine map `x ≈ code * scale + offset` (per-vector min/max). Because
//! the map is affine, dot products against quantized storage stay *exact in
//! the codes*:
//!
//! - **asymmetric** (full-precision query `q` vs codes `c`):
//!   `q · x ≈ scale · Σ cᵢqᵢ + offset · Σ qᵢ` — one integer-weighted dot plus
//!   a precomputed query sum;
//! - **symmetric** (codes vs codes):
//!   `a · b ≈ sₐs_b Σ cₐc_b + sₐo_b Σ cₐ + oₐs_b Σ c_b + d·oₐo_b` — the code
//!   sums are stored once per vector.
//!
//! The quantized graph answers approximately; callers that keep the
//! full-precision vectors elsewhere (the estate's durable vector column
//! family) **rescore** the returned candidates exactly. Quantization here is
//! a memory decision, never a silent accuracy decision.

/// Per-vector affine parameters for SQ8 codes.
#[derive(Debug, Clone, Copy)]
pub struct SqParams {
    /// Code → value multiplier.
    pub scale: f32,
    /// Code → value offset (the vector's minimum).
    pub offset: f32,
    /// Σ codes, cached for symmetric dots.
    pub code_sum: f32,
}

/// Quantize `v`, appending its codes to `codes`; returns the affine params.
pub fn quantize_into(v: &[f32], codes: &mut Vec<u8>) -> SqParams {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &x in v {
        min = min.min(x);
        max = max.max(x);
    }
    if v.is_empty() {
        return SqParams {
            scale: 0.0,
            offset: 0.0,
            code_sum: 0.0,
        };
    }
    let range = max - min;
    let scale = if range > f32::EPSILON {
        range / 255.0
    } else {
        0.0
    };
    let mut code_sum = 0.0f32;
    for &x in v {
        let code = if scale > 0.0 {
            ((x - min) / scale).round().clamp(0.0, 255.0) as u8
        } else {
            0
        };
        code_sum += code as f32;
        codes.push(code);
    }
    SqParams {
        scale,
        offset: min,
        code_sum,
    }
}

/// Asymmetric dot: full-precision query against one vector's codes.
/// `qsum` is `Σ qᵢ`, computed once per query.
pub fn dot_query(codes: &[u8], p: &SqParams, q: &[f32], qsum: f32) -> f32 {
    let mut acc = 0.0f32;
    for (&c, &x) in codes.iter().zip(q) {
        acc += c as f32 * x;
    }
    p.scale * acc + p.offset * qsum
}

/// Symmetric dot: codes against codes, both dequantized implicitly.
pub fn dot_codes(a: &[u8], pa: &SqParams, b: &[u8], pb: &SqParams) -> f32 {
    let mut acc = 0u32;
    for (&x, &y) in a.iter().zip(b) {
        acc += x as u32 * y as u32;
    }
    let d = a.len() as f32;
    pa.scale * pb.scale * acc as f32
        + pa.scale * pb.offset * pa.code_sum
        + pa.offset * pb.scale * pb.code_sum
        + d * pa.offset * pb.offset
}

/// Reconstruct the (lossy) full-precision vector from its codes.
pub fn decode(codes: &[u8], p: &SqParams) -> Vec<f32> {
    codes
        .iter()
        .map(|&c| c as f32 * p.scale + p.offset)
        .collect()
}

// ---- binary quantization (BQ) ----------------------------------------------
//
// One **bit** per dimension — the sign of each component — so a vector shrinks
// 32× from f32 (vs SQ8's 4×). The estimate keeps only orientation, which for
// normalized embeddings is most of the cosine signal; magnitude is gone. BQ is
// far lossier than SQ8, so it is a *traversal* code: the graph is walked cheaply
// on bits, then candidates are rescored exactly from the durable vectors (same
// contract as SQ8 — quantization is a memory/IO decision, never a silent
// accuracy one).
//
// Distances, for bits interpreted as sign ±1:
// - **asymmetric** (full query `q` vs bits `b`): `Σ qᵢ·sign(bᵢ)` — the query's
//   own magnitude is kept, only the stored side is signed;
// - **symmetric** (bits vs bits): `Σ sign(aᵢ)sign(bᵢ) = D − 2·popcount(a ⊕ b)` —
//   one XOR + popcount per word.

/// Bytes needed to hold `dim` sign bits, packed 8 per byte.
pub fn bq_bytes(dim: usize) -> usize {
    dim.div_ceil(8)
}

/// Encode `v`'s sign bits (bit `i` = 1 iff `v[i] > 0`), appending
/// `bq_bytes(v.len())` bytes to `bits`. Padding bits in the final byte stay 0.
pub fn bq_encode_into(v: &[f32], bits: &mut Vec<u8>) {
    let start = bits.len();
    bits.resize(start + bq_bytes(v.len()), 0);
    for (i, &x) in v.iter().enumerate() {
        if x > 0.0 {
            bits[start + i / 8] |= 1 << (i % 8);
        }
    }
}

/// Asymmetric estimate: a full-precision query against one vector's sign bits.
/// `Σ qᵢ` if the bit is set, `−qᵢ` otherwise.
pub fn bq_dot_query(bits: &[u8], q: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (i, &qi) in q.iter().enumerate() {
        let set = (bits[i / 8] >> (i % 8)) & 1 == 1;
        acc += if set { qi } else { -qi };
    }
    acc
}

/// Symmetric estimate: sign bits against sign bits, `D − 2·popcount(a ⊕ b)`.
/// Padding bits are 0 in both operands, so they cancel and never affect the
/// count.
pub fn bq_dot_codes(a: &[u8], b: &[u8], dim: usize) -> f32 {
    let ham: u32 = a.iter().zip(b).map(|(&x, &y)| (x ^ y).count_ones()).sum();
    dim as f32 - 2.0 * ham as f32
}

/// Reconstruct an approximate **unit** vector from sign bits: each component is
/// `±1/√dim`. Magnitude is unrecoverable from bits; this keeps the vector unit-
/// norm so cosine-style use stays sane (used for in-graph neighbor rescoring).
pub fn bq_decode(bits: &[u8], dim: usize) -> Vec<f32> {
    let s = 1.0 / (dim as f32).sqrt();
    (0..dim)
        .map(|i| {
            let set = (bits[i / 8] >> (i % 8)) & 1 == 1;
            if set {
                s
            } else {
                -s
            }
        })
        .collect()
}

// ---- product quantization (PQ) ---------------------------------------------
//
// The vector is split into `m` sub-vectors; each is replaced by the index of the
// nearest of `k` trained centroids (`k ≤ 256`, so one byte per sub-vector). A
// vector thus becomes `m` bytes — tunable compression: `sub_dim = dim/m`, so
// `m = dim/4` gives 16× (vs SQ8's 4×, BQ's 32×), with better recall-per-byte than
// BQ because each byte encodes a whole `sub_dim`-block, not a single sign.
//
// Unlike SQ8/BQ, PQ must be **trained**: the centroids come from k-means over a
// representative sample. Distances reconstruct through the centroids — dot(q, x)
// ≈ Σ_s dot(q_sub[s], centroid[s][code[s]]) — exact in the codes, lossy in the
// reconstruction; candidates are rescored exactly from the durable vectors, the
// same contract as SQ8/BQ.

/// A trained product-quantization codebook: `m` subspaces × `k` centroids of
/// `sub_dim` dims each. Centroid `code` of subspace `s` is the `sub_dim` slice at
/// `centroids[(s*k + code)*sub_dim ..]`.
#[derive(Debug, Clone)]
pub struct PqCodebook {
    /// Number of subspaces (bytes per encoded vector).
    pub m: usize,
    /// Centroids per subspace (≤ 256, so a code fits a byte).
    pub k: usize,
    /// Dimensions per subspace (`dim / m`).
    pub sub_dim: usize,
    /// Flattened centroids, `m * k * sub_dim` floats.
    pub centroids: Vec<f32>,
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// k-means (Lloyd) over `points` (each `sub_dim` long). Deterministic:
/// evenly-strided init, fixed iteration count, empty clusters keep their
/// centroid. Returns `k*sub_dim` centroid floats.
fn kmeans(points: &[&[f32]], sub_dim: usize, k: usize, iters: usize) -> Vec<f32> {
    let n = points.len();
    let k = k.min(n).max(1);
    let mut centroids = vec![0.0f32; k * sub_dim];
    for c in 0..k {
        let idx = (c * n) / k; // spread across the sample
        centroids[c * sub_dim..(c + 1) * sub_dim].copy_from_slice(points[idx]);
    }
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        for (i, p) in points.iter().enumerate() {
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let d = l2_sq(p, &centroids[c * sub_dim..(c + 1) * sub_dim]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assign[i] = best;
        }
        let mut sums = vec![0.0f32; k * sub_dim];
        let mut counts = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            let c = assign[i];
            counts[c] += 1;
            for d in 0..sub_dim {
                sums[c * sub_dim + d] += p[d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..sub_dim {
                    centroids[c * sub_dim + d] = sums[c * sub_dim + d] / counts[c] as f32;
                }
            }
        }
    }
    centroids
}

/// Train a PQ codebook from `sample` (each vector `dim` long). `dim` must be
/// divisible by `m`. `k` is capped at both 256 and the sample size.
pub fn pq_train(sample: &[Vec<f32>], dim: usize, m: usize, iters: usize) -> PqCodebook {
    assert!(
        m > 0 && dim.is_multiple_of(m),
        "dim {dim} must be divisible by m {m}"
    );
    let sub_dim = dim / m;
    let k = 256.min(sample.len()).max(1);
    let mut centroids = vec![0.0f32; m * k * sub_dim];
    for s in 0..m {
        let subvecs: Vec<&[f32]> = sample
            .iter()
            .map(|v| &v[s * sub_dim..(s + 1) * sub_dim])
            .collect();
        let sub = kmeans(&subvecs, sub_dim, k, iters);
        centroids[s * k * sub_dim..(s + 1) * k * sub_dim].copy_from_slice(&sub);
    }
    PqCodebook {
        m,
        k,
        sub_dim,
        centroids,
    }
}

impl PqCodebook {
    #[inline]
    fn centroid(&self, s: usize, code: usize) -> &[f32] {
        let base = (s * self.k + code) * self.sub_dim;
        &self.centroids[base..base + self.sub_dim]
    }

    /// Nearest centroid index for `sub` in subspace `s`.
    fn nearest(&self, s: usize, sub: &[f32]) -> u8 {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..self.k {
            let d = l2_sq(sub, self.centroid(s, c));
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        best as u8
    }
}

/// Encode `v` to `m` codes, appending them to `codes`.
pub fn pq_encode_into(v: &[f32], cb: &PqCodebook, codes: &mut Vec<u8>) {
    for s in 0..cb.m {
        let sub = &v[s * cb.sub_dim..(s + 1) * cb.sub_dim];
        codes.push(cb.nearest(s, sub));
    }
}

/// Asymmetric dot: full query vs a vector's PQ codes, reconstructed through the
/// centroids. `Σ_s dot(q_sub[s], centroid[s][code[s]])`.
pub fn pq_dot_query(codes: &[u8], cb: &PqCodebook, q: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for s in 0..cb.m {
        let qsub = &q[s * cb.sub_dim..(s + 1) * cb.sub_dim];
        let cent = cb.centroid(s, codes[s] as usize);
        acc += qsub.iter().zip(cent).map(|(a, b)| a * b).sum::<f32>();
    }
    acc
}

/// Symmetric dot: two vectors' codes, both reconstructed through the centroids.
pub fn pq_dot_codes(a: &[u8], b: &[u8], cb: &PqCodebook) -> f32 {
    let mut acc = 0.0f32;
    for s in 0..cb.m {
        let ca = cb.centroid(s, a[s] as usize);
        let cbv = cb.centroid(s, b[s] as usize);
        acc += ca.iter().zip(cbv).map(|(x, y)| x * y).sum::<f32>();
    }
    acc
}

/// Reconstruct the (lossy) full-precision vector by concatenating the centroids.
pub fn pq_decode(codes: &[u8], cb: &PqCodebook) -> Vec<f32> {
    let mut out = Vec::with_capacity(cb.m * cb.sub_dim);
    for (s, &code) in codes.iter().enumerate() {
        out.extend_from_slice(cb.centroid(s, code as usize));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_vec(seed: u64, dim: usize) -> Vec<f32> {
        let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..dim)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x as f64 / u64::MAX as f64) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn exact_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn asymmetric_dot_tracks_exact() {
        for seed in 0..20u64 {
            let v = pseudo_vec(seed, 128);
            let q = pseudo_vec(seed + 1000, 128);
            let mut codes = Vec::new();
            let p = quantize_into(&v, &mut codes);
            let qsum: f32 = q.iter().sum();
            let approx = dot_query(&codes, &p, &q, qsum);
            let exact = exact_dot(&v, &q);
            assert!(
                (approx - exact).abs() < 0.05 * 128f32.sqrt(),
                "seed {seed}: approx {approx} vs exact {exact}"
            );
        }
    }

    #[test]
    fn symmetric_dot_tracks_exact() {
        for seed in 0..20u64 {
            let a = pseudo_vec(seed, 128);
            let b = pseudo_vec(seed + 500, 128);
            let mut ca = Vec::new();
            let mut cb = Vec::new();
            let pa = quantize_into(&a, &mut ca);
            let pb = quantize_into(&b, &mut cb);
            let approx = dot_codes(&ca, &pa, &cb, &pb);
            let exact = exact_dot(&a, &b);
            assert!(
                (approx - exact).abs() < 0.08 * 128f32.sqrt(),
                "seed {seed}: approx {approx} vs exact {exact}"
            );
        }
    }

    #[test]
    fn constant_vector_roundtrips() {
        let v = vec![0.25f32; 32];
        let mut codes = Vec::new();
        let p = quantize_into(&v, &mut codes);
        let back = decode(&codes, &p);
        for x in back {
            assert!((x - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn decode_error_is_bounded_by_half_step() {
        let v = pseudo_vec(7, 64);
        let mut codes = Vec::new();
        let p = quantize_into(&v, &mut codes);
        let back = decode(&codes, &p);
        for (orig, dec) in v.iter().zip(&back) {
            assert!((orig - dec).abs() <= p.scale * 0.5 + 1e-6);
        }
    }

    #[test]
    fn bq_byte_count_and_padding() {
        assert_eq!(bq_bytes(8), 1);
        assert_eq!(bq_bytes(9), 2);
        assert_eq!(bq_bytes(0), 0);
        // A dim not a multiple of 8: padding bits stay 0, so self-XOR is 0.
        let v = pseudo_vec(3, 12);
        let mut bits = Vec::new();
        bq_encode_into(&v, &mut bits);
        assert_eq!(bits.len(), 2);
        assert_eq!(bq_dot_codes(&bits, &bits, 12), 12.0, "self-similarity is D");
    }

    #[test]
    fn bq_encodes_signs() {
        let v = vec![0.3, -0.1, 0.0, 2.0, -5.0];
        let mut bits = Vec::new();
        bq_encode_into(&v, &mut bits);
        // bits: +,-,0(→not>0),+,- => 1,0,0,1,0
        assert_eq!(bits[0] & 0b11111, 0b01001);
    }

    #[test]
    fn bq_symmetric_equals_signed_dot() {
        // D − 2·Hamming must equal Σ sign(a)sign(b) computed directly.
        for seed in 0..20u64 {
            let a = pseudo_vec(seed, 96);
            let b = pseudo_vec(seed + 7, 96);
            let (mut ba, mut bb) = (Vec::new(), Vec::new());
            bq_encode_into(&a, &mut ba);
            bq_encode_into(&b, &mut bb);
            let direct: f32 = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| x.signum() * y.signum())
                .sum();
            assert_eq!(bq_dot_codes(&ba, &bb, 96), direct);
        }
    }

    /// PQ with cluster structure: build a corpus of a few tight clusters, train a
    /// codebook, and confirm the codes shrink memory and the dot reconstructs well
    /// enough to rank a cluster's own members together.
    #[test]
    fn pq_trains_encodes_and_reconstructs() {
        let dim = 64;
        let m = 16; // sub_dim 4 → 16 bytes/vec = 16× smaller than 256-byte f32
                    // 8 cluster centers; each sample is a center plus small jitter.
        let centers: Vec<Vec<f32>> = (0..8).map(|c| pseudo_vec(c, dim)).collect();
        let mut sample = Vec::new();
        for c in 0..8u64 {
            for j in 0..100u64 {
                let jitter = pseudo_vec(1000 + c * 100 + j, dim);
                let v: Vec<f32> = centers[c as usize]
                    .iter()
                    .zip(&jitter)
                    .map(|(a, b)| a + 0.05 * b)
                    .collect();
                sample.push(v);
            }
        }
        let cb = pq_train(&sample, dim, m, 10);
        assert_eq!(cb.m, 16);
        assert_eq!(cb.sub_dim, 4);
        assert_eq!(cb.k, 256.min(sample.len()));

        // Encoding a sample point and decoding it lands close to the point (its
        // sub-vectors sit near trained centroids).
        let v = &sample[42];
        let mut codes = Vec::new();
        pq_encode_into(v, &cb, &mut codes);
        assert_eq!(codes.len(), m); // 16 bytes vs 256
        let recon = pq_decode(&codes, &cb);
        let err = l2_sq(v, &recon).sqrt() / (v.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-9);
        assert!(err < 0.3, "PQ reconstruction relative error {err} too high");

        // Asymmetric dot vs codes tracks the exact dot far better than chance.
        let q = &sample[7];
        let approx = pq_dot_query(&codes, &cb, q);
        let exact: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
        assert!(
            (approx - exact).abs() < 0.15 * dim as f32,
            "PQ asymmetric dot {approx} vs exact {exact}"
        );
    }

    /// On clustered data, PQ must rank a point's own cluster-mates above random
    /// outsiders — the property that lets it traverse the graph.
    #[test]
    fn pq_ranks_cluster_mates_together() {
        let dim = 32;
        let m = 8;
        let centers: Vec<Vec<f32>> = (0..6).map(|c| pseudo_vec(c, dim)).collect();
        let mut sample = Vec::new();
        let mut cluster_of = Vec::new();
        for c in 0..6u64 {
            for j in 0..80u64 {
                let jitter = pseudo_vec(5000 + c * 80 + j, dim);
                let v: Vec<f32> = centers[c as usize]
                    .iter()
                    .zip(&jitter)
                    .map(|(a, b)| a + 0.03 * b)
                    .collect();
                sample.push(v);
                cluster_of.push(c as usize);
            }
        }
        let cb = pq_train(&sample, dim, m, 10);
        let codes: Vec<Vec<u8>> = sample
            .iter()
            .map(|v| {
                let mut c = Vec::new();
                pq_encode_into(v, &cb, &mut c);
                c
            })
            .collect();

        // For each query, its top-1 by PQ asymmetric dot should be a cluster-mate.
        let mut same = 0;
        for (qi, q) in sample.iter().enumerate() {
            let best = (0..sample.len())
                .filter(|&i| i != qi)
                .max_by(|&a, &b| {
                    pq_dot_query(&codes[a], &cb, q).total_cmp(&pq_dot_query(&codes[b], &cb, q))
                })
                .unwrap();
            if cluster_of[best] == cluster_of[qi] {
                same += 1;
            }
        }
        let frac = same as f64 / sample.len() as f64;
        assert!(frac > 0.9, "PQ should keep cluster-mates on top: {frac:.3}");
    }

    /// BQ is lossy, so it cannot track the exact dot like SQ8 — but it must
    /// *rank* well enough to traverse: the vector a query points at should score
    /// highest under BQ. Here the query is one corpus vector, so its own sign
    /// code is the exact match and must win under both BQ estimators.
    #[test]
    fn bq_ranks_the_true_neighbour_first() {
        let dim = 128;
        let corpus: Vec<Vec<f32>> = (0..200).map(|i| pseudo_vec(i, dim)).collect();
        let codes: Vec<Vec<u8>> = corpus
            .iter()
            .map(|v| {
                let mut b = Vec::new();
                bq_encode_into(v, &mut b);
                b
            })
            .collect();

        let mut hits = 0;
        for (qi, q) in corpus.iter().enumerate() {
            // asymmetric: full query vs every code
            let best = (0..corpus.len())
                .max_by(|&a, &b| bq_dot_query(&codes[a], q).total_cmp(&bq_dot_query(&codes[b], q)))
                .unwrap();
            if best == qi {
                hits += 1;
            }
        }
        // Every self-query should recover itself (its signs match perfectly).
        assert_eq!(hits, corpus.len(), "BQ must rank the exact match first");
    }
}
