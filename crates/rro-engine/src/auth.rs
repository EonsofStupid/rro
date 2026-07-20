//! Local-key authentication: HS256 JWTs + per-verb RBAC + namespace scope.
//!
//! ## Why hand-rolled, and why HS256
//!
//! The clean-engine law is *pull nothing in* — the HTTP responder, the GraphQL
//! parser and RRQL are all hand-rolled and zero-dep, and there is no crypto crate
//! anywhere in the tree. So this module carries its own SHA-256 and HMAC, each
//! pinned to the published NIST/RFC test vectors (see the tests), and mints
//! **HS256** JWTs signed by a single **local** key. HS256 is symmetric — the same
//! key signs and verifies — which is exactly right for a self-hosted engine with
//! no outbound network and no JWKS endpoint to fetch. (Asymmetric RS/ES256 exists
//! to distribute verification to parties who must not hold the signing key; RRO
//! has no such split, so the extra machinery would buy nothing.)
//!
//! ## What a token authorizes
//!
//! A token carries a [`Role`] and an optional namespace. The role gates verbs by
//! capability (a `reader` cannot write); the namespace scopes the token to one
//! tenant — presented to a node serving a *different* namespace, it is refused.
//! `ping` stays open (the liveness probe needs no capability), matching the a2a
//! shared-secret gate this replaces.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ===========================================================================
// Roles + the verb → capability policy
// ===========================================================================

/// A capability tier. Higher tiers include every lower tier's verbs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Read-only: search, ask, health, graph reads — no mutation.
    Reader,
    /// Reader + data mutation: index, transactions, payload edits, aliases.
    Writer,
    /// Writer + estate administration: compaction, flush, drop.
    Admin,
}

impl Role {
    /// The minimum role a verb requires, or `None` if the verb is open to any
    /// authenticated caller at reader level. `ping` is handled before this (it
    /// needs no token at all).
    fn min_role(verb: &str) -> Role {
        match verb {
            // Estate administration — the destructive/structural surface.
            "compact" | "flush" | "drop_collection" => Role::Admin,

            // Data mutation.
            "index"
            | "tx"
            | "relate"
            | "set_payload"
            | "overwrite_payload"
            | "delete_payload_keys"
            | "clear_payload"
            | "create_alias"
            | "delete_alias" => Role::Writer,

            // `sql` is admitted at reader level; the handler additionally refuses
            // a write statement from a non-writer (a reader's sql is read-only).
            // Everything else — ask/query/recall/map/changes/graphql/health/
            // recommend/discover/traverse/sample/info/collections/aliases/matrix/
            // watch/live — is a read.
            _ => Role::Reader,
        }
    }

    /// Whether this role may invoke `verb` on capability grounds alone (namespace
    /// scope is checked separately).
    pub fn allows(&self, verb: &str) -> bool {
        *self >= Role::min_role(verb)
    }
}

// ===========================================================================
// Claims + the signing policy
// ===========================================================================

/// The JWT payload RRO issues and verifies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject — who the token is for (a user or service id). Informational.
    #[serde(default)]
    pub sub: String,
    /// The capability tier.
    pub role: Role,
    /// Namespace scope. `None` = unscoped (may be used on any node); `Some(ns)` =
    /// usable only on a node serving namespace `ns`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ns: Option<String>,
    /// Issued-at (unix seconds).
    #[serde(default)]
    pub iat: u64,
    /// Expiry (unix seconds). A token past this is rejected.
    pub exp: u64,
}

/// Why a token was refused. All map to `unauthorized` on the wire — the reason is
/// for the engine's own event log, never leaked to the caller (a precise reason
/// is an oracle for an attacker).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthError {
    /// No token presented on a guarded node.
    Missing,
    /// Malformed token (not three base64url segments, or bad JSON).
    Malformed,
    /// Signature did not verify against the local key.
    BadSignature,
    /// `exp` is in the past.
    Expired,
    /// The role lacks the capability for this verb.
    Forbidden,
    /// The token's namespace scope does not match this node.
    WrongNamespace,
}

/// The local signing policy for one node: the HS256 key and the node's namespace
/// identity (if it serves a specific tenant).
#[derive(Clone)]
pub struct AuthPolicy {
    key: Vec<u8>,
    namespace: Option<String>,
}

impl AuthPolicy {
    /// A policy signing with `key`, serving all namespaces (unscoped node).
    pub fn new(key: impl Into<Vec<u8>>) -> Self {
        AuthPolicy {
            key: key.into(),
            namespace: None,
        }
    }

    /// Bind this node to a single namespace, so a token scoped to any *other*
    /// namespace is refused here even if its role and signature are valid.
    pub fn for_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Mint a signed token for `claims`.
    pub fn issue(&self, claims: &Claims) -> String {
        sign_hs256(claims, &self.key)
    }

    /// Issue with a role, namespace scope, and a TTL in seconds from now.
    pub fn issue_for(&self, sub: &str, role: Role, ns: Option<&str>, ttl_secs: u64) -> String {
        let now = now_secs();
        self.issue(&Claims {
            sub: sub.to_string(),
            role,
            ns: ns.map(str::to_string),
            iat: now,
            exp: now + ttl_secs,
        })
    }

    /// Authorize `token` for `verb`: verify the signature, check expiry, the
    /// role's capability, and the namespace scope. Returns the caller's role so
    /// the handler can make finer decisions (e.g. a reader's `sql` is read-only).
    pub fn authorize(
        &self,
        token: Option<&str>,
        verb: &str,
    ) -> std::result::Result<Role, AuthError> {
        let token = token.ok_or(AuthError::Missing)?;
        let claims = verify_hs256(token, &self.key)?;
        if claims.exp <= now_secs() {
            return Err(AuthError::Expired);
        }
        // Namespace scope: a scoped token may only be used on a node serving that
        // namespace. An unscoped token (ns = None) works anywhere; a scoped node
        // still admits unscoped (global/admin) tokens.
        if let (Some(node_ns), Some(tok_ns)) = (&self.namespace, &claims.ns) {
            if node_ns != tok_ns {
                return Err(AuthError::WrongNamespace);
            }
        }
        if !claims.role.allows(verb) {
            return Err(AuthError::Forbidden);
        }
        Ok(claims.role)
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ===========================================================================
// JWT (HS256) encode / decode
// ===========================================================================

fn sign_hs256(claims: &Claims, key: &[u8]) -> String {
    // Fixed header for HS256.
    let header = b"{\"alg\":\"HS256\",\"typ\":\"JWT\"}";
    let payload = serde_json::to_vec(claims).expect("claims serialize");
    let signing_input = format!("{}.{}", b64url_encode(header), b64url_encode(&payload));
    let sig = hmac_sha256(key, signing_input.as_bytes());
    format!("{}.{}", signing_input, b64url_encode(&sig))
}

fn verify_hs256(token: &str, key: &[u8]) -> std::result::Result<Claims, AuthError> {
    let mut parts = token.split('.');
    let (h, p, s) = match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(h), Some(p), Some(s), None) => (h, p, s),
        _ => return Err(AuthError::Malformed),
    };
    let signing_input = format!("{h}.{p}");
    let expected = hmac_sha256(key, signing_input.as_bytes());
    let given = b64url_decode(s).ok_or(AuthError::Malformed)?;
    // Constant-time compare: the whole point of a MAC is defeated by an early-out
    // that leaks how many leading bytes matched.
    if !constant_time_eq(&expected, &given) {
        return Err(AuthError::BadSignature);
    }
    let payload = b64url_decode(p).ok_or(AuthError::Malformed)?;
    serde_json::from_slice(&payload).map_err(|_| AuthError::Malformed)
}

/// Constant-time byte comparison — no data-dependent branch or early return.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ===========================================================================
// base64url (RFC 4648 §5, no padding)
// ===========================================================================

const B64URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn b64url_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64URL[(n >> 18) as usize & 63] as char);
        out.push(B64URL[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(B64URL[(n >> 6) as usize & 63] as char);
        }
        if chunk.len() > 2 {
            out.push(B64URL[n as usize & 63] as char);
        }
    }
    out
}

fn b64url_decode(input: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'-' => Some(62),
            b'_' => Some(63),
            _ => None,
        }
    }
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        if chunk.len() < 2 {
            return None; // a lone trailing char cannot encode any byte
        }
        let mut n = 0u32;
        for &c in chunk {
            n = (n << 6) | val(c)?;
        }
        // Left-align the accumulated bits for the number of bytes this chunk holds.
        n <<= 6 * (4 - chunk.len());
        out.push((n >> 16) as u8);
        if chunk.len() >= 3 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() >= 4 {
            out.push(n as u8);
        }
    }
    Some(out)
}

// ===========================================================================
// SHA-256 + HMAC-SHA256 (FIPS 180-4 / RFC 2104), pinned to test vectors below
// ===========================================================================

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-256 of `msg` → 32 bytes.
pub fn sha256(msg: &[u8]) -> [u8; 32] {
    let mut h = SHA256_H0;

    // Pad: 0x80, then zeros to 56 mod 64, then the 64-bit big-endian bit length.
    let bit_len = (msg.len() as u64).wrapping_mul(8);
    let mut data = msg.to_vec();
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());

    let mut w = [0u32; 64];
    for block in data.chunks_exact(64) {
        for (i, word) in w.iter_mut().take(16).enumerate() {
            let j = i * 4;
            *word = u32::from_be_bytes([block[j], block[j + 1], block[j + 2], block[j + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (hi, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *hi = hi.wrapping_add(v);
        }
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// HMAC-SHA256 (RFC 2104) of `msg` under `key` → 32 bytes.
pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    // A key longer than the block is hashed down first.
    let mut k = if key.len() > BLOCK {
        sha256(key).to_vec()
    } else {
        key.to_vec()
    };
    k.resize(BLOCK, 0);

    let mut ipad = [0u8; BLOCK];
    let mut opad = [0u8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5c;
    }

    let mut inner = Vec::with_capacity(BLOCK + msg.len());
    inner.extend_from_slice(&ipad);
    inner.extend_from_slice(msg);
    let inner_hash = sha256(&inner);

    let mut outer = Vec::with_capacity(BLOCK + 32);
    outer.extend_from_slice(&opad);
    outer.extend_from_slice(&inner_hash);
    sha256(&outer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn sha256_nist_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(&sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn hmac_sha256_rfc4231_vectors() {
        // RFC 4231 test case 1.
        assert_eq!(
            hex(&hmac_sha256(&[0x0b; 20], b"Hi There")),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        // Test case 2: key "Jefe", data "what do ya want for nothing?".
        assert_eq!(
            hex(&hmac_sha256(b"Jefe", b"what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        // Test case 6: key longer than the block (131 bytes) is hashed first.
        assert_eq!(
            hex(&hmac_sha256(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn base64url_roundtrips_all_lengths() {
        for n in 0..40usize {
            let data: Vec<u8> = (0..n).map(|i| (i * 37 + 5) as u8).collect();
            let enc = b64url_encode(&data);
            assert!(!enc.contains('='), "no padding: {enc}");
            assert!(
                !enc.contains('+') && !enc.contains('/'),
                "url-safe alphabet: {enc}"
            );
            assert_eq!(b64url_decode(&enc).unwrap(), data, "roundtrip len {n}");
        }
    }

    #[test]
    fn jwt_roundtrip_and_tamper_detection() {
        let policy = AuthPolicy::new(b"local-signing-key".to_vec());
        let token = policy.issue_for("alice", Role::Writer, Some("acme"), 3600);
        let claims = verify_hs256(&token, b"local-signing-key").unwrap();
        assert_eq!(claims.sub, "alice");
        assert_eq!(claims.role, Role::Writer);
        assert_eq!(claims.ns.as_deref(), Some("acme"));

        // Wrong key → bad signature.
        assert_eq!(
            verify_hs256(&token, b"other-key").unwrap_err(),
            AuthError::BadSignature
        );

        // Flip a payload character → signature no longer matches.
        let mut chars: Vec<char> = token.chars().collect();
        let mid = chars.len() / 2;
        chars[mid] = if chars[mid] == 'A' { 'B' } else { 'A' };
        let tampered: String = chars.into_iter().collect();
        assert!(verify_hs256(&tampered, b"local-signing-key").is_err());
    }

    #[test]
    fn expiry_and_role_and_namespace_are_enforced() {
        let policy = AuthPolicy::new(b"k".to_vec()).for_namespace("acme");

        // An expired token is refused regardless of role.
        let expired = policy.issue(&Claims {
            sub: "x".into(),
            role: Role::Admin,
            ns: None,
            iat: 0,
            exp: 1, // 1970
        });
        assert_eq!(
            policy.authorize(Some(&expired), "health").unwrap_err(),
            AuthError::Expired
        );

        // A reader may query but not index or compact.
        let reader = policy.issue_for("r", Role::Reader, Some("acme"), 3600);
        assert_eq!(
            policy.authorize(Some(&reader), "query").unwrap(),
            Role::Reader
        );
        assert_eq!(
            policy.authorize(Some(&reader), "ask").unwrap(),
            Role::Reader
        );
        assert_eq!(
            policy.authorize(Some(&reader), "index").unwrap_err(),
            AuthError::Forbidden
        );
        assert_eq!(
            policy.authorize(Some(&reader), "compact").unwrap_err(),
            AuthError::Forbidden
        );

        // A writer may index but not compact (admin).
        let writer = policy.issue_for("w", Role::Writer, Some("acme"), 3600);
        assert_eq!(
            policy.authorize(Some(&writer), "index").unwrap(),
            Role::Writer
        );
        assert_eq!(
            policy.authorize(Some(&writer), "compact").unwrap_err(),
            AuthError::Forbidden
        );

        // An admin may compact.
        let admin = policy.issue_for("a", Role::Admin, Some("acme"), 3600);
        assert_eq!(
            policy.authorize(Some(&admin), "compact").unwrap(),
            Role::Admin
        );

        // A token scoped to another namespace is refused on this node.
        let globex = policy.issue_for("g", Role::Admin, Some("globex"), 3600);
        assert_eq!(
            policy.authorize(Some(&globex), "query").unwrap_err(),
            AuthError::WrongNamespace
        );

        // An unscoped (global) token works on the scoped node.
        let global = policy.issue_for("root", Role::Reader, None, 3600);
        assert_eq!(
            policy.authorize(Some(&global), "query").unwrap(),
            Role::Reader
        );

        // No token at all → missing.
        assert_eq!(
            policy.authorize(None, "query").unwrap_err(),
            AuthError::Missing
        );
    }
}
