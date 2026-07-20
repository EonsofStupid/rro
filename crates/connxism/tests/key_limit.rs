//! Key-length guard: an over-limit key is rejected cleanly, up front.
//!
//! Fjall caps keys at 65_536 bytes. Without the guard a huge doc id (or a long
//! indexed metadata value) would only fail deep inside the store, mid-write. The
//! guard closes it: the key is rejected at the call site, before any write.

use connxism::Estate;
use rro_core::{Embedding, Recall, VectorRecord};

#[tokio::test(flavor = "multi_thread")]
async fn over_limit_document_id_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "keylimit").unwrap();
    let recall = estate.recall();

    // A normal id upserts fine.
    let ok = VectorRecord::new("doc-1", Embedding(vec![1.0, 0.0]).normalized(), "hello");
    recall.upsert(vec![ok]).await.unwrap();

    // An id past the ceiling is refused, not silently accepted on one backend.
    let huge = "x".repeat(connxism::keys::MAX_KEY_LEN + 1);
    let bad = VectorRecord::new(huge, Embedding(vec![1.0, 0.0]).normalized(), "world");
    assert!(recall.upsert(vec![bad]).await.is_err());

    // The rejected upsert changed nothing: the good doc is still the only one.
    assert_eq!(recall.len().await.unwrap(), 1);
}
