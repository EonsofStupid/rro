//! The KV storage seam: the Fjall 3.x LSM, behind a backend-agnostic handle.
//!
//! `connxism` speaks to one key/value backend — the pure-Rust Fjall 3.x LSM.
//! Everything above this seam (`estate`, `store`, `txn`, `query`, `filter`,
//! `rels`) uses the re-exported `Db`/`Batch`/`KvItem` and never names the
//! concrete backend, so the store stays swappable if that ever changes.

mod fjall;
pub(crate) use fjall::{Batch, Db, KvItem};
