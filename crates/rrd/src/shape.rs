//! Shape fingerprinting: the JIT's "hidden class".
//!
//! Canonical key format `field:type,field:type,…` (BTreeMap order) — the same
//! canonical form the estate's shape census uses, so RRD shape ids and estate
//! census keys line up one-to-one.

use std::collections::BTreeMap;

use rro_core::Metadata;

/// A schema fingerprint: field name → JSON type name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ShapeFingerprint(pub BTreeMap<String, String>);

impl ShapeFingerprint {
    /// Fingerprint a metadata map.
    pub fn of(metadata: &Metadata) -> Self {
        let mut m = BTreeMap::new();
        for (k, v) in metadata {
            let ty = match v {
                serde_json::Value::Null => "null",
                serde_json::Value::Bool(_) => "bool",
                serde_json::Value::Number(_) => "number",
                serde_json::Value::String(_) => "string",
                serde_json::Value::Array(_) => "array",
                serde_json::Value::Object(_) => "object",
            };
            m.insert(k.clone(), ty.to_string());
        }
        ShapeFingerprint(m)
    }

    /// Canonical key (stable field order).
    pub fn key(&self) -> String {
        let parts: Vec<String> = self.0.iter().map(|(k, t)| format!("{k}:{t}")).collect();
        parts.join(",")
    }

    /// Field names.
    pub fn fields(&self) -> impl Iterator<Item = (&String, &String)> {
        self.0.iter()
    }
}
