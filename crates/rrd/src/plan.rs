//! Plans: the compiled artifact of the JIT.
//!
//! A plan is compiled **once per sliver** and cached; every payload of that
//! shape then rides the compiled path. v1 compilation is rule-derived (field
//! names + types → roles). The seam for a learned compiler — including a
//! zero-shot/NLI pass at *compile time only*, never per document — is this
//! module; per-document work must stay model-free.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::mode::Mode;
use crate::shape::ShapeFingerprint;

/// What a field is *for*, from the reasoner's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldRole {
    /// Identifies the object (ids, keys, addresses).
    Identity,
    /// Anchors it in time.
    Time,
    /// Names it for humans.
    Title,
    /// Carries the substance.
    Content,
    /// Weighs it (amounts, scores, priorities).
    Salience,
    /// Everything else — kept, but not privileged.
    Other,
}

/// A compiled per-sliver distillation plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// The sliver this plan is compiled for.
    pub sliver_id: u64,
    /// The sliver's mode.
    pub mode: Mode,
    /// Field → role.
    pub roles: BTreeMap<String, FieldRole>,
    /// Plan version (bumps when the compiler improves).
    pub version: u32,
}

/// Current plan-compiler version.
pub const COMPILER_VERSION: u32 = 1;

/// Compile a plan for a sliver: rule-derived field roles.
pub fn compile(sliver_id: u64, mode: Mode, shape: &ShapeFingerprint) -> Plan {
    let mut roles = BTreeMap::new();
    for (field, ty) in shape.fields() {
        roles.insert(field.clone(), infer_role(field, ty));
    }
    Plan {
        sliver_id,
        mode,
        roles,
        version: COMPILER_VERSION,
    }
}

fn infer_role(field: &str, ty: &str) -> FieldRole {
    let f = field.to_lowercase();

    // Identity: ids, keys, unique handles.
    if f == "id"
        || f.ends_with("_id")
        || f.ends_with("_key")
        || f == "uuid"
        || f == "sku"
        || f == "email"
        || f == "from"
        || f == "to"
        || f == "sender"
        || f == "recipient"
    {
        return FieldRole::Identity;
    }
    // Time anchors.
    if f.ends_with("_at")
        || f == "date"
        || f == "time"
        || f == "timestamp"
        || f == "occurred"
        || f == "created"
        || f == "updated"
    {
        return FieldRole::Time;
    }
    // Titles.
    if f == "title" || f == "subject" || f == "name" || f == "heading" {
        return FieldRole::Title;
    }
    // Substance.
    if f == "body" || f == "content" || f == "text" || f == "description" || f == "message" {
        return FieldRole::Content;
    }
    // Weight: numeric business fields.
    if ty == "number"
        && (f.contains("amount")
            || f.contains("price")
            || f.contains("score")
            || f.contains("priority")
            || f.contains("total")
            || f.contains("severity"))
    {
        return FieldRole::Salience;
    }
    FieldRole::Other
}

#[cfg(test)]
mod tests {
    use super::*;
    use rro_core::Metadata;

    #[test]
    fn mail_fields_get_sensible_roles() {
        let m: Metadata = [
            ("from", serde_json::json!("a@b")),
            ("subject", serde_json::json!("hi")),
            ("body", serde_json::json!("...")),
            ("sent_at", serde_json::json!(1)),
            ("priority_score", serde_json::json!(5)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let plan = compile(0, Mode::Mail, &ShapeFingerprint::of(&m));
        assert_eq!(plan.roles["from"], FieldRole::Identity);
        assert_eq!(plan.roles["subject"], FieldRole::Title);
        assert_eq!(plan.roles["body"], FieldRole::Content);
        assert_eq!(plan.roles["sent_at"], FieldRole::Time);
        assert_eq!(plan.roles["priority_score"], FieldRole::Salience);
    }
}
