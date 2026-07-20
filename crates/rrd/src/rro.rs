//! The RRO — Reason-Ready Object. The engine's unit of structured evidence.

use serde::{Deserialize, Serialize};

use rro_core::Metadata;

use crate::gates::{GateVerdict, LexicalSignals, SourceStamp};
use crate::mode::Mode;
use crate::router::RoutedTag;

/// Structured hints the readiness classifier consumes instead of guessing
/// from raw text.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReadinessHints {
    /// An identity field was present and non-empty.
    pub has_identity: bool,
    /// A time anchor was present.
    pub has_time: bool,
    /// A human title was present.
    pub has_title: bool,
    /// Substance (content-role field or document text) was present.
    pub has_content: bool,
    /// Best tag-route score (0 when nothing routed).
    pub tag_confidence: f32,
    /// L2 ambiguity: margin between the top two routed intents (small =
    /// ambiguous = confirmation/escalation candidate).
    pub ambiguity_margin: f32,
}

/// A reason-ready object: typed, tagged, provenance-carrying.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rro {
    /// The source document id.
    pub doc_id: String,
    /// Which sliver (shape) produced it.
    pub sliver_id: u64,
    /// The sliver's mode.
    pub mode: Mode,
    /// Fields grouped by role, ready for a reasoner: role name → {field: value}.
    pub fields: Metadata,
    /// Tags routed in embedding space, best first.
    pub tags: Vec<RoutedTag>,
    /// Evidence hints for the readiness gate.
    pub hints: ReadinessHints,
    /// Plan version that distilled this object (provenance of the distillation).
    pub plan_version: u32,
    /// Who/where this came from (stamped before any gate ran).
    pub stamp: SourceStamp,
    /// The worst verdict any gate tier returned (Pass / Flag / Block).
    pub gate: GateVerdict,
    /// L1 lexical signals (secrets / injection / unicode / operation).
    pub signals: LexicalSignals,
}
