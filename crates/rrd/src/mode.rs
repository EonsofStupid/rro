//! Modes: the base shapes of the sliver lattice.
//!
//! Mode identification is **structural, not semantic** — field names and
//! types already say what a payload is. No model runs here; this tier is
//! deterministic and free, which is what lets RRD classify at ingest speed.

use serde::{Deserialize, Serialize};

use rro_core::Metadata;

/// The base shapes. Every observed shape lives under exactly one mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Correspondence: from/to/subject-shaped payloads.
    Mail,
    /// Structured rows: typed fields, amounts, statuses.
    Record,
    /// Prose payloads: title + body dominate.
    Document,
    /// Time-anchored happenings: timestamps + kinds.
    Event,
    /// Places and paths.
    Location,
    /// Binary/media references.
    Media,
    /// Nothing matched; the lattice still holds it, under the open mode.
    Unshaped,
}

impl Mode {
    /// All modes, for registry roots.
    pub const ALL: &'static [Mode] = &[
        Mode::Mail,
        Mode::Record,
        Mode::Document,
        Mode::Event,
        Mode::Location,
        Mode::Media,
        Mode::Unshaped,
    ];

    /// Stable lowercase name.
    pub fn name(&self) -> &'static str {
        match self {
            Mode::Mail => "mail",
            Mode::Record => "record",
            Mode::Document => "document",
            Mode::Event => "event",
            Mode::Location => "location",
            Mode::Media => "media",
            Mode::Unshaped => "unshaped",
        }
    }
}

/// Field-name evidence per mode: (names that vote for the mode).
const MODE_VOTES: &[(Mode, &[&str])] = &[
    (
        Mode::Mail,
        &[
            "from",
            "to",
            "cc",
            "bcc",
            "subject",
            "sender",
            "recipient",
            "mailbox",
        ],
    ),
    (
        Mode::Record,
        &[
            "amount", "price", "status", "quantity", "sku", "total", "currency", "row",
        ],
    ),
    (
        Mode::Document,
        &[
            "title", "body", "content", "text", "author", "abstract", "section",
        ],
    ),
    (
        Mode::Event,
        &[
            "event", "kind", "action", "occurred", "started", "ended", "duration", "severity",
        ],
    ),
    (
        Mode::Location,
        &[
            "lat",
            "lon",
            "latitude",
            "longitude",
            "address",
            "city",
            "country",
            "geo",
            "path",
        ],
    ),
    (
        Mode::Media,
        &[
            "mime",
            "bytes",
            "width",
            "height",
            "codec",
            "duration_ms",
            "url",
            "blob",
        ],
    ),
];

/// Identify the mode of a payload from its metadata field names.
///
/// Highest vote count wins; time-ish fields alone push toward [`Mode::Event`];
/// no votes at all → [`Mode::Unshaped`]. Ties resolve in `MODE_VOTES` order
/// (mail before record before document …), which is deliberate and stable.
pub fn identify(metadata: &Metadata) -> Mode {
    let fields: Vec<String> = metadata.keys().map(|k| k.to_lowercase()).collect();
    if fields.is_empty() {
        return Mode::Unshaped;
    }

    let mut best = Mode::Unshaped;
    let mut best_votes = 0usize;
    for (mode, names) in MODE_VOTES {
        let votes = fields
            .iter()
            .filter(|f| {
                names
                    .iter()
                    .any(|n| f == n || f.starts_with(&format!("{n}_")))
            })
            .count();
        if votes > best_votes {
            best = *mode;
            best_votes = votes;
        }
    }

    if best_votes == 0 {
        let timeish = fields
            .iter()
            .any(|f| f.ends_with("_at") || f == "date" || f == "time" || f == "timestamp");
        if timeish {
            return Mode::Event;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(keys: &[&str]) -> Metadata {
        keys.iter()
            .map(|k| (k.to_string(), serde_json::Value::Null))
            .collect()
    }

    #[test]
    fn mail_record_document_identify() {
        assert_eq!(
            identify(&meta(&["from", "to", "subject", "body"])),
            Mode::Mail
        );
        assert_eq!(identify(&meta(&["amount", "status", "sku"])), Mode::Record);
        assert_eq!(
            identify(&meta(&["title", "body", "author"])),
            Mode::Document
        );
        assert_eq!(identify(&meta(&["lat", "lon"])), Mode::Location);
    }

    #[test]
    fn timeish_only_is_event_and_empty_is_unshaped() {
        assert_eq!(identify(&meta(&["created_at", "zzz"])), Mode::Event);
        assert_eq!(identify(&meta(&[])), Mode::Unshaped);
        assert_eq!(identify(&meta(&["xyzzy"])), Mode::Unshaped);
    }
}
