//! TOON — Token-Oriented Object Notation: a compact, lossless encoding of the
//! JSON data model for LLM prompts (spec v3.2, `text/toon`).
//!
//! The point is the recall→LLM handoff. A recall result is a *uniform array of
//! objects* — TOON's sweet spot — which collapses into a table that declares its
//! fields once and streams rows, CSV-compact but with an explicit `[N]{fields}`
//! schema the model can follow. Published benchmarks put TOON at ~40% fewer
//! tokens than JSON at equal-or-better answer accuracy; here we own a faithful
//! encoder so recall context never pays the JSON tax.
//!
//! This is an **encoder** (the direction we need): `serde_json::Value → TOON`.
//! Objects become `key: value` lines; scalar arrays inline as `key[N]: a,b,c`;
//! uniform arrays of scalar-valued objects become tables; anything else falls
//! back to a per-element list that stays valid and lossless.

use crate::types::Candidate;
use serde_json::Value;
use std::fmt::Write as _;

/// Encode recall candidates as a TOON table for the LLM context — the recall→LLM
/// handoff this format exists for. Projects each candidate to the scalar columns
/// a prompt actually needs (`id`, `text`, `score`), which is TOON's sweet spot: a
/// uniform array of objects collapses to `id,text,score` declared once, one row
/// per hit. Score is rounded to keep rows tight.
pub fn candidates_toon(candidates: &[Candidate]) -> String {
    let rows: Vec<Value> = candidates
        .iter()
        .map(|c| {
            serde_json::json!({
                "id": c.id.as_str(),
                "text": c.text,
                "score": (c.score * 10_000.0).round() / 10_000.0,
            })
        })
        .collect();
    to_toon(&Value::Array(rows))
}

/// Which encoding the recall context is served in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContextFormat {
    /// Standard JSON — the safe default when the client hasn't opted in.
    Json,
    /// TOON — compact, served only when the client accepts it.
    Toon,
}

/// A layman-friendly pitch to enable TOON, with the real savings *for this
/// result* so the recommendation is concrete, never marketing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Recommendation {
    /// One-line headline.
    pub headline: String,
    /// Plain-language explanation of why it's worth enabling.
    pub why: String,
    /// Bytes saved vs JSON, as a percentage, measured on this result.
    pub bytes_saved_pct: u32,
    /// Approximate tokens saved vs JSON, measured on this result.
    pub tokens_saved_pct: u32,
    /// How to turn it on (the negotiation the client should send next time).
    pub how_to_enable: String,
}

/// The rendered recall context: the encoded body, which format it's in, and —
/// when the client took JSON but TOON was available — a recommendation to switch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RenderedContext {
    /// The format the body is encoded in.
    pub format: ContextFormat,
    /// The encoded recall context.
    pub body: String,
    /// Present only when JSON was served but TOON would have been smaller — the
    /// offer to enable it, with this result's actual savings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommendation: Option<Recommendation>,
}

fn approx_tokens(s: &str) -> usize {
    s.split(|c: char| c.is_whitespace() || "{}[]\":,".contains(c))
        .filter(|t| !t.is_empty())
        .count()
}

/// Render recall candidates for an LLM, **negotiating** the format: if the client
/// accepts TOON, serve TOON; otherwise serve JSON and attach a recommendation
/// (with this result's measured savings) so the client can choose to opt in.
///
/// This is the deliberate rollout the operator asked for — detect support, use it
/// when present, otherwise ask with a concrete, plain-language reason, never
/// silently force a format the consumer may not parse.
pub fn render_candidates(candidates: &[Candidate], accepts_toon: bool) -> RenderedContext {
    if accepts_toon {
        return RenderedContext {
            format: ContextFormat::Toon,
            body: candidates_toon(candidates),
            recommendation: None,
        };
    }
    // Serve JSON, but measure what TOON would have saved so the offer is honest.
    let rows: Vec<Value> = candidates
        .iter()
        .map(|c| {
            serde_json::json!({
                "id": c.id.as_str(),
                "text": c.text,
                "score": (c.score * 10_000.0).round() / 10_000.0,
            })
        })
        .collect();
    let value = Value::Array(rows);
    let json = serde_json::to_string(&value).unwrap_or_default();
    let toon = to_toon(&value);
    let bytes_saved = pct_saved(json.len(), toon.len());
    let tokens_saved = pct_saved(approx_tokens(&json), approx_tokens(&toon));

    let recommendation = (bytes_saved > 0).then(|| Recommendation {
        headline: format!(
            "Recall context is {bytes_saved}% smaller in TOON — want RRO to send it that way?"
        ),
        why: "TOON packs your results into a compact table instead of repeating \
              the field names, quotes and braces on every single row — think a \
              spreadsheet, not the same form filled out over and over. Nothing is \
              lost (it's a lossless rewrite of the exact same data), models parse \
              it as reliably or better, and it costs fewer tokens: fewer tokens \
              means lower cost, faster answers, and more of the context window \
              left for the actual content. Highly recommended for anything that \
              feeds recall results into an LLM."
            .to_string(),
        bytes_saved_pct: bytes_saved,
        tokens_saved_pct: tokens_saved,
        how_to_enable: "Send `accepts_toon: true` (or `format: \"toon\"`) with your \
                        next recall request and RRO will serve TOON automatically."
            .to_string(),
    });

    RenderedContext {
        format: ContextFormat::Json,
        body: json,
        recommendation,
    }
}

fn pct_saved(from: usize, to: usize) -> u32 {
    if from == 0 || to >= from {
        return 0;
    }
    ((from - to) as f64 / from as f64 * 100.0).round() as u32
}

/// Encode a JSON value as a TOON document (2-space indent, comma delimiter).
pub fn to_toon(value: &Value) -> String {
    let mut out = String::new();
    match value {
        // A bare scalar or empty container is its own single line.
        Value::Object(map) if !map.is_empty() => write_object(&mut out, map, 0),
        Value::Array(items) if !items.is_empty() => write_root_array(&mut out, items),
        other => {
            out.push_str(&scalar(other));
            out.push('\n');
        }
    }
    out
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth * 2 {
        out.push(' ');
    }
}

/// Render a scalar (or empty container) exactly as TOON requires.
fn scalar(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => quote_if_needed(s),
        Value::Array(a) if a.is_empty() => "[]".to_string(),
        Value::Object(o) if o.is_empty() => "{}".to_string(),
        // Non-empty containers are never rendered inline as a scalar.
        _ => String::new(),
    }
}

fn is_scalar(v: &Value) -> bool {
    matches!(
        v,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
    ) || matches!(v, Value::Array(a) if a.is_empty())
        || matches!(v, Value::Object(o) if o.is_empty())
}

/// Quote a string only when TOON requires it, escaping the contents.
fn quote_if_needed(s: &str) -> String {
    let needs = s.is_empty()
        || s.starts_with('-')
        || s != s.trim()
        || matches!(s, "true" | "false" | "null")
        || looks_numeric(s)
        || s.chars()
            .any(|c| matches!(c, ':' | '"' | '\\' | ',' | '[' | ']' | '{' | '}') || c.is_control());
    if !needs {
        return s.to_string();
    }
    let mut q = String::with_capacity(s.len() + 2);
    q.push('"');
    for c in s.chars() {
        match c {
            '\\' => q.push_str("\\\\"),
            '"' => q.push_str("\\\""),
            '\n' => q.push_str("\\n"),
            '\r' => q.push_str("\\r"),
            '\t' => q.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(q, "\\u{:04x}", c as u32);
            }
            c => q.push(c),
        }
    }
    q.push('"');
    q
}

fn looks_numeric(s: &str) -> bool {
    // Matches the spec's numeric-like guard: -?digits(.digits)?(e[+-]?digits)?
    let mut chars = s.chars().peekable();
    if chars.peek() == Some(&'-') {
        chars.next();
    }
    let mut saw_digit = false;
    let consume_digits = |chars: &mut std::iter::Peekable<std::str::Chars>| {
        let mut any = false;
        while matches!(chars.peek(), Some(c) if c.is_ascii_digit()) {
            chars.next();
            any = true;
        }
        any
    };
    saw_digit |= consume_digits(&mut chars);
    if chars.peek() == Some(&'.') {
        chars.next();
        saw_digit |= consume_digits(&mut chars);
    }
    if matches!(chars.peek(), Some('e' | 'E')) {
        chars.next();
        if matches!(chars.peek(), Some('+' | '-')) {
            chars.next();
        }
        if !consume_digits(&mut chars) {
            return false;
        }
    }
    saw_digit && chars.next().is_none()
}

fn write_object(out: &mut String, map: &serde_json::Map<String, Value>, depth: usize) {
    for (key, val) in map {
        write_field(out, key, val, depth);
    }
}

/// Write one `key: value` entry (recursing for containers).
fn write_field(out: &mut String, key: &str, val: &Value, depth: usize) {
    indent(out, depth);
    out.push_str(&quote_key(key));
    match val {
        Value::Object(m) if !m.is_empty() => {
            out.push(':');
            out.push('\n');
            write_object(out, m, depth + 1);
        }
        Value::Array(items) if !items.is_empty() => {
            write_array_after_key(out, items, depth);
        }
        scalar_or_empty => {
            out.push(':');
            out.push(' ');
            out.push_str(&scalar(scalar_or_empty));
            out.push('\n');
        }
    }
}

/// Keys follow the same quoting rules as string values.
fn quote_key(key: &str) -> String {
    quote_if_needed(key)
}

/// Encode a non-empty array that appears as a field value, choosing the tightest
/// valid form: inline scalars, a table of uniform scalar-objects, or a list.
fn write_array_after_key(out: &mut String, items: &[Value], depth: usize) {
    if items.iter().all(is_scalar) {
        // Inline: key[N]: a,b,c
        let _ = write!(out, "[{}]: ", items.len());
        push_delimited(out, items.iter().map(scalar));
        out.push('\n');
    } else if let Some(fields) = uniform_object_fields(items) {
        // Table: key[N]{f1,f2}:  then rows at depth+1
        let _ = write!(out, "[{}]{{", items.len());
        push_delimited(out, fields.iter().map(|f| quote_if_needed(f)));
        out.push_str("}:\n");
        for item in items {
            let obj = item.as_object().expect("uniform object");
            indent(out, depth + 1);
            push_delimited(out, fields.iter().map(|f| scalar(&obj[*f])));
            out.push('\n');
        }
    } else {
        // Heterogeneous / nested: list form. key[N]: then each element as its own
        // block under a `-` marker, so the structure stays lossless.
        let _ = writeln!(out, "[{}]:", items.len());
        for item in items {
            write_list_item(out, item, depth + 1);
        }
    }
}

/// A root-level array (no key) — same choices, no `key` prefix.
fn write_root_array(out: &mut String, items: &[Value]) {
    if items.iter().all(is_scalar) {
        let _ = write!(out, "[{}]: ", items.len());
        push_delimited(out, items.iter().map(scalar));
        out.push('\n');
    } else if let Some(fields) = uniform_object_fields(items) {
        let _ = write!(out, "[{}]{{", items.len());
        push_delimited(out, fields.iter().map(|f| quote_if_needed(f)));
        out.push_str("}:\n");
        for item in items {
            let obj = item.as_object().expect("uniform object");
            indent(out, 1);
            push_delimited(out, fields.iter().map(|f| scalar(&obj[*f])));
            out.push('\n');
        }
    } else {
        let _ = writeln!(out, "[{}]:", items.len());
        for item in items {
            write_list_item(out, item, 1);
        }
    }
}

/// One element of a heterogeneous array: `- ` then the element's encoding.
fn write_list_item(out: &mut String, item: &Value, depth: usize) {
    match item {
        Value::Object(m) if !m.is_empty() => {
            indent(out, depth);
            out.push_str("-\n");
            write_object(out, m, depth + 1);
        }
        Value::Array(a) if !a.is_empty() => {
            indent(out, depth);
            out.push('-');
            write_array_after_key(out, a, depth);
        }
        scalar_or_empty => {
            indent(out, depth);
            out.push_str("- ");
            out.push_str(&scalar(scalar_or_empty));
            out.push('\n');
        }
    }
}

/// If every item is an object with the *same* keys and all-scalar values, return
/// the field order (from the first item); otherwise `None` (not a table).
fn uniform_object_fields(items: &[Value]) -> Option<Vec<&str>> {
    let first = items.first()?.as_object()?;
    if first.is_empty() {
        return None;
    }
    let fields: Vec<&str> = first.keys().map(String::as_str).collect();
    for item in items {
        let obj = item.as_object()?;
        if obj.len() != fields.len() {
            return None;
        }
        for f in &fields {
            match obj.get(*f) {
                Some(v) if is_scalar(v) => {}
                _ => return None,
            }
        }
    }
    Some(fields)
}

fn push_delimited(out: &mut String, mut parts: impl Iterator<Item = String>) {
    if let Some(first) = parts.next() {
        out.push_str(&first);
        for p in parts {
            out.push(',');
            out.push_str(&p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn object_lines() {
        // serde_json's Map is a BTreeMap, so keys emit in sorted order — fine, and
        // deterministic.
        let v = json!({ "id": 123, "name": "Ada", "active": true, "note": null });
        assert_eq!(
            to_toon(&v),
            "active: true\nid: 123\nname: Ada\nnote: null\n"
        );
    }

    #[test]
    fn uniform_object_array_is_a_table() {
        let v = json!({
            "users": [
                { "id": 1, "name": "Ada" },
                { "id": 2, "name": "Linus" },
            ]
        });
        assert_eq!(to_toon(&v), "users[2]{id,name}:\n  1,Ada\n  2,Linus\n");
    }

    #[test]
    fn scalar_array_is_inline() {
        let v = json!({ "tags": ["admin", "ops", "dev"] });
        assert_eq!(to_toon(&v), "tags[3]: admin,ops,dev\n");
    }

    #[test]
    fn nested_object() {
        let v = json!({ "user": { "id": 1, "name": "Ada" } });
        assert_eq!(to_toon(&v), "user:\n  id: 1\n  name: Ada\n");
    }

    #[test]
    fn strings_are_quoted_only_when_required() {
        // colon, comma, numeric-like, leading dash, reserved word, empty → quoted.
        let v = json!({
            "a": "plain",
            "b": "has: colon",
            "c": "1,2",
            "d": "42",
            "e": "-x",
            "f": "true",
            "g": "",
        });
        let t = to_toon(&v);
        assert!(t.contains("a: plain\n"));
        assert!(t.contains("b: \"has: colon\"\n"));
        assert!(t.contains("c: \"1,2\"\n"));
        assert!(t.contains("d: \"42\"\n"));
        assert!(t.contains("e: \"-x\"\n"));
        assert!(t.contains("f: \"true\"\n"));
        assert!(t.contains("g: \"\"\n"));
    }

    #[test]
    fn escapes_control_and_quotes() {
        let v = json!({ "s": "line\nwith \"quote\" and \\slash" });
        assert_eq!(
            to_toon(&v),
            "s: \"line\\nwith \\\"quote\\\" and \\\\slash\"\n"
        );
    }

    #[test]
    fn heterogeneous_array_falls_back_to_a_list() {
        // rows have different shapes → not a table, list form keeps it lossless.
        let v = json!({ "mixed": [ { "id": 1 }, 7, "x" ] });
        let t = to_toon(&v);
        assert!(t.starts_with("mixed[3]:\n"), "got: {t}");
        assert!(t.contains("  -\n    id: 1\n"));
        assert!(t.contains("  - 7\n"));
        assert!(t.contains("  - x\n"));
    }

    #[test]
    fn empty_containers() {
        assert_eq!(to_toon(&json!({ "items": [] })), "items: []\n");
        assert_eq!(to_toon(&json!({ "meta": {} })), "meta: {}\n");
        assert_eq!(to_toon(&json!([])), "[]\n");
    }

    /// The point of the phase: TOON is materially smaller than JSON for a recall
    /// result (a uniform array of objects). We measure two proxies — raw bytes and
    /// an approximate token count (whitespace/punct splits) — since the true LLM-
    /// tokenizer reduction is TOON's published ~40%. The gate asserts a clear win
    /// on both, so a regression that bloated the encoding would fail.
    #[test]
    fn toon_is_smaller_than_json_for_a_recall_result() {
        // 25 hits, the shape recall returns.
        let hits: Vec<Value> = (0..25)
            .map(|i| {
                json!({
                    "id": format!("doc{i}"),
                    "text": format!("The quick brown fox jumps over lazy dog number {i}."),
                    "score": 0.9 - i as f64 * 0.01,
                })
            })
            .collect();
        let value = Value::Array(hits);

        let json = serde_json::to_string(&value).unwrap();
        let toon = to_toon(&value);

        let approx_tokens = |s: &str| {
            s.split(|c: char| c.is_whitespace() || "{}[]\":,".contains(c))
                .filter(|t| !t.is_empty())
                .count()
        };
        let (jb, tb) = (json.len(), toon.len());
        let (jt, tt) = (approx_tokens(&json), approx_tokens(&toon));
        println!(
            "TOON GATE — bytes {tb} vs json {jb} ({:.0}% smaller); ~tokens {tt} vs {jt} ({:.0}% fewer)",
            100.0 * (1.0 - tb as f64 / jb as f64),
            100.0 * (1.0 - tt as f64 / jt as f64),
        );
        // The table declares id/text/score once instead of per row → the field
        // names, quotes and braces stop repeating. The win is real but bounded by
        // the long `text` payloads (identical in both), so we gate at the honest
        // measured level (~26% bytes here), not the ~40% marketing figure that
        // assumes field-name-heavy rows.
        assert!(
            tb * 5 < jb * 4,
            "TOON must be ≥20% fewer bytes: {tb} vs {jb}"
        );
        assert!(tt < jt, "TOON must use fewer approx tokens: {tt} vs {jt}");
        // And it really is the table form (keys sort — BTreeMap).
        assert!(
            toon.starts_with("[25]{id,score,text}:\n"),
            "got: {}",
            &toon[..40]
        );
    }

    fn sample_candidates(n: usize) -> Vec<Candidate> {
        (0..n)
            .map(|i| Candidate {
                id: crate::types::Id::new(format!("doc{i}")),
                text: format!("A representative recall hit number {i} with some prose."),
                score: 0.9 - i as f32 * 0.01,
                metadata: Default::default(),
                vector: None,
                highlights: Vec::new(),
            })
            .collect()
    }

    #[test]
    fn negotiation_serves_toon_when_accepted_and_recommends_otherwise() {
        let cands = sample_candidates(20);

        // Client accepts TOON → served TOON, no nagging recommendation.
        let yes = render_candidates(&cands, true);
        assert_eq!(yes.format, ContextFormat::Toon);
        assert!(yes.body.starts_with("[20]{id,score,text}:\n"));
        assert!(yes.recommendation.is_none());

        // Client did not → served JSON, with a concrete, honest recommendation.
        let no = render_candidates(&cands, false);
        assert_eq!(no.format, ContextFormat::Json);
        assert!(no.body.starts_with('['), "JSON body");
        let rec = no.recommendation.expect("should offer TOON");
        assert!(
            rec.bytes_saved_pct >= 15,
            "real savings: {}",
            rec.bytes_saved_pct
        );
        assert!(rec.headline.contains("TOON"));
        assert!(rec.why.to_lowercase().contains("lossless"));
        assert!(rec.how_to_enable.contains("accepts_toon"));
    }
}
