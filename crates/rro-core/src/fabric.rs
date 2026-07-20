//! The fabric envelope — the baseline metadata every fabric signal carries.
//!
//! The devstation fabric emits a signal as content flows (a `clyffy code` turn,
//! later a file/git event, …). Each signal carries a small, well-known set of
//! metadata **at emit time** so the rest of the engine — rrd classification,
//! recall filtering, connectome growth, security redaction — works precisely
//! and cheaply instead of reconstructing context later.
//!
//! This is the **baseline**, deliberately minimal and **evolvable**: the values
//! live as namespaced `fab.*` keys on the ordinary [`Metadata`] bag rather than
//! a rigid struct, so a new dimension is a plain key today and gets promoted to
//! a typed field once DuckDB analytics show it matters. Keys already in the bag
//! that are not part of the envelope are never disturbed.

use crate::types::Metadata;
use serde_json::Value;

/// Namespaced keys for the baseline fabric envelope. Every fabric key shares the
/// `fab.` prefix so it never collides with a consumer's own metadata.
pub mod keys {
    /// Which tap emitted the signal. See [`super::Source`].
    pub const SOURCE: &str = "fab.source";
    /// The session / locus the signal belongs to (e.g. a `clyffy code` session id).
    pub const SESSION: &str = "fab.session";
    /// Who or what caused the signal. See [`super::Actor`].
    pub const ACTOR: &str = "fab.actor";
    /// rrd domain hint, refined by the classifier downstream. Free-form.
    pub const DOMAIN: &str = "fab.domain";
    /// Tenant / project / scope region — feeds connectome growth. Free-form.
    pub const BOUNDARY: &str = "fab.boundary";
    /// Security class gating recall + redaction. See [`super::Security`].
    pub const SECURITY: &str = "fab.security";
    /// Emit time, epoch milliseconds.
    pub const TS_MS: &str = "fab.ts_ms";
}

/// Which tap emitted a fabric signal.
///
/// Open by design: [`Source::Other`] means a new tap is a *value*, not a
/// breaking change — the whole point of an evolvable baseline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A `clyffy code` turn (the v1 tap).
    ClyffyCode,
    /// A filesystem change.
    FileEvent,
    /// A git event (commit, checkout, …).
    GitEvent,
    /// A shell command.
    Shell,
    /// An editor / LSP event.
    Editor,
    /// Any tap not yet promoted to a named variant.
    Other(String),
}

impl Source {
    /// The canonical wire string stored under [`keys::SOURCE`].
    pub fn as_str(&self) -> &str {
        match self {
            Source::ClyffyCode => "clyffy_code",
            Source::FileEvent => "file_event",
            Source::GitEvent => "git_event",
            Source::Shell => "shell",
            Source::Editor => "editor",
            Source::Other(s) => s,
        }
    }

    /// Parse from the wire string; an unrecognized value becomes [`Source::Other`].
    pub fn parse(s: &str) -> Source {
        match s {
            "clyffy_code" => Source::ClyffyCode,
            "file_event" => Source::FileEvent,
            "git_event" => Source::GitEvent,
            "shell" => Source::Shell,
            "editor" => Source::Editor,
            other => Source::Other(other.to_string()),
        }
    }
}

/// Who or what caused a fabric signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Actor {
    /// The human operator.
    Operator,
    /// An AI agent acting on the operator's behalf.
    Agent,
    /// The system itself (a scheduled cycle, a daemon).
    System,
    /// Any actor not yet promoted to a named variant.
    Other(String),
}

impl Actor {
    /// The canonical wire string stored under [`keys::ACTOR`].
    pub fn as_str(&self) -> &str {
        match self {
            Actor::Operator => "operator",
            Actor::Agent => "agent",
            Actor::System => "system",
            Actor::Other(s) => s,
        }
    }

    /// Parse from the wire string; an unrecognized value becomes [`Actor::Other`].
    pub fn parse(s: &str) -> Actor {
        match s {
            "operator" => Actor::Operator,
            "agent" => Actor::Agent,
            "system" => Actor::System,
            other => Actor::Other(other.to_string()),
        }
    }
}

/// Security class of a fabric signal — gates recall visibility and redaction.
///
/// Closed on purpose (unlike [`Source`]/[`Actor`]): a *present but unrecognized*
/// value fails safe to [`Security::Secret`] (most restrictive) rather than
/// silently widening access. A *missing* key defaults to [`Security::Operator`]
/// — the operator's own devstation is the baseline scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Security {
    /// Freely recallable, safe to surface anywhere.
    Public,
    /// The operator's own context (the baseline default).
    #[default]
    Operator,
    /// Sensitive — redacted from general recall.
    Secret,
}

impl Security {
    /// The canonical wire string stored under [`keys::SECURITY`].
    pub fn as_str(&self) -> &str {
        match self {
            Security::Public => "public",
            Security::Operator => "operator",
            Security::Secret => "secret",
        }
    }

    /// Parse from the wire string. An unrecognized value fails safe to
    /// [`Security::Secret`] — we never widen access on an unknown class.
    pub fn parse(s: &str) -> Security {
        match s {
            "public" => Security::Public,
            "operator" => Security::Operator,
            "secret" => Security::Secret,
            _ => Security::Secret,
        }
    }
}

/// The baseline fabric envelope: the typed view of the `fab.*` keys on a
/// [`Metadata`] bag. Round-trips losslessly through [`FabricMeta::write_into`]
/// and [`FabricMeta::read`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FabricMeta {
    /// Which tap emitted the signal.
    pub source: Source,
    /// The session / locus the signal belongs to.
    pub session: String,
    /// Who or what caused it.
    pub actor: Actor,
    /// rrd domain hint (refined downstream), if the tap already knows one.
    pub domain: Option<String>,
    /// Tenant / project / scope region, if known.
    pub boundary: Option<String>,
    /// Security class.
    pub security: Security,
    /// Emit time, epoch milliseconds.
    pub ts_ms: u64,
}

impl FabricMeta {
    /// A minimal envelope: a source, its session, and the emit time. Actor
    /// defaults to [`Actor::Operator`], security to [`Security::Operator`], and
    /// domain/boundary are unset — fill them with the builder setters.
    pub fn new(source: Source, session: impl Into<String>, ts_ms: u64) -> Self {
        FabricMeta {
            source,
            session: session.into(),
            actor: Actor::Operator,
            domain: None,
            boundary: None,
            security: Security::Operator,
            ts_ms,
        }
    }

    /// Set the actor.
    pub fn with_actor(mut self, actor: Actor) -> Self {
        self.actor = actor;
        self
    }

    /// Set the rrd domain hint.
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Set the boundary (tenant / project / scope).
    pub fn with_boundary(mut self, boundary: impl Into<String>) -> Self {
        self.boundary = Some(boundary.into());
        self
    }

    /// Set the security class.
    pub fn with_security(mut self, security: Security) -> Self {
        self.security = security;
        self
    }

    /// Write the envelope's `fab.*` keys into an existing bag, leaving every
    /// other key untouched. Unset optionals write nothing.
    pub fn write_into(&self, m: &mut Metadata) {
        m.insert(keys::SOURCE.into(), Value::from(self.source.as_str()));
        m.insert(keys::SESSION.into(), Value::from(self.session.clone()));
        m.insert(keys::ACTOR.into(), Value::from(self.actor.as_str()));
        m.insert(keys::SECURITY.into(), Value::from(self.security.as_str()));
        m.insert(keys::TS_MS.into(), Value::from(self.ts_ms));
        if let Some(d) = &self.domain {
            m.insert(keys::DOMAIN.into(), Value::from(d.clone()));
        }
        if let Some(b) = &self.boundary {
            m.insert(keys::BOUNDARY.into(), Value::from(b.clone()));
        }
    }

    /// A fresh bag carrying only this envelope.
    pub fn to_metadata(&self) -> Metadata {
        let mut m = Metadata::new();
        self.write_into(&mut m);
        m
    }

    /// Read the envelope back from a bag. Returns `None` unless the identity
    /// keys ([`keys::SOURCE`], [`keys::SESSION`], [`keys::TS_MS`]) are all
    /// present — a bag without them is not a fabric signal. Missing optionals
    /// fall back to their defaults; a present-but-unknown security class fails
    /// safe (see [`Security::parse`]).
    pub fn read(m: &Metadata) -> Option<FabricMeta> {
        let source = Source::parse(str_at(m, keys::SOURCE)?);
        let session = str_at(m, keys::SESSION)?.to_string();
        let ts_ms = m.get(keys::TS_MS).and_then(Value::as_u64)?;
        let actor = str_at(m, keys::ACTOR).map(Actor::parse).unwrap_or(Actor::Operator);
        let security = str_at(m, keys::SECURITY)
            .map(Security::parse)
            .unwrap_or_default();
        Some(FabricMeta {
            source,
            session,
            actor,
            domain: str_at(m, keys::DOMAIN).map(str::to_string),
            boundary: str_at(m, keys::BOUNDARY).map(str::to_string),
            security,
            ts_ms,
        })
    }
}

/// Borrow a string value from the bag, if the key holds a JSON string.
fn str_at<'a>(m: &'a Metadata, key: &str) -> Option<&'a str> {
    m.get(key).and_then(Value::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_the_full_envelope() {
        let meta = FabricMeta::new(Source::ClyffyCode, "sess-1", 1_700_000_000_000)
            .with_actor(Actor::Operator)
            .with_domain("rust")
            .with_boundary("rro")
            .with_security(Security::Operator);
        let bag = meta.to_metadata();
        assert_eq!(FabricMeta::read(&bag), Some(meta));
    }

    #[test]
    fn read_none_without_identity_keys() {
        let mut bag = Metadata::new();
        bag.insert("fab.actor".into(), Value::from("operator"));
        assert_eq!(FabricMeta::read(&bag), None);
    }

    #[test]
    fn write_into_preserves_foreign_keys() {
        let mut bag = Metadata::new();
        bag.insert("mime".into(), Value::from("text/markdown"));
        FabricMeta::new(Source::FileEvent, "s", 1).write_into(&mut bag);
        assert_eq!(bag.get("mime").and_then(Value::as_str), Some("text/markdown"));
        assert_eq!(bag.get("fab.source").and_then(Value::as_str), Some("file_event"));
    }

    #[test]
    fn unknown_security_fails_safe_to_secret() {
        let mut bag = FabricMeta::new(Source::ClyffyCode, "s", 1).to_metadata();
        bag.insert(keys::SECURITY.into(), Value::from("who-knows"));
        assert_eq!(FabricMeta::read(&bag).unwrap().security, Security::Secret);
    }

    #[test]
    fn open_source_variant_round_trips() {
        assert_eq!(Source::parse("browser"), Source::Other("browser".into()));
        assert_eq!(Source::Other("browser".into()).as_str(), "browser");
    }
}
