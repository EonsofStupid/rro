//! Session triggers: *when* RRD runs.
//!
//! RRD fires at the start of every new conversation, and on resume after the
//! session has sat idle past a threshold — the moments a system is "coming
//! back" and must re-orient. On fire, the caller routes the fresh context
//! through the intent router: the operator *should* pick a mode (Dev,
//! Creative/media, …), but the engine also detects "we actually need to be
//! in X mode", switches, and the expert state absorbs the standing task
//! list. Intent and tags are how RRD *evolves* session over session.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Why RRD fired (or would).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FireReason {
    /// A brand-new conversation.
    ConversationStart,
    /// The session resumed after idling past the threshold.
    IdleResume,
}

/// Session events the trigger observes.
#[derive(Debug, Clone, Copy)]
pub enum SessionEvent {
    /// A new conversation began.
    ConversationStart,
    /// Activity arrived in an existing session.
    Activity,
}

/// Decides when RRD must re-orient the session.
#[derive(Debug)]
pub struct SessionTrigger {
    idle_threshold: Duration,
    last_activity: Option<Instant>,
}

impl SessionTrigger {
    /// A trigger with the given idle threshold.
    pub fn new(idle_threshold: Duration) -> Self {
        SessionTrigger {
            idle_threshold,
            last_activity: None,
        }
    }

    /// Observe an event; returns `Some(reason)` when RRD must fire.
    ///
    /// Conversation start always fires. Activity fires only when the gap
    /// since the previous activity exceeds the idle threshold (a resume).
    pub fn observe(&mut self, event: SessionEvent) -> Option<FireReason> {
        let now = Instant::now();
        let fired = match event {
            SessionEvent::ConversationStart => Some(FireReason::ConversationStart),
            SessionEvent::Activity => match self.last_activity {
                Some(prev) if now.duration_since(prev) > self.idle_threshold => {
                    Some(FireReason::IdleResume)
                }
                Some(_) => None,
                // First activity with no start event recorded: treat as start.
                None => Some(FireReason::ConversationStart),
            },
        };
        self.last_activity = Some(now);
        if let Some(reason) = fired {
            rro_core::events::emit("rrd.trigger", serde_json::json!({ "reason": reason }));
        }
        fired
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_fires_and_rapid_activity_does_not() {
        let mut t = SessionTrigger::new(Duration::from_millis(50));
        assert_eq!(
            t.observe(SessionEvent::ConversationStart),
            Some(FireReason::ConversationStart)
        );
        assert_eq!(t.observe(SessionEvent::Activity), None);
        assert_eq!(t.observe(SessionEvent::Activity), None);
    }

    #[test]
    fn idle_resume_fires() {
        let mut t = SessionTrigger::new(Duration::from_millis(10));
        t.observe(SessionEvent::ConversationStart);
        std::thread::sleep(Duration::from_millis(25));
        assert_eq!(
            t.observe(SessionEvent::Activity),
            Some(FireReason::IdleResume)
        );
        // Immediately after, no refire.
        assert_eq!(t.observe(SessionEvent::Activity), None);
    }
}
