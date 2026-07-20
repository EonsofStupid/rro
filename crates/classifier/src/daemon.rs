//! The Reason Ready daemon: a long-lived, message-driven classifier service.
//!
//! Any [`Classifier`] can be run as an embedded daemon. Requests arrive over a
//! channel and are answered on a per-request oneshot, so the flow — or a remote
//! peer over `rro-net` — can ask "is this ready to reason on?" without owning
//! the model. This is the shape the tuned DevPULSE classifier will run in.

use std::sync::Arc;

use rro_core::{Candidate, Classifier, Readiness, Result, RroError};
use tokio::sync::{mpsc, oneshot};

/// A single readiness request routed to the daemon.
struct Job {
    query: String,
    context: Vec<Candidate>,
    reply: oneshot::Sender<Result<Readiness>>,
}

/// A cheap, cloneable handle for submitting readiness requests to the daemon.
#[derive(Clone)]
pub struct DaemonHandle {
    tx: mpsc::Sender<Job>,
}

impl DaemonHandle {
    /// Ask the daemon to judge readiness. Awaits the daemon's answer.
    pub async fn classify(&self, query: &str, context: Vec<Candidate>) -> Result<Readiness> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job {
                query: query.to_string(),
                context,
                reply,
            })
            .await
            .map_err(|_| RroError::Classify("daemon stopped".into()))?;
        rx.await
            .map_err(|_| RroError::Classify("daemon dropped the request".into()))?
    }
}

/// The running daemon task; drop or `shutdown` to stop it.
pub struct ReasonReadyDaemon {
    handle: DaemonHandle,
    task: tokio::task::JoinHandle<()>,
}

impl ReasonReadyDaemon {
    /// Spawn a daemon around `classifier`. Returns immediately.
    pub fn spawn(classifier: Arc<dyn Classifier>) -> Self {
        let (tx, mut rx) = mpsc::channel::<Job>(256);
        let task = tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                let verdict = classifier.classify(&job.query, &job.context).await;
                // If the requester is gone, drop the answer silently.
                let _ = job.reply.send(verdict);
            }
            tracing::debug!("reason-ready daemon: channel closed, stopping");
        });
        ReasonReadyDaemon {
            handle: DaemonHandle { tx },
            task,
        }
    }

    /// A handle for submitting requests.
    pub fn handle(&self) -> DaemonHandle {
        self.handle.clone()
    }

    /// Stop the daemon deterministically.
    ///
    /// The task is aborted rather than waiting for every outstanding
    /// [`DaemonHandle`] to drop — handles are cloneable and may outlive the
    /// daemon, so relying on the channel closing would hang. In-flight requests
    /// whose reply has not been sent resolve to a "daemon stopped" error on the
    /// caller side.
    pub async fn shutdown(self) {
        self.task.abort();
        let _ = self.task.await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HeuristicClassifier;

    #[tokio::test]
    async fn daemon_answers_requests() {
        let daemon = ReasonReadyDaemon::spawn(Arc::new(HeuristicClassifier::new()));
        let h = daemon.handle();
        let ctx = vec![Candidate::new("1", "alpha beta gamma", 1.0)];
        let r = h.classify("alpha beta gamma", ctx).await.unwrap();
        assert!(r.ready);
        daemon.shutdown().await;
    }
}
