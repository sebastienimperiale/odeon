//! The messages and routes shared by the server and the client. The job
//! itself is a [`JobSpec`](ode_observers::jobs::JobSpec) and its result a
//! [`JobOutput`](ode_observers::jobs::JobOutput), both JSON as `ode-observers`
//! serializes them.
//!
//! | Route                     | Body / answer                                   |
//! |---------------------------|-------------------------------------------------|
//! | `POST /runs`              | `JobSpec` → [`NewRun`] (400 if the configuration's dimension is not the model's) |
//! | `GET /runs/{id}`          | [`RunStatus`] (404 unknown id)                  |
//! | `GET /runs/{id}/output`   | `JobOutput` (404 unknown, 409 not finished, 500 failed) |
//! | `DELETE /runs/{id}`       | cancels at the next iteration and forgets the run (204; 404 unknown) |

use serde::{Deserialize, Serialize};

/// Answer to `POST /runs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewRun {
    pub id: u64,
}

/// Answer to `GET /runs/{id}`: the run's progress.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunStatus {
    pub id: u64,
    /// Steps done so far.
    pub done: usize,
    /// Steps requested.
    pub total: usize,
    /// The job returned (its output can be fetched) — also `true` after a
    /// failure, see `error`.
    pub finished: bool,
    /// Seconds since the run started on the server.
    pub elapsed: f64,
    /// Why the run failed on the server (a panic in the job), if it did.
    pub error: Option<String>,
}

/// `{base}/runs`.
pub fn runs_url(base: &str) -> String {
    format!("{}/runs", base.trim_end_matches('/'))
}

/// `{base}/runs/{id}`.
pub fn run_url(base: &str, id: u64) -> String {
    format!("{}/runs/{id}", base.trim_end_matches('/'))
}

/// `{base}/runs/{id}/output`.
pub fn output_url(base: &str, id: u64) -> String {
    format!("{}/runs/{id}/output", base.trim_end_matches('/'))
}
