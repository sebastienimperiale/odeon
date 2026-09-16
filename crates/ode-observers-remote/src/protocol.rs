//! The messages and routes shared by the server and the client. The job
//! itself is a [`JobSpec`](ode_observers::jobs::JobSpec) and its result a
//! [`JobOutput`](ode_observers::jobs::JobOutput), both JSON as `ode-observers`
//! serializes them.
//!
//! | Route                     | Body / answer                                   |
//! |---------------------------|-------------------------------------------------|
//! | `POST /runs`              | `JobSpec` (model, reference, estimator, configuration) → [`NewRun`] (400 if the configuration's or the reference's dimension is not the model's) |
//! | `POST /twin-runs`         | [`TwinJob`] (a twin experiment instead of a reference: the server generates it) → [`NewRun`] |
//! | `GET /runs/{id}`          | [`RunStatus`] (404 unknown id)                  |
//! | `GET /runs/{id}/output`   | `JobOutput` (404 unknown, 409 not finished, 500 failed) |
//! | `DELETE /runs/{id}`       | cancels at the next iteration and forgets the run (204; 404 unknown) |
//!
//! When the server runs with a token (`ODEON_TOKEN`), every one of these
//! needs `Authorization: Bearer <token>` (401 otherwise); a job beyond the
//! server's limits is refused with 400 and a message naming the limit, and
//! one more run than it accepts at once with 429.

use ode_observers::jobs::{Estimator, FilterConfig};
use ode_models_spec::spec::TwinSpec;
use serde::{Deserialize, Serialize};

/// Body of `POST /twin-runs`: an estimator run whose reference the server
/// generates from a twin experiment — for clients without a model
/// implementation of their own (a script, a page with only the forms).
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct TwinJob {
    pub twin: TwinSpec,
    pub estimator: Estimator,
    pub config: FilterConfig,
}

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

/// `{base}/twin-runs`.
pub fn twin_runs_url(base: &str) -> String {
    format!("{}/twin-runs", base.trim_end_matches('/'))
}

/// `{base}/runs/{id}`.
pub fn run_url(base: &str, id: u64) -> String {
    format!("{}/runs/{id}", base.trim_end_matches('/'))
}

/// `{base}/runs/{id}/output`.
pub fn output_url(base: &str, id: u64) -> String {
    format!("{}/runs/{id}/output", base.trim_end_matches('/'))
}
