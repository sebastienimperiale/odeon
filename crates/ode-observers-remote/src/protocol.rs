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
//! | `GET /runs/{id}/output`   | `JobOutput` (404 unknown, 409 not finished, 500 failed) — JSON, or the compact [`BINARY`] encoding when the request accepts it |
//! | `DELETE /runs/{id}`       | cancels at the next iteration and forgets the run (204; 404 unknown) |
//!
//! Answers are gzip-compressed for clients announcing `Accept-Encoding:
//! gzip` (browsers and ureq do). When the server runs with a token (`ODEON_TOKEN`), every one of these
//! needs `Authorization: Bearer <token>` (401 otherwise); a job beyond the
//! server's limits is refused with 400 and a message naming the limit, and
//! one more run than it accepts at once with 429.

use ode_observers::jobs::{BoxOutput, Estimator, FilterConfig, FilterOutput, JobOutput, ParticleOutput, TrackerOutput};
use ode_models_spec::spec::TwinSpec;
use serde::{Deserialize, Serialize};

/// Content type of the binary encoding of a `JobOutput`
/// ([`encode_output`] / [`decode_output`]): what [`RemoteRun`] asks for
/// with `Accept`, since a filter output is tens of megabytes as JSON text
/// and parsing it in the browser is the slow part of a run.
///
/// [`RemoteRun`]: crate::RemoteRun
pub const BINARY: &str = "application/octet-stream";

/// The binary encoding of an output: one tag byte naming the estimator
/// (0 filter, 1 window, 2 particles, 3 tracker), then the output itself in
/// `postcard` (serde-derived; floats as 8 little-endian bytes, exact).
/// The tag replaces `JobOutput`'s adjacently tagged JSON form, which only
/// self-describing formats can read.
pub fn encode_output(out: &JobOutput) -> postcard::Result<Vec<u8>> {
    match out {
        JobOutput::Filter(o) => postcard::to_allocvec(&(0u8, o)),
        JobOutput::Window(o) => postcard::to_allocvec(&(1u8, o)),
        JobOutput::Particles(o) => postcard::to_allocvec(&(2u8, o)),
        JobOutput::Tracker(o) => postcard::to_allocvec(&(3u8, o)),
    }
}

/// Inverse of [`encode_output`].
pub fn decode_output(bytes: &[u8]) -> postcard::Result<JobOutput> {
    let (tag, rest): (u8, &[u8]) = postcard::take_from_bytes(bytes)?;
    Ok(match tag {
        0 => JobOutput::Filter(postcard::from_bytes::<FilterOutput>(rest)?),
        1 => JobOutput::Window(postcard::from_bytes::<BoxOutput>(rest)?),
        2 => JobOutput::Particles(postcard::from_bytes::<ParticleOutput>(rest)?),
        3 => JobOutput::Tracker(postcard::from_bytes::<TrackerOutput>(rest)?),
        _ => return Err(postcard::Error::DeserializeBadEnum),
    })
}

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
