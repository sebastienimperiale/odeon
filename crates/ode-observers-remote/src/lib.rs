//! `ode-observers-remote`: the wire protocol of the Odeon observers server
//! and its client.
//!
//! * [`protocol`] — the JSON messages and the routes: a
//!   [`JobSpec`] is POSTed to `/runs` and
//!   answered by a [`protocol::NewRun`]; `GET /runs/{id}` returns a
//!   [`protocol::RunStatus`]; `GET /runs/{id}/output` the [`JobOutput`]
//!   once finished;
//!   `DELETE /runs/{id}` cancels (and forgets) the run.
//! * [`RemoteRun`] — the client: a handle on a run executing on the server,
//!   with the same surface as a local run handle (`progress`, `elapsed`,
//!   `try_take`) plus `error`, so a front-end can hold either. A run is
//!   posted either as a twin experiment ([`RemoteRun::spawn_twin`], a few
//!   kilobytes: the server generates the reference — what the viewers
//!   use) or with its reference ([`RemoteRun::spawn`], for observations
//!   that are not a twin experiment's). Built on
//!   `ehttp`, which issues the requests from a thread on native and through
//!   `fetch` on wasm; nothing here is UI-specific.
//!
//! Jobs travel as JSON; with `serde_json`'s `float_roundtrip` feature (on
//! here and on the server) a run fetched from the server equals the local
//! run bit for bit. The output comes back in the binary encoding
//! [`protocol::BINARY`] (`postcard`, exact floats, a third of the JSON's
//! size and a copy rather than a text scan to decode) and the server
//! gzips it in transit. Time is `web_time::Instant` (std's on native, the
//! browser clock on wasm), so the client builds for the web.

pub mod protocol;

use ode_observers::jobs::{JobOutput, JobSpec};
use protocol::{NewRun, RunStatus, TwinJob};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use web_time::Instant;

/// Minimum time between two progress requests of one run.
const POLL_INTERVAL: Duration = Duration::from_millis(150);

/// What the client knows about its run; updated by the request callbacks.
#[derive(Default)]
struct State {
    /// The server's id of the run, once `POST /runs` answered.
    id: Option<u64>,
    /// Steps done, from the last status.
    done: usize,
    /// The server reported the run finished.
    finished: bool,
    /// A request is in flight (status or output): don't send another.
    in_flight: bool,
    last_poll: Option<Instant>,
    /// The fetched output, waiting to be taken.
    output: Option<JobOutput>,
    /// The first error met (connection, HTTP status, decoding, or the
    /// server's own error for the run); sticky.
    error: Option<String>,
}

/// A run executing on the observers server: created by [`spawn`](Self::spawn),
/// polled by [`try_take`](Self::try_take) (call it every frame — it issues
/// at most one request per `POLL_INTERVAL`, 150 ms), cancelled on drop. `T` is
/// the output type expected by the caller, extracted from the
/// [`JobOutput`] the server returns (`FilterOutput`, `TrackerOutput`, …,
/// which implement `TryFrom<JobOutput>`).
pub struct RemoteRun<T> {
    base: String,
    total: usize,
    started: Instant,
    state: Arc<Mutex<State>>,
    _out: PhantomData<fn() -> T>,
}

impl<T: TryFrom<JobOutput, Error = String>> RemoteRun<T> {
    /// POST the job description — reference included — to `/runs` of the
    /// server at `base` (e.g. `http://127.0.0.1:8787`) and return the
    /// handle at once; the id arrives asynchronously, and any failure
    /// surfaces through [`error`](Self::error).
    pub fn spawn(base: &str, job: &JobSpec) -> Self {
        Self::post(base, protocol::runs_url(base), serde_json::to_vec(job), job.steps())
    }

    /// POST a twin experiment to `/twin-runs`: the server generates the
    /// reference (deterministic in the twin's seeds, so it is the run the
    /// client computed for its own display) and runs the estimator on it.
    /// A few kilobytes whatever the run length.
    pub fn spawn_twin(base: &str, job: &TwinJob) -> Self {
        Self::post(base, protocol::twin_runs_url(base), serde_json::to_vec(job), job.twin.steps)
    }

    fn post(base: &str, url: String, body: serde_json::Result<Vec<u8>>, steps: usize) -> Self {
        let base = base.trim_end_matches('/').to_string();
        let state = Arc::new(Mutex::new(State::default()));
        let run = RemoteRun {
            base,
            total: steps.max(1),
            started: Instant::now(),
            state: state.clone(),
            _out: PhantomData,
        };
        match body {
            Ok(body) => {
                let mut request = ehttp::Request::post(url, body);
                request.headers = ehttp::Headers::new(&[
                    ("Accept", "application/json"),
                    ("Content-Type", "application/json"),
                ]);
                ehttp::fetch(request, move |response| {
                    let mut st = state.lock().unwrap();
                    match decode::<NewRun>(response) {
                        Ok(new) => st.id = Some(new.id),
                        Err(e) => st.error = Some(e),
                    }
                });
            }
            Err(e) => state.lock().unwrap().error = Some(format!("cannot encode the job: {e}")),
        }
        run
    }

    /// Fraction of steps done, in [0, 1].
    pub fn progress(&self) -> f32 {
        self.state.lock().unwrap().done as f32 / self.total as f32
    }

    /// Seconds since the job was posted.
    pub fn elapsed(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// The first error met, if any: the run is then dead (the server never
    /// answered, refused the job, or the job failed there).
    pub fn error(&self) -> Option<String> {
        self.state.lock().unwrap().error.clone()
    }

    /// The finished result, if the server is done and the output has
    /// arrived; otherwise issues the next status (or output) request when
    /// due. An output of the wrong kind is reported through
    /// [`error`](Self::error).
    pub fn try_take(&self) -> Option<T> {
        self.poll();
        let mut st = self.state.lock().unwrap();
        let out = st.output.take()?;
        match T::try_from(out) {
            Ok(t) => Some(t),
            Err(e) => {
                st.error = Some(e);
                None
            }
        }
    }

    /// Send a status request if none is in flight and the interval passed;
    /// its callback chains the output request once the run is finished.
    fn poll(&self) {
        let mut st = self.state.lock().unwrap();
        let Some(id) = st.id else { return };
        if st.error.is_some() || st.output.is_some() || st.finished || st.in_flight {
            return;
        }
        if st.last_poll.is_some_and(|t| t.elapsed() < POLL_INTERVAL) {
            return;
        }
        st.in_flight = true;
        st.last_poll = Some(Instant::now());
        drop(st);

        let state = self.state.clone();
        let base = self.base.clone();
        let request = ehttp::Request::get(protocol::run_url(&base, id));
        let mut output_request = ehttp::Request::get(protocol::output_url(&base, id));
        output_request.headers.insert("Accept", protocol::BINARY);
        ehttp::fetch(request, move |response| {
            let status = match decode::<RunStatus>(response) {
                Ok(s) => s,
                Err(e) => {
                    let mut st = state.lock().unwrap();
                    st.error = Some(e);
                    st.in_flight = false;
                    return;
                }
            };
            let mut st = state.lock().unwrap();
            st.done = status.done;
            if let Some(e) = status.error {
                st.error = Some(format!("the server's run failed: {e}"));
                st.in_flight = false;
                return;
            }
            if !status.finished {
                st.in_flight = false;
                return;
            }
            st.finished = true;
            drop(st);
            let state = state.clone();
            ehttp::fetch(output_request, move |response| {
                let mut st = state.lock().unwrap();
                match decode_output(response) {
                    Ok(out) => st.output = Some(out),
                    Err(e) => st.error = Some(e),
                }
                st.in_flight = false;
            });
        });
    }
}

impl<T> Drop for RemoteRun<T> {
    /// Cancel and forget the run on the server (a no-op for a run that
    /// already finished, apart from freeing its output there).
    fn drop(&mut self) {
        if let Some(id) = self.state.lock().unwrap().id {
            let mut request = ehttp::Request::get(protocol::run_url(&self.base, id));
            request.method = "DELETE".to_owned();
            ehttp::fetch(request, |_| {});
        }
    }
}

/// Decode a JSON response, turning transport failures and non-2xx statuses
/// into one error string.
fn decode<D: serde::de::DeserializeOwned>(response: ehttp::Result<ehttp::Response>) -> Result<D, String> {
    let response = check(response)?;
    serde_json::from_slice(&response.bytes).map_err(|e| format!("cannot decode the server's answer: {e}"))
}

/// Decode the output, in the binary encoding when the server used it
/// (its content type says so) and as JSON otherwise.
fn decode_output(response: ehttp::Result<ehttp::Response>) -> Result<JobOutput, String> {
    let response = check(response)?;
    let binary = response.headers.get("content-type").is_some_and(|t| t.starts_with(protocol::BINARY));
    if binary {
        protocol::decode_output(&response.bytes).map_err(|e| format!("cannot decode the server's binary output: {e}"))
    } else {
        serde_json::from_slice(&response.bytes).map_err(|e| format!("cannot decode the server's output: {e}"))
    }
}

/// Transport failures and non-2xx statuses as one error string.
fn check(response: ehttp::Result<ehttp::Response>) -> Result<ehttp::Response, String> {
    let response = response.map_err(|e| format!("no answer from the server: {e}"))?;
    if !response.ok {
        let text = response.text().unwrap_or("").trim();
        return Err(format!("server answered {} {}{}{}", response.status, response.status_text, if text.is_empty() { "" } else { ": " }, text));
    }
    Ok(response)
}
