//! The observers job server: receives job descriptions
//! (`ode_observers::jobs::JobSpec` — model, estimator, configuration, dt,
//! steps) over HTTP, runs them on blocking worker threads through
//! `JobSpec::run`, and serves their progress and outputs. The routes and
//! messages are those of `ode_observers_remote::protocol`; the desktop
//! observers viewer (and, later, the web front-end) talk to it through
//! `ode_observers_remote::RemoteRun`.
//!
//! ```sh
//! observers-server                       # listens on 127.0.0.1:8787
//! ODEON_SERVER_ADDR=0.0.0.0:9000 observers-server
//! ODEON_WEB_DIR=/path/to/dist observers-server   # serves that web client at /
//! ```
//!
//! Cross-origin requests are allowed from anywhere (the web client may be
//! served by `trunk serve` on another port). The web page (a `trunk
//! build` of `observers-client`) is served at `/` from `ODEON_WEB_DIR`
//! or, by default, from `apps/observers-client/dist` of this workspace
//! (an absolute path fixed at compile time, so the working directory does
//! not matter) whenever that directory exists — one origin for the page
//! and its jobs, no configuration.
//!
//! Runs live in memory until they are deleted (the client deletes its run
//! when it drops the handle, i.e. once it has taken the output or
//! cancelled). A job that panics (a malformed description) is reported in
//! its status rather than killing the server.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use ode_observers::jobs::{JobOutput, JobSpec};
use ode_observers::progress::Progress;
use ode_observers_remote::protocol::{NewRun, RunStatus};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

/// One run: its progress handle (shared with the worker), and its result
/// once the worker returned.
struct Run {
    total: usize,
    progress: Arc<Progress>,
    started: Instant,
    result: Mutex<Option<Result<Arc<JobOutput>, String>>>,
}

#[derive(Clone, Default)]
struct Runs {
    runs: Arc<Mutex<HashMap<u64, Arc<Run>>>>,
    next_id: Arc<AtomicU64>,
}

impl Runs {
    fn get(&self, id: u64) -> Option<Arc<Run>> {
        self.runs.lock().unwrap().get(&id).cloned()
    }
}

/// The server's routes (see `ode_observers_remote::protocol`), open to
/// cross-origin callers; with `web_dir`, the static web client at `/`.
pub fn router(web_dir: Option<&str>) -> Router {
    let api = Router::new()
        .route("/runs", post(create))
        .route("/runs/{id}", get(status).delete(cancel))
        .route("/runs/{id}/output", get(output))
        .with_state(Runs::default());
    let router = match web_dir {
        Some(dir) => api.fallback_service(ServeDir::new(dir)),
        None => api,
    };
    router.layer(CorsLayer::permissive())
}

/// `POST /runs`: start the job on a blocking worker thread.
async fn create(State(runs): State<Runs>, Json(job): Json<JobSpec>) -> Result<Json<NewRun>, (StatusCode, String)> {
    let (dim, vars) = (job.model.dim(), job.config.vars.len());
    if vars != dim {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("the configuration has {vars} variables, the {} model {dim}", job.model.kind()),
        ));
    }
    let id = runs.next_id.fetch_add(1, Ordering::Relaxed) + 1;
    let run = Arc::new(Run {
        total: job.steps.max(1),
        progress: Arc::new(Progress::default()),
        started: Instant::now(),
        result: Mutex::new(None),
    });
    runs.runs.lock().unwrap().insert(id, run.clone());
    let worker = run.clone();
    tokio::spawn(async move {
        let job_run = worker.clone();
        let joined = tokio::task::spawn_blocking(move || {
            let out = job.run(&job_run.progress);
            *job_run.result.lock().unwrap() = Some(Ok(Arc::new(out)));
        })
        .await;
        if let Err(e) = joined {
            *worker.result.lock().unwrap() = Some(Err(format!("the job panicked: {e}")));
        }
    });
    Ok(Json(NewRun { id }))
}

/// `GET /runs/{id}`: progress of the run.
async fn status(State(runs): State<Runs>, Path(id): Path<u64>) -> Result<Json<RunStatus>, StatusCode> {
    let run = runs.get(id).ok_or(StatusCode::NOT_FOUND)?;
    let result = run.result.lock().unwrap();
    Ok(Json(RunStatus {
        id,
        done: run.progress.done(),
        total: run.total,
        finished: result.is_some(),
        elapsed: run.started.elapsed().as_secs_f64(),
        error: match &*result {
            Some(Err(e)) => Some(e.clone()),
            _ => None,
        },
    }))
}

/// `GET /runs/{id}/output`: the output of a finished run.
async fn output(State(runs): State<Runs>, Path(id): Path<u64>) -> Result<Json<Arc<JobOutput>>, (StatusCode, String)> {
    let run = runs.get(id).ok_or((StatusCode::NOT_FOUND, format!("no run {id}")))?;
    let result = run.result.lock().unwrap();
    match &*result {
        None => Err((StatusCode::CONFLICT, format!("run {id} is not finished"))),
        Some(Err(e)) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.clone())),
        Some(Ok(out)) => Ok(Json(out.clone())),
    }
}

/// `DELETE /runs/{id}`: ask the job to stop at its next iteration and
/// forget the run.
async fn cancel(State(runs): State<Runs>, Path(id): Path<u64>) -> StatusCode {
    match runs.runs.lock().unwrap().remove(&id) {
        Some(run) => {
            run.progress.cancel();
            StatusCode::NO_CONTENT
        }
        None => StatusCode::NOT_FOUND,
    }
}

/// The directory of the web page to serve: `ODEON_WEB_DIR` (relative to
/// the working directory), else the workspace's
/// `apps/observers-client/dist`; `None`, with a warning, when it does not
/// exist.
fn web_dir() -> Option<String> {
    let default = concat!(env!("CARGO_MANIFEST_DIR"), "/../observers-client/dist");
    let (dir, from_env) = match std::env::var("ODEON_WEB_DIR").ok().filter(|d| !d.trim().is_empty()) {
        Some(d) => (d, true),
        None => (default.to_string(), false),
    };
    let path = std::path::Path::new(&dir);
    if path.join("index.html").is_file() {
        return Some(path.canonicalize().map_or(dir, |p| p.display().to_string()));
    }
    if from_env {
        eprintln!(
            "warning: ODEON_WEB_DIR={dir} has no index.html (resolved from {}), not serving a web client",
            std::env::current_dir().map_or("?".into(), |d| d.display().to_string())
        );
    }
    None
}

#[tokio::main]
async fn main() {
    let addr = std::env::var("ODEON_SERVER_ADDR").unwrap_or_else(|_| "127.0.0.1:8787".to_string());
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("cannot listen on {addr}: {e}"));
    let web_dir = web_dir();
    println!("observers-server listening on http://{}", listener.local_addr().unwrap());
    match &web_dir {
        Some(dir) => println!("serving the web client from {dir} at /"),
        None => println!("no web client to serve (build one: cd apps/observers-client && trunk build --release)"),
    }
    axum::serve(listener, router(web_dir.as_deref())).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ode_models::noise::NoiseModel;
    use ode_models::spec::ModelSpec;
    use ode_observers::jobs::{Estimator, FilterConfig, FilterOutput, TrackerOutput};
    use ode_observers_remote::RemoteRun;
    use std::time::Duration;

    /// The server on an ephemeral port of the test's runtime; returns its
    /// base URL.
    async fn serve() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router(None)).await.unwrap() });
        format!("http://{addr}")
    }

    fn spring_job(estimator: Estimator, steps: usize) -> JobSpec {
        let dt = 0.02;
        let mut config = FilterConfig::defaults(
            vec!["y_1".into(), "v_1".into()],
            vec![false, false],
            &[(0, 1)],
            &vec![vec![0.15, 0.0]; 11],
        );
        for v in &mut config.vars {
            v.sigma = 5.0;
        }
        config.n_snapshots = 3;
        JobSpec {
            model: ModelSpec::Spring {
                n: 1,
                rho: 1.0,
                a: 1.0,
                y0: vec![0.15],
                v0: vec![0.0],
                obs: 0,
                noise: NoiseModel::Gaussian { std: 0.01 },
                noise_seed: 1000,
            },
            estimator,
            config,
            dt,
            steps,
        }
    }

    /// Drive a remote run to its end from a blocking thread (the client is
    /// callback-based and polls at most every 150 ms).
    async fn take<T: TryFrom<JobOutput, Error = String> + Send + 'static>(run: RemoteRun<T>) -> Result<T, String> {
        tokio::task::spawn_blocking(move || {
            let deadline = Instant::now() + Duration::from_secs(60);
            loop {
                if let Some(e) = run.error() {
                    return Err(e);
                }
                if let Some(out) = run.try_take() {
                    return Ok(out);
                }
                assert!(Instant::now() < deadline, "the remote run did not finish in time");
                std::thread::sleep(Duration::from_millis(20));
            }
        })
        .await
        .unwrap()
    }

    /// A filter job posted through the client runs on the server and comes
    /// back equal to the local run, bit for bit (JSON with exact floats).
    #[tokio::test]
    async fn remote_filter_run_equals_the_local_run() {
        let base = serve().await;
        let job = spring_job(Estimator::Filter, 20);
        let remote = take(RemoteRun::<FilterOutput>::spawn(&base, &job)).await.unwrap();
        let JobOutput::Filter(local) = job.run(&Progress::default()) else { unreachable!() };
        assert_eq!(remote.estimates, local.estimates);
        assert_eq!(remote.marginals, local.marginals);
        assert_eq!(remote.reference, local.reference);
        assert_eq!(remote.saved_steps, local.saved_steps);
        assert_eq!(remote.axes, local.axes);
    }

    /// The output type the client asks for must match the estimator: a
    /// tracker output requested from a filter run is an error, not a
    /// silent mismatch.
    #[tokio::test]
    async fn wrong_output_kind_is_an_error() {
        let base = serve().await;
        let job = spring_job(Estimator::Filter, 5);
        let err = take(RemoteRun::<TrackerOutput>::spawn(&base, &job)).await.err().expect("an error was expected");
        assert!(err.contains("expected the tracker output"), "{err}");
    }

    /// A configuration of the wrong dimension is refused with 400 and the
    /// message reaches the client; an unknown id is 404.
    #[tokio::test]
    async fn bad_jobs_are_refused() {
        let base = serve().await;
        let mut job = spring_job(Estimator::Tracker, 5);
        job.config.vars.pop();
        let err = take(RemoteRun::<TrackerOutput>::spawn(&base, &job)).await.err().expect("an error was expected");
        assert!(err.contains("400") && err.contains("1 variables"), "{err}");

        let response = tokio::task::spawn_blocking(move || {
            ehttp::fetch_blocking(&ehttp::Request::get(format!("{base}/runs/999"))).unwrap()
        })
        .await
        .unwrap();
        assert_eq!(response.status, 404);
    }

    /// Dropping the handle deletes the run on the server: its status is
    /// gone afterwards.
    #[tokio::test]
    async fn dropping_the_handle_cancels_the_run() {
        let base = serve().await;
        let job = spring_job(Estimator::Filter, 100_000);
        let run = RemoteRun::<FilterOutput>::spawn(&base, &job);
        // Wait for the id, then drop.
        let b = base.clone();
        let id = tokio::task::spawn_blocking(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                run.try_take();
                let list = ehttp::fetch_blocking(&ehttp::Request::get(format!("{b}/runs/1"))).unwrap();
                if list.status == 200 {
                    drop(run);
                    return 1u64;
                }
                assert!(Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(20));
            }
        })
        .await
        .unwrap();
        let b = base.clone();
        let gone = tokio::task::spawn_blocking(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                let r = ehttp::fetch_blocking(&ehttp::Request::get(format!("{b}/runs/{id}"))).unwrap();
                if r.status == 404 {
                    return true;
                }
                if Instant::now() > deadline {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        })
        .await
        .unwrap();
        assert!(gone, "the run was not deleted");
    }
}
