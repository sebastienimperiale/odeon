//! The observers job server: receives job descriptions
//! (`ode_observers::jobs::JobSpec` — model, reference, estimator,
//! configuration; or a twin experiment whose reference it generates) over
//! HTTP, runs them on blocking worker threads through
//! `JobSpec::run`, and serves their progress and outputs. The routes and
//! messages are those of `ode_observers_remote::protocol`; the desktop
//! observers viewer and the web page talk to it through
//! `ode_observers_remote::RemoteRun`.
//!
//! ```sh
//! observers-server                       # listens on 127.0.0.1:8787, open, default limits
//! ODEON_SERVER_ADDR=0.0.0.0:9000 observers-server
//! ODEON_WEB_DIR=/path/to/dist observers-server   # serves that web client at /
//! ODEON_TOKEN=secret observers-server            # every API request must carry it
//! ```
//!
//! Deployment settings, all from the environment ([`Settings::from_env`]):
//!
//! | Variable              | Default   | Meaning |
//! |-----------------------|-----------|---------|
//! | `ODEON_SERVER_ADDR`   | `127.0.0.1:8787` | listening address |
//! | `ODEON_WEB_DIR`       | the workspace's `apps/observers-client/dist` | the page served at `/`; `none` serves no page |
//! | `ODEON_TOKEN`         | unset = open | bearer token required on `/runs*` (`Authorization: Bearer …`), 401 otherwise |
//! | `ODEON_MAX_RUNS`      | 2         | runs in progress at once; further jobs get 429 |
//! | `ODEON_MAX_STEPS`     | 200000    | steps of one run |
//! | `ODEON_MAX_DOFS`      | 2000000   | grid points of a filter or window run |
//! | `ODEON_MAX_PARTICLES` | 200000    | particles of a particle run |
//! | `ODEON_MAX_OUTPUT`    | 50000000  | floats stored by one run's output (snapshots, particle positions, covariances) |
//! | `ODEON_RUN_TTL`       | 600       | seconds; a run nobody asked about for that long is cancelled and forgotten |
//!
//! A job beyond a limit is refused with 400 and a message naming the
//! limit; the limits bound the memory one request can claim (the filter
//! allocates ≈ (8 + 8·M) bytes per grid point plus its snapshots, the
//! particles store the whole population at every step).
//!
//! Cross-origin requests are allowed from anywhere (the page may be
//! published elsewhere, e.g. on GitHub Pages, and name this server with
//! `?server=`). The web page (a `trunk build` of `observers-client`) is
//! served at `/` from `ODEON_WEB_DIR` or, by default, from
//! `apps/observers-client/dist` of this workspace (an absolute path fixed
//! at compile time, so the working directory does not matter) whenever
//! that directory exists — one origin for the page and its jobs, no
//! configuration. The page never needs the token to load; only its jobs
//! carry it.
//!
//! Runs live in memory until they are deleted (the client deletes its run
//! when it drops the handle, i.e. once it has taken the output or
//! cancelled) or until nobody has polled them for `ODEON_RUN_TTL` (a closed
//! tab). A job that panics (a malformed description) is reported in its
//! status rather than killing the server.

use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use ode_models_spec::progress::Progress;
use ode_observers::jobs::{Estimator, FilterConfig, JobOutput, JobSpec};
use ode_observers_remote::protocol::{self, NewRun, RunStatus, TwinJob};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

/// Deployment settings: access and the limits protecting the machine.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Bearer token every API request must carry; `None` = open server.
    pub token: Option<String>,
    /// Runs in progress accepted at once.
    pub max_runs: usize,
    /// Steps of one run.
    pub max_steps: usize,
    /// Grid points of a filter or window run.
    pub max_dofs: usize,
    /// Particles of a particle run.
    pub max_particles: usize,
    /// Floats stored by one run's output.
    pub max_output: usize,
    /// A run nobody asked about for this long is cancelled and forgotten.
    pub ttl: Duration,
}

impl Default for Settings {
    /// Open server with the default limits (what a bare `observers-server`
    /// runs with).
    fn default() -> Self {
        Settings {
            token: None,
            max_runs: 2,
            max_steps: 200_000,
            max_dofs: 2_000_000,
            max_particles: 200_000,
            max_output: 50_000_000,
            ttl: Duration::from_secs(600),
        }
    }
}

impl Settings {
    /// The settings from the `ODEON_*` variables (see the crate doc);
    /// unset or empty = the default. Panics on an unparsable number.
    pub fn from_env() -> Self {
        fn var(name: &str) -> Option<String> {
            std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
        }
        fn num(name: &str, default: usize) -> usize {
            var(name).map_or(default, |v| v.parse().unwrap_or_else(|_| panic!("{name}={v} is not a number")))
        }
        let d = Settings::default();
        Settings {
            token: var("ODEON_TOKEN"),
            max_runs: num("ODEON_MAX_RUNS", d.max_runs),
            max_steps: num("ODEON_MAX_STEPS", d.max_steps),
            max_dofs: num("ODEON_MAX_DOFS", d.max_dofs),
            max_particles: num("ODEON_MAX_PARTICLES", d.max_particles),
            max_output: num("ODEON_MAX_OUTPUT", d.max_output),
            ttl: Duration::from_secs(num("ODEON_RUN_TTL", d.ttl.as_secs() as usize) as u64),
        }
    }
}

/// One run: its progress handle (shared with the worker), and its result
/// once the worker returned.
struct Run {
    total: usize,
    progress: Arc<Progress>,
    started: Instant,
    /// Last time a client asked about this run (status or output); the
    /// sweeper drops runs idle for longer than the TTL.
    last_seen: Mutex<Instant>,
    result: Mutex<Option<Result<Arc<JobOutput>, String>>>,
}

impl Run {
    fn touch(&self) {
        *self.last_seen.lock().unwrap() = Instant::now();
    }
    fn finished(&self) -> bool {
        self.result.lock().unwrap().is_some()
    }
}

#[derive(Clone)]
struct Runs {
    runs: Arc<Mutex<HashMap<u64, Arc<Run>>>>,
    next_id: Arc<AtomicU64>,
    settings: Arc<Settings>,
}

impl Runs {
    fn new(settings: Settings) -> Self {
        Runs { runs: Default::default(), next_id: Default::default(), settings: Arc::new(settings) }
    }

    fn get(&self, id: u64) -> Option<Arc<Run>> {
        self.runs.lock().unwrap().get(&id).cloned()
    }

    /// Runs whose worker has not returned.
    fn in_progress(&self) -> usize {
        self.runs.lock().unwrap().values().filter(|r| !r.finished()).count()
    }

    /// Cancel and forget the runs nobody asked about for the TTL.
    fn sweep(&self) {
        let ttl = self.settings.ttl;
        let mut runs = self.runs.lock().unwrap();
        runs.retain(|_, run| {
            let idle = run.last_seen.lock().unwrap().elapsed() > ttl;
            if idle {
                run.progress.cancel();
            }
            !idle
        });
    }
}

/// The server's routes (see `ode_observers_remote::protocol`), open to
/// cross-origin callers, guarded by the token and the limits of
/// `settings`; with `web_dir`, the static web client at `/`. Starts the
/// sweeper of idle runs on the current tokio runtime.
pub fn router(web_dir: Option<&str>, settings: Settings) -> Router {
    let runs = Runs::new(settings);
    let sweeper = runs.clone();
    let period = (sweeper.settings.ttl / 4).clamp(Duration::from_millis(50), Duration::from_secs(30));
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(period).await;
            sweeper.sweep();
        }
    });
    let api = Router::new()
        .route("/runs", post(create))
        .route("/twin-runs", post(create_twin))
        .route("/runs/{id}", get(status).delete(cancel))
        .route("/runs/{id}/output", get(output))
        .route_layer(middleware::from_fn_with_state(runs.clone(), require_token))
        .with_state(runs);
    let router = match web_dir {
        Some(dir) => api.fallback_service(ServeDir::new(dir)),
        None => api,
    };
    // A job carries its reference (states and observations of every
    // step): a long run is tens of megabytes of JSON, far beyond axum's
    // default 2 MB body limit. The CORS layer is outermost so that
    // preflight requests are answered before the token check; answers
    // are gzipped for clients accepting it (an output is megabytes).
    router
        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(1 << 30))
}

/// With a token configured, refuse (401) any API request not carrying it
/// as `Authorization: Bearer <token>`.
async fn require_token(State(runs): State<Runs>, request: Request, next: Next) -> Response {
    if let Some(token) = &runs.settings.token {
        let sent = request
            .headers()
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(str::trim);
        if sent != Some(token.as_str()) {
            // Read the body before answering: a client still uploading its
            // job would otherwise see a reset connection instead of the 401.
            let _ = axum::body::to_bytes(request.into_body(), 1 << 30).await;
            return (StatusCode::UNAUTHORIZED, "missing or wrong token (Authorization: Bearer …)").into_response();
        }
    }
    next.run(request).await
}

/// Nodes per direction of the grid a configuration describes for the
/// estimator: the full box of the filter (Neumann n_el·p_ord + 1,
/// periodic n_el·p_ord, Dirichlet n_el·p_ord − 1) or the Dirichlet window.
fn nodes(config: &FilterConfig, estimator: Estimator) -> Vec<usize> {
    config
        .vars
        .iter()
        .map(|v| match estimator {
            Estimator::Window => (v.win_n_el * v.p_ord).saturating_sub(1),
            _ if v.periodic => v.n_el * v.p_ord,
            _ if v.dirichlet => (v.n_el * v.p_ord).saturating_sub(1),
            _ => v.n_el * v.p_ord + 1,
        })
        .collect()
}

/// Floats the run's output will hold: the estimates, plus the 2D snapshots
/// of the grid estimators, the whole population per step of the particles,
/// the covariance per step of the tracker.
fn output_floats(config: &FilterConfig, estimator: Estimator, steps: usize, dim: usize) -> usize {
    let records = steps + 1;
    let estimates = records * dim;
    match estimator {
        Estimator::Filter | Estimator::Window => {
            let n = nodes(config, estimator);
            let plane: usize = config
                .pairs
                .iter()
                .filter(|(_, selected)| *selected)
                .map(|((a, b), _)| n.get(*a).copied().unwrap_or(0) * n.get(*b).copied().unwrap_or(0))
                .sum();
            estimates + config.n_snapshots.min(records) * plane
        }
        Estimator::Particles => records * config.particles.n_particles.saturating_mul(dim + 2),
        Estimator::Tracker => records * (dim + dim * dim),
    }
}

/// Refuse (400) a job beyond the limits, or (429) one more run than the
/// server accepts at once.
fn admit(runs: &Runs, estimator: Estimator, config: &FilterConfig, steps: usize, dim: usize) -> Result<(), (StatusCode, String)> {
    let s = &runs.settings;
    let refuse = |what: String| Err((StatusCode::BAD_REQUEST, what));
    if steps > s.max_steps {
        return refuse(format!("{steps} steps, the server accepts at most {}", s.max_steps));
    }
    if matches!(estimator, Estimator::Filter | Estimator::Window) {
        let dofs = nodes(config, estimator).iter().fold(1usize, |p, n| p.saturating_mul(*n));
        if dofs > s.max_dofs {
            return refuse(format!("{dofs} grid points, the server accepts at most {}", s.max_dofs));
        }
    }
    if estimator == Estimator::Particles && config.particles.n_particles > s.max_particles {
        return refuse(format!("{} particles, the server accepts at most {}", config.particles.n_particles, s.max_particles));
    }
    let floats = output_floats(config, estimator, steps, dim);
    if floats > s.max_output {
        return refuse(format!(
            "an output of {floats} values (snapshots, population or covariances), the server accepts at most {}",
            s.max_output
        ));
    }
    let busy = runs.in_progress();
    if busy >= s.max_runs {
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            format!("the server is busy: {busy} run(s) in progress, its limit — try again later"),
        ));
    }
    Ok(())
}

/// `POST /runs`: start the job on a blocking worker thread.
async fn create(State(runs): State<Runs>, Json(job): Json<JobSpec>) -> Result<Json<NewRun>, (StatusCode, String)> {
    let (dim, vars, rdim) = (job.model.dim(), job.config.vars.len(), job.reference.dim());
    if vars != dim {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("the configuration has {vars} variables, the {} model {dim}", job.model.kind()),
        ));
    }
    if rdim != dim {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("the reference has {rdim} components, the {} model {dim}", job.model.kind()),
        ));
    }
    admit(&runs, job.estimator, &job.config, job.steps(), dim)?;
    Ok(Json(NewRun { id: start(&runs, job) }))
}

/// `POST /twin-runs`: generate the reference of a twin experiment here,
/// then start the job.
async fn create_twin(State(runs): State<Runs>, Json(twin): Json<TwinJob>) -> Result<Json<NewRun>, (StatusCode, String)> {
    let (dim, x0) = (twin.twin.model.dim(), twin.twin.x0.len());
    if x0 != dim {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("x0 has {x0} entries, the {} model {dim}", twin.twin.model.kind()),
        ));
    }
    if twin.config.vars.len() != dim {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("the configuration has {} variables, the {} model {dim}", twin.config.vars.len(), twin.twin.model.kind()),
        ));
    }
    admit(&runs, twin.estimator, &twin.config, twin.twin.steps, dim)?;
    let job = tokio::task::spawn_blocking(move || {
        JobSpec::from_twin(&twin.twin, twin.estimator, twin.config, &Progress::default())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("the reference generation panicked: {e}")))?;
    Ok(Json(NewRun { id: start(&runs, job) }))
}

/// Register a run and execute `job` on a blocking worker thread.
fn start(runs: &Runs, job: JobSpec) -> u64 {
    let id = runs.next_id.fetch_add(1, Ordering::Relaxed) + 1;
    let run = Arc::new(Run {
        total: job.steps().max(1),
        progress: Arc::new(Progress::default()),
        started: Instant::now(),
        last_seen: Mutex::new(Instant::now()),
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
    id
}

/// `GET /runs/{id}`: progress of the run.
async fn status(State(runs): State<Runs>, Path(id): Path<u64>) -> Result<Json<RunStatus>, StatusCode> {
    let run = runs.get(id).ok_or(StatusCode::NOT_FOUND)?;
    run.touch();
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

/// `GET /runs/{id}/output`: the output of a finished run — the binary
/// encoding (`postcard`, [`protocol::BINARY`]) when the request accepts
/// it, JSON otherwise (curl, scripts).
async fn output(State(runs): State<Runs>, Path(id): Path<u64>, headers: HeaderMap) -> Result<Response, (StatusCode, String)> {
    let run = runs.get(id).ok_or((StatusCode::NOT_FOUND, format!("no run {id}")))?;
    run.touch();
    let out = {
        let result = run.result.lock().unwrap();
        match &*result {
            None => return Err((StatusCode::CONFLICT, format!("run {id} is not finished"))),
            Some(Err(e)) => return Err((StatusCode::INTERNAL_SERVER_ERROR, e.clone())),
            Some(Ok(out)) => out.clone(),
        }
    };
    let binary = headers.get(ACCEPT).and_then(|v| v.to_str().ok()).is_some_and(|a| a.contains(protocol::BINARY));
    if !binary {
        return Ok(Json(out).into_response());
    }
    // Encoding tens of megabytes: off the request threads.
    let bytes = tokio::task::spawn_blocking(move || protocol::encode_output(&out))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("encoding panicked: {e}")))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("cannot encode the output: {e}")))?;
    Ok(([(CONTENT_TYPE, protocol::BINARY)], bytes).into_response())
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
/// the working directory; `none` = serve no page), else the workspace's
/// `apps/observers-client/dist`; `None`, with a warning, when it does not
/// exist.
fn web_dir() -> Option<String> {
    let default = concat!(env!("CARGO_MANIFEST_DIR"), "/../observers-client/dist");
    let (dir, from_env) = match std::env::var("ODEON_WEB_DIR").ok().map(|d| d.trim().to_string()).filter(|d| !d.is_empty()) {
        Some(d) if d.eq_ignore_ascii_case("none") => return None,
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
    let settings = Settings::from_env();
    println!("observers-server listening on http://{}", listener.local_addr().unwrap());
    match &web_dir {
        Some(dir) => println!("serving the web client from {dir} at /"),
        None => println!("no web client served (ODEON_WEB_DIR=<dist> to serve one)"),
    }
    println!(
        "access: {}; limits: {} run(s) at once, {} steps, {} grid points, {} particles, {} output values; idle runs dropped after {} s",
        if settings.token.is_some() { "token required (ODEON_TOKEN)" } else { "OPEN — set ODEON_TOKEN before exposing this server" },
        settings.max_runs,
        settings.max_steps,
        settings.max_dofs,
        settings.max_particles,
        settings.max_output,
        settings.ttl.as_secs()
    );
    axum::serve(listener, router(web_dir.as_deref(), settings)).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ode_models_spec::noise::NoiseModel;
    use ode_models_spec::spec::{ModelSpec, TwinSpec};
    use ode_observers::jobs::{Estimator, FilterConfig, FilterOutput, TrackerOutput};
    use ode_observers_remote::RemoteRun;
    use std::time::Duration;

    /// The server on an ephemeral port of the test's runtime, open with the
    /// default limits; returns its base URL.
    async fn serve() -> String {
        serve_with(Settings::default()).await
    }

    async fn serve_with(settings: Settings) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router(None, settings)).await.unwrap() });
        format!("http://{addr}")
    }

    /// One request from a blocking thread.
    async fn fetch(request: ehttp::Request) -> ehttp::Response {
        tokio::task::spawn_blocking(move || ehttp::fetch_blocking(&request).unwrap()).await.unwrap()
    }

    fn spring_twin(steps: usize) -> TwinSpec {
        TwinSpec {
            model: ModelSpec::Spring { n: 1, rho: 1.0, a: 1.0, obs: 0 },
            x0: vec![0.15, 0.0],
            dt: 0.02,
            steps,
            noise: NoiseModel::Gaussian { std: 0.01 },
            seed: 1000,
            walk: None,
        }
    }

    fn spring_config() -> FilterConfig {
        let mut config = FilterConfig::defaults(
            vec!["y_1".into(), "v_1".into()],
            vec![None, None],
            &[(0, 1)],
            &vec![vec![0.15, 0.0]; 11],
        );
        for v in &mut config.vars {
            v.sigma = 5.0;
        }
        config.n_snapshots = 3;
        config
    }

    fn spring_job(estimator: Estimator, steps: usize) -> JobSpec {
        JobSpec::from_twin(&spring_twin(steps), estimator, spring_config(), &Progress::default())
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

    /// A twin job posted to `/twin-runs` through the client handle — the
    /// server generating the reference — gives the same run as the job
    /// run locally along its own reference.
    #[tokio::test]
    async fn twin_runs_generate_the_reference_on_the_server() {
        let base = serve().await;
        let twin = TwinJob { twin: spring_twin(20), estimator: Estimator::Tracker, config: spring_config() };
        let remote = take(RemoteRun::<TrackerOutput>::spawn_twin(&base, &twin)).await.unwrap();
        let JobOutput::Tracker(local) = spring_job(Estimator::Tracker, 20).run(&Progress::default()) else { unreachable!() };
        assert_eq!(remote.estimates, local.estimates);
        assert_eq!(remote.reference, local.reference);
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
        // A long job that does not saturate the rayon pool (the tracker is
        // sequential), so the other tests of this binary keep running
        // alongside without starving the request threads — and whose
        // reference (generated here, 200k steps) stays small.
        let job = spring_job(Estimator::Tracker, 200_000);
        let run = RemoteRun::<TrackerOutput>::spawn(&base, &job);
        // Wait for the id, then drop.
        let b = base.clone();
        let id = tokio::task::spawn_blocking(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                run.try_take();
                if let Some(e) = run.error() {
                    panic!("the run failed: {e}");
                }
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

    /// With a token configured, jobs without it (or with a wrong one) are
    /// refused with 401, and the client handle carrying it runs as usual.
    #[tokio::test]
    async fn the_token_guards_the_api() {
        let base = serve_with(Settings { token: Some("s3cret".into()), ..Settings::default() }).await;
        let twin = TwinJob { twin: spring_twin(5), estimator: Estimator::Tracker, config: spring_config() };
        let err = take(RemoteRun::<TrackerOutput>::spawn_twin(&base, &twin)).await.err().expect("refused");
        assert!(err.contains("401") && err.contains("token"), "{err}");
        let err = take(RemoteRun::<TrackerOutput>::spawn_twin_with_token(&base, Some("wrong"), &twin))
            .await
            .err()
            .expect("refused");
        assert!(err.contains("401"), "{err}");
        let out = take(RemoteRun::<TrackerOutput>::spawn_twin_with_token(&base, Some("s3cret"), &twin)).await.unwrap();
        assert_eq!(out.estimates.len(), 6);
        // The page's preflight must pass without the token.
        let mut preflight = ehttp::Request::get(format!("{base}/twin-runs"));
        preflight.method = "OPTIONS".into();
        preflight.headers = ehttp::Headers::new(&[
            ("Origin", "https://example.github.io"),
            ("Access-Control-Request-Method", "POST"),
            ("Access-Control-Request-Headers", "authorization,content-type"),
        ]);
        let r = fetch(preflight).await;
        assert!(r.ok, "preflight answered {}", r.status);
        assert_eq!(r.headers.get("access-control-allow-origin"), Some("*"));
    }

    /// Jobs beyond the limits are refused with 400 and a message naming
    /// the limit: steps, grid points, particles, output size.
    #[tokio::test]
    async fn oversized_jobs_are_refused() {
        let settings = Settings { max_steps: 10, max_dofs: 100, max_particles: 50, max_output: 1_000, ..Settings::default() };
        let base = serve_with(settings).await;
        let post = |estimator, steps, config| {
            let (base, twin) = (base.clone(), TwinJob { twin: spring_twin(steps), estimator, config });
            async move { take(RemoteRun::<TrackerOutput>::spawn_twin(&base, &twin)).await.err().expect("refused") }
        };
        let err = post(Estimator::Tracker, 11, spring_config()).await;
        assert!(err.contains("400") && err.contains("11 steps"), "{err}");
        let mut config = spring_config();
        config.vars.iter_mut().for_each(|v| (v.n_el, v.p_ord) = (5, 4));
        let err = post(Estimator::Filter, 5, config).await;
        assert!(err.contains("441 grid points"), "{err}");
        let mut config = spring_config();
        config.particles.n_particles = 60;
        let err = post(Estimator::Particles, 5, config).await;
        assert!(err.contains("60 particles"), "{err}");
        let mut config = spring_config();
        config.particles.n_particles = 50;
        let err = post(Estimator::Particles, 10, config).await;
        assert!(err.contains("output of 2200 values"), "{err}");
        // Within every limit, the job runs.
        let twin = TwinJob { twin: spring_twin(5), estimator: Estimator::Tracker, config: spring_config() };
        take(RemoteRun::<TrackerOutput>::spawn_twin(&base, &twin)).await.unwrap();
    }

    /// The (max_runs + 1)-th run in progress is refused with 429; once a
    /// run finished, the next one is accepted.
    #[tokio::test]
    async fn a_busy_server_answers_429() {
        let base = serve_with(Settings { max_runs: 1, ..Settings::default() }).await;
        let long = spring_job(Estimator::Tracker, 200_000);
        let first = RemoteRun::<TrackerOutput>::spawn(&base, &long);
        let b = base.clone();
        tokio::task::spawn_blocking(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            while ehttp::fetch_blocking(&ehttp::Request::get(format!("{b}/runs/1"))).unwrap().status != 200 {
                assert!(Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(20));
            }
        })
        .await
        .unwrap();
        let short = spring_job(Estimator::Tracker, 5);
        let err = take(RemoteRun::<TrackerOutput>::spawn(&base, &short)).await.err().expect("refused");
        assert!(err.contains("429") && err.contains("busy"), "{err}");
        drop(first);
        let b = base.clone();
        tokio::task::spawn_blocking(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            while ehttp::fetch_blocking(&ehttp::Request::get(format!("{b}/runs/1"))).unwrap().status != 404 {
                assert!(Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(20));
            }
        })
        .await
        .unwrap();
        take(RemoteRun::<TrackerOutput>::spawn(&base, &short)).await.unwrap();
    }

    /// A run nobody polls is cancelled and forgotten after the TTL.
    #[tokio::test]
    async fn idle_runs_expire() {
        let base = serve_with(Settings { ttl: Duration::from_millis(200), ..Settings::default() }).await;
        let job = spring_job(Estimator::Tracker, 200_000);
        let mut request = ehttp::Request::post(format!("{base}/runs"), serde_json::to_vec(&job).unwrap());
        request.headers = ehttp::Headers::new(&[("Content-Type", "application/json")]);
        let r = fetch(request).await;
        assert!(r.ok, "{}", r.status);
        let id = serde_json::from_slice::<NewRun>(&r.bytes).unwrap().id;
        let r = fetch(ehttp::Request::get(format!("{base}/runs/{id}"))).await;
        assert_eq!(r.status, 200);
        tokio::time::sleep(Duration::from_millis(600)).await;
        let r = fetch(ehttp::Request::get(format!("{base}/runs/{id}"))).await;
        assert_eq!(r.status, 404, "the idle run was not dropped");
    }

    /// The output route serves JSON by default (curl, scripts) and the
    /// binary encoding on request; both decode to the same output, and the
    /// client handle gets the binary one.
    #[tokio::test]
    async fn output_formats_agree() {
        let base = serve().await;
        let job = spring_job(Estimator::Filter, 10);
        let run = RemoteRun::<FilterOutput>::spawn(&base, &job);
        let out = take(run).await.unwrap();
        // The handle deleted the run on drop; post it again and fetch the
        // output by hand in both formats.
        let mut request = ehttp::Request::post(format!("{base}/runs"), serde_json::to_vec(&job).unwrap());
        request.headers = ehttp::Headers::new(&[("Content-Type", "application/json")]);
        let id = serde_json::from_slice::<NewRun>(&fetch(request).await.bytes).unwrap().id;
        let deadline = Instant::now() + Duration::from_secs(30);
        let json = loop {
            let r = fetch(ehttp::Request::get(format!("{base}/runs/{id}/output"))).await;
            if r.status == 200 {
                break r;
            }
            assert!(Instant::now() < deadline && r.status == 409, "{}", r.status);
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        assert!(json.headers.get("content-type").unwrap().starts_with("application/json"));
        let mut request = ehttp::Request::get(format!("{base}/runs/{id}/output"));
        request.headers = ehttp::Headers::new(&[("Accept", protocol::BINARY)]);
        let binary = fetch(request).await;
        assert_eq!(binary.headers.get("content-type"), Some(protocol::BINARY));
        assert!(binary.bytes.len() * 2 < json.bytes.len(), "binary {} vs json {}", binary.bytes.len(), json.bytes.len());
        let from_json: JobOutput = serde_json::from_slice(&json.bytes).unwrap();
        let from_binary = protocol::decode_output(&binary.bytes).unwrap();
        let (JobOutput::Filter(a), JobOutput::Filter(b)) = (from_json, from_binary) else { panic!("not filter outputs") };
        assert_eq!(a.estimates, b.estimates);
        assert_eq!(a.marginals, b.marginals);
        assert_eq!(a.estimates, out.estimates);
        assert_eq!(a.marginals, out.marginals);
    }
}
