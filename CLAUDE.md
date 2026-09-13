# CLAUDE.md — Odeon

Odeon is the successor of the `mortensen` research repository: the same
code (Mortensen filter on Gauss–Lobatto grids, its Gaussian closure, the
translating window, the particle approximation, the forward models, the
egui viewer), reorganised as a Cargo workspace of four library crates plus
applications, so that desktop viewers, web front-ends and a compute server
can be built on top without any of them owning the science. The reference
PDFs are in `bib/` (git-ignored). The detailed design notes inherited from
the old repository follow the plan below and remain the authoritative
description of the algorithms; their file paths are mapped at their top.

## Target layout

```
Odeon/
  Cargo.toml                  workspace (resolver 3), shared version/edition, release profile with symbols
  crates/ode-models/          forward models: Model trait, models/*, gl4 stepper, noise      [DONE]
  crates/ode-observers/       observers: filter, tracker, box_tracker, particles, output,
                              jobs (dimension-erased runners + plain outputs), progress     [DONE]
  crates/ode-models-egui/     VizModel trait, scenes, parameter forms, palette, playback,
                              and app::App — the whole models-only application            [DONE]
  crates/ode-observers-egui/  the four views (filter, box, particles, tracker), panels,
                              and app::App<C: Compute> — the whole observers application  [DONE]
  crates/ode-observers-remote/ wire protocol of the job server + RemoteRun client (ehttp)   [DONE]
  apps/models-viewer/         the models-only viewer: scenes, no estimator
                              (package odeon-models-viewer, bin models-viewer)              [DONE]
  apps/observers-viewer/      the observers viewer, computing locally
                              (package odeon-observers-viewer, bin observers-viewer)        [DONE]
  apps/observers-client/      the same viewer as a client of observers-server: a desktop
                              binary AND the web page (trunk → dist/)
                              (package odeon-observers-client, bin observers-client)        [DONE]
  apps/observers-server/      HTTP job server on ode-observers::jobs (axum; package
                              odeon-observers-server, bin observers-server); serves the
                              built web page at / (apps/observers-client/dist by default)  [DONE]
  docs/                       LaTeX design notes (hybrid_closure.tex, density_filter.tex, figures/)
  scripts/                    Python plotting scripts for the CLI examples
```

Dependency graph (a diamond): `ode-models` at the bottom; `ode-observers`
depends on it through the `Model` trait only; `ode-models-egui` depends on
`ode-models` (+ egui); `ode-observers-egui` depends on `ode-observers` (+
egui) and, for the palette of the tracker view, on `ode-models-egui`;
`ode-observers-remote` depends on `ode-observers` (+ ehttp) only; the apps
depend on what they show. The five library crates are the ones that may be
published to crates.io; the apps never are.

## Migration plan and status

1. **Copy** the working tree of `~/Codes/Mortensen/mortensen` (with its
   uncommitted state) and `~/Codes/Mortensen/bib` into `~/Documents/Odeon`.
   Done 2026-09-13. Not a git repository yet (no `git init` was run —
   decide whether to start fresh or import the old history).
2. **Extract `ode-models`**: `model.rs`, `models/`, `noise.rs` (the PRNG
   `SplitMix64` made `pub` so that `ode-observers::particles` can use it).
   The model tests that exercised an observer (tracker identification of θ
   on kepler_mu and spring_mass, the 3D/5D grid smoke runs, the random
   walk's ring) moved to `crates/ode-observers/tests/models_with_observers.rs`
   (`kepler_mu::augmented_state` made `pub` for them). Done.
3. **Extract `ode-observers`**: `filter.rs`, `tracker.rs`, `box_tracker.rs`,
   `particles.rs`, `output.rs`, the examples and the benches; plus the
   **job layer** that used to live in the viewer (`viz/src/filtering.rs` →
   `jobs.rs`: `FilterConfig`/`VarConfig`/`ParticleConfig`, the four
   `run_*` functions, the four `*Output` types) and `Progress` (→
   `progress.rs`), both UI-free so that a server can reuse them. The
   viewer keeps `crate::filtering::…` and `crate::playback::Progress`
   paths through re-exports. Done — workspace builds without warnings,
   87 tests pass (10 viz, 31 models, 30 observers, 6 integration, 10
   doctests).
4. **Break the model↔observer coupling in the viewer** (the visitor).
   Done 2026-09-13. `ode_models::spec` holds `ModelSpec`, a plain-data
   enum with one variant per model (kind, physical parameters, initial
   state, observation choice, `NoiseModel` + seed; the seeds are part of
   the spec so a description rebuilds the very same reference and
   observation sequence — test `spec_reproduces_the_hand_built_model`),
   with `dim()`, `kind()` and `visit(dt, visitor)`, which builds the
   concrete system and hands it to a `ModelVisitor` (`fn visit<const M,
   Mod: Model<M> + Send + Sync + 'static>(self, model) -> Output`) at
   the right compile-time dimension (springs dispatch n = 1..3 → M = 2,
   4, 6; unknown-mass chains n = 1..3 → M = 3, 5, 7). `ode_observers::jobs`
   adds `Estimator { Filter, Window, Particles, Tracker }`, `JobSpec {
   model, estimator, config, dt, steps }` with `run(&Progress) ->
   JobOutput` (one variant per estimator), and the typed
   `run_filter_spec` / `run_tracker_spec` / `run_box_spec` /
   `run_particles_spec`, each a visitor calling the matching generic
   runner (dimension of the config checked against `ModelSpec::dim`
   first). The viewer's `VizModel` lost the four `make_*_job` methods
   and gained `model_spec()` (the drawn parameters as a `ModelSpec`);
   `app.rs` builds a `JobSpec` and spawns it through `spawn_job` with the
   runner of the selected slot type, and uses `jobs::Estimator` instead
   of its own enum. No model file names an observer any more, and the
   spring's opaque observation closure is built inside the spec from the
   component index. Numerics untouched: test
   `spec_run_equals_the_hand_built_run` pins the spec path to the
   hand-built run bit for bit. 92 tests (10 viz, 33 models, 33
   observers, 6 integration, 10 doctests), no warnings. Not done, by
   choice: the trajectory job `VizModel::make_job` (the forward run with
   its plotted observation series) still builds the model in each viewer
   file — the web front-end will need it through the spec too (a
   forward-run visitor in `jobs`), a natural piece of step 6.
5. **Split `viz/`**. Done 2026-09-13. `crates/ode-models-egui` (lib
   `ode_models_egui`: `models/` with `VizModel` and the eight entries,
   `noise`, `palette`, `playback` — depends on `ode-models` and `egui`
   only), `crates/ode-observers-egui` (lib `ode_observers_egui`:
   `filter_view`, `box_view`, `particles_view`, `tracker_view`, and
   `panels` — the estimator selector, the Initial data / Observation /
   Diffusion / Grid / Window / LParticles / Outputs panels editing a
   `FilterConfig`, `prior_ok`, `run_label`, `running_text`, the progress
   row with Cancel — extracted from the old `app.rs`; depends on
   `ode-observers`, `egui`, `egui_plot` and, for the series colours of the
   tracker view, `ode-models-egui`), and `apps/viewer` (renamed
   `apps/observers-viewer` in 5c; `publish = false`: `main.rs`, `app.rs`
   with the slots, the runs, the tabs, the scene and observation panels).
   `viz/` is gone. Two contract changes were needed to keep
   `ode-models-egui` free of `ode-observers`: `Progress` moved down to
   `ode_models::progress` (`ode_observers::progress` re-exports it, all
   paths unchanged), and `VizModel::tune_filter_defaults(&mut
   FilterConfig)` became `estimator_hints() -> EstimatorHints` — a
   `VarHint { domain, n_el, p_ord, q, x0 }` (all `Option`) per state
   component plus an optional ε — applied onto the computed defaults by
   the app's `apply_hints`; Kepler-μ, the unknown-mass chains and the
   random walk return the same overrides as before. The three
   "tuned defaults + filter run" tests split accordingly: the hints are
   tested in `ode-models-egui`, the runs through the whole pipeline
   (`model_spec` → `JobSpec` → `run_filter_spec`) in `apps/viewer`. The
   library crates depend on `egui` directly (not `eframe`); the app is
   the only `eframe` user. 94 tests (33 models, 3 models-egui, 33
   observers, 6 integration, 7 observers-egui, 2 viewer, 10 doctests),
   no warnings; the viewer launches.
5b. **Two apps** (2026-09-13, added on request). The model side of the old
   `app.rs` moved into `ode-models-egui`: `slot::ModelSlot` (model, dt,
   T, trajectory, playback clock, run in flight; `refresh_preview`,
   `start_run`, `poll`, `completed`, `advance`, `busy`) and `panels`
   (`model_selector`, `params_ui` — parameter form + Observation +
   Simulation sections, refreshing the preview on any edit —, `run_ui`,
   `observation_panel` — playback controls + y(t) plot —, `scene_ui` —
   "About this model" + the scene). `apps/models-viewer` (package
   `odeon-models-viewer`, bin `models-viewer`, window "Odeon — models")
   is those panels and nothing else, ≈ 70 lines; `apps/viewer` is the
   same plus the estimator state (`Slot { base: ModelSlot, … }`), the
   estimator panels and the result tabs. **Unknown-mass springs are no
   longer separate entries**: the "Single/Two/Three springs" entries got
   an estimator-only option "unknown free-end mass" (+ θ_true) through
   the new `VizModel::estimator_options_ui` hook, drawn by the estimator
   viewer in its estimator section and never by the models-only viewer.
   With the option on the *whole run* (scene, observation plot, and
   estimators) uses `SpringMassSystem` with m = m₀ = ρℓ, k = a/ℓ, the
   same state (y, v) plus θ, the same equations (only the integrator
   differs: GL4 vs midpoint — test `augmented_run_is_the_same_chain`,
   agreement to 1e-6 at θ_true = 0), positions only as observations, and
   the θ hints of the old entry; the estimators' twin reference is thus
   exactly the trajectory on screen, and toggling the option discards
   the run and the configuration (its dimension changes; the app also
   recomputes the configuration whenever a completed run's dimension
   differs from it). The two library models were deliberately **not**
   merged (different parametrisations and integrators; too risky). The
   menu then had 9 entries, "Kepler orbit (unknown μ)" still separate
   (made an option in 5c). Library crates depend
   on `egui`/`egui_plot`, the apps on `eframe`. 96 tests, no warnings;
   both binaries launch. **Spring convention fixed in the viewer** (same
   day): `SpringSystem`'s state y holds *displacements* from rest (its
   force is −K y with zero-length springs, equilibrium y = 0), but the
   spring entry fed it absolute positions with rest at iℓ and drew y
   directly — the default single spring swung ±1.15 through the wall.
   The entry now edits displacements (default: last mass at 0.15), draws
   mass i at iℓ + yᵢ (`draw_chain` offsets, as the unknown-mass option
   already did) and says so in its form and description; the library's
   doc comment and the four spring examples' comments, which called
   (ℓ, 2ℓ, …) the rest configuration, are corrected — no code or run
   changed on the library side (the examples still start from Y = iℓ,
   now described as a stretched chain).
5c. **Kepler-μ as an option, viewer renamed** (same day, on request).
   "Kepler orbit (unknown μ)" is no longer an entry: the Kepler entry got
   the estimator-only option "unknown μ" (+ θ_true, default 0.4) through
   `estimator_options_ui`, exactly like the springs — with it on, the
   whole run uses `KeplerMuSystem` with μ₀ = the form's μ, the same
   potential and the same GL4 stepper (the θ stage decouples), so at
   θ_true = 0 the runs coincide to 1e-10 (test
   `augmented_run_is_the_same_orbit`); the θ hints, the (q₁, θ) plane
   first, θ not drawn. `kepler_mu.rs` of the viewer is gone; the menu
   has 8 entries (springs 0–2, pendulum 3, rod 4, Kepler 5, Lorenz 6,
   random walk 7). `apps/viewer` is renamed `apps/observers-viewer`
   (package `odeon-observers-viewer`, bin `observers-viewer`, window
   "Odeon — observers"). 97 tests, no warnings.
6. **Serde, server, web.**
   6a. **Serde derives** — done 2026-09-13. `serde` (with `derive`) is a
   plain dependency of `ode-models` and `ode-observers` (not a feature:
   the job layer exists to travel, and the derive cost is small).
   `Serialize`/`Deserialize` on `ModelSpec` (internally tagged:
   `{"kind": "spring", …}` — the tag values are exactly
   `ModelSpec::kind()`), on the parameter structs (`KeplerParams`,
   `KeplerMuParams`, `LorenzParams`, `PendulumParams`,
   `PendulumRodParams`, `SpringMassParams`), on the enums
   `NoiseModel`, `KeplerObservation`, `LorenzObservation`,
   `DiffusionScheme`, `Estimator` (all `rename_all = "snake_case"`:
   `"gaussian"`, `"range"`, `"split_euler"`, `"tracker"`…), on
   `VarConfig`/`FilterConfig`/`ParticleConfig`/`JobSpec`, on the four
   outputs and on `JobOutput` (adjacently tagged: `{"estimator":
   "tracker", "output": {…}}`). Tests: `every_spec_round_trips_through_json`
   (ode-models) and `job_spec_and_output_round_trip_through_json`
   (ode-observers, a tracker run serialized and read back field by
   field). **Finding:** `serde_json` parses floats one ULP off by
   default; bit-exact round trips need its `float_roundtrip` feature
   (enabled in the dev-dependencies; the server and the web client
   must enable it too, or use a binary format such as `postcard` /
   `bincode`). 99 tests, no warnings.
   6b. **Server and remote runs** — done 2026-09-13.
   `crates/ode-observers-remote` (UI-free): `protocol` — `NewRun { id }`,
   `RunStatus { id, done, total, finished, elapsed, error }` and the
   routes `POST /runs` (a `JobSpec` → `NewRun`; 400 when the
   configuration's dimension is not the model's), `GET /runs/{id}`
   (`RunStatus`, 404 unknown), `GET /runs/{id}/output` (the `JobOutput`;
   409 not finished, 500 failed), `DELETE /runs/{id}` (cancel at the next
   iteration and forget, 204); and `RemoteRun<T>` — the client handle
   with the local `RunHandle`'s surface (`spawn(base, &JobSpec)`,
   `progress`, `elapsed`, `try_take`) plus `error()`; `T` is any output
   with `TryFrom<JobOutput, Error = String>` (the four outputs implement
   it — asking for the wrong kind is an error naming both). Built on
   `ehttp` 0.5 (callbacks; a thread per request on native, `fetch` on
   wasm), polling at most every 150 ms from `try_take`, chaining the
   output request when the status says finished, DELETE on drop.
   `apps/observers-server`: axum 0.8 + tokio; runs in a `HashMap` of
   `Arc<Run { progress, started, result }>`, each job on
   `spawn_blocking` through `JobSpec::run`, a panicking job recorded as
   the run's error; `ODEON_SERVER_ADDR` (default `127.0.0.1:8787`);
   `serde` needs its `rc` feature there (`Json<Arc<JobOutput>>`), and
   `serde_json` `float_roundtrip` on both sides. Tests (in the server,
   `#[tokio::test]` on an ephemeral port, driven by `RemoteRun`):
   `remote_filter_run_equals_the_local_run` (bit for bit),
   `wrong_output_kind_is_an_error`, `bad_jobs_are_refused` (400 with the
   message, 404), `dropping_the_handle_cancels_the_run`. The observers
   viewer has an "on server" checkbox + URL next to the estimator header
   (`Compute`; `ODEON_SERVER=http://host:port` pre-selects it), every run
   slot is an `EstimatorRun<T> { Local(RunHandle<T>), Remote(RemoteRun<T>) }`
   collected by one generic `collect` (a failure — server down, refused
   job, wrong kind — is shown in red under the progress row until the
   next run). Trajectories (`make_job`) stay local. A sample job for
   curl is `apps/observers-server/jobs/spring_tracker.json` (a one-spring
   tracker run, 200 steps). **Then split into two apps** (same day, on
   request): the observers application moved into
   `ode_observers_egui::app` as `App<C: Compute>` — `trait Compute {
   fn spawn<T>(&self, job: JobSpec, local: LocalRunner<T>) -> Box<dyn
   EstimatorRun<T>>; fn ui(&mut self, ui) {} }` with `trait
   EstimatorRun<T> { progress, elapsed, try_take, error }` (implemented
   for the local `RunHandle<T>`), `LocalCompute` (worker thread), and
   `App::ui(&mut self, &mut egui::Ui)` (no eframe in the library
   crates); the "on server" checkbox is gone. `apps/observers-viewer`
   (bin `observers-viewer`) is `App::new(LocalCompute)` + eframe
   boilerplate and links no HTTP code; `apps/observers-client` (bin
   `observers-client`, window "Odeon — observers (server)") supplies
   `RemoteCompute { url }` (from `ODEON_SERVER`, editable in the
   estimator section) wrapping `RemoteRun` in a newtype `Remote<T>` for
   the trait. `ode-observers-egui` now depends on `ode-models` directly
   (for `ModelSpec` in the runner type). The app's tests (hinted
   defaults + filter runs, the ring) live in `ode_observers_egui::app`.
   103 tests, no warnings; both binaries launch. The models-only
   application got the same treatment: `ode_models_egui::app::App`
   (`new()`, `ui(&mut egui::Ui)`), `apps/models-viewer` being the eframe
   wrapper. Note for 6c: `RemoteRun` uses `std::time::Instant`, which
   panics on wasm — switch to `web_time::Instant` there.
   6c. **The web page** — done 2026-09-13. No separate `apps/web`: the
   `observers-client` package builds for `wasm32-unknown-unknown` too
   (`apps/observers-client/index.html` + `Trunk.toml`; `trunk build
   --release` → `dist/`, git-ignored; the wasm entry point in `main.rs`
   starts `eframe::WebRunner` on `<canvas id="odeon">`, the server URL
   defaults to the page's origin via `web_sys`). Made wasm-ready:
   `web_time::Instant` in `ode_models_egui::playback` and
   `ode-observers-remote`; `RunHandle::spawn` runs the job synchronously
   on wasm (no threads — trajectories are cheap; estimator jobs are on
   the server anyway); every library crate, `ode-observers` and the
   lobatto solvers included, compiles for wasm untouched. The server
   gained `tower-http`: `CorsLayer::permissive()` (a `trunk serve` page
   on another port may call it) and a `ServeDir` fallback serving the
   page at `/` (same origin): from `ODEON_WEB_DIR` (relative to the
   working directory) or, by default, from the workspace's
   `apps/observers-client/dist` — an absolute path fixed at compile time
   (`CARGO_MANIFEST_DIR`), so `cargo run -p odeon-observers-server`
   serves the page from any directory once it is built; a missing
   directory is a startup warning naming the resolved path (the first
   attempt failed exactly so: the server had been started from
   `apps/observers-client` with a relative `ODEON_WEB_DIR`). Checked: `/` → 200 text/html, the `.wasm` → 200
   application/wasm, `/runs/1` → 404, CORS preflight → 200 with
   `access-control-allow-origin: *`. Trajectories (`VizModel::make_job`)
   are still computed in the page; a forward-run visitor through the
   spec, computing the reference on the server too, remains open.
   **Publishing the page elsewhere** (GitHub Pages, …; same day): the
   web client takes the server URL from `?server=https://…` in the
   page's address (remembered in the browser's local storage under
   `odeon.server`), else the remembered value, else the page's origin;
   editing the field re-remembers it (`web` module of the client's
   `main.rs`, `web-sys` `Storage` + `UrlSearchParams`). Build with
   `trunk build --release --public-url /<repo>/` for a project page; an
   HTTPS page can only call an HTTPS server (a tunnel with TLS in front
   of the Mac), and the server has no authentication yet — add a token
   and run limits before pointing a public page at it. The
   `.wasm` is ≈ 8 MB because the workspace release profile keeps debug
   symbols and `wasm-opt` is not installed (trunk skips it): install
   `binaryen` for a smaller page. 103 native tests, no warnings.

Conventions carried over: all linear solves through nalgebra; English,
Unicode-math doc comments citing the papers by name; tests pin every
numerical claim; no `Default` on parameter structs of the solvers.

## Commands (workspace)

```sh
cargo test --release --workspace                          # everything
cargo run -p ode-observers --release --example one_spring # CLI examples (write examples/output/<name>/ under the cwd)
cargo bench -p ode-observers --bench transport            # transport paths (fixed box and window)
cargo bench -p ode-observers --bench diffusion            # lobatto-spectral vs lobatto-fft
cargo run -p odeon-models-viewer --release                # the models-only viewer (apps/models-viewer)
cargo run -p odeon-observers-viewer --release             # the observers viewer (apps/observers-viewer)
cargo run -p odeon-observers-server --release             # the job server on 127.0.0.1:8787 (ODEON_SERVER_ADDR)
cargo run -p odeon-observers-client --release             # the same viewer sending its jobs to the server (ODEON_SERVER, default http://127.0.0.1:8787)
(cd apps/observers-client && trunk build)                # the web page → apps/observers-client/dist (release by Trunk.toml)
cargo run -p odeon-observers-server --release             # …then serves it at http://127.0.0.1:8787/ (or ODEON_WEB_DIR=<dist>)
python scripts/visualize_filter.py one_spring             # figures from a CLI run
```

# Inherited design notes (from the `mortensen` repository, 2026-09-13)

**Paths in these notes are the old ones**: read `src/model.rs`, `src/models/`,
`src/noise.rs` as `crates/ode-models/src/...`; `src/filter.rs`, `tracker.rs`,
`box_tracker.rs`, `particles.rs`, `output.rs` as `crates/ode-observers/src/...`;
`viz/src/filtering.rs` as `crates/ode-observers/src/jobs.rs` (with `Progress`
in `crates/ode-models/src/progress.rs`); `examples/` and `benches/` as
`crates/ode-observers/...`; the old crate name `mortensen::` as
`ode_models::` / `ode_observers::`. The viewer `viz/` is split:
`viz/src/models/`, `noise.rs`, `palette.rs`, `playback.rs` are
`crates/ode-models-egui/src/...`; `viz/src/filter_view.rs`, `box_view.rs`,
`particles_view.rs`, `tracker_view.rs` are `crates/ode-observers-egui/src/...`
(the estimator panels of the old `app.rs` are `panels.rs` there);
`viz/src/app.rs`, `main.rs` are `apps/observers-viewer/src/...`; the old
crate name `mortensen-viz` is `odeon-observers-viewer`. Everything else below is current.

## What this project is

A Rust research library for the **Mortensen filter** (deterministic optimal
filtering): the value function V solves a Hamilton–Jacobi–Bellman equation, and
the code evolves p = exp(−V/ε) instead, discretized on a tensor-product
Gauss–Lobatto grid; the diffusion is solved by dense fast diagonalisation
of the per-direction spectral-element operators (`lobatto-spectral` crate;
the FFT-across-elements solver `lobatto-fft` is a dev-dependency kept for
the comparison bench only). One
filter iteration is the 4-step cycle in `src/filter.rs`:

1. **Observation**: p ← p · exp(−ε⁻¹ · dt · γ · d(y_n, h(ξ))²/2), d the
   distance in observation space (Euclidean; angular for a bearing —
   `discrepancy`), γ the observation weight (`FilterParams::gamma` /
   `TrackerParams::gamma`, 1 by default, 0 = observations off — in the
   tracker it multiplies the HᵀH and Hᵀr terms; the viewer exposes it in a
   shared Observation panel)
2. **Model forward**: advance the reference state t^n → t^{n+1}
3. **Transport**: p(ξ) ← p(φ⁻¹(ξ)) (spectral convection via `lobatto-grid`)
4. **Diffusion**: ∂τ p = (ε/2) Σ_d q_d ∂²_d p over one dt (one implicit-Euler
   step, directionally split by default — see `DiffusionScheme`; q = diagonal
   of the model-noise covariance, `FilterParams::q_diag`, q ≡ 1 is Δ)

This is intended as a **long-lived research library** (shared with
collaborators/students), not one-off paper code. Doc comments cite the
reference papers by name (Algo-Mortensen for the algorithm, N-oscilator and
pendulum for the models); the PDFs themselves are not stored in the repo.

**Status: the filter runs and results look plausible, but the implementation
is still under verification.** Treat the core cycle as "probably right, not
yet proven right". The verification and research programme (Kalman
comparison of the curvature, self-convergence, ε studies, sub-grid argmax,
new models) is deliberately *not* tracked in this file: Odeon is the
software reorganisation of that code, and the algorithms are frozen while
it proceeds — the to-do list below is architecture only.

## Repository layout

- `src/model.rs` — the `Model<M>` trait: the contract between the filter and
  a forward model. Neutral ground — neither the filter nor the models own it.
- `src/models/` — the forward models (one file each, re-exported from
  `models/mod.rs`):
  - `spring.rs` — `SpringSystem`: linear N-mass spring chain (Chapelle &
    Moireau), implicit midpoint, constant transition matrix T and T⁻¹.
  - `pendulum.rs` — `PendulumSystem`: planar double pendulum, point masses
    (pendulum.pdf §5.2), Gauss–Legendre 4 with Newton + analytic Jacobian;
    symmetric scheme so flow_inv = one step with −dt.
  - `pendulum_rod.rs` — `PendulumRodSystem`: planar double *compound*
    pendulum (two identical uniform rigid rods; the swaptube model,
    github.com/2swap/swaptube), same GL4 scheme. Released from rest at
    (2.453, −2.7727) its tip traces the heart-shaped orbit.
  - `kepler.rs` — `KeplerSystem`: planar Kepler problem (unit mass around a
    fixed center, state (q₁,q₂,p₁,p₂)) with the **Plummer-softened**
    potential −μ/√(|q|²+a²), `KeplerParams { mu, softening }` (defaults 1,
    0.1). Softening is a filter necessity, not a modelling choice: the
    state-space box contains the origin and φ⁻¹ is evaluated at every grid
    point. Three scalar observations via `KeplerObservation` (`Q1`, `Range`,
    `Bearing`; the bearing discrepancy is the shortest angular difference).
    `perihelion_state(e)` gives the orbit of semi-major axis 1 (period 2π at
    μ = 1). Same GL4 scheme; tests pin energy, *exact* angular-momentum
    conservation (quadratic invariant), the Jacobian, and that the flow is
    finite at the origin.
  - `kepler_mu.rs` — `KeplerMuSystem`: the joint state–parameter estimation
    variant of Kepler (M = 5), augmented state (q₁,q₂,p₁,p₂,θ) with the
    gravitational parameter μ(θ) = μ₀·2^θ and θ̇ = 0
    (`KeplerMuParams { mu0, softening }`). The exponential parameterization
    keeps μ > 0 for every real θ (no domain constraint; Gaussian prior in θ
    = log-normal in μ); θ is exactly conserved by GL4 (its stage equation
    decouples), and q_θ = 0 in `q_diag` realizes dθ/dt = 0 in the filter
    (each θ-slice runs its own Kepler flow; the observation step
    discriminates slices). Reuses `KeplerObservation` (h depends on q only,
    zero θ-gradient component). μ is identifiable from position observations
    (period, Kepler III) — the *satellite's* mass would not be (it cancels
    from q̈). Tests: energy + exact θ/L conservation, 5×5 Jacobian (ln 2
    θ-column), round-trip, 5D grid-filter smoke run, and tracker
    identification of θ from q₁ — in a deliberately local regime (stiff
    prior on the known (q,p), θ_true = 0.1): for large θ the q₁ likelihood
    is phase-aliased/multimodal and the EKF falls into a secondary basin —
    the global regime is the grid filter's job.
  - `spring_mass.rs` — `SpringMassSystem<N>`: the spring chain of N = 1,
    2, 3 bodies with an **unknown free-end mass** (M = 2N+1; `Model<M>`
    instantiated by a macro for (1,3), (2,5), (3,7)), augmented state
    (y₁..y_N, v₁..v_N, θ), m_N = m₀·2^θ, θ̇ = 0, `SpringMassParams { m,
    m0, k }` (y = displacements from rest, every spring stiffness k, body
    1 at the wall; for N = 1 the single body is the unknown one),
    observation = one position (`obs` index), `state(y, v, θ)` builds the
    array. **Conditionally linear**: given θ the dynamics are linear in
    (y, v) — the class where the hybrid grid–Gaussian closure of
    `docs/hybrid_closure.tex` is exact — and the discrete GL4 flow keeps
    that structure (`transition(θ)` returns the 2N×2N map; test
    `flow_is_conditionally_linear`). Energy is a quadratic invariant at
    fixed θ, conserved *exactly* by GL4 (test to 1e-10, N = 1 and 2).
    Other tests: Jacobian vs finite differences (N = 2, 3; ln 2
    θ-column), round-trip, 3D and 5D grid smoke runs, tracker
    identification of θ from a position (N = 1 and 2, local regime: stiff
    prior on (y,v), θ_true = 0.15, P_θθ ≈ 0.08 after t = 30).
  - `lorenz.rs` — `LorenzSystem`: Lorenz-63 (M = 3), `LorenzParams
    { sigma, rho, beta }` (defaults 10, 28, 8/3), observation one coordinate
    (`LorenzObservation::{X, Y, Z}`), `fixed_points()` = C±. First
    non-Hamiltonian, dissipative model: volume contracts at rate −(σ+1+β),
    so `q_diag` model noise is what keeps p from collapsing; the inverse
    flow is expanding, so filter domains need generous margins. Tests:
    Jacobian, round-trip, C± stationary, boundedness, fourth order.
  - `random_walk.rs` — `RandomWalkSystem`: the **random walk** toy (M = 2): a static
    point (x, y) in the plane, d_t(x, y) = 0 (identity flow, Φ = I, no
    stepper), observed through its squared distance to a lamppost at the
    origin, h = x² + y²; reference (0, 1) (`DEFAULT_STATE`). The
    posterior is a **ring** x² + y² = 1 — a continuum of maxima: the grid
    filter shows it (test `grid_filter_finds_the_ring`), the tracker
    collapses onto one point of it chosen by its prior (test
    `tracker_collapses_onto_one_point_of_the_ring`). The drunkness is the
    filter's model noise q (diffusion keeps the ring from collapsing);
    optional `with_walk(std, seed)` makes the reference itself
    random-walk. Viz entry "Random walk" (scene: lamppost, man, walk trace,
    observed circle; `estimator_hints` sets the (−2, 2)² box at
    16×4 so the whole ring is on the grid).
  - `gl4.rs` — the shared **const-generic** Gauss–Legendre 4 stepper
    (Newton + analytic Jacobian, nalgebra dynamic LU — the const-generic LU
    needs typenum bounds; **all linear solves go through nalgebra, never
    hand-written**). Symmetric ⇒ flow_inv = step with −dt. Used by every
    nonlinear model (pendulums, Kepler, Lorenz): GL4 everywhere, by
    decision, even where there is no energy to conserve.
- `src/filter.rs` — `MortensenFilter<M, Mod>`: the filter driver (grid,
  spectral diffusion solver, p field, the 4-step cycle, run loop).
- `src/tracker.rs` — `MortensenTracker<M, Mod>`: the **second-order
  (Gaussian) closure** of the same filter — one state x̂ and the curvature
  S = ∇²V(x̂), i.e. Mortensen's minimum-energy estimator / EKF in
  information form, ε-free. Mirrors the filter cycle step by step (obs:
  S += dt HᵀH, x̂ += dt S⁻¹Hᵀ(y − h); transport: x̂ ← φ(x̂), S ← Φ⁻ᵀSΦ⁻¹
  with the *exact* discrete-flow tangent Φ; diffusion: S ← S(I + dt Q S)⁻¹).
  Unimodal by construction. Tests: equals an independent covariance-form
  Kalman filter on the spring to 1e-9 (`matches_kalman_filter_on_linear_model`),
  the grid filter's argmax follows it on the spring to grid accuracy
  (`grid_filter_argmax_follows_the_tracker_on_linear_model` — the first
  filter-vs-Kalman validation), and it locks onto Lorenz from x alone.
- `src/box_tracker.rs` — `BoxTracker<M, Mod>`: the **translating
  window** (docs/density_filter.tex §3, hybrid note J.4): the grid filter
  on a small box ∏[−L_d, L_d] in ξ = x − x̂ that follows the mode of p.
  `Shifted<Mod>` is a `Model<M>` adapter reading the physical model at
  x̂ + ξ (flow_inv(ξ) = φ⁻¹(x̂ + ξ) − x̂, discrepancy, h, Jacobians;
  `is_autonomous = false`); `BoxTracker` owns a `MortensenFilter<M,
  Shifted<Mod>>` (Dirichlet in every direction, no plan — `convect`
  path) with `BoxTrackerParams { half_width, eps, gamma, n_el, p_ord,
  diffusion, q_diag }`. One `forward()` = the filter cycle with the frame
  frozen, then `recentre()`: the element containing the node argmax of ρ
  becomes the central element (⌊n_el/2⌋), x̂ += k·h and the field is
  translated by whole elements — `MortensenFilter::shift_by_elements`, an
  **exact index shift** (Lobatto nodes map onto nodes; entering nodes get
  0), no interpolation and no equation for x̂: the gauge "mode at the
  origin" is enforced up to one element, which is all the truncation
  needs. Estimate = x̂ + argmax ρ (node accuracy; sub-grid refinement by
  Newton on the spectral interpolant is the planned next step, giving
  the curvature too). Requirements: dt·|mode velocity| ≪ h, L_d = a few
  widths √(ε(S⁻¹)_dd) chosen a priori, n_el ≥ 3 (odd centres ξ = 0).
  **Preimage cache** (`BoxTrackerParams::pre_compute_flow_inv`, autonomous
  models only, the viewer sets it from `is_autonomous`): φ⁻¹ of a
  *physical* point never changes and an element shift maps nodes onto
  nodes, so the tracker caches φ⁻¹(x̂ + ξ_i) per node in physical
  coordinates, installs them minus x̂ as the filter's transport plan
  (`MortensenFilter::set_transport_preimages` — plan from explicit
  preimages via `eval_extended_plan` plus the Dirichlet mask;
  `clear_transport_plan` reverts to `convect`), and on a move translates
  the cache by the field's index shift (`filter::shifted_source`, shared
  with `shift_by_elements`), evaluating φ⁻¹ only at the entering nodes
  before re-installing the plan. Measured (`benches/transport.rs`,
  pendulum window 19⁴ ≈ 130k DOFs, 7 moves in 10 steps): 0.046 →
  0.012 s/iteration (×3.9). Tests:
  `shift_by_elements_is_an_exact_node_translation` (filter),
  `window_follows_the_tracker_on_linear_model` (spring, half-width 1
  while the reference swings ±1.15: follows the Kalman tracker to node
  accuracy over half a period, window moves, mode stays in the central
  element), `preimage_cache_matches_the_convect_path` (both paths in
  lockstep, fields equal to 1e-10 across several moves),
  `gamma_zero_follows_the_flow`. Viz entry "box tracker"; no CLI example
  yet.
- `src/particles.rs` — `ParticleSystem<M, Mod>`: the **particle
  approximation** of the filter, a Fleming–Viot-type population after §3
  of `bib/Notes_Mortensen_et_dynamique_des_populations.pdf` (2026-09-11,
  observations in since the same day). `ParticleParams { n_particles,
  center, sigma, domain, periodic, eps, q_diag, noise_scale, gamma,
  seed }`; `new` draws N positions from the Gaussian initial data (centre
  x_c, variance ε/σ_d per direction — the filter's prior; uniform in
  `domain[d]` where σ_d = 0; periodic directions wrapped) and an
  **exponential clock** e_i ~ Exp(1) per particle. One `forward()`
  mirrors the filter cycle: (1) observation = killing: the accumulated
  misfit a_i += dt·γ·d(y_n, h(Z^i)) (`Model::discrepancy`, the squared
  distance) and the particle dies when a_i ≥ e_i — the note's (1.25),
  so the death time is not known in advance; γ = 0 ⇒ immortal; (2)
  births: each dead particle copies a survivor of the step drawn
  uniformly (all deaths decided first: order-independent), fresh clock,
  misfit 0; **panics if nobody survives** (dt·γ·d too large for the
  whole cloud, or N too small); (3) model forward (reference +
  observation); (4) move by the exact discrete flow φ then the Brownian
  increment s·√(ε q_d dt)·G_d (s = 1 reproduces the filter's diffusion —
  a library knob only, the viewer fixes s = 1; periodic directions
  wrapped). The killing rate is γ·d, *without* the filter's 1/(2ε)
  (the note's ε⁻¹f would make the jump frequency explode at small ε —
  its own caveat). Deterministic in the seed (`noise::SplitMix64`, now
  `pub(crate)`). No estimate yet. Tests: Gaussian initial draw
  (mean/variance, Exp(1) clocks), pure flow with γ = 0 and no noise,
  misfit kills and births copy a survivor (cloud pulled to the observed
  position), reproducibility in the seed, periodic wrap.
- `src/output.rs` — output routines of the filter (binary snapshots of p,
  trajectory writer/printer, progress bar), kept out of the numerical core.
  New models go in `src/models/`; new output formats in `src/output.rs`.
- `examples/` — `one_spring` (2D), `two_springs` (4D, heavy), `pendulum`
  (4D, heavy). Each writes binary snapshots, `estimate.bin` (x̂ = argmax p
  per step) + `meta.txt` to `examples/output/<name>/`. Each also has a
  `<name>_forward` variant running the model alone (no filter) via
  `output::save_forward`: prints/saves the state ranges — use it to choose
  the filter-domain intervals before a filter run. `pendulum_rod` runs the
  filter on the heart orbit (periodic angles, ≈26.6M DOFs at p_ord 6 — the
  target problem size). `kepler` / `kepler_forward`: the e = 0.5 softened
  Kepler orbit over two periods, range observation by default (≈1.2M DOFs).
  `lorenz` / `lorenz_forward`: Lorenz-63 over t ∈ [0, 20] observing x
  (3D, 65³ ≈ 275k DOFs, ε = 1 — the attractor spans tens of units).
- `scripts/visualize_filter.py` — plots the 2D max-marginal snapshots with
  the reference trajectory overlaid; snapshots are redisplayed through the
  run's own spectral-element basis (per-element Lagrange interpolation onto
  a fine grid, from `n_el`/`p_ord`/`periodic` in meta.txt) so figures keep
  the solver's resolution instead of Q1 cells (raw-grid fallback for older
  runs).
- `scripts/visualize_model.py` — plots a trajectory (one time-series subplot
  per component, ranges printed); optional second argument overlays an
  estimator trajectory in red and prints the max error (e.g.
  `visualize_model.py pendulum estimate`).
- `scripts/visualize_pendulum.py` — double pendulum in the physical (x, y)
  plane from a `pendulum_forward` or pendulum filter run: static overlay
  (pendulum.png), live animation on interactive backends, `--gif` for an
  animated export; optional second argument draws an estimator trajectory in
  red. Reads L₁/L₂ from the run's meta.txt (the pendulum filter example
  appends them after `run()`).
- `benches/transport.rs` — plain-main bench (`harness = false`) timing the
  transport step's two paths on the pendulum (fixed box 21⁴), then the
  translating window's two paths (`convect` vs the preimage cache) on a
  19⁴ window of the same pendulum; run with `cargo bench --bench transport`.
- `benches/diffusion.rs` — plain-main bench timing the diffusion step's
  spectral solve with `lobatto-spectral` (dense per-direction
  eigendecomposition, real data — the one the filter uses) against
  `lobatto-fft` (HoFFT, complex data) on the repository's 4D grids, and
  checking that the two agree; `cargo bench --bench diffusion` runs the
  21⁴, 33⁴ and 9.5M grids, `-- --heavy` adds the 26.6M (p_ord 6) and 43M
  ones. Results in the To-do's profile notes.
- `viz/` (now `crates/ode-models-egui`, `crates/ode-observers-egui`,
  `apps/viewer`, see the path mapping above) — interactive egui/eframe
  viewer for the forward models, which can also launch the Mortensen filter
  on a completed run: `viz/src/filtering.rs` holds the dimension-erased
  `FilterConfig` (per-variable domains defaulted from the run's state ranges,
  fixed (−π, π) for periodic angles; per-variable n_el/p_ord; ε = 0.01 and
  per-variable σ_d = 0.01 by default for every model (2026-09; was 0 = flat), per-variable Gaussian
  center x_c defaulted to the run's initial state (`init_filter_gaussian`,
  V₀ = Σ_d σ_d (x_d − x_c,d)²/2); per-variable model-noise q_d; diffusion scheme;
  models may adjust these computed defaults via
  `VizModel::estimator_hints` (per-variable `VarHint` overrides applied by
  the app's `apply_hints`) — Kepler-μ uses it for its twin
  experiment (fixed θ domain (−1, 1), cheap θ resolution, q_θ = 0, Gaussian
  center at θ = 0 instead of the run's true θ, since the generic defaults
  would hand the filter the answer);
  selectable 2D output planes) and the generic `run_filter` each model
  instantiates at its compile-time dimension (springs dispatch M = 2/4/6 by
  chain length). Outputs (`FilterOutput`: axes, max-marginal snapshots per
  selected plane via the library's `marginal_2d`, per-step argmax estimates)
  are displayed by `viz/src/filter_view.rs` (spectral heatmaps of the selected
  planes with the reference overlaid; the plot grid is the filter's own mesh —
  element boundaries strong, Gauss–Lobatto nodes faint — via `mesh_marks`;
  when a tracker run exists, the "tracker x̂" checkbox in the view header
  overlays the tracker's estimate trajectory up to the playback time in red
  (`run_tracker` wraps the periodic components of x̂ to (−π, π) as the
  models wrap the reference; the tracker itself works with unwrapped
  angles, harmless since its flow and innovation are 2π-periodic — before
  2026-09-11 a winding pendulum's red curve ran past π while the reference
  jumped to −π, which looked like a non-periodic q₁)
  with a diamond at the current point — `FilterView::show_tracker`, on by
  default — so the Gaussian closure's single mode can be compared with the
  density, e.g. on the random-walk ring or the θ-marginals; and the
  "tracker density" checkbox (`show_tracker_density`, on by default) adds
  a pane with the tracker's own density at the playback step, the
  Gaussian p ∝ exp(−½(x−x̂)ᵀS(x−x̂)/ε) max-marginalized on a plane of its
  choice — exp(−½ξᵀP_ab⁻¹ξ/ε) with P_ab the plane's block of P = S⁻¹ —
  normalized by its max, same colormap and mesh as the filter panes
  (`tracker_image`, cached per (step, plane), flat where S is singular;
  on a periodic direction the offset x − x̂ is taken to the nearest image
  of x̂, so the drawn Gaussian is periodic like the density beside it —
  2026-09-11));
  a **"color floor"** slider (`ColorFocus`, shared with the
  Tracking-density view; 0 by default = plain scale) makes the color scale
  of every heatmap run from fraction·max to max of the displayed snapshot
  (values below the floor take the lowest color — `shade`;
  textures are rebuilt when the floor changes; test
  `color_floor_focuses_the_scale_on_the_max`).
  Left panel: model selector +
  per-model parameter form (physical parameters, initial condition,
  observation choice, presets — e.g. the heart orbit) + dt/T + Run; runs
  compute on a worker thread (progress bar), then replay: animated per-model
  scene (oscillating spring chain with zigzag springs; pendulums with rods
  and fading tip trace; Kepler orbit in a fixed, clipped [−2, 2]² window
  with a unit momentum-direction arrow on the body and the observed quantity — q₁ projection, range
  circle or bearing ray — in the accent; initial data bounded to ±2 per
  component; the Kepler entry's estimator-only option "unknown μ" (see
  migration step 5c) runs `KeplerMuSystem` at M = 5 with an editable
  θ_true (default 0.4, the multimodal regime where the tracker fails)
  and draws θ nowhere — it is read in the state plots and the (q₁, θ)
  filter marginal, pre-selected as an output plane; the
  spring entries' estimator-only option "unknown free-end mass"
  (`SpringViz`, `VizModel::estimator_options_ui`; `SpringMassSystem<N>`
  at M = 2N + 1, N = 1..3, θ_true default 0.3, (y₁, θ) plane first — see
  migration step 5b) does the same for the unknown-mass chain; Lorenz-63 as a 3D projected polyline with an orbit camera —
  drag to rotate, scroll to zoom, camera kept in egui temp memory — with a
  wireframe box, axis triad, depth-shaded history and the observed axis in
  the accent) above an observation plot y_i(t) with a time cursor
  synced to the play/pause/speed/scrub controls. The "Mortensen estimator"
  section has a filter / **box tracker** / tracker selector: Initial data,
  Observation and Diffusion panels are shared (ε and scheme greyed out for
  the tracker, which is ε-free), Grid is filter-only, **Window** (per
  variable: half-width L, n elements, order — `VarConfig::half_width`,
  `win_n_el`; defaults L = ¼ of the run's range, n = 5; a warning and a
  disabled Run when some σ = 0, since a flat p₀ has no mode to follow) is
  box-tracker-only, Outputs (planes, snapshots, memory estimate) serves
  both grid estimators; the Run button launches the selected estimator
  (`VizModel::model_spec` gives the drawn parameters as an
  `ode_models::spec::ModelSpec`; the app wraps it with the estimator, the
  config, dt and steps in a `jobs::JobSpec` and spawns the typed
  `run_filter_spec` / `run_box_spec` / `run_particles_spec` /
  `run_tracker_spec`, which build the model through the visitor and call
  `run_filter` / `run_box` / `run_particles` / `run_tracker`; `run_box` drives
  `mortensen::box_tracker::BoxTracker` from the config, starts the window
  at the Gaussian centre, and returns a `BoxOutput { window:
  FilterOutput in ξ-coordinates (domain (−L, L), Dirichlet, marginals of
  ρ, physical estimates and reference), centers: x̂ per step }`, periodic
  components wrapped to (−π, π) like the reference). The **"Tracking
  density"** tab (`viz/src/box_view.rs`, between "Filter density" and
  "Tracker") draws, on a white background whatever the theme, the window
  at the snapshot nearest the playback time: its box and mesh in dark
  green (element boundaries strong, nodes faint), the max-marginal of ρ
  inside it (checkbox "density", reusing `filter_view::heatmap_image` /
  `mesh_lines`, now `pub(crate)`), the window estimate trajectory x̂ +
  argmax ρ in dark green with a square at the current point (checkbox
  "window x̂", on by default), the reference
  in black, and the tracker's x̂ in red when a tracker run exists
  (checkbox); plane selectable; bounds fixed to the reference and every
  window position. Test `spring_window_job_produces_outputs`. The
  fourth estimator, **LParticles** (`Estimator::Particles`; every
  user-facing label — selector, panel, Run button, progress, tab — reads
  "LParticles", 2026-09-11; panel: N only — `FilterConfig::particles`, a `ParticleConfig`
  defaulting to 500; the initial cloud is the Gaussian initial data
  (x_c, σ) of the shared Initial-data panel, γ comes from the
  Observation panel (the killing rate), ε and q from the Diffusion panel
  (the scheme selector is greyed out for the particles), the noise scale
  is fixed to 1 and the seed is drawn from the clock at every run — no
  seed in the UI; Outputs is not shown for it), runs
  `filtering::run_particles` → `ParticleOutput { positions (N·M flat per
  step), budget e and spent a per step, reference, domain, periodic }`,
  the whole population stored at every step. The **"LParticles"** tab
  (`viz/src/particles_view.rs`, after "Tracking density") draws, on a
  selectable plane, the particles as red dots and the smoothed cloud
  Σ_i exp(−|ξ − z_i|²/2h_i²) as a heatmap (same colormap and colour
  floor; "width h" slider with a fixed log range from one heatmap pixel
  to the box, sized at reset to 3 % of the box), where
  **h_i = h·(e_i − a_i)/e_i** — a newborn's Gaussian has width h and
  shrinks linearly to 5 % of h as the particle's misfit approaches its
  clock;
  Gaussians splatted within 3.5 h_i and on their periodic images;
  reference in black, tracker in red on request; checkboxes for dots
  and density; textures cached per (step, plane) and rebuilt when h or
  the floor changes. Test `cloud_image_is_the_sum_of_life_scaled_gaussians`; while it runs, a **Cancel**
  button next to the progress bar stops it after the current iteration and
  discards the partial result. Jobs receive a `playback::Progress` handle
  (step counter + cancel flag; `step()`, `done()`, `cancel()`,
  `cancelled()`), the run loops check `cancelled()` once per iteration
  (the filter's construction — transport plan — is not interruptible), and
  dropping a `RunHandle` raises the flag, so abandoning a run (Cancel,
  editing a parameter) also frees the CPU. Test
  `cancelled_jobs_stop_at_once`. Tracker results
  (`TrackerOutput`: x̂ and P = S⁻¹ per step, plus the run's ε) are shown in
  the "Tracker" tab (`viz/src/tracker_view.rs`: one plot per component,
  reference vs x̂ with the **±2√(εP) band** — P is the inverse *energy*
  curvature and the density p ∝ e^{−V/ε} has covariance εP, so ±2√P would
  be 1/√ε too wide — drawn as per-interval convex quads (egui fills
  polygons as convex fans) and **hidden by default**: its legend entry
  starts unchecked, via the plot memory's `hidden_items` seeded on first
  display, and toggles it as usual; a faint legend-less playback cursor;
  the y-range follows the estimate only, skipping the first 5 % of the
  steps). Above the
  scene, a collapsed-by-default "About this model" panel shows
  `VizModel::description()` — physics, parameters, state variables, what
  the scene draws, and the observation choices — one paragraph per topic. The viewer-side contract is
  the dimension-erased `VizModel` trait (`viz/src/models/mod.rs`) — separate
  from the filter's `Model<M>`; observations are vector-valued (ready for
  the decoupled-observation refactor) and scenes draw an arbitrary state, so
  a filter-estimate overlay can plug in later. GUI deps stay out of the
  library crate. Series/accent colors follow the validated reference palette
  in `viz/src/palette.rs` (theme-aware); the observed element in the scene
  wears its series color.
- `examples/output/` — run artifacts (generated, git-ignored, not source).
- `docs/` — design notes in LaTeX (build with `latexmk -pdf` in `docs/`;
  the PDFs are git-ignored). `hybrid_closure.tex`: pedagogical derivation
  of the Mortensen HJB equation (dynamic programming → HJB → viscous
  regularization → Hopf–Cole → the linear equation for p and its 4-step
  splitting; §1, with skippable refinements on the stochastic origin of
  the viscous term and on Dirichlet/Neumann walls as state constraints),
  and of the **hybrid grid–Gaussian filter** in its triangular case
  (§2: grid directions y with ẏ = f_y(y), model noise allowed in both
  blocks, Gaussian directions z with mean/information fields ẑ(y), S(y)
  on the y-grid, derived directly in the density representation
  p = ρ(y)·Gaussian — trackers along the characteristics f_y + Q_y∇W
  with the deviation noise Q_z + AQ_yAᵀ, weighted by the max-marginal ρ;
  exact for conditionally linear models — the deterministic
  Rao–Blackwellization — with the tracker as the k = 0 case and the
  bank of independent trackers as the q_y = 0 case; for kepler_mu this
  is the proposed fix for the θ-multimodality).
  The general closure (coupling f_y(y,z), model noise in y, weighted
  variables ρẑ, ρS, stability properties, discrete cycle) is kept in the
  appendices. Appendix J (`sec:frame`, 2026-09): the **moving frame**
  ansatz p = ρ(R(t)(x − x̂(t))) — a grid that follows the mode ("moving
  window"): exact change of frame (same drift/diffusion/misfit structure,
  Q_R = RQRᵀ full in general), small-ξ expansion of W = −ε log ρ with the
  gauge conditions (origin = minimiser, Σ = I whitening or principal
  axes) giving the *mode equations* — the tracker corrected by the
  second-order observation term and by ∇³V, ∇⁴V, which the grid
  provides — the translating frame R = I (`sec:frame_translate`: the
  closed exact system for ρ and x̂ alone, ẋ̂ = f(x̂) + (γ/ε)C⁻¹Hᵀr +
  (ε/2)C⁻¹(Q:∇³ρ)/ρ with C = −∇²ρ(0)/ρ(0), i.e. the current filter with
  its box re-centred on the argmax each step — the minimal moving window
  and the first implementation target), the orientation gauge (symmetric, co-rotating, or the unique
  whitening frame with axis-aligned diffusion R = D O Q^{-1/2} from the
  eigendecomposition of Q^{1/2} S Q^{1/2}, usable by the existing
  solver), the √ε scaling (ρ = standard Gaussian + O(√ε) on an
  ε-independent box of ≈ 5 standard deviations), the discrete cycle
  (tracker-driven frame, one composed interpolation per step on the
  convect path, re-anchoring on the grid's argmax) and its validation
  plan. Not implemented yet. `density_filter.tex` (2026-09): a
  self-contained extract, **purely deterministic by request** (no
  stochastic filtering / Zakai / Feynman–Kac / Brownian content, no
  probabilistic vocabulary: the viscous term is presented as the soft
  minimum of the inf-convolution step, Dirichlet as infinite cost outside
  the box, Gaussians have "width matrices", the Kalman filter is named
  only as a deterministic least-squares estimator) — §1 of the hybrid
  note (estimation problem → HJB → viscous → Hopf–Cole → linear equation
  for p, with the refinements), then the **tracker derived on the density** (Gaussian
  ansatz p = a·exp(−½ξᵀSξ/ε) inserted in the linear p-equation, orders
  0/1/2; each step of the discrete cycle as an exact Gaussian → Gaussian
  map: product of Gaussians with the *updated* S⁺ in the mean update as
  in `tracker.rs`, p∘φ⁻¹ with Φ, Gaussian convolution S(I + dt QS)⁻¹) and
  the **translating window** p = ρ(x − x̂(t), t) of Appendix J.4 (exact
  closed system for ρ and x̂, tracker = window with ρ frozen Gaussian,
  discrete cycle with the element-wise shift and the preimage cache, exact-vs-approximate table, validation plan) **§4 the rotating window** ξ = R(t)(x − x̂(t)), R orthogonal (no scaling): the ρ-equation with the rotation term Ωξ, Ω = ṘRᵀ skew, and Q_R = RQRᵀ (full unless Q isotropic — the price of rotating with anisotropic noise); the mode equation with RᵀC⁻¹R = C_x⁻¹ (frame-free); the rotation gauge — principal axes (C diagonal, eigenvector-derivative formulas, crossings; re-diagonalised at the discrete level rather than integrated) — and the remark that a rotation alone makes the leading Gaussian separable (rank one for §5), scaling only making the box ε-independent; discrete cycle: frame frozen, composed transport map, re-centre + re-orient by one interpolation ρ_{n+1}(ξ') = ρ(R_nR_{n+1}ᵀξ' + ξ*), which also invalidates the preimage cache — hence translate every step, rotate rarely; the gain ∏(w₁/w_d) in nodes. And **§5 the trace ansatz** ρ = tr(G₁(ξ₁)⋯G_n(ξ_n)) in the window with a single given rank r, every factor an r×r matrix (the rank-r train contained by zero-padding its end factors), stated as a *programme* only, nothing implemented: initialise (rank one from the quadratic prior; TT-SVD from a full grid), diffuse (§4.2, continuous level: entrywise 1D heat equations ∂_tG_d = (ε/2)q_d∂²G_d solve the diffusion exactly, rank unchanged, plus the gauge terms G_dΛ_d − Λ_{d−1}G_d that telescope in the trace; Gaussian convolution per factor; Dirichlet ends per factor; a non-diagonal Q would couple factors and grow the rank), drift (§5.3, worked out: exact iff the map is separable — translations, element shifts — with a 2D linear counter-example otherwise; the fit of the new factors to samples of F = ρ∘ψ: linear in each factor via tr(W_dᵀG_d), samples on fibres through K ≥ r² anchors so the least squares split node by node into N_d systems of size K×r²; the samples are grid nodes so the preimage cache supplies φ⁻¹; F = ρ + O(dt) keeps the rank moderate, shear being the rotating window's job; Figure 1 = the 2D sample set for r = 3, a TikZ file `docs/figures/sample_points.tex` generated by `docs/figures/make_sample_points.py` (both versioned; the note needs `tikz`); explicit factorised drift as the alternative for polynomial fields), multiply (exact and rank-preserving for a one-coordinate observation; Kronecker products of factors otherwise, then compress), compress (SVD sweep truncated to r, or alternating least squares), read off the mode (§4.6: derivatives by replacing factors with their 1D derivatives; with the other coordinates fixed ρ is the linear combination tr(W_dᵀG_d(ξ_d)) of the r² entry functions — a 1D maximisation; alternating maximisation sweeps from the previous mode, monotone, converge to a coordinate-wise maximum; Newton polish on −ε log ρ gives the mode and S = εC; local only), and the order to do it in (rank diagnostics on the full-grid window first). **§6 a sum of Gaussians, the min-plus algorithm on the density** (2026-09-11): superposition (linearity of the p-equation, cycle term by term), the ansatz Σ_k a_k Gaussians = K trackers plus the *heights* updated by the observation (completion-of-square factor) and the diffusion (det factor), exact for linear models; heights as accumulated costs c_k = −ε log a_k, V = softmin_ε(q_k) → min_k q_k as ε → 0 (min-plus), estimate = centre of least cost; splitting by the nonlinearity criterion η_k = max_v |∂²φ[v,v]| w_k(v) ≪ 1 with 1D split tables (the one non-linearisation approximation); pruning by pairwise domination q_k − q_j ≥ δ (Hessian S_k − S_j ≻ 0 + minimum test, relative error e^{−δ/ε}) and merging by matching mode and curvature of the pair's sum (deterministic, not moment matching); the step-by-step algorithm (O(Kn³) + O(K²n³) housekeeping); the tracker = K = 1, the hybrid note's bank = pinned centres without housekeeping; McEneaney's curse-of-dimensionality-free method as the ε = 0 version (min-plus linearity of the HJB semigroup, switched-linear Riccati branching, curse of complexity, pruning with convergence rates) with references from memory. Reviewed 2026-09-10 (notation: x for grid points, ξ = x − x̂ only in the window; N_d nodes per direction; n the state dimension); §§5–6 with their side points in remark environments; 25 pages. Same
  build (`latexmk -pdf`), PDFs git-ignored.

## Commands

```sh
cargo test                                 # unit tests + doctests
cargo run --release --example one_spring   # fast (2D)
cargo run --release --example two_springs  # heavy: ~43M DOFs (4D)
cargo run --release --example pendulum     # heavy: ~43M DOFs + Newton per DOF
cargo run --release --example kepler       # softened Kepler orbit, range observation (4D, ~1.2M DOFs)
cargo run --release --example lorenz       # Lorenz-63 observing x (3D, ~275k DOFs)
python scripts/visualize_filter.py one_spring   # figures into examples/output/one_spring/
cargo run --release --example pendulum_forward  # model-only solve + state ranges
python scripts/visualize_model.py pendulum_forward    # trajectory time series
python scripts/visualize_pendulum.py pendulum_forward # (x,y)-plane animation
cargo bench --bench transport              # time the two transport paths (pendulum)
cargo bench --bench diffusion              # lobatto-spectral vs lobatto-fft diffusion step (4D grids; -- --heavy for 26.6M/43M)
cargo run -p odeon-observers-viewer --release   # observers viewer (GUI); odeon-models-viewer = models only
```

Always use `--release` for the 4D examples.

## Key design points (current state)

- **`Model` trait has two faces**: the grid filter consumes `discrepancy`
  (scalar, fast per-grid-point path) and `flow_inv`; the tracker consumes
  `flow`, `flow_jacobian` (exact tangent of the GL4 step from the stage
  system — `gl4_step_with_jacobian`, one extra LU solve; the spring returns
  T), `n_obs`/`h`/`y_obs`/`obs_jacobian` (analytic; the spring's opaque
  observation closure is the one finite-difference exception, exact for its
  linear observations) and the provided `innovation` (overridden by Kepler
  for the bearing's angular difference). Vector observations (pendulum tip)
  are supported on the tracker side.
- **Flat state convention**: `Model` exposes the state as one flat vector.
  `discrepancy(&[f64])`, `flow_inv([f64; M])`, `states() -> &[DVector<f64>]`
  and `state_labels()` all use the same component order (e.g. `[Y; V]` for
  springs, `(q₁,q₂,p₁,p₂)` for the pendulum). Any (position, momentum)
  structure lives inside the model, never in the filter.
- **Pendulum physical parameters are runtime values**: `PendulumParams`
  (ℓ₁, ℓ₂, m₁, m₂, g) and `PendulumRodParams` (m, ℓ, g), stored on the
  systems as `params`, with `Default` = the paper/swaptube values. `new`
  keeps its old signature (defaults); `with_params` takes explicit values
  (used by the viewer's sliders). Tests pin energy conservation and the
  analytic Jacobian at deliberately non-unit parameters, where errors no
  longer cancel.
- **Compile-time dimension**: `M` is a const generic because the spectral
  solver and grid work on `[f64; M]` grid points. `SpringSystem` implements `Model<M>` for
  any M; `MortensenFilter::new` checks at runtime that M matches the model's
  state dimension.
- **Solver parameters via `FilterParams<M>`**: `MortensenFilter::new(model,
  params)` takes a parameter struct (per-direction `domain` intervals, ε,
  per-direction `n_el` and `p_ord`, `diffusion` scheme), stored whole on the filter as
  `filter.params` — the filter keeps
  no loose copies of individual parameters. Deliberately no `Default` — every
  run states its parameters explicitly, for reproducibility. Resolution may
  differ per direction: `params.ndof()[d] = n_el[d]·p_ord[d] + 1`, `meta.txt`
  stores `n_axis` (plus `n_el`, `p_ord`, `periodic`) as comma-separated
  per-direction lists, and `visualize.py` reads them as such. New solver parameters go in `FilterParams`, not as extra
  `new` arguments; initial-condition (`sigma`/`v0`) and run/output settings
  stay out of it.
- **dt plays three roles**: model time step, observation weight, and diffusion
  horizon per iteration (one implicit-Euler step — deliberately no
  sub-stepping, see the diffusion scheme below).
- **Diffusion scheme via `FilterParams::diffusion`** (`DiffusionScheme`):
  `SplitEuler` (default in every example) is the *directionally split*
  implicit Euler step p ← ∏_d (I + dt·(ε/2) M_d⁻¹K_d)⁻¹ p
  (`PoissonND::euler_split_step_in_place`): the 1D factors commute, so it is
  exact as a tensor product and costs one solver application, and each 1D
  factor obeys the discrete maximum principle **for dt·ε/2 large enough
  relative to the local mesh size squared** ⇒ **nodal positivity** of p,
  which the unsplit step (`Euler`, `PoissonND::euler_step_in_place`, kept for
  comparisons) does not guarantee. That is why there is **no diffusion
  sub-stepping** (the former `n_sub` was removed): smaller steps work
  against positivity. The splitting error is O(dt²), same order as implicit
  Euler itself. Tests: `split_euler_diffusion_is_nodally_positive` (a Dirac
  nodal vector through the step alone — the whole cycle cannot be
  positivity-preserving, since the degree-p Lagrange interpolation of the
  transport overshoots) and `split_and_unsplit_diffusion_are_consistent`.
  The operator coefficients are passed at each application, not to
  `PoissonND::new` (same interface in lobatto-spectral and lobatto-fft ≥ 0.4). **Diagonally anisotropic diffusion**
  (`FilterParams::q_diag`, the diagonal of the model-noise covariance Q):
  the diffusion is (ε/2) Σ_d q_d ∂²_d p, implemented through
  `PoissonND::apply_in_place_dirs`, whose symbol receives the per-direction
  stiffness eigenvalues μ_d of each mode — split: ∏_d (1 + dt·β_d μ_d)⁻¹,
  unsplit: (1 + dt Σ_d β_d μ_d)⁻¹ with β_d = ε q_d/2. q_d = 0 turns
  diffusion off in a direction (test
  `anisotropic_diffusion_acts_only_on_weighted_directions`). The viewer
  exposes q per variable in the Diffusion panel.
- **Autonomy flag**: `Model::is_autonomous()` says whether the discrete flow
  map is time-independent (`true` for both current models). With
  `FilterParams::pre_compute_flow_inv` the filter exploits it: an `EvalPlan`
  (`lobatto_grid::EvalPlan`, built by `GridND::convect_plan`) is built at
  construction — φ⁻¹ of
  every grid point *plus* its extension folding, element localization and DOF
  gather tables — and the transport step is `eval_with_plan` (pure gather +
  contraction) instead of `convect` (which redoes flow evaluation and element
  search per DOF each iteration). Costs ~(8 + 8·M) bytes per DOF plus one
  gather table per occupied element; `new` panics if requested on a
  non-autonomous model. A test (`precomputed_transport_matches_convect`) pins
  both paths to agree. Measured (2026-07, M1-class laptop, pendulum at
  21⁴ ≈ 194k DOFs, `benches/transport.rs`): ×2.5 per iteration
  (~0.03 s → ~0.01 s), plan build ~0.01 s (amortized in ~1 iteration) — worth
  it for any autonomous model, not just expensive flows.
- **p is real, end to end**: `p_field` is `Vec<f64>` — p = exp(−V/ε) is
  real; transport (`eval_with_plan`/`convect` via `InterpScalar`) and the
  `lobatto-spectral` diffusion (`apply_in_place_dirs` on real nodal vectors)
  both run on real data. The former `lobatto-fft` diffusion was complex-only
  and needed a complex scratch buffer `c_buf` plus two copies per iteration;
  that buffer and the `num-complex` dependency are gone (2026-09).
- **The cycle is allocation-free on the mortensen side**: transport writes
  into `p_buf`, diffusion steps `p_buf` in place and swaps it with
  `p_field`; both are sized once in `init_filter`. Removing allocations did
  not change the timings (memory churn, not a bottleneck); the remaining
  per-iteration allocations are internal to the crates (lobatto-spectral's
  sweep allocates two small fibre vectors per fibre).
- **Observations are model-internal, optionally noisy**: the model caches
  y_n = h(current reference state) + η_n at each `forward()`; the filter
  queries `discrepancy(ξ) = y_n − h(ξ)`. η ≡ 0 by default; each model's
  `with_obs_noise(noise, seed)` builder enables a `src/noise.rs`
  `NoiseModel` (white Gaussian σ; bounded uniform a; colored AR(1) σ, τ) —
  deterministic in the seed, so a run is reproducible and the viewer can
  plot exactly the sequence the filter consumed. Decoupling observation
  generation from the models entirely is still on the roadmap (see Future
  state).
- **p is unnormalized**: overall mass decays over time; only the shape (and
  argmax) matters. `visualize.py` normalizes each snapshot by its max.
- **Boundary conditions**: per direction via `FilterParams::periodic` —
  `true` = periodic (interval identified end-to-end: diffusion uses periodic
  BCs, transport wraps, ndof = n_el·p_ord), `false` = Neumann on the box
  (ndof = n_el·p_ord + 1); transport folds outside points per the extension.
  The pendulum example marks its two angles periodic on (−π, π). A test
  (`periodic_directions_are_consistent`) pins `FilterParams::ndof` to the
  solver's grid under mixed BCs, and `periodic_angle_wraps_across_the_boundary`
  checks the semantics on the pendulum: a bump just below q₁ = π shows up
  above −π after a few diffusive steps when the direction is periodic, and
  not at all under Neumann (added 2026-09-11 after a report that q₁ looked
  non-periodic — the library was right; the visible effect was the
  tracker's unwrapped x̂, see viz). **The Gaussian prior is wrapped on
  periodic directions** (`init_filter_gaussian`, 2026-09-11): the plain
  Gaussian in x − x_c is not a function on the circle (its ends differ ⇒
  a jump at the seam that the periodic diffusion then rings on), so on a
  periodic direction g_d is the sum over the images of the centre,
  Σ_k exp(−σ_d(x − x_c + kL_d)²/2ε) — the wrapped normal / theta function,
  the heat kernel of the circle — truncated where the next image is below
  e⁻⁴⁰ (one or two images for a narrow prior; the wide-prior limit is
  flat). Non-periodic directions keep the plain Gaussian; `init_filter_quadratic*`
  route through it. Test `gaussian_prior_is_wrapped_on_periodic_directions`
  (node values vs an independent image sum; p(−π) = p(π⁻) ≈ 0.41 for a
  centre 0.3 below π, where the plain Gaussian gives ≈ 0). **Dirichlet wall option**
  (`FilterParams::dirichlet`, per direction, invalid with periodic): p = 0
  on both endpoints — V = +∞ at the wall, *absorbing* — with the semantics
  **p ≡ 0 outside the box**. Diffusion enforces it spectrally (sine basis,
  boundary nodes not stored, ndof = n_el·p_ord − 1); transport enforces it
  *explicitly*: grid points whose preimage φ⁻¹(ξ) leaves the box in a
  Dirichlet direction are zeroed (the grid's odd folding would give
  −p(reflected) — deliberately not used). The mask of such points is
  precomputed with the transport plan (autonomous models, zero per-step
  cost); on the `convect` path it costs one extra φ⁻¹ per grid point per
  step. meta.txt gains a `dirichlet=` line (`visualize_filter.py` falls
  back to raw-grid display for such runs). Tests:
  `dirichlet_diffusion_absorbs_at_the_boundary` (constant field: Neumann
  invariant, Dirichlet sags at the wall, stays in [0, 1]),
  `dirichlet_transport_zeroes_outside_preimages` (outside preimage ⇒
  exactly 0, both transport paths agree). In the viewer the wall is a
  per-variable checkbox ("wall" column of the Grid panel, disabled for
  periodic angles); the heatmap resampler treats the unstored boundary
  zeros correctly (`dirichlet_axis_resampling_is_exact`).
- **`lobatto-spectral` / `lobatto-fft` / `lobatto-grid` are also our
  crates**: the grid geometry and interpolation layer (`Grid1D`/`GridND`,
  `eval`/`convect`/`convect_plan`/`eval_with_plan`, `EvalPlan`) lives in
  `lobatto-grid` (a direct dependency here). The diffusion solver is
  `lobatto-spectral` (since 2026-09; `PoissonND<N>` with the same interface
  as lobatto-fft's, restricted to real data: per-direction dense generalized
  eigendecomposition K_d V = M_d V diag(μ_d) at construction, one
  application = mass scaling + N forward sweeps Vᵀ_d + per-mode symbol + N
  inverse sweeps V_d, cost O(Σ_d n_d r_d · N_dof) — the FFT solver is
  O(Σ_d r_d² · N_dof) plus the FFT, so the eigen solver's edge shrinks as
  n_d·r_d grows; measured ×3 faster at 21–33 DOFs per direction, ×1.3–1.6
  at 81–121, see `benches/diffusion.rs`). `PoissonND::new` takes no (α, β)
  — they go to `solve_in_place`/`apply_in_place*`/`euler_*_step_in_place` —
  and `apply_in_place_dirs` gives any tensor-product operator function
  (basis of the split Euler step); the grid is exposed via
  `PoissonND::grid()`. `lobatto-fft` (HoFFT: FFT across the elements, r × r
  symbol per Fourier mode, complex data) stays a dev-dependency for the
  comparison bench. Filter work may require changes in any of the three.

## Conventions

- English everywhere; math-heavy doc comments in the current style (Unicode
  math, citations of the reference papers by name).
- Models implement `Model<M>` and own their physics; the filter stays
  model-agnostic.
- Tests validate models today (energy conservation, Jacobian vs finite
  differences, flow round-trip); filter-level validation is a to-do, not done.

## Future state (where this is going)

1. **Realistic twin experiments**: observations decoupled from the model — the
   model only provides h(x) and the reference trajectory; a separate
   observation component generates the sequence y_n = h(x_ref) + noise and
   feeds the filter. This is a deliberate refactor of the `Model` trait /
   filter interface.
2. **Tractable 4D+ runs**: performance work, starting with profiling the
   current CPU/Rayon code (transport Newton per DOF and the diffusion
   solves are the suspected hotspots).

## To-do list (architecture only, in priority order)

1. **Decouple observations + add noise**: refactor so observation generation
   (reference + noise) lives outside the model; the model keeps only h(x).
   An interface change (the `Model` trait / filter interface), not an
   algorithm change; it fits naturally after the crate split.
2. **Profile & optimize**: done so far: allocation-free transport on the
   mortensen side (`p_buf`); real `p_field` (`Vec<f64>` +
   `solve_in_place_real`); and `pre_compute_flow_inv` now builds a lobatto-fft
   0.2.2 `EvalPlan` (`convect_plan`/`eval_with_plan`) — this was the "big
   lever (b)" identified by the 2026-07 profile (precomputed element
   localization + basis weights for fixed evaluation points) and delivers the
   predicted ×2.5 per iteration (`benches/transport.rs`, pendulum at 21⁴).
   Fresh profile on the plan path (2026-07, macOS `sample` 15 s, `pendulum_rod`
   at 1.96M DOFs, ~0.10 s/iteration — was ~0.34 s; `debug = true` is set in
   the release profile for symbols):
   - wall-clock per iteration is now **~48% transport / ~51% diffusion**
     (observation ~1.5%) — diffusion went from 3% to half the iteration
     because transport shrank ×3+ while diffusion is unchanged;
   - transport busy time is still interpolation math: `contract` (~30k
     samples) + `lagrangian_interpolation_into` (~25k) — the plan stores
     local coordinates but **re-evaluates the 1D basis weights every apply**
     (`fill_phi`); good parallel occupancy (~8 threads busy);
   - diffusion has only ~10k busy samples but poor occupancy (~2–3 threads;
     large rayon idle/backoff counts) — it is parallel-inefficient /
     memory-bound, not compute-heavy;
   - malloc/free churn is still ~22k samples: per-group `u_local` gathers in
     `eval_with_plan`, the per-mode `DVector` in the symbol solve, and
     `solve_in_place_real`'s temporary complex buffer.
   Same profile at the **target size** (60×60×121×61 = 26.6M DOFs, p_ord 6,
   ~2.1 s/iteration — ~13.6× the DOFs for ~21× the time, mildly superlinear):
   - wall-clock is **~76% transport / ~22% diffusion** (observation ~1%) —
     the higher p_ord tips the balance back to transport;
   - transport busy time is dominated by the **tensor contraction itself**
     (`contract` ≈ 70k samples vs ≈ 23k for the basis evaluation — 3:1,
     where at p_ord 4 they were 1:1; the 7⁴-DOF local element makes the
     contraction the cost). Occupancy is good (~9 threads);
   - alloc churn is relatively minor here (~5% of busy samples); diffusion
     still runs at ~2–3 threads (occupancy, not compute, is its problem).
   Next levers at target size, in order: (a) the contraction kernel itself
   in lobatto-grid (SIMD/layout; it is also exactly the GPU-shaped hotspot
   — one big gather + contract is now 3/4 of the iteration, which is real
   data for the open GPU question); (b) ~~diffusion parallel efficiency in
   lobatto-fft's `solve_in_place`~~ — superseded by the **switch of the
   diffusion solver to `lobatto-spectral`** (2026-09, M4 Pro, 14 threads,
   `benches/diffusion.rs`, split-Euler symbol, identical results to
   1e-14 — 1e-12 at p_ord 6): per application lobatto-fft → lobatto-spectral
   is 21⁴ 0.0084 s → 0.0027 s (×3.1), kepler 33⁴ 0.034 s → 0.0098 s (×3.5),
   pendulum 9.5M 0.19 s → 0.10 s (×1.9), pendulum_rod 26.6M at p_ord 6
   0.45 s → 0.34 s (×1.3), two_springs 43M 0.84 s → 0.53 s (×1.6). Full
   pendulum iteration at 21⁴ on the plan path: 0.0084 s (`benches/transport`,
   was ≈ 0.014 s: the diffusion step alone used to cost as much as the
   whole iteration does now). At the target size the gain is ≈ 5 % of the
   iteration (diffusion was 22 % of it); the remaining lever there in
   lobatto-spectral is the sweep (two Vec allocations per fibre, strided
   gathers); (c) store per-point basis weights in the `EvalPlan` (targets
   ~23% of transport busy time at p_ord 6, ~45% at p_ord 4). Not worth
   touching: observation, mortensen-side code. GPU offload vs algorithmic
   DOF reduction (adaptive grids, moving window) is **deliberately
   undecided** — revisit with the data above.

## Open questions (recorded, not decided)

- Performance path beyond CPU optimization: GPU vs algorithmic reduction.
- Exact interface of the decoupled observation component (noise model,
  non-scalar observations).
