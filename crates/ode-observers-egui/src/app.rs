//! The observers application: the models-only viewer (model choice +
//! parameter form + run settings on the left, the animated scene in the
//! middle, the observation plot below — all drawn by
//! `ode_models_egui::panels`) plus the Mortensen estimators: their
//! configuration panels and Run button in the left panel
//! ([`crate::panels`]), and the result views as tabs of the central panel.
//! Each model keeps its own trajectory, playback clock and estimator
//! outputs, so switching models never loses a run.
//!
//! [`App`] is generic over a [`Compute`] backend — *how* an estimator job
//! (`ode_observers::jobs::JobSpec`) is started: [`LocalCompute`] runs it on
//! a worker thread of this process; the `observers-client` binary supplies
//! a backend posting it to an `observers-server`. The same job description
//! and the same views either way; a binary is `App::new(backend)` plus the
//! eframe boilerplate.

use crate::box_view::BoxView;
use crate::filter_view::FilterView;
use crate::panels;
use crate::particles_view::ParticlesView;
use crate::tracker_view;
use ode_models::spec::ModelSpec;
use ode_models_egui::models::{self, EstimatorHints, VizModel};
use ode_models_egui::panels as model_panels;
use ode_models_egui::playback::{Progress, RunHandle};
use ode_models_egui::slot::ModelSlot;
use ode_observers::jobs::{
    self, BoxOutput, Estimator, FilterConfig, FilterOutput, JobOutput, JobSpec, ParticleOutput,
    TrackerOutput,
};

/// What the central panel shows: the animated model scene, the filter's
/// density heatmaps, the translating window's density, the particles, or
/// the tracker's component plots.
#[derive(Clone, Copy, PartialEq)]
enum CentralView {
    Scene,
    Filter,
    Window,
    Particles,
    Tracker,
}

/// The typed runner of `ode_observers::jobs` a backend may execute locally
/// (`run_filter_spec`, `run_tracker_spec`, …): the slot's result type picks
/// it, `JobSpec::estimator` is what the user selected.
pub type LocalRunner<T> = fn(&ModelSpec, &FilterConfig, f64, usize, &Progress) -> T;

/// An estimator run in flight, whatever executes it: progress for the bar,
/// the result when done, and — for backends that can fail — why it died.
pub trait EstimatorRun<T> {
    /// Fraction of steps done, in [0, 1].
    fn progress(&self) -> f32;
    /// Seconds since the run started.
    fn elapsed(&self) -> f64;
    /// The finished result, once.
    fn try_take(&self) -> Option<T>;
    /// The failure that killed the run, if any (a local thread never
    /// fails; a server may be down or refuse the job).
    fn error(&self) -> Option<String> {
        None
    }
}

impl<T: Send + 'static> EstimatorRun<T> for RunHandle<T> {
    fn progress(&self) -> f32 {
        RunHandle::progress(self)
    }
    fn elapsed(&self) -> f64 {
        RunHandle::elapsed(self)
    }
    fn try_take(&self) -> Option<T> {
        RunHandle::try_take(self)
    }
}

/// Where and how estimator jobs execute. Implemented by [`LocalCompute`]
/// (this process) and, in the `observers-client` binary, by a backend
/// talking to an `observers-server`.
pub trait Compute {
    /// Start `job`, whose result is a `T` (one of the four outputs);
    /// `local` is the typed runner a local backend calls.
    fn spawn<T>(&self, job: JobSpec, local: LocalRunner<T>) -> Box<dyn EstimatorRun<T>>
    where
        T: TryFrom<JobOutput, Error = String> + Send + 'static;

    /// The backend's own settings row in the estimator section (a server
    /// URL, say). Nothing by default.
    fn ui(&mut self, _ui: &mut egui::Ui) {}
}

/// Jobs run on a worker thread of this process.
pub struct LocalCompute;

impl Compute for LocalCompute {
    fn spawn<T>(&self, job: JobSpec, local: LocalRunner<T>) -> Box<dyn EstimatorRun<T>>
    where
        T: TryFrom<JobOutput, Error = String> + Send + 'static,
    {
        let steps = job.steps;
        Box::new(RunHandle::spawn(
            Box::new(move |p| local(&job.model, &job.config, job.dt, job.steps, p)),
            steps,
        ))
    }
}

/// Collect a finished run: its output, or — on a failure — its message
/// into `error`; the handle is dropped in both cases.
fn collect<T>(run: &mut Option<Box<dyn EstimatorRun<T>>>, error: &mut Option<String>) -> Option<T> {
    let r = run.as_ref()?;
    if let Some(e) = r.error() {
        *error = Some(e);
        *run = None;
        return None;
    }
    let out = r.try_take()?;
    *run = None;
    Some(out)
}

/// One model plus everything the viewer remembers about it: the model
/// slot (model, run settings, trajectory, playback) and the estimator
/// state beside it.
struct Slot {
    base: ModelSlot,
    /// Estimator configuration; created (with defaults computed from the
    /// run) when a simulation completes, cleared by a change of dimension.
    filter_cfg: Option<FilterConfig>,
    estimator: Estimator,
    /// Why the last estimator run failed (backends that can), shown until
    /// the next one starts.
    run_error: Option<String>,
    filter_run: Option<Box<dyn EstimatorRun<FilterOutput>>>,
    /// Outputs of the last completed filter run, kept in memory and shown
    /// by the Filter view.
    filter_out: Option<FilterOutput>,
    tracker_run: Option<Box<dyn EstimatorRun<TrackerOutput>>>,
    /// Outputs of the last completed tracker run.
    tracker_out: Option<TrackerOutput>,
    box_run: Option<Box<dyn EstimatorRun<BoxOutput>>>,
    /// Outputs of the last completed translating-window run.
    box_out: Option<BoxOutput>,
    particle_run: Option<Box<dyn EstimatorRun<ParticleOutput>>>,
    /// Outputs of the last completed particle run.
    particle_out: Option<ParticleOutput>,
    view: CentralView,
    filter_view: FilterView,
    box_view: BoxView,
    particles_view: ParticlesView,
}

impl Slot {
    fn new(model: Box<dyn VizModel>) -> Self {
        Slot {
            base: ModelSlot::new(model),
            filter_cfg: None,
            estimator: Estimator::Filter,
            run_error: None,
            filter_run: None,
            filter_out: None,
            tracker_run: None,
            tracker_out: None,
            box_run: None,
            box_out: None,
            particle_run: None,
            particle_out: None,
            view: CentralView::Scene,
            filter_view: FilterView::default(),
            box_view: BoxView::default(),
            particles_view: ParticlesView::default(),
        }
    }

    /// An estimator run is in flight.
    fn estimating(&self) -> bool {
        self.filter_run.is_some()
            || self.tracker_run.is_some()
            || self.box_run.is_some()
            || self.particle_run.is_some()
    }

    fn cancel_estimators(&mut self) {
        self.filter_run = None;
        self.tracker_run = None;
        self.box_run = None;
        self.particle_run = None;
    }

    /// After an edit of the model or the run settings: discard any
    /// finished or in-flight estimator run and the stored *outputs* (they
    /// no longer correspond to the edited model); the estimator
    /// *configuration* is kept — edits must not reset the user's tuning
    /// (the `default` button recomputes it on demand).
    fn clear_estimator_state(&mut self) {
        self.cancel_estimators();
        self.filter_out = None;
        self.tracker_out = None;
        self.box_out = None;
        self.particle_out = None;
        self.filter_view.clear_cache();
        self.box_view.clear_cache();
        self.particles_view.clear_cache();
        self.view = CentralView::Scene;
    }

    /// Recompute the estimator configuration's defaults from the current
    /// run, with the model's hints applied.
    fn default_filter_cfg(&self) -> Option<FilterConfig> {
        let traj = self.base.completed()?;
        let mut cfg = FilterConfig::defaults(
            self.base.model.state_labels(),
            self.base.model.periodic(),
            &self.base.model.default_pairs(),
            &traj.states,
        );
        apply_hints(&mut cfg, &self.base.model.estimator_hints());
        Some(cfg)
    }
}

/// The observers application over a [`Compute`] backend. Call
/// [`ui`](Self::ui) once per frame from the eframe `App::ui`.
pub struct App<C: Compute> {
    slots: Vec<Slot>,
    selected: usize,
    compute: C,
}

impl<C: Compute> App<C> {
    /// Every model of the menu at its initial condition, jobs on `compute`.
    pub fn new(compute: C) -> Self {
        App {
            slots: models::all().into_iter().map(Slot::new).collect(),
            selected: 0,
            compute,
        }
    }

    /// One frame: the three panels.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // Collect finished background runs (all slots, not just the visible
        // one) and start their playback.
        for slot in &mut self.slots {
            if slot.base.poll() {
                // First run only — later runs keep the user's tuned
                // configuration (the `default` button recomputes it) —
                // unless the run's dimension changed (an estimator option
                // augmented the state).
                let dim = slot.base.model.state_labels().len();
                if slot.filter_cfg.as_ref().is_none_or(|c| c.vars.len() != dim) {
                    slot.filter_cfg = slot.default_filter_cfg();
                }
            }
            if let Some(out) = collect(&mut slot.filter_run, &mut slot.run_error) {
                slot.filter_view.reset(&out);
                slot.view = CentralView::Filter;
                slot.filter_out = Some(out);
                slot.base.playback.restart();
            }
            if let Some(out) = collect(&mut slot.tracker_run, &mut slot.run_error) {
                slot.view = CentralView::Tracker;
                slot.filter_view.clear_tracker_cache();
                slot.tracker_out = Some(out);
                slot.base.playback.restart();
            }
            if let Some(out) = collect(&mut slot.box_run, &mut slot.run_error) {
                slot.box_view.reset(&out);
                slot.view = CentralView::Window;
                slot.box_out = Some(out);
                slot.base.playback.restart();
            }
            if let Some(out) = collect(&mut slot.particle_run, &mut slot.run_error) {
                slot.particles_view.reset(&out);
                slot.view = CentralView::Particles;
                slot.particle_out = Some(out);
                slot.base.playback.restart();
            }
        }

        egui::Panel::left("params")
            .resizable(true)
            .default_size(280.0)
            .min_size(250.0)
            .show(ui, |ui| {
                ui.add_space(4.0);
                ui.heading("Model");
                // While an estimator runs, the whole panel is frozen
                // (grayed): its configuration must stay exactly what the
                // run uses.
                let estimating = self.slots[self.selected].estimating();
                ui.add_enabled_ui(!estimating, |ui| {
                    let names: Vec<&str> = self.slots.iter().map(|s| s.base.model.name()).collect();
                    model_panels::model_selector(ui, &names, &mut self.selected);
                    let compute = &mut self.compute;
                    let slot = &mut self.slots[self.selected];
                    ui.separator();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        if model_panels::params_ui(ui, &mut slot.base) {
                            slot.clear_estimator_state();
                        }
                        ui.add_space(6.0);
                        let estimating = slot.estimating();
                        model_panels::run_ui(ui, &mut slot.base, !estimating);
                        let run_done = slot.base.run_done();

                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Mortensen estimator");
                            if ui
                                .add_enabled(run_done, egui::Button::new("default").small())
                                .clicked()
                            {
                                slot.filter_cfg = slot.default_filter_cfg();
                            }
                        });
                        compute.ui(ui);
                        // Estimator-only options of the model (an unknown
                        // parameter augmenting the state): the run is made
                        // of a different model, so it and the configuration
                        // (whose dimension may change) are discarded.
                        if slot.base.model.estimator_options_ui(ui) {
                            slot.base.refresh_preview();
                            slot.clear_estimator_state();
                            slot.filter_cfg = None;
                        }
                        panels::estimator_selector(ui, &mut slot.estimator);
                        let is_tracker = slot.estimator == Estimator::Tracker;
                        let is_window = slot.estimator == Estimator::Window;
                        let obs_ok = slot.base.model.obs_selected().iter().any(|&s| s);
                        let steps = slot.base.steps();
                        let mut domains_ok = true;
                        match &mut slot.filter_cfg {
                            None => {
                                ui.weak(
                                    "Run a simulation first — domains and defaults \
                                     are computed from the run.",
                                );
                            }
                            Some(cfg) => {
                                domains_ok = panels::config_ui(ui, cfg, slot.estimator, steps);
                            }
                        }

                        ui.add_space(6.0);
                        let prior_ok = slot
                            .filter_cfg
                            .as_ref()
                            .is_none_or(|c| panels::prior_ok(c, slot.estimator));
                        let can_run = run_done
                            && !estimating
                            && slot.filter_cfg.is_some()
                            && obs_ok
                            && prior_ok
                            && (domains_ok || is_tracker || is_window);
                        let clicked = ui
                            .add_enabled(
                                can_run,
                                egui::Button::new(panels::run_label(slot.estimator))
                                    .min_size(egui::vec2(ui.available_width() - 8.0, 28.0)),
                            )
                            .on_hover_text("Run the Mortensen estimator on this trajectory")
                            .on_disabled_hover_text(if !obs_ok {
                                "Select at least one observation first"
                            } else if !domains_ok {
                                "Each domain needs min < max"
                            } else if !prior_ok {
                                "The window needs σ > 0 in every direction (a mode to follow)"
                            } else {
                                "Run a simulation first"
                            })
                            .clicked();
                        if clicked
                            && let Some(cfg) = &slot.filter_cfg
                        {
                            let job = JobSpec {
                                model: slot.base.model.model_spec(),
                                estimator: slot.estimator,
                                config: cfg.clone(),
                                dt: slot.base.dt,
                                steps,
                            };
                            slot.run_error = None;
                            match slot.estimator {
                                Estimator::Filter => {
                                    slot.filter_run =
                                        Some(compute.spawn(job, jobs::run_filter_spec));
                                }
                                Estimator::Window => {
                                    slot.box_run =
                                        Some(compute.spawn(job, jobs::run_box_spec));
                                }
                                Estimator::Particles => {
                                    slot.particle_run =
                                        Some(compute.spawn(job, jobs::run_particles_spec));
                                }
                                Estimator::Tracker => {
                                    slot.tracker_run =
                                        Some(compute.spawn(job, jobs::run_tracker_spec));
                                }
                            }
                        }
                        if let Some(out) = &slot.filter_out {
                            ui.small(format!(
                                "In memory: {} snapshots × {} plane(s)",
                                out.saved_steps.len(),
                                out.pairs.len(),
                            ));
                        }
                    });
                }); // add_enabled_ui

                // Progress of a running estimator — outside the frozen
                // region, so it renders at full strength.
                let sel = &self.slots[self.selected];
                let running: Option<(f32, f64, &str)> =
                    match (&sel.filter_run, &sel.tracker_run, &sel.box_run, &sel.particle_run) {
                        (Some(f), _, _, _) => {
                            Some((f.progress(), f.elapsed(), panels::running_text(Estimator::Filter)))
                        }
                        (_, Some(t), _, _) => {
                            Some((t.progress(), t.elapsed(), panels::running_text(Estimator::Tracker)))
                        }
                        (_, _, Some(b), _) => {
                            Some((b.progress(), b.elapsed(), panels::running_text(Estimator::Window)))
                        }
                        (_, _, _, Some(p)) => {
                            Some((p.progress(), p.elapsed(), panels::running_text(Estimator::Particles)))
                        }
                        _ => None,
                    };
                if let Some((progress, elapsed, text)) = running {
                    ui.add_space(4.0);
                    if panels::progress_ui(ui, progress, elapsed, text) {
                        // Dropping the handles raises the cancel flag (or
                        // deletes the run on the server): the worker stops
                        // after its current iteration and its partial
                        // result is discarded.
                        self.slots[self.selected].cancel_estimators();
                    }
                }
                if let Some(e) = &self.slots[self.selected].run_error {
                    ui.add_space(4.0);
                    ui.colored_label(ui.visuals().error_fg_color, format!("Estimator run failed: {e}"));
                }
                // Pin the content width to the panel width (same creep
                // prevention as the bottom panel's take_available_height).
                ui.take_available_width();
            });

        egui::Panel::bottom("obs_panel")
            .resizable(true)
            .default_size(230.0)
            .min_size(170.0)
            .show(ui, |ui| {
                model_panels::observation_panel(ui, &mut self.slots[self.selected].base);
            });

        egui::CentralPanel::default().show(ui, |ui| {
            let slot = &mut self.slots[self.selected];
            // Advance the clock with wall time while playing.
            let wall_dt = ui.input(|i| i.stable_dt) as f64;
            slot.base.advance(wall_dt);

            // View tabs, shown once an estimator output exists.
            if slot.filter_out.is_some()
                || slot.tracker_out.is_some()
                || slot.box_out.is_some()
                || slot.particle_out.is_some()
            {
                let n_pairs = slot.filter_out.as_ref().map_or(0, |o| o.pairs.len());
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut slot.view, CentralView::Scene, "Scene");
                    if slot.filter_out.is_some() {
                        ui.selectable_value(&mut slot.view, CentralView::Filter, "Filter density");
                    }
                    if slot.box_out.is_some() {
                        ui.selectable_value(&mut slot.view, CentralView::Window, "Tracking density");
                    }
                    if slot.particle_out.is_some() {
                        ui.selectable_value(&mut slot.view, CentralView::Particles, "LParticles");
                    }
                    if slot.tracker_out.is_some() {
                        ui.selectable_value(&mut slot.view, CentralView::Tracker, "Tracker");
                    }
                    if slot.view == CentralView::Filter && n_pairs > 1 {
                        ui.separator();
                        ui.selectable_value(&mut slot.filter_view.n_panes, 1, "1 panel");
                        ui.selectable_value(&mut slot.filter_view.n_panes, 2, "2 panels");
                    }
                });
                ui.separator();
            }

            let t = slot.base.playback.t;
            if slot.view == CentralView::Filter
                && let Some(out) = &slot.filter_out
            {
                let step = (t / out.dt).round().max(0.0) as usize;
                slot.filter_view.ui(ui, out, slot.tracker_out.as_ref(), step);
                return;
            }
            if slot.view == CentralView::Window
                && let Some(out) = &slot.box_out
            {
                let step = (t / out.window.dt).round().max(0.0) as usize;
                slot.box_view.ui(ui, out, slot.tracker_out.as_ref(), step);
                return;
            }
            if slot.view == CentralView::Particles
                && let Some(out) = &slot.particle_out
            {
                let step = (t / out.dt).round().max(0.0) as usize;
                slot.particles_view.ui(ui, out, slot.tracker_out.as_ref(), step);
                return;
            }
            if slot.view == CentralView::Tracker
                && let Some(out) = &slot.tracker_out
            {
                tracker_view::ui(ui, out, t);
                return;
            }

            model_panels::scene_ui(ui, &slot.base, self.selected);
        });

        // Keep animating while something moves or computes.
        let busy = self.slots[self.selected].base.playback.playing
            || self.slots.iter().any(|s| s.base.run.is_some() || s.estimating());
        if busy {
            ui.ctx().request_repaint();
        }
    }
}

/// Apply a model's estimator hints onto the freshly computed defaults:
/// every `Some` field of a [`VarHint`](ode_models_egui::VarHint) replaces
/// the default of that variable, and `eps` replaces the temperature.
/// Hints for more variables than the configuration has are ignored.
fn apply_hints(cfg: &mut FilterConfig, hints: &EstimatorHints) {
    for (v, h) in cfg.vars.iter_mut().zip(&hints.vars) {
        if let Some(d) = h.domain {
            v.domain = d;
        }
        if let Some(n) = h.n_el {
            v.n_el = n;
        }
        if let Some(p) = h.p_ord {
            v.p_ord = p;
        }
        if let Some(q) = h.q {
            v.q = q;
        }
        if let Some(x0) = h.x0 {
            v.x0 = x0;
        }
    }
    if let Some(eps) = hints.eps {
        cfg.eps = eps;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ode_models_egui::models::kepler::KeplerViz;
    use ode_models_egui::playback::Progress;
    use ode_models_egui::models::spring::SpringViz;

    /// The default configuration of a viewer entry: the generic defaults
    /// from a fake completed run at the model's initial state, with the
    /// model's hints applied.
    fn default_cfg(viz: &dyn VizModel, x0: &[f64]) -> FilterConfig {
        let states = vec![x0.to_vec(); 11];
        let mut cfg = FilterConfig::defaults(
            viz.state_labels(),
            viz.periodic(),
            &viz.default_pairs(),
            &states,
        );
        apply_hints(&mut cfg, &viz.estimator_hints());
        cfg
    }

    /// The initial state of a model entry, through its own model spec.
    fn initial_state(viz: &dyn VizModel) -> Vec<f64> {
        struct X0;
        impl ode_models::spec::ModelVisitor for X0 {
            type Output = Vec<f64>;
            fn visit<const M: usize, Mod: ode_models::model::Model<M>>(self, model: Mod) -> Vec<f64> {
                model.states()[0].as_slice().to_vec()
            }
        }
        viz.model_spec().visit(0.02, X0)
    }

    /// A spring entry with the estimator-only "unknown free-end mass"
    /// option on, snapshotted (a zero-step run) so its spec is augmented.
    fn spring_unknown_mass(n: usize) -> Box<dyn VizModel> {
        let mut viz = SpringViz::new(n, "spring");
        viz.set_unknown_mass(true);
        let _ = viz.make_job(0.01, 0)(&Progress::default());
        Box::new(viz)
    }

    /// The Kepler entry with the estimator-only "unknown μ" option on.
    fn kepler_unknown_mu() -> Box<dyn VizModel> {
        let mut viz = KeplerViz::default();
        viz.set_unknown_mu(true);
        let _ = viz.make_job(0.01, 0)(&Progress::default());
        Box::new(viz)
    }

    /// The hints of the twin-experiment entries reach the configuration
    /// (fixed cheap θ direction, q_θ = 0, prior at θ = 0, (·, θ) plane
    /// pre-selected) and a small filter run through the whole pipeline —
    /// `VizModel::model_spec` → `JobSpec` → `run_filter_spec` — stays
    /// finite: Kepler with the unknown-μ option (5D) and the spring chains
    /// with the unknown-mass option (3D, 5D).
    #[test]
    fn hinted_defaults_and_filter_runs() {
        let cases: Vec<(Box<dyn VizModel>, usize)> =
            vec![(kepler_unknown_mu(), 4), (spring_unknown_mass(1), 2), (spring_unknown_mass(2), 4)];
        for (viz, theta) in cases {
            let x0 = initial_state(viz.as_ref());
            assert_eq!(x0.len(), theta + 1, "{}", viz.name());
            let mut cfg = default_cfg(viz.as_ref(), &x0);
            let th = &cfg.vars[theta];
            assert_eq!(th.label, "θ");
            assert_eq!(th.domain, (-1.0, 1.0), "{}", viz.name());
            assert_eq!(th.q, 0.0, "{}: dθ/dt = 0 requires q_θ = 0", viz.name());
            assert_eq!(th.x0, 0.0, "{}: the prior must not be centered on θ_true", viz.name());
            assert!(th.n_el * th.p_ord <= 8, "{}: the θ direction should be cheap", viz.name());
            assert!(cfg.selected_pairs().contains(&(0, theta)), "{}: (·, θ) plane off", viz.name());

            for v in &mut cfg.vars[..theta] {
                v.n_el = 2;
                v.p_ord = 2;
                v.sigma = 1.0;
            }
            cfg.n_snapshots = 2;
            let job = JobSpec {
                model: viz.model_spec(),
                estimator: Estimator::Filter,
                config: cfg,
                dt: 0.02,
                steps: 5,
            };
            let progress = Progress::default();
            let out = match job.run(&progress) {
                jobs::JobOutput::Filter(out) => out,
                _ => unreachable!(),
            };
            assert_eq!(progress.done(), 5);
            let last = out.marginals.last().unwrap();
            assert!(last.iter().flatten().all(|v| v.is_finite()));
            assert!(last.iter().flatten().any(|&v| v > 0.0), "{}: p degenerated", viz.name());
        }
    }

    /// The same spring entry without the option is the plain 2N-dimensional
    /// chain, with no hint applied.
    #[test]
    fn spring_without_the_option_is_the_plain_chain() {
        let viz = models::make(1);
        assert_eq!(viz.name(), "Two springs");
        let x0 = initial_state(viz.as_ref());
        assert_eq!(x0.len(), 4);
        let cfg = default_cfg(viz.as_ref(), &x0);
        assert_eq!(cfg.vars.len(), 4);
        assert!(cfg.vars.iter().all(|v| v.q == 1.0 && v.n_el == 5));
    }

    /// The random walk's hints cover the ring, and a short filter run
    /// through the pipeline produces it: high p at the four axis points of
    /// the unit circle, low at the origin.
    #[test]
    fn hinted_defaults_and_ring_through_the_viewer() {
        let viz = models::make(7);
        assert_eq!(viz.name(), "Random walk");
        let x0 = initial_state(viz.as_ref());
        let mut cfg = default_cfg(viz.as_ref(), &x0);
        assert!(cfg.vars.iter().all(|v| v.domain == (-2.0, 2.0)));
        cfg.n_snapshots = 2;
        let out = jobs::run_filter_spec(&viz.model_spec(), &cfg, 0.05, 40, &Progress::default());
        let marg = &out.marginals.last().unwrap()[0];
        let (ax, ay) = (&out.axes[0], &out.axes[1]);
        let max = marg.iter().copied().fold(f64::MIN, f64::max);
        let at = |x: f64, y: f64| {
            let i = ax.iter().enumerate().min_by(|a, b| (a.1 - x).abs().total_cmp(&(b.1 - x).abs())).unwrap().0;
            let j = ay.iter().enumerate().min_by(|a, b| (a.1 - y).abs().total_cmp(&(b.1 - y).abs())).unwrap().0;
            marg[i + j * ax.len()] / max
        };
        for (x, y) in [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)] {
            assert!(at(x, y) > 0.5, "no ring at ({x}, {y}): {}", at(x, y));
        }
        assert!(at(0.0, 0.0) < 0.05, "p not low at the origin: {}", at(0.0, 0.0));
    }
}
