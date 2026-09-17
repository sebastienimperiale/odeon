//! The viewer-side model contract ([`VizModel`]) and its implementations —
//! one per forward model, each owning its parameter form and its scene
//! drawing. This is deliberately separate from the filter's `Model<M>` trait:
//! the viewer needs a dimension-erased, drawable interface, not the filter's
//! compile-time grid contract.

mod lamppost;
pub mod kepler;
mod lorenz;
mod pendulum;
mod pendulum_rod;
pub mod spring;

use crate::playback::{Job, Trajectory};
use ode_models_spec::spec::{ModelSpec, TwinSpec};

/// One selectable model in the viewer.
///
/// Observations are vector-valued (`obs_labels` names the components) so
/// models with several y_i(t) plot naturally — and so the interface already
/// matches the roadmap's decoupled-observation refactor.
pub trait VizModel {
    fn name(&self) -> &'static str;

    /// Named destination of the model's section in the models note
    /// (`crates/ode-models/docs/models.tex`, `\modelsection{<name>}{…}`),
    /// which the "About this model…" button opens in the browser as the
    /// local `models.pdf` at that section's page. A model with an
    /// estimator option augmenting its state points at the section of the
    /// augmented model while the option is on. `None` = no section.
    fn doc_dest(&self) -> Option<&'static str> {
        None
    }

    /// The model-specific parameter form (physical parameters, initial
    /// condition, defaults buttons). Returns `true` when any value was
    /// edited this frame, so the app can refresh the scene preview. The
    /// observation choice is *not* part of this: the app renders a uniform
    /// Observation section from the `obs_*` methods below, so the section
    /// exists for every model.
    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool;

    /// Options that only make sense when the model is run for the
    /// estimators — e.g. declaring a physical parameter unknown, which
    /// augments the state with its log-value θ and changes the dimension
    /// of the run. Drawn by the estimator viewer in its estimator section,
    /// never by a models-only viewer. Returns `true` when edited: the model
    /// the run is made of changed, so the caller discards the run (and its
    /// estimator configuration, whose dimension may have changed) and
    /// re-arms Run. The default draws nothing.
    fn estimator_options_ui(&mut self, _ui: &mut egui::Ui) -> bool {
        false
    }

    /// The observation components on offer, one entry per choice. Always
    /// non-empty — models with a single hardcoded observation return one
    /// entry.
    fn obs_options(&self) -> Vec<String>;

    /// Whether the choices are mutually exclusive (radio selection). The
    /// filter consumes a single scalar observation, so models with several
    /// options keep exactly one selected — showing several while filtering
    /// on one would misrepresent what the filter sees.
    fn obs_exclusive(&self) -> bool;

    /// Which `obs_options` entries are currently selected (same length).
    fn obs_selected(&self) -> Vec<bool>;

    /// Toggle the selection of `obs_options` entry `idx`.
    fn toggle_obs(&mut self, idx: usize);

    /// Noise-model row of observation `idx` (selector + parameters), shown
    /// in the Observation section under the selected entry. Returns `true`
    /// when edited. With noise enabled, the observation series of the run
    /// — plotted, and consumed by the estimators — is the noisy one.
    fn noise_ui(&mut self, ui: &mut egui::Ui, idx: usize) -> bool;

    /// Snapshot the current form as the *drawn* parameters: the scene, the
    /// model description and the twin experiment are read from the
    /// snapshot, so editing the form after a run cannot desynchronize the
    /// scene from the trajectory being replayed.
    fn snapshot(&mut self);

    /// The twin experiment of the drawn parameters: the model, its initial
    /// state, the observation noise with its seed, an optional walk of the
    /// reference — everything the reference generator needs for `steps`
    /// steps of `dt`.
    fn twin_spec(&self, dt: f64, steps: usize) -> TwinSpec;

    /// Labels of the plotted observation components of the drawn
    /// parameters ("(noisy)" appended when a noise model is on); empty
    /// when the model's observation is switched off in the form, in which
    /// case the trajectory carries no observation to plot.
    fn obs_labels(&self) -> Vec<String>;

    /// Snapshot the current form and build the trajectory computation: the
    /// reference of [`twin_spec`](Self::twin_spec) — states and
    /// observations from `ode_models_spec::reference::Reference::twin` — with
    /// the observation labels; the job runs on a worker thread and bumps
    /// the counter once per step.
    fn make_job(&mut self, dt: f64, steps: usize) -> Job {
        self.snapshot();
        let twin = self.twin_spec(dt, steps);
        let obs_labels = self.obs_labels();
        Box::new(move |progress| {
            let r = twin.reference(progress);
            let observations = if obs_labels.is_empty() { vec![Vec::new(); r.states.len()] } else { r.observations };
            Trajectory { dt, states: r.states, observations, obs_labels }
        })
    }

    /// Flat state-component labels, in the model's filter order (matches
    /// `Trajectory.states` components).
    fn state_labels(&self) -> Vec<String>;

    /// Per-component periodicity: `Some((lo, hi))` for a component defined
    /// modulo hi − lo (the model's `Model::periodic`), which gets that
    /// interval as filter domain and periodic boundary conditions.
    fn periodic(&self) -> Vec<Option<(f64, f64)>>;

    /// The model's natural 2D output planes (a, b), pre-selected in the
    /// filter configuration (e.g. (qᵢ, pᵢ)).
    fn default_pairs(&self) -> Vec<(usize, usize)>;

    /// Model-specific overrides of the estimator defaults a front-end
    /// computes from a run (domains from the state ranges, resolution from
    /// the dimension, Gaussian centre at the initial state). The default
    /// implementation overrides nothing. Kepler-μ and the unknown-mass
    /// chain use it for their twin experiment: the generic defaults would
    /// center the θ domain and the prior on the run's constant θ_true —
    /// handing the estimator the answer — so they fix the θ domain, make
    /// the direction cheap, set q_θ = 0 and center the prior at θ = 0. The
    /// lamppost widens the box to the whole ring. Applied by the app
    /// onto its configuration; this crate knows no observer.
    fn estimator_hints(&self) -> EstimatorHints {
        EstimatorHints::default()
    }

    /// The drawn model as a plain-data description (parameters and
    /// observation choice; the model of [`twin_spec`](Self::twin_spec)).
    /// The app pairs it with the run's trajectory, the estimator choice and
    /// the configuration into a `JobSpec` (`ode_observers::jobs`), which
    /// builds the concrete model at its compile-time dimension: no model
    /// file names an observer, and the same description can be sent to a
    /// compute server.
    fn model_spec(&self) -> ModelSpec;

    /// Draw frame `frame` of `traj` into `rect`. Styling comes from
    /// `visuals` (theme neutrals) and [`crate::palette`] (observation
    /// accents).
    fn draw(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        traj: &Trajectory,
        frame: usize,
        visuals: &egui::Visuals,
    );
}

/// Per-variable overrides a model may request for the estimator defaults
/// (see [`VizModel::estimator_hints`]); `None` keeps the computed default.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VarHint {
    /// Domain interval of the grid filter in this direction.
    pub domain: Option<(f64, f64)>,
    /// Elements per direction.
    pub n_el: Option<usize>,
    /// Polynomial order per direction.
    pub p_ord: Option<usize>,
    /// Model-noise variance q_d (0 = no diffusion in this direction).
    pub q: Option<f64>,
    /// Centre x_{c,d} of the Gaussian initial data.
    pub x0: Option<f64>,
}

/// Overrides of the estimator defaults requested by a model: one
/// [`VarHint`] per state component (an empty `vars` overrides nothing) and
/// optionally the temperature ε.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EstimatorHints {
    pub vars: Vec<VarHint>,
    pub eps: Option<f64>,
}

impl EstimatorHints {
    /// No override in any of `dim` directions — a starting point to set a
    /// few fields on.
    pub fn for_dim(dim: usize) -> Self {
        EstimatorHints { vars: vec![VarHint::default(); dim], eps: None }
    }
}

/// Number of models in the menu.
pub const COUNT: usize = 8;

/// Fresh model `idx` with default parameters — also what the Reset button
/// and a model switch fall back to.
pub fn make(idx: usize) -> Box<dyn VizModel> {
    match idx {
        0 => Box::new(spring::SpringViz::new(1, "Single spring")),
        1 => Box::new(spring::SpringViz::new(2, "Two springs")),
        2 => Box::new(spring::SpringViz::new(3, "Three springs")),
        3 => Box::new(pendulum::PendulumViz::default()),
        4 => Box::new(pendulum_rod::PendulumRodViz::default()),
        5 => Box::new(kepler::KeplerViz::default()),
        6 => Box::new(lorenz::LorenzViz::default()),
        _ => Box::new(lamppost::LamppostViz::default()),
    }
}

/// All models, in menu order.
pub fn all() -> Vec<Box<dyn VizModel>> {
    (0..COUNT).map(make).collect()
}

/// Orbit trace: the full history faint, the recent tail stronger — the eye
/// follows the moving end without losing the overall orbit shape.
pub fn draw_trace(painter: &egui::Painter, pts: &[egui::Pos2], color: egui::Color32) {
    if pts.len() < 2 {
        return;
    }
    painter.add(egui::Shape::line(
        pts.to_vec(),
        egui::Stroke::new(1.0, color.gamma_multiply(0.25)),
    ));
    let tail = pts.len().saturating_sub(150);
    if pts.len() - tail >= 2 {
        painter.add(egui::Shape::line(
            pts[tail..].to_vec(),
            egui::Stroke::new(2.0, color.gamma_multiply(0.9)),
        ));
    }
}

/// Aspect-preserving map from world coordinates (y pointing up) to screen
/// points inside a rect, with a small margin.
pub struct SceneMap {
    scale: f32,
    center: egui::Pos2,
    world_center: (f64, f64),
}

impl SceneMap {
    pub fn fit(rect: egui::Rect, x_range: (f64, f64), y_range: (f64, f64)) -> Self {
        let w = (x_range.1 - x_range.0).max(1e-9) as f32;
        let h = (y_range.1 - y_range.0).max(1e-9) as f32;
        let scale = (rect.width() / w).min(rect.height() / h) * 0.85;
        SceneMap {
            scale,
            center: rect.center(),
            world_center: (
                0.5 * (x_range.0 + x_range.1),
                0.5 * (y_range.0 + y_range.1),
            ),
        }
    }

    pub fn pt(&self, x: f64, y: f64) -> egui::Pos2 {
        egui::pos2(
            self.center.x + self.scale * (x - self.world_center.0) as f32,
            self.center.y - self.scale * (y - self.world_center.1) as f32,
        )
    }

    /// A world length in pixels.
    pub fn len(&self, d: f64) -> f32 {
        self.scale * d as f32
    }
}
