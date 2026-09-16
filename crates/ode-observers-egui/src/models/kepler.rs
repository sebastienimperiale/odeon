//! Softened Kepler scene: the central mass at the origin (its softening
//! length drawn as a faint disc), the orbiting body with its trace, and the
//! observed quantity drawn in the observation accent — a projection line to
//! the q₁ axis for q₁, a circle of radius |q| for the range, a ray from the
//! center for the bearing.
//!
//! **Unknown μ** — an option of the estimator viewer only
//! ([`VizModel::estimator_options_ui`]; a models-only viewer never shows
//! it): the same orbit driven by `KeplerMuSystem` instead of
//! `KeplerSystem`. Its augmented state (q₁, q₂, p₁, p₂, θ) carries the
//! log-parameter θ with μ = μ₀·2^θ and θ̇ = 0, where μ₀ is the form's μ
//! (the estimators' prior: θ = 0 ⇔ μ = μ₀) and the reference runs at
//! θ_true. Same potential, same Gauss–Legendre 4 stepper (the θ stage
//! equation decouples), so at θ_true = 0 the two runs coincide (test
//! `augmented_run_is_the_same_orbit`). The whole run — scene, observation
//! plot and estimators — uses the augmented model, so the estimators' twin
//! reference is exactly the trajectory on screen; toggling the option
//! therefore requires a new run. The twin experiment: the θ direction gets
//! a fixed (−1, 1) domain, cheap resolution, q_θ = 0 and a Gaussian center
//! at θ = 0 (the prior μ₀, *not* the true value), with the (q₁, θ) plane
//! among the default outputs. μ is identifiable from position observations
//! (it sets the period — Kepler's third law).

use super::{EstimatorHints, SceneMap, VarHint, VizModel, draw_trace};
use crate::noise::NoiseModel;
use crate::palette;
use crate::playback::Trajectory;
use ode_models::models::kepler::{KeplerObservation, KeplerParams, perihelion_state};
use ode_models::models::kepler_mu::{KeplerMuParams, augmented_state};
use ode_models_spec::spec::{ModelSpec, TwinSpec};

/// Eccentricity of the default orbit.
const DEFAULT_E: f64 = 0.5;
/// Half-width of the scene window (world units, centered on the origin);
/// the initial data are bounded to stay inside it.
const WINDOW: f64 = 2.0;
/// Bound on each initial component (|qᵢ|, |pᵢ| ≤ IC_BOUND).
const IC_BOUND: f64 = 2.0;
/// Default true log-parameter of the reference when μ is unknown:
/// μ = 2^0.4·μ₀ ≈ 1.32·μ₀. Deliberately *large*: at this θ the q₁
/// likelihood is phase-aliased over a few periods (multimodal in θ) and the
/// tracker provably falls into a secondary basin — the regime the grid
/// filter is for.
const DEFAULT_THETA: f64 = 0.4;
/// Fixed filter domain of the θ direction (the parameterization needs no
/// positivity constraint: μ = μ₀·2^θ > 0 for every θ).
const THETA_DOMAIN: (f64, f64) = (-1.0, 1.0);

#[derive(Clone, PartialEq)]
struct Params {
    phys: KeplerParams,
    x0: [f64; 4],
    /// Selected observation, index into [`KeplerObservation::ALL`] (exactly
    /// one: the filter consumes a single scalar observation).
    obs_idx: usize,
    /// Noise model of each observation choice (same indexing).
    obs_noise: [NoiseModel; 3],
    /// Estimator-only option: μ is unknown — the run uses the augmented
    /// model (state (q, p, θ)), with the form's μ as the prior μ₀.
    unknown_mu: bool,
    /// True log-parameter θ of the reference when `unknown_mu`
    /// (μ_true = μ₀·2^θ).
    theta: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            phys: KeplerParams::default(),
            x0: perihelion_state(DEFAULT_E),
            obs_idx: 0,
            obs_noise: [NoiseModel::None; 3],
            unknown_mu: false,
            theta: DEFAULT_THETA,
        }
    }
}

impl Params {
    fn observation(&self) -> KeplerObservation {
        KeplerObservation::ALL[self.obs_idx]
    }

    /// State dimension of the run: 4, or 5 with the unknown μ.
    fn dim(&self) -> usize {
        4 + usize::from(self.unknown_mu)
    }

    /// Seed of the observation noise of the twin experiment.
    fn noise_seed(&self) -> u64 {
        4000 + self.obs_idx as u64
    }

    /// Parameters of the augmented model: the form's μ as the prior μ₀.
    fn mu_params(&self) -> KeplerMuParams {
        KeplerMuParams { mu0: self.phys.mu, softening: self.phys.softening }
    }
}

#[derive(Default)]
pub struct KeplerViz {
    params: Params,
    /// Snapshot taken by `make_job`; the scene is drawn with these, so form
    /// edits after a run don't desynchronize scene and trajectory.
    drawn: Params,
}

impl KeplerViz {
    /// Set the estimator-only "unknown μ" option programmatically (what the
    /// checkbox of [`estimator_options_ui`](VizModel::estimator_options_ui)
    /// does). Takes effect at the next `make_job`.
    pub fn set_unknown_mu(&mut self, on: bool) {
        self.params.unknown_mu = on;
    }
}

impl VizModel for KeplerViz {
    fn name(&self) -> &'static str {
        "Kepler orbit (softened)"
    }

    fn description(&self) -> String {
        "Planar Kepler problem: a body of unit mass orbiting a fixed center at the origin \
         in the softened gravitational potential −μ/√(|q|² + a²). With a = 0 this is \
         exactly Newtonian gravity (closed ellipses); the softening a bounds the force at \
         the center so that the flow is well defined everywhere in the filter's state box \
         (which contains the origin) — it makes the orbit precess slightly. Conservative: \
         energy and angular momentum are conserved.\n\n\
         Parameters: μ = GM — gravitational parameter (at μ = 1 the circular orbit of \
         radius 1 has speed 1 and period 2π); a — softening length.\n\n\
         State (filter order): q₁, q₂ — Cartesian position; p₁, p₂ — momentum = velocity \
         (unit mass). The default is the perihelion of the e = 0.5 ellipse of semi-major \
         axis 1: q = (0.5, 0), p = (0, √3).\n\n\
         Scene: fixed window [−2, 2]², the center (its softening disc in grey), the orbit \
         trace, the body with a unit arrow in the direction of its momentum, and the \
         observed quantity in the accent: the projection onto the q₁ axis, the range circle, \
         or the bearing ray.\n\n\
         Observation (one of): q₁ — the line-of-sight coordinate (linear); |q| — the range \
         (the angle is unobserved: expect ring-shaped densities in the (q₁, q₂) plane); \
         atan2(q₂, q₁) — the bearing (the range is unobserved; differences are taken mod 2π).\n\n\
         Unknown μ (option of the estimator viewer): the same orbit with the gravitational \
         parameter unknown, written μ = μ₀·2^θ with μ₀ the μ above (the estimators' prior) \
         and θ an extra state variable obeying dθ/dt = 0 (μ > 0 for every real θ; a Gaussian \
         prior in θ is a log-normal prior in μ; θ reads in doublings of μ₀). The reference \
         runs at θ_true and the estimators, started at θ = 0, must recover it — μ is \
         identifiable from position observations, since it sets the orbital period (Kepler's \
         third law); the satellite's own mass would not be. The run then uses the augmented \
         state (q₁, q₂, p₁, p₂, θ) on a 5D grid whose θ direction is cheap by default (fixed \
         domain (−1, 1), few DOFs, q_θ = 0 so nothing diffuses in θ): each θ-slice runs its \
         own Kepler flow and the observation step discriminates the slices; watch the \
         (q₁, θ) plane sharpen around θ_true. The default θ_true = 0.4 is deliberately \
         large: over a few periods the q₁ likelihood is phase-aliased (multimodal in θ) and \
         the tracker — a Gaussian closure — falls into a secondary basin; the grid filter \
         can carry the multimodal θ-density until the phase disambiguates. For a regime \
         where the tracker succeeds, set θ_true ≈ 0.1 and stiffen the (q, p) prior. θ is \
         not drawn in the scene; read it in the state plots and the θ marginals."
            .to_string()
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let before = self.params.clone();
        let p = &mut self.params;
        ui.horizontal(|ui| {
            ui.label("Physical parameters");
            if ui.small_button("default").clicked() {
                p.phys = KeplerParams::default();
            }
        });
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut p.phys.mu).speed(0.02).range(0.05..=10.0));
            ui.label("μ = GM").on_hover_text(
                "Gravitational parameter of the central mass (G·M, unit orbiting mass). \
                 Sets the strength of the attraction: at μ = 1 a circular orbit of \
                 radius 1 has speed 1 and period 2π; the period scales as 1/√μ. With \
                 the unknown-μ option this is the prior μ₀ and the reference runs at \
                 μ₀·2^θ_true.",
            );
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut p.phys.softening)
                    .speed(0.005)
                    .range(0.0..=1.0),
            );
            ui.label("softening a").on_hover_text(
                "Plummer softening length: the potential is −μ/√(|q|² + a²) instead of \
                 −μ/|q|, so the force stays bounded (≤ μ/a²) at the center. Needed by \
                 the filter, whose state box contains the origin. Effect on an orbit \
                 of radius r is O((a/r)²): a slow precession of the ellipse. a = 0 is \
                 exact Kepler — fine for the forward run, but the filter will fail \
                 at grid points near the origin.",
            );
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Initial condition").on_hover_text(
                "Bounded to ±2 per component so the body starts inside the fixed \
                 scene window [−2, 2]² (an orbit leaving it is clipped at the edge). \
                 The default is the perihelion of the e = 0.5 \
                 Kepler ellipse of semi-major axis 1: q = (0.5, 0), p = (0, √3), \
                 period 2π at μ = 1.",
            );
            if ui.small_button("default").clicked() {
                p.x0 = perihelion_state(DEFAULT_E);
            }
        });
        const IC_HELP: [&str; 4] = [
            "Initial position, component 1 (the center is at the origin).",
            "Initial position, component 2.",
            "Initial momentum (= velocity, unit mass), component 1.",
            "Initial momentum, component 2. Bound orbit iff |p|²/2 < μ/|q|.",
        ];
        for (i, label) in ["q₁", "q₂", "p₁", "p₂"].iter().enumerate() {
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut p.x0[i])
                        .speed(0.02)
                        .range(-IC_BOUND..=IC_BOUND),
                );
                ui.label(*label).on_hover_text(IC_HELP[i]);
            });
        }
        before != self.params
    }

    fn estimator_options_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let before = self.params.clone();
        let p = &mut self.params;
        ui.checkbox(&mut p.unknown_mu, "unknown μ").on_hover_text(
            "Run the orbit with the gravitational parameter unknown: the state is \
             augmented with its log-value θ (μ = μ₀·2^θ, μ₀ the μ of the form, \
             dθ/dt = 0), the reference runs at θ_true and the estimators start at \
             θ = 0. Same potential and stepper. Requires a new run.",
        );
        if p.unknown_mu {
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut p.theta)
                        .speed(0.01)
                        .range(THETA_DOMAIN.0..=THETA_DOMAIN.1),
                );
                ui.label("θ_true").on_hover_text(
                    "True log-parameter of the reference: μ_true = μ₀·2^θ_true (θ in \
                     doublings of μ₀, bounded to the filter's fixed θ domain (−1, 1)). \
                     The default 0.4 is the multimodal regime where the tracker fails \
                     and the grid filter is the right tool; θ_true ≈ 0.1 is the local \
                     regime where the tracker also converges.",
                );
            });
        }
        before != self.params
    }

    fn obs_options(&self) -> Vec<String> {
        vec![
            "q₁  (line-of-sight coordinate)".into(),
            "|q|  (range)".into(),
            "atan2(q₂, q₁)  (bearing)".into(),
        ]
    }

    fn obs_exclusive(&self) -> bool {
        true
    }

    fn obs_selected(&self) -> Vec<bool> {
        (0..3).map(|k| k == self.params.obs_idx).collect()
    }

    fn toggle_obs(&mut self, idx: usize) {
        self.params.obs_idx = idx.min(2);
    }

    fn noise_ui(&mut self, ui: &mut egui::Ui, idx: usize) -> bool {
        crate::noise::noise_ui(
            &mut self.params.obs_noise[idx],
            ui,
            &format!("kepler_noise_{idx}"),
        )
    }

    fn state_labels(&self) -> Vec<String> {
        let mut labels = ["q_1", "q_2", "p_1", "p_2"].map(String::from).to_vec();
        if self.params.unknown_mu {
            labels.push("θ".to_string());
        }
        labels
    }

    fn periodic(&self) -> Vec<Option<(f64, f64)>> {
        vec![None; self.params.dim()]
    }

    /// The position and momentum planes; with the unknown μ the (q₁, θ)
    /// plane — where the parameter estimation is watched — comes first.
    fn default_pairs(&self) -> Vec<(usize, usize)> {
        let theta = self.params.unknown_mu.then_some((0, 4));
        theta.into_iter().chain([(0, 1), (2, 3)]).collect()
    }

    /// Twin-experiment defaults for the θ direction when μ is unknown. The
    /// generic defaults would center the θ domain and the Gaussian prior on
    /// the run's (constant) θ_true — i.e. hand the filter the answer.
    /// Instead: fixed domain (−1, 1); cheap resolution (the direction
    /// carries a parameter, not dynamics); q_θ = 0, realizing dθ/dt = 0
    /// exactly; and the Gaussian center at θ = 0, the prior μ₀.
    fn estimator_hints(&self) -> EstimatorHints {
        let mut hints = EstimatorHints::for_dim(self.params.dim());
        if self.params.unknown_mu {
            hints.vars[4] = VarHint {
                domain: Some(THETA_DOMAIN),
                n_el: Some(2),
                p_ord: Some(2),
                q: Some(0.0),
                x0: Some(0.0),
            };
        }
        hints
    }

    fn snapshot(&mut self) {
        self.drawn = self.params.clone();
    }

    fn model_spec(&self) -> ModelSpec {
        let p = &self.drawn;
        if p.unknown_mu {
            ModelSpec::KeplerMu { params: p.mu_params(), observation: p.observation() }
        } else {
            ModelSpec::Kepler { params: p.phys, observation: p.observation() }
        }
    }

    /// The orbit from (q, p), augmented with θ_true when μ is unknown.
    fn twin_spec(&self, dt: f64, steps: usize) -> TwinSpec {
        let p = &self.drawn;
        TwinSpec {
            model: self.model_spec(),
            x0: if p.unknown_mu { augmented_state(p.x0, p.theta).to_vec() } else { p.x0.to_vec() },
            dt,
            steps,
            noise: p.obs_noise[p.obs_idx],
            seed: p.noise_seed(),
            walk: None,
        }
    }

    fn obs_labels(&self) -> Vec<String> {
        let p = &self.drawn;
        let obs = p.observation();
        let noise = p.obs_noise[p.obs_idx];
        vec![if noise.is_none() { obs.label().to_string() } else { format!("{} (noisy)", obs.label()) }]
    }

    fn draw(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        traj: &Trajectory,
        frame: usize,
        visuals: &egui::Visuals,
    ) {
        draw_scene(
            painter,
            rect,
            traj,
            frame,
            visuals,
            self.drawn.phys.softening,
            self.drawn.observation(),
        );
    }
}

/// Kepler scene drawing: the state is read positionally (q in components
/// 0–1, p in 2–3), so the augmented (q, p, θ) state of the unknown-μ
/// option draws the same scene.
fn draw_scene(
    painter: &egui::Painter,
    rect: egui::Rect,
    traj: &Trajectory,
    frame: usize,
    visuals: &egui::Visuals,
    softening: f64,
    obs: KeplerObservation,
) {
    {
        // Fixed window [−WINDOW, WINDOW]² around the center, whatever the
        // data: the initial data are bounded to lie inside it, and an orbit
        // that leaves it is simply clipped at the edge (nothing is ever
        // rescaled, so a change of the initial condition is seen as such).
        let r_max = WINDOW;
        let map = SceneMap::fit(rect, (-r_max, r_max), (-r_max, r_max));
        let painter = painter.with_clip_rect(rect);
        let painter = &painter;

        let neutral = visuals.widgets.noninteractive.fg_stroke.color;
        let accent = palette::series(0, visuals.dark_mode);
        let trace_color = palette::series(2, visuals.dark_mode); // scene decoration

        // Central mass: its softening length as a faint disc, a dot on top.
        let center = map.pt(0.0, 0.0);
        let a = softening;
        if a > 0.0 {
            painter.circle_filled(center, map.len(a), neutral.gamma_multiply(0.15));
        }
        painter.circle_filled(center, 5.0, neutral);

        // Orbit trace up to the current frame.
        let trace: Vec<egui::Pos2> = traj.states[..=frame]
            .iter()
            .map(|s| map.pt(s[0], s[1]))
            .collect();
        draw_trace(painter, &trace, trace_color);

        // The observed quantity, in the observation accent.
        let [q1, q2] = [traj.states[frame][0], traj.states[frame][1]];
        let body = map.pt(q1, q2);
        let thin = egui::Stroke::new(1.5, accent.gamma_multiply(0.7));
        match obs {
            KeplerObservation::Q1 => {
                // Projection onto the q₁ axis (the axis itself, faint).
                painter.line_segment(
                    [map.pt(-r_max, 0.0), map.pt(r_max, 0.0)],
                    egui::Stroke::new(1.0, neutral.gamma_multiply(0.3)),
                );
                painter.line_segment([body, map.pt(q1, 0.0)], thin);
                painter.circle_filled(map.pt(q1, 0.0), 3.0, accent);
            }
            KeplerObservation::Range => {
                painter.circle_stroke(center, map.len(q1.hypot(q2)), thin);
                painter.line_segment([center, body], thin);
            }
            KeplerObservation::Bearing => {
                let r = q1.hypot(q2).max(1e-12);
                let far = map.pt(q1 / r * 1.5 * r_max, q2 / r * 1.5 * r_max);
                painter.line_segment([center, far], thin);
            }
        }

        // Momentum *direction* arrow from the body: unit vector p/|p|, fixed
        // world length 0.3 (the magnitude is not shown; it is readable in
        // the state plots).
        let [p1, p2] = [traj.states[frame][2], traj.states[frame][3]];
        let p_norm = p1.hypot(p2);
        if p_norm > 1e-9 {
            let arrow = egui::vec2(map.len(0.3 * p1 / p_norm), -map.len(0.3 * p2 / p_norm));
            painter.arrow(body, arrow, egui::Stroke::new(1.5, neutral));
        }

        // The body, wearing the accent (it is always the observed element).
        painter.circle_filled(body, 6.0, accent);
        painter.circle_stroke(body, 6.0, egui::Stroke::new(1.0, visuals.window_fill()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playback::Progress;

    /// The option augments the run: dimension 5, θ label, the (q₁, θ)
    /// plane first, the twin-experiment hints on θ; off, everything is the
    /// plain orbit and no hint is set.
    #[test]
    fn unknown_mu_option_augments_the_run() {
        let mut viz = KeplerViz::default();
        assert_eq!(viz.state_labels().len(), 4);
        assert!(viz.estimator_hints().vars.iter().all(|v| *v == VarHint::default()));
        viz.set_unknown_mu(true);
        assert_eq!(viz.state_labels(), ["q_1", "q_2", "p_1", "p_2", "θ"]);
        assert_eq!(viz.default_pairs()[0], (0, 4));
        let hints = viz.estimator_hints();
        assert_eq!(hints.vars.len(), 5);
        let th = hints.vars[4];
        assert_eq!(th.domain, Some(THETA_DOMAIN));
        assert_eq!(th.q, Some(0.0), "dθ/dt = 0 requires q_θ = 0");
        assert_eq!(th.x0, Some(0.0), "the prior must not be centered on θ_true");
        assert!(th.n_el.unwrap() * th.p_ord.unwrap() <= 8, "the θ direction should be cheap");
        let _ = viz.make_job(0.01, 0)(&Progress::default());
        assert_eq!(viz.model_spec().dim(), 5);
        assert_eq!(viz.model_spec().kind(), "kepler_mu");
    }

    /// The augmented run is the same orbit: at θ_true = 0 its (q, p)
    /// trajectory equals the plain one (same potential, same stepper) and
    /// the plotted observation is the same quantity of each.
    #[test]
    fn augmented_run_is_the_same_orbit() {
        let dt = 0.01;
        let steps = 100;
        let mut plain = KeplerViz::default();
        let traj = plain.make_job(dt, steps)(&Progress::default());
        let mut aug = KeplerViz::default();
        aug.set_unknown_mu(true);
        aug.params.theta = 0.0;
        let traj_aug = aug.make_job(dt, steps)(&Progress::default());
        assert_eq!(traj.states[0].len(), 4);
        assert_eq!(traj_aug.states[0].len(), 5);
        for (s, t) in traj.states.iter().zip(&traj_aug.states) {
            for i in 0..4 {
                assert!((s[i] - t[i]).abs() < 1e-10, "component {i}: {} vs {}", s[i], t[i]);
            }
            assert_eq!(t[4], 0.0);
        }
        assert_eq!(traj.obs_labels, traj_aug.obs_labels);
        assert!((traj.observations[50][0] - traj_aug.observations[50][0]).abs() < 1e-10);
    }
}
