//! Spring-chain scene: N masses on a horizontal axis, springs drawn as
//! zigzag coils, wall at x = 0. One viewer instance per chain length
//! ("Single spring", "Two springs", "Three springs"); the full initial data
//! (every yᵢ, vᵢ) is editable. The model's state y holds *displacements*
//! from the rest positions iℓ (its equilibrium is y = 0); the scene draws
//! mass i at iℓ + yᵢ. (Before 2026-09-13 the entry fed the model absolute
//! positions and drew y directly, so the default single spring swung
//! ±1.15 through the wall.) A position observation colors the mass disc
//! with its series color; a velocity observation draws a diamond on the mass
//! in its series color.
//!
//! **Unknown free-end mass** — an option of the estimator viewer only
//! ([`VizModel::estimator_options_ui`]; a models-only viewer never shows
//! it): the same chain driven by `SpringMassSystem` instead of
//! `SpringSystem`. Its augmented state (y₁..y_N, v₁..v_N, θ) carries the
//! log-mass θ of the free-end body, m_N = m₀·2^θ with θ̇ = 0, where m₀ = ρℓ
//! is the mass of the other bodies (the estimators' prior, θ = 0 ⇔ all
//! masses equal) and the reference runs at θ_true. The physical chain is
//! the same — masses ρℓ, stiffness a/ℓ, the same state (y, v) and the same
//! equations m ÿ = −K y with K = k·tridiag(−1, 2, −1), K_NN = k — so at
//! θ_true = 0 the two runs agree to the integrators' accuracy
//! (Gauss–Legendre 4 instead of the implicit midpoint; test
//! `augmented_run_is_the_same_chain`); only positions can be observed. The
//! whole run —
//! scene, observation plot and estimators — uses the augmented model, so
//! the estimators' twin reference is exactly the trajectory on screen;
//! toggling the option therefore requires a new run. The model is
//! conditionally linear given θ — the case where the hybrid grid–Gaussian
//! closure is exact — and the twin experiment is set up like Kepler-μ's:
//! the θ direction gets a fixed (−1, 1) domain, cheap resolution, q_θ = 0
//! and a Gaussian center at θ = 0, with the (y₁, θ) plane among the
//! default outputs.

use super::{EstimatorHints, SceneMap, VarHint, VizModel};
use crate::noise::NoiseModel;
use crate::palette;
use crate::playback::Trajectory;
use ode_models::models::SpringMassParams;
use ode_models_spec::spec::{ModelSpec, TwinSpec};

/// Default true log-mass of the reference when the free-end mass is
/// unknown: m_N = 2^0.3·m₀ ≈ 1.23·m₀. The normal-mode frequencies then
/// differ from the prior's by ~10 %, enough for the θ-likelihood to be
/// aliased after a few periods.
const DEFAULT_THETA: f64 = 0.3;
/// Fixed filter domain of the θ direction (m_N = m₀·2^θ > 0 for every θ).
const THETA_DOMAIN: (f64, f64) = (-1.0, 1.0);

#[derive(Clone, PartialEq)]
struct Params {
    n: usize,
    rho: f64,
    a: f64,
    /// Full initial data, one entry per mass: displacements from rest and
    /// velocities.
    y0: Vec<f64>,
    v0: Vec<f64>,
    /// Selected observation component, flat [Y; V] order: entries 0..n are
    /// the positions, n..2n the velocities. Exactly one entry is true (the
    /// filter consumes a single scalar observation).
    obs_sel: Vec<bool>,
    /// Noise model of each observation component (same indexing).
    obs_noise: Vec<NoiseModel>,
    /// Estimator-only option: the free-end mass is unknown — the run uses
    /// the augmented model (state (y, v, θ), displacements from rest).
    unknown_mass: bool,
    /// True log-mass θ of the reference when `unknown_mass` (m_N = m₀·2^θ).
    theta: f64,
}

impl Params {
    /// Rest positions iℓ, ℓ = 1/N.
    fn rest(n: usize) -> Vec<f64> {
        let ell = 1.0 / n as f64;
        (1..=n).map(|i| i as f64 * ell).collect()
    }

    /// Default initial data: at rest except the last mass, displaced by
    /// 0.15, so the default run already moves.
    fn default_ic(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut y0 = vec![0.0; n];
        y0[n - 1] = 0.15;
        (y0, vec![0.0; n])
    }

    fn new(n: usize) -> Self {
        let (y0, v0) = Self::default_ic(n);
        let mut obs_sel = vec![false; 2 * n];
        obs_sel[n - 1] = true; // position of the last mass
        Params {
            n,
            rho: 1.0,
            a: 1.0,
            y0,
            v0,
            obs_sel,
            obs_noise: vec![NoiseModel::None; 2 * n],
            unknown_mass: false,
            theta: DEFAULT_THETA,
        }
    }

    /// The single selected component, flat [Y; V] index — the model's
    /// scalar h.
    fn obs_index(&self) -> usize {
        self.obs_sel.iter().position(|&s| s).unwrap_or(0)
    }

    /// State dimension of the run: 2N, or 2N + 1 with the unknown mass.
    fn dim(&self) -> usize {
        2 * self.n + usize::from(self.unknown_mass)
    }

    /// Seed of the observation noise of the twin experiment.
    fn noise_seed(&self) -> u64 {
        1000 + self.obs_index() as u64
    }

    /// Physical parameters of the augmented model: every body of mass
    /// m = ρℓ, the free end's prior m₀ = m too, springs of stiffness a/ℓ —
    /// the chain of `SpringSystem` with the last mass made unknown.
    fn mass_params(&self) -> SpringMassParams {
        let ell = 1.0 / self.n as f64;
        let m = self.rho * ell;
        SpringMassParams { m, m0: m, k: self.a / ell }
    }

    /// Initial state of the augmented model: the same (y, v), then θ_true.
    fn augmented_x0(&self) -> Vec<f64> {
        self.y0
            .iter()
            .chain(self.v0.iter())
            .copied()
            .chain(std::iter::once(self.theta))
            .collect()
    }
}

pub struct SpringViz {
    name: &'static str,
    params: Params,
    /// Snapshot taken by `make_job`; the scene is drawn with these, so form
    /// edits after a run don't desynchronize scene and trajectory.
    drawn: Params,
}

impl SpringViz {
    pub fn new(n: usize, name: &'static str) -> Self {
        assert!((1..=3).contains(&n), "viewer spring chains have 1 to 3 masses");
        SpringViz {
            name,
            params: Params::new(n),
            drawn: Params::new(n),
        }
    }

    /// Set the estimator-only "unknown free-end mass" option
    /// programmatically (what the checkbox of
    /// [`estimator_options_ui`](VizModel::estimator_options_ui) does); with
    /// the option on, a selected velocity observation falls back to the
    /// free end's position. Takes effect at the next `make_job`.
    pub fn set_unknown_mass(&mut self, on: bool) {
        let p = &mut self.params;
        p.unknown_mass = on;
        if on && p.obs_index() >= p.n {
            for (k, sel) in p.obs_sel.iter_mut().enumerate() {
                *sel = k == p.n - 1;
            }
        }
    }
}

impl VizModel for SpringViz {
    fn name(&self) -> &'static str {
        self.name
    }

    fn doc_dest(&self) -> Option<&'static str> {
        Some(if self.params.unknown_mass { "spring_mass" } else { "spring" })
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let before = self.params.clone();
        let p = &mut self.params;
        ui.horizontal(|ui| {
            ui.label("Physical parameters");
            if ui.small_button("default").clicked() {
                p.rho = 1.0;
                p.a = 1.0;
            }
        });
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut p.rho).speed(0.05).range(0.01..=100.0));
            ui.label("density ρ");
        });
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut p.a).speed(0.05).range(0.01..=100.0));
            ui.label("elastic modulus a");
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Initial condition").on_hover_text(
                "Displacements from rest and velocities of the masses (the scene adds \
                 the rest positions iℓ, ℓ = 1/N).",
            );
            if ui.small_button("default").clicked() {
                (p.y0, p.v0) = Params::default_ic(p.n);
            }
        });
        for i in 0..p.n {
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut p.y0[i]).speed(0.01));
                ui.label(format!("y_{}", i + 1));
                ui.add(egui::DragValue::new(&mut p.v0[i]).speed(0.01));
                ui.label(format!("v_{}", i + 1));
            });
        }
        before != self.params
    }

    fn estimator_options_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let before = self.params.clone();
        let mut on = self.params.unknown_mass;
        ui.checkbox(&mut on, "unknown free-end mass").on_hover_text(
            "Run the chain with the mass of the last body unknown: the state is \
             augmented with its log-mass θ (m_N = m₀·2^θ, m₀ = ρℓ the mass of the \
             others, dθ/dt = 0), the reference runs at θ_true and the estimators start \
             at θ = 0. Same chain and state, Gauss–Legendre 4 instead of the midpoint \
             rule, positions only as observations. Requires a new run.",
        );
        self.set_unknown_mass(on);
        if on {
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut self.params.theta)
                        .speed(0.01)
                        .range(THETA_DOMAIN.0..=THETA_DOMAIN.1),
                );
                ui.label("θ_true").on_hover_text(
                    "True log-mass of the reference: m_N = m₀·2^θ_true (θ in doublings \
                     of m₀, bounded to the filter's fixed θ domain (−1, 1)).",
                );
            });
        }
        before != self.params
    }

    /// Positions and velocities; positions only with the unknown mass
    /// (the augmented model observes a position).
    fn obs_options(&self) -> Vec<String> {
        let n = self.params.n;
        let positions = (1..=n).map(|i| format!("position y_{i}"));
        if self.params.unknown_mass {
            positions.collect()
        } else {
            positions.chain((1..=n).map(|i| format!("velocity v_{i}"))).collect()
        }
    }

    fn obs_exclusive(&self) -> bool {
        true
    }

    fn obs_selected(&self) -> Vec<bool> {
        let k = self.obs_options().len();
        self.params.obs_sel[..k].to_vec()
    }

    /// Exclusive selection: pick `idx`, deselect everything else.
    fn toggle_obs(&mut self, idx: usize) {
        for (k, sel) in self.params.obs_sel.iter_mut().enumerate() {
            *sel = k == idx;
        }
    }

    fn noise_ui(&mut self, ui: &mut egui::Ui, idx: usize) -> bool {
        crate::noise::noise_ui(
            &mut self.params.obs_noise[idx],
            ui,
            ("spring_noise", self.params.n, idx),
        )
    }

    fn state_labels(&self) -> Vec<String> {
        let n = self.params.n;
        (1..=n)
            .map(|i| format!("y_{i}"))
            .chain((1..=n).map(|i| format!("v_{i}")))
            .chain(self.params.unknown_mass.then(|| "θ".to_string()))
            .collect()
    }

    fn periodic(&self) -> Vec<Option<(f64, f64)>> {
        vec![None; self.params.dim()]
    }

    /// The (yᵢ, vᵢ) phase planes; with the unknown mass the (y₁, θ) plane
    /// — where the parameter estimation is watched — comes first.
    fn default_pairs(&self) -> Vec<(usize, usize)> {
        let n = self.params.n;
        let theta = self.params.unknown_mass.then_some((0, 2 * n));
        theta.into_iter().chain((0..n).map(|i| (i, n + i))).collect()
    }

    /// Twin-experiment defaults for the θ direction when the mass is
    /// unknown, as for Kepler-μ: fixed domain, cheap resolution, q_θ = 0
    /// (dθ/dt = 0), and the Gaussian center at θ = 0 (the prior m₀)
    /// instead of the run's true value.
    fn estimator_hints(&self) -> EstimatorHints {
        let mut hints = EstimatorHints::for_dim(self.params.dim());
        if self.params.unknown_mass {
            hints.vars[2 * self.params.n] = VarHint {
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
        if p.unknown_mass {
            ModelSpec::SpringMass { n: p.n, params: p.mass_params(), obs: p.obs_index() }
        } else {
            ModelSpec::Spring { n: p.n, rho: p.rho, a: p.a, obs: p.obs_index() }
        }
    }

    /// The plain chain from (y, v), or the augmented one from (y, v, θ_true)
    /// — the same (y, v) either way — with the selected component's noise.
    fn twin_spec(&self, dt: f64, steps: usize) -> TwinSpec {
        let p = &self.drawn;
        let x0 = if p.unknown_mass {
            p.augmented_x0()
        } else {
            p.y0.iter().chain(p.v0.iter()).copied().collect()
        };
        TwinSpec {
            model: self.model_spec(),
            x0,
            dt,
            steps,
            noise: p.obs_noise[p.obs_index()],
            seed: p.noise_seed(),
            walk: None,
        }
    }

    fn obs_labels(&self) -> Vec<String> {
        let p = &self.drawn;
        let k = p.obs_index();
        let name = if k < p.n { format!("y_{}", k + 1) } else { format!("v_{}", k - p.n + 1) };
        vec![if p.obs_noise[k].is_none() { name } else { format!("{name} (noisy)") }]
    }

    fn draw(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        traj: &Trajectory,
        frame: usize,
        visuals: &egui::Visuals,
    ) {
        // The state holds displacements: draw at the rest positions plus
        // them (same state either way; a trailing θ component is not drawn).
        let n = self.drawn.n;
        draw_chain(painter, rect, traj, frame, n, &Params::rest(n), &self.drawn.obs_sel, visuals);
    }
}

/// Spring-chain scene drawing: the first `n` state components are the
/// displacements of the masses, drawn at `offsets` (the rest positions)
/// plus them; `obs_sel` has `2n` entries in [Y; V] order
/// (positions color the disc, velocities draw a diamond).
fn draw_chain(
    painter: &egui::Painter,
    rect: egui::Rect,
    traj: &Trajectory,
    frame: usize,
    n: usize,
    offsets: &[f64],
    obs_sel: &[bool],
    visuals: &egui::Visuals,
) {
    {
        let state = &traj.states[frame];
        let positions: Vec<f64> = (0..n).map(|i| state[i] + offsets[i]).collect();
        let positions = &positions[..];
        // Series rank of component k = number of selected components before
        // it, i.e. its color slot in the observation plot.
        let rank = |k: usize| obs_sel[..k].iter().filter(|&&s| s).count();

        // World x range over the whole trajectory (so the frame doesn't jump).
        let mut x_max = f64::MIN;
        let mut x_min = 0.0_f64;
        for s in &traj.states {
            for (i, &y) in s[..n].iter().enumerate() {
                x_max = x_max.max(y + offsets[i]);
                x_min = x_min.min(y + offsets[i]);
            }
        }
        let span = (x_max - x_min).max(1e-6);
        let map = SceneMap::fit(
            rect,
            (x_min - 0.05 * span, x_max + 0.08 * span),
            (-0.25 * span, 0.25 * span),
        );

        let neutral = visuals.widgets.noninteractive.fg_stroke.color;
        let stroke = egui::Stroke::new(1.8, neutral);

        // Wall at x = 0 with hatching.
        let wall_top = map.pt(0.0, 0.18 * span);
        let wall_bot = map.pt(0.0, -0.18 * span);
        painter.line_segment([wall_top, wall_bot], egui::Stroke::new(3.0, neutral));
        let hatch = map.len(0.03 * span).max(6.0);
        let n_hatch = 7;
        for i in 0..n_hatch {
            let y = egui::lerp(wall_top.y..=wall_bot.y, (i as f32 + 0.5) / n_hatch as f32);
            painter.line_segment(
                [
                    egui::pos2(wall_top.x, y),
                    egui::pos2(wall_top.x - hatch, y + hatch),
                ],
                egui::Stroke::new(1.0, neutral),
            );
        }

        // Springs: wall→mass 1, then mass i→mass i+1.
        let r_mass = (map.len(0.018 * span)).clamp(6.0, 14.0);
        let mut left = map.pt(0.0, 0.0);
        for &pos in positions {
            let right = map.pt(pos, 0.0);
            painter.add(egui::Shape::line(
                zigzag(left, right, 8, (0.6 * r_mass).max(5.0)),
                stroke,
            ));
            left = right;
        }

        // Masses; each observed one wears its plot-series color (position
        // selection colors the disc, velocity selection draws a diamond on
        // the mass in the velocity series' color).
        for i in 0..n {
            let c = map.pt(positions[i], 0.0);
            let fill = if obs_sel[i] {
                palette::series(rank(i), visuals.dark_mode)
            } else {
                visuals.text_color()
            };
            painter.circle_filled(c, r_mass, fill);
            painter.circle_stroke(c, r_mass, egui::Stroke::new(1.0, visuals.window_fill()));
            if obs_sel[n + i] {
                let d = r_mass;
                let diamond = vec![
                    c + egui::vec2(0.0, -d),
                    c + egui::vec2(d, 0.0),
                    c + egui::vec2(0.0, d),
                    c + egui::vec2(-d, 0.0),
                ];
                painter.add(egui::Shape::convex_polygon(
                    diamond,
                    palette::series(rank(n + i), visuals.dark_mode),
                    egui::Stroke::new(1.0, visuals.window_fill()),
                ));
            }
        }
    }
}

/// Zigzag polyline from `a` to `b`: straight leads at both ends, `coils`
/// alternating perpendicular offsets in between.
fn zigzag(a: egui::Pos2, b: egui::Pos2, coils: usize, amp: f32) -> Vec<egui::Pos2> {
    let d = b - a;
    let len = d.length();
    if len < 1.0 {
        return vec![a, b];
    }
    let dir = d / len;
    let perp = egui::vec2(-dir.y, dir.x);
    let lead = (0.12 * len).min(12.0);
    let mut pts = vec![a, a + dir * lead];
    let inner = len - 2.0 * lead;
    for i in 0..coils {
        let f = (i as f32 + 0.5) / coils as f32;
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };
        pts.push(a + dir * (lead + f * inner) + perp * amp * side);
    }
    pts.push(b - dir * lead);
    pts.push(b);
    pts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playback::Progress;

    /// The option augments the run: dimension 2N + 1, θ label, positions
    /// only as observations (a selected velocity falls back to the free
    /// end's position), the (y₁, θ) plane first, and the twin-experiment
    /// hints on θ; off, everything is the plain chain and no hint is set.
    #[test]
    fn unknown_mass_option_augments_the_run() {
        for n in [1usize, 2, 3] {
            let mut viz = SpringViz::new(n, "test");
            assert_eq!(viz.state_labels().len(), 2 * n);
            assert_eq!(viz.obs_options().len(), 2 * n);
            assert!(viz.estimator_hints().vars.iter().all(|v| *v == VarHint::default()));
            viz.toggle_obs(2 * n - 1); // a velocity
            viz.set_unknown_mass(true);
            assert_eq!(viz.state_labels().len(), 2 * n + 1);
            assert_eq!(viz.state_labels()[2 * n], "θ");
            assert_eq!(viz.obs_options().len(), n);
            assert_eq!(viz.obs_selected(), (0..n).map(|k| k == n - 1).collect::<Vec<_>>());
            assert_eq!(viz.default_pairs()[0], (0, 2 * n));
            let hints = viz.estimator_hints();
            assert_eq!(hints.vars.len(), 2 * n + 1);
            let th = hints.vars[2 * n];
            assert_eq!(th.domain, Some(THETA_DOMAIN));
            assert_eq!(th.q, Some(0.0), "dθ/dt = 0 requires q_θ = 0");
            assert_eq!(th.x0, Some(0.0), "the prior must not be centered on θ_true");
            // The snapshot drives the spec: augmented after a (zero-step) run.
            let _ = viz.make_job(0.01, 0)(&Progress::default());
            assert_eq!(viz.model_spec().dim(), 2 * n + 1);
            assert_eq!(viz.model_spec().kind(), "spring_mass");
        }
    }

    /// The augmented run is the same physical chain: at θ_true = 0 its
    /// (y, v) trajectory agrees with the plain chain's to the integrators'
    /// accuracy (Gauss–Legendre 4 vs implicit midpoint: O(dt²) apart over
    /// a short run), and the plotted observation is the selected component
    /// of each.
    #[test]
    fn augmented_run_is_the_same_chain() {
        let dt = 0.002;
        let steps = 50;
        let mut plain = SpringViz::new(2, "plain");
        let traj = plain.make_job(dt, steps)(&Progress::default());
        let mut aug = SpringViz::new(2, "aug");
        aug.set_unknown_mass(true);
        aug.params.theta = 0.0;
        let traj_aug = aug.make_job(dt, steps)(&Progress::default());
        assert_eq!(traj.states[0].len(), 4);
        assert_eq!(traj_aug.states[0].len(), 5);
        for (s, t) in traj.states.iter().zip(&traj_aug.states) {
            for i in 0..4 {
                assert!((s[i] - t[i]).abs() < 1e-6, "component {i}: {} vs {}", s[i], t[i]);
            }
            assert_eq!(t[4], 0.0);
        }
        assert_eq!(traj.obs_labels, vec!["y_2"]);
        assert_eq!(traj_aug.obs_labels, vec!["y_2"]);
        assert!((traj.observations[10][0] - traj_aug.observations[10][0]).abs() < 1e-6);
    }
}
