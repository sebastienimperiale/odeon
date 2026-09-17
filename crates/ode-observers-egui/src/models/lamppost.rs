//! Random-walk scene: the lamppost at the origin, the man as a disc at (x, y)
//! with the trace of his (optional) walk, and the observed quantity — the
//! circle of radius √(x² + y²) — in the accent. The filter's density is a
//! ring on that circle; the tracker collapses onto one of its points.

use super::{EstimatorHints, SceneMap, VarHint, VizModel, draw_trace};
use crate::noise::NoiseModel;
use crate::palette;
use crate::playback::Trajectory;
use ode_models::models::lamppost::{DEFAULT_STATE, DIM};
use ode_models_spec::reference::Walk;
use ode_models_spec::spec::{ModelSpec, TwinSpec};

/// Half-width of the scene window (world units, centered on the lamppost).
const WINDOW: f64 = 2.0;

#[derive(Clone, PartialEq)]
struct Params {
    x0: [f64; 2],
    /// Standard deviation of the reference's random walk (0 = static man).
    walk_std: f64,
    obs_noise: NoiseModel,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            x0: DEFAULT_STATE,
            walk_std: 0.0,
            obs_noise: NoiseModel::None,
        }
    }
}

#[derive(Default)]
pub struct LamppostViz {
    params: Params,
    /// Snapshot taken by `make_job`; the scene is drawn with these.
    drawn: Params,
}

impl VizModel for LamppostViz {
    fn name(&self) -> &'static str {
        "Lamppost"
    }

    fn doc_dest(&self) -> Option<&'static str> {
        Some("lamppost")
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let before = self.params.clone();
        let p = &mut self.params;
        ui.horizontal(|ui| {
            ui.label("Initial position").on_hover_text("The toy's reference is (0, 1): on the unit circle.");
            if ui.small_button("default").clicked() {
                p.x0 = DEFAULT_STATE;
            }
        });
        for (i, label) in ["x₀", "y₀"].iter().enumerate() {
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut p.x0[i]).speed(0.02).range(-WINDOW..=WINDOW));
                ui.label(*label);
            });
        }
        ui.separator();
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut p.walk_std).speed(0.01).range(0.0..=2.0));
            ui.label("walk σ").on_hover_text(
                "Standard deviation of the reference's random walk per unit time (0: the man \
                 stands still — the drunkness is then only the filter's model noise q).",
            );
        });
        before != self.params
    }

    fn obs_options(&self) -> Vec<String> {
        vec!["x² + y²  (squared distance to the lamppost)".into()]
    }

    fn obs_exclusive(&self) -> bool {
        true
    }

    fn obs_selected(&self) -> Vec<bool> {
        vec![true]
    }

    fn toggle_obs(&mut self, _idx: usize) {}

    fn noise_ui(&mut self, ui: &mut egui::Ui, _idx: usize) -> bool {
        crate::noise::noise_ui(&mut self.params.obs_noise, ui, "lamppost_noise")
    }

    fn state_labels(&self) -> Vec<String> {
        vec!["x".into(), "y".into()]
    }

    fn periodic(&self) -> Vec<Option<(f64, f64)>> {
        vec![None; DIM]
    }

    fn default_pairs(&self) -> Vec<(usize, usize)> {
        vec![(0, 1)]
    }

    /// The reference is a point, so the range-based domain default would
    /// be a small box around it: use the scene window, at the resolution
    /// of the one-spring example, so that the whole ring is on the grid.
    fn estimator_hints(&self) -> EstimatorHints {
        let var = VarHint { domain: Some((-WINDOW, WINDOW)), n_el: Some(16), p_ord: Some(4), ..VarHint::default() };
        EstimatorHints { vars: vec![var; DIM], eps: Some(0.05) }
    }

    fn snapshot(&mut self) {
        self.drawn = self.params.clone();
    }

    fn model_spec(&self) -> ModelSpec {
        ModelSpec::Lamppost
    }

    /// The man at x₀, walking when the form asks for it (seed 7000), the
    /// squared distance observed with the form's noise (seed 7100).
    fn twin_spec(&self, dt: f64, steps: usize) -> TwinSpec {
        let p = &self.drawn;
        TwinSpec {
            model: ModelSpec::Lamppost,
            x0: p.x0.to_vec(),
            dt,
            steps,
            noise: p.obs_noise,
            seed: 7100,
            walk: (p.walk_std > 0.0).then_some(Walk { std: p.walk_std, seed: 7000 }),
        }
    }

    fn obs_labels(&self) -> Vec<String> {
        vec![if self.drawn.obs_noise.is_none() { "x² + y²".to_string() } else { "x² + y² (noisy)".to_string() }]
    }

    fn draw(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        traj: &Trajectory,
        frame: usize,
        visuals: &egui::Visuals,
    ) {
        let map = SceneMap::fit(rect, (-WINDOW, WINDOW), (-WINDOW, WINDOW));
        let painter = painter.with_clip_rect(rect);
        let painter = &painter;
        let neutral = visuals.widgets.noninteractive.fg_stroke.color;
        let accent = palette::series(0, visuals.dark_mode);
        let trace_color = palette::series(2, visuals.dark_mode);

        // The lamppost.
        let origin = map.pt(0.0, 0.0);
        painter.circle_filled(origin, 5.0, neutral);
        painter.line_segment([origin, origin + egui::vec2(0.0, -map.len(0.25))], egui::Stroke::new(2.0, neutral));

        // The walk so far.
        let trace: Vec<egui::Pos2> = traj.states[..=frame].iter().map(|s| map.pt(s[0], s[1])).collect();
        draw_trace(painter, &trace, trace_color);

        // The observed circle and the man, in the accent.
        let [x, y] = [traj.states[frame][0], traj.states[frame][1]];
        let man = map.pt(x, y);
        painter.circle_stroke(origin, map.len(x.hypot(y)), egui::Stroke::new(1.5, accent.gamma_multiply(0.7)));
        painter.line_segment([origin, man], egui::Stroke::new(1.0, accent.gamma_multiply(0.5)));
        painter.circle_filled(man, 6.0, accent);
        painter.circle_stroke(man, 6.0, egui::Stroke::new(1.0, visuals.window_fill()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hints put the whole ring on the grid: the (−2, 2)² box at 16×4
    /// in both directions, ε = 0.05. (The ring itself, through the whole
    /// pipeline, is tested in the viewer app.)
    #[test]
    fn hints_cover_the_ring() {
        let viz = LamppostViz::default();
        let hints = viz.estimator_hints();
        assert_eq!(hints.vars.len(), DIM);
        assert!(hints.vars.iter().all(|v| v.domain == Some((-WINDOW, WINDOW))));
        assert!(hints.vars.iter().all(|v| v.n_el == Some(16) && v.p_ord == Some(4)));
        assert_eq!(hints.eps, Some(0.05));
        assert_eq!(viz.model_spec().dim(), DIM);
    }
}
