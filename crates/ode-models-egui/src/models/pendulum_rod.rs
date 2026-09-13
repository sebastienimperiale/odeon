//! Compound (rigid-rod) double-pendulum scene: two thick uniform rods, tip
//! trace (the heart-shaped orbit from its preset). The observation is
//! h(x) = cos q₁, so the *first rod* wears the observation accent; the trace
//! is scene decoration and stays in a secondary hue.

use super::{SceneMap, VizModel, draw_trace};
use crate::noise::NoiseModel;
use crate::palette;
use crate::playback::{Job, Trajectory};
use ode_models::spec::ModelSpec;
use ode_models::models::pendulum_rod::{PendulumRodParams, PendulumRodSystem};

/// Initial state of the heart-shaped tip orbit (released from rest).
const HEART: [f64; 4] = [2.453, -2.7727, 0.0, 0.0];

#[derive(Clone, PartialEq)]
struct Params {
    phys: PendulumRodParams,
    x0: [f64; 4],
    /// Whether the (single) observation, cos q₁, is selected.
    obs_on: bool,
    /// Noise model of the observation.
    obs_noise: NoiseModel,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            phys: PendulumRodParams::default(),
            x0: HEART,
            obs_on: true,
            obs_noise: NoiseModel::None,
        }
    }
}

#[derive(Default)]
pub struct PendulumRodViz {
    params: Params,
    /// Snapshot taken by `make_job`; the scene is drawn with these, so form
    /// edits after a run don't desynchronize scene and trajectory.
    drawn: Params,
}

impl VizModel for PendulumRodViz {
    fn name(&self) -> &'static str {
        "Double pendulum (rigid rods)"
    }

    fn description(&self) -> String {
        "Planar double compound pendulum: two identical uniform rigid rods of mass m and \
         length ℓ, pinned end to end below a fixed pivot, gravity g downward (the swaptube \
         model). The mass is distributed along the rods (moment of inertia mℓ²/3), which \
         changes the dynamics from the point-mass pendulum. Conservative, chaotic at large \
         amplitudes; released from rest at the 'heart' preset the tip traces a heart-shaped \
         orbit.\n\n\
         Parameters: m — mass of each rod; ℓ — length of each rod; g — gravity.\n\n\
         State (filter order): q₁, q₂ — angles of the rods from the downward vertical \
         (radians, periodic on (−π, π)); p₁, p₂ — conjugate momenta.\n\n\
         Scene: pivot, the two rods (drawn thick, as rigid bodies) and the fading trace of \
         the free end. The first rod wears the observation accent.\n\n\
         Observation: cos q₁, the vertical elevation of the first rod. It is even in q₁, \
         so ±q₁ cannot be told apart from the observation alone — expect bimodal densities \
         until the dynamics break the symmetry."
            .to_string()
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let before = self.params.clone();
        let p = &mut self.params;
        ui.horizontal(|ui| {
            ui.label("Physical parameters (identical rods)");
            if ui.small_button("default").clicked() {
                p.phys = PendulumRodParams::default();
            }
        });
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut p.phys.m).speed(0.02).range(0.05..=10.0));
            ui.label("rod mass m");
        });
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut p.phys.l).speed(0.02).range(0.05..=10.0));
            ui.label("rod length ℓ");
        });
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut p.phys.g).speed(0.1).range(0.0..=30.0));
            ui.label("gravity g");
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Initial condition");
            if ui.small_button("default").clicked() {
                p.x0 = HEART;
            }
        });
        //ui.label("(heart orbit: 2.453, −2.7727, 0, 0)");
        for (i, label) in ["q₁", "q₂", "p₁", "p₂"].iter().enumerate() {
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut p.x0[i]).speed(0.02));
                ui.label(*label);
            });
        }
        before != self.params
    }

    fn obs_options(&self) -> Vec<String> {
        vec!["cos q₁  (first-rod elevation)".into()]
    }

    fn obs_exclusive(&self) -> bool {
        false
    }

    fn obs_selected(&self) -> Vec<bool> {
        vec![self.params.obs_on]
    }

    fn toggle_obs(&mut self, _idx: usize) {
        self.params.obs_on = !self.params.obs_on;
    }

    fn noise_ui(&mut self, ui: &mut egui::Ui, _idx: usize) -> bool {
        crate::noise::noise_ui(&mut self.params.obs_noise, ui, "pendulum_rod_noise")
    }

    fn state_labels(&self) -> Vec<String> {
        ["q_1", "q_2", "p_1", "p_2"].map(String::from).to_vec()
    }

    fn periodic(&self) -> Vec<bool> {
        vec![true, true, false, false]
    }

    fn default_pairs(&self) -> Vec<(usize, usize)> {
        vec![(0, 2), (1, 3)]
    }

    fn model_spec(&self) -> ModelSpec {
        let p = &self.drawn;
        ModelSpec::PendulumRod { params: p.phys, x0: p.x0, noise: p.obs_noise, noise_seed: 3000 }
    }

    fn make_job(&mut self, dt: f64, steps: usize) -> Job {
        self.drawn = self.params.clone();
        let p = self.params.clone();
        Box::new(move |progress| {
            let mut sys = PendulumRodSystem::with_params(p.phys, p.x0, dt);
            for _ in 0..steps {
                sys.forward();
                progress.step();
            }
            let states: Vec<Vec<f64>> = sys.states.iter().map(|s| s.iter().copied().collect()).collect();
            let mut observations: Vec<Vec<f64>> = states
                .iter()
                .map(|s| if p.obs_on { vec![s[0].cos()] } else { Vec::new() })
                .collect();
            if p.obs_on {
                let eta = p.obs_noise.realize(observations.len(), dt, 3000);
                for (obs, e) in observations.iter_mut().zip(eta) {
                    obs[0] += e;
                }
            }
            let obs_labels = if p.obs_on {
                vec![if p.obs_noise.is_none() {
                    "cos q₁".to_string()
                } else {
                    "cos q₁ (noisy)".to_string()
                }]
            } else {
                Vec::new()
            };
            Trajectory {
                dt,
                states,
                observations,
                obs_labels,
            }
        })
    }

    fn draw(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        traj: &Trajectory,
        frame: usize,
        visuals: &egui::Visuals,
    ) {
        let phys = &self.drawn.phys;
        let l = phys.l;
        let map = SceneMap::fit(rect, (-2.1 * l, 2.1 * l), (-0.05 * l, 4.05 * l));

        let neutral = visuals.widgets.noninteractive.fg_stroke.color;
        // Rod 1 is the observed element; it keeps the neutral ink when the
        // observation is deselected.
        let accent = if self.drawn.obs_on {
            palette::series(0, visuals.dark_mode)
        } else {
            neutral
        };
        let trace_color = palette::series(2, visuals.dark_mode); // scene decoration

        // Tip trace up to the current frame (the heart orbit, on the preset).
        let trace: Vec<egui::Pos2> = traj.states[..=frame]
            .iter()
            .map(|s| {
                map.pt(
                    l * (s[0].sin() + s[1].sin()),
                    2.0 * l - l * (s[0].cos() + s[1].cos()),
                )
            })
            .collect();
        draw_trace(painter, &trace, trace_color);

        // The two rods; rod 1 is the observed element.
        let [q1, q2] = [traj.states[frame][0], traj.states[frame][1]];
        let pivot = map.pt(0.0, 2.0 * l);
        let joint = map.pt(l * q1.sin(), 2.0 * l - l * q1.cos());
        let tip = map.pt(
            l * (q1.sin() + q2.sin()),
            2.0 * l - l * (q1.cos() + q2.cos()),
        );
        let w = map.len(0.05 * l).clamp(4.0, 10.0);
        painter.line_segment([pivot, joint], egui::Stroke::new(w, accent));
        painter.line_segment([joint, tip], egui::Stroke::new(w, neutral));
        // Rounded ends and hinges.
        painter.circle_filled(pivot, 0.7 * w, accent);
        painter.circle_filled(joint, 0.7 * w, neutral);
        painter.circle_filled(tip, 0.5 * w, neutral);
        painter.circle_filled(pivot, 0.3 * w, visuals.window_fill());
    }
}
