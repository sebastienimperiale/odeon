//! Point-mass double-pendulum scene: two rods from the pivot, discs at the
//! masses, fading tip trace. The tip is the observed quantity (h(x) = tip
//! position), so its disc and trace wear the observation accent of the first
//! series; the y-component series shares the tip but gets its own plot color.

use super::{SceneMap, VizModel, draw_trace};
use crate::noise::NoiseModel;
use crate::palette;
use crate::playback::{Job, Trajectory};
use ode_models::spec::ModelSpec;
use ode_models::models::pendulum::{DIM, PendulumParams, PendulumSystem};

/// The paper's target trajectory (pendulum.pdf §5.2).
const PAPER_IC: [f64; 4] = [1.5, 1.4, 0.0, 0.0];

#[derive(Clone, PartialEq)]
struct Params {
    phys: PendulumParams,
    x0: [f64; 4],
    /// Whether the (single) observation, the tip position, is selected.
    obs_on: bool,
    /// Noise model of the observation (applied to both tip components,
    /// independent realizations).
    obs_noise: NoiseModel,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            phys: PendulumParams::default(),
            x0: PAPER_IC,
            obs_on: true,
            obs_noise: NoiseModel::None,
        }
    }
}

#[derive(Default)]
pub struct PendulumViz {
    params: Params,
    /// Snapshot taken by `make_job`; the scene is drawn with these, so form
    /// edits after a run don't desynchronize scene and trajectory.
    drawn: Params,
}

impl VizModel for PendulumViz {
    fn name(&self) -> &'static str {
        "Double pendulum (point masses)"
    }

    fn description(&self) -> String {
        "Planar double pendulum with point masses (pendulum.pdf §5.2): mass m₁ hangs from \
         a fixed pivot by a massless rigid rod of length ℓ₁, mass m₂ hangs from m₁ by a rod \
         of length ℓ₂, gravity g acts downward. Conservative Hamiltonian system; chaotic at \
         large amplitudes.\n\n\
         Parameters: ℓ₁, ℓ₂ — rod lengths; m₁, m₂ — masses; g — gravity.\n\n\
         State (filter order): q₁, q₂ — angles of the rods measured from the downward \
         vertical (radians, periodic on (−π, π)); p₁, p₂ — the conjugate momenta (not the \
         angular velocities: p = ∂L/∂q̇, they mix both rods' motion).\n\n\
         Scene: the pivot at the top, the two rods and the two masses, and the fading trace \
         of the tip (mass m₂). The tip wears the observation accent.\n\n\
         Observation: the tip position (x, y) — two scalar series, both functions of the \
         angles only; the momenta are unobserved and must be recovered by the filter \
         through the dynamics."
            .to_string()
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let before = self.params.clone();
        let p = &mut self.params;
        ui.horizontal(|ui| {
            ui.label("Physical parameters");
            if ui.small_button("default").clicked() {
                p.phys = PendulumParams::default();
            }
        });
        for (v, label) in [
            (&mut p.phys.l1, "length ℓ₁"),
            (&mut p.phys.l2, "length ℓ₂"),
            (&mut p.phys.m1, "mass m₁"),
            (&mut p.phys.m2, "mass m₂"),
        ] {
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(v).speed(0.02).range(0.05..=10.0));
                ui.label(label);
            });
        }
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut p.phys.g).speed(0.1).range(0.0..=30.0));
            ui.label("gravity g");
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Initial condition");
            if ui.small_button("default").clicked() {
                p.x0 = PAPER_IC;
            }
        });
       // ui.label("(paper trajectory: 1.5, 1.4, 0, 0)");
        for (i, label) in ["q₁", "q₂", "p₁", "p₂"].iter().enumerate() {
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut p.x0[i]).speed(0.02));
                ui.label(*label);
            });
        }
        before != self.params
    }

    fn obs_options(&self) -> Vec<String> {
        vec!["tip position (x, y)".into()]
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
        crate::noise::noise_ui(&mut self.params.obs_noise, ui, "pendulum_noise")
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
        ModelSpec::Pendulum { params: p.phys, x0: p.x0, noise: p.obs_noise, noise_seed: 2000 }
    }

    fn make_job(&mut self, dt: f64, steps: usize) -> Job {
        self.drawn = self.params.clone();
        let p = self.params.clone();
        Box::new(move |progress| {
            let mut sys = PendulumSystem::with_params(p.phys, p.x0, dt);
            for _ in 0..steps {
                sys.forward();
                progress.step();
            }
            let states: Vec<Vec<f64>> = sys.states.iter().map(|s| s.iter().copied().collect()).collect();
            let mut observations: Vec<Vec<f64>> = states
                .iter()
                .map(|s| {
                    if p.obs_on {
                        sys.tip_position(s[0], s[1]).to_vec()
                    } else {
                        Vec::new()
                    }
                })
                .collect();
            if p.obs_on {
                for j in 0..2 {
                    let eta = p.obs_noise.realize(observations.len(), dt, 2000 + j as u64);
                    for (obs, e) in observations.iter_mut().zip(eta) {
                        obs[j] += e;
                    }
                }
            }
            let suffix = if p.obs_noise.is_none() { "" } else { " (noisy)" };
            let obs_labels = if p.obs_on {
                vec![format!("tip x{suffix}"), format!("tip y{suffix}")]
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
        let (l1, l2) = (phys.l1, phys.l2);
        let big_l = l1 + l2;
        let map = SceneMap::fit(rect, (-1.05 * big_l, 1.05 * big_l), (-0.05 * big_l, 2.05 * big_l));

        let neutral = visuals.widgets.noninteractive.fg_stroke.color;
        // The tip is the observed element; it keeps the neutral ink when the
        // observation is deselected.
        let accent = if self.drawn.obs_on {
            palette::series(0, visuals.dark_mode)
        } else {
            visuals.text_color()
        };

        // Tip trace up to the current frame.
        let trace: Vec<egui::Pos2> = traj.states[..=frame]
            .iter()
            .map(|s| {
                map.pt(
                    l1 * s[0].sin() + l2 * s[1].sin(),
                    big_l - l1 * s[0].cos() - l2 * s[1].cos(),
                )
            })
            .collect();
        draw_trace(painter, &trace, accent);

        // Rods and masses at the current frame.
        let [q1, q2] = [traj.states[frame][0], traj.states[frame][1]];
        let pivot = map.pt(0.0, big_l);
        let mid = map.pt(l1 * q1.sin(), big_l - l1 * q1.cos());
        let tip = map.pt(
            l1 * q1.sin() + l2 * q2.sin(),
            big_l - l1 * q1.cos() - l2 * q2.cos(),
        );
        let rod = egui::Stroke::new(2.5, neutral);
        painter.line_segment([pivot, mid], rod);
        painter.line_segment([mid, tip], rod);

        // Disc radii grow slowly with the mass (∛m), pinned to a readable range.
        let r1 = (6.0 * phys.m1.cbrt() as f32).clamp(4.0, 16.0);
        let r2 = (6.0 * phys.m2.cbrt() as f32).clamp(4.0, 16.0);
        painter.circle_filled(pivot, 3.0, neutral);
        painter.circle_filled(mid, r1, visuals.text_color());
        painter.circle_filled(tip, r2, accent); // observed mass
    }
}

const _: () = assert!(DIM == 4);
