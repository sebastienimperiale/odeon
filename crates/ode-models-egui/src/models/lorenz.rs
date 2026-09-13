//! Lorenz-63 scene: the attractor as a 3D polyline seen through an
//! orbitable camera (drag to rotate, scroll to zoom), drawn with the
//! painter's 2D primitives after our own projection. Depth cues: a
//! wireframe bounding box, an axis triad, and per-segment ink that fades
//! with distance from the camera. The observed coordinate wears the
//! accent: the state's projection line onto that axis.

use super::{VizModel, draw_trace};
use crate::noise::NoiseModel;
use crate::palette;
use crate::playback::{Job, Trajectory};
use ode_models::spec::ModelSpec;
use ode_models::models::lorenz::{LorenzObservation, LorenzParams, LorenzSystem};

/// Default initial condition: on the attractor — the state reached from the
/// usual (1, 1, 1) start after t = 1.4, then a further t = 10 (dt = 0.01).
const DEFAULT_X0: [f64; 3] = [-3.716171, -4.204785, 20.339103];

#[derive(Clone, PartialEq)]
struct Params {
    phys: LorenzParams,
    x0: [f64; 3],
    /// Selected observation, index into [`LorenzObservation::ALL`].
    obs_idx: usize,
    obs_noise: [NoiseModel; 3],
}

impl Default for Params {
    fn default() -> Self {
        Params {
            phys: LorenzParams::default(),
            x0: DEFAULT_X0,
            obs_idx: 0,
            obs_noise: [NoiseModel::None; 3],
        }
    }
}

impl Params {
    fn observation(&self) -> LorenzObservation {
        LorenzObservation::ALL[self.obs_idx]
    }
}

/// Orbit camera: yaw/pitch around the attractor's center, orthographic
/// projection scaled to the widget. Kept in egui's per-frame memory so the
/// scene can be interactive without a mutable draw interface.
#[derive(Clone, Copy)]
struct Camera {
    yaw: f32,
    pitch: f32,
    zoom: f32,
}

impl Default for Camera {
    fn default() -> Self {
        // The classic view: looking along −y with a slight elevation, so the
        // two wings and the z axis read at once.
        Camera {
            yaw: -0.5,
            pitch: 0.35,
            zoom: 1.0,
        }
    }
}

impl Camera {
    /// World (x, y, z) → camera coordinates (right, up, depth toward the
    /// viewer): yaw about the world z axis, then pitch about the screen x
    /// axis. Scene convention: z is up.
    fn view(&self, p: [f64; 3]) -> [f32; 3] {
        let (x, y, z) = (p[0] as f32, p[1] as f32, p[2] as f32);
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let xr = cy * x - sy * y;
        let yr = sy * x + cy * y;
        // Pitch: rotate (yr, z) — yr is depth before pitch.
        let up = cp * z - sp * yr;
        let depth = sp * z + cp * yr;
        [xr, up, depth]
    }
}

#[derive(Default)]
pub struct LorenzViz {
    params: Params,
    drawn: Params,
}

impl VizModel for LorenzViz {
    fn name(&self) -> &'static str {
        "Lorenz-63"
    }

    fn description(&self) -> String {
        "Lorenz-63: ẋ = σ(y − x), ẏ = x(ρ − z) − y, ż = xy − βz — the three-mode \
         truncation of Rayleigh–Bénard convection (Lorenz 1963), the classic dissipative \
         chaotic system. Phase-space volume contracts at the constant rate −(σ + 1 + β), \
         and for the default parameters every trajectory settles on the butterfly-shaped \
         strange attractor, switching irregularly between its two wings. Not Hamiltonian: \
         nothing is conserved, and nearby trajectories separate exponentially, so any \
         estimate loses track without observations.\n\n\
         Parameters: σ — Prandtl number (default 10); ρ — Rayleigh number (default 28; \
         for ρ < 1 the origin attracts everything, the attractor exists for ρ ≳ 24.7); \
         β — geometric factor (default 8/3). The two wings are centered on the fixed \
         points C± = (±√(β(ρ−1)), ±√(β(ρ−1)), ρ−1).\n\n\
         State (filter order): x — convective intensity; y — temperature difference between \
         rising and sinking currents; z — deviation of the vertical temperature profile from \
         linear. Typical ranges |x| ≲ 20, |y| ≲ 28, 0 < z ≲ 50. The default initial \
         condition is a state on the attractor (t = 11.4 of the (1, 1, 1) start), so \
         the default run has no transient.\n\n\
         Scene: the trajectory as a 3D curve (z up), seen through an orbit camera — drag to \
         rotate, scroll to zoom. Wireframe box of the trajectory's extent, axis triad at its \
         corner, the history shaded by depth (nearer is darker) with the recent tail strong, \
         the current state as the accent dot with a projection line onto the observed axis \
         (drawn in the accent).\n\n\
         Observation (one of): the coordinate x, y or z. Observing x alone is the standard \
         hard case: y and z must be inferred through the dynamics."
            .to_string()
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let before = self.params.clone();
        let p = &mut self.params;
        ui.horizontal(|ui| {
            ui.label("Parameters");
            if ui.small_button("default").clicked() {
                p.phys = LorenzParams::default();
            }
        });
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut p.phys.sigma).speed(0.1).range(0.0..=50.0));
            ui.label("σ (Prandtl)");
        });
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut p.phys.rho).speed(0.1).range(0.0..=100.0));
            ui.label("ρ (Rayleigh)").on_hover_text(
                "ρ < 1: the origin attracts; ρ ≈ 28 with σ = 10, β = 8/3: the strange \
                 attractor",
            );
        });
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut p.phys.beta).speed(0.05).range(0.0..=10.0));
            ui.label("β");
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Initial condition");
            if ui.small_button("default").clicked() {
                p.x0 = DEFAULT_X0;
            }
        });
        for (i, label) in ["x", "y", "z"].iter().enumerate() {
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut p.x0[i]).speed(0.1).range(-50.0..=80.0));
                ui.label(*label);
            });
        }
        ui.weak("Scene: drag to orbit, scroll to zoom");
        before != self.params
    }

    fn obs_options(&self) -> Vec<String> {
        vec!["x".into(), "y".into(), "z".into()]
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
            &format!("lorenz_noise_{idx}"),
        )
    }

    fn state_labels(&self) -> Vec<String> {
        ["x", "y", "z"].map(String::from).to_vec()
    }

    fn periodic(&self) -> Vec<bool> {
        vec![false; 3]
    }

    fn default_pairs(&self) -> Vec<(usize, usize)> {
        vec![(0, 2), (0, 1)]
    }

    fn model_spec(&self) -> ModelSpec {
        let p = &self.drawn;
        ModelSpec::Lorenz {
            params: p.phys,
            x0: p.x0,
            observation: p.observation(),
            noise: p.obs_noise[p.obs_idx],
            noise_seed: 5000 + p.obs_idx as u64,
        }
    }

    fn make_job(&mut self, dt: f64, steps: usize) -> Job {
        self.drawn = self.params.clone();
        let p = self.params.clone();
        Box::new(move |progress| {
            let obs = p.observation();
            let mut sys = LorenzSystem::with_params(p.phys, p.x0, dt, obs);
            for _ in 0..steps {
                sys.forward();
                progress.step();
            }
            let states: Vec<Vec<f64>> =
                sys.states.iter().map(|s| s.iter().copied().collect()).collect();
            let mut observations: Vec<Vec<f64>> =
                states.iter().map(|s| vec![obs.h(s)]).collect();
            let noise = p.obs_noise[p.obs_idx];
            let eta = noise.realize(observations.len(), dt, 5000 + p.obs_idx as u64);
            for (o, e) in observations.iter_mut().zip(eta) {
                o[0] += e;
            }
            let obs_labels = vec![if noise.is_none() {
                obs.label().to_string()
            } else {
                format!("{} (noisy)", obs.label())
            }];
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
        let ctx = painter.ctx();
        let cam_id = egui::Id::new("lorenz_camera");

        // ── Interaction: orbit with the primary button, zoom with scroll ──
        let mut cam: Camera = ctx.data(|d| d.get_temp(cam_id)).unwrap_or_default();
        ctx.input(|i| {
            let over = i.pointer.interact_pos().is_some_and(|p| rect.contains(p));
            if over && i.pointer.primary_down() {
                let d = i.pointer.delta();
                cam.yaw += 0.01 * d.x;
                cam.pitch = (cam.pitch + 0.01 * d.y).clamp(-1.5, 1.5);
            }
            if over {
                let s = i.smooth_scroll_delta.y;
                if s != 0.0 {
                    cam.zoom = (cam.zoom * (1.0 + 0.002 * s)).clamp(0.3, 5.0);
                }
            }
        });
        ctx.data_mut(|d| d.insert_temp(cam_id, cam));

        // ── Projection: world box from the trajectory, orthographic ───────
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for s in &traj.states {
            for d in 0..3 {
                lo[d] = lo[d].min(s[d]);
                hi[d] = hi[d].max(s[d]);
            }
        }
        // A preview has one state: pad to a sensible box around it.
        for d in 0..3 {
            if hi[d] - lo[d] < 1.0 {
                lo[d] -= 20.0;
                hi[d] += 20.0;
            }
        }
        let center = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])];
        let radius = (0..3)
            .map(|d| 0.5 * (hi[d] - lo[d]))
            .fold(0.0_f64, |m, r| m.max(r))
            * 1.6;
        let scale = 0.5 * rect.width().min(rect.height()) / radius as f32 * cam.zoom;
        let view = |p: [f64; 3]| -> [f32; 3] {
            cam.view([p[0] - center[0], p[1] - center[1], p[2] - center[2]])
        };
        let to_screen = |v: [f32; 3]| egui::pos2(rect.center().x + scale * v[0], rect.center().y - scale * v[1]);
        let painter = painter.with_clip_rect(rect);

        let neutral = visuals.widgets.noninteractive.fg_stroke.color;
        let faint = neutral.gamma_multiply(0.25);
        let accent = palette::series(0, visuals.dark_mode);
        let trace_color = palette::series(2, visuals.dark_mode);

        // Bounding box wireframe.
        let corner = |i: usize| {
            [
                if i & 1 == 0 { lo[0] } else { hi[0] },
                if i & 2 == 0 { lo[1] } else { hi[1] },
                if i & 4 == 0 { lo[2] } else { hi[2] },
            ]
        };
        for i in 0..8 {
            for bit in [1, 2, 4] {
                if i & bit == 0 {
                    let (a, b) = (view(corner(i)), view(corner(i | bit)));
                    painter.line_segment(
                        [to_screen(a), to_screen(b)],
                        egui::Stroke::new(1.0, faint),
                    );
                }
            }
        }
        // Axis triad at the box corner (lo, lo, lo), labelled.
        let o = corner(0);
        let len = 0.25 * radius;
        for (d, label) in ["x", "y", "z"].iter().enumerate() {
            let mut e = o;
            e[d] += len;
            let (a, b) = (to_screen(view(o)), to_screen(view(e)));
            let col = if d == self.drawn.obs_idx { accent } else { neutral };
            painter.arrow(a, b - a, egui::Stroke::new(1.5, col));
            painter.text(
                b + (b - a).normalized() * 8.0,
                egui::Align2::CENTER_CENTER,
                *label,
                egui::FontId::proportional(13.0),
                col,
            );
        }

        // Trajectory up to the frame, faint full history + strong tail, with
        // depth shading on the history: farther segments lighter.
        let pts: Vec<[f32; 3]> = traj.states[..=frame]
            .iter()
            .map(|s| view([s[0], s[1], s[2]]))
            .collect();
        let depth_range = 2.0 * radius as f32;
        for w in pts.windows(2) {
            let t = ((w[0][2] + w[1][2]) * 0.5 / depth_range + 0.5).clamp(0.0, 1.0);
            let ink = 0.12 + 0.3 * t; // near = darker
            painter.line_segment(
                [to_screen(w[0]), to_screen(w[1])],
                egui::Stroke::new(1.0, trace_color.gamma_multiply(ink)),
            );
        }
        let tail: Vec<egui::Pos2> = pts[pts.len().saturating_sub(150)..]
            .iter()
            .map(|&v| to_screen(v))
            .collect();
        draw_trace(&painter, &tail, trace_color);

        // The current state: projection line onto the observed axis (through
        // the box corner) and the accent dot.
        let s = &traj.states[frame];
        let cur = [s[0], s[1], s[2]];
        let mut foot = o;
        foot[self.drawn.obs_idx] = cur[self.drawn.obs_idx];
        painter.line_segment(
            [to_screen(view(cur)), to_screen(view(foot))],
            egui::Stroke::new(1.0, accent.gamma_multiply(0.6)),
        );
        painter.circle_filled(to_screen(view(foot)), 3.0, accent);
        let body = to_screen(view(cur));
        painter.circle_filled(body, 6.0, accent);
        painter.circle_stroke(body, 6.0, egui::Stroke::new(1.0, visuals.window_fill()));
    }
}
