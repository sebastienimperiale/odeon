//! The estimator side of a viewer's parameter panel: the estimator
//! selector, the configuration panels editing a
//! [`FilterConfig`] (Initial data, Observation, Diffusion — shared by every
//! estimator; Grid — filter only; Window — box tracker only; LParticles —
//! particles only; Quadrature — unscented only; Outputs — the two grid
//! estimators), the Run-button
//! label, and the progress row of a running job with its Cancel button.
//! The app owns the configuration, the runs and the button itself; this
//! module only draws.

use ode_observers::jobs::{Estimator, FilterConfig};
use ode_observers::methods::unscented_kalman::Quadrature;

/// The estimator selector (filter / box tracker / LParticles / tracker /
/// unscented), with the one-line description of each on hover.
pub fn estimator_selector(ui: &mut egui::Ui, estimator: &mut Estimator) {
    ui.horizontal(|ui| {
        ui.selectable_value(estimator, Estimator::Filter, "filter")
            .on_hover_text(
                "Grid filter: the density p = exp(−V/ε) on the HoFFT \
                 grid — any shape, several maxima",
            );
        ui.selectable_value(estimator, Estimator::Window, "box tracker")
            .on_hover_text(
                "Translating window: the grid density on a small box \
                 ∏[−L_d, L_d] around the mode, re-centred by whole \
                 elements after every step (exact index shift). Exact \
                 inside the box, blind outside it; needs a Gaussian \
                 prior (σ > 0) to have a mode to follow",
            );
        ui.selectable_value(estimator, Estimator::Particles, "LParticles")
            .on_hover_text(
                "Particle solver: N particles drawn from the Gaussian initial \
                 data, moved by the flow plus a Brownian increment √(ε q dt), \
                 each killed when its accumulated misfit dt·γ·d exceeds an \
                 Exp(1) clock and reborn on a survivor (Fleming–Viot type)",
            );
        ui.selectable_value(estimator, Estimator::Tracker, "tracker")
            .on_hover_text(
                "Tracker: the Gaussian closure — one state x̂ and the \
                 curvature S of V at it (minimum-energy / EKF form, \
                 ε-free); cannot track several maxima",
            );
        ui.selectable_value(estimator, Estimator::Unscented, "unscented")
            .on_hover_text(
                "Unscented closure: the same Gaussian matched by its integrals \
                 (centre x̄, width Σ = εP) — the averages of the exact flow and \
                 observation computed by a quadrature rule at a few points, no \
                 Jacobian; equals the tracker on a linear model, keeps the \
                 second-order term of the flow across the bump otherwise; \
                 depends on ε (the points spread with √ε); needs σ > 0",
            );
    });
}

/// The configuration panels of `estimator`, editing `cfg` in place;
/// `steps` (the run length) bounds the snapshot slider and sizes the memory
/// estimate. Returns whether every grid domain is valid (min < max) — the
/// app disables the Run button of the grid filter otherwise.
pub fn config_ui(
    ui: &mut egui::Ui,
    cfg: &mut FilterConfig,
    estimator: Estimator,
    steps: usize,
) -> bool {
    let is_filter = estimator == Estimator::Filter;
    let is_window = estimator == Estimator::Window;
    let is_particles = estimator == Estimator::Particles;
    let is_tracker = estimator == Estimator::Tracker;
    let is_unscented = estimator == Estimator::Unscented;
    let mut domains_ok = true;
    egui::CollapsingHeader::new("Initial data")
        .id_salt("filter_init")
        .default_open(true)
        .show(ui, |ui| {
            ui.label("Gaussian p₀: center x_c and stiffness σ per variable")
                .on_hover_text(
                    "V₀ = Σ_d σ_d (x_d − x_c,d)²/2, i.e. \
                     p₀ ∝ exp(−V₀/ε): variance ε/σ_d in \
                     direction d. σ_d = 0 leaves that \
                     direction flat (no prior); all zero \
                     is the flat p₀. The center defaults \
                     to the run's initial state.",
                );
            egui::Grid::new("filter_init_grid")
                .num_columns(cfg.vars.len() + 1)
                .spacing([6.0, 2.0])
                .show(ui, |ui| {
                    ui.small("");
                    for v in &cfg.vars {
                        ui.small(&v.label);
                    }
                    ui.end_row();
                    ui.small("x_c");
                    for v in &mut cfg.vars {
                        ui.add(egui::DragValue::new(&mut v.x0).speed(0.02).max_decimals(3));
                    }
                    ui.end_row();
                    ui.small("σ");
                    for v in &mut cfg.vars {
                        ui.add(
                            egui::DragValue::new(&mut v.sigma)
                                .speed(0.1)
                                .range(0.0..=1e3),
                        );
                    }
                    ui.end_row();
                });
        });

    egui::CollapsingHeader::new("Observation")
        .id_salt("filter_obs")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut cfg.gamma)
                        .speed(0.02)
                        .range(0.0..=1e3),
                );
                ui.label("γ (observation weight)").on_hover_text(
                    "Scales the observation term γ·d(y, h(x))²/2 \
                     in both estimators: 1 is the standard cycle, \
                     larger trusts the observations more, 0 \
                     switches them off",
                );
            });
        });

    egui::CollapsingHeader::new("Diffusion")
        .id_salt("filter_diffusion")
        .default_open(true)
        .show(ui, |ui| {
            // ε: every estimator but the (ε-free) tracker;
            // the scheme: only the two grid solvers.
            ui.add_enabled_ui(!is_tracker, |ui| {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut cfg.eps)
                            .speed(0.001)
                            .range(1e-5..=10.0),
                    );
                    ui.label("ε (temperature)").on_hover_text(
                        "Diffusion coefficient ε/2 of p and scale \
                         of the observation weight; smaller sharpens \
                         p but needs a finer grid",
                    );
                });
            });
            ui.add_enabled_ui(is_filter || is_window, |ui| {
                ui.horizontal(|ui| {
                    use ode_observers::methods::mortensen::DiffusionScheme;
                    ui.selectable_value(
                        &mut cfg.diffusion,
                        DiffusionScheme::SplitEuler,
                        "split Euler",
                    )
                    .on_hover_text(
                        "Directionally split implicit Euler: \
                         nodal positivity of p",
                    );
                    ui.selectable_value(
                        &mut cfg.diffusion,
                        DiffusionScheme::Euler,
                        "unsplit Euler",
                    )
                    .on_hover_text("Implicit Euler on the full tensor Laplacian");
                    ui.label("scheme");
                });
            }); // grid solvers only (scheme)
            ui.label("model-noise variances q (diagonal of Q)")
                .on_hover_text(
                    "Diagonally anisotropic diffusion \
                     (ε/2)·Σ_d q_d ∂²_d p: 1 in every \
                     direction is the isotropic Laplacian, \
                     0 switches diffusion off in a direction",
                );
            egui::Grid::new("filter_q")
                .num_columns(cfg.vars.len().max(1))
                .spacing([6.0, 2.0])
                .show(ui, |ui| {
                    for v in &cfg.vars {
                        ui.small(&v.label);
                    }
                    ui.end_row();
                    for v in &mut cfg.vars {
                        ui.add(
                            egui::DragValue::new(&mut v.q)
                                .speed(0.05)
                                .range(0.0..=100.0)
                                .max_decimals(3),
                        );
                    }
                    ui.end_row();
                });
        });

    if is_filter {
        egui::CollapsingHeader::new("Grid")
            .id_salt("filter_grid")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("filter_vars")
                    .num_columns(6)
                    .spacing([6.0, 2.0])
                    .show(ui, |ui| {
                        ui.small("");
                        ui.small("min");
                        ui.small("max");
                        ui.small("n");
                        ui.small("order");
                        ui.small("wall").on_hover_text(
                            "Boundary condition of this direction: checked = \
                             Dirichlet wall, p = 0 on the domain boundary \
                             (V = +∞ outside: absorbing — mass reaching the \
                             wall is removed, and a grid point whose preimage \
                             leaves the box gets p = 0); unchecked = Neumann \
                             (reflecting). Periodic angles have no wall.",
                        );
                        ui.end_row();
                        for v in &mut cfg.vars {
                            // Periodic variables live on the
                            // fixed circle (−π, π): visible but
                            // not editable.
                            ui.label(&v.label).on_hover_text(if v.periodic {
                                "periodic angle: domain fixed to (−π, π), \
                                 periodic boundary conditions"
                            } else {
                                "domain chosen from the run's state range"
                            });
                            ui.add_enabled(
                                !v.periodic,
                                egui::DragValue::new(&mut v.domain.0)
                                    .speed(0.05)
                                    .max_decimals(2),
                            );
                            ui.add_enabled(
                                !v.periodic,
                                egui::DragValue::new(&mut v.domain.1)
                                    .speed(0.05)
                                    .max_decimals(2),
                            );
                            ui.add(egui::DragValue::new(&mut v.n_el).range(1..=200));
                            ui.add(egui::DragValue::new(&mut v.p_ord).range(1..=12));
                            ui.add_enabled(
                                !v.periodic,
                                egui::Checkbox::without_text(&mut v.dirichlet),
                            )
                            .on_hover_text(if v.periodic {
                                "periodic direction: no wall"
                            } else {
                                "Dirichlet wall: p = 0 on this direction's \
                                 domain boundary (absorbing)"
                            });
                            ui.end_row();
                            domains_ok &= v.domain.0 < v.domain.1;
                        }
                    });
            });
    } // Grid (filter only)

    if is_window {
        egui::CollapsingHeader::new("Window")
            .id_salt("box_window")
            .default_open(true)
            .show(ui, |ui| {
                ui.label("box ξ_d ∈ [−L_d, L_d] around the mode")
                    .on_hover_text(
                        "Half-width L_d per variable: a few widths of p, \
                         √(ε (S⁻¹)_dd) — the window is blind outside it. \
                         n elements of the given order per direction (n ≥ 3, \
                         odd puts the centre in one element). The mode must \
                         travel much less than one element per step.",
                    );
                egui::Grid::new("box_vars")
                    .num_columns(4)
                    .spacing([6.0, 2.0])
                    .show(ui, |ui| {
                        ui.small("");
                        ui.small("L");
                        ui.small("n");
                        ui.small("order");
                        ui.end_row();
                        for v in &mut cfg.vars {
                            ui.label(&v.label);
                            ui.add(
                                egui::DragValue::new(&mut v.half_width)
                                    .speed(0.02)
                                    .range(1e-3..=1e3)
                                    .max_decimals(3),
                            );
                            ui.add(egui::DragValue::new(&mut v.win_n_el).range(1..=100));
                            ui.add(egui::DragValue::new(&mut v.p_ord).range(1..=12));
                            ui.end_row();
                        }
                    });
                if cfg.vars.iter().any(|v| v.sigma <= 0.0) {
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        "σ = 0 in some direction: p₀ is flat there and the \
                         window has no mode to follow — set σ > 0 in Initial data.",
                    );
                }
            });
    } // Window (box tracker only)

    if is_particles {
        egui::CollapsingHeader::new("LParticles")
                                .id_salt("particles_panel")
                                .default_open(true)
                                .show(ui, |ui| {
                                    let pc = &mut cfg.particles;
                                    egui::Grid::new("particles_grid")
                                        .num_columns(2)
                                        .spacing([6.0, 2.0])
                                        .show(ui, |ui| {
                                            ui.add(egui::DragValue::new(&mut pc.n_particles).range(2..=200_000));
                                            ui.label("N particles").on_hover_text(
                                                "Drawn at t = 0 from the Gaussian initial data \
                                                 (centre x_c, variance ε/σ per direction; uniform \
                                                 in the domain where σ = 0); constant along the run",
                                            );
                                            ui.end_row();
                                        });
                                    ui.small(
                                        "Initial cloud: the Gaussian initial data (x_c, σ). Killing: each \
                                         particle has an Exp(1) clock consumed by dt·γ·d(y, h(x)) (γ from \
                                         the Observation panel) and is reborn on a survivor. New random seed \
                                         at every run.",
                                    );
                                    ui.small(format!(
                                        "stored: {} × {} positions per step",
                                        pc.n_particles,
                                        cfg.vars.len()
                                    ));
                                });
    } // Particles only

    if is_unscented {
        egui::CollapsingHeader::new("Quadrature")
            .id_salt("unscented_panel")
            .default_open(true)
            .show(ui, |ui| {
                quadrature_ui(ui, &mut cfg.unscented.rule, cfg.vars.len());
                if cfg.vars.iter().any(|v| v.sigma <= 0.0) {
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        "σ = 0 in some direction: the Gaussian has no width there — \
                         set σ > 0 in Initial data.",
                    );
                }
            });
    } // Quadrature (unscented only)

    if is_filter || is_window {
        egui::CollapsingHeader::new("Outputs")
            .id_salt("filter_outputs")
            .default_open(false)
            .show(ui, |ui| {
                ui.label("2D planes (max over the other directions)");
                let labels: Vec<String> = cfg.vars.iter().map(|v| v.label.clone()).collect();
                for ((a, b), on) in &mut cfg.pairs {
                    ui.checkbox(on, format!("{} × {}", labels[*a], labels[*b]));
                }
                ui.add(egui::Slider::new(&mut cfg.n_snapshots, 1..=steps + 1).text("snapshots"));

                // Memory estimate: the solver's working set
                // (p field + transport buffer + φ⁻¹ plan) and
                // the outputs kept in memory.
                let ndof: Vec<usize> = cfg
                    .vars
                    .iter()
                    .map(|v| {
                        if is_window {
                            (v.win_n_el * v.p_ord).saturating_sub(1).max(1)
                        } else {
                            v.n_el * v.p_ord + if v.periodic { 0 } else { 1 }
                        }
                    })
                    .collect();
                let dofs: usize = ndof.iter().product();
                let dim = ndof.len();
                let plane_dofs: usize = cfg
                    .pairs
                    .iter()
                    .filter(|(_, on)| *on)
                    .map(|((a, b), _)| ndof[*a] * ndof[*b])
                    .sum();
                let solver = dofs * (16 + 8 + 8 * dim); // p + buf + plan
                let outputs =
                    cfg.n_snapshots.min(steps + 1) * plane_dofs * 8 + (steps + 1) * dim * 8;
                ui.small(format!(
                    "grid {} = {} DOFs — solver ≈ {}, outputs ≈ {}",
                    ndof.iter()
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join("×"),
                    dofs,
                    fmt_bytes(solver),
                    fmt_bytes(outputs),
                ));
            });
    } // Outputs (filter and box tracker)
    domains_ok
}

/// The quadrature-rule selector of the unscented closure: the rule, its
/// parameter (the symmetric rule's spread, the Gauss–Hermite points per
/// direction) and the resulting number of evaluations per set in `dim`
/// dimensions.
pub fn quadrature_ui(ui: &mut egui::Ui, rule: &mut Quadrature, dim: usize) {
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("quadrature_rule")
            .selected_text(rule.name())
            .show_ui(ui, |ui| {
                let symmetric = matches!(rule, Quadrature::Symmetric { .. });
                if ui.selectable_label(symmetric, "symmetric").on_hover_text(
                    "The centre and the 2n points x̄ ± h s_i (s_i the columns of a square root \
                     of Σ), weights 1 − n/h² and 1/(2h²): exact to degree three for every h; \
                     h = √3 matches the fourth moment along each axis (Julier–Uhlmann's \
                     unscented transform is h² = n + κ)",
                ).clicked() && !symmetric {
                    *rule = Quadrature::DEFAULT;
                }
                ui.selectable_value(rule, Quadrature::Cubature, "cubature").on_hover_text(
                    "Spherical–radial cubature: the 2n points x̄ ± √n s_i with equal weights, \
                     no centre, no parameter, all weights positive; degree three \
                     (Arasaratnam–Haykin)",
                );
                ui.selectable_value(rule, Quadrature::DegreeFive, "degree five").on_hover_text(
                    "Fully symmetric degree-five rule (McNamee–Stenger): the centre, the 2n \
                     axis points and the 2n(n−1) pair points at spread √3; 2n² + 1 evaluations",
                );
                let hermite = matches!(rule, Quadrature::GaussHermite { .. });
                if ui.selectable_label(hermite, "Gauss–Hermite").on_hover_text(
                    "Tensor product of the one-dimensional Gauss–Hermite rule with m points per \
                     direction: mⁿ evaluations, exact to degree 2m − 1 (Ito–Xiong)",
                ).clicked() && !hermite {
                    *rule = Quadrature::GaussHermite { points: 3 };
                }
            });
        match rule {
            Quadrature::Symmetric { spread } => {
                ui.add(egui::DragValue::new(spread).speed(0.02).range(0.1..=10.0).max_decimals(3));
                ui.label("h").on_hover_text("Spread of the points in units of the width; √3 ≈ 1.732 by default");
                if ui.small_button("√3").clicked() {
                    *rule = Quadrature::DEFAULT;
                }
            }
            Quadrature::GaussHermite { points } => {
                ui.add(egui::DragValue::new(points).range(1..=9));
                ui.label("points per direction");
            }
            _ => {}
        }
    });
    let count = rule.count(dim);
    ui.small(format!("{count} evaluations of the flow and of h per step"));
    if count > 20_000 {
        ui.colored_label(ui.visuals().warn_fg_color, "many points: this run will be slow");
    }
}

/// Whether the estimator can start from the configuration's prior: the
/// translating window and the unscented closure need σ > 0 in every
/// direction (a flat p₀ has no mode to follow, no width to spread points
/// over); the others accept any prior.
pub fn prior_ok(cfg: &FilterConfig, estimator: Estimator) -> bool {
    !matches!(estimator, Estimator::Window | Estimator::Unscented) || cfg.vars.iter().all(|v| v.sigma > 0.0)
}

/// Label of the Run button.
pub fn run_label(estimator: Estimator) -> &'static str {
    match estimator {
        Estimator::Filter => "▶  Run filter",
        Estimator::Window => "▶  Run box tracker",
        Estimator::Particles => "▶  Run LParticles",
        Estimator::Tracker => "▶  Run tracker",
        Estimator::Unscented => "▶  Run unscented",
    }
}

/// Text of the progress bar while the estimator runs.
pub fn running_text(estimator: Estimator) -> &'static str {
    match estimator {
        Estimator::Filter => "filtering…",
        Estimator::Window => "box tracking…",
        Estimator::Particles => "LParticles…",
        Estimator::Tracker => "tracking…",
        Estimator::Unscented => "unscented…",
    }
}

/// The progress row of a running job: bar with `text`, elapsed / remaining
/// estimate from `progress` ∈ [0, 1] and `elapsed` seconds, and a Cancel
/// button. Returns `true` when Cancel was clicked (the app drops the run
/// handle, which raises the cancel flag).
pub fn progress_ui(ui: &mut egui::Ui, progress: f32, elapsed: f64, text: &str) -> bool {
    let mut cancel = false;
    ui.add(
        egui::ProgressBar::new(progress)
            .show_percentage()
            .text(text),
    );
    let frac = progress as f64;
    ui.horizontal(|ui| {
        ui.small(if frac > 0.01 {
            format!(
                "elapsed {} — remaining ~{}",
                fmt_duration(elapsed),
                fmt_duration(elapsed * (1.0 - frac) / frac),
            )
        } else {
            format!("elapsed {} — estimating…", fmt_duration(elapsed))
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Dropping the handle raises the cancel flag: the
            // worker stops after its current iteration and
            // its partial result is discarded.
            if ui
                .small_button("✖ Cancel")
                .on_hover_text("Stop this run after the current iteration (result discarded)")
                .clicked()
            {
                cancel = true;
            }
        });
    });
    cancel
}

/// Human-readable byte count (B / kB / MB / GB).
fn fmt_bytes(bytes: usize) -> String {
    let b = bytes as f64;
    if b < 1e3 {
        format!("{bytes} B")
    } else if b < 1e6 {
        format!("{:.1} kB", b / 1e3)
    } else if b < 1e9 {
        format!("{:.1} MB", b / 1e6)
    } else {
        format!("{:.2} GB", b / 1e9)
    }
}

/// Human-readable duration ("42 s", "3 min 05 s", "1 h 12 min").
fn fmt_duration(secs: f64) -> String {
    let s = secs.round().max(0.0) as u64;
    if s < 60 {
        format!("{s} s")
    } else if s < 3600 {
        format!("{} min {:02} s", s / 60, s % 60)
    } else {
        format!("{} h {:02} min", s / 3600, (s % 3600) / 60)
    }
}
