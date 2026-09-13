//! Central-panel view "Tracking density": the translating window's density
//! on a selectable plane, drawn where the window is. The window's box and
//! mesh (dark green) and its estimate trajectory x̂ + argmax ρ (dark green)
//! move with the mode over a white background; the reference trajectory is
//! black; the tracker's estimate can be overlaid in red. The heatmap inside
//! the box is the max-marginal of ρ at the snapshot nearest the playback
//! time, redisplayed through the window's spectral-element basis (same
//! resampler as the filter view), and can be hidden to see the mesh alone.

use crate::filter_view::{ColorFocus, TRACKER_RED, heatmap_image, mesh_lines};
use ode_observers::jobs::{BoxOutput, TrackerOutput};
use std::collections::HashMap;

/// The window and its estimate: dark green, the same in both themes (the
/// plot background is forced white).
const WINDOW_GREEN: egui::Color32 = egui::Color32::from_rgb(0, 100, 0);
/// The reference trajectory.
const REFERENCE_BLACK: egui::Color32 = egui::Color32::BLACK;

/// Persistent state of the window view: plane choice, the overlay toggles
/// and the texture cache (keyed by snapshot × plane; cleared when a new
/// window output arrives).
pub struct BoxView {
    /// Overlay the tracker's estimate in red when a tracker run exists.
    pub show_tracker: bool,
    /// Draw the density inside the window (off: mesh and box only).
    pub show_density: bool,
    /// Draw the window's estimate trajectory x̂ + argmax ρ (dark green).
    pub show_estimate: bool,
    /// Color-scale focus, see [`ColorFocus`].
    pub focus: ColorFocus,
    plane: usize,
    cache: HashMap<(usize, usize), egui::TextureHandle>,
    /// The color floor the cached textures were built with.
    cached_floor: f64,
}

impl Default for BoxView {
    fn default() -> Self {
        BoxView {
            show_tracker: true,
            show_density: true,
            show_estimate: true,
            focus: ColorFocus::default(),
            plane: 0,
            cache: HashMap::new(),
            cached_floor: 0.0,
        }
    }
}

impl BoxView {
    /// Drop the cached textures (the output they were built from is gone).
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Forget cached textures and reset the plane for a fresh output.
    pub fn reset(&mut self, _out: &BoxOutput) {
        self.clear_cache();
        self.plane = 0;
    }

    /// Render the view at playback `step`.
    pub fn ui(&mut self, ui: &mut egui::Ui, out: &BoxOutput, tracker: Option<&TrackerOutput>, step: usize) {
        let w = &out.window;
        if w.pairs.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("No 2D plane was stored by this window run.");
            });
            return;
        }
        // Snapshot nearest to the playback step, and the window's position
        // at that snapshot (the density is drawn where it was computed).
        let snap = w
            .saved_steps
            .iter()
            .enumerate()
            .min_by_key(|&(_, &s)| s.abs_diff(step))
            .map(|(k, _)| k)
            .unwrap_or(0);
        let snap_step = w.saved_steps[snap].min(out.centers.len() - 1);
        let t_snap = w.saved_steps[snap] as f64 * w.dt;

        ui.horizontal(|ui| {
            self.plane = self.plane.min(w.pairs.len() - 1);
            let pair_label = |&(a, b): &(usize, usize)| format!("{} × {}", w.labels[a], w.labels[b]);
            egui::ComboBox::from_id_salt("box_plane")
                .width(ui.available_width().min(180.0))
                .selected_text(pair_label(&w.pairs[self.plane]))
                .show_ui(ui, |ui| {
                    for (j, pair) in w.pairs.iter().enumerate() {
                        ui.selectable_value(&mut self.plane, j, pair_label(pair));
                    }
                });
            ui.small(format!(
                "snapshot {}/{} at t = {t_snap:.2} s (ρ normalized by its max)",
                snap + 1,
                w.saved_steps.len(),
            ));
            ui.separator();
            ui.checkbox(&mut self.show_density, "density")
                .on_hover_text("Draw the window's density ρ inside the box (off: box and mesh only)");
            ui.checkbox(&mut self.show_estimate, "window x̂")
                .on_hover_text("Draw the window's estimate x̂ + argmax ρ up to the playback time (dark green)");
            ui.add_enabled(tracker.is_some(), egui::Checkbox::new(&mut self.show_tracker, "tracker x̂"))
                .on_hover_text(if tracker.is_some() {
                    "Overlay the tracker's estimate x̂(t) up to the playback time, in red"
                } else {
                    "Run the tracker to overlay its estimate"
                });
            ui.separator();
            self.focus.ui(ui);
        });
        let floor = self.focus.floor();
        if floor != self.cached_floor {
            self.cache.clear();
            self.cached_floor = floor;
        }

        let plane = self.plane;
        let (a, b) = w.pairs[plane];
        let (la, lb) = (w.domain[a].1, w.domain[b].1); // half-widths
        let tex = self
            .cache
            .entry((snap, plane))
            .or_insert_with(|| {
                let image = heatmap_image(w, snap, plane, floor);
                ui.ctx().load_texture(format!("box2d_{snap}_{plane}"), image, egui::TextureOptions::LINEAR)
            })
            .clone();

        // Plot bounds: the reference and every window position, with margin.
        let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for s in &w.reference {
            x0 = x0.min(s[a]);
            x1 = x1.max(s[a]);
            y0 = y0.min(s[b]);
            y1 = y1.max(s[b]);
        }
        for c in &out.centers {
            x0 = x0.min(c[a] - la);
            x1 = x1.max(c[a] + la);
            y0 = y0.min(c[b] - lb);
            y1 = y1.max(c[b] + lb);
        }
        let (mx, my) = (0.05 * (x1 - x0).max(1e-9), 0.05 * (y1 - y0).max(1e-9));
        let (x0, x1, y0, y1) = (x0 - mx, x1 + mx, y0 - my, y1 + my);

        // Window at the snapshot: box, mesh, density.
        let c = &out.centers[snap_step];
        let (cx, cy) = (c[a], c[b]);
        let box_x = (cx - la, cx + la);
        let box_y = (cy - lb, cy + lb);
        let mesh_x: Vec<(f64, bool)> = mesh_lines(w, a).into_iter().map(|(x, e)| (cx + x, e)).collect();
        let mesh_y: Vec<(f64, bool)> = mesh_lines(w, b).into_iter().map(|(y, e)| (cy + y, e)).collect();

        // Trajectories up to the playback step.
        let clip = step.min(w.reference.len() - 1);
        let reference: Vec<[f64; 2]> = w.reference[..=clip].iter().map(|s| [s[a], s[b]]).collect();
        let clip_e = step.min(w.estimates.len() - 1);
        let estimate: Vec<[f64; 2]> = w.estimates[..=clip_e].iter().map(|s| [s[a], s[b]]).collect();
        let tracker_path = tracker.filter(|_| self.show_tracker).map(|t| {
            let clip_t = step.min(t.estimates.len().saturating_sub(1));
            let pts: Vec<[f64; 2]> = t.estimates[..=clip_t].iter().map(|e| [e[a], e[b]]).collect();
            let last = &t.estimates[clip_t];
            (pts, [last[a], last[b]])
        });
        let show_density = self.show_density;
        let show_estimate = self.show_estimate;
        let strong = WINDOW_GREEN;
        let faint = WINDOW_GREEN.gamma_multiply(0.35);

        // White background whatever the theme: light visuals for the plot's
        // own text and axes, and an explicit white frame behind it.
        ui.scope(|ui| {
            *ui.visuals_mut() = egui::Visuals::light();
            egui::Frame::new().fill(egui::Color32::WHITE).show(ui, |ui| {
                egui_plot::Plot::new("box_plot")
                    .x_axis_label(&w.labels[a])
                    .y_axis_label(&w.labels[b])
                    .show_background(false)
                    .show_grid(false)
                    .set_margin_fraction(egui::vec2(0.0, 0.0))
                    .allow_drag(false)
                    .allow_zoom(false)
                    .allow_scroll(false)
                    .allow_boxed_zoom(false)
                    .allow_double_click_reset(false)
                    .show(ui, |plot_ui| {
                        plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max([x0, y0], [x1, y1]));
                        if show_density {
                            plot_ui.image(egui_plot::PlotImage::new(
                                "ρ",
                                &tex,
                                egui_plot::PlotPoint::new(cx, cy),
                                egui::vec2((2.0 * la) as f32, (2.0 * lb) as f32),
                            ));
                        }
                        // The window's mesh, clipped to the box: element
                        // boundaries strong, interior nodes faint.
                        for &(x, boundary) in &mesh_x {
                            let (wd, col) = if boundary { (1.2, strong) } else { (0.5, faint) };
                            plot_ui.line(
                                egui_plot::Line::new("", vec![[x, box_y.0], [x, box_y.1]]).color(col).width(wd),
                            );
                        }
                        for &(y, boundary) in &mesh_y {
                            let (wd, col) = if boundary { (1.2, strong) } else { (0.5, faint) };
                            plot_ui.line(
                                egui_plot::Line::new("", vec![[box_x.0, y], [box_x.1, y]]).color(col).width(wd),
                            );
                        }
                        plot_ui.line(
                            egui_plot::Line::new(
                                "window",
                                vec![
                                    [box_x.0, box_y.0],
                                    [box_x.1, box_y.0],
                                    [box_x.1, box_y.1],
                                    [box_x.0, box_y.1],
                                    [box_x.0, box_y.0],
                                ],
                            )
                            .color(strong)
                            .width(2.0),
                        );
                        // Reference (black) and window estimate (dark green).
                        plot_ui.line(egui_plot::Line::new("reference", reference).color(REFERENCE_BLACK).width(1.5));
                        let s = &w.reference[clip];
                        plot_ui.points(
                            egui_plot::Points::new("", vec![[s[a], s[b]]]).radius(4.0).color(REFERENCE_BLACK),
                        );
                        if show_estimate {
                            plot_ui.line(egui_plot::Line::new("window x̂", estimate).color(strong).width(1.5));
                            let e = &w.estimates[clip_e];
                            plot_ui.points(
                                egui_plot::Points::new("", vec![[e[a], e[b]]])
                                    .radius(4.0)
                                    .shape(egui_plot::MarkerShape::Square)
                                    .color(strong),
                            );
                        }
                        if let Some((path, last)) = tracker_path {
                            plot_ui.line(egui_plot::Line::new("tracker x̂", path).color(TRACKER_RED).width(1.5));
                            plot_ui.points(
                                egui_plot::Points::new("", vec![last])
                                    .radius(4.0)
                                    .shape(egui_plot::MarkerShape::Diamond)
                                    .color(TRACKER_RED),
                            );
                        }
                    });
            });
        });
    }
}
