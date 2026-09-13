//! Central-panel view of the stored filter outputs: up to two side-by-side
//! heatmap panes, each showing the max-marginal of p on a selectable plane
//! at the snapshot nearest the playback time, with the reference trajectory
//! overlaid in plot coordinates. Heatmaps are redisplayed through the run's
//! own spectral-element basis (per-element Lagrange interpolation, as in
//! scripts/visualize_filter.py), so figures keep the solver's resolution.

use ode_observers::jobs::{FilterOutput, TrackerOutput};
use std::collections::HashMap;

/// Pixel resolution (per side) of the resampled heatmap textures.
const RES: usize = 220;

/// Color of the tracker overlay (red in both themes, as asked: the tracker
/// estimate is a foreign object on the density plot, not a plot series).
pub(crate) const TRACKER_RED: egui::Color32 = egui::Color32::from_rgb(220, 40, 40);

/// Persistent state of the filter view: pane count, per-pane plane choice,
/// the tracker-overlay toggle, and the texture cache (keyed by snapshot ×
/// plane; cleared when a new filter output arrives).
pub struct FilterView {
    pub n_panes: usize,
    /// Draw the tracker's estimate x̂(t) on top of the density when a
    /// tracker run is available.
    pub show_tracker: bool,
    /// Show an additional pane with the tracker's Gaussian density on a
    /// plane of its own, when a tracker run is available.
    pub show_tracker_density: bool,
    /// Color-scale focus, see [`ColorFocus`].
    pub focus: ColorFocus,
    plane: [usize; 2],
    tracker_plane: usize,
    cache: HashMap<(usize, usize), egui::TextureHandle>,
    /// Tracker-density textures, keyed by (tracker step, plane).
    tracker_cache: HashMap<(usize, usize), egui::TextureHandle>,
    /// The color floor the cached textures were built with.
    cached_floor: f64,
}

impl Default for FilterView {
    fn default() -> Self {
        FilterView {
            n_panes: 2,
            show_tracker: true,
            show_tracker_density: true,
            focus: ColorFocus::default(),
            plane: [0, 1],
            tracker_plane: 0,
            cache: HashMap::new(),
            tracker_cache: HashMap::new(),
            cached_floor: 0.0,
        }
    }
}

/// Color-scale floor of the density heatmaps: the colormap of each
/// displayed snapshot runs from `fraction · max` to `max` (max = the
/// largest |p| of the snapshot), every value below the floor taking the
/// lowest color — the color dynamics is spent on the peak of p. 0 (the
/// default) is the plain 0-to-max scale.
#[derive(Clone, Copy, PartialEq)]
pub struct ColorFocus {
    /// Lower end of the scale as a fraction of the max, in [0, 0.99].
    pub fraction: f64,
}

impl Default for ColorFocus {
    fn default() -> Self {
        ColorFocus { fraction: 0.0 }
    }
}

impl ColorFocus {
    /// The floor in effect.
    pub fn floor(&self) -> f64 {
        self.fraction.clamp(0.0, 0.99)
    }

    /// The slider row shared by the density views.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("color floor").on_hover_text(
            "Lower end of the color scale as a fraction of the snapshot's max: the colormap \
             runs from (fraction · max) to max, values below take the lowest color — raise it \
             to spend the colors on the peak; 0 is the full 0-to-max scale",
        );
        ui.add(egui::Slider::new(&mut self.fraction, 0.0..=0.99).step_by(0.01).show_value(true));
    }
}

impl FilterView {
    /// Drop the cached textures (the output they were built from is gone).
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.tracker_cache.clear();
    }

    /// Drop the cached textures if the color floor changed since they were
    /// built, and record the floor in effect.
    fn sync_floor(&mut self) {
        let floor = self.focus.floor();
        if floor != self.cached_floor {
            self.clear_cache();
            self.cached_floor = floor;
        }
    }

    /// Drop the cached tracker-density textures (a new tracker run arrived).
    pub fn clear_tracker_cache(&mut self) {
        self.tracker_cache.clear();
    }

    /// Forget cached textures and reset the pane planes for a fresh output.
    pub fn reset(&mut self, out: &FilterOutput) {
        self.clear_cache();
        self.plane = [0, 1.min(out.pairs.len().saturating_sub(1))];
        self.tracker_plane = 0;
        self.n_panes = self.n_panes.clamp(1, 2);
    }

    /// Render the view; `step` is the current playback step (used to pick
    /// the nearest stored snapshot and to clip the overlay trajectories).
    /// `tracker`, when a tracker run exists, is overlaid as the estimate
    /// trajectory x̂(t) in red if `show_tracker` is on, and shown as an
    /// additional pane — the Gaussian density of the closure on a chosen
    /// plane — if `show_tracker_density` is on. The plots are locked to
    /// the filter's domain box.
    pub fn ui(&mut self, ui: &mut egui::Ui, out: &FilterOutput, tracker: Option<&TrackerOutput>, step: usize) {
        if out.pairs.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("No 2D plane was stored by this filter run.");
            });
            return;
        }
        // Snapshot nearest to the playback step.
        let snap = out
            .saved_steps
            .iter()
            .enumerate()
            .min_by_key(|&(_, &s)| s.abs_diff(step))
            .map(|(k, _)| k)
            .unwrap_or(0);
        let t_snap = out.saved_steps[snap] as f64 * out.dt;
        ui.horizontal(|ui| {
            ui.small(format!(
                "snapshot {}/{} at t = {t_snap:.2} s (p normalized by its max)",
                snap + 1,
                out.saved_steps.len(),
            ));
            ui.separator();
            ui.add_enabled(tracker.is_some(), egui::Checkbox::new(&mut self.show_tracker, "tracker x̂"))
                .on_hover_text(if tracker.is_some() {
                    "Overlay the tracker's estimate x̂(t) up to the playback time, in red \
                     (the Gaussian closure's single mode, to compare with the density)"
                } else {
                    "Run the tracker to overlay its estimate on the density"
                });
            ui.add_enabled(
                tracker.is_some(),
                egui::Checkbox::new(&mut self.show_tracker_density, "tracker density"),
            )
            .on_hover_text(if tracker.is_some() {
                "Additional pane: the density implied by the tracker at the playback time, \
                 p ∝ exp(−½ (x−x̂)ᵀ S (x−x̂) / ε) — a Gaussian of covariance εP, P = S⁻¹ — \
                 max-marginalized on the selected plane (the (a, b) block of P) and \
                 normalized by its max, like the filter's snapshots"
            } else {
                "Run the tracker to show its Gaussian density next to the filter's"
            });
            ui.separator();
            self.focus.ui(ui);
        });
        self.sync_floor();
        let overlay = if self.show_tracker { tracker } else { None };
        let density = if self.show_tracker_density { tracker } else { None };

        let n = self.n_panes.clamp(1, 2).min(out.pairs.len());
        let n_cols = n + usize::from(density.is_some());
        ui.columns(n_cols, |cols| {
            for (i, col) in cols.iter_mut().enumerate() {
                if i < n {
                    self.pane(col, out, overlay, i, snap, step);
                } else if let Some(t) = density {
                    self.tracker_pane(col, out, t, overlay, step);
                }
            }
        });
    }

    /// Plane selector shared by the panes; returns the selected plane index.
    fn plane_selector(
        ui: &mut egui::Ui,
        out: &FilterOutput,
        sel: &mut usize,
        id: impl std::hash::Hash + std::fmt::Debug,
    ) -> usize {
        let pair_label = |&(a, b): &(usize, usize)| format!("{} × {}", out.labels[a], out.labels[b]);
        *sel = (*sel).min(out.pairs.len() - 1);
        egui::ComboBox::from_id_salt(id)
            .width(ui.available_width().min(180.0))
            .selected_text(pair_label(&out.pairs[*sel]))
            .show_ui(ui, |ui| {
                for (j, pair) in out.pairs.iter().enumerate() {
                    ui.selectable_value(sel, j, pair_label(pair));
                }
            });
        *sel
    }

    /// Tracker estimate up to the playback step, on plane (a, b): the path
    /// and its last point (the tracker run may have a different length than
    /// the filter's: clip to its own history).
    fn tracker_path(tracker: Option<&TrackerOutput>, step: usize, (a, b): (usize, usize)) -> Option<(egui_plot::PlotPoints<'static>, [f64; 2])> {
        tracker.map(|t| {
            let clip_t = step.min(t.estimates.len().saturating_sub(1));
            let last = &t.estimates[clip_t];
            (
                t.estimates[..=clip_t].iter().map(|e| [e[a], e[b]]).collect(),
                [last[a], last[b]],
            )
        })
    }

    /// A filter pane: the max-marginal snapshot of the selected plane.
    fn pane(
        &mut self,
        ui: &mut egui::Ui,
        out: &FilterOutput,
        tracker: Option<&TrackerOutput>,
        pane: usize,
        snap: usize,
        step: usize,
    ) {
        let plane = Self::plane_selector(ui, out, &mut self.plane[pane], ("filter_plane", pane));
        let tex = self
            .cache
            .entry((snap, plane))
            .or_insert_with(|| {
                let image = heatmap_image(out, snap, plane, self.focus.floor());
                ui.ctx().load_texture(format!("p2d_{snap}_{plane}"), image, egui::TextureOptions::LINEAR)
            })
            .clone();
        let path = Self::tracker_path(tracker, step, out.pairs[plane]);
        draw_plane(ui, out, plane, &tex, path, step, ("filter_plot", pane));
    }

    /// The tracker pane: the Gaussian density of the closure at the
    /// playback step, on its own selected plane.
    fn tracker_pane(
        &mut self,
        ui: &mut egui::Ui,
        out: &FilterOutput,
        tracker: &TrackerOutput,
        overlay: Option<&TrackerOutput>,
        step: usize,
    ) {
        ui.horizontal(|ui| {
            Self::plane_selector(ui, out, &mut self.tracker_plane, "tracker_density_plane");
            ui.small("tracker density");
        });
        let plane = self.tracker_plane;
        let step_t = step.min(tracker.estimates.len().saturating_sub(1));
        let tex = self
            .tracker_cache
            .entry((step_t, plane))
            .or_insert_with(|| {
                let image = tracker_image(out, tracker, step_t, plane, self.focus.floor());
                ui.ctx().load_texture(format!("tracker2d_{step_t}_{plane}"), image, egui::TextureOptions::LINEAR)
            })
            .clone();
        let path = Self::tracker_path(overlay, step, out.pairs[plane]);
        draw_plane(ui, out, plane, &tex, path, step, "tracker_density_plot");
    }
}

/// Draw one plane: the heatmap `tex` over the filter's domain box, the
/// filter's mesh, the reference up to `step`, and the optional tracker path.
fn draw_plane(
    ui: &mut egui::Ui,
    out: &FilterOutput,
    plane: usize,
    tex: &egui::TextureHandle,
    tracker_path: Option<(egui_plot::PlotPoints<'static>, [f64; 2])>,
    step: usize,
    id: impl std::hash::Hash + std::fmt::Debug,
) {
    let (a, b) = out.pairs[plane];
    let (x0, x1) = out.domain[a];
    let (y0, y1) = out.domain[b];
    let center = egui_plot::PlotPoint::new(0.5 * (x0 + x1), 0.5 * (y0 + y1));
    let size = egui::vec2((x1 - x0) as f32, (y1 - y0) as f32);

    let neutral = ui.visuals().text_color();
    let clip = step.min(out.reference.len() - 1);
    let reference: egui_plot::PlotPoints = out.reference[..=clip].iter().map(|s| [s[a], s[b]]).collect();

    // Grid = the filter's actual mesh, drawn explicitly (egui_plot's own
    // grid fades lines that are close on screen, which would hide the
    // Gauss–Lobatto clustering at high order): element boundaries strong,
    // interior nodes faint. Only the boundaries feed the axis tick labels.
    let mesh_x = mesh_lines(out, a);
    let mesh_y = mesh_lines(out, b);
    let x_marks = boundary_marks(&mesh_x, out, a);
    let y_marks = boundary_marks(&mesh_y, out, b);
    let strong = neutral.gamma_multiply(0.55);
    let faint = neutral.gamma_multiply(0.22);

    // The plot is fixed to the filter's domain box: no drag, zoom or
    // scroll — it always shows exactly the discretized region.
    egui_plot::Plot::new(id)
        .x_axis_label(&out.labels[a])
        .y_axis_label(&out.labels[b])
        .set_margin_fraction(egui::vec2(0.0, 0.0))
        .show_grid(false)
        .x_grid_spacer(move |_| x_marks.clone())
        .y_grid_spacer(move |_| y_marks.clone())
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .allow_double_click_reset(false)
        .show(ui, |plot_ui| {
            plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max([x0, y0], [x1, y1]));
            plot_ui.image(egui_plot::PlotImage::new("p", tex, center, size));
            for &(x, boundary) in &mesh_x {
                let (w, c) = if boundary { (1.0, strong) } else { (0.5, faint) };
                plot_ui.vline(egui_plot::VLine::new("", x).color(c).width(w));
            }
            for &(y, boundary) in &mesh_y {
                let (w, c) = if boundary { (1.0, strong) } else { (0.5, faint) };
                plot_ui.hline(egui_plot::HLine::new("", y).color(c).width(w));
            }
            plot_ui.line(egui_plot::Line::new("reference", reference).color(neutral).width(1.5));
            let s = &out.reference[clip];
            plot_ui.points(egui_plot::Points::new("", vec![[s[a], s[b]]]).radius(4.0).color(neutral));
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
}

/// The tracker's density on plane (a, b) at `step`, as a RES×RES image over
/// the filter's domain box: p ∝ exp(−½ ξᵀ P_ab⁻¹ ξ / ε) with ξ = x − x̂ and
/// P_ab the (a, b) block of P = S⁻¹ — the max-marginal of the closure's
/// Gaussian (minimizing the quadratic form over the other directions
/// leaves the Schur complement of S, i.e. the inverse of that block) —
/// normalized by its max (1 at x̂), through the colormap with the color
/// floor `floor` (see [`shade`]). Flat (uniform) where the covariance is
/// undefined (S singular) or the block degenerate.
fn tracker_image(
    out: &FilterOutput,
    tracker: &TrackerOutput,
    step: usize,
    plane: usize,
    floor: f64,
) -> egui::ColorImage {
    let (a, b) = out.pairs[plane];
    let (x0, x1) = out.domain[a];
    let (y0, y1) = out.domain[b];
    let m = out.labels.len();
    let e = &tracker.estimates[step];
    // Inverse of the 2×2 block, scaled by 1/(2ε); None ⇒ flat image.
    let quad = tracker.covariances[step].as_ref().and_then(|p| {
        let (paa, pab, pbb) = (p[a * m + a], p[a * m + b], p[b * m + b]);
        let det = paa * pbb - pab * pab;
        (det > 0.0 && tracker.eps > 0.0).then(|| {
            let s = 1.0 / (2.0 * tracker.eps * det);
            (pbb * s, -pab * s, paa * s) // coefficients of ξ₁², 2ξ₁ξ₂, ξ₂² in q/(2ε)
        })
    });
    // On a periodic direction the offset to the mode is taken to the
    // nearest image of x̂, so the drawn Gaussian is periodic like the
    // density it stands next to (the tracker itself works with unwrapped
    // angles, which is harmless: its flow and innovation are 2π-periodic).
    let wrap = |u: f64, d: usize| {
        if out.periodic[d] {
            let l = out.domain[d].1 - out.domain[d].0;
            (u + 0.5 * l).rem_euclid(l) - 0.5 * l
        } else {
            u
        }
    };
    let mut pixels = Vec::with_capacity(RES * RES);
    for r in 0..RES {
        // Image rows run top to bottom = decreasing b.
        let y = y0 + (y1 - y0) * ((RES - 1 - r) as f64 + 0.5) / RES as f64;
        for c in 0..RES {
            let x = x0 + (x1 - x0) * (c as f64 + 0.5) / RES as f64;
            let v = match quad {
                Some((qaa, qab, qbb)) => {
                    let (u, w) = (wrap(x - e[a], a), wrap(y - e[b], b));
                    (-(qaa * u * u + 2.0 * qab * u * w + qbb * w * w)).exp()
                }
                None => 1.0,
            };
            pixels.push(shade(v, floor));
        }
    }
    egui::ColorImage::new([RES, RES], pixels)
}

/// The mesh lines of direction `d`: every Gauss–Lobatto node of the run's
/// axis, flagged `true` at element boundaries (every p_ord-th node of the
/// *full* numbering — a Dirichlet axis starts at full node 1, its two
/// boundary zeros being unstored). Endpoints absent from the axis are added
/// as boundaries: the right one in a periodic direction (identified with
/// node 0), both in a Dirichlet direction.
pub(crate) fn mesh_lines(out: &FilterOutput, d: usize) -> Vec<(f64, bool)> {
    let p = out.p_ord[d].max(1);
    let shift = usize::from(out.dirichlet[d]);
    let mut lines: Vec<(f64, bool)> = out.axes[d]
        .iter()
        .enumerate()
        .map(|(g, &x)| (x, (g + shift) % p == 0))
        .collect();
    if out.periodic[d] {
        lines.push((out.domain[d].1, true));
    } else if out.dirichlet[d] {
        lines.push((out.domain[d].0, true));
        lines.push((out.domain[d].1, true));
    }
    lines
}

/// Axis tick marks at the element boundaries only (the plot's grid lines
/// are drawn separately by the pane).
fn boundary_marks(lines: &[(f64, bool)], out: &FilterOutput, d: usize) -> Vec<egui_plot::GridMark> {
    let (x0, x1) = out.domain[d];
    let step_size = (x1 - x0) / out.n_el[d].max(1) as f64;
    lines
        .iter()
        .filter(|&&(_, boundary)| boundary)
        .map(|&(value, _)| egui_plot::GridMark { value, step_size })
        .collect()
}

/// Per-pixel spectral evaluation data of one axis: for each of the RES
/// uniform sample points, the global node indices of its element's p+1
/// Lobatto nodes and the Lagrange basis weights at the point.
struct AxisEval {
    nodes: Vec<Vec<usize>>,
    weights: Vec<Vec<f64>>,
}

/// Locate each sample point's element and evaluate the element's degree-p
/// Lagrange basis there. In a periodic direction the last element's right
/// node wraps to global node 0, at coordinate `x1`. In a Dirichlet
/// direction the axis stores only the interior nodes (full node g ↦ stored
/// index g − 1): the interpolation still runs on all p + 1 element nodes,
/// but the two boundary nodes carry the value 0, so their (node, weight)
/// pairs are simply dropped from the contraction.
fn axis_eval(out: &FilterOutput, d: usize) -> AxisEval {
    let axis = &out.axes[d];
    let (n_el, p) = (out.n_el[d], out.p_ord[d]);
    let (x0, x1) = out.domain[d];
    let h = (x1 - x0) / n_el as f64;
    let na = axis.len();
    let dirichlet = out.dirichlet[d];

    let mut nodes = Vec::with_capacity(RES);
    let mut weights = Vec::with_capacity(RES);
    for pix in 0..RES {
        let x = x0 + (x1 - x0) * (pix as f64 + 0.5) / RES as f64;
        let e = (((x - x0) / h).floor() as usize).min(n_el - 1);
        // Stored index (`None` = Dirichlet boundary zero) and coordinate of
        // the element's p + 1 nodes.
        let elem: Vec<(Option<usize>, f64)> = (0..=p)
            .map(|i| {
                let g = e * p + i;
                if dirichlet {
                    if g == 0 {
                        (None, x0)
                    } else if g == n_el * p {
                        (None, x1)
                    } else {
                        (Some(g - 1), axis[g - 1])
                    }
                } else if g == na {
                    (Some(0), x1) // periodic wrap: right endpoint ≡ left node
                } else {
                    (Some(g), axis[g])
                }
            })
            .collect();
        let xs: Vec<f64> = elem.iter().map(|&(_, x)| x).collect();
        let (idx, w): (Vec<usize>, Vec<f64>) = elem
            .iter()
            .enumerate()
            .filter_map(|(i, &(g, _))| {
                let g = g?;
                let w: f64 = (0..=p)
                    .filter(|&j| j != i)
                    .map(|j| (x - xs[j]) / (xs[i] - xs[j]))
                    .product();
                Some((g, w))
            })
            .unzip();
        nodes.push(idx);
        weights.push(w);
    }
    AxisEval { nodes, weights }
}

/// Resample the (snapshot, plane) marginal onto a uniform RES×RES image
/// through the run's own spectral-element basis (tensor-product Lagrange
/// interpolation per element), normalized by the largest nodal |value|,
/// through the colormap with the color floor `floor` (see [`shade`]).
/// Polynomial over/undershoot is clamped by the colormap.
pub(crate) fn heatmap_image(out: &FilterOutput, snap: usize, plane: usize, floor: f64) -> egui::ColorImage {
    let (a, b) = out.pairs[plane];
    let marg = &out.marginals[snap][plane]; // na × nb, direction a fastest
    let na = out.axes[a].len();
    let max = marg.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300);

    let ea = axis_eval(out, a);
    let eb = axis_eval(out, b);

    let mut pixels = Vec::with_capacity(RES * RES);
    // Image rows run top to bottom = decreasing b.
    for r in 0..RES {
        let (bn, bw) = (&eb.nodes[RES - 1 - r], &eb.weights[RES - 1 - r]);
        for c in 0..RES {
            let (an, aw) = (&ea.nodes[c], &ea.weights[c]);
            let mut v = 0.0;
            for (gb, wb) in bn.iter().zip(bw) {
                let row = gb * na;
                let mut s = 0.0;
                for (ga, wa) in an.iter().zip(aw) {
                    s += wa * marg[ga + row];
                }
                v += wb * s;
            }
            pixels.push(shade(v / max, floor));
        }
    }
    egui::ColorImage::new([RES, RES], pixels)
}

/// Color of a normalized value v ∈ [0, 1] under the color floor `floor`:
/// the colormap runs from `floor` to 1 (v ≤ floor gives the lowest color,
/// v = 1 the highest); `floor = 0` is the plain colormap.
pub(crate) fn shade(v: f64, floor: f64) -> egui::Color32 {
    let floor = floor.clamp(0.0, 0.99);
    colormap(((v - floor) / (1.0 - floor)) as f32)
}

/// Colormap: blue (0) → green (0.5) → yellow (1), same in both themes.
fn colormap(t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let blue = egui::Rgba::from_srgba_unmultiplied(0x27, 0x44, 0xa8, 0xff);
    let green = egui::Rgba::from_srgba_unmultiplied(0x1e, 0x9c, 0x50, 0xff);
    let yellow = egui::Rgba::from_srgba_unmultiplied(0xfd, 0xe7, 0x25, 0xff);
    if t < 0.5 {
        let s = 2.0 * t;
        egui::Color32::from(blue * (1.0 - s) + green * s)
    } else {
        let s = 2.0 * t - 1.0;
        egui::Color32::from(green * (1.0 - s) + yellow * s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ode_observers::jobs::TrackerOutput;

    /// The tracker pane's image is the closure's Gaussian on the plane:
    /// 1 at x̂, exp(−½ ξᵀ P_ab⁻¹ ξ / ε) elsewhere, with P_ab the plane's
    /// block of P; flat when the covariance is undefined.
    #[test]
    fn tracker_image_is_the_plane_gaussian() {
        let out = make_out([4, 4], [3, 3], [false, false], [(-2.0, 2.0), (-2.0, 2.0)]);
        let eps = 0.5;
        let p = vec![0.5, 0.1, 0.1, 2.0]; // P, row-major 2×2
        let tracker = TrackerOutput {
            dt: 0.1,
            eps,
            labels: vec!["a".into(), "b".into()],
            estimates: vec![vec![0.25, -0.5]],
            covariances: vec![Some(p.clone())],
            reference: vec![vec![0.0, 0.0]],
        };
        let img = tracker_image(&out, &tracker, 0, 0, 0.0);
        // Texel containing a point (x, y).
        let texel = |x: f64, y: f64| {
            let c = (((x + 2.0) / 4.0) * RES as f64).floor() as usize;
            let r = RES - 1 - (((y + 2.0) / 4.0) * RES as f64).floor() as usize;
            img.pixels[r * RES + c]
        };
        let close = |a: egui::Color32, b: egui::Color32| {
            (a.r() as i32 - b.r() as i32).abs() <= 3
                && (a.g() as i32 - b.g() as i32).abs() <= 3
                && (a.b() as i32 - b.b() as i32).abs() <= 3
        };
        assert!(close(texel(0.25, -0.5), colormap(1.0)), "not maximal at x̂");
        // At ξ = (1, 0): q = ξᵀ P⁻¹ ξ = P_bb/det, value exp(−q/(2ε)).
        let det = 0.5 * 2.0 - 0.01;
        let expect = (-(2.0 / det) / (2.0 * eps)) as f32;
        let (x, y) = (0.25 + 1.0, -0.5); // texel centers are off by ≤ half a texel: compare loosely
        let got = texel(x + 4.0 / RES as f64 * 0.5, y + 4.0 / RES as f64 * 0.5);
        assert!(close(got, colormap(expect.exp())) || close(texel(x, y), colormap(expect.exp())), "wrong decay");
        // Undefined covariance ⇒ flat.
        let flat = TrackerOutput { covariances: vec![None], ..tracker };
        let img = tracker_image(&out, &flat, 0, 0, 0.0);
        assert!(img.pixels.iter().all(|&px| close(px, colormap(1.0))));
    }

    /// A spectral-element axis with the run's global numbering (shared
    /// element endpoints; a periodic direction omits the right domain
    /// endpoint). Chebyshev–Lobatto interior nodes stand in for
    /// Gauss–Lobatto — the interpolation identities under test hold for any
    /// distinct nodes at the element endpoints.
    fn make_axis(n_el: usize, p: usize, (x0, x1): (f64, f64), periodic: bool) -> Vec<f64> {
        let h = (x1 - x0) / n_el as f64;
        let n = n_el * p + if periodic { 0 } else { 1 };
        (0..n)
            .map(|g| {
                let (e, i) = (g / p, g % p);
                let u = 0.5 * (1.0 - (std::f64::consts::PI * i as f64 / p as f64).cos());
                x0 + (e as f64 + u) * h
            })
            .collect()
    }

    /// A 2D `FilterOutput` shell with hand-built axes (no filter run) —
    /// enough for `axis_eval` / `heatmap_image`.
    fn make_out(
        n_el: [usize; 2],
        p_ord: [usize; 2],
        periodic: [bool; 2],
        domain: [(f64, f64); 2],
    ) -> FilterOutput {
        FilterOutput {
            dt: 0.1,
            labels: vec!["a".into(), "b".into()],
            axes: (0..2)
                .map(|d| make_axis(n_el[d], p_ord[d], domain[d], periodic[d]))
                .collect(),
            n_el: n_el.to_vec(),
            p_ord: p_ord.to_vec(),
            periodic: periodic.to_vec(),
            dirichlet: vec![false; 2],
            domain: domain.to_vec(),
            pairs: vec![(0, 1)],
            saved_steps: vec![0],
            marginals: vec![vec![Vec::new()]],
            estimates: Vec::new(),
            reference: Vec::new(),
        }
    }

    /// Texel-center sample coordinate `k` of an axis, as in `axis_eval`.
    fn sample((x0, x1): (f64, f64), k: usize) -> f64 {
        x0 + (x1 - x0) * (k as f64 + 0.5) / RES as f64
    }

    /// The interpolant is exact on functions that are polynomials of degree
    /// ≤ p_ord on every element: each texel value equals the analytic
    /// product P(x)·Q(y) to round-off — the heatmap samples are exact
    /// spectral evaluations, not an approximation of them.
    #[test]
    fn resampling_is_exact_on_element_polynomials() {
        let out = make_out([4, 3], [3, 4], [false, false], [(-1.2, 0.8), (0.4, 2.0)]);
        let pa = |x: f64| 1.0 + x * (0.5 + x * (-2.0 + x * 1.5)); // degree 3
        let pb = |y: f64| 0.3 + y * (1.0 + y * (0.2 + y * (-0.7 + y * 0.1))); // degree 4
        let (na, nb) = (out.axes[0].len(), out.axes[1].len());
        let mut marg = vec![0.0; na * nb];
        for gb in 0..nb {
            for ga in 0..na {
                marg[ga + gb * na] = pa(out.axes[0][ga]) * pb(out.axes[1][gb]);
            }
        }
        let (ea, eb) = (axis_eval(&out, 0), axis_eval(&out, 1));
        for r in 0..RES {
            for c in 0..RES {
                let mut v = 0.0;
                for (gb, wb) in eb.nodes[r].iter().zip(&eb.weights[r]) {
                    let mut s = 0.0;
                    for (ga, wa) in ea.nodes[c].iter().zip(&ea.weights[c]) {
                        s += wa * marg[ga + gb * na];
                    }
                    v += wb * s;
                }
                let exact = pa(sample(out.domain[0], c)) * pb(sample(out.domain[1], r));
                assert!(
                    (v - exact).abs() <= 1e-10 * (1.0 + exact.abs()),
                    "texel ({r},{c}): {v} vs {exact}"
                );
            }
        }
    }

    /// A Dirichlet axis stores only the interior nodes; the resampler
    /// treats the two missing endpoints as zeros, so it is exact on
    /// element-wise polynomials of degree ≤ p_ord that vanish at the
    /// domain boundary.
    #[test]
    fn dirichlet_axis_resampling_is_exact() {
        let mut out = make_out([4, 3], [3, 4], [false, false], [(-1.2, 0.8), (0.4, 2.0)]);
        out.dirichlet = vec![true, false];
        // Drop the two boundary nodes of direction 0 (they are zeros).
        let full = out.axes[0].clone();
        out.axes[0] = full[1..full.len() - 1].to_vec();

        let (x0, x1) = out.domain[0];
        let pa = |x: f64| (x - x0) * (x1 - x) * (0.7 + 0.9 * x); // deg 3, 0 at both walls
        let pb = |y: f64| 0.3 + y * (1.0 + y * (0.2 + y * (-0.7 + y * 0.1))); // deg 4
        let (na, nb) = (out.axes[0].len(), out.axes[1].len());
        assert_eq!(na, 4 * 3 - 1);
        let mut marg = vec![0.0; na * nb];
        for gb in 0..nb {
            for ga in 0..na {
                marg[ga + gb * na] = pa(out.axes[0][ga]) * pb(out.axes[1][gb]);
            }
        }
        let (ea, eb) = (axis_eval(&out, 0), axis_eval(&out, 1));
        for r in 0..RES {
            for c in 0..RES {
                let mut v = 0.0;
                for (gb, wb) in eb.nodes[r].iter().zip(&eb.weights[r]) {
                    let mut s = 0.0;
                    for (ga, wa) in ea.nodes[c].iter().zip(&ea.weights[c]) {
                        s += wa * marg[ga + gb * na];
                    }
                    v += wb * s;
                }
                let exact = pa(sample(out.domain[0], c)) * pb(sample(out.domain[1], r));
                assert!(
                    (v - exact).abs() <= 1e-10 * (1.0 + exact.abs()),
                    "texel ({r},{c}): {v} vs {exact}"
                );
            }
        }
    }

    /// Neville's algorithm: the unique interpolating polynomial through
    /// (xs, fs) at x — an independent evaluation path from the product-form
    /// Lagrange weights of `axis_eval`.
    fn neville(xs: &[f64], fs: &[f64], x: f64) -> f64 {
        let mut q = fs.to_vec();
        for k in 1..xs.len() {
            for i in 0..xs.len() - k {
                q[i] = ((x - xs[i + k]) * q[i] - (x - xs[i]) * q[i + 1]) / (xs[i] - xs[i + k]);
            }
        }
        q[0]
    }

    /// On a periodic axis every sample point's weights form a partition of
    /// unity, agree with Neville's evaluation on the element's node set, and
    /// the last element's right endpoint wraps to global node 0 at
    /// coordinate x1.
    #[test]
    fn periodic_axis_wraps_and_matches_neville() {
        use std::f64::consts::PI;
        let (n_el, p) = (5, 4);
        let out = make_out([n_el, 3], [p, 3], [true, false], [(-PI, PI), (0.0, 1.0)]);
        let na = out.axes[0].len();
        assert_eq!(na, n_el * p); // right endpoint omitted
        let f: Vec<f64> = (0..na).map(|g| ((g * g % 17) as f64) * 0.37 - 1.0).collect();
        let (x0, x1) = out.domain[0];
        let h = (x1 - x0) / n_el as f64;
        let ea = axis_eval(&out, 0);
        for c in 0..RES {
            let x = sample(out.domain[0], c);
            let unity: f64 = ea.weights[c].iter().sum();
            assert!((unity - 1.0).abs() < 1e-12, "texel {c}: Σw = {unity}");
            let e = (((x - x0) / h).floor() as usize).min(n_el - 1);
            if e == n_el - 1 {
                assert_eq!(*ea.nodes[c].last().unwrap(), 0, "no wrap at texel {c}");
            }
            let (xs, fs): (Vec<f64>, Vec<f64>) = (0..=p)
                .map(|i| {
                    let g = e * p + i;
                    if g == na { (x1, f[0]) } else { (out.axes[0][g], f[g]) }
                })
                .unzip();
            let expect = neville(&xs, &fs, x);
            let got: f64 = ea.nodes[c]
                .iter()
                .zip(&ea.weights[c])
                .map(|(&g, w)| w * f[g])
                .sum();
            assert!(
                (got - expect).abs() <= 1e-9 * (1.0 + expect.abs()),
                "texel {c}: {got} vs {expect}"
            );
        }
    }

    /// The color floor rescales the colormap to [floor, 1]: on the ramp in
    /// b with floor 0.5, texels at y ≤ 0.5 take the lowest color, y = 0.75
    /// the middle one and y → 1 the highest; floor 0 is the plain colormap.
    #[test]
    fn color_floor_focuses_the_scale_on_the_max() {
        let mut out = make_out([4, 4], [3, 3], [false, false], [(0.0, 1.0), (0.0, 1.0)]);
        let (na, nb) = (out.axes[0].len(), out.axes[1].len());
        out.marginals = vec![vec![(0..na * nb).map(|k| out.axes[1][k / na]).collect()]];
        let close = |a: egui::Color32, b: egui::Color32| {
            (a.r() as i32 - b.r() as i32).abs() <= 2
                && (a.g() as i32 - b.g() as i32).abs() <= 2
                && (a.b() as i32 - b.b() as i32).abs() <= 2
        };
        let img = heatmap_image(&out, 0, 0, 0.5);
        // Row r has y = (RES − 1 − r + ½)/RES.
        let row_at = |y: f64| RES - 1 - ((y * RES as f64).floor() as usize).min(RES - 1);
        for y in [0.05, 0.3, 0.5] {
            let got = img.pixels[row_at(y) * RES + RES / 2];
            assert!(close(got, colormap(0.0)), "y = {y}: {got:?} should be the lowest color");
        }
        // Above the floor the scale is linear from floor to 1: compare
        // with the texel-centre value of each row.
        let y_of_row = |r: usize| ((RES - 1 - r) as f64 + 0.5) / RES as f64;
        for r in [row_at(0.75), row_at(0.9), 0] {
            let got = img.pixels[r * RES + RES / 2];
            let want = colormap(((y_of_row(r) - 0.5) / 0.5) as f32);
            assert!(close(got, want), "row {r}: {got:?} vs {want:?}");
        }
        assert_eq!(shade(0.2, 0.0), colormap(0.2));
    }

    /// Orientation and colormap of the produced image: a nodal ramp in
    /// direction b is interpolated exactly, so row r must be
    /// colormap((RES−1−r+½)/RES) — rows run top-to-bottom = decreasing b;
    /// likewise a ramp in direction a fixes columns left-to-right.
    #[test]
    fn heatmap_orientation_matches_axes() {
        let mut out = make_out([4, 4], [3, 3], [false, false], [(0.0, 1.0), (0.0, 1.0)]);
        let (na, nb) = (out.axes[0].len(), out.axes[1].len());
        let close = |a: egui::Color32, b: egui::Color32| {
            (a.r() as i32 - b.r() as i32).abs() <= 2
                && (a.g() as i32 - b.g() as i32).abs() <= 2
                && (a.b() as i32 - b.b() as i32).abs() <= 2
        };

        // Ramp in b (nodal max 1 at the right endpoint ⇒ v/max = y exactly).
        out.marginals = vec![vec![(0..na * nb).map(|k| out.axes[1][k / na]).collect()]];
        let img = heatmap_image(&out, 0, 0, 0.0);
        for r in [0, RES / 3, RES - 1] {
            let want = colormap(((RES - 1 - r) as f32 + 0.5) / RES as f32);
            for c in [0, RES / 2, RES - 1] {
                let got = img.pixels[r * RES + c];
                assert!(close(want, got), "row {r}, col {c}: {got:?} vs {want:?}");
            }
        }

        // Ramp in a.
        out.marginals = vec![vec![(0..na * nb).map(|k| out.axes[0][k % na]).collect()]];
        let img = heatmap_image(&out, 0, 0, 0.0);
        for c in [0, RES / 3, RES - 1] {
            let want = colormap((c as f32 + 0.5) / RES as f32);
            for r in [0, RES / 2, RES - 1] {
                let got = img.pixels[r * RES + c];
                assert!(close(want, got), "row {r}, col {c}: {got:?} vs {want:?}");
            }
        }
    }
}
