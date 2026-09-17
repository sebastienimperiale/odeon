//! Central-panel view "Particles": the population of the particle solver at
//! the playback time on a selectable plane — red dots at the particle
//! positions, and the sum of an isotropic Gaussian per particle as a
//! heatmap (same colormap and colour floor as the other density views).
//! The Gaussian of a particle has width h·(e − a)/e, e its exponential
//! clock and a its accumulated misfit: h at birth, shrinking linearly with
//! the particle's relative life expectancy, so a particle about to die
//! fades from the smoothed cloud (a floor keeps the width from vanishing). The reference trajectory is drawn in black,
//! the tracker's estimate in red on request. The plot is locked to the box
//! of the initial draw.

use crate::filter_view::{ColorFocus, Overlays, shade};
use ode_observers::jobs::ParticleOutput;
use std::collections::HashMap;

/// Pixel resolution (per side) of the smoothed-cloud textures.
const RES: usize = 220;
/// Colour of the particles (red, as asked).
const PARTICLE_RED: egui::Color32 = egui::Color32::from_rgb(200, 30, 30);
/// Smallest width of a particle's Gaussian, as a fraction of h.
const MIN_WIDTH: f64 = 0.05;

/// Persistent state of the particles view.
pub struct ParticlesView {
    /// Draw the smoothed cloud (sum of Gaussians) under the dots.
    pub show_density: bool,
    /// Draw the particles themselves.
    pub show_dots: bool,
    /// Overlay the tracker's estimate in red when a tracker run exists.
    pub show_tracker: bool,
    /// Overlay the unscented closure's centre in blue when its run exists.
    pub show_unscented: bool,
    /// Width h of a newborn particle's Gaussian, in state units.
    pub width: f64,
    /// Colour-scale floor, see [`ColorFocus`].
    pub focus: ColorFocus,
    plane: usize,
    cache: HashMap<(usize, usize), egui::TextureHandle>,
    /// (width, floor) the cached textures were built with.
    cached_key: (u64, u64),
}

impl Default for ParticlesView {
    fn default() -> Self {
        ParticlesView {
            show_density: true,
            show_dots: true,
            show_tracker: true,
            show_unscented: true,
            width: 0.1,
            focus: ColorFocus::default(),
            plane: 0,
            cache: HashMap::new(),
            cached_key: (0, 0),
        }
    }
}

impl ParticlesView {
    /// Drop the cached textures.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Fresh output: drop the cache, pick the first plane, and size h to
    /// the box (a few percent of the smallest extent).
    pub fn reset(&mut self, out: &ParticleOutput) {
        self.clear_cache();
        self.plane = 0;
        let extent = out.domain.iter().map(|(a, b)| b - a).fold(f64::MAX, f64::min);
        if extent.is_finite() && extent > 0.0 {
            self.width = 0.03 * extent;
        }
    }

    /// All planes (a, b), a < b, of the state.
    fn pairs(out: &ParticleOutput) -> Vec<(usize, usize)> {
        let n = out.labels.len();
        (0..n).flat_map(|a| ((a + 1)..n).map(move |b| (a, b))).collect()
    }

    /// Render the view at playback `step`.
    pub fn ui(&mut self, ui: &mut egui::Ui, out: &ParticleOutput, overlays: Overlays, step: usize) {
        let pairs = Self::pairs(out);
        if pairs.is_empty() || out.positions.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("Nothing to draw: the state needs two components.");
            });
            return;
        }
        let step = step.min(out.positions.len() - 1);
        ui.horizontal(|ui| {
            self.plane = self.plane.min(pairs.len() - 1);
            let pair_label = |&(a, b): &(usize, usize)| format!("{} × {}", out.labels[a], out.labels[b]);
            egui::ComboBox::from_id_salt("particles_plane")
                .width(ui.available_width().min(180.0))
                .selected_text(pair_label(&pairs[self.plane]))
                .show_ui(ui, |ui| {
                    for (j, pair) in pairs.iter().enumerate() {
                        ui.selectable_value(&mut self.plane, j, pair_label(pair));
                    }
                });
            ui.small(format!(
                "step {step}, t = {:.2} s, {} particles",
                step as f64 * out.dt,
                out.n_particles
            ));
            ui.separator();
            ui.checkbox(&mut self.show_dots, "particles");
            ui.checkbox(&mut self.show_density, "density")
                .on_hover_text("Sum of one isotropic Gaussian per particle, normalised by its max");
            ui.label("width h").on_hover_text(
                "Width of a newborn particle's Gaussian (state units); it shrinks \
                 linearly with the particle's remaining life, to 5 % of h",
            );
            // Fixed range from the box: one heatmap pixel of the smallest
            // extent (nothing narrower can be drawn) up to the largest
            // extent (the box itself); logarithmic in between.
            let (e_min, e_max) = out.domain.iter().fold((f64::MAX, 0.0f64), |(mn, mx), (a, b)| {
                (mn.min(b - a), mx.max(b - a))
            });
            let (lo, hi) = ((e_min / RES as f64).max(1e-9), e_max.max(1e-6));
            self.width = self.width.clamp(lo, hi);
            ui.add(egui::Slider::new(&mut self.width, lo..=hi).logarithmic(true).show_value(true));
            overlays.checkboxes(ui, &mut self.show_tracker, &mut self.show_unscented);
            ui.separator();
            self.focus.ui(ui);
        });
        // Rebuild the cache when h or the floor changed.
        let key = (self.width.to_bits(), self.focus.floor().to_bits());
        if key != self.cached_key {
            self.cache.clear();
            self.cached_key = key;
        }

        let plane = self.plane;
        let (a, b) = pairs[plane];
        let (x0, x1) = out.domain[a];
        let (y0, y1) = out.domain[b];
        let tex = if self.show_density {
            Some(
                self.cache
                    .entry((step, plane))
                    .or_insert_with(|| {
                        let image = cloud_image(out, step, (a, b), self.width, self.focus.floor());
                        ui.ctx().load_texture(format!("cloud_{step}_{plane}"), image, egui::TextureOptions::LINEAR)
                    })
                    .clone(),
            )
        } else {
            None
        };

        let m = out.labels.len();
        let dots: Vec<[f64; 2]> = if self.show_dots {
            let pos = &out.positions[step];
            (0..out.n_particles).map(|i| [pos[i * m + a], pos[i * m + b]]).collect()
        } else {
            Vec::new()
        };
        let clip = step.min(out.reference.len() - 1);
        let reference: Vec<[f64; 2]> = out.reference[..=clip].iter().map(|s| [s[a], s[b]]).collect();
        let paths = overlays.paths(self.show_tracker, self.show_unscented, step, (a, b));
        let neutral = ui.visuals().text_color();

        egui_plot::Plot::new("particles_plot")
            .x_axis_label(&out.labels[a])
            .y_axis_label(&out.labels[b])
            .set_margin_fraction(egui::vec2(0.0, 0.0))
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .allow_double_click_reset(false)
            .show(ui, |plot_ui| {
                plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max([x0, y0], [x1, y1]));
                if let Some(tex) = &tex {
                    plot_ui.image(egui_plot::PlotImage::new(
                        "cloud",
                        tex,
                        egui_plot::PlotPoint::new(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
                        egui::vec2((x1 - x0) as f32, (y1 - y0) as f32),
                    ));
                }
                if !dots.is_empty() {
                    plot_ui.points(egui_plot::Points::new("particles", dots).radius(1.8).color(PARTICLE_RED));
                }
                plot_ui.line(egui_plot::Line::new("reference", reference).color(neutral).width(1.5));
                let s = &out.reference[clip];
                plot_ui.points(egui_plot::Points::new("", vec![[s[a], s[b]]]).radius(4.0).color(neutral));
                Overlays::draw(plot_ui, paths);
            });
    }
}

/// The smoothed cloud on plane (a, b) at `step`, as a RES×RES image over the
/// box: Σ_i exp(−|ξ − z_i|²/(2 h_i²)) with h_i = h·max((e_i − a_i)/e_i,
/// MIN_WIDTH), each Gaussian splatted on the pixels within 3.5 h_i (and on
/// its images across a periodic direction), normalised by the max, through
/// the colormap with the colour floor.
fn cloud_image(out: &ParticleOutput, step: usize, (a, b): (usize, usize), h: f64, floor: f64) -> egui::ColorImage {
    let m = out.labels.len();
    let (x0, x1) = out.domain[a];
    let (y0, y1) = out.domain[b];
    let (lx, ly) = (x1 - x0, y1 - y0);
    let (px, py) = (lx / RES as f64, ly / RES as f64); // pixel sizes
    let pos = &out.positions[step];
    let budget = &out.budget[step];
    let spent = &out.spent[step];
    let mut field = vec![0.0f64; RES * RES];
    // Image offsets to consider per direction: 0, and ±period if periodic.
    let images = |periodic: bool, l: f64| if periodic { vec![0.0, -l, l] } else { vec![0.0] };
    let (ima, imb) = (images(out.periodic[a], lx), images(out.periodic[b], ly));
    for i in 0..out.n_particles {
        let rel = if budget[i] > 0.0 { ((budget[i] - spent[i]) / budget[i]).clamp(0.0, 1.0) } else { 1.0 };
        let hi = h * rel.max(MIN_WIDTH);
        let inv = 1.0 / (2.0 * hi * hi);
        let reach = 3.5 * hi;
        for &ka in &ima {
            for &kb in &imb {
                let (zx, zy) = (pos[i * m + a] + ka, pos[i * m + b] + kb);
                // Pixel range touched by this (image of the) particle.
                let c0 = (((zx - reach - x0) / px).floor().max(0.0)) as usize;
                let c1 = (((zx + reach - x0) / px).ceil().min(RES as f64 - 1.0)) as isize;
                let r0 = (((zy - reach - y0) / py).floor().max(0.0)) as usize;
                let r1 = (((zy + reach - y0) / py).ceil().min(RES as f64 - 1.0)) as isize;
                if c1 < c0 as isize || r1 < r0 as isize {
                    continue;
                }
                for r in r0..=(r1 as usize) {
                    let y = y0 + (r as f64 + 0.5) * py;
                    let dy = y - zy;
                    // Image rows run top to bottom = decreasing b.
                    let row = (RES - 1 - r) * RES;
                    for c in c0..=(c1 as usize) {
                        let x = x0 + (c as f64 + 0.5) * px;
                        let dx = x - zx;
                        field[row + c] += (-(dx * dx + dy * dy) * inv).exp();
                    }
                }
            }
        }
    }
    let max = field.iter().cloned().fold(0.0, f64::max).max(1e-300);
    let pixels = field.iter().map(|&v| shade(v / max, floor)).collect();
    egui::ColorImage::new([RES, RES], pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One newborn particle gives a Gaussian of width h centred on it (max
    /// at its pixel, e^{−1/2} one h away); a particle with half its life
    /// left gets half the width; the image row order follows the axes.
    #[test]
    fn cloud_image_is_the_sum_of_life_scaled_gaussians() {
        // Particle on a pixel centre, h a whole number of pixels, so the
        // sampled values are the analytic ones.
        let px = 2.0 / RES as f64;
        let (zx, zy) = (-1.0 + 132.5 * px, -1.0 + 66.5 * px);
        let h = 22.0 * px;
        let out = ParticleOutput {
            dt: 0.1,
            labels: vec!["a".into(), "b".into()],
            periodic: vec![false, false],
            domain: vec![(-1.0, 1.0), (-1.0, 1.0)],
            n_particles: 1,
            positions: vec![vec![zx, zy], vec![zx, zy]],
            budget: vec![vec![10.0], vec![10.0]],
            spent: vec![vec![0.0], vec![5.0]],
            reference: vec![vec![0.0, 0.0]; 2],
        };
        let texel = |img: &egui::ColorImage, x: f64, y: f64| {
            let c = (((x + 1.0) / 2.0) * RES as f64).floor() as usize;
            let r = RES - 1 - (((y + 1.0) / 2.0) * RES as f64).floor() as usize;
            img.pixels[r * RES + c]
        };
        let close = |a: egui::Color32, b: egui::Color32| {
            (a.r() as i32 - b.r() as i32).abs() <= 6
                && (a.g() as i32 - b.g() as i32).abs() <= 6
                && (a.b() as i32 - b.b() as i32).abs() <= 6
        };
        let img = cloud_image(&out, 0, (0, 1), h, 0.0);
        assert!(close(texel(&img, zx, zy), shade(1.0, 0.0)), "not maximal at the particle");
        assert!(close(texel(&img, zx + h, zy), shade((-0.5f64).exp(), 0.0)), "wrong width at full life");
        // Half life left: width h/2, so one h away the value is e^{−2}.
        let img = cloud_image(&out, 1, (0, 1), h, 0.0);
        assert!(close(texel(&img, zx + h, zy), shade((-2.0f64).exp(), 0.0)), "width did not shrink with life");
        // Far away: the lowest colour.
        assert!(close(texel(&img, -0.8, 0.8), shade(0.0, 0.0)));
    }
}
