//! Tracker view: one time-series plot per state component — the true
//! (reference) trajectory as a wide neutral line, the tracker estimate x̂
//! as a thin dashed accent line on top (so both stay visible where they
//! coincide), its 2σ band, and a faint playback cursor. The y-range of each
//! plot follows the *estimate* only, ignoring the first 5 % of the steps
//! (the prior-to-data transient), so that a wild start or a runaway
//! reference cannot flatten the interesting part.
//!
//! The band is ±2√(ε P_dd) with P = S⁻¹: the tracker is ε-free, but P is
//! the inverse curvature of the *energy* V, and the density p ∝ exp(−V/ε)
//! has covariance ε·P — the 2σ band of the density is ±2√(εP), not ±2√P.

use ode_observers::jobs::TrackerOutput;
use ode_models_egui::palette;

/// Fraction of the run's steps skipped at the start when scaling the
/// y-axis on the estimate.
const SKIP_FRACTION: f64 = 0.05;

/// Legend name of the band. It is hidden by default (its legend entry is
/// unchecked on first display); clicking the entry shows it.
const BAND_NAME: &str = "±2√(εP)";

pub fn ui(ui: &mut egui::Ui, out: &TrackerOutput, t: f64) {
    let m = out.labels.len();
    if m == 0 {
        return;
    }
    let n = out.estimates.len();
    let t_final = (n.saturating_sub(1)) as f64 * out.dt;
    let dark = ui.visuals().dark_mode;
    let neutral = ui.visuals().text_color();
    let accent = palette::series(0, dark);
    let band = accent.gamma_multiply(0.18);
    let cursor = neutral.gamma_multiply(0.35);
    // First step used for the y-range: skip the initial transient.
    let skip = ((n as f64 * SKIP_FRACTION).ceil() as usize).min(n.saturating_sub(1));

    let spacing = ui.spacing().item_spacing.y;
    let h = ((ui.available_height() - spacing * m as f32) / m as f32).max(80.0);
    for d in 0..m {
        let reference: egui_plot::PlotPoints = out
            .reference
            .iter()
            .enumerate()
            .map(|(i, s)| [i as f64 * out.dt, s[d]])
            .collect();
        let estimate: egui_plot::PlotPoints = out
            .estimates
            .iter()
            .enumerate()
            .map(|(i, s)| [i as f64 * out.dt, s[d]])
            .collect();
        // y-range from the estimate only, after the initial transient
        // (with a margin); the reference and the band are clipped to it.
        let (mut y_lo, mut y_hi) = (f64::MAX, f64::MIN);
        for s in &out.estimates[skip..] {
            y_lo = y_lo.min(s[d]);
            y_hi = y_hi.max(s[d]);
        }
        let margin = 0.1 * (y_hi - y_lo).max(1e-6);
        let (y_lo, y_hi) = (y_lo - margin, y_hi + margin);

        // 2σ band of the density, one convex quad per time interval (egui
        // fills polygons as convex fans, so the band cannot be a single
        // concave polygon); intervals missing a covariance at either end
        // are left undrawn.
        let mut quads: Vec<[[f64; 2]; 4]> = Vec::new();
        for i in 0..n.saturating_sub(1) {
            let (Some(p0), Some(p1)) = (&out.covariances[i], &out.covariances[i + 1]) else {
                continue;
            };
            let sd0 = (out.eps * p0[d * m + d]).max(0.0).sqrt();
            let sd1 = (out.eps * p1[d * m + d]).max(0.0).sqrt();
            let (e0, e1) = (&out.estimates[i], &out.estimates[i + 1]);
            let (t0, t1) = (i as f64 * out.dt, (i + 1) as f64 * out.dt);
            quads.push([
                [t0, (e0[d] + 2.0 * sd0).min(y_hi)],
                [t1, (e1[d] + 2.0 * sd1).min(y_hi)],
                [t1, (e1[d] - 2.0 * sd1).max(y_lo)],
                [t0, (e0[d] - 2.0 * sd0).max(y_lo)],
            ]);
        }

        let plot_id = egui::Id::new(("tracker_plot", d));
        egui_plot::Plot::new(("tracker_plot", d))
            .id(plot_id)
            .height(h)
            .y_axis_label(&out.labels[d])
            .include_x(0.0)
            .include_x(t_final)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .legend(egui_plot::Legend::default().position(egui_plot::Corner::RightTop))
            .show(ui, |plot_ui| {
                plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                    [0.0, y_lo],
                    [t_final.max(out.dt), y_hi],
                ));
                for quad in &quads {
                    plot_ui.polygon(
                        egui_plot::Polygon::new(BAND_NAME, quad.to_vec())
                            .fill_color(band)
                            .stroke(egui::Stroke::NONE),
                    );
                }
                plot_ui.line(
                    egui_plot::Line::new("true state", reference)
                        .color(neutral.gamma_multiply(0.7))
                        .width(3.0),
                );
                plot_ui.line(
                    egui_plot::Line::new("tracker x̂", estimate)
                        .color(accent)
                        .width(1.5)
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
                // Playback cursor, kept out of the legend.
                plot_ui.vline(egui_plot::VLine::new("", t).color(cursor).width(1.0));
            });

        // Band hidden by default: on the plot's first display, mark its
        // legend item as hidden in the plot memory (egui_plot keys legend
        // items by the hash of their name); the legend entry then shows
        // unchecked and toggles it as usual.
        let init_id = plot_id.with("band_hidden_by_default");
        let initialized: bool = ui.ctx().data(|d| d.get_temp(init_id)).unwrap_or(false);
        if !initialized
            && let Some(mut mem) = egui_plot::PlotMemory::load(ui.ctx(), plot_id)
        {
            mem.hidden_items.insert(egui::Id::new(BAND_NAME));
            mem.store(ui.ctx(), plot_id);
            ui.ctx().data_mut(|d| d.insert_temp(init_id, true));
        }
    }
}
