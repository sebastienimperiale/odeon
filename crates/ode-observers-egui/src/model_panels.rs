//! The model side of a viewer's panels: the model selector, the parameter
//! form with the Observation and Simulation sections, the Run-simulation
//! button, the bottom observation panel (playback controls + y(t) plot with
//! a time cursor) and the central scene with its "About this model…"
//! button, which opens the models note (`crates/ode-models/docs/models.pdf`,
//! built locally with `latexmk -pdf models.tex`) in the browser at the
//! model's section. The application adds its estimator sections beside
//! them.

use crate::palette;
use crate::slot::ModelSlot;

/// The model selector: a combo box over `names`, editing `selected`.
pub fn model_selector(ui: &mut egui::Ui, names: &[&str], selected: &mut usize) {
    egui::ComboBox::from_id_salt("model_select")
        .width(ui.available_width() - 8.0)
        .selected_text(names[*selected])
        .show_ui(ui, |ui| {
            for (i, name) in names.iter().enumerate() {
                ui.selectable_value(selected, i, *name);
            }
        });
}

/// The parameter form of the slot's model, the Observation section (one
/// entry per observation choice, with its noise row under the selected
/// ones) and the Simulation section (dt, T). Any edit re-plots the scene at
/// the new initial condition and discards stale runs
/// ([`ModelSlot::refresh_preview`]); returns `true` in that case so the
/// caller can drop its own dependent state.
pub fn params_ui(ui: &mut egui::Ui, slot: &mut ModelSlot) -> bool {
    let params_changed = slot.model.params_ui(ui);

    // Observation section — always present, one checkable entry per
    // observation type the model offers.
    ui.separator();
    ui.label("Observation h(x)");
    let options = slot.model.obs_options();
    let selected_obs = slot.model.obs_selected();
    let exclusive = slot.model.obs_exclusive();
    let mut obs_changed = false;
    for (i, option) in options.iter().enumerate() {
        let mut on = selected_obs[i];
        if exclusive {
            // One observation at a time: the filter consumes a single
            // scalar h, and the plot must show exactly what the filter
            // sees.
            if ui.radio(on, option).clicked() && !on {
                slot.model.toggle_obs(i);
                obs_changed = true;
                on = true;
            }
        } else if ui.checkbox(&mut on, option).changed() {
            slot.model.toggle_obs(i);
            obs_changed = true;
        }
        // Noise model of this observation (plotted series becomes the
        // noisy one; unset = noiseless).
        if on {
            ui.indent(("noise_row", i), |ui| {
                obs_changed |= slot.model.noise_ui(ui, i);
            });
        }
    }

    ui.separator();
    let mut sim_changed = false;
    ui.horizontal(|ui| {
        ui.label("Simulation");
        if ui.small_button("default").clicked() {
            slot.dt = 0.01;
            slot.t_final = 10.0;
            sim_changed = true;
        }
    });
    ui.horizontal(|ui| {
        sim_changed |= ui
            .add(egui::DragValue::new(&mut slot.dt).speed(0.001).range(1e-4..=1.0))
            .changed();
        ui.label("time step dt");
    });
    ui.horizontal(|ui| {
        sim_changed |= ui
            .add(egui::DragValue::new(&mut slot.t_final).speed(0.5).range(0.1..=10_000.0))
            .changed();
        ui.label("duration T");
    });

    // Any edit re-plots the scene at the (new) initial condition
    // immediately, discards stale runs, and re-arms the Run button.
    let changed = params_changed || obs_changed || sim_changed;
    if changed {
        slot.refresh_preview();
    }
    changed
}

/// The "Run simulation" button — armed until a run completes; editing any
/// parameter re-arms it — or the progress bar of the run in flight.
/// `enabled` lets the caller veto the button (an estimator run in flight,
/// say).
pub fn run_ui(ui: &mut egui::Ui, slot: &mut ModelSlot, enabled: bool) {
    match &slot.run {
        Some(run) => {
            ui.add(egui::ProgressBar::new(run.progress()).show_percentage());
        }
        None => {
            let clicked = ui
                .add_enabled(
                    enabled && !slot.run_done(),
                    egui::Button::new("▶  Run simulation")
                        .min_size(egui::vec2(ui.available_width() - 8.0, 28.0)),
                )
                .clicked();
            if clicked {
                slot.start_run();
            }
        }
    }
}

/// The bottom panel: playback controls (play/pause, restart, speed, fit,
/// scrub) and the observation plot y(t) up to the cursor, with a legend row
/// below it. A placeholder until a run completes. Meant for an
/// `egui::Panel::bottom`: the content fills the panel exactly (see the
/// note inside).
pub fn observation_panel(ui: &mut egui::Ui, slot: &mut ModelSlot) {
    // A panel stores its own content height as next frame's size, so the
    // content must fill it *exactly*: everything below is sized to
    // undershoot slightly, and `take_available_height` at the end absorbs
    // the slack — otherwise any ±ε mismatch makes the panel creep bigger
    // or smaller every frame.
    let traj = match &slot.traj {
        Some(traj) if !slot.is_preview => traj,
        _ => {
            ui.centered_and_justified(|ui| {
                ui.label("The observation y(t) appears here after a run.");
            });
            return;
        }
    };
    let t_final = traj.t_final();

    ui.add_space(4.0);
    let mut fit_clicked = false;
    ui.horizontal(|ui| {
        let icon = if slot.playback.playing { "⏸" } else { "▶" };
        if ui.button(icon).clicked() {
            if !slot.playback.playing && slot.playback.t >= t_final {
                slot.playback.t = 0.0;
            }
            slot.playback.playing = !slot.playback.playing;
        }
        if ui.button("⏮").clicked() {
            slot.playback.restart();
        }
        egui::ComboBox::from_id_salt("speed")
            .width(70.0)
            .selected_text(format!("{}×", slot.playback.speed))
            .show_ui(ui, |ui| {
                for s in [0.25, 0.5, 1.0, 2.0, 4.0] {
                    ui.selectable_value(&mut slot.playback.speed, s, format!("{s}×"));
                }
            });
        fit_clicked = ui
            .button("⛶")
            .on_hover_text("Rescale the plot axes to the data")
            .clicked();
        ui.monospace(format!("t = {:6.2} / {:.2} s", slot.playback.t, t_final));
        // The slider comes last and takes exactly what is left, so the row
        // can never overflow the window.
        ui.spacing_mut().slider_width = (ui.available_width() - 16.0).max(60.0);
        let slider = egui::Slider::new(&mut slot.playback.t, 0.0..=t_final)
            .show_value(false)
            .clamping(egui::SliderClamping::Always);
        if ui.add(slider).dragged() {
            slot.playback.playing = false;
        }
    });
    slot.playback.t = slot.playback.t.clamp(0.0, t_final);

    // Observation plot up to the playback cursor, with a manual legend row
    // *below* the plot (egui_plot can only place its legend inside the
    // plot area).
    let frame = traj.frame_at(slot.playback.t);
    let labels = traj.obs_labels.clone();
    let dark = ui.visuals().dark_mode;
    let n_obs = traj.observations.first().map_or(0, Vec::len);
    let cursor_color = ui.visuals().text_color().gamma_multiply(0.5);
    let legend_h = ui.text_style_height(&egui::TextStyle::Body);
    let spacing_y = ui.spacing().item_spacing.y;
    let plot_h = (ui.available_height() - legend_h - 2.0 * spacing_y - 4.0).max(60.0);
    egui_plot::Plot::new("obs_plot")
        .height(plot_h)
        .set_margin_fraction(egui::vec2(0.0, 0.05))
        .include_x(0.0)
        .include_x(t_final)
        .show(ui, |plot_ui| {
            for k in 0..n_obs {
                let pts: egui_plot::PlotPoints = (0..=frame)
                    .map(|i| [i as f64 * traj.dt, traj.observations[i][k]])
                    .collect();
                let name = labels.get(k).cloned().unwrap_or_else(|| format!("y_{k}"));
                plot_ui.line(
                    egui_plot::Line::new(name, pts)
                        .color(palette::series(k, dark))
                        .width(2.0),
                );
            }
            plot_ui.vline(
                egui_plot::VLine::new("t", slot.playback.t)
                    .color(cursor_color)
                    .width(1.0),
            );
            if fit_clicked {
                plot_ui.set_auto_bounds(egui::Vec2b::TRUE);
            }
            // Time never goes negative: translate panned/zoomed bounds back
            // to x ≥ 0 (auto bounds already start at 0 thanks to the zero
            // x-margin above).
            let b = plot_ui.plot_bounds();
            if b.min()[0] < 0.0 {
                plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                    [0.0, b.min()[1]],
                    [b.width(), b.max()[1]],
                ));
            }
        });
    ui.horizontal(|ui| {
        for k in 0..n_obs {
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(18.0, ui.text_style_height(&egui::TextStyle::Body)),
                egui::Sense::hover(),
            );
            ui.painter().line_segment(
                [
                    rect.left_center() + egui::vec2(2.0, 0.0),
                    rect.right_center() - egui::vec2(2.0, 0.0),
                ],
                egui::Stroke::new(3.0, palette::series(k, dark)),
            );
            ui.label(labels.get(k).cloned().unwrap_or_else(|| format!("y_{k}")));
            ui.add_space(14.0);
        }
    });
    ui.take_available_height();
}

/// The central scene: the "About this model…" button (with the outcome of
/// the last click beside it, kept in egui's temporary memory under `id`)
/// and the model's drawing of the frame at the playback time. The caller
/// advances the playback clock ([`ModelSlot::advance`]) before.
pub fn scene_ui(ui: &mut egui::Ui, slot: &ModelSlot, id: usize) {
    let status_id = egui::Id::new(("about_model_status", id));
    ui.horizontal(|ui| {
        if ui
            .button("About this model…")
            .on_hover_text(
                "Open the models note (crates/ode-models/docs/models.pdf of this checkout, \
                 built with `latexmk -pdf models.tex` there) in the browser at this model's \
                 page; ODEON_BROWSER selects the browser — Safari's PDF viewer ignores the \
                 page, Firefox's and Chrome's honour it. On the web page: models.pdf next to \
                 the page, at the section's named destination.",
            )
            .clicked()
        {
            let status = match slot.model.doc_dest() {
                Some(dest) => match open_models_pdf(dest) {
                    Ok(url) => format!("opened {url}"),
                    Err(e) => format!("could not open the models note: {e}"),
                },
                None => "no section for this model in models.pdf".to_string(),
            };
            ui.ctx().data_mut(|d| d.insert_temp(status_id, status));
        }
        if let Some(s) = ui.ctx().data(|d| d.get_temp::<String>(status_id)) {
            ui.small(s);
        }
    });
    ui.add_space(4.0);

    let (response, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::hover());
    match &slot.traj {
        Some(traj) => {
            let frame = traj.frame_at(slot.playback.t);
            slot.model.draw(&painter, response.rect, traj, frame, ui.visuals());
        }
        None => {
            painter.text(
                response.rect.center(),
                egui::Align2::CENTER_CENTER,
                "Choose a model and its parameters, then press Run.",
                egui::FontId::proportional(16.0),
                ui.visuals().weak_text_color(),
            );
        }
    }
}

/// Open the browser on the models note at the section of the model `dest`
/// (`\modelsection{<dest>}{…}` of `crates/ode-models/docs/models.tex`).
///
/// Native: the PDF is the local `models.pdf` of this checkout — the path of
/// the source tree at compile time (`CARGO_MANIFEST_DIR/../ode-models/docs`),
/// by decision: the link is local to this installation. The page of the
/// section is read from `models.aux`, written beside the PDF by the same
/// build ([`models_page`]), so the link follows the document as it grows;
/// without it the named destination `#nameddest=<dest>` is used. The
/// browser: `$ODEON_BROWSER` when set (an application name or the path of
/// an executable), otherwise on macOS the first installed of Firefox,
/// Google Chrome, Chromium, Microsoft Edge, Brave Browser, Safari — Safari
/// last because its PDF viewer ignores the page fragment. On macOS the
/// browser's executable is launched directly: Launch Services (`open`,
/// even `open -u`) resolves a `file://` URL to the file and drops its
/// fragment. On Linux `$ODEON_BROWSER` or `xdg-open`, on Windows `start`.
/// Returns the URL opened.
///
/// Web: opens `models.pdf#nameddest=<dest>` relative to the page in a new
/// tab (the PDF must be published next to the page).
pub fn open_models_pdf(dest: &str) -> std::io::Result<String> {
    #[cfg(target_arch = "wasm32")]
    {
        let url = format!("models.pdf#nameddest={dest}");
        web_sys::window()
            .and_then(|w| w.open_with_url_and_target(&url, "_blank").ok().flatten())
            .ok_or_else(|| std::io::Error::other("the browser refused to open a new tab (pop-up blocked?)"))?;
        return Ok(url);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let docs = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../ode-models/docs"));
        let path = docs.join("models.pdf");
        let path = std::fs::canonicalize(&path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("{} ({e}); build it with `latexmk -pdf models.tex` in crates/ode-models/docs", path.display()),
            )
        })?;
        let fragment = match std::fs::read_to_string(docs.join("models.aux")).ok().and_then(|aux| models_page(&aux, dest)) {
            Some(page) => format!("page={page}"),
            None => format!("nameddest={dest}"),
        };
        let url = format!("file://{}#{fragment}", path.to_string_lossy().replace(' ', "%20"));
        let browser = std::env::var("ODEON_BROWSER").ok();
        #[cfg(target_os = "macos")]
        macos_launch(&browser.unwrap_or_else(macos_browser), &url)?;
        #[cfg(target_os = "linux")]
        {
            let status = std::process::Command::new(browser.as_deref().unwrap_or("xdg-open")).arg(&url).status()?;
            if !status.success() {
                return Err(std::io::Error::other(format!("browser exited with {status}")));
            }
        }
        #[cfg(target_os = "windows")]
        {
            let status = std::process::Command::new("cmd")
                .args(["/C", "start", "", browser.as_deref().unwrap_or(""), &url])
                .status()?;
            if !status.success() {
                return Err(std::io::Error::other(format!("browser exited with {status}")));
            }
        }
        Ok(url)
    }
}

/// The page of the model `dest` in `models.pdf`, from the `.aux` file of
/// its build: `\modelsection` labels every section `model:<dest>`, and
/// LaTeX records `\newlabel{model:<dest>}{{<section>}{<page>}…}`.
#[cfg(not(target_arch = "wasm32"))]
fn models_page(aux: &str, dest: &str) -> Option<u32> {
    let key = format!("\\newlabel{{model:{dest}}}{{{{");
    let rest = &aux[aux.find(&key)? + key.len()..];
    let rest = &rest[rest.find('}')? + 1..];
    let rest = rest.strip_prefix('{')?;
    rest[..rest.find('}')?].trim().parse().ok()
}

/// The first installed browser of the preference list, by bundle.
#[cfg(target_os = "macos")]
fn macos_browser() -> String {
    const CANDIDATES: [&str; 6] = ["Firefox", "Google Chrome", "Chromium", "Microsoft Edge", "Brave Browser", "Safari"];
    CANDIDATES.iter().find(|app| macos_bundle(app).is_some()).unwrap_or(&"Safari").to_string()
}

/// The `.app` bundle of `app` in /Applications or ~/Applications.
#[cfg(target_os = "macos")]
fn macos_bundle(app: &str) -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    [format!("/Applications/{app}.app"), format!("{home}/Applications/{app}.app")]
        .into_iter()
        .map(std::path::PathBuf::from)
        .find(|p| p.is_dir())
}

/// Launch the browser `app` (a name in /Applications, or an executable
/// path) directly on `url`, so that the URL's fragment survives; Safari
/// through `open -u` (its viewer ignores the fragment anyway).
#[cfg(target_os = "macos")]
fn macos_launch(app: &str, url: &str) -> std::io::Result<()> {
    let exe = if std::path::Path::new(app).is_file() {
        Some(std::path::PathBuf::from(app))
    } else if app == "Safari" {
        None
    } else {
        let bundle = macos_bundle(app)
            .ok_or_else(|| std::io::Error::other(format!("browser {app} not found in /Applications")))?;
        let name = std::process::Command::new("defaults")
            .args(["read", &bundle.join("Contents/Info").to_string_lossy(), "CFBundleExecutable"])
            .output()?;
        let name = String::from_utf8_lossy(&name.stdout).trim().to_string();
        Some(bundle.join("Contents/MacOS").join(name)).filter(|p| p.is_file())
    };
    match exe {
        Some(exe) => {
            std::process::Command::new(exe).arg(url).spawn()?;
        }
        None => {
            let status = std::process::Command::new("open").args(["-a", app, "-u", url]).status()?;
            if !status.success() {
                return Err(std::io::Error::other(format!("open exited with {status}")));
            }
        }
    }
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::models_page;

    /// The section's page is read from the `.aux` of the note's build.
    #[test]
    fn page_of_a_model_is_read_from_the_aux_file() {
        let aux = "\\newlabel{model:spring}{{1}{2}{Linear spring chain}{section.1}{}}\n\
                   \\newlabel{model:kepler_mu}{{6}{7}{Kepler problem with an unknown gravitational parameter}{section.6}{}}\n";
        assert_eq!(models_page(aux, "spring"), Some(2));
        assert_eq!(models_page(aux, "kepler_mu"), Some(7));
        assert_eq!(models_page(aux, "kepler"), None);
        assert_eq!(models_page("", "spring"), None);
    }

    /// Every viewer entry points at a section of the note, and the note
    /// of this checkout has that section (its label in the source).
    #[test]
    fn every_entry_has_a_section_in_the_models_note() {
        let tex = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../ode-models/docs/models.tex")).unwrap();
        for viz in crate::models::all() {
            let dest = viz.doc_dest().unwrap_or_else(|| panic!("{} has no section", viz.name()));
            assert!(tex.contains(&format!("\\modelsection{{{dest}}}")), "{}: no \\modelsection{{{dest}}}", viz.name());
        }
        for dest in ["spring_mass", "kepler_mu"] {
            assert!(tex.contains(&format!("\\modelsection{{{dest}}}")), "augmented model {dest} has no section");
        }
    }
}
