//! The models-only application ([`App`]): left panel = model choice +
//! parameter form + run settings; central panel = the animated scene;
//! bottom panel = playback controls and the observation plot. Each model
//! keeps its own trajectory and playback clock, so switching models never
//! loses a run. Everything is drawn by [`crate::panels`]; this module only
//! wires the three panels together, and a binary is `App::new()` plus the
//! eframe boilerplate (this crate knows egui, not eframe). The observers
//! application of `ode-observers-egui` is the same with the estimators
//! beside it.

use crate::models;
use crate::panels;
use crate::slot::ModelSlot;

/// The models-only application. Call [`ui`](Self::ui) once per frame from
/// the eframe `App::ui`.
pub struct App {
    slots: Vec<ModelSlot>,
    selected: usize,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Every model of the menu at its initial condition.
    pub fn new() -> Self {
        App {
            slots: models::all().into_iter().map(ModelSlot::new).collect(),
            selected: 0,
        }
    }

    /// One frame: the three panels.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // Collect finished background runs (all slots, not just the visible
        // one) and start their playback.
        for slot in &mut self.slots {
            slot.poll();
        }

        egui::Panel::left("params")
            .resizable(true)
            .default_size(280.0)
            .min_size(250.0)
            .show(ui, |ui| {
                ui.add_space(4.0);
                ui.heading("Model");
                let names: Vec<&str> = self.slots.iter().map(|s| s.model.name()).collect();
                panels::model_selector(ui, &names, &mut self.selected);
                let slot = &mut self.slots[self.selected];
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    panels::params_ui(ui, slot);
                    ui.add_space(6.0);
                    panels::run_ui(ui, slot, true);
                });
                // Pin the content width to the panel width (same creep
                // prevention as the bottom panel's take_available_height).
                ui.take_available_width();
            });

        egui::Panel::bottom("obs_panel")
            .resizable(true)
            .default_size(230.0)
            .min_size(170.0)
            .show(ui, |ui| {
                panels::observation_panel(ui, &mut self.slots[self.selected]);
            });

        egui::CentralPanel::default().show(ui, |ui| {
            let slot = &mut self.slots[self.selected];
            let wall_dt = ui.input(|i| i.stable_dt) as f64;
            slot.advance(wall_dt);
            panels::scene_ui(ui, slot, self.selected);
        });

        // Keep animating while something moves or computes.
        if self.slots[self.selected].playback.playing || self.slots.iter().any(|s| s.run.is_some()) {
            ui.ctx().request_repaint();
        }
    }
}
