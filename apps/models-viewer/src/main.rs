//! The models-only viewer: pick a forward model, set its parameters, run
//! the trajectory and replay it as an animated scene with the observation
//! signal below. No estimator — that is `observers-viewer` (local) or
//! `observers-client` (on a server).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use ode_models_egui::app::App;

struct Viewer(App);

impl eframe::App for Viewer {
    fn ui(&mut self, ui: &mut eframe::egui::Ui, _frame: &mut eframe::Frame) {
        self.0.ui(ui);
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0])
            .with_title("Odeon — models"),
        ..Default::default()
    };
    eframe::run_native("Odeon models", options, Box::new(|_cc| Ok(Box::new(Viewer(App::new())))))
}
