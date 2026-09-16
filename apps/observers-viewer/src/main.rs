//! The observers viewer, computing locally: pick a model, set its
//! parameters, run the trajectory, replay it as an animated scene with the
//! observation signal below, then run the filter, the box tracker, the
//! particles or the tracker on it — on a worker thread of this process.
//! `observers-client` is the same viewer sending its jobs to an
//! `observers-server`.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use ode_observers_egui::app::{App, LocalCompute};

struct Viewer(App<LocalCompute>);

impl eframe::App for Viewer {
    fn ui(&mut self, ui: &mut eframe::egui::Ui, _frame: &mut eframe::Frame) {
        self.0.ui(ui);
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1150.0, 780.0])
            .with_title("Odeon — observers"),
        ..Default::default()
    };
    eframe::run_native(
        "Odeon observers",
        options,
        Box::new(|_cc| Ok(Box::new(Viewer(App::new(LocalCompute))))),
    )
}
