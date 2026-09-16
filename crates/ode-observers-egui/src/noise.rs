//! UI for the library's observation-noise models
//! ([`ode_models_spec::noise::NoiseModel`]): a selector plus parameter fields,
//! one row per observation, shown in the Observation section. The noise
//! itself lives in the library: the reference generator draws it with the
//! entry's seed, and the run's observations are what the estimators
//! consume — what is plotted is exactly what the filter sees.

pub use ode_models_spec::noise::NoiseModel;

const KIND_NAMES: [&str; 4] = ["no noise", "Gaussian", "Uniform", "AR(1)"];

fn kind(m: &NoiseModel) -> usize {
    match m {
        NoiseModel::None => 0,
        NoiseModel::Gaussian { .. } => 1,
        NoiseModel::Uniform { .. } => 2,
        NoiseModel::Ar1 { .. } => 3,
    }
}

fn of_kind(k: usize) -> NoiseModel {
    match k {
        1 => NoiseModel::Gaussian { std: 0.05 },
        2 => NoiseModel::Uniform { half_width: 0.05 },
        3 => NoiseModel::Ar1 {
            std: 0.05,
            tau: 0.5,
        },
        _ => NoiseModel::None,
    }
}

/// Model selector + parameter fields on one row; returns `true` when
/// anything was edited.
pub fn noise_ui(
    model: &mut NoiseModel,
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        let mut k = kind(model);
        egui::ComboBox::from_id_salt(id)
            .width(92.0)
            .selected_text(KIND_NAMES[k])
            .show_ui(ui, |ui| {
                for (i, name) in KIND_NAMES.iter().enumerate() {
                    ui.selectable_value(&mut k, i, *name);
                }
            });
        if k != kind(model) {
            *model = of_kind(k);
            changed = true;
        }
        let drag = |ui: &mut egui::Ui, v: &mut f64| {
            ui.add(egui::DragValue::new(v).speed(0.005).range(0.0..=1e3))
                .changed()
        };
        match model {
            NoiseModel::None => {}
            NoiseModel::Gaussian { std } => {
                changed |= drag(ui, std);
                ui.label("σ");
            }
            NoiseModel::Uniform { half_width } => {
                changed |= drag(ui, half_width);
                ui.label("a");
            }
            NoiseModel::Ar1 { std, tau } => {
                changed |= drag(ui, std);
                ui.label("σ");
                changed |= ui
                    .add(egui::DragValue::new(tau).speed(0.01).range(1e-3..=1e3))
                    .changed();
                ui.label("τ");
            }
        }
    });
    changed
}
