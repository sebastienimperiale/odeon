//! Forward solve only: the softened planar Kepler reference orbit, no
//! filter.
//!
//! Prints the per-component min/max (use them to choose the filter domain
//! in `kepler`) and writes the trajectory for `visualize_model.py`.
//!
//! Outputs go to examples/output/kepler_forward/.

use ode_models::models::kepler::{DIM, KeplerObservation, KeplerSystem, perihelion_state};
use ode_observers::output::save_forward;
use ode_models_spec::noise::NoiseModel;
use ode_models_spec::progress::Progress;
use ode_models_spec::reference::Reference;

fn main() {
    let dt = 0.01;
    let t_end = 4.0 * std::f64::consts::PI; // two periods

    // Initial data: perihelion of the e = 0.5 orbit of semi-major axis 1.
    let x0 = perihelion_state(0.5);

    let sys = KeplerSystem::new(dt, KeplerObservation::Q1);

    let n_steps = (t_end / dt).ceil() as usize;
    let reference = Reference::twin(&sys, x0, n_steps, NoiseModel::None, 0, None, &Progress::default());

    save_forward::<DIM, _>(&sys, &reference, "examples/output/kepler_forward");
}
