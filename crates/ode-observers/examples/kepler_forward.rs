//! Forward solve only: the softened planar Kepler reference orbit, no
//! filter.
//!
//! Prints the per-component min/max (use them to choose the filter domain
//! in `kepler`) and writes the trajectory for `visualize_model.py`.
//!
//! Outputs go to examples/output/kepler_forward/.

use ode_models::models::kepler::{DIM, KeplerObservation, KeplerSystem, perihelion_state};
use ode_observers::output::save_forward;

fn main() {
    let dt = 0.01;
    let t_end = 4.0 * std::f64::consts::PI; // two periods

    // Initial data: perihelion of the e = 0.5 orbit of semi-major axis 1.
    let x0 = perihelion_state(0.5);

    let mut sys = KeplerSystem::new(x0, dt, KeplerObservation::Q1);

    let n_steps = (t_end / dt).ceil() as usize;
    for _ in 0..n_steps {
        sys.forward();
    }

    save_forward::<DIM, _>(&sys, "examples/output/kepler_forward");
}
