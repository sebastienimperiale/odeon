//! Forward solve only: the Lorenz-63 reference trajectory, no filter.
//!
//! Prints the per-component min/max (use them to choose the filter domain
//! in `lorenz`) and writes the trajectory for `visualize_model.py`.
//!
//! Outputs go to examples/output/lorenz_forward/.

use ode_models::models::lorenz::{DIM, LorenzObservation, LorenzSystem};
use ode_observers::output::save_forward;

fn main() {
    let dt = 0.01;
    let t_end = 20.0;

    // Initial data on the attractor (the state at t = 11.4 of the usual
    // (1, 1, 1) start, which has a short transient).
    let x0 = [-3.716171, -4.204785, 20.339103]; // on the attractor (t = 11.4 of the (1, 1, 1) start)

    let mut sys = LorenzSystem::new(x0, dt, LorenzObservation::X);

    let n_steps = (t_end / dt).ceil() as usize;
    for _ in 0..n_steps {
        sys.forward();
    }

    save_forward::<DIM, _>(&sys, "examples/output/lorenz_forward");
}
