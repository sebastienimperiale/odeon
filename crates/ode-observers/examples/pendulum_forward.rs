//! Forward solve only: the double pendulum reference trajectory, no filter.
//!
//! Use this to inspect the range each state component actually visits before
//! choosing the filter-domain intervals of the `pendulum` example. Prints the
//! per-component min/max and writes the trajectory for `visualize_model.py`.
//!
//! Outputs go to examples/output/pendulum_forward/.

use ode_models::models::pendulum::{DIM, PendulumSystem};
use ode_observers::output::save_forward;
use ode_models_spec::noise::NoiseModel;
use ode_models_spec::progress::Progress;
use ode_models_spec::reference::Reference;
use std::io::Write;

fn main() {
    let dt = 0.01;
    let t_end = 20.0;

    // Initial data: the paper's target trajectory (same as the pendulum example)
    let x0 = [2.453, -2.7727, 0.0, 0.0];

    let sys = PendulumSystem::new(dt);

    let n_steps = (t_end / dt).ceil() as usize;
    let reference = Reference::twin(&sys, x0, n_steps, NoiseModel::None, 0, None, &Progress::default());

    let out_dir = "examples/output/pendulum_forward";
    save_forward::<DIM, _>(&sys, &reference, out_dir);

    // The (x, y)-plane script needs the rod lengths; the model-agnostic meta
    // written by save_forward cannot know them, so append.
    let mut meta = std::fs::OpenOptions::new()
        .append(true)
        .open(format!("{out_dir}/meta.txt"))
        .expect("meta.txt should have been written by save_forward");
    let phys = ode_models::models::pendulum::PendulumParams::default();
    writeln!(meta, "l1={}\nl2={}", phys.l1, phys.l2).unwrap();
}
