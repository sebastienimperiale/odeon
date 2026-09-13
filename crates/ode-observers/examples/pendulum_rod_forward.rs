//! Forward solve only: the double compound (rigid-rod) pendulum reference
//! trajectory, no filter — the swaptube model.
//!
//! Prints the per-component min/max and writes the trajectory for
//! `visualize_model.py` / `visualize_pendulum.py`.
//!
//! Outputs go to examples/output/pendulum_rod_forward/.

use ode_models::models::pendulum_rod::{DIM, PendulumRodSystem};
use ode_observers::output::save_forward;
use std::io::Write;

fn main() {
    let dt = 0.01;
    let t_end = 10.0;

    // Initial data: released from rest.
    let x0 = [2.453, -2.7727, 0.0, 0.0];

    let mut sys = PendulumRodSystem::new(x0, dt);

    let n_steps = (t_end / dt).ceil() as usize;
    for _ in 0..n_steps {
        sys.forward();
    }

    let out_dir = "examples/output/pendulum_rod_forward";
    save_forward::<DIM, _>(&sys, out_dir);

    // The (x, y)-plane script needs the rod lengths; the model-agnostic meta
    // written by save_forward cannot know them, so append (both rods = L).
    let mut meta = std::fs::OpenOptions::new()
        .append(true)
        .open(format!("{out_dir}/meta.txt"))
        .expect("meta.txt should have been written by save_forward");
    let phys = ode_models::models::pendulum_rod::PendulumRodParams::default();
    writeln!(meta, "l1={}\nl2={}", phys.l, phys.l).unwrap();
}
