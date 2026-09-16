//! Forward solve only: the single mass–spring reference trajectory, no
//! filter (same setup as the `one_spring` example).
//!
//! Use this to inspect the range each state component actually visits before
//! choosing the filter-domain intervals of the `one_spring` example. Prints
//! the per-component min/max and writes the trajectory for
//! `visualize_model.py`.
//!
//! Outputs go to examples/output/one_spring_forward/.

use ode_models::models::SpringSystem;
use ode_observers::output::save_forward;
use ode_models_spec::noise::NoiseModel;
use ode_models_spec::progress::Progress;
use ode_models_spec::reference::Reference;

fn main() {
    const N: usize = 1;
    const M: usize = 2 * N;

    let rho = 1.0;
    let a = 1.0;
    let dt = 0.005;
    let t_end = 3.0;

    // Initial data: displacements Y(0) = [ℓ, 2ℓ, …, Nℓ] — a stretched chain (the
    // model's equilibrium is Y = 0: y are displacements from rest), V(0) = 0
    let ell = 1.0 / N as f64;
    let x0: [f64; M] = std::array::from_fn(|d| if d < N { (d + 1) as f64 * ell } else { 0.0 });

    // The observation is a filter-only concern; a pure forward solve never
    // uses it, so pass a placeholder.
    let sys = SpringSystem::new(N, rho, a, dt, |x: &[f64]| x[0]);

    let n_steps = (t_end / dt).ceil() as usize;
    let reference = Reference::twin(&sys, x0, n_steps, NoiseModel::None, 0, None, &Progress::default());

    save_forward::<M, _>(&sys, &reference, "examples/output/one_spring_forward");
}
