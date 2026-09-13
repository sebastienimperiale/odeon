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
use nalgebra::DVector;

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
    let y0 = DVector::from_fn(N, |n, _| (n + 1) as f64 * ell);
    let v0 = DVector::zeros(N);

    // The observation is a filter-only concern; a pure forward solve never
    // uses it, so pass a placeholder.
    let mut sys = SpringSystem::new(N, rho, a, dt, y0, v0, |x: &[f64]| x[0]);

    let n_steps = (t_end / dt).ceil() as usize;
    for _ in 0..n_steps {
        sys.forward();
    }

    save_forward::<M, _>(&sys, "examples/output/one_spring_forward");
}
