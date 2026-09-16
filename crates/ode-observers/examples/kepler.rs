//! Mortensen filter example: softened planar Kepler orbit (4D state space).
//!
//! Reference trajectory: the e = 0.5 orbit of semi-major axis 1 started at
//! perihelion (see `kepler_forward`), two periods. Observation h(x) = |q|
//! (the range): the polar angle is unobserved, so the density should be
//! ring-like in the (q₁, q₂) plane until the dynamics pin the phase.
//! Change `OBSERVATION` to `Q1` or `Bearing` for the other two.
//!
//! Outputs go to examples/output/kepler/.

use ode_observers::methods::mortensen::{DiffusionScheme, FilterParams, MortensenFilter};
use ode_models::models::kepler::{DIM, KeplerObservation, KeplerSystem, perihelion_state};
use ode_models_spec::noise::NoiseModel;
use ode_models_spec::progress::Progress;
use ode_models_spec::reference::Reference;

fn main() {
    const M: usize = DIM;
    /// 2D outputs: the orbital plane (q₁, q₂), the momentum plane (p₁, p₂),
    /// and the two (qᵢ, pᵢ) phase planes.
    const PLOT_PAIRS: &[(usize, usize)] = &[(0, 1), (2, 3), (0, 2), (1, 3)];
    const OBSERVATION: KeplerObservation = KeplerObservation::Range;

    let dt = 0.01;
    let t_end = 4.0 * std::f64::consts::PI;

    let x0 = perihelion_state(0.5);

    // Filter-domain interval per state-space direction, from the ranges
    // measured by `kepler_forward` over two periods:
    //   q₁ ∈ [−1.68, 0.50]  → (−2.2, 1.0)
    //   q₂ ∈ [−0.86, 1.05]  → (−1.4, 1.6)
    //   p₁ ∈ [−1.10, 1.19]  → (−1.6, 1.7)
    //   p₂ ∈ [−0.53, 1.73]  → (−1.1, 2.3)
    let domain = [(-2.2, 1.0), (-1.4, 1.6), (-1.6, 1.7), (-1.1, 2.3)];

    println!(
        "softened Kepler: x0 = {x0:?}, dt = {dt}, t_end = {t_end:.3}, observation {OBSERVATION:?}\ndomain = {domain:?}"
    );

    let sys = KeplerSystem::new(dt, OBSERVATION);

    // ── Mortensen filter ─────────────────────────────────────────────────────
    let params = FilterParams {
        domain,
        eps: 0.05, // ε — diffusion / temperature parameter
        gamma: 1.0, // γ — observation weight (1 = standard cycle)
        n_el: [8; M],
        p_ord: [4; M],
        diffusion: DiffusionScheme::SplitEuler, // directionally split → nodal positivity
        q_diag: [1.0; M], // isotropic model noise (diagonal of Q)
        periodic: [false; M], // Cartesian box, no periodic direction
        dirichlet: [false; M], // Neumann (reflecting) walls; true = absorbing p = 0
        // Autonomous flow: cache φ⁻¹ of the grid once instead of one Newton
        // solve per DOF per iteration.
        pre_compute_flow_inv: true,
    };
    let sigma = 2.0; // initial condition: V_0(x) = σ·|x|²/2

    println!("building HoFFT solver and φ⁻¹ cache…");
    let mut filter = MortensenFilter::<M, _>::new(sys, params);
    filter.init_filter_quadratic(sigma);

    let n_steps = (t_end / dt).ceil() as usize;
    // The twin experiment: the reference orbit and its (noiseless) observations.
    let reference = Reference::twin(&filter.model, x0, n_steps, NoiseModel::None, 0, None, &Progress::default());
    filter.run_and_save(&reference, 25, PLOT_PAIRS, "examples/output/kepler");
}
