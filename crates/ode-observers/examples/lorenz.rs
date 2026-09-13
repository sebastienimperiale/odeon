//! Mortensen filter example: Lorenz-63 (3D state space), observing x only.
//!
//! Reference trajectory: the classic parameters from the attractor state
//! (−3.72, −4.20, 20.34) (= t = 11.4 of the (1, 1, 1) start) over t ∈ [0, 20]
//! (see `lorenz_forward`). With x observed, y and z must be
//! recovered through the dynamics. Being dissipative, the flow contracts
//! the density; the model noise `q_diag` keeps it spread — this example
//! uses noise on every component.
//!
//! Outputs go to examples/output/lorenz/.

use ode_observers::filter::{DiffusionScheme, FilterParams, MortensenFilter};
use ode_models::models::lorenz::{DIM, LorenzObservation, LorenzSystem};

fn main() {
    const M: usize = DIM;
    /// 2D outputs: the three coordinate planes.
    const PLOT_PAIRS: &[(usize, usize)] = &[(0, 1), (0, 2), (1, 2)];

    let dt = 0.01;
    let t_end = 20.0;
    let x0 = [-3.716171, -4.204785, 20.339103]; // on the attractor (t = 11.4 of the (1, 1, 1) start)

    // Filter-domain interval per direction, from the ranges measured by
    // `lorenz_forward` (|x| ≲ 20, |y| ≲ 27, 0 < z ≲ 48) with a generous
    // margin: the inverse flow of a dissipative system is expanding, so
    // boundary points are mapped far out and folded back.
    let domain = [(-28.0, 28.0), (-36.0, 36.0), (-10.0, 60.0)];

    println!("Lorenz-63: x0 = {x0:?}, dt = {dt}, t_end = {t_end}\ndomain = {domain:?}");

    let sys = LorenzSystem::new(x0, dt, LorenzObservation::X);

    // ── Mortensen filter ─────────────────────────────────────────────────────
    let params = FilterParams {
        domain,
        eps: 1.0, // ε — the attractor spans tens of units, so ε is O(1) here
        gamma: 1.0, // γ — observation weight (1 = standard cycle)
        n_el: [16; M],
        p_ord: [4; M],
        diffusion: DiffusionScheme::SplitEuler, // directionally split → nodal positivity
        q_diag: [1.0; M], // model noise on every component (keeps p from collapsing)
        periodic: [false; M],
        dirichlet: [false; M],
        pre_compute_flow_inv: true,
    };
    let sigma = 0.1; // initial condition: V_0(x) = σ·|x − x0|²/2

    println!("building HoFFT solver and φ⁻¹ cache…");
    let mut filter = MortensenFilter::<M, _>::new(sys, params);
    filter.init_filter_quadratic_at(sigma, x0);

    let n_steps = (t_end / dt).ceil() as usize;
    filter.run(n_steps, 25, PLOT_PAIRS, "examples/output/lorenz");
}
