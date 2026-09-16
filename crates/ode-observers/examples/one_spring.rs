//! Mortensen filter example: single massâspring (N = 1, 2D state space).
//!
//! Forward problem: N-harmonic oscillator (Chapelle & Moireau)
//!
//!   M Å¸ + K Y = 0
//!
//! with  M = Ïâ I   and  K = (a/â) tridiag(-1, 2, -1)
//!   (last row [-1, 1] : right end free, left end fixed to wall)
//!
//! Outputs go to examples/output/one_spring/.

use ode_observers::methods::mortensen::{DiffusionScheme, FilterParams, MortensenFilter};
use ode_models::models::SpringSystem;
use ode_models_spec::noise::NoiseModel;
use ode_models_spec::progress::Progress;
use ode_models_spec::reference::Reference;

fn main() {
    const N: usize = 1;
    const M: usize = 2 * N;
    /// 2D outputs: pairs of state-space directions (indices 0..N are y_1..y_N,
    /// indices N..2N are v_1..v_N).
    const PLOT_PAIRS: &[(usize, usize)] = &[(0, 1)];

    let rho = 1.0;
    let a = 1.0;
    let dt = 0.005;
    let t_end = 3.0;

    // Observation function h(x), x = [Y; V]: observe only y_1, the position
    // of the first mass (partial observation).
    let h_obs = |x: &[f64]| x[0];

    // Initial data: displacements Y(0) = [ℓ, 2ℓ, …, Nℓ] — a stretched chain (the
    // model's equilibrium is Y = 0: y are displacements from rest), V(0) = 0
    let ell = 1.0 / N as f64;
    let x0: [f64; M] = std::array::from_fn(|d| if d < N { (d + 1) as f64 * ell } else { 0.0 });

    let sys = SpringSystem::new(N, rho, a, dt, h_obs);

    println!("\nTransition matrix T (2NÃ2N, state = [Y; V]):");
    println!("{:.6}", sys.trans);
    println!("Inverse transition matrix Tâ»Â¹:");
    println!("{:.6}", sys.trans_inv);

    // ââ Mortensen filter âââââââââââââââââââââââââââââââââââââââââââââââââââââ
    let params = FilterParams {
        // Filter-domain interval per state-space direction
        domain: [(-3.0, 3.0); M],
        eps: 0.01,      // Îµ â diffusion / temperature parameter
        gamma: 1.0, // γ — observation weight (1 = standard cycle)
        n_el: [20; M],  // elements per direction
        p_ord: [4; M],  // polynomial order per direction
        diffusion: DiffusionScheme::SplitEuler, // directionally split → nodal positivity
        q_diag: [1.0; M], // isotropic model noise (diagonal of Q)
        periodic: [false; M], // positions/velocities live on a plain box
        dirichlet: [false; M], // Neumann (reflecting) walls; true = absorbing p = 0
        // The linear flow (one matvec per DOF) is too cheap to be worth caching.
        pre_compute_flow_inv: false,
    };
    let sigma = 10.0_f64; // initial condition: V_0(x) = ÏÂ·|x|Â²/2

    let mut filter = MortensenFilter::<M, _>::new(sys, params);
    filter.init_filter_quadratic(sigma);

    let n_steps = (t_end / dt).ceil() as usize;
    // The twin experiment: the reference trajectory and its (noiseless) observations.
    let reference = Reference::twin(&filter.model, x0, n_steps, NoiseModel::None, 0, None, &Progress::default());
    filter.run_and_save(&reference, 25, PLOT_PAIRS, "examples/output/one_spring");
}
