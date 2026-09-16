//! Mortensen filter example: planar double pendulum (4D state space).
//!
//! Model from pendulum.pdf Â§5.2: state x = (qâ, qâ, pâ, pâ), reference
//! trajectory from xâ = (1.5, 1.4, 0, 0) with dt = 10â»Â² (GaussâLegendre 4,
//! Newton). Observation h(x) = (x, y) tip position (angles observed through
//! the geometry, momenta unobserved). The filter domain is
//! sized from the state ranges measured by the `pendulum_forward` example
//! (angles stay within Â±2.5; the momenta reach |pâ| â 26 and |pâ| â 9),
//! plus margin for the density spread.
//!
//! Heavy run: the grid has 41Â·41Â·121Â·49 â 10M DOFs at the default
//! resolution; the flow's Newton iteration runs once per DOF at
//! construction (`pre_compute_flow_inv`), not at every transport step.
//!
//! Outputs go to examples/output/pendulum/.

use ode_observers::methods::mortensen::{DiffusionScheme, FilterParams, MortensenFilter};
use ode_models::models::pendulum::{DIM, PendulumSystem};
use ode_models_spec::noise::NoiseModel;
use ode_models_spec::progress::Progress;
use ode_models_spec::reference::Reference;
use std::f64::consts::PI;

fn main() {
    const M: usize = DIM;
    /// 2D outputs: pairs of state-space directions (0, 1 are qâ, qâ; 2, 3 are
    /// pâ, pâ).
    const PLOT_PAIRS: &[(usize, usize)] = &[(0, 2), (1, 3)];

    let dt = 0.01;
    let t_end = 10.0;

    // Initial data: the paper's target trajectory
    let x0 = [1.5, 1.4, 0.0, 0.0];

    // Filter-domain interval per state-space direction (angles then momenta),
    // from the trajectory ranges measured by `pendulum_forward` over t â [0, 6]:
    //   qâ â [â1.45, 1.69], qâ â [â2.41, 1.41] â (âÏ, Ï)
    //   pâ â [â25.6, 24.3]                     â (â32, 32)
    //   pâ â [â9.0, 9.2]                       â (â13, 13)
    let domain = [(-PI, PI), (-PI, PI), (-32.0, 32.0), (-13.0, 13.0)];

    println!("double pendulum: x0 = {x0:?}, dt = {dt}, t_end = {t_end}\ndomain = {domain:?}");

    let sys = PendulumSystem::new(dt);

    // ââ Mortensen filter âââââââââââââââââââââââââââââââââââââââââââââââââââââ
    let params = FilterParams {
        domain,
        eps: 0.1, // Îµ â diffusion / temperature parameter
        gamma: 1.0, // γ — observation weight (1 = standard cycle)
        // More elements where the domain is wide (pâ, pâ), so the element
        // size stays comparable across directions: 0.63, 0.63, 2.0, 2.0.
        n_el: [10, 10, 30, 12],
        p_ord: [4; M], // polynomial order per direction
        diffusion: DiffusionScheme::SplitEuler, // directionally split → nodal positivity
        q_diag: [1.0; M], // isotropic model noise (diagonal of Q)
        // The two angles are periodic on (âÏ, Ï): the transport wraps them
        // mod 2Ï and the diffusion uses periodic boundary conditions there.
        periodic: [true, true, false, false],
        dirichlet: [false; 4],
        // Autonomous flow: cache Ïâ»Â¹ of the grid once instead of one Newton
        // solve per DOF per iteration (costs one [f64; 4] per DOF in memory).
        pre_compute_flow_inv: true,
    };
    let sigma = 1.0; // initial condition: V_0(x) = ÏÂ·|x|Â²/2

    println!("building HoFFT solver and Ïâ»Â¹ cache (may take a moment at high resolution)â¦");
    let mut filter = MortensenFilter::<M, _>::new(sys, params);
    filter.init_filter_quadratic(sigma);

    let n_steps = (t_end / dt).ceil() as usize;
    let reference = Reference::twin(&filter.model, x0, n_steps, NoiseModel::None, 0, None, &Progress::default());
    filter.run_and_save(&reference, 25, PLOT_PAIRS, "examples/output/pendulum");

    // The (x, y)-plane script needs the rod lengths; the model-agnostic
    // filter meta cannot know them, so append them here.
    use std::io::Write;
    let mut meta = std::fs::OpenOptions::new()
        .append(true)
        .open("examples/output/pendulum/meta.txt")
        .expect("meta.txt should have been written by run()");
    let phys = ode_models::models::pendulum::PendulumParams::default();
    writeln!(meta, "l1={}\nl2={}", phys.l1, phys.l2).unwrap();
}
