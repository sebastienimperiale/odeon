//! Mortensen filter example: planar double compound (rigid-rod) pendulum
//! (4D state space) on the heart-shaped orbit.
//!
//! Reference trajectory: released from rest at xâ = (2.453, â2.7727, 0, 0)
//! (the swaptube heart orbit; see `pendulum_rod_forward`). Observation
//! h(x) = (x, y) tip position (angles observed through the geometry,
//! momenta unobserved).
//!
//! Both angles are periodic: the orbit winds qâ through full rotations
//! (qâ reaches â 15 rad over t â [0, 10]), so the angle box (âÏ, Ï) only
//! makes sense with periodic wrap.
//!
//! Outputs go to examples/output/pendulum_rod/.

use ode_observers::methods::mortensen::{DiffusionScheme, FilterParams, MortensenFilter};
use ode_models::models::pendulum_rod::{DIM, PendulumRodSystem};
use ode_models_spec::noise::NoiseModel;
use ode_models_spec::progress::Progress;
use ode_models_spec::reference::Reference;
use std::f64::consts::PI;

fn main() {
    const M: usize = DIM;
    /// 2D outputs: pairs of state-space directions (0, 1 are qâ, qâ; 2, 3 are
    /// pâ, pâ).
    const PLOT_PAIRS: &[(usize, usize)] = &[(0, 2), (1, 3), (0, 1), (2, 3)];

    let dt = 0.01;
    let t_end = 10.0;

    // Initial data: the heart orbit, released from rest.
    let x0 = [2.453, -2.7727, 0.0, 0.0];

    // Filter-domain interval per state-space direction (angles then momenta),
    // from the trajectory ranges measured by `pendulum_rod_forward` over
    // t â [0, 10]:
    //   qâ â [â2.46, 2.46], qâ winds (periodic wrap) â (âÏ, Ï) each
    //   pâ â [â9.2, 9.2]                             â (â12, 12)
    //   pâ â [â3.3, 3.3]                             â (â5, 5)
    let domain = [(-PI, PI), (-PI, PI), (-14.0, 14.0), (-7.0, 7.0)];

    println!("double rod pendulum: x0 = {x0:?}, dt = {dt}, t_end = {t_end}\ndomain = {domain:?}");

    let sys = PendulumRodSystem::new(dt);

    // ââ Mortensen filter âââââââââââââââââââââââââââââââââââââââââââââââââââââ
    let params = FilterParams {
        domain,
        eps: 0.1, // Îµ â diffusion / temperature parameter
        gamma: 1.0, // γ — observation weight (1 = standard cycle)
        // More elements where the domain is wide (pâ), so the element size
        // stays comparable across directions: 0.63, 0.63, 2.0, 1.67.
        n_el: [10, 10, 20, 10],
        p_ord: [6; M], // polynomial order per direction
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
    filter.run_and_save(&reference, 25, PLOT_PAIRS, "examples/output/pendulum_rod");

    // The (x, y)-plane script needs the rod lengths; the model-agnostic
    // filter meta cannot know them, so append them here (both rods = L).
    use std::io::Write;
    let mut meta = std::fs::OpenOptions::new()
        .append(true)
        .open("examples/output/pendulum_rod/meta.txt")
        .expect("meta.txt should have been written by run()");
    let phys = ode_models::models::pendulum_rod::PendulumRodParams::default();
    writeln!(meta, "l1={}\nl2={}", phys.l, phys.l).unwrap();
}
