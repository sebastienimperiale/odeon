//! Timing: pendulum transport with per-iteration flow evaluation (`convect`)
//! vs the precomputed transport plan (`pre_compute_flow_inv`, `eval_with_plan`).
//!
//! Pendulum setup at a reduced resolution (21⁴ ≈ 194k DOFs), and only a few
//! filter iterations: enough to time a steady per-iteration cost, cheap
//! enough to run routinely.
//!
//! Run with `cargo bench --bench transport`.

use ode_observers::filter::{DiffusionScheme, FilterParams, MortensenFilter};
use ode_models::models::pendulum::{DIM, PendulumSystem};
use std::f64::consts::PI;
use std::time::Instant;

const N_STEPS: usize = 10;

/// Build the pendulum filter; returns the filter and the construction time
/// (which includes building the transport plan when `pre_compute_flow_inv`
/// is on).
fn build(pre_compute_flow_inv: bool) -> (MortensenFilter<DIM, PendulumSystem>, f64) {
    let sys = PendulumSystem::new([1.5, 1.4, 0.0, 0.0], 0.01);
    let t0 = Instant::now();
    let mut filter = MortensenFilter::<DIM, _>::new(
        sys,
        FilterParams {
            domain: [(-PI, PI), (-PI, PI), (-3.0, 3.0), (-3.0, 3.0)],
            eps: 0.1,
            gamma: 1.0, // γ — observation weight (1 = standard cycle)
            n_el: [5; DIM],
            p_ord: [4; DIM],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: [1.0; DIM],
            periodic: [false; DIM],
            dirichlet: [false; DIM],
            pre_compute_flow_inv,
        },
    );
    let setup = t0.elapsed().as_secs_f64();
    filter.init_filter_quadratic(5.0);
    (filter, setup)
}

/// Returns (setup seconds, seconds per iteration).
fn bench(label: &str, pre: bool) -> (f64, f64) {
    let (mut filter, setup) = build(pre);
    let t0 = Instant::now();
    for _ in 0..N_STEPS {
        filter.forward();
    }
    let per_step = t0.elapsed().as_secs_f64() / N_STEPS as f64;
    println!("{label:<22}  setup {setup:7.3}s   {per_step:7.4}s / iteration");
    (setup, per_step)
}

/// The translating window on the same pendulum: a 5⁴-element box of
/// half-width 1 (19⁴ ≈ 130k DOFs), preimage cache (plan path, refreshed on
/// the entering elements when the window moves) vs `convect`.
fn bench_window(label: &str, pre: bool) -> f64 {
    use ode_observers::box_tracker::{BoxTracker, BoxTrackerParams};
    let sys = PendulumSystem::new([1.5, 1.4, 0.0, 0.0], 0.01);
    let t0 = Instant::now();
    let mut window = BoxTracker::<DIM, _>::new(
        sys,
        BoxTrackerParams {
            half_width: [1.0; DIM],
            eps: 0.1,
            gamma: 1.0,
            n_el: [5; DIM],
            p_ord: [4; DIM],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: [1.0; DIM],
            pre_compute_flow_inv: pre,
        },
    );
    window.init_gaussian([5.0; DIM], [1.5, 1.4, 0.0, 0.0]);
    let setup = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let mut moves = 0;
    for _ in 0..N_STEPS {
        moves += usize::from(window.forward().iter().any(|&k| k != 0));
    }
    let per_step = t0.elapsed().as_secs_f64() / N_STEPS as f64;
    println!("{label:<22}  setup {setup:7.3}s   {per_step:7.4}s / iteration   ({moves} window moves)");
    per_step
}

fn main() {
    println!("Pendulum, {N_STEPS} iterations each:\n");
    let (setup_conv, convect) = bench("convect (on the fly)", false);
    let (setup_cache, cached) = bench("transport plan", true);
    let saved_per_step = convect - cached;
    println!(
        "\nper-iteration speedup: ×{:.2}   extra setup {:.2}s ≈ {:.1} iterations to amortize",
        convect / cached,
        setup_cache - setup_conv,
        (setup_cache - setup_conv) / saved_per_step,
    );

    println!("\nTranslating window on the pendulum, {N_STEPS} iterations each:\n");
    let w_conv = bench_window("window, convect", false);
    let w_plan = bench_window("window, preimage cache", true);
    println!("\nper-iteration speedup: ×{:.2}", w_conv / w_plan);
}
