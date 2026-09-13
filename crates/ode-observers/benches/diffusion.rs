//! Timing: the diffusion step's spectral solve, `lobatto-fft` (HoFFT: FFT
//! across the elements, r × r symbol per Fourier mode, complex data) vs
//! `lobatto-spectral` (dense generalized eigendecomposition of the n·r × n·r
//! 1D operator, real data), both through their `apply_in_place_dirs` with the
//! filter's split-Euler symbol ∏_d (1 + dt·β_d μ_d)⁻¹.
//!
//! The grids are the 4D ones of the repository: the transport bench (21⁴),
//! `kepler` (33⁴), `pendulum` (9.5M), `pendulum_rod` (26.6M, p_ord 6) and
//! `two_springs` (43M), mixed periodic/Neumann where the example has them.
//! The HoFFT path is timed as the filter runs it — real → complex copy, apply,
//! real part back — and also as the bare apply; the eigen path works on the
//! real field directly. The two results are compared (max abs difference).
//!
//! Run with `cargo bench --bench diffusion`; `--heavy` adds the two largest
//! grids (≈ 2–3 GB of buffers, minutes of runtime).

use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::time::Instant;

const N_REPS: usize = 7;
const DIM: usize = 4;

struct Case {
    name: &'static str,
    domain: [(f64, f64); DIM],
    n_el: [usize; DIM],
    p_ord: [usize; DIM],
    periodic: [bool; DIM],
    eps: f64,
    dt: f64,
}

fn cases(heavy: bool) -> Vec<Case> {
    let mut v = vec![
        Case {
            name: "transport bench 21⁴",
            domain: [(-PI, PI), (-PI, PI), (-3.0, 3.0), (-3.0, 3.0)],
            n_el: [5; DIM],
            p_ord: [4; DIM],
            periodic: [false; DIM],
            eps: 0.1,
            dt: 0.01,
        },
        Case {
            name: "kepler 33⁴",
            domain: [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
            n_el: [8; DIM],
            p_ord: [4; DIM],
            periodic: [false; DIM],
            eps: 0.05,
            dt: 0.02,
        },
        Case {
            name: "pendulum 9.5M",
            domain: [(-PI, PI), (-PI, PI), (-30.0, 30.0), (-12.0, 12.0)],
            n_el: [10, 10, 30, 12],
            p_ord: [4; DIM],
            periodic: [true, true, false, false],
            eps: 0.1,
            dt: 0.01,
        },
    ];
    if heavy {
        v.push(Case {
            name: "pendulum_rod 26.6M p6",
            domain: [(-PI, PI), (-PI, PI), (-20.0, 20.0), (-8.0, 8.0)],
            n_el: [10, 10, 20, 10],
            p_ord: [6; DIM],
            periodic: [true, true, false, false],
            eps: 0.1,
            dt: 0.01,
        });
        v.push(Case {
            name: "two_springs 43M",
            domain: [(-2.0, 2.0); DIM],
            n_el: [20; DIM],
            p_ord: [4; DIM],
            periodic: [false; DIM],
            eps: 0.01,
            dt: 0.01,
        });
    }
    v
}

/// Deterministic pseudo-random positive field (a density-like p ∈ (0, 1]).
fn random_field(n: usize) -> Vec<f64> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64).max(1e-12)
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn run_case(c: &Case) {
    let ls: [f64; DIM] = std::array::from_fn(|d| c.domain[d].1 - c.domain[d].0);
    let xls: [f64; DIM] = std::array::from_fn(|d| c.domain[d].0);
    let beta: [f64; DIM] = std::array::from_fn(|_| c.eps / 2.0); // q_d = 1
    let dt = c.dt;
    let symbol = move |mu: [f64; DIM]| -> f64 {
        (0..DIM)
            .map(|d| 1.0 / (1.0 + dt * beta[d] * mu[d]))
            .product::<f64>()
    };

    // ── lobatto-fft (HoFFT) ────────────────────────────────────────────────
    let t0 = Instant::now();
    let fft = lobatto_fft::solver::PoissonND::<DIM>::new(
        ls,
        xls,
        c.n_el,
        c.p_ord,
        std::array::from_fn(|d| {
            if c.periodic[d] {
                lobatto_fft::solver::BoundaryCondition::Periodic
            } else {
                lobatto_fft::solver::BoundaryCondition::Neumann
            }
        }),
    );
    let setup_fft = t0.elapsed().as_secs_f64();

    // ── lobatto-spectral (dense eigendecomposition) ────────────────────────
    let t0 = Instant::now();
    let spec = lobatto_spectral::solver::PoissonND::<DIM>::new(
        ls,
        xls,
        c.n_el,
        c.p_ord,
        std::array::from_fn(|d| {
            if c.periodic[d] {
                lobatto_spectral::solver::BoundaryCondition::Periodic
            } else {
                lobatto_spectral::solver::BoundaryCondition::Neumann
            }
        }),
    );
    let setup_spec = t0.elapsed().as_secs_f64();

    let n = fft.grid().total();
    assert_eq!(n, spec.grid().total());
    let p0 = random_field(n);

    // HoFFT as the filter runs it: copy to complex, apply, real part back.
    let mut c_buf = vec![Complex::new(0.0, 0.0); n];
    let mut p_fft = vec![0.0; n];
    let mut t_fft_cycle = Vec::new();
    let mut t_fft_apply = Vec::new();
    for _ in 0..N_REPS {
        let t0 = Instant::now();
        c_buf
            .par_iter_mut()
            .zip(p0.par_iter())
            .for_each(|(ci, &pi)| *ci = Complex::new(pi, 0.0));
        let t1 = Instant::now();
        fft.apply_in_place_dirs(&mut c_buf, |mu| Complex::from(symbol(mu)));
        t_fft_apply.push(t1.elapsed().as_secs_f64());
        p_fft
            .par_iter_mut()
            .zip(c_buf.par_iter())
            .for_each(|(pi, ci)| *pi = ci.re);
        t_fft_cycle.push(t0.elapsed().as_secs_f64());
    }
    drop(c_buf);

    // Eigen path: real field, in place.
    let mut p_spec = vec![0.0; n];
    let mut t_spec = Vec::new();
    for _ in 0..N_REPS {
        p_spec.copy_from_slice(&p0);
        let t0 = Instant::now();
        spec.apply_in_place_dirs(&mut p_spec, symbol);
        t_spec.push(t0.elapsed().as_secs_f64());
    }

    let max_diff = p_fft
        .par_iter()
        .zip(p_spec.par_iter())
        .map(|(a, b)| (a - b).abs())
        .reduce(|| 0.0, f64::max);
    let max_val = p_spec.par_iter().cloned().reduce(|| 0.0, f64::max);

    let ndofs = fft.grid().ndofs();
    let cyc = median(t_fft_cycle);
    let app = median(t_fft_apply);
    let sp = median(t_spec);
    println!(
        "{:<24} {:>3}×{:>3}×{:>3}×{:>3} = {:>10} DOFs\n\
         {:<24} setup {:6.3}s   apply {:7.4}s   with copies {:7.4}s\n\
         {:<24} setup {:6.3}s   apply {:7.4}s\n\
         {:<24} speed-up ×{:.2} (vs filter cycle)  ×{:.2} (bare apply)   max |Δp| = {:.2e} (max p = {:.2e})\n",
        c.name,
        ndofs[0],
        ndofs[1],
        ndofs[2],
        ndofs[3],
        n,
        "  lobatto-fft",
        setup_fft,
        app,
        cyc,
        "  lobatto-spectral",
        setup_spec,
        sp,
        "",
        cyc / sp,
        app / sp,
        max_diff,
        max_val
    );
}

fn main() {
    let heavy = std::env::args().any(|a| a == "--heavy");
    println!(
        "Diffusion step (split implicit Euler, 4D), {N_REPS} applications each, median times; \
         {} rayon threads\n",
        rayon::current_num_threads()
    );
    for c in cases(heavy) {
        run_case(&c);
    }
}
