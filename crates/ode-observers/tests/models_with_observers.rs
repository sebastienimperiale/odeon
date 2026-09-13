//! The model tests that exercise an observer (moved out of `ode-models`,
//! which must not depend on `ode-observers`): the tracker's parameter
//! identification on the unknown-μ Kepler problem and the unknown-mass
//! spring chain, the grid filter's smoke runs in 3D and 5D, and the
//! random walk's ring (found by the grid filter, collapsed by the tracker).

#![allow(clippy::excessive_precision)]

use ode_models::models::kepler::perihelion_state;
use ode_models::models::kepler_mu::{augmented_state, KeplerMuSystem, DIM};
use ode_models::models::random_walk::{RandomWalkSystem, DEFAULT_STATE};
use ode_models::models::spring_mass::SpringMassSystem;
use ode_models::models::KeplerObservation;
use ode_observers::filter::{DiffusionScheme, FilterParams, MortensenFilter};
use ode_observers::tracker::{MortensenTracker, TrackerParams};

/// Identifiability of the parameter: observing q₁ alone, the tracker
/// started at the *correct* (q, p) but the *wrong* θ converges to the
/// true θ (μ sets the orbital period, so position observations see it).
/// Q = 0: pure minimum-energy estimation of a constant parameter.
///
/// The regime is deliberately *local*: a stiff prior on the
/// correctly-known (q, p) — so the innovations must be explained by θ,
/// not absorbed into the phase — and a small true θ. For large θ errors
/// the q₁ likelihood is phase-aliased over several periods (multimodal
/// in θ), the EKF linearization breaks, and the estimate falls into a
/// secondary basin (observed at θ_true = 0.4: a transient through the
/// truth, then collapse to ≈ 0.13). That global, multimodal regime is
/// exactly what the grid filter — not the tracker — is for.
#[test]
fn tracker_estimates_theta_from_q1_observations() {
    let theta_true = 0.1;
    let dt = 0.01;
    let x0 = augmented_state(perihelion_state(0.3), theta_true);
    let sys = KeplerMuSystem::new(x0, dt, KeplerObservation::Q1);
    let mut tracker =
        MortensenTracker::<DIM, _>::new(sys, TrackerParams { q_diag: [0.0; DIM], gamma: 1.0 });
    // Prior centered at the true (q, p) but θ̂₀ = 0 (i.e. μ̂ = μ₀).
    tracker.init(
        [x0[0], x0[1], x0[2], x0[3], 0.0],
        [1000.0, 1000.0, 1000.0, 1000.0, 1.0],
    );
    for _ in 0..2000 {
        tracker.forward(); // t = 20 ≈ 3 orbital periods
    }
    let theta_hat = tracker.estimate()[4];
    assert!(
        (theta_hat - theta_true).abs() < 0.03,
        "tracker did not identify θ: θ̂ = {theta_hat} vs {theta_true}"
    );
    // Information about θ accumulated: P_θθ down from the prior 1.
    let p_theta = tracker.covariance().expect("S is definite")[(4, 4)];
    assert!(
        p_theta < 0.05,
        "no information gained about θ: P_θθ = {p_theta}"
    );
}

/// 5D smoke test of the grid filter on the augmented model: a small run
/// with q_θ = 0 (dθ/dt = 0, no diffusion in θ) keeps p finite and
/// positive, with the transport plan active.
#[test]
fn grid_filter_runs_in_5d() {
    use ode_observers::filter::{DiffusionScheme, FilterParams, MortensenFilter};
    let theta_true = 0.3;
    let x0 = augmented_state(perihelion_state(0.3), theta_true);
    let sys = KeplerMuSystem::new(x0, 0.02, KeplerObservation::Range);
    let mut filter = MortensenFilter::<DIM, _>::new(
        sys,
        FilterParams {
            domain: [(-2.0, 2.0), (-2.0, 2.0), (-2.5, 2.5), (-2.5, 2.5), (-1.0, 1.0)],
            eps: 0.1,
            gamma: 1.0,
            n_el: [3, 3, 3, 3, 2],
            p_ord: [2; DIM],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: [1.0, 1.0, 1.0, 1.0, 0.0],
            periodic: [false; DIM],
            dirichlet: [false; DIM],
            pre_compute_flow_inv: true,
        },
    );
    filter.init_filter_gaussian([5.0, 5.0, 5.0, 5.0, 1.0], [x0[0], x0[1], x0[2], x0[3], 0.0]);
    for _ in 0..5 {
        filter.forward();
    }
    let max = filter.p_field().iter().copied().fold(f64::MIN, f64::max);
    assert!(max.is_finite() && max > 0.0, "p degenerated: max = {max}");
    let xhat = filter.argmax_p();
    assert!(xhat.iter().all(|v| v.is_finite()), "argmax not finite: {xhat:?}");
}

/// The grid filter finds the ring: from a flat prior, after a few
/// observation/diffusion cycles, p is large on the whole unit circle
/// (checked at the four axis points, which are grid nodes) and small
/// inside and outside it; the argmax lies on the ring.
#[test]
fn grid_filter_finds_the_ring() {
    let dt = 0.05;
    let mut filter = MortensenFilter::<2, _>::new(
        RandomWalkSystem::new(DEFAULT_STATE, dt),
        FilterParams {
            domain: [(-2.0, 2.0); 2],
            eps: 0.05,
            gamma: 1.0,
            n_el: [16; 2],
            p_ord: [4; 2],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: [1.0; 2],
            periodic: [false; 2],
            dirichlet: [false; 2],
            pre_compute_flow_inv: true,
        },
    );
    filter.init_filter_gaussian([0.0; 2], [0.0; 2]); // flat prior
    for _ in 0..40 {
        filter.forward();
    }
    let p = filter.p_field();
    let grid = filter.grid();
    let max = p.iter().copied().fold(f64::MIN, f64::max);
    let at = |pt: [f64; 2]| {
        let (i, _) = grid
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = (a[0] - pt[0]).powi(2) + (a[1] - pt[1]).powi(2);
                let db = (b[0] - pt[0]).powi(2) + (b[1] - pt[1]).powi(2);
                da.total_cmp(&db)
            })
            .unwrap();
        p[i] / max
    };
    for pt in [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]] {
        assert!(at(pt) > 0.5, "p is not high on the ring at {pt:?}: {}", at(pt));
    }
    assert!(at([0.0, 0.0]) < 0.05, "p is not low inside the ring: {}", at([0.0, 0.0]));
    assert!(at([1.5, 0.0]) < 0.05, "p is not low outside the ring: {}", at([1.5, 0.0]));
    let r = filter.argmax_p().iter().map(|c| c * c).sum::<f64>().sqrt();
    assert!((r - 1.0).abs() < 0.15, "argmax not on the ring: radius {r}");
}

/// The tracker cannot represent the ring: it converges to *one* point
/// at distance 1 — the one selected by its prior — with a definite S.
#[test]
fn tracker_collapses_onto_one_point_of_the_ring() {
    let mut tracker = MortensenTracker::<2, _>::new(
        RandomWalkSystem::new(DEFAULT_STATE, 0.05),
        TrackerParams { q_diag: [0.0; 2], gamma: 1.0 },
    );
    tracker.init([0.5, 0.3], [1.0; 2]);
    for _ in 0..600 {
        tracker.forward();
    }
    let e = tracker.estimate();
    let r = (e[0] * e[0] + e[1] * e[1]).sqrt();
    assert!((r - 1.0).abs() < 0.02, "tracker not on the ring: {e:?}");
    // It went to the ring along its prior's direction, not to (0, 1).
    assert!((e[0] - DEFAULT_STATE[0]).abs() > 0.3, "tracker landed on the reference by luck: {e:?}");
}

/// Identifiability of the mass from one position: the tracker started
/// at the correct (y, v) and the wrong θ converges to the true θ — in
/// the local regime (stiff prior on the known (y, v), small θ_true),
/// as for Kepler-μ. N = 2 observing y₁, and N = 1.
#[test]
fn tracker_estimates_theta_from_a_position() {
    let theta_true = 0.15;
    let dt = 0.01;

    let x0 = SpringMassSystem::<2>::state([0.0, 0.5], [0.0, 0.0], theta_true);
    let sys = SpringMassSystem::<2>::new(&x0, dt, 0);
    let mut tracker = MortensenTracker::<5, _>::new(sys, TrackerParams { q_diag: [0.0; 5], gamma: 1.0 });
    tracker.init([x0[0], x0[1], x0[2], x0[3], 0.0], [1000.0, 1000.0, 1000.0, 1000.0, 1.0]);
    for _ in 0..3000 {
        tracker.forward(); // t = 30, several periods
    }
    let theta_hat = tracker.estimate()[4];
    assert!((theta_hat - theta_true).abs() < 0.03, "N=2: θ̂ = {theta_hat} vs {theta_true}");
    // Information about θ accumulated (measured P_θθ ≈ 0.08).
    let p_theta = tracker.covariance().expect("S is definite")[(4, 4)];
    assert!(p_theta < 0.2, "N=2: no information gained about θ: P_θθ = {p_theta}");

    let x0 = SpringMassSystem::<1>::state([0.5], [0.0], theta_true);
    let sys = SpringMassSystem::<1>::new(&x0, dt, 0);
    let mut tracker = MortensenTracker::<3, _>::new(sys, TrackerParams { q_diag: [0.0; 3], gamma: 1.0 });
    tracker.init([x0[0], x0[1], 0.0], [1000.0, 1000.0, 1.0]);
    for _ in 0..3000 {
        tracker.forward();
    }
    let theta_hat = tracker.estimate()[2];
    assert!((theta_hat - theta_true).abs() < 0.03, "N=1: θ̂ = {theta_hat} vs {theta_true}");
}

/// Grid-filter smoke tests on the augmented model: a small 3D run
/// (N = 1) and a small 5D run (N = 2) with q_θ = 0 keep p finite and
/// positive, with the transport plan.
#[test]
fn grid_filter_runs_in_3d_and_5d() {
    use ode_observers::filter::{DiffusionScheme, FilterParams, MortensenFilter};

    let x0 = SpringMassSystem::<1>::state([0.5], [0.0], 0.3);
    let mut filter = MortensenFilter::<3, _>::new(
        SpringMassSystem::<1>::new(&x0, 0.02, 0),
        FilterParams {
            domain: [(-1.0, 1.0), (-1.5, 1.5), (-1.0, 1.0)],
            eps: 0.1,
            gamma: 1.0,
            n_el: [6, 6, 4],
            p_ord: [3; 3],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: [1.0, 1.0, 0.0],
            periodic: [false; 3],
            dirichlet: [false; 3],
            pre_compute_flow_inv: true,
        },
    );
    filter.init_filter_gaussian([5.0, 5.0, 1.0], [x0[0], x0[1], 0.0]);
    for _ in 0..10 {
        filter.forward();
    }
    let max = filter.p_field().iter().copied().fold(f64::MIN, f64::max);
    assert!(max.is_finite() && max > 0.0, "N=1: p degenerated: max = {max}");

    let x0 = SpringMassSystem::<2>::state([0.0, 0.5], [0.0, 0.0], 0.3);
    let mut filter = MortensenFilter::<5, _>::new(
        SpringMassSystem::<2>::new(&x0, 0.02, 0),
        FilterParams {
            domain: [(-1.0, 1.0), (-1.0, 1.0), (-1.5, 1.5), (-1.5, 1.5), (-1.0, 1.0)],
            eps: 0.1,
            gamma: 1.0,
            n_el: [3, 3, 3, 3, 2],
            p_ord: [2; 5],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: [1.0, 1.0, 1.0, 1.0, 0.0],
            periodic: [false; 5],
            dirichlet: [false; 5],
            pre_compute_flow_inv: true,
        },
    );
    filter.init_filter_gaussian([5.0, 5.0, 5.0, 5.0, 1.0], [x0[0], x0[1], x0[2], x0[3], 0.0]);
    for _ in 0..5 {
        filter.forward();
    }
    let max = filter.p_field().iter().copied().fold(f64::MIN, f64::max);
    assert!(max.is_finite() && max > 0.0, "N=2: p degenerated: max = {max}");
    assert!(filter.argmax_p().iter().all(|v| v.is_finite()));
}
