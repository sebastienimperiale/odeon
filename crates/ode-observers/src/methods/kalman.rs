//! Mortensen tracker: the second-order (Gaussian) closure of the Mortensen
//! filter — one state x̂ and the curvature S of the value function at it,
//! instead of V on a grid.
//!
//! With V = −ε log p the filter cycle is the viscous HJB equation
//!
//!   ∂ₜV + f·∇V + ½ ∇VᵀQ∇V − (ε/2) tr(Q∇²V) = ½ d(y, h(x))²,
//!
//! (Q = diag(q) the model-noise covariance, unit observation weight, d the
//! distance in observation space — Euclidean in general, the angular
//! distance for a bearing; see [`Model::discrepancy`]). An observation
//! weight γ ([`TrackerParams::gamma`], 1 by default, matching
//! [`crate::methods::mortensen::FilterParams::gamma`]) scales the right-hand side to
//! γ·d²/2, i.e. an observation-noise covariance R = I/γ. The ansatz
//! V(x) ≈ V₀ + ½ (x − x̂)ᵀ S (x − x̂), with f and h linearized at x̂
//! (F = ∂f/∂x, H = ∇h) and d(y, h(x))² ≈ |r − H(x − x̂)|² where
//! r = [`Model::innovation`] is the signed difference y ⊖ h(x̂) in the
//! observation's own geometry, gives by matching orders in (x − x̂)
//!
//!   dx̂/dt = f(x̂) + γ S⁻¹ Hᵀ r(x̂),
//!   dS/dt  = γHᵀH − FᵀS − SF − SQS     (P = S⁻¹:  Ṗ = FP + PFᵀ + Q − γPHᵀHP),
//!
//! ε drops out: this is the ε → 0 limit, Mortensen's minimum-energy
//! estimator (Mortensen 1968, Hijab), the continuous-time extended Kalman
//! filter in information form. On a linear model with linear observations
//! the closure is exact and the tracker *is* the Kalman filter — the
//! reference the grid filter is validated against.
//!
//! The discretization mirrors [`crate::methods::mortensen::MortensenFilter::forward`]
//! step by step, with the same dt and the same order, so both estimators can
//! be compared at every step:
//!
//!   1. observation:  S ← S + dt·γ·HᵀH,   x̂ ← x̂ + dt·γ·S⁻¹Hᵀ r(x̂)   (with y_n)
//!   2. transport:    x̂ ← φ(x̂),   S ← Φ⁻ᵀ S Φ⁻¹   (Φ = ∂φ/∂x, exact)
//!   3. diffusion:    S ← (S⁻¹ + dt·Q)⁻¹ = S (I + dt·Q S)⁻¹
//!
//! The information form S (rather than P = S⁻¹) is used throughout because
//! it tolerates a flat prior (S₀ singular); only the x̂ correction needs
//! S⁻¹, and it is regularized until the observations have made S definite.
//! The tracker is unimodal by construction: it cannot represent a density
//! with several local maxima — that is precisely what the grid filter adds.

use super::Observer;
use ode_models::model::Model;
use ode_models_spec::reference::Reference;
use nalgebra::{DMatrix, DVector};

/// Parameters of the [`MortensenTracker`].
#[derive(Clone, Copy, Debug)]
pub struct TrackerParams<const M: usize> {
    /// Diagonal of the model-noise covariance Q (same meaning as
    /// [`crate::methods::mortensen::FilterParams::q_diag`]).
    pub q_diag: [f64; M],
    /// γ — observation weight (same meaning as
    /// [`crate::methods::mortensen::FilterParams::gamma`]): the observation step becomes
    /// S ← S + dt·γ·HᵀH, x̂ ← x̂ + dt·γ·S⁻¹Hᵀr. 1 is the standard cycle;
    /// 0 switches the observations off. Must be ≥ 0.
    pub gamma: f64,
}

/// Second-order Mortensen estimator: state x̂ and information matrix S.
///
/// # Example
/// ```
/// use ode_observers::methods::kalman::{MortensenTracker, TrackerParams};
/// use ode_models::models::SpringSystem;
///
/// let sys = SpringSystem::new(1, 1.0, 1.0, 0.01, |x: &[f64]| x[0]);
/// let mut tracker = MortensenTracker::<2, _>::new(sys,
///     TrackerParams { q_diag: [1.0; 2], gamma: 1.0 });
/// tracker.init([0.4, 0.1], [10.0, 10.0]); // Gaussian prior: center, stiffness σ_d
/// tracker.forward(&[0.5]); // one cycle, with the observation y₀ of this step
/// let x_hat = tracker.estimate();
/// let p = tracker.covariance().expect("S is definite");
/// ```
pub struct MortensenTracker<const M: usize, Mod: Model<M>> {
    /// The forward model (parameters and maps).
    pub model: Mod,
    /// Parameters the tracker was built with.
    pub params: TrackerParams<M>,
    x_hat: [f64; M],
    /// Information matrix S = ∇²V(x̂) (M × M, symmetric PSD).
    s: DMatrix<f64>,
}

impl<const M: usize, Mod: Model<M>> MortensenTracker<M, Mod> {
    /// Build the tracker around a model; call [`init`](Self::init) before
    /// [`forward`](Self::forward).
    pub fn new(model: Mod, params: TrackerParams<M>) -> Self {
        assert_eq!(
            model.dim(),
            M,
            "tracker dimension M = {M} does not match the model's state dimension"
        );
        assert!(
            params.q_diag.iter().all(|&q| q >= 0.0),
            "q_diag must be nonnegative, got {:?}",
            params.q_diag
        );
        assert!(
            params.gamma >= 0.0,
            "gamma must be nonnegative, got {}",
            params.gamma
        );
        MortensenTracker {
            model,
            params,
            x_hat: [0.0; M],
            s: DMatrix::zeros(M, M),
        }
    }

    /// Gaussian initial data V₀ = Σ_d σ_d (x_d − x_c,d)²/2 — the same prior
    /// as [`crate::methods::mortensen::MortensenFilter::init_filter_gaussian`]: x̂₀ = x_c,
    /// S₀ = diag(σ). σ_d = 0 leaves direction d flat (no prior).
    pub fn init(&mut self, center: [f64; M], sigma: [f64; M]) {
        assert!(sigma.iter().all(|&s| s >= 0.0), "sigma must be nonnegative");
        self.x_hat = center;
        self.s = DMatrix::from_diagonal(&DVector::from_row_slice(&sigma));
    }

    /// Current estimate x̂ (the minimizer of the quadratic V).
    pub fn estimate(&self) -> [f64; M] {
        self.x_hat
    }

    /// Information matrix S = ∇²V(x̂).
    pub fn information(&self) -> &DMatrix<f64> {
        &self.s
    }

    /// Covariance P = S⁻¹ (in the filter's units: the density p ∝ exp(−V/ε)
    /// has covariance ε·P); `None` while S is singular (flat directions not
    /// yet observed).
    pub fn covariance(&self) -> Option<DMatrix<f64>> {
        self.s.clone().try_inverse()
    }

    /// One tracker iteration, t^n → t^{n+1}, given the observation `y` =
    /// y_n of step n: observation → transport → diffusion (see the module
    /// docs).
    pub fn forward(&mut self, y: &[f64]) {
        let dt = self.model.dt();

        // Step 1 — observation at x̂ₙ with yₙ, weighted by γ.
        let gamma = self.params.gamma;
        let h_jac = self.model.obs_jacobian(&self.x_hat);
        let innov = self.model.innovation(y, &self.x_hat);
        self.s += dt * gamma * h_jac.transpose() * &h_jac;
        let rhs = dt * gamma * h_jac.transpose() * innov;
        // S may still be singular (flat prior, partially observed): solve
        // with a tiny Tikhonov shift, which vanishes once S is definite.
        let shift = 1e-12 * (1.0 + self.s.diagonal().amax());
        let s_reg = &self.s + DMatrix::<f64>::identity(M, M) * shift;
        let delta = s_reg
            .lu()
            .solve(&rhs)
            .expect("regularized information matrix is invertible");
        for d in 0..M {
            self.x_hat[d] += delta[d];
        }

        // Step 2 — transport along the discrete flow, S ← Φ⁻ᵀ S Φ⁻¹.
        let phi = self.model.flow_jacobian(self.x_hat);
        self.x_hat = self.model.flow(self.x_hat);
        let phi = DMatrix::from_fn(M, M, |r, c| phi[r][c]);
        let phi_inv = phi
            .try_inverse()
            .expect("discrete flow Jacobian is invertible (symmetric scheme)");
        self.s = phi_inv.transpose() * &self.s * &phi_inv;

        // Step 3 — diffusion: S ← S (I + dt·Q S)⁻¹, valid for singular S too.
        let q = DMatrix::from_diagonal(&DVector::from_row_slice(&self.params.q_diag));
        let m = DMatrix::<f64>::identity(M, M) + dt * &q * &self.s;
        let m_inv = m
            .try_inverse()
            .expect("I + dt·Q·S is invertible (S, Q ⪰ 0)");
        self.s = &self.s * m_inv;
        // Keep S exactly symmetric against round-off drift.
        self.s = 0.5 * (&self.s + self.s.transpose());
    }

    /// Run one iteration per step of `reference` (with its observations),
    /// returning the estimate at every step (including t = 0) and the
    /// covariance where defined.
    pub fn run_with_covariances(&mut self, reference: &Reference) -> (Vec<[f64; M]>, Vec<Option<DMatrix<f64>>>) {
        let n_steps = reference.steps();
        let mut estimates = Vec::with_capacity(n_steps + 1);
        let mut covariances = Vec::with_capacity(n_steps + 1);
        estimates.push(self.estimate());
        covariances.push(self.covariance());
        for y in &reference.observations[..n_steps] {
            self.forward(y);
            estimates.push(self.estimate());
            covariances.push(self.covariance());
        }
        (estimates, covariances)
    }
}

/// The common observer surface.
impl<const M: usize, Mod: Model<M>> Observer<M> for MortensenTracker<M, Mod> {
    fn dt(&self) -> f64 {
        self.model.dt()
    }

    fn init_gaussian(&mut self, center: [f64; M], sigma: [f64; M]) {
        self.init(center, sigma);
    }

    fn forward(&mut self, y: &[f64]) {
        MortensenTracker::forward(self, y);
    }

    fn estimate(&self) -> [f64; M] {
        MortensenTracker::estimate(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::mortensen::{DiffusionScheme, FilterParams, MortensenFilter};
    use ode_models::models::{LorenzObservation, LorenzSystem, SpringSystem};
    use ode_models_spec::noise::NoiseModel;
    use ode_models_spec::progress::Progress;

    fn spring(dt: f64) -> SpringSystem {
        SpringSystem::new(1, 1.0, 1.0, dt, |x: &[f64]| x[0])
    }

    /// The spring's reference from (1.15, 0) with the given observation
    /// noise.
    fn spring_reference(dt: f64, steps: usize, noise: NoiseModel, seed: u64) -> Reference {
        Reference::twin(&spring(dt), [1.15, 0.0], steps, noise, seed, None, &Progress::default())
    }

    /// On the linear spring the tracker must reproduce the discrete Kalman
    /// filter written independently in *covariance* form (the tracker works
    /// in information form): observation update with R = I/dt via the gain
    /// K = P⁻Hᵀ(HP⁻Hᵀ + I/dt)⁻¹, prediction P ← ΦPΦᵀ + dt·Q. Equality is
    /// the Woodbury identity, so this checks the implementation, not a
    /// tautology.
    #[test]
    fn matches_kalman_filter_on_linear_model() {
        let dt = 0.01;
        let q = [0.3, 0.7];
        let sigma = [4.0, 9.0];
        let x_c = [1.0, 0.2];
        let r = spring_reference(dt, 200, NoiseModel::Gaussian { std: 0.05 }, 7);
        let mut tracker =
            MortensenTracker::<2, _>::new(spring(dt), TrackerParams { q_diag: q, gamma: 1.0 });
        tracker.init(x_c, sigma);

        // Independent Kalman filter on the same observation sequence.
        let mut x = DVector::from_row_slice(&x_c);
        let mut p = DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0 / sigma[0], 1.0 / sigma[1]]));
        let phi = spring(dt).trans.clone();
        let qm = DMatrix::from_diagonal(&DVector::from_row_slice(&q));
        let h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);

        for n in 0..200 {
            tracker.forward(&r.observations[n]);

            let y = DVector::from_row_slice(&r.observations[n]);
            let innov = &y - &h * &x;
            let s_inn = &h * &p * h.transpose() + DMatrix::identity(1, 1) / dt;
            let k = &p * h.transpose() * s_inn.try_inverse().unwrap();
            x = &x + &k * innov;
            p = (DMatrix::<f64>::identity(2, 2) - &k * &h) * &p;
            x = &phi * &x;
            p = &phi * &p * phi.transpose() + dt * &qm;

            let xt = tracker.estimate();
            let pt = tracker.covariance().expect("definite");
            for d in 0..2 {
                assert!((xt[d] - x[d]).abs() < 1e-9, "x̂ differs: {xt:?} vs {x}");
            }
            assert!((&pt - &p).amax() < 1e-9, "P differs:\n{pt}\nvs\n{p}");
        }
    }

    /// Validation of the grid filter against its Gaussian closure: on the
    /// linear spring (where the closure is exact) the filter's argmax must
    /// follow the tracker's x̂ to grid accuracy.
    #[test]
    fn grid_filter_argmax_follows_the_tracker_on_linear_model() {
        let dt = 0.01;
        let q = [1.0; 2];
        let sigma = [20.0; 2];
        let x_c = [1.0, 0.0];
        let params = |pre| FilterParams {
            domain: [(-3.0, 3.0); 2],
            eps: 0.05,
            gamma: 1.0,
            n_el: [24; 2],
            p_ord: [4; 2],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: q,
            periodic: [false; 2],
            dirichlet: [false; 2],
            pre_compute_flow_inv: pre,
        };
        let mut filter = MortensenFilter::<2, _>::new(spring(dt), params(true));
        filter.init_filter_gaussian(sigma, x_c);
        let mut tracker =
            MortensenTracker::<2, _>::new(spring(dt), TrackerParams { q_diag: q, gamma: 1.0 });
        tracker.init(x_c, sigma);

        // Node spacing near the center ≈ 6/(24·4)·(Lobatto clustering) ≲ 0.08.
        let tol = 0.12;
        let r = spring_reference(dt, 100, NoiseModel::None, 0);
        for y in &r.observations[..100] {
            filter.forward(y);
            tracker.forward(y);
            let (a, b) = (filter.argmax_p(), tracker.estimate());
            for d in 0..2 {
                assert!(
                    (a[d] - b[d]).abs() < tol,
                    "grid argmax {a:?} vs tracker {b:?}"
                );
            }
        }
    }

    /// γ = 0 switches the observations off: the estimate is then the pure
    /// discrete flow of its initial condition (the diffusion step moves only
    /// S), regardless of the observation sequence.
    #[test]
    fn gamma_zero_reduces_to_the_flow() {
        let dt = 0.01;
        let x0 = [0.9, -0.3];
        let r = spring_reference(dt, 50, NoiseModel::Gaussian { std: 0.05 }, 3);
        let mut tracker =
            MortensenTracker::<2, _>::new(spring(dt), TrackerParams { q_diag: [1.0; 2], gamma: 0.0 });
        tracker.init(x0, [5.0; 2]);
        let mut flow = x0;
        for y in &r.observations[..50] {
            tracker.forward(y);
            flow = tracker.model.flow(flow);
            let e = tracker.estimate();
            for d in 0..2 {
                assert!(
                    (e[d] - flow[d]).abs() < 1e-12,
                    "γ = 0 estimate left the flow: {e:?} vs {flow:?}"
                );
            }
        }
    }

    /// Nonlinear sanity: with noise-free x observations on Lorenz the
    /// tracker started off the reference locks onto it (bounded, small
    /// error after a transient), and S stays symmetric positive definite.
    #[test]
    fn tracks_lorenz_from_x_observations() {
        let dt = 0.01;
        let x0 = [-3.716171, -4.204785, 20.339103];
        let sys = LorenzSystem::new(dt, LorenzObservation::X);
        let r = Reference::twin(&sys, x0, 1000, NoiseModel::None, 0, None, &Progress::default());
        let mut tracker =
            MortensenTracker::<3, _>::new(sys, TrackerParams { q_diag: [1.0; 3], gamma: 1.0 });
        tracker.init([x0[0] + 2.0, x0[1] - 2.0, x0[2] + 3.0], [1.0; 3]);
        let mut max_err_late = 0.0_f64;
        for n in 0..1000 {
            tracker.forward(&r.observations[n]);
            let s = &r.states[n + 1];
            let e = tracker.estimate();
            let err = (0..3).map(|d| (e[d] - s[d]).abs()).fold(0.0, f64::max);
            assert!(err.is_finite() && err < 50.0, "diverged at step {n}: {err}");
            if n >= 500 {
                max_err_late = max_err_late.max(err);
            }
            let p = tracker.covariance().expect("definite");
            assert!(p.diagonal().iter().all(|&v| v > 0.0), "P not positive: {p}");
        }
        assert!(max_err_late < 1.0, "did not lock onto the reference: max late error {max_err_late}");
    }
}
