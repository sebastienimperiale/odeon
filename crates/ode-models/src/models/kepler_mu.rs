//! Planar Kepler problem with an *unknown gravitational parameter* — the
//! joint state–parameter estimation variant of [`super::kepler`].
//!
//! The state is augmented to x = (q₁, q₂, p₁, p₂, θ) with the gravitational
//! parameter written
//!
//!   μ(θ) = μ₀ · 2^θ,      dθ/dt = 0,
//!
//! where μ₀ is the prior value ([`KeplerMuParams::mu0`]) and θ the
//! dimensionless parameter to estimate, in *doublings* of μ₀ (θ = 0 ⇔
//! μ = μ₀, θ = ±1 ⇔ μ = 2μ₀ / μ₀/2). The exponential parameterization
//! guarantees μ > 0 for every real θ, so the filter's state-space box needs
//! no positivity constraint in the θ direction, and a Gaussian prior in θ is
//! a log-normal prior in μ — the natural prior for a scale parameter.
//!
//! The dynamics are those of [`super::kepler`] (Plummer-softened potential,
//! softening a) at the pointwise parameter μ(θ):
//!
//!   q̇ = p,   ṗ = −μ(θ) q / (|q|² + a²)^{3/2},   θ̇ = 0.
//!
//! In the filter, dθ/dt = 0 means: transport leaves the θ coordinate of
//! every grid point unchanged (each θ-slice runs its own Kepler flow), and
//! `q_diag[4] = 0` switches diffusion off in θ — the observation step alone
//! discriminates the slices. μ is identifiable from the position-type
//! observations (it sets the orbital period, Kepler's third law); the
//! *satellite's* mass would not be, since it cancels from q̈.
//!
//! Conserved quantities along the flow (θ constant on trajectories): θ
//! itself — exactly, its stage equation k_θ = 0 decouples — the angular
//! momentum L = q₁p₂ − q₂p₁ (quadratic invariant, exact under
//! Gauss–Legendre), and the energy H = |p|²/2 − μ(θ)/√(|q|² + a²) up to the
//! bounded O(dt⁴) oscillation of a symplectic scheme.
//!
//! Observations are the same three scalars as the base model
//! ([`KeplerObservation`]: q₁, range, bearing) — they depend on q only, and
//! their gradients get a zero θ-component.
//!
//! Time stepping: the shared symplectic Gauss–Legendre 4 scheme
//! ([`crate::gl4`]) at M = 5, so flow_inv = one step with −dt is exact and
//! the tracker's Φ = ∂φ/∂x (including the ∂φ/∂θ sensitivity column) comes
//! from the tangent of the stage system.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use super::kepler::KeplerObservation;
use crate::model::Model;
use nalgebra::{DMatrix, DVector};
use std::f64::consts::LN_2;

/// State dimension: (q₁, q₂, p₁, p₂, θ).
pub const DIM: usize = 5;

/// Physical parameters. [`Default`]: μ₀ = 1, a = 0.1.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KeplerMuParams {
    /// Prior gravitational parameter μ₀; the true value is μ₀·2^θ.
    pub mu0: f64,
    /// Plummer softening length a (see [`super::kepler`]).
    pub softening: f64,
}

impl Default for KeplerMuParams {
    fn default() -> Self {
        KeplerMuParams {
            mu0: 1.0,
            softening: 0.1,
        }
    }
}

impl KeplerMuParams {
    /// μ(θ) = μ₀ · 2^θ (> 0 for every real θ).
    pub fn mu(&self, theta: f64) -> f64 {
        self.mu0 * theta.exp2()
    }
}

/// The augmented state (q₁, q₂, p₁, p₂, θ) from a base Kepler state and θ
/// (e.g. `augmented_state(perihelion_state(e), theta)`).
pub fn augmented_state(x: [f64; 4], theta: f64) -> [f64; DIM] {
    [x[0], x[1], x[2], x[3], theta]
}

/// Softened planar Kepler forward problem with the augmented state
/// (q₁, q₂, p₁, p₂, θ), μ = μ₀·2^θ.
///
/// # Example
/// ```
/// use ode_models::models::kepler::{perihelion_state, KeplerObservation};
/// use ode_models::models::kepler_mu::{augmented_state, KeplerMuSystem};
///
/// // Reference orbit at the true parameter μ = 2^0.4·μ₀:
/// let x0 = augmented_state(perihelion_state(0.5), 0.4);
/// let sys = KeplerMuSystem::new(0.01, KeplerObservation::Range);
/// let x1 = sys.flow(&x0); // one step: t^0 → t^1 (θ stays 0.4 exactly)
/// ```
pub struct KeplerMuSystem {
    // ── Public ───────────────────────────────────────────────────────────────
    pub dt: f64,
    /// Physical parameters (μ₀, softening).
    pub params: KeplerMuParams,
    /// Which scalar observation h(x) the system reports.
    pub observation: KeplerObservation,

}

impl KeplerMuSystem {
    /// Build the system with the default physical parameters
    /// ([`KeplerMuParams::default`]).
    ///
    /// The state is (q₁, q₂, p₁, p₂, θ); the θ component of a reference
    /// trajectory is the *true* log-parameter.
    ///
    /// * `dt`          — time step
    /// * `observation` — the scalar observation h(x)
    pub fn new(dt: f64, observation: KeplerObservation) -> Self {
        Self::with_params(KeplerMuParams::default(), dt, observation)
    }

    /// Build the system from explicit physical parameters (see
    /// [`new`](Self::new) for the other arguments).
    pub fn with_params(params: KeplerMuParams, dt: f64, observation: KeplerObservation) -> Self {
        KeplerMuSystem { dt, params, observation }
    }

    /// Squared discrepancy |y − h(xi)|² (angular difference for the
    /// bearing), as in [`super::kepler`].
    pub fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        let d = self.observation.difference(y[0], self.observation.h(xi));
        d * d
    }

    /// Hamiltonian H(x) = |p|²/2 − μ(θ)/√(|q|² + a²), conserved along the
    /// flow (θ is constant on trajectories).
    pub fn hamiltonian(&self, x: &[f64; DIM]) -> f64 {
        let a = self.params.softening;
        let [q1, q2, p1, p2, theta] = *x;
        0.5 * (p1 * p1 + p2 * p2) - self.params.mu(theta) / (q1 * q1 + q2 * q2 + a * a).sqrt()
    }

    /// Angular momentum L = q₁p₂ − q₂p₁ (the force stays central for every θ).
    pub fn angular_momentum(&self, x: &[f64; DIM]) -> f64 {
        x[0] * x[3] - x[1] * x[2]
    }

    /// Right-hand side:  q̇ = p, ṗ = −μ(θ) q / s^{3/2}, θ̇ = 0, with
    /// s = |q|² + a².
    fn rhs(&self, x: &[f64; DIM]) -> [f64; DIM] {
        let a = self.params.softening;
        let [q1, q2, p1, p2, theta] = *x;
        let s = q1 * q1 + q2 * q2 + a * a;
        let f = -self.params.mu(theta) / (s * s.sqrt());
        [p1, p2, f * q1, f * q2, 0.0]
    }

    /// Analytic Jacobian ∂rhs/∂x: the 4×4 block of [`super::kepler`] at
    /// μ = μ(θ), plus the θ-column ∂ṗᵢ/∂θ = ln2 · f · qᵢ (since
    /// dμ/dθ = ln2 · μ(θ)) and a zero θ-row.
    fn rhs_jacobian(&self, x: &[f64; DIM]) -> [[f64; DIM]; DIM] {
        let a = self.params.softening;
        let [q1, q2, _, _, theta] = *x;
        let s = q1 * q1 + q2 * q2 + a * a;
        let f = -self.params.mu(theta) / (s * s.sqrt());
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [
                f * (1.0 - 3.0 * q1 * q1 / s),
                f * (-3.0 * q1 * q2 / s),
                0.0,
                0.0,
                LN_2 * f * q1,
            ],
            [
                f * (-3.0 * q1 * q2 / s),
                f * (1.0 - 3.0 * q2 * q2 / s),
                0.0,
                0.0,
                LN_2 * f * q2,
            ],
            [0.0; DIM],
        ]
    }

    /// One step of the fourth-order Gauss–Legendre method (shared solver,
    /// see [`crate::gl4`]).
    fn gl4_step(&self, x: &[f64; DIM], h: f64) -> [f64; DIM] {
        crate::gl4::gl4_step(|x| self.rhs(x), |x| self.rhs_jacobian(x), x, h)
    }

    /// Discrete flow map φ: one Gauss–Legendre step of size `dt`.
    pub fn flow(&self, x: &[f64; DIM]) -> [f64; DIM] {
        self.gl4_step(x, self.dt)
    }

    /// φ(x) together with its exact Jacobian ∂φ/∂x, including the
    /// ∂φ/∂θ sensitivity column the tracker needs (see
    /// [`crate::gl4::gl4_step_with_jacobian`]).
    pub fn flow_with_jacobian(&self, x: &[f64; DIM]) -> ([f64; DIM], [[f64; DIM]; DIM]) {
        crate::gl4::gl4_step_with_jacobian(|x| self.rhs(x), |x| self.rhs_jacobian(x), x, self.dt)
    }

    /// Inverse discrete flow map φ⁻¹: one Gauss–Legendre step of size `−dt`.
    /// Exact inverse of [`flow`](Self::flow) because the scheme is symmetric.
    pub fn flow_inv(&self, x: &[f64; DIM]) -> [f64; DIM] {
        self.gl4_step(x, -self.dt)
    }
}

/// [`Model`] interface for the Mortensen filter, at the fixed dimension
/// [`DIM`] = 5.
impl Model<DIM> for KeplerMuSystem {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        KeplerMuSystem::discrepancy(self, y, xi)
    }

    fn flow_inv(&self, xi: [f64; DIM]) -> [f64; DIM] {
        KeplerMuSystem::flow_inv(self, &xi)
    }

    fn flow(&self, xi: [f64; DIM]) -> [f64; DIM] {
        KeplerMuSystem::flow(self, &xi)
    }

    fn flow_jacobian(&self, xi: [f64; DIM]) -> [[f64; DIM]; DIM] {
        self.flow_with_jacobian(&xi).1
    }

    fn obs_dim(&self) -> usize {
        1
    }

    fn obs(&self, xi: &[f64]) -> DVector<f64> {
        DVector::from_element(1, self.observation.h(xi))
    }

    /// ∇h of the base observations with a zero θ-component (h depends on q
    /// only).
    fn obs_jacobian(&self, xi: &[f64]) -> DMatrix<f64> {
        let g4 = self.observation.gradient(xi);
        let mut g = [0.0; DIM];
        g[..4].copy_from_slice(&g4);
        DMatrix::from_row_slice(1, DIM, &g)
    }

    /// The bearing innovation is the shortest angular difference.
    fn innovation(&self, y: &[f64], xi: &[f64]) -> DVector<f64> {
        DVector::from_element(1, self.observation.difference(y[0], self.observation.h(xi)))
    }

    /// Time-independent flow: autonomous (so the transport plan applies).
    fn is_autonomous(&self) -> bool {
        true
    }

    fn state_labels(&self) -> Vec<String> {
        ["q_1", "q_2", "p_1", "p_2", "θ"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::kepler::perihelion_state;

    /// Deliberately non-unit parameters and θ ≠ 0, so the θ-dependence
    /// enters every check.
    const ODD_PARAMS: KeplerMuParams = KeplerMuParams {
        mu0: 1.7,
        softening: 0.3,
    };
    const ODD_THETA: f64 = -0.6;

    /// The trajectory from `x0` (x0 included) and the system.
    fn run(params: KeplerMuParams, x0: [f64; DIM], dt: f64, steps: usize) -> (KeplerMuSystem, Vec<[f64; DIM]>) {
        let sys = KeplerMuSystem::with_params(params, dt, KeplerObservation::Q1);
        let mut states = vec![x0];
        for _ in 0..steps {
            states.push(sys.flow(states.last().unwrap()));
        }
        (sys, states)
    }

    /// Energy conservation at μ(θ) ≠ μ₀, and θ exactly constant along the
    /// trajectory (its stage equation decouples: k_θ = 0).
    #[test]
    fn energy_is_conserved_and_theta_is_constant() {
        let x0 = augmented_state(perihelion_state(0.5), ODD_THETA);
        let (sys, states) = run(ODD_PARAMS, x0, 0.005, 1000);
        let h0 = sys.hamiltonian(&x0);
        let l0 = sys.angular_momentum(&x0);
        for x in &states {
            assert!(
                (sys.hamiltonian(&x) - h0).abs() < 1e-7 * h0.abs(),
                "energy drift: H = {} vs H0 = {h0}",
                sys.hamiltonian(&x)
            );
            assert!(
                (sys.angular_momentum(&x) - l0).abs() < 1e-10,
                "angular momentum drift"
            );
            assert_eq!(x[4], ODD_THETA, "θ moved along the flow");
        }
    }

    /// The 5×5 analytic Jacobian — in particular the θ-column with its ln 2
    /// factor — against central finite differences of the rhs.
    #[test]
    fn jacobian_matches_finite_differences() {
        let sys = KeplerMuSystem::with_params(ODD_PARAMS, 0.01, KeplerObservation::Q1);
        let x = [0.6, -0.35, 0.8, -1.9, 0.45];
        let jac = sys.rhs_jacobian(&x);
        let eps = 1e-6;
        for col in 0..DIM {
            let mut xp = x;
            let mut xm = x;
            xp[col] += eps;
            xm[col] -= eps;
            let fp = sys.rhs(&xp);
            let fm = sys.rhs(&xm);
            for row in 0..DIM {
                let fd = (fp[row] - fm[row]) / (2.0 * eps);
                assert!(
                    (jac[row][col] - fd).abs() < 1e-7 * (1.0 + fd.abs()),
                    "J[{row}][{col}] = {} but finite difference gives {fd}",
                    jac[row][col]
                );
            }
        }
    }

    #[test]
    fn flow_inv_inverts_flow() {
        let x0 = augmented_state(perihelion_state(0.5), 0.3);
        let sys = KeplerMuSystem::new(0.01, KeplerObservation::Range);
        let y = sys.flow(&x0);
        let z = sys.flow_inv(&y);
        for i in 0..DIM {
            assert!(
                (z[i] - x0[i]).abs() < 1e-11,
                "round-trip error at {i}: {}",
                z[i] - x0[i]
            );
        }
    }

    /// The flow must be evaluable everywhere in a 5D filter box: at the
    /// origin (softening) and at extreme θ, where μ is far from μ₀.
    #[test]
    fn flow_is_defined_at_the_origin_and_extreme_theta() {
        let sys = KeplerMuSystem::new(0.01, KeplerObservation::Q1);
        for x in [
            [0.0; DIM],
            [0.0, 0.0, 3.0, -3.0, 2.0],
            [0.01, -0.01, -5.0, 5.0, -2.0],
        ] {
            let y = sys.flow_inv(&x);
            assert!(y.iter().all(|v| v.is_finite()), "flow_inv({x:?}) = {y:?}");
        }
    }

}
