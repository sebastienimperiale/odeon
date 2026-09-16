//! Planar double *compound* pendulum — forward problem solver.
//!
//! Two identical uniform rigid rods of mass m and length ℓ, pinned end to
//! end, gravity g in the −y direction — the model of the swaptube
//! simulations (github.com/2swap/swaptube, `PendulumHelpers.h`). The state
//! is x = (q₁, q₂, p₁, p₂): angles of the rods with the downward vertical
//! and their conjugate momenta. The rod moment of inertia I = mℓ²/3 gives,
//! with Δ = q₁ − q₂, the Hamiltonian
//!
//!   H(x) = 6·(p₁² + 4p₂² − 3p₁p₂cos Δ) / (mℓ²(16 − 9cos²Δ))
//!        + ½mgℓ(3(1 − cos q₁) + (1 − cos q₂)),
//!
//! whose ∂H/∂p reproduce the swaptube equations
//!
//!   q̇₁ = (6/mℓ²)(2p₁ − 3p₂cos Δ)/(16 − 9cos²Δ),
//!   q̇₂ = (6/mℓ²)(8p₂ − 3p₁cos Δ)/(16 − 9cos²Δ),
//!   ṗ₁ = −½mℓ²(q̇₁q̇₂ sin Δ + 3(g/ℓ) sin q₁),
//!   ṗ₂ = −½mℓ²(−q̇₁q̇₂ sin Δ + (g/ℓ) sin q₂).
//!
//! m, ℓ and g are runtime parameters (see [`PendulumRodParams`]; the defaults
//! are the swaptube values).
//!
//! (swaptube integrates with explicit RK4; here the same symplectic
//! Gauss–Legendre 4 scheme as [`crate::models::pendulum`] is used, so
//! flow_inv = one step with −dt is exact.)

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use crate::model::Model;
use nalgebra::{DMatrix, DVector};

/// State dimension: (q₁, q₂, p₁, p₂).
pub const DIM: usize = 4;

/// Physical parameters of the compound pendulum (both rods identical).
/// [`Default`] gives the swaptube values: m = 1, ℓ = 1, g = 9.8.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PendulumRodParams {
    /// Mass of each rod.
    pub m: f64,
    /// Length of each rod.
    pub l: f64,
    /// Gravity g (−y direction).
    pub g: f64,
}

impl Default for PendulumRodParams {
    fn default() -> Self {
        PendulumRodParams {
            m: 1.0,
            l: 1.0,
            g: 9.8,
        }
    }
}

/// Solver for the planar double compound (rigid-rod) pendulum forward
/// problem.
///
/// The observation is hardcoded: h(x) = cos q₁, the vertical component of
/// the first rod's direction. Everything else is unobserved — note cos is
/// even in q₁, so the observation cannot distinguish ±q₁ (expect bimodal
/// densities until the dynamics break the symmetry).
///
/// # Example
/// ```
/// use ode_models::models::pendulum_rod::{DIM, PendulumRodSystem};
///
/// let sys = PendulumRodSystem::new(0.01);
/// let x1 = sys.flow(&[2.453, -2.7727, 0.0, 0.0]); // one step: t^0 → t^1
/// ```
pub struct PendulumRodSystem {
    // ── Public ───────────────────────────────────────────────────────────────
    pub dt: f64,
    /// Physical parameters (rod mass, rod length, gravity).
    pub params: PendulumRodParams,

}

impl PendulumRodSystem {
    /// Build the system with the swaptube physical parameters
    /// ([`PendulumRodParams::default`]).
    ///
    /// The state is (q₁, q₂, p₁, p₂).
    ///
    /// * `dt`    — time step
    pub fn new(dt: f64) -> Self {
        Self::with_params(PendulumRodParams::default(), dt)
    }

    /// Build the system from explicit physical parameters (see [`new`](Self::new)
    /// for the other arguments).
    pub fn with_params(params: PendulumRodParams, dt: f64) -> Self {
        PendulumRodSystem { dt, params }
    }

    /// (x, y) position of the free end of the second rod — depends only on
    /// the two angles and the rod length. Same convention as
    /// scripts/visualize_pendulum.py: the pivot sits at (0, 2ℓ) and the
    /// angles are measured from the downward vertical, so q₁ = q₂ = 0 puts
    /// the tip at the origin.
    pub fn tip_position(&self, q1: f64, q2: f64) -> [f64; 2] {
        let l = self.params.l;
        [
            l * (q1.sin() + q2.sin()),
            2.0 * l - l * (q1.cos() + q2.cos()),
        ]
    }

    /// Squared discrepancy |y − cos q₁|² between an observation `y` (one
    /// component) and h(xi) = cos q₁ of an arbitrary input state.
    pub fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        let d = y[0] - xi[0].cos();
        d * d
    }

    /// Hamiltonian H(x); conserved by the exact flow, and by the
    /// Gauss–Legendre scheme up to O(dt⁴) over long times (symplectic).
    pub fn hamiltonian(&self, x: &[f64; DIM]) -> f64 {
        let PendulumRodParams { m, l, g } = self.params;
        let [q1, q2, p1, p2] = *x;
        let c = (q1 - q2).cos();
        let kinetic = 6.0 * (p1 * p1 + 4.0 * p2 * p2 - 3.0 * p1 * p2 * c)
            / (m * l * l * (16.0 - 9.0 * c * c));
        let potential = 0.5 * m * g * l * (3.0 * (1.0 - q1.cos()) + (1.0 - q2.cos()));
        kinetic + potential
    }

    /// Right-hand side of Hamilton's equations:  ẋ = (∂H/∂p, −∂H/∂q).
    fn rhs(&self, x: &[f64; DIM]) -> [f64; DIM] {
        let PendulumRodParams { m, l, g } = self.params;
        let [q1, q2, p1, p2] = *x;
        let (s, c) = (q1 - q2).sin_cos();

        // q̇ᵢ = ∂H/∂pᵢ, common factor k = 6/(mℓ²(16 − 9cos²Δ))
        let k = 6.0 / (m * l * l * (16.0 - 9.0 * c * c));
        let q1_dot = k * (2.0 * p1 - 3.0 * c * p2);
        let q2_dot = k * (8.0 * p2 - 3.0 * c * p1);

        // ṗᵢ = −∂H/∂qᵢ, with ∂K/∂Δ = ½mℓ²·q̇₁q̇₂·sin Δ
        let w = q1_dot * q2_dot * s;
        let a = 0.5 * m * l * l;
        let p1_dot = -a * w - 1.5 * m * g * l * q1.sin();
        let p2_dot = a * w - 0.5 * m * g * l * q2.sin();

        [q1_dot, q2_dot, p1_dot, p2_dot]
    }

    /// Analytic Jacobian ∂rhs/∂x of Hamilton's equations, used by the Newton
    /// solver in [`gl4_step`]. Row i holds the gradient of component i of
    /// [`rhs`] with respect to (q₁, q₂, p₁, p₂); q-derivatives go through
    /// Δ = q₁ − q₂ (∂Δ/∂q₁ = 1, ∂Δ/∂q₂ = −1).
    fn rhs_jacobian(&self, x: &[f64; DIM]) -> [[f64; DIM]; DIM] {
        let PendulumRodParams { m, l, g } = self.params;
        let [q1, q2, p1, p2] = *x;
        let (s, c) = (q1 - q2).sin_cos();

        let denom = 16.0 - 9.0 * c * c;
        let k = 6.0 / (m * l * l * denom);
        let k_d = -k * (18.0 * s * c) / denom; // ∂k/∂Δ  (∂denom/∂Δ = 18sc)

        let q1_dot = k * (2.0 * p1 - 3.0 * c * p2);
        let q2_dot = k * (8.0 * p2 - 3.0 * c * p1);

        // ∂q̇ᵢ/∂Δ and ∂q̇ᵢ/∂pⱼ
        let dq1_d = k_d * (2.0 * p1 - 3.0 * c * p2) + k * 3.0 * s * p2;
        let dq2_d = k_d * (8.0 * p2 - 3.0 * c * p1) + k * 3.0 * s * p1;
        let (dq1_p1, dq1_p2) = (2.0 * k, -3.0 * c * k);
        let (dq2_p1, dq2_p2) = (-3.0 * c * k, 8.0 * k);

        // w = q̇₁q̇₂ sin Δ and its partials
        let w_d = (dq1_d * q2_dot + q1_dot * dq2_d) * s + q1_dot * q2_dot * c;
        let w_p1 = (dq1_p1 * q2_dot + q1_dot * dq2_p1) * s;
        let w_p2 = (dq1_p2 * q2_dot + q1_dot * dq2_p2) * s;

        // ṗ₁ = −a·w − (3/2)mgℓ sin q₁,  ṗ₂ = a·w − ½mgℓ sin q₂
        let a = 0.5 * m * l * l;
        [
            [dq1_d, -dq1_d, dq1_p1, dq1_p2],
            [dq2_d, -dq2_d, dq2_p1, dq2_p2],
            [
                -a * w_d - 1.5 * m * g * l * q1.cos(),
                a * w_d,
                -a * w_p1,
                -a * w_p2,
            ],
            [
                a * w_d,
                -a * w_d - 0.5 * m * g * l * q2.cos(),
                a * w_p1,
                a * w_p2,
            ],
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

    /// φ(x) together with its exact Jacobian ∂φ/∂x (see
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
/// [`DIM`] = 4.
impl Model<DIM> for PendulumRodSystem {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        PendulumRodSystem::discrepancy(self, y, xi)
    }

    fn flow_inv(&self, xi: [f64; DIM]) -> [f64; DIM] {
        PendulumRodSystem::flow_inv(self, &xi)
    }

    fn flow(&self, xi: [f64; DIM]) -> [f64; DIM] {
        PendulumRodSystem::flow(self, &xi)
    }

    fn flow_jacobian(&self, xi: [f64; DIM]) -> [[f64; DIM]; DIM] {
        self.flow_with_jacobian(&xi).1
    }

    /// The two angles live on [−π, π) (the dynamics are 2π-periodic in
    /// each); the momenta are plain.
    fn periodic(&self) -> [Option<(f64, f64)>; DIM] {
        use std::f64::consts::PI;
        [Some((-PI, PI)), Some((-PI, PI)), None, None]
    }

    fn obs_dim(&self) -> usize {
        1
    }

    fn obs(&self, xi: &[f64]) -> DVector<f64> {
        DVector::from_element(1, xi[0].cos())
    }

    /// ∇(cos q₁) = (−sin q₁, 0, 0, 0).
    fn obs_jacobian(&self, xi: &[f64]) -> DMatrix<f64> {
        DMatrix::from_row_slice(1, DIM, &[-xi[0].sin(), 0.0, 0.0, 0.0])
    }

    /// The Hamiltonian is time-independent, so the Gauss–Legendre flow is the
    /// same at every step: autonomous.
    fn is_autonomous(&self) -> bool {
        true
    }

    fn state_labels(&self) -> Vec<String> {
        ["q_1", "q_2", "p_1", "p_2"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X0: [f64; DIM] = [2.453, -2.7727, 0.0, 0.0];

    /// Deliberately non-unit parameters, to exercise every parameter in the
    /// dynamics (the defaults hide errors that cancel at m = ℓ = 1).
    const ODD_PARAMS: PendulumRodParams = PendulumRodParams {
        m: 2.5,
        l: 0.8,
        g: 3.7,
    };

    #[test]
    fn energy_is_conserved() {
        // The rod pendulum's fast bottom swings make the (bounded, O(dt⁴))
        // energy oscillation of the scheme much larger than the point-mass
        // pendulum's at equal dt: measured max |H − H0| over t ∈ [0, 5] from
        // (1.5, 1.4, 0, 0) is 1.2e-4 at dt = 0.01, shrinking ×16 per dt
        // halving (verified 4th order). At dt = 0.0025 it is 4.1e-7 absolute
        // ≈ 2.3e-8 relative, so 1e-7 relative bounds it with margin while
        // still catching any rhs/Hamiltonian inconsistency (those drift
        // orders of magnitude more, independently of dt).
        let x0 = [1.5, 1.4, 0.0, 0.0];
        let dt = 0.0025;
        for sys in [PendulumRodSystem::new(dt), PendulumRodSystem::with_params(ODD_PARAMS, dt)] {
            let h0 = sys.hamiltonian(&x0);
            let mut x = x0;
            for _ in 0..2000 {
                x = sys.flow(&x); // 2000 steps of dt = 0.0025 → t = 5
                assert!(
                    (sys.hamiltonian(&x) - h0).abs() < 1e-7 * h0.abs(),
                    "energy drift: H = {} vs H0 = {h0}",
                    sys.hamiltonian(&x)
                );
            }
        }
    }

    #[test]
    fn jacobian_matches_finite_differences() {
        // Run at non-unit parameters so every parameter enters the check.
        let sys = PendulumRodSystem::with_params(ODD_PARAMS, 0.01);
        let x = [1.3, -0.7, 0.8, -1.9];
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
        let sys = PendulumRodSystem::new(0.01);
        let y = sys.flow(&X0);
        let z = sys.flow_inv(&y);
        for i in 0..DIM {
            assert!(
                (z[i] - X0[i]).abs() < 1e-11,
                "round-trip error at {i}: {}",
                z[i] - X0[i]
            );
        }
    }
}
