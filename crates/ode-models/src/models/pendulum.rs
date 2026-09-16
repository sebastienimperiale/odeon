//! Planar double pendulum — forward problem solver.
//!
//! Model from pendulum.pdf, §5.2 "The planar double pendulum":
//! two point masses m₁, m₂ connected by massless rigid rods of lengths
//! ℓ₁, ℓ₂ (runtime parameters, see [`PendulumParams`]; the defaults are the
//! paper's values), gravity g in the −y direction. The state is
//! x = (q₁, q₂, p₁, p₂), where qᵢ is the angle of rod i with the y-axis and
//! pᵢ its conjugate momentum. With Δ = q₁ − q₂, the Hamiltonian is
//!
//!   H(x) = (m₂ℓ₂²p₁² + (m₁+m₂)ℓ₁²p₂² − 2m₂p₁p₂ℓ₁ℓ₂cos Δ)
//!              / (2m₂ℓ₁²ℓ₂²(m₁ + m₂sin²Δ))
//!        + (m₁+m₂)gℓ₁(1 − cos q₁) + m₂gℓ₂(1 − cos q₂) + m₂gℓ₂,
//!
//! which at m₁ = m₂ = 1 is the paper's Hamiltonian.
//!
//! Hamilton's equations are integrated with the fourth-order Gauss–Legendre
//! method (2-stage implicit Runge–Kutta, symplectic), as in the paper.
//!
//! Unlike the linear [`crate::spring::SpringSystem`], there is no constant
//! transition matrix: the flow map and its inverse are exposed as
//! [`PendulumSystem::flow`] / [`PendulumSystem::flow_inv`] (one Gauss–Legendre
//! step with +dt / −dt; the scheme is symmetric, so stepping with −dt is the
//! exact inverse of the discrete flow). The Mortensen transport step should
//! convect with `|xi| sys.flow_inv(xi)`.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use crate::model::Model;
use nalgebra::{DMatrix, DVector};

/// State dimension: (q₁, q₂, p₁, p₂).
pub const DIM: usize = 4;

/// Physical parameters of the double pendulum. [`Default`] gives the values
/// of pendulum.pdf, §5.2: ℓ₁ = ℓ₂ = 1, m₁ = m₂ = 1, g = 9.8.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PendulumParams {
    /// Rod lengths ℓ₁, ℓ₂.
    pub l1: f64,
    pub l2: f64,
    /// Point masses m₁ (middle) and m₂ (tip).
    pub m1: f64,
    pub m2: f64,
    /// Gravity g (−y direction).
    pub g: f64,
}

impl Default for PendulumParams {
    fn default() -> Self {
        PendulumParams {
            l1: 1.0,
            l2: 1.0,
            m1: 1.0,
            m2: 1.0,
            g: 9.8,
        }
    }
}

/// Solver for the planar double pendulum forward problem (N fixed to 2).
///
/// The observation is hardcoded: h(x) = (x, y) position of the tip mass
/// ([`tip_position`](Self::tip_position)), which depends only on the two
/// angles and the rod lengths; the momenta are unobserved.
///
/// # Example
/// ```
/// use ode_models::models::pendulum::{DIM, PendulumSystem};
///
/// let sys = PendulumSystem::new(0.01);
/// let x1 = sys.flow(&[2.453, -2.7727, 0.0, 0.0]); // one step: t^0 → t^1
/// ```
pub struct PendulumSystem {
    // ── Public ───────────────────────────────────────────────────────────────
    pub dt: f64,
    /// Physical parameters (rod lengths, masses, gravity).
    pub params: PendulumParams,

}

impl PendulumSystem {
    /// Number of angles / momenta (hardcoded).
    pub const N: usize = 2;

    /// Build the system with the paper's physical parameters
    /// ([`PendulumParams::default`]).
    ///
    /// The state is (q₁, q₂, p₁, p₂); the paper's target trajectory starts
    /// from (1.5, 1.4, 0, 0).
    ///
    /// * `dt`    — time step (the paper uses 10⁻²)
    pub fn new(dt: f64) -> Self {
        Self::with_params(PendulumParams::default(), dt)
    }

    /// Build the system from explicit physical parameters (see [`new`](Self::new)
    /// for the other arguments).
    pub fn with_params(params: PendulumParams, dt: f64) -> Self {
        PendulumSystem { dt, params }
    }

    /// (x, y) position of the tip mass (mass 2) — depends only on the two
    /// angles and the rod lengths. Same convention as
    /// scripts/visualize_pendulum.py: the pivot sits at (0, ℓ₁ + ℓ₂) and the
    /// angles are measured from the downward vertical, so q₁ = q₂ = 0 puts
    /// the tip at the origin.
    pub fn tip_position(&self, q1: f64, q2: f64) -> [f64; 2] {
        let (l1, l2) = (self.params.l1, self.params.l2);
        [
            l1 * q1.sin() + l2 * q2.sin(),
            (l1 + l2) - l1 * q1.cos() - l2 * q2.cos(),
        ]
    }

    /// Squared discrepancy |y − h(xi)|² between an observation `y` (a tip
    /// position) and the observation of an arbitrary input state, with
    /// h(x) = [`tip_position`](Self::tip_position): the squared Euclidean
    /// distance between the two tip positions.
    pub fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        let h = self.tip_position(xi[0], xi[1]);
        let dx = y[0] - h[0];
        let dy = y[1] - h[1];
        dx * dx + dy * dy
    }

    /// Hamiltonian H(x); conserved by the exact flow, and by the
    /// Gauss–Legendre scheme up to O(dt⁴) over long times (symplectic).
    pub fn hamiltonian(&self, x: &[f64; DIM]) -> f64 {
        let PendulumParams { l1, l2, m1, m2, g } = self.params;
        let mu = m1 + m2;
        let [q1, q2, p1, p2] = *x;
        let (s, c) = (q1 - q2).sin_cos();
        let kinetic = (m2 * p1 * p1 * l2 * l2 + mu * p2 * p2 * l1 * l1
            - 2.0 * m2 * p1 * p2 * l1 * l2 * c)
            / (2.0 * m2 * l1 * l1 * l2 * l2 * (m1 + m2 * s * s));
        let potential =
            mu * g * l1 * (1.0 - q1.cos()) + m2 * g * l2 * (1.0 - q2.cos()) + m2 * g * l2;
        kinetic + potential
    }

    /// Right-hand side of Hamilton's equations:  ẋ = (∂H/∂p, −∂H/∂q).
    fn rhs(&self, x: &[f64; DIM]) -> [f64; DIM] {
        let PendulumParams { l1, l2, m1, m2, g } = self.params;
        let mu = m1 + m2;
        let [q1, q2, p1, p2] = *x;
        let (s, c) = (q1 - q2).sin_cos();
        let ll = l1 * l1 * l2 * l2;
        let denom = m2 * ll * (m1 + m2 * s * s);

        // q̇ᵢ = ∂H/∂pᵢ
        let q1_dot = (m2 * p1 * l2 * l2 - m2 * p2 * l1 * l2 * c) / denom;
        let q2_dot = (mu * p2 * l1 * l1 - m2 * p1 * l1 * l2 * c) / denom;

        // ∂K/∂(q₁−q₂), with K the kinetic term: quotient rule on T/(2·denom)
        let t = m2 * p1 * p1 * l2 * l2 + mu * p2 * p2 * l1 * l1 - 2.0 * m2 * p1 * p2 * l1 * l2 * c;
        let ddenom = 2.0 * m2 * m2 * ll * s * c; // ∂denom/∂(q₁−q₂)
        let c1 = (m2 * p1 * p2 * l1 * l2 * s) / denom - t * ddenom / (2.0 * denom * denom);

        // ṗᵢ = −∂H/∂qᵢ
        let p1_dot = -c1 - mu * g * l1 * q1.sin();
        let p2_dot = c1 - m2 * g * l2 * q2.sin();

        [q1_dot, q2_dot, p1_dot, p2_dot]
    }

    /// Analytic Jacobian ∂rhs/∂x of Hamilton's equations, used by the Newton
    /// solver in [`gl4_step`]. Row i holds the gradient of component i of
    /// [`rhs`] with respect to (q₁, q₂, p₁, p₂); q-derivatives go through
    /// δ = q₁ − q₂ (∂δ/∂q₁ = 1, ∂δ/∂q₂ = −1).
    fn rhs_jacobian(&self, x: &[f64; DIM]) -> [[f64; DIM]; DIM] {
        let PendulumParams { l1, l2, m1, m2, g } = self.params;
        let mu = m1 + m2;
        let [q1, q2, p1, p2] = *x;
        let (s, c) = (q1 - q2).sin_cos();
        let ll = l1 * l1 * l2 * l2;
        let denom = m2 * ll * (m1 + m2 * s * s);
        let ddenom = 2.0 * m2 * m2 * ll * s * c; // ∂denom/∂δ
        let d2denom = 2.0 * m2 * m2 * ll * (c * c - s * s); // ∂²denom/∂δ²
        let d2 = denom * denom;
        let d3 = d2 * denom;

        // Kinetic numerator T and its partials (see `rhs`).
        let t = m2 * p1 * p1 * l2 * l2 + mu * p2 * p2 * l1 * l1 - 2.0 * m2 * p1 * p2 * l1 * l2 * c;
        let t_d = 2.0 * m2 * p1 * p2 * l1 * l2 * s;
        let t_p1 = 2.0 * m2 * p1 * l2 * l2 - 2.0 * m2 * p2 * l1 * l2 * c;
        let t_p2 = 2.0 * mu * p2 * l1 * l1 - 2.0 * m2 * p1 * l1 * l2 * c;

        // q̇₁ = N₁/denom with N₁ = m₂(p₁l₂² − p₂l₁l₂c)
        let n1 = m2 * (p1 * l2 * l2 - p2 * l1 * l2 * c);
        let f1_d = (m2 * p2 * l1 * l2 * s) / denom - n1 * ddenom / d2;
        let f1_p1 = m2 * l2 * l2 / denom;
        let f1_p2 = -m2 * l1 * l2 * c / denom;

        // q̇₂ = N₂/denom with N₂ = (m₁+m₂)p₂l₁² − m₂p₁l₁l₂c
        let n2 = mu * p2 * l1 * l1 - m2 * p1 * l1 * l2 * c;
        let f2_d = (m2 * p1 * l1 * l2 * s) / denom - n2 * ddenom / d2;
        let f2_p1 = -m2 * l1 * l2 * c / denom;
        let f2_p2 = mu * l1 * l1 / denom;

        // c₁ = m₂p₁p₂l₁l₂s/denom − T·denom'/(2·denom²)  (quotient rules)
        let c1_d = m2 * p1 * p2 * l1 * l2 * (c / denom - s * ddenom / d2)
            - (t_d * ddenom + t * d2denom) / (2.0 * d2)
            + t * ddenom * ddenom / d3;
        let c1_p1 = m2 * p2 * l1 * l2 * s / denom - t_p1 * ddenom / (2.0 * d2);
        let c1_p2 = m2 * p1 * l1 * l2 * s / denom - t_p2 * ddenom / (2.0 * d2);

        // ṗ₁ = −c₁ − (m₁+m₂)gl₁ sin q₁,  ṗ₂ = c₁ − m₂gl₂ sin q₂
        [
            [f1_d, -f1_d, f1_p1, f1_p2],
            [f2_d, -f2_d, f2_p1, f2_p2],
            [-c1_d - mu * g * l1 * q1.cos(), c1_d, -c1_p1, -c1_p2],
            [c1_d, -c1_d - m2 * g * l2 * q2.cos(), c1_p1, c1_p2],
        ]
    }

    /// One step of the fourth-order Gauss–Legendre method (2-stage implicit
    /// Runge–Kutta) with step `h`. The 8-dimensional stage system
    /// kᵢ = f(x + h Σⱼ aᵢⱼ kⱼ) is solved by Newton's method with the analytic
    /// Jacobian [`rhs_jacobian`]; panics if it does not converge.
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
impl Model<DIM> for PendulumSystem {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        PendulumSystem::discrepancy(self, y, xi)
    }

    fn flow_inv(&self, xi: [f64; DIM]) -> [f64; DIM] {
        PendulumSystem::flow_inv(self, &xi)
    }

    fn flow(&self, xi: [f64; DIM]) -> [f64; DIM] {
        PendulumSystem::flow(self, &xi)
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
        2
    }

    fn obs(&self, xi: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(&self.tip_position(xi[0], xi[1]))
    }

    /// Jacobian of the tip position with respect to (q₁, q₂, p₁, p₂):
    /// ∂x/∂qᵢ = ℓᵢ cos qᵢ, ∂y/∂qᵢ = ℓᵢ sin qᵢ, no momentum dependence.
    fn obs_jacobian(&self, xi: &[f64]) -> DMatrix<f64> {
        let (l1, l2) = (self.params.l1, self.params.l2);
        DMatrix::from_row_slice(
            2,
            DIM,
            &[
                l1 * xi[0].cos(), l2 * xi[1].cos(), 0.0, 0.0,
                l1 * xi[0].sin(), l2 * xi[1].sin(), 0.0, 0.0,
            ],
        )
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

    const X0: [f64; DIM] = [1.5, 1.4, 0.0, 0.0];

    /// Deliberately asymmetric parameters, to exercise every parameter in the
    /// dynamics (the defaults hide errors that cancel at ℓᵢ = mᵢ = 1).
    const ODD_PARAMS: PendulumParams = PendulumParams {
        l1: 0.7,
        l2: 1.3,
        m1: 2.0,
        m2: 0.5,
        g: 3.7,
    };

    /// Symplectic scheme: the energy error oscillates at O(dt⁴) without
    /// secular drift; 1e-7 relative comfortably bounds it at dt = 0.01
    /// over 500 steps (t = 5), at the paper's and at asymmetric parameters.
    #[test]
    fn energy_is_conserved() {
        for sys in [PendulumSystem::new(0.01), PendulumSystem::with_params(ODD_PARAMS, 0.01)] {
            let h0 = sys.hamiltonian(&X0);
            let mut x = X0;
            for _ in 0..500 {
                x = sys.flow(&x);
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
        // Run at asymmetric parameters so every parameter enters the check.
        let sys = PendulumSystem::with_params(ODD_PARAMS, 0.01);
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
        let sys = PendulumSystem::new(0.01);
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
