//! Lorenz-63 system — forward problem solver.
//!
//! The classic dissipative chaotic benchmark (Lorenz 1963):
//!
//!   ẋ = σ (y − x),   ẏ = x (ρ − z) − y,   ż = x y − β z,
//!
//! state (x, y, z) ∈ ℝ³, parameters σ, ρ, β ([`LorenzParams`]; the defaults
//! σ = 10, ρ = 28, β = 8/3 give the strange attractor). No Hamiltonian
//! structure: the flow contracts phase-space volume at the constant rate
//! −(σ + 1 + β), so without model noise the filter density would collapse
//! onto the attractor — the model-noise covariance `FilterParams::q_diag`
//! is what keeps it spread.
//!
//! Fixed points: the origin, and for ρ > 1 the pair
//! C± = (±√(β(ρ−1)), ±√(β(ρ−1)), ρ−1) (the centers of the two wings).
//!
//! Time stepping: the same Gauss–Legendre 4 scheme as the other nonlinear
//! models ([`crate::gl4`]; no energy to conserve here, but it is A-stable
//! and symmetric, so flow_inv = one step with −dt is exact). Note the inverse flow of a dissipative system is expanding: grid
//! points near the filter-domain boundary are mapped far outside and folded
//! back by the extension, so leave a generous margin around the attractor.
//!
//! The observation is one coordinate ([`LorenzObservation`]: X, Y or Z).

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use crate::model::Model;
use nalgebra::{DMatrix, DVector};

/// State dimension: (x, y, z).
pub const DIM: usize = 3;

/// Parameters of the Lorenz-63 system. [`Default`]: σ = 10, ρ = 28, β = 8/3.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LorenzParams {
    /// Prandtl number σ.
    pub sigma: f64,
    /// Rayleigh number ρ.
    pub rho: f64,
    /// Geometric factor β.
    pub beta: f64,
}

impl Default for LorenzParams {
    fn default() -> Self {
        LorenzParams {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        }
    }
}

/// The observed coordinate h(x) of the Lorenz system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum LorenzObservation {
    #[default]
    X,
    Y,
    Z,
}

impl LorenzObservation {
    /// All variants, in display order.
    pub const ALL: [LorenzObservation; 3] =
        [LorenzObservation::X, LorenzObservation::Y, LorenzObservation::Z];

    /// Index of the observed component in the flat state.
    pub fn index(self) -> usize {
        match self {
            LorenzObservation::X => 0,
            LorenzObservation::Y => 1,
            LorenzObservation::Z => 2,
        }
    }

    /// h(x) for a flat state.
    pub fn h(self, x: &[f64]) -> f64 {
        x[self.index()]
    }

    /// Short human-readable label.
    pub fn label(self) -> &'static str {
        ["x", "y", "z"][self.index()]
    }
}

/// Solver for the Lorenz-63 forward problem.
///
/// # Example
/// ```
/// use ode_models::models::lorenz::{DIM, LorenzObservation, LorenzSystem};
///
/// let sys = LorenzSystem::new(0.01, LorenzObservation::X);
/// let x1 = sys.flow(&[1.0, 1.0, 1.0]); // one step: t^0 → t^1
/// ```
pub struct LorenzSystem {
    // ── Public ───────────────────────────────────────────────────────────────
    pub dt: f64,
    /// Parameters (σ, ρ, β).
    pub params: LorenzParams,
    /// Which coordinate the system reports as observation.
    pub observation: LorenzObservation,

}

impl LorenzSystem {
    /// Build the system with the classic parameters ([`LorenzParams::default`]).
    ///
    /// The state is (x, y, z).
    ///
    /// * `dt`          — time step
    /// * `observation` — the observed coordinate
    pub fn new(dt: f64, observation: LorenzObservation) -> Self {
        Self::with_params(LorenzParams::default(), dt, observation)
    }

    /// Build the system from explicit parameters (see [`new`](Self::new) for
    /// the other arguments).
    pub fn with_params(params: LorenzParams, dt: f64, observation: LorenzObservation) -> Self {
        LorenzSystem { dt, params, observation }
    }

    /// The non-trivial fixed points C± for the current parameters (ρ > 1):
    /// (±√(β(ρ−1)), ±√(β(ρ−1)), ρ−1); `None` when ρ ≤ 1.
    pub fn fixed_points(&self) -> Option<[[f64; DIM]; 2]> {
        let LorenzParams { rho, beta, .. } = self.params;
        (rho > 1.0).then(|| {
            let c = (beta * (rho - 1.0)).sqrt();
            [[c, c, rho - 1.0], [-c, -c, rho - 1.0]]
        })
    }

    /// Squared discrepancy |y − h(xi)|² between an observation `y` (one
    /// component) and the observation of an arbitrary input state.
    pub fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        let d = y[0] - self.observation.h(xi);
        d * d
    }

    /// Right-hand side of the Lorenz equations.
    fn rhs(&self, s: &[f64; DIM]) -> [f64; DIM] {
        let LorenzParams { sigma, rho, beta } = self.params;
        let [x, y, z] = *s;
        [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    }

    /// Analytic Jacobian ∂rhs/∂(x, y, z), used by the Newton solver.
    fn rhs_jacobian(&self, s: &[f64; DIM]) -> [[f64; DIM]; DIM] {
        let LorenzParams { sigma, rho, beta } = self.params;
        let [x, y, z] = *s;
        [
            [-sigma, sigma, 0.0],
            [rho - z, -1.0, -x],
            [y, x, -beta],
        ]
    }

    /// One step of the fourth-order Gauss–Legendre method (shared solver,
    /// see [`crate::gl4`]).
    fn step(&self, x: &[f64; DIM], h: f64) -> [f64; DIM] {
        crate::gl4::gl4_step(|x| self.rhs(x), |x| self.rhs_jacobian(x), x, h)
    }

    /// Discrete flow map φ: one Gauss–Legendre step of size `dt`.
    pub fn flow(&self, x: &[f64; DIM]) -> [f64; DIM] {
        self.step(x, self.dt)
    }

    /// φ(x) together with its exact Jacobian ∂φ/∂x (see
    /// [`crate::gl4::gl4_step_with_jacobian`]).
    pub fn flow_with_jacobian(&self, x: &[f64; DIM]) -> ([f64; DIM], [[f64; DIM]; DIM]) {
        crate::gl4::gl4_step_with_jacobian(|x| self.rhs(x), |x| self.rhs_jacobian(x), x, self.dt)
    }

    /// Inverse discrete flow map φ⁻¹: one Gauss–Legendre step of size `−dt`.
    /// Exact inverse of [`flow`](Self::flow) because the scheme is symmetric.
    pub fn flow_inv(&self, x: &[f64; DIM]) -> [f64; DIM] {
        self.step(x, -self.dt)
    }
}

/// [`Model`] interface for the Mortensen filter, at the fixed dimension
/// [`DIM`] = 3.
impl Model<DIM> for LorenzSystem {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        LorenzSystem::discrepancy(self, y, xi)
    }

    fn flow_inv(&self, xi: [f64; DIM]) -> [f64; DIM] {
        LorenzSystem::flow_inv(self, &xi)
    }

    fn flow(&self, xi: [f64; DIM]) -> [f64; DIM] {
        LorenzSystem::flow(self, &xi)
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

    fn obs_jacobian(&self, _xi: &[f64]) -> DMatrix<f64> {
        let mut row = [0.0; DIM];
        row[self.observation.index()] = 1.0;
        DMatrix::from_row_slice(1, DIM, &row)
    }

    /// The vector field is time-independent: autonomous.
    fn is_autonomous(&self) -> bool {
        true
    }

    fn state_labels(&self) -> Vec<String> {
        ["x", "y", "z"].iter().map(|s| s.to_string()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X0: [f64; DIM] = [1.0, 1.0, 1.0];

    /// Non-default parameters so every one of them enters the checks.
    const ODD_PARAMS: LorenzParams = LorenzParams {
        sigma: 7.3,
        rho: 21.0,
        beta: 1.9,
    };

    #[test]
    fn jacobian_matches_finite_differences() {
        let sys = LorenzSystem::with_params(ODD_PARAMS, 0.01, LorenzObservation::X);
        let x = [-3.2, 4.1, 17.5];
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
                    (jac[row][col] - fd).abs() < 1e-6 * (1.0 + fd.abs()),
                    "J[{row}][{col}] = {} but finite difference gives {fd}",
                    jac[row][col]
                );
            }
        }
    }

    #[test]
    fn flow_inv_inverts_flow() {
        let sys = LorenzSystem::new(0.01, LorenzObservation::Y);
        let x0 = [-8.0, 7.0, 27.0];
        let y = sys.flow(&x0);
        let z = sys.flow_inv(&y);
        for i in 0..DIM {
            assert!(
                (z[i] - x0[i]).abs() < 1e-10,
                "round-trip error at {i}: {}",
                z[i] - x0[i]
            );
        }
    }

    /// The fixed points C± are stationary for the discrete flow too.
    #[test]
    fn fixed_points_are_stationary() {
        let sys = LorenzSystem::with_params(ODD_PARAMS, 0.02, LorenzObservation::Z);
        for c in sys.fixed_points().expect("ρ > 1") {
            let y = sys.flow(&c);
            for i in 0..DIM {
                assert!((y[i] - c[i]).abs() < 1e-10, "fixed point moved: {c:?} → {y:?}");
            }
            let f = sys.rhs(&c);
            assert!(f.iter().all(|v| v.abs() < 1e-10), "rhs(C) = {f:?}");
        }
    }

    /// The trajectory stays on the attractor: bounded, and z never
    /// negative after the transient (the classic ranges are
    /// |x| ≲ 20, |y| ≲ 28, 0 < z ≲ 50).
    #[test]
    fn trajectory_is_bounded() {
        let sys = LorenzSystem::new(0.01, LorenzObservation::X);
        let mut s = X0;
        for n in 0..5000 {
            s = sys.flow(&s);
            if n >= 500 {
                assert!(s[0].abs() < 25.0 && s[1].abs() < 35.0 && s[2] > 0.0 && s[2] < 60.0,
                    "left the attractor: {s:?}");
            }
        }
    }

    /// Gauss–Legendre 4 is fourth order: halving dt divides the error
    /// against a reference (tiny-step) solution by ≈ 16 over a fixed time.
    #[test]
    fn scheme_is_fourth_order() {
        let t = 0.5;
        let solve = |dt: f64| {
            let sys = LorenzSystem::new(dt, LorenzObservation::X);
            let mut s = X0;
            for _ in 0..(t / dt).round() as usize {
                s = sys.flow(&s);
            }
            s
        };
        let reference = solve(2.5e-4);
        let err = |dt: f64| {
            let s = solve(dt);
            (0..DIM).map(|i| (s[i] - reference[i]).abs()).fold(0.0, f64::max)
        };
        let (e1, e2) = (err(0.02), err(0.01));
        let ratio = e1 / e2;
        assert!((13.0..19.0).contains(&ratio), "error ratio {ratio} (errors {e1:e}, {e2:e})");
    }
}
