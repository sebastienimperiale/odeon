//! Planar Kepler problem with a softened potential — forward problem solver.
//!
//! One body of unit mass in the gravitational field of a fixed center at the
//! origin (the reduced two-body problem in the orbital plane). The state is
//! x = (q₁, q₂, p₁, p₂): Cartesian position and momentum. The Hamiltonian
//! uses the Plummer-softened potential
//!
//!   H(x) = |p|²/2 − μ / √(|q|² + a²),
//!
//! i.e. q̇ = p, ṗ = −μ q / (|q|² + a²)^{3/2}. With a = 0 this is exactly
//! Kepler's problem (μ = GM); the softening length a > 0 (default 0.1)
//! bounds the force by μ/a² so that the flow — and its Newton solve — is
//! well defined at *every* point of the filter's state-space box, which
//! necessarily contains the singular origin. For an orbit at |q| ≈ 1 the
//! softening changes the dynamics by O(a²).
//!
//! Units: μ = 1 gives a circular orbit of radius 1 with speed 1 and period
//! 2π; more generally [`perihelion_state`] builds the orbit of semi-major
//! axis 1 and eccentricity e, started at perihelion.
//!
//! Conserved quantities: the energy H and the angular momentum
//! L = q₁p₂ − q₂p₁ (the softened force is still central). L is a quadratic
//! invariant, which the Gauss–Legendre scheme conserves exactly; H is
//! conserved up to the bounded O(dt⁴) oscillation of a symplectic scheme.
//!
//! Three scalar observations are available ([`KeplerObservation`]): a
//! Cartesian coordinate q₁ (line-of-sight), the range |q| (leaves the polar
//! angle unobserved: expect ring-shaped densities in the (q₁, q₂) plane),
//! and the bearing atan2(q₂, q₁) (bearings-only tracking, the range is
//! unobserved).
//!
//! Time stepping: the same symplectic Gauss–Legendre 4 scheme as the
//! pendulums ([`crate::gl4`]), so flow_inv = one step with −dt is exact.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use crate::model::Model;
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

/// State dimension: (q₁, q₂, p₁, p₂).
pub const DIM: usize = 4;

/// Physical parameters. [`Default`]: μ = 1, a = 0.1.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KeplerParams {
    /// Gravitational parameter μ = GM of the central mass.
    pub mu: f64,
    /// Plummer softening length a (0 = exact Kepler potential, singular at
    /// the origin).
    pub softening: f64,
}

impl Default for KeplerParams {
    fn default() -> Self {
        KeplerParams {
            mu: 1.0,
            softening: 0.1,
        }
    }
}

/// The scalar observation h(x) of the Kepler system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum KeplerObservation {
    /// h(x) = q₁ — one Cartesian coordinate (line-of-sight position).
    #[default]
    Q1,
    /// h(x) = |q| — the range (distance to the center); the polar angle is
    /// unobserved.
    Range,
    /// h(x) = atan2(q₂, q₁) ∈ (−π, π] — the bearing; the range is
    /// unobserved. Discrepancies are taken modulo 2π (the shortest angular
    /// difference), so the branch cut at ±π is invisible to the filter.
    Bearing,
}

impl KeplerObservation {
    /// All variants, in display order.
    pub const ALL: [KeplerObservation; 3] = [
        KeplerObservation::Q1,
        KeplerObservation::Range,
        KeplerObservation::Bearing,
    ];

    /// h(x) for a flat state (only q₁, q₂ enter).
    pub fn h(self, x: &[f64]) -> f64 {
        match self {
            KeplerObservation::Q1 => x[0],
            KeplerObservation::Range => x[0].hypot(x[1]),
            KeplerObservation::Bearing => x[1].atan2(x[0]),
        }
    }

    /// Signed difference y − h with the observation's own geometry: plain
    /// for Q1 and Range, shortest angular difference in (−π, π] for Bearing.
    pub fn difference(self, y: f64, h: f64) -> f64 {
        let d = y - h;
        match self {
            KeplerObservation::Bearing => (d + PI).rem_euclid(2.0 * PI) - PI,
            _ => d,
        }
    }

    /// Gradient ∇h(x) with respect to the flat state (only q₁, q₂ enter).
    pub fn gradient(self, x: &[f64]) -> [f64; DIM] {
        match self {
            KeplerObservation::Q1 => [1.0, 0.0, 0.0, 0.0],
            KeplerObservation::Range => {
                let r = x[0].hypot(x[1]).max(1e-300);
                [x[0] / r, x[1] / r, 0.0, 0.0]
            }
            KeplerObservation::Bearing => {
                let r2 = (x[0] * x[0] + x[1] * x[1]).max(1e-300);
                [-x[1] / r2, x[0] / r2, 0.0, 0.0]
            }
        }
    }

    /// Short human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            KeplerObservation::Q1 => "q₁",
            KeplerObservation::Range => "|q|",
            KeplerObservation::Bearing => "bearing",
        }
    }
}

/// State at the perihelion of the orbit of semi-major axis 1 and
/// eccentricity `e` ∈ [0, 1) for μ = 1 (exact Kepler): q = (1 − e, 0),
/// p = (0, √((1 + e)/(1 − e))). The orbit period is 2π. With softening the
/// orbit is slightly perturbed (precesses), not exactly this ellipse.
pub fn perihelion_state(e: f64) -> [f64; DIM] {
    assert!((0.0..1.0).contains(&e), "eccentricity must be in [0, 1)");
    [1.0 - e, 0.0, 0.0, ((1.0 + e) / (1.0 - e)).sqrt()]
}

/// Solver for the softened planar Kepler forward problem.
///
/// # Example
/// ```
/// use ode_models::models::kepler::{DIM, KeplerObservation, KeplerSystem, perihelion_state};
///
/// let sys = KeplerSystem::new(0.01, KeplerObservation::Q1);
/// let x1 = sys.flow(&perihelion_state(0.5)); // one step: t^0 → t^1
/// ```
pub struct KeplerSystem {
    // ── Public ───────────────────────────────────────────────────────────────
    pub dt: f64,
    /// Physical parameters (μ, softening).
    pub params: KeplerParams,
    /// Which scalar observation h(x) the system reports.
    pub observation: KeplerObservation,

}

impl KeplerSystem {
    /// Build the system with the default physical parameters
    /// ([`KeplerParams::default`]).
    ///
    /// The state is (q₁, q₂, p₁, p₂).
    ///
    /// * `dt`          — time step
    /// * `observation` — the scalar observation h(x)
    pub fn new(dt: f64, observation: KeplerObservation) -> Self {
        Self::with_params(KeplerParams::default(), dt, observation)
    }

    /// Build the system from explicit physical parameters (see
    /// [`new`](Self::new) for the other arguments).
    pub fn with_params(params: KeplerParams, dt: f64, observation: KeplerObservation) -> Self {
        KeplerSystem { dt, params, observation }
    }

    /// Squared discrepancy |y − h(xi)|² between an observation `y` (one
    /// component) and the observation of an arbitrary input state (angular
    /// difference for the bearing).
    pub fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        let d = self.observation.difference(y[0], self.observation.h(xi));
        d * d
    }

    /// Hamiltonian H(x) = |p|²/2 − μ/√(|q|² + a²).
    pub fn hamiltonian(&self, x: &[f64; DIM]) -> f64 {
        let KeplerParams { mu, softening } = self.params;
        let [q1, q2, p1, p2] = *x;
        0.5 * (p1 * p1 + p2 * p2) - mu / (q1 * q1 + q2 * q2 + softening * softening).sqrt()
    }

    /// Angular momentum L = q₁p₂ − q₂p₁ (conserved: the force is central).
    pub fn angular_momentum(&self, x: &[f64; DIM]) -> f64 {
        x[0] * x[3] - x[1] * x[2]
    }

    /// Right-hand side of Hamilton's equations:  q̇ = p, ṗ = −μ q / s^{3/2}
    /// with s = |q|² + a².
    fn rhs(&self, x: &[f64; DIM]) -> [f64; DIM] {
        let KeplerParams { mu, softening } = self.params;
        let [q1, q2, p1, p2] = *x;
        let s = q1 * q1 + q2 * q2 + softening * softening;
        let f = -mu / (s * s.sqrt());
        [p1, p2, f * q1, f * q2]
    }

    /// Analytic Jacobian ∂rhs/∂x, used by the Newton solver in
    /// [`gl4_step`]. The force block is −μ (I/s^{3/2} − 3 q qᵀ/s^{5/2}).
    fn rhs_jacobian(&self, x: &[f64; DIM]) -> [[f64; DIM]; DIM] {
        let KeplerParams { mu, softening } = self.params;
        let [q1, q2, _, _] = *x;
        let s = q1 * q1 + q2 * q2 + softening * softening;
        let s32 = s * s.sqrt();
        let f = -mu / s32; // ∂ṗᵢ/∂qⱼ = f·(δᵢⱼ − 3 qᵢqⱼ/s)
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [f * (1.0 - 3.0 * q1 * q1 / s), f * (-3.0 * q1 * q2 / s), 0.0, 0.0],
            [f * (-3.0 * q1 * q2 / s), f * (1.0 - 3.0 * q2 * q2 / s), 0.0, 0.0],
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
impl Model<DIM> for KeplerSystem {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        KeplerSystem::discrepancy(self, y, xi)
    }

    fn flow_inv(&self, xi: [f64; DIM]) -> [f64; DIM] {
        KeplerSystem::flow_inv(self, &xi)
    }

    fn flow(&self, xi: [f64; DIM]) -> [f64; DIM] {
        KeplerSystem::flow(self, &xi)
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

    fn obs_jacobian(&self, xi: &[f64]) -> DMatrix<f64> {
        DMatrix::from_row_slice(1, DIM, &self.observation.gradient(xi))
    }

    /// The bearing innovation is the shortest angular difference.
    fn innovation(&self, y: &[f64], xi: &[f64]) -> DVector<f64> {
        DVector::from_element(1, self.observation.difference(y[0], self.observation.h(xi)))
    }

    /// The Hamiltonian is time-independent: autonomous.
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

    /// Deliberately non-unit parameters, so every parameter enters the
    /// checks (μ = 1 and a small softening hide errors that cancel).
    const ODD_PARAMS: KeplerParams = KeplerParams {
        mu: 1.7,
        softening: 0.3,
    };

    /// The trajectory from `x0` (x0 included) and the system.
    fn run(params: KeplerParams, x0: [f64; DIM], dt: f64, steps: usize) -> (KeplerSystem, Vec<[f64; DIM]>) {
        let sys = KeplerSystem::with_params(params, dt, KeplerObservation::Q1);
        let mut states = vec![x0];
        for _ in 0..steps {
            states.push(sys.flow(states.last().unwrap()));
        }
        (sys, states)
    }

    /// Energy over one full period of the e = 0.5 orbit (perihelion speed
    /// √3, the fastest part), at both parameter sets.
    #[test]
    fn energy_is_conserved() {
        for params in [KeplerParams::default(), ODD_PARAMS] {
            let x0 = perihelion_state(0.5);
            let (sys, states) = run(params, x0, 0.005, 1260); // t ≈ 2π
            let h0 = sys.hamiltonian(&x0);
            for x in &states {
                assert!(
                    (sys.hamiltonian(&x) - h0).abs() < 1e-7 * h0.abs(),
                    "energy drift with {params:?}: H = {} vs H0 = {h0}",
                    sys.hamiltonian(&x)
                );
            }
        }
    }

    /// L is a quadratic invariant of a central force field: Gauss–Legendre
    /// conserves it to the Newton tolerance, whatever dt.
    #[test]
    fn angular_momentum_is_conserved_exactly() {
        let x0 = perihelion_state(0.7);
        let (sys, states) = run(ODD_PARAMS, x0, 0.02, 500);
        let l0 = sys.angular_momentum(&x0);
        for x in &states {
            assert!(
                (sys.angular_momentum(&x) - l0).abs() < 1e-10,
                "angular momentum drift: L = {} vs L0 = {l0}",
                sys.angular_momentum(&x)
            );
        }
    }

    #[test]
    fn jacobian_matches_finite_differences() {
        let sys = KeplerSystem::with_params(ODD_PARAMS, 0.01, KeplerObservation::Q1);
        let x = [0.6, -0.35, 0.8, -1.9];
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
        let x0 = perihelion_state(0.5);
        let sys = KeplerSystem::new(0.01, KeplerObservation::Range);
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

    /// The softened flow must be evaluable everywhere in a filter box,
    /// including at the origin and at fast points that fly through it.
    #[test]
    fn flow_is_defined_at_the_origin() {
        let sys = KeplerSystem::new(0.01, KeplerObservation::Q1);
        for x in [[0.0; 4], [0.0, 0.0, 3.0, -3.0], [0.01, -0.01, -5.0, 5.0]] {
            let y = sys.flow_inv(&x);
            assert!(y.iter().all(|v| v.is_finite()), "flow_inv({x:?}) = {y:?}");
        }
    }

    /// Bearing discrepancies cross the branch cut without a jump.
    #[test]
    fn bearing_discrepancy_wraps() {
        let obs = KeplerObservation::Bearing;
        let just_below = [-1.0, -1e-9]; // bearing ≈ −π
        let just_above = [-1.0, 1e-9]; // bearing ≈ +π
        let d = obs.difference(obs.h(&just_below), obs.h(&just_above));
        assert!(d.abs() < 1e-8, "bearing difference across the cut: {d}");
        assert!((obs.difference(0.5, -0.5) - 1.0).abs() < 1e-15);
    }
}
