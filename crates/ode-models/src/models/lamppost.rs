//! Lamppost — a two-dimensional toy model for multimodality: a man seen
//! only by his distance to a lamppost (formerly `lamppost`).
//!
//! A man stands in the plane at (x, y), next to a lamppost at the origin,
//! and does not move: d_t (x, y) = 0. All we observe is his **squared
//! distance to the lamppost**, h(x, y) = x² + y². The reference starts at
//! (x, y) = (0, 1), so the observation is y_obs ≡ 1 and every point of the
//! unit circle explains the data equally well: the value function V is
//! flat on the ring x² + y² = 1 and the density p = e^{−V/ε} is a **ring**
//! — a continuum of maxima. The grid filter represents it as is; the
//! tracker, a single Gaussian, must pick one point of the ring and does
//! (it converges to *some* point at distance 1, chosen by the prior), which
//! makes the model the smallest possible illustration of what the closure
//! loses.
//!
//! The drunkness is in the filter's eyes: with model noise q_d > 0 the
//! filter believes the man random-walks, and the ring is kept from
//! collapsing to a curve by the diffusion step; with q ≡ 0 it sharpens
//! forever. Optionally the reference itself random-walks (the `Walk` of
//! the reference generator in `ode-models-spec`), so the ring has to
//! follow a moving radius.
//!
//! The flow is the identity, exactly: no time stepping, no Jacobian to
//! compute (Φ = I). Observation noise comes from the reference generator,
//! as for every model.

use crate::model::Model;
use nalgebra::{DMatrix, DVector};

/// State dimension: (x, y).
pub const DIM: usize = 2;

/// The reference initial state of the toy: (x, y) = (0, 1).
pub const DEFAULT_STATE: [f64; DIM] = [0.0, 1.0];

/// The drunk man: a point in the plane observed through its squared
/// distance to the origin, with the identity flow.
///
/// # Example
/// ```
/// use ode_models::models::lamppost::{LamppostSystem, DEFAULT_STATE};
/// use ode_models::model::Model;
///
/// let sys = LamppostSystem::new(0.05);
/// assert_eq!(Model::<2>::flow(&sys, DEFAULT_STATE), DEFAULT_STATE); // the man stays put
/// assert_eq!(sys.obs(&[0.6, 0.8]), 1.0); // the ring x² + y² = 1
/// ```
pub struct LamppostSystem {
    pub dt: f64,
}

impl LamppostSystem {
    /// Build the system.
    ///
    /// * `dt` — time step (only sets the observation cadence and, in the
    ///   filter, the observation weight and diffusion horizon)
    pub fn new(dt: f64) -> Self {
        LamppostSystem { dt }
    }

    fn h_static(x: &[f64]) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }

    /// Observation h(x, y) = x² + y², the squared distance to the lamppost.
    pub fn obs(&self, x: &[f64]) -> f64 {
        Self::h_static(x)
    }

    /// Squared discrepancy |y − h(xi)|² between an observation `y` (one
    /// component) and the observation of an arbitrary input state.
    pub fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        let d = y[0] - Self::h_static(xi);
        d * d
    }
}

/// [`Model`] interface for the Mortensen filter, at the fixed dimension
/// [`DIM`] = 2. The flow is the identity.
impl Model<DIM> for LamppostSystem {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        LamppostSystem::discrepancy(self, y, xi)
    }

    /// φ⁻¹ = identity.
    fn flow_inv(&self, xi: [f64; DIM]) -> [f64; DIM] {
        xi
    }

    /// φ = identity.
    fn flow(&self, xi: [f64; DIM]) -> [f64; DIM] {
        xi
    }

    /// Φ = I.
    fn flow_jacobian(&self, _xi: [f64; DIM]) -> [[f64; DIM]; DIM] {
        [[1.0, 0.0], [0.0, 1.0]]
    }

    fn obs_dim(&self) -> usize {
        1
    }

    fn obs(&self, xi: &[f64]) -> DVector<f64> {
        DVector::from_element(1, Self::h_static(xi))
    }

    /// ∇h = (2x, 2y).
    fn obs_jacobian(&self, xi: &[f64]) -> DMatrix<f64> {
        DMatrix::from_row_slice(1, DIM, &[2.0 * xi[0], 2.0 * xi[1]])
    }

    /// The identity flow is trivially autonomous.
    fn is_autonomous(&self) -> bool {
        true
    }

    fn state_labels(&self) -> Vec<String> {
        vec!["x".to_string(), "y".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flow is the identity, the observation is the squared distance
    /// with gradient (2x, 2y), and the discrepancy vanishes on the ring.
    #[test]
    fn identity_flow_and_ring_observation() {
        let sys = LamppostSystem::new(0.1);
        let x = [0.3, -0.7];
        assert_eq!(Model::<2>::flow(&sys, x), x);
        assert_eq!(Model::<2>::flow_inv(&sys, x), x);
        let g = Model::<2>::obs_jacobian(&sys, &x);
        let eps = 1e-6;
        for c in 0..2 {
            let (mut xp, mut xm) = (x, x);
            xp[c] += eps;
            xm[c] -= eps;
            let fd = (sys.obs(&xp) - sys.obs(&xm)) / (2.0 * eps);
            assert!((g[(0, c)] - fd).abs() < 1e-8);
        }
        // The reference (0, 1) observes 1: discrepancy vanishes on the ring, not off it.
        assert!(sys.discrepancy(&[1.0], &[0.6, 0.8]).abs() < 1e-15);
        assert!(sys.discrepancy(&[1.0], &[0.0, 0.0]) > 0.9);
    }

}
