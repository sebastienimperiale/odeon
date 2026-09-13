//! Random walk — a two-dimensional toy model for multimodality.
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
//! forever. Optionally ([`with_walk`](RandomWalkSystem::with_walk)) the
//! reference itself random-walks, so the ring has to follow a moving
//! radius.
//!
//! The flow is the identity, exactly: no time stepping, no Jacobian to
//! compute (Φ = I). Observation noise is available as for every model.

use crate::model::Model;
use crate::noise::{NoiseModel, NoiseSampler};
use nalgebra::{DMatrix, DVector};

/// State dimension: (x, y).
pub const DIM: usize = 2;

/// The reference initial state of the toy: (x, y) = (0, 1).
pub const DEFAULT_STATE: [f64; DIM] = [0.0, 1.0];

/// The drunk man: static (or optionally random-walking) point in the plane
/// observed through its squared distance to the origin.
///
/// # Example
/// ```
/// use ode_models::models::random_walk::{RandomWalkSystem, DEFAULT_STATE};
///
/// let mut sys = RandomWalkSystem::new(DEFAULT_STATE, 0.05);
/// sys.forward();              // the man stays at (0, 1)
/// assert_eq!(sys.h(&[0.6, 0.8]), 1.0); // the ring x² + y² = 1
/// ```
pub struct RandomWalkSystem {
    // ── Public ───────────────────────────────────────────────────────────────
    pub dt: f64,
    /// Trajectory so far: `states[n]` = (x, y) at t^n — constant unless
    /// [`with_walk`](Self::with_walk) is on.
    pub states: Vec<DVector<f64>>,

    // ── Private ──────────────────────────────────────────────────────────────
    /// Cached observation of the current state, y_n = h(x_n) + η_n.
    y_obs: f64,
    /// Observation-noise draws, one per step.
    noise: NoiseSampler,
    /// Optional random walk of the reference: Gaussian increments of
    /// standard deviation σ√dt per component, one sampler per component.
    walk: Option<[NoiseSampler; 2]>,
}

impl RandomWalkSystem {
    /// Build the system.
    ///
    /// * `x0` — initial position (the toy's reference is [`DEFAULT_STATE`])
    /// * `dt` — time step (only sets the observation cadence and, in the
    ///   filter, the observation weight and diffusion horizon)
    pub fn new(x0: [f64; DIM], dt: f64) -> Self {
        RandomWalkSystem {
            dt,
            states: vec![DVector::from_row_slice(&x0)],
            y_obs: Self::h_static(&x0),
            noise: NoiseModel::None.sampler(dt, 0),
            walk: None,
        }
    }

    /// Add observation noise: from now on the cached observation is
    /// y_n = h(x_n) + η_n with η drawn from `noise` (deterministic in
    /// `seed`). Re-caches the current observation with the first draw, so
    /// call this right after construction.
    pub fn with_obs_noise(mut self, noise: NoiseModel, seed: u64) -> Self {
        self.noise = noise.sampler(self.dt, seed);
        let x = self.states.last().expect("states holds the initial condition");
        self.y_obs = Self::h_static(x.as_slice()) + self.noise.next_sample();
        self
    }

    /// Let the reference random-walk: each step adds independent Gaussian
    /// increments of standard deviation `std·√dt` to x and y (a Brownian
    /// motion of diffusion coefficient std²), deterministic in `seed`. Off
    /// by default: the toy's reference is static.
    pub fn with_walk(mut self, std: f64, seed: u64) -> Self {
        let inc = NoiseModel::Gaussian {
            std: std * self.dt.sqrt(),
        };
        self.walk = Some([inc.sampler(self.dt, seed), inc.sampler(self.dt, seed.wrapping_add(1))]);
        self
    }

    fn h_static(x: &[f64]) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }

    /// Observation h(x, y) = x² + y², the squared distance to the lamppost.
    pub fn h(&self, x: &[f64]) -> f64 {
        Self::h_static(x)
    }

    /// Forward operator: the man stays where he is (plus the optional
    /// random-walk increment); the observation is refreshed.
    pub fn forward(&mut self) {
        let mut x = self.states.last().expect("states holds the initial condition").clone();
        if let Some(walk) = &mut self.walk {
            x[0] += walk[0].next_sample();
            x[1] += walk[1].next_sample();
        }
        self.y_obs = Self::h_static(x.as_slice()) + self.noise.next_sample();
        self.states.push(x);
    }

    /// Squared discrepancy |y_n − h(xi)|².
    pub fn discrepancy(&self, xi: &[f64]) -> f64 {
        let d = self.y_obs - Self::h_static(xi);
        d * d
    }
}

/// [`Model`] interface for the Mortensen filter, at the fixed dimension
/// [`DIM`] = 2. The flow is the identity.
impl Model<DIM> for RandomWalkSystem {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn forward(&mut self) {
        RandomWalkSystem::forward(self);
    }

    fn discrepancy(&self, xi: &[f64]) -> f64 {
        RandomWalkSystem::discrepancy(self, xi)
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

    fn n_obs(&self) -> usize {
        1
    }

    fn h(&self, xi: &[f64]) -> DVector<f64> {
        DVector::from_element(1, Self::h_static(xi))
    }

    fn y_obs(&self) -> DVector<f64> {
        DVector::from_element(1, self.y_obs)
    }

    /// ∇h = (2x, 2y).
    fn obs_jacobian(&self, xi: &[f64]) -> DMatrix<f64> {
        DMatrix::from_row_slice(1, DIM, &[2.0 * xi[0], 2.0 * xi[1]])
    }

    /// The identity flow is trivially autonomous.
    fn is_autonomous(&self) -> bool {
        true
    }

    fn states(&self) -> &[DVector<f64>] {
        &self.states
    }

    fn state_labels(&self) -> Vec<String> {
        vec!["x".to_string(), "y".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference stays put, the flow is the identity, the observation
    /// is the squared distance and its gradient (2x, 2y).
    #[test]
    fn static_reference_identity_flow_and_observation() {
        let mut sys = RandomWalkSystem::new(DEFAULT_STATE, 0.1);
        for _ in 0..5 {
            sys.forward();
        }
        assert_eq!(sys.states.last().unwrap().as_slice(), &DEFAULT_STATE);
        assert_eq!(Model::<2>::y_obs(&sys)[0], 1.0);
        let x = [0.3, -0.7];
        assert_eq!(Model::<2>::flow(&sys, x), x);
        assert_eq!(Model::<2>::flow_inv(&sys, x), x);
        let g = Model::<2>::obs_jacobian(&sys, &x);
        let eps = 1e-6;
        for c in 0..2 {
            let (mut xp, mut xm) = (x, x);
            xp[c] += eps;
            xm[c] -= eps;
            let fd = (sys.h(&xp) - sys.h(&xm)) / (2.0 * eps);
            assert!((g[(0, c)] - fd).abs() < 1e-8);
        }
        // Discrepancy vanishes on the ring, not off it.
        assert!(sys.discrepancy(&[0.6, 0.8]).abs() < 1e-15);
        assert!(sys.discrepancy(&[0.0, 0.0]) > 0.9);
    }

    /// The reference random walk moves the reference and is reproducible
    /// in its seed.
    #[test]
    fn optional_walk_moves_the_reference_reproducibly() {
        let run = |seed| {
            let mut sys = RandomWalkSystem::new(DEFAULT_STATE, 0.1).with_walk(0.5, seed);
            for _ in 0..20 {
                sys.forward();
            }
            sys.states.last().unwrap().clone()
        };
        let (a, b, c) = (run(1), run(1), run(2));
        assert_eq!(a, b, "not reproducible in the seed");
        assert!((&a - &c).norm() > 1e-6, "different seeds gave the same walk");
        assert!((&a - DVector::from_row_slice(&DEFAULT_STATE)).norm() > 1e-6, "the walk did not move");
    }

}
