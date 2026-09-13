//! N-harmonic oscillator — forward problem solver.
//!
//! Encapsulates physical/time parameters, the transition matrix T and its
//! inverse, and the sequence of states produced by successive calls to
//! [`SpringSystem::forward`].

use crate::model::Model;
use crate::noise::{NoiseModel, NoiseSampler};
use nalgebra::{DMatrix, DVector, Dyn, LU};

/// Solver for the N-spring forward problem.
///
/// # Example
/// ```
/// use ode_models::models::SpringSystem;
///
/// let y0 = nalgebra::DVector::from_row_slice(&[0.5, 1.0]);
/// let v0 = nalgebra::DVector::zeros(2);
/// let mut sys = SpringSystem::new(2, 1.0, 1.0, 0.005, y0, v0,
///                                 |x: &[f64]| x.iter().sum());
/// sys.forward();                    // one step: t^0 → t^1
/// println!("{:.6}", sys.trans);     // transition matrix
/// println!("{:.6}", sys.trans_inv); // its inverse
/// // sys.states[n] = [Y_n; V_n]
/// ```
pub struct SpringSystem {
    // ── Public ───────────────────────────────────────────────────────────────
    pub dt: f64,
    /// Transition matrix T (2N×2N): [Y_{n+1}; V_{n+1}] = T [Y_n; V_n]
    pub trans: DMatrix<f64>,
    /// Inverse transition matrix T⁻¹
    pub trans_inv: DMatrix<f64>,
    /// Trajectory so far: `states[n]` = [Y_n; V_n] (flat, positions then
    /// velocities). Holds the initial condition after [`new`]; each
    /// [`forward`] call appends one state. The last entry is the current
    /// state.
    pub states: Vec<DVector<f64>>,

    // ── Private (needed for time-stepping) ───────────────────────────────────
    lu: LU<f64, Dyn, Dyn>,
    mat_b: DMatrix<f64>,
    big_k: DMatrix<f64>,
    m: f64,

    // ── Private (observation) ────────────────────────────────────────────────
    /// Observation function h: state slice [Y; V] ↦ scalar observation.
    obs: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    /// Cached observation of the current state, y_n = h([Y_n; V_n]) + η_n
    /// (η ≡ 0 without [`with_obs_noise`](Self::with_obs_noise)); refreshed
    /// by [`forward`].
    y_obs: f64,
    /// Observation-noise draws, one per step.
    noise: NoiseSampler,
}

impl SpringSystem {
    /// Build the system from physical and time parameters and the initial
    /// data (Y₀, V₀).
    ///
    /// The state y holds the *displacements* of the masses from their rest
    /// positions (the springs enter through K y with K = (a/ℓ)·tridiag(−1, 2, −1),
    /// K_NN = a/ℓ, i.e. they have zero natural length in these coordinates):
    /// the equilibrium is Y = 0, V = 0, and a mass sits at iℓ + y_i.
    ///
    /// * `n`  — number of masses / springs
    /// * `rho`    — linear density ρ
    /// * `a`      — elastic modulus
    /// * `dt`      — time step
    /// * `y0`, `v0` — initial displacements and velocities (n entries each)
    /// * `obs`    — observation function h: state slice [Y; V] ↦ scalar
    pub fn new(
        n: usize,
        rho: f64,
        a: f64,
        dt: f64,
        y0: DVector<f64>,
        v0: DVector<f64>,
        obs: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
    ) -> Self {
        assert_eq!(y0.len(), n, "y0 must have n = {n} entries");
        assert_eq!(v0.len(), n, "v0 must have n = {n} entries");
        let ell = 1.0 / n as f64;
        let m = rho * ell;
        let k = a / ell;

        // ── Stiffness matrix K = (a/ℓ) tridiag(−1, 2, −1) ───────────────────
        let mut big_k = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            big_k[(i, i)] = if i == n - 1 { k } else { 2.0 * k };
            if i > 0 {
                big_k[(i, i - 1)] = -k;
            }
            if i < n - 1 {
                big_k[(i, i + 1)] = -k;
            }
        }

        // ── Midpoint matrices  A = I + (h²/4m) K,  B = I − (h²/4m) K ────────
        let id = DMatrix::<f64>::identity(n, n);
        let coeff = dt * dt / (4.0 * m);
        let mat_a = &id + coeff * &big_k;
        let mat_b = &id - coeff * &big_k;
        let lu = mat_a.lu();

        // ── Transition matrix T ───────────────────────────────────────────────
        //
        //        ┌  A⁻¹B                   h A⁻¹              ┐
        //   T =  │                                              │
        //        └  −(h/2m) K (I + A⁻¹B)  I − (h²/2m) K A⁻¹  ┘
        //
        let a_inv_b = lu.solve(&mat_b).expect("singular A");
        let a_inv = lu.solve(&id).expect("singular A");

        let top_left = a_inv_b.clone();
        let top_right = dt * &a_inv;
        let bot_left = -(dt / (2.0 * m)) * &big_k * (&id + &a_inv_b);
        let bot_right = &id - (2.0 * coeff) * &big_k * &a_inv; // h²/2m = 2·coeff

        let mut trans = DMatrix::<f64>::zeros(2 * n, 2 * n);
        trans.view_mut((0, 0), (n, n)).copy_from(&top_left);
        trans.view_mut((0, n), (n, n)).copy_from(&top_right);
        trans.view_mut((n, 0), (n, n)).copy_from(&bot_left);
        trans.view_mut((n, n), (n, n)).copy_from(&bot_right);

        let trans_inv = trans.clone().try_inverse().expect("T is not invertible");

        // ── Initial conditions ────────────────────────────────────────────────
        let state0 = DVector::from_iterator(2 * n, y0.iter().chain(v0.iter()).copied());
        let y_obs = obs(state0.as_slice());

        SpringSystem {
            dt,
            trans,
            trans_inv,
            states: vec![state0],
            lu,
            mat_b,
            big_k,
            m,
            obs: Box::new(obs),
            y_obs,
            noise: NoiseModel::None.sampler(dt, 0),
        }
    }

    /// Add observation noise: from now on the cached observation is
    /// y_n = h(x_n) + η_n with η drawn from `noise` (deterministic in
    /// `seed`). Re-caches the current observation with the first draw, so
    /// call this right after construction.
    pub fn with_obs_noise(mut self, noise: NoiseModel, seed: u64) -> Self {
        self.noise = noise.sampler(self.dt, seed);
        let state = self.states.last().expect("states holds the initial condition");
        self.y_obs = (self.obs)(state.as_slice()) + self.noise.next_sample();
        self
    }

    /// Forward operator: advance the current state one implicit-midpoint
    /// step, t^n → t^{n+1}.
    ///
    /// The current state is the last entry of [`states`]; the new state is
    /// appended and becomes the current one.
    pub fn forward(&mut self) {
        let state = self.states.last().expect("states holds the initial condition");
        let n = self.big_k.nrows();
        let y = state.rows(0, n);
        let v = state.rows(n, n);
        let rhs = &self.mat_b * y + self.dt * v;
        let y_next = self.lu.solve(&rhs).expect("singular system");
        let v_next = v - (self.dt / (2.0 * self.m)) * &self.big_k * (y + &y_next);
        let next = DVector::from_iterator(2 * n, y_next.iter().chain(v_next.iter()).copied());
        self.y_obs = (self.obs)(next.as_slice()) + self.noise.next_sample();
        self.states.push(next);
    }

    /// Squared discrepancy |y_n − h(xi)|² between the current observation
    /// y_n = h([Y_n; V_n]) (the last entry of [`states`]) and the
    /// observation of an arbitrary input state.
    pub fn discrepancy(&self, xi: &[f64]) -> f64 {
        let d = self.y_obs - (self.obs)(xi);
        d * d
    }
}

/// [`Model`] interface for the Mortensen filter. The dimension check
/// (M = 2N) happens at runtime, in `MortensenFilter::new`.
impl<const M: usize> Model<M> for SpringSystem {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn forward(&mut self) {
        SpringSystem::forward(self);
    }

    fn discrepancy(&self, xi: &[f64]) -> f64 {
        SpringSystem::discrepancy(self, xi)
    }

    /// φ⁻¹(xi) = T⁻¹ · xi (the discrete flow is linear).
    fn flow_inv(&self, xi: [f64; M]) -> [f64; M] {
        let zi = &self.trans_inv * DVector::from_column_slice(&xi);
        std::array::from_fn(|d| zi[d])
    }

    /// φ(xi) = T · xi.
    fn flow(&self, xi: [f64; M]) -> [f64; M] {
        let zi = &self.trans * DVector::from_column_slice(&xi);
        std::array::from_fn(|d| zi[d])
    }

    /// The flow is linear: its Jacobian is T itself.
    fn flow_jacobian(&self, _xi: [f64; M]) -> [[f64; M]; M] {
        std::array::from_fn(|r| std::array::from_fn(|c| self.trans[(r, c)]))
    }

    fn n_obs(&self) -> usize {
        1
    }

    fn h(&self, xi: &[f64]) -> DVector<f64> {
        DVector::from_element(1, (self.obs)(xi))
    }

    fn y_obs(&self) -> DVector<f64> {
        DVector::from_element(1, self.y_obs)
    }

    /// The observation is an opaque closure, so its gradient is taken by
    /// central finite differences — exact (to round-off) for the linear
    /// observations the spring is used with (a component, a sum of
    /// components).
    fn obs_jacobian(&self, xi: &[f64]) -> DMatrix<f64> {
        let eps = 1e-6;
        let mut xp = xi.to_vec();
        let mut xm = xi.to_vec();
        DMatrix::from_fn(1, M, |_, c| {
            xp[c] = xi[c] + eps;
            xm[c] = xi[c] - eps;
            let g = ((self.obs)(&xp) - (self.obs)(&xm)) / (2.0 * eps);
            xp[c] = xi[c];
            xm[c] = xi[c];
            g
        })
    }

    /// The flow is the constant transition matrix T: autonomous.
    fn is_autonomous(&self) -> bool {
        true
    }

    fn states(&self) -> &[DVector<f64>] {
        &self.states
    }

    fn state_labels(&self) -> Vec<String> {
        let n = self.states[0].len() / 2;
        (1..=n)
            .map(|i| format!("y_{i}"))
            .chain((1..=n).map(|i| format!("v_{i}")))
            .collect()
    }
}
