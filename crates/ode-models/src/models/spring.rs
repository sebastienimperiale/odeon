//! N-harmonic oscillator — the linear spring chain.
//!
//! Encapsulates the physical/time parameters, the transition matrix T of
//! the implicit-midpoint step and its inverse, and the observation
//! function; the discrete flow is x ↦ T x. (Until 2026-09-14 the system
//! also carried its trajectory and a cached observation; a reference is
//! now a `Reference` of `ode-models-spec`.)

use crate::model::Model;
use nalgebra::{DMatrix, DVector};

/// Solver for the N-spring forward problem.
///
/// # Example
/// ```
/// use ode_models::models::SpringSystem;
///
/// use ode_models::model::Model;
///
/// let sys = SpringSystem::new(2, 1.0, 1.0, 0.005, |x: &[f64]| x.iter().sum());
/// let x1 = Model::<4>::flow(&sys, [0.5, 1.0, 0.0, 0.0]); // one step: [Y; V] ↦ T [Y; V]
/// println!("{:.6}", sys.trans);     // transition matrix
/// println!("{:.6}", sys.trans_inv); // its inverse
/// ```
pub struct SpringSystem {
    // ── Public ───────────────────────────────────────────────────────────────
    pub dt: f64,
    /// Transition matrix T (2N×2N): [Y_{n+1}; V_{n+1}] = T [Y_n; V_n]
    pub trans: DMatrix<f64>,
    /// Inverse transition matrix T⁻¹
    pub trans_inv: DMatrix<f64>,
    /// Number of masses N (state dimension 2N).
    pub n: usize,

    // ── Private ──────────────────────────────────────────────────────────────
    /// Observation function h: state slice [Y; V] ↦ scalar observation.
    obs: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
}

impl SpringSystem {
    /// Build the system from physical and time parameters.
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
    /// * `obs`    — observation function h: state slice [Y; V] ↦ scalar
    pub fn new(
        n: usize,
        rho: f64,
        a: f64,
        dt: f64,
        obs: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
    ) -> Self {
        assert!(n >= 1, "the chain needs at least one mass");
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

        SpringSystem {
            dt,
            trans,
            trans_inv,
            n,
            obs: Box::new(obs),
        }
    }

    /// Squared discrepancy |y − h(xi)|² between an observation `y` (one
    /// component) and the observation of an arbitrary input state.
    pub fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        let d = y[0] - (self.obs)(xi);
        d * d
    }
}

/// [`Model`] interface for the Mortensen filter. The dimension check
/// (M = 2N) happens at runtime, in `MortensenFilter::new`.
impl<const M: usize> Model<M> for SpringSystem {
    fn dt(&self) -> f64 {
        self.dt
    }

    fn dim(&self) -> usize {
        2 * self.n
    }

    fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        SpringSystem::discrepancy(self, y, xi)
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

    fn obs_dim(&self) -> usize {
        1
    }

    fn obs(&self, xi: &[f64]) -> DVector<f64> {
        DVector::from_element(1, (self.obs)(xi))
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

    fn state_labels(&self) -> Vec<String> {
        let n = self.n;
        (1..=n)
            .map(|i| format!("y_{i}"))
            .chain((1..=n).map(|i| format!("v_{i}")))
            .collect()
    }
}
