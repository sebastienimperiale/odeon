//! Spring chain with an *unknown mass* — the joint state–parameter
//! estimation variant of [`super::spring`], and the simplest
//! **conditionally linear** models of the library (N = 1, 2, 3 masses).
//!
//! N bodies on a line, body 1 attached to a wall by a spring and each body
//! to the next by a further spring, all of stiffness k; y₁..y_N are the
//! displacements from rest (the rest configuration is y = 0, where every
//! spring is unstretched). The bodies 1..N−1 have the known mass m, the
//! free-end body N has the unknown mass
//!
//!   m_N(θ) = m₀ · 2^θ,      dθ/dt = 0,
//!
//! with m₀ the prior value ([`SpringMassParams::m0`]) and θ the
//! log-parameter to estimate, in doublings of m₀ (exactly as μ = μ₀·2^θ in
//! [`super::kepler_mu`]: positive for every real θ, log-normal prior). For
//! N = 1 the single body, attached to the wall, is the unknown one. The
//! augmented state is x = (y₁..y_N, v₁..v_N, θ) and the dynamics
//!
//!   ẏ = v,   m_i v̇_i = −(K y)_i,   θ̇ = 0,   K = k·tridiag(−1, 2, −1) with K_NN = k.
//!
//! **Given θ the dynamics are linear** in (y, v): this is the textbook
//! conditionally linear-Gaussian model, for which the hybrid grid–Gaussian
//! closure (grid in θ, Gaussian in (y, v)) is *exact* — the deterministic
//! Rao–Blackwellization of `docs/hybrid_closure.tex`. The discrete flow
//! inherits the structure: at fixed θ the Gauss–Legendre step is a linear
//! map T(θ) of (y, v) (a rational approximation of exp(dt·A(θ))), as the
//! test `flow_is_conditionally_linear` pins.
//!
//! θ is identifiable from a single position: it sets the normal-mode
//! frequencies (for N = 1, ω = √(k/m(θ))). As for Kepler-μ, a large θ-error
//! makes the likelihood in θ multimodal after a few periods (frequency
//! aliasing), where the single tracker fails and the grid in θ is needed.
//!
//! Conserved quantities along the flow: θ (exactly — its stage equation
//! decouples), and the energy H = ½Σ m_i v_i² + ½ yᵀKy, a quadratic
//! invariant at fixed θ, which the Gauss–Legendre scheme conserves
//! *exactly* (to the Newton tolerance).
//!
//! Time stepping: the shared Gauss–Legendre 4 scheme ([`crate::gl4`]) at
//! M = 2N + 1, by the library's convention for every nonlinear model (the θ
//! dependence is nonlinear); Newton converges in one iteration since the
//! stage equations are linear in (y, v) at the frozen θ. The `Model<M>`
//! implementations are instantiated for N = 1, 2, 3 (M = 3, 5, 7).

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use crate::model::Model;
use nalgebra::{DMatrix, DVector};
use std::f64::consts::LN_2;

/// Physical parameters. [`Default`]: m = 1, m₀ = 1, k = 1.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpringMassParams {
    /// Known mass of the bodies 1..N−1 (unused for N = 1).
    pub m: f64,
    /// Prior mass of the free-end body N; the true value is m₀·2^θ.
    pub m0: f64,
    /// Stiffness of every spring.
    pub k: f64,
}

impl Default for SpringMassParams {
    fn default() -> Self {
        SpringMassParams {
            m: 1.0,
            m0: 1.0,
            k: 1.0,
        }
    }
}

impl SpringMassParams {
    /// m_N(θ) = m₀ · 2^θ (> 0 for every real θ).
    pub fn m_end(&self, theta: f64) -> f64 {
        self.m0 * theta.exp2()
    }
}

/// Spring chain of `N` bodies with the augmented state
/// (y₁..y_N, v₁..v_N, θ), m_N = m₀·2^θ; state dimension 2N + 1.
///
/// # Example
/// ```
/// use ode_models::models::spring_mass::SpringMassSystem;
///
/// // Two bodies, free end displaced, true mass m₂ = 2^0.3·m₀, observing y₁:
/// let x0 = SpringMassSystem::<2>::state([0.0, 0.3], [0.0, 0.0], 0.3);
/// let sys = SpringMassSystem::<2>::new(0.01, 0);
/// let x1 = sys.flow(&x0); // one step: t^0 → t^1 (θ stays 0.3 exactly)
/// ```
pub struct SpringMassSystem<const N: usize> {
    // ── Public ───────────────────────────────────────────────────────────────
    pub dt: f64,
    /// Physical parameters (m, m₀, k).
    pub params: SpringMassParams,
    /// Index (0-based) of the observed position y_{obs+1}.
    pub obs: usize,
}

impl<const N: usize> SpringMassSystem<N> {
    /// State dimension 2N + 1.
    pub const DIM: usize = 2 * N + 1;

    /// Build the system with the default physical parameters
    /// ([`SpringMassParams::default`]).
    ///
    /// The state is (y₁..y_N, v₁..v_N, θ), 2N + 1 entries; the θ component
    /// of a reference trajectory is the *true* log-parameter.
    ///
    /// * `dt`  — time step
    /// * `obs` — index (0-based) of the observed position
    pub fn new(dt: f64, obs: usize) -> Self {
        Self::with_params(SpringMassParams::default(), dt, obs)
    }

    /// Build the system from explicit physical parameters (see
    /// [`new`](Self::new) for the other arguments).
    pub fn with_params(params: SpringMassParams, dt: f64, obs: usize) -> Self {
        assert!(N >= 1, "the chain needs at least one body");
        assert!(obs < N, "observed position index {obs} out of range (N = {N})");
        assert!(params.m > 0.0 && params.m0 > 0.0 && params.k > 0.0, "masses and stiffness must be positive");
        SpringMassSystem { dt, params, obs }
    }

    /// Observation h(x) = y_{obs+1}.
    pub fn obs(&self, x: &[f64]) -> f64 {
        x[self.obs]
    }

    /// Squared discrepancy |y − h(xi)|².
    pub fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        let d = y[0] - self.obs(xi);
        d * d
    }

    /// Mass of body i (0-based): m for i < N−1, m₀·2^θ for the free end.
    fn mass(&self, i: usize, theta: f64) -> f64 {
        if i + 1 == N {
            self.params.m_end(theta)
        } else {
            self.params.m
        }
    }

    /// (K y)_i for the chain stiffness K = k·tridiag(−1, 2, −1), K_NN = k:
    /// the spring to the wall or previous body, plus the spring to the next
    /// body if any.
    fn ky(&self, y: &[f64], i: usize) -> f64 {
        let k = self.params.k;
        let prev = if i == 0 { 0.0 } else { y[i - 1] };
        let mut f = k * (y[i] - prev);
        if i + 1 < N {
            f += k * (y[i] - y[i + 1]);
        }
        f
    }

    /// Energy H(x) = ½Σ m_i v_i² + ½k[y₁² + Σ(y_{i+1} − y_i)²], conserved
    /// along the flow (θ is constant on trajectories).
    pub fn energy(&self, x: &[f64]) -> f64 {
        let theta = x[2 * N];
        let (y, v) = (&x[..N], &x[N..2 * N]);
        let kin: f64 = (0..N).map(|i| 0.5 * self.mass(i, theta) * v[i] * v[i]).sum();
        let mut pot = 0.5 * self.params.k * y[0] * y[0];
        for i in 1..N {
            pot += 0.5 * self.params.k * (y[i] - y[i - 1]) * (y[i] - y[i - 1]);
        }
        kin + pot
    }

    /// Right-hand side, for the const dimension M = 2N + 1 of the stepper.
    fn rhs<const M: usize>(&self, x: &[f64; M]) -> [f64; M] {
        let theta = x[2 * N];
        let mut f = [0.0; M];
        for i in 0..N {
            f[i] = x[N + i];
            f[N + i] = -self.ky(&x[..N], i) / self.mass(i, theta);
        }
        f
    }

    /// Analytic Jacobian ∂rhs/∂x: the linear (y, v) block at the current
    /// masses, the θ-column ∂v̇_N/∂θ = ln2·(Ky)_N/m_N (since
    /// d(1/m_N)/dθ = −ln2/m_N), and a zero θ-row.
    fn rhs_jacobian<const M: usize>(&self, x: &[f64; M]) -> [[f64; M]; M] {
        let theta = x[2 * N];
        let k = self.params.k;
        let mut j = [[0.0; M]; M];
        for i in 0..N {
            j[i][N + i] = 1.0;
            let mi = self.mass(i, theta);
            // −K_{i,·}/m_i
            j[N + i][i] = -(if i + 1 < N { 2.0 * k } else { k }) / mi;
            if i > 0 {
                j[N + i][i - 1] = k / mi;
            }
            if i + 1 < N {
                j[N + i][i + 1] = k / mi;
            }
        }
        let m_end = self.params.m_end(theta);
        j[2 * N - 1][2 * N] = LN_2 * self.ky(&x[..N], N - 1) / m_end;
        j
    }
}

/// The per-dimension part: stepper calls and the [`Model`] impl, for the
/// pairs (N, M = 2N + 1) the library instantiates.
macro_rules! impl_spring_mass {
    ($n:literal, $m:literal) => {
        impl SpringMassSystem<$n> {
            /// The augmented state (y, v, θ) as a fixed-size array.
            pub fn state(y: [f64; $n], v: [f64; $n], theta: f64) -> [f64; $m] {
                let mut x = [0.0; $m];
                x[..$n].copy_from_slice(&y);
                x[$n..2 * $n].copy_from_slice(&v);
                x[2 * $n] = theta;
                x
            }

            /// One step of the fourth-order Gauss–Legendre method (shared
            /// solver, see [`crate::gl4`]).
            fn gl4_step(&self, x: &[f64; $m], h: f64) -> [f64; $m] {
                crate::gl4::gl4_step(|x| self.rhs(x), |x| self.rhs_jacobian(x), x, h)
            }

            /// Discrete flow map φ: one Gauss–Legendre step of size `dt`.
            pub fn flow(&self, x: &[f64; $m]) -> [f64; $m] {
                self.gl4_step(x, self.dt)
            }

            /// φ(x) together with its exact Jacobian ∂φ/∂x, including the
            /// ∂φ/∂θ sensitivity column the tracker needs.
            pub fn flow_with_jacobian(&self, x: &[f64; $m]) -> ([f64; $m], [[f64; $m]; $m]) {
                crate::gl4::gl4_step_with_jacobian(|x| self.rhs(x), |x| self.rhs_jacobian(x), x, self.dt)
            }

            /// Inverse discrete flow map φ⁻¹: one Gauss–Legendre step of
            /// size `−dt` (exact inverse: the scheme is symmetric).
            pub fn flow_inv(&self, x: &[f64; $m]) -> [f64; $m] {
                self.gl4_step(x, -self.dt)
            }

            /// The linear map T(θ) of (y, v) realized by the discrete flow at
            /// the frozen log-parameter θ — the conditionally linear
            /// structure made explicit (columns = images of the basis
            /// vectors), 2N × 2N.
            pub fn transition(&self, theta: f64) -> DMatrix<f64> {
                DMatrix::from_fn(2 * $n, 2 * $n, |r, c| {
                    let mut e = [0.0; $m];
                    e[c] = 1.0;
                    e[2 * $n] = theta;
                    self.flow(&e)[r]
                })
            }
        }

        impl Model<$m> for SpringMassSystem<$n> {
            fn dt(&self) -> f64 {
                self.dt
            }

            fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
                SpringMassSystem::<$n>::discrepancy(self, y, xi)
            }

            fn flow_inv(&self, xi: [f64; $m]) -> [f64; $m] {
                SpringMassSystem::<$n>::flow_inv(self, &xi)
            }

            fn flow(&self, xi: [f64; $m]) -> [f64; $m] {
                SpringMassSystem::<$n>::flow(self, &xi)
            }

            fn flow_jacobian(&self, xi: [f64; $m]) -> [[f64; $m]; $m] {
                self.flow_with_jacobian(&xi).1
            }

            fn obs_dim(&self) -> usize {
                1
            }

            fn obs(&self, xi: &[f64]) -> DVector<f64> {
                DVector::from_element(1, SpringMassSystem::<$n>::obs(self, xi))
            }

            /// ∇h is the unit vector of the observed position (zero
            /// θ-component).
            fn obs_jacobian(&self, _xi: &[f64]) -> DMatrix<f64> {
                let mut g = [0.0; $m];
                g[self.obs] = 1.0;
                DMatrix::from_row_slice(1, $m, &g)
            }

            /// Time-independent flow: autonomous (so the transport plan applies).
            fn is_autonomous(&self) -> bool {
                true
            }

            fn state_labels(&self) -> Vec<String> {
                (1..=$n)
                    .map(|i| format!("y_{i}"))
                    .chain((1..=$n).map(|i| format!("v_{i}")))
                    .chain(std::iter::once("θ".to_string()))
                    .collect()
            }
        }
    };
}

impl_spring_mass!(1, 3);
impl_spring_mass!(2, 5);
impl_spring_mass!(3, 7);

#[cfg(test)]
mod tests {
    use super::*;

    /// Deliberately non-unit parameters and θ ≠ 0, so every parameter
    /// enters the checks.
    const ODD_PARAMS: SpringMassParams = SpringMassParams {
        m: 0.7,
        m0: 1.3,
        k: 2.1,
    };
    const ODD_THETA: f64 = -0.45;

    /// The energy is a quadratic invariant at fixed θ, so Gauss–Legendre
    /// conserves it to the Newton tolerance; θ itself is exactly constant.
    /// Checked for N = 1 and N = 2.
    #[test]
    fn energy_is_conserved_exactly_and_theta_is_constant() {
        let x0 = SpringMassSystem::<2>::state([0.2, -0.4], [0.1, 0.3], ODD_THETA);
        let sys = SpringMassSystem::<2>::with_params(ODD_PARAMS, 0.05, 0);
        let e0 = sys.energy(&x0);
        let mut x = x0;
        for _ in 0..400 {
            x = sys.flow(&x);
            assert!((sys.energy(&x) - e0).abs() < 1e-10 * e0, "N=2 energy drift");
            assert_eq!(x[4], ODD_THETA, "θ moved along the flow");
        }
        let x0 = SpringMassSystem::<1>::state([0.3], [-0.2], ODD_THETA);
        let sys = SpringMassSystem::<1>::with_params(ODD_PARAMS, 0.05, 0);
        let e0 = sys.energy(&x0);
        let mut x = x0;
        for _ in 0..400 {
            x = sys.flow(&x);
            assert!((sys.energy(&x) - e0).abs() < 1e-10 * e0, "N=1 energy drift");
            assert_eq!(x[2], ODD_THETA);
        }
    }

    /// The analytic Jacobian — in particular the θ-column with its ln 2
    /// factor — against central finite differences of the rhs (N = 2 and
    /// N = 3, where the interior body exercises both couplings).
    #[test]
    fn jacobian_matches_finite_differences() {
        fn check<const N: usize, const M: usize>(sys: &SpringMassSystem<N>, x: [f64; M]) {
            let jac = sys.rhs_jacobian(&x);
            let eps = 1e-6;
            for col in 0..M {
                let mut xp = x;
                let mut xm = x;
                xp[col] += eps;
                xm[col] -= eps;
                let (fp, fm) = (sys.rhs(&xp), sys.rhs(&xm));
                for row in 0..M {
                    let fd = (fp[row] - fm[row]) / (2.0 * eps);
                    assert!(
                        (jac[row][col] - fd).abs() < 1e-7 * (1.0 + fd.abs()),
                        "N={N}: J[{row}][{col}] = {} but finite difference gives {fd}",
                        jac[row][col]
                    );
                }
            }
        }
        check(
            &SpringMassSystem::<2>::with_params(ODD_PARAMS, 0.01, 1),
            [0.6, -0.35, 0.8, -1.9, 0.45],
        );
        check(
            &SpringMassSystem::<3>::with_params(ODD_PARAMS, 0.01, 2),
            [0.6, -0.35, 0.2, 0.8, -1.9, 0.4, 0.45],
        );
    }

    #[test]
    fn flow_inv_inverts_flow() {
        let x0 = SpringMassSystem::<2>::state([0.3, -0.2], [0.0, 0.5], 0.3);
        let sys = SpringMassSystem::<2>::new(0.02, 0);
        let z = sys.flow_inv(&sys.flow(&x0));
        for i in 0..5 {
            assert!((z[i] - x0[i]).abs() < 1e-11, "round-trip error at {i}: {}", z[i] - x0[i]);
        }
    }

    /// Conditionally linear structure of the discrete flow: at a frozen θ
    /// the (y, v) part of φ is linear, so φ(αx + βx') = αφ(x) + βφ(x') on
    /// the (y, v) components, and `transition(θ)` reproduces φ.
    #[test]
    fn flow_is_conditionally_linear() {
        let theta = 0.35;
        let sys = SpringMassSystem::<2>::with_params(ODD_PARAMS, 0.03, 0);
        let x = SpringMassSystem::<2>::state([0.4, -0.1], [0.2, 0.7], theta);
        let xp = SpringMassSystem::<2>::state([-0.3, 0.5], [-0.6, 0.1], theta);
        let (a, b) = (1.7, -0.4);
        let mut comb = [0.0; 5];
        for i in 0..4 {
            comb[i] = a * x[i] + b * xp[i];
        }
        comb[4] = theta;
        let (fx, fxp, fc) = (sys.flow(&x), sys.flow(&xp), sys.flow(&comb));
        let t = sys.transition(theta);
        for i in 0..4 {
            assert!((fc[i] - (a * fx[i] + b * fxp[i])).abs() < 1e-12, "flow not linear in (y, v) at fixed θ");
            let tx: f64 = (0..4).map(|j| t[(i, j)] * x[j]).sum();
            assert!((tx - fx[i]).abs() < 1e-12, "transition(θ) disagrees with flow");
        }
        assert_eq!(fc[4], theta);
    }

}
