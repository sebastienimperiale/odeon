//! The [`Model`] trait — the contract between a forward model and the
//! observers. Implementations live in [`crate::models`].
//!
//! A model is a *definition*, not a simulation: it holds its physical
//! parameters, its time step and its observation choice, and exposes pure
//! functions of a state — the discrete flow and its inverse and tangent,
//! the observation operator and its Jacobian, the discrepancy between an
//! observation and a state. It owns no state: the reference trajectory of
//! a twin experiment and the observation sequence live in a `Reference`
//! (`ode_models_spec::reference`), produced by its `twin` generator from
//! the same maps or coming from anywhere else, and the observers receive the
//! observation of each step as an argument. (Until 2026-09-14 a model
//! carried its trajectory and a cached noisy observation; the observers
//! called `forward()` on it.)
//!
//! `M` is the state-space dimension, fixed at compile time because the
//! grid solver works on `[f64; M]` points.

use nalgebra::{DMatrix, DVector};

/// Interface the observers need from a forward model.
pub trait Model<const M: usize> {
    /// Time step of the discrete flow; also weights the observation update
    /// and sets the diffusion step of the filter.
    fn dt(&self) -> f64;

    /// State dimension of this instance: `M`, except for models
    /// implementing the trait for every `M` and checked at runtime (the
    /// spring chain: 2N).
    fn dim(&self) -> usize {
        M
    }


    /// Which components live on a circle, and on which interval: `None` for
    /// a plain component, `Some((lo, hi))` for one defined modulo hi − lo
    /// (an angle on [−π, π) for the pendulums). The maps `flow`/`flow_inv`
    /// stay smooth on the covering space (they never wrap); the reference
    /// generator stores every state with these components wrapped into
    /// their interval, so a winding reference stays inside a periodic box,
    /// and the estimators' default domains use the same interval. All
    /// `None` by default.
    fn periodic(&self) -> [Option<(f64, f64)>; M] {
        [None; M]
    }



    /// Discrete flow map φ (one step forward) of an arbitrary state.
    fn flow(&self, x: [f64; M]) -> [f64; M];

    /// Inverse discrete flow map φ⁻¹ (one step backward), used by the
    /// transport step of the grid filter.
    fn flow_inv(&self, x: [f64; M]) -> [f64; M];

    /// Exact Jacobian Φ = ∂φ/∂x of the discrete flow at `x` (row i =
    /// gradient of component i), analytic — for the Gauss–Legendre models
    /// from the tangent of the stage system, for the linear spring the
    /// transition matrix itself.
    fn flow_jacobian(&self, x: [f64; M]) -> [[f64; M]; M];

    /// Whether the model is autonomous: `true` iff the discrete flow map φ
    /// (and hence [`flow_inv`](Self::flow_inv)) does not depend on time,
    /// i.e. is the same at every step. An autonomous flow allows the
    /// transport step to be precomputed once and reused across iterations.
    fn is_autonomous(&self) -> bool;

    /// Dimension of the observation vector h(x).
    fn obs_dim(&self) -> usize;

    /// Observation operator h(x) (length [`obs_dim`](Self::obs_dim)).
    fn obs(&self, x: &[f64]) -> DVector<f64>;

    /// Jacobian ∇h(x), `n_obs × M`, analytic.
    fn obs_jacobian(&self, x: &[f64]) -> DMatrix<f64>;

    /// Squared discrepancy d(y, h(x))² between an observation `y` (length
    /// [`obs_dim`](Self::obs_dim)) and the observation of the state `x` — the
    /// scalar the grid filter multiplies p by exp(−ε⁻¹·dt·γ·d²/2) with,
    /// evaluated at every grid point. The squared Euclidean distance by
    /// default; models with angular observations override it with the
    /// squared shortest angular difference. Implementations avoid
    /// allocating (this is the observation step's inner loop).
    fn discrepancy(&self, y: &[f64], x: &[f64]) -> f64 {
        let hx = self.obs(x);
        let mut d2 = 0.0;
        for (yi, hi) in y.iter().zip(hx.iter()) {
            let d = yi - hi;
            d2 += d * d;
        }
        d2
    }

    /// Innovation y ⊖ h(x) (length [`obs_dim`](Self::obs_dim)), the signed
    /// difference in the observation's own geometry — the plain difference
    /// by default; models with angular observations override it with the
    /// shortest angular difference (the same geometry their
    /// [`discrepancy`](Self::discrepancy) uses).
    fn innovation(&self, y: &[f64], x: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(y) - self.obs(x)
    }

    /// Human-readable labels of the state components, in flat state order
    /// (e.g. `["y_1", "v_1"]` or `["q_1", "q_2", "p_1", "p_2"]`).
    fn state_labels(&self) -> Vec<String>;
}
