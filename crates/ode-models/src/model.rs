//! The [`Model`] trait — the contract between the Mortensen filter and a
//! forward model. Implementations live in [`crate::models`].

use nalgebra::{DMatrix, DVector};

/// Interface the Mortensen filter needs from a forward model.
///
/// `M` is the state-space dimension, fixed at compile time because the FFT
/// solver works on `[f64; M]` grid points.
pub trait Model<const M: usize> {


    // pub fn forward(
    //     &mut self,
    //      T0: f64,  -> The state is advanced from T0
    //     T1: f64,  -> to T1
    //     state: &DVector<f64>,
    // )   -> Note the argument  state: &DVector<f64> in the forward, implies that the function     fn states(&self) -> &[DVector<f64>]; must be removed


    /// Time step of the model; also weights the observation update and sets
    /// the diffusion step of the filter.
    fn dt(&self) -> f64; // dt intern au model, c'est une limite superieure, lors du forward un nombre de pas fixé est chosii de telle sorte que le dt correspondant soit plus petit que la limite superieure

    /// Forward operator: advance the reference state one step, t^n → t^{n+1}.
    fn forward(&mut self);

    /// Discrepancy d(y_n,h(xi)) between the current observation
    /// y_n = h(current state) and the observation of an arbitrary state xi.
    fn discrepancy(&self, xi: &[f64]) -> f64;

    /// Inverse discrete flow map φ_n⁻¹ (one step backward), used by the
    /// transport step.
    fn flow_inv(&self, xi: [f64; M]) -> [f64; M];

    // ── Tracker interface (second-order / Gaussian closure, see
    //    `crate::tracker`): the forward map with its exact tangent, and the
    //    observation split into h, y_n and ∇h instead of the scalar
    //    discrepancy the grid filter consumes. ────────────────────────────

    /// Discrete flow map φ_n (one step forward) of an arbitrary state.
    fn flow(&self, xi: [f64; M]) -> [f64; M];

    /// Exact Jacobian Φ = ∂φ_n/∂x of the discrete flow at `xi` (row i =
    /// gradient of component i), analytic — for the Gauss–Legendre models
    /// from the tangent of the stage system, for the linear spring the
    /// transition matrix itself.
    fn flow_jacobian(&self, xi: [f64; M]) -> [[f64; M]; M];

    /// Dimension of the observation vector h(x).
    fn n_obs(&self) -> usize;

    /// Observation operator h(xi) (length [`n_obs`](Self::n_obs)).
    fn h(&self, xi: &[f64]) -> DVector<f64>;

    /// The current observation y_n = h(current state) + η_n.
    fn y_obs(&self) -> DVector<f64>;

    /// Jacobian ∇h(xi), `n_obs × M`, analytic.
    fn obs_jacobian(&self, xi: &[f64]) -> DMatrix<f64>;

    /// Innovation y_n − h(xi). Provided as the plain difference; models with
    /// angular observations override it with the shortest angular
    /// difference (the same geometry their [`discrepancy`](Self::discrepancy)
    /// uses).
    fn innovation(&self, xi: &[f64]) -> DVector<f64> {
        self.y_obs() - self.h(xi)
    }

    /// Whether the model is autonomous: `true` iff the discrete flow map φ
    /// (and hence [`flow_inv`](Self::flow_inv)) does not depend on time, i.e.
    /// is the same at every step. An autonomous flow allows the transport
    /// step to be precomputed once and reused across iterations.
    fn is_autonomous(&self) -> bool;

    /// Reference trajectory so far: `states()[n]` is the flat state vector at
    /// step n, in the same component order as [`flow_inv`](Self::flow_inv) and
    /// [`state_labels`](Self::state_labels); the last entry is the current
    /// state.
    fn states(&self) -> &[DVector<f64>]; // A degager

    /// Human-readable labels of the state components, in flat state order
    /// (e.g. `["y_1", "v_1"]` or `["q_1", "q_2", "p_1", "p_2"]`).
    fn state_labels(&self) -> Vec<String>;
}



pub trait ModelEstimator<const M: usize> {


    // pub fn get_init_free_state(&self) -> &DVector<f64> {
    //     &self.init_free_state
    // }

    // pub fn get_init_uq_state(&self) -> &DVector<f64> {
    //     &self.init_uq_state
    // }

    // pub fn get_inv_cov_uq_state(&self) -> &DMatrix<f64> {
    //     &self.inv_cov_uq_state
    // }

    // pub fn get_n_free_state(&self) -> usize {
    //     self.n_free_state
    // }

    // pub fn get_n_uq_state(&self) -> usize {
    //     self.n_uq_state
    // }

    // let innovations = or.get_innovations(time_iter as usize, &free_state);

    // let new_state = merge_free_and_uq_states(&free_state, &uq_state);


    // // generate inv cov observ matrix
    // // For most model it should be a sparse matrix
    // // for us it just a diagonal but we play along Akilles API
    // let mut inv_cov_observation = CooMatrix::<f64>::zeros(n_observation, n_observation);
    // for i in 0..n_observation {
    //     inv_cov_observation.push(i, i, 1000 as f64);
    // }


    //     /// Discrepancy d(y_n,h(xi)) between the current observation
    // /// y_n = h(current state) and the observation of an arbitrary state xi.
    // fn discrepancy(&self, xi: &[f64]) -> f64;
}