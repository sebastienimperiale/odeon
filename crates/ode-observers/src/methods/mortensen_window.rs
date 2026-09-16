//! Translating window ("box tracker"): the grid filter on a small box that
//! follows the mode of p.
//!
//! The ansatz is p(x, t) = ρ(ξ, t) with ξ = x − x̂(t): ρ lives on a fixed
//! grid in ξ (the *window*, a box [−L, L] per direction, Dirichlet walls:
//! p ≡ 0 outside), and x̂(t) is the position of the window in state space.
//! For any x̂(t) this is an exact change of variables — the linear equation
//! for p becomes, in the window,
//!
//!   ∂τ ρ + (f(x̂ + ξ) − ẋ̂)·∇ρ = (ε/2) Σ_d q_d ∂²_d ρ − (γ/2ε) d(y, h(x̂ + ξ))² ρ,
//!
//! the same drift/diffusion/misfit structure with the flow and the
//! observation read at the physical point x̂ + ξ (docs/density_filter.tex,
//! §3). Its only approximation is the truncation of ρ to the box, which is
//! harmless as long as the mode stays inside.
//!
//! This implementation keeps the frame *piecewise constant*: during one
//! filter step the window does not move (so the −ẋ̂ drift vanishes and the
//! step is exactly [`MortensenFilter::forward`] on the box, with the model's
//! flow and observation composed with the shift — the [`Shifted`] adapter),
//! and after the step the window is re-centred on the mode **by whole
//! elements**: the element containing the node argmax of ρ becomes the
//! central element. A shift by whole elements maps Gauss–Lobatto nodes onto
//! nodes, so the re-centring is an exact index translation
//! ([`MortensenFilter::shift_by_elements`]) — no interpolation, no accuracy
//! loss, and no equation for x̂ to integrate: the gauge "the mode is at the
//! origin" is enforced up to one element, which is all the truncation
//! needs. The estimate is x̂ + argmax ρ, at node accuracy.
//!
//! What the window requires, beyond the fixed box: the per-step
//! displacement of the mode, dt·|f(x̂) + (γ/ε) C⁻¹Hᵀr|, must stay well
//! below the element size h, so that the mode never leaves the central
//! element's neighbours between two re-centrings; and the box half-widths
//! L_d must be a few widths of p, √(ε (S⁻¹)_dd), chosen a priori (the
//! curvature is not estimated here). The grid is smaller by orders of
//! magnitude than a fixed box covering the trajectory.
//!
//! **Transport plan for an autonomous flow** (`pre_compute_flow_inv`): the
//! preimage φ⁻¹(x) of a *physical* point never changes, and a shift by
//! whole elements maps the window's nodes onto nodes. So the tracker
//! caches φ⁻¹(x̂ + ξ_i) for every node in physical coordinates, installs
//! them (minus x̂) as the filter's transport plan
//! ([`MortensenFilter::set_transport_preimages`]), and when the window
//! moves by k elements it translates the cache by the same index shift as
//! the field, evaluates φ⁻¹ only at the nodes that entered the box, and
//! re-installs the plan. Between moves the transport is the pure
//! gather + contraction of the fixed box; a move costs the plan rebuild
//! (localization, no flow) plus Newton solves on the entering elements
//! only — a fraction ≈ |k_d|/n_el,d of the nodes per moved direction.

use super::Observer;
use crate::methods::mortensen::{DiffusionScheme, FilterParams, MortensenFilter, shifted_source};
use ode_models::model::Model;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// A model seen from a window centred at `offset`: every state-dependent
/// quantity is evaluated at the physical point x̂ + ξ, and the flow is
/// expressed in window coordinates, φ_ξ(ξ) = φ(x̂ + ξ) − x̂ (same for φ⁻¹).
/// The observations and the time step are the inner model's, untouched.
/// [`Model::is_autonomous`] is `false`: the window map changes whenever the
/// offset does.
pub struct Shifted<Mod> {
    /// The physical model.
    pub inner: Mod,
    /// Window position x̂ (physical coordinates of the window's origin).
    offset: Vec<f64>,
}

impl<Mod> Shifted<Mod> {
    /// Wrap `inner` in a window at `offset`.
    pub fn new(inner: Mod, offset: &[f64]) -> Self {
        Shifted {
            inner,
            offset: offset.to_vec(),
        }
    }

    /// Current window position x̂.
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    /// Move the window to x̂ = `offset`.
    pub fn set_offset(&mut self, offset: &[f64]) {
        self.offset.copy_from_slice(offset);
    }

    /// Physical point of the window coordinate ξ.
    fn physical(&self, xi: &[f64]) -> Vec<f64> {
        xi.iter().zip(&self.offset).map(|(a, b)| a + b).collect()
    }
}

impl<const M: usize, Mod: Model<M>> Model<M> for Shifted<Mod> {
    fn dt(&self) -> f64 {
        self.inner.dt()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn discrepancy(&self, y: &[f64], xi: &[f64]) -> f64 {
        let mut x = [0.0; M];
        for d in 0..M {
            x[d] = xi[d] + self.offset[d];
        }
        self.inner.discrepancy(y, &x)
    }

    fn flow_inv(&self, xi: [f64; M]) -> [f64; M] {
        let x: [f64; M] = std::array::from_fn(|d| xi[d] + self.offset[d]);
        let y = self.inner.flow_inv(x);
        std::array::from_fn(|d| y[d] - self.offset[d])
    }

    fn flow(&self, xi: [f64; M]) -> [f64; M] {
        let x: [f64; M] = std::array::from_fn(|d| xi[d] + self.offset[d]);
        let y = self.inner.flow(x);
        std::array::from_fn(|d| y[d] - self.offset[d])
    }

    fn flow_jacobian(&self, xi: [f64; M]) -> [[f64; M]; M] {
        self.inner
            .flow_jacobian(std::array::from_fn(|d| xi[d] + self.offset[d]))
    }

    fn obs_dim(&self) -> usize {
        self.inner.obs_dim()
    }

    fn obs(&self, xi: &[f64]) -> DVector<f64> {
        self.inner.obs(&self.physical(xi))
    }

    fn obs_jacobian(&self, xi: &[f64]) -> DMatrix<f64> {
        self.inner.obs_jacobian(&self.physical(xi))
    }

    fn innovation(&self, y: &[f64], xi: &[f64]) -> DVector<f64> {
        self.inner.innovation(y, &self.physical(xi))
    }

    /// The inner periodic intervals, translated to window coordinates.
    fn periodic(&self) -> [Option<(f64, f64)>; M] {
        let p = self.inner.periodic();
        std::array::from_fn(|d| p[d].map(|(lo, hi)| (lo - self.offset[d], hi - self.offset[d])))
    }

    fn is_autonomous(&self) -> bool {
        false
    }

    fn state_labels(&self) -> Vec<String> {
        self.inner.state_labels()
    }
}

/// Numerical parameters of the [`BoxTracker`]. The window is the box
/// ∏_d [−L_d, L_d] in ξ = x − x̂, always with Dirichlet walls (p ≡ 0
/// outside); the remaining fields are those of [`FilterParams`].
/// Deliberately no `Default`.
#[derive(Clone, Copy, Debug)]
pub struct BoxTrackerParams<const M: usize> {
    /// Half-width L_d of the window per direction: a few widths of p,
    /// √(ε (S⁻¹)_dd), where S is the expected curvature of V at the mode.
    pub half_width: [f64; M],
    /// ε — diffusion / temperature parameter.
    pub eps: f64,
    /// γ — observation weight (see [`FilterParams::gamma`]).
    pub gamma: f64,
    /// Elements per direction (odd numbers put ξ = 0 at the centre of the
    /// central element; at least 3 so the central element has neighbours).
    pub n_el: [usize; M],
    /// Polynomial order per direction.
    pub p_ord: [usize; M],
    /// Diffusion scheme (see [`DiffusionScheme`]).
    pub diffusion: DiffusionScheme,
    /// Diagonal of the model-noise covariance (see [`FilterParams::q_diag`]).
    pub q_diag: [f64; M],
    /// Cache φ⁻¹ of the window's physical nodes and run the transport on a
    /// plan, refreshing only the entering elements when the window moves
    /// (see the module doc). Requires an autonomous model
    /// ([`BoxTracker::new`] panics otherwise); `false` evaluates φ⁻¹ at
    /// every node at every step (`convect`).
    pub pre_compute_flow_inv: bool,
}

impl<const M: usize> BoxTrackerParams<M> {
    /// The [`FilterParams`] of the window: box (−L, L), Dirichlet in every
    /// direction, no periodic direction, no precomputed transport plan.
    pub fn filter_params(&self) -> FilterParams<M> {
        assert!(
            (0..M).all(|d| self.half_width[d] > 0.0 && self.n_el[d] >= 1),
            "half_width must be positive and n_el ≥ 1: {:?} / {:?}",
            self.half_width,
            self.n_el
        );
        FilterParams {
            domain: std::array::from_fn(|d| (-self.half_width[d], self.half_width[d])),
            eps: self.eps,
            gamma: self.gamma,
            n_el: self.n_el,
            p_ord: self.p_ord,
            diffusion: self.diffusion,
            q_diag: self.q_diag,
            periodic: [false; M],
            dirichlet: [true; M],
            pre_compute_flow_inv: false,
        }
    }
}

/// The translating-window filter: a [`MortensenFilter`] on the window,
/// driving a [`Shifted`] model, re-centred by whole elements after each
/// step.
pub struct BoxTracker<const M: usize, Mod: Model<M> + Sync> {
    /// The grid filter on the window (its model is the shifted one; the
    /// physical model is `filter.model.inner`).
    pub filter: MortensenFilter<M, Shifted<Mod>>,
    /// Parameters the window was built with.
    pub params: BoxTrackerParams<M>,
    /// Window position x̂ (physical coordinates of ξ = 0).
    center: [f64; M],
    /// Element size h_d of the window.
    h: [f64; M],
    /// Index of the element containing ξ = 0 (⌊n_el/2⌋).
    home: [usize; M],
    /// φ⁻¹(x̂ + ξ_i) of every window node, *physical* coordinates (valid
    /// for any x̂: autonomous flow); `Some` iff `pre_compute_flow_inv`.
    preimages: Option<Vec<[f64; M]>>,
}

impl<const M: usize, Mod: Model<M> + Sync> BoxTracker<M, Mod> {
    /// Build the window around a model. The window is placed at the origin
    /// until initialised ([`init_gaussian`](Self::init_gaussian) /
    /// [`init`](Self::init)).
    pub fn new(model: Mod, params: BoxTrackerParams<M>) -> Self {
        assert!(
            !params.pre_compute_flow_inv || model.is_autonomous(),
            "pre_compute_flow_inv requires an autonomous model: \
             the window's preimage cache assumes φ⁻¹ is time-independent"
        );
        let center = [0.0; M];
        let filter = MortensenFilter::<M, _>::new(
            Shifted::new(model, &center),
            params.filter_params(),
        );
        let h = filter.element_size();
        let home = std::array::from_fn(|d| params.n_el[d] / 2);
        let preimages = params.pre_compute_flow_inv.then(Vec::new);
        BoxTracker {
            filter,
            params,
            center,
            h,
            home,
            preimages,
        }
    }

    /// Place the window at `center` and set p_0 = exp(−ε⁻¹ V_0) from a
    /// value function V_0 given in *physical* coordinates.
    pub fn init(&mut self, center: [f64; M], v0: impl Fn(&[f64]) -> f64 + Sync) {
        self.set_center(center);
        let c = center;
        self.filter.init_filter(|xi| {
            let mut x = [0.0; M];
            for d in 0..M {
                x[d] = xi[d] + c[d];
            }
            v0(&x)
        });
    }

    /// Place the window at `center` (arbitrary move: the whole preimage
    /// cache is recomputed).
    fn set_center(&mut self, center: [f64; M]) {
        self.center = center;
        self.filter.model.set_offset(&center);
        if self.preimages.is_some() {
            let model = &self.filter.model.inner;
            let pre: Vec<[f64; M]> = self
                .filter
                .grid()
                .par_iter()
                .map(|xi| model.flow_inv(std::array::from_fn(|d| center[d] + xi[d])))
                .collect();
            self.install_preimages(pre);
        }
    }

    /// Make `pre` (physical preimages of the window's nodes) the cache and
    /// the filter's transport plan (in window coordinates).
    fn install_preimages(&mut self, pre: Vec<[f64; M]>) {
        let c = self.center;
        let local: Vec<[f64; M]> = pre
            .par_iter()
            .map(|y| std::array::from_fn(|d| y[d] - c[d]))
            .collect();
        self.filter.set_transport_preimages(&local);
        self.preimages = Some(pre);
    }

    /// Window position x̂ (physical coordinates of ξ = 0).
    pub fn center(&self) -> [f64; M] {
        self.center
    }

    /// Element size of the window per direction.
    pub fn element_size(&self) -> [f64; M] {
        self.h
    }

    /// The physical model.
    pub fn model(&self) -> &Mod {
        &self.filter.model.inner
    }

    /// State estimate x̂ + argmax ρ (node accuracy, see
    /// [`MortensenFilter::argmax_p`]).
    pub fn estimate(&self) -> [f64; M] {
        let xi = self.filter.argmax_p();
        std::array::from_fn(|d| self.center[d] + xi[d])
    }

    /// Mode of ρ in window coordinates (node accuracy). After
    /// [`forward`](Self::forward) it lies in the central element.
    pub fn mode_offset(&self) -> [f64; M] {
        self.filter.argmax_p()
    }

    /// One step, given the observation `y` of this step: the filter cycle
    /// on the window (observation and flow read at x̂ + ξ, transport in the
    /// frozen frame, diffusion), then the re-centring by whole elements.
    /// Returns the element shift applied ([`Observer::forward`] is the
    /// same step without it).
    pub fn step(&mut self, y: &[f64]) -> [isize; M] {
        self.filter.forward(y);
        self.recentre()
    }

    /// Move the window so that the element containing the node argmax of ρ
    /// becomes the central element: x̂ ← x̂ + k·h, ρ shifted by −k elements
    /// (exact index translation). Returns k.
    pub fn recentre(&mut self) -> [isize; M] {
        let xi = self.filter.argmax_p();
        let n = self.params.n_el;
        let k: [isize; M] = std::array::from_fn(|d| {
            let e = ((xi[d] + self.params.half_width[d]) / self.h[d]).floor();
            let e = (e.max(0.0) as usize).min(n[d] - 1);
            e as isize - self.home[d] as isize
        });
        if k.iter().any(|&kd| kd != 0) {
            self.filter.shift_by_elements(k);
            let c: [f64; M] = std::array::from_fn(|d| self.center[d] + k[d] as f64 * self.h[d]);
            match self.preimages.take() {
                None => self.set_center(c),
                Some(old) => {
                    // Same index translation as the field: node i of the
                    // moved window is node i + k·p_ord of the old one, whose
                    // physical preimage is already known; the nodes that
                    // entered the box are evaluated afresh.
                    self.center = c;
                    self.filter.model.set_offset(&c);
                    let ndof = self.filter.params.ndof();
                    let step: [isize; M] =
                        std::array::from_fn(|d| k[d] * self.params.p_ord[d] as isize);
                    let model = &self.filter.model.inner;
                    let grid = self.filter.grid();
                    let pre: Vec<[f64; M]> = (0..grid.len())
                        .into_par_iter()
                        .map(|i| match shifted_source(&ndof, &step, i) {
                            Some(j) => old[j],
                            None => model.flow_inv(std::array::from_fn(|d| c[d] + grid[i][d])),
                        })
                        .collect();
                    self.install_preimages(pre);
                }
            }
        }
        k
    }
}

/// The common observer surface: the Gaussian prior places the window at
/// its centre; estimate = x̂ + argmax ρ.
impl<const M: usize, Mod: Model<M> + Sync> Observer<M> for BoxTracker<M, Mod> {
    fn dt(&self) -> f64 {
        self.filter.model.dt()
    }

    /// Place the window at `center` with the Gaussian prior
    /// V_0 = Σ_d σ_d (x_d − center_d)²/2, i.e. p_0 centred on the window.
    fn init_gaussian(&mut self, center: [f64; M], sigma: [f64; M]) {
        self.set_center(center);
        self.filter.init_filter_gaussian(sigma, [0.0; M]);
    }

    fn forward(&mut self, y: &[f64]) {
        self.step(y);
    }

    fn estimate(&self) -> [f64; M] {
        BoxTracker::estimate(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::kalman::{MortensenTracker, TrackerParams};
    use ode_models::models::SpringSystem;
    use ode_models_spec::noise::NoiseModel;
    use ode_models_spec::progress::Progress;
    use ode_models_spec::reference::Reference;

    fn spring(dt: f64) -> SpringSystem {
        SpringSystem::new(1, 1.0, 1.0, dt, |x: &[f64]| x[0])
    }

    /// The noiseless reference of the spring from (1.15, 0).
    fn reference(dt: f64, steps: usize) -> Reference {
        Reference::twin(&spring(dt), [1.15, 0.0], steps, NoiseModel::None, 0, None, &Progress::default())
    }

    /// The window on the linear spring: a box of half-width 1 (the
    /// reference swings between ±1.15, so a fixed box of that size would
    /// lose it) follows the tracker — the Kalman filter — to node accuracy
    /// over half a period, the window moves, and the mode never leaves the
    /// central element's neighbours.
    #[test]
    fn window_follows_the_tracker_on_linear_model() {
        let dt = 0.01;
        let q = [1.0; 2];
        let sigma = [20.0; 2];
        let x_c = [1.0, 0.0];
        let params = BoxTrackerParams {
            half_width: [1.0; 2],
            eps: 0.05,
            gamma: 1.0,
            n_el: [5; 2],
            p_ord: [4; 2],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: q,
            pre_compute_flow_inv: true,
        };
        let mut window = BoxTracker::<2, _>::new(spring(dt), params);
        window.init_gaussian(x_c, sigma);
        let mut tracker =
            MortensenTracker::<2, _>::new(spring(dt), TrackerParams { q_diag: q, gamma: 1.0 });
        tracker.init(x_c, sigma);

        // Node spacing: h = 0.4, p_ord 4 ⇒ ≲ 0.14 at the element ends.
        let tol = 0.15;
        let h = window.element_size();
        let mut moved = false;
        let r = reference(dt, 300);
        for (step, y) in r.observations[..300].iter().enumerate() {
            let k = window.step(y);
            tracker.forward(y);
            moved |= k.iter().any(|&kd| kd != 0);
            let (a, b) = (window.estimate(), tracker.estimate());
            for d in 0..2 {
                assert!(
                    (a[d] - b[d]).abs() < tol,
                    "step {step}: window estimate {a:?} vs tracker {b:?}"
                );
            }
            // After re-centring the mode is in the central element.
            let xi = window.mode_offset();
            for d in 0..2 {
                assert!(xi[d].abs() <= h[d] + 1e-12, "mode {xi:?} left the central element");
            }
            // The shift per step never exceeds one element (dt·|f| ≪ h).
            assert!(k.iter().all(|kd| kd.abs() <= 1), "shift {k:?} > 1 element");
        }
        assert!(moved, "the window never moved");
        let c = window.center();
        assert!(c[0] < 0.0, "after half a period the window should be at y < 0: {c:?}");
    }

    /// The plan path (preimage cache, shifted with the window and completed
    /// on the entering elements) and the convect path (φ⁻¹ at every node
    /// every step) are the same algorithm: run in lockstep on the spring,
    /// with several window moves, their fields and positions agree to
    /// round-off at every step.
    #[test]
    fn preimage_cache_matches_the_convect_path() {
        let dt = 0.01;
        let params = |pre| BoxTrackerParams {
            half_width: [0.7; 2],
            eps: 0.05,
            gamma: 1.0,
            n_el: [5; 2],
            p_ord: [4; 2],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: [1.0; 2],
            pre_compute_flow_inv: pre,
        };
        let mut a = BoxTracker::<2, _>::new(spring(dt), params(true));
        let mut b = BoxTracker::<2, _>::new(spring(dt), params(false));
        a.init_gaussian([1.0, 0.0], [20.0; 2]);
        b.init_gaussian([1.0, 0.0], [20.0; 2]);
        let mut moves = 0;
        let r = reference(dt, 250);
        for (step, y) in r.observations[..250].iter().enumerate() {
            let ka = a.step(y);
            let kb = b.step(y);
            assert_eq!(ka, kb, "step {step}: shifts differ");
            moves += usize::from(ka.iter().any(|&k| k != 0));
            assert_eq!(a.center(), b.center(), "step {step}: centres differ");
            let (pa, pb) = (a.filter.p_field(), b.filter.p_field());
            let scale = pa.iter().cloned().fold(0.0, f64::max).max(1e-300);
            let diff = pa.iter().zip(pb).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            assert!(diff <= 1e-10 * scale, "step {step}: fields differ by {diff:.2e} (max {scale:.2e})");
        }
        assert!(moves >= 2, "the window should have moved several times ({moves})");
    }

    /// With the observations off the window follows the pure flow of its
    /// initial condition (mode transported, no pull): the estimate stays on
    /// the discrete flow of x_c to node accuracy while the window travels.
    #[test]
    fn gamma_zero_follows_the_flow() {
        let dt = 0.01;
        let x0 = [0.9, -0.3];
        let params = BoxTrackerParams {
            half_width: [0.8; 2],
            eps: 0.05,
            gamma: 0.0,
            n_el: [5; 2],
            p_ord: [4; 2],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: [1.0; 2],
            pre_compute_flow_inv: false,
        };
        let mut window = BoxTracker::<2, _>::new(spring(dt), params);
        window.init_gaussian(x0, [20.0; 2]);
        let mut flow = x0;
        let r = reference(dt, 200);
        for y in &r.observations[..200] {
            window.forward(y);
            flow = window.model().flow(flow);
            let e = window.estimate();
            for d in 0..2 {
                assert!((e[d] - flow[d]).abs() < 0.15, "estimate {e:?} left the flow {flow:?}");
            }
        }
    }
}
