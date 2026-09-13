//! Mortensen filter on p = exp(−V/ε).
//!
//! One filter iteration ([`MortensenFilter::forward`]) runs the cycle
//!
//!   1. observation:  p ← p · exp( −ε⁻¹ · dt · γ · d(y_n, h(xi))² / 2 )
//!                    (d = distance in observation space: Euclidean, or the
//!                    angular distance for a bearing — `Model::discrepancy`;
//!                    γ = observation weight [`FilterParams::gamma`], 1 by
//!                    default)
//!   2. model forward: reference state t^n → t^{n+1}
//!   3. transport:     p(xi) ← p( φ⁻¹(xi) )
//!   4. diffusion:     ∂τ p = (ε/2) Σ_d q_d ∂²_d p over one time step
//!                     (q = diag of the model-noise covariance; q ≡ 1 is Δ)
//!
//! discretized on a tensor-product Gauss–Lobatto grid; the diffusion step is
//! solved by dense fast diagonalisation of the per-direction spectral-element
//! operators (`lobatto-spectral`; the FFT-across-elements alternative of
//! `lobatto-fft` is kept in `benches/diffusion.rs` for comparison).
//! The diffusion step is, by default, the *directionally split* implicit
//! Euler scheme ([`DiffusionScheme::SplitEuler`]), chosen for nodal
//! positivity of p.

use serde::{Deserialize, Serialize};
use ode_models::model::Model;
use crate::output::print_progress;
use lobatto_grid::EvalPlan;
use lobatto_spectral::solver::{BoundaryCondition, PoissonND};
use rayon::prelude::*;

/// Time-stepping scheme of the diffusion step ∂τ p = (ε/2) Δp, taken as a
/// single implicit-Euler step of length δτ = dt per filter iteration.
///
/// There is deliberately no sub-stepping: the nodal positivity of the split
/// scheme holds for steps *large enough* relative to the local mesh size
/// squared, so shrinking the step would work against it. In both cases the
/// step costs one solver application (forward transforms, per-mode symbol,
/// inverse transforms): the split scheme is *not* more expensive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffusionScheme {
    /// Directionally split implicit Euler (product splitting of the tensor
    /// Laplacian into its 1D factors):
    ///
    ///   p_new = ∏_d (I + dt·(ε/2) M_d⁻¹K_d)⁻¹ p_old,
    ///
    /// where M_d, K_d are the 1D Gauss–Lobatto mass and stiffness matrices
    /// of direction d. The 1D factors commute (they act on different tensor
    /// factors), so the product is exact and evaluated in one solver
    /// application (`PoissonND::euler_split_step_in_place`). Each 1D factor
    /// (I + dt·(ε/2) M_d⁻¹K_d)⁻¹ satisfies the one-dimensional discrete
    /// maximum principle for dt·ε/2 large enough relative to the local mesh
    /// size squared, hence maps nonnegative nodal vectors to nonnegative
    /// nodal vectors — **nodal positivity** of p = exp(−V/ε), which the
    /// unsplit step does not guarantee. Splitting error is O(dt²), i.e. of
    /// the same order as the implicit-Euler error itself.
    SplitEuler,
    /// Unsplit implicit Euler on the full tensor Laplacian:
    ///
    ///   (I + dt·(ε/2) M⁻¹K) p_new = p_old,   K = Σ_d K_d ⊗ (mass in the others).
    ///
    /// The historical scheme (`PoissonND::euler_step_in_place`); kept for
    /// comparisons. May produce (small) negative nodal values of p.
    Euler,
}

/// Numerical parameters of the [`MortensenFilter`], passed to
/// [`MortensenFilter::new`] and kept on the filter as `filter.params`.
///
/// All fields are public and there is deliberately no `Default`: every run
/// states its parameters explicitly (reproducibility over convenience).
#[derive(Clone, Copy, Debug)]
pub struct FilterParams<const M: usize> {
    /// Per-direction domain intervals (left, right) on which the filter
    /// discretizes p. A filter concern, not a model property: choose it from
    /// the state ranges of the reference trajectory (the `*_forward`
    /// examples print them) plus margin for the density spread.
    pub domain: [(f64, f64); M],
    /// ε — diffusion / temperature parameter.
    pub eps: f64,
    /// γ — observation weight: the observation step multiplies p by
    /// exp(−ε⁻¹·dt·γ·d²/2), i.e. γ scales the impact of the observations on
    /// the density (equivalently, an observation-noise covariance R = I/γ).
    /// 1 is the standard cycle; 0 switches the observations off. Must be ≥ 0.
    pub gamma: f64,
    /// Elements per direction.
    pub n_el: [usize; M],
    /// Polynomial order per direction.
    pub p_ord: [usize; M],
    /// Time-stepping scheme of the diffusion step, see [`DiffusionScheme`]
    /// (`SplitEuler` for nodal positivity of p).
    pub diffusion: DiffusionScheme,
    /// Diagonal of the model-noise covariance Q = diag(q_0, …, q_{M−1}):
    /// the diffusion step is ∂τ p = (ε/2) Σ_d q_d ∂²_d p, i.e. diagonally
    /// anisotropic. `[1.0; M]` is the isotropic Laplacian; q_d = 0 switches
    /// diffusion off in direction d (e.g. noise entering only through the
    /// momenta of a mechanical system). Must be ≥ 0.
    pub q_diag: [f64; M],
    /// Per-direction periodicity: `true` marks a periodic direction (the
    /// domain interval is identified end-to-end, e.g. an angle on (−π, π));
    /// `false` keeps Neumann boundary conditions (or Dirichlet, see
    /// [`dirichlet`](Self::dirichlet)). Affects the boundary
    /// conditions of the diffusion solver, the folding of the transport step
    /// (periodic wrap instead of even reflection), and the DOF count
    /// ([`ndof`](Self::ndof)).
    pub periodic: [bool; M],
    /// Per-direction homogeneous Dirichlet option: `true` imposes p = 0 on
    /// both endpoints of direction d — V → +∞ at the wall, an *absorbing* /
    /// confining boundary (the state is asserted to be strictly inside the
    /// box; mass reaching the wall is removed) — instead of the reflecting
    /// Neumann default. Semantics: **p ≡ 0 outside the box**. The diffusion
    /// solver enforces it spectrally (sine basis; the two boundary nodes
    /// are zeros and not stored, ndof = n_el·p_ord − 1), and the transport
    /// step enforces it explicitly: a grid point whose preimage φ⁻¹(ξ)
    /// leaves the box in a Dirichlet direction receives exactly p = 0 — the
    /// grid's odd folding (−p at the reflected point) is deliberately *not*
    /// used for transport. Invalid on a periodic direction
    /// ([`MortensenFilter::new`] panics).
    pub dirichlet: [bool; M],
    /// Precompute the whole transport step's position-dependent work once at
    /// construction: an [`EvalPlan`] built by `convect_plan` holds φ⁻¹ of every
    /// grid point *and* its element localization, basis coordinates and DOF
    /// gather tables, so each iteration's transport is a pure gather +
    /// contraction (`eval_with_plan`) instead of `convect`. Requires an
    /// autonomous model ([`Model::is_autonomous`]); [`MortensenFilter::new`]
    /// panics otherwise. Trades memory (~(8 + 8·M) bytes per DOF plus one
    /// gather table per occupied element) for skipping both the per-DOF flow
    /// evaluation and the per-DOF element search / basis setup at every
    /// iteration.
    pub pre_compute_flow_inv: bool,
}

impl<const M: usize> FilterParams<M> {
    /// DOFs of direction d: n_el[d]·p_ord[d] + 1 under Neumann boundary
    /// conditions (both endpoints stored), n_el[d]·p_ord[d] when the
    /// direction is periodic (right endpoint identified with the left),
    /// n_el[d]·p_ord[d] − 1 under Dirichlet (both endpoints are zeros, not
    /// stored).
    pub fn ndof(&self) -> [usize; M] {
        std::array::from_fn(|d| {
            let np = self.n_el[d] * self.p_ord[d];
            if self.periodic[d] {
                np
            } else if self.dirichlet[d] {
                np - 1
            } else {
                np + 1
            }
        })
    }
}

/// Source index of node `i` under a translation of the frame by
/// `step[d]` nodes in direction d (direction 0 fastest): the node whose
/// multi-index is that of `i` plus `step`, or `None` if it falls outside
/// the grid. Shared by the field shift and the window's preimage cache.
pub(crate) fn shifted_source<const M: usize>(
    ndof: &[usize; M],
    step: &[isize; M],
    i: usize,
) -> Option<usize> {
    let mut rem = i;
    let mut j = 0usize;
    let mut stride = 1usize;
    for d in 0..M {
        let jd = (rem % ndof[d]) as isize + step[d];
        rem /= ndof[d];
        if jd < 0 || jd >= ndof[d] as isize {
            return None;
        }
        j += jd as usize * stride;
        stride *= ndof[d];
    }
    Some(j)
}

/// Is `y` outside the box in some Dirichlet direction? p ≡ 0 there: the
/// transport step zeroes grid points whose preimage satisfies this.
fn outside_dirichlet<const M: usize>(params: &FilterParams<M>, y: &[f64; M]) -> bool {
    (0..M).any(|d| {
        params.dirichlet[d] && (y[d] < params.domain[d].0 || y[d] > params.domain[d].1)
    })
}

/// Mortensen filter loop driver; owns the forward model and the spectral
/// diffusion solver.
///
/// # Example
/// ```
/// use ode_observers::filter::{DiffusionScheme, FilterParams, MortensenFilter};
/// use ode_models::models::SpringSystem;
/// use nalgebra::DVector;
///
/// let sys = SpringSystem::new(1, 1.0, 1.0, 0.005,
///     DVector::from_row_slice(&[0.5]), DVector::zeros(1),
///     |x: &[f64]| x[0]);
/// let mut filter = MortensenFilter::<2, _>::new(sys, FilterParams {
///     domain: [(-3.0, 3.0); 2],
///     eps: 0.01,
///     gamma: 1.0,
///     n_el: [4; 2],
///     p_ord: [2; 2],
///     diffusion: DiffusionScheme::SplitEuler,
///     q_diag: [1.0; 2],
///     periodic: [false; 2],
///     dirichlet: [false; 2],
///     pre_compute_flow_inv: false,
/// });
/// filter.init_filter_quadratic(10.0);
/// filter.forward();
/// ```
pub struct MortensenFilter<const M: usize, Mod: Model<M> + Sync> {
    // ── Public ───────────────────────────────────────────────────────────────
    /// The forward model; its `states` hold the reference trajectory so far.
    pub model: Mod,
    /// Numerical parameters the filter was built with (see [`FilterParams`]).
    pub params: FilterParams<M>,

    // ── Private ──────────────────────────────────────────────────────────────
    /// Spectral diffusion solver (grid, per-direction generalized
    /// eigendecompositions K_d V = M_d V diag(μ_d), dense per direction —
    /// `lobatto-spectral`); the operator coefficients are passed at each
    /// application. Works on real nodal vectors.
    solver: PoissonND<M>,
    /// Tensor-product Gauss–Lobatto grid (direction 0 varies fastest).
    grid: Vec<[f64; M]>,
    /// Precomputed transport plan (φ⁻¹ of every grid point + interpolation
    /// localization, see `convect_plan`); `Some` iff
    /// `params.pre_compute_flow_inv` (autonomous flow: valid at every step).
    transport_plan: Option<EvalPlan<M>>,
    /// Dirichlet outside mask: grid indices whose preimage φ⁻¹(ξ) leaves
    /// the box in a Dirichlet direction — p ≡ 0 outside, so the transport
    /// step zeroes them explicitly (the grid's odd folding would give
    /// −p(reflected) instead). `Some` iff a direction is Dirichlet *and*
    /// the transport plan is precomputed (autonomous flow: the mask is
    /// fixed); on the `convect` path it is recomputed each step.
    dirichlet_mask: Option<Vec<usize>>,
    /// p on the grid (p = exp(−V/ε) is real); empty until [`init_filter`] /
    /// [`init_filter_quadratic`] (the grid is never empty, so emptiness ⇔ not
    /// initialized).
    p_field: Vec<f64>,
    /// Output buffer of the transport step, consumed in place by the
    /// diffusion step (which then swaps it with `p_field`). Sized once in
    /// [`init_filter`] alongside `p_field`, reused afterwards (no
    /// per-iteration allocation on our side).
    p_buf: Vec<f64>,
}

impl<const M: usize, Mod: Model<M> + Sync> MortensenFilter<M, Mod> {
    /// Build the filter around a model.
    ///
    /// The grid covers `params.domain` with, per direction, Neumann or
    /// periodic boundary conditions (`params.periodic`). The diffusion step
    /// ∂τ p = (ε/2) Δp over δτ = dt is one implicit-Euler step of the scheme
    /// `params.diffusion` (see [`DiffusionScheme`]).
    ///
    /// * `model`  — forward model (taken by value; access it via `self.model`)
    /// * `params` — numerical parameters, see [`FilterParams`]
    pub fn new(model: Mod, params: FilterParams<M>) -> Self {
        assert_eq!(
            model.states().last().map(|s| s.len()),
            Some(M),
            "filter dimension M = {M} does not match the model's state dimension"
        );
        assert!(
            params.q_diag.iter().all(|&q| q >= 0.0),
            "q_diag must be nonnegative, got {:?}",
            params.q_diag
        );
        assert!(
            params.gamma >= 0.0,
            "gamma must be nonnegative, got {}",
            params.gamma
        );
        assert!(
            (0..M).all(|d| !(params.periodic[d] && params.dirichlet[d])),
            "a direction cannot be periodic and Dirichlet at once: periodic = {:?}, dirichlet = {:?}",
            params.periodic,
            params.dirichlet
        );
        let domain = params.domain;
        let solver = PoissonND::<M>::new(
            std::array::from_fn(|d| domain[d].1 - domain[d].0), // ls  : lengths
            std::array::from_fn(|d| domain[d].0),               // xls : left boundaries
            params.n_el,
            params.p_ord,
            std::array::from_fn(|d| {
                if params.periodic[d] {
                    BoundaryCondition::Periodic
                } else if params.dirichlet[d] {
                    BoundaryCondition::Dirichlet
                } else {
                    BoundaryCondition::Neumann
                }
            }),
        );
        let grid = solver.get_x();

        let transport_plan = if params.pre_compute_flow_inv {
            assert!(
                model.is_autonomous(),
                "pre_compute_flow_inv requires an autonomous model: \
                 a time-dependent flow cannot be precomputed"
            );
            Some(solver.grid().convect_plan(|xi| model.flow_inv(xi)))
        } else {
            None
        };

        // With an autonomous (precomputed) flow the Dirichlet outside mask
        // is fixed: build it once, one φ⁻¹ evaluation per grid point.
        let dirichlet_mask = if params.dirichlet.iter().any(|&d| d) && transport_plan.is_some() {
            Some(
                grid.par_iter()
                    .enumerate()
                    .filter(|(_, xi)| outside_dirichlet(&params, &model.flow_inv(**xi)))
                    .map(|(i, _)| i)
                    .collect(),
            )
        } else {
            None
        };

        MortensenFilter {
            model,
            params,
            solver,
            grid,
            transport_plan,
            dirichlet_mask,
            p_field: Vec::new(),
            p_buf: Vec::new(),
        }
    }

    /// The tensor-product Gauss–Lobatto grid (direction 0 varies fastest).
    pub fn grid(&self) -> &[[f64; M]] {
        &self.grid
    }

    /// p on the grid.
    ///
    /// Panics if called before [`init_filter`] / [`init_filter_quadratic`].
    pub fn p_field(&self) -> &[f64] {
        assert!(
            !self.p_field.is_empty(),
            "filter not initialized: call init_filter(...) or init_filter_quadratic(...) first"
        );
        &self.p_field
    }

    /// Grid point where p is maximal — the filter's state estimate
    /// x̂ = argmax p = argmin V (V = −ε log p).
    ///
    /// Grid-resolution accuracy: the true maximizer is located up to the
    /// local grid spacing (no sub-grid refinement). Ties resolve to an
    /// arbitrary maximizing point.
    ///
    /// Panics if called before [`init_filter`] / [`init_filter_quadratic`].
    pub fn argmax_p(&self) -> [f64; M] {
        let (i, _) = self
            .p_field()
            .par_iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("p_field is non-empty");
        self.grid[i]
    }

    /// Element size h_d = (right − left)/n_el per direction.
    pub fn element_size(&self) -> [f64; M] {
        std::array::from_fn(|d| {
            (self.params.domain[d].1 - self.params.domain[d].0) / self.params.n_el[d] as f64
        })
    }

    /// Translate the field by whole elements — the re-centring step of the
    /// translating window (`box_tracker`): the frame moves by `k[d]`
    /// elements in direction d, so the new field is the old one read
    /// `k[d]·h_d` further,
    ///
    ///   p_new(ξ) = p_old(ξ + k·h),   p_old ≡ 0 outside the box.
    ///
    /// Since the Gauss–Lobatto nodes of every element are the same, a shift
    /// by whole elements maps nodes onto nodes: this is a pure index shift
    /// (by `k[d]·p_ord[d]` nodes), *exact*, no interpolation; nodes whose
    /// source lies outside the box receive 0 (the Dirichlet semantics of the
    /// window). Not meaningful on a periodic direction (panics).
    ///
    /// Panics if called before [`init_filter`] / [`init_filter_quadratic`].
    pub fn shift_by_elements(&mut self, k: [isize; M]) {
        assert!(
            (0..M).all(|d| !self.params.periodic[d] || k[d] == 0),
            "shift_by_elements: cannot shift a periodic direction (k = {k:?})"
        );
        assert!(!self.p_field.is_empty(), "filter not initialized");
        if k.iter().all(|&kd| kd == 0) {
            return;
        }
        let ndof = self.params.ndof();
        let step: [isize; M] = std::array::from_fn(|d| k[d] * self.params.p_ord[d] as isize);
        let src = &self.p_field;
        self.p_buf.par_iter_mut().enumerate().for_each(|(i, out)| {
            *out = shifted_source(&ndof, &step, i).map_or(0.0, |j| src[j]);
        });
        std::mem::swap(&mut self.p_field, &mut self.p_buf);
    }

    /// Install the transport plan from explicit preimages: `preimages[i]` =
    /// φ⁻¹(ξ_i) of grid point i, in the filter's own coordinates (folded
    /// into the box per the extension, as `convect_plan` does), together
    /// with the Dirichlet outside mask derived from them. The transport then
    /// runs on the plan path (`eval_with_plan`) until the plan is replaced
    /// or [`clear_transport_plan`](Self::clear_transport_plan) is called —
    /// the caller is responsible for the preimages being those of the
    /// current flow (the translating window uses this: it caches φ⁻¹ of the
    /// physical grid points of an autonomous flow and re-installs them,
    /// shifted, when the window moves).
    pub fn set_transport_preimages(&mut self, preimages: &[[f64; M]]) {
        assert_eq!(
            preimages.len(),
            self.grid.len(),
            "set_transport_preimages: one preimage per grid point"
        );
        self.transport_plan = Some(self.solver.grid().eval_extended_plan(preimages));
        self.dirichlet_mask = if self.params.dirichlet.iter().any(|&d| d) {
            let params = &self.params;
            Some(
                preimages
                    .par_iter()
                    .enumerate()
                    .filter(|(_, y)| outside_dirichlet(params, y))
                    .map(|(i, _)| i)
                    .collect(),
            )
        } else {
            None
        };
    }

    /// Drop the transport plan (and its Dirichlet mask): the transport falls
    /// back to `convect`, evaluating φ⁻¹ at every step.
    pub fn clear_transport_plan(&mut self) {
        self.transport_plan = None;
        self.dirichlet_mask = None;
    }

    /// Set the initial condition p_0 = exp( −ε⁻¹ V_0(x) ) from a value
    /// function V_0 evaluated at every grid point.
    pub fn init_filter(&mut self, v0: impl Fn(&[f64]) -> f64 + Sync) {
        let eps = self.params.eps;
        self.p_field = self
            .grid
            .par_iter()
            .map(|xi| (-v0(xi) / eps).exp())
            .collect();
        self.p_buf = vec![0.0; self.p_field.len()];
    }

    /// Set the quadratic initial condition V_0(x) = σ·|x|²/2 centered at the
    /// origin (see [`init_filter_quadratic_at`](Self::init_filter_quadratic_at)).
    pub fn init_filter_quadratic(&mut self, sigma: f64) {
        self.init_filter_quadratic_at(sigma, [0.0; M]);
    }

    /// Set the quadratic initial condition V_0(x) = σ·|x − x_c|²/2 (isotropic
    /// case of [`init_filter_gaussian`](Self::init_filter_gaussian)).
    pub fn init_filter_quadratic_at(&mut self, sigma: f64, center: [f64; M]) {
        self.init_filter_gaussian([sigma; M], center);
    }

    /// Set the diagonal Gaussian initial condition
    /// p_0 ∝ ∏_d g_d(x_d), g_d(x) = exp(−σ_d (x − x_{c,d})²/(2ε)), i.e.
    /// V_0(x) = Σ_d σ_d (x_d − x_{c,d})²/2 of mean `center` = x_c and
    /// variance ε/σ_d in direction d; σ_d = 0 leaves direction d flat (no
    /// prior there), σ ≡ 0 gives p_0 ≡ 1.
    ///
    /// On a **periodic direction** the plain Gaussian is not a function on
    /// the circle (its values at the two identified ends differ: a jump at
    /// the seam), so g_d is the *wrapped* Gaussian, the sum over the images
    /// of the centre,
    ///
    ///   g_d(x) = Σ_k exp(−σ_d (x − x_{c,d} + k·L_d)²/(2ε)),   L_d the period,
    ///
    /// smooth and L_d-periodic (the wrapped normal / theta function — also
    /// the heat kernel of the circle, hence what the diffusion step itself
    /// would produce). The sum is truncated where the next image is below
    /// e⁻⁴⁰ relative to the peak: one or two images for a prior narrow
    /// against the period, more for a wide one, which tends to the flat
    /// prior. For a centre in the middle of a wide box the difference from
    /// the plain Gaussian is exponentially small; near the seam it is the
    /// difference between a continuous prior and a jump.
    pub fn init_filter_gaussian(&mut self, sigma: [f64; M], center: [f64; M]) {
        let eps = self.params.eps;
        let periodic = self.params.periodic;
        let period: [f64; M] =
            std::array::from_fn(|d| self.params.domain[d].1 - self.params.domain[d].0);
        // Images needed so that the first neglected one is below e⁻⁴⁰:
        // σ (K L)² / (2ε) ≥ 40  ⇔  K ≥ √(80 ε / σ) / L  (capped).
        let n_images: [i64; M] = std::array::from_fn(|d| {
            if periodic[d] && sigma[d] > 0.0 {
                (((80.0 * eps / sigma[d]).sqrt() / period[d]).ceil() as i64 + 1).min(200)
            } else {
                0
            }
        });
        self.init_filter(|x| {
            (0..M)
                .map(|d| {
                    let u = x[d] - center[d];
                    if periodic[d] && sigma[d] > 0.0 {
                        let g: f64 = (-n_images[d]..=n_images[d])
                            .map(|k| {
                                let v = u + k as f64 * period[d];
                                (-sigma[d] * v * v / (2.0 * eps)).exp()
                            })
                            .sum();
                        -eps * g.ln() // V_0 contribution −ε log g_d (g = 0 ⇒ +∞ ⇒ p = 0)
                    } else {
                        sigma[d] * u * u / 2.0
                    }
                })
                .sum::<f64>()
        });
    }

    /// One Mortensen filter iteration, t^n → t^{n+1}:
    /// observation → model forward → transport → diffusion.
    ///
    /// Panics if called before [`init_filter`] / [`init_filter_quadratic`].
    pub fn forward(&mut self) {
        assert!(
            !self.p_field.is_empty(),
            "filter not initialized: call init_filter(...) or init_filter_quadratic(...) first"
        );
        // Step 1 — Observation (with the current reference state).
        self.observe();
        // Step 2 — Forward: evolve the reference state t^n → t^{n+1}.
        self.model.forward();
        // Step 3 — Transport:  p(xi) ← p( φ⁻¹(xi) ), into `p_buf`.
        self.transport();
        // Step 4 — Diffusion:  ∂τ p = (ε/2) Δp over δτ = dt, `p_buf` → `p_field`.
        self.diffuse();
    }

    /// Observation step, in place on `p_field`:
    /// p ← p · exp( −ε⁻¹·dt·γ·d(y_n,h(xi))/2 ), with d(y_n,h(xi)) the model's
    /// discrepancy at the current reference state and γ the observation
    /// weight [`FilterParams::gamma`].
    fn observe(&mut self) {
        let dt = self.model.dt();
        let eps = self.params.eps;
        let gamma = self.params.gamma;
        let model = &self.model;
        self.grid
            .par_iter()
            .zip(self.p_field.par_iter_mut())
            .for_each(|(xi, pi)| {
                let d = model.discrepancy(xi);
                *pi *= (-dt * gamma * d / (2.0 * eps)).exp();
            });
    }

    /// Transport step, `p_field` → `p_buf`: p(xi) ← p( φ⁻¹(xi) ) (points
    /// mapped outside the domain are folded back internally according to the
    /// extension). With `pre_compute_flow_inv` all position-dependent work —
    /// φ⁻¹, folding, element localization, gather tables — is baked into the
    /// transport plan (autonomous flow), so only the gather + contraction
    /// remains per iteration. The result lands in the reused `p_buf`, which
    /// [`diffuse`](Self::diffuse) consumes in place: no per-iteration
    /// allocation here.
    fn transport(&mut self) {
        let model = &self.model;
        let grid = self.solver.grid();
        match &self.transport_plan {
            Some(plan) => grid.eval_with_plan(plan, &self.p_field, &mut self.p_buf),
            None => grid.convect(|xi| model.flow_inv(xi), &self.p_field, &mut self.p_buf),
        }
        // Dirichlet semantics: p ≡ 0 outside the box, so any grid point
        // whose preimage left the box in a Dirichlet direction gets exactly
        // 0 (overwriting the odd-folded −p the interpolation produced).
        if let Some(mask) = &self.dirichlet_mask {
            for &i in mask {
                self.p_buf[i] = 0.0;
            }
        } else if self.params.dirichlet.iter().any(|&d| d) {
            // convect path: the flow may be time-dependent — recompute the
            // preimages (one extra φ⁻¹ evaluation per grid point).
            let params = &self.params;
            self.grid
                .par_iter()
                .zip(self.p_buf.par_iter_mut())
                .for_each(|(xi, pi)| {
                    if outside_dirichlet(params, &model.flow_inv(*xi)) {
                        *pi = 0.0;
                    }
                });
        }
    }

    /// Diffusion step, `p_buf` → `p_field`: ∂τ p = (ε/2) Σ_d q_d ∂²_d p over
    /// δτ = dt as one implicit-Euler step of the scheme `params.diffusion`
    /// (split by default, for nodal positivity). The per-direction operators
    /// are L_d = β_d M_d⁻¹K_d with β_d = ε·q_d/2 (weak form of −(ε q_d/2)∂²_d),
    /// applied as a function of the per-direction stiffness eigenvalues μ_d
    /// of each mode (`apply_in_place_dirs`):
    ///
    ///   split:    ∏_d (1 + dt·β_d μ_d)⁻¹,      unsplit:  (1 + dt·Σ_d β_d μ_d)⁻¹.
    ///
    /// The solver works on real nodal vectors in place: `p_buf` is stepped
    /// where it stands and then swapped with `p_field` (the old `p_field`
    /// becomes the scratch buffer of the next transport step).
    fn diffuse(&mut self) {
        let dt = self.model.dt();
        let eps = self.params.eps;
        let beta: [f64; M] = std::array::from_fn(|d| eps * self.params.q_diag[d] / 2.0);
        let p = &mut self.p_buf;
        match self.params.diffusion {
            DiffusionScheme::SplitEuler => self.solver.apply_in_place_dirs(p, |mu| {
                (0..M)
                    .map(|d| 1.0 / (1.0 + dt * beta[d] * mu[d]))
                    .product::<f64>()
            }),
            DiffusionScheme::Euler => self.solver.apply_in_place_dirs(p, |mu| {
                let lam: f64 = (0..M).map(|d| beta[d] * mu[d]).sum();
                1.0 / (1.0 + dt * lam)
            }),
        }
        std::mem::swap(&mut self.p_field, &mut self.p_buf);
    }

    /// Run the full filter loop with outputs: `n_steps` iterations of
    /// [`forward`](Self::forward), `n_snapshots` evenly spaced marginal
    /// snapshots (including t = 0; e.g. 25 gives a 5×5 grid in the plotting
    /// script), a
    /// progress bar on stderr, the reference-trajectory table on stdout, and
    /// the binary outputs in `out_dir`:
    ///
    /// * `axis_{d}.bin` — 1D Gauss–Lobatto axis of direction d (ndof_1d f64 LE)
    /// * `p2d_{a}_{b}_{step:05}.bin` — max-marginal of p on each `plot_pairs`
    ///   plane at every saved step
    /// * `trajectory.bin` — reference states, interleaved f64 LE
    /// * `estimate.bin` — estimator x̂_n = argmax p at every step, same layout
    ///   as trajectory.bin
    /// * `meta.txt` — metadata for the plotting script
    ///
    /// Panics if the filter is not initialized or a `plot_pairs` entry is out
    /// of range.
    pub fn run(
        &mut self,
        n_steps: usize,
        n_snapshots: usize,
        plot_pairs: &[(usize, usize)],
        out_dir: &str,
    ) {
        for &(a, b) in plot_pairs {
            assert!(
                a < M && b < M && a != b,
                "plot pair ({a}, {b}) is invalid for M = {M}"
            );
        }
        self.p_field(); // fail early (with the init message) if not initialized
        std::fs::create_dir_all(out_dir).expect("cannot create output dir");

        println!(
            "\nSpectral solver: {} = {} DOFs",
            self.params.ndof().map(|n| n.to_string()).join("×"),
            self.grid.len()
        );

        self.save_axes(out_dir);

        // n_snapshots − 1 saves after the t=0 snapshot (div_ceil so the final
        // forced save at n_steps never adds an extra one).
        let save_every = n_steps.div_ceil(n_snapshots.saturating_sub(1).max(1)).max(1);
        let mut saved_steps: Vec<usize> = vec![0];
        self.save_marginals_2d(0, plot_pairs, out_dir);

        let dt = self.model.dt();
        println!(
            "Mortensen filter: {n_steps} iterations (dt={dt}, t_end={}), diffusion scheme {:?}",
            n_steps as f64 * dt,
            self.params.diffusion
        );
        let t0 = std::time::Instant::now();

        let mut estimates: Vec<[f64; M]> = Vec::with_capacity(n_steps + 1);
        estimates.push(self.argmax_p());

        for n in 0..n_steps {
            self.forward();
            estimates.push(self.argmax_p());

            let step = n + 1;
            if step % save_every == 0 || step == n_steps {
                self.save_marginals_2d(step, plot_pairs, out_dir);
                saved_steps.push(step);
            }

            print_progress(step, n_steps, t0);
        }
        eprintln!();

        self.print_trajectory();
        self.save_trajectory(out_dir);
        self.save_estimate(&estimates, out_dir);

        let meta = format!(
            "n_axis={}\nM={M}\ndt={dt}\nn_el={}\np_ord={}\nperiodic={}\ndirichlet={}\nlabels={}\npairs={}\nsaved_steps={}\n",
            self.params.ndof().map(|n| n.to_string()).join(","),
            self.params.n_el.map(|n| n.to_string()).join(","),
            self.params.p_ord.map(|p| p.to_string()).join(","),
            self.params
                .periodic
                .map(|p| if p { "1" } else { "0" }.to_string())
                .join(","),
            self.params
                .dirichlet
                .map(|p| if p { "1" } else { "0" }.to_string())
                .join(","),
            self.model.state_labels().join(","),
            plot_pairs
                .iter()
                .map(|(a, b)| format!("{a}:{b}"))
                .collect::<Vec<_>>()
                .join(","),
            saved_steps
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
        std::fs::write(format!("{out_dir}/meta.txt"), meta).expect("cannot write meta.txt");
        println!("Saved {} snapshots to {out_dir}/", saved_steps.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ode_models::models::{PendulumSystem, SpringSystem};
    use nalgebra::DVector;
    use std::f64::consts::PI;

    /// `shift_by_elements` is an exact node-to-node translation: after a
    /// shift by k elements, p_new at a node equals p_old at the node k·h
    /// further (checked through the grid coordinates on a Gaussian that is
    /// not element-aligned), and the nodes whose source left the box are 0.
    #[test]
    fn shift_by_elements_is_an_exact_node_translation() {
        let params = FilterParams {
            domain: [(-1.0, 1.0), (-1.5, 1.5)],
            eps: 0.1,
            gamma: 1.0,
            n_el: [5, 6],
            p_ord: [4, 3],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: [1.0; 2],
            periodic: [false; 2],
            dirichlet: [true; 2],
            pre_compute_flow_inv: false,
        };
        let sys = SpringSystem::new(
            1,
            1.0,
            1.0,
            0.01,
            DVector::from_row_slice(&[0.3]),
            DVector::zeros(1),
            |x: &[f64]| x[0],
        );
        let mut filter = MortensenFilter::<2, _>::new(sys, params);
        filter.init_filter_gaussian([3.0, 2.0], [0.17, -0.31]);
        let old = filter.p_field().to_vec();
        let grid = filter.grid().to_vec();
        let h = filter.element_size();
        let k = [1isize, -2];
        filter.shift_by_elements(k);
        let new = filter.p_field();
        let dom = params.domain;
        let mut checked = 0;
        for (i, xi) in grid.iter().enumerate() {
            let src: [f64; 2] = std::array::from_fn(|d| xi[d] + k[d] as f64 * h[d]);
            let inside = (0..2).all(|d| src[d] > dom[d].0 && src[d] < dom[d].1);
            if inside {
                // Find the node at `src` (exists: whole-element shift).
                let j = grid
                    .iter()
                    .position(|g| (0..2).all(|d| (g[d] - src[d]).abs() < 1e-10))
                    .expect("shifted node is a grid node");
                assert_eq!(new[i], old[j], "p_new at node {i} ≠ p_old at its source");
                checked += 1;
            } else {
                assert_eq!(new[i], 0.0, "node {i} entered from outside: must be 0");
            }
        }
        assert!(checked > 0);
    }

    /// A periodic angle really wraps: a bump placed just below q₁ = π, left
    /// to diffuse (γ = 0, pendulum at rest near the top) for a few steps,
    /// shows up just above q₁ = −π when the direction is periodic, and not
    /// at all when it is Neumann (the bump then reflects at the wall).
    #[test]
    fn periodic_angle_wraps_across_the_boundary() {
        let run = |periodic: bool| {
            let x0 = [PI - 0.1, 0.0, 0.0, 0.0];
            let sys = PendulumSystem::new(x0, 0.01);
            let mut filter = MortensenFilter::<4, _>::new(
                sys,
                FilterParams {
                    domain: [(-PI, PI), (-PI, PI), (-3.0, 3.0), (-3.0, 3.0)],
                    eps: 0.1,
                    gamma: 0.0,
                    n_el: [12, 4, 4, 4],
                    p_ord: [4; 4],
                    diffusion: DiffusionScheme::SplitEuler,
                    q_diag: [1.0; 4],
                    periodic: [periodic, periodic, false, false],
                    dirichlet: [false; 4],
                    pre_compute_flow_inv: true,
                },
            );
            filter.init_filter_gaussian([10.0; 4], x0);
            for _ in 0..30 {
                filter.forward();
            }
            let p = filter.p_field();
            let grid = filter.grid();
            let max = p.iter().cloned().fold(0.0, f64::max);
            // Largest value on the far side of the boundary, q₁ ∈ (−π, −π + 0.4).
            let far = grid
                .iter()
                .zip(p)
                .filter(|(x, _)| x[0] < -PI + 0.4)
                .map(|(_, &v)| v)
                .fold(0.0, f64::max);
            far / max
        };
        let wrapped = run(true);
        let walled = run(false);
        assert!(wrapped > 0.05, "periodic q₁: nothing crossed the boundary ({wrapped:.2e})");
        assert!(walled < 1e-6, "Neumann q₁: mass crossed the wall ({walled:.2e})");
    }

    /// The Gaussian prior on a periodic direction is the wrapped Gaussian:
    /// centred just below q = π it is large again just above q = −π (the
    /// image through the seam) and continuous across it, where the plain
    /// Gaussian would be ≈ 0 there and jump at the seam; the non-periodic
    /// direction keeps the plain Gaussian. Checked against an independent
    /// image sum at every node.
    #[test]
    fn gaussian_prior_is_wrapped_on_periodic_directions() {
        let (eps, sig, c) = (0.1, 2.0, PI - 0.3);
        let sys = SpringSystem::new(
            1,
            1.0,
            1.0,
            0.01,
            DVector::from_row_slice(&[0.0]),
            DVector::zeros(1),
            |x: &[f64]| x[0],
        );
        let mut filter = MortensenFilter::<2, _>::new(
            sys,
            FilterParams {
                domain: [(-PI, PI), (-2.0, 2.0)],
                eps,
                gamma: 1.0,
                n_el: [8, 4],
                p_ord: [4, 4],
                diffusion: DiffusionScheme::SplitEuler,
                q_diag: [1.0; 2],
                periodic: [true, false],
                dirichlet: [false; 2],
                pre_compute_flow_inv: false,
            },
        );
        filter.init_filter_gaussian([sig, sig], [c, 0.5]);
        let (grid, p) = (filter.grid(), filter.p_field());
        // Independent wrapped Gaussian (5 images each side) × plain Gaussian.
        let expect = |x: [f64; 2]| {
            let g0: f64 = (-5..=5)
                .map(|k| {
                    let v = x[0] - c + k as f64 * 2.0 * PI;
                    (-sig * v * v / (2.0 * eps)).exp()
                })
                .sum();
            g0 * (-sig * (x[1] - 0.5) * (x[1] - 0.5) / (2.0 * eps)).exp()
        };
        for (x, &v) in grid.iter().zip(p) {
            let e = expect(*x);
            assert!((v - e).abs() <= 1e-12 * (1.0 + e), "at {x:?}: {v} vs {e}");
        }
        // The image through the seam: at q = −π (the stored first node) the
        // prior is exp(−σ 0.3²/2ε) ≈ 0.41, not ≈ 0.
        let row = |q: f64| {
            grid.iter()
                .zip(p)
                .filter(|(x, _)| (x[0] - q).abs() < 1e-12 && (x[1] - 0.5).abs() < 0.3)
                .map(|(_, &v)| v)
                .fold(0.0, f64::max)
        };
        let at_seam = row(-PI);
        assert!(at_seam > 0.3, "no image through the seam: p(−π) = {at_seam}");
        // Periodicity: the value at −π is the value the Gaussian has at +π,
        // the same point of the circle (0.3 below the centre), exp(−σ 0.3²/2ε)
        // up to the farther images — the plain Gaussian would give ≈ 0 here.
        let at_pi = (-sig * 0.3 * 0.3 / (2.0 * eps)).exp();
        assert!((at_seam - at_pi).abs() < 1e-3, "p(−π) = {at_seam} ≠ p(π) = {at_pi}");
    }

    /// Mixed periodic/Neumann directions: the solver's grid size must match
    /// `FilterParams::ndof` (periodic drops the duplicated endpoint), and a
    /// few filter iterations must keep p finite.
    #[test]
    fn periodic_directions_are_consistent() {
        let params = FilterParams {
            domain: [(-PI, PI), (-PI, PI), (-3.0, 3.0), (-3.0, 3.0)],
            eps: 0.1,
            gamma: 1.0,
            n_el: [2; 4],
            p_ord: [2; 4],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: [1.0; 4],
            periodic: [true, true, false, false],
            dirichlet: [false; 4],
            pre_compute_flow_inv: false,
        };
        let sys = PendulumSystem::new([1.5, 1.4, 0.0, 0.0], 0.01);
        let mut filter = MortensenFilter::<4, _>::new(sys, params);
        assert_eq!(
            filter.grid().len(),
            params.ndof().iter().product::<usize>(),
            "FilterParams::ndof disagrees with the solver's grid"
        );

        filter.init_filter_quadratic(1.0);
        for _ in 0..3 {
            filter.forward();
        }
        let max = filter
            .p_field()
            .iter()
            .copied()
            .fold(f64::MIN, f64::max);
        assert!(max.is_finite() && max > 0.0, "p degenerated: max = {max}");
    }

    /// The plan-based transport (`pre_compute_flow_inv`, `eval_with_plan`)
    /// must produce the same p field as the per-iteration `convect` path.
    #[test]
    fn precomputed_transport_matches_convect() {
        let sys = || {
            SpringSystem::new(
                1,
                1.0,
                1.0,
                0.005,
                DVector::from_row_slice(&[0.5]),
                DVector::zeros(1),
                |x: &[f64]| x[0],
            )
        };
        let params = |pre| FilterParams {
            domain: [(-3.0, 3.0); 2],
            eps: 0.01,
            gamma: 1.0,
            n_el: [4; 2],
            p_ord: [2; 2],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: [1.0; 2],
            periodic: [false; 2],
            dirichlet: [false; 2],
            pre_compute_flow_inv: pre,
        };

        let mut on_the_fly = MortensenFilter::<2, _>::new(sys(), params(false));
        let mut precomputed = MortensenFilter::<2, _>::new(sys(), params(true));

        // V_0 = σ·|x|²/2 is minimal at the origin, which is a grid point
        // (element boundaries at -3, -1.5, 0, 1.5, 3): argmax p = 0 exactly.
        on_the_fly.init_filter_quadratic(10.0);
        let xhat = on_the_fly.argmax_p();
        assert!(
            xhat.iter().all(|c| c.abs() < 1e-12),
            "argmax of the initial quadratic should be the origin, got {xhat:?}"
        );
        on_the_fly.init_filter_quadratic(10.0);
        precomputed.init_filter_quadratic(10.0);

        for _ in 0..5 {
            on_the_fly.forward();
            precomputed.forward();
        }
        for (pa, pb) in on_the_fly.p_field().iter().zip(precomputed.p_field()) {
            assert!(
                (pa - pb).abs() < 1e-14,
                "transport paths diverge: {pa} vs {pb}"
            );
        }
    }

    fn spring_2d(dt: f64) -> SpringSystem {
        SpringSystem::new(
            1,
            1.0,
            1.0,
            dt,
            DVector::from_row_slice(&[0.5]),
            DVector::zeros(1),
            |x: &[f64]| x[0],
        )
    }

    /// γ weights the observation step: γ = 0 leaves p unchanged, and the
    /// pointwise multiplier exp(−dt·γ·d/(2ε)) is exponential in γ, so
    /// doubling γ squares it: p_{2γ}/p₀ = (p_γ/p₀)².
    #[test]
    fn gamma_weights_the_observation_step() {
        let params = |gamma| FilterParams {
            domain: [(-3.0, 3.0); 2],
            eps: 0.5,
            gamma,
            n_el: [4; 2],
            p_ord: [3; 2],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: [1.0; 2],
            periodic: [false; 2],
            dirichlet: [false; 2],
            pre_compute_flow_inv: false,
        };
        let observe = |gamma: f64| {
            let mut f = MortensenFilter::<2, _>::new(spring_2d(0.01), params(gamma));
            f.init_filter_quadratic(1.0);
            let before = f.p_field().to_vec();
            f.observe();
            (before, f.p_field().to_vec())
        };
        let (before, off) = observe(0.0);
        assert_eq!(before, off, "γ = 0 must switch the observation off");
        let (p0, g1) = observe(1.0);
        let (_, g2) = observe(2.0);
        let mut changed = false;
        for ((b, p1), p2) in p0.iter().zip(&g1).zip(&g2) {
            assert!((p2 / b - (p1 / b) * (p1 / b)).abs() < 1e-12);
            changed |= (p1 - b).abs() > 1e-6;
        }
        assert!(changed, "γ = 1 observation had no effect anywhere");
    }

    /// Nodal positivity of the split diffusion step, tested on the step
    /// alone (the degree-p Lagrange interpolation of the transport step is
    /// not positivity-preserving, so the whole cycle cannot be): a Dirac
    /// nodal vector (the extreme case) diffused by one split step with
    /// dt·ε/2 large relative to the local mesh size squared — the regime
    /// where the 1D discrete maximum principle holds — must stay
    /// nonnegative at every node.
    #[test]
    fn split_euler_diffusion_is_nodally_positive() {
        let dt = 0.01;
        let mut filter = MortensenFilter::<2, _>::new(
            spring_2d(dt),
            FilterParams {
                domain: [(-3.0, 3.0); 2],
                eps: 20.0, // dt·ε/2 = 0.1 ≫ smallest nodal spacing² ≈ 0.017
                gamma: 1.0,
                n_el: [8; 2],
                p_ord: [4; 2],
                diffusion: DiffusionScheme::SplitEuler,
                q_diag: [1.0; 2],
                periodic: [false; 2],
                dirichlet: [false; 2],
                pre_compute_flow_inv: false,
            },
        );
        filter.init_filter_quadratic(1.0); // sizes the buffers
        // Dirac at an interior node (near the origin) written straight into
        // the transport output buffer that `diffuse` consumes.
        let i = filter
            .grid
            .iter()
            .position(|x| x[0].abs() < 1e-12 && x[1].abs() < 1e-12)
            .expect("the origin is a grid point");
        filter.p_buf.iter_mut().for_each(|v| *v = 0.0);
        filter.p_buf[i] = 1.0;
        filter.diffuse();

        let min = filter.p_field().iter().copied().fold(f64::MAX, f64::min);
        let max = filter.p_field().iter().copied().fold(f64::MIN, f64::max);
        assert!(max > 0.0 && max < 1.0, "not a diffusion: max = {max}");
        assert!(min >= 0.0, "split scheme lost nodal positivity: min p = {min:e}");
    }

    /// The split and unsplit steps discretize the same diffusion: on the
    /// full cycle with a smooth p and a small step they agree to
    /// splitting-error accuracy, O(dt²) relative.
    #[test]
    fn split_and_unsplit_diffusion_are_consistent() {
        let params = |diffusion| FilterParams {
            domain: [(-3.0, 3.0); 2],
            eps: 0.5,
            gamma: 1.0,
            n_el: [8; 2],
            p_ord: [4; 2],
            diffusion,
            q_diag: [1.0; 2],
            periodic: [false; 2],
            dirichlet: [false; 2],
            pre_compute_flow_inv: false,
        };
        let mut split =
            MortensenFilter::<2, _>::new(spring_2d(0.01), params(DiffusionScheme::SplitEuler));
        let mut unsplit =
            MortensenFilter::<2, _>::new(spring_2d(0.01), params(DiffusionScheme::Euler));
        split.init_filter_quadratic(10.0);
        unsplit.init_filter_quadratic(10.0);
        for _ in 0..10 {
            split.forward();
            unsplit.forward();
        }
        let max = split.p_field().iter().copied().fold(f64::MIN, f64::max);
        let diff = split
            .p_field()
            .iter()
            .zip(unsplit.p_field())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max > 0.0 && max.is_finite(), "p degenerated: max = {max}");
        assert!(
            diff / max < 1e-2,
            "split and unsplit diffusion disagree: relative max difference {:e}",
            diff / max
        );
    }

    /// Diagonal anisotropy: with q = (1, 0) the step diffuses in direction 0
    /// only. A field constant along direction 0 (a function of x₁ alone)
    /// must then be left exactly unchanged by both schemes, while the same
    /// field diffuses with the isotropic q = (1, 1).
    #[test]
    fn anisotropic_diffusion_acts_only_on_weighted_directions() {
        for scheme in [DiffusionScheme::SplitEuler, DiffusionScheme::Euler] {
            let params = |q_diag| FilterParams {
                domain: [(-3.0, 3.0); 2],
                eps: 1.0,
                gamma: 1.0,
                n_el: [6; 2],
                p_ord: [4; 2],
                diffusion: scheme,
                q_diag,
                periodic: [false; 2],
                dirichlet: [false; 2],
                pre_compute_flow_inv: false,
            };
            let field = |x: &[f64]| (0.7 * x[1]).cos();
            let apply = |q_diag| {
                let mut f = MortensenFilter::<2, _>::new(spring_2d(0.05), params(q_diag));
                f.init_filter_quadratic(1.0); // sizes the buffers
                let grid = f.grid().to_vec();
                for (b, x) in f.p_buf.iter_mut().zip(&grid) {
                    *b = field(x);
                }
                f.diffuse();
                (grid, f.p_field().to_vec())
            };
            let (grid, aniso) = apply([1.0, 0.0]);
            let (_, iso) = apply([1.0, 1.0]);
            let err = grid
                .iter()
                .zip(&aniso)
                .map(|(x, p)| (p - field(x)).abs())
                .fold(0.0, f64::max);
            assert!(err < 1e-12, "{scheme:?}: q = (1, 0) changed an x₁-only field by {err:e}");
            let change = iso
                .iter()
                .zip(&aniso)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(change > 1e-3, "{scheme:?}: isotropic diffusion had no effect");
        }
    }

    /// Dirichlet diffusion absorbs at the wall: one implicit-Euler step of
    /// the constant field p ≡ 1 leaves it exactly invariant under Neumann
    /// (constants are in the kernel) but sags near the boundary under
    /// Dirichlet — staying within [0, 1] and ≈ 1 deep inside. Also pins the
    /// Dirichlet DOF count (no stored node on the boundary).
    #[test]
    fn dirichlet_diffusion_absorbs_at_the_boundary() {
        for scheme in [DiffusionScheme::SplitEuler, DiffusionScheme::Euler] {
            let params = |dirichlet| FilterParams {
                domain: [(-3.0, 3.0); 2],
                eps: 1.0,
                gamma: 1.0,
                n_el: [6; 2],
                p_ord: [4; 2],
                diffusion: scheme,
                q_diag: [1.0; 2],
                periodic: [false; 2],
                dirichlet,
                pre_compute_flow_inv: false,
            };
            let diffuse_const = |dirichlet: [bool; 2]| {
                let mut f = MortensenFilter::<2, _>::new(spring_2d(0.05), params(dirichlet));
                assert_eq!(
                    f.grid().len(),
                    params(dirichlet).ndof().iter().product::<usize>(),
                    "grid size disagrees with FilterParams::ndof"
                );
                f.init_filter_quadratic(1.0); // sizes the buffers
                f.p_buf.iter_mut().for_each(|v| *v = 1.0);
                f.diffuse();
                (f.grid().to_vec(), f.p_field().to_vec())
            };

            let (_, pn) = diffuse_const([false; 2]);
            let drift = pn.iter().map(|v| (v - 1.0).abs()).fold(0.0, f64::max);
            assert!(drift < 1e-10, "{scheme:?}: Neumann moved the constant by {drift:e}");

            let (grid, pd) = diffuse_const([true; 2]);
            assert!(
                grid.iter().all(|x| x.iter().all(|c| c.abs() < 3.0)),
                "{scheme:?}: a Dirichlet grid stores a boundary node"
            );
            let (mut near_wall, mut center) = (f64::MAX, 0.0);
            for (x, &v) in grid.iter().zip(&pd) {
                assert!((-1e-12..=1.0 + 1e-12).contains(&v), "{scheme:?}: p = {v} at {x:?}");
                let d_wall = x.iter().map(|c| 3.0 - c.abs()).fold(f64::MAX, f64::min);
                if d_wall < 0.2 {
                    near_wall = near_wall.min(v);
                }
                if x[0].abs() < 0.3 && x[1].abs() < 0.3 {
                    center = v;
                }
            }
            assert!(near_wall < 0.8, "{scheme:?}: no absorption at the wall: {near_wall}");
            assert!(center > 0.99, "{scheme:?}: interior should be untouched: {center}");
        }
    }

    /// Dirichlet semantics of the transport step: p ≡ 0 outside the box, so
    /// a grid point whose preimage φ⁻¹(ξ) leaves the box in a Dirichlet
    /// direction receives exactly 0 — not the odd extension's −p(reflected)
    /// — and the precomputed-mask and per-step paths agree.
    #[test]
    fn dirichlet_transport_zeroes_outside_preimages() {
        let dt = 0.4; // large step: the rotation pushes many preimages out
        let params = |pre| FilterParams {
            domain: [(-2.0, 2.0); 2],
            eps: 0.1,
            gamma: 1.0,
            n_el: [5; 2],
            p_ord: [3; 2],
            diffusion: DiffusionScheme::SplitEuler,
            q_diag: [1.0; 2],
            periodic: [false; 2],
            dirichlet: [true; 2],
            pre_compute_flow_inv: pre,
        };
        let run = |pre| {
            let mut f = MortensenFilter::<2, _>::new(spring_2d(dt), params(pre));
            f.init_filter(|_| 0.0); // p ≡ 1 at every stored node
            f.transport();
            (f.grid().to_vec(), f.p_buf.clone())
        };
        let (grid, masked) = run(true);
        let (_, per_step) = run(false);

        let model = spring_2d(dt);
        let mut n_out = 0;
        for (i, xi) in grid.iter().enumerate() {
            let y = Model::<2>::flow_inv(&model, *xi);
            if y.iter().any(|&c| !(-2.0..=2.0).contains(&c)) {
                n_out += 1;
                assert_eq!(masked[i], 0.0, "preimage outside but p = {} at {xi:?}", masked[i]);
            }
            assert!(
                (masked[i] - per_step[i]).abs() < 1e-14,
                "mask paths disagree at {xi:?}: {} vs {}",
                masked[i],
                per_step[i]
            );
        }
        assert!(n_out > 0, "test is vacuous: no preimage left the box");
    }
}
