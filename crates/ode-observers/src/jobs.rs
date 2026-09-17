//! The dimension-erased job layer: the observer configuration edited by a
//! user interface ([`FilterConfig`]), the plain-data outputs kept for
//! display ([`FilterOutput`], [`TrackerOutput`], [`BoxOutput`],
//! [`ParticleOutput`]), the generic runners ([`run_filter`],
//! [`run_tracker`], [`run_unscented`], [`run_box`], [`run_particles`])
//! driving each observer
//! along a [`Reference`] (a trajectory with its observations) at the
//! model's compile-time dimension, and the **job description**
//! ([`JobSpec`]: a [`ModelSpec`] + the `Reference` + an [`Estimator`] + the
//! configuration) whose [`run`](JobSpec::run) dispatches to the right
//! runner through the [`ModelVisitor`] of `ode-models` — so a front end
//! (or a server receiving the description) never names a model type or a
//! dimension. The observations travel with the job: they may come from a
//! twin experiment ([`JobSpec::from_twin`]) or from anywhere else. UI-free.

use crate::methods::Observer;
use crate::methods::mortensen_window::{BoxTracker, BoxTrackerParams};
use crate::methods::mortensen::{DiffusionScheme, FilterParams, MortensenFilter};
use crate::methods::fleming_viot::{ParticleParams, ParticleSystem};
use ode_models_spec::progress::Progress;
use crate::methods::kalman::{MortensenTracker, TrackerParams};
use crate::methods::unscented_kalman::{Quadrature, UnscentedParams, UnscentedTracker};
use ode_models::model::Model;
use ode_models_spec::reference::Reference;
use ode_models_spec::spec::{ModelSpec, ModelVisitor, TwinSpec};
use serde::{Deserialize, Serialize};

/// Filter settings of one state variable.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct VarConfig {
    pub label: String,
    /// Periodic variables (angles) have their domain fixed to the model's
    /// interval and use periodic boundary conditions — shown in the UI but
    /// not editable.
    pub periodic: bool,
    /// Dirichlet wall in this direction: p = 0 on the domain boundary
    /// (V = +∞ outside — absorbing; preimages leaving the box get p = 0)
    /// instead of the reflecting Neumann default. Ignored (kept `false`)
    /// for periodic variables.
    pub dirichlet: bool,
    /// Per-direction domain interval on which the filter discretizes p.
    pub domain: (f64, f64),
    pub n_el: usize,
    pub p_ord: usize,
    /// Half-width L_d of the translating window (`box_tracker`) in this
    /// direction: the window is ξ_d ∈ [−L_d, L_d] around its centre x̂. A
    /// few widths of p; defaults to a quarter of the run's state range.
    pub half_width: f64,
    /// Elements per direction of the window (its order is `p_ord`).
    pub win_n_el: usize,
    /// Model-noise variance q_d of this direction (diagonal of Q): the
    /// diffusion is (ε/2)·q_d·∂²_d; 1 = isotropic, 0 = none.
    pub q: f64,
    /// Center x_{c,d} of the Gaussian initial data in this direction
    /// (defaults to the run's initial state).
    pub x0: f64,
    /// Initial-data stiffness σ_d: V₀ = Σ_d σ_d (x_d − x_{c,d})²/2, i.e. a
    /// Gaussian of variance ε/σ_d in this direction; 0 = flat.
    pub sigma: f64,
}

/// Dimension-erased filter configuration edited in the UI; converted to a
/// `FilterParams<M>` by [`run_filter`].
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterConfig {
    pub eps: f64,
    /// Observation weight γ (shared by the filter and the tracker): scales
    /// the observation term γ·d²/2; 1 is the standard cycle, 0 switches the
    /// observations off.
    pub gamma: f64,
    /// Diffusion time-stepping scheme (split implicit Euler for nodal
    /// positivity, or the unsplit step).
    pub diffusion: DiffusionScheme,
    /// Number of 2D marginal snapshots stored, evenly spaced over the run
    /// (2+ include t = 0; 1 stores only the final step). Clamped to
    /// n_steps + 1 by [`run_filter`].
    pub n_snapshots: usize,
    pub vars: Vec<VarConfig>,
    /// All candidate 2D output planes (a, b) with a < b, and whether each is
    /// selected (the marginal is the max of p over the other directions).
    pub pairs: Vec<((usize, usize), bool)>,
    /// Settings of the particle solver (`crate::methods::fleming_viot`).
    pub particles: ParticleConfig,
    /// Settings of the unscented closure (`crate::methods::unscented_kalman`);
    /// absent in older descriptions, hence the default.
    #[serde(default)]
    pub unscented: UnscentedConfig,
}

/// Settings of the unscented closure, edited in the viewer's Quadrature
/// panel: the rule alone — ε, γ, q and the initial data come from the
/// shared configuration.
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UnscentedConfig {
    pub rule: Quadrature,
}

impl Default for UnscentedConfig {
    /// The symmetric rule at h = √3.
    fn default() -> Self {
        UnscentedConfig { rule: Quadrature::DEFAULT }
    }
}

/// Settings of the particle solver, edited in the viewer's Particles panel;
/// ε, q and the domains come from the shared configuration.
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParticleConfig {
    /// Number of particles N.
    pub n_particles: usize,
    /// Seed of the particle system's generator; `None` draws one from the
    /// clock at each run (the viewer's choice: every run a fresh sample).
    pub seed: Option<u64>,
}

impl FilterConfig {
    /// Defaults computed from a completed run: non-periodic domains cover
    /// the reference trajectory's range with a ×1.3 margin, periodic
    /// components (`periodic[d] = Some((lo, hi))`, the model's interval)
    /// get that interval as domain; the resolution default scales with the
    /// dimension (2D can afford the `one_spring` example's resolution, 4D+
    /// starts small enough to stay interactive). ε = 0.01 and σ_d = 0.01 (a wide
    /// p₀) for every model; the Gaussian center defaults to the run's
    /// initial state.
    pub fn defaults(
        labels: Vec<String>,
        periodic: Vec<Option<(f64, f64)>>,
        default_pairs: &[(usize, usize)],
        states: &[Vec<f64>],
    ) -> Self {
        let dim = labels.len();
        let n_el = if dim <= 2 { 20 } else { 5 };
        let vars = labels
            .into_iter()
            .enumerate()
            .map(|(d, label)| {
                let (mut mn, mut mx) = (f64::MAX, f64::MIN);
                for s in states {
                    mn = mn.min(s[d]);
                    mx = mx.max(s[d]);
                }
                let domain = match periodic[d] {
                    Some(interval) => interval,
                    None => {
                        let center = 0.5 * (mn + mx);
                        let half = (0.65 * (mx - mn)).max(0.5);
                        (center - half, center + half)
                    }
                };
                VarConfig {
                    label,
                    periodic: periodic[d].is_some(),
                    dirichlet: false,
                    domain,
                    n_el,
                    p_ord: 4,
                    half_width: (0.25 * (mx - mn)).max(0.2),
                    win_n_el: 5,
                    q: 1.0,
                    x0: states.first().map_or(0.0, |s| s[d]),
                    sigma: 0.01,
                }
            })
            .collect();
        let pairs = (0..dim)
            .flat_map(|a| ((a + 1)..dim).map(move |b| (a, b)))
            .map(|p| (p, default_pairs.contains(&p)))
            .collect();
        FilterConfig {
            eps: 0.01,
            gamma: 1.0,
            diffusion: DiffusionScheme::SplitEuler,
            // Every step (run_filter clamps to n_steps + 1 anyway).
            n_snapshots: states.len().max(1),
            vars,
            pairs,
            particles: ParticleConfig { n_particles: 500, seed: None },
            unscented: UnscentedConfig::default(),
        }
    }

    pub fn selected_pairs(&self) -> Vec<(usize, usize)> {
        self.pairs
            .iter()
            .filter(|(_, on)| *on)
            .map(|(p, _)| *p)
            .collect()
    }
}

/// In-memory outputs of one filter run, kept on the slot for the filter
/// view. Self-contained: it also carries the reference trajectory the
/// filter ran against, so editing the model afterwards cannot desynchronize
/// the display.
#[derive(Serialize, Deserialize)]
#[allow(dead_code)] // estimates/periodic: stored outputs, not displayed yet
pub struct FilterOutput {
    pub dt: f64,
    pub labels: Vec<String>,
    /// Per-direction 1D Gauss–Lobatto axes (nodal coordinates; a periodic
    /// direction omits the right endpoint, identified with the left).
    pub axes: Vec<Vec<f64>>,
    /// Per-direction element structure, for spectral redisplay: the
    /// heatmaps re-evaluate the degree-`p_ord` element polynomials rather
    /// than connecting nodes linearly.
    pub n_el: Vec<usize>,
    pub p_ord: Vec<usize>,
    pub periodic: Vec<bool>,
    /// Per-direction Dirichlet wall (p = 0 at the boundary): such an axis
    /// stores only the interior nodes (n_el·p_ord − 1), the two boundary
    /// zeros being implicit.
    pub dirichlet: Vec<bool>,
    pub domain: Vec<(f64, f64)>,
    /// The stored 2D planes (a, b).
    pub pairs: Vec<(usize, usize)>,
    /// Steps at which marginal snapshots were stored (starts at 0).
    pub saved_steps: Vec<usize>,
    /// `marginals[k][j]`: max-marginal of pair `pairs[j]` at step
    /// `saved_steps[k]`, ndof[a] × ndof[b] with direction a varying fastest.
    pub marginals: Vec<Vec<Vec<f64>>>,
    /// Estimator x̂_n = argmax p at every step (including t = 0).
    pub estimates: Vec<Vec<f64>>,
    /// The reference trajectory the filter ran against, one flat state per
    /// step (including t = 0).
    pub reference: Vec<Vec<f64>>,
}

/// In-memory outputs of one tracker run (the second-order / Gaussian
/// closure, `crate::methods::kalman`): the estimate and its covariance at every
/// step, with the reference trajectory it ran against.
#[derive(Serialize, Deserialize)]
pub struct TrackerOutput {
    pub dt: f64,
    /// The configuration's ε. The tracker itself is ε-free, but its P = S⁻¹
    /// is the inverse *energy* curvature: the density p ∝ exp(−V/ε) has
    /// covariance ε·P, which is what an uncertainty band must show.
    pub eps: f64,
    pub labels: Vec<String>,
    /// x̂ₙ at every step (including t = 0).
    pub estimates: Vec<Vec<f64>>,
    /// Covariance P = S⁻¹ at every step, row-major M×M; `None` while S is
    /// singular (flat prior, not yet observed).
    pub covariances: Vec<Option<Vec<f64>>>,
    /// The reference trajectory, one flat state per step.
    pub reference: Vec<Vec<f64>>,
}

/// Run the Mortensen tracker along `reference` (same initial data and
/// model-noise q as the filter configuration; ε, grid and outputs are
/// irrelevant to it), bumping `progress` once per iteration and stopping
/// early (partial output) if it is cancelled.
pub fn run_tracker<const M: usize, Mod: Model<M>>(
    model: Mod,
    reference: &Reference,
    cfg: &FilterConfig,
    progress: &Progress,
) -> TrackerOutput {
    check_dimensions::<M, _>(&model, reference, cfg);
    let labels = model.state_labels();
    let dt = model.dt();
    let mut tracker = MortensenTracker::<M, _>::new(
        model,
        TrackerParams {
            q_diag: std::array::from_fn(|d| cfg.vars[d].q),
            gamma: cfg.gamma,
        },
    );
    tracker.init(
        std::array::from_fn(|d| cfg.vars[d].x0),
        std::array::from_fn(|d| cfg.vars[d].sigma),
    );
    // The tracker works with unwrapped angles (its flow and innovation are
    // 2π-periodic, so nothing depends on the branch), while the reference
    // stores its angles wrapped to [−π, π): wrap the estimate the same way
    // for display, or a winding pendulum's x̂ runs past π while the
    // reference jumps back to −π.
    let wrap = wrapper::<M>(cfg);
    let mut estimates = vec![wrap(tracker.estimate())];
    let mut covariances = vec![tracker.covariance().map(|p| p.transpose().as_slice().to_vec())];
    for y in &reference.observations[..reference.steps()] {
        if progress.cancelled() {
            break;
        }
        tracker.forward(y);
        estimates.push(wrap(tracker.estimate()));
        // nalgebra is column-major: transpose to store row-major.
        covariances.push(tracker.covariance().map(|p| p.transpose().as_slice().to_vec()));
        progress.step();
    }
    TrackerOutput {
        dt,
        eps: cfg.eps,
        labels,
        reference: reference.states[..estimates.len()].to_vec(),
        estimates,
        covariances,
    }
}

/// Run the unscented closure along `reference` (the configuration's initial
/// data, model-noise q, γ, ε and quadrature rule), producing the tracker's
/// kind of output — centre x̄ as the estimate, P = Σ/ε as the covariance —
/// so the Tracker view serves it unchanged. The initial stiffness must be
/// positive in every direction. Bumps `progress` once per iteration; stops
/// early (partial output) if cancelled.
pub fn run_unscented<const M: usize, Mod: Model<M>>(
    model: Mod,
    reference: &Reference,
    cfg: &FilterConfig,
    progress: &Progress,
) -> TrackerOutput {
    check_dimensions::<M, _>(&model, reference, cfg);
    let labels = model.state_labels();
    let dt = model.dt();
    let mut ukf = UnscentedTracker::<M, _>::new(
        model,
        UnscentedParams {
            q_diag: std::array::from_fn(|d| cfg.vars[d].q),
            gamma: cfg.gamma,
            eps: cfg.eps,
            rule: cfg.unscented.rule,
        },
    );
    ukf.init(
        std::array::from_fn(|d| cfg.vars[d].x0),
        std::array::from_fn(|d| cfg.vars[d].sigma),
    );
    // Unwrapped angles inside, wrapped for display (see `run_tracker`).
    let wrap = wrapper::<M>(cfg);
    let mut estimates = vec![wrap(ukf.estimate())];
    let mut covariances = vec![ukf.covariance().map(|p| p.transpose().as_slice().to_vec())];
    for y in &reference.observations[..reference.steps()] {
        if progress.cancelled() {
            break;
        }
        ukf.forward(y);
        estimates.push(wrap(ukf.estimate()));
        covariances.push(ukf.covariance().map(|p| p.transpose().as_slice().to_vec()));
        progress.step();
    }
    TrackerOutput {
        dt,
        eps: cfg.eps,
        labels,
        reference: reference.states[..estimates.len()].to_vec(),
        estimates,
        covariances,
    }
}

/// The configuration's periodic components wrapped into their domain, as
/// the reference generator stores them.
fn wrapper<const M: usize>(cfg: &FilterConfig) -> impl Fn([f64; M]) -> Vec<f64> {
    let periodic: Vec<Option<(f64, f64)>> = cfg.vars.iter().map(|v| v.periodic.then_some(v.domain)).collect();
    move |x: [f64; M]| {
        (0..M)
            .map(|d| match periodic[d] {
                Some((lo, hi)) => lo + (x[d] - lo).rem_euclid(hi - lo),
                None => x[d],
            })
            .collect()
    }
}

/// The model, the reference and the configuration must agree on the
/// dimension, and the reference on the model's time step.
fn check_dimensions<const M: usize, Mod: Model<M>>(model: &Mod, reference: &Reference, cfg: &FilterConfig) {
    assert_eq!(cfg.vars.len(), M, "FilterConfig dimension mismatch");
    assert_eq!(model.dim(), M, "model dimension mismatch");
    assert_eq!(reference.dim(), M, "reference dimension mismatch");
    assert_eq!(reference.obs_dim(), model.obs_dim(), "reference observation dimension mismatch");
    assert_eq!(reference.dt, model.dt(), "the reference and the model disagree on dt");
}

/// Run the Mortensen filter along `reference`, bumping `progress` once per
/// iteration and stopping early (partial output) if it is cancelled — the
/// check is once per iteration, after the (uninterruptible) construction of
/// the filter and its transport plan.
pub fn run_filter<const M: usize, Mod: Model<M> + Sync>(
    model: Mod,
    reference: &Reference,
    cfg: &FilterConfig,
    progress: &Progress,
) -> FilterOutput {
    check_dimensions::<M, _>(&model, reference, cfg);
    let params = FilterParams::<M> {
        domain: std::array::from_fn(|d| cfg.vars[d].domain),
        eps: cfg.eps,
        gamma: cfg.gamma,
        n_el: std::array::from_fn(|d| cfg.vars[d].n_el),
        p_ord: std::array::from_fn(|d| cfg.vars[d].p_ord),
        diffusion: cfg.diffusion,
        q_diag: std::array::from_fn(|d| cfg.vars[d].q),
        periodic: std::array::from_fn(|d| cfg.vars[d].periodic),
        dirichlet: std::array::from_fn(|d| cfg.vars[d].dirichlet && !cfg.vars[d].periodic),
        // All viewer models are autonomous; caching φ⁻¹ of the grid is a
        // pure win (chosen automatically, not exposed in the UI).
        pre_compute_flow_inv: true,
    };
    let labels = model.state_labels();
    let dt = model.dt();
    let mut filter = MortensenFilter::<M, _>::new(model, params);
    filter.init_filter_gaussian(
        std::array::from_fn(|d| cfg.vars[d].sigma),
        std::array::from_fn(|d| cfg.vars[d].x0),
    );

    let pairs = cfg.selected_pairs();
    let snapshot =
        |f: &MortensenFilter<M, Mod>| pairs.iter().map(|&p| f.marginal_2d(p)).collect::<Vec<_>>();
    let n_steps = reference.steps();
    let save_at = snapshot_schedule(n_steps, cfg.n_snapshots);

    let mut saved_steps = Vec::new();
    let mut marginals = Vec::new();
    if save_at[0] {
        saved_steps.push(0);
        marginals.push(snapshot(&filter));
    }
    let mut estimates = vec![filter.argmax_p().to_vec()];
    for (n, y) in reference.observations[..n_steps].iter().enumerate() {
        if progress.cancelled() {
            break;
        }
        filter.forward(y);
        estimates.push(filter.argmax_p().to_vec());
        let step = n + 1;
        if save_at[step] {
            marginals.push(snapshot(&filter));
            saved_steps.push(step);
        }
        progress.step();
    }

    FilterOutput {
        dt,
        labels,
        axes: (0..M).map(|d| filter.axis(d)).collect(),
        n_el: cfg.vars.iter().map(|v| v.n_el).collect(),
        p_ord: cfg.vars.iter().map(|v| v.p_ord).collect(),
        periodic: cfg.vars.iter().map(|v| v.periodic).collect(),
        dirichlet: cfg.vars.iter().map(|v| v.dirichlet && !v.periodic).collect(),
        domain: cfg.vars.iter().map(|v| v.domain).collect(),
        pairs,
        saved_steps,
        marginals,
        reference: reference.states[..estimates.len()].to_vec(),
        estimates,
    }
}

/// Exact evenly spaced snapshot schedule: `n_snapshots` distinct steps
/// ending at `n_steps`, including t = 0 when n_snapshots ≥ 2 (a plain
/// integer stride cannot hit the requested count exactly). Rounding keeps
/// the steps distinct because the spacing is ≥ 1 after the clamp.
fn snapshot_schedule(n_steps: usize, n_snapshots: usize) -> Vec<bool> {
    let n_snapshots = n_snapshots.clamp(1, n_steps + 1);
    let mut save_at = vec![false; n_steps + 1];
    if n_snapshots == 1 {
        save_at[n_steps] = true;
    } else {
        for k in 0..n_snapshots {
            let s = (k as f64 * n_steps as f64 / (n_snapshots - 1) as f64).round() as usize;
            save_at[s.min(n_steps)] = true;
        }
    }
    save_at
}

/// In-memory outputs of one particle run (`crate::methods::fleming_viot`): the whole
/// population at every step, with each particle's exponential budget e and
/// accumulated misfit a (the Particles view shrinks a particle's Gaussian
/// with its relative life expectancy (e − a)/e), and the reference
/// trajectory.
#[derive(Serialize, Deserialize)]
pub struct ParticleOutput {
    pub dt: f64,
    pub labels: Vec<String>,
    pub periodic: Vec<bool>,
    /// Per-direction box of the initial draw (the plot box of the view).
    pub domain: Vec<(f64, f64)>,
    pub n_particles: usize,
    /// `positions[step]`: the N positions flattened, particle-major
    /// (`positions[step][i * M + d]`), including t = 0.
    pub positions: Vec<Vec<f64>>,
    /// `budget[step][i]`: the exponential clock e of particle i.
    pub budget: Vec<Vec<f64>>,
    /// `spent[step][i]`: the misfit Σ dt·γ·d accumulated by particle i
    /// since its creation (< budget: the others were reborn).
    pub spent: Vec<Vec<f64>>,
    pub reference: Vec<Vec<f64>>,
}

/// Run the particle solver along `reference`: the killing uses the
/// configuration's γ (misfit dt·γ·d per step against an Exp(1) clock), the
/// initial population is drawn from the configuration's Gaussian initial
/// data (centre x_c, variance ε/σ_d; uniform in the domain where σ_d = 0),
/// the Brownian increment is the filter's √(ε q_d dt) (ε drives the noise;
/// the library's scale s is fixed to 1 here), and the seed is the
/// configuration's or, without one, drawn from the clock — every run a
/// fresh sample. Bumps `progress` once per step; stops early if cancelled.
pub fn run_particles<const M: usize, Mod: Model<M> + Sync>(
    model: Mod,
    reference: &Reference,
    cfg: &FilterConfig,
    progress: &Progress,
) -> ParticleOutput {
    check_dimensions::<M, _>(&model, reference, cfg);
    let pc = cfg.particles;
    let seed = pc.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    });
    let params = ParticleParams::<M> {
        n_particles: pc.n_particles.max(2),
        center: std::array::from_fn(|d| cfg.vars[d].x0),
        sigma: std::array::from_fn(|d| cfg.vars[d].sigma.max(0.0)),
        domain: std::array::from_fn(|d| cfg.vars[d].domain),
        periodic: std::array::from_fn(|d| cfg.vars[d].periodic),
        eps: cfg.eps,
        q_diag: std::array::from_fn(|d| cfg.vars[d].q),
        noise_scale: 1.0,
        gamma: cfg.gamma,
        seed,
    };
    let labels = model.state_labels();
    let dt = model.dt();
    let mut sys = ParticleSystem::<M, _>::new(model, params);
    let record = |s: &ParticleSystem<M, Mod>| {
        (
            s.positions().iter().flat_map(|z| z.iter().copied()).collect::<Vec<f64>>(),
            s.budgets().to_vec(),
            s.spent().to_vec(),
        )
    };
    let (mut positions, mut budget, mut spent) = (Vec::new(), Vec::new(), Vec::new());
    let push = |s: &ParticleSystem<M, Mod>, p: &mut Vec<Vec<f64>>, e: &mut Vec<Vec<f64>>, a: &mut Vec<Vec<f64>>| {
        let (x, y, z) = record(s);
        p.push(x);
        e.push(y);
        a.push(z);
    };
    push(&sys, &mut positions, &mut budget, &mut spent);
    for y in &reference.observations[..reference.steps()] {
        if progress.cancelled() {
            break;
        }
        sys.forward(y);
        push(&sys, &mut positions, &mut budget, &mut spent);
        progress.step();
    }
    ParticleOutput {
        dt,
        labels,
        periodic: cfg.vars.iter().map(|v| v.periodic).collect(),
        domain: cfg.vars.iter().map(|v| v.domain).collect(),
        n_particles: params.n_particles,
        reference: reference.states[..positions.len()].to_vec(),
        positions,
        budget,
        spent,
    }
}

/// In-memory outputs of one translating-window run (`crate::methods::mortensen_window`).
#[derive(Serialize, Deserialize)]
pub struct BoxOutput {
    /// The window's own grid and snapshots, in *window coordinates*
    /// ξ = x − x̂: domain (−L_d, L_d), Dirichlet in every direction, the
    /// max-marginals of ρ on the selected planes. Its `estimates` (x̂ +
    /// argmax ρ) and `reference` are in physical coordinates.
    pub window: FilterOutput,
    /// Window position x̂_n at every step (physical coordinates of ξ = 0;
    /// index = step, including t = 0). Periodic components are wrapped to
    /// (−π, π) like the reference.
    pub centers: Vec<Vec<f64>>,
}

/// Run the translating window along `reference`: the grid filter on the
/// box ∏[−L_d, L_d] around x̂, re-centred by whole elements after every
/// step (`BoxTracker`), with the configuration's initial data (the window
/// starts at the Gaussian centre x_c), model noise, ε, γ and scheme; the
/// per-variable window size is `half_width` / `win_n_el` / `p_ord`.
/// Bumps `progress` once per iteration; stops early if cancelled.
pub fn run_box<const M: usize, Mod: Model<M> + Sync>(
    model: Mod,
    reference: &Reference,
    cfg: &FilterConfig,
    progress: &Progress,
) -> BoxOutput {
    check_dimensions::<M, _>(&model, reference, cfg);
    let params = BoxTrackerParams::<M> {
        half_width: std::array::from_fn(|d| cfg.vars[d].half_width),
        eps: cfg.eps,
        gamma: cfg.gamma,
        n_el: std::array::from_fn(|d| cfg.vars[d].win_n_el),
        p_ord: std::array::from_fn(|d| cfg.vars[d].p_ord),
        diffusion: cfg.diffusion,
        q_diag: std::array::from_fn(|d| cfg.vars[d].q),
        // Autonomous flow (every viewer model): cache φ⁻¹ of the window's
        // nodes and refresh only the entering elements on a move.
        pre_compute_flow_inv: model.is_autonomous(),
    };
    let labels = model.state_labels();
    let dt = model.dt();
    let mut window = BoxTracker::<M, _>::new(model, params);
    window.init_gaussian(
        std::array::from_fn(|d| cfg.vars[d].x0),
        std::array::from_fn(|d| cfg.vars[d].sigma),
    );
    // The window's position is not wrapped; the reference is — wrap it
    // the same way for display.
    let wrap = wrapper::<M>(cfg);

    let pairs = cfg.selected_pairs();
    let snapshot = |w: &BoxTracker<M, Mod>| {
        pairs.iter().map(|&p| w.filter.marginal_2d(p)).collect::<Vec<_>>()
    };
    let n_steps = reference.steps();
    let save_at = snapshot_schedule(n_steps, cfg.n_snapshots);
    let mut saved_steps = Vec::new();
    let mut marginals = Vec::new();
    if save_at[0] {
        saved_steps.push(0);
        marginals.push(snapshot(&window));
    }
    let mut estimates = vec![wrap(window.estimate())];
    let mut centers = vec![wrap(window.center())];
    for (n, y) in reference.observations[..n_steps].iter().enumerate() {
        if progress.cancelled() {
            break;
        }
        window.forward(y);
        estimates.push(wrap(window.estimate()));
        centers.push(wrap(window.center()));
        let step = n + 1;
        if save_at[step] {
            marginals.push(snapshot(&window));
            saved_steps.push(step);
        }
        progress.step();
    }

    BoxOutput {
        window: FilterOutput {
            dt,
            labels,
            axes: (0..M).map(|d| window.filter.axis(d)).collect(),
            n_el: cfg.vars.iter().map(|v| v.win_n_el).collect(),
            p_ord: cfg.vars.iter().map(|v| v.p_ord).collect(),
            periodic: vec![false; M],
            dirichlet: vec![true; M],
            domain: cfg.vars.iter().map(|v| (-v.half_width, v.half_width)).collect(),
            pairs,
            saved_steps,
            marginals,
            reference: reference.states[..estimates.len()].to_vec(),
            estimates,
        },
        centers,
    }
}

// ── Job descriptions: model + reference + estimator + configuration,
//    dispatched through the model visitor. ───────────────────────────────

/// Which observer a job runs: the grid filter (density p on the spectral
/// grid), the translating window (the same density on a small box following
/// the mode — `box_tracker`), the particle approximation, the tracker (the
/// Gaussian closure at the maximum: one state and its curvature), or the
/// unscented closure (the same Gaussian matched by its integrals through a
/// quadrature rule). All share the configuration's initial data and model
/// noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Estimator {
    Filter,
    Window,
    Particles,
    Tracker,
    Unscented,
}

impl Estimator {
    /// All variants, in the order a selector shows them.
    pub const ALL: [Estimator; 5] =
        [Estimator::Filter, Estimator::Window, Estimator::Particles, Estimator::Tracker, Estimator::Unscented];
}

/// A complete, self-contained description of one observer run: the model
/// as data, the reference (trajectory + observations) to run along, the
/// estimator and its configuration. Plain data — what a viewer builds from
/// its forms and its run, and what a server receives; [`run`](Self::run)
/// executes it. The number of steps and the time step are the
/// reference's.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct JobSpec {
    pub model: ModelSpec,
    pub reference: Reference,
    pub estimator: Estimator,
    pub config: FilterConfig,
}

/// The output of a [`JobSpec`], one variant per [`Estimator`] (the
/// unscented closure produces the tracker's kind of output).
#[derive(Serialize, Deserialize)]
#[serde(tag = "estimator", content = "output", rename_all = "snake_case")]
pub enum JobOutput {
    Filter(FilterOutput),
    Window(BoxOutput),
    Particles(ParticleOutput),
    Tracker(TrackerOutput),
    Unscented(TrackerOutput),
}

impl JobOutput {
    /// The estimator that produced this output.
    pub fn estimator(&self) -> Estimator {
        match self {
            JobOutput::Filter(_) => Estimator::Filter,
            JobOutput::Window(_) => Estimator::Window,
            JobOutput::Particles(_) => Estimator::Particles,
            JobOutput::Tracker(_) => Estimator::Tracker,
            JobOutput::Unscented(_) => Estimator::Unscented,
        }
    }
}

/// The typed outputs can be extracted from a [`JobOutput`] (what a remote
/// run returns); asking for the wrong kind is an error naming both.
macro_rules! output_from_job {
    ($t:ty, $variant:ident, $name:literal) => {
        impl TryFrom<JobOutput> for $t {
            type Error = String;
            fn try_from(out: JobOutput) -> Result<Self, String> {
                match out {
                    JobOutput::$variant(x) => Ok(x),
                    other => Err(format!("expected the {} output, got the {:?} one", $name, other.estimator())),
                }
            }
        }
    };
}
output_from_job!(FilterOutput, Filter, "filter");
output_from_job!(BoxOutput, Window, "window");
output_from_job!(ParticleOutput, Particles, "particles");

/// The tracker's output type serves the tracker and the unscented closure.
impl TryFrom<JobOutput> for TrackerOutput {
    type Error = String;
    fn try_from(out: JobOutput) -> Result<Self, String> {
        match out {
            JobOutput::Tracker(x) | JobOutput::Unscented(x) => Ok(x),
            other => Err(format!("expected the tracker output, got the {:?} one", other.estimator())),
        }
    }
}

impl JobSpec {
    /// The job of a twin experiment: the reference generated from `twin`
    /// (bumping `progress` once per step), then `estimator` with `config`.
    pub fn from_twin(twin: &TwinSpec, estimator: Estimator, config: FilterConfig, progress: &Progress) -> JobSpec {
        JobSpec {
            model: twin.model.clone(),
            reference: twin.reference(progress),
            estimator,
            config,
        }
    }

    /// Number of steps of the run (the reference's).
    pub fn steps(&self) -> usize {
        self.reference.steps()
    }

    /// Run the described job to completion (or until `progress` is
    /// cancelled), bumping `progress` once per step. Panics if the
    /// configuration's or the reference's dimension is not the model's.
    pub fn run(&self, progress: &Progress) -> JobOutput {
        let (m, r, cfg) = (&self.model, &self.reference, &self.config);
        match self.estimator {
            Estimator::Filter => JobOutput::Filter(run_filter_spec(m, r, cfg, progress)),
            Estimator::Window => JobOutput::Window(run_box_spec(m, r, cfg, progress)),
            Estimator::Particles => JobOutput::Particles(run_particles_spec(m, r, cfg, progress)),
            Estimator::Tracker => JobOutput::Tracker(run_tracker_spec(m, r, cfg, progress)),
            Estimator::Unscented => JobOutput::Unscented(run_unscented_spec(m, r, cfg, progress)),
        }
    }
}

/// The visitor behind the `run_*_spec` functions: the reference, the
/// configuration and the progress handle of one run, applied to whatever
/// model the spec builds.
struct Runner<'a> {
    reference: &'a Reference,
    cfg: &'a FilterConfig,
    progress: &'a Progress,
}

macro_rules! spec_runner {
    ($(#[$doc:meta])* $name:ident, $visitor:ident, $run:ident, $out:ty) => {
        struct $visitor<'a>(Runner<'a>);
        impl ModelVisitor for $visitor<'_> {
            type Output = $out;
            fn visit<const M: usize, Mod: Model<M> + Send + Sync + 'static>(self, model: Mod) -> $out {
                $run::<M, _>(model, self.0.reference, self.0.cfg, self.0.progress)
            }
        }
        $(#[$doc])*
        pub fn $name(
            model: &ModelSpec,
            reference: &Reference,
            cfg: &FilterConfig,
            progress: &Progress,
        ) -> $out {
            assert_eq!(
                cfg.vars.len(),
                model.dim(),
                "FilterConfig has {} variables, the {} model {}",
                cfg.vars.len(),
                model.kind(),
                model.dim()
            );
            assert_eq!(
                reference.dim(),
                model.dim(),
                "the reference has {} components, the {} model {}",
                reference.dim(),
                model.kind(),
                model.dim()
            );
            model.visit(reference.dt, $visitor(Runner { reference, cfg, progress }))
        }
    };
}

spec_runner!(
    /// [`run_filter`] on the model a [`ModelSpec`] describes, built with the
    /// reference's time step.
    run_filter_spec, FilterVisitor, run_filter, FilterOutput
);
spec_runner!(
    /// [`run_tracker`] on the model a [`ModelSpec`] describes.
    run_tracker_spec, TrackerVisitor, run_tracker, TrackerOutput
);
spec_runner!(
    /// [`run_unscented`] on the model a [`ModelSpec`] describes.
    run_unscented_spec, UnscentedVisitor, run_unscented, TrackerOutput
);
spec_runner!(
    /// [`run_box`] on the model a [`ModelSpec`] describes.
    run_box_spec, BoxVisitor, run_box, BoxOutput
);
spec_runner!(
    /// [`run_particles`] on the model a [`ModelSpec`] describes.
    run_particles_spec, ParticlesVisitor, run_particles, ParticleOutput
);

#[cfg(test)]
mod tests {
    use super::*;
    use ode_models::models::{LorenzObservation, LorenzParams, SpringSystem};
    use ode_models_spec::noise::NoiseModel;

    fn spring(dt: f64) -> SpringSystem {
        SpringSystem::new(1, 1.0, 1.0, dt, |x: &[f64]| x[0])
    }

    /// The single spring's noiseless reference from (1.15, 0).
    fn spring_reference(dt: f64, steps: usize) -> Reference {
        Reference::twin(&spring(dt), [1.15, 0.0], steps, NoiseModel::None, 0, None, &Progress::default())
    }

    /// The configuration defaults of a spring run.
    fn spring_cfg(reference: &Reference) -> FilterConfig {
        FilterConfig::defaults(
            vec!["y_1".into(), "v_1".into()],
            vec![None, None],
            &[(0, 1)],
            &reference.states,
        )
    }

    /// The window path on the single spring: a box a quarter of the state
    /// range wide follows the reference (which swings beyond it), the
    /// window moves, the snapshots are the window's own (Dirichlet, (−L, L))
    /// grid, and the estimates stay close to the reference.
    #[test]
    fn spring_window_job_produces_outputs() {
        let r = spring_reference(0.01, 300);
        let mut cfg = spring_cfg(&r);
        for v in &mut cfg.vars {
            v.sigma = 20.0;
            v.half_width = 0.8;
            v.win_n_el = 5;
        }
        cfg.eps = 0.05;
        cfg.n_snapshots = 4;
        let progress = Progress::default();
        let out = run_box::<2, _>(spring(0.01), &r, &cfg, &progress);
        assert_eq!(progress.done(), 300);
        let w = &out.window;
        assert_eq!(w.dirichlet, vec![true, true]);
        assert_eq!(w.domain, vec![(-0.8, 0.8), (-0.8, 0.8)]);
        assert_eq!(w.axes[0].len(), 5 * 4 - 1);
        assert_eq!(w.saved_steps, vec![0, 100, 200, 300]);
        assert_eq!(out.centers.len(), 301);
        assert_eq!(w.estimates.len(), 301);
        assert_eq!(w.reference, r.states);
        assert!(w.marginals.iter().all(|m| m[0].iter().all(|v| v.is_finite())));
        // The window moved, and the estimate followed the reference.
        assert!(out.centers[0][0] != out.centers[300][0]);
        for (e, r) in w.estimates.iter().zip(&w.reference).skip(20) {
            assert!((e[0] - r[0]).abs() < 0.25, "estimate {e:?} vs reference {r:?}");
        }
    }

    /// A cancelled job stops before its next iteration and returns what it
    /// has: with the flag raised up front, neither the filter nor the tracker
    /// takes a single step (the counters stay at 0, the outputs hold t = 0
    /// only). This is what the viewer's Cancel button relies on.
    #[test]
    fn cancelled_jobs_stop_at_once() {
        let r = spring_reference(0.005, 20);
        let cfg = spring_cfg(&r);
        let progress = Progress::default();
        progress.cancel();
        let out = run_filter::<2, _>(spring(0.005), &r, &cfg, &progress);
        assert_eq!(progress.done(), 0);
        assert_eq!(out.estimates.len(), 1);
        assert_eq!(out.reference.len(), 1);
        let out = run_tracker::<2, _>(spring(0.005), &r, &cfg, &progress);
        assert_eq!(progress.done(), 0);
        assert_eq!(out.estimates.len(), 1);
    }

    /// End-to-end check of the viewer's filter path on the single spring:
    /// defaults from a run, filter execution, output shapes and finiteness.
    #[test]
    fn spring_filter_job_produces_outputs() {
        let r = spring_reference(0.005, 20);
        let cfg = spring_cfg(&r);
        assert_eq!(cfg.n_snapshots, 21);
        assert!(cfg.vars.iter().all(|v| v.domain.0 < v.domain.1));

        let progress = Progress::default();
        let out = run_filter::<2, _>(spring(0.005), &r, &cfg, &progress);

        assert_eq!(progress.done(), 20);
        assert_eq!(out.estimates.len(), 21);
        assert_eq!(out.reference, r.states);
        assert_eq!(out.reference[0].len(), 2);
        assert_eq!(out.pairs, vec![(0, 1)]);
        assert_eq!(out.saved_steps.len(), out.marginals.len());
        assert_eq!(out.saved_steps.first(), Some(&0));
        assert_eq!(out.saved_steps.last(), Some(&20));
        let (n0, n1) = (out.axes[0].len(), out.axes[1].len());
        for snap in &out.marginals {
            assert_eq!(snap[0].len(), n0 * n1);
        }
        let last = &out.marginals.last().unwrap()[0];
        assert!(last.iter().all(|v| v.is_finite()));
        assert!(last.iter().any(|&v| v > 0.0), "p degenerated to zero");
    }

    /// The tracker path of the viewer: same configuration, per-step
    /// estimates and covariances, converging toward the reference on the
    /// observed component.
    #[test]
    fn spring_tracker_job_produces_outputs() {
        let r = spring_reference(0.01, 200);
        let mut cfg = spring_cfg(&r);
        for v in &mut cfg.vars {
            v.sigma = 1.0;
            v.x0 += 0.3; // start off the reference
        }
        let progress = Progress::default();
        let out = run_tracker::<2, _>(spring(0.01), &r, &cfg, &progress);
        assert_eq!(progress.done(), 200);
        assert_eq!(out.estimates.len(), 201);
        assert_eq!(out.covariances.len(), 201);
        assert!(out.covariances[1].as_ref().is_some_and(|p| p.len() == 4));
        let err = |i: usize| (out.estimates[i][0] - out.reference[i][0]).abs();
        assert!(err(200) < 0.1 * err(0), "tracker did not converge: {} vs {}", err(200), err(0));
    }

    /// The per-variable Dirichlet walls of the GUI configuration must reach
    /// the filter: the axes drop the two boundary nodes (ndof = n·p − 1)
    /// and the run stays finite with positive mass.
    #[test]
    fn config_dirichlet_reaches_the_filter() {
        let r = spring_reference(0.02, 10);
        let mut cfg = spring_cfg(&r);
        for v in &mut cfg.vars {
            v.sigma = 5.0;
            v.dirichlet = true;
        }
        cfg.eps = 1.0;
        let out = run_filter::<2, _>(spring(0.02), &r, &cfg, &Progress::default());
        for d in 0..2 {
            assert!(out.dirichlet[d]);
            assert_eq!(
                out.axes[d].len(),
                cfg.vars[d].n_el * cfg.vars[d].p_ord - 1,
                "Dirichlet axis must drop the two boundary nodes"
            );
            let (lo, hi) = out.domain[d];
            assert!(
                out.axes[d].iter().all(|&x| x > lo && x < hi),
                "a Dirichlet axis stores a boundary node"
            );
        }
        let last = &out.marginals.last().unwrap()[0];
        assert!(last.iter().all(|v| v.is_finite()));
        assert!(last.iter().any(|&v| v > 0.0), "p degenerated to zero");
    }

    /// The per-variable model-noise variances of the GUI configuration must
    /// reach the filter: q = (1, 0) and q = (1, 1) give different densities.
    #[test]
    fn config_q_changes_the_filter_output() {
        let r = spring_reference(0.02, 10);
        let mut cfg = spring_cfg(&r);
        cfg.eps = 1.0;
        for v in &mut cfg.vars {
            v.sigma = 5.0;
        }
        let run = |q1: f64| {
            let mut cfg = cfg.clone();
            cfg.vars[1].q = q1;
            run_filter::<2, _>(spring(0.02), &r, &cfg, &Progress::default())
        };
        let (a, b) = (run(0.0), run(1.0));
        let (ma, mb) = (&a.marginals.last().unwrap()[0], &b.marginals.last().unwrap()[0]);
        let diff = ma.iter().zip(mb).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        let max = mb.iter().copied().fold(0.0, f64::max);
        assert!(diff / max > 1e-3, "q had no effect: relative difference {:e}", diff / max);
    }

    /// The single-spring twin experiment of the viewer's "Single spring"
    /// entry: Gaussian observation noise, seed 1000.
    fn spring_twin(dt: f64, steps: usize) -> TwinSpec {
        TwinSpec {
            model: ModelSpec::Spring { n: 1, rho: 1.0, a: 1.0, obs: 0 },
            x0: vec![0.15, 0.0],
            dt,
            steps,
            noise: NoiseModel::Gaussian { std: 0.01 },
            seed: 1000,
            walk: None,
        }
    }

    /// Running through the job description gives exactly the run of the
    /// hand-built model along the same reference: same estimates and
    /// marginals, bit for bit (the visitor rebuilds the very same system).
    #[test]
    fn spec_run_equals_the_hand_built_run() {
        let twin = spring_twin(0.02, 10);
        let r = twin.reference(&Progress::default());
        let cfg = spring_cfg(&r);
        let direct = run_filter::<2, _>(spring(0.02), &r, &cfg, &Progress::default());
        let via_spec = run_filter_spec(&twin.model, &r, &cfg, &Progress::default());
        assert_eq!(direct.estimates, via_spec.estimates);
        assert_eq!(direct.reference, via_spec.reference);
        assert_eq!(direct.saved_steps, via_spec.saved_steps);
        assert_eq!(direct.marginals, via_spec.marginals);
        assert_eq!(direct.axes, via_spec.axes);
    }

    /// A job description runs every estimator on the spring without any
    /// caller naming a model type or a dimension, and the output variant
    /// matches the requested estimator.
    #[test]
    fn job_spec_runs_every_estimator() {
        let twin = spring_twin(0.02, 10);
        let r = twin.reference(&Progress::default());
        let mut cfg = spring_cfg(&r);
        for v in &mut cfg.vars {
            v.sigma = 5.0;
            v.half_width = 0.8;
        }
        cfg.particles.n_particles = 20;
        for estimator in Estimator::ALL {
            let job = JobSpec::from_twin(&twin, estimator, cfg.clone(), &Progress::default());
            assert_eq!(job.steps(), 10);
            let progress = Progress::default();
            let out = job.run(&progress);
            assert_eq!(progress.done(), 10, "{estimator:?}");
            assert_eq!(out.estimator(), estimator);
            let (dt_out, reference) = match &out {
                JobOutput::Filter(o) => (o.dt, &o.reference),
                JobOutput::Window(o) => (o.window.dt, &o.window.reference),
                JobOutput::Particles(o) => (o.dt, &o.reference),
                JobOutput::Tracker(o) | JobOutput::Unscented(o) => (o.dt, &o.reference),
            };
            assert_eq!(dt_out, 0.02);
            assert_eq!(reference, &r.states);
        }
    }

    /// The unscented path of the viewer equals the tracker path on the
    /// linear spring, for every rule of the selector — same estimates and
    /// covariances to round-off — and produces the tracker's kind of output
    /// (the Tracker view's data), with the periodic wrap applied alike.
    #[test]
    fn spring_unscented_job_equals_the_tracker_job() {
        let r = spring_reference(0.01, 100);
        let mut cfg = spring_cfg(&r);
        for v in &mut cfg.vars {
            v.sigma = 1.0;
            v.x0 += 0.3;
        }
        cfg.eps = 0.02;
        let tracker = run_tracker::<2, _>(spring(0.01), &r, &cfg, &Progress::default());
        for rule in [
            Quadrature::DEFAULT,
            Quadrature::Cubature,
            Quadrature::DegreeFive,
            Quadrature::GaussHermite { points: 3 },
        ] {
            cfg.unscented.rule = rule;
            let progress = Progress::default();
            let out = run_unscented::<2, _>(spring(0.01), &r, &cfg, &progress);
            assert_eq!(progress.done(), 100);
            assert_eq!(out.estimates.len(), 101);
            for n in 0..101 {
                for d in 0..2 {
                    assert!((out.estimates[n][d] - tracker.estimates[n][d]).abs() < 1e-9, "{rule:?}, step {n}");
                }
                let (a, b) = (out.covariances[n].as_ref().unwrap(), tracker.covariances[n].as_ref().unwrap());
                for i in 0..4 {
                    assert!((a[i] - b[i]).abs() < 1e-9, "{rule:?}, step {n}: P differs");
                }
            }
        }
        // Through the job description and its JSON form (the rule travels).
        let twin = spring_twin(0.01, 20);
        cfg.unscented.rule = Quadrature::GaussHermite { points: 3 };
        let job = JobSpec::from_twin(&twin, Estimator::Unscented, cfg.clone(), &Progress::default());
        let json = serde_json::to_string(&job).unwrap();
        assert!(json.contains("\"estimator\":\"unscented\"") && json.contains("\"gauss_hermite\":{\"points\":3}"), "{json}");
        let back: JobSpec = serde_json::from_str(&json).unwrap();
        assert!(back == job);
        let out = back.run(&Progress::default());
        assert_eq!(out.estimator(), Estimator::Unscented);
        assert!(TrackerOutput::try_from(out).is_ok());
    }

    /// A seeded particle configuration is reproducible; an unseeded one is
    /// a fresh sample.
    #[test]
    fn particle_seed_makes_runs_reproducible() {
        let r = spring_reference(0.02, 10);
        let mut cfg = spring_cfg(&r);
        for v in &mut cfg.vars {
            v.sigma = 5.0;
        }
        cfg.particles = ParticleConfig { n_particles: 20, seed: Some(3) };
        let a = run_particles::<2, _>(spring(0.02), &r, &cfg, &Progress::default());
        let b = run_particles::<2, _>(spring(0.02), &r, &cfg, &Progress::default());
        assert_eq!(a.positions, b.positions);
        cfg.particles.seed = None;
        let c = run_particles::<2, _>(spring(0.02), &r, &cfg, &Progress::default());
        assert_ne!(a.positions, c.positions);
    }

    /// A job description and its output survive a JSON round trip: the
    /// description unchanged (its reference included), the output field by
    /// field — what a server receives and returns.
    #[test]
    fn job_spec_and_output_round_trip_through_json() {
        let twin = spring_twin(0.02, 10);
        let r = twin.reference(&Progress::default());
        let mut cfg = spring_cfg(&r);
        cfg.vars[1].dirichlet = true;
        cfg.diffusion = DiffusionScheme::Euler;
        let job = JobSpec::from_twin(&twin, Estimator::Tracker, cfg, &Progress::default());
        let json = serde_json::to_string(&job).unwrap();
        assert!(json.contains("\"estimator\":\"tracker\"") && json.contains("\"diffusion\":\"euler\""), "{json}");
        assert!(json.contains("\"observations\":[["), "{json}");
        let back: JobSpec = serde_json::from_str(&json).unwrap();
        assert!(back == job);

        let out = back.run(&Progress::default());
        let json = serde_json::to_string(&out).unwrap();
        assert!(json.starts_with("{\"estimator\":\"tracker\",\"output\":{"), "{}", &json[..60]);
        let (JobOutput::Tracker(a), JobOutput::Tracker(b)) =
            (&out, &serde_json::from_str::<JobOutput>(&json).unwrap())
        else {
            panic!("wrong variant");
        };
        assert_eq!(a.estimates, b.estimates);
        assert_eq!(a.covariances, b.covariances);
        assert_eq!(a.reference, b.reference);
        assert_eq!((a.dt, a.eps, &a.labels), (b.dt, b.eps, &b.labels));
    }

    /// A configuration of the wrong dimension is refused before anything
    /// is built.
    #[test]
    #[should_panic(expected = "FilterConfig has 2 variables")]
    fn job_spec_checks_the_dimension() {
        let r = spring_reference(0.01, 1);
        let cfg = spring_cfg(&r);
        let model = ModelSpec::Lorenz {
            params: LorenzParams::default(),
            observation: LorenzObservation::X,
        };
        run_tracker_spec(&model, &r, &cfg, &Progress::default());
    }
}

/// Anisotropy check through the viewer's own pipeline: the standard
/// deviation of the stored max-marginal in each direction.
#[cfg(test)]
mod anisotropy {
    use super::*;
    use ode_models::models::SpringSystem;
    use ode_models_spec::noise::NoiseModel;

    /// (std_y, std_v) of the max-marginal of plane 0 at snapshot `k`.
    fn widths(out: &FilterOutput, k: usize) -> (f64, f64) {
        let (na, nb) = (out.axes[0].len(), out.axes[1].len());
        let m = &out.marginals[k][0];
        let mut w = [0.0; 2];
        for (d, n, other) in [(0usize, na, nb), (1, nb, na)] {
            let ax = &out.axes[d];
            let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
            for i in 0..n {
                let acc: f64 = (0..other)
                    .map(|j| m[if d == 0 { i + j * na } else { j + i * na }])
                    .sum();
                s0 += acc;
                s1 += acc * ax[i];
                s2 += acc * ax[i] * ax[i];
            }
            let mean = s1 / s0;
            w[d] = (s2 / s0 - mean * mean).max(0.0).sqrt();
        }
        (w[0], w[1])
    }

    /// With q = (100, 0) the density spreads in y only, with q = (0, 100) in
    /// v only — over a short time (20 steps ≪ the period 2π, before the
    /// oscillator's rotation mixes the two directions). Measured on the
    /// viewer's stored marginals, i.e. through `FilterConfig` → `run_filter`.
    #[test]
    fn q_from_the_config_diffuses_anisotropically() {
        let dt = 0.005;
        let n = 20;
        let sys = || SpringSystem::new(1, 1.0, 1.0, dt, |x: &[f64]| x[0]);
        let r = Reference::twin(&sys(), [1.15, 0.0], n, NoiseModel::None, 0, None, &Progress::default());
        let mut cfg = FilterConfig::defaults(
            vec!["y".into(), "v".into()],
            vec![None, None],
            &[(0, 1)],
            &r.states,
        );
        for v in &mut cfg.vars {
            v.sigma = 10.0;
            v.domain = (-3.0, 3.0);
            v.n_el = 20;
            v.p_ord = 4;
        }
        let run = |q: [f64; 2]| {
            let mut c = cfg.clone();
            c.vars[0].q = q[0];
            c.vars[1].q = q[1];
            let out = run_filter::<2, _>(sys(), &r, &c, &Progress::default());
            widths(&out, n)
        };
        let (y_only_y, y_only_v) = run([100.0, 0.0]);
        let (v_only_y, v_only_v) = run([0.0, 100.0]);
        // Measured: (0.278, 0.000) and (0.044, 0.324).
        assert!(y_only_y > 0.2 && y_only_v < 0.05, "q = (100, 0): {y_only_y} / {y_only_v}");
        assert!(v_only_v > 0.2 && v_only_y < 0.1, "q = (0, 100): {v_only_y} / {v_only_v}");
    }
}
