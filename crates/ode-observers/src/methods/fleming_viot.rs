//! Particle approximation of the Mortensen filter: a Fleming–Viot-type
//! population, after §3 of the note "Links between Mortensen observer and
//! population dynamics" (`bib/Notes_Mortensen_et_dynamique_des_populations.pdf`).
//!
//! The density p is represented by the empirical measure of N particles
//! Z^i ∈ ℝ^M. One step of [`ParticleSystem::forward`] mirrors the filter's
//! cycle:
//!
//! 1. **Observation — killing**: every particle carries a *budget* e_i,
//!    drawn from the exponential law of parameter 1 at its creation, and a
//!    running *misfit* a_i. With the observation y_n of the step (given to
//!    [`ParticleSystem::forward`]) and the model's discrepancy d(y_n, h(x))
//!    (the squared distance the filter multiplies p by exp(−dt·γ·d/2ε)
//!    with), the misfit grows by
//!
//!      a_i += dt · γ · d(y_n, h(Z^i)),
//!
//!    and the particle dies when a_i ≥ e_i — the note's (1.25): the death
//!    time is the first time the accumulated misfit exceeds an exponential
//!    clock, so it cannot be known in advance. A large γ or a large
//!    discrepancy kills fast; γ = 0 makes the particles immortal.
//! 2. **Births**: each dead particle is resurrected at the position of a
//!    survivor of the step, drawn uniformly, with a fresh budget and a
//!    misfit reset to 0; N stays constant. All deaths of a step are
//!    decided before any birth (order-independent). If no particle
//!    survives the run panics: the misfit of the whole cloud exceeded the
//!    clocks in one step (γ·dt·d too large, or N too small).
//! 3. **Move**: every particle, the newborns included, follows the model's
//!    exact discrete flow and receives a Brownian increment,
//!
//!      Z^i ← φ(Z^i) + s·√(ε q_d dt)·G^i_d   (G^i_d standard normal, per direction),
//!
//!    the Euler–Maruyama step (1.24)/(3.4) of the note with the flow in
//!    place of the Euler drift and the variance ε q_d dt of the filter's
//!    diffusion (ε/2) Σ_d q_d ∂²_d, times a scale s (1 = the filter's
//!    diffusion, 0 = no noise). Periodic directions are wrapped.
//!
//! The initial population is drawn from the filter's Gaussian initial
//! data. Everything is deterministic in the seed. Compared with the note's
//! rate ε⁻¹ f of (3.5), the killing rate here is γ·d without the factor
//! 1/(2ε) of the filter's observation step: the population is killed at
//! the rate of the energy \eqref{eq:energy}, not of the density's misfit
//! factor, so that a small ε does not make the jump frequency explode
//! (the note's own caveat).
//!
//! **Parallelism and randomness** (2026-09): steps 1, 2 and 3 run on
//! rayon's thread pool, particle by particle. Every particle has its own
//! random stream at every step — a `SplitMix64` seeded by mixing the run
//! seed, the step number, the particle index and the purpose (birth or
//! move), see `stream` — so the Brownian increments, and the parent and
//! clock of a birth, are drawn inside the parallel loops without any
//! shared state. A run is deterministic in its seed and independent of the
//! number of threads (test `runs_do_not_depend_on_the_thread_count`);
//! its random sequence differs from the former sequential implementation,
//! which drew everything from one stream in index order, with the same
//! statistics. Only the initial draw uses one sequential stream.

use super::Observer;
use ode_models::model::Model;
use ode_models_spec::rng::SplitMix64;
use rayon::prelude::*;

/// Parameters of the [`ParticleSystem`]. Deliberately no `Default`.
#[derive(Clone, Copy, Debug)]
pub struct ParticleParams<const M: usize> {
    /// Number of particles N (constant along the run).
    pub n_particles: usize,
    /// Centre x_c of the Gaussian initial draw (the filter's initial data).
    pub center: [f64; M],
    /// Stiffness σ_d of the initial data: particles are drawn with variance
    /// ε/σ_d in direction d (the prior p₀ ∝ exp(−σ_d (x_d − x_{c,d})²/2ε)
    /// of the filter); σ_d = 0 (no prior) draws direction d uniformly in
    /// `domain[d]` instead.
    pub sigma: [f64; M],
    /// Box per direction (the filter's domains in the viewer): the range of
    /// the uniform draw in directions without prior, and the period of the
    /// periodic directions, onto which positions are wrapped. Particles are
    /// otherwise free to leave it.
    pub domain: [(f64, f64); M],
    /// Per-direction periodicity: positions are wrapped onto `domain[d]`
    /// after every move.
    pub periodic: [bool; M],
    /// ε of the filter: the Brownian increment has variance ε q_d dt.
    pub eps: f64,
    /// Diagonal of the model-noise covariance Q (as `FilterParams::q_diag`).
    pub q_diag: [f64; M],
    /// Multiplier s of the Brownian increment (1 = the filter's diffusion,
    /// 0 = pure flow).
    pub noise_scale: f64,
    /// Observation weight γ: the misfit of a particle grows by
    /// dt·γ·d(y, h(Z)) per step; 0 switches the killing off.
    pub gamma: f64,
    /// Seed of the deterministic generator (initial draw, noise, budgets,
    /// parents).
    pub seed: u64,
}

/// The population of particles driving a [`Model`]'s flow.
pub struct ParticleSystem<const M: usize, Mod: Model<M>> {
    /// The forward model (parameters and maps).
    pub model: Mod,
    /// Parameters the system was built with.
    pub params: ParticleParams<M>,
    positions: Vec<[f64; M]>,
    /// Budget e_i ~ Exp(1) of every particle, drawn at its creation.
    budget: Vec<f64>,
    /// Accumulated misfit a_i = Σ dt·γ·d of every particle since its
    /// creation; it dies when a_i ≥ e_i.
    spent: Vec<f64>,
    step: usize,
    births: usize,
}

/// The random stream of particle `i` at step `step` for `purpose` (0 =
/// move, 1 = birth): a `SplitMix64` seeded by mixing the run seed with the
/// three indices, so that no two (step, particle, purpose) share a stream
/// and no state is shared between threads.
fn stream(seed: u64, step: usize, i: usize, purpose: u64) -> SplitMix64 {
    let mut mix = SplitMix64::new(
        seed ^ (step as u64).wrapping_mul(0xA24B_AED4_963E_E407)
            ^ (i as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25)
            ^ purpose.wrapping_mul(0xD6E8_FEB8_6659_FD93),
    );
    SplitMix64::new(mix.next_u64())
}

impl<const M: usize, Mod: Model<M> + Sync> ParticleSystem<M, Mod> {
    /// Draw the initial population: N positions from the Gaussian prior
    /// (centre `center`, variance ε/σ_d per direction; uniform in the box
    /// where σ_d = 0; periodic directions wrapped), each with a budget
    /// e ~ Exp(1) and no misfit yet.
    pub fn new(model: Mod, params: ParticleParams<M>) -> Self {
        assert!(params.n_particles >= 2, "at least two particles are needed (births copy a survivor)");
        assert!(
            params.eps >= 0.0 && params.noise_scale >= 0.0 && params.gamma >= 0.0,
            "eps, noise_scale and gamma must be ≥ 0"
        );
        assert!(
            (0..M).all(|d| {
                params.domain[d].0 < params.domain[d].1 && params.q_diag[d] >= 0.0 && params.sigma[d] >= 0.0
            }),
            "each domain needs min < max, q_d ≥ 0 and σ_d ≥ 0"
        );
        let mut sys = ParticleSystem {
            model,
            params,
            positions: Vec::new(),
            budget: Vec::new(),
            spent: Vec::new(),
            step: 0,
            births: 0,
        };
        sys.draw_initial();
        sys
    }

    /// (Re)draw the initial population from `params` (centre, stiffness,
    /// box, seed): N positions, N budgets, no misfit, step 0.
    fn draw_initial(&mut self) {
        let params = self.params;
        let mut rng = SplitMix64::new(params.seed);
        let n = params.n_particles;
        let positions = (0..n)
            .map(|_| {
                std::array::from_fn(|d| {
                    let (lo, hi) = params.domain[d];
                    let x = if params.sigma[d] > 0.0 {
                        params.center[d] + (params.eps / params.sigma[d]).sqrt() * rng.normal()
                    } else {
                        lo + (hi - lo) * rng.uniform()
                    };
                    if params.periodic[d] { lo + (x - lo).rem_euclid(hi - lo) } else { x }
                })
            })
            .collect();
        let budget: Vec<f64> = (0..n).map(|_| draw_budget(&mut rng)).collect();
        self.positions = positions;
        self.budget = budget;
        self.spent = vec![0.0; n];
        self.step = 0;
        self.births = 0;
    }

    /// Empirical mean of the cloud, the estimate the common
    /// [`Observer`] surface reports (a plain mean: on a periodic direction
    /// it is only meaningful while the cloud does not straddle the seam).
    pub fn mean(&self) -> [f64; M] {
        let n = self.positions.len() as f64;
        std::array::from_fn(|d| self.positions.iter().map(|z| z[d]).sum::<f64>() / n)
    }

    /// Particle positions (length N).
    pub fn positions(&self) -> &[[f64; M]] {
        &self.positions
    }

    /// Budget e_i of every particle (its exponential clock).
    pub fn budgets(&self) -> &[f64] {
        &self.budget
    }

    /// Accumulated misfit a_i of every particle since its creation
    /// (always < e_i at the end of a step: the others were reborn).
    pub fn spent(&self) -> &[f64] {
        &self.spent
    }

    /// Steps taken so far.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Total number of resurrections so far.
    pub fn births(&self) -> usize {
        self.births
    }

    /// One step, given the observation `y` of this step: killing by the
    /// observation, births, move.
    ///
    /// Panics if no particle survives the step.
    pub fn forward(&mut self, y: &[f64]) {
        let dt = self.model.dt();
        let gamma = self.params.gamma;

        // 1. Observation: the misfit of every particle grows by dt·γ·d at
        //    its current position, with the observation y. The
        //    discrepancies are pure functions of the positions: evaluated
        //    in parallel, then applied in index order.
        if gamma > 0.0 {
            let misfit: Vec<f64> =
                self.positions.par_iter().map(|z| dt * gamma * self.model.discrepancy(y, z)).collect();
            for (a, m) in self.spent.iter_mut().zip(misfit) {
                *a += m;
            }
        }
        let (dead, alive): (Vec<usize>, Vec<usize>) =
            (0..self.positions.len()).partition(|&i| self.spent[i] >= self.budget[i]);
        assert!(
            !alive.is_empty(),
            "particle system: every particle died at step {} (N = {}): the misfit dt·γ·d of the \
             whole cloud exceeded its clocks in one step — reduce γ or dt, or increase N",
            self.step + 1,
            self.params.n_particles
        );

        // 2. Births: each dead particle copies a survivor drawn uniformly
        //    (deaths were all decided above: order-independent), with its
        //    own stream for the parent and the fresh clock.
        let seed = self.params.seed;
        let step = self.step;
        let births: Vec<(usize, [f64; M], f64)> = dead
            .par_iter()
            .map(|&i| {
                let mut r = stream(seed, step, i, 1);
                let parent = alive[(r.uniform() * alive.len() as f64).floor().min(alive.len() as f64 - 1.0) as usize];
                (i, self.positions[parent], draw_budget(&mut r))
            })
            .collect();
        for (i, position, budget) in births {
            self.positions[i] = position;
            self.budget[i] = budget;
            self.spent[i] = 0.0;
        }
        self.births += dead.len();

        // 3. Move: exact flow, then the Brownian increment from the
        //    particle's own stream, then the wrap — all in parallel.
        let p = &self.params;
        let model = &self.model;
        let std: [f64; M] = std::array::from_fn(|d| p.noise_scale * (p.eps * p.q_diag[d] * dt).sqrt());
        self.positions.par_iter_mut().enumerate().for_each(|(i, z)| {
            let mut r = stream(seed, step, i, 0);
            let mut y = model.flow(*z);
            for d in 0..M {
                if std[d] > 0.0 {
                    y[d] += std[d] * r.normal();
                }
                if p.periodic[d] {
                    let (lo, hi) = p.domain[d];
                    y[d] = lo + (y[d] - lo).rem_euclid(hi - lo);
                }
            }
            *z = y;
        });
        self.step += 1;
    }
}

/// A budget e ~ Exp(1): −ln u with u uniform in (0, 1].
fn draw_budget(rng: &mut SplitMix64) -> f64 {
    -rng.uniform().ln()
}

/// The common observer surface: the Gaussian prior redraws the cloud
/// (same seed: the same population as a fresh system with these
/// parameters); estimate = the empirical mean.
impl<const M: usize, Mod: Model<M> + Sync> Observer<M> for ParticleSystem<M, Mod> {
    fn dt(&self) -> f64 {
        self.model.dt()
    }

    fn init_gaussian(&mut self, center: [f64; M], sigma: [f64; M]) {
        assert!(sigma.iter().all(|&s| s >= 0.0), "sigma must be nonnegative");
        self.params.center = center;
        self.params.sigma = sigma;
        self.draw_initial();
    }

    fn forward(&mut self, y: &[f64]) {
        ParticleSystem::forward(self, y);
    }

    fn estimate(&self) -> [f64; M] {
        self.mean()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ode_models::models::SpringSystem;
    use ode_models_spec::noise::NoiseModel;
    use ode_models_spec::progress::Progress;
    use ode_models_spec::reference::Reference;
    use std::f64::consts::PI;

    fn spring(dt: f64) -> SpringSystem {
        SpringSystem::new(1, 1.0, 1.0, dt, |x: &[f64]| x[0])
    }

    /// The observations of the spring's noiseless reference from (1.15, 0).
    fn observations(dt: f64, steps: usize) -> Vec<Vec<f64>> {
        Reference::twin(&spring(dt), [1.15, 0.0], steps, NoiseModel::None, 0, None, &Progress::default()).observations
    }

    /// No prior (σ = 0): the initial draw is uniform in the box.
    fn params(n: usize, s: f64, gamma: f64, seed: u64) -> ParticleParams<2> {
        ParticleParams {
            n_particles: n,
            center: [0.0; 2],
            sigma: [0.0; 2],
            domain: [(-2.0, 2.0); 2],
            periodic: [false; 2],
            eps: 0.05,
            q_diag: [1.0; 2],
            noise_scale: s,
            gamma,
            seed,
        }
    }

    /// With a prior the initial population is Gaussian: mean x_c and
    /// variance ε/σ_d per direction (to sampling accuracy), and a direction
    /// with σ_d = 0 is uniform in its box.
    #[test]
    fn initial_draw_follows_the_gaussian_prior() {
        let n = 40_000;
        let mut p = params(n, 0.0, 0.0, 5);
        p.center = [0.7, -0.3];
        p.sigma = [2.0, 0.0];
        let sys = ParticleSystem::<2, _>::new(spring(0.01), p);
        let z = sys.positions();
        let mean = |d: usize| z.iter().map(|x| x[d]).sum::<f64>() / n as f64;
        let var = |d: usize| {
            let m = mean(d);
            z.iter().map(|x| (x[d] - m).powi(2)).sum::<f64>() / n as f64
        };
        let v0 = p.eps / p.sigma[0]; // 0.025
        assert!((mean(0) - 0.7).abs() < 4.0 * (v0 / n as f64).sqrt(), "mean {}", mean(0));
        assert!((var(0) - v0).abs() < 0.05 * v0, "variance {} vs {v0}", var(0));
        // Direction 1: uniform on (−2, 2): mean 0, variance 4/3.
        assert!(mean(1).abs() < 0.05 && (var(1) - 4.0 / 3.0).abs() < 0.05, "uniform: {} {}", mean(1), var(1));
        // Budgets are exponential of mean 1.
        let mean_e = sys.budgets().iter().sum::<f64>() / n as f64;
        assert!((mean_e - 1.0).abs() < 0.03, "budget mean {mean_e}");
    }

    /// Without noise and without observations (γ = 0) every particle is
    /// exactly the discrete flow of its initial position, and nobody dies.
    #[test]
    fn without_noise_and_observations_particles_follow_the_flow() {
        let mut sys = ParticleSystem::<2, _>::new(spring(0.01), params(50, 0.0, 0.0, 1));
        let z0 = sys.positions().to_vec();
        let flow = spring(0.01);
        for y in &observations(0.01, 40)[..40] {
            sys.forward(y);
        }
        for (z, &z0) in sys.positions().iter().zip(&z0) {
            let mut y = z0;
            for _ in 0..40 {
                y = flow.flow(y);
            }
            assert!((z[0] - y[0]).abs() < 1e-12 && (z[1] - y[1]).abs() < 1e-12, "{z:?} vs {y:?}");
        }
        assert_eq!(sys.births(), 0);
    }

    /// With observations on (γ > 0): particles far from the observed
    /// position die and are reborn on survivors; N is constant, every
    /// surviving misfit stays below its budget, and (without noise) each
    /// newborn sits exactly on another particle at the end of the step.
    #[test]
    fn misfit_kills_and_births_copy_a_survivor() {
        let n = 300;
        let mut sys = ParticleSystem::<2, _>::new(spring(0.05), params(n, 0.0, 5.0, 7));
        for y in &observations(0.05, 40)[..40] {
            sys.forward(y);
            assert_eq!(sys.positions().len(), n);
            let (pos, e, a) = (sys.positions(), sys.budgets(), sys.spent());
            for i in 0..n {
                assert!(a[i] < e[i], "particle {i} outlived its budget");
                if a[i] == 0.0 {
                    assert!(
                        (0..n).any(|j| j != i && pos[j] == pos[i]),
                        "newborn {i} at {:?} has no parent", pos[i]
                    );
                }
            }
        }
        assert!(sys.births() > 0, "no death in 40 steps with γ = 5");
        // The cloud has been pulled toward the observed position y₁ = 1.15·cos t.
        let t: f64 = 40.0 * 0.05;
        let y_obs = 1.15 * t.cos();
        let mean_y = sys.positions().iter().map(|z| z[0]).sum::<f64>() / n as f64;
        assert!((mean_y - y_obs).abs() < 0.5, "cloud mean {mean_y} vs observed {y_obs}");
    }

    /// Same seed, same cloud; another seed, another cloud.
    #[test]
    fn runs_are_reproducible_in_the_seed() {
        let run = |seed| {
            let mut sys = ParticleSystem::<2, _>::new(spring(0.01), params(100, 1.0, 1.0, seed));
            for y in &observations(0.01, 25)[..25] {
                sys.forward(y);
            }
            sys.positions().to_vec()
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3), run(4));
    }

    /// The parallel evaluations (discrepancies, flows) are pure per-particle
    /// functions and every random draw stays in index order: a run does not
    /// depend on the number of threads, bit for bit.
    #[test]
    fn runs_do_not_depend_on_the_thread_count() {
        let run = || {
            let mut sys = ParticleSystem::<2, _>::new(spring(0.01), params(300, 1.0, 1.0, 5));
            for y in &observations(0.01, 25)[..25] {
                sys.forward(y);
            }
            (sys.positions().to_vec(), sys.spent().to_vec(), sys.budgets().to_vec(), sys.births())
        };
        let pool = |n| rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
        let one = pool(1).install(run);
        let four = pool(4).install(run);
        assert!(one.3 > 0, "no birth in the test run");
        assert_eq!(one, four);
    }

    /// Every particle has its own stream: particles started at the same
    /// point receive different increments at the first step, and the same
    /// particle receives different increments at successive steps (the
    /// spring's flow is linear, so equal positions would stay equal
    /// without noise).
    #[test]
    fn particles_have_independent_streams() {
        let mut p = params(200, 1.0, 0.0, 9); // unit noise, no killing
        p.sigma = [1e12; 2]; // a Dirac initial draw
        let mut sys = ParticleSystem::<2, _>::new(spring(0.01), p);
        let start = sys.positions().to_vec();
        assert!(start.iter().all(|z| (z[0] - start[0][0]).abs() < 1e-5));
        // Increment of every particle at a step = new position − flow(old).
        let increments = |sys: &ParticleSystem<2, SpringSystem>, before: &[[f64; 2]]| -> Vec<f64> {
            sys.positions().iter().zip(before).map(|(z, b)| z[0] - sys.model.flow(*b)[0]).collect()
        };
        sys.forward(&[1.15]);
        let first = sys.positions().to_vec();
        let inc1 = increments(&sys, &start);
        let mut sorted = inc1.clone();
        sorted.sort_by(f64::total_cmp);
        sorted.dedup();
        assert_eq!(sorted.len(), 200, "some particles received the same increment");
        sys.forward(&[1.15]);
        let inc2 = increments(&sys, &first);
        assert!(inc1.iter().zip(&inc2).all(|(a, b)| (a - b).abs() > 1e-12), "a stream repeated across steps");
    }

    /// Periodic directions are wrapped onto the box after every move.
    #[test]
    fn periodic_directions_are_wrapped() {
        let mut p = params(100, 20.0, 0.0, 11);
        p.domain[0] = (-PI, PI);
        p.periodic[0] = true;
        let mut sys = ParticleSystem::<2, _>::new(spring(0.01), p);
        for y in &observations(0.01, 20)[..20] {
            sys.forward(y);
            assert!(sys.positions().iter().all(|z| (-PI..PI).contains(&z[0])));
        }
    }
}
