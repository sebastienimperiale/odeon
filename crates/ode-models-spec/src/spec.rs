//! Dimension-erased model descriptions ([`ModelSpec`]) and the visitor that
//! turns one into a concrete [`Model<M>`] ([`ModelVisitor`]).
//!
//! The observers are generic over the state dimension `M` (a const generic,
//! because the grid works on `[f64; M]` points) and over the model type.
//! A user interface or a job server, on the other hand, handles *values*:
//! a model chosen from a menu with its parameters, initial state,
//! observation choice and observation noise. `ModelSpec` is that value —
//! plain data, one variant per model of [`ode_models::models`], with no closure
//! and no generic — and [`ModelSpec::visit`] is the bridge: it builds the
//! concrete system and hands it to a [`ModelVisitor`] whose `visit` method
//! is generic over `M` and the model type. The visitor is where an observer
//! gets instantiated at the right compile-time dimension; this crate knows
//! nothing about observers.
//!
//! A `ModelSpec` describes the model only — parameters and observation
//! choice, no state. A twin experiment on it (initial state, steps, noise
//! and seeds, optional walk) is a [`TwinSpec`], whose
//! [`reference`](TwinSpec::reference) is the [`Reference`] the observers
//! consume; both
//! are plain serde data, so a description sent to a server reproduces
//! what a viewer computed locally.

use serde::{Deserialize, Serialize};
use ode_models::model::Model;
use ode_models::models::{
    KeplerMuParams, KeplerMuSystem, KeplerObservation, KeplerParams, KeplerSystem, LamppostSystem,
    LorenzObservation, LorenzParams, LorenzSystem, PendulumParams, PendulumRodParams,
    PendulumRodSystem, PendulumSystem, SpringMassParams, SpringMassSystem, SpringSystem,
};
use crate::noise::NoiseModel;
use crate::progress::Progress;
use crate::reference::{Reference, Walk};

/// A forward model as plain data: which model, its physical parameters and
/// its observation choice. Built into a concrete system by
/// [`visit`](Self::visit); the time step is passed at that point, not
/// stored here — it is a run setting shared with the observer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelSpec {
    /// [`SpringSystem`]: the linear chain of `n` masses (state dimension
    /// 2n), observing the flat component `obs` of [Y; V].
    Spring { n: usize, rho: f64, a: f64, obs: usize },
    /// [`SpringMassSystem`]: the chain of `n` ∈ {1, 2, 3} bodies with an
    /// unknown free-end mass, augmented state (y, v, θ) of dimension
    /// 2n + 1; `obs` is the index of the observed position.
    SpringMass { n: usize, params: SpringMassParams, obs: usize },
    /// [`PendulumSystem`]: the double pendulum with point masses, state
    /// (q₁, q₂, p₁, p₂), observing the tip position.
    Pendulum { params: PendulumParams },
    /// [`PendulumRodSystem`]: the double compound pendulum, same state.
    PendulumRod { params: PendulumRodParams },
    /// [`KeplerSystem`]: the softened Kepler problem, state (q₁, q₂, p₁, p₂).
    Kepler { params: KeplerParams, observation: KeplerObservation },
    /// [`KeplerMuSystem`]: Kepler with the unknown μ = μ₀·2^θ, augmented
    /// state (q₁, q₂, p₁, p₂, θ).
    KeplerMu { params: KeplerMuParams, observation: KeplerObservation },
    /// [`LorenzSystem`]: Lorenz-63, state (x, y, z).
    Lorenz { params: LorenzParams, observation: LorenzObservation },
    /// [`LamppostSystem`]: the point observed through its squared distance
    /// to the origin, identity flow.
    Lamppost,
}

/// A computation generic over the model: [`ModelSpec::visit`] builds the
/// system described by the spec and calls [`visit`](Self::visit) with it,
/// at the model's compile-time dimension `M`. Implement it with a struct
/// holding whatever the computation needs (an observer configuration, a
/// reference, a progress handle) and returning its output.
pub trait ModelVisitor {
    type Output;

    fn visit<const M: usize, Mod>(self, model: Mod) -> Self::Output
    where
        Mod: Model<M> + Send + Sync + 'static;
}

impl ModelSpec {
    /// State dimension `M` of the described model.
    pub fn dim(&self) -> usize {
        match self {
            ModelSpec::Spring { n, .. } => 2 * n,
            ModelSpec::SpringMass { n, .. } => 2 * n + 1,
            ModelSpec::Pendulum { .. } | ModelSpec::PendulumRod { .. } | ModelSpec::Kepler { .. } => 4,
            ModelSpec::KeplerMu { .. } => 5,
            ModelSpec::Lorenz { .. } => 3,
            ModelSpec::Lamppost => 2,
        }
    }

    /// Short stable identifier of the model kind (`"spring"`,
    /// `"kepler_mu"`, …), the name a job description or a server route
    /// can key on — also the JSON tag of the spec.
    pub fn kind(&self) -> &'static str {
        match self {
            ModelSpec::Spring { .. } => "spring",
            ModelSpec::SpringMass { .. } => "spring_mass",
            ModelSpec::Pendulum { .. } => "pendulum",
            ModelSpec::PendulumRod { .. } => "pendulum_rod",
            ModelSpec::Kepler { .. } => "kepler",
            ModelSpec::KeplerMu { .. } => "kepler_mu",
            ModelSpec::Lorenz { .. } => "lorenz",
            ModelSpec::Lamppost => "lamppost",
        }
    }

    /// Build the described system with time step `dt` and hand it to
    /// `visitor` at its compile-time dimension.
    ///
    /// # Panics
    ///
    /// On a chain length outside the supported range (springs and
    /// unknown-mass chains: 1–3) — a malformed description, not a runtime
    /// condition.
    pub fn visit<V: ModelVisitor>(&self, dt: f64, visitor: V) -> V::Output {
        match self {
            ModelSpec::Spring { n, rho, a, obs } => {
                let obs = *obs;
                let sys = SpringSystem::new(*n, *rho, *a, dt, move |x: &[f64]| x[obs]);
                match n {
                    1 => visitor.visit::<2, _>(sys),
                    2 => visitor.visit::<4, _>(sys),
                    3 => visitor.visit::<6, _>(sys),
                    _ => panic!("spring chains of 1 to 3 masses are supported, got n = {n}"),
                }
            }
            ModelSpec::SpringMass { n, params, obs } => match n {
                1 => visitor.visit::<3, _>(SpringMassSystem::<1>::with_params(*params, dt, *obs)),
                2 => visitor.visit::<5, _>(SpringMassSystem::<2>::with_params(*params, dt, *obs)),
                3 => visitor.visit::<7, _>(SpringMassSystem::<3>::with_params(*params, dt, *obs)),
                _ => panic!("unknown-mass chains of 1 to 3 bodies are supported, got n = {n}"),
            },
            ModelSpec::Pendulum { params } => visitor.visit::<4, _>(PendulumSystem::with_params(*params, dt)),
            ModelSpec::PendulumRod { params } => {
                visitor.visit::<4, _>(PendulumRodSystem::with_params(*params, dt))
            }
            ModelSpec::Kepler { params, observation } => {
                visitor.visit::<4, _>(KeplerSystem::with_params(*params, dt, *observation))
            }
            ModelSpec::KeplerMu { params, observation } => {
                visitor.visit::<5, _>(KeplerMuSystem::with_params(*params, dt, *observation))
            }
            ModelSpec::Lorenz { params, observation } => {
                visitor.visit::<3, _>(LorenzSystem::with_params(*params, dt, *observation))
            }
            ModelSpec::Lamppost => visitor.visit::<2, _>(LamppostSystem::new(dt)),
        }
    }
}

/// A twin experiment as plain data: a model, the initial state of its
/// reference, the run length and time step, the observation noise with its
/// seed, and an optional random walk of the reference. Everything
/// [`Reference::twin`] needs, so that the same description reproduces the
/// same [`Reference`] anywhere.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwinSpec {
    pub model: ModelSpec,
    /// Initial state x₀ (length `model.dim()`).
    pub x0: Vec<f64>,
    pub dt: f64,
    pub steps: usize,
    pub noise: NoiseModel,
    /// Seed of the observation noise (component j uses `seed + j`).
    pub seed: u64,
    pub walk: Option<Walk>,
}

impl TwinSpec {
    /// Generate the reference (bumping `progress` once per step; a
    /// cancelled progress stops early).
    ///
    /// # Panics
    ///
    /// If `x0` does not have the model's dimension.
    pub fn reference(&self, progress: &Progress) -> Reference {
        assert_eq!(self.x0.len(), self.model.dim(), "x0 has {} entries, the {} model {}", self.x0.len(), self.model.kind(), self.model.dim());
        struct Twin<'a>(&'a TwinSpec, &'a Progress);
        impl ModelVisitor for Twin<'_> {
            type Output = Reference;
            fn visit<const M: usize, Mod: Model<M>>(self, model: Mod) -> Reference {
                let t = self.0;
                let x0: [f64; M] = std::array::from_fn(|d| t.x0[d]);
                Reference::twin(&model, x0, t.steps, t.noise, t.seed, t.walk, self.1)
            }
        }
        self.model.visit(self.dt, Twin(self, progress))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ode_models::models::kepler::perihelion_state;
    use ode_models::models::kepler_mu::augmented_state;

    /// A visitor reporting the compile-time dimension it was called at and
    /// the model's own state dimension.
    struct Dims;
    impl ModelVisitor for Dims {
        type Output = (usize, usize, usize);
        fn visit<const M: usize, Mod: Model<M>>(self, model: Mod) -> (usize, usize, usize) {
            (M, model.dim(), model.state_labels().len())
        }
    }

    fn every_spec() -> Vec<ModelSpec> {
        let mut specs = Vec::new();
        for n in 1..=3 {
            specs.push(ModelSpec::Spring { n, rho: 1.0, a: 1.0, obs: n - 1 });
            specs.push(ModelSpec::SpringMass { n, params: SpringMassParams::default(), obs: 0 });
        }
        specs.push(ModelSpec::Pendulum { params: PendulumParams::default() });
        specs.push(ModelSpec::PendulumRod { params: PendulumRodParams::default() });
        specs.push(ModelSpec::Kepler { params: KeplerParams::default(), observation: KeplerObservation::Range });
        specs.push(ModelSpec::KeplerMu { params: KeplerMuParams::default(), observation: KeplerObservation::Q1 });
        specs.push(ModelSpec::Lorenz { params: LorenzParams::default(), observation: LorenzObservation::X });
        specs.push(ModelSpec::Lamppost);
        specs
    }

    /// Every variant is built at the compile-time dimension `dim()`
    /// announces, and the model agrees.
    #[test]
    fn every_spec_is_built_at_its_dimension() {
        let specs = every_spec();
        assert_eq!(specs.len(), 12);
        for spec in &specs {
            let (m, dim, labels) = spec.visit(0.01, Dims);
            assert_eq!(m, spec.dim(), "{}: visited at M = {m}", spec.kind());
            assert_eq!(dim, spec.dim(), "{}: model.dim() = {dim}", spec.kind());
            assert_eq!(labels, spec.dim(), "{}: model has {labels} components", spec.kind());
        }
    }

    /// Every variant survives a JSON round trip unchanged, and the JSON is
    /// tagged by the same name `kind()` returns (`{"kind": "spring", …}`);
    /// so does a twin description.
    #[test]
    fn every_spec_round_trips_through_json() {
        for spec in every_spec() {
            let json = serde_json::to_string(&spec).unwrap();
            assert!(json.starts_with(&format!("{{\"kind\":\"{}\"", spec.kind())), "{json}");
            let back: ModelSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(back, spec);
        }
        let twin = TwinSpec {
            model: ModelSpec::Lorenz {
                params: LorenzParams { sigma: 10.0, rho: 28.0, beta: 8.0 / 3.0 },
                observation: LorenzObservation::Z,
            },
            x0: vec![1.0, 1.0, 1.0],
            dt: 0.01,
            steps: 5,
            noise: NoiseModel::Ar1 { std: 0.05, tau: 0.5 },
            seed: 42,
            walk: Some(Walk { std: 0.1, seed: 7 }),
        };
        let json = serde_json::to_string(&twin).unwrap();
        assert!(json.contains("\"observation\":\"z\"") && json.contains("\"ar1\""), "{json}");
        assert_eq!(serde_json::from_str::<TwinSpec>(&json).unwrap(), twin);
    }

    /// A twin description generates the reference the hand-built model
    /// generates, bit for bit: same trajectory, same (noisy) observation
    /// sequence — the seeds travel with the description.
    #[test]
    fn twin_spec_reproduces_the_hand_built_reference() {
        let p = Progress::default();
        let twin = TwinSpec {
            model: ModelSpec::Kepler {
                params: KeplerParams { mu: 1.2, softening: 0.1 },
                observation: KeplerObservation::Bearing,
            },
            x0: perihelion_state(0.5).to_vec(),
            dt: 0.02,
            steps: 25,
            noise: NoiseModel::Gaussian { std: 0.05 },
            seed: 4002,
            walk: None,
        };
        let by_spec = twin.reference(&p);
        let sys = KeplerSystem::with_params(KeplerParams { mu: 1.2, softening: 0.1 }, 0.02, KeplerObservation::Bearing);
        let by_hand = Reference::twin(&sys, perihelion_state(0.5), 25, NoiseModel::Gaussian { std: 0.05 }, 4002, None, &p);
        assert_eq!(by_spec, by_hand);
        assert_eq!(by_spec.steps(), 25);
        assert_eq!(by_spec.dim(), 4);
        assert_eq!(by_spec.obs_dim(), 1);
        assert_eq!(p.done(), 50);

        // The augmented Kepler: θ is carried by the state, not the model.
        let twin = TwinSpec {
            model: ModelSpec::KeplerMu { params: KeplerMuParams::default(), observation: KeplerObservation::Q1 },
            x0: augmented_state(perihelion_state(0.3), 0.4).to_vec(),
            dt: 0.01,
            steps: 3,
            noise: NoiseModel::None,
            seed: 0,
            walk: None,
        };
        let r = twin.reference(&Progress::default());
        assert!(r.states.iter().all(|s| s[4] == 0.4));
    }
}
