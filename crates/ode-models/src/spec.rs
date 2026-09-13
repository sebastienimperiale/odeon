//! Dimension-erased model descriptions ([`ModelSpec`]) and the visitor that
//! turns one into a concrete [`Model<M>`] ([`ModelVisitor`]).
//!
//! The observers are generic over the state dimension `M` (a const generic,
//! because the grid works on `[f64; M]` points) and over the model type.
//! A user interface or a job server, on the other hand, handles *values*:
//! a model chosen from a menu with its parameters, initial state,
//! observation choice and observation noise. `ModelSpec` is that value —
//! plain data, one variant per model of [`crate::models`], with no closure
//! and no generic — and [`ModelSpec::visit`] is the bridge: it builds the
//! concrete system and hands it to a [`ModelVisitor`] whose `visit` method
//! is generic over `M` and the model type. The visitor is where an observer
//! gets instantiated at the right compile-time dimension; this crate knows
//! nothing about observers.
//!
//! Every field a model needs to be rebuilt *exactly* (same reference
//! trajectory, same observation sequence) is in the spec, including the
//! seeds of the deterministic noise models, so that a description sent to
//! a server reproduces what a viewer computed locally.

use serde::{Deserialize, Serialize};
use crate::model::Model;
use crate::models::{
    KeplerMuParams, KeplerMuSystem, KeplerObservation, KeplerParams, KeplerSystem,
    LorenzObservation, LorenzParams, LorenzSystem, PendulumParams, PendulumRodParams,
    PendulumRodSystem, PendulumSystem, RandomWalkSystem, SpringMassParams, SpringMassSystem,
    SpringSystem,
};
use crate::noise::NoiseModel;
use nalgebra::DVector;

/// A forward model as plain data: which model, its physical parameters,
/// its initial state, its observation and the observation noise. Built
/// into a concrete system by [`visit`](Self::visit); the time step is
/// passed at that point, not stored here — it is a run setting shared with
/// the observer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelSpec {
    /// [`SpringSystem`]: the linear chain of `n` masses (state dimension
    /// 2n), observing the flat component `obs` of [Y; V].
    Spring {
        n: usize,
        rho: f64,
        a: f64,
        y0: Vec<f64>,
        v0: Vec<f64>,
        obs: usize,
        noise: NoiseModel,
        noise_seed: u64,
    },
    /// [`SpringMassSystem`]: the chain of `n` ∈ {1, 2, 3} bodies with an
    /// unknown free-end mass, augmented state (y, v, θ) of dimension
    /// 2n + 1; `theta` is the *true* log-mass of the reference; `obs` is the
    /// index of the observed position.
    SpringMass {
        n: usize,
        params: SpringMassParams,
        y0: Vec<f64>,
        v0: Vec<f64>,
        theta: f64,
        obs: usize,
        noise: NoiseModel,
        noise_seed: u64,
    },
    /// [`PendulumSystem`]: the double pendulum with point masses, state
    /// (q₁, q₂, p₁, p₂), observing the tip position.
    Pendulum {
        params: PendulumParams,
        x0: [f64; 4],
        noise: NoiseModel,
        noise_seed: u64,
    },
    /// [`PendulumRodSystem`]: the double compound pendulum, same state.
    PendulumRod {
        params: PendulumRodParams,
        x0: [f64; 4],
        noise: NoiseModel,
        noise_seed: u64,
    },
    /// [`KeplerSystem`]: the softened Kepler problem, state (q₁, q₂, p₁, p₂).
    Kepler {
        params: KeplerParams,
        x0: [f64; 4],
        observation: KeplerObservation,
        noise: NoiseModel,
        noise_seed: u64,
    },
    /// [`KeplerMuSystem`]: Kepler with the unknown μ = μ₀·2^θ, augmented
    /// state (q₁, q₂, p₁, p₂, θ); `x0` is the base Kepler state and `theta`
    /// the *true* log-parameter of the reference.
    KeplerMu {
        params: KeplerMuParams,
        x0: [f64; 4],
        theta: f64,
        observation: KeplerObservation,
        noise: NoiseModel,
        noise_seed: u64,
    },
    /// [`LorenzSystem`]: Lorenz-63, state (x, y, z).
    Lorenz {
        params: LorenzParams,
        x0: [f64; 3],
        observation: LorenzObservation,
        noise: NoiseModel,
        noise_seed: u64,
    },
    /// [`RandomWalkSystem`]: the static point observed through its squared
    /// distance to the origin; `walk_std > 0` makes the reference itself a
    /// random walk of that standard deviation (seeded by `walk_seed`).
    RandomWalk {
        x0: [f64; 2],
        walk_std: f64,
        walk_seed: u64,
        noise: NoiseModel,
        noise_seed: u64,
    },
}

/// A computation generic over the model: [`ModelSpec::visit`] builds the
/// system described by the spec and calls [`visit`](Self::visit) with it,
/// at the model's compile-time dimension `M`. Implement it with a struct
/// holding whatever the computation needs (an observer configuration, a
/// progress handle) and returning its output.
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
            ModelSpec::RandomWalk { .. } => 2,
        }
    }

    /// Short stable identifier of the model kind (`"spring"`,
    /// `"kepler_mu"`, …), the name a job description or a server route
    /// can key on.
    pub fn kind(&self) -> &'static str {
        match self {
            ModelSpec::Spring { .. } => "spring",
            ModelSpec::SpringMass { .. } => "spring_mass",
            ModelSpec::Pendulum { .. } => "pendulum",
            ModelSpec::PendulumRod { .. } => "pendulum_rod",
            ModelSpec::Kepler { .. } => "kepler",
            ModelSpec::KeplerMu { .. } => "kepler_mu",
            ModelSpec::Lorenz { .. } => "lorenz",
            ModelSpec::RandomWalk { .. } => "random_walk",
        }
    }

    /// Build the described system with time step `dt` and hand it to
    /// `visitor` at its compile-time dimension. The observation noise is
    /// installed right after construction (the models re-cache y₀ with the
    /// first draw), exactly as the CLI examples and the viewer do.
    ///
    /// # Panics
    ///
    /// On a chain length outside the supported range (springs: 1–3,
    /// unknown-mass chains: 1–3), or on an initial state of the wrong
    /// length — a malformed description, not a runtime condition.
    pub fn visit<V: ModelVisitor>(&self, dt: f64, visitor: V) -> V::Output {
        match self {
            ModelSpec::Spring { n, rho, a, y0, v0, obs, noise, noise_seed } => {
                let obs = *obs;
                let sys = SpringSystem::new(
                    *n,
                    *rho,
                    *a,
                    dt,
                    DVector::from_vec(y0.clone()),
                    DVector::from_vec(v0.clone()),
                    move |x: &[f64]| x[obs],
                )
                .with_obs_noise(*noise, *noise_seed);
                match n {
                    1 => visitor.visit::<2, _>(sys),
                    2 => visitor.visit::<4, _>(sys),
                    3 => visitor.visit::<6, _>(sys),
                    _ => panic!("spring chains of 1 to 3 masses are supported, got n = {n}"),
                }
            }
            ModelSpec::SpringMass { n, params, y0, v0, theta, obs, noise, noise_seed } => {
                let x0: Vec<f64> =
                    y0.iter().chain(v0.iter()).copied().chain(std::iter::once(*theta)).collect();
                macro_rules! build {
                    ($n:literal, $m:literal) => {
                        visitor.visit::<$m, _>(
                            SpringMassSystem::<$n>::with_params(*params, &x0, dt, *obs)
                                .with_obs_noise(*noise, *noise_seed),
                        )
                    };
                }
                match n {
                    1 => build!(1, 3),
                    2 => build!(2, 5),
                    3 => build!(3, 7),
                    _ => panic!("unknown-mass chains of 1 to 3 bodies are supported, got n = {n}"),
                }
            }
            ModelSpec::Pendulum { params, x0, noise, noise_seed } => visitor.visit::<4, _>(
                PendulumSystem::with_params(*params, *x0, dt).with_obs_noise(*noise, *noise_seed),
            ),
            ModelSpec::PendulumRod { params, x0, noise, noise_seed } => visitor.visit::<4, _>(
                PendulumRodSystem::with_params(*params, *x0, dt).with_obs_noise(*noise, *noise_seed),
            ),
            ModelSpec::Kepler { params, x0, observation, noise, noise_seed } => visitor.visit::<4, _>(
                KeplerSystem::with_params(*params, *x0, dt, *observation)
                    .with_obs_noise(*noise, *noise_seed),
            ),
            ModelSpec::KeplerMu { params, x0, theta, observation, noise, noise_seed } => {
                let x0 = crate::models::kepler_mu::augmented_state(*x0, *theta);
                visitor.visit::<5, _>(
                    KeplerMuSystem::with_params(*params, x0, dt, *observation)
                        .with_obs_noise(*noise, *noise_seed),
                )
            }
            ModelSpec::Lorenz { params, x0, observation, noise, noise_seed } => visitor.visit::<3, _>(
                LorenzSystem::with_params(*params, *x0, dt, *observation)
                    .with_obs_noise(*noise, *noise_seed),
            ),
            ModelSpec::RandomWalk { x0, walk_std, walk_seed, noise, noise_seed } => {
                let sys = RandomWalkSystem::new(*x0, dt);
                let sys = if *walk_std > 0.0 { sys.with_walk(*walk_std, *walk_seed) } else { sys };
                visitor.visit::<2, _>(sys.with_obs_noise(*noise, *noise_seed))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::kepler::perihelion_state;

    /// A visitor reporting the compile-time dimension it was called at and
    /// the model's own state dimension.
    struct Dims;
    impl ModelVisitor for Dims {
        type Output = (usize, usize);
        fn visit<const M: usize, Mod: Model<M>>(self, model: Mod) -> (usize, usize) {
            (M, model.state_labels().len())
        }
    }

    fn every_spec() -> Vec<ModelSpec> {
        let (noise, noise_seed) = (NoiseModel::None, 0);
        let mut specs = Vec::new();
        for n in 1..=3 {
            let mut y0: Vec<f64> = (1..=n).map(|i| i as f64 / n as f64).collect();
            y0[n - 1] += 0.15;
            specs.push(ModelSpec::Spring {
                n,
                rho: 1.0,
                a: 1.0,
                y0,
                v0: vec![0.0; n],
                obs: n - 1,
                noise,
                noise_seed,
            });
            specs.push(ModelSpec::SpringMass {
                n,
                params: SpringMassParams::default(),
                y0: vec![0.1; n],
                v0: vec![0.0; n],
                theta: 0.3,
                obs: 0,
                noise,
                noise_seed,
            });
        }
        specs.push(ModelSpec::Pendulum {
            params: PendulumParams::default(),
            x0: [1.5, 1.4, 0.0, 0.0],
            noise,
            noise_seed,
        });
        specs.push(ModelSpec::PendulumRod {
            params: PendulumRodParams::default(),
            x0: [2.453, -2.7727, 0.0, 0.0],
            noise,
            noise_seed,
        });
        specs.push(ModelSpec::Kepler {
            params: KeplerParams::default(),
            x0: perihelion_state(0.5),
            observation: KeplerObservation::Range,
            noise,
            noise_seed,
        });
        specs.push(ModelSpec::KeplerMu {
            params: KeplerMuParams::default(),
            x0: perihelion_state(0.5),
            theta: 0.4,
            observation: KeplerObservation::Q1,
            noise,
            noise_seed,
        });
        specs.push(ModelSpec::Lorenz {
            params: LorenzParams::default(),
            x0: [1.0, 1.0, 1.0],
            observation: LorenzObservation::X,
            noise,
            noise_seed,
        });
        specs.push(ModelSpec::RandomWalk {
            x0: [0.0, 1.0],
            walk_std: 0.0,
            walk_seed: 0,
            noise,
            noise_seed,
        });
        specs
    }

    /// Every variant is built at the compile-time dimension `dim()`
    /// announces, and the model agrees.
    #[test]
    fn every_spec_is_built_at_its_dimension() {
        let specs = every_spec();
        assert_eq!(specs.len(), 12);
        for spec in &specs {
            let (m, labels) = spec.visit(0.01, Dims);
            assert_eq!(m, spec.dim(), "{}: visited at M = {m}", spec.kind());
            assert_eq!(labels, spec.dim(), "{}: model has {labels} components", spec.kind());
        }
    }

    /// Every variant survives a JSON round trip unchanged, and the JSON is
    /// tagged by the same name `kind()` returns (`{"kind": "spring", …}`).
    #[test]
    fn every_spec_round_trips_through_json() {
        for spec in every_spec() {
            let json = serde_json::to_string(&spec).unwrap();
            assert!(json.starts_with(&format!("{{\"kind\":\"{}\"", spec.kind())), "{json}");
            let back: ModelSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(back, spec);
        }
        let noisy = ModelSpec::Lorenz {
            params: LorenzParams { sigma: 10.0, rho: 28.0, beta: 8.0 / 3.0 },
            x0: [1.0, 1.0, 1.0],
            observation: LorenzObservation::Z,
            noise: NoiseModel::Ar1 { std: 0.05, tau: 0.5 },
            noise_seed: 42,
        };
        let json = serde_json::to_string(&noisy).unwrap();
        assert!(json.contains("\"observation\":\"z\"") && json.contains("\"ar1\""), "{json}");
        assert_eq!(serde_json::from_str::<ModelSpec>(&json).unwrap(), noisy);
    }

    /// The spec rebuilds the model the hand-written constructor builds:
    /// same reference trajectory and same (noisy) observation sequence,
    /// bit for bit — the noise seed travels with the spec.
    #[test]
    fn spec_reproduces_the_hand_built_model() {
        struct Run(usize);
        impl ModelVisitor for Run {
            type Output = (Vec<Vec<f64>>, Vec<f64>);
            fn visit<const M: usize, Mod: Model<M>>(self, mut model: Mod) -> Self::Output {
                let mut ys = vec![model.y_obs()[0]];
                for _ in 0..self.0 {
                    model.forward();
                    ys.push(model.y_obs()[0]);
                }
                (model.states().iter().map(|s| s.as_slice().to_vec()).collect(), ys)
            }
        }
        let noise = NoiseModel::Gaussian { std: 0.05 };
        let spec = ModelSpec::Kepler {
            params: KeplerParams { mu: 1.2, softening: 0.1 },
            x0: perihelion_state(0.5),
            observation: KeplerObservation::Bearing,
            noise,
            noise_seed: 4002,
        };
        let (states, ys) = spec.visit(0.02, Run(25));

        let mut sys = KeplerSystem::with_params(
            KeplerParams { mu: 1.2, softening: 0.1 },
            perihelion_state(0.5),
            0.02,
            KeplerObservation::Bearing,
        )
        .with_obs_noise(noise, 4002);
        let mut ys_ref = vec![sys.y_obs()[0]];
        for _ in 0..25 {
            sys.forward();
            ys_ref.push(sys.y_obs()[0]);
        }
        assert_eq!(ys, ys_ref);
        assert_eq!(states.len(), 26);
        for (s, r) in states.iter().zip(sys.states()) {
            assert_eq!(s.as_slice(), r.as_slice());
        }
    }
}
