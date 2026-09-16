//! The reference of a twin experiment: a trajectory and the observation
//! sequence made from it — the data every observer consumes, one
//! observation per step. [`Reference::twin`] generates it from a
//! [`Model`] (the discrete flow, the observation operator, a noise model
//! and its seed, optionally a random walk of the reference); a
//! `Reference` can as well be built from real measurements, since nothing
//! in it depends on the model.

use ode_models::model::Model;
use crate::noise::{NoiseModel, NoiseSampler};
use crate::progress::Progress;
use serde::{Deserialize, Serialize};

/// A reference trajectory with its observations, at a fixed time step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reference {
    /// Time step between two consecutive entries.
    pub dt: f64,
    /// `states[n]`: the flat state x_n at step n, for n = 0..=steps.
    pub states: Vec<Vec<f64>>,
    /// `observations[n]`: the observation y_n of step n (length n_obs),
    /// same indexing as `states`.
    pub observations: Vec<Vec<f64>>,
}

/// A random walk of the reference: independent Gaussian increments of
/// standard deviation `std·√dt` added to every state component after each
/// step (a Brownian motion of diffusion coefficient std²), deterministic in
/// `seed` (component d draws from the stream seeded `seed + d`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Walk {
    pub std: f64,
    pub seed: u64,
}

impl Reference {
    /// Number of steps (entries minus one).
    pub fn steps(&self) -> usize {
        self.states.len().saturating_sub(1)
    }

    /// State dimension (of the first entry; 0 when empty).
    pub fn dim(&self) -> usize {
        self.states.first().map_or(0, Vec::len)
    }

    /// Observation dimension (of the first entry; 0 when empty).
    pub fn obs_dim(&self) -> usize {
        self.observations.first().map_or(0, Vec::len)
    }

    /// Final time, steps·dt.
    pub fn t_final(&self) -> f64 {
        self.steps() as f64 * self.dt
    }

    /// The twin experiment: from x₀, `steps` applications of the model's
    /// discrete flow, every stored state (x₀ included) having its periodic
    /// components wrapped into their interval ([`Model::periodic`]: a
    /// pendulum's angles are stored in [−π, π)) and being displaced by the
    /// optional `walk`; and at every step the observation
    /// y_n = h(x_n) + η_n with η drawn from `noise` — component j of the
    /// observation from the stream seeded `seed + j`, one draw per step
    /// starting at n = 0. Deterministic in the seeds. Bumps `progress`
    /// once per step (and stops early if it is cancelled).
    pub fn twin<const M: usize, Mod: Model<M>>(
        model: &Mod,
        x0: [f64; M],
        steps: usize,
        noise: NoiseModel,
        seed: u64,
        walk: Option<Walk>,
        progress: &Progress,
    ) -> Reference {
        let dt = model.dt();
        let n_obs = model.obs_dim();
        let mut samplers: Vec<NoiseSampler> =
            (0..n_obs).map(|j| noise.sampler(dt, seed.wrapping_add(j as u64))).collect();
        let mut walkers: Option<Vec<NoiseSampler>> = walk.map(|w| {
            let inc = NoiseModel::Gaussian { std: w.std * dt.sqrt() };
            (0..M).map(|d| inc.sampler(dt, w.seed.wrapping_add(d as u64))).collect()
        });
        let observe = |x: &[f64; M], samplers: &mut Vec<NoiseSampler>| -> Vec<f64> {
            let h = model.obs(x);
            (0..n_obs).map(|j| h[j] + samplers[j].next_sample()).collect()
        };
        let periodic = model.periodic();
        let wrap = |mut x: [f64; M]| {
            for d in 0..M {
                if let Some((lo, hi)) = periodic[d] {
                    x[d] = lo + (x[d] - lo).rem_euclid(hi - lo);
                }
            }
            x
        };
        let mut x = wrap(x0);
        let mut states = Vec::with_capacity(steps + 1);
        let mut observations = Vec::with_capacity(steps + 1);
        states.push(x.to_vec());
        observations.push(observe(&x, &mut samplers));
        for _ in 0..steps {
            if progress.cancelled() {
                break;
            }
            x = wrap(model.flow(x));
            if let Some(w) = &mut walkers {
                for d in 0..M {
                    x[d] += w[d].next_sample();
                }
            }
            states.push(x.to_vec());
            observations.push(observe(&x, &mut samplers));
            progress.step();
        }
        Reference { dt, states, observations }
    }

    /// The states as fixed-size arrays (panics if the dimension is not M).
    pub fn states_array<const M: usize>(&self) -> Vec<[f64; M]> {
        self.states
            .iter()
            .map(|s| {
                assert_eq!(s.len(), M, "reference dimension {} is not {M}", s.len());
                std::array::from_fn(|d| s[d])
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ode_models::models::lamppost::{DEFAULT_STATE, LamppostSystem};

    /// A static reference stays at its initial state and observes 1 at
    /// every step; with a walk it moves, reproducibly in the seed.
    #[test]
    fn reference_is_static_unless_it_walks() {
        let sys = LamppostSystem::new(0.1);
        let p = Progress::default();
        let r = Reference::twin(&sys, DEFAULT_STATE, 5, NoiseModel::None, 0, None, &p);
        assert!(r.states.iter().all(|s| s == &DEFAULT_STATE.to_vec()));
        assert!(r.observations.iter().all(|y| y == &vec![1.0]));
        assert_eq!((r.steps(), r.dim(), r.obs_dim()), (5, 2, 1));
        let walk = |seed| {
            Reference::twin(&sys, DEFAULT_STATE, 20, NoiseModel::None, 0, Some(Walk { std: 0.5, seed }), &p)
                .states
                .last()
                .unwrap()
                .clone()
        };
        let (a, b, c) = (walk(1), walk(1), walk(2));
        assert_eq!(a, b, "not reproducible in the seed");
        assert!((a[0] - c[0]).abs() + (a[1] - c[1]).abs() > 1e-6, "different seeds gave the same walk");
        assert!((a[0] - DEFAULT_STATE[0]).abs() + (a[1] - DEFAULT_STATE[1]).abs() > 1e-6, "the walk did not move");
    }
}
