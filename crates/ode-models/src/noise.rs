//! Observation-noise models: y_n = h(x_n) + η_n with η white Gaussian
//! (the assumption matching the filter's quadratic discrepancy), bounded
//! uniform (deterministic / minimum-energy worldview), or colored AR(1)
//! (violates whiteness, for robustness studies).
//!
//! Noise is currently model-internal: a model configured with
//! `with_obs_noise` caches the noisy observation at each forward step, and
//! the filter's discrepancy consumes it transparently. (Decoupling
//! observation generation from the models is a separate roadmap item.)
//!
//! Generation is deterministic in the seed: [`NoiseModel::realize`] and a
//! [`NoiseSampler`] with the same (dt, seed) produce the same sequence, so
//! a display series and a model's internal observations can be kept
//! identical.

use serde::{Deserialize, Serialize};

/// A noise model for one scalar observation component.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseModel {
    None,
    /// White Gaussian: η ~ N(0, σ²), i.i.d.
    Gaussian { std: f64 },
    /// Bounded uniform: η ~ U(−a, a), i.i.d.
    Uniform { half_width: f64 },
    /// Colored AR(1) (discrete Ornstein–Uhlenbeck): η_{n+1} = ρ η_n +
    /// √(1−ρ²) σ ξ_n with ρ = exp(−dt/τ); stationary std σ, correlation
    /// time τ.
    Ar1 { std: f64, tau: f64 },
}

impl NoiseModel {
    pub fn is_none(&self) -> bool {
        matches!(self, NoiseModel::None)
    }

    /// Stateful sequential sampler (one draw per time step).
    pub fn sampler(&self, dt: f64, seed: u64) -> NoiseSampler {
        let mut rng = SplitMix64::new(seed);
        let ar1 = match *self {
            NoiseModel::Ar1 { std, tau } => {
                let rho = (-dt / tau.max(1e-12)).exp();
                Some(Ar1State {
                    rho,
                    drive: (1.0 - rho * rho).sqrt() * std,
                    x: std * rng.normal(), // stationary start
                })
            }
            _ => None,
        };
        NoiseSampler {
            model: *self,
            rng,
            ar1,
        }
    }

    /// A length-`n` realization: the first `n` draws of
    /// [`sampler`](Self::sampler).
    pub fn realize(&self, n: usize, dt: f64, seed: u64) -> Vec<f64> {
        let mut s = self.sampler(dt, seed);
        (0..n).map(|_| s.next_sample()).collect()
    }
}

struct Ar1State {
    rho: f64,
    drive: f64,
    x: f64,
}

/// Sequential noise draws, one per time step (see [`NoiseModel::sampler`]).
pub struct NoiseSampler {
    model: NoiseModel,
    rng: SplitMix64,
    ar1: Option<Ar1State>,
}

impl NoiseSampler {
    pub fn next_sample(&mut self) -> f64 {
        match self.model {
            NoiseModel::None => 0.0,
            NoiseModel::Gaussian { std } => std * self.rng.normal(),
            NoiseModel::Uniform { half_width } => half_width * (2.0 * self.rng.uniform() - 1.0),
            NoiseModel::Ar1 { .. } => {
                let s = self.ar1.as_mut().expect("AR(1) state exists");
                let y = s.x;
                s.x = s.rho * s.x + s.drive * self.rng.normal();
                y
            }
        }
    }
}

/// SplitMix64 — tiny deterministic PRNG, good enough for observation noise
/// and the particle system, and dependency-free.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box–Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realizations_have_expected_statistics() {
        let n = 100_000;
        for model in [
            NoiseModel::Gaussian { std: 0.5 },
            NoiseModel::Uniform { half_width: 0.5 },
            NoiseModel::Ar1 { std: 0.5, tau: 0.3 },
        ] {
            let x = model.realize(n, 0.01, 42);
            let mean = x.iter().sum::<f64>() / n as f64;
            let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
            let expected_var = match model {
                NoiseModel::Uniform { half_width } => half_width * half_width / 3.0,
                NoiseModel::Gaussian { std } | NoiseModel::Ar1 { std, .. } => std * std,
                NoiseModel::None => 0.0,
            };
            assert!(mean.abs() < 0.02, "mean {mean} too far from 0");
            assert!(
                (var - expected_var).abs() < 0.1 * expected_var,
                "variance {var} vs expected {expected_var}"
            );
        }
        // Determinism, and sampler ≡ realize.
        let m = NoiseModel::Ar1 { std: 1.0, tau: 0.2 };
        let mut s = m.sampler(0.01, 7);
        let r = m.realize(5, 0.01, 7);
        for v in r {
            assert_eq!(v, s.next_sample());
        }
    }
}
