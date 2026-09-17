//! The estimation methods — five observers of one density p = exp(−V/ε),
//! sharing the same cycle (observation → transport → diffusion) and the
//! same [`Model`](ode_models::model::Model) contract:
//!
//! * [`mortensen`]        — the Mortensen filter on a tensor-product
//!   Gauss–Lobatto grid: the density itself, any shape, several maxima.
//! * [`mortensen_window`] — the translating window: the same grid density
//!   on a small box following the mode by whole-element shifts.
//! * [`kalman`]           — the second-order (Gaussian) closure: one state
//!   and its curvature, the minimum-energy estimator, i.e. the extended
//!   Kalman filter in information form.
//! * [`unscented_kalman`] — the unscented closure: the same Gaussian
//!   matched by its integrals, computed by a quadrature rule at a few
//!   points of the exact flow — no Jacobian.
//! * [`fleming_viot`]     — the Fleming–Viot-type particle approximation.
//!
//! Each takes the observation of the step as an argument of its `forward`
//! and knows nothing about where it came from; the job layer
//! ([`crate::jobs`]) drives them along a reference. What they have in
//! common is the [`Observer`] trait: Gaussian initial data, one step per
//! observation, an estimate — enough to drive any of them along a
//! [`Reference`] with the provided [`run`](Observer::run), to test them
//! together, or to hold one as `Box<dyn Observer<M>>`.

pub mod fleming_viot;
pub mod kalman;
pub mod mortensen;
pub mod mortensen_window;
pub mod unscented_kalman;

use ode_models_spec::progress::Progress;
use ode_models_spec::reference::Reference;

/// The common surface of the five estimators. Every method keeps its own
/// richer interface (the filter's density, the tracker's curvature, the
/// window's shifts, the particles' cloud); this is the part a driver can
/// use without knowing which method it holds.
pub trait Observer<const M: usize> {
    /// The model's time step, also the step of the observation sequence.
    fn dt(&self) -> f64;

    /// Gaussian initial data V₀ = Σ_d σ_d (x_d − x_c,d)²/2, i.e. p₀ ∝
    /// exp(−V₀/ε): centre x_c and stiffness σ per direction (σ_d = 0 leaves
    /// direction d flat where the method allows it). Resets the observer.
    fn init_gaussian(&mut self, center: [f64; M], sigma: [f64; M]);

    /// One step, t^n → t^{n+1}, given the observation y_n of step n.
    fn forward(&mut self, y: &[f64]);

    /// The current state estimate.
    fn estimate(&self) -> [f64; M];

    /// Run one step per observation of `reference`, returning the estimate
    /// at every step (t = 0 included); bumps `progress` once per step and
    /// stops early (partial output) if it is cancelled.
    fn run(&mut self, reference: &Reference, progress: &Progress) -> Vec<[f64; M]> {
        let mut estimates = Vec::with_capacity(reference.steps() + 1);
        estimates.push(self.estimate());
        for y in &reference.observations[..reference.steps()] {
            if progress.cancelled() {
                break;
            }
            self.forward(y);
            estimates.push(self.estimate());
            progress.step();
        }
        estimates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::fleming_viot::{ParticleParams, ParticleSystem};
    use crate::methods::kalman::{MortensenTracker, TrackerParams};
    use crate::methods::mortensen::{DiffusionScheme, FilterParams, MortensenFilter};
    use crate::methods::mortensen_window::{BoxTracker, BoxTrackerParams};
    use crate::methods::unscented_kalman::{Quadrature, UnscentedParams, UnscentedTracker};
    use ode_models::models::SpringSystem;
    use ode_models_spec::noise::NoiseModel;

    fn spring(dt: f64) -> SpringSystem {
        SpringSystem::new(1, 1.0, 1.0, dt, |x: &[f64]| x[0])
    }

    /// The five methods behind one trait: started off the reference with
    /// the same Gaussian prior and driven along the same noisy
    /// observations (observation weight γ = 20, so that the particles'
    /// killing selects within the run) by the provided `run`, each one's
    /// estimate of the observed component ends at least twice closer to
    /// the reference than it started, and all report the same number of
    /// estimates.
    #[test]
    fn every_observer_converges_on_the_spring() {
        let dt = 0.01;
        let steps = 200;
        let r = Reference::twin(&spring(dt), [1.15, 0.0], steps, NoiseModel::Gaussian { std: 0.02 }, 7, None, &Progress::default());
        let (center, sigma) = ([0.8, 0.3], [4.0, 9.0]);
        let q = [1.0; 2];
        let filter = MortensenFilter::<2, _>::new(
            spring(dt),
            FilterParams {
                domain: [(-3.0, 3.0); 2],
                eps: 0.05,
                gamma: 20.0,
                n_el: [12; 2],
                p_ord: [4; 2],
                diffusion: DiffusionScheme::SplitEuler,
                q_diag: q,
                periodic: [false; 2],
                dirichlet: [false; 2],
                pre_compute_flow_inv: true,
            },
        );
        let window = BoxTracker::<2, _>::new(
            spring(dt),
            BoxTrackerParams {
                half_width: [1.0; 2],
                eps: 0.05,
                gamma: 20.0,
                n_el: [5; 2],
                p_ord: [4; 2],
                diffusion: DiffusionScheme::SplitEuler,
                q_diag: q,
                pre_compute_flow_inv: true,
            },
        );
        let tracker = MortensenTracker::<2, _>::new(spring(dt), TrackerParams { q_diag: q, gamma: 20.0 });
        let unscented = UnscentedTracker::<2, _>::new(
            spring(dt),
            UnscentedParams { q_diag: q, gamma: 20.0, eps: 0.05, rule: Quadrature::DEFAULT },
        );
        let particles = ParticleSystem::<2, _>::new(
            spring(dt),
            ParticleParams {
                n_particles: 400,
                center,
                sigma,
                domain: [(-3.0, 3.0); 2],
                periodic: [false; 2],
                eps: 0.05,
                q_diag: q,
                noise_scale: 1.0,
                gamma: 20.0,
                seed: 3,
            },
        );
        let observers: Vec<(&str, Box<dyn Observer<2>>)> = vec![
            ("filter", Box::new(filter)),
            ("window", Box::new(window)),
            ("tracker", Box::new(tracker)),
            ("unscented", Box::new(unscented)),
            ("particles", Box::new(particles)),
        ];
        for (name, mut o) in observers {
            assert_eq!(o.dt(), dt);
            o.init_gaussian(center, sigma);
            let e0 = (o.estimate()[0] - r.states[0][0]).abs();
            let progress = Progress::default();
            let estimates = o.run(&r, &progress);
            assert_eq!(estimates.len(), steps + 1, "{name}");
            assert_eq!(progress.done(), steps, "{name}");
            let e1 = (estimates[steps][0] - r.states[steps][0]).abs();
            // Measured at γ = 20: filter 0.40 → 0.021, window 0.35 → 0.079
            // (node accuracy of its coarser box), tracker 0.35 → 0.027,
            // unscented = tracker on this linear model, particles 0.35 → 0.14 (the mean of a cloud selected by
            // killing, the weakest of the four here).
            assert!(e1 < 0.5 * e0, "{name}: error {e0} → {e1}");
        }
    }
}
