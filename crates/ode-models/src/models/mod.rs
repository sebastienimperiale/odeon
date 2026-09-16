//! Forward models implementing the [`Model`](crate::model::Model) trait:
//!
//! * [`spring`]       — linear N-harmonic oscillator chain (implicit midpoint)
//! * [`pendulum`]     — planar double pendulum, point masses (Gauss–Legendre 4)
//! * [`pendulum_rod`] — planar double pendulum, uniform rigid rods
//!   (Gauss–Legendre 4; the swaptube model)
//! * [`kepler`]       — planar Kepler problem with Plummer softening
//!   (Gauss–Legendre 4)
//! * [`kepler_mu`]    — Kepler with an unknown gravitational parameter
//!   μ = μ₀·2^θ, augmented state (q, p, θ), θ̇ = 0 (joint state–parameter
//!   estimation)
//! * [`spring_mass`]  — spring chain (N = 1, 2, 3) with an unknown free-end
//!   mass m_N = m₀·2^θ, augmented state (y, v, θ), θ̇ = 0: the conditionally
//!   linear parameter-estimation model (Gauss–Legendre 4)
//! * [`lorenz`]       — Lorenz-63, dissipative chaotic (Gauss–Legendre 4)
//! * [`lamppost`]     — lamppost: a point in the plane observed through its
//!   squared distance to the origin — the ring-shaped density, a toy for
//!   multimodality (identity flow)
//!
//! Every model is parameters plus pure maps of a state; a reference
//! trajectory with its observations is a `Reference` of `ode-models-spec`.

pub mod lamppost;
pub mod kepler;
pub mod kepler_mu;
pub mod lorenz;
pub mod pendulum;
pub mod pendulum_rod;
pub mod spring;
pub mod spring_mass;

pub use lamppost::LamppostSystem;
pub use kepler::{KeplerObservation, KeplerParams, KeplerSystem};
pub use kepler_mu::{KeplerMuParams, KeplerMuSystem};
pub use lorenz::{LorenzObservation, LorenzParams, LorenzSystem};
pub use pendulum::{PendulumParams, PendulumSystem};
pub use pendulum_rod::{PendulumRodParams, PendulumRodSystem};
pub use spring::SpringSystem;
pub use spring_mass::{SpringMassParams, SpringMassSystem};
