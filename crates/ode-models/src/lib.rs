//! `ode-models`: forward models of dynamical systems for the Odeon observers.
//!
//! * [`model`]  — the [`model::Model`] trait every forward model implements:
//!   the contract between a model and an observer (flow, inverse flow and its
//!   exact tangent, observation operator with its Jacobian, discrepancy).
//! * [`models`] — the models: spring chains, double pendulums, Kepler (with
//!   or without an unknown μ), the spring chain with an unknown mass,
//!   Lorenz-63, the random walk; the shared Gauss–Legendre 4 stepper.
//! * [`noise`]  — deterministic observation-noise models and the small PRNG.
//! * [`progress`] — the step counter / cancel flag a run shares with
//!   whoever drives it (a progress bar, a server), UI-free.
//! * [`spec`]   — [`spec::ModelSpec`], a model as plain data (kind,
//!   parameters, initial state, observation, noise), and the
//!   [`spec::ModelVisitor`] that builds the concrete `Model<M>` from it at
//!   its compile-time dimension — how a user interface or a job server
//!   names a model without being generic over it.
//!
//! No observer lives here: `ode-observers` depends on this crate, never the
//! reverse.

pub mod model;
pub mod models;
pub mod noise;
pub mod progress;
pub mod spec;
