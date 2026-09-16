//! `ode-models`: forward models of dynamical systems for the Odeon observers.
//!
//! * [`model`]  — the [`model::Model`] trait every forward model implements:
//!   the contract between a model and an observer (flow, inverse flow and its
//!   exact tangent, observation operator with its Jacobian, discrepancy) —
//!   parameters and pure maps, no state.
//! * [`models`] — the models: spring chains, double pendulums, Kepler (with
//!   or without an unknown μ), the spring chain with an unknown mass,
//!   Lorenz-63, the lamppost; the shared Gauss–Legendre 4 stepper.
//! * [`gl4`]    — the shared Gauss–Legendre 4 time step (Newton + analytic
//!   Jacobian, exact tangent of the discrete map) used by every nonlinear
//!   model.
//!
//! Nothing else: the models as plain data, the twin-experiment generator,
//! the noise models and the progress handle are `ode-models-spec`, which
//! depends on this crate. The `serde` feature (off by default) derives
//! `Serialize`/`Deserialize` on the parameter structs and observation
//! enums for that crate.
//!
//! No observer lives here: `ode-observers` depends on this crate, never the
//! reverse.

pub mod gl4;
pub mod model;
pub mod models;
