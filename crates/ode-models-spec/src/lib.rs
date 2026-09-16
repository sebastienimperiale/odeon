//! `ode-models-spec`: the layer that *drives* the models of `ode-models` —
//! everything an observer, a server, a script or a viewer needs to name a
//! model, run it and feed it, without being generic over it. `ode-models`
//! itself stays the physics (the `Model` trait, the models, the stepper).
//!
//! * [`spec`]      — [`spec::ModelSpec`], a model as plain data (kind,
//!   parameters, observation choice), and the [`spec::ModelVisitor`] that
//!   builds the concrete `Model<M>` from it at its compile-time dimension;
//!   [`spec::TwinSpec`], a twin experiment as data.
//! * [`reference`](mod@crate::reference) — [`reference::Reference`], a trajectory with its
//!   observation sequence, and the twin-experiment generator that makes
//!   one from a model, a noise model and seeds.
//! * [`noise`]     — deterministic observation-noise models and the small
//!   PRNG.
//! * [`progress`]  — the step counter / cancel flag a run shares with
//!   whoever drives it (a progress bar, a server), UI-free.
//! * [`rng`]       — the small deterministic PRNG behind the noise models
//!   and the particles' per-particle streams.
//!
//! No observer lives here: `ode-observers` depends on this crate, never
//! the reverse.

pub mod noise;
pub mod progress;
pub mod reference;
pub mod rng;
pub mod spec;
