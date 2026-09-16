//! `ode-observers`: observers (estimators) for the models of `ode-models`.
//!
//! * [`methods`]     — the four observers: [`methods::mortensen`] (the
//!   Mortensen filter on a Gauss–Lobatto grid), [`methods::mortensen_window`]
//!   (the translating window), [`methods::kalman`] (the Gaussian
//!   closure) and [`methods::fleming_viot`] (the particle approximation).
//! * [`output`]      — file outputs of the filter (snapshots, trajectories).
//! * [`jobs`]        — the dimension-erased job layer: a configuration edited
//!   by a user interface, the runners instantiating each observer at the
//!   model's compile-time dimension, and plain-data outputs for display.
//!
//! The step counter / cancel flag every job takes is
//! `ode_models_spec::progress::Progress`, shared with the reference generator.

pub mod jobs;
pub mod methods;
pub mod output;
