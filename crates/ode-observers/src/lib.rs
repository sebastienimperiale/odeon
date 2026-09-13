//! `ode-observers`: observers (estimators) for the models of `ode-models`.
//!
//! * [`filter`]      — the Mortensen filter on p = exp(−V/ε): a tensor-product
//!   Gauss–Lobatto grid, the 4-step cycle (observation, model forward,
//!   spectral transport, split implicit-Euler diffusion).
//! * [`tracker`]     — its second-order (Gaussian) closure: one state and its
//!   curvature — the minimum-energy / EKF estimator.
//! * [`box_tracker`] — the translating window: the grid filter on a small box
//!   following the mode by whole-element shifts.
//! * [`particles`]   — the Fleming–Viot-type particle approximation.
//! * [`output`]      — file outputs of the filter (snapshots, trajectories).
//! * [`jobs`]        — the dimension-erased job layer: a configuration edited
//!   by a user interface, the runners instantiating each observer at the
//!   model's compile-time dimension, and plain-data outputs for display.
//! * [`progress`]    — the step counter / cancel flag shared by jobs and their
//!   drivers.

pub mod box_tracker;
pub mod filter;
pub mod jobs;
pub mod output;
pub mod particles;
pub mod progress;
pub mod tracker;
