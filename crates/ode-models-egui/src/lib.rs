//! `ode-models-egui`: the egui front-end of `ode-models`.
//!
//! * [`models`]   — the dimension-erased [`models::VizModel`] contract (name,
//!   description, parameter form, observation choices, trajectory job,
//!   scene drawing, estimator hints, [`ode_models::spec::ModelSpec`]) and
//!   one implementation per forward model, with its animated scene.
//! * [`noise`]    — the observation-noise selector row.
//! * [`palette`]  — the chart / accent colours, theme-aware.
//! * [`playback`] — the computed [`playback::Trajectory`], the playback
//!   clock and the background [`playback::RunHandle`] of a job.
//! * [`slot`]     — [`slot::ModelSlot`]: one model with its run settings,
//!   trajectory, playback clock and run in flight.
//! * [`panels`]   — the model side of a viewer's panels: model selector,
//!   parameter form with the Observation and Simulation sections, the
//!   Run-simulation button, the observation plot with playback controls,
//!   the scene. A models-only viewer is these and nothing else; the
//!   estimator viewer adds its own sections beside them.
//! * [`app`]      — [`app::App`], the models-only application wired from
//!   those panels; the `models-viewer` binary is it plus eframe.
//!
//! No observer lives here: this crate depends on `ode-models` and egui
//! only, so a viewer for the models alone can be built without the
//! estimators. The estimator views and panels are in `ode-observers-egui`.

pub mod app;
pub mod models;
pub mod noise;
pub mod palette;
pub mod panels;
pub mod playback;
pub mod slot;

pub use models::{EstimatorHints, VarHint, VizModel};
pub use slot::ModelSlot;
