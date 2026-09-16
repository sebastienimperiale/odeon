//! `ode-observers-egui`: the egui front-end of `ode-models` and
//! `ode-observers` (one crate since 2026-09-15; the model side used to be
//! `ode-models-egui`).
//!
//! The model side:
//! * [`models`]       — the dimension-erased [`models::VizModel`] contract
//!   (name, description, parameter form, observation choices, twin
//!   experiment, scene drawing, estimator hints and options) and one
//!   implementation per forward model, with its animated scene.
//! * [`noise`]        — the observation-noise selector row.
//! * [`palette`]      — the chart / accent colours, theme-aware.
//! * [`playback`]     — the computed [`playback::Trajectory`], the playback
//!   clock and the background [`playback::RunHandle`] of a job.
//! * [`slot`]         — [`slot::ModelSlot`]: one model with its run
//!   settings, trajectory, playback clock and run in flight.
//! * [`model_panels`] — the model side of the panels: model selector,
//!   parameter form with the Observation and Simulation sections, the
//!   Run-simulation button, the observation plot with playback controls,
//!   the scene.
//!
//! The observer side:
//! * [`panels`]         — the estimator selector, the configuration panels
//!   of a [`ode_observers::jobs::FilterConfig`], the progress row.
//! * [`filter_view`]    — the "Filter density" view: spectral heatmaps of
//!   the stored max-marginals, the tracker overlay, the colour floor.
//! * [`box_view`]       — the "Tracking density" view of the translating
//!   window.
//! * [`particles_view`] — the "LParticles" view: dots and smoothed cloud.
//! * [`tracker_view`]   — the "Tracker" view: one plot per component with
//!   the ±2√(εP) band.
//! * [`app`]            — the whole observers application ([`app::App`]),
//!   generic over a [`app::Compute`] backend (jobs on a local thread, or
//!   wherever a binary's backend sends them); the binaries are a few lines.
//!
//! Everything draws from the plain-data outputs of `ode_observers::jobs`;
//! the app owns the runs and the playback clock.

pub mod app;
pub mod box_view;
pub mod filter_view;
pub mod model_panels;
pub mod models;
pub mod noise;
pub mod palette;
pub mod panels;
pub mod particles_view;
pub mod playback;
pub mod slot;
pub mod tracker_view;

pub use models::{EstimatorHints, VarHint, VizModel};
pub use slot::ModelSlot;
