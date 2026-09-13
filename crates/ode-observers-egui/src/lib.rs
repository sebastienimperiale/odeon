//! `ode-observers-egui`: the egui front-end of `ode-observers`.
//!
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
//! the app owns the runs and the playback clock. The only model-side
//! dependency is the palette of `ode-models-egui`, so that the series
//! colours match the scene's.

pub mod app;
pub mod box_view;
pub mod filter_view;
pub mod panels;
pub mod particles_view;
pub mod tracker_view;
