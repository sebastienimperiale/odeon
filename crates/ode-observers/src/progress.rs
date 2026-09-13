//! The step counter / cancel flag shared by jobs and their drivers. It
//! lives in `ode-models` (a forward run needs it as much as an observer
//! run); re-exported here so `ode_observers::progress::Progress` keeps
//! working.
pub use ode_models::progress::Progress;
