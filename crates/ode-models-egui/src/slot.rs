//! One model in a viewer: the model, its run settings, its trajectory and
//! playback clock, and the background run producing the trajectory. Shared
//! by the models-only viewer and the estimator viewer (which keeps its
//! estimator state beside it).

use crate::models::VizModel;
use crate::playback::{Playback, Progress, RunHandle, Trajectory};

pub struct ModelSlot {
    pub model: Box<dyn VizModel>,
    pub dt: f64,
    pub t_final: f64,
    pub traj: Option<Trajectory>,
    /// `traj` is a one-frame preview of the initial condition (drawn in the
    /// scene, but not a run: no playback, no observation plot).
    pub is_preview: bool,
    pub playback: Playback,
    /// The trajectory computation in flight, if any.
    pub run: Option<RunHandle>,
}

impl ModelSlot {
    /// A slot showing the model's initial condition, with the default run
    /// settings (dt = 0.01, T = 10).
    pub fn new(model: Box<dyn VizModel>) -> Self {
        let mut slot = ModelSlot {
            model,
            dt: 0.01,
            t_final: 10.0,
            traj: None,
            is_preview: true,
            playback: Playback::default(),
            run: None,
        };
        slot.refresh_preview();
        slot
    }

    /// Show the current initial condition in the scene: a zero-step "run",
    /// computed synchronously (it is just the initial state). Discards any
    /// finished or in-flight run and resets the playback.
    pub fn refresh_preview(&mut self) {
        let job = self.model.make_job(self.dt, 0);
        self.traj = Some(job(&Progress::default()));
        self.is_preview = true;
        self.playback = Playback::default();
        self.run = None;
    }

    /// Number of steps of a run with the current settings.
    pub fn steps(&self) -> usize {
        (self.t_final / self.dt).round().max(1.0) as usize
    }

    /// A completed run is on screen (not a preview, nothing in flight).
    pub fn run_done(&self) -> bool {
        !self.is_preview && self.run.is_none() && self.traj.is_some()
    }

    /// The trajectory of a completed run, if any.
    pub fn completed(&self) -> Option<&Trajectory> {
        self.traj.as_ref().filter(|_| !self.is_preview)
    }

    /// Start the trajectory computation on a worker thread.
    pub fn start_run(&mut self) {
        let steps = self.steps();
        let job = self.model.make_job(self.dt, steps);
        self.run = Some(RunHandle::spawn(job, steps));
    }

    /// Collect a finished run: install its trajectory and start the
    /// playback. Returns `true` when a run completed on this call.
    pub fn poll(&mut self) -> bool {
        if let Some(run) = &self.run
            && let Some(traj) = run.try_take()
        {
            self.traj = Some(traj);
            self.is_preview = false;
            self.run = None;
            self.playback.restart();
            return true;
        }
        false
    }

    /// Advance the playback clock by one wall-clock frame (a preview has a
    /// single frame, so this is a no-op there).
    pub fn advance(&mut self, wall_dt: f64) {
        if let Some(traj) = &self.traj {
            self.playback.advance(wall_dt, traj.t_final());
        }
    }

    /// Whether something moves or computes — the app keeps repainting.
    pub fn busy(&self) -> bool {
        self.playback.playing || self.run.is_some()
    }
}
