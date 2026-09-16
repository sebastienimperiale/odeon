//! Run-then-replay machinery: a computed [`Trajectory`], the playback clock,
//! and the background worker that produces trajectories without blocking the
//! UI thread.

use ode_models_spec::reference::Reference;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use web_time::Instant;

/// A fully computed forward run: states and observations at every step
/// (index n ↔ time n·dt). Observations are vector-valued; `obs_labels`
/// names the components. Labels live here (snapshotted by the job) so the
/// plot legend always matches the data even if the observation choice is
/// edited after the run.
pub struct Trajectory {
    pub dt: f64,
    pub states: Vec<Vec<f64>>,
    pub observations: Vec<Vec<f64>>,
    pub obs_labels: Vec<String>,
}

impl Trajectory {
    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn t_final(&self) -> f64 {
        (self.len().saturating_sub(1)) as f64 * self.dt
    }

    /// Frame index closest to time `t`, clamped to the trajectory.
    pub fn frame_at(&self, t: f64) -> usize {
        ((t / self.dt).round().max(0.0) as usize).min(self.len().saturating_sub(1))
    }

    /// The trajectory as the [`Reference`] an estimator job runs along:
    /// the same states and observations (a copy).
    pub fn reference(&self) -> Reference {
        Reference { dt: self.dt, states: self.states.clone(), observations: self.observations.clone() }
    }
}

/// The playback clock of one model's trajectory.
pub struct Playback {
    pub t: f64,
    pub playing: bool,
    pub speed: f64,
}

impl Default for Playback {
    fn default() -> Self {
        Playback {
            t: 0.0,
            playing: false,
            speed: 1.0,
        }
    }
}

impl Playback {
    /// Advance by one wall-clock frame; stops (and clamps) at the end.
    pub fn advance(&mut self, wall_dt: f64, t_final: f64) {
        if self.playing {
            self.t += self.speed * wall_dt;
            if self.t >= t_final {
                self.t = t_final;
                self.playing = false;
            }
        }
    }

    pub fn restart(&mut self) {
        self.t = 0.0;
        self.playing = true;
    }
}

pub use ode_models_spec::progress::Progress;

/// A computation snapshotted from the current parameter form and ready to
/// run on a worker thread, producing a `T`. Bumps the [`Progress`] counter
/// once per step and stops early when it is cancelled.
pub type JobOf<T> = Box<dyn FnOnce(&Progress) -> T + Send>;

/// The trajectory job of a simulation run.
pub type Job = JobOf<Trajectory>;

/// A running background computation; poll from the UI thread every frame.
/// Dropping the handle cancels the job: the worker stops at its next check
/// and its result is discarded (the receiver is gone). On wasm there are
/// no threads: the job runs to completion inside `spawn` (fine for the
/// trajectories, which are cheap; estimator jobs go to a server there).
pub struct RunHandle<T = Trajectory> {
    rx: Receiver<T>,
    progress: Arc<Progress>,
    total: usize,
    started: Instant,
}

impl<T> Drop for RunHandle<T> {
    fn drop(&mut self) {
        self.progress.cancel();
    }
}

impl<T: Send + 'static> RunHandle<T> {
    pub fn spawn(job: JobOf<T>, total_steps: usize) -> Self {
        let (tx, rx) = channel();
        let progress = Arc::new(Progress::default());
        let counter = progress.clone();
        let work = move || {
            let traj = job(&counter);
            let _ = tx.send(traj); // receiver gone = run abandoned/cancelled, fine
        };
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(work);
        #[cfg(target_arch = "wasm32")]
        work();
        RunHandle {
            rx,
            progress,
            total: total_steps.max(1),
            started: Instant::now(),
        }
    }

    /// Seconds since the job was spawned.
    pub fn elapsed(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// Fraction of steps done, in [0, 1].
    pub fn progress(&self) -> f32 {
        self.progress.done() as f32 / self.total as f32
    }

    /// The finished result, if the worker is done.
    pub fn try_take(&self) -> Option<T> {
        self.rx.try_recv().ok()
    }
}
