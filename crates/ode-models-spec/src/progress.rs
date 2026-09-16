//! The link between a running job and whoever drives it: a step counter the
//! job bumps once per step (a progress bar) and a cancel flag the driver
//! raises to ask the job to stop. A cancelled job returns early with what it
//! has. UI-free: shared by the egui viewer and any server front.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Default)]
pub struct Progress {
    done: AtomicUsize,
    cancel: AtomicBool,
}

impl Progress {
    /// One more step done.
    pub fn step(&self) {
        self.done.fetch_add(1, Ordering::Relaxed);
    }

    /// Steps done so far.
    pub fn done(&self) -> usize {
        self.done.load(Ordering::Relaxed)
    }

    /// Ask the job to stop at its next check.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Has the job been asked to stop? Checked by the run loops once per
    /// iteration (the filter's construction — transport plan — and each
    /// iteration are atomic units: a cancel takes effect after the current one).
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}
