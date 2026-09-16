//! Output routines of the [`MortensenFilter`]: binary snapshots of p, the
//! reference trajectory (binary + stdout table), and the progress bar.
//! The numerical cycle itself lives in [`crate::methods::mortensen`].

use crate::methods::mortensen::MortensenFilter;
use ode_models::model::Model;
use ode_models_spec::reference::Reference;
use rayon::prelude::*;
use std::io::Write;

impl<const M: usize, Mod: Model<M> + Sync> MortensenFilter<M, Mod> {
    /// The 1D Gauss–Lobatto axis of direction d (ndof[d] coordinates):
    /// direction d sits at stride ndof[0]·…·ndof[d−1] in the grid.
    pub fn axis(&self, d: usize) -> Vec<f64> {
        let ndof = self.params.ndof();
        let stride: usize = ndof[..d].iter().product();
        (0..ndof[d]).map(|i| self.grid()[i * stride][d]).collect()
    }

    /// Max-marginal of p on the (a, b) plane: p reduced over all other
    /// directions with a max (= min-plus marginal of V = −ε log p), as an
    /// ndof[a] × ndof[b] field with direction `a` varying fastest.
    pub fn marginal_2d(&self, (a, b): (usize, usize)) -> Vec<f64> {
        let p = self.p_field();
        let ndof = self.params.ndof();
        let stride = |d: usize| ndof[..d].iter().product::<usize>();
        let (sa, sb) = (stride(a), stride(b));
        let (na, nb) = (ndof[a], ndof[b]);
        let empty = || vec![f64::NEG_INFINITY; na * nb];
        p.par_iter()
            .enumerate()
            .fold(empty, |mut acc, (i, pi)| {
                let k = (i / sa) % na + ((i / sb) % nb) * na;
                acc[k] = acc[k].max(*pi);
                acc
            })
            .reduce(empty, |mut u, v| {
                for (ui, vi) in u.iter_mut().zip(v) {
                    *ui = ui.max(vi);
                }
                u
            })
    }

    /// Write the 1D Gauss–Lobatto axis of every direction to
    /// `{out_dir}/axis_{d}.bin` (ndof[d] f64 LE each).
    pub fn save_axes(&self, out_dir: &str) {
        for d in 0..M {
            let f = std::fs::File::create(format!("{out_dir}/axis_{d}.bin"))
                .expect("cannot create axis file");
            let mut f = std::io::BufWriter::new(f);
            for x in self.axis(d) {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }
    }

    /// Write the [`marginal_2d`](Self::marginal_2d) of each pair (a, b) to
    /// `{out_dir}/p2d_{a}_{b}_{step:05}.bin` as f64 LE.
    pub fn save_marginals_2d(&self, step: usize, pairs: &[(usize, usize)], out_dir: &str) {
        for &(a, b) in pairs {
            let f = std::fs::File::create(format!("{out_dir}/p2d_{a}_{b}_{step:05}.bin"))
                .expect("cannot create 2D output file");
            let mut f = std::io::BufWriter::new(f);
            for v in self.marginal_2d((a, b)) {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }

    /// Write the reference trajectory to `{out_dir}/trajectory.bin`: the flat
    /// state of every step, f64 LE.
    pub fn save_trajectory(&self, reference: &Reference, out_dir: &str) {
        write_states(&reference.states, &format!("{out_dir}/trajectory.bin"));
    }

    /// Write the estimator trajectory x̂_n = argmax p to
    /// `{out_dir}/estimate.bin` (one flat state per step, f64 LE — same
    /// layout as trajectory.bin).
    pub fn save_estimate(&self, estimates: &[[f64; M]], out_dir: &str) {
        let f = std::fs::File::create(format!("{out_dir}/estimate.bin"))
            .expect("cannot create estimate.bin");
        let mut f = std::io::BufWriter::new(f);
        for x in estimates {
            for &c in x {
                f.write_all(&c.to_le_bytes()).unwrap();
            }
        }
    }

    /// Print the reference trajectory as a table on stdout (one row every
    /// ~0.1 time units), with the model's state labels as columns.
    pub fn print_trajectory(&self, reference: &Reference) {
        let labels = self.model.state_labels();
        let dt = self.model.dt();
        let print_every = ((0.1 / dt).round() as usize).max(1);

        print!("{:<8}", "t");
        for label in &labels {
            print!("  {label:>9}");
        }
        println!();
        print!("{:-<8}", "");
        for _ in &labels {
            print!("  {:->9}", "");
        }
        println!();

        for (step, x) in reference.states.iter().enumerate() {
            if step == 0 || step % print_every == 0 {
                print!("{:<8.4}", step as f64 * dt);
                for c in x.iter() {
                    print!("  {:>9.5}", c);
                }
                println!();
            }
        }
    }
}

/// Save a model-only run (no filter): print the per-component state ranges
/// of `reference` and write `trajectory.bin` + `meta.txt` (M, dt, labels)
/// to `out_dir` — the layout `scripts/visualize_model.py` reads. Used by
/// the `*_forward` examples to choose filter-domain intervals.
pub fn save_forward<const M: usize, Mod: Model<M>>(model: &Mod, reference: &Reference, out_dir: &str) {
    let labels = model.state_labels();
    let states = &reference.states;
    let dt = reference.dt;
    let n_steps = reference.steps();

    println!(
        "state ranges over t ∈ [0, {}]  (dt = {dt}, {n_steps} steps):\n",
        n_steps as f64 * dt
    );
    for (d, label) in labels.iter().enumerate() {
        let (min, max) = states
            .iter()
            .map(|x| x[d])
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(v), hi.max(v))
            });
        println!("  {label:>4}:  [{min:8.4}, {max:8.4}]");
    }

    std::fs::create_dir_all(out_dir).expect("cannot create output dir");
    write_states(states, &format!("{out_dir}/trajectory.bin"));

    let meta = format!("M={M}\ndt={dt}\nlabels={}\n", labels.join(","));
    std::fs::write(format!("{out_dir}/meta.txt"), meta).expect("cannot write meta.txt");
    println!("\nSaved trajectory to {out_dir}/");
}

/// Write flat states one after the other as f64 LE.
fn write_states(states: &[Vec<f64>], path: &str) {
    let f = std::fs::File::create(path).unwrap_or_else(|e| panic!("cannot create {path}: {e}"));
    let mut f = std::io::BufWriter::new(f);
    for x in states {
        for &c in x {
            f.write_all(&c.to_le_bytes()).unwrap();
        }
    }
}

/// In-place progress bar on stderr:  [#####-----] 123/600  20%  12.3s elapsed, ETA 49.1s
pub(crate) fn print_progress(done: usize, total: usize, t0: std::time::Instant) {
    const WIDTH: usize = 40;
    let filled = done * WIDTH / total;
    let elapsed = t0.elapsed().as_secs_f64();
    let eta = elapsed / done as f64 * (total - done) as f64;
    eprint!(
        "\r[{}{}] {done}/{total}  {:3}%  {elapsed:.1}s elapsed, ETA {eta:.1}s   ",
        "#".repeat(filled),
        "-".repeat(WIDTH - filled),
        done * 100 / total,
    );
    std::io::stderr().flush().ok();
}
