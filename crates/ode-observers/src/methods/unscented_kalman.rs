//! Unscented closure: the Gaussian with the same *integrals* as the density
//! (`docs/density_filter.tex` §3) — the same ansatz as the tracker of
//! [`kalman`](crate::methods::kalman), one centre x̄ and one width matrix
//! Σ, closed by a different rule. The tracker matches the density at its
//! maximum (value, gradient, Hessian), which replaces φ and h by their
//! tangents at x̂; this closure matches the mass, centre and width, whose
//! exact evolution under the linear p-equation asks for the Gaussian
//! averages ⟨φ⟩, ⟨(x − x̄)φᵀ⟩, … of the *nonlinear* maps, evaluated by a
//! quadrature rule at a few points χ_j with weights w_j built from
//! (x̄, Σ) — no Jacobian is ever needed. Every rule is exact to degree two
//! at least (Σ w = 1, Σ w χ = x̄, Σ w (χ − x̄)(χ − x̄)ᵀ = Σ), which makes
//! the closure the tracker — and the Kalman filter — on an affine model,
//! whatever the rule; on a nonlinear model the rules keep the second-order
//! term ½ Σ_ij Σ_ij ∂_ij φ of the flow across the bump, which the tangent
//! drops. Σ = εP in the tracker's variables (the density p ∝ e^{−V/ε} has
//! covariance εP), so the closure is *not* ε-free: the points spread with
//! √ε and collapse onto the tracker as ε → 0.
//!
//! The discrete cycle mirrors [`MortensenTracker::forward`](crate::methods::kalman::MortensenTracker::forward)
//! step by step, with the same dt and order (§3.4 of the note):
//!
//!   1. observation: points χ_j from (x̄, Σ); η_j = h(χ_j); the integrals
//!      of the pair (x, h(x)) — ȳ, Σ_yy, Σ_xy — by the rule; the product
//!      with the Gaussian factor g = exp(−dt·γ·d²/2ε) in y gives
//!      K = Σ_xy (Σ_yy + εR)⁻¹ with R = I/(γ dt),
//!      x̄ ← x̄ + K (y ⊖ ȳ),   Σ ← Σ − K (Σ_yy + εR) Kᵀ
//!      (differences of observations are taken with the model's
//!      `innovation`, relative to h(x̄), so angles never wrap inside a sum);
//!   2. transport: points χ_j from the updated (x̄, Σ); χ⁺_j = φ(χ_j);
//!      x̄ ← Σ w_j χ⁺_j,   Σ ← Σ w_j (χ⁺_j − x̄)(χ⁺_j − x̄)ᵀ;
//!   3. diffusion: Σ ← Σ + ε dt Q, exact for any density.
//!
//! Per step: two square roots of Σ (Cholesky, nalgebra) and two sets of
//! evaluations of the model at the points of the rule, 2(2n + 1) for the
//! default. The estimate is x̄ and the covariance reported to the viewer is
//! P = Σ/ε, so that the Tracker tab's band ±2√(εP) = ±2√Σ applies unchanged.
//!
//! References (quoted from memory in the note): Julier & Uhlmann (1997,
//! 2004) for the symmetric rule; Wan & van der Merwe (2000) for its scaled
//! variant; Arasaratnam & Haykin (2009) for the spherical–radial cubature;
//! McNamee & Stenger (1967) for the degree-five rules; Ito & Xiong (2000)
//! for the Gauss–Hermite filter.

use super::Observer;
use nalgebra::{Cholesky, DMatrix, DVector, SymmetricEigen};
use ode_models::model::Model;
use ode_models_spec::reference::Reference;
use serde::{Deserialize, Serialize};

/// The quadrature rule for the Gaussian averages: points χ_j = x̄ + L ξ_j
/// (L a square root of Σ, Σ = L Lᵀ, columns s_i) and weights w_j, exact
/// for polynomials up to the rule's degree. Every rule is exact to degree
/// two, so all agree on an affine model.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quadrature {
    /// The symmetric rule with spread h: the centre and the 2n points
    /// x̄ ± h s_i, weights w₀ = 1 − n/h², w_±i = 1/(2h²); 2n + 1
    /// evaluations, exact to degree three for every h, the fourth moment
    /// along each axis (3 for a Gaussian) matched at h² = 3 — the default.
    /// Julier–Uhlmann's unscented rule is h² = n + κ; for n > h² the
    /// centre weight is negative and Σ can come out indefinite (see
    /// [`square_root`]).
    Symmetric { spread: f64 },
    /// The spherical–radial cubature: the 2n points x̄ ± √n s_i with equal
    /// weights 1/(2n), no centre, no parameter, all weights positive;
    /// exact to degree three (Arasaratnam–Haykin).
    Cubature,
    /// The fully symmetric degree-five rule of McNamee–Stenger: the centre,
    /// the 2n axis points x̄ ± √3 s_i and the 2n(n − 1) points
    /// x̄ ± √3 (s_i ± s_k); 2n² + 1 evaluations, exact to degree five. The
    /// moment conditions fix the spread at √3 and the weights at
    /// w₀ = (n² − 7n + 18)/18, w_axis = (4 − n)/18, w_pair = 1/36 (the
    /// axis weight is negative for n ≥ 5).
    DegreeFive,
    /// The tensor product of the one-dimensional Gauss–Hermite rule with
    /// `points` nodes along each s_i: pointsⁿ evaluations, exact to degree
    /// 2·points − 1 (three points per direction give degree five with 3ⁿ
    /// evaluations). Exponential in n.
    GaussHermite { points: usize },
}

impl Quadrature {
    /// The default: the symmetric rule at h = √3 (fourth axis moment
    /// matched).
    pub const DEFAULT: Quadrature = Quadrature::Symmetric { spread: 1.732_050_807_568_877_2 };

    /// Number of evaluations per set of points in dimension `n`.
    pub fn count(&self, n: usize) -> usize {
        match self {
            Quadrature::Symmetric { .. } => 2 * n + 1,
            Quadrature::Cubature => 2 * n,
            Quadrature::DegreeFive => 2 * n * n + 1,
            Quadrature::GaussHermite { points } => (*points).max(1).pow(n as u32),
        }
    }

    /// Short name for labels.
    pub fn name(&self) -> &'static str {
        match self {
            Quadrature::Symmetric { .. } => "symmetric",
            Quadrature::Cubature => "cubature",
            Quadrature::DegreeFive => "degree five",
            Quadrature::GaussHermite { .. } => "Gauss–Hermite",
        }
    }

    /// The points and weights of the rule for the Gaussian of centre
    /// `center` and square root `root` (Σ = root·rootᵀ, its columns the
    /// s_i); `hermite` holds the one-dimensional Gauss–Hermite nodes and
    /// weights when the rule needs them ([`gauss_hermite`]).
    pub fn points<const M: usize>(
        &self,
        center: &[f64; M],
        root: &DMatrix<f64>,
        hermite: Option<&(Vec<f64>, Vec<f64>)>,
    ) -> (Vec<[f64; M]>, Vec<f64>) {
        let n = M;
        let col = |i: usize| -> [f64; M] { std::array::from_fn(|d| root[(d, i)]) };
        let at = |coef: &[(usize, f64)]| -> [f64; M] {
            let mut x = *center;
            for &(i, c) in coef {
                let s = col(i);
                for d in 0..n {
                    x[d] += c * s[d];
                }
            }
            x
        };
        let mut points = Vec::with_capacity(self.count(n));
        let mut weights = Vec::with_capacity(self.count(n));
        match *self {
            Quadrature::Symmetric { spread: h } => {
                assert!(h > 0.0, "the spread must be positive, got {h}");
                points.push(*center);
                weights.push(1.0 - n as f64 / (h * h));
                for i in 0..n {
                    for sign in [1.0, -1.0] {
                        points.push(at(&[(i, sign * h)]));
                        weights.push(0.5 / (h * h));
                    }
                }
            }
            Quadrature::Cubature => {
                let h = (n as f64).sqrt();
                for i in 0..n {
                    for sign in [1.0, -1.0] {
                        points.push(at(&[(i, sign * h)]));
                        weights.push(0.5 / n as f64);
                    }
                }
            }
            Quadrature::DegreeFive => {
                let h = 3f64.sqrt();
                let nf = n as f64;
                points.push(*center);
                weights.push((nf * nf - 7.0 * nf + 18.0) / 18.0);
                for i in 0..n {
                    for sign in [1.0, -1.0] {
                        points.push(at(&[(i, sign * h)]));
                        weights.push((4.0 - nf) / 18.0);
                    }
                }
                for i in 0..n {
                    for k in (i + 1)..n {
                        for (si, sk) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                            points.push(at(&[(i, si * h), (k, sk * h)]));
                            weights.push(1.0 / 36.0);
                        }
                    }
                }
            }
            Quadrature::GaussHermite { points: m } => {
                let (nodes, w1) = hermite.expect("Gauss–Hermite nodes are computed at construction");
                assert_eq!(nodes.len(), m.max(1));
                // Every multi-index (j_1, …, j_n) in [0, m)ⁿ.
                let mut index = vec![0usize; n];
                loop {
                    let coef: Vec<(usize, f64)> = (0..n).map(|i| (i, nodes[index[i]])).collect();
                    points.push(at(&coef));
                    weights.push((0..n).map(|i| w1[index[i]]).product());
                    let mut d = 0;
                    loop {
                        if d == n {
                            return (points, weights);
                        }
                        index[d] += 1;
                        if index[d] < nodes.len() {
                            break;
                        }
                        index[d] = 0;
                        d += 1;
                    }
                }
            }
        }
        (points, weights)
    }
}

/// The one-dimensional Gauss–Hermite rule with `m` points for the standard
/// Gaussian weight e^{−ξ²/2}/√(2π) (probabilists' form): nodes and
/// weights by Golub–Welsch — the eigenvalues of the Jacobi matrix with
/// off-diagonal √k, and the squared first components of its eigenvectors.
/// Exact to degree 2m − 1; Σ w = 1.
pub fn gauss_hermite(m: usize) -> (Vec<f64>, Vec<f64>) {
    let m = m.max(1);
    if m == 1 {
        return (vec![0.0], vec![1.0]);
    }
    let mut jacobi = DMatrix::<f64>::zeros(m, m);
    for k in 1..m {
        let b = (k as f64).sqrt();
        jacobi[(k - 1, k)] = b;
        jacobi[(k, k - 1)] = b;
    }
    let eig = SymmetricEigen::new(jacobi);
    let mut rule: Vec<(f64, f64)> = (0..m)
        .map(|i| {
            let v0 = eig.eigenvectors[(0, i)];
            (eig.eigenvalues[i], v0 * v0)
        })
        .collect();
    rule.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Symmetrize against round-off: the rule is symmetric about 0.
    let total: f64 = rule.iter().map(|r| r.1).sum();
    (rule.iter().map(|r| r.0).collect(), rule.iter().map(|r| r.1 / total).collect())
}

/// A square root L of a symmetric matrix, Σ = L Lᵀ: the Cholesky factor
/// when Σ is definite; otherwise (a rule with negative weights can make the
/// width indefinite, a flat direction makes it singular) the symmetric
/// square root V √Λ with the negative eigenvalues clamped to zero — still
/// Σ = Σ_i s_i s_iᵀ over its columns for the projected Σ.
pub fn square_root(sigma: &DMatrix<f64>) -> DMatrix<f64> {
    if let Some(chol) = Cholesky::new(sigma.clone()) {
        return chol.l();
    }
    let eig = SymmetricEigen::new(sigma.clone());
    let sqrt = eig.eigenvalues.map(|l| l.max(0.0).sqrt());
    &eig.eigenvectors * DMatrix::from_diagonal(&sqrt)
}

/// Parameters of the [`UnscentedTracker`].
#[derive(Clone, Copy, Debug)]
pub struct UnscentedParams<const M: usize> {
    /// Diagonal of the model-noise covariance Q (same meaning as
    /// [`crate::methods::mortensen::FilterParams::q_diag`]).
    pub q_diag: [f64; M],
    /// γ — observation weight (same meaning as
    /// [`crate::methods::mortensen::FilterParams::gamma`]): the observation
    /// factor is exp(−dt·γ·d²/2ε), i.e. R = I/(γ dt). 0 switches the
    /// observations off. Must be ≥ 0.
    pub gamma: f64,
    /// ε — the temperature of the density: the width is Σ = εP and the
    /// points of the rule spread with √ε. Must be > 0.
    pub eps: f64,
    /// The quadrature rule.
    pub rule: Quadrature,
}

/// The unscented estimator: centre x̄ and width matrix Σ of the density.
///
/// # Example
/// ```
/// use ode_observers::methods::unscented_kalman::{Quadrature, UnscentedParams, UnscentedTracker};
/// use ode_models::models::SpringSystem;
///
/// let sys = SpringSystem::new(1, 1.0, 1.0, 0.01, |x: &[f64]| x[0]);
/// let mut ukf = UnscentedTracker::<2, _>::new(sys,
///     UnscentedParams { q_diag: [1.0; 2], gamma: 1.0, eps: 0.05, rule: Quadrature::DEFAULT });
/// ukf.init([0.4, 0.1], [10.0, 10.0]); // Gaussian prior: centre, stiffness σ_d
/// ukf.forward(&[0.5]);                // one cycle, with the observation y₀ of this step
/// let x_bar = ukf.estimate();
/// let p = ukf.covariance().expect("Σ/ε is always defined");
/// ```
pub struct UnscentedTracker<const M: usize, Mod: Model<M>> {
    /// The forward model (parameters and maps).
    pub model: Mod,
    /// Parameters the estimator was built with.
    pub params: UnscentedParams<M>,
    x_bar: [f64; M],
    /// Width matrix Σ of the density (M × M, symmetric).
    sigma: DMatrix<f64>,
    /// The one-dimensional Gauss–Hermite rule, when the rule needs it.
    hermite: Option<(Vec<f64>, Vec<f64>)>,
}

impl<const M: usize, Mod: Model<M>> UnscentedTracker<M, Mod> {
    /// Build the estimator around a model; call [`init`](Self::init) before
    /// [`forward`](Self::forward).
    pub fn new(model: Mod, params: UnscentedParams<M>) -> Self {
        assert_eq!(model.dim(), M, "dimension M = {M} does not match the model's state dimension");
        assert!(params.q_diag.iter().all(|&q| q >= 0.0), "q_diag must be nonnegative, got {:?}", params.q_diag);
        assert!(params.gamma >= 0.0, "gamma must be nonnegative, got {}", params.gamma);
        assert!(params.eps > 0.0, "eps must be positive, got {}", params.eps);
        let hermite = match params.rule {
            Quadrature::GaussHermite { points } => Some(gauss_hermite(points)),
            _ => None,
        };
        UnscentedTracker {
            model,
            params,
            x_bar: [0.0; M],
            sigma: DMatrix::zeros(M, M),
            hermite,
        }
    }

    /// Gaussian initial data V₀ = Σ_d σ_d (x_d − x_c,d)²/2 — the same prior
    /// as the filter's and the tracker's: x̄₀ = x_c, Σ₀ = diag(ε/σ_d).
    /// Every σ_d must be positive: a flat direction has no width.
    pub fn init(&mut self, center: [f64; M], sigma: [f64; M]) {
        assert!(sigma.iter().all(|&s| s > 0.0), "sigma must be positive in every direction, got {sigma:?}");
        self.x_bar = center;
        self.sigma = DMatrix::from_diagonal(&DVector::from_iterator(M, sigma.iter().map(|&s| self.params.eps / s)));
    }

    /// Current estimate x̄ (the centre of the density).
    pub fn estimate(&self) -> [f64; M] {
        self.x_bar
    }

    /// Width matrix Σ of the density.
    pub fn width(&self) -> &DMatrix<f64> {
        &self.sigma
    }

    /// Covariance in the tracker's units, P = Σ/ε (the density has
    /// covariance εP). Always defined; an `Option` to match the tracker's
    /// accessor.
    pub fn covariance(&self) -> Option<DMatrix<f64>> {
        Some(&self.sigma / self.params.eps)
    }

    /// The points and weights of the rule at the current (x̄, Σ).
    fn points(&self) -> (Vec<[f64; M]>, Vec<f64>) {
        let root = square_root(&self.sigma);
        self.params.rule.points(&self.x_bar, &root, self.hermite.as_ref())
    }

    /// One iteration, t^n → t^{n+1}, given the observation `y` = y_n of
    /// step n: observation → transport → diffusion (see the module docs).
    pub fn forward(&mut self, y: &[f64]) {
        let dt = self.model.dt();
        let (gamma, eps) = (self.params.gamma, self.params.eps);

        // Step 1 — observation: the pair (x, h(x)) by quadrature, then the
        // Gaussian product with the observation factor.
        if gamma > 0.0 {
            let (chi, w) = self.points();
            let k = self.model.obs_dim();
            // δ_j = h(χ_j) ⊖ h(x̄), in the observation's own geometry.
            let deltas: Vec<DVector<f64>> = chi
                .iter()
                .map(|c| self.model.innovation(self.model.obs(c).as_slice(), &self.x_bar))
                .collect();
            let mut delta_bar = DVector::<f64>::zeros(k);
            for (d, &wj) in deltas.iter().zip(&w) {
                delta_bar += wj * d;
            }
            let mut syy = DMatrix::<f64>::zeros(k, k);
            let mut sxy = DMatrix::<f64>::zeros(M, k);
            for (j, &wj) in w.iter().enumerate() {
                let dd = &deltas[j] - &delta_bar;
                let dx = DVector::from_iterator(M, (0..M).map(|d| chi[j][d] - self.x_bar[d]));
                syy += wj * &dd * dd.transpose();
                sxy += wj * &dx * dd.transpose();
            }
            // y ⊖ ȳ = (y ⊖ h(x̄)) − δ̄.
            let r = self.model.innovation(y, &self.x_bar) - &delta_bar;
            let s = &syy + DMatrix::<f64>::identity(k, k) * (eps / (gamma * dt));
            let gain = &sxy * s.clone().try_inverse().expect("Σ_yy + εR is definite (εR ≻ 0)");
            let shift = &gain * r;
            for d in 0..M {
                self.x_bar[d] += shift[d];
            }
            self.sigma -= &gain * &s * gain.transpose();
            self.symmetrize();
        }

        // Step 2 — transport: the integrals of φ by quadrature.
        let (chi, w) = self.points();
        let images: Vec<[f64; M]> = chi.iter().map(|c| self.model.flow(*c)).collect();
        let mut mean = [0.0; M];
        for (img, &wj) in images.iter().zip(&w) {
            for d in 0..M {
                mean[d] += wj * img[d];
            }
        }
        let mut sigma = DMatrix::<f64>::zeros(M, M);
        for (img, &wj) in images.iter().zip(&w) {
            let dx = DVector::from_iterator(M, (0..M).map(|d| img[d] - mean[d]));
            sigma += wj * &dx * dx.transpose();
        }
        self.x_bar = mean;
        self.sigma = sigma;

        // Step 3 — diffusion: Σ ← Σ + ε dt Q, exact.
        for d in 0..M {
            self.sigma[(d, d)] += eps * dt * self.params.q_diag[d];
        }
        self.symmetrize();
    }

    /// Keep Σ exactly symmetric against round-off drift.
    fn symmetrize(&mut self) {
        self.sigma = 0.5 * (&self.sigma + self.sigma.transpose());
    }

    /// Run one iteration per step of `reference` (with its observations),
    /// returning the estimate at every step (including t = 0) and the
    /// covariance P = Σ/ε.
    pub fn run_with_covariances(&mut self, reference: &Reference) -> (Vec<[f64; M]>, Vec<Option<DMatrix<f64>>>) {
        let n_steps = reference.steps();
        let mut estimates = Vec::with_capacity(n_steps + 1);
        let mut covariances = Vec::with_capacity(n_steps + 1);
        estimates.push(self.estimate());
        covariances.push(self.covariance());
        for y in &reference.observations[..n_steps] {
            self.forward(y);
            estimates.push(self.estimate());
            covariances.push(self.covariance());
        }
        (estimates, covariances)
    }
}

/// The common observer surface.
impl<const M: usize, Mod: Model<M>> Observer<M> for UnscentedTracker<M, Mod> {
    fn dt(&self) -> f64 {
        self.model.dt()
    }

    fn init_gaussian(&mut self, center: [f64; M], sigma: [f64; M]) {
        self.init(center, sigma);
    }

    fn forward(&mut self, y: &[f64]) {
        UnscentedTracker::forward(self, y);
    }

    fn estimate(&self) -> [f64; M] {
        UnscentedTracker::estimate(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::kalman::{MortensenTracker, TrackerParams};
    use ode_models::models::{LorenzObservation, LorenzSystem, SpringSystem};
    use ode_models_spec::noise::NoiseModel;
    use ode_models_spec::progress::Progress;

    const RULES: [Quadrature; 5] = [
        Quadrature::DEFAULT,
        Quadrature::Symmetric { spread: 1.1 },
        Quadrature::Cubature,
        Quadrature::DegreeFive,
        Quadrature::GaussHermite { points: 3 },
    ];

    fn spring(dt: f64) -> SpringSystem {
        SpringSystem::new(1, 1.0, 1.0, dt, |x: &[f64]| x[0])
    }

    fn spring_reference(dt: f64, steps: usize, noise: NoiseModel, seed: u64) -> Reference {
        Reference::twin(&spring(dt), [1.15, 0.0], steps, noise, seed, None, &Progress::default())
    }

    /// The one-dimensional Gauss–Hermite rule: three points are 0, ±√3
    /// with weights 2/3, 1/6, and the m-point rule reproduces the Gaussian
    /// moments E[ξ^k] = (k − 1)!! (0 for odd k) up to degree 2m − 1.
    #[test]
    fn gauss_hermite_rule_is_exact() {
        let (nodes, w) = gauss_hermite(3);
        assert!((nodes[0] + 3f64.sqrt()).abs() < 1e-12 && nodes[1].abs() < 1e-12 && (nodes[2] - 3f64.sqrt()).abs() < 1e-12);
        assert!((w[0] - 1.0 / 6.0).abs() < 1e-12 && (w[1] - 2.0 / 3.0).abs() < 1e-12);
        for m in 1..=8 {
            let (nodes, w) = gauss_hermite(m);
            for k in 0..(2 * m) {
                let terms: Vec<f64> = nodes.iter().zip(&w).map(|(x, w)| w * x.powi(k as i32)).collect();
                let moment: f64 = terms.iter().sum();
                let scale: f64 = terms.iter().map(|t| t.abs()).sum();
                let exact = if k % 2 == 1 { 0.0 } else { (1..k as u64).step_by(2).product::<u64>() as f64 };
                assert!((moment - exact).abs() < 1e-12 * (1.0 + scale), "m = {m}, degree {k}: {moment} vs {exact}");
            }
        }
    }

    /// Every rule reproduces the mass, the centre and the width of a
    /// Gaussian with a full width matrix (degree two); the degree-five
    /// rules also its fourth moments E[ξ_i⁴] = 3Σ_ii², E[ξ_i²ξ_k²] =
    /// Σ_iiΣ_kk + 2Σ_ik².
    #[test]
    fn rules_reproduce_the_gaussian_moments() {
        let center = [0.3, -1.2, 2.0];
        let sigma = DMatrix::from_row_slice(3, 3, &[2.0, 0.5, -0.3, 0.5, 1.0, 0.2, -0.3, 0.2, 0.7]);
        let root = square_root(&sigma);
        assert!((&root * root.transpose() - &sigma).amax() < 1e-12);
        for rule in RULES {
            let hermite = match rule {
                Quadrature::GaussHermite { points } => Some(gauss_hermite(points)),
                _ => None,
            };
            let (chi, w) = rule.points::<3>(&center, &root, hermite.as_ref());
            assert_eq!(chi.len(), rule.count(3), "{rule:?}");
            let mass: f64 = w.iter().sum();
            assert!((mass - 1.0).abs() < 1e-12, "{rule:?}: mass {mass}");
            for d in 0..3 {
                let m1: f64 = chi.iter().zip(&w).map(|(c, w)| w * c[d]).sum();
                assert!((m1 - center[d]).abs() < 1e-12, "{rule:?}: centre");
            }
            for i in 0..3 {
                for k in 0..3 {
                    let m2: f64 = chi.iter().zip(&w).map(|(c, w)| w * (c[i] - center[i]) * (c[k] - center[k])).sum();
                    assert!((m2 - sigma[(i, k)]).abs() < 1e-12, "{rule:?}: width ({i}, {k}) {m2} vs {}", sigma[(i, k)]);
                }
            }
            if matches!(rule, Quadrature::DegreeFive | Quadrature::GaussHermite { .. }) {
                for i in 0..3 {
                    for k in 0..3 {
                        let m4: f64 = chi
                            .iter()
                            .zip(&w)
                            .map(|(c, w)| w * (c[i] - center[i]).powi(2) * (c[k] - center[k]).powi(2))
                            .sum();
                        let exact = if i == k {
                            3.0 * sigma[(i, i)].powi(2)
                        } else {
                            sigma[(i, i)] * sigma[(k, k)] + 2.0 * sigma[(i, k)].powi(2)
                        };
                        assert!((m4 - exact).abs() < 1e-11, "{rule:?}: fourth moment ({i}, {k}) {m4} vs {exact}");
                    }
                }
            }
        }
    }

    /// On the linear spring the closure is exact for every rule: it equals
    /// the tracker of `kalman` (and hence the Kalman filter, which that
    /// module pins the tracker to) — centre vs x̂ and Σ/ε vs P = S⁻¹ — to
    /// round-off.
    #[test]
    fn matches_the_tracker_on_the_spring_for_every_rule() {
        let dt = 0.01;
        let q = [0.3, 0.7];
        let sigma = [4.0, 9.0];
        let x_c = [1.0, 0.2];
        let eps = 0.05;
        let r = spring_reference(dt, 200, NoiseModel::Gaussian { std: 0.05 }, 7);
        for rule in RULES {
            let mut tracker = MortensenTracker::<2, _>::new(spring(dt), TrackerParams { q_diag: q, gamma: 1.0 });
            tracker.init(x_c, sigma);
            let mut ukf =
                UnscentedTracker::<2, _>::new(spring(dt), UnscentedParams { q_diag: q, gamma: 1.0, eps, rule });
            ukf.init(x_c, sigma);
            for n in 0..200 {
                tracker.forward(&r.observations[n]);
                ukf.forward(&r.observations[n]);
                let (a, b) = (tracker.estimate(), ukf.estimate());
                for d in 0..2 {
                    assert!((a[d] - b[d]).abs() < 1e-9, "{rule:?}, step {n}: x̂ {a:?} vs x̄ {b:?}");
                }
                let (pa, pb) = (tracker.covariance().unwrap(), ukf.covariance().unwrap());
                assert!((&pa - &pb).amax() < 1e-9, "{rule:?}, step {n}: P differs:\n{pa}\nvs\n{pb}");
            }
        }
    }

    /// γ = 0 switches the observations off: on the linear spring the centre
    /// is then the pure discrete flow of its initial condition (the rule is
    /// exact on an affine map), for every rule.
    #[test]
    fn gamma_zero_reduces_to_the_flow() {
        let dt = 0.01;
        let x0 = [0.9, -0.3];
        let r = spring_reference(dt, 50, NoiseModel::Gaussian { std: 0.05 }, 3);
        for rule in RULES {
            let mut ukf = UnscentedTracker::<2, _>::new(
                spring(dt),
                UnscentedParams { q_diag: [1.0; 2], gamma: 0.0, eps: 0.1, rule },
            );
            ukf.init(x0, [5.0; 2]);
            let mut flow = x0;
            for y in &r.observations[..50] {
                ukf.forward(y);
                flow = ukf.model.flow(flow);
                let e = ukf.estimate();
                for d in 0..2 {
                    assert!((e[d] - flow[d]).abs() < 1e-12, "{rule:?}: γ = 0 estimate left the flow: {e:?} vs {flow:?}");
                }
            }
        }
    }

    /// A quadratic map, φ(x₁, x₂) = (x₁ + a x₂², x₂), observed through x₁,
    /// for the transport test: the Gaussian integrals of a quadratic map are
    /// explicit.
    struct Quadratic {
        a: f64,
    }

    impl Model<2> for Quadratic {
        fn dt(&self) -> f64 {
            1.0
        }
        fn flow(&self, x: [f64; 2]) -> [f64; 2] {
            [x[0] + self.a * x[1] * x[1], x[1]]
        }
        fn flow_inv(&self, x: [f64; 2]) -> [f64; 2] {
            [x[0] - self.a * x[1] * x[1], x[1]]
        }
        fn flow_jacobian(&self, x: [f64; 2]) -> [[f64; 2]; 2] {
            [[1.0, 2.0 * self.a * x[1]], [0.0, 1.0]]
        }
        fn is_autonomous(&self) -> bool {
            true
        }
        fn obs_dim(&self) -> usize {
            1
        }
        fn obs(&self, x: &[f64]) -> DVector<f64> {
            DVector::from_row_slice(&[x[0]])
        }
        fn obs_jacobian(&self, _x: &[f64]) -> DMatrix<f64> {
            DMatrix::from_row_slice(1, 2, &[1.0, 0.0])
        }
        fn state_labels(&self) -> Vec<String> {
            vec!["x1".into(), "x2".into()]
        }
    }

    /// The transported centre and width against the exact Gaussian
    /// integrals of the quadratic map, one step with γ = 0 and q = 0: with
    /// ξ = x − x̄ ~ N(0, Σ), E[φ] = (x̄₁ + a(x̄₂² + Σ₂₂), x̄₂),
    /// Var(x₁ + a x₂²) = Σ₁₁ + 4a x̄₂ Σ₁₂ + a²(4x̄₂² Σ₂₂ + 2Σ₂₂²),
    /// Cov(x₁ + a x₂², x₂) = Σ₁₂ + 2a x̄₂ Σ₂₂ — moments up to degree four,
    /// which the degree-five rules match on any Σ and the symmetric rule at
    /// h = √3 only on a diagonal one (fourth axis moment matched); the
    /// cubature (h² = n = 2) gets E[ξ₂⁴] = 2Σ₂₂² instead of 3Σ₂₂², so its
    /// Σ₁₁ is short by a²Σ₂₂² — and the tracker's tangent misses the
    /// a Σ₂₂ term of the centre altogether.
    #[test]
    fn transport_of_a_quadratic_map_matches_the_gaussian_integrals() {
        let a = 0.7;
        let x_c = [0.2, 0.5];
        let eps = 1.0;
        let cases: Vec<(Quadrature, [f64; 2], bool)> = vec![
            (Quadrature::DegreeFive, [0.5, 0.3], true),
            (Quadrature::GaussHermite { points: 3 }, [0.5, 0.3], true),
            (Quadrature::GaussHermite { points: 4 }, [0.5, 0.3], true),
            (Quadrature::DEFAULT, [0.5, 0.3], false),
        ];
        for (rule, sig, full) in cases {
            let mut ukf = UnscentedTracker::<2, _>::new(
                Quadratic { a },
                UnscentedParams { q_diag: [0.0; 2], gamma: 0.0, eps, rule },
            );
            // Σ₀ = diag(ε/σ); a first step with the cubature rule of an
            // off-diagonal Σ is not needed: build the off-diagonal case by
            // hand through a diagonal prior transported once (the map
            // correlates the components), then test the second step.
            ukf.init(x_c, sig);
            if full {
                ukf.forward(&[0.0]);
            }
            let (xb, s) = (ukf.estimate(), ukf.width().clone());
            assert!(!full || s[(0, 1)].abs() > 1e-3, "{rule:?}: the first step should have correlated the components");
            ukf.forward(&[0.0]);
            let mean = [xb[0] + a * (xb[1] * xb[1] + s[(1, 1)]), xb[1]];
            let var11 = s[(0, 0)] + 4.0 * a * xb[1] * s[(0, 1)] + a * a * (4.0 * xb[1] * xb[1] * s[(1, 1)] + 2.0 * s[(1, 1)].powi(2));
            let cov12 = s[(0, 1)] + 2.0 * a * xb[1] * s[(1, 1)];
            let (e, w) = (ukf.estimate(), ukf.width());
            assert!((e[0] - mean[0]).abs() < 1e-12 && (e[1] - mean[1]).abs() < 1e-12, "{rule:?}: centre {e:?} vs {mean:?}");
            assert!((w[(0, 0)] - var11).abs() < 1e-12, "{rule:?}: Σ₁₁ {} vs {var11}", w[(0, 0)]);
            assert!((w[(0, 1)] - cov12).abs() < 1e-12, "{rule:?}: Σ₁₂ {} vs {cov12}", w[(0, 1)]);
            assert!((w[(1, 1)] - s[(1, 1)]).abs() < 1e-12, "{rule:?}: Σ₂₂");
            // The tracker's centre is φ(x̄): off by a·Σ₂₂.
            assert!((ukf.model.flow(xb)[0] - mean[0]).abs() > 1e-3);
        }
        // The cubature's fourth moment is 2Σ₂₂² (degree three only).
        let mut ukf = UnscentedTracker::<2, _>::new(
            Quadratic { a },
            UnscentedParams { q_diag: [0.0; 2], gamma: 0.0, eps, rule: Quadrature::Cubature },
        );
        ukf.init(x_c, [0.5, 0.3]);
        let s22 = ukf.width()[(1, 1)];
        let var11 = ukf.width()[(0, 0)] + a * a * (4.0 * x_c[1] * x_c[1] * s22 + 2.0 * s22 * s22);
        ukf.forward(&[0.0]);
        assert!((ukf.width()[(0, 0)] - var11 + a * a * s22 * s22).abs() < 1e-12, "cubature: Σ₁₁ {} vs {var11}", ukf.width()[(0, 0)]);
    }

    /// Nonlinear sanity, as for the tracker: with noise-free x observations
    /// on Lorenz the estimator started off the reference locks onto it, for
    /// every rule, and Σ stays positive. The closure is not ε-free: the
    /// points spread with the width Σ = εP, and on Lorenz P reaches ≈ 50 in
    /// the unobserved directions, so at ε = 1 the cloud straddles the
    /// attractor and the averaged flow no longer follows it (late error
    /// 6.2, all rules alike); it converges to the tracker (late error 0.08)
    /// as ε shrinks — measured 3.3 at ε = 0.3, 1.5 at 0.1, 0.19 at 0.01,
    /// 0.05 at 0.003 — which is Remark "it is not ε-free" of the note.
    #[test]
    fn tracks_lorenz_from_x_observations() {
        let dt = 0.01;
        let x0 = [-3.716171, -4.204785, 20.339103];
        let sys = || LorenzSystem::new(dt, LorenzObservation::X);
        let r = Reference::twin(&sys(), x0, 1000, NoiseModel::None, 0, None, &Progress::default());
        for rule in RULES {
            let mut ukf = UnscentedTracker::<3, _>::new(
                sys(),
                UnscentedParams { q_diag: [1.0; 3], gamma: 1.0, eps: 0.01, rule },
            );
            ukf.init([x0[0] + 2.0, x0[1] - 2.0, x0[2] + 3.0], [1.0; 3]);
            let mut max_err_late = 0.0_f64;
            for n in 0..1000 {
                ukf.forward(&r.observations[n]);
                let s = &r.states[n + 1];
                let e = ukf.estimate();
                let err = (0..3).map(|d| (e[d] - s[d]).abs()).fold(0.0, f64::max);
                assert!(err.is_finite() && err < 50.0, "{rule:?}: diverged at step {n}: {err}");
                if n >= 500 {
                    max_err_late = max_err_late.max(err);
                }
                assert!(ukf.width().diagonal().iter().all(|&v| v > 0.0), "{rule:?}: Σ not positive");
            }
            assert!(max_err_late < 0.5, "{rule:?}: did not lock onto the reference: max late error {max_err_late}");
        }
    }
}
