//! Shared 2-stage Gauss–Legendre time step (order 4, implicit, symplectic,
//! A-stable) for `M`-dimensional models, solved by Newton's method with an
//! analytic Jacobian. Used by every model with a nonlinear flow
//! ([`super::pendulum`], [`super::pendulum_rod`], [`super::kepler`],
//! [`super::lorenz`]).
//!
//! [`gl4_step_with_jacobian`] additionally returns the exact tangent
//! Φ = ∂x₁/∂x₀ of the discrete map (needed by the tracker's covariance
//! propagation): differentiating the stage equations gives
//! (I − h A⊗J)[dK₁; dK₂] = [J₁; J₂] — the Newton matrix at the converged
//! stages — and Φ = I + (h/2)(dK₁ + dK₂). One extra LU solve with M
//! right-hand sides, no finite differences.

use nalgebra::{DMatrix, DVector};

const SQRT3_6: f64 = 0.288_675_134_594_812_9; // √3 / 6
const A11: f64 = 0.25;
const A12: f64 = 0.25 - SQRT3_6;
const A21: f64 = 0.25 + SQRT3_6;
const A22: f64 = 0.25;
const TOL: f64 = 1e-13;
const MAX_ITER: usize = 25;

/// Newton matrix I₂ₘ − h (A ⊗ J) with J evaluated at each stage.
fn newton_matrix<const M: usize>(
    j1: &[[f64; M]; M],
    j2: &[[f64; M]; M],
    h: f64,
) -> DMatrix<f64> {
    let mut mat = DMatrix::<f64>::identity(2 * M, 2 * M);
    for r in 0..M {
        for c in 0..M {
            mat[(r, c)] -= h * A11 * j1[r][c];
            mat[(r, M + c)] -= h * A12 * j1[r][c];
            mat[(M + r, c)] -= h * A21 * j2[r][c];
            mat[(M + r, M + c)] -= h * A22 * j2[r][c];
        }
    }
    mat
}

/// Solve the stage system kᵢ = rhs(x + h Σⱼ aᵢⱼ kⱼ) by Newton's method;
/// returns the converged stage slopes (k₁, k₂).
///
/// The Newton system uses nalgebra's dynamic LU (`DMatrix`): the
/// const-generic LU needs typenum bounds on `2M` that would leak into every
/// caller, and the allocation is negligible next to the rhs / Jacobian
/// evaluations.
fn stages<const M: usize>(
    rhs: &impl Fn(&[f64; M]) -> [f64; M],
    jacobian: &impl Fn(&[f64; M]) -> [[f64; M]; M],
    x: &[f64; M],
    h: f64,
) -> ([f64; M], [f64; M]) {
    let mut k1 = rhs(x);
    let mut k2 = k1;

    for iter in 0..=MAX_ITER {
        let x1: [f64; M] = std::array::from_fn(|i| x[i] + h * (A11 * k1[i] + A12 * k2[i]));
        let x2: [f64; M] = std::array::from_fn(|i| x[i] + h * (A21 * k1[i] + A22 * k2[i]));
        let f1 = rhs(&x1);
        let f2 = rhs(&x2);

        // Residual F(K) = K − f(stages), converged when small relative to f.
        let res = DVector::<f64>::from_fn(2 * M, |i, _| {
            if i < M {
                k1[i] - f1[i]
            } else {
                k2[i - M] - f2[i - M]
            }
        });
        let scale = f1
            .iter()
            .chain(f2.iter())
            .fold(1.0_f64, |m, v| m.max(v.abs()));
        if res.amax() < TOL * scale {
            break;
        }
        assert!(
            iter < MAX_ITER,
            "Newton iteration in gl4_step did not converge (x = {x:?}, h = {h})"
        );

        let mat = newton_matrix(&jacobian(&x1), &jacobian(&x2), h);
        let delta = mat
            .lu()
            .solve(&res)
            .expect("singular Newton matrix in gl4_step");
        for i in 0..M {
            k1[i] -= delta[i];
            k2[i] -= delta[M + i];
        }
    }
    (k1, k2)
}

/// One Gauss–Legendre step of size `h` for ẋ = rhs(x). The 2M-dimensional
/// stage system kᵢ = rhs(x + h Σⱼ aᵢⱼ kⱼ) is solved by Newton's method with
/// the analytic `jacobian` (∂rhs/∂x); panics if it does not converge.
///
/// The scheme is symmetric: a step of size −h is the exact inverse of a step
/// of size +h, so callers get an exact discrete inverse flow for free.
pub(crate) fn gl4_step<const M: usize>(
    rhs: impl Fn(&[f64; M]) -> [f64; M],
    jacobian: impl Fn(&[f64; M]) -> [[f64; M]; M],
    x: &[f64; M],
    h: f64,
) -> [f64; M] {
    let (k1, k2) = stages(&rhs, &jacobian, x, h);
    std::array::from_fn(|i| x[i] + h * 0.5 * (k1[i] + k2[i]))
}

/// [`gl4_step`] plus the exact Jacobian Φ = ∂x₁/∂x₀ of the discrete map
/// (row i = gradient of component i of x₁), see the module docs.
pub(crate) fn gl4_step_with_jacobian<const M: usize>(
    rhs: impl Fn(&[f64; M]) -> [f64; M],
    jacobian: impl Fn(&[f64; M]) -> [[f64; M]; M],
    x: &[f64; M],
    h: f64,
) -> ([f64; M], [[f64; M]; M]) {
    let (k1, k2) = stages(&rhs, &jacobian, x, h);
    let x1: [f64; M] = std::array::from_fn(|i| x[i] + h * (A11 * k1[i] + A12 * k2[i]));
    let x2: [f64; M] = std::array::from_fn(|i| x[i] + h * (A21 * k1[i] + A22 * k2[i]));
    let (j1, j2) = (jacobian(&x1), jacobian(&x2));

    // (I − h A⊗J) dK = [J₁; J₂], one right-hand side per state component.
    let mat = newton_matrix(&j1, &j2, h);
    let rhs_mat = DMatrix::<f64>::from_fn(2 * M, M, |r, c| {
        if r < M {
            j1[r][c]
        } else {
            j2[r - M][c]
        }
    });
    let dk = mat
        .lu()
        .solve(&rhs_mat)
        .expect("singular Newton matrix in gl4_step_with_jacobian");

    let phi: [[f64; M]; M] = std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            (if r == c { 1.0 } else { 0.0 }) + 0.5 * h * (dk[(r, c)] + dk[(M + r, c)])
        })
    });
    (std::array::from_fn(|i| x[i] + h * 0.5 * (k1[i] + k2[i])), phi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The analytic tangent of the step must match central finite
    /// differences of the step itself (nonlinear test problem: Lorenz rhs).
    #[test]
    fn step_jacobian_matches_finite_differences() {
        let rhs = |s: &[f64; 3]| [10.0 * (s[1] - s[0]), s[0] * (28.0 - s[2]) - s[1], s[0] * s[1] - 8.0 / 3.0 * s[2]];
        let jac = |s: &[f64; 3]| [[-10.0, 10.0, 0.0], [28.0 - s[2], -1.0, -s[0]], [s[1], s[0], -8.0 / 3.0]];
        let x = [-3.7, -4.2, 20.3];
        let h = 0.02;
        let (x1, phi) = gl4_step_with_jacobian(rhs, jac, &x, h);
        let x1_plain = gl4_step(rhs, jac, &x, h);
        assert_eq!(x1, x1_plain);
        let eps = 1e-6;
        for c in 0..3 {
            let mut xp = x;
            let mut xm = x;
            xp[c] += eps;
            xm[c] -= eps;
            let (fp, fm) = (gl4_step(rhs, jac, &xp, h), gl4_step(rhs, jac, &xm, h));
            for r in 0..3 {
                let fd = (fp[r] - fm[r]) / (2.0 * eps);
                assert!(
                    (phi[r][c] - fd).abs() < 1e-7 * (1.0 + fd.abs()),
                    "Φ[{r}][{c}] = {} vs finite difference {fd}",
                    phi[r][c]
                );
            }
        }
    }
}
