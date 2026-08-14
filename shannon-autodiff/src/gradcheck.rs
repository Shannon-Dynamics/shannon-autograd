//! Central-difference gradient checking.

/// Central-difference gradient:  ∂f/∂xᵢ ≈ ( f(x + ε·eᵢ) − f(x − ε·eᵢ) ) / 2ε
pub fn grad_fd<F: Fn(&[f32]) -> f32>(f: &F, x: &[f32], eps: f32) -> Vec<f32> {
    let mut g = vec![0.0; x.len()];
    let mut probe = x.to_vec();
    for i in 0..x.len() {
        let orig = probe[i];
        probe[i] = orig + eps;
        let hi = f(&probe);
        probe[i] = orig - eps;
        let lo = f(&probe);
        probe[i] = orig;
        g[i] = (hi - lo) / (2.0 * eps);
    }
    g
}

#[derive(Debug)]
pub struct GradError {
    pub index: usize,
    pub analytic: f32,
    pub finite_diff: f32,
    pub rel_error: f32,
}

/// Compare an analytic gradient against finite differences.
///
/// RELATIVE tolerance, never exact equality: GPU float atomics commit in
/// nondeterministic order, so accumulated gradients are not bit-reproducible.
///
/// Recommended start: `eps = 1e-3`, `tol = 1e-4`.
/// If a check fails marginally, sweep `eps` BEFORE suspecting the adjoint —
/// at f32 precision, too-small eps means cancellation, too-large means truncation.
pub fn gradcheck<F: Fn(&[f32]) -> f32>(
    f: F,
    x: &[f32],
    analytic: &[f32],
    eps: f32,
    tol: f32,
) -> Result<(), GradError> {
    assert_eq!(
        x.len(),
        analytic.len(),
        "gradcheck: x has {} elements but analytic gradient has {}",
        x.len(),
        analytic.len()
    );
    let fd = grad_fd(&f, x, eps);
    for i in 0..x.len() {
        let denom = fd[i].abs().max(analytic[i].abs()).max(1.0);
        let rel = (fd[i] - analytic[i]).abs() / denom;
        if rel > tol {
            return Err(GradError {
                index: i,
                analytic: analytic[i],
                finite_diff: fd[i],
                rel_error: rel,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fd_matches_product_rule() {
        // f(x) = x0 * x1  ⟹  ∇f = (x1, x0)
        let x = [3.0f32, 4.0];
        gradcheck(|v| v[0] * v[1], &x, &[4.0, 3.0], 1e-3, 1e-4).unwrap();
    }

    #[test]
    fn fd_matches_affine() {
        // f(x) = 2·x0 + 1  ⟹  ∇f = (2,)   ← the W0 rule
        gradcheck(|v| v[0] * 2.0 + 1.0, &[5.0], &[2.0], 1e-3, 1e-4).unwrap();
    }

    #[test]
    fn detects_a_wrong_gradient() {
        // The oracle must FAIL on a deliberately wrong analytic gradient,
        // otherwise it proves nothing.
        assert!(gradcheck(|v| v[0] * v[1], &[3.0, 4.0], &[99.0, 3.0], 1e-3, 1e-4).is_err());
    }

    #[test]
    fn fd_handles_nonlinear() {
        // f(x) = sin(x0)·x1²  ⟹  ∇f = (cos(x0)·x1², 2·sin(x0)·x1)
        let x = [0.7f32, 1.3];
        let analytic = [x[0].cos() * x[1] * x[1], 2.0 * x[0].sin() * x[1]];
        gradcheck(|v| v[0].sin() * v[1] * v[1], &x, &analytic, 1e-3, 1e-3).unwrap();
    }
}
