// Update rules derived from NVIDIA Warp (https://github.com/NVIDIA/warp,
// warp/_src/optim/adam.py and warp/_src/optim/sgd.py),
// SPDX-License-Identifier: Apache-2.0 — see NOTICE at the workspace root.

//! Host-side optimizers over flat `f32` slices (Day-6 plan §4.4, §5.3). 📦
//!
//! Deliberately host code over downloaded gradients — Warp steps parameters
//! with GPU kernels (warp/_src/optim/adam.py:30); at W3's 642 vertices the
//! round trip is ~15 KB/iteration, noise next to the mesh queries. The GPU
//! `adam_step` kernel is the week-2 upgrade, behind these same signatures.
//! Vec3 parameter buffers pass through `shannon_core::vec::vec3s_as_f32s_mut`.

/// Plain SGD — stage 1's provably-monotone oracle. On the L2 loss the
/// iteration is x ← x − lr·(x − t): every residual contracts by (1 − lr), so
/// loss contracts by (1 − lr)² per step — geometric, strictly monotone, no
/// tuning. (The momentum-0 core of warp/_src/optim/sgd.py:39.)
pub struct Sgd {
    pub lr: f32,
}

impl Sgd {
    pub fn step(&self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len());
        for (p, g) in params.iter_mut().zip(grads) {
            *p -= self.lr * g;
        }
    }
}

/// Adam with bias correction — the float port of Warp's
/// `adam_step_kernel_float` (warp/_src/optim/adam.py:30). Stage 2's
/// optimizer: momentum absorbs the gradient discontinuities at Chamfer
/// correspondence switches (week-1 plan §5, W3 caveats).
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    t: i32,
    m: Vec<f32>, // 1st moment
    v: Vec<f32>, // 2nd moment
}

impl Adam {
    /// `n` = flat parameter count (3 × vertex count for Vec3 buffers).
    pub fn new(n: usize, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            t: 0,
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(
            params.len(),
            self.m.len(),
            "Adam sized for a different parameter count"
        );
        assert_eq!(params.len(), grads.len());
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t);
        let bc2 = 1.0 - self.beta2.powi(self.t);
        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            params[i] -= self.lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + self.eps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_matches_hand_formula() {
        let sgd = Sgd { lr: 0.1 };
        let mut p = [1.0f32, -2.0];
        sgd.step(&mut p, &[0.5, -1.0]);
        assert_eq!(p, [1.0 - 0.05, -2.0 + 0.1]);
    }

    /// Adam's first step has magnitude ≈ lr regardless of gradient SCALE —
    /// the bias-correction signature (m̂/√v̂ = g/|g| at t = 1).
    #[test]
    fn adam_first_step_is_lr_sized() {
        for scale in [1e-3f32, 1.0, 1e3] {
            let mut adam = Adam::new(1, 0.01);
            let mut p = [0.0f32];
            adam.step(&mut p, &[scale]);
            assert!(
                (p[0].abs() - 0.01).abs() < 1e-4,
                "first step {} for gradient scale {scale}",
                p[0]
            );
        }
    }

    /// Converges a scalar quadratic ½x² (gradient x) well below 1e-6.
    #[test]
    fn adam_converges_a_quadratic() {
        let mut adam = Adam::new(1, 0.05);
        let mut p = [3.0f32];
        for _ in 0..300 {
            let g = [p[0]];
            adam.step(&mut p, &g);
        }
        assert!(0.5 * p[0] * p[0] < 1e-6, "did not converge: x = {}", p[0]);
    }
}
