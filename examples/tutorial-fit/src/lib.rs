//! The USER's crate — everything docs/TUTORIAL.md builds lives here, not in
//! the SDK. Layout mirrors the SDK's own invariant at user scale:
//!
//!   math bodies (this file, top)  →  the only arithmetic
//!   `shannon_gpu_kernels!` block  →  GPU adapter (one row + @raw adjoints)
//!   `define_module_cache!()`      →  this crate's PTX-module accessor
//!   `mod cpu`                     →  CPU adapter (same rows, same bodies)
//!
//! Model: a damped sine wave with three scalar parameters,
//!
//! ```text
//! y(t) = A · exp(−d·t) · sin(ω·t),      params = [A, d, ω]
//! ```

use cuda_device::{kernel, thread};
use shannon_core::GradSink;
use shannon_core::math;
use shannon_device::DeviceGradF32;

// ─────────────────────────────────────────────────────────────────────────────
// The math bodies — plain functions, testable on the host, no CUDA anywhere.
// Body contract for a kernel row: `fn(i: usize, params…) -> Ret`.
// ─────────────────────────────────────────────────────────────────────────────

/// FORWARD body (map shape): `y[i] = A·exp(−d·tᵢ)·sin(ω·tᵢ)`.
pub fn damped_wave_at(i: usize, t: &[f32], params: &[f32]) -> f32 {
    let (a, d, w) = (params[0], params[1], params[2]);
    a * math::exp(-d * t[i]) * math::sin(w * t[i])
}

/// ADJOINT body (scatter shape): every sample accumulates into the SAME three
/// parameter cells, so this cannot be a map row — it goes through a
/// [`GradSink`] (hardware atomic add on GPU, CAS loop on host).
///
/// With `e = exp(−d·t)`, `s = sin(ω·t)`, `c = cos(ω·t)`:
///
/// ```text
/// ∂y/∂A = e·s        ∂y/∂d = −t·A·e·s        ∂y/∂ω = t·A·e·c
/// ```
pub fn adj_damped_wave_at<S: GradSink<f32>>(
    i: usize,
    t: &[f32],
    params: &[f32],
    adj_y: &[f32],
    adj_params: &S,
) {
    let (a, d, w) = (params[0], params[1], params[2]);
    let ti = t[i];
    let e = math::exp(-d * ti);
    let s = math::sin(w * ti);
    let c = math::cos(w * ti);
    let g = adj_y[i];
    adj_params.accumulate(0, g * e * s);
    adj_params.accumulate(1, g * (-ti) * a * e * s);
    adj_params.accumulate(2, g * ti * a * e * c);
}

/// FORWARD loss term: `½(yᵢ − targetᵢ)²`. The kernel accumulates the sum
/// into `loss[0]` through a `GradSink` — which is why the taped phase never
/// needs `&mut loss`, and why the host must zero the loss value each
/// iteration.
pub fn wave_loss_term_at(i: usize, y: &[f32], target: &[f32]) -> f32 {
    let r = y[i] - target[i];
    0.5 * r * r
}

/// ADJOINT of the loss: `ȳᵢ += (yᵢ − targetᵢ)·loss̄`. Reads the seed
/// `loss̄ = ∂loss/∂loss = 1` planted before recording.
pub fn adj_wave_loss_at<S: GradSink<f32>>(
    i: usize,
    y: &[f32],
    target: &[f32],
    adj_loss: &[f32],
    adj_y: &S,
) {
    adj_y.accumulate(i, (y[i] - target[i]) * adj_loss[0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU adapter — one declaration row per map kernel; scatter kernels in @raw.
// Emits `pub mod kernels`, compiled to PTX by cargo-oxide and embedded in
// this binary keyed by this package's name.
// ─────────────────────────────────────────────────────────────────────────────

shannon_core::shannon_gpu_kernels! {
    /// FORWARD: y[i] = A·exp(−d·tᵢ)·sin(ω·tᵢ) — race-free via DisjointSlice.
    damped_wave(t: &[f32], params: &[f32]) -> f32 = crate::damped_wave_at;

    @raw {
        /// ADJOINT: a 3-cell scatter — every one of the N threads accumulates
        /// into the SAME three parameter cells, which is exactly why this is
        /// @raw (GradSink atomics) and not a map row.
        #[kernel]
        pub fn adj_damped_wave(t: &[f32], params: &[f32], adj_y: &[f32], adj_params: &[f32]) {
            let i = thread::index_1d().get();
            if i >= t.len() {
                return; // manual guard — no DisjointSlice to do it for us
            }
            crate::adj_damped_wave_at(i, t, params, adj_y, &DeviceGradF32(adj_params));
        }

        /// FORWARD reduction: loss[0] += Σ ½(yᵢ − targetᵢ)².
        #[kernel]
        pub fn wave_loss(y: &[f32], target: &[f32], loss: &[f32]) {
            let i = thread::index_1d().get();
            if i >= y.len() {
                return;
            }
            use shannon_core::GradSink;
            DeviceGradF32(loss).accumulate(0, crate::wave_loss_term_at(i, y, target));
        }

        /// ADJOINT of the loss: ȳᵢ += (yᵢ − targetᵢ)·loss̄.
        #[kernel]
        pub fn adj_wave_loss(y: &[f32], target: &[f32], adj_loss: &[f32], adj_y: &[f32]) {
            let i = thread::index_1d().get();
            if i >= y.len() {
                return;
            }
            crate::adj_wave_loss_at(i, y, target, adj_loss, &DeviceGradF32(adj_y));
        }
    }
}

// This crate's cached PTX-module accessor: `pub fn module(&Device) -> …`.
// Pass it as the first argument of `shannon_rt::launch_in!`.
shannon_rt::define_module_cache!();

// ─────────────────────────────────────────────────────────────────────────────
// CPU adapter — the SAME rows and bodies, under rayon. No arithmetic here.
// ─────────────────────────────────────────────────────────────────────────────

pub mod cpu {
    use shannon_rt::HostGradF32;

    shannon_core::shannon_cpu_kernels! {
        /// FORWARD — same shared body, under rayon.
        damped_wave(t: &[f32], params: &[f32]) -> f32 = crate::damped_wave_at;
    }

    /// ADJOINT — same 3-cell scatter as the GPU kernel, via CAS loops.
    pub fn adj_damped_wave(t: &[f32], params: &[f32], adj_y: &[f32], adj_params: &mut [f32]) {
        use rayon::prelude::*;
        let sink = HostGradF32::new(adj_params);
        (0..t.len())
            .into_par_iter()
            .for_each(|i| crate::adj_damped_wave_at(i, t, params, adj_y, &sink));
    }

    /// FORWARD reduction: loss[0] += Σ ½(yᵢ − targetᵢ)².
    pub fn wave_loss(y: &[f32], target: &[f32], loss: &mut [f32]) {
        use rayon::prelude::*;
        let sink = HostGradF32::new(loss);
        (0..y.len()).into_par_iter().for_each(|i| {
            use shannon_core::GradSink;
            sink.accumulate(0, crate::wave_loss_term_at(i, y, target));
        });
    }

    /// ADJOINT of the loss: ȳᵢ += (yᵢ − targetᵢ)·loss̄.
    pub fn adj_wave_loss(y: &[f32], target: &[f32], adj_loss: &[f32], adj_y: &mut [f32]) {
        use rayon::prelude::*;
        let n = y.len();
        let sink = HostGradF32::new(adj_y);
        (0..n)
            .into_par_iter()
            .for_each(|i| crate::adj_wave_loss_at(i, y, target, adj_loss, &sink));
    }
}
