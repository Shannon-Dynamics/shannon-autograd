//! Forward bodies and their adjoints — written ONCE, shared by both backends.
//!
//! Day 1 shipped the W0 rule as the shape template; Day 6 adds the remaining
//! eleven of the week-1 adjoint set (week-1 plan §11.4). Every function is
//! `#[inline(always)]`, GradSink-generic, and row-shaped
//! `fn(i, inputs…, adj_out, sinks…)` — monomorphization erases the
//! abstraction on both backends.
//!
//! Convention: `ḡ` is the incoming gradient `adj_y[i]`; every rule
//! ACCUMULATES (`+=` through the sink), never assigns. Two rules thread the
//! FORWARD OUTPUT (`adj_div_at`, `adj_sqrt_at`) — the tape keeps forward
//! buffers alive precisely so this is legal. Two rules are singular at the
//! origin (`adj_length_at`, `adj_normalize_at`) and return zero gradient
//! there (the subgradient convention) rather than NaN — one NaN scatter-added
//! into a grad buffer poisons everything downstream.

use crate::{GradSink, Vec3, math};

/// Forward body of W0:  `y[i] = a[i] * scale + bias`.
///
/// The core function RETURNS a value; the backend adapter writes it. That is
/// what lets the identical body serve `DisjointSlice` on GPU and
/// `par_iter_mut` on CPU (week-1 plan §8.2).
#[inline(always)]
pub fn affine_at(i: usize, a: &[f32], scale: f32, bias: f32) -> f32 {
    a[i] * scale + bias
}

/// Adjoint of `y[i] = a[i] * scale + bias`  (scale, bias are constants).
///   ∂y/∂a = scale  ⟹  ā[i] += ȳ[i] · scale
///
/// Written ONCE, generic over GradSink — runs on both backends unchanged.
#[inline(always)]
pub fn adj_affine_at<S: GradSink<f32>>(i: usize, adj_y: &[f32], scale: f32, adj_a: &S) {
    adj_a.accumulate(i, adj_y[i] * scale);
}

// ─────────────────────────────────────────────────────────────────────────────
// The week-1 adjoint set — the other eleven (Day-6 plan §3.1)
// ─────────────────────────────────────────────────────────────────────────────

/// y = a + b.   ā += ḡ,  b̄ += ḡ
#[inline(always)]
pub fn adj_add_at<Sa: GradSink<f32>, Sb: GradSink<f32>>(
    i: usize,
    adj_y: &[f32],
    adj_a: &Sa,
    adj_b: &Sb,
) {
    let g = adj_y[i];
    adj_a.accumulate(i, g);
    adj_b.accumulate(i, g);
}

/// y = a − b.   ā += ḡ,  b̄ −= ḡ
#[inline(always)]
pub fn adj_sub_at<Sa: GradSink<f32>, Sb: GradSink<f32>>(
    i: usize,
    adj_y: &[f32],
    adj_a: &Sa,
    adj_b: &Sb,
) {
    let g = adj_y[i];
    adj_a.accumulate(i, g);
    adj_b.accumulate(i, -g);
}

/// y = a · b.   ā += ḡ·b,  b̄ += ḡ·a
#[inline(always)]
pub fn adj_mul_at<Sa: GradSink<f32>, Sb: GradSink<f32>>(
    i: usize,
    a: &[f32],
    b: &[f32],
    adj_y: &[f32],
    adj_a: &Sa,
    adj_b: &Sb,
) {
    let g = adj_y[i];
    adj_a.accumulate(i, g * b[i]);
    adj_b.accumulate(i, g * a[i]);
}

/// y = a / b.   ā += ḡ/b;  b̄ −= ḡ·a/b² — computed as ḡ·y/b with the FORWARD
/// OUTPUT threaded in, saving a division and forcing the caller to keep `y`
/// alive (the tape does — records reference forward buffers).
#[inline(always)]
pub fn adj_div_at<Sa: GradSink<f32>, Sb: GradSink<f32>>(
    i: usize,
    b: &[f32],
    y: &[f32],
    adj_y: &[f32],
    adj_a: &Sa,
    adj_b: &Sb,
) {
    let g = adj_y[i];
    adj_a.accumulate(i, g / b[i]);
    adj_b.accumulate(i, -(g * y[i] / b[i]));
}

/// y = −a.   ā −= ḡ
#[inline(always)]
pub fn adj_neg_at<S: GradSink<f32>>(i: usize, adj_y: &[f32], adj_a: &S) {
    adj_a.accumulate(i, -adj_y[i]);
}

/// y = sin a.   ā += ḡ·cos a
#[inline(always)]
pub fn adj_sin_at<S: GradSink<f32>>(i: usize, a: &[f32], adj_y: &[f32], adj_a: &S) {
    adj_a.accumulate(i, adj_y[i] * math::cos(a[i]));
}

/// y = cos a.   ā −= ḡ·sin a
#[inline(always)]
pub fn adj_cos_at<S: GradSink<f32>>(i: usize, a: &[f32], adj_y: &[f32], adj_a: &S) {
    adj_a.accumulate(i, -adj_y[i] * math::sin(a[i]));
}

/// y = √a.   ā += ḡ/(2y) — the forward output threaded in. EPS-guarded:
/// √ is singular at 0, and a NaN here would poison the whole grad buffer.
#[inline(always)]
pub fn adj_sqrt_at<S: GradSink<f32>>(i: usize, y: &[f32], adj_y: &[f32], adj_a: &S) {
    let d = 2.0 * y[i];
    if d > crate::EPS {
        adj_a.accumulate(i, adj_y[i] / d);
    }
}

/// y = a·b (Vec3 dot).   ā += ḡ·b,  b̄ += ḡ·a
#[inline(always)]
pub fn adj_dot_at<Sa: GradSink<Vec3>, Sb: GradSink<Vec3>>(
    i: usize,
    a: &[Vec3],
    b: &[Vec3],
    adj_y: &[f32],
    adj_a: &Sa,
    adj_b: &Sb,
) {
    let g = adj_y[i];
    adj_a.accumulate(i, b[i] * g);
    adj_b.accumulate(i, a[i] * g);
}

/// y = ‖v‖.   v̄ += ḡ·v/‖v‖ — zero gradient at the origin (subgradient
/// convention), never NaN.
#[inline(always)]
pub fn adj_length_at<S: GradSink<Vec3>>(i: usize, v: &[Vec3], adj_y: &[f32], adj_v: &S) {
    let len = v[i].length();
    if len > crate::EPS {
        adj_v.accumulate(i, v[i] * (adj_y[i] / len));
    }
}

/// y = v/‖v‖.   v̄ += (ḡ − n̂(n̂·ḡ))/‖v‖ — the incoming gradient projected onto
/// the tangent plane of the unit sphere, scaled by 1/‖v‖. Same EPS convention.
#[inline(always)]
pub fn adj_normalize_at<S: GradSink<Vec3>>(i: usize, v: &[Vec3], adj_y: &[Vec3], adj_v: &S) {
    let len = v[i].length();
    if len > crate::EPS {
        let n = v[i] * (1.0 / len);
        let g = adj_y[i];
        adj_v.accumulate(i, (g - n * n.dot(g)) * (1.0 / len));
    }
}

/// y[i] = x[idx[i]] — the array-indexing rule. Forward is a gather; the
/// adjoint is a SCATTER-ADD (several i may target one slot) — the one
/// operation GradSink was built for (week-1 plan §8.3).
#[inline(always)]
pub fn adj_gather_at<S: GradSink<f32>>(i: usize, idx: &[i32], adj_y: &[f32], adj_x: &S) {
    adj_x.accumulate(idx[i] as usize, adj_y[i]);
}
