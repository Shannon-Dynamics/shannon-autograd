//! W3 loss bodies — forward terms and their adjoints (Day-6 plan §3.2). 📦
//!
//! Forward terms RETURN per-element values; the kernels accumulate them into
//! `loss[0]` through a GradSink (reduction = scatter to slot 0, the same
//! atomics the adjoints use, on the forward path). Consequence carried by
//! every caller: the loss buffer accumulates and MUST be zeroed each
//! iteration.
//!
//! The Chamfer gradients are EXACT under held correspondence — the envelope
//! theorem (week-1 plan §5, W3): ∇ₓ ½‖x − p(x)‖² = x − p exactly, because
//! (x − p) is normal to the target surface at p while ∂p/∂x maps into its
//! tangent plane; they annihilate. Holding (face, u, v) constant is what
//! makes the query-free adjoint correct, not an approximation.

use crate::mesh::{MeshQuery, mesh_eval_position};
use crate::{GradSink, Vec3, math};

// ─────────────────────────────────────────────────────────────────────────────
// Stage 1 — L2 with known correspondence (the oracle loss)
// ─────────────────────────────────────────────────────────────────────────────

/// ½‖xᵢ − tᵢ‖² — same-index correspondence.
#[inline(always)]
pub fn l2_term_at(i: usize, x: &[Vec3], t: &[Vec3]) -> f32 {
    0.5 * (x[i] - t[i]).length_sq()
}

/// ∂/∂xᵢ Σ ½‖x − t‖² = (xᵢ − tᵢ)·l̄ — l̄ is loss.grad[0], broadcast.
#[inline(always)]
pub fn adj_l2_at<S: GradSink<Vec3>>(i: usize, x: &[Vec3], t: &[Vec3], adj_loss: &[f32], adj_x: &S) {
    adj_x.accumulate(i, (x[i] - t[i]) * adj_loss[0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2 — symmetric Chamfer under held correspondence
// ─────────────────────────────────────────────────────────────────────────────

/// A→B term: source vertex → held closest point ON the target surface.
/// `corr[i]` came from an UNTAPED `mesh_query` against the target.
#[inline(always)]
pub fn chamfer_ab_term_at(
    i: usize,
    src: &[Vec3],
    corr: &[MeshQuery],
    tpoints: &[Vec3],
    tindices: &[i32],
) -> f32 {
    let q = corr[i];
    if q.face < 0 {
        return 0.0; // out of max_dist — contributes nothing (and no gradient)
    }
    let p = mesh_eval_position(tpoints, tindices, q.face, q.u, q.v);
    0.5 * (src[i] - p).length_sq()
}

/// Envelope theorem: the ENTIRE A→B gradient is (x − p)·l̄. The query's own
/// dependence on x contributes exactly zero — no query adjoint needed.
#[inline(always)]
pub fn adj_chamfer_ab_at<S: GradSink<Vec3>>(
    i: usize,
    src: &[Vec3],
    corr: &[MeshQuery],
    tpoints: &[Vec3],
    tindices: &[i32],
    adj_loss: &[f32],
    adj_src: &S,
) {
    let q = corr[i];
    if q.face < 0 {
        return;
    }
    let p = mesh_eval_position(tpoints, tindices, q.face, q.u, q.v);
    adj_src.accumulate(i, (src[i] - p) * adj_loss[0]);
}

/// B→A term: target vertex → held closest point on the SOURCE surface.
/// The gradient flows to the source VERTICES through the barycentric weights.
#[inline(always)]
pub fn chamfer_ba_term_at(
    i: usize,
    tverts: &[Vec3],
    corr: &[MeshQuery],
    spoints: &[Vec3],
    sindices: &[i32],
) -> f32 {
    let q = corr[i];
    if q.face < 0 {
        return 0.0;
    }
    let p = mesh_eval_position(spoints, sindices, q.face, q.u, q.v);
    0.5 * (p - tverts[i]).length_sq()
}

/// THE canonical gather→scatter (forecast by Day 5's unlocks table):
/// p = a·u + b·v + c·w  ⟹  ā += u·r·l̄, b̄ += v·r·l̄, c̄ += w·r·l̄  with r = p − t.
/// Neighbouring target vertices share source triangles — the accumulate MUST
/// be atomic, which is exactly what GradSink guarantees on both backends.
#[inline(always)]
pub fn adj_chamfer_ba_at<S: GradSink<Vec3>>(
    i: usize,
    tverts: &[Vec3],
    corr: &[MeshQuery],
    spoints: &[Vec3],
    sindices: &[i32],
    adj_loss: &[f32],
    adj_spoints: &S,
) {
    let q = corr[i];
    if q.face < 0 {
        return;
    }
    let p = mesh_eval_position(spoints, sindices, q.face, q.u, q.v);
    let r = (p - tverts[i]) * adj_loss[0];
    let base = (q.face as usize) * 3;
    let w = 1.0 - q.u - q.v;
    adj_spoints.accumulate(sindices[base] as usize, r * q.u);
    adj_spoints.accumulate(sindices[base + 1] as usize, r * q.v);
    adj_spoints.accumulate(sindices[base + 2] as usize, r * w);
}

// ─────────────────────────────────────────────────────────────────────────────
// Chain-check bodies — affine → sin_map → sum_scalar (Day-6 plan §5.8, Part 0)
// ─────────────────────────────────────────────────────────────────────────────

/// y = sin(a) — the middle link of the tape-chain check; a macro-row body.
#[inline(always)]
pub fn sin_at(i: usize, a: &[f32]) -> f32 {
    math::sin(a[i])
}

/// Adjoint of the scalar reduction loss = Σ x: a broadcast — every element
/// receives loss̄. (The forward reduction is a one-line scatter in the kernel.)
#[inline(always)]
pub fn adj_sum_at<S: GradSink<f32>>(i: usize, adj_out: &[f32], adj_x: &S) {
    adj_x.accumulate(i, adj_out[0]);
}
