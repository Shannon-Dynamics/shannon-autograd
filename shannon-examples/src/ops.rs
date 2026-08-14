//! The taped-op layer (Day-6 plan §5.7). 🧪
//!
//! One function per differentiable op the W3 demo uses: forward `launch!`,
//! then `tape.record` of the adjoint launch. Two shapes:
//!
//! - **Accumulate-shaped** (loss reductions): everything by shared reference —
//!   the forward kernel writes `loss[0]` through const-ref scatter, so the
//!   taped phase never needs `&mut` on the loss at all.
//! - **Map-shaped** (`DisjointSlice` outputs): the op takes its output as
//!   `&'a mut`, launches forward, then DOWNGRADES the reference and returns
//!   `&'a` — statically freezing the taped intermediate until `backward`
//!   consumes the tape. The overwrite hazard as a compile error (§4.1).
//!
//! Plus `begin_iteration` — phase 1 of the three-phase choreography (§4.2)
//! as one call, so the zero+reset+seed checklist cannot be forgotten in
//! parts (pitfalls 1, 2, 7; the Warp-`tape.zero()` idea, warp/_src/tape.py:288,
//! adapted to the borrow-scoped design).

use anyhow::{Result, anyhow};
use shannon_autodiff::Tape;
use shannon_core::{MeshQuery, Vec3};
use shannon_rt::{Array, launch};

/// Phase 1 in one call: zero every listed gradient, reset the accumulated
/// loss VALUE, seed `loss.grad = 1` — all BEFORE anything records. Two
/// concrete buffer lists because `Array<T>` is generic; Vec3 parameters/
/// intermediates and the f32 loss (plus f32 chain intermediates) are Day 6's
/// only cases.
pub fn begin_iteration(
    vec3_grads: &mut [&mut Array<Vec3>],
    f32_grads: &mut [&mut Array<f32>],
    loss: &mut Array<f32>,
) -> Result<()> {
    for a in vec3_grads.iter_mut() {
        a.zero_grad()?;
    }
    for a in f32_grads.iter_mut() {
        a.zero_grad()?;
    }
    loss.zero_grad()?;
    loss.copy_from_slice(&[0.0])?; // forward loss accumulates — reset the VALUE
    loss.grad_mut()
        .ok_or_else(|| anyhow!("loss has no grad — call requires_grad_(true) first"))?
        .copy_from_slice(&[1.0]) // the seed, BEFORE any record (§4.2)
}

// ─────────────────────────────────────────────────────────────────────────────
// Map-shaped ops — forward &mut, downgrade, return the frozen shared view
// ─────────────────────────────────────────────────────────────────────────────

/// y = a·scale + bias — Day-1's W0 kernels as a tape citizen.
pub fn affine_op<'a>(
    tape: &mut Tape<'a>,
    a: &'a Array<f32>,
    scale: f32,
    bias: f32,
    y: &'a mut Array<f32>,
) -> Result<&'a Array<f32>> {
    let n = a.len();
    launch!(affine, dim = n, (a, scale, bias, &mut *y))?;
    let y_sh: &'a Array<f32> = y; // ← the downgrade: y is frozen until backward
    tape.record("affine", n, move || {
        let gy = y_sh.grad().ok_or_else(|| anyhow!("affine: y has no grad"))?;
        let ga = a.grad().ok_or_else(|| anyhow!("affine: a has no grad"))?;
        launch!(adj_affine, dim = n, (gy, scale, ga))
    });
    Ok(y_sh)
}

/// y = sin(a).
pub fn sin_map_op<'a>(
    tape: &mut Tape<'a>,
    a: &'a Array<f32>,
    y: &'a mut Array<f32>,
) -> Result<&'a Array<f32>> {
    let n = a.len();
    launch!(sin_map, dim = n, (a, &mut *y))?;
    let y_sh: &'a Array<f32> = y;
    tape.record("sin_map", n, move || {
        let gy = y_sh.grad().ok_or_else(|| anyhow!("sin_map: y has no grad"))?;
        let ga = a.grad().ok_or_else(|| anyhow!("sin_map: a has no grad"))?;
        launch!(adj_sin_map, dim = n, (a, gy, ga))
    });
    Ok(y_sh)
}

// ─────────────────────────────────────────────────────────────────────────────
// Accumulate-shaped ops — shared refs only; loss written by const-ref scatter
// ─────────────────────────────────────────────────────────────────────────────

/// loss[0] += Σ x[i].
pub fn sum_op<'a>(tape: &mut Tape<'a>, x: &'a Array<f32>, loss: &'a Array<f32>) -> Result<()> {
    let n = x.len();
    launch!(sum_scalar, dim = n, (x, loss))?;
    tape.record("sum_scalar", n, move || {
        let gl = loss.grad().ok_or_else(|| anyhow!("sum: loss has no grad"))?;
        let gx = x.grad().ok_or_else(|| anyhow!("sum: x has no grad"))?;
        launch!(adj_sum_scalar, dim = n, (gl, gx))
    });
    Ok(())
}

/// loss[0] += Σ ½‖xᵢ − tᵢ‖² — stage 1's oracle loss.
pub fn l2_loss_op<'a>(
    tape: &mut Tape<'a>,
    x: &'a Array<Vec3>,
    t: &'a Array<Vec3>,
    loss: &'a Array<f32>,
) -> Result<()> {
    let n = x.len();
    launch!(l2_loss, dim = n, (x, t, loss))?;
    tape.record("l2_loss", n, move || {
        let gl = loss.grad().ok_or_else(|| anyhow!("l2: loss has no grad"))?;
        let gx = x.grad().ok_or_else(|| anyhow!("l2: x has no grad"))?;
        launch!(adj_l2_loss, dim = n, (x, t, gl, gx))
    });
    Ok(())
}

/// A→B Chamfer under held correspondence (envelope theorem — the adjoint
/// needs no query).
pub fn chamfer_ab_op<'a>(
    tape: &mut Tape<'a>,
    src: &'a Array<Vec3>,
    corr: &'a Array<MeshQuery>,
    tpoints: &'a Array<Vec3>,
    tindices: &'a Array<i32>,
    loss: &'a Array<f32>,
) -> Result<()> {
    let n = src.len();
    launch!(chamfer_ab, dim = n, (src, corr, tpoints, tindices, loss))?;
    tape.record("chamfer_ab", n, move || {
        let gl = loss.grad().ok_or_else(|| anyhow!("chamfer_ab: loss has no grad"))?;
        let gs = src.grad().ok_or_else(|| anyhow!("chamfer_ab: src has no grad"))?;
        launch!(adj_chamfer_ab, dim = n, (src, corr, tpoints, tindices, gl, gs))
    });
    Ok(())
}

/// B→A Chamfer: gradients flow to the SOURCE mesh points via the canonical
/// barycentric gather→scatter.
pub fn chamfer_ba_op<'a>(
    tape: &mut Tape<'a>,
    tverts: &'a Array<Vec3>,
    corr: &'a Array<MeshQuery>,
    spoints: &'a Array<Vec3>,
    sindices: &'a Array<i32>,
    loss: &'a Array<f32>,
) -> Result<()> {
    let n = tverts.len();
    launch!(chamfer_ba, dim = n, (tverts, corr, spoints, sindices, loss))?;
    tape.record("chamfer_ba", n, move || {
        let gl = loss.grad().ok_or_else(|| anyhow!("chamfer_ba: loss has no grad"))?;
        let gp = spoints.grad().ok_or_else(|| anyhow!("chamfer_ba: spoints has no grad"))?;
        launch!(adj_chamfer_ba, dim = n, (tverts, corr, spoints, sindices, gl, gp))
    });
    Ok(())
}
