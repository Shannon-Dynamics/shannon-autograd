//! Signed-distance functions and CSG operators.
//!
//! Ported from the reference scene in Warp's `example_raymarch.py`, itself
//! following <https://iquilezles.org/articles/distfunctions/>

use crate::{Vec3, Vec4, math};

/// Distance to a sphere of radius `r` centred at the origin.
#[inline(always)]
pub fn sphere(p: Vec3, r: f32) -> f32 {
    p.length() - r
}

/// Distance to an axis-aligned box with half-extents `upper`, centred at the origin.
#[inline(always)]
pub fn box_(upper: Vec3, p: Vec3) -> f32 {
    let qx = math::abs(p.x) - upper.x;
    let qy = math::abs(p.y) - upper.y;
    let qz = math::abs(p.z) - upper.z;
    let e = Vec3::new(math::max(qx, 0.0), math::max(qy, 0.0), math::max(qz, 0.0));
    e.length() + math::min(math::max(qx, math::max(qy, qz)), 0.0)
}

/// Distance to the plane `n·p + d = 0`, with `plane = (n.x, n.y, n.z, d)`.
#[inline(always)]
pub fn plane(p: Vec3, plane: Vec4) -> f32 {
    plane.xyz().dot(p) + plane.w
}

/// Distance to a capsule (line segment `a`→`b` with radius `r`).
#[inline(always)]
pub fn capsule(p: Vec3, a: Vec3, b: Vec3, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let denom = ba.dot(ba);
    let h = if denom > crate::EPS { math::clamp(pa.dot(ba) / denom, 0.0, 1.0) } else { 0.0 };
    (pa - ba * h).length() - r
}

/// 2D signed distance to a box with half-extents `(hx, hy)`, centred at origin.
#[inline(always)]
pub fn box2(px: f32, py: f32, hx: f32, hy: f32) -> f32 {
    let qx = math::abs(px) - hx;
    let qy = math::abs(py) - hy;
    let ox = math::max(qx, 0.0);
    let oy = math::max(qy, 0.0);
    math::sqrt(ox * ox + oy * oy) + math::min(math::max(qx, qy), 0.0)
}

/// Extrude a 2D SDF value `d2` along z with half-depth `hz` (exact for boxes).
#[inline(always)]
pub fn extrude(d2: f32, pz: f32, hz: f32) -> f32 {
    let wz = math::abs(pz) - hz;
    let ox = math::max(d2, 0.0);
    let oy = math::max(wz, 0.0);
    math::min(math::max(d2, wz), 0.0) + math::sqrt(ox * ox + oy * oy)
}

// ── CSG ─────────────────────────────────────────────────────────────────────

#[inline(always)]
pub fn op_union(d1: f32, d2: f32) -> f32 {
    math::min(d1, d2)
}

/// NOTE the asymmetry: subtracts `d1` FROM `d2` (matches Warp's `op_subtract`).
#[inline(always)]
pub fn op_subtract(d1: f32, d2: f32) -> f32 {
    math::max(-d1, d2)
}

#[inline(always)]
pub fn op_intersect(d1: f32, d2: f32) -> f32 {
    math::max(d1, d2)
}

/// Polynomial smooth union with blend radius `k` (Quilez's smin).
#[inline(always)]
pub fn op_smooth_union(d1: f32, d2: f32, k: f32) -> f32 {
    let h = math::clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    math::lerp(d2, d1, h) - k * h * (1.0 - h)
}

/// Capped cylinder about the Y axis: half-height `h`, radius `r`, centred at
/// the origin of `p`'s frame.
#[inline(always)]
pub fn capped_cylinder(p: Vec3, h: f32, r: f32) -> f32 {
    let dx = math::sqrt(p.x * p.x + p.z * p.z) - r;
    let dy = math::abs(p.y) - h;
    let ox = math::max(dx, 0.0);
    let oy = math::max(dy, 0.0);
    math::min(math::max(dx, dy), 0.0) + math::sqrt(ox * ox + oy * oy)
}
