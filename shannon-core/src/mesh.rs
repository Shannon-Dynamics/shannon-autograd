// Portions derived from NVIDIA Warp (https://github.com/NVIDIA/warp),
// SPDX-License-Identifier: Apache-2.0 — see NOTICE at the workspace root.
// Specifically: the 7-region closest-point-on-triangle method
// (warp/native/intersect.h) and the distance-culled, nearest-child-first
// BVH descent structure (warp/native/mesh.h), ported to Rust. The existing
// file:line citations in doc-comments below identify the exact sources.

//! Triangle-mesh geometry: closest-point queries and the W2 simulation bodies
//! (Day-5 plan §3–§4). Pure `no_std` functions compiled identically for both
//! backends — the builder and the host `Mesh` wrapper live in `shannon-spatial`.
//!
//! Barycentric convention, pinned once and tested forever: `(u, v, w)` weigh
//! `(a, b, c)`, `u + v + w = 1`, and every function returns `(…, u, v)` — `w`
//! is ALWAYS derived. Matches Warp (`u` is the `a`-weight, intersect.h:44).

use crate::bvh::{BVH_STACK, BvhNode};
use crate::math;
use crate::vec::Vec3;

/// Result of a closest-point query. POD with a sentinel rather than `Option`:
/// consistent with the `right < 0` leaf convention, valid at all-zero bits
/// (required by `Array::zeros`), and usable as a `DisjointSlice<MeshQuery>`
/// kernel output — which is what lets the GPU query be a macro row.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeshQuery {
    /// Triangle index; miss ⇔ `face == -1`.
    pub face: i32,
    /// Barycentric `a`-weight.
    pub u: f32,
    /// Barycentric `b`-weight (`w = 1 − u − v`).
    pub v: f32,
    /// Unsigned distance; `f32::INFINITY` on miss.
    pub dist: f32,
}

impl MeshQuery {
    pub const fn miss() -> Self {
        Self {
            face: -1,
            u: 0.0,
            v: 0.0,
            dist: f32::INFINITY,
        }
    }
}

/// One simulated particle. A single value so the W2 step kernel fits the
/// macro row shape (`parts[i] → Particle`) — the two-outputs problem (pos AND
/// vel) dissolves into one struct return plus host-side double buffering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Particle {
    pub pos: Vec3,
    pub vel: Vec3,
}

// ─────────────────────────────────────────────────────────────────────────────
// Closest point on a triangle — the 7-region method
// ─────────────────────────────────────────────────────────────────────────────

/// Closest point on triangle `(a, b, c)` to `p`, with barycentrics `(u, v)`.
///
/// Exact port of Warp's `closest_point_to_triangle` (intersect.h:44) — the
/// Ericson (*Real-Time Collision Detection* §5.1.5) Voronoi-region method:
/// six dot products (`d1..d6`) plus three scalar products (`va, vb, vc`)
/// classify `p` into one of 3 vertex, 3 edge, or 1 interior region.
///
/// Branch-heavy but branch-cheap — every path is a handful of FMAs, no trig,
/// no sqrt. The function never leaves squared-distance land; callers compare
/// squares and take ONE `math::sqrt` at the very end.
pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, f32, f32) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, 1.0, 0.0); // region A: vertex a
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, 0.0, 1.0); // region B: vertex b
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, 1.0 - v, v); // region AB: edge ab
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, 0.0, 0.0); // region C: vertex c
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, 1.0 - w, 0.0); // region AC: edge ac
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, 0.0, 1.0 - w); // region BC: edge bc
    }

    let denom = 1.0 / (va + vb + vc); // interior face
    let v = vb * denom;
    let w = vc * denom;
    let u = 1.0 - v - w;
    (a * u + b * v + c * w, u, v)
}

/// Squared distance from `p` to the AABB `[lower, upper]` — clamp `p` into the
/// box per axis, squared length of the offset. Zero when `p` is inside.
/// The culling metric of `mesh_query_point` (Warp mesh.h:88).
#[inline(always)]
pub fn distance_to_aabb_sq(p: Vec3, lower: Vec3, upper: Vec3) -> f32 {
    let dx = math::min(upper.x, math::max(lower.x, p.x)) - p.x;
    let dy = math::min(upper.y, math::max(lower.y, p.y)) - p.y;
    let dz = math::min(upper.z, math::max(lower.z, p.z)) - p.z;
    dx * dx + dy * dy + dz * dz
}

/// AABB of one triangle — `cw_min`/`cw_max` chains, exact float ops.
#[inline(always)]
pub fn triangle_aabb(a: Vec3, b: Vec3, c: Vec3) -> (Vec3, Vec3) {
    (a.cw_min(b).cw_min(c), a.cw_max(b).cw_max(c))
}

/// True when the triangle is too degenerate for a trustworthy closest point:
/// `|e0 × e1| < 1e-6 · Σ|eᵢ|²` — the multiplied form of Warp's sliver ratio
/// test (mesh.h:539): no division, and a zero-area triangle with distinct
/// vertices is skipped without a 0/0.
///
/// SHARED PREDICATE (Day-4 discipline): `mesh_query_point` and the spatial
/// crate's brute-force oracle both call THIS function, so per-triangle
/// accept/skip decisions agree between tree and reference by construction.
#[inline(always)]
pub fn triangle_is_sliver(a: Vec3, b: Vec3, c: Vec3) -> bool {
    let e0 = b - a;
    let e1 = c - b;
    let e2 = a - c;
    let scale = e0.dot(e0) + e1.dot(e1) + e2.dot(e2);
    e0.cross(e1).length() < 1.0e-6 * scale
}

/// Barycentric position on triangle `face`: `a·u + b·v + c·(1 − u − v)`.
/// Shared by the Day-6 Chamfer loss (Warp mesh.h:2570).
#[inline(always)]
pub fn mesh_eval_position(points: &[Vec3], indices: &[i32], face: i32, u: f32, v: f32) -> Vec3 {
    let base = (face as usize) * 3;
    let a = points[indices[base] as usize];
    let b = points[indices[base + 1] as usize];
    let c = points[indices[base + 2] as usize];
    a * u + b * v + c * (1.0 - u - v)
}

// ─────────────────────────────────────────────────────────────────────────────
// The distance-culled query
// ─────────────────────────────────────────────────────────────────────────────

/// Closest point on the mesh to `p`, strictly within `max_dist`.
///
/// Port of Warp's `mesh_query_point_no_sign` (mesh.h:497): a best-first
/// bounded search — NOT an `AabbQuery`-style iterator, and it cannot be one:
/// the running bound `best_sq` shrinks mid-traversal and must influence both
/// the pop-time re-test and the push guards, feedback a `next()` iterator has
/// no channel for. Hand-written stack loop, same `[i32; BVH_STACK]` bones.
///
/// Everything is SQUARED distance; the one `math::sqrt` happens at the end.
/// Acceptance is strict `< max_dist²` (Warp parity — a face exactly at
/// `max_dist` is a miss).
pub fn mesh_query_point(
    nodes: &[BvhNode],
    points: &[Vec3],
    indices: &[i32],
    p: Vec3,
    max_dist: f32,
) -> MeshQuery {
    let mut best = MeshQuery::miss();
    if nodes.is_empty() {
        return best;
    }
    let mut best_sq = max_dist * max_dist; // the shrinking bound
    let mut stack = [0i32; BVH_STACK]; // stack[0] = 0: root is node 0 (Day-4 contract)
    let mut sp = 1usize;

    while sp > 0 {
        sp -= 1;
        let n = nodes[stack[sp] as usize];

        // Pop-time re-test: best_sq may have shrunk since this node was pushed.
        if distance_to_aabb_sq(p, n.lower, n.upper) > best_sq {
            continue;
        }

        if n.right < 0 {
            // Leaf: n.left IS the triangle index — Day 4 built leaf_size = 1
            // with the primitive permutation baked into the leaves, so Warp's
            // leaf-range loop (mesh.h:528) degenerates to a single triangle.
            let face = n.left;
            let base = (face as usize) * 3;
            let a = points[indices[base] as usize];
            let b = points[indices[base + 1] as usize];
            let c = points[indices[base + 2] as usize];

            if triangle_is_sliver(a, b, c) {
                continue;
            }

            let (cp, u, v) = closest_point_on_triangle(p, a, b, c);
            let d_sq = (cp - p).length_sq();
            if d_sq < best_sq {
                best_sq = d_sq;
                best = MeshQuery {
                    face,
                    u,
                    v,
                    dist: 0.0,
                }; // dist patched below
            }
        } else {
            // Interior: order children so the NEARER one is popped first — it
            // tightens best_sq early, which is what makes the culling bite.
            let l = nodes[n.left as usize];
            let r = nodes[n.right as usize];
            let dl = distance_to_aabb_sq(p, l.lower, l.upper);
            let dr = distance_to_aabb_sq(p, r.lower, r.upper);
            let (near, near_d, far, far_d) = if dl <= dr {
                (n.left, dl, n.right, dr)
            } else {
                (n.right, dr, n.left, dl)
            };
            // Push-time guard AND the Day-4 capacity guard. Far first, near last.
            if far_d < best_sq && sp < BVH_STACK {
                stack[sp] = far;
                sp += 1;
            }
            if near_d < best_sq && sp < BVH_STACK {
                stack[sp] = near;
                sp += 1;
            }
        }
    }

    if best.face >= 0 {
        best.dist = math::sqrt(best_sq); // the ONE sqrt in the whole query
    }
    best
}

/// Row-shaped body for the `mesh_query` kernel: one thread per query point.
#[inline(always)]
pub fn mesh_query_point_at(
    i: usize,
    nodes: &[BvhNode],
    points: &[Vec3],
    indices: &[i32],
    queries: &[Vec3],
    max_dist: f32,
) -> MeshQuery {
    mesh_query_point(nodes, points, indices, queries[i], max_dist)
}

// ─────────────────────────────────────────────────────────────────────────────
// W2 simulation bodies
// ─────────────────────────────────────────────────────────────────────────────

/// W2 deform: absolute positions from immutable REST positions — a standing
/// wave `y = rest.y + amp · sin(x) · cos(z) · sin(phase)`, with `phase = ω·t`
/// fed by the host.
///
/// Pure-of-rest (vs Warp's cumulative in-place deform): deterministic, no
/// accumulated drift, and frame n is a function of n — which is what makes the
/// mesh_sim tunnelling predicate ANALYTIC.
///
/// DEVIATION from the Day-5 plan §4.2 (recorded): the plan wrote a travelling
/// wave `amp·sin(x+t)·cos(z+t)`, whose valley pattern translates diagonally at
/// speed √2 — "settled" particles would chase moving minima at |v| far above
/// the settle bound, so the acceptance predicate would be false by
/// construction. The standing wave keeps every CFL number (surface speed
/// ≤ amp·ω < the plan's amp bound) and mirrors Warp's own example_mesh.py
/// deform (fixed spatial pattern × sin(t)) more closely anyway.
#[inline(always)]
pub fn deform_at(i: usize, rest: &[Vec3], phase: f32, amp: f32) -> Vec3 {
    let p = rest[i];
    Vec3::new(
        p.x,
        p.y + amp * math::sin(p.x) * math::cos(p.z) * math::sin(phase),
        p.z,
    )
}

/// Coulomb-style contact friction coefficient. Without friction the pushout
/// is a frictionless bead-on-wire — the normal projection removes bounce, but
/// gravity re-feeds tangential sliding every frame, and on a slope of angle θ
/// particles slide indefinitely at terminal velocity. Coulomb friction gives
/// the real settling criterion: sliding decelerates whenever tan θ < μ, and
/// the demo surface's maximum slope is amp·max|∇(sin x cos z)| = 0.25 < μ —
/// so every particle provably STOPS, on any slope, like sand.
/// (Recorded deviation from strict example_mesh.py force parity, which has no
/// friction and no settle requirement.)
pub const FRICTION_MU: f32 = 0.35;

/// W2 particle step — Warp `example_mesh.py::simulate` forces, minus the sign
/// query, plus the two ingredients that make UNSIGNED collision sound AND
/// settling provable: a CFL speed clamp (max step = margin/2, so a particle
/// at distance ≥ margin can never cross the surface in one frame) and Coulomb
/// contact friction ([`FRICTION_MU`]). Reads `parts[i]`, returns the updated
/// particle — the host double-buffers and swaps (out-of-place, Day-6 tape
/// discipline).
#[allow(clippy::too_many_arguments)] // arity mirrors the kernel row
pub fn sim_particle_at(
    i: usize,
    parts: &[Particle],
    nodes: &[BvhNode],
    points: &[Vec3],
    indices: &[i32],
    margin: f32,
    dt: f32,
    max_dist: f32,
    y_floor: f32,
) -> Particle {
    let x = parts[i].pos;
    let mut v = parts[i].vel;

    // Gravity + drag (Warp parity).
    v = v + Vec3::new(0.0, -9.8, 0.0) * dt - v * (0.1 * dt);

    // CFL clamp — the no-tunnelling guarantee: max step = 0.5·margin.
    let v_max = 0.5 * margin / dt;
    let speed = v.length();
    if speed > v_max {
        v = v * (v_max / speed);
    }

    let mut xpred = x + v * dt;
    let mut in_contact = false;
    let mut contact_n = Vec3::ZERO; // contact normal for the friction split

    let q = mesh_query_point(nodes, points, indices, xpred, max_dist);
    if q.face >= 0 {
        let cp = mesh_eval_position(points, indices, q.face, q.u, q.v);
        let delta = xpred - cp;
        if delta.length() < margin {
            // Unsigned pushout: park the particle on the margin shell.
            // normalize() is EPS-guarded — delta ≈ 0 degenerates to
            // xpred = cp, corrected next frame.
            contact_n = delta.normalize();
            xpred = cp + contact_n * margin;
            in_contact = true;
        }
    }

    // Belt-and-braces floor for anything that leaves the mesh footprint.
    if xpred.y < y_floor + margin {
        xpred.y = y_floor + margin;
        contact_n = Vec3::new(0.0, 1.0, 0.0);
        in_contact = true;
    }

    // PBD velocity update (Warp parity)…
    let mut v_out = (xpred - x) / dt;
    // …then Coulomb friction on contact frames: remove up to μ·g·dt from the
    // TANGENTIAL speed (a degenerate contact_n = ZERO treats all of v_out as
    // tangential, which is the conservative choice). Sliding stops and stays
    // stopped wherever tan θ < μ — which is everywhere on the demo surface.
    if in_contact {
        let v_n = contact_n * v_out.dot(contact_n);
        let v_t = v_out - v_n;
        let t_speed = v_t.length();
        let drop = FRICTION_MU * 9.8 * dt;
        let v_t = if t_speed > drop {
            v_t * ((t_speed - drop) / t_speed)
        } else {
            Vec3::ZERO
        };
        v_out = v_n + v_t;
    }
    Particle {
        pos: xpred,
        vel: v_out,
    }
}
