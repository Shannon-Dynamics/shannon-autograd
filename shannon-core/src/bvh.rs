//! BVH node layout and the shared traversal (Day-4 plan §4).
//!
//! The BUILDER lives in `shannon-spatial` — it needs `Vec` and sorting, which
//! this `no_std` crate deliberately cannot have. This module is the
//! device-safe half: the node struct and the two stack traversals, compiled
//! identically into GPU kernels and the rayon CPU adapters.
//!
//! Layout contract: `right < 0` marks a leaf whose PRIMITIVE index is `left`;
//! interior nodes hold child indices in `left`/`right` (both ≥ 0). Node 0 is
//! the root. Every node's bounds contain everything beneath it.
//! (Deviation from week-plan §9.4's `!right` packing, recorded in the Day-4
//! plan §4.1: one uniform sign test, no bit tricks, same 32 bytes.)

use crate::math;
use crate::vec::Vec3;

/// One BVH node. 32 bytes, two per cache line.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BvhNode {
    pub lower: Vec3,
    pub left: i32,
    pub upper: Vec3,
    pub right: i32,
}

const _: () = assert!(core::mem::size_of::<BvhNode>() == 32);

/// Traversal stack capacity. A balanced median split is ⌈log₂ n⌉ deep — 17 at
/// 100k primitives — so 64 is margin, not hope. The push is guarded anyway:
/// an overflow drops work instead of trapping the GPU, and the three-way
/// hit-set validation (Day-4 plan §2) would surface any drop.
pub const BVH_STACK: usize = 64;

/// Componentwise closed-interval overlap between two AABBs. Symmetric.
#[inline(always)]
pub fn aabb_overlaps(a_lo: Vec3, a_hi: Vec3, b_lo: Vec3, b_hi: Vec3) -> bool {
    a_lo.x <= b_hi.x
        && a_hi.x >= b_lo.x
        && a_lo.y <= b_hi.y
        && a_hi.y >= b_lo.y
        && a_lo.z <= b_hi.z
        && a_hi.z >= b_lo.z
}

/// Per-component reciprocal of a ray direction. Zero components become ±∞,
/// which `ray_hits_aabb` is built to digest.
#[inline(always)]
pub fn inv_dir(dir: Vec3) -> Vec3 {
    Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z)
}

/// Slab test over the ray segment `[0, t_max]`, with precomputed `inv`.
///
/// NaN discipline (Day-4 plan §4.4): a zero direction component with the
/// origin EXACTLY on that slab's boundary yields `0 × ∞ = NaN` slab times.
/// `math::min`/`math::max` are single-comparison selects, so those NaNs
/// resolve deterministically instead of poisoning the interval — and, decisive
/// for validation, the brute-force reference calls THIS function, so per-box
/// decisions agree between tree and reference by construction.
#[inline(always)]
pub fn ray_hits_aabb(lower: Vec3, upper: Vec3, origin: Vec3, inv: Vec3, t_max: f32) -> bool {
    let t0x = (lower.x - origin.x) * inv.x;
    let t1x = (upper.x - origin.x) * inv.x;
    let t0y = (lower.y - origin.y) * inv.y;
    let t1y = (upper.y - origin.y) * inv.y;
    let t0z = (lower.z - origin.z) * inv.z;
    let t1z = (upper.z - origin.z) * inv.z;
    let tmin = math::max(
        math::max(math::min(t0x, t1x), math::min(t0y, t1y)),
        math::max(math::min(t0z, t1z), 0.0),
    );
    let tmax = math::min(
        math::min(math::max(t0x, t1x), math::max(t0y, t1y)),
        math::min(math::max(t0z, t1z), t_max),
    );
    tmin <= tmax
}

// ─────────────────────────────────────────────────────────────────────────────
// AABB query
// ─────────────────────────────────────────────────────────────────────────────

/// Iterator over the indices of primitives whose bounds overlap `[lo, hi]`.
pub struct AabbQuery<'a> {
    nodes: &'a [BvhNode],
    stack: [i32; BVH_STACK],
    sp: usize,
    lo: Vec3,
    hi: Vec3,
}

/// Start an AABB query. An empty tree yields an exhausted iterator.
#[inline(always)]
pub fn query_aabb(nodes: &[BvhNode], lo: Vec3, hi: Vec3) -> AabbQuery<'_> {
    // stack[0] starts as 0 — the root index — so a non-empty tree only sets sp.
    let sp = usize::from(!nodes.is_empty());
    AabbQuery { nodes, stack: [0; BVH_STACK], sp, lo, hi }
}

impl Iterator for AabbQuery<'_> {
    type Item = i32;

    #[inline(always)]
    fn next(&mut self) -> Option<i32> {
        while self.sp > 0 {
            self.sp -= 1;
            let n = self.nodes[self.stack[self.sp] as usize];
            if !aabb_overlaps(self.lo, self.hi, n.lower, n.upper) {
                continue;
            }
            if n.right < 0 {
                return Some(n.left); // leaf → primitive index
            }
            if self.sp + 2 <= BVH_STACK {
                self.stack[self.sp] = n.left;
                self.stack[self.sp + 1] = n.right;
                self.sp += 2;
            }
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray query
// ─────────────────────────────────────────────────────────────────────────────

/// Iterator over the indices of primitives whose bounds the ray segment hits.
pub struct RayQuery<'a> {
    nodes: &'a [BvhNode],
    stack: [i32; BVH_STACK],
    sp: usize,
    origin: Vec3,
    inv: Vec3,
    t_max: f32,
}

/// Start a ray query over `[0, t_max]`. An empty tree yields an exhausted
/// iterator; a zero direction component is legal (see `ray_hits_aabb`).
#[inline(always)]
pub fn query_ray(nodes: &[BvhNode], origin: Vec3, dir: Vec3, t_max: f32) -> RayQuery<'_> {
    let sp = usize::from(!nodes.is_empty());
    RayQuery { nodes, stack: [0; BVH_STACK], sp, origin, inv: inv_dir(dir), t_max }
}

impl Iterator for RayQuery<'_> {
    type Item = i32;

    #[inline(always)]
    fn next(&mut self) -> Option<i32> {
        while self.sp > 0 {
            self.sp -= 1;
            let n = self.nodes[self.stack[self.sp] as usize];
            if !ray_hits_aabb(n.lower, n.upper, self.origin, self.inv, self.t_max) {
                continue;
            }
            if n.right < 0 {
                return Some(n.left); // leaf → primitive index
            }
            if self.sp + 2 <= BVH_STACK {
                self.stack[self.sp] = n.left;
                self.stack[self.sp + 1] = n.right;
                self.sp += 2;
            }
        }
        None
    }
}
