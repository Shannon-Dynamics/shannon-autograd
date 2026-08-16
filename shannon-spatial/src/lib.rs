//! shannon-spatial — spatial acceleration structures. 📦 Shipped.
//!
//! Day 4: a median-split BVH — built HERE on the host (this is the project's
//! first host-only geometry code: it needs `Vec` and sorting, which `no_std`
//! shannon-core cannot have), traversed by `shannon_core::bvh` on both
//! backends. Day 5 adds `Mesh` — a BVH over triangle AABBs plus closest-point
//! queries — in this same crate ([`mesh`]), with procedural generators
//! ([`shapes`] — not `gen`, a reserved keyword in edition 2024).

pub mod mesh;
pub mod shapes;

pub use mesh::{Mesh, brute_force_closest_point, refit_nodes, triangle_aabbs};

use anyhow::Result;
use shannon_core::bvh::{aabb_overlaps, inv_dir, ray_hits_aabb};
use shannon_core::{BvhNode, Vec3};
use shannon_rt::Array;

/// Build a median-split BVH over primitive AABBs. Pure host function — the
/// structure and traversal tests need no GPU.
///
/// Recursion: split the primitive indices at the median of the longest axis
/// of the CENTROID bounds (`select_nth_unstable_by` — O(n), duplicate-safe),
/// one primitive per leaf. Balanced by construction ⟹ depth ≈ ⌈log₂ n⌉, which
/// is what makes `shannon_core::bvh::BVH_STACK` = 64 margin rather than hope.
///
/// Returns `2n − 1` nodes with the root at index 0; empty input ⟹ empty tree.
pub fn build_median_split(aabbs: &[(Vec3, Vec3)]) -> Vec<BvhNode> {
    if aabbs.is_empty() {
        return Vec::new();
    }
    let centroids: Vec<Vec3> = aabbs.iter().map(|&(lo, hi)| (lo + hi) * 0.5).collect();
    let mut indices: Vec<i32> = (0..aabbs.len() as i32).collect();
    let mut nodes = Vec::with_capacity(2 * aabbs.len() - 1);
    recurse(aabbs, &centroids, &mut indices, &mut nodes);
    nodes
}

/// Depth-first node emission: reserve this node's slot pre-order (so the root
/// is index 0), then fill it once the children report their indices.
fn recurse(
    aabbs: &[(Vec3, Vec3)],
    centroids: &[Vec3],
    indices: &mut [i32],
    nodes: &mut Vec<BvhNode>,
) -> i32 {
    let id = nodes.len() as i32;
    nodes.push(BvhNode::default()); // reserved; overwritten below

    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &i in indices.iter() {
        lo = lo.cw_min(aabbs[i as usize].0);
        hi = hi.cw_max(aabbs[i as usize].1);
    }

    if let [prim] = *indices {
        nodes[id as usize] = BvhNode {
            lower: lo,
            left: prim,
            upper: hi,
            right: -1,
        };
        return id;
    }

    let mut c_lo = Vec3::splat(f32::INFINITY);
    let mut c_hi = Vec3::splat(f32::NEG_INFINITY);
    for &i in indices.iter() {
        c_lo = c_lo.cw_min(centroids[i as usize]);
        c_hi = c_hi.cw_max(centroids[i as usize]);
    }
    let ext = c_hi - c_lo;
    let axis = if ext.x >= ext.y && ext.x >= ext.z {
        0
    } else if ext.y >= ext.z {
        1
    } else {
        2
    };

    let mid = indices.len() / 2; // len ≥ 2 ⟹ both halves non-empty
    indices.select_nth_unstable_by(mid, |&a, &b| {
        centroids[a as usize]
            .component(axis)
            .partial_cmp(&centroids[b as usize].component(axis))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (left_half, right_half) = indices.split_at_mut(mid);
    let left = recurse(aabbs, centroids, left_half, nodes);
    let right = recurse(aabbs, centroids, right_half, nodes);
    nodes[id as usize] = BvhNode {
        lower: lo,
        left,
        upper: hi,
        right,
    };
    id
}

// ─────────────────────────────────────────────────────────────────────────────
// Brute-force references — O(n) per query
// ─────────────────────────────────────────────────────────────────────────────
// The validation oracle for both backends AND the honest denominator behind
// any "BVH speedup" claim. They call the SAME per-box predicates as the tree
// traversal, so per-box decisions agree by construction; what a mismatch would
// prove is a broken TREE (missed subtree, duplicate visit), which is exactly
// what the three-way hit-set comparison exists to catch.

/// Indices of all primitives whose AABB overlaps `[lo, hi]`, ascending.
pub fn brute_force_aabb(aabbs: &[(Vec3, Vec3)], lo: Vec3, hi: Vec3) -> Vec<i32> {
    aabbs
        .iter()
        .enumerate()
        .filter(|&(_, &(b_lo, b_hi))| aabb_overlaps(lo, hi, b_lo, b_hi))
        .map(|(i, _)| i as i32)
        .collect()
}

/// Indices of all primitives whose AABB the ray segment hits, ascending.
pub fn brute_force_ray(aabbs: &[(Vec3, Vec3)], origin: Vec3, dir: Vec3, t_max: f32) -> Vec<i32> {
    let inv = inv_dir(dir);
    aabbs
        .iter()
        .enumerate()
        .filter(|&(_, &(b_lo, b_hi))| ray_hits_aabb(b_lo, b_hi, origin, inv, t_max))
        .map(|(i, _)| i as i32)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Host wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// A built BVH resident on the device, ready to pass to kernels via
/// `.nodes()`. Day 5's `Mesh` composes one of these over triangle AABBs.
pub struct Bvh {
    nodes: Array<BvhNode>,
    n_prims: usize,
}

impl Bvh {
    /// Build on the host (median split) and upload.
    pub fn build(aabbs: &[(Vec3, Vec3)]) -> Result<Self> {
        let nodes = build_median_split(aabbs);
        Ok(Self {
            nodes: Array::from_slice(&nodes)?,
            n_prims: aabbs.len(),
        })
    }

    /// The device-resident node buffer — what kernels take as `&[BvhNode]`.
    pub fn nodes(&self) -> &Array<BvhNode> {
        &self.nodes
    }

    pub fn n_prims(&self) -> usize {
        self.n_prims
    }
}
