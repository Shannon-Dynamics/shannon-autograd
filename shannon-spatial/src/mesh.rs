//! `Mesh` — a triangle mesh with a BVH over its triangle AABBs, plus the host
//! refit path and the closest-point brute-force oracle (Day-5 plan §5.4).

use anyhow::Result;
use shannon_core::mesh::{MeshQuery, closest_point_on_triangle, triangle_aabb, triangle_is_sliver};
use shannon_core::{BvhNode, Vec3};
use shannon_rt::Array;

use crate::build_median_split;

/// Per-triangle AABBs in face order — what `build_median_split` consumes.
pub fn triangle_aabbs(points: &[Vec3], indices: &[i32]) -> Vec<(Vec3, Vec3)> {
    (0..indices.len() / 3)
        .map(|face| {
            let base = face * 3;
            triangle_aabb(
                points[indices[base] as usize],
                points[indices[base + 1] as usize],
                points[indices[base + 2] as usize],
            )
        })
        .collect()
}

/// Refresh every node AABB from the current points, in place.
///
/// A single REVERSE index loop is a post-order traversal here because the
/// Day-4 pre-order builder reserves the parent's slot before recursing — every
/// child index is strictly greater than its parent's, so children are always
/// refreshed before their parent reads them. No recursion, no parent pointers,
/// no stack. (Warp's host refit is the recursive equivalent, bvh.cpp:696; the
/// GPU refit — parent pointers + atomic arrival counters, bvh.cu:42 — is the
/// week-2 upgrade.)
///
/// This is a silent dependency on a builder implementation detail: the
/// `debug_assert!` below and the `builder_children_follow_parent` contract
/// test make a future builder swap fail loudly instead of corrupting bounds.
pub fn refit_nodes(nodes: &mut [BvhNode], points: &[Vec3], indices: &[i32]) {
    for id in (0..nodes.len()).rev() {
        let n = nodes[id];
        let (lo, hi) = if n.right < 0 {
            let base = (n.left as usize) * 3;
            triangle_aabb(
                points[indices[base] as usize],
                points[indices[base + 1] as usize],
                points[indices[base + 2] as usize],
            )
        } else {
            debug_assert!(
                n.left as usize > id && n.right as usize > id,
                "refit's reverse loop requires the builder's child > parent pre-order"
            );
            let l = nodes[n.left as usize];
            let r = nodes[n.right as usize];
            (l.lower.cw_min(r.lower), l.upper.cw_max(r.upper))
        };
        nodes[id].lower = lo;
        nodes[id].upper = hi;
    }
}

/// Closest point over ALL triangles — the validation oracle. Same sliver
/// predicate as the real query (Day-4 same-predicate discipline), so a
/// mismatch proves a broken TREE, not a divergent per-triangle decision.
pub fn brute_force_closest_point(
    points: &[Vec3],
    indices: &[i32],
    p: Vec3,
    max_dist: f32,
) -> MeshQuery {
    let mut best = MeshQuery::miss();
    let mut best_sq = max_dist * max_dist;
    for face in 0..indices.len() / 3 {
        let base = face * 3;
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
                face: face as i32,
                u,
                v,
                dist: 0.0,
            };
        }
    }
    if best.face >= 0 {
        best.dist = best_sq.sqrt();
    }
    best
}

/// A triangle mesh resident on the device: points (the deform target), an
/// immutable index buffer, and a BVH over triangle AABBs refreshed by
/// [`Mesh::refit`].
///
/// Deliberately does NOT compose Day-4's `Bvh` wrapper — refit needs a host
/// node mirror, which `Bvh` doesn't carry; wrapping it would only hide state.
/// Kernels never see `Mesh` either: they take three plain slices
/// `&[BvhNode]`, `&[Vec3]`, `&[i32]` (Day-4 precedent — no mesh-id
/// indirection; cuda-oxide slices are already fat pointers).
pub struct Mesh {
    points: Array<Vec3>,      // device — single source of truth for geometry
    indices: Array<i32>,      // device, immutable, 3 per triangle
    nodes: Array<BvhNode>,    // device, refreshed by refit()
    host_nodes: Vec<BvhNode>, // host mirror; refit updates bounds in place
    host_indices: Vec<i32>,   // host mirror for refit / CPU adapters / oracles
    n_tris: usize,
}

impl Mesh {
    /// Build the BVH over triangle AABBs on the host and upload everything.
    pub fn new(points: &[Vec3], indices: &[i32]) -> Result<Self> {
        anyhow::ensure!(
            indices.len().is_multiple_of(3),
            "indices must be 3 per triangle"
        );
        anyhow::ensure!(
            indices
                .iter()
                .all(|&i| (i as usize) < points.len() && i >= 0),
            "triangle index out of range"
        );
        let host_nodes = build_median_split(&triangle_aabbs(points, indices));
        Ok(Self {
            points: Array::from_slice(points)?,
            indices: Array::from_slice(indices)?,
            nodes: Array::from_slice(&host_nodes)?,
            host_nodes,
            host_indices: indices.to_vec(),
            n_tris: indices.len() / 3,
        })
    }

    /// Refresh every node AABB from the CURRENT device points.
    ///
    /// Downloads the true device positions rather than recomputing the deform
    /// on a host mirror: host `libm` sin and device `__nv_sinf` differ by
    /// ulps, and host-recomputed bounds could fail to contain the actual
    /// device triangles — silently breaking the one invariant distance
    /// culling rests on. ~51 KB down + ~0.5 MB up per frame at demo sizes.
    pub fn refit(&mut self) -> Result<()> {
        let pts = self.points.to_vec()?;
        refit_nodes(&mut self.host_nodes, &pts, &self.host_indices);
        self.nodes.copy_from_slice(&self.host_nodes)
    }

    pub fn points(&self) -> &Array<Vec3> {
        &self.points
    }
    /// The deform kernel's output target.
    pub fn points_mut(&mut self) -> &mut Array<Vec3> {
        &mut self.points
    }
    pub fn indices(&self) -> &Array<i32> {
        &self.indices
    }
    pub fn nodes(&self) -> &Array<BvhNode> {
        &self.nodes
    }
    /// Current host node mirror (post-refit) — for CPU adapters and oracles.
    pub fn host_nodes(&self) -> &[BvhNode] {
        &self.host_nodes
    }
    pub fn host_indices(&self) -> &[i32] {
        &self.host_indices
    }
    pub fn n_tris(&self) -> usize {
        self.n_tris
    }
}
