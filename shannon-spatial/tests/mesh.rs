//! Mesh structure, refit, closest-point, and generator invariants — host-only,
//! no GPU (Day-5 plan §5.6). The CPU query is FULLY validated against brute
//! force here, before any device build exists.

use shannon_core::mesh::{mesh_eval_position, mesh_query_point, triangle_is_sliver};
use shannon_core::{BvhNode, Vec3};
use shannon_spatial::shapes::{grid, icosphere, torus};
use shannon_spatial::{
    brute_force_closest_point, build_median_split, refit_nodes, triangle_aabbs,
};
use std::collections::HashSet;

/// Minimal LCG (Knuth MMIX constants), 24-bit mantissa → uniform [0, 1).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) & 0x00FF_FFFF) as f32 / 16_777_216.0
    }
    fn vec3(&mut self) -> Vec3 {
        Vec3::new(self.next_f32(), self.next_f32(), self.next_f32())
    }
}

/// The Day-5 comparison rule (plan §5.9): never compare face/u/v across
/// implementations — equidistant-triangle ties make the winning face
/// legitimately implementation-dependent. Compare found-flags, distances, and
/// per-implementation barycentric self-consistency only.
fn assert_queries_agree(
    label: &str,
    got: &shannon_core::MeshQuery,
    brute: &shannon_core::MeshQuery,
    points: &[Vec3],
    indices: &[i32],
    query_point: Vec3,
) {
    assert_eq!(got.face >= 0, brute.face >= 0, "{label}: found-flags diverge");
    if got.face < 0 {
        return;
    }
    let tol = 1e-4 * 1.0f32.max(brute.dist);
    assert!(
        (got.dist - brute.dist).abs() <= tol,
        "{label}: dist {} vs brute {} (tol {tol})",
        got.dist,
        brute.dist
    );
    // Self-consistency pins the u/v/w convention end-to-end.
    let cp = mesh_eval_position(points, indices, got.face, got.u, got.v);
    assert!(
        ((cp - query_point).length() - got.dist).abs() <= 1e-4 * 1.0f32.max(got.dist),
        "{label}: eval_position(face,u,v) disagrees with reported dist"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The refit contract on the builder
// ─────────────────────────────────────────────────────────────────────────────

/// Every interior node's children have strictly greater indices — the
/// pre-order property `refit_nodes`'s reverse loop depends on (plan §4.4).
/// If a future builder (SAH, LBVH) breaks this, THIS test fails loudly
/// instead of refit silently corrupting every bound.
#[test]
fn builder_children_follow_parent() {
    let (points, indices) = icosphere(3, 1.0);
    let nodes = build_median_split(&triangle_aabbs(&points, &indices));
    for (id, n) in nodes.iter().enumerate() {
        if n.right >= 0 {
            assert!(
                n.left as usize > id && n.right as usize > id,
                "node {id}: child indices {} / {} not strictly greater",
                n.left,
                n.right
            );
        }
    }
}

fn assert_containment(nodes: &[BvhNode], points: &[Vec3], indices: &[i32]) {
    for (id, n) in nodes.iter().enumerate() {
        if n.right < 0 {
            let base = (n.left as usize) * 3;
            let (lo, hi) = shannon_core::mesh::triangle_aabb(
                points[indices[base] as usize],
                points[indices[base + 1] as usize],
                points[indices[base + 2] as usize],
            );
            assert!(
                n.lower.x <= lo.x
                    && n.lower.y <= lo.y
                    && n.lower.z <= lo.z
                    && n.upper.x >= hi.x
                    && n.upper.y >= hi.y
                    && n.upper.z >= hi.z,
                "leaf {id} does not contain its triangle after refit"
            );
        } else {
            for child in [n.left, n.right] {
                let c = &nodes[child as usize];
                assert!(
                    n.lower.x <= c.lower.x
                        && n.lower.y <= c.lower.y
                        && n.lower.z <= c.lower.z
                        && n.upper.x >= c.upper.x
                        && n.upper.y >= c.upper.y
                        && n.upper.z >= c.upper.z,
                    "child {child} escapes parent {id} after refit"
                );
            }
        }
    }
}

/// Scramble the points, refit, and demand exact containment everywhere —
/// union bounds are cw_min/cw_max (exact float ops), so no epsilon.
#[test]
fn refit_restores_containment() {
    let (mut points, indices) = icosphere(3, 1.0);
    let mut nodes = build_median_split(&triangle_aabbs(&points, &indices));

    let mut rng = Lcg::new(42);
    for p in points.iter_mut() {
        *p += (rng.vec3() - Vec3::splat(0.5)) * 0.7; // aggressive scramble
    }
    refit_nodes(&mut nodes, &points, &indices);
    assert_containment(&nodes, &points, &indices);
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU query vs brute force — the oracle, no GPU involved
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpu_query_matches_brute_force() {
    let mut rng = Lcg::new(42);
    let meshes =
        [("icosphere", icosphere(3, 1.0)), ("torus", torus(24, 16, 1.0, 0.35)), ("grid", grid(16, 2.0))];
    for (name, (points, indices)) in &meshes {
        let nodes = build_median_split(&triangle_aabbs(points, indices));
        for _ in 0..256 {
            // Queries in a box 2× the unit-ish meshes — a near/far mix.
            let p = (rng.vec3() - Vec3::splat(0.5)) * 4.0;
            let got = mesh_query_point(&nodes, points, indices, p, 1.0e10);
            let brute = brute_force_closest_point(points, indices, p, 1.0e10);
            assert_queries_agree(name, &got, &brute, points, indices, p);
        }
    }
}

/// found ⟺ brute dist strictly < max_dist, swept across a distance ladder.
#[test]
fn query_respects_max_dist() {
    let (points, indices) = icosphere(2, 1.0);
    let nodes = build_median_split(&triangle_aabbs(&points, &indices));
    let mut rng = Lcg::new(7);
    for _ in 0..64 {
        let p = (rng.vec3() - Vec3::splat(0.5)) * 6.0;
        let true_dist = brute_force_closest_point(&points, &indices, p, 1.0e10).dist;
        for max_dist in [0.05f32, 0.2, 0.5, 1.0, 2.0, 4.0] {
            let q = mesh_query_point(&nodes, &points, &indices, p, max_dist);
            // Skip the knife-edge: strict-< under float noise is genuinely
            // ambiguous within a hair of the boundary.
            if (true_dist - max_dist).abs() < 1e-5 {
                continue;
            }
            assert_eq!(
                q.face >= 0,
                true_dist < max_dist,
                "max_dist {max_dist}: found={} but true dist {true_dist}",
                q.face >= 0
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Generator invariants
// ─────────────────────────────────────────────────────────────────────────────

fn euler_characteristic(n_verts: usize, indices: &[i32]) -> i64 {
    let mut edges: HashSet<(i32, i32)> = HashSet::new();
    for tri in indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            edges.insert(if a < b { (a, b) } else { (b, a) });
        }
    }
    n_verts as i64 - edges.len() as i64 + (indices.len() / 3) as i64
}

#[test]
fn generators_satisfy_euler() {
    for k in 0..4u32 {
        let (points, indices) = icosphere(k, 1.0);
        assert_eq!(points.len(), 10 * 4usize.pow(k) + 2, "icosphere V at subdiv {k}");
        assert_eq!(indices.len() / 3, 20 * 4usize.pow(k), "icosphere F at subdiv {k}");
        assert_eq!(euler_characteristic(points.len(), &indices), 2, "sphere χ at subdiv {k}");
    }
    let (points, indices) = torus(24, 16, 1.0, 0.35);
    assert_eq!(points.len(), 24 * 16);
    assert_eq!(indices.len() / 3, 2 * 24 * 16);
    assert_eq!(euler_characteristic(points.len(), &indices), 0, "torus χ");

    let (points, indices) = grid(16, 2.0);
    assert_eq!(points.len(), 17 * 17);
    assert_eq!(indices.len() / 3, 2 * 16 * 16);
    assert_eq!(euler_characteristic(points.len(), &indices), 1, "grid (disk) χ");
}

#[test]
fn icosphere_is_spherical_and_outward() {
    let r = 2.5f32;
    let (points, indices) = icosphere(3, r);
    for p in &points {
        assert!((p.length() - r).abs() < 1e-5, "vertex off the sphere: |v| = {}", p.length());
    }
    for tri in indices.chunks_exact(3) {
        let (a, b, c) =
            (points[tri[0] as usize], points[tri[1] as usize], points[tri[2] as usize]);
        let normal = (b - a).cross(c - a);
        let centroid = (a + b + c) * (1.0 / 3.0);
        assert!(normal.dot(centroid) > 0.0, "inward-facing icosphere triangle");
    }
}

#[test]
fn grid_normals_point_up() {
    let (points, indices) = grid(8, 2.0);
    for tri in indices.chunks_exact(3) {
        let (a, b, c) =
            (points[tri[0] as usize], points[tri[1] as usize], points[tri[2] as usize]);
        assert!((b - a).cross(c - a).y > 0.0, "grid triangle not wound for +y");
    }
}

#[test]
fn torus_normals_point_outward() {
    let (points, indices) = torus(24, 16, 1.0, 0.35);
    for tri in indices.chunks_exact(3) {
        let (a, b, c) =
            (points[tri[0] as usize], points[tri[1] as usize], points[tri[2] as usize]);
        let centroid = (a + b + c) * (1.0 / 3.0);
        // Outward = away from the tube's center circle at the same major angle.
        let theta = centroid.z.atan2(centroid.x);
        let tube_center = Vec3::new(theta.cos(), 0.0, theta.sin());
        let outward = centroid - tube_center;
        assert!((b - a).cross(c - a).dot(outward) > 0.0, "inward-facing torus triangle");
    }
}

#[test]
fn no_degenerate_triangles() {
    for (name, (points, indices)) in [
        ("icosphere", icosphere(3, 1.0)),
        ("torus", torus(24, 16, 1.0, 0.35)),
        ("grid", grid(16, 2.0)),
    ] {
        for (face, tri) in indices.chunks_exact(3).enumerate() {
            assert!(
                !triangle_is_sliver(
                    points[tri[0] as usize],
                    points[tri[1] as usize],
                    points[tri[2] as usize]
                ),
                "{name}: degenerate triangle {face}"
            );
        }
    }
}
