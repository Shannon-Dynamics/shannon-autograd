//! Closest-point-on-triangle + query-primitive invariants — host-only, no GPU
//! (Day-5 plan §5.7).
//!
//! Seven named region tests pin the Ericson classification on the canonical
//! right triangle with hand-computed answers; the LCG upper-bound property
//! then catches wrong-region bugs on random data without needing a second
//! closest-point oracle.

use shannon_core::Vec3;
use shannon_core::mesh::{
    MeshQuery, closest_point_on_triangle, distance_to_aabb_sq, mesh_eval_position,
    mesh_query_point, triangle_aabb, triangle_is_sliver,
};

/// Minimal LCG (Knuth MMIX constants), 24-bit mantissa → uniform [0, 1).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) & 0x00FF_FFFF) as f32 / 16_777_216.0
    }
    fn vec3(&mut self) -> Vec3 {
        Vec3::new(self.next_f32(), self.next_f32(), self.next_f32())
    }
}

/// The canonical right triangle used by every region test.
const A: Vec3 = Vec3::new(0.0, 0.0, 0.0);
const B: Vec3 = Vec3::new(1.0, 0.0, 0.0);
const C: Vec3 = Vec3::new(0.0, 1.0, 0.0);

fn assert_close(cp: Vec3, expect: Vec3, u: f32, eu: f32, v: f32, ev: f32) {
    assert!(
        (cp - expect).length() < 1e-6,
        "closest point {cp:?} != expected {expect:?}"
    );
    assert!((u - eu).abs() < 1e-6, "u {u} != {eu}");
    assert!((v - ev).abs() < 1e-6, "v {v} != {ev}");
}

// ─────────────────────────────────────────────────────────────────────────────
// The seven regions, hand-computed
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn region_a_vertex() {
    let (cp, u, v) = closest_point_on_triangle(Vec3::new(-1.0, -1.0, 0.5), A, B, C);
    assert_close(cp, A, u, 1.0, v, 0.0);
}

#[test]
fn region_b_vertex() {
    let (cp, u, v) = closest_point_on_triangle(Vec3::new(2.0, -0.5, -0.3), A, B, C);
    assert_close(cp, B, u, 0.0, v, 1.0);
}

#[test]
fn region_c_vertex() {
    let (cp, u, v) = closest_point_on_triangle(Vec3::new(-0.5, 2.0, 0.1), A, B, C);
    assert_close(cp, C, u, 0.0, v, 0.0); // w = 1
}

#[test]
fn region_ab_edge() {
    let (cp, u, v) = closest_point_on_triangle(Vec3::new(0.5, -1.0, 0.0), A, B, C);
    assert_close(cp, Vec3::new(0.5, 0.0, 0.0), u, 0.5, v, 0.5);
}

#[test]
fn region_ac_edge() {
    let (cp, u, v) = closest_point_on_triangle(Vec3::new(-1.0, 0.5, 0.0), A, B, C);
    assert_close(cp, Vec3::new(0.0, 0.5, 0.0), u, 0.5, v, 0.0); // w = 0.5
}

#[test]
fn region_bc_edge() {
    // Projection of (1, 1) onto the segment b→c lands at its midpoint.
    let (cp, u, v) = closest_point_on_triangle(Vec3::new(1.0, 1.0, 0.0), A, B, C);
    assert_close(cp, Vec3::new(0.5, 0.5, 0.0), u, 0.0, v, 0.5); // w = 0.5
}

#[test]
fn region_interior() {
    let (cp, u, v) = closest_point_on_triangle(Vec3::new(0.25, 0.25, 5.0), A, B, C);
    assert_close(cp, Vec3::new(0.25, 0.25, 0.0), u, 0.5, v, 0.25); // w = 0.25
}

// ─────────────────────────────────────────────────────────────────────────────
// Invariants on random data
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn barycentrics_valid_and_reconstruct() {
    let mut rng = Lcg::new(42);
    for _ in 0..200 {
        let (a, b, c) = (rng.vec3() * 4.0, rng.vec3() * 4.0, rng.vec3() * 4.0);
        if triangle_is_sliver(a, b, c) {
            continue;
        }
        let p = rng.vec3() * 8.0 - Vec3::splat(2.0);
        let (cp, u, v) = closest_point_on_triangle(p, a, b, c);
        let w = 1.0 - u - v;
        let eps = 1e-5;
        assert!((-eps..=1.0 + eps).contains(&u), "u out of range: {u}");
        assert!((-eps..=1.0 + eps).contains(&v), "v out of range: {v}");
        assert!((-eps..=1.0 + eps).contains(&w), "w out of range: {w}");
        let rebuilt = a * u + b * v + c * w;
        assert!(
            (rebuilt - cp).length() < 1e-4,
            "barycentric reconstruction diverged: {rebuilt:?} vs {cp:?}"
        );
    }
}

#[test]
fn on_surface_point_has_zero_distance() {
    let mut rng = Lcg::new(7);
    for _ in 0..100 {
        let (a, b, c) = (rng.vec3() * 4.0, rng.vec3() * 4.0, rng.vec3() * 4.0);
        if triangle_is_sliver(a, b, c) {
            continue;
        }
        // Random convex combination = a point ON the triangle.
        let (r1, r2) = (rng.next_f32(), rng.next_f32());
        let (u, v) = if r1 + r2 > 1.0 {
            (1.0 - r1, 1.0 - r2)
        } else {
            (r1, r2)
        };
        let p = a * u + b * v + c * (1.0 - u - v);
        let (cp, _, _) = closest_point_on_triangle(p, a, b, c);
        assert!(
            (cp - p).length() < 1e-4,
            "on-surface point should be its own closest point"
        );
    }
}

/// The returned distance must lower-bound the distance to ANY on-triangle
/// point — catches wrong-region bugs without a second closest-point oracle.
#[test]
fn closest_point_is_an_infimum() {
    let mut rng = Lcg::new(11);
    for _ in 0..50 {
        let (a, b, c) = (rng.vec3() * 4.0, rng.vec3() * 4.0, rng.vec3() * 4.0);
        if triangle_is_sliver(a, b, c) {
            continue;
        }
        let p = rng.vec3() * 10.0 - Vec3::splat(3.0);
        let (cp, _, _) = closest_point_on_triangle(p, a, b, c);
        let d = (cp - p).length();
        for _ in 0..100 {
            let (r1, r2) = (rng.next_f32(), rng.next_f32());
            let (u, v) = if r1 + r2 > 1.0 {
                (1.0 - r1, 1.0 - r2)
            } else {
                (r1, r2)
            };
            let sample = a * u + b * v + c * (1.0 - u - v);
            assert!(
                d <= (sample - p).length() + 1e-4,
                "found an on-triangle point closer than the 'closest' point"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// distance_to_aabb_sq — inside / face / edge / corner
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn aabb_distance_cases() {
    let (lo, hi) = (Vec3::ZERO, Vec3::ONE);
    assert_eq!(distance_to_aabb_sq(Vec3::splat(0.5), lo, hi), 0.0, "inside");
    assert_eq!(
        distance_to_aabb_sq(Vec3::new(0.5, 0.5, 2.0), lo, hi),
        1.0,
        "face"
    );
    assert_eq!(
        distance_to_aabb_sq(Vec3::new(2.0, 2.0, 0.5), lo, hi),
        2.0,
        "edge"
    );
    assert_eq!(
        distance_to_aabb_sq(Vec3::new(2.0, 2.0, 2.0), lo, hi),
        3.0,
        "corner"
    );
    // On the boundary counts as inside (distance 0) — closed box.
    assert_eq!(
        distance_to_aabb_sq(Vec3::new(1.0, 0.5, 0.5), lo, hi),
        0.0,
        "boundary"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Small pieces
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn triangle_aabb_bounds_all_vertices() {
    let (lo, hi) = triangle_aabb(A, B, C);
    assert_eq!(lo, Vec3::ZERO);
    assert_eq!(hi, Vec3::new(1.0, 1.0, 0.0));
}

#[test]
fn sliver_predicate() {
    assert!(
        !triangle_is_sliver(A, B, C),
        "healthy triangle flagged as sliver"
    );
    // Collinear — zero area with distinct vertices.
    assert!(triangle_is_sliver(
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0)
    ));
    // Zero-length edge.
    assert!(triangle_is_sliver(A, A, C));
}

#[test]
fn eval_position_matches_barycentrics() {
    let points = [A, B, C];
    let indices = [0i32, 1, 2];
    let p = mesh_eval_position(&points, &indices, 0, 0.25, 0.5);
    assert!((p - (A * 0.25 + B * 0.5 + C * 0.25)).length() < 1e-6);
}

#[test]
fn query_on_empty_tree_is_a_miss() {
    let q = mesh_query_point(&[], &[], &[], Vec3::ZERO, 1.0e10);
    assert_eq!(q.face, -1);
    assert_eq!(q.dist, f32::INFINITY);
    assert_eq!(q, MeshQuery::miss());
}
