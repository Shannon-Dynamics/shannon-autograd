//! BVH structure + traversal invariants — host-only, no GPU (Day-4 plan §5.3).
//!
//! Everything here is deterministic: a fixed-seed LCG replaces `rand` (no new
//! dependency for a test), and the traversal/brute-force comparison is on
//! sorted hit LISTS — the week plan's "identical hit sets, not just counts".

use shannon_core::bvh::{query_aabb, query_ray};
use shannon_core::{BvhNode, Vec3};
use shannon_spatial::{brute_force_aabb, brute_force_ray, build_median_split};

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

/// Random boxes: lowers in [0, pos)³, extents in [0, ext)³ per axis.
fn random_bounds(rng: &mut Lcg, n: usize, pos: f32, ext: f32) -> Vec<(Vec3, Vec3)> {
    (0..n)
        .map(|_| {
            let lo = rng.vec3() * pos;
            (lo, lo + rng.vec3() * ext)
        })
        .collect()
}

fn sorted<I: Iterator<Item = i32>>(it: I) -> Vec<i32> {
    let mut v: Vec<i32> = it.collect();
    v.sort_unstable();
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// Structure invariants
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn every_prim_in_exactly_one_leaf() {
    let aabbs = random_bounds(&mut Lcg::new(1), 1000, 100.0, 10.0);
    let nodes = build_median_split(&aabbs);
    let leaves = sorted(nodes.iter().filter(|n| n.right < 0).map(|n| n.left));
    let expected: Vec<i32> = (0..1000).collect();
    assert_eq!(
        leaves, expected,
        "leaf primitive indices must be a permutation of 0..n"
    );
}

#[test]
fn child_aabbs_contained_in_parent() {
    let aabbs = random_bounds(&mut Lcg::new(2), 500, 100.0, 10.0);
    let nodes = build_median_split(&aabbs);
    // Union bounds are cw_min/cw_max — EXACT float ops — so containment must
    // hold with no epsilon at all.
    fn contains(outer: &BvhNode, inner: &BvhNode) -> bool {
        outer.lower.x <= inner.lower.x
            && outer.lower.y <= inner.lower.y
            && outer.lower.z <= inner.lower.z
            && outer.upper.x >= inner.upper.x
            && outer.upper.y >= inner.upper.y
            && outer.upper.z >= inner.upper.z
    }
    for n in &nodes {
        if n.right >= 0 {
            assert!(
                contains(n, &nodes[n.left as usize]),
                "left child escapes parent bounds"
            );
            assert!(
                contains(n, &nodes[n.right as usize]),
                "right child escapes parent bounds"
            );
        }
    }
}

#[test]
fn node_count_is_2n_minus_1() {
    let mut rng = Lcg::new(3);
    for n in [1usize, 2, 3, 7, 64, 1000] {
        let aabbs = random_bounds(&mut rng, n, 100.0, 10.0);
        let nodes = build_median_split(&aabbs);
        assert_eq!(nodes.len(), 2 * n - 1, "full binary tree over {n} leaves");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Traversal vs brute force — sorted hit lists
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn traversal_matches_brute_force() {
    let mut rng = Lcg::new(42);
    let aabbs = random_bounds(&mut rng, 1000, 100.0, 10.0);
    let nodes = build_median_split(&aabbs);

    for q in 0..64 {
        let lo = rng.vec3() * 80.0;
        let hi = lo + rng.vec3() * 20.0;
        let tree = sorted(query_aabb(&nodes, lo, hi));
        let brute = sorted(brute_force_aabb(&aabbs, lo, hi).into_iter());
        assert_eq!(tree, brute, "AABB query {q} diverged from brute force");
    }

    for q in 0..64 {
        let origin = rng.vec3() * 50.0;
        let dir = (rng.vec3() - Vec3::splat(0.5)).normalize();
        let tree = sorted(query_ray(&nodes, origin, dir, 1.0e30));
        let brute = sorted(brute_force_ray(&aabbs, origin, dir, 1.0e30).into_iter());
        assert_eq!(tree, brute, "ray query {q} diverged from brute force");
    }
}

#[test]
fn ray_on_axis_origin_is_robust() {
    // One box, rays with a ZERO direction component — the 0 × ∞ = NaN slab
    // path (Day-4 plan §4.4). The robustness claims worth pinning:
    //   1. parallel-inside-the-slab must HIT (the axis must not constrain);
    //   2. parallel-outside must MISS;
    //   3. the exact-graze knife edge resolves DETERMINISTICALLY and
    //      IDENTICALLY between traversal and brute force (its boolean value is
    //      a boundary convention, not a correctness claim).
    let aabbs = vec![(Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0))];
    let nodes = build_median_split(&aabbs);
    let x = Vec3::new(1.0, 0.0, 0.0);

    let inside = Vec3::new(0.0, 2.0, 2.0);
    assert_eq!(
        sorted(query_ray(&nodes, inside, x, 1.0e30)),
        vec![0],
        "parallel-inside must hit"
    );
    assert_eq!(brute_force_ray(&aabbs, inside, x, 1.0e30), vec![0]);

    let outside = Vec3::new(0.0, 5.0, 2.0);
    assert!(
        sorted(query_ray(&nodes, outside, x, 1.0e30)).is_empty(),
        "parallel-outside must miss"
    );
    assert!(brute_force_ray(&aabbs, outside, x, 1.0e30).is_empty());

    for graze in [
        Vec3::new(0.0, 1.0, 2.0), // origin.y exactly on lower.y
        Vec3::new(0.0, 3.0, 2.0), // origin.y exactly on upper.y
        Vec3::new(0.0, 1.0, 1.0), // grazing two boundaries at once
    ] {
        let tree = sorted(query_ray(&nodes, graze, x, 1.0e30));
        let brute = sorted(brute_force_ray(&aabbs, graze, x, 1.0e30).into_iter());
        assert_eq!(
            tree, brute,
            "graze at {graze:?} must agree between tree and brute force"
        );
    }
}

#[test]
fn single_prim_and_empty_tree() {
    let aabbs = vec![(Vec3::ZERO, Vec3::ONE)];
    let nodes = build_median_split(&aabbs);
    assert_eq!(nodes.len(), 1);
    assert_eq!(
        sorted(query_aabb(&nodes, Vec3::splat(0.5), Vec3::splat(0.6))),
        vec![0]
    );
    assert!(sorted(query_aabb(&nodes, Vec3::splat(5.0), Vec3::splat(6.0))).is_empty());

    let empty: Vec<(Vec3, Vec3)> = Vec::new();
    let nodes = build_median_split(&empty);
    assert!(nodes.is_empty());
    assert!(query_aabb(&nodes, Vec3::ZERO, Vec3::ONE).next().is_none());
    assert!(
        query_ray(&nodes, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 1.0e30)
            .next()
            .is_none()
    );
}
