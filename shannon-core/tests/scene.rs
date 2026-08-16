//! Host tests for Day-2 tasks 2.1–2.2 — the SDF library and the W1 scene.

use shannon_core::{Quat, Vec3, scene, sdf};

const TOL: f32 = 1e-5;

// ── sdf.rs ──────────────────────────────────────────────────────────────────

#[test]
fn sphere_outside_and_inside() {
    assert!((sdf::sphere(Vec3::new(2.0, 0.0, 0.0), 1.0) - 1.0).abs() <= TOL);
    assert!((sdf::sphere(Vec3::ZERO, 1.0) + 1.0).abs() <= TOL); // inside is negative
}

#[test]
fn box_corner_is_outside() {
    assert!(sdf::box_(Vec3::splat(1.0), Vec3::splat(2.0)) > 0.0);
}

#[test]
fn box_center_is_inside() {
    assert!(sdf::box_(Vec3::splat(1.0), Vec3::ZERO) < 0.0);
}

#[test]
fn csg_operators() {
    assert_eq!(sdf::op_union(3.0, 5.0), 3.0);
    assert_eq!(sdf::op_intersect(3.0, 5.0), 5.0);
    assert_eq!(sdf::op_subtract(3.0, 5.0), 5.0); // max(-3, 5)
}

// ── scene.rs ────────────────────────────────────────────────────────────────

#[test]
fn ground_plane_is_zero_level() {
    assert!(scene::scene(Vec3::new(0.0, -1.0, 0.0)).abs() <= TOL);
}

#[test]
fn ground_normal_points_up() {
    // Sample away from the box so the plane dominates the neighbourhood.
    let n = scene::normal(Vec3::new(4.0, -1.0, 4.0));
    assert!(n.y > 0.9, "ground normal was {n:?}");
}

#[test]
fn normals_are_unit_length() {
    for p in [
        Vec3::new(4.0, -1.0, 4.0),
        Vec3::new(0.0, 0.5, 0.5), // box surface-ish
        Vec3::new(0.0, 1.5, 0.0), // above the carved sphere
    ] {
        let n = scene::normal(p);
        assert!((n.length() - 1.0).abs() <= 1e-3, "at {p:?}: {n:?}");
    }
}

#[test]
fn shadow_straight_up_is_unshadowed() {
    let s = scene::shadow(Vec3::new(0.0, 10.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(s, 1.0);
}

#[test]
fn draw_at_output_is_gamma_clamped() {
    // Every pixel — hit or sky — must land in [0, 1] per channel.
    let cam_rot = Quat::from_rpy(-0.5, -0.5, 0.0);
    let cam_pos = Vec3::new(-1.25, 1.0, 2.0);
    const W: u32 = 32;
    const H: u32 = 16;
    for i in 0..(W * H) as usize {
        let c = scene::draw_at(i, cam_pos, cam_rot, W, H);
        for v in [c.x, c.y, c.z] {
            assert!((0.0..=1.0).contains(&v), "pixel {i}: {c:?}");
        }
    }
}

#[test]
fn draw_hits_both_geometry_and_sky() {
    // Sanity at the 128×64 parity resolution. NOTE: 32×16 is TOO COARSE for
    // this check — its topmost row is sy = 0.875, where every ray still tilts
    // slightly downward and converges onto the ground within 128 steps, so no
    // pixel takes the sky branch at all. (Diagnosed on Day 2: the reference
    // image's grey top band is DISTANT SHADED GROUND — our far-ground colour
    // matches it to 1 LSB — with true sky only above sy ≈ 0.93.)
    let cam_rot = Quat::from_rpy(-0.5, -0.5, 0.0);
    let cam_pos = Vec3::new(-1.25, 1.0, 2.0);
    let sky = Vec3::new(0.4, 0.45, 0.5) * 1.5;
    let (mut hits, mut skies) = (0, 0);
    for i in 0..(128 * 64) as usize {
        let c = scene::draw_at(i, cam_pos, cam_rot, 128, 64);
        if (c - sky).length() <= 1e-6 {
            skies += 1
        } else {
            hits += 1
        }
    }
    assert!(hits > 0, "no geometry visible — camera convention wrong?");
    assert!(skies > 0, "no sky visible — camera convention wrong?");
}
