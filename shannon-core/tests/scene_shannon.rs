//! Host tests for the animated-demo geometry (Day-2 addendum tasks A1–A2).

use shannon_core::scene_shannon as sc;
use shannon_core::{Quat, Vec3, sdf};

const TOL: f32 = 1e-5;

// ── sdf.rs additions ────────────────────────────────────────────────────────

#[test]
fn capsule_distances() {
    let a = Vec3::new(0.0, 0.0, 0.0);
    let b = Vec3::new(2.0, 0.0, 0.0);
    // Beside the middle of the segment.
    assert!((sdf::capsule(Vec3::new(1.0, 1.0, 0.0), a, b, 0.25) - 0.75).abs() <= TOL);
    // Beyond an endpoint.
    assert!((sdf::capsule(Vec3::new(3.0, 0.0, 0.0), a, b, 0.25) - 0.75).abs() <= TOL);
    // Inside.
    assert!(sdf::capsule(Vec3::new(1.0, 0.1, 0.0), a, b, 0.25) < 0.0);
    // Degenerate (a == b) falls back to a sphere.
    assert!((sdf::capsule(Vec3::new(0.0, 1.0, 0.0), a, a, 0.25) - 0.75).abs() <= TOL);
}

#[test]
fn box2_and_extrude() {
    // Outside along +x.
    assert!((sdf::box2(2.0, 0.0, 1.0, 1.0) - 1.0).abs() <= TOL);
    // Inside is negative.
    assert!(sdf::box2(0.0, 0.0, 1.0, 1.0) < 0.0);
    // Extrusion: inside the 2D shape, past the depth → distance is z overshoot.
    assert!((sdf::extrude(-0.5, 1.0, 0.5,) - 0.5).abs() <= TOL);
    // Inside both → negative.
    assert!(sdf::extrude(-0.5, 0.0, 0.5) < 0.0);
}

// ── scene_shannon.rs ────────────────────────────────────────────────────────

fn rest_args() -> (Vec3, Vec3, Vec3, f32, Vec3, Quat) {
    // A plausible rest pose: arm folded right, H flat on the table.
    let s1 = Vec3::new(0.5, 2.0, -0.5);
    let s2 = Vec3::new(0.9, 1.35, 0.1);
    let gd = (s2 - s1).normalize();
    let h_pos = Vec3::new(-0.5, 0.985, 0.30);
    let h_rot = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), -core::f32::consts::FRAC_PI_2);
    (s1, s2, gd, 0.0, h_pos, h_rot)
}

#[test]
fn scene_is_finite_over_a_grid() {
    let (s1, s2, gd, g, hp, hr) = rest_args();
    for ix in -8..=8 {
        for iy in 0..=8 {
            for iz in -8..=8 {
                let p = Vec3::new(ix as f32 * 0.4, iy as f32 * 0.4, iz as f32 * 0.4);
                let d = sc::scene(p, s1, s2, gd, g, hp, hr);
                assert!(d.is_finite(), "non-finite SDF at {p:?}");
            }
        }
    }
}

#[test]
fn ground_is_zero_level() {
    let (s1, s2, gd, g, hp, hr) = rest_args();
    let d = sc::scene(Vec3::new(3.5, 0.0, 2.5), s1, s2, gd, g, hp, hr);
    assert!(d.abs() <= 1e-4, "ground SDF = {d}");
}

#[test]
fn letter_interior_is_negative() {
    let (s1, s2, gd, g, hp, hr) = rest_args();
    // Centre of the S's middle bar (slot 0).
    let p = Vec3::new(sc::SLOT_X[0], sc::LETTER_Y, sc::LETTER_Z);
    let d = sc::scene(p, s1, s2, gd, g, hp, hr);
    assert!(d < 0.0, "inside the S should be negative, got {d}");
}

#[test]
fn h_slot_is_empty_but_dynamic_h_renders() {
    let (s1, s2, gd, g, _hp, hr) = rest_args();
    // With the H parked at its slot, the slot centre is inside geometry…
    let slot_c = Vec3::new(sc::SLOT_X[sc::H_SLOT], sc::LETTER_Y, sc::LETTER_Z);
    let with_h = sc::scene(
        slot_c + Vec3::new(-0.09, 0.0, 0.0),
        s1,
        s2,
        gd,
        g,
        slot_c,
        Quat::IDENTITY,
    );
    assert!(
        with_h < 0.0,
        "H at slot: left stroke interior should be negative, got {with_h}"
    );
    // …and with the H far away, the same point is empty space.
    let far = Vec3::new(50.0, 50.0, 50.0);
    let without = sc::scene(slot_c + Vec3::new(-0.09, 0.0, 0.0), s1, s2, gd, g, far, hr);
    assert!(
        without > 0.05,
        "empty slot should be open space, got {without}"
    );
}

#[test]
fn arm_link_interior_is_negative() {
    let (s1, s2, gd, g, hp, hr) = rest_args();
    let mid = (sc::ARM_SHOULDER + s1) * 0.5;
    let d = sc::scene(mid, s1, s2, gd, g, hp, hr);
    assert!(
        d < 0.0,
        "inside the upper-arm capsule should be negative, got {d}"
    );
}

#[test]
fn draw_rt_output_is_clamped() {
    let (s1, s2, gd, g, hp, hr) = rest_args();
    let cam = Quat::from_rpy(sc::CAM_TILT, 0.0, 0.0);
    for i in (0..64 * 36).step_by(7) {
        let c = sc::draw_rt_at(i, s1, s2, gd, g, hp, hr, cam, 64, 36);
        for v in [c.x, c.y, c.z] {
            assert!((0.0..=1.0).contains(&v), "pixel {i}: {c:?}");
        }
    }
}
