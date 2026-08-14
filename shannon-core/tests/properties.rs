//! Property tests for shannon-core — the minimum set from Day-1 plan §7.
//!
//! Host-side: reproducibility is guaranteed, so `1e-5` absolute tolerance.
//! (GPU-accumulated gradients would need relative tolerance — pitfall #7.)

use shannon_core::{Mat33, Quat, Vec3};

const TOL: f32 = 1e-5;

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() <= TOL
}
fn approx_v(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

/// Deterministic pseudo-random test points — no rand dependency needed.
fn test_vectors() -> Vec<Vec3> {
    let mut out = Vec::new();
    let mut s = 0x9e3779b9u32;
    for _ in 0..32 {
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1 << 24) as f32 * 20.0 - 10.0
        };
        out.push(Vec3::new(next(), next(), next()));
    }
    out
}

// ── vec.rs ──────────────────────────────────────────────────────────────────

#[test]
fn normalize_gives_unit_length() {
    for v in test_vectors() {
        if v.length() > shannon_core::EPS {
            assert!(approx(v.normalize().length(), 1.0), "v = {v:?}");
        }
    }
}

#[test]
fn normalize_zero_is_zero_not_nan() {
    // The guard — must not be NaN.
    assert_eq!(Vec3::ZERO.normalize(), Vec3::ZERO);
}

#[test]
fn dot_is_commutative() {
    let vs = test_vectors();
    for pair in vs.chunks(2) {
        if let [a, b] = pair {
            assert!(approx(a.dot(*b), b.dot(*a)));
        }
    }
}

#[test]
fn cross_is_anticommutative_and_perpendicular() {
    let vs = test_vectors();
    for pair in vs.chunks(2) {
        if let [a, b] = pair {
            assert!(approx_v(a.cross(*b), -(b.cross(*a))));
            // dot(cross(a,b), a) ≈ 0 — scale tolerance by magnitude.
            let c = a.cross(*b);
            let scale = (a.length() * b.length() * a.length()).max(1.0);
            assert!(c.dot(*a).abs() / scale <= TOL, "a={a:?} b={b:?}");
        }
    }
}

#[test]
fn add_sub_roundtrip() {
    let vs = test_vectors();
    for pair in vs.chunks(2) {
        if let [a, b] = pair {
            assert!(approx_v((*a + *b) - *b, *a));
        }
    }
}

#[test]
fn length_sq_matches_length_squared() {
    for v in test_vectors() {
        let l = v.length();
        // relative comparison — values span orders of magnitude
        assert!((v.length_sq() - l * l).abs() <= TOL * (1.0 + l * l));
    }
}

// ── quat.rs ─────────────────────────────────────────────────────────────────

fn test_quats() -> Vec<Quat> {
    let axes = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, -2.0, 0.5),
        Vec3::new(-0.3, 0.7, 2.0),
    ];
    let angles = [0.0, 0.5, -1.2, 3.0];
    axes.iter()
        .zip(angles.iter())
        .map(|(a, t)| Quat::from_axis_angle(*a, *t))
        .collect()
}

#[test]
fn identity_rotation_is_noop() {
    for v in test_vectors() {
        assert!(approx_v(Quat::IDENTITY.rotate(v), v));
    }
}

#[test]
fn rotation_preserves_length() {
    for q in test_quats() {
        for v in test_vectors() {
            assert!(
                (q.rotate(v).length() - v.length()).abs() <= TOL * (1.0 + v.length()),
                "q={q:?} v={v:?}"
            );
        }
    }
}

#[test]
fn quat_times_inverse_is_identity() {
    for q in test_quats() {
        let r = q.mul(q.inverse());
        assert!(approx(r.x, 0.0) && approx(r.y, 0.0) && approx(r.z, 0.0) && approx(r.w, 1.0));
    }
}

#[test]
fn rotate_inv_undoes_rotate() {
    for q in test_quats() {
        for v in test_vectors() {
            let back = q.rotate_inv(q.rotate(v));
            let scale = 1.0 + v.length();
            assert!((back - v).length() <= TOL * scale, "q={q:?} v={v:?}");
        }
    }
}

#[test]
fn from_rpy_zero_is_identity() {
    let q = Quat::from_rpy(0.0, 0.0, 0.0);
    assert!(approx(q.x, 0.0) && approx(q.y, 0.0) && approx(q.z, 0.0) && approx(q.w, 1.0));
}

// ── mat.rs ──────────────────────────────────────────────────────────────────

#[test]
fn identity_mul_vec_is_noop() {
    for v in test_vectors() {
        assert!(approx_v(Mat33::IDENTITY.mul_vec(v), v));
    }
}

#[test]
fn transpose_is_involutive() {
    let m = Mat33::from_rows(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 10.0),
    );
    assert_eq!(m.transpose().transpose(), m);
}

#[test]
fn mul_by_identity_is_noop() {
    let m = Mat33::from_rows(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 10.0),
    );
    assert_eq!(m.mul_mat(Mat33::IDENTITY), m);
}

#[test]
fn inverse_times_matrix_is_identity() {
    let m = Mat33::from_rows(
        Vec3::new(2.0, 0.0, 1.0),
        Vec3::new(-1.0, 3.0, 0.5),
        Vec3::new(0.0, 1.0, 4.0),
    );
    let prod = m.mul_mat(m.inverse());
    for r in 0..3 {
        for c in 0..3 {
            let expect = if r == c { 1.0 } else { 0.0 };
            assert!(approx(prod.m[r][c], expect), "prod = {prod:?}");
        }
    }
}
