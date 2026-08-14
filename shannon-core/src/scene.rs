//! The W1 reference scene — a box with a sphere carved out of it, on a ground
//! plane. Shared verbatim by the GPU kernel and the CPU adapter.
//!
//! Ported line-for-line from Warp's `example_raymarch.py`; the porting traps
//! (loop-variable scoping, no early exit in the march, gamma direction) are
//! called out inline. Day-2 plan §5.2.

use crate::{Quat, Vec3, Vec4, math, sdf};

/// The scene SDF.
#[inline(always)]
pub fn scene(p: Vec3) -> f32 {
    let sphere_1 = Vec3::new(0.0, 0.75, 0.0);
    let d = sdf::op_subtract(
        sdf::sphere(p - sphere_1, 0.75),
        sdf::box_(Vec3::new(1.0, 0.5, 0.5), p),
    );
    sdf::op_union(d, sdf::plane(p, Vec4::new(0.0, 1.0, 0.0, 1.0)))
}

/// Surface normal by central differences on the SDF.
#[inline(always)]
pub fn normal(p: Vec3) -> Vec3 {
    const EPS: f32 = 1.0e-5;
    let dx = scene(p + Vec3::new(EPS, 0.0, 0.0)) - scene(p - Vec3::new(EPS, 0.0, 0.0));
    let dy = scene(p + Vec3::new(0.0, EPS, 0.0)) - scene(p - Vec3::new(0.0, EPS, 0.0));
    let dz = scene(p + Vec3::new(0.0, 0.0, EPS)) - scene(p - Vec3::new(0.0, 0.0, EPS));
    Vec3::new(dx, dy, dz).normalize()
}

/// Soft shadow by sphere tracing toward the light.
#[inline(always)]
pub fn shadow(ro: Vec3, rd: Vec3) -> f32 {
    let mut t = 0.0f32;
    let mut s = 1.0f32;
    for _ in 0..64 {
        let d = scene(ro + rd * t);
        t += math::clamp(d, 0.0001, 2.0);
        let h = math::clamp(4.0 * d / t, 0.0, 1.0);
        s = math::min(s, h * h * (3.0 - 2.0 * h));
        if t > 8.0 {
            return 1.0; // ← early exit IS present here (unlike the march loop)
        }
    }
    s
}

/// Shade one pixel. Returns the gamma-encoded colour.
///
/// `i` is the flattened pixel index; `width`/`height` are the framebuffer dims.
#[inline(always)]
pub fn draw_at(i: usize, cam_pos: Vec3, cam_rot: Quat, width: u32, height: u32) -> Vec3 {
    let x = (i as u32) % width;
    let y = (i as u32) / width;

    // Screen coords. BOTH divide by `height` — that is what makes pixels square
    // and bakes in the aspect ratio. Do not "fix" sx to divide by width.
    let sx = (2.0 * x as f32 - width as f32) / height as f32;
    let sy = (2.0 * y as f32 - height as f32) / height as f32;

    let ro = cam_pos; // camera pos is NOT rotated
    let rd = cam_rot.rotate(Vec3::new(sx, sy, -2.0).normalize()); // normalize BEFORE rotating

    // ── The march ───────────────────────────────────────────────────────────
    // `d` MUST be hoisted: Python leaks the loop variable, Rust does not.
    // There is deliberately NO early exit — 128 iterations always — and `p`
    // below uses the POST-increment `t`. Replicate exactly for pixel parity.
    let mut t = 0.0f32;
    let mut d = 0.0f32;
    for _ in 0..128 {
        d = scene(ro + rd * t);
        t += d;
    }

    if d < 0.01 {
        let p = ro + rd * t;
        let n = normal(p);
        let l = Vec3::new(0.6, 0.4, 0.5).normalize();
        let h = (l - rd).normalize(); // half-vector

        let diffuse = n.dot(l);
        let specular = math::pow(math::clamp(n.dot(h), 0.0, 1.0), 80.0);
        let fresnel = 0.04 + 0.96 * math::pow(math::clamp(1.0 - h.dot(l), 0.0, 1.0), 5.0);

        let intensity = 2.0;
        let result = Vec3::new(0.85, 0.9, 0.95)
            * (diffuse * (1.0 - fresnel) + specular * fresnel * 10.0)
            * shadow(p, l)
            * intensity;

        // Gamma. NOTE: pow(c, 2.2) is the ENCODING direction (darkening), not the
        // usual 1/2.2. Matches the reference exactly — keep it.
        Vec3::new(
            math::clamp(math::pow(result.x, 2.2), 0.0, 1.0),
            math::clamp(math::pow(result.y, 2.2), 0.0, 1.0),
            math::clamp(math::pow(result.z, 2.2), 0.0, 1.0),
        )
    } else {
        Vec3::new(0.4, 0.45, 0.5) * 1.5 // sky
    }
}
