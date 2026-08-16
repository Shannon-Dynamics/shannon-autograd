//! The animated demo scene: a robot arm fixing the "S H A N N O N" sign.
//!
//! PURE function of its per-frame parameters — all animation (state machine,
//! IK, the H letter's fall) runs on the HOST and feeds transforms in. This
//! file evaluates geometry only, on both backends.
//!
//! ⚠️ Glyph data is stored as FLAT `[f32; N]` arrays (5 floats per segment:
//! cx, cy, hx, hy, rot). cuda-oxide does not materialize struct-element array
//! constants on device; primitive-element arrays are supported.

// Kernel-shaped code passes flat per-frame parameters by design — a GPU ABI
// has no notion of a convenience struct (and cuda-oxide's scalar marshalling
// is per-argument). Silencing arity lints for the whole scene module.
#![allow(clippy::too_many_arguments)]

use crate::{Quat, Vec3, Vec4, math, sdf};

// ── Fixed staging (camera + light + arm base are scene constants) ───────────

pub const CAM_POS: Vec3 = Vec3::new(0.0, 1.55, 3.9);
/// Slight downward tilt; host builds the quat via `Quat::from_rpy(CAM_TILT, 0.0, 0.0)`.
pub const CAM_TILT: f32 = -0.10;

const LIGHT: Vec3 = Vec3::new(0.5, 0.65, 0.45); // normalized in draw

/// Arm shoulder pivot — floor column behind-right of the table.
pub const ARM_BASE: Vec3 = Vec3::new(0.0, 0.0, -0.9);
pub const ARM_SHOULDER: Vec3 = Vec3::new(0.0, 1.45, -0.9);
pub const ARM_L1: f32 = 0.88; // upper-arm length
pub const ARM_L2: f32 = 0.82; // forearm length

// Table.
const TABLE_C: Vec3 = Vec3::new(0.0, 0.90, 0.0);
const TABLE_H: Vec3 = Vec3::new(1.9, 0.05, 0.55);
pub const TABLE_TOP_Y: f32 = 0.95;

// Letter row.
pub const LETTER_HH: f32 = 0.14; // half-height
pub const LETTER_HZ: f32 = 0.035; // half-depth
pub const SLOT_X: [f32; 7] = [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2];
pub const H_SLOT: usize = 1;
pub const LETTER_Z: f32 = -0.10;
/// Standing letter centre height.
pub const LETTER_Y: f32 = TABLE_TOP_Y + LETTER_HH;

// ── Glyphs: 5 floats per segment (cx, cy, hx, hy, rot) ──────────────────────

#[rustfmt::skip]
const GLYPH_S: [f32; 25] = [
     0.0,   0.11, 0.12, 0.03, 0.0,
     0.0,   0.0,  0.12, 0.03, 0.0,
     0.0,  -0.11, 0.12, 0.03, 0.0,
    -0.09,  0.055, 0.03, 0.055, 0.0,
     0.09, -0.055, 0.03, 0.055, 0.0,
];
#[rustfmt::skip]
const GLYPH_H: [f32; 15] = [
    -0.09, 0.0, 0.03, 0.14, 0.0,
     0.09, 0.0, 0.03, 0.14, 0.0,
     0.0,  0.0, 0.06, 0.03, 0.0,
];
#[rustfmt::skip]
const GLYPH_A: [f32; 20] = [
    -0.09,  0.0,  0.03, 0.14, 0.0,
     0.09,  0.0,  0.03, 0.14, 0.0,
     0.0,   0.11, 0.12, 0.03, 0.0,
     0.0,  -0.01, 0.06, 0.03, 0.0,
];
#[rustfmt::skip]
const GLYPH_N: [f32; 15] = [
    -0.09, 0.0, 0.03, 0.14, 0.0,
     0.09, 0.0, 0.03, 0.14, 0.0,
     0.0,  0.0, 0.155, 0.03, -1.00, // diagonal top-left → bottom-right
];
#[rustfmt::skip]
const GLYPH_O: [f32; 20] = [
     0.0,   0.11, 0.12, 0.03, 0.0,
     0.0,  -0.11, 0.12, 0.03, 0.0,
    -0.09,  0.0,  0.03, 0.14, 0.0,
     0.09,  0.0,  0.03, 0.14, 0.0,
];

/// SDF of one glyph in letter-local space (origin at letter centre).
#[inline(always)]
fn glyph_sdf(local: Vec3, segs: &[f32]) -> f32 {
    let mut d = f32::MAX;
    let mut k = 0;
    while k + 5 <= segs.len() {
        let (cx, cy, hx, hy, rot) = (segs[k], segs[k + 1], segs[k + 2], segs[k + 3], segs[k + 4]);
        let mut qx = local.x - cx;
        let mut qy = local.y - cy;
        if rot != 0.0 {
            let (s, c) = (math::sin(-rot), math::cos(-rot));
            let rx = c * qx - s * qy;
            qy = s * qx + c * qy;
            qx = rx;
        }
        let d2 = sdf::box2(qx, qy, hx, hy);
        d = math::min(d, sdf::extrude(d2, local.z, LETTER_HZ));
        k += 5;
    }
    d
}

#[inline(always)]
fn glyph_for_slot(slot: usize, local: Vec3) -> f32 {
    // S H A N N O N
    match slot {
        0 => glyph_sdf(local, &GLYPH_S),
        2 => glyph_sdf(local, &GLYPH_A),
        5 => glyph_sdf(local, &GLYPH_O),
        _ => glyph_sdf(local, &GLYPH_N), // 3, 4, 6 (slot 1 = H is dynamic)
    }
}

// ── Scene assembly ──────────────────────────────────────────────────────────

/// Static letters (all slots except the dynamic H).
#[inline(always)]
fn letters_sdf(p: Vec3) -> f32 {
    let mut d = f32::MAX;
    let mut slot = 0;
    while slot < 7 {
        if slot != H_SLOT {
            let sx = SLOT_X[slot];
            // Cheap prune: skip glyphs whose x-slab cannot be nearest.
            if math::abs(p.x - sx) < 0.45 {
                let local = Vec3::new(p.x - sx, p.y - LETTER_Y, p.z - LETTER_Z);
                d = math::min(d, glyph_for_slot(slot, local));
            }
        }
        slot += 1;
    }
    // Distance lower bound when everything was pruned: horizontal slab distance.
    if d == f32::MAX {
        // Nearest possible letter surface is at least this far in x alone.
        let mut best = f32::MAX;
        let mut s2 = 0;
        while s2 < 7 {
            if s2 != H_SLOT {
                best = math::min(best, math::abs(p.x - SLOT_X[s2]) - 0.13);
            }
            s2 += 1;
        }
        d = math::max(best, 0.05);
    }
    d
}

#[inline(always)]
fn table_sdf(p: Vec3) -> f32 {
    let mut d = sdf::box_(TABLE_H, p - TABLE_C);
    let leg_h = Vec3::new(0.06, 0.425, 0.06);
    let mut i = 0;
    while i < 4 {
        let lx = if i % 2 == 0 { -1.75 } else { 1.75 };
        let lz = if i < 2 { -0.42 } else { 0.42 };
        d = math::min(d, sdf::box_(leg_h, p - Vec3::new(lx, 0.425, lz)));
        i += 1;
    }
    d
}

#[inline(always)]
fn arm_sdf(p: Vec3, s1: Vec3, s2: Vec3, grip_dir: Vec3, grip: f32) -> f32 {
    // Pedestal column + two links.
    let mut d = sdf::capsule(p, ARM_BASE, ARM_SHOULDER, 0.09);
    d = math::min(d, sdf::capsule(p, ARM_SHOULDER, s1, 0.062));
    d = math::min(d, sdf::capsule(p, s1, s2, 0.052));
    // Gripper: two fingers straddling grip_dir; `grip`=1 closed, 0 open.
    // Lateral axis degenerates when grip_dir ∥ up (top-down grabs) — fall back to x.
    let up = Vec3::new(0.0, 1.0, 0.0);
    let raw_lat = grip_dir.cross(up);
    let lat = if raw_lat.length_sq() > 1.0e-6 {
        raw_lat.normalize()
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let w = 0.028 + (1.0 - grip) * 0.038;
    let fa = s2 + grip_dir * 0.02;
    let fb = s2 + grip_dir * 0.17;
    d = math::min(d, sdf::capsule(p, fa + lat * w, fb + lat * w, 0.022));
    d = math::min(d, sdf::capsule(p, fa - lat * w, fb - lat * w, 0.022));
    d
}

#[inline(always)]
fn h_letter_sdf(p: Vec3, h_pos: Vec3, h_rot: Quat) -> f32 {
    // Bounding prune before the full glyph.
    let rel = p - h_pos;
    if rel.length_sq() > 0.40 * 0.40 {
        return rel.length() - 0.28;
    }
    glyph_sdf(h_rot.rotate_inv(rel), &GLYPH_H)
}

/// Full scene SDF.
#[inline(always)]
pub fn scene(
    p: Vec3,
    s1: Vec3,
    s2: Vec3,
    grip_dir: Vec3,
    grip: f32,
    h_pos: Vec3,
    h_rot: Quat,
) -> f32 {
    let ground = sdf::plane(p, Vec4::new(0.0, 1.0, 0.0, 0.0));
    let mut d = ground;
    d = math::min(d, table_sdf(p));
    d = math::min(d, letters_sdf(p));
    d = math::min(d, h_letter_sdf(p, h_pos, h_rot));
    d = math::min(d, arm_sdf(p, s1, s2, grip_dir, grip));
    d
}

/// Material id at a hit point (argmin re-evaluation — hit points only).
#[inline(always)]
fn material(
    p: Vec3,
    s1: Vec3,
    s2: Vec3,
    grip_dir: Vec3,
    grip: f32,
    h_pos: Vec3,
    h_rot: Quat,
) -> u32 {
    let dg = sdf::plane(p, Vec4::new(0.0, 1.0, 0.0, 0.0));
    let dt = table_sdf(p);
    let dl = letters_sdf(p);
    let dh = h_letter_sdf(p, h_pos, h_rot);
    let da = arm_sdf(p, s1, s2, grip_dir, grip);
    let mut id = 0u32;
    let mut best = dg;
    if dt < best {
        best = dt;
        id = 1;
    }
    if dl < best {
        best = dl;
        id = 2;
    }
    if dh < best {
        best = dh;
        id = 3;
    }
    if da < best {
        id = 4;
    }
    id
}

#[inline(always)]
fn albedo(id: u32) -> Vec3 {
    match id {
        0 => Vec3::new(0.50, 0.52, 0.57), // ground
        1 => Vec3::new(0.58, 0.38, 0.22), // table
        2 => Vec3::new(0.92, 0.92, 0.95), // letters
        3 => Vec3::new(1.00, 0.62, 0.12), // the H — amber
        _ => Vec3::new(0.26, 0.30, 0.38), // arm
    }
}

#[inline(always)]
fn normal(p: Vec3, s1: Vec3, s2: Vec3, gd: Vec3, g: f32, hp: Vec3, hr: Quat) -> Vec3 {
    // Tetrahedral gradient — 4 scene evals instead of 6 (standard RT trick).
    const E: f32 = 1.0e-4;
    let e1 = Vec3::new(1.0, -1.0, -1.0);
    let e2 = Vec3::new(-1.0, -1.0, 1.0);
    let e3 = Vec3::new(-1.0, 1.0, -1.0);
    let e4 = Vec3::new(1.0, 1.0, 1.0);
    (e1 * scene(p + e1 * E, s1, s2, gd, g, hp, hr)
        + e2 * scene(p + e2 * E, s1, s2, gd, g, hp, hr)
        + e3 * scene(p + e3 * E, s1, s2, gd, g, hp, hr)
        + e4 * scene(p + e4 * E, s1, s2, gd, g, hp, hr))
    .normalize()
}

/// Soft shadow — same formulation as the proven W1 scene (Warp's reference).
#[inline(always)]
fn shadow(ro: Vec3, rd: Vec3, s1: Vec3, s2: Vec3, gd: Vec3, g: f32, hp: Vec3, hr: Quat) -> f32 {
    let mut t = 0.0f32;
    let mut s = 1.0f32;
    let mut i = 0;
    while i < 40 {
        let d = scene(ro + rd * t, s1, s2, gd, g, hp, hr);
        t += math::clamp(d, 0.0001, 2.0);
        let h = math::clamp(4.0 * d / t, 0.0, 1.0);
        s = math::min(s, h * h * (3.0 - 2.0 * h));
        if t > 6.5 {
            return 1.0;
        }
        i += 1;
    }
    s
}

/// Shade one pixel of the animated scene. Early-exit march (real-time variant —
/// the parity-exact W1 `scene::draw_at` is untouched).
#[inline(always)]
pub fn draw_rt_at(
    i: usize,
    s1: Vec3,
    s2: Vec3,
    grip_dir: Vec3,
    grip: f32,
    h_pos: Vec3,
    h_rot: Quat,
    cam_rot: Quat,
    width: u32,
    height: u32,
) -> Vec3 {
    let x = (i as u32) % width;
    let y = (i as u32) / width;
    let sx = (2.0 * x as f32 - width as f32) / height as f32;
    let sy = (2.0 * y as f32 - height as f32) / height as f32;

    let ro = CAM_POS;
    let rd = cam_rot.rotate(Vec3::new(sx, sy, -2.0).normalize());

    // Early-exit sphere trace.
    let mut t = 0.0f32;
    let mut hit = false;
    let mut steps = 0;
    while steps < 72 {
        let d = scene(ro + rd * t, s1, s2, grip_dir, grip, h_pos, h_rot);
        if d < 1.0e-3 {
            hit = true;
            break;
        }
        t += d;
        if t > 12.0 {
            break;
        }
        steps += 1;
    }

    if hit {
        let p = ro + rd * t;
        let n = normal(p, s1, s2, grip_dir, grip, h_pos, h_rot);
        let l = LIGHT.normalize();
        let h = (l - rd).normalize();

        let diffuse = math::max(n.dot(l), 0.0);
        let ambient = 0.18;
        let specular = math::pow(math::clamp(n.dot(h), 0.0, 1.0), 60.0);
        let sh = shadow(p + n * 0.02, l, s1, s2, grip_dir, grip, h_pos, h_rot);

        let id = material(p, s1, s2, grip_dir, grip, h_pos, h_rot);
        let base = albedo(id);
        let c = base * (ambient + diffuse * sh) + Vec3::splat(specular * sh * 0.6);

        // Same encoding gamma as W1 for a consistent look.
        Vec3::new(
            math::clamp(math::pow(c.x, 1.6), 0.0, 1.0),
            math::clamp(math::pow(c.y, 1.6), 0.0, 1.0),
            math::clamp(math::pow(c.z, 1.6), 0.0, 1.0),
        )
    } else {
        // Simple vertical sky gradient.
        let g = math::clamp(0.5 + 0.5 * rd.y, 0.0, 1.0);
        Vec3::new(0.55 + 0.15 * g, 0.65 + 0.12 * g, 0.80 + 0.10 * g)
    }
}
