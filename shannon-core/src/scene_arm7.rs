//! The ARM-7 pick-and-place scene: an industrial 4-DOF robot arm (yaw /
//! shoulder / elbow / wrist + two-finger gripper) restoring the fallen H of a
//! flat-tiled "S H A N N O N" sign, on a dark studio floor with a glowing
//! grid. Ported from the `docs/robot-arm-shannon.html` WebGL shader.
//!
//! PURE function of its per-frame parameters — the keyframe timeline, forward
//! kinematics, and the H's flight all run on the HOST and feed joint
//! positions in. This file evaluates geometry and shading only, identically
//! on both backends.

// Kernel-shaped code passes flat per-frame parameters by design — a GPU ABI
// has no notion of a convenience struct (and cuda-oxide's scalar marshalling
// is per-argument). Silencing arity lints for the whole scene module.
#![allow(clippy::too_many_arguments)]

use crate::{Vec3, math, sdf};

// ── Arm proportions (shared with the host FK — single source of truth) ──────

pub const BASE_H: f32 = 0.16;
pub const BASE_R: f32 = 0.42;
pub const ARM_L1: f32 = 0.95;
pub const ARM_L2: f32 = 0.78;
pub const ARM_L3: f32 = 0.22;
pub const ARM_L4: f32 = 0.16;
/// Shoulder pivot sits on top of the base pedestal.
pub const SHOULDER: Vec3 = Vec3::new(0.0, BASE_H * 2.0, 0.0);
/// Finger spread at gripper value `g` ∈ 0..1.
#[inline(always)]
pub fn finger_spread(g: f32) -> f32 {
    0.02 + 0.09 * g
}

// ── Sign tiles ──────────────────────────────────────────────────────────────

const CX: f32 = 0.13; // letter half-width
const CZ: f32 = 0.20; // letter half-height (flat on the table, so along z)
const BT: f32 = 0.028; // bar half-thickness
const TY: f32 = 0.025; // tile half-depth (y)

pub const TABLE_TOP: f32 = 0.50;
pub const TILE_Y: f32 = 0.025;
pub const SPACING: f32 = 0.34;
pub const TABLE_Z: f32 = 1.30;
/// Letter tiles rest ON the tabletop.
pub const LETTER_Y: f32 = TABLE_TOP + TILE_Y;
/// The H's slot: index 1 of S _ A N N O N (offsets −3,−2,−1,0,1,2,3 ·SPACING).
pub const H_SLOT_X: f32 = -2.0 * SPACING;

/// Camera orbit target (matches the HTML default view).
pub const CAM_TARGET: Vec3 = Vec3::new(0.0, 0.6, 0.75);

// ── Block-segment letters (local frame: x across, z up the glyph, y depth) ──

#[inline(always)]
fn bar_h(p: Vec3, zc: f32) -> f32 {
    sdf::box_(Vec3::new(CX, TY, BT), p - Vec3::new(0.0, 0.0, zc))
}
#[inline(always)]
fn bar_m(p: Vec3) -> f32 {
    sdf::box_(Vec3::new(CX, TY, BT), p)
}
#[inline(always)]
fn bar_full_v(p: Vec3, xc: f32) -> f32 {
    sdf::box_(Vec3::new(BT, TY, CZ), p - Vec3::new(xc, 0.0, 0.0))
}
#[inline(always)]
fn bar_half_v(p: Vec3, xc: f32, zc: f32) -> f32 {
    sdf::box_(Vec3::new(BT, TY, CZ * 0.5), p - Vec3::new(xc, 0.0, zc))
}
#[inline(always)]
fn bar_diag(p: Vec3) -> f32 {
    sdf::capsule(
        p,
        Vec3::new(-CX + BT, 0.0, -CZ + BT),
        Vec3::new(CX - BT, 0.0, CZ - BT),
        BT * 1.1,
    )
}

#[inline(always)]
fn letter_s(p: Vec3) -> f32 {
    let mut d = bar_h(p, CZ - BT);
    d = math::min(d, bar_half_v(p, CX - BT, CZ * 0.5));
    d = math::min(d, bar_m(p));
    d = math::min(d, bar_half_v(p, -CX + BT, -CZ * 0.5));
    math::min(d, bar_h(p, -CZ + BT))
}
#[inline(always)]
fn letter_h(p: Vec3) -> f32 {
    let mut d = bar_full_v(p, -CX + BT);
    d = math::min(d, bar_full_v(p, CX - BT));
    math::min(d, bar_m(p))
}
#[inline(always)]
fn letter_a(p: Vec3) -> f32 {
    let mut d = bar_h(p, CZ - BT);
    d = math::min(d, bar_full_v(p, -CX + BT));
    d = math::min(d, bar_full_v(p, CX - BT));
    math::min(d, bar_m(p))
}
#[inline(always)]
fn letter_n(p: Vec3) -> f32 {
    let mut d = bar_full_v(p, -CX + BT);
    d = math::min(d, bar_full_v(p, CX - BT));
    math::min(d, bar_diag(p))
}
#[inline(always)]
fn letter_o(p: Vec3) -> f32 {
    let mut d = bar_h(p, CZ - BT);
    d = math::min(d, bar_h(p, -CZ + BT));
    d = math::min(d, bar_full_v(p, -CX + BT));
    math::min(d, bar_full_v(p, CX - BT))
}

// ── Per-frame parameters, bundled for the internal call tree ────────────────

/// Everything the host feeds per frame. Joint POSITIONS, not angles — the
/// forward kinematics runs once on the host, not once per pixel.
#[derive(Clone, Copy)]
pub struct Arm7Frame {
    pub elbow: Vec3,
    pub wrist: Vec3,
    pub hand_end: Vec3,
    pub f1: Vec3,
    pub f2: Vec3,
    pub h_pos: Vec3,
    pub h_yaw: f32,
    pub carrying: f32,
}

// ── Scene assembly ──────────────────────────────────────────────────────────

/// Full scene SDF + material id (GLSL `map`). Materials:
/// 0 ground · 1 base · 2 links · 3 joints · 4 fingers · 5 letters ·
/// 6 carried H · 7 table.
#[inline(always)]
fn map(p: Vec3, f: &Arm7Frame) -> (f32, f32) {
    // Ground.
    let mut d = p.y;
    let mut m = 0.0f32;

    // Base pedestal + hub.
    let base = sdf::capped_cylinder(p - Vec3::new(0.0, BASE_H, 0.0), BASE_H, BASE_R);
    let hub = sdf::capped_cylinder(p - Vec3::new(0.0, BASE_H * 2.0, 0.0), 0.06, BASE_R * 0.55);
    let base = math::min(base, hub);
    if base < d {
        d = base;
        m = 1.0;
    }

    // Links: three smooth-blended capsules.
    let mut links = sdf::capsule(p, SHOULDER, f.elbow, 0.14);
    links = sdf::op_smooth_union(links, sdf::capsule(p, f.elbow, f.wrist, 0.115), 0.05);
    links = sdf::op_smooth_union(links, sdf::capsule(p, f.wrist, f.hand_end, 0.085), 0.05);
    if links < d {
        d = links;
        m = 2.0;
    }

    // Joints: three smooth-blended spheres.
    let mut joints = sdf::sphere(p - SHOULDER, 0.185);
    joints = sdf::op_smooth_union(joints, sdf::sphere(p - f.elbow, 0.155), 0.05);
    joints = sdf::op_smooth_union(joints, sdf::sphere(p - f.wrist, 0.125), 0.04);
    if joints < d {
        d = joints;
        m = 3.0;
    }

    // Gripper fingers.
    let fingers = math::min(
        sdf::capsule(p, f.hand_end, f.f1, 0.045),
        sdf::capsule(p, f.hand_end, f.f2, 0.045),
    );
    if fingers < d {
        d = fingers;
        m = 4.0;
    }

    // Table: top + four legs.
    let table_top = sdf::box_(
        Vec3::new(1.25, 0.05, 0.32),
        p - Vec3::new(0.0, TABLE_TOP - 0.05, TABLE_Z),
    );
    let leg_y = (TABLE_TOP - 0.10) * 0.5;
    let leg_h = Vec3::new(0.03, leg_y, 0.03);
    let mut legs = sdf::box_(leg_h, p - Vec3::new(-1.15, leg_y, TABLE_Z - 0.24));
    legs = math::min(legs, sdf::box_(leg_h, p - Vec3::new(1.15, leg_y, TABLE_Z - 0.24)));
    legs = math::min(legs, sdf::box_(leg_h, p - Vec3::new(-1.15, leg_y, TABLE_Z + 0.24)));
    legs = math::min(legs, sdf::box_(leg_h, p - Vec3::new(1.15, leg_y, TABLE_Z + 0.24)));
    let table = math::min(table_top, legs);
    if table < d {
        d = table;
        m = 7.0;
    }

    // Static letters: S _ A N N O N (slot −2 / the H is dynamic). Each glyph
    // is evaluated in a 180°-yawed local frame (negate x and z) so the sign
    // reads upright from the front camera.
    let ly = LETTER_Y;
    let lp = |n: f32| {
        let l = p - Vec3::new(n * SPACING, ly, TABLE_Z);
        Vec3::new(-l.x, l.y, -l.z)
    };
    let mut letters = letter_s(lp(-3.0));
    letters = math::min(letters, letter_a(lp(-1.0)));
    letters = math::min(letters, letter_n(lp(0.0)));
    letters = math::min(letters, letter_n(lp(1.0)));
    letters = math::min(letters, letter_o(lp(2.0)));
    letters = math::min(letters, letter_n(lp(3.0)));
    if letters < d {
        d = letters;
        m = 5.0;
    }

    // The dynamic H, yawed about Y (same 180° reading-frame flip).
    let rel = p - f.h_pos;
    let (s, c) = (math::sin(-f.h_yaw), math::cos(-f.h_yaw));
    let pl = Vec3::new(c * rel.x + s * rel.z, rel.y, -s * rel.x + c * rel.z);
    let pl = Vec3::new(-pl.x, pl.y, -pl.z);
    let h = letter_h(pl);
    if h < d {
        d = h;
        m = if f.carrying > 0.5 { 6.0 } else { 5.0 };
    }

    (d, m)
}

#[inline(always)]
fn normal(p: Vec3, f: &Arm7Frame) -> Vec3 {
    const E: f32 = 0.0005;
    let e1 = Vec3::new(1.0, -1.0, -1.0);
    let e2 = Vec3::new(-1.0, -1.0, 1.0);
    let e3 = Vec3::new(-1.0, 1.0, -1.0);
    let e4 = Vec3::new(1.0, 1.0, 1.0);
    (e1 * map(p + e1 * E, f).0
        + e2 * map(p + e2 * E, f).0
        + e3 * map(p + e3 * E, f).0
        + e4 * map(p + e4 * E, f).0)
        .normalize()
}

#[inline(always)]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = math::clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline(always)]
fn soft_shadow(ro: Vec3, rd: Vec3, f: &Arm7Frame) -> f32 {
    let mut res = 1.0f32;
    let mut t = 0.02f32;
    let mut i = 0;
    while i < 28 {
        if t >= 6.0 {
            break;
        }
        let h = map(ro + rd * t, f).0;
        if h < 0.001 {
            return 0.0;
        }
        res = math::min(res, 12.0 * h / t);
        t += math::clamp(h, 0.02, 0.2);
        i += 1;
    }
    math::clamp(res, 0.0, 1.0)
}

#[inline(always)]
fn ambient_occlusion(p: Vec3, n: Vec3, f: &Arm7Frame) -> f32 {
    let mut occ = 0.0f32;
    let mut sca = 1.0f32;
    let mut i = 0;
    while i < 4 {
        let h = 0.01 + 0.13 * i as f32 / 3.0;
        let d = map(p + n * h, f).0;
        occ += (h - d) * sca;
        sca *= 0.72;
        i += 1;
    }
    math::clamp(1.0 - 1.8 * occ, 0.0, 1.0)
}

#[inline(always)]
fn sky_color(rd: Vec3) -> Vec3 {
    let top = Vec3::new(0.03, 0.035, 0.045);
    let hor = Vec3::new(0.055, 0.065, 0.075);
    let u = math::clamp(rd.y * 0.7 + 0.3, 0.0, 1.0);
    hor + (top - hor) * u
}

#[inline(always)]
fn fract(x: f32) -> f32 {
    x - math::floor(x)
}

#[inline(always)]
fn shade(p: Vec3, n: Vec3, rd: Vec3, mat: f32, f: &Arm7Frame) -> Vec3 {
    let base = if mat < 0.5 {
        // Studio floor: radial-faded glowing grid.
        let gx = math::abs(fract(p.x - 0.5) - 0.5);
        let gz = math::abs(fract(p.z - 0.5) - 0.5);
        let line = smoothstep(0.0, 0.02, math::min(gx, gz));
        let dark = Vec3::new(0.035, 0.04, 0.045);
        let grid = Vec3::new(0.10, 0.22, 0.28);
        let rad = math::clamp(1.0 - math::sqrt(p.x * p.x + p.z * p.z) / 9.0, 0.0, 1.0);
        dark + (dark + (grid - dark) * (1.0 - line) - dark) * (rad * 0.9)
    } else if mat < 1.5 {
        Vec3::new(0.22, 0.24, 0.26) // base pedestal
    } else if mat < 2.5 {
        Vec3::new(0.92, 0.40, 0.20) // links — safety orange
    } else if mat < 3.5 {
        Vec3::new(0.10, 0.14, 0.16) // joints — gloss black
    } else if mat < 4.5 {
        Vec3::new(0.72, 0.75, 0.77) // fingers
    } else if mat < 5.5 {
        Vec3::new(0.95, 0.76, 0.28) // letters — brushed gold
    } else if mat < 6.5 {
        Vec3::new(0.98, 0.82, 0.35) // the carried H — brighter
    } else {
        Vec3::new(0.10, 0.09, 0.085) // table — dark wood
    };

    let lig = Vec3::new(0.55, 0.85, 0.35).normalize();
    let dif = math::clamp(n.dot(lig), 0.0, 1.0);
    let sh = soft_shadow(p + n * 0.02, lig, f);
    let ao = ambient_occlusion(p, n, f);
    let hal = (lig - rd).normalize();
    let spe_pow = if mat > 2.5 && mat < 3.5 { 60.0 } else { 28.0 };
    let spe = math::pow(math::clamp(n.dot(hal), 0.0, 1.0), spe_pow);
    let fre = math::pow(math::clamp(1.0 + n.dot(rd), 0.0, 1.0), 4.0);

    let amb = sky_color(n) * 1.4;
    let mut col = Vec3::new(
        base.x * (amb.x * ao + dif * sh * 1.4 * 1.0),
        base.y * (amb.y * ao + dif * sh * 1.4 * 0.97),
        base.z * (amb.z * ao + dif * sh * 1.4 * 0.9),
    );
    col += Vec3::splat(spe * sh * 0.6);

    if mat > 2.5 && mat < 3.5 {
        col += Vec3::new(0.35, 0.75, 0.95) * (fre * 0.9); // cyan rim on joints
    } else if mat > 5.5 && mat < 6.5 {
        col += Vec3::new(0.4, 0.85, 1.0) * (fre * 1.1); // carried-H highlight
    } else {
        col += Vec3::splat(fre * 0.06);
    }
    col
}

/// Shade one pixel of the ARM-7 scene (early-exit march, orbit camera).
#[inline(always)]
pub fn draw_arm7_at(
    i: usize,
    elbow: Vec3,
    wrist: Vec3,
    hand_end: Vec3,
    f1: Vec3,
    f2: Vec3,
    h_pos: Vec3,
    h_yaw: f32,
    carrying: f32,
    cam_az: f32,
    cam_el: f32,
    cam_dist: f32,
    width: u32,
    height: u32,
) -> Vec3 {
    let f = Arm7Frame { elbow, wrist, hand_end, f1, f2, h_pos, h_yaw, carrying };

    let x = (i as u32) % width;
    let y = (i as u32) / width;
    let sx = (2.0 * x as f32 - width as f32) / height as f32 * 0.5;
    let sy = (2.0 * y as f32 - height as f32) / height as f32 * 0.5;

    let ro = CAM_TARGET
        + Vec3::new(
            math::cos(cam_el) * math::sin(cam_az),
            math::sin(cam_el),
            math::cos(cam_el) * math::cos(cam_az),
        ) * cam_dist;
    let ww = (CAM_TARGET - ro).normalize();
    let uu = ww.cross(Vec3::new(0.0, 1.0, 0.0)).normalize();
    let vv = uu.cross(ww);
    let rd = (uu * sx + vv * sy + ww * 1.7).normalize();

    // March (GLSL-faithful: eps 0.0008, tmax 40, 100 steps).
    let mut t = 0.0f32;
    let mut mat = -1.0f32;
    let mut p = ro;
    let mut steps = 0;
    while steps < 100 {
        p = ro + rd * t;
        let (d, m) = map(p, &f);
        if d < 0.0008 || t > 40.0 {
            mat = m;
            break;
        }
        t += d;
        steps += 1;
    }

    let mut col = if t <= 40.0 && mat >= 0.0 {
        let n = normal(p, &f);
        let c = shade(p, n, rd, mat, &f);
        let fog = math::clamp(1.0 - math::exp(-0.012 * t * t * 0.35), 0.0, 1.0);
        c + (sky_color(rd) - c) * fog
    } else {
        sky_color(rd)
    };

    // Gamma, then the HTML's vignette.
    col = Vec3::new(
        math::pow(math::max(col.x, 0.0), 0.4545),
        math::pow(math::max(col.y, 0.0), 0.4545),
        math::pow(math::max(col.z, 0.0), 0.4545),
    );
    let qx = x as f32 / width as f32;
    let qy = y as f32 / height as f32;
    let vig = 0.75 + 0.25 * math::pow(16.0 * qx * qy * (1.0 - qx) * (1.0 - qy), 0.25);
    col = col * vig;

    Vec3::new(
        math::clamp(col.x, 0.0, 1.0),
        math::clamp(col.y, 0.0, 1.0),
        math::clamp(col.z, 0.0, 1.0),
    )
}
