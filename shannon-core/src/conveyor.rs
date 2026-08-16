//! Conveyor-sort simulation bodies: a grooved belt surface that conveys
//! particles along +X while its lane grooves channel them across Z, sorting
//! them into discrete lanes before they drop off the end.
//!
//! Original force model (no upstream derivation): position-based pushout onto
//! the margin shell, a belt-grip term that drives the tangential velocity
//! toward the belt's conveying velocity (exponential approach — kinetic grip,
//! not Coulomb friction), and a plain damped floor backstop. Sorting is pure
//! geometry: groove walls give every pushout normal a Z-component pointing at
//! the nearest lane center, so riding the belt IS being funneled.
//!
//! No-tunnelling holds by the same CFL-by-construction argument as the mesh
//! sim: the per-step displacement is clamped to margin/2, and the surface
//! moves ≤ `ripple_amp·ripple_k·belt_speed·dt` per frame (two orders below
//! margin/2 at the demo constants), so the closing displacement can never
//! cross the margin in one frame. The belt is a heightfield, so unsigned
//! queries are sound: a particle that never crosses is never "inside".

use crate::bvh::BvhNode;
use crate::math;
use crate::mesh::{Particle, mesh_eval_position, mesh_query_point};
use crate::vec::Vec3;

/// How hard the belt grips a contacting particle: tangential velocity
/// approaches the belt velocity as `1 − exp(−GRIP·dt)` per second.
pub const BELT_GRIP: f32 = 6.0;
/// Tangential damping on the (static) floor backstop, per second.
pub const FLOOR_DAMP: f32 = 4.0;
/// Linear air drag, per second.
pub const AIR_DRAG: f32 = 0.15;

/// Belt surface height offset at `(x, z)` and time `t` — the closed form the
/// deform kernel evaluates and the tunnelling predicate re-evaluates on the
/// host. Grooves are valleys at `z = n·lane_w` (lane centers); the ripple
/// travels along +X at `belt_speed`, which is what makes the belt visibly run.
#[allow(clippy::too_many_arguments)] // arity mirrors the kernel row
#[inline(always)]
pub fn belt_surface_y(
    x: f32,
    z: f32,
    t: f32,
    groove_amp: f32,
    lane_w: f32,
    ripple_amp: f32,
    ripple_k: f32,
    belt_speed: f32,
) -> f32 {
    let tau = 2.0 * core::f32::consts::PI;
    let groove = groove_amp * 0.5 * (1.0 - math::cos(tau * z / lane_w));
    let ripple = ripple_amp * math::sin(ripple_k * (x - belt_speed * t));
    groove + ripple
}

/// Belt deform — pure function of REST positions + time (no accumulation, no
/// drift; same discipline as `mesh::deform_at`).
#[allow(clippy::too_many_arguments)] // arity mirrors the kernel row
#[inline(always)]
pub fn belt_deform_at(
    i: usize,
    rest: &[Vec3],
    t: f32,
    groove_amp: f32,
    lane_w: f32,
    ripple_amp: f32,
    ripple_k: f32,
    belt_speed: f32,
) -> Vec3 {
    let p = rest[i];
    let y = belt_surface_y(
        p.x, p.z, t, groove_amp, lane_w, ripple_amp, ripple_k, belt_speed,
    );
    Vec3::new(p.x, y, p.z)
}

/// Conveyor particle step — reads `parts[i]` + the belt mesh, returns the
/// updated particle. The host double-buffers and swaps.
#[allow(clippy::too_many_arguments)] // arity mirrors the kernel row
#[inline(always)]
pub fn conveyor_step_at(
    i: usize,
    parts: &[Particle],
    nodes: &[BvhNode],
    points: &[Vec3],
    indices: &[i32],
    margin: f32,
    dt: f32,
    max_dist: f32,
    y_floor: f32,
    belt_speed: f32,
) -> Particle {
    let x = parts[i].pos;
    let mut v = parts[i].vel;

    // Gravity + air drag.
    v = v + Vec3::new(0.0, -9.8, 0.0) * dt - v * (AIR_DRAG * dt);

    // CFL clamp — the no-tunnelling guarantee: max step = 0.5·margin.
    let v_max = 0.5 * margin / dt;
    let speed = v.length();
    if speed > v_max {
        v = v * (v_max / speed);
    }

    let mut xpred = x + v * dt;
    let mut contact_n = Vec3::ZERO;
    let mut on_belt = false;
    let mut on_floor = false;

    let q = mesh_query_point(nodes, points, indices, xpred, max_dist);
    if q.face >= 0 {
        let cp = mesh_eval_position(points, indices, q.face, q.u, q.v);
        let delta = xpred - cp;
        if delta.length() < margin {
            // Unsigned pushout: park the particle on the margin shell.
            // normalize() is EPS-guarded — delta ≈ 0 degenerates to
            // xpred = cp, corrected next frame.
            contact_n = delta.normalize();
            xpred = cp + contact_n * margin;
            on_belt = true;
        }
    }

    // Floor backstop for everything past the belt's end (and any stray).
    if xpred.y < y_floor + margin {
        xpred.y = y_floor + margin;
        contact_n = Vec3::new(0.0, 1.0, 0.0);
        on_floor = true;
        on_belt = false;
    }

    // Velocity from positions, then the contact response on the tangential
    // component: belt contact drives it toward the belt's conveying velocity
    // (the grip term — this is what a conveyor IS); floor contact just damps.
    let mut v_out = (xpred - x) / dt;
    if on_belt || on_floor {
        let v_n = contact_n * v_out.dot(contact_n);
        let v_t = v_out - v_n;
        let v_t = if on_belt {
            let belt_v = Vec3::new(belt_speed, 0.0, 0.0);
            let belt_t = belt_v - contact_n * belt_v.dot(contact_n);
            let a = 1.0 - math::exp(-BELT_GRIP * dt);
            v_t + (belt_t - v_t) * a
        } else {
            let a = 1.0 - math::exp(-FLOOR_DAMP * dt);
            v_t * (1.0 - a)
        };
        v_out = v_n + v_t;
    }

    Particle {
        pos: xpred,
        vel: v_out,
    }
}
