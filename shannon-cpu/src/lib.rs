//! shannon-cpu — W0/W1's CPU adapters. 🧪 First-party example.
//!
//! Elementwise map adapters are DECLARED via `shannon_cpu_kernels!` — the same
//! rows as the GPU side (Day-2 task 2.7). Adjoints stay hand-written: scatter
//! via GradSink does not fit the map shape. Note: no function here contains a
//! line of arithmetic — the math lives in `shannon-core`, called from both
//! backends. That is the invariant, visible.

// The kernel-row macro is a token-muncher; the row count now exceeds the
// default recursion limit of 128 expansions.
#![recursion_limit = "512"]

use rayon::prelude::*;
use shannon_core::{BvhNode, MeshQuery, Particle, Quat, Vec3};
use shannon_rt::{HostGradF32, HostGradVec3};

shannon_core::shannon_cpu_kernels! {
    /// FORWARD: y[i] = a[i] * scale + bias — same body the GPU kernel calls.
    affine(a: &[f32], scale: f32, bias: f32) -> f32 = shannon_core::adjoint::affine_at;

    /// W1 pixel shader — same shared body the GPU kernel calls, under rayon.
    draw(cam_pos: Vec3, cam_rot: Quat, width: u32, height: u32) -> Vec3
        = shannon_core::scene::draw_at;

    /// Animated SHANNON-sign scene — same shared body, under rayon.
    draw_shannon(s1: Vec3, s2: Vec3, grip_dir: Vec3, grip: f32,
                 h_pos: Vec3, h_rot: Quat, cam_rot: Quat,
                 width: u32, height: u32) -> Vec3
        = shannon_core::scene_shannon::draw_rt_at;

    /// W2 closest-point query — same shared traversal, under rayon.
    mesh_query(nodes: &[BvhNode], points: &[Vec3], indices: &[i32],
               queries: &[Vec3], max_dist: f32)
        -> MeshQuery = shannon_core::mesh::mesh_query_point_at;

    /// W2 deform — same shared body, under rayon.
    mesh_deform(rest: &[Vec3], phase: f32, amp: f32) -> Vec3
        = shannon_core::mesh::deform_at;

    /// W2 particle step — same shared body, under rayon.
    sim_particles(parts: &[Particle], nodes: &[BvhNode], points: &[Vec3], indices: &[i32],
                  margin: f32, dt: f32, max_dist: f32, y_floor: f32)
        -> Particle = shannon_core::mesh::sim_particle_at;

    /// ARM-7 pick-and-place scene — same shared body, under rayon.
    draw_arm7(elbow: Vec3, wrist: Vec3, hand_end: Vec3, f1: Vec3, f2: Vec3,
              h_pos: Vec3, h_yaw: f32, carrying: f32,
              cam_az: f32, cam_el: f32, cam_dist: f32,
              width: u32, height: u32) -> Vec3
        = shannon_core::scene_arm7::draw_arm7_at;

    /// Conveyor belt deform — same shared body, under rayon.
    belt_deform(rest: &[Vec3], t: f32, groove_amp: f32, lane_w: f32,
                ripple_amp: f32, ripple_k: f32, belt_speed: f32)
        -> Vec3 = shannon_core::conveyor::belt_deform_at;

    /// Conveyor particle step — same shared body, under rayon.
    conveyor_step(parts: &[Particle], nodes: &[BvhNode], points: &[Vec3], indices: &[i32],
                  margin: f32, dt: f32, max_dist: f32, y_floor: f32, belt_speed: f32)
        -> Particle = shannon_core::conveyor::conveyor_step_at;

    /// W3 chain check: y = sin(a) — same shared body, under rayon.
    sin_map(a: &[f32]) -> f32 = shannon_core::loss::sin_at;
}

/// ADJOINT: ā[i] += ȳ[i] * scale — same adjoint body, host sink.
pub fn adj_affine(adj_y: &[f32], scale: f32, adj_a: &mut [f32]) {
    let sink = HostGradF32::new(adj_a);
    (0..adj_y.len())
        .into_par_iter()
        .for_each(|i| shannon_core::adjoint::adj_affine_at(i, adj_y, scale, &sink));
}

// ── W3 loss + adjoint counterparts (Day-6 plan §5.6) ────────────────────────
// Same shapes as the GPU @raw kernels: scatter through Host sinks, zero
// arithmetic in this file — every body is a shannon_core::loss call.

/// ADJOINT of sin_map: ā[i] += ȳ[i]·cos(a[i]).
pub fn adj_sin_map(a: &[f32], adj_y: &[f32], adj_a: &mut [f32]) {
    let sink = HostGradF32::new(adj_a);
    (0..a.len())
        .into_par_iter()
        .for_each(|i| shannon_core::adjoint::adj_sin_at(i, a, adj_y, &sink));
}

/// FORWARD reduction: loss[0] += Σ x[i] — scatter to slot 0, like the GPU.
pub fn sum_scalar(x: &[f32], loss: &mut [f32]) {
    let sink = HostGradF32::new(loss);
    (0..x.len()).into_par_iter().for_each(|i| {
        use shannon_core::GradSink;
        sink.accumulate(0, x[i]);
    });
}

/// ADJOINT of sum_scalar: broadcast — x̄[i] += loss̄[0].
pub fn adj_sum_scalar(adj_loss: &[f32], adj_x: &mut [f32]) {
    let n = adj_x.len();
    let sink = HostGradF32::new(adj_x);
    (0..n).into_par_iter().for_each(|i| shannon_core::loss::adj_sum_at(i, adj_loss, &sink));
}

/// FORWARD reduction: loss[0] += Σ ½‖xᵢ − tᵢ‖².
pub fn l2_loss(x: &[Vec3], t: &[Vec3], loss: &mut [f32]) {
    let sink = HostGradF32::new(loss);
    (0..x.len()).into_par_iter().for_each(|i| {
        use shannon_core::GradSink;
        sink.accumulate(0, shannon_core::loss::l2_term_at(i, x, t));
    });
}

/// ADJOINT of l2_loss: x̄ᵢ += (xᵢ − tᵢ)·loss̄.
pub fn adj_l2_loss(x: &[Vec3], t: &[Vec3], adj_loss: &[f32], adj_x: &mut [Vec3]) {
    let sink = HostGradVec3::new(adj_x);
    (0..x.len())
        .into_par_iter()
        .for_each(|i| shannon_core::loss::adj_l2_at(i, x, t, adj_loss, &sink));
}

/// FORWARD A→B Chamfer: loss[0] += Σ ½‖srcᵢ − eval(corrᵢ on target)‖².
pub fn chamfer_ab(src: &[Vec3], corr: &[MeshQuery], tp: &[Vec3], ti: &[i32], loss: &mut [f32]) {
    let sink = HostGradF32::new(loss);
    (0..src.len()).into_par_iter().for_each(|i| {
        use shannon_core::GradSink;
        sink.accumulate(0, shannon_core::loss::chamfer_ab_term_at(i, src, corr, tp, ti));
    });
}

/// ADJOINT of chamfer_ab: envelope — src̄ᵢ += (srcᵢ − p)·loss̄.
pub fn adj_chamfer_ab(
    src: &[Vec3], corr: &[MeshQuery], tp: &[Vec3], ti: &[i32],
    adj_loss: &[f32], adj_src: &mut [Vec3],
) {
    let sink = HostGradVec3::new(adj_src);
    (0..src.len())
        .into_par_iter()
        .for_each(|i| shannon_core::loss::adj_chamfer_ab_at(i, src, corr, tp, ti, adj_loss, &sink));
}

/// FORWARD B→A Chamfer: loss[0] += Σ ½‖eval(corrᵢ on source) − tvᵢ‖².
pub fn chamfer_ba(tv: &[Vec3], corr: &[MeshQuery], sp: &[Vec3], si: &[i32], loss: &mut [f32]) {
    let sink = HostGradF32::new(loss);
    (0..tv.len()).into_par_iter().for_each(|i| {
        use shannon_core::GradSink;
        sink.accumulate(0, shannon_core::loss::chamfer_ba_term_at(i, tv, corr, sp, si));
    });
}

/// ADJOINT of chamfer_ba: the canonical gather→scatter into the source
/// points' gradients through the barycentric weights.
pub fn adj_chamfer_ba(
    tv: &[Vec3], corr: &[MeshQuery], sp: &[Vec3], si: &[i32],
    adj_loss: &[f32], adj_sp: &mut [Vec3],
) {
    let sink = HostGradVec3::new(adj_sp);
    (0..tv.len())
        .into_par_iter()
        .for_each(|i| shannon_core::loss::adj_chamfer_ba_at(i, tv, corr, sp, si, adj_loss, &sink));
}

// ── W4 CPU counterparts (Day-3 plan §5.4) ───────────────────────────────────
// #[inline(never)] is LOAD-BEARING: without it the optimizer deletes empty
// calls and the CPU column measures an empty loop. Callers must additionally
// wrap args + call in std::hint::black_box (Day-3 plan §4.C). GPU launches
// need neither — they bottom out in FFI, which cannot be elided.

use shannon_kernels::bench::{BenchS0, BenchSa, BenchSf, BenchSm, BenchSv, BenchSz};
use shannon_core::Mat33;

#[inline(never)]
pub fn bench_k0() {}
#[inline(never)]
pub fn bench_kf(_x: f32, _y: f32, _z: f32) {}
#[inline(never)]
pub fn bench_kv(_u: Vec3, _v: Vec3, _w: Vec3) {}
#[inline(never)]
pub fn bench_km(_m: Mat33, _n: Mat33, _o: Mat33) {}
#[inline(never)]
pub fn bench_ka(_a: &[f32], _b: &[f32], _c: &[f32]) {}
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn bench_kz(
    _a: &[f32], _b: &[f32], _c: &[f32],
    _x: f32, _y: f32, _z: f32,
    _u: Vec3, _v: Vec3, _w: Vec3,
) {}
#[inline(never)]
pub fn bench_s0(_s: BenchS0) {}
#[inline(never)]
pub fn bench_sf(_s: BenchSf) {}
#[inline(never)]
pub fn bench_sv(_s: BenchSv) {}
#[inline(never)]
pub fn bench_sm(_s: BenchSm) {}
#[inline(never)]
pub fn bench_sa(_s: BenchSa) {}
#[inline(never)]
pub fn bench_sz(_s: BenchSz) {}

// ── W5 BVH counterparts (Day-4 plan §5.5) ───────────────────────────────────
// Rayon over queries; the walk is the same `shannon_core::bvh` the GPU runs.

pub fn bvh_count_aabb(nodes: &[BvhNode], lowers: &[Vec3], uppers: &[Vec3], counts: &mut [i32]) {
    counts.par_iter_mut().enumerate().for_each(|(i, out)| {
        *out = shannon_core::bvh::query_aabb(nodes, lowers[i], uppers[i]).count() as i32;
    });
}

pub fn bvh_count_ray(
    nodes: &[BvhNode],
    starts: &[Vec3],
    dirs: &[Vec3],
    t_max: f32,
    counts: &mut [i32],
) {
    counts.par_iter_mut().enumerate().for_each(|(i, out)| {
        *out = shannon_core::bvh::query_ray(nodes, starts[i], dirs[i], t_max).count() as i32;
    });
}

/// `hits` is chunked per query (`max_hits` slots each) — disjoint ranges, no
/// unsafe. Counts are stored UNclamped, mirroring the GPU kernel's overflow
/// contract (host asserts `count < max_hits`).
pub fn bvh_hits_aabb(
    nodes: &[BvhNode],
    lowers: &[Vec3],
    uppers: &[Vec3],
    max_hits: usize,
    hits: &mut [i32],
    counts: &mut [i32],
) {
    hits.par_chunks_mut(max_hits).zip(counts.par_iter_mut()).enumerate().for_each(
        |(i, (chunk, out))| {
            let mut c = 0usize;
            for prim in shannon_core::bvh::query_aabb(nodes, lowers[i], uppers[i]) {
                if c < max_hits {
                    chunk[c] = prim;
                }
                c += 1;
            }
            *out = c as i32;
        },
    );
}

#[allow(clippy::too_many_arguments)] // arity mirrors the GPU kernel
pub fn bvh_hits_ray(
    nodes: &[BvhNode],
    starts: &[Vec3],
    dirs: &[Vec3],
    t_max: f32,
    max_hits: usize,
    hits: &mut [i32],
    counts: &mut [i32],
) {
    hits.par_chunks_mut(max_hits).zip(counts.par_iter_mut()).enumerate().for_each(
        |(i, (chunk, out))| {
            let mut c = 0usize;
            for prim in shannon_core::bvh::query_ray(nodes, starts[i], dirs[i], t_max) {
                if c < max_hits {
                    chunk[c] = prim;
                }
                c += 1;
            }
            *out = c as i32;
        },
    );
}
