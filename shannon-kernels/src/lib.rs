//! shannon-kernels — W0/W1's GPU kernels. 🧪 First-party example.
//!
//! Elementwise map kernels are DECLARED, not written: one row per kernel via
//! `shannon_gpu_kernels!` (Day-2 task 2.7). Kernels that do not fit the map
//! shape — adjoints (scatter via GradSink), spikes — live in the `raw` block.
//!
//! Output-type rule (Day-1 plan §5.4):
//!   forward  → `DisjointSlice<T>`   (race-free by construction; the macro emits this)
//!   adjoint  → `&[T]` + `GradSink`  (scatter-add must opt out)

// The kernel-row macro is a token-muncher; the row count now exceeds the
// default recursion limit of 128 expansions.
#![recursion_limit = "512"]

use cuda_device::{kernel, thread};
use shannon_core::{BvhNode, Mat33, MeshQuery, Particle, Quat, Vec3};

pub mod bench;
use bench::{BenchS0, BenchSa, BenchSf, BenchSm, BenchSv, BenchSz};
use shannon_device::{DeviceGradF32, DeviceGradVec3};

shannon_core::shannon_gpu_kernels! {
    /// FORWARD: y[i] = a[i] * scale + bias
    affine(a: &[f32], scale: f32, bias: f32) -> f32 = shannon_core::adjoint::affine_at;

    /// W1 pixel shader: one thread shades one pixel via the shared scene body.
    draw(cam_pos: Vec3, cam_rot: Quat, width: u32, height: u32) -> Vec3
        = shannon_core::scene::draw_at;

    /// Animated SHANNON-sign scene (early-exit march; host feeds all transforms).
    draw_shannon(s1: Vec3, s2: Vec3, grip_dir: Vec3, grip: f32,
                 h_pos: Vec3, h_rot: Quat, cam_rot: Quat,
                 width: u32, height: u32) -> Vec3
        = shannon_core::scene_shannon::draw_rt_at;

    /// W2 closest-point query: one thread per query point (Day-5 plan §5.8).
    /// The traversal is a macro row — not @raw like the Day-4 BVH kernels —
    /// because MeshQuery is a single POD return, which is all the row needs.
    mesh_query(nodes: &[BvhNode], points: &[Vec3], indices: &[i32],
               queries: &[Vec3], max_dist: f32)
        -> MeshQuery = shannon_core::mesh::mesh_query_point_at;

    /// W2 deform: pure function of REST positions + phase — no accumulation,
    /// no drift (Day-5 plan §4.2).
    mesh_deform(rest: &[Vec3], phase: f32, amp: f32) -> Vec3
        = shannon_core::mesh::deform_at;

    /// W2 particle step: reads parts[i] + the mesh, returns the updated
    /// particle. The host double-buffers and swaps (Day-5 plan §4.1/§4.3).
    sim_particles(parts: &[Particle], nodes: &[BvhNode], points: &[Vec3], indices: &[i32],
                  margin: f32, dt: f32, max_dist: f32, y_floor: f32)
        -> Particle = shannon_core::mesh::sim_particle_at;

    /// ARM-7 pick-and-place scene (early-exit march; host feeds joint
    /// positions from its forward kinematics).
    draw_arm7(elbow: Vec3, wrist: Vec3, hand_end: Vec3, f1: Vec3, f2: Vec3,
              h_pos: Vec3, h_yaw: f32, carrying: f32,
              cam_az: f32, cam_el: f32, cam_dist: f32,
              width: u32, height: u32) -> Vec3
        = shannon_core::scene_arm7::draw_arm7_at;

    /// Conveyor belt deform: grooved lanes + traveling ripple, pure function
    /// of REST positions + time.
    belt_deform(rest: &[Vec3], t: f32, groove_amp: f32, lane_w: f32,
                ripple_amp: f32, ripple_k: f32, belt_speed: f32)
        -> Vec3 = shannon_core::conveyor::belt_deform_at;

    /// Conveyor particle step: pushout + belt-grip drive toward the belt
    /// velocity. Host double-buffers and swaps.
    conveyor_step(parts: &[Particle], nodes: &[BvhNode], points: &[Vec3], indices: &[i32],
                  margin: f32, dt: f32, max_dist: f32, y_floor: f32, belt_speed: f32)
        -> Particle = shannon_core::conveyor::conveyor_step_at;

    /// W3 chain check: y = sin(a) — the middle link of the tape-chain oracle.
    sin_map(a: &[f32]) -> f32 = shannon_core::loss::sin_at;

    @raw {
        /// ADJOINT: ā[i] += ȳ[i] * scale
        /// NOTE: every parameter is `&[f32]` — no DisjointSlice anywhere. Adjoints scatter.
        #[kernel]
        pub fn adj_affine(adj_y: &[f32], scale: f32, adj_a: &[f32]) {
            let i = thread::index_1d().get();
            if i >= adj_y.len() {
                return; // manual guard — no DisjointSlice to do it for us
            }
            shannon_core::adjoint::adj_affine_at(i, adj_y, scale, &DeviceGradF32(adj_a));
        }

        // ── W3 loss + adjoint kernels (Day-6 plan §5.6) ─────────────────────
        // All scatter-shaped, which is WHY they are @raw: forward reductions
        // accumulate into loss[0] and adjoints scatter into grad buffers —
        // both through const-ref GradSink atomics, never DisjointSlice.
        // Consequence the host carries: loss and every grad buffer must be
        // zeroed each iteration (ops::begin_iteration).


        /// ADJOINT of sin_map: ā[i] += ȳ[i]·cos(a[i]).
        #[kernel]
        pub fn adj_sin_map(a: &[f32], adj_y: &[f32], adj_a: &[f32]) {
            let i = thread::index_1d().get();
            if i >= a.len() {
                return;
            }
            shannon_core::adjoint::adj_sin_at(i, a, adj_y, &DeviceGradF32(adj_a));
        }

        /// FORWARD reduction: loss[0] += Σ x[i].
        #[kernel]
        pub fn sum_scalar(x: &[f32], loss: &[f32]) {
            let i = thread::index_1d().get();
            if i >= x.len() {
                return;
            }
            use shannon_core::GradSink;
            DeviceGradF32(loss).accumulate(0, x[i]);
        }

        /// ADJOINT of sum_scalar: broadcast — x̄[i] += loss̄[0].
        #[kernel]
        pub fn adj_sum_scalar(adj_loss: &[f32], adj_x: &[f32]) {
            let i = thread::index_1d().get();
            if i >= adj_x.len() {
                return;
            }
            shannon_core::loss::adj_sum_at(i, adj_loss, &DeviceGradF32(adj_x));
        }

        /// FORWARD reduction: loss[0] += Σ ½‖xᵢ − tᵢ‖² (stage 1).
        #[kernel]
        pub fn l2_loss(x: &[Vec3], t: &[Vec3], loss: &[f32]) {
            let i = thread::index_1d().get();
            if i >= x.len() {
                return;
            }
            use shannon_core::GradSink;
            DeviceGradF32(loss).accumulate(0, shannon_core::loss::l2_term_at(i, x, t));
        }

        /// ADJOINT of l2_loss: x̄ᵢ += (xᵢ − tᵢ)·loss̄.
        #[kernel]
        pub fn adj_l2_loss(x: &[Vec3], t: &[Vec3], adj_loss: &[f32], adj_x: &[Vec3]) {
            let i = thread::index_1d().get();
            if i >= x.len() {
                return;
            }
            shannon_core::loss::adj_l2_at(i, x, t, adj_loss, &DeviceGradVec3(adj_x));
        }

        /// FORWARD A→B Chamfer: loss[0] += Σ ½‖srcᵢ − eval(corrᵢ on target)‖².
        #[kernel]
        pub fn chamfer_ab(src: &[Vec3], corr: &[MeshQuery], tpoints: &[Vec3],
                          tindices: &[i32], loss: &[f32]) {
            let i = thread::index_1d().get();
            if i >= src.len() {
                return;
            }
            use shannon_core::GradSink;
            DeviceGradF32(loss)
                .accumulate(0, shannon_core::loss::chamfer_ab_term_at(i, src, corr, tpoints, tindices));
        }

        /// ADJOINT of chamfer_ab: envelope theorem — src̄ᵢ += (srcᵢ − p)·loss̄.
        #[kernel]
        pub fn adj_chamfer_ab(src: &[Vec3], corr: &[MeshQuery], tpoints: &[Vec3],
                              tindices: &[i32], adj_loss: &[f32], adj_src: &[Vec3]) {
            let i = thread::index_1d().get();
            if i >= src.len() {
                return;
            }
            shannon_core::loss::adj_chamfer_ab_at(
                i, src, corr, tpoints, tindices, adj_loss, &DeviceGradVec3(adj_src));
        }

        /// FORWARD B→A Chamfer: loss[0] += Σ ½‖eval(corrᵢ on source) − tvᵢ‖².
        #[kernel]
        pub fn chamfer_ba(tverts: &[Vec3], corr: &[MeshQuery], spoints: &[Vec3],
                          sindices: &[i32], loss: &[f32]) {
            let i = thread::index_1d().get();
            if i >= tverts.len() {
                return;
            }
            use shannon_core::GradSink;
            DeviceGradF32(loss)
                .accumulate(0, shannon_core::loss::chamfer_ba_term_at(i, tverts, corr, spoints, sindices));
        }

        /// ADJOINT of chamfer_ba: the canonical gather→scatter — barycentric
        /// weights route the gradient to the source triangle's three vertices.
        #[kernel]
        pub fn adj_chamfer_ba(tverts: &[Vec3], corr: &[MeshQuery], spoints: &[Vec3],
                              sindices: &[i32], adj_loss: &[f32], adj_spoints: &[Vec3]) {
            let i = thread::index_1d().get();
            if i >= tverts.len() {
                return;
            }
            shannon_core::loss::adj_chamfer_ba_at(
                i, tverts, corr, spoints, sindices, adj_loss, &DeviceGradVec3(adj_spoints));
        }

        // ── W4 launch-overhead benchmark kernels (Day-3 plan §5.3) ─────────
        // Deliberately EMPTY bodies mirroring Warp's `tid = wp.tid()` — the
        // measurement is pure launch + argument marshalling. These cannot be
        // macro rows: rows always append a DisjointSlice output and a write,
        // and the whole point here is having NO outputs.

        #[kernel]
        pub fn bench_k0() {
            let _ = thread::index_1d().get();
        }

        #[kernel]
        pub fn bench_kf(_x: f32, _y: f32, _z: f32) {
            let _ = thread::index_1d().get();
        }

        #[kernel]
        pub fn bench_kv(_u: Vec3, _v: Vec3, _w: Vec3) {
            let _ = thread::index_1d().get();
        }

        #[kernel]
        pub fn bench_km(_m: Mat33, _n: Mat33, _o: Mat33) {
            let _ = thread::index_1d().get();
        }

        #[kernel]
        pub fn bench_ka(_a: &[f32], _b: &[f32], _c: &[f32]) {
            let _ = thread::index_1d().get();
        }

        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn bench_kz(
            _a: &[f32], _b: &[f32], _c: &[f32],
            _x: f32, _y: f32, _z: f32,
            _u: Vec3, _v: Vec3, _w: Vec3,
        ) {
            let _ = thread::index_1d().get();
        }

        #[kernel]
        pub fn bench_s0(_s: BenchS0) {
            let _ = thread::index_1d().get();
        }

        #[kernel]
        pub fn bench_sf(_s: BenchSf) {
            let _ = thread::index_1d().get();
        }

        #[kernel]
        pub fn bench_sv(_s: BenchSv) {
            let _ = thread::index_1d().get();
        }

        #[kernel]
        pub fn bench_sm(_s: BenchSm) {
            let _ = thread::index_1d().get();
        }

        #[kernel]
        pub fn bench_sa(_s: BenchSa) {
            let _ = thread::index_1d().get();
        }

        #[kernel]
        pub fn bench_sz(_s: BenchSz) {
            let _ = thread::index_1d().get();
        }

        // ── W5 BVH kernels (Day-4 plan §5.4) ───────────────────────────────
        // Traversal loops don't fit the macro row shape (no single-return
        // elementwise body). One thread = one query; the walk itself is
        // `shannon_core::bvh` — the same code the CPU adapters run.
        //
        // Count kernels (benchmark): one count per query → DisjointSlice,
        // race-free by construction.

        #[kernel]
        pub fn bvh_count_aabb(
            nodes: &[shannon_core::BvhNode],
            lowers: &[Vec3],
            uppers: &[Vec3],
            mut counts: cuda_device::DisjointSlice<i32>,
        ) {
            let idx = thread::index_1d();
            let i = idx.get();
            if let Some(out) = counts.get_mut(idx) {
                let mut q = shannon_core::bvh::query_aabb(nodes, lowers[i], uppers[i]);
                let mut c = 0i32;
                while q.next().is_some() {
                    c += 1;
                }
                *out = c;
            }
        }

        #[kernel]
        pub fn bvh_count_ray(
            nodes: &[shannon_core::BvhNode],
            starts: &[Vec3],
            dirs: &[Vec3],
            t_max: f32,
            mut counts: cuda_device::DisjointSlice<i32>,
        ) {
            let idx = thread::index_1d();
            let i = idx.get();
            if let Some(out) = counts.get_mut(idx) {
                let mut q = shannon_core::bvh::query_ray(nodes, starts[i], dirs[i], t_max);
                let mut c = 0i32;
                while q.next().is_some() {
                    c += 1;
                }
                *out = c;
            }
        }

        // Hit-list kernels (validation): each thread writes the DISJOINT range
        // hits[i*max_hits .. i*max_hits+c] — not expressible in DisjointSlice's
        // Index1D, so `&mut [i32]` is the documented §10.4 opt-out. The stored
        // count is UNclamped: the host detects buffer overflow as
        // `count >= max_hits` instead of comparing silently-truncated sets.

        #[kernel]
        pub fn bvh_hits_aabb(
            nodes: &[shannon_core::BvhNode],
            lowers: &[Vec3],
            uppers: &[Vec3],
            max_hits: u32,
            hits: &mut [i32],
            mut counts: cuda_device::DisjointSlice<i32>,
        ) {
            let idx = thread::index_1d();
            let i = idx.get();
            if let Some(out) = counts.get_mut(idx) {
                let base = i * max_hits as usize;
                let q = shannon_core::bvh::query_aabb(nodes, lowers[i], uppers[i]);
                let mut c = 0usize;
                for prim in q {
                    if c < max_hits as usize && base + c < hits.len() {
                        hits[base + c] = prim;
                    }
                    c += 1;
                }
                *out = c as i32;
            }
        }

        #[kernel]
        pub fn bvh_hits_ray(
            nodes: &[shannon_core::BvhNode],
            starts: &[Vec3],
            dirs: &[Vec3],
            t_max: f32,
            max_hits: u32,
            hits: &mut [i32],
            mut counts: cuda_device::DisjointSlice<i32>,
        ) {
            let idx = thread::index_1d();
            let i = idx.get();
            if let Some(out) = counts.get_mut(idx) {
                let base = i * max_hits as usize;
                let q = shannon_core::bvh::query_ray(nodes, starts[i], dirs[i], t_max);
                let mut c = 0usize;
                for prim in q {
                    if c < max_hits as usize && base + c < hits.len() {
                        hits[base + c] = prim;
                    }
                    c += 1;
                }
                *out = c as i32;
            }
        }
    }
}

// This crate's cached PTX-module accessor: `pub fn module(&Device) -> …`.
// Same call every kernel crate (first- or third-party) makes; the embedded
// bundle is keyed by this crate's package name.
shannon_rt::define_module_cache!();

/// First-party `launch!` — `shannon_rt::launch_in!` bound to THIS crate's
/// module cache, so demo call sites keep the original one-argument syntax:
/// `launch!(affine, dim = n, (&a, scale, bias, &mut y))?`.
#[macro_export]
macro_rules! launch {
    ($kernel:ident, dim = $n:expr, ($($arg:expr),* $(,)?)) => {
        $crate::__shannon_rt::launch_in!($crate::module, $kernel, dim = $n, ($($arg),*))
    };
}

// Macro plumbing — `launch!` reaches shannon-rt through `$crate::…` so
// callers do not need shannon-rt's macros imported.
#[doc(hidden)]
pub use shannon_rt as __shannon_rt;
