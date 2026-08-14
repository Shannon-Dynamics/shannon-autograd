// Workload structure derived from NVIDIA Warp (https://github.com/NVIDIA/warp,
// warp/examples/core/example_mesh.py), SPDX-License-Identifier: Apache-2.0 —
// see NOTICE at the workspace root. The particle force model and the
// deform → refit → step frame loop follow that example; the analytic
// tunnelling predicate and settle acceptance are original.

//! W2 — 1000 particles fall onto a deforming mesh, collide, and settle
//! (Day-5 plan §5.11). This is the week plan's actual W2 workload; the other
//! `w2_*` binary, `shannon_anim`, is Day 2's bonus SDF demo — the `w`
//! number tracks the week-plan WORKLOAD, not the day.
//!
//! Per frame: deform (GPU, pure function of rest positions) → refit (host,
//! over downloaded device points) → particle step (GPU, double-buffered) →
//! tunnelling predicate (host, ANALYTIC — the deform is a standing wave, so
//! the exact surface height under every particle is known in closed form).
//!
//! No-tunnelling is CFL-by-construction, not luck: the particle step is
//! clamped to margin/2 and the surface moves ≤ amp·ω·dt ≈ 0.002 per frame,
//! so max closing displacement < margin. Unsigned queries are SOUND here
//! because the sheet is a heightfield — a particle that never crosses is
//! never "inside", so the pushout direction is always correct.

use anyhow::Result;
use shannon_core::mesh::mesh_eval_position;
use shannon_core::{MeshQuery, Particle, Vec3};
use shannon_examples::obj::{write_obj, write_obj_points};
use shannon_rt::{Array, Device, launch};
use shannon_spatial::shapes::grid;
use shannon_spatial::{Mesh, brute_force_closest_point};
use std::path::PathBuf;

const DT: f32 = 1.0 / 60.0;
const MARGIN: f32 = 0.1;
const AMP: f32 = 0.25;
/// Slow phase: sin(ω·t) stays in (0, 1] for all 200 frames — the valley
/// pattern never inverts, so particles settle into FIXED valleys.
const OMEGA: f32 = 0.5;
/// Query radius for the sim step (particles further than this skip collision).
const MAX_DIST: f32 = 1.0;
/// Unbounded-ish radius for the validation checks — real distances, no misses.
const CHECK_MAX_DIST: f32 = 1.0e10;
const Y_FLOOR: f32 = -2.0;
const GRID_EXTENT: f32 = 5.0;

/// Minimal LCG (Knuth MMIX constants), seed 42 — the house generator.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) & 0x00FF_FFFF) as f32 / 16_777_216.0
    }
}

struct Args {
    frames: usize,
    particles: usize,
    grid: usize,
    dump_every: usize,
    out_dir: PathBuf,
}

fn parse_args() -> Args {
    let mut args = Args {
        frames: 200,
        particles: 1000,
        grid: 64,
        dump_every: 10,
        out_dir: PathBuf::from("frames"),
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--frames" => {
                args.frames = argv[i + 1].parse().expect("--frames N");
                i += 2;
            }
            "--particles" => {
                args.particles = argv[i + 1].parse().expect("--particles N");
                i += 2;
            }
            "--grid" => {
                args.grid = argv[i + 1].parse().expect("--grid N");
                i += 2;
            }
            "--dump-every" => {
                args.dump_every = argv[i + 1].parse().expect("--dump-every N (0 = off)");
                i += 2;
            }
            "--out-dir" => {
                args.out_dir = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
    args
}

/// Spawn strictly inside the grid footprint so every particle lands on mesh
/// (the floor at Y_FLOOR is a backstop, not a crutch).
fn spawn_particles(rng: &mut Lcg, n: usize) -> Vec<Particle> {
    (0..n)
        .map(|_| Particle {
            pos: Vec3::new(
                -4.0 + 8.0 * rng.next_f32(),
                1.0 + rng.next_f32(),
                -4.0 + 8.0 * rng.next_f32(),
            ),
            vel: Vec3::ZERO,
        })
        .collect()
}

/// Analytic surface height under (x, z) — the same standing wave the deform
/// kernel evaluates (rest y = 0 on the demo grid).
fn surface_y(x: f32, z: f32, phase: f32) -> f32 {
    AMP * x.sin() * z.cos() * phase.sin()
}

/// The tunnelling predicate — every particle, every frame. Inside the grid
/// footprint the mesh chord tracks the analytic surface to ≲ 1e-3 at this
/// resolution, two orders under the margin/2 tolerance.
fn assert_no_tunnelling(ps: &[Particle], phase: f32, frame: usize) {
    for (i, p) in ps.iter().enumerate() {
        let (x, z) = (p.pos.x, p.pos.z);
        if x.abs() <= GRID_EXTENT && z.abs() <= GRID_EXTENT {
            let floor = surface_y(x, z, phase) - 0.5 * MARGIN;
            assert!(
                p.pos.y >= floor,
                "frame {frame}: particle {i} tunnelled — y {} < surface bound {floor}",
                p.pos.y
            );
        } else {
            // Off the mesh footprint (should not happen — spawns are ≥ 1
            // inside the edge and valleys are interior): floor backstop.
            assert!(
                p.pos.y >= Y_FLOOR + MARGIN - 1e-4,
                "frame {frame}: escaped particle {i} below the floor"
            );
        }
    }
}

/// The Day-5 comparison rule (plan §5.9): found-flags + distance +
/// per-backend barycentric self-consistency. NEVER face indices across
/// backends — equidistant-triangle ties are legitimately backend-dependent.
fn assert_queries_agree(
    label: &str,
    i: usize,
    got: &MeshQuery,
    reference: &MeshQuery,
    points: &[Vec3],
    indices: &[i32],
    query_point: Vec3,
) {
    assert_eq!(got.face >= 0, reference.face >= 0, "{label} query {i}: found-flags diverge");
    if got.face < 0 {
        return;
    }
    let tol = 1e-4 * 1.0f32.max(reference.dist);
    assert!(
        (got.dist - reference.dist).abs() <= tol,
        "{label} query {i}: dist {} vs reference {} (tol {tol})",
        got.dist,
        reference.dist
    );
    let cp = mesh_eval_position(points, indices, got.face, got.u, got.v);
    assert!(
        ((cp - query_point).length() - got.dist).abs() <= 1e-4 * 1.0f32.max(got.dist),
        "{label} query {i}: eval_position(face,u,v) disagrees with reported dist"
    );
}

/// Three-way closest-point check on up to 128 sampled particle positions:
/// GPU == brute AND CPU-rayon == brute, all three against the SAME downloaded
/// points and current (post-refit) nodes.
fn three_way_check(mesh: &Mesh, ps: &[Particle], frame: usize) -> Result<()> {
    let pts = mesh.points().to_vec()?;
    let queries: Vec<Vec3> = ps.iter().take(128).map(|p| p.pos).collect();
    let m = queries.len();

    let q_dev = Array::from_slice(&queries)?;
    let mut out_gpu = Array::<MeshQuery>::zeros(m)?;
    launch!(mesh_query, dim = m, (mesh.nodes(), mesh.points(), mesh.indices(), &q_dev, CHECK_MAX_DIST, &mut out_gpu))?;
    let gpu = out_gpu.to_vec()?;

    let mut cpu: Vec<MeshQuery> = vec![MeshQuery::default(); m];
    shannon_cpu::mesh_query(
        mesh.host_nodes(),
        &pts,
        mesh.host_indices(),
        &queries,
        CHECK_MAX_DIST,
        &mut cpu,
    );

    for i in 0..m {
        let brute = brute_force_closest_point(&pts, mesh.host_indices(), queries[i], CHECK_MAX_DIST);
        assert_queries_agree("GPU", i, &gpu[i], &brute, &pts, mesh.host_indices(), queries[i]);
        assert_queries_agree("CPU", i, &cpu[i], &brute, &pts, mesh.host_indices(), queries[i]);
    }
    println!("✓ frame {frame}: three-way closest-point check — GPU == brute == CPU ({m} samples)");
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args();
    Device::default()?;
    let mut rng = Lcg::new(42);

    let (rest, idx) = grid(args.grid, GRID_EXTENT);
    let mut mesh = Mesh::new(&rest, &idx)?;
    let rest_arr = Array::from_slice(&rest)?;
    let n_verts = rest.len();

    let spawned = spawn_particles(&mut rng, args.particles);
    let mut parts_a = Array::from_slice(&spawned)?;
    let mut parts_b = Array::<Particle>::zeros(args.particles)?;

    if args.dump_every > 0 {
        std::fs::create_dir_all(&args.out_dir)?;
    }
    println!(
        "Mesh sim — {} particles on a {}×{} grid ({} tris), {} frames\n",
        args.particles,
        args.grid,
        args.grid,
        idx.len() / 3,
        args.frames
    );

    let mut last_ps: Vec<Particle> = spawned;
    for frame in 0..args.frames {
        let phase = OMEGA * (frame as f32 * DT);

        // deform → refit → simulate: the critical ordering. The query kernel
        // must traverse bounds refreshed from the very points it reads.
        launch!(mesh_deform, dim = n_verts, (&rest_arr, phase, AMP, mesh.points_mut()))?;
        mesh.refit()?;
        launch!(sim_particles, dim = args.particles,
                (&parts_a, mesh.nodes(), mesh.points(), mesh.indices(),
                 MARGIN, DT, MAX_DIST, Y_FLOOR, &mut parts_b))?;
        std::mem::swap(&mut parts_a, &mut parts_b);

        let ps = parts_a.to_vec()?;
        assert_no_tunnelling(&ps, phase, frame);
        if frame == 0 || frame == 100 {
            three_way_check(&mesh, &ps, frame)?;
        }
        if args.dump_every > 0 && frame % args.dump_every == 0 {
            let pts = mesh.points().to_vec()?;
            write_obj(&args.out_dir.join(format!("mesh_{frame:04}.obj")), &pts, &idx)?;
            write_obj_points(
                &args.out_dir.join(format!("parts_{frame:04}.obj")),
                &ps.iter().map(|p| p.pos).collect::<Vec<_>>(),
            )?;
        }
        if frame % 50 == 49 {
            let max_v = ps.iter().map(|p| p.vel.length()).fold(0.0f32, f32::max);
            println!("  frame {:>3}: max |v| = {max_v:.3}", frame + 1);
        }
        last_ps = ps;
    }

    // ── Settle predicate (frame N): resting particles track the surface ────
    let max_v = last_ps.iter().map(|p| p.vel.length()).fold(0.0f32, f32::max);
    let mean_v =
        last_ps.iter().map(|p| p.vel.length()).sum::<f32>() / last_ps.len() as f32;
    println!("\nsettle @ frame {}: max |v| = {max_v:.4}, mean |v| = {mean_v:.4}", args.frames);
    assert!(max_v < 0.25, "settle failed: max |v| = {max_v} ≥ 0.25");
    assert!(mean_v < 0.05, "settle failed: mean |v| = {mean_v} ≥ 0.05");

    if args.dump_every > 0 {
        println!("OBJ frames in {}/", args.out_dir.display());
    }
    println!("\n✅ MESH SIM ACCEPTANCE PASSED");
    Ok(())
}
