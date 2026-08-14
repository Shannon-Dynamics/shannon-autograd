//! Conveyor sort — particles rain onto a grooved, running conveyor belt; the
//! belt conveys them along +X while its grooves channel them across Z into
//! discrete lanes; at the belt's end they drop off and land sorted into bins.
//!
//! Original scenario and force model (`shannon_core::conveyor`): pushout onto
//! the margin shell + a belt-grip term driving tangential velocity toward the
//! belt's conveying velocity. Sorting is geometry — groove walls give every
//! pushout normal a Z-component pointing at the nearest lane center.
//!
//! Per frame: deform (GPU, pure function of rest positions + time) → refit
//! (host) → particle step (GPU, double-buffered) → tunnelling predicate
//! (host, ANALYTIC — the belt surface height is known in closed form).
//!
//! Acceptance: zero tunnelling violations, three-way closest-point agreement
//! at frames 0 and 100, a minimum delivery fraction off the belt's end, and a
//! minimum lane-sorting accuracy among the delivered particles.

use anyhow::Result;
use shannon_core::conveyor::belt_surface_y;
use shannon_core::mesh::mesh_eval_position;
use shannon_core::{MeshQuery, Particle, Vec3};
use shannon_examples::obj::{write_obj, write_obj_points};
use shannon_rt::{Array, Device, launch};
use shannon_spatial::shapes::grid;
use shannon_spatial::{Mesh, brute_force_closest_point};
use std::path::PathBuf;

const DT: f32 = 1.0 / 60.0;
const MARGIN: f32 = 0.1;
const MAX_DIST: f32 = 1.0;
/// Unbounded-ish radius for the validation checks — real distances, no misses.
const CHECK_MAX_DIST: f32 = 1.0e10;
const Y_FLOOR: f32 = -2.0;
const EXTENT: f32 = 5.0;

const GROOVE_AMP: f32 = 0.35;
const LANE_W: f32 = 2.0;
const RIPPLE_AMP: f32 = 0.06;
const RIPPLE_K: f32 = 2.0;
const BELT_SPEED: f32 = 1.2;

/// Surface-motion bound per frame: `RIPPLE_AMP·RIPPLE_K·BELT_SPEED·DT`
/// = 0.0024, two orders under the CFL step clamp of `MARGIN/2` = 0.05 —
/// no-tunnelling is by construction, not luck.
const _SURFACE_MOTION_BOUND: f32 = RIPPLE_AMP * RIPPLE_K * BELT_SPEED * DT;

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
        frames: 600,
        particles: 600,
        grid: 64,
        dump_every: 0,
        out_dir: PathBuf::from("conveyor_frames"),
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

/// Spawn over the upstream half of the belt, strictly inside the footprint.
fn spawn_particles(rng: &mut Lcg, n: usize) -> Vec<Particle> {
    (0..n)
        .map(|_| Particle {
            pos: Vec3::new(
                -4.5 + 3.0 * rng.next_f32(),
                0.8 + 0.8 * rng.next_f32(),
                -4.6 + 9.2 * rng.next_f32(),
            ),
            vel: Vec3::ZERO,
        })
        .collect()
}

fn surface_y(x: f32, z: f32, t: f32) -> f32 {
    belt_surface_y(x, z, t, GROOVE_AMP, LANE_W, RIPPLE_AMP, RIPPLE_K, BELT_SPEED)
}

/// The tunnelling predicate — every particle, every frame. Inside the belt
/// footprint the mesh chord tracks the analytic surface to ≲ 6e-3 at this
/// resolution (chord sag ≈ h²/8·max|f''|), an order under margin/2.
fn assert_no_tunnelling(ps: &[Particle], t: f32, frame: usize) {
    for (i, p) in ps.iter().enumerate() {
        let (x, z) = (p.pos.x, p.pos.z);
        if x.abs() <= EXTENT && z.abs() <= EXTENT {
            let floor = surface_y(x, z, t) - 0.5 * MARGIN;
            assert!(
                p.pos.y >= floor,
                "frame {frame}: particle {i} tunnelled — y {} < surface bound {floor}",
                p.pos.y
            );
        } else {
            assert!(
                p.pos.y >= Y_FLOOR + MARGIN - 1e-4,
                "frame {frame}: particle {i} below the floor off-belt"
            );
        }
    }
}

/// Comparison rule shared with the mesh sim: found-flags + distance +
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

fn nearest_lane(z: f32) -> i32 {
    (z / LANE_W).round() as i32
}

fn main() -> Result<()> {
    let args = parse_args();
    Device::default()?;
    let mut rng = Lcg::new(42);

    let (rest, idx) = grid(args.grid, EXTENT);
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
        "Conveyor sort — {} particles onto a {}×{} belt ({} tris), lanes every {LANE_W}, {} frames\n",
        args.particles,
        args.grid,
        args.grid,
        idx.len() / 3,
        args.frames
    );

    let mut last_ps: Vec<Particle> = spawned;
    for frame in 0..args.frames {
        let t = frame as f32 * DT;

        // deform → refit → step: the query kernel must traverse bounds
        // refreshed from the very points it reads.
        launch!(belt_deform, dim = n_verts,
                (&rest_arr, t, GROOVE_AMP, LANE_W, RIPPLE_AMP, RIPPLE_K, BELT_SPEED, mesh.points_mut()))?;
        mesh.refit()?;
        launch!(conveyor_step, dim = args.particles,
                (&parts_a, mesh.nodes(), mesh.points(), mesh.indices(),
                 MARGIN, DT, MAX_DIST, Y_FLOOR, BELT_SPEED, &mut parts_b))?;
        std::mem::swap(&mut parts_a, &mut parts_b);

        let ps = parts_a.to_vec()?;
        assert_no_tunnelling(&ps, t, frame);
        if frame == 0 || frame == 100 {
            three_way_check(&mesh, &ps, frame)?;
        }
        if args.dump_every > 0 && frame % args.dump_every == 0 {
            let pts = mesh.points().to_vec()?;
            write_obj(&args.out_dir.join(format!("belt_{frame:04}.obj")), &pts, &idx)?;
            write_obj_points(
                &args.out_dir.join(format!("parts_{frame:04}.obj")),
                &ps.iter().map(|p| p.pos).collect::<Vec<_>>(),
            )?;
        }
        if frame % 100 == 99 {
            let delivered = ps.iter().filter(|p| p.pos.x > EXTENT).count();
            println!("  frame {:>3}: delivered {delivered}/{}", frame + 1, args.particles);
        }
        last_ps = ps;
    }

    // ── Delivery + sorting predicates ───────────────────────────────────────
    let delivered: Vec<&Particle> = last_ps.iter().filter(|p| p.pos.x > EXTENT).collect();
    let frac = delivered.len() as f32 / last_ps.len() as f32;

    let mut hist = std::collections::BTreeMap::<i32, usize>::new();
    let mut sorted_ok = 0usize;
    for p in &delivered {
        let lane = nearest_lane(p.pos.z);
        *hist.entry(lane).or_default() += 1;
        if (p.pos.z - lane as f32 * LANE_W).abs() <= 0.25 * LANE_W {
            sorted_ok += 1;
        }
    }
    let sort_acc = if delivered.is_empty() { 0.0 } else { sorted_ok as f32 / delivered.len() as f32 };

    println!("\ndelivered off the belt end: {}/{} ({:.0}%)", delivered.len(), last_ps.len(), frac * 100.0);
    print!("lane histogram:");
    for (lane, n) in &hist {
        print!("  z={:+.0}: {n}", *lane as f32 * LANE_W);
    }
    println!("\nsorted within ±{}·lane: {:.0}%", 0.25, sort_acc * 100.0);

    assert!(frac >= 0.60, "delivery failed: {:.0}% < 60%", frac * 100.0);
    assert!(sort_acc >= 0.80, "sorting failed: {:.0}% < 80%", sort_acc * 100.0);

    if args.dump_every > 0 {
        println!("OBJ frames in {}/", args.out_dir.display());
    }
    println!("\n✅ CONVEYOR SORT ACCEPTANCE PASSED");
    Ok(())
}
