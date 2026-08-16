//! W2 mesh-closest-point validation + benchmark (Day-5 plan §5.9).
//!
//! A NEW binary rather than rows bolted onto `bvh_bench` — that binary is
//! Day 4's frozen acceptance artifact for the BVH workload (recorded deviation
//! 4 from the week plan's W6 note; the RESULTS.md *table* stays unified).
//!
//! Phase 1 (validation): three-way agreement at 128 queries — GPU == brute
//! AND CPU-rayon == brute — under the float-tie-proof comparison rule (found
//! flags + distance + per-backend barycentric self-consistency; NEVER face
//! indices across backends). At every benchmarked size, GPU is additionally
//! checked against CPU before any timing is believed (Day-4 pattern).
//!
//! Phase 2 (benchmark): GPU cells are DEVICE time via `GpuTimer` (5 warm-up +
//! `--iters` event-timed launches) — the W5 methodology, NOT W4's
//! wall-clock-enqueue. CPU cells are rayon wall-clock over the same loop.

use anyhow::Result;
use shannon_core::mesh::mesh_eval_position;
use shannon_core::{MeshQuery, Vec3};
use shannon_examples::obj::write_obj;
use shannon_kernels::launch;
use shannon_rt::{Array, Device, GpuTimer, ScopedTimer};
use shannon_spatial::shapes::icosphere;
use shannon_spatial::{Mesh, brute_force_closest_point};

/// Finite (the strict-< accept and the Warp API both want a real number).
const MAX_DIST: f32 = 1.0e10;

/// Minimal LCG (Knuth MMIX constants), 24-bit mantissa → uniform [0, 1).
/// Seed 42, same generator as the test suites — and reimplemented verbatim in
/// the Warp baseline script so both stacks query IDENTICAL points.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) & 0x00FF_FFFF) as f32 / 16_777_216.0
    }
    fn vec3(&mut self) -> Vec3 {
        Vec3::new(self.next_f32(), self.next_f32(), self.next_f32())
    }
}

struct Args {
    sizes: Vec<usize>,
    iters: u32,
    subdiv: u32,
    dump_obj: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        sizes: vec![128, 16_384],
        iters: 100,
        subdiv: 5,
        dump_obj: None,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--queries" => {
                args.sizes = vec![argv[i + 1].parse().expect("--queries N")];
                i += 2;
            }
            "--iters" => {
                args.iters = argv[i + 1].parse().expect("--iters N");
                i += 2;
            }
            "--subdiv" => {
                args.subdiv = argv[i + 1].parse().expect("--subdiv N");
                i += 2;
            }
            "--dump-obj" => {
                args.dump_obj = Some(argv[i + 1].clone());
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
    args
}

/// Query points in the unit sphere's AABB scaled 2× — a near/far mix.
fn gen_queries(rng: &mut Lcg, m: usize) -> Vec<Vec3> {
    (0..m)
        .map(|_| rng.vec3() * 4.0 - Vec3::splat(2.0))
        .collect()
}

/// The Day-5 comparison rule (plan §5.9). `reference` is brute force in phase
/// 1 and the CPU tree at benchmark sizes; faces are never compared.
fn assert_queries_agree(
    label: &str,
    i: usize,
    got: &MeshQuery,
    reference: &MeshQuery,
    points: &[Vec3],
    indices: &[i32],
    query_point: Vec3,
) {
    assert_eq!(
        got.face >= 0,
        reference.face >= 0,
        "{label} query {i}: found-flags diverge"
    );
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

fn mean_stdev(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// One GPU cell: 5 warm-up launches, sync, then `iters` × event-timed launch.
fn bench_gpu<F: FnMut() -> Result<()>>(dev: &Device, iters: u32, mut one: F) -> Result<(f64, f64)> {
    for _ in 0..5 {
        one()?;
    }
    dev.synchronize()?;
    let t = GpuTimer::new(dev)?;
    let mut xs = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        t.start(dev)?;
        one()?;
        t.stop(dev)?;
        xs.push(t.elapsed_ms()? as f64);
    }
    Ok(mean_stdev(&xs))
}

/// One CPU cell: same warm-up + iteration count, rayon wall-clock.
fn bench_cpu<F: FnMut()>(iters: u32, mut one: F) -> (f64, f64) {
    for _ in 0..5 {
        one();
    }
    let mut xs = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = ScopedTimer::quiet("cell");
        one();
        xs.push(t.elapsed_ms());
    }
    mean_stdev(&xs)
}

fn main() -> Result<()> {
    let args = parse_args();
    let dev = Device::default()?;
    let mut rng = Lcg::new(42);

    let (points, indices) = icosphere(args.subdiv, 1.0);
    let n_tris = indices.len() / 3;
    let mesh = Mesh::new(&points, &indices)?;
    println!(
        "Mesh-cp bench — icosphere subdiv {} ({} tris / {} verts), {} iterations, \
         GPU cells in DEVICE time (GpuTimer)\n",
        args.subdiv,
        n_tris,
        points.len(),
        args.iters
    );

    if let Some(path) = &args.dump_obj {
        write_obj(std::path::Path::new(path), &points, &indices)?;
        println!("mesh dumped to {path} (for the Warp baseline)\n");
    }

    // ── Phase 1: three-way validation at 128 — before any timing ──────────
    let m = 128;
    let qs = gen_queries(&mut rng, m);
    let q_dev = Array::from_slice(&qs)?;
    let mut out_gpu = Array::<MeshQuery>::zeros(m)?;
    let mut out_cpu = vec![MeshQuery::default(); m];

    launch!(
        mesh_query,
        dim = m,
        (
            mesh.nodes(),
            mesh.points(),
            mesh.indices(),
            &q_dev,
            MAX_DIST,
            &mut out_gpu
        )
    )?;
    shannon_cpu::mesh_query(
        mesh.host_nodes(),
        &points,
        mesh.host_indices(),
        &qs,
        MAX_DIST,
        &mut out_cpu,
    );
    let gpu_results = out_gpu.to_vec()?;
    for i in 0..m {
        let brute = brute_force_closest_point(&points, &indices, qs[i], MAX_DIST);
        assert_queries_agree("GPU", i, &gpu_results[i], &brute, &points, &indices, qs[i]);
        assert_queries_agree("CPU", i, &out_cpu[i], &brute, &points, &indices, qs[i]);
    }
    println!("✓ validation: closest points — GPU == brute == CPU ({m} queries × {n_tris} tris)\n");

    // ── Phase 2: benchmark ─────────────────────────────────────────────────
    struct Row {
        m: usize,
        cpu: (f64, f64),
        gpu: (f64, f64),
    }
    let mut rows: Vec<Row> = Vec::new();

    for &m in &args.sizes {
        let qs = gen_queries(&mut rng, m);
        let q_dev = Array::from_slice(&qs)?;
        let mut out_gpu = Array::<MeshQuery>::zeros(m)?;
        let mut out_cpu = vec![MeshQuery::default(); m];

        let gpu = bench_gpu(dev, args.iters, || {
            launch!(
                mesh_query,
                dim = m,
                (
                    mesh.nodes(),
                    mesh.points(),
                    mesh.indices(),
                    &q_dev,
                    MAX_DIST,
                    &mut out_gpu
                )
            )
        })?;
        let cpu = bench_cpu(args.iters, || {
            shannon_cpu::mesh_query(
                mesh.host_nodes(),
                &points,
                mesh.host_indices(),
                &qs,
                MAX_DIST,
                &mut out_cpu,
            );
        });

        // GPU must agree with CPU at EVERY benchmarked size — phase 1
        // validated against brute at 128; this pins the at-scale path too.
        let gpu_results = out_gpu.to_vec()?;
        for i in 0..m {
            assert_queries_agree(
                "GPU@scale",
                i,
                &gpu_results[i],
                &out_cpu[i],
                &points,
                &indices,
                qs[i],
            );
        }

        println!(
            "  mesh-cp @ {m:>5}: cpu {:>8.3} ± {:>6.3} ms | gpu {:>8.3} ± {:>6.3} ms | {:>7.2}×",
            cpu.0,
            cpu.1,
            gpu.0,
            gpu.1,
            cpu.0 / gpu.0
        );
        rows.push(Row { m, cpu, gpu });
    }

    // ── Markdown rows (paste-ready; same format as bvh_bench) ───────────
    println!("\n| workload | queries | CPU ms (rayon) | GPU ms (device) | GPU speedup |");
    println!("|----------|---------|----------------|-----------------|-------------|");
    for r in &rows {
        println!(
            "| {:<8} | {:>7} | {:>14.3} | {:>15.3} | {:>10.2}× |",
            "mesh-cp",
            r.m,
            r.cpu.0,
            r.gpu.0,
            r.cpu.0 / r.gpu.0
        );
    }

    // ── Sanity — bounds only, never performance gates ──────────────────────
    for r in &rows {
        assert!(
            r.cpu.0.is_finite() && r.cpu.0 > 0.0 && r.gpu.0.is_finite() && r.gpu.0 > 0.0,
            "non-finite or zero mean at {} queries",
            r.m
        );
    }

    println!("\n✅ MESH BENCH PASSED");
    Ok(())
}
