//! W5 — BVH validation + spatial benchmark (Day-4 plan §5.6).
//!
//! Phase 1 (validation, acceptance item 1): three-way hit-SET comparison —
//! GPU == brute force AND CPU-rayon == brute force, sorted lists per query.
//! Sets, not counts: a traversal that prunes a wrong subtree and double-visits
//! another can still count correctly.
//!
//! Phase 2 (benchmark, acceptance item 2): GPU cells are DEVICE time via
//! `GpuTimer` — 5 warm-up launches, then one event pair per iteration —
//! deliberately the opposite of W4's wall-clock-enqueue methodology, because
//! W5 asks what the TRAVERSAL costs, not the dispatch (Day-4 plan §3, matching
//! Warp's `benchmark_bvh.py` under `wp.TIMING_KERNEL`). CPU cells are rayon
//! wall-clock over the same 100-iteration loop.

use anyhow::Result;
use shannon_core::Vec3;
use shannon_kernels::launch;
use shannon_rt::{Array, Device, GpuTimer, ScopedTimer};
use shannon_spatial::{Bvh, brute_force_aabb, brute_force_ray, build_median_split};

/// Per-query hit-list capacity for the validation kernels. At the Warp-mirrored
/// densities the mean is a few dozen hits per query; the host asserts every
/// count stays BELOW this, so clamping can never silently truncate a set.
const MAX_HITS: usize = 1024;
const T_MAX: f32 = 1.0e30;

/// Minimal LCG (Knuth MMIX constants), 24-bit mantissa → uniform [0, 1).
/// Same generator as the shannon-spatial test suite; seed 42 mirrors Warp.
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

fn parse_args() -> (usize, Vec<usize>, u32) {
    let (mut bounds, mut sizes, mut iters) = (10_000usize, vec![128usize, 16_384], 100u32);
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--bounds" => {
                bounds = argv[i + 1].parse().expect("--bounds N");
                i += 2;
            }
            "--queries" => {
                sizes = vec![argv[i + 1].parse().expect("--queries N")];
                i += 2;
            }
            "--iters" => {
                iters = argv[i + 1].parse().expect("--iters N");
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
    (bounds, sizes, iters)
}

/// Warp-mirrored data: box lowers uniform in [0,100)³, extents in [0,10)³.
fn gen_bounds(rng: &mut Lcg, n: usize) -> Vec<(Vec3, Vec3)> {
    (0..n)
        .map(|_| {
            let lo = rng.vec3() * 100.0;
            (lo, lo + rng.vec3() * 10.0)
        })
        .collect()
}

struct Queries {
    lo: Vec<Vec3>,
    hi: Vec<Vec3>,
    start: Vec<Vec3>,
    dir: Vec<Vec3>,
}

/// AABB queries: lowers in [0,80)³ + extents in [0,20)³. Rays: origins in
/// [0,50)³, directions normalized uniform-cube offsets — Warp's distributions.
fn gen_queries(rng: &mut Lcg, m: usize) -> Queries {
    let mut q = Queries {
        lo: Vec::with_capacity(m),
        hi: Vec::with_capacity(m),
        start: Vec::with_capacity(m),
        dir: Vec::with_capacity(m),
    };
    for _ in 0..m {
        let lo = rng.vec3() * 80.0;
        q.lo.push(lo);
        q.hi.push(lo + rng.vec3() * 20.0);
    }
    for _ in 0..m {
        q.start.push(rng.vec3() * 50.0);
        q.dir.push((rng.vec3() - Vec3::splat(0.5)).normalize());
    }
    q
}

fn mean_stdev(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// One GPU cell: 5 warm-up launches, sync, then `iters` × (event pair around
/// one launch, sync via `elapsed_ms`). Returns (mean, stdev) device ms.
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

/// One CPU cell: same warm-up + iteration count, wall-clock (rayon wall time
/// IS the CPU cost — there is no device to event-time). Returns (mean, stdev).
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

/// Assert sorted per-query hit lists agree three ways: GPU == brute AND
/// CPU == brute. `brute` yields the (already ascending) reference list.
fn check_three_way(
    kind: &str,
    gpu_hits: &[i32],
    gpu_counts: &[i32],
    cpu_hits: &[i32],
    cpu_counts: &[i32],
    brute: impl Fn(usize) -> Vec<i32>,
) {
    for i in 0..gpu_counts.len() {
        let reference = brute(i);
        for (label, hits, counts) in [("GPU", gpu_hits, gpu_counts), ("CPU", cpu_hits, cpu_counts)]
        {
            let c = counts[i];
            assert!(
                (c as usize) < MAX_HITS,
                "{kind} query {i}: {label} count {c} reached MAX_HITS — raise MAX_HITS"
            );
            let mut list = hits[i * MAX_HITS..i * MAX_HITS + c as usize].to_vec();
            list.sort_unstable();
            assert_eq!(
                list, reference,
                "{kind} query {i}: {label} hit set diverged from brute force"
            );
        }
    }
}

fn main() -> Result<()> {
    let (n_bounds, sizes, iters) = parse_args();
    let dev = Device::default()?;
    let mut rng = Lcg::new(42);

    let aabbs = gen_bounds(&mut rng, n_bounds);
    let nodes_host = build_median_split(&aabbs);
    let bvh = Bvh::build(&aabbs)?; // the same deterministic tree, uploaded
    assert_eq!(bvh.n_prims(), n_bounds);

    println!(
        "BVH bench — {n_bounds} bounds, {iters} iterations, GPU cells in DEVICE time (GpuTimer)\n"
    );

    // ── Phase 1: validation — before any timing is believed ────────────────
    let m = 128;
    let vq = gen_queries(&mut rng, m);
    let qlo = Array::from_slice(&vq.lo)?;
    let qhi = Array::from_slice(&vq.hi)?;
    let qstart = Array::from_slice(&vq.start)?;
    let qdir = Array::from_slice(&vq.dir)?;

    let mut hits = Array::<i32>::zeros(m * MAX_HITS)?;
    let mut counts = Array::<i32>::zeros(m)?;
    let mut cpu_hits = vec![0i32; m * MAX_HITS];
    let mut cpu_counts = vec![0i32; m];

    launch!(
        bvh_hits_aabb,
        dim = m,
        (
            bvh.nodes(),
            &qlo,
            &qhi,
            MAX_HITS as u32,
            &mut hits,
            &mut counts
        )
    )?;
    shannon_cpu::bvh_hits_aabb(
        &nodes_host,
        &vq.lo,
        &vq.hi,
        MAX_HITS,
        &mut cpu_hits,
        &mut cpu_counts,
    );
    check_three_way(
        "AABB",
        &hits.to_vec()?,
        &counts.to_vec()?,
        &cpu_hits,
        &cpu_counts,
        |i| brute_force_aabb(&aabbs, vq.lo[i], vq.hi[i]),
    );
    println!("✓ validation: AABB hit sets — GPU == brute == CPU ({m} queries × {n_bounds} bounds)");

    launch!(
        bvh_hits_ray,
        dim = m,
        (
            bvh.nodes(),
            &qstart,
            &qdir,
            T_MAX,
            MAX_HITS as u32,
            &mut hits,
            &mut counts
        )
    )?;
    shannon_cpu::bvh_hits_ray(
        &nodes_host,
        &vq.start,
        &vq.dir,
        T_MAX,
        MAX_HITS,
        &mut cpu_hits,
        &mut cpu_counts,
    );
    check_three_way(
        "ray",
        &hits.to_vec()?,
        &counts.to_vec()?,
        &cpu_hits,
        &cpu_counts,
        |i| brute_force_ray(&aabbs, vq.start[i], vq.dir[i], T_MAX),
    );
    println!(
        "✓ validation: ray  hit sets — GPU == brute == CPU ({m} queries × {n_bounds} bounds)\n"
    );

    // ── Phase 2: benchmark ─────────────────────────────────────────────────
    struct Row {
        workload: &'static str,
        m: usize,
        cpu: (f64, f64),
        gpu: (f64, f64),
    }
    let mut rows: Vec<Row> = Vec::new();

    for &m in &sizes {
        let qs = gen_queries(&mut rng, m);
        let qlo = Array::from_slice(&qs.lo)?;
        let qhi = Array::from_slice(&qs.hi)?;
        let qstart = Array::from_slice(&qs.start)?;
        let qdir = Array::from_slice(&qs.dir)?;
        let mut counts_gpu = Array::<i32>::zeros(m)?;
        let mut counts_cpu = vec![0i32; m];

        for workload in ["AABB", "ray"] {
            let gpu = bench_gpu(dev, iters, || match workload {
                "AABB" => launch!(
                    bvh_count_aabb,
                    dim = m,
                    (bvh.nodes(), &qlo, &qhi, &mut counts_gpu)
                ),
                _ => launch!(
                    bvh_count_ray,
                    dim = m,
                    (bvh.nodes(), &qstart, &qdir, T_MAX, &mut counts_gpu)
                ),
            })?;
            let cpu = bench_cpu(iters, || match workload {
                "AABB" => shannon_cpu::bvh_count_aabb(&nodes_host, &qs.lo, &qs.hi, &mut counts_cpu),
                _ => shannon_cpu::bvh_count_ray(
                    &nodes_host,
                    &qs.start,
                    &qs.dir,
                    T_MAX,
                    &mut counts_cpu,
                ),
            });
            // The count kernels must agree at EVERY benchmarked size — phase 1
            // validated sets at 128; this pins the at-scale path too.
            assert_eq!(
                counts_gpu.to_vec()?,
                counts_cpu,
                "{workload} counts diverged at {m} queries"
            );
            println!(
                "  {workload:<4} @ {m:>5}: cpu {:>8.3} ± {:>6.3} ms | gpu {:>8.3} ± {:>6.3} ms | {:>7.2}×",
                cpu.0,
                cpu.1,
                gpu.0,
                gpu.1,
                cpu.0 / gpu.0
            );
            rows.push(Row {
                workload,
                m,
                cpu,
                gpu,
            });
        }
    }

    // ── Markdown table (paste-ready for RESULTS.md), workload-major ────────
    rows.sort_by(|a, b| a.workload.cmp(b.workload).then(a.m.cmp(&b.m)));
    println!("\n| workload | queries | CPU ms (rayon) | GPU ms (device) | GPU speedup |");
    println!("|----------|---------|----------------|-----------------|-------------|");
    for r in &rows {
        println!(
            "| {:<8} | {:>7} | {:>14.3} | {:>15.3} | {:>10.2}× |",
            r.workload,
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
            "non-finite or zero mean in row {} @ {}",
            r.workload,
            r.m
        );
    }

    println!("\n✅ BVH BENCH ACCEPTANCE PASSED");
    Ok(())
}
