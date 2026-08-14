//! W4 — launch-overhead benchmark. Ports Warp's `benchmark_launches.py`.
//!
//! Methodology (Day-3 plan §3): wall-clock around the ASYNC enqueue loop,
//! synchronization OUTSIDE the timed region, 100-launch warm-up per cell,
//! median of `--reps`. The number blends enqueue rate and steady-state
//! throughput once the stream queue fills — the same blend Warp reports,
//! comparable by construction.
//!
//! CPU column: a direct function call IS the CPU dispatch path; reported in
//! ns/call (no queue exists to imply). #[inline(never)] callees + black_box
//! on args and call keep the optimizer honest (§4.C).

use anyhow::Result;
use shannon_core::{Mat33, Vec3};
use shannon_kernels::bench::{BenchS0, BenchSa, BenchSf, BenchSm, BenchSv, BenchSz};
use shannon_rt::{Array, Device, GpuTimer, ScopedTimer, launch};
use std::hint::black_box;
use std::time::Instant;

fn parse_args() -> (u32, u32) {
    let (mut launches, mut reps) = (20_000u32, 3u32);
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--launches" => {
                launches = argv[i + 1].parse().expect("--launches N");
                i += 2;
            }
            "--reps" => {
                reps = argv[i + 1].parse().expect("--reps N");
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
    (launches, reps)
}

/// Time one GPU cell: warm-up, sync, timed enqueue loop, sync outside.
/// Returns µs/launch.
fn time_gpu<F: FnMut() -> Result<()>>(launches: u32, dev: &Device, mut one: F) -> Result<f64> {
    for _ in 0..100 {
        one()?;
    }
    dev.synchronize()?;
    let t = ScopedTimer::quiet("cell");
    for _ in 0..launches {
        one()?;
    }
    let elapsed_ms = t.elapsed_ms();
    dev.synchronize()?; // drain — deliberately OUTSIDE the measurement
    Ok(elapsed_ms * 1e3 / launches as f64)
}

/// Time one CPU cell. Returns ns/call.
fn time_cpu<F: FnMut()>(calls: u32, mut one: F) -> f64 {
    for _ in 0..100 {
        one();
    }
    let start = Instant::now();
    for _ in 0..calls {
        one();
    }
    start.elapsed().as_secs_f64() * 1e9 / calls as f64
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn median_of<F: FnMut() -> Result<f64>>(reps: u32, mut f: F) -> Result<f64> {
    let mut xs = Vec::with_capacity(reps as usize);
    for _ in 0..reps {
        xs.push(f()?);
    }
    Ok(median(xs))
}

struct Row {
    shape: &'static str,
    cpu_ns: f64,
    gpu_direct_us: f64,
    gpu_struct_us: f64,
}

// `black_box(unit_returning_call())` is the deliberate §4.C elision guard —
// clippy's unit_arg lint fires on exactly that shape.
#[allow(clippy::unit_arg)]
fn main() -> Result<()> {
    let (launches, reps) = parse_args();
    let dev = Device::default()?;

    // ── Argument values (mirroring Warp's) ─────────────────────────────────
    let (x, y, z) = (17.0f32, 42.0f32, 99.0f32);
    let u = Vec3::new(1.0, 2.0, 3.0);
    let v = Vec3::new(10.0, 20.0, 30.0);
    let w = Vec3::new(100.0, 200.0, 300.0);
    let m = Mat33::IDENTITY;

    // Device arrays for ka/kz — and the device pointers for Sa/Sz. Held alive
    // for the whole run so the pointers stay valid.
    let a = Array::<f32>::zeros(1)?;
    let b = Array::<f32>::zeros(1)?;
    let c = Array::<f32>::zeros(1)?;
    // The one sanctioned use of __buf outside launch!: extracting DEVICE
    // pointers for the struct payloads (Day-3 plan §4.A / §5.6).
    let (pa, pb, pc) = (
        a.__buf().cu_deviceptr() as *const f32,
        b.__buf().cu_deviceptr() as *const f32,
        c.__buf().cu_deviceptr() as *const f32,
    );

    let s0 = BenchS0 {};
    let sf = BenchSf { x, y, z };
    let sv = BenchSv { u, v, w };
    let sm = BenchSm { m, n: m, o: m };
    let sa = BenchSa { a: pa, a_len: 1, b: pb, b_len: 1, c: pc, c_len: 1 };
    let sz = BenchSz {
        a: pa, a_len: 1, b: pb, b_len: 1, c: pc, c_len: 1,
        x, y, z, u, v, w,
    };

    println!(
        "Launch-overhead benchmark — {launches} launches/cell, median of {reps} reps\n"
    );

    // ── The six shapes ─────────────────────────────────────────────────────
    // Each closure is one launch through the REAL user-facing path.
    let mut rows: Vec<Row> = Vec::new();

    macro_rules! cell {
        ($shape:literal, cpu: $cpu:expr, direct: $direct:expr, strct: $strct:expr) => {{
            let cpu_ns = time_cpu(launches, $cpu);
            let gpu_direct_us = median_of(reps, || time_gpu(launches, dev, $direct))?;
            let gpu_struct_us = median_of(reps, || time_gpu(launches, dev, $strct))?;
            println!(
                "  {:<4} cpu {:>8.1} ns   gpu direct {:>7.2} µs   gpu struct {:>7.2} µs",
                $shape, cpu_ns, gpu_direct_us, gpu_struct_us
            );
            rows.push(Row { shape: $shape, cpu_ns, gpu_direct_us, gpu_struct_us });
        }};
    }

    cell!("k0",
        cpu: || black_box(shannon_cpu::bench_k0()),
        direct: || launch!(bench_k0, dim = 1, ()),
        strct:  || launch!(bench_s0, dim = 1, (black_box(s0))));

    cell!("kf",
        cpu: || black_box(shannon_cpu::bench_kf(black_box(x), black_box(y), black_box(z))),
        direct: || launch!(bench_kf, dim = 1, (x, y, z)),
        strct:  || launch!(bench_sf, dim = 1, (sf)));

    cell!("kv",
        cpu: || black_box(shannon_cpu::bench_kv(black_box(u), black_box(v), black_box(w))),
        direct: || launch!(bench_kv, dim = 1, (u, v, w)),
        strct:  || launch!(bench_sv, dim = 1, (sv)));

    cell!("km",
        cpu: || black_box(shannon_cpu::bench_km(black_box(m), black_box(m), black_box(m))),
        direct: || launch!(bench_km, dim = 1, (m, m, m)),
        strct:  || launch!(bench_sm, dim = 1, (sm)));

    cell!("ka",
        cpu: || {
            let (ha, hb, hc) = ([0.0f32], [0.0f32], [0.0f32]);
            black_box(shannon_cpu::bench_ka(black_box(&ha), black_box(&hb), black_box(&hc)))
        },
        direct: || launch!(bench_ka, dim = 1, (&a, &b, &c)),
        strct:  || launch!(bench_sa, dim = 1, (sa)));

    cell!("kz",
        cpu: || {
            let (ha, hb, hc) = ([0.0f32], [0.0f32], [0.0f32]);
            black_box(shannon_cpu::bench_kz(
                black_box(&ha), black_box(&hb), black_box(&hc),
                black_box(x), black_box(y), black_box(z),
                black_box(u), black_box(v), black_box(w)))
        },
        direct: || launch!(bench_kz, dim = 1, (&a, &b, &c, x, y, z, u, v, w)),
        strct:  || launch!(bench_sz, dim = 1, (sz)));

    // ── Garnish: device-side execution time of the empty kernel ───────────
    // Shows the enqueue/execute split once (Day-3 plan §3.2).
    let gpu_t = GpuTimer::new(dev)?;
    dev.synchronize()?;
    gpu_t.start(dev)?;
    for _ in 0..1000 {
        launch!(bench_k0, dim = 1, ())?;
    }
    gpu_t.stop(dev)?;
    let device_us = gpu_t.elapsed_ms()? as f64 * 1e3 / 1000.0;

    // ── Markdown table (paste-ready for RESULTS.md) ────────────────────────
    println!("\n| shape | CPU ns/call | GPU direct µs | GPU struct µs |");
    println!("|-------|-------------|---------------|---------------|");
    for r in &rows {
        println!(
            "| {:<5} | {:>11.1} | {:>13.2} | {:>13.2} |",
            r.shape, r.cpu_ns, r.gpu_direct_us, r.gpu_struct_us
        );
    }
    println!("\nbench_k0 device-side execution (CudaEvent, 1000 launches): {device_us:.2} µs/kernel");

    // ── Sanity assertions — bounds only, never performance gates ───────────
    for r in &rows {
        assert!(
            r.cpu_ns.is_finite() && r.cpu_ns > 0.0
                && r.gpu_direct_us.is_finite() && r.gpu_direct_us > 0.0
                && r.gpu_struct_us.is_finite() && r.gpu_struct_us > 0.0,
            "non-finite or zero median in row {}",
            r.shape
        );
    }
    let k0 = &rows[0];
    assert!(
        k0.gpu_direct_us < 1000.0,
        "k0 at {:.1} µs/launch — a synchronize leaked into the timed loop?",
        k0.gpu_direct_us
    );

    println!("\n✅ LAUNCH BENCH ACCEPTANCE PASSED");
    Ok(())
}
