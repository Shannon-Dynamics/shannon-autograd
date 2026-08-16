//! W1 — the SDF ray marcher. The "SDK works" milestone.
//!
//! Renders the reference scene on the GPU, writes a PPM, and proves the
//! identical source produces a matching image on the CPU at 128×64.

use anyhow::Result;
use shannon_core::{Quat, Vec3};
use shannon_kernels::launch;
use shannon_rt::Array;
use std::time::Instant;

const CAM_POS: Vec3 = Vec3::new(-1.25, 1.0, 2.0);

/// Write a binary PPM (P6). Rows are emitted BOTTOM-TO-TOP.
///
/// The reference renders with matplotlib's `origin="lower"`, i.e. buffer row 0
/// is the BOTTOM of the image. PPM defines row 0 as the TOP scanline. Emit
/// rows in reverse or the render comes out vertically flipped — ground plane
/// in the sky.
fn write_ppm(path: &str, pixels: &[Vec3], width: u32, height: u32) -> Result<()> {
    let mut out = Vec::with_capacity(20 + (width * height * 3) as usize);
    out.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    for y in (0..height).rev() {
        for x in 0..width {
            let p = pixels[(y * width + x) as usize];
            out.push((p.x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            out.push((p.y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            out.push((p.z.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn parse_args() -> (u32, u32) {
    let (mut width, mut height) = (2048u32, 1024u32);
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--width" if i + 1 < args.len() => {
                width = args[i + 1].parse().expect("--width takes an integer");
                i += 2;
            }
            "--height" if i + 1 < args.len() => {
                height = args[i + 1].parse().expect("--height takes an integer");
                i += 2;
            }
            other => panic!("unknown argument {other}; supported: --width N --height N"),
        }
    }
    (width, height)
}

fn main() -> Result<()> {
    let (width, height) = parse_args();
    let cam_rot = Quat::from_rpy(-0.5, -0.5, 0.0);
    let n = (width * height) as usize;

    // ---- GPU render at full resolution ------------------------------------
    let mut pixels = Array::<Vec3>::zeros(n)?;
    let t0 = Instant::now();
    launch!(
        draw,
        dim = n,
        (CAM_POS, cam_rot, width, height, &mut pixels)
    )?;
    let gpu = pixels.to_vec()?; // implicitly synchronizes
    println!("GPU render {width}×{height}: {:?}", t0.elapsed());

    write_ppm("raymarch.ppm", &gpu, width, height)?;
    println!("✓ wrote raymarch.ppm");

    // ---- CPU parity at low resolution (full res would stall the day) ------
    const PW: u32 = 128;
    const PH: u32 = 64;
    let pn = (PW * PH) as usize;
    let mut gpu_small = Array::<Vec3>::zeros(pn)?;
    launch!(draw, dim = pn, (CAM_POS, cam_rot, PW, PH, &mut gpu_small))?;
    let a = gpu_small.to_vec()?;

    let t1 = Instant::now();
    let mut b = vec![Vec3::ZERO; pn];
    shannon_cpu::draw(CAM_POS, cam_rot, PW, PH, &mut b);
    println!("CPU render {PW}×{PH}: {:?}", t1.elapsed());

    // Robust parity criterion. A flat 1e-3 on every pixel is unachievable
    // across two float pipelines: GPU FMA contraction (--fmad on, as nvcc)
    // plus libdevice powf ULP differences get amplified by pow(·, 80) in the
    // specular term and 128 marched steps. Measured on Day 2: 8182/8192
    // pixels agree < 1e-4; 10 pixels land near 1e-2 at shading boundaries.
    // Real convention errors (flipped camera, wrong gamma) shift MANY pixels
    // by O(0.1) — caught by both thresholds below.
    let mut worst = 0.0f32;
    let mut over_tol = 0usize;
    for i in 0..pn {
        let d = a[i] - b[i];
        let m = d.x.abs().max(d.y.abs()).max(d.z.abs());
        worst = worst.max(m);
        if m > 1e-3 {
            over_tol += 1;
        }
    }
    let over_frac = over_tol as f32 / pn as f32;
    assert!(
        over_frac <= 0.005,
        "CPU/GPU divergence: {over_tol}/{pn} pixels exceed 1e-3 (> 0.5%)"
    );
    assert!(
        worst <= 5e-2,
        "CPU/GPU divergence: worst channel delta {worst}"
    );
    println!(
        "✓ CPU == GPU at {PW}×{PH} ({:.2}% of pixels >1e-3, worst {worst:.2e})",
        over_frac * 100.0
    );

    // Sanity: both branches of the shader are exercised at parity resolution.
    let sky = Vec3::new(0.4, 0.45, 0.5) * 1.5;
    let skies = a.iter().filter(|c| (**c - sky).length() <= 1e-6).count();
    assert!(
        skies > 0 && skies < pn,
        "expected both sky and geometry; sky = {skies}/{pn}"
    );
    println!("✓ scene shows geometry and sky ({skies}/{pn} sky pixels)");

    println!("\n✅ RAYMARCH ACCEPTANCE PASSED");
    Ok(())
}
