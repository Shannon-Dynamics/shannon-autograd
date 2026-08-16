//! W0 — the Day-1 vertical slice.
//!
//! FORWARD    y[i] = a[i] · scale + bias          scale = 2.0, bias = 1.0
//! BACKWARD   ā[i] += ȳ[i] · scale                seed ȳ = 1  ⟹  ā[i] == 2.0 exactly
//!
//! Chosen because the correct answer is checkable by inspection.

use anyhow::Result;
use shannon_kernels::launch;
use shannon_rt::Array;

const N: usize = 1024;
const SCALE: f32 = 2.0;
const BIAS: f32 = 1.0;

fn main() -> Result<()> {
    // No explicit device setup, no module loading — Array and launch! resolve both.
    let a_host: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();

    // ---- FORWARD ----------------------------------------------------------
    let a = Array::from_slice(&a_host)?;
    let mut y = Array::<f32>::zeros(N)?;
    launch!(affine, dim = N, (&a, SCALE, BIAS, &mut y))?;
    let y_gpu = y.to_vec()?;

    let mut y_cpu = vec![0.0f32; N];
    shannon_cpu::affine(&a_host, SCALE, BIAS, &mut y_cpu);

    for i in 0..N {
        let expected = a_host[i] * SCALE + BIAS;
        assert!(
            (y_gpu[i] - expected).abs() <= 1e-6,
            "GPU forward mismatch at {i}"
        );
        assert!(
            (y_cpu[i] - expected).abs() <= 1e-6,
            "CPU forward mismatch at {i}"
        );
    }
    println!("✓ forward: GPU == CPU == analytic  ({N} elements)");

    // ---- BACKWARD ---------------------------------------------------------
    let adj_y = Array::from_slice(&vec![1.0f32; N])?; // seed ȳ = 1
    let adj_a = Array::<f32>::zeros(N)?; // zeroed: adjoints ACCUMULATE
    launch!(adj_affine, dim = N, (&adj_y, SCALE, &adj_a))?;
    let g_gpu = adj_a.to_vec()?;

    let mut g_cpu = vec![0.0f32; N];
    shannon_cpu::adj_affine(&vec![1.0f32; N], SCALE, &mut g_cpu);

    for i in 0..N {
        assert!(
            (g_gpu[i] - SCALE).abs() <= 1e-6,
            "GPU gradient mismatch at {i}"
        );
        assert!(
            (g_cpu[i] - SCALE).abs() <= 1e-6,
            "CPU gradient mismatch at {i}"
        );
    }
    println!("✓ backward: GPU == CPU == {SCALE} (analytic)");

    // ---- ORACLE -----------------------------------------------------------
    shannon_autodiff::gradcheck(|v| v[0] * SCALE + BIAS, &[3.0], &[SCALE], 1e-3, 1e-4)
        .map_err(|e| anyhow::anyhow!("gradcheck failed: {e:?}"))?;
    println!("✓ gradcheck: analytic gradient confirmed by finite differences");

    println!("\n✅ AFFINE ACCEPTANCE PASSED");
    Ok(())
}
