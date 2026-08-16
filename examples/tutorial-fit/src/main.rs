//! The USER's program — docs/TUTORIAL.md, steps 5–7. Fits the three
//! parameters of a damped sine wave to sampled data by gradient descent
//! through the GPU, using shannon-autograd strictly as an imported library.
//!
//!   part 1 — the forward kernel runs on both backends and they agree
//!   part 2 — the hand-written adjoint survives a finite-difference gradcheck
//!   part 3 — the taped GPU pipeline recovers the true parameters with Adam
//!
//! Every line of arithmetic lives in this package's `lib.rs`; this file only
//! moves buffers, records the tape, and asserts.

use anyhow::{Result, anyhow, ensure};
use shannon_autodiff::{Adam, Tape, gradcheck};
use shannon_rt::{Array, launch_in};
use tutorial_fit::{cpu, module};

/// Ground truth [A, d, ω] the optimizer must recover.
const TRUTH: [f32; 3] = [1.5, 0.35, 2.2];
/// Deliberately wrong starting point.
const INIT: [f32; 3] = [1.0, 0.2, 2.0];
const N: usize = 512;
const T_MAX: f32 = 4.0;
const ITERS: usize = 400;
const LR: f32 = 0.05;

fn sample_times() -> Vec<f32> {
    (0..N).map(|i| i as f32 / (N - 1) as f32 * T_MAX).collect()
}

fn max_abs(g: &[f32]) -> f32 {
    g.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

/// Zero every gradient, zero the loss VALUE, seed ∂loss/∂loss = 1 — all
/// before the first tape record. Reductions accumulate through GradSink
/// atomics, so the taped phase never takes `&mut loss`; by backward time
/// there is no way to write the seed. Plant it first.
fn begin_iteration(f32_grads: &mut [&mut Array<f32>], loss: &mut Array<f32>) -> Result<()> {
    for a in f32_grads.iter_mut() {
        a.zero_grad()?;
    }
    loss.zero_grad()?;
    loss.copy_from_slice(&[0.0])?; // forward loss accumulates — reset the VALUE
    loss.grad_mut()
        .ok_or_else(|| anyhow!("loss has no grad — call requires_grad_(true) first"))?
        .copy_from_slice(&[1.0])
}

// ─────────────────────────────────────────────────────────────────────────────
// The two ops — wrap each kernel pair (forward + adjoint) so the forward
// launch and the recorded backward launch stay in one place. The signature
// move is the DOWNGRADE: take the output `&mut`, launch, rebind it shared —
// the record's shared borrow then freezes `y` until `backward` runs.
// ─────────────────────────────────────────────────────────────────────────────

fn damped_wave_op<'a>(
    tape: &mut Tape<'a>,
    t: &'a Array<f32>,
    params: &'a Array<f32>,
    y: &'a mut Array<f32>,
) -> Result<&'a Array<f32>> {
    let n = t.len();
    launch_in!(module, damped_wave, dim = n, (t, params, &mut *y))?;
    let y_sh: &'a Array<f32> = y; // ← the downgrade: y is frozen until backward
    tape.record("damped_wave", n, move || {
        let gy = y_sh
            .grad()
            .ok_or_else(|| anyhow!("damped_wave: y has no grad"))?;
        let gp = params
            .grad()
            .ok_or_else(|| anyhow!("damped_wave: params has no grad"))?;
        launch_in!(module, adj_damped_wave, dim = n, (t, params, gy, gp))
    });
    Ok(y_sh)
}

fn wave_loss_op<'a>(
    tape: &mut Tape<'a>,
    y: &'a Array<f32>,
    target: &'a Array<f32>,
    loss: &'a Array<f32>,
) -> Result<()> {
    let n = y.len();
    launch_in!(module, wave_loss, dim = n, (y, target, loss))?;
    tape.record("wave_loss", n, move || {
        let gl = loss
            .grad()
            .ok_or_else(|| anyhow!("wave_loss: loss has no grad"))?;
        let gy = y
            .grad()
            .ok_or_else(|| anyhow!("wave_loss: y has no grad"))?;
        launch_in!(module, adj_wave_loss, dim = n, (y, target, gl, gy))
    });
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Part 1 — one kernel, two backends, and they agree
// ─────────────────────────────────────────────────────────────────────────────

fn part1_forward_parity(t: &[f32], target: &[f32]) -> Result<()> {
    // GPU
    let t_arr = Array::from_slice(t)?;
    let p_arr = Array::from_slice(&TRUTH)?;
    let mut y_arr = Array::<f32>::zeros(N)?;
    launch_in!(module, damped_wave, dim = N, (&t_arr, &p_arr, &mut y_arr))?;
    let y_gpu = y_arr.to_vec()?;

    // CPU — the same body, expanded from the same kernel row, under rayon
    let mut y_cpu = vec![0.0f32; N];
    cpu::damped_wave(t, &TRUTH, &mut y_cpu);

    // Direct body call — the arithmetic itself, no backend at all
    for i in 0..N {
        let direct = tutorial_fit::damped_wave_at(i, t, &TRUTH);
        ensure!(
            (y_gpu[i] - direct).abs() <= 1e-5 && (y_cpu[i] - direct).abs() <= 1e-6,
            "forward parity broke at sample {i}: gpu {} cpu {} direct {direct}",
            y_gpu[i],
            y_cpu[i],
        );
    }
    // `target` is the CPU forward at TRUTH — confirm we produced what part 3 chases
    ensure!(
        y_cpu == target,
        "target buffer drifted from the CPU forward"
    );
    println!("✓ forward: GPU == CPU == direct body call  ({N} samples)");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Part 2 — the adjoint vs finite differences (host backend)
// ─────────────────────────────────────────────────────────────────────────────

fn full_loss_cpu(t: &[f32], target: &[f32], params: &[f32]) -> f32 {
    let mut y = vec![0.0f32; N];
    cpu::damped_wave(t, params, &mut y);
    let mut loss = [0.0f32];
    cpu::wave_loss(&y, target, &mut loss);
    loss[0]
}

fn part2_gradcheck(t: &[f32], target: &[f32]) -> Result<()> {
    // Analytic ∇: run the same two adjoints the tape will replay, seeded with
    // ∂loss/∂loss = 1, in reverse order — loss adjoint first, then the wave's.
    let mut y = vec![0.0f32; N];
    cpu::damped_wave(t, &INIT, &mut y);
    let mut adj_y = vec![0.0f32; N];
    cpu::adj_wave_loss(&y, target, &[1.0], &mut adj_y);
    let mut adj_params = [0.0f32; 3];
    cpu::adj_damped_wave(t, &INIT, &adj_y, &mut adj_params);

    // tol 5e-3, arrived at by the gradcheck discipline (sweep eps BEFORE
    // suspecting the adjoint). The d-component is ill-conditioned for f32
    // central differences: measured error 3.9e-3 at eps 1e-3 (cancellation —
    // the loss is a 512-term sum) and 6.6e-2 at eps 1e-2 (truncation — d's
    // third derivative carries factors of t³). Fitting err = a·eps² + b/eps
    // to those two points puts the best achievable FD accuracy at ~3.5e-3
    // near eps 1.3e-3 — no eps passes tol 1e-4. An f64 reference settles who
    // is right: analytic ∂loss/∂d = −0.7881650…, f64 finite difference
    // −0.7881650… — the adjoint is exact; the f32 probe is the noise.
    gradcheck(
        |p| full_loss_cpu(t, target, p),
        &INIT,
        &adj_params,
        1e-3,
        5e-3,
    )
    .map_err(|e| anyhow!("gradcheck failed: {e:?}"))?;
    println!("✓ gradcheck: analytic ∇[A, d, ω] confirmed by finite differences");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Part 3 — tape it, and let Adam recover the truth
// ─────────────────────────────────────────────────────────────────────────────

fn part3_fit(t: &[f32], target: &[f32]) -> Result<()> {
    let t_arr = Array::from_slice(t)?;
    let target_arr = Array::from_slice(target)?;
    let mut params = Array::from_slice(&INIT)?;
    let mut y = Array::<f32>::zeros(N)?;
    let mut loss = Array::<f32>::zeros(1)?;
    params.requires_grad_(true)?;
    y.requires_grad_(true)?;
    loss.requires_grad_(true)?;

    let mut host_params = INIT.to_vec();
    let mut adam = Adam::new(3, LR);
    let mut loss0 = 0.0f32;
    let mut last = f32::INFINITY;

    for iter in 0..ITERS {
        // MUTATE — upload params; zero grads, reset + seed the loss.
        params.copy_from_slice(&host_params)?;
        begin_iteration(&mut [&mut params, &mut y], &mut loss)?;

        // TAPE — forward launches recorded; everything borrowed shared.
        let mut tape = Tape::new();
        let y_sh = damped_wave_op(&mut tape, &t_arr, &params, &mut y)?;
        wave_loss_op(&mut tape, y_sh, &target_arr, &loss)?;
        tape.backward()?; // replays adj_wave_loss then adj_damped_wave, consumes the tape

        // STEP — host optimizer over the 3-element downloaded gradient.
        let l = loss.to_vec()?[0];
        let g = params
            .grad()
            .ok_or_else(|| anyhow!("params has no grad"))?
            .to_vec()?;

        if iter == 0 {
            loss0 = l;
            // Iteration-0 tripwire: INIT ≠ TRUTH, so the true gradient is
            // nonzero — an all-zero gradient here is ALWAYS a seeding bug,
            // and a flat loss would pass a decrease check for 400 iterations.
            ensure!(
                max_abs(&g) > 0.0,
                "iteration-0 tripwire: all gradients zero — seed/plumbing bug"
            );
            // Cross-backend check: the GPU gradient must match the host
            // backend's recomputation of the same adjoints (rel 1e-3 — float
            // atomics commit in nondeterministic order, so never exact).
            let mut y_c = vec![0.0f32; N];
            cpu::damped_wave(t, &host_params, &mut y_c);
            let mut adj_y = vec![0.0f32; N];
            cpu::adj_wave_loss(&y_c, target, &[1.0], &mut adj_y);
            let mut g_cpu = [0.0f32; 3];
            cpu::adj_damped_wave(t, &host_params, &adj_y, &mut g_cpu);
            for k in 0..3 {
                let denom = g_cpu[k].abs().max(1e-6);
                ensure!(
                    (g[k] - g_cpu[k]).abs() / denom <= 1e-3,
                    "GPU/CPU gradient mismatch at param {k}: {} vs {}",
                    g[k],
                    g_cpu[k]
                );
            }
            println!("  ✓ iter 0: GPU gradient matches CPU backend (rel 1e-3)");
        }
        last = l;

        adam.step(&mut host_params, &g);

        if iter % 50 == 0 || iter + 1 == ITERS {
            println!(
                "  iter {iter:3}  loss {l:.6e}  params [{:.4}, {:.4}, {:.4}]",
                host_params[0], host_params[1], host_params[2]
            );
        }
    }

    ensure!(
        last <= 1e-4 * loss0,
        "fit: final loss {last} not ≥4 orders below initial {loss0}"
    );
    for k in 0..3 {
        let rel = (host_params[k] - TRUTH[k]).abs() / TRUTH[k].abs();
        ensure!(
            rel <= 0.01,
            "param {k} not recovered: {} vs truth {} (rel {rel:.4})",
            host_params[k],
            TRUTH[k]
        );
    }
    println!(
        "✓ fit: {loss0:.3e} → {last:.3e} ({:.1} orders); [A, d, ω] recovered within 1 %",
        (loss0 / last).log10()
    );
    Ok(())
}

fn main() -> Result<()> {
    let t = sample_times();
    // The observations: the CPU backend evaluated at TRUTH. Part 3 must
    // recover TRUTH from these samples alone.
    let mut target = vec![0.0f32; N];
    cpu::damped_wave(&t, &TRUTH, &mut target);

    println!("— Part 1: one kernel, two backends —");
    part1_forward_parity(&t, &target)?;

    println!("— Part 2: the adjoint vs finite differences —");
    part2_gradcheck(&t, &target)?;

    println!("— Part 3: taped GPU fit, Adam (lr {LR}) —");
    part3_fit(&t, &target)?;

    println!("\n✅ TUTORIAL FIT ACCEPTANCE PASSED");
    Ok(())
}
