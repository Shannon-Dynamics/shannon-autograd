//! Tape machinery, tested on the CPU backend — no GPU anywhere (Day-6 plan
//! §5.4). The adjoints themselves are proven innocent by tests/adjoints.rs;
//! what is under test HERE is the tape: record order, the reverse walk, the
//! cross-op gradient hand-off through shared buffers, pause, and the
//! borrow-release contract of the consuming `backward`.
//!
//! Test-harness note: gradient buffers live in `RefCell<Vec<f32>>` cells so
//! `Fn` closures can reach `HostGradF32::new(&mut …)` per replay. That is a
//! TEST-LOCAL crutch — the real `Array` needs none of it, because GPU adjoint
//! kernels take grad buffers as `&Array` (const-ref scatter) and the Host
//! sinks exist to make CPU mirrors match.

use std::cell::RefCell;

use shannon_autodiff::{Tape, grad_fd};

const N: usize = 8;
const SCALE: f32 = 2.0;
const BIAS: f32 = 0.5;

/// The 3-op chain: y1 = affine(x) → y2 = sin(y1) → loss = Σ y2, taped via
/// the real shannon_cpu backend functions, then walked backward.
/// Returns (x_grad, loss).
fn run_chain(x: &[f32]) -> (Vec<f32>, f32) {
    // Forward values (mutation phase — computed before anything records).
    let mut y1 = vec![0.0f32; N];
    shannon_cpu::affine(x, SCALE, BIAS, &mut y1);
    let mut y2 = vec![0.0f32; N];
    shannon_cpu::sin_map(&y1, &mut y2);
    let mut loss_buf = vec![0.0f32; 1];
    shannon_cpu::sum_scalar(&y2, &mut loss_buf);

    // Grad cells, zeroed; the loss grad SEEDED BEFORE recording (§4.2).
    let gx = RefCell::new(vec![0.0f32; N]);
    let gy1 = RefCell::new(vec![0.0f32; N]);
    let gy2 = RefCell::new(vec![0.0f32; N]);
    let gloss = RefCell::new(vec![1.0f32]);

    // Taped phase: records in FORWARD order, replay must run in reverse —
    // sum fills gy2, sin consumes gy2 into gy1, affine consumes gy1 into gx.
    let mut tape = Tape::new();
    tape.record("affine", N, || {
        shannon_cpu::adj_affine(&gy1.borrow(), SCALE, &mut gx.borrow_mut());
        Ok(())
    });
    let y1_ref = &y1;
    tape.record("sin_map", N, || {
        shannon_cpu::adj_sin_map(y1_ref, &gy2.borrow(), &mut gy1.borrow_mut());
        Ok(())
    });
    tape.record("sum_scalar", N, || {
        shannon_cpu::adj_sum_scalar(&gloss.borrow(), &mut gy2.borrow_mut());
        Ok(())
    });
    assert_eq!(tape.len(), 3);
    tape.backward().unwrap();

    (gx.into_inner(), loss_buf[0])
}

/// The week-plan acceptance clause: the tape's gradient for a 3-kernel chain
/// matches finite differences of the COMPOSED function.
#[test]
fn chain_matches_finite_differences() {
    let x: Vec<f32> = (0..N).map(|i| 0.11 * i as f32 - 0.3).collect();
    let (gx, loss) = run_chain(&x);

    let f = |v: &[f32]| -> f32 { v.iter().map(|&a| (a * SCALE + BIAS).sin()).sum() };
    assert!(
        (loss - f(&x)).abs() < 1e-5,
        "forward chain disagrees: {loss} vs {}",
        f(&x)
    );

    let fd = grad_fd(&f, &x, 1e-3);
    for i in 0..N {
        // Analytic: d/dx sin(2x + 0.5) = 2·cos(2x + 0.5) — check both ways.
        let analytic = SCALE * (x[i] * SCALE + BIAS).cos();
        assert!(
            (gx[i] - analytic).abs() < 1e-5,
            "i={i}: tape {} vs analytic {analytic}",
            gx[i]
        );
        assert!(
            (gx[i] - fd[i]).abs() < 1e-3,
            "i={i}: tape {} vs fd {}",
            gx[i],
            fd[i]
        );
    }
}

/// Replay order is exactly reversed record order — "the ordering is the whole
/// point" (reference doc, Sample C).
#[test]
fn backward_runs_in_reverse() {
    let log = RefCell::new(Vec::<&'static str>::new());
    let mut tape = Tape::new();
    for label in ["first", "second", "third"] {
        tape.record(label, 1, || {
            log.borrow_mut().push(label);
            Ok(())
        });
    }
    assert_eq!(
        tape.labels().collect::<Vec<_>>(),
        ["first", "second", "third"]
    );
    tape.backward().unwrap();
    assert_eq!(log.into_inner(), ["third", "second", "first"]);
}

#[test]
fn pause_skips_recording() {
    let mut tape = Tape::new();
    tape.record("kept", 1, || Ok(()));
    tape.pause();
    assert!(tape.is_paused());
    tape.record("dropped", 1, || Ok(()));
    tape.resume();
    tape.record("kept_too", 1, || Ok(()));
    assert_eq!(tape.labels().collect::<Vec<_>>(), ["kept", "kept_too"]);
}

/// The §4.1 contract: `backward(self)` consumes the tape, so the borrows its
/// records held END — and the parameter buffer is mutable again. This test
/// exists primarily to COMPILE; the asserts are a formality.
#[test]
fn backward_consumes_and_releases_borrows() {
    let mut x = vec![1.0f32; 4];
    {
        let x_ref = &x; // the borrow a record would hold
        let mut tape = Tape::new();
        tape.record("op", 4, move || {
            let _ = x_ref[0];
            Ok(())
        });
        tape.backward().unwrap();
    } // ← borrow region ends with the consumed tape
    x[0] = 99.0; // must compile: &mut is legal again
    assert_eq!(x[0], 99.0);
}

/// Adjoints ACCUMULATE: two backwards into the same grads without zeroing
/// give exactly 2× — this is WHY zero_grad is law (pitfall 1). Exact
/// equality is safe here: single-threaded, identical order, both replays.
#[test]
fn grads_accumulate_across_backwards() {
    let x: Vec<f32> = (0..N).map(|i| 0.2 * i as f32).collect();
    let (g1, _) = run_chain(&x);

    // One harness, two tapes backward, no zero in between.
    let gx = RefCell::new(vec![0.0f32; N]);
    let gloss = RefCell::new(vec![1.0f32]);
    for _ in 0..2 {
        let mut tape = Tape::new();
        tape.record("affine", N, || {
            shannon_cpu::adj_affine(&gloss.borrow().repeat(N), SCALE, &mut gx.borrow_mut());
            Ok(())
        });
        tape.backward().unwrap();
    }
    let doubled = gx.into_inner();
    for g in &doubled {
        assert_eq!(*g, 2.0 * SCALE, "second backward must ADD, not overwrite");
    }
    drop(g1);
}

/// A failing record surfaces as an error naming the op, not a panic.
#[test]
fn backward_propagates_record_errors_with_label() {
    let mut tape = Tape::new();
    tape.record("exploder", 1, || Err(anyhow::anyhow!("boom")));
    let err = tape.backward().unwrap_err().to_string();
    assert!(
        err.contains("exploder"),
        "error must name the record: {err}"
    );
}
