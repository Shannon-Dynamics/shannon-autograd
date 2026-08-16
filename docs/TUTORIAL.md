# Tutorial — your own kernel, in your own crate, made differentiable

This walkthrough builds a complete user project from nothing: **your own
crate** that imports shannon-autograd as a library, declares its own GPU
kernel with its CPU twin, hand-writes and gradchecks the adjoint, and runs a
taped optimization loop that recovers physical parameters by gradient descent.
**No SDK file is edited at any point** — the SDK crates are dependencies, your
kernels live in your directory, and your binary embeds its own PTX bundle
beside any others.

**What you'll build:** `tutorial-fit`, which fits the three parameters of a
damped sine wave

```text
y(t) = A · exp(−d·t) · sin(ω·t),      params = [A, d, ω]
```

to sampled data, on the GPU, by reverse-mode autodiff. Every snippet below is
real code from the finished project, and the final output is pasted from an
actual run.

**Prerequisites:** the [README quickstart](../README.md#quickstart) runs to
its banner. **The finished project**, if you'd rather read along than type:

| File | What lands there |
|---|---|
| [`examples/tutorial-fit/Cargo.toml`](../examples/tutorial-fit/Cargo.toml) | step 0 — the project scaffold |
| [`examples/tutorial-fit/src/lib.rs`](../examples/tutorial-fit/src/lib.rs) | steps 1–4 — math bodies, GPU kernels, CPU mirrors |
| [`examples/tutorial-fit/src/main.rs`](../examples/tutorial-fit/src/main.rs) | steps 5–7 — the program |

Run it any time with:

```bash
cd examples/tutorial-fit && cargo oxide run
```

---

## 0. Scaffold the project

A user project is an ordinary cargo package. Three things distinguish it:

```toml
[workspace]          # standalone — not a member of the SDK workspace

[package]
name = "tutorial-fit"    # ⚠️ this name is the embedded-PTX bundle lookup key
version = "0.1.0"
edition = "2024"

[dependencies]
# The SDK. A published release would use versions ("0.1"); until then, paths.
shannon-core     = { path = "../../shannon-core" }
shannon-device   = { path = "../../shannon-device" }
shannon-rt       = { path = "../../shannon-rt" }
shannon-autodiff = { path = "../../shannon-autodiff" }

# ⚠️ Required as DIRECT dependencies, not re-exported through the SDK:
# `#[cuda_module]` (inside `shannon_gpu_kernels!`) emits absolute
# `::cuda_core::…` paths, so these must resolve at your crate's root.
cuda-device = { path = "../../../cuda-oxide/crates/cuda-device" }
cuda-core   = { path = "../../../cuda-oxide/crates/cuda-core" }
cuda-host   = { path = "../../../cuda-oxide/crates/cuda-host" }

rayon  = "1.10"      # `shannon_cpu_kernels!` expands to rayon loops
anyhow = "1"
```

Copy the SDK's `rust-toolchain.toml` next to it — consumers must build with
the same pinned nightly. `cargo oxide run` works from the project directory:
cargo-oxide treats any directory with a `Cargo.toml` as a standalone project
and device-compiles **every** crate in the build by default, so your kernels
compile to PTX with zero extra configuration. (Fully out-of-tree projects
also need the codegen backend discoverable — the `~/.cargo/cuda-oxide/` cache
that `cargo oxide doctor` validates, or a `.cargo/cuda-oxide.toml`.)

## 1. Write the math — one plain function, in your crate

A kernel body is an ordinary function with one contract: **it computes element
`i`**, with signature `fn(i: usize, params…) -> Ret`.

```rust
// examples/tutorial-fit/src/lib.rs
use shannon_core::math;

/// FORWARD body (map shape): `y[i] = A·exp(−d·tᵢ)·sin(ω·tᵢ)`.
pub fn damped_wave_at(i: usize, t: &[f32], params: &[f32]) -> f32 {
    let (a, d, w) = (params[0], params[1], params[2]);
    a * math::exp(-d * t[i]) * math::sin(w * t[i])
}
```

Note what is *absent*: no CUDA, no thread indexing, no output buffer. The
function is testable on the host with `cargo test` before a GPU ever sees it.
(`shannon_core::math` is `libm` on both backends, so the two sides agree to
float precision.)

## 2. Declare the GPU kernel — one row, plus your module accessor

`shannon_core::shannon_gpu_kernels!` works in **any** crate that has the three
`cuda-*` dependencies. A map-shaped kernel is one declaration line, pointing
at your body:

```rust
// examples/tutorial-fit/src/lib.rs
shannon_core::shannon_gpu_kernels! {
    /// FORWARD: y[i] = A·exp(−d·tᵢ)·sin(ω·tᵢ) — race-free via DisjointSlice.
    damped_wave(t: &[f32], params: &[f32]) -> f32 = crate::damped_wave_at;

    @raw { /* the adjoints — step 4 */ }
}

// This crate's cached PTX-module accessor: `pub fn module(&Device) -> …`.
shannon_rt::define_module_cache!();
```

The macro expands the row to a real `#[kernel]` that computes its thread
index, bounds-checks it, calls your body, and writes the return through
`DisjointSlice<f32>` — each thread may write only its own index, so the
forward pass is **race-free by construction** (the output-type rule,
[SAFETY.md](../SAFETY.md) §2).

`define_module_cache!()` generates `pub fn module(...)`: a `OnceLock`-cached
handle to *this crate's* embedded PTX, looked up by your package name. That
is how several kernel crates — yours and the SDK's demo crate — coexist in
one binary, each with its own bundle.

## 3. Declare the CPU twin — the same row

```rust
// examples/tutorial-fit/src/lib.rs
pub mod cpu {
    shannon_core::shannon_cpu_kernels! {
        /// FORWARD — same shared body, under rayon.
        damped_wave(t: &[f32], params: &[f32]) -> f32 = crate::damped_wave_at;
    }
    // …hand-written CPU adjoints — step 4
}
```

This expands to `cpu::damped_wave(t, params, out: &mut [f32])` running the
identical arithmetic under rayon. Nothing was ported; there is nothing to
drift. The program's part 1 holds the backends to that claim — GPU launch,
CPU call, and direct body calls, compared elementwise:

```rust
launch_in!(module, damped_wave, dim = N, (&t_arr, &p_arr, &mut y_arr))?;  // GPU
cpu::damped_wave(t, &TRUTH, &mut y_cpu);                                  // CPU
// ✓ forward: GPU == CPU == direct body call  (512 samples)
```

`shannon_rt::launch_in!` is the whole host-side ceremony: first argument is
*your* module accessor, then the kernel name, `dim`, and the arguments —
`Array<f32>` buffers go up, `dim` threads run, `to_vec()` brings results back.

## 4. Write the adjoint — where autodiff actually lives

Reverse-mode AD needs the derivative of your kernel. For
`y = A·e^(−d·t)·sin(ωt)`, with `e = exp(−d·t)`, `s = sin(ωt)`, `c = cos(ωt)`:

| Parameter | ∂y/∂p | Backward contribution (given ȳᵢ) |
|---|---|---|
| `A` | `e·s` | `Ā += ȳᵢ · e·s` |
| `d` | `−t·A·e·s` | `d̄ += ȳᵢ · (−tᵢ)·A·e·s` |
| `ω` | `t·A·e·c` | `ω̄ += ȳᵢ · tᵢ·A·e·c` |

Here the shape changes, and this is the tutorial's most important idea: the
forward pass was a *map* (thread `i` owns output `i`), but the backward pass
is a *scatter* — **all 512 threads accumulate into the same three parameter
cells**. A `DisjointSlice` cannot express that; a `GradSink` (hardware atomic
add on GPU, CAS loop on host) is built for it. The body is generic over the
sink, so one function serves both backends:

```rust
// examples/tutorial-fit/src/lib.rs — the body, generic over the sink
pub fn adj_damped_wave_at<S: GradSink<f32>>(
    i: usize, t: &[f32], params: &[f32], adj_y: &[f32], adj_params: &S,
) {
    let (a, d, w) = (params[0], params[1], params[2]);
    let ti = t[i];
    let e = math::exp(-d * ti);
    let s = math::sin(w * ti);
    let c = math::cos(w * ti);
    let g = adj_y[i];
    adj_params.accumulate(0, g * e * s);
    adj_params.accumulate(1, g * (-ti) * a * e * s);
    adj_params.accumulate(2, g * ti * a * e * c);
}
```

Because scatter does not fit the map row, the GPU side goes in the block's
`@raw { … }` section as a hand-written kernel — index, guard, call the body
with the *device* sink:

```rust
// inside the shannon_gpu_kernels! block's @raw section
#[kernel]
pub fn adj_damped_wave(t: &[f32], params: &[f32], adj_y: &[f32], adj_params: &[f32]) {
    let i = thread::index_1d().get();
    if i >= t.len() {
        return; // manual guard — no DisjointSlice to do it for us
    }
    crate::adj_damped_wave_at(i, t, params, adj_y, &DeviceGradF32(adj_params));
}
```

and the CPU mirror calls the same body with the *host* sink
(`shannon_rt::HostGradF32::new(adj_params)`). The loss needs the same
treatment — a forward reduction `wave_loss` accumulating `½(yᵢ−targetᵢ)²`
into `loss[0]`, and its adjoint `adj_wave_loss` broadcasting
`ȳᵢ += (yᵢ−targetᵢ)·loss̄` — four short functions, all in your `lib.rs`.

## 5. Gradcheck it — before believing it

A hand-written adjoint is guilty until finite differences acquit it. The
program composes the CPU pipeline into a scalar function of the parameters
and compares against the analytic gradient:

```rust
gradcheck(|p| full_loss_cpu(t, target, p), &INIT, &adj_params, 1e-3, 5e-3)?;
```

Those tolerances are themselves a lesson the SDK insists on: at `eps = 1e-3`
the check misses by 3.9e-3 on the damping parameter, and at `eps = 1e-2` by
6.6e-2. The discipline is **sweep eps before suspecting the adjoint** — here
no eps passes 1e-4, because the `d`-component's best achievable f32
central-difference accuracy is ~3.5e-3 (a 512-term sum eats small probes by
cancellation; `d`'s third derivative carries `t³` and eats large ones by
truncation). An f64 reference settles it: analytic `−0.7881650…`, f64 finite
difference `−0.7881650…` — the adjoint is exact and the f32 probe is the
noise, so the tolerance says 5e-3 and the comment in the code says why. Never
loosen a tolerance without being able to write that comment.

## 6. Tape it — the three-phase iteration

Wrap each kernel pair in an *op*: launch the forward, then record a closure
that will launch the adjoint. The signature move is the **downgrade** — take
the output `&mut`, launch, then rebind it shared:

```rust
// examples/tutorial-fit/src/main.rs
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
```

The record holds a shared borrow of `y`, so the borrow checker statically
rejects any later `&mut y` — nothing can overwrite a taped buffer before its
adjoint replays. `tape.backward()` walks the records in reverse and *consumes
the tape*, releasing the borrows; that is what makes the optimizer's mutation
legal again. Each iteration is then three phases, in an order that is not
negotiable:

```rust
for iter in 0..ITERS {
    // MUTATE — upload params; zero grads, reset + seed the loss.
    params.copy_from_slice(&host_params)?;
    begin_iteration(&mut [&mut params, &mut y], &mut loss)?;

    // TAPE — forward launches recorded; everything borrowed shared.
    let mut tape = Tape::new();
    let y_sh = damped_wave_op(&mut tape, &t_arr, &params, &mut y)?;
    wave_loss_op(&mut tape, y_sh, &target_arr, &loss)?;
    tape.backward()?;

    // STEP — host optimizer over the 3-element downloaded gradient.
    let g = params.grad().unwrap().to_vec()?;
    adam.step(&mut host_params, &g);
}
```

`begin_iteration` is ~10 lines you write yourself (see `main.rs`): zero every
grad, zero the loss *value*, and seed `loss̄ = 1` — all before the first
record. Why seed *before* recording? Because reductions accumulate through
`GradSink` atomics, the taped phase never takes `&mut loss` — so by backward
time there is no way to write the seed. The iteration-0 companion tripwire
(`max|∇| > 0` after the first backward) catches a forgotten seed in one
iteration instead of letting a flat loss impersonate convergence for four
hundred.

## 7. Run it

```bash
cd examples/tutorial-fit && cargo oxide run
```

```text
— Part 1: one kernel, two backends —
✓ forward: GPU == CPU == direct body call  (512 samples)
— Part 2: the adjoint vs finite differences —
✓ gradcheck: analytic ∇[A, d, ω] confirmed by finite differences
— Part 3: taped GPU fit, Adam (lr 0.05) —
  ✓ iter 0: GPU gradient matches CPU backend (rel 1e-3)
  iter   0  loss 1.255231e1  params [1.0500, 0.2500, 2.0500]
  iter  50  loss 3.025381e-2  params [1.4713, 0.3403, 2.2101]
  iter 100  loss 7.394206e-5  params [1.5021, 0.3505, 2.2001]
  iter 200  loss 6.685728e-9  params [1.5000, 0.3500, 2.2000]
  iter 399  loss 3.340401e-13  params [1.5000, 0.3500, 2.2000]
✓ fit: 1.255e1 → 3.340e-13 (13.6 orders); [A, d, ω] recovered within 1 %

✅ TUTORIAL FIT ACCEPTANCE PASSED
```

Started from `[1.0, 0.2, 2.0]`, the optimizer recovers `[1.5, 0.35, 2.2]` to
four decimal places (last-digit wobble across runs is expected — gradient
atomics commit in nondeterministic order, which is also why the iteration-0
cross-backend check compares at relative 1e-3, never `==`). And the SDK
workspace is untouched: `git status` there shows nothing.

## 8. Where to go from here

- **Custom by-value kernel params**: POD structs defined in your crate pass
  by value after one line — `shannon_rt::impl_kernel_arg!(MyParams);`.
- **Vec3 parameters and losses**: the same pattern at `Vec3` scale — the
  SDK's `shape_fit` demo is the worked example, up to differentiating
  *through* BVH closest-point queries (untaped correspondence, envelope
  theorem).
- **The one rule to respect while taped**: between the first record and
  `backward`, touch taped buffers only through op wrappers and read-only
  launches — a raw scatter launch that aliases a taped buffer compiles and
  silently corrupts gradients ([LIMITATIONS.md](LIMITATIONS.md) #1).
- **This boilerplate is scheduled to shrink**: the kernel registry + ambient
  tape ([ROADMAP.md](../ROADMAP.md) Phase 2) makes the op-wrapper pattern the
  SDK's job instead of yours.
