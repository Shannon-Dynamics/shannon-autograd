# Safety model

What "safety" means in this repository, what has been verified, where the
`unsafe` code is, and what is deliberately not claimed. This document is about
Rust memory safety; it is not a functional-safety argument, and a robot
optimized with this library can still converge to a bad answer for reasons
that have nothing to do with memory safety.

## 1. Where the `unsafe` code is

Fifteen `unsafe` occurrences across five files, all at the boundaries between
safe Rust and either the GPU or a reinterpreted buffer. There is no `unsafe`
in any kernel body, adjoint, spatial structure, or optimizer.

| Site | What it does | Why it is sound |
|---|---|---|
| `shannon-core/src/vec.rs` (2) | `vec3s_as_f32s{,_mut}`: view `&[Vec3]` as `&[f32]` of 3× the length | `Vec3` is `#[repr(C)]` with three `f32` fields and no padding; the SAFETY comment states the layout argument |
| `shannon-core/src/device_copy.rs` (8) | `unsafe impl DeviceCopy` marker for the eight `#[repr(C)]` POD types crossing the PCIe boundary | Plain-old-data by construction: `Copy`, no pointers, no invariants beyond bit validity |
| `shannon-rt/src/grad.rs` (2) | View an `f32` buffer as `&[AtomicU32]` for the host CAS-loop grad sinks | Same size/alignment; atomics are the only writers during accumulation |
| `shannon-rt/src/launch.rs` (1) | The `launch_in!` macro's actual kernel call into cuda-host's unsafe API (the block expands at each launch call site) | The macro is the single chokepoint; argument marshalling goes through the typed `AsKernelArg` trait |
| `shannon-device/src/lib.rs` (2) | The device-side atomic-add intrinsics behind `DeviceGradF32`/`DeviceGradVec3` | Wraps a hardware atomic on addresses derived from checked slice indexing |

Eight of these carry explicit `SAFETY:` comments stating the invariant. There
is currently no crate-level `#![forbid(unsafe_code)]` anywhere — the policy is
containment at these boundary files, enforced by review rather than by the
compiler. Making the math crates `forbid` outside the two cast helpers is a
cheap hardening step a contributor could take.

## 2. The race-freedom argument: the output-type rule

Data races on the GPU are prevented by construction, not by testing:

- **Forward kernels** write through `DisjointSlice<T>` — each thread may write
  only its own index. Two threads cannot alias an output element without
  changing the kernel's type signature.
- **Adjoint kernels** never take `&mut`. They read `&[T]` and accumulate
  through a `GradSink` (device: hardware atomic add; host: CAS loop), so
  concurrent writes to the same gradient cell are ordered by the hardware.

The cost of that choice is stated in §4.

## 3. What has been verified

| Property | How | Result |
|---|---|---|
| Host memory safety outside the 15 sites | Safe Rust; enforced by the compiler | Holds by construction |
| GPU/CPU agreement | 105 host tests + per-binary acceptance runs: every spatial query validated GPU == CPU == brute force; adjoints gradient-checked against finite differences | Green ([docs/RESULTS.md](docs/RESULTS.md)) |
| Lint state | `cargo clippy --workspace --all-targets` | Zero warnings |

Tests cover what they execute, which is not every path. This is evidence, not
a proof.

## 4. Known safety boundaries

- **Gradient accumulation is not bit-reproducible.** Float atomics commit in
  nondeterministic order; two identical runs differ in the last ulps. All
  validation is relative-tolerance, never `==`
  ([docs/LIMITATIONS.md](docs/LIMITATIONS.md) #6).
- **The tape's raw-launch hole.** The borrow-scoped tape statically freezes
  taped buffers against `&mut`, but a raw `launch!` of a const-ref scatter
  kernel aliasing a taped buffer compiles and corrupts backward — wrong
  numbers, not memory unsafety. Convention-guarded
  ([docs/LIMITATIONS.md](docs/LIMITATIONS.md) #1) until the planned verifier
  ([ROADMAP.md](ROADMAP.md)).
- **Untrusted input** enters only through the OBJ loader; its failure mode on
  malformed input is an error or a panic in the demo binary, never a silent
  wrong mesh. Hostile-input handling is a [SECURITY.md](SECURITY.md) matter.

## 5. Panics

Library code returns `Result` for fallible operations (device errors, size
mismatches). Panics remain where a programming error is the only cause:
slice-index violations (checked indexing is on in device code — its measured
13–16 % cost is in [docs/RESULTS.md](docs/RESULTS.md)) and demo-binary
`expect`s on CLI arguments.

## 6. What is not claimed

- **Not functional correctness of user kernels.** The SDK differentiates what
  you wrote; if the forward model is wrong, the gradients are faithfully
  wrong.
- **Not bit-reproducibility** of any accumulated quantity (§4).
- **Not whole-application safety.** The `launch_in!` chokepoint trusts kernel
  dimension arguments; a dimension larger than the smallest buffer is caught
  by device-side bounds checks, not at the call site.
- **Not thread safety beyond what the types say.** The tape is deliberately
  not `Send` ([docs/LIMITATIONS.md](docs/LIMITATIONS.md) #5).

## 7. Reporting a safety problem

Memory-unsafety with a security dimension: follow [SECURITY.md](SECURITY.md).
Everything else: open a public issue with the smallest reproduction you have.
