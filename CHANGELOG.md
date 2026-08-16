# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/) and the
project follows [semantic versioning](https://semver.org/) — with the 0.x
convention that the minor version is the breaking one until 1.0.

Every measured number below is reproduced, with hardware and methodology, in
[docs/RESULTS.md](docs/RESULTS.md).

## [Unreleased]

Nothing has been released yet. This section describes the repository as it
stands, which is what the first release will contain.

### Added

- **A differentiable GPU computing SDK in Rust**: kernels written as ordinary
  Rust functions, compiled ahead-of-time to PTX via cuda-oxide and to a rayon
  CPU backend from the same source, with reverse-mode autodiff over a
  borrow-scoped tape of launches.
- **Eight workspace crates** — `shannon-core` (all math, `no_std`),
  `shannon-device`, `shannon-rt`, `shannon-autodiff`, `shannon-spatial`,
  `shannon-kernels`, `shannon-cpu`, `shannon-examples` — about 9,200 lines of
  Rust.
- **Ten demo binaries**, each ending at an acceptance banner: `affine`,
  `raymarch`, `shannon_anim`, `arm_pick_place`, `mesh_sim`, `conveyor_sort`,
  `mesh_bench`, `shape_fit`, `launch_bench`, `bvh_bench`.
- **Module-agnostic launching**: `shannon_rt::launch_in!` +
  `define_module_cache!` + `impl_kernel_arg!` let ANY crate — first- or
  third-party — declare kernels with `shannon_gpu_kernels!` and launch them;
  each kernel crate owns its PTX-module cache, keyed by package name, and
  several bundles coexist in one binary. `shannon-rt` no longer depends on the
  example kernel crate (the recorded dependency-direction wart), whose
  `launch!` is now a thin wrapper over `launch_in!`.
- **A user tutorial with a living example** ([docs/TUTORIAL.md](docs/TUTORIAL.md)
  + the standalone project `examples/tutorial-fit/`): a user crate that
  imports the SDK as a library and goes from math body to taped optimization
  in its own directory — map row on both backends, scatter adjoint through
  GradSink atomics, finite-difference gradcheck, and Adam recovering the
  parameters of a damped sine wave to four decimal places (13.6 orders of
  loss decrease). No SDK file is edited.
- **105 host tests** (102 across the library crates, 3 in the examples
  library), including per-adjoint gradient checks against finite differences
  and three-way GPU == CPU == brute-force validation for every spatial query.

### Documentation

- README with a verified quickstart, [docs/RESULTS.md](docs/RESULTS.md) as the
  single source of measured numbers, [docs/LIMITATIONS.md](docs/LIMITATIONS.md)
  as the honest ledger, [ROADMAP.md](ROADMAP.md), [SAFETY.md](SAFETY.md),
  [SECURITY.md](SECURITY.md), [CONTRIBUTING.md](CONTRIBUTING.md), and
  NOTICE/LICENSE attribution for cuda-oxide and NVIDIA Warp.

### Known limitations

The thirteen known limitations — the tape's raw-launch hole, host-side refit
and optimizers, unsigned mesh queries, non-bit-reproducible gradient
accumulation, f32-only, and the rest — are catalogued with symptoms,
workarounds, and planned fixes in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

---

## Earlier work — showcases and release polish

### Added

- **`--smooth <λ>` for `shape_fit`**: an untaped, host-side Laplacian
  shrink-wrap step after each optimizer iteration. Without it, vertex-wise
  Chamfer stalls in "tenting" minima on highly non-convex targets — every
  vertex lands on the target surface while large faces bridge the concavities
  at zero loss cost.
- **Three arbitrary-OBJ fitting showcases** (car, cat, robot) with auto-
  normalization of target meshes, assembled with the blob acceptance fit into
  one side-by-side GIF (`docs/shape_fit.gif`).
- **Two original demos with no Warp derivation**: `conveyor_sort` (a grooved,
  running conveyor belt sorts particles into lanes) and `arm_pick_place`
  (a 4-DOF arm with analytic IK restores a sign's letter in a seamless loop).

### Measured

- Car / cat / robot fits: 3.1 / 2.5 / 3.6 orders of loss decrease, with GPU
  gradients matching the CPU backend at relative 1e-3 on every run.
- `arm_pick_place`: ~30 fps at 960×540, worst CPU/GPU channel delta 2.2e-4.
- `conveyor_sort`: 600/600 particles delivered, 100 % lane-sorted, zero
  tunnelling over 600 frames.

### Changed

- **Assets audited for licensing.** The license-unknown research-heritage car
  mesh was replaced with a CC0 sedan (Kenney Car Kit); the cat is CC BY 4.0
  (attributed in NOTICE); the robot is original. See `assets/README.md`.
- Architecture-specific warning wording was generalized: the docs describe the
  default-arch JIT trap for any GPU rather than one machine.

---

## Earlier work — autodiff and shape fitting

### Added

- **The borrow-scoped tape** (`shannon-autodiff`): records hold shared borrows
  of their buffers, so the borrow checker statically forbids mutating a taped
  buffer; `backward(self)` consumes the tape, which is what makes the
  optimizer's mutation legal again.
- **Twelve hand-written adjoints** in `shannon-core`, each gradient-checked;
  `GradSink` abstraction so the same adjoint body accumulates through device
  atomics on GPU and CAS loops on host.
- **Symmetric-Chamfer shape fitting through BVH closest-point queries**:
  correspondence is requeried untaped each iteration and the gradient is exact
  under held correspondence (envelope theorem) — differentiating through an
  imperative spatial query without writing a query adjoint.
- Host `Sgd` and `Adam` optimizers behind flat-slice signatures.

### Measured

- Stage 1 (L2 + SGD, the provable oracle): 8.9 orders of loss decrease over
  200 iterations, matching the (1−lr)² contraction theorem; monotone at every
  step.
- Stage 2 (Chamfer + Adam): 3.5 orders against a 2-order bar; full acceptance
  run 1.08 s wall.

---

## Earlier work — spatial structures and benchmarks

### Added

- Median-split BVH with AABB, ray, and mesh closest-point queries; host-side
  refit; mesh generators and OBJ I/O; a particle-vs-deforming-mesh simulation
  with an analytic no-tunnelling predicate.
- Launch-overhead and BVH/mesh benchmarks, each validating GPU == CPU ==
  brute force before timing anything.

### Measured

- GPU launch overhead 1.1–3.0× lower than the Warp baseline in 11 of 12
  cells on the same machine; CPU dispatch is a direct call (ns) versus an
  interpreted path (µs).
- Mesh closest-point at parity with Warp at 128 queries (1.234 vs 1.301 ms),
  1.45× behind at 16,384; Warp's tuned LBVH leads 4.5–5.8× on enumerate-all
  BVH queries at full occupancy.
- Mesh simulation: zero tunnelling violations across 1,000 particles × 200
  frames on a deforming mesh.

---

## Earlier work — the core SDK and the dual backend

### Added

- `shannon-core` as a `no_std`, zero-CUDA-dependency crate holding every line
  of math; GPU (`shannon-kernels`) and CPU (`shannon-cpu`) adapters expand the
  same kernel-row declarations into both backends.
- The output-type rule: forward kernels write through `DisjointSlice` (each
  thread owns its index); adjoint kernels accumulate through `GradSink`
  atomics, never `&mut`.
- `Array<T>` with attached `.grad`, the `launch!` macro, timers, and an SDF
  ray-marching renderer as the first two-backend workload.

### Measured

- Ray-march CPU/GPU parity: 99.88 % of pixels within 1e-4 at 2048×1024.
