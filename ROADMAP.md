# Roadmap

Phases, not dates. Each phase is a state the project can sit in indefinitely
without being half-finished, and nothing below promises a feature. Items carry
a pointer to the groundwork that makes them tractable. The matched pair to
this file is [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — what's missing ↔
what's next.

The project is unreleased; the current state is the `## [Unreleased]` section
of [CHANGELOG.md](CHANGELOG.md).

---

## Phase 1 — Runs for someone else

**Goal: a stranger with an NVIDIA GPU reproduces the README quickstart.**

| | Item |
|---|---|
| ✅ | Quickstart rehearsed verbatim in a fresh shell, ending at the `affine` banner |
| ✅ | Release document set (this file, CHANGELOG, CONTRIBUTING, SECURITY, SAFETY, NOTICE) |
| ⬜ | Verified on a second machine / second GPU architecture |
| ⬜ | First tagged release. Publishing to crates.io is blocked while the workspace depends on a sibling cuda-oxide checkout by path — resolving that (git or crates.io dependency) is part of this item |

## Phase 2 — Ergonomics

**Goal: using the tape does not require knowing its implementation.**

| | Item |
|---|---|
| ✅ | Module-agnostic launching — `launch_in!` + `define_module_cache!` + `impl_kernel_arg!`: any crate declares and launches its own kernels; `shannon-rt` no longer depends on the example kernel crate; the standalone [tutorial project](examples/tutorial-fit/) is the proof |
| ⬜ | Kernel registry + ambient-recording launch + runtime backend dispatch — dissolves the remaining warts: explicit `kernels::` vs `cpu::` call sites and the tape's raw-launch hole (LIMITATIONS #1, #9). Groundwork: the tape (`shannon-autodiff/src/tape.rs`), the `ops::` layer (`shannon-examples/src/ops.rs`), and `launch_in!` |
| ⬜ | `grad(f)` functional veneer — ~15 lines over `Tape` + `requires_grad_`; the shape-fitting three-phase loop is the shape it wraps |
| ⬜ | `#[shannon_kernel]` proc macro, return form — the `macro_rules!` rows in `shannon-kernels`/`shannon-cpu` are the expansion spec |

## Phase 3 — GPU completeness

**Goal: a training iteration touches the host only to read the loss.**

| | Item |
|---|---|
| ⬜ | Signed closest-point via pseudonormals — the main mesh-query capability gap (LIMITATIONS #7); the unsigned query and its brute-force oracle in `shannon-core/src/mesh.rs` are the groundwork |
| ⬜ | GPU refit (parent pointers + atomic arrival counters) — removes the ~0.5 MB/frame host round trip (LIMITATIONS #3); host refit in `shannon-spatial/src/mesh.rs` is the oracle |
| ⬜ | GPU `adam_step` kernel behind the same `step()` signature (LIMITATIONS #4) |

## Phase 4 — Performance

**Goal: close the measured gaps in [docs/RESULTS.md](docs/RESULTS.md).**

| | Item |
|---|---|
| ⬜ | SAH/LBVH build + traversal-order upgrades (test-children-before-push, packed nodes, fast-math device codegen) — the measured 4.5–5.8× enumerate-all gap and 1.45× mesh-closest-point gap at 16,384 queries name the order to try (LIMITATIONS #10) |
| ⬜ | 2-D/3-D launch grids · CUDA graphs · pooled allocator — the launch-overhead bench is the before/after metric |

## Phase 5 — Verification depth

**Goal: the conventions in [CONTRIBUTING.md](CONTRIBUTING.md) become checks.**

| | Item |
|---|---|
| ⬜ | Compile-time read/write verifier for launches — closes LIMITATIONS #1 properly, beyond the Phase-2 convention |
| ⬜ | `#[differentiable]` adjoint-generation macro — the twelve hand adjoints are the target output and the gradcheck suite is the ready-made oracle |

---

## Not on the roadmap

Named because their absence is a design choice, not an oversight.

- **f64 kernels and multi-GPU.** The f32 / one-GPU / 1-D-grid contract
  (LIMITATIONS #13) is what keeps the SDK small enough to verify.
- **Production hardening.** This is an experimental research prototype; the
  [README](README.md) status banner is the contract.
- **A physics engine, planner, or SLAM layer.** The SDK differentiates
  kernels; what you build with the gradients is a different project.
- **Beating Warp on every benchmark.** The measured claims are launch
  overhead and mesh-closest-point parity; Warp's decade-tuned traversal
  interiors are acknowledged, not chased.
