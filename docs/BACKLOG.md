# Backlog

Planned work, **ranked** — top of list = highest value per unit of effort. Each item
names *what*, *why now*, an effort signal (S = hours, M = ~a focused session,
L = multi-session), and *where the groundwork sits* — no vague wishes. The matched
pair to this file is [LIMITATIONS.md](LIMITATIONS.md) — what's missing ↔ what's next.

1. **Kernel registry + ambient-recording `launch!` + runtime backend dispatch.**
   Why now: dissolves THREE recorded warts at once — the `AsKernelArg`
   dependency-direction smell (`shannon-rt/src/launch.rs`, LIMITATIONS #8), explicit
   `kernels::` vs `cpu::` call sites (LIMITATIONS #9), and the tape's raw-launch hole
   (LIMITATIONS #1: with ambient recording, `pause()` becomes load-bearing and a
   Warp-style grad registry gives one-call `tape.zero()`, warp `tape.py:157/:288`).
   Effort: L — the headliner. Groundwork: the tape
   (`shannon-autodiff/src/tape.rs`) + the `ops::` layer in `shannon-examples/src/ops.rs`.
2. **Signed closest-point (pseudonormals).** The main mesh-query capability gap
   (LIMITATIONS #7). Effort: M. Groundwork: the unsigned query + brute-force
   oracle pattern in `shannon-core/src/mesh.rs` and the three-way validation harness.
3. **`#[shannon_kernel]` proc macro, return form** (the next rung of the macro
   ladder). Effort: M–L. Groundwork: the `macro_rules!` rows ARE the expansion
   spec (`shannon-kernels/src/lib.rs`, `shannon-cpu/src/lib.rs`); build against a
   host-only stub crate first.
4. **`grad(f)` functional veneer.** ~15 lines over `Tape` + `requires_grad_`.
   Effort: S. Groundwork: `shannon-autodiff/src/tape.rs`; the shape-fitting
   three-phase loop is the shape it wraps.
5. **GPU refit** (parent pointers + atomic arrival counters, warp `bvh.cu:42`).
   Removes the ~0.5 MB/frame host round trip (LIMITATIONS #3). Effort: M.
   Groundwork: host refit in `shannon-spatial/src/mesh.rs` is the oracle.
6. **GPU `adam_step` kernel** behind the same `step()` signature (warp
   `_src/optim/adam.py` parity; LIMITATIONS #4). Effort: S. Groundwork:
   `shannon-autodiff/src/optim.rs` + the adjoint-kernel row pattern.
7. **`#[differentiable]` adjoint-generation macro.** Effort: exploratory. Groundwork:
   the 12 hand adjoints in `shannon-core/src/adjoint.rs` are the target output; the
   gradcheck suite (`shannon-autodiff/tests/adjoints.rs`) is the ready-made oracle.
8. **SAH/LBVH build + traversal-order upgrades** (test-children-before-push, packed
   nodes, fast-math device codegen). The measured gaps: 4.5–5.8× on the BVH
   enumerate-all queries at 16 384, 1.45× on mesh closest-point (LIMITATIONS #10;
   RESULTS.md findings name the order to try). Effort: M–L. Groundwork:
   `shannon-core/src/bvh.rs` + both bench binaries.
9. **Compile-time read/write verifier** for launches (closes LIMITATIONS #1
   properly, beyond the item-1 convention). Effort: exploratory — likely a
   `launch!`-macro analysis. Groundwork: warp `tape.py:285`
   (`verify_autograd_array_access`) is the semantic spec.
10. **2-D/3-D launch grids · CUDA graphs · pooled allocator.** Effort: each M.
    Groundwork: `shannon-rt/src/launch.rs` (grids), the launch-overhead bench
    harness (graphs — the µs/launch table is the before/after metric).
