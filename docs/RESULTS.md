# shannon-autograd — Measured Results

*The single source of truth for every number the README cites.*

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1650, 4 GiB, sm_75, under WSL2 |
| Driver / Toolkit | Driver 13.3 · CUDA Toolkit 13.3 (ours) / 12.9 (Warp wheel) |
| CPU baseline path | direct `#[inline(never)]` call + `black_box` |
| Warp baseline | **warp-lang 1.13.0 (pip wheel, prebuilt binaries), same machine** |
| Date | 2026-08-08 (launch overhead) · 2026-08-11 (BVH, mesh) · 2026-08-12 (shape fitting) · collated 2026-08-13 · 2026-08-14 (car/cat/robot fits) |

## Headline summary

One row per workload; every number is reproduced in full, with methodology, in the
linked section below.

| Workload | Headline result | Full section |
|---|---|---|
| Launch overhead | GPU launches **1.1–3.0× faster than Warp in 11 of 12 cells** (e.g. k0 direct 17.87 vs 45.3 µs); CPU dispatch ns vs µs (direct call vs interpreter) | [launch overhead](#launch-overhead-20-000-launchescell-dim--1-median-of-3) |
| BVH queries | Correctness three ways (GPU == CPU == brute force, hit **sets**); CPU↔GPU crossover as predicted (0.16×@128 → 1.6×@16 384); Warp's tuned LBVH traversal 4.5–5.8× faster at full occupancy | [BVH queries](#bvh-queries-10-000-bounds-100-iterations-mean-of-per-launch-device-times) |
| Mesh closest-point | **Parity with Warp at 128 queries** (1.234 vs 1.301 ms); 1.45× behind at 16 384; three-way validation before every timing | [mesh bench](#mesh-closest-point-queries-icosphere-subdiv-5-20-480-tris--10-242-verts) |
| Mesh simulation | 1000 particles × 200 frames on a deforming mesh: zero tunnelling violations, settle predicate met, three-way closest-point green at frames 0 and 100 | [mesh sim](#mesh-simulation-mesh_sim-acceptance-run) |
| Differentiable fitting | Stage 1 SGD: **8.9 orders** of loss decrease (theorem-predicted); stage 2 Chamfer+Adam: **3.5 orders**, GPU grads == CPU grads (rel 1e-3) at iters 0/100; car/cat/robot OBJ fits via `--smooth` (2.5–3.6 orders); full run **1.08 s** | [shape fitting](#differentiable-shape-fitting-shape_fit-acceptance-run) |

## Launch overhead (20 000 launches/cell, dim = 1, median of 3)

Methodology: wall-clock around the **async enqueue loop**, synchronization outside the
timed region, 100-launch warm-up per cell — identical on both stacks (Warp's own
`benchmark_launches.py`, trimmed to 20k, run twice with the second run reported).

### shannon-autograd

| shape | CPU ns/call | GPU direct µs | GPU struct µs |
|-------|-------------|---------------|---------------|
| k0    |         1.3 |         17.87 |         14.58 |
| kf    |         1.9 |         13.15 |         20.05 |
| kv    |         1.8 |         12.78 |         20.75 |
| km    |         3.7 |         16.96 |         21.71 |
| ka    |         3.8 |         19.98 |         18.49 |
| kz    |        23.1 |         14.27 |         22.47 |

`bench_k0` device-side execution (CudaEvent, 1000 launches): **13.16 µs/kernel** — on this
WSL2 stack the pipeline is launch-latency-bound; enqueue and drain rates converge.

### Warp 1.13.0 (same machine, same methodology, µs/launch)

| shape | CPU direct | CPU struct | CUDA direct | CUDA struct |
|-------|-----------:|-----------:|------------:|------------:|
| k0    |   182.4 ¹  |        6.8 |        45.3 |        32.0 |
| kf    |       10.2 |        6.5 |        37.2 |        28.1 |
| kv    |        9.6 |        7.0 |        35.8 |        32.4 |
| km    |        8.5 |        6.5 |        27.6 |        24.3 |
| ka    |       10.0 |        6.8 |        33.7 |        23.2 |
| kz    |       18.9 |        7.5 |        43.4 |        21.1 |

¹ First CPU cell absorbs Warp's per-device warm-up (no per-cell warm-up in the upstream
script); treat as an artifact, not a representative number.

### Head-to-head (GPU, µs/launch — lower is better)

| shape | shannon direct | Warp direct | × | shannon struct | Warp struct | × |
|-------|---------------:|------------:|---:|---------------:|------------:|---:|
| k0    | **17.87** | 45.3 | 2.5 | **14.58** | 32.0 | 2.2 |
| kf    | **13.15** | 37.2 | 2.8 | **20.05** | 28.1 | 1.4 |
| kv    | **12.78** | 35.8 | 2.8 | **20.75** | 32.4 | 1.6 |
| km    | **16.96** | 27.6 | 1.6 | **21.71** | 24.3 | 1.1 |
| ka    | **19.98** | 33.7 | 1.7 | **18.49** | 23.2 | 1.3 |
| kz    | **14.27** | 43.4 | 3.0 | 22.47 | **21.1** | 0.9 |

**Findings.**
- shannon-autograd launches are **faster in 11 of 12 GPU cells, by 1.1–3.0×**, with the
  single exception (`kz` struct) within noise of parity. The thesis-statement claim —
  typed Rust dispatch beats interpreter + ctypes marshalling — holds on measured data.
- **CPU dispatch is ~3 orders of magnitude apart**: a shannon CPU "launch" is a direct
  function call (1.3–23 **ns**); Warp's CPU launch goes through the Python interpreter
  (6.5–19 **µs**). Different mechanisms, honestly labelled — but that difference is the
  per-step overhead a simulation loop actually pays.
- Direct-vs-struct diverges between stacks: Warp consistently benefits from struct packing
  (one ctypes marshal instead of N); shannon's direct path is often *cheaper* than struct
  (typed calls have no per-arg interpreter cost to amortize, while the struct adds a byval
  copy). Warp's published guidance "prefer structs for many args" does not transfer to us.
- Run-to-run spread on this shared GPU is ~±20% per cell (WSLg compositor shares the
  1650); medians of 3 reported. The cross-stack gaps exceed the noise band except `kz`
  struct.

## BVH queries (10 000 bounds, 100 iterations, mean of per-launch device times)

Methodology: **device time**, deliberately the opposite of the launch bench's wall-clock enqueue —
this section asks what the *traversal* costs, not the dispatch. Both stacks: 5
warm-up launches, then 100 timed iterations of ONE launch over all queries;
mean ± stdev of per-launch device times (ours via CudaEvent pairs, Warp via
`cuda_filter=wp.TIMING_KERNEL`). CPU cells are rayon wall-clock over the same loop.

Correctness first: sorted per-query hit **sets** (not counts) identical three ways —
GPU == brute force AND CPU-rayon == brute force — for 128 AABB + 128 ray queries
against the 10 000 bounds, plus GPU==CPU count equality re-checked at every
benchmarked size. All green before any number below was believed.

### shannon-autograd (median-split BVH, host-built; checked indexing)

| workload | queries | CPU ms (rayon) | GPU ms (device) | GPU speedup |
|----------|---------|----------------|-----------------|-------------|
| AABB     |     128 |          0.058 |           0.361 |       0.16× |
| AABB     |   16384 |          6.168 |           4.456 |       1.38× |
| ray      |     128 |          0.290 |           0.395 |       0.74× |
| ray      |   16384 |          8.053 |           4.908 |       1.64× |

With `--unchecked-indexing` (elides device bounds checks; validation still green):
GPU AABB @16384 **3.859 ms** (−13%), ray @16384 **4.121 ms** (−16%).

CPU cells at 128 queries carry heavy run-to-run jitter (the whole workload is tens of
µs — rayon wake-up and the WSLg compositor dominate); treat those two cells as
order-of-magnitude only.

### Warp 1.13.0 (same machine, same seed-42 distributions, device time, ms)

LBVH tree (GPU-built; `wp.Bvh` default for device arrays), `fast_math` on — the
upstream benchmark's own configuration, run unmodified for the 128-query rows; the
16 384-query rows are the same script with only `NUM_QUERIES` raised and the tiled
section skipped. Mean hits/query on its data: AABB 34.9, ray 16.9.

| workload | queries | Warp single-threaded | shannon GPU | Warp faster by |
|----------|---------|---------------------:|------------:|---------------:|
| AABB     |     128 |        0.179 ± 0.028 |       0.361 |           2.0× |
| ray      |     128 |        0.255 ± 0.049 |       0.395 |           1.5× |
| AABB     |   16384 |        1.000 ± 0.081 |       4.456 |           4.5× |
| ray      |   16384 |        0.848 ± 0.051 |       4.908 |           5.8× |

Context (tiled queries were out of scope): Warp's tile-parallel variants reach
0.062 ms (AABB, BD=32, 2.9× over their single) and 0.064 ms (ray, BD=32, 4.0×) at
128 queries, degrading again at BD ≥ 128 on this GPU.

**Findings.**
- **The acceptance is correctness, and it holds**: identical hit sets, three ways, on
  both backends, from one shared `shannon_core::bvh` traversal. The core invariant
  (no arithmetic outside shannon-core) survived its first data structure.
- **The CPU↔GPU crossover behaves exactly as predicted**: at 128
  queries the 14-SM GPU is idle hardware and the CPU wins; at 16 384 the GPU wins
  1.4–1.6×. Neither cell alone is the story; the crossover is.
- **Warp's traversal throughput beats ours** — ~2× at 128 queries, 4.5–5.8× at full
  occupancy. Honest reading: the dispatch win does not transfer to kernel *interior*
  performance. Their inner loop is decade-tuned CUDA C over an LBVH with fast_math;
  ours is first-pass generic Rust through MIR→LLVM→PTX with checked indexing and a
  push-both-children-test-on-pop walk. Bounds checks account for 13–16% (measured);
  the rest is tree quality + inner-loop shape.
- **Backlog leads, in expected-value order**: test children before pushing (halves
  stack traffic), packed/SAH nodes, fast-math codegen flags for device crates.

## Mesh closest-point queries (icosphere subdiv 5: 20 480 tris / 10 242 verts)

Methodology: identical to the BVH bench — device time (CudaEvent pairs vs Warp
`cuda_filter=wp.TIMING_KERNEL`), 5 warm-up + 100 timed launches, mean ± stdev; CPU
cells rayon wall-clock. Both stacks query the **bit-identical mesh** (the OBJ dumped
by `mesh_bench --dump-obj`, round-tripped through Display-exact floats) and the
**bit-identical seed-42 query points** ([−2, 2]³, a near/far mix), `max_dist = 1e10`,
`wp.mesh_query_point_no_sign` (the unsigned variant — the signed ones time their sign
machinery too). Run twice, second run reported.

Correctness first: closest-point distances agree three ways at 128 queries —
GPU == brute force AND CPU-rayon == brute force — under the float-tie rule (found
flags + distance + per-backend barycentric self-consistency; face indices are never
compared across backends), and GPU == CPU re-checked at every benchmarked size.

### shannon-autograd (median-split BVH, host-built; checked indexing)

| workload | queries | CPU ms (rayon) | GPU ms (device) | GPU speedup |
|----------|---------|----------------|-----------------|-------------|
| mesh-cp  |     128 |          0.252 |           1.234 |       0.20× |
| mesh-cp  |   16384 |         27.124 |           9.353 |       2.90× |

(The 128-query CPU cell carries the same tens-of-µs jitter caveat as the BVH bench's small
cells: 0.25–0.37 ms across runs; order-of-magnitude only.)

### Warp 1.13.0 (same machine, same mesh, same queries, device time, ms)

| workload | queries | Warp GPU | shannon GPU | Warp faster by |
|----------|---------|---------:|------------:|---------------:|
| mesh-cp  |     128 | 1.301 ± 0.085 | **1.234** | 0.95× (parity) |
| mesh-cp  |   16384 | 6.448 ± 0.345 | 9.353 | 1.45× |

**Findings.**
- **The enumerate-all traversal gap collapses on this workload**: 4.5–5.8× behind on the BVH bench's
  enumerate-all BVH counts → **parity at 128 queries and 1.45× at 16 384** on mesh
  closest-point. The difference is the algorithm, not the codegen: `mesh_query_point`
  is a best-first descent with squared-distance culling and nearest-child-first
  ordering (Warp's own structure, ported), so the shared inner loop does far less
  work than the BVH bench's push-both-visit-all walk. The backlog's traversal upgrades apply
  to those kernels, not this one.
- The remaining 1.45× at full occupancy is consistent with tree quality (their
  GPU-built tree vs our host median split) plus checked indexing — the same two
  leads already on the backlog.
- The CPU↔GPU crossover repeats a third time (0.20× at 128 → 2.90× at 16 384),
  now on the heaviest per-thread kernel yet.
- Warp-side sanity: mean reported distance on identical data — 0.9748 (128) /
  0.9548 (16 384) — matches the geometry of uniform [−2, 2]³ points against a unit
  sphere, and the two stacks' validation paths agree to 1e-4 before timing.

## Mesh simulation (mesh_sim acceptance run)

1000 particles, 64×64 grid (8192 tris), 200 frames of deform → host refit → GPU
particle step. All binary predicates green on the acceptance run:

| Check | Result |
|---|---|
| Three-way closest-point (frames 0 AND 100, 128 samples) | ✅ GPU == brute == CPU |
| Analytic tunnelling predicate, 1000 particles × 200 frames | ✅ zero violations |
| Settle @ frame 200 (bounds: max < 0.25, mean < 0.05) | ✅ max \|v\| = 0.0104, mean = 0.0041 |
| OBJ frame dumps (mesh + particle cloud every 10 frames) | ✅ 40 files |

Sim-loop cost note: the per-frame host refit (download 51 KB of points, reverse-loop
refit, re-upload 0.5 MB of nodes) is a recorded design trade — exact
device-consistency over speed. GPU refit (parent pointers + atomic arrival counters)
is on the roadmap ([ROADMAP.md](../ROADMAP.md) Phase 3).

## Differentiable shape fitting (shape_fit acceptance run)

The capability demonstration: a 642-vertex icosphere fitted to a same-topology "blob"
target (displaced icosphere) by gradients flowing backward through GPU kernels — tape
of launches, hand-written adjoints, envelope-theorem Chamfer with untaped
correspondence queries, host-side optimizers.

**Part 0 — tape chain oracle.** The 3-kernel GPU chain (`affine → sin_map →
sum_scalar`) taped, walked backward, and checked elementwise against the analytic
gradient `2·cos(2x + 0.5)` at 1e-3: ✅. The CPU-backend mirror of the same chain
matches composed finite differences (host test suite).

**Stage 1 — L2 + SGD (lr 0.05), the provable oracle.** SGD on the L2 loss contracts
every residual by (1 − lr) per step — loss by (1 − lr)² — so ~8.9 orders over 200
iterations is a theorem, and the run reproduces it:

| iter | 0 | 40 | 100 | 160 | 199 |
|---|---|---|---|---|---|
| loss | 5.337e0 | 8.814e-2 | 1.871e-4 | 3.972e-7 | 7.281e-9 |

Monotone at every step (1e-4 headroom), iteration-0 gradient tripwire clean,
**8.9 orders** of decrease against a 4-order acceptance bar.

**Stage 2 — symmetric Chamfer + Adam (lr 0.01).** Correspondence requeried untaped
every iteration (two `mesh_query` launches + source-BVH refit); gradients are exact
under held correspondence (envelope theorem — no query adjoint exists or is needed):

| iter | 0 | 40 | 100 | 160 | 199 |
|---|---|---|---|---|---|
| loss | 9.887e0 | 6.492e-2 | 5.536e-3 | 3.844e-3 | 2.918e-3 |

**3.5 orders** against a 2-order bar; the 1.25×-running-min envelope held at every
iterate; GPU gradients matched the host-recomputed CPU-backend gradients at
iterations 0 and 100 (relative 1e-3 — never bit-exact, float atomics commit in
nondeterministic order); zero query misses; OBJ morph frames on disk.

Showcase (`--target torus`, predicates waived by topology): the genus-0 sphere wraps
onto the genus-1 torus surface, 6.475e1 → 2.107e-2 (3.5 orders) — the residual floor
is the covering degeneracy, not the machinery.

Showcase (`--target-obj assets/{car,cat,robot}.obj --subdiv 5 --iters 400
--smooth 0.2`, 2026-08-14): three arbitrary-OBJ targets, all non-convex — a sedan
(Kenney Car Kit, CC0; densified to 16 981 verts / 31 060 tris, auto-normalized
×0.7843), a stretching cat (7949 verts / 15 894 tris, ×0.5771), and an original
procedurally generated toy robot (11 760 verts / 3920 tris, ×0.7828). Plain
vertex-wise Chamfer stalls on such targets in "tenting" minima (every vertex on the
target surface, but large faces bridging the concavities — wheel wells, the gap
between the cat's arched back and head, the robot's limb gaps — at zero loss cost).
The `--smooth λ` shrink-wrap regularizer — an untaped, host-side pull of each vertex
λ of the way toward its 1-ring centroid after every optimizer step — resolves it:
the 10 242-vertex sphere wraps into recognizable shapes (the car's cabin and wheel
bulges; the cat's stretch pose; the robot's antenna, head, and arms). Measured, with
both cross-backend gradient checks green (rel 1e-3 at iters 0 and 100):

| target | loss trajectory | orders |
|---|---|---|
| car   | 1.555e3 → 1.227e0  | 3.1 (smoothing raises the floor — it trades exact fit for shape quality) |
| cat   | 2.212e3 → 7.401e0  | 2.5 |
| robot | 2.144e3 → 5.586e-1 | 3.6 |

All four mesh-to-mesh fits (blob, car, cat, robot) are assembled side-by-side in
`docs/shape_fit.gif`.

Cost note: the full acceptance run — chain check + 200 SGD + 200 Adam iterations,
including 400 correspondence queries, 200 host refits, and the ~15 KB/iteration
gradient/parameter round trip to the host optimizer — is **1.08 s wall**
(~2.7 ms/iteration for stage 2). The GPU `adam_step` kernel and ambient-recording
tape are roadmap items ([ROADMAP.md](../ROADMAP.md) Phases 2 and 3); at this size neither is the bottleneck.

## Prior measurements (context)

| Result | Date |
|---|---|
| Ray march (`raymarch`), 2048×1024, warm GPU | 26.5 ms (2026-08-07) |
| Animated robot-arm scene (`shannon_anim`), 960×540 | 33.4 fps headless / ~29 fps windowed (2026-08-07) |
| Ray-march CPU/GPU parity | 99.88 % of pixels < 1e-4; robust criterion passes (2026-08-07) |
| Animated-scene CPU/GPU parity at fixed t | worst channel delta 1.2e-4 (2026-08-07) |
| ARM-7 pick & place (`arm_pick_place`), 960×540 | 29.7 fps headless; parity 0.00 % > 1e-3, worst 2.2e-4 (2026-08-13) |
| Conveyor sort (`conveyor_sort`) acceptance | 600/600 delivered off the belt, 100 % lane-sorted (±lane/4), zero tunnelling over 600 frames, three-way checks green (2026-08-13) |
