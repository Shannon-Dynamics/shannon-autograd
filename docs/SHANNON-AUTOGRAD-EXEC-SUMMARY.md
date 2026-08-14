# shannon-autograd — Executive Summary

**A differentiable GPU computing SDK for Rust.**

*Proof-of-concept proposal · 7 days · 1 engineer · Prepared 2026-08-07*

---

## In one sentence

**shannon-autograd lets an engineer write a physics or graphics routine once, in Rust, and get
three things automatically: it runs fast on the GPU, it runs on the CPU without a GPU present,
and it computes its own derivatives — which is what makes optimization, calibration, and
learning possible.**

---

## Why this matters

**Derivatives are the difference between simulating and solving.** A simulator answers "given
these parameters, what happens?" A *differentiable* simulator answers the far more valuable
question: **"what parameters would produce the outcome I want?"** That turns simulation from a
test tool into an optimization engine — fitting models to sensor data, calibrating physical
parameters, and training controllers against a virtual environment.

Today, building one forces an unattractive choice:

| Option | The cost |
|---|---|
| **Python frameworks** | Fast to prototype, awkward to ship. Interpreter overhead on every operation; heavyweight runtime; errors surface at runtime, in production |
| **Raw CUDA C++** | Fast, but no automatic derivatives, manual memory safety, and a separate toolchain |
| **Machine-learning frameworks** | Excellent for neural networks; structurally unable to express geometric queries like "what is the nearest surface point?" |

Rust has become a default choice for robotics, simulation, and infrastructure — it delivers
C++-level performance with memory safety and a single-binary deployment story. **There is
currently no production-grade differentiable GPU computing SDK for Rust.**

## Why now

In 2026 NVIDIA released **cuda-oxide**, a compiler backend that turns ordinary Rust into GPU
code. Until this existed, writing GPU kernels in Rust meant accepting a restricted mini-language.
That constraint is gone.

**The compiler problem — historically the hard, expensive part — is now solved by someone else,
and it is open source.** What does not yet exist is the layer above it: the types, the geometry,
and the automatic differentiation that make it usable for real work. That layer is a
tractable amount of engineering, and the window to build it is open now.

---

## What we would build

A layered SDK. The engineer writes the middle line; the system supplies the rest.

```
   Write once, in ordinary Rust
              │
      ┌───────┴────────┐
      ▼                ▼
  Runs on GPU      Runs on CPU          ← same source, no rewrite
      │                │
      └───────┬────────┘
              ▼
     Computes its own gradients          ← the differentiating capability
```

Concretely, week 1 delivers four demonstrations and two performance measurements:

| # | Deliverable | Proves |
|---|---|---|
| 1 | Vector arithmetic + gradient verification | The foundation is correct, checked against an independent oracle |
| 2 | Real-time 3D renderer | The SDK handles genuine graphics workloads, identically on CPU and GPU |
| 3 | Mesh collision simulation | Geometric queries and spatial data structures work |
| 4 | **Shape fitting by optimization** | **The core capability** — one 3D shape deforms into another, driven purely by gradients |
| 5 | Dispatch-overhead benchmark | Quantifies the performance advantage over interpreted frameworks |
| 6 | Spatial query benchmark | Establishes the geometry performance baseline |

**Deliverable 4 is the one to watch.** It is the smallest complete demonstration that the whole
system works: a shape is described only by a target, and the software discovers the deformation
that reaches it — no human specifying how.

---

## What success looks like

Each day carries a **binary pass/fail test**. There is no partial credit and no room for
"mostly working."

The week succeeds if, on day 7, we can demonstrate:

1. ✅ A 3D scene rendered on the GPU, and the **identical source** producing a matching image on
   the CPU.
2. ✅ A shape optimizing into a target shape, with the error decreasing by **at least 100×**.
3. ✅ Every gradient independently verified against a mathematical reference — **not** merely
   "the demo looked right."
4. ✅ Published performance numbers with reproducible methodology.

**If items 1 and 2 both work, the architecture is validated** and a longer roadmap becomes a
question of scope, not feasibility.

---

## Risks, stated plainly

| Risk | Likelihood | If it happens | Our response |
|---|---|---|---|
| **Development GPU is below the supported floor** | Medium | Blocks GPU work entirely | **Verified on day zero, before any code.** The compiler's source shows our hardware *should* work despite the documentation stating otherwise. If not: a cloud GPU costs roughly a day's engineering time. **And because the CPU path is a first-class design decision, ~80% of the SDK still ships even in the worst case.** |
| **Toolchain setup overruns** | High | Loses ~1 day of 7 | Scheduled *before* the week starts, not inside it |
| **The underlying compiler is alpha software** | Medium | Intermittent bugs | Keep our code simple and conventional; the project is actively maintained with a public issue tracker |
| **Silent mathematical errors** | Medium | Wrong results that *look* correct | The independent verification tool is built **on day 1, before the code it checks.** This is the single most important process decision in the plan |

**The honest framing:** this is a one-week proof of concept, not a product. It validates a
technical hypothesis on a small budget. A production SDK is a multi-month effort, and this week
is precisely how we decide whether that investment is warranted.

---

## Cost and decision points

| | |
|---|---|
| **Investment** | 1 engineer × 7 days |
| **External cost** | £0 — all dependencies are open source. Contingency: ~1 day of cloud GPU if the hardware risk materialises |
| **Existing assets** | Development machine with GPU already available |

**Two decision points:**

- **End of day 0** — hardware viability confirmed. If the GPU is unsuitable, we decide between a
  cloud instance or proceeding CPU-only. *A ~1-hour decision, not a re-plan.*
- **End of day 7** — architecture validated or not. If validated, we scope a longer roadmap. If
  not, we have spent one week to learn it, with a documented account of why.

---

## What we are *not* claiming

Presented deliberately, so the proposal is judged on what it actually is:

- ❌ This will **not** be feature-complete against mature alternatives. Those represent many
  engineer-years.
- ❌ We are **not** claiming a performance win across the board. We expect an advantage in one
  specific, measurable dimension — dispatch overhead — and will publish the methodology so the
  claim can be checked.
- ❌ This is **not** production-ready after one week. It is a validated foundation, or a
  documented dead end.

---

## Positioning and intellectual property

shannon-autograd is an original Rust SDK, architecturally distinct from existing systems — it
compiles ahead of time rather than at runtime, uses Rust's type system rather than a Python
subset, and treats CPU execution as a first-class target rather than a fallback.

It builds on open-source components under the permissive **Apache 2.0** licence, which permits
commercial use. Standard attribution obligations are itemised in the technical plan and
scheduled as a day-7 task. **No copyleft or licence-contamination exposure.**

Where our design deliberately follows established practice, we say so and cite it. That is
normal engineering discipline and it strengthens, rather than weakens, the work.

---

## Recommendation

**Proceed.** The cost is one engineer-week, the downside is bounded and informative, and the
upside is a validated foundation in a domain where no Rust option currently exists.

Run day 0 — the hardware check — **before** the week formally begins. It is the only step that
can invalidate the plan, it takes hours rather than days, and doing it early converts the
largest risk into a known quantity.

---

### Appendix — Terms

| Term | Meaning |
|---|---|
| **GPU kernel** | A small program run simultaneously across thousands of GPU cores — the unit of parallel work |
| **Gradient / derivative** | The direction and rate a result changes as an input changes. Knowing this lets software *search* for good parameters instead of guessing |
| **Automatic differentiation** | Computing those derivatives mechanically from the code itself, rather than deriving them by hand — eliminating a slow and error-prone step |
| **Differentiable simulation** | A simulation you can run *backwards*: given a desired outcome, find the inputs that produce it |
| **SDK** | Software Development Kit — the library other engineers build applications on |
| **Ahead-of-time compilation** | Translating code to machine instructions when the software is built, rather than while it runs. Catches errors earlier and removes startup cost |
| **Proof of concept** | A deliberately minimal build that tests whether an approach works, before committing to full development |

---

*Technical detail: [SHANNON-AUTOGRAD-WEEK-1-PLAN.md](SHANNON-AUTOGRAD-WEEK-1-PLAN.md)*
