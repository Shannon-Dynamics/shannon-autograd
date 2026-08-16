# Security policy

## Scope

The project is pre-1.0 and has not been released. Until the first tagged
release, the supported version is the current `main` branch.

This is an experimental research prototype that runs locally against files the
user chooses. Its one untrusted-input surface is the OBJ loader in
`shannon-examples` (`--target-obj` and the mesh benchmarks): a hostile mesh
file is the realistic attack input here.

## Reporting a vulnerability

Report privately through GitHub's **Report a vulnerability** button (Security
tab of the repository). Please do not open a public issue for anything you
believe is exploitable.

> **Note for the repository owner:** private vulnerability reporting has to be
> enabled in the repository settings for that button to exist. Until it is,
> this section describes a route that is not yet open. No alternative contact
> is published here, because publishing one is the owner's decision to make.

Include what you observed, a minimal input that triggers it, and your
environment (GPU, driver, toolkit). Reports get an acknowledgement within a
few working days, and any disclosure timeline is agreed with the reporter
before anything is published.

## What counts as a security issue here

In scope:

- Undefined behaviour reachable through the safe API, including through the
  `unsafe` blocks inventoried in [SAFETY.md](SAFETY.md).
- Panics, out-of-bounds access, or unbounded resource use triggered by a
  small, malformed OBJ file.
- Silent memory corruption from the launch layer (a kernel writing outside
  its buffers).

Out of scope:

- Vulnerabilities in upstream dependencies — cuda-oxide, the CUDA driver and
  toolkit, rayon — reported on their own trackers. Issues in how this
  repository *uses* them are in scope.
- Resource exhaustion from legitimately large meshes; the demos load what
  they are given.
- Numerical-accuracy or convergence bugs on valid input. They matter, but
  they are correctness issues — open a public issue.
- Results of running the benchmarks against the Warp baseline; benchmark
  variance is not a vulnerability.
