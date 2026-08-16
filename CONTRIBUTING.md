# Contributing

Thank you for considering a contribution. This is an experimental research
prototype; the bar for merging is not size or polish, it is whether the change
keeps the project's claims measurable and true.

## Getting set up

You need an NVIDIA GPU, CUDA Toolkit 12+, and the
[cuda-oxide](https://github.com/NVlabs/cuda-oxide) checkout as a sibling
directory of this workspace (`Cargo.toml` uses path dependencies), with
`cargo-oxide` installed from it. `rust-toolchain.toml` pins the nightly
toolchain; rustup fetches it on first build.

```bash
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:$PATH"
cargo oxide doctor        # validates driver / toolkit / toolchain — run once
cargo oxide run --bin affine   # the vertical slice; ends at its ✅ banner
```

The full development loop, which is also the pre-push check:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets    # zero warnings is the bar
cargo test -p shannon-core -p shannon-rt -p shannon-autodiff \
           -p shannon-spatial -p shannon-cpu   # 102 tests
cargo test -p shannon-examples --lib           # 3 tests
```

The host tests need no GPU — only the demo binaries do. Workspace-wide
`cargo test` cannot link the example binaries (the PTX-bundle symbol exists
only under cargo-oxide's build; [docs/LIMITATIONS.md](docs/LIMITATIONS.md)
#12), which is why the two-command form above is canonical.

## The rules that keep this SDK honest

These are not style preferences. Each one guards a claim the README makes;
breaking one silently invalidates that claim.

1. **Every line of math lives in `shannon-core`.** The GPU and CPU backends
   are thin adapters expanding the same kernel rows. Arithmetic added to an
   adapter exists on one backend only, and the parity results stop meaning
   anything.
2. **Respect the output-type rule.** Forward kernels write through
   `DisjointSlice` (each thread owns its index); adjoint kernels read `&[T]`
   and accumulate through a `GradSink`, never `&mut`. This is the race-freedom
   argument ([SAFETY.md](SAFETY.md)); a kernel that bypasses it can compile
   and corrupt.
3. **Never relax a parity or gradient-check tolerance to make a test pass.**
   If a comparison fails, assume the change is wrong until proven otherwise.
   Marginal gradcheck failures get an epsilon sweep before the adjoint is
   suspected (f32 cancellation is real); they never get a looser bar.
4. **Between the first taped record and `backward`, touch taped buffers only
   through `ops::` calls and read-only launches.** The raw-launch hole
   ([docs/LIMITATIONS.md](docs/LIMITATIONS.md) #1) means a scatter launch that
   aliases a taped buffer compiles fine and produces silently wrong gradients.
5. **This project measures rather than claims.** Every number in the README
   must exist in [docs/RESULTS.md](docs/RESULTS.md) with hardware and
   methodology beside it. If you cannot measure something, say so instead of
   estimating.
6. **Demos end at a binary acceptance banner.** A demo that "mostly works" is
   a demo that fails; predicates are pass/fail, not vibes.

## Submitting a change

1. For behaviour or dependency changes, open an issue first — the design
   invariants above are cheap to discuss and expensive to unwind.
2. Run the full pre-push block above; it must be green.
3. If the change affects a measured number, re-run the relevant binary and
   update [docs/RESULTS.md](docs/RESULTS.md) in the same PR — the README is
   audited against it.
4. Add a [CHANGELOG.md](CHANGELOG.md) entry under `## [Unreleased]`.
5. If a [ROADMAP.md](ROADMAP.md) item opens or closes, update its checkbox.

There is no commit-message convention beyond keeping the history readable —
one logical change per commit.

## Lockfile policy

`Cargo.lock` is committed. This is an application-style workspace (nothing is
published to crates.io yet), so reproducible builds win over dependency
freshness.

## Licensing of contributions

Contributions are accepted under the repository's Apache-2.0 license. Do not
add assets without a clear license; `assets/README.md` shows the required
provenance format, and non-Apache material must be listed in [NOTICE](NOTICE).
