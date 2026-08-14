//! W3 — differentiable shape fitting: one shape becomes another because
//! gradients flowed backward through the GPU (Day-6 plan §5.8).
//!
//! Three parts, each the oracle for the next:
//!
//!   Part 0  GPU tape chain (affine → sin_map → sum_scalar) vs the ANALYTIC
//!           gradient — the only test where the tape itself is the thing
//!           under test, every adjoint already proven by its own gradcheck.
//!   Part 1  Stage 1: L2 with known correspondence under SGD. Convergence is
//!           a THEOREM (x ← x − lr·(x − t) contracts every residual by
//!           (1 − lr) per step) — so a failure here is broken machinery,
//!           never tuning.
//!   Part 2  Stage 2: symmetric Chamfer under Adam. Correspondence requeried
//!           each iteration (UNTAPED — the envelope theorem makes the held-
//!           correspondence gradient exact), gradients cross-checked against
//!           the CPU backend at iterations 0 and 100.
//!
//! Per-iteration choreography (Day-6 plan §4.2): MUTATE (upload, refit,
//! begin_iteration = zero + reset + SEED) → TAPE (shared borrows only) →
//! backward consumes the tape → STEP (&mut legal again: download, Adam, dump).

use anyhow::{Result, anyhow, ensure};
use shannon_autodiff::{Adam, Sgd, Tape};
use shannon_core::vec::{vec3s_as_f32s, vec3s_as_f32s_mut};
use shannon_core::{MeshQuery, Vec3};
use shannon_examples::obj::{read_obj, write_obj};
use shannon_examples::ops;
use shannon_rt::{Array, launch};
use shannon_spatial::Mesh;
use shannon_spatial::shapes::{icosphere, torus};
use std::path::PathBuf;

/// Queries stay well inside this radius on the demo shapes — a miss at a
/// check iteration means a broken refit, not geometry.
const MAX_DIST: f32 = 2.0;

/// Minimal LCG (Knuth MMIX constants), seed 42 — the house generator.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) & 0x00FF_FFFF) as f32 / 16_777_216.0
    }
}

enum TargetKind {
    Blob,
    Torus,
    Obj(PathBuf),
}

struct Args {
    iters: usize,
    lr1: f32,
    lr2: f32,
    subdiv: u32,
    dump_every: usize,
    out_dir: PathBuf,
    target: TargetKind,
    no_normalize: bool,
    smooth: f32,
}

fn parse_args() -> Args {
    let mut args = Args {
        iters: 200,
        lr1: 0.05,
        lr2: 0.01,
        subdiv: 3,
        dump_every: 20,
        out_dir: PathBuf::from("fit_frames"),
        target: TargetKind::Blob,
        no_normalize: false,
        smooth: 0.0,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--iters" => {
                args.iters = argv[i + 1].parse().expect("--iters N");
                i += 2;
            }
            "--lr1" => {
                args.lr1 = argv[i + 1].parse().expect("--lr1 F");
                i += 2;
            }
            "--lr2" => {
                args.lr2 = argv[i + 1].parse().expect("--lr2 F");
                i += 2;
            }
            "--subdiv" => {
                args.subdiv = argv[i + 1].parse().expect("--subdiv N");
                i += 2;
            }
            "--dump-every" => {
                args.dump_every = argv[i + 1].parse().expect("--dump-every N (0 = off)");
                i += 2;
            }
            "--out-dir" => {
                args.out_dir = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--target" => {
                args.target = match argv[i + 1].as_str() {
                    "blob" => TargetKind::Blob,
                    "torus" => TargetKind::Torus,
                    other => panic!("--target blob|torus (got {other}); OBJ via --target-obj"),
                };
                i += 2;
            }
            "--target-obj" => {
                args.target = TargetKind::Obj(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--no-normalize" => {
                args.no_normalize = true;
                i += 1;
            }
            "--smooth" => {
                args.smooth = argv[i + 1].parse().expect("--smooth F (0 = off)");
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
    args
}

/// The default acceptance target: a same-topology displaced icosphere —
/// t = n̂·(1 + 0.3·sin(3n̂ₓ)·cos(2n̂ᵧ)), radius ∈ [0.7, 1.3], C^∞. Same vertex
/// count as the source, so ZERO Chamfer loss is attainable (source verts =
/// target verts is a global minimum — exactly the configuration stage 1
/// converges to; "stage 1 is the oracle for stage 2", made literal).
fn blob(subdiv: u32) -> (Vec<Vec3>, Vec<i32>) {
    let (mut pts, idx) = icosphere(subdiv, 1.0);
    for p in pts.iter_mut() {
        let n = *p; // unit-sphere vertex
        let r = 1.0 + 0.3 * (3.0 * n.x).sin() * (2.0 * n.y).cos();
        *p = n * r;
    }
    (pts, idx)
}

/// max |gᵢ| over a flat f32 view — the iteration-0 tripwire's measurement.
fn max_abs(g: &[f32]) -> f32 {
    g.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Part 0 — the GPU tape chain vs the analytic gradient
// ─────────────────────────────────────────────────────────────────────────────

fn part0_chain_check() -> Result<()> {
    const N: usize = 64;
    const SCALE: f32 = 2.0;
    const BIAS: f32 = 0.5;

    let mut rng = Lcg::new(42);
    let x_host: Vec<f32> = (0..N).map(|_| rng.next_f32()).collect();

    let mut x = Array::from_slice(&x_host)?;
    let mut y1 = Array::<f32>::zeros(N)?;
    let mut y2 = Array::<f32>::zeros(N)?;
    let mut loss = Array::<f32>::zeros(1)?;
    x.requires_grad_(true)?;
    y1.requires_grad_(true)?;
    y2.requires_grad_(true)?;
    loss.requires_grad_(true)?;

    // MUTATE: zero + reset + seed, before anything records.
    ops::begin_iteration(&mut [], &mut [&mut x, &mut y1, &mut y2], &mut loss)?;

    // TAPE: the chain threads downgraded shared views forward.
    let mut tape = Tape::new();
    let y1v = ops::affine_op(&mut tape, &x, SCALE, BIAS, &mut y1)?;
    let y2v = ops::sin_map_op(&mut tape, y1v, &mut y2)?;
    ops::sum_op(&mut tape, y2v, &loss)?;
    ensure!(tape.len() == 3, "expected 3 records, got {}", tape.len());
    ensure!(
        tape.labels().collect::<Vec<_>>() == ["affine", "sin_map", "sum_scalar"],
        "record order wrong"
    );
    tape.backward()?; // replays sum_scalar, sin_map, affine — in that order

    // STEP: check against d/dx Σ sin(2x + 0.5) = 2·cos(2x + 0.5), exact form.
    let g = x.grad().ok_or_else(|| anyhow!("x has no grad"))?.to_vec()?;
    let l = loss.to_vec()?[0];
    let expect_l: f32 = x_host.iter().map(|&a| (a * SCALE + BIAS).sin()).sum();
    ensure!((l - expect_l).abs() < 1e-4, "chain forward: {l} vs {expect_l}");
    for i in 0..N {
        let want = SCALE * (x_host[i] * SCALE + BIAS).cos();
        ensure!(
            (g[i] - want).abs() <= 1e-3 * want.abs().max(1.0),
            "chain grad {i}: {} vs analytic {want}",
            g[i]
        );
    }
    println!("✓ tape chain (3 kernels, reverse replay, analytic match)");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Part 1 — stage 1: L2 + SGD, provably monotone
// ─────────────────────────────────────────────────────────────────────────────

fn part1_l2_sgd(args: &Args, source: &[Vec3], target_verts: &[Vec3]) -> Result<()> {
    let n = source.len();
    let mut host_params = source.to_vec();

    let mut params = Array::from_slice(&host_params)?;
    let target = Array::from_slice(target_verts)?;
    let mut loss = Array::<f32>::zeros(1)?;
    params.requires_grad_(true)?;
    loss.requires_grad_(true)?;

    let sgd = Sgd { lr: args.lr1 };
    let mut loss0 = 0.0f32;
    let mut prev = f32::INFINITY;

    for iter in 0..args.iters {
        // MUTATE
        params.copy_from_slice(&host_params)?;
        ops::begin_iteration(&mut [&mut params], &mut [], &mut loss)?;

        // TAPE
        let mut tape = Tape::new();
        ops::l2_loss_op(&mut tape, &params, &target, &loss)?;
        tape.backward()?;

        // STEP
        let l = loss.to_vec()?[0];
        let g = params.grad().ok_or_else(|| anyhow!("params has no grad"))?.to_vec()?;

        if iter == 0 {
            loss0 = l;
            // The iteration-0 tripwire: the stage-1 gradient is analytically
            // x − t ≠ 0, so zero gradient here is ALWAYS a seeding/plumbing
            // bug (a flat loss would PASS the monotone check — this is the
            // predicate that catches a missing seed in 1 iteration, not 200).
            ensure!(
                max_abs(vec3s_as_f32s(&g)) > 0.0,
                "iteration-0 tripwire: all gradients zero — seed/plumbing bug"
            );
        }
        ensure!(
            l <= prev * (1.0 + 1e-4),
            "stage 1 monotonicity violated at iter {iter}: {l} > {prev}"
        );
        prev = l;

        sgd.step(vec3s_as_f32s_mut(&mut host_params), vec3s_as_f32s(&g));

        if iter % 20 == 0 || iter + 1 == args.iters {
            println!("  stage1 iter {iter:3}  loss {l:.6e}");
        }
    }

    ensure!(
        prev <= 1e-4 * loss0,
        "stage 1: final loss {prev} not ≥4 orders below initial {loss0}"
    );
    println!("✓ stage 1: monotone, {loss0:.3e} → {prev:.3e} ({:.1} orders) over {} iters ({n} verts)",
        (loss0 / prev).log10(), args.iters);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Part 2 — stage 2: symmetric Chamfer + Adam
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn cross_backend_grad_check(
    iter: usize,
    host_params: &[Vec3],
    src_indices: &[i32],
    target_verts: &[Vec3],
    target_indices: &[i32],
    corr_ab: &[MeshQuery],
    corr_ba: &[MeshQuery],
    gpu_grad: &[Vec3],
) -> Result<()> {
    // Zero query misses at check iterations: everything is well inside
    // MAX_DIST on these shapes — a miss means a broken refit, not geometry.
    ensure!(
        corr_ab.iter().chain(corr_ba).all(|q| q.face >= 0),
        "iter {iter}: query miss during grad check — refit is lying"
    );

    // Recompute the FULL gradient on the host via the CPU backend adapters
    // over the same downloaded state, adj_loss seeded to [1.0].
    let adj_loss = [1.0f32];
    let mut host_grad = vec![Vec3::ZERO; host_params.len()];
    shannon_cpu::adj_chamfer_ab(
        host_params, corr_ab, target_verts, target_indices, &adj_loss, &mut host_grad,
    );
    shannon_cpu::adj_chamfer_ba(
        target_verts, corr_ba, host_params, src_indices, &adj_loss, &mut host_grad,
    );

    // Relative 1e-3 with max(1,·) denominators — NEVER bit-exact: float
    // atomics commit in nondeterministic order (gradcheck.rs:29).
    let (g, h) = (vec3s_as_f32s(gpu_grad), vec3s_as_f32s(&host_grad));
    for k in 0..g.len() {
        let denom = g[k].abs().max(h[k].abs()).max(1.0);
        ensure!(
            (g[k] - h[k]).abs() / denom <= 1e-3,
            "iter {iter}: GPU/CPU grad disagree at flat index {k}: {} vs {}",
            g[k],
            h[k]
        );
    }
    println!("  ✓ iter {iter}: GPU gradient matches CPU backend (rel 1e-3)");
    Ok(())
}

fn part2_chamfer_adam(
    args: &Args,
    source: (&[Vec3], &[i32]),
    target: (&[Vec3], &[i32]),
    enforce: bool,
) -> Result<()> {
    let (src_verts, src_idx) = source;
    let (tgt_verts, tgt_idx) = target;
    let (n_src, n_tgt) = (src_verts.len(), tgt_verts.len());
    let mut host_params = src_verts.to_vec();

    let mut src_mesh = Mesh::new(src_verts, src_idx)?;
    src_mesh.points_mut().requires_grad_(true)?;
    let tgt_mesh = Mesh::new(tgt_verts, tgt_idx)?;
    let tgt_verts_arr = Array::from_slice(tgt_verts)?;
    let mut corr_ab = Array::<MeshQuery>::zeros(n_src)?;
    let mut corr_ba = Array::<MeshQuery>::zeros(n_tgt)?;
    let mut loss = Array::<f32>::zeros(1)?;
    loss.requires_grad_(true)?;

    let mut adam = Adam::new(n_src * 3, args.lr2);
    let mut loss0 = 0.0f32;
    let mut running_min = f32::INFINITY;

    // Shrink-wrap regularizer (--smooth λ). Vertex-wise Chamfer pays nothing
    // for a stretched triangle bridging a concavity, so highly non-convex
    // targets (a cat's arched back, the gaps between a robot's limbs) stall
    // in "tenting" minima with every vertex on the surface but large faces
    // spanning the gaps. Pulling each vertex λ of the way toward its 1-ring
    // centroid after the optimizer step keeps the wrap evenly spread so the
    // next iteration's queries can sink it into the concavity. Host-side and
    // untaped — a projection between iterations, not part of the gradient.
    let neighbors: Vec<Vec<u32>> = if args.smooth > 0.0 {
        let mut nb = vec![Vec::new(); n_src];
        for tri in src_idx.chunks(3) {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            nb[a].extend([tri[1] as u32, tri[2] as u32]);
            nb[b].extend([tri[0] as u32, tri[2] as u32]);
            nb[c].extend([tri[0] as u32, tri[1] as u32]);
        }
        for l in &mut nb {
            l.sort_unstable();
            l.dedup();
        }
        nb
    } else {
        Vec::new()
    };

    if args.dump_every > 0 {
        std::fs::create_dir_all(&args.out_dir)?;
        write_obj(&args.out_dir.join("target.obj"), tgt_verts, tgt_idx)?;
    }

    for iter in 0..args.iters {
        // MUTATE: upload current params, refresh the source BVH, zero + seed.
        src_mesh.points_mut().copy_from_slice(&host_params)?;
        src_mesh.refit()?;
        ops::begin_iteration(&mut [src_mesh.points_mut()], &mut [], &mut loss)?;

        // TAPE. Correspondence is requeried EVERY iteration, untaped — the
        // envelope theorem makes the held-correspondence gradient exact; a
        // stale correspondence would optimize toward a frozen snapshot.
        let mut tape = Tape::new();
        tape.pause();
        launch!(mesh_query, dim = n_src,
            (tgt_mesh.nodes(), tgt_mesh.points(), tgt_mesh.indices(),
             src_mesh.points(), MAX_DIST, &mut corr_ab))?;
        launch!(mesh_query, dim = n_tgt,
            (src_mesh.nodes(), src_mesh.points(), src_mesh.indices(),
             &tgt_verts_arr, MAX_DIST, &mut corr_ba))?;
        tape.resume();

        let spts = src_mesh.points();
        ops::chamfer_ab_op(&mut tape, spts, &corr_ab, tgt_mesh.points(), tgt_mesh.indices(), &loss)?;
        ops::chamfer_ba_op(&mut tape, &tgt_verts_arr, &corr_ba, spts, src_mesh.indices(), &loss)?;
        tape.backward()?;

        // STEP
        let l = loss.to_vec()?[0];
        let g = src_mesh
            .points()
            .grad()
            .ok_or_else(|| anyhow!("source points have no grad"))?
            .to_vec()?;

        if iter == 0 {
            loss0 = l;
            // Iteration-0 tripwire: the initial sphere is nowhere on the
            // target surface, so the true gradient is nonzero.
            ensure!(
                max_abs(vec3s_as_f32s(&g)) > 0.0,
                "iteration-0 tripwire: all gradients zero — seed/plumbing bug"
            );
        }
        if iter == 0 || iter == 100 {
            cross_backend_grad_check(
                iter, &host_params, src_idx, tgt_verts, tgt_idx,
                &corr_ab.to_vec()?, &corr_ba.to_vec()?, &g,
            )?;
        }
        running_min = running_min.min(l);
        if enforce {
            // The envelope rule: Adam may wobble at correspondence switches;
            // it may not climb. (Strict monotonicity is stage 1's job.)
            ensure!(
                l <= 1.25 * running_min,
                "stage 2 envelope violated at iter {iter}: {l} > 1.25 × {running_min}"
            );
        }

        adam.step(vec3s_as_f32s_mut(&mut host_params), vec3s_as_f32s(&g));

        if args.smooth > 0.0 {
            let prev = host_params.clone();
            for (v, nbs) in host_params.iter_mut().zip(&neighbors) {
                let mut c = Vec3::ZERO;
                for &j in nbs {
                    c += prev[j as usize];
                }
                let c = c * (1.0 / nbs.len() as f32);
                *v += (c - *v) * args.smooth;
            }
        }

        if iter % 20 == 0 || iter + 1 == args.iters {
            println!("  stage2 iter {iter:3}  loss {l:.6e}");
        }
        if args.dump_every > 0 && iter % args.dump_every == 0 {
            write_obj(&args.out_dir.join(format!("fit_{iter:04}.obj")), &host_params, src_idx)?;
        }
    }

    if args.dump_every > 0 {
        write_obj(&args.out_dir.join("fit_final.obj"), &host_params, src_idx)?;
    }

    let l_final = running_min;
    if enforce {
        ensure!(
            l_final <= 1e-2 * loss0,
            "stage 2: final loss {l_final} not ≥2 orders below initial {loss0}"
        );
    }
    println!(
        "✓ stage 2: {loss0:.3e} → {l_final:.3e} ({:.1} orders) over {} iters ({n_src}→{n_tgt} verts)",
        (loss0 / l_final).log10(),
        args.iters
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args();

    println!("— Part 0: GPU tape chain check —");
    part0_chain_check()?;

    // Source is always the unit icosphere; stage 1's target is always the
    // blob (identity correspondence needs matching vertex counts — stage 1
    // is the oracle regardless of what stage 2 fits).
    let (src_verts, src_idx) = icosphere(args.subdiv, 1.0);
    let (blob_verts, blob_idx) = blob(args.subdiv);

    println!("— Part 1: stage 1, L2 + SGD (lr {}) —", args.lr1);
    part1_l2_sgd(&args, &src_verts, &blob_verts)?;

    // Stage 2 target: blob (acceptance, predicates enforced) or the showcase
    // shapes (curves printed, hard predicates waived — a genus-1 torus has a
    // nonzero Chamfer floor from a genus-0 source, by topology, not by bug).
    let (tgt_verts, tgt_idx, enforce) = match &args.target {
        TargetKind::Blob => (blob_verts, blob_idx, true),
        TargetKind::Torus => {
            let (v, i) = torus(24, 16, 1.0, 0.35);
            (v, i, false)
        }
        TargetKind::Obj(path) => {
            let (mut v, i) = read_obj(path)?;
            // A v-only point cloud has no surface to query — fail HERE with a
            // clear message, not 30 lines later with "query miss — refit is
            // lying" (which means something else entirely).
            ensure!(
                i.len() >= 3,
                "target OBJ {} has no triangles (point-cloud OBJ?)",
                path.display()
            );
            if args.no_normalize {
                println!("target OBJ used as-is (--no-normalize)");
            } else {
                // Arbitrary OBJs are rarely unit-sized at the origin, but the
                // source sphere is, and MAX_DIST = 2.0 assumes it. Center the
                // AABB and scale the max half-extent to 1.0 — sphere-sized.
                let (mut lo, mut hi) = (v[0], v[0]);
                for p in &v {
                    lo = lo.cw_min(*p);
                    hi = hi.cw_max(*p);
                }
                let center = (lo + hi) * 0.5;
                let half = (hi - lo) * 0.5;
                let extent = half.x.max(half.y).max(half.z);
                ensure!(extent > 0.0, "target OBJ is degenerate: zero extent");
                let s = 1.0 / extent;
                for p in v.iter_mut() {
                    *p = (*p - center) * s;
                }
                println!(
                    "normalized target: center ({:.3}, {:.3}, {:.3}) → origin, scale ×{:.4}  \
                     ({} verts, {} tris; --no-normalize to disable)",
                    center.x,
                    center.y,
                    center.z,
                    s,
                    v.len(),
                    i.len() / 3
                );
            }
            (v, i, false)
        }
    };
    println!("— Part 2: stage 2, symmetric Chamfer + Adam (lr {}) —", args.lr2);
    part2_chamfer_adam(&args, (&src_verts, &src_idx), (&tgt_verts, &tgt_idx), enforce)?;

    if enforce {
        println!("\n✅ SHAPE FIT ACCEPTANCE PASSED");
    } else {
        println!("\n(showcase target — acceptance predicates waived; run without --target for the acceptance form)");
    }
    Ok(())
}
