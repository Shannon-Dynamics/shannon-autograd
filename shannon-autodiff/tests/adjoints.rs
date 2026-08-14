//! Every week-1 adjoint validated against finite differences BEFORE any GPU
//! build exists (Day-6 plan §5.4; week-1 plan §11.3: "no adjoint is
//! considered written until gradcheck passes at 1e-4 relative tolerance").
//!
//! Skeleton per adjoint: LCG-seeded inputs (seed 42, house convention);
//! analytic gradient computed by running the adjoint through the Host sinks
//! with seed ḡ = 1 per output; compared against `gradcheck` of the scalarized
//! forward (f = Σ yᵢ, whose per-element incoming gradient is exactly 1).

use shannon_autodiff::gradcheck;
use shannon_core::adjoint::*;
use shannon_core::loss::*;
use shannon_core::mesh::MeshQuery;
use shannon_core::vec::vec3s_as_f32s;
use shannon_core::{GradSink, Vec3};
use shannon_rt::{HostGradF32, HostGradVec3};

/// Minimal LCG (Knuth MMIX constants), 24-bit mantissa → uniform [0, 1).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) & 0x00FF_FFFF) as f32 / 16_777_216.0
    }
    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
    fn vec3(&mut self, lo: f32, hi: f32) -> Vec3 {
        Vec3::new(self.range(lo, hi), self.range(lo, hi), self.range(lo, hi))
    }
}

const N: usize = 16;
const EPS_FD: f32 = 1e-3;
const TOL: f32 = 1e-3; // f32 central differences on multi-input sums

/// Run an f32 adjoint through the host sink with ḡ = 1 seeds, returning the
/// accumulated analytic gradient buffer.
fn analytic_f32(n: usize, run: impl Fn(&HostGradF32)) -> Vec<f32> {
    let mut buf = vec![0.0f32; n];
    let sink = HostGradF32::new(&mut buf);
    run(&sink);
    buf
}

fn analytic_vec3(n: usize, run: impl Fn(&HostGradVec3)) -> Vec<Vec3> {
    let mut buf = vec![Vec3::ZERO; n];
    let sink = HostGradVec3::new(&mut buf);
    run(&sink);
    buf
}

// ─────────────────────────────────────────────────────────────────────────────
// The twelve, in §11.4 order
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn adjoint_01_add() {
    let mut rng = Lcg::new(42);
    let a: Vec<f32> = (0..N).map(|_| rng.range(-2.0, 2.0)).collect();
    let b: Vec<f32> = (0..N).map(|_| rng.range(-2.0, 2.0)).collect();
    let ones = vec![1.0f32; N];
    let ga = analytic_f32(N, |s| (0..N).for_each(|i| adj_add_at(i, &ones, s, &HostGradF32Null)));
    let gb = analytic_f32(N, |s| (0..N).for_each(|i| adj_add_at(i, &ones, &HostGradF32Null, s)));
    // f(a) = Σ (aᵢ + bᵢ) with b fixed, then symmetrically for b.
    gradcheck(|v| v.iter().zip(&b).map(|(x, y)| x + y).sum(), &a, &ga, EPS_FD, TOL).unwrap();
    gradcheck(|v| a.iter().zip(v).map(|(x, y)| x + y).sum(), &b, &gb, EPS_FD, TOL).unwrap();
}

#[test]
fn adjoint_02_sub() {
    let mut rng = Lcg::new(43);
    let a: Vec<f32> = (0..N).map(|_| rng.range(-2.0, 2.0)).collect();
    let b: Vec<f32> = (0..N).map(|_| rng.range(-2.0, 2.0)).collect();
    let ones = vec![1.0f32; N];
    let ga = analytic_f32(N, |s| (0..N).for_each(|i| adj_sub_at(i, &ones, s, &HostGradF32Null)));
    let gb = analytic_f32(N, |s| (0..N).for_each(|i| adj_sub_at(i, &ones, &HostGradF32Null, s)));
    gradcheck(|v| v.iter().zip(&b).map(|(x, y)| x - y).sum(), &a, &ga, EPS_FD, TOL).unwrap();
    gradcheck(|v| a.iter().zip(v).map(|(x, y)| x - y).sum(), &b, &gb, EPS_FD, TOL).unwrap();
}

#[test]
fn adjoint_03_mul() {
    let mut rng = Lcg::new(44);
    let a: Vec<f32> = (0..N).map(|_| rng.range(-2.0, 2.0)).collect();
    let b: Vec<f32> = (0..N).map(|_| rng.range(-2.0, 2.0)).collect();
    let ones = vec![1.0f32; N];
    let ga =
        analytic_f32(N, |s| (0..N).for_each(|i| adj_mul_at(i, &a, &b, &ones, s, &HostGradF32Null)));
    let gb =
        analytic_f32(N, |s| (0..N).for_each(|i| adj_mul_at(i, &a, &b, &ones, &HostGradF32Null, s)));
    gradcheck(|v| v.iter().zip(&b).map(|(x, y)| x * y).sum(), &a, &ga, EPS_FD, TOL).unwrap();
    gradcheck(|v| a.iter().zip(v).map(|(x, y)| x * y).sum(), &b, &gb, EPS_FD, TOL).unwrap();
}

#[test]
fn adjoint_04_div_threads_forward_output() {
    let mut rng = Lcg::new(45);
    let a: Vec<f32> = (0..N).map(|_| rng.range(-2.0, 2.0)).collect();
    // b bounded away from zero — division singularity is not under test here.
    let b: Vec<f32> = (0..N).map(|_| rng.range(0.5, 2.0)).collect();
    let y: Vec<f32> = (0..N).map(|i| a[i] / b[i]).collect(); // the threaded forward output
    let ones = vec![1.0f32; N];
    let ga = analytic_f32(N, |s| {
        (0..N).for_each(|i| adj_div_at(i, &b, &y, &ones, s, &HostGradF32Null))
    });
    let gb = analytic_f32(N, |s| {
        (0..N).for_each(|i| adj_div_at(i, &b, &y, &ones, &HostGradF32Null, s))
    });
    gradcheck(|v| v.iter().zip(&b).map(|(x, y)| x / y).sum(), &a, &ga, EPS_FD, TOL).unwrap();
    gradcheck(|v| a.iter().zip(v).map(|(x, y)| x / y).sum(), &b, &gb, EPS_FD, TOL).unwrap();
}

/// The oracle must FAIL on the canonical adj_div bug (dropping the /b, i.e.
/// b̄ −= ḡ·y instead of ḡ·y/b) — otherwise these tests prove nothing
/// (the Day-1 `detects_a_wrong_gradient` discipline).
#[test]
fn adjoint_04_div_wrong_variant_is_caught() {
    let mut rng = Lcg::new(46);
    let a: Vec<f32> = (0..N).map(|_| rng.range(-2.0, 2.0)).collect();
    let b: Vec<f32> = (0..N).map(|_| rng.range(0.4, 0.9)).collect(); // /b matters: b ≠ 1
    let y: Vec<f32> = (0..N).map(|i| a[i] / b[i]).collect();
    let wrong: Vec<f32> = (0..N).map(|i| -y[i]).collect(); // missing /b
    assert!(
        gradcheck(|v| a.iter().zip(v).map(|(x, y)| x / y).sum(), &b, &wrong, EPS_FD, TOL).is_err(),
        "gradcheck failed to catch the dropped-division adj_div bug"
    );
}

#[test]
fn adjoint_05_neg() {
    let mut rng = Lcg::new(47);
    let a: Vec<f32> = (0..N).map(|_| rng.range(-2.0, 2.0)).collect();
    let ones = vec![1.0f32; N];
    let ga = analytic_f32(N, |s| (0..N).for_each(|i| adj_neg_at(i, &ones, s)));
    gradcheck(|v| v.iter().map(|x| -x).sum(), &a, &ga, EPS_FD, TOL).unwrap();
}

#[test]
fn adjoint_06_sin() {
    let mut rng = Lcg::new(48);
    let a: Vec<f32> = (0..N).map(|_| rng.range(-3.0, 3.0)).collect();
    let ones = vec![1.0f32; N];
    let ga = analytic_f32(N, |s| (0..N).for_each(|i| adj_sin_at(i, &a, &ones, s)));
    gradcheck(|v| v.iter().map(|x| x.sin()).sum(), &a, &ga, EPS_FD, TOL).unwrap();
}

#[test]
fn adjoint_07_cos() {
    let mut rng = Lcg::new(49);
    let a: Vec<f32> = (0..N).map(|_| rng.range(-3.0, 3.0)).collect();
    let ones = vec![1.0f32; N];
    let ga = analytic_f32(N, |s| (0..N).for_each(|i| adj_cos_at(i, &a, &ones, s)));
    gradcheck(|v| v.iter().map(|x| x.cos()).sum(), &a, &ga, EPS_FD, TOL).unwrap();
}

#[test]
fn adjoint_08_sqrt_threads_forward_output() {
    let mut rng = Lcg::new(50);
    let a: Vec<f32> = (0..N).map(|_| rng.range(0.25, 4.0)).collect();
    let y: Vec<f32> = a.iter().map(|x| x.sqrt()).collect(); // threaded forward output
    let ones = vec![1.0f32; N];
    let ga = analytic_f32(N, |s| (0..N).for_each(|i| adj_sqrt_at(i, &y, &ones, s)));
    gradcheck(|v| v.iter().map(|x| x.sqrt()).sum(), &a, &ga, EPS_FD, TOL).unwrap();
}

/// √ at the singularity: zero gradient, finite, no NaN (Day-6 pitfall 9 class).
#[test]
fn adjoint_08_sqrt_at_zero_is_finite() {
    let y = [0.0f32];
    let ones = [1.0f32];
    let ga = analytic_f32(1, |s| adj_sqrt_at(0, &y, &ones, s));
    assert!(ga[0].is_finite());
    assert_eq!(ga[0], 0.0, "subgradient convention: zero gradient at the singularity");
}

#[test]
fn adjoint_09_dot() {
    let mut rng = Lcg::new(51);
    let a: Vec<Vec3> = (0..N).map(|_| rng.vec3(-2.0, 2.0)).collect();
    let b: Vec<Vec3> = (0..N).map(|_| rng.vec3(-2.0, 2.0)).collect();
    let ones = vec![1.0f32; N];
    let ga = analytic_vec3(N, |s| {
        (0..N).for_each(|i| adj_dot_at(i, &a, &b, &ones, s, &HostGradVec3Null))
    });
    // f over the flattened a (3N scalars): Σᵢ aᵢ·bᵢ.
    let b_flat = b.clone();
    gradcheck(
        |v| {
            (0..N)
                .map(|i| Vec3::new(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]).dot(b_flat[i]))
                .sum()
        },
        vec3s_as_f32s(&a),
        vec3s_as_f32s(&ga),
        EPS_FD,
        TOL,
    )
    .unwrap();
}

#[test]
fn adjoint_10_length() {
    let mut rng = Lcg::new(52);
    // Bounded away from the origin: the singular case has its own test below.
    let v: Vec<Vec3> = (0..N).map(|_| rng.vec3(0.5, 2.0)).collect();
    let ones = vec![1.0f32; N];
    let gv = analytic_vec3(N, |s| (0..N).for_each(|i| adj_length_at(i, &v, &ones, s)));
    // eps 1e-2, not 1e-3: the objective is a sum of 16 O(1) lengths, so a
    // 1e-3 perturbation sits at f32 cancellation level relative to the base
    // sum. The gradcheck discipline (gradcheck.rs:33): sweep eps BEFORE
    // suspecting the adjoint — and at 1e-2 the check passes at 1e-3 relative.
    gradcheck(
        |x| (0..N).map(|i| Vec3::new(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]).length()).sum(),
        vec3s_as_f32s(&v),
        vec3s_as_f32s(&gv),
        1e-2,
        TOL,
    )
    .unwrap();
}

#[test]
fn adjoint_11_normalize() {
    let mut rng = Lcg::new(53);
    let v: Vec<Vec3> = (0..N).map(|_| rng.vec3(0.5, 2.0)).collect();
    // Weighted sum of normalized components so the tangent projection matters:
    // f = Σᵢ wᵢ · n̂(vᵢ), with per-element incoming gradient ḡᵢ = wᵢ.
    let w: Vec<Vec3> = (0..N).map(|_| rng.vec3(-1.0, 1.0)).collect();
    let gv = analytic_vec3(N, |s| (0..N).for_each(|i| adj_normalize_at(i, &v, &w, s)));
    let w2 = w.clone();
    gradcheck(
        |x| {
            (0..N)
                .map(|i| {
                    let n = Vec3::new(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]).normalize();
                    w2[i].dot(n)
                })
                .sum()
        },
        vec3s_as_f32s(&v),
        vec3s_as_f32s(&gv),
        EPS_FD,
        TOL,
    )
    .unwrap();
}

/// The two origin-singular adjoints at v = (0,0,0): zero gradient, no NaN.
/// One NaN scatter-added into a grad buffer poisons everything downstream —
/// this is the whole-fit-dies-in-one-iteration failure mode (pitfall 9).
#[test]
fn adjoints_10_11_zero_vector_is_finite() {
    let v = [Vec3::ZERO];
    let ones = [1.0f32];
    let seed = [Vec3::ONE];
    let g_len = analytic_vec3(1, |s| adj_length_at(0, &v, &ones, s));
    let g_nrm = analytic_vec3(1, |s| adj_normalize_at(0, &v, &seed, s));
    for g in [g_len[0], g_nrm[0]] {
        assert!(g.x.is_finite() && g.y.is_finite() && g.z.is_finite());
        assert_eq!(g, Vec3::ZERO, "subgradient convention at the origin");
    }
}

#[test]
fn adjoint_12_gather_scatter_accumulates_on_collision() {
    // Forward: y[i] = x[idx[i]] with idx = [0, 1, 1, 2] — slot 1 is hit twice.
    // The adjoint must ACCUMULATE both contributions, not overwrite.
    // x values are O(0.1): with O(10) values the base sum drowns a 1e-3
    // perturbation in f32 rounding (the function is linear — FD error here is
    // pure cancellation, and shrinking the base kills it).
    let idx = [0i32, 1, 1, 2];
    let x = [0.10f32, 0.20, 0.30];
    let adj_y = [1.0f32, 2.0, 3.0, 4.0];
    let gx = analytic_f32(3, |s| (0..idx.len()).for_each(|i| adj_gather_at(i, &idx, &adj_y, s)));
    assert_eq!(gx, vec![1.0, 5.0, 4.0], "collided slot must hold the SUM (2+3)");
    // And the FD cross-check of the same structure:
    let idx2 = idx;
    let ay = adj_y;
    gradcheck(
        |v| (0..idx2.len()).map(|i| ay[i] * v[idx2[i] as usize]).sum(),
        &x,
        &gx,
        EPS_FD,
        TOL,
    )
    .unwrap();
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss adjoints (Day-6 plan §3.2) over a 4-triangle mini-mesh, correspondence
// held fixed — the envelope-theorem gradients checked against raw FD.
// ─────────────────────────────────────────────────────────────────────────────

/// A little tent: 4 triangles over 5 vertices.
fn mini_mesh() -> (Vec<Vec3>, Vec<i32>) {
    let points = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.5, 0.6, 0.5),
    ];
    let indices = vec![0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4];
    (points, indices)
}

#[test]
fn loss_adj_l2_matches_fd() {
    let mut rng = Lcg::new(54);
    let x: Vec<Vec3> = (0..8).map(|_| rng.vec3(-1.0, 1.0)).collect();
    let t: Vec<Vec3> = (0..8).map(|_| rng.vec3(-1.0, 1.0)).collect();
    let adj_loss = [1.0f32];
    let gx = analytic_vec3(8, |s| (0..8).for_each(|i| adj_l2_at(i, &x, &t, &adj_loss, s)));
    let t2 = t.clone();
    gradcheck(
        |v| {
            (0..8)
                .map(|i| {
                    0.5 * (Vec3::new(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]) - t2[i]).length_sq()
                })
                .sum()
        },
        vec3s_as_f32s(&x),
        vec3s_as_f32s(&gx),
        EPS_FD,
        TOL,
    )
    .unwrap();
}

#[test]
fn loss_adj_chamfer_ab_matches_fd() {
    // Held correspondence: source vertices against fixed points on the tent.
    let (tpoints, tindices) = mini_mesh();
    let mut rng = Lcg::new(55);
    let src: Vec<Vec3> = (0..6).map(|_| rng.vec3(-0.5, 1.5)).collect();
    let corr: Vec<MeshQuery> = (0i32..6)
        .map(|i| MeshQuery { face: i % 4, u: 0.3, v: 0.4, dist: 0.0 })
        .collect();
    let adj_loss = [1.0f32];
    let gs = analytic_vec3(6, |s| {
        (0..6).for_each(|i| adj_chamfer_ab_at(i, &src, &corr, &tpoints, &tindices, &adj_loss, s))
    });
    let (tp, ti, c) = (tpoints.clone(), tindices.clone(), corr.clone());
    gradcheck(
        |v| {
            (0..6)
                .map(|i| {
                    let x = Vec3::new(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
                    let q = c[i];
                    let p = shannon_core::mesh::mesh_eval_position(&tp, &ti, q.face, q.u, q.v);
                    0.5 * (x - p).length_sq()
                })
                .sum()
        },
        vec3s_as_f32s(&src),
        vec3s_as_f32s(&gs),
        EPS_FD,
        TOL,
    )
    .unwrap();
}

#[test]
fn loss_adj_chamfer_ba_matches_fd() {
    // The gather→scatter: gradient w.r.t. the SOURCE mesh points, with several
    // target verts referencing shared source triangles (collisions exercised).
    let (spoints, sindices) = mini_mesh();
    let mut rng = Lcg::new(56);
    let tverts: Vec<Vec3> = (0..6).map(|_| rng.vec3(-0.5, 1.5)).collect();
    let corr: Vec<MeshQuery> = (0i32..6)
        .map(|i| MeshQuery { face: i % 3, u: 0.25, v: 0.5, dist: 0.0 })
        .collect();
    let adj_loss = [1.0f32];
    let gp = analytic_vec3(spoints.len(), |s| {
        (0..6).for_each(|i| adj_chamfer_ba_at(i, &tverts, &corr, &spoints, &sindices, &adj_loss, s))
    });
    let (si, tv, c) = (sindices.clone(), tverts.clone(), corr.clone());
    let n_pts = spoints.len();
    gradcheck(
        |v| {
            let pts: Vec<Vec3> =
                (0..n_pts).map(|k| Vec3::new(v[k * 3], v[k * 3 + 1], v[k * 3 + 2])).collect();
            (0..6)
                .map(|i| {
                    let q = c[i];
                    let p = shannon_core::mesh::mesh_eval_position(&pts, &si, q.face, q.u, q.v);
                    0.5 * (p - tv[i]).length_sq()
                })
                .sum()
        },
        vec3s_as_f32s(&spoints),
        vec3s_as_f32s(&gp),
        EPS_FD,
        TOL,
    )
    .unwrap();
}

#[test]
fn loss_chamfer_miss_contributes_nothing() {
    let (tpoints, tindices) = mini_mesh();
    let src = [Vec3::new(9.0, 9.0, 9.0)];
    let corr = [MeshQuery::miss()];
    let adj_loss = [1.0f32];
    assert_eq!(chamfer_ab_term_at(0, &src, &corr, &tpoints, &tindices), 0.0);
    let gs = analytic_vec3(1, |s| {
        adj_chamfer_ab_at(0, &src, &corr, &tpoints, &tindices, &adj_loss, s)
    });
    assert_eq!(gs[0], Vec3::ZERO, "a missed query must produce no gradient");
}

#[test]
fn loss_adj_sum_broadcasts() {
    let adj_out = [2.5f32];
    let gx = analytic_f32(4, |s| (0..4).for_each(|i| adj_sum_at(i, &adj_out, s)));
    assert_eq!(gx, vec![2.5; 4]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Null sinks — "this operand's gradient is not under test".
// ─────────────────────────────────────────────────────────────────────────────

struct HostGradF32Null;
impl GradSink<f32> for HostGradF32Null {
    fn accumulate(&self, _i: usize, _g: f32) {}
}
struct HostGradVec3Null;
impl GradSink<Vec3> for HostGradVec3Null {
    fn accumulate(&self, _i: usize, _g: Vec3) {}
}
