//! Procedural mesh generators (Day-5 plan §5.5): grid (the W2 demo floor),
//! icosphere (the W2 benchmark mesh), torus (tests + Day-6 fitting shapes).
//!
//! Correctness is pinned by the invariant tests (Euler characteristic,
//! sphericity, outward normals, no degenerate triangles) — not by eyeballing.
//! Warp ships Python UV-sphere/grid references (render_opengl.py:3568,
//! benchmarks/benchmark_mesh.py:119); icosphere and torus are written fresh.

use shannon_core::Vec3;
use std::collections::HashMap;
use std::f32::consts::TAU;

/// n×n quads on y = 0 over [−extent, extent]²: (n+1)² vertices, 2n² triangles,
/// wound for +y normals.
pub fn grid(n: usize, extent: f32) -> (Vec<Vec3>, Vec<i32>) {
    assert!(n >= 1, "grid needs at least one quad");
    let step = 2.0 * extent / n as f32;
    let mut points = Vec::with_capacity((n + 1) * (n + 1));
    for iz in 0..=n {
        for ix in 0..=n {
            points.push(Vec3::new(-extent + ix as f32 * step, 0.0, -extent + iz as f32 * step));
        }
    }
    let vid = |ix: usize, iz: usize| (iz * (n + 1) + ix) as i32;
    let mut indices = Vec::with_capacity(6 * n * n);
    for iz in 0..n {
        for ix in 0..n {
            // Both triangles wound so (e1 × e2).y > 0 — normals point up.
            indices.extend_from_slice(&[vid(ix, iz), vid(ix + 1, iz + 1), vid(ix + 1, iz)]);
            indices.extend_from_slice(&[vid(ix, iz), vid(ix, iz + 1), vid(ix + 1, iz + 1)]);
        }
    }
    (points, indices)
}

/// Icosahedron + `subdiv` rounds of edge-midpoint subdivision, every vertex
/// normalized to `radius`: V = 10·4ᵏ + 2, F = 20·4ᵏ.
///
/// The midpoint cache is keyed on the SORTED edge (i, j) so the two faces
/// sharing an edge reuse one midpoint vertex — that sharing is exactly what
/// makes V − E + F = 2 hold.
pub fn icosphere(subdiv: u32, radius: f32) -> (Vec<Vec3>, Vec<i32>) {
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;
    // The 12 icosahedron vertices (three orthogonal golden rectangles).
    let mut points: Vec<Vec3> = [
        (-1.0, t, 0.0),
        (1.0, t, 0.0),
        (-1.0, -t, 0.0),
        (1.0, -t, 0.0),
        (0.0, -1.0, t),
        (0.0, 1.0, t),
        (0.0, -1.0, -t),
        (0.0, 1.0, -t),
        (t, 0.0, -1.0),
        (t, 0.0, 1.0),
        (-t, 0.0, -1.0),
        (-t, 0.0, 1.0),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalize() * radius)
    .collect();
    // The 20 faces, outward-wound (canonical listing).
    let mut faces: Vec<[i32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    for _ in 0..subdiv {
        let mut cache: HashMap<(i32, i32), i32> = HashMap::new();
        let mut midpoint = |a: i32, b: i32, points: &mut Vec<Vec3>| -> i32 {
            let key = if a < b { (a, b) } else { (b, a) };
            *cache.entry(key).or_insert_with(|| {
                let m = (points[a as usize] + points[b as usize]) * 0.5;
                points.push(m.normalize() * radius);
                (points.len() - 1) as i32
            })
        };
        let mut next = Vec::with_capacity(faces.len() * 4);
        for [a, b, c] in faces {
            let ab = midpoint(a, b, &mut points);
            let bc = midpoint(b, c, &mut points);
            let ca = midpoint(c, a, &mut points);
            next.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        faces = next;
    }

    (points, faces.into_iter().flatten().collect())
}

/// Parametric torus: nu major × nv minor segments, both wrapped —
/// V = nu·nv, F = 2·nu·nv, χ = 0. Major radius `big_r`, tube radius `small_r`.
pub fn torus(nu: usize, nv: usize, big_r: f32, small_r: f32) -> (Vec<Vec3>, Vec<i32>) {
    assert!(nu >= 3 && nv >= 3, "torus needs at least 3 segments per direction");
    let mut points = Vec::with_capacity(nu * nv);
    for iu in 0..nu {
        let theta = TAU * iu as f32 / nu as f32; // major angle
        for iv in 0..nv {
            let phi = TAU * iv as f32 / nv as f32; // minor angle
            let ring = big_r + small_r * phi.cos();
            points.push(Vec3::new(ring * theta.cos(), small_r * phi.sin(), ring * theta.sin()));
        }
    }
    let vid = |iu: usize, iv: usize| ((iu % nu) * nv + (iv % nv)) as i32;
    let mut indices = Vec::with_capacity(6 * nu * nv);
    for iu in 0..nu {
        for iv in 0..nv {
            // Outward winding (verified against the surface normal at θ=φ=0).
            let (v00, v01) = (vid(iu, iv), vid(iu, iv + 1));
            let (v10, v11) = (vid(iu + 1, iv), vid(iu + 1, iv + 1));
            indices.extend_from_slice(&[v00, v01, v11]);
            indices.extend_from_slice(&[v00, v11, v10]);
        }
    }
    (points, indices)
}
