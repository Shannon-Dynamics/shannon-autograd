//! Minimal Wavefront OBJ I/O — dependency-free, like `image.rs` (Day-5 plan
//! §5.10). `v` and `f` lines only; no materials, no groups, no normals.
//!
//! The writer prints `f32` via `{}` Display, whose shortest-round-trip
//! guarantee makes write → read bit-exact. Indices are 1-BASED in the file
//! (the OBJ convention) and 0-based in memory — both functions convert.

use anyhow::{Context, Result, bail};
use shannon_core::Vec3;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write a triangle mesh: `v x y z` lines then 1-based `f a b c` lines.
pub fn write_obj(path: &Path, points: &[Vec3], indices: &[i32]) -> Result<()> {
    anyhow::ensure!(
        indices.len().is_multiple_of(3),
        "indices must be 3 per triangle"
    );
    let file =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(file);
    for p in points {
        writeln!(w, "v {} {} {}", p.x, p.y, p.z)?;
    }
    for tri in indices.chunks_exact(3) {
        writeln!(w, "f {} {} {}", tri[0] + 1, tri[1] + 1, tri[2] + 1)?;
    }
    w.flush()?;
    Ok(())
}

/// Write a point cloud as a v-only OBJ — valid OBJ; viewers render vertices.
pub fn write_obj_points(path: &Path, points: &[Vec3]) -> Result<()> {
    let file =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(file);
    for p in points {
        writeln!(w, "v {} {} {}", p.x, p.y, p.z)?;
    }
    w.flush()?;
    Ok(())
}

/// Read `v`/`f` lines. Face tokens are split at the first `/` (vt/vn refs
/// ignored); faces with more than 3 vertices are fan-triangulated; negative
/// (relative) or out-of-range indices are errors; every other line type is
/// silently skipped.
pub fn read_obj(path: &Path) -> Result<(Vec<Vec3>, Vec<i32>)> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut points: Vec<Vec3> = Vec::new();
    let mut indices: Vec<i32> = Vec::new();

    for (lineno, line) in text.lines().enumerate() {
        let mut tokens = line.split_whitespace();
        match tokens.next() {
            Some("v") => {
                let mut coord = || -> Result<f32> {
                    tokens
                        .next()
                        .with_context(|| format!("line {}: v needs 3 coords", lineno + 1))?
                        .parse()
                        .with_context(|| format!("line {}: bad coordinate", lineno + 1))
                };
                points.push(Vec3::new(coord()?, coord()?, coord()?));
            }
            Some("f") => {
                let mut face: Vec<i32> = Vec::with_capacity(4);
                for token in tokens {
                    // "7", "7/1", "7/1/3", "7//3" all reference vertex 7.
                    let vertex_ref = token.split('/').next().unwrap_or(token);
                    let idx: i64 = vertex_ref
                        .parse()
                        .with_context(|| format!("line {}: bad face index", lineno + 1))?;
                    if idx < 1 {
                        bail!(
                            "line {}: index {idx} — negative/relative indices unsupported",
                            lineno + 1
                        );
                    }
                    if idx as usize > points.len() {
                        bail!("line {}: index {idx} out of range", lineno + 1);
                    }
                    face.push((idx - 1) as i32);
                }
                if face.len() < 3 {
                    bail!("line {}: face needs at least 3 vertices", lineno + 1);
                }
                // Fan-triangulate anything beyond a triangle.
                for k in 1..face.len() - 1 {
                    indices.extend_from_slice(&[face[0], face[k], face[k + 1]]);
                }
            }
            _ => {} // comments, vn/vt/o/g/s/mtllib/usemtl, blank lines
        }
    }
    Ok((points, indices))
}

// Inline rather than tests/: `cargo test` on an integration test also builds
// this package's BINARIES, which only link under cargo-oxide (the PTX bundle
// symbol). `cargo test -p shannon-examples --lib` builds just this library.
#[cfg(test)]
mod tests {
    use super::*;
    use shannon_spatial::shapes::icosphere;

    /// The writer's `{}` Display guarantees shortest-round-trip floats, so
    /// read-back must be BIT-exact.
    #[test]
    fn round_trip_is_bit_exact() {
        let (points, indices) = icosphere(2, 1.3);
        let path =
            std::env::temp_dir().join(format!("shannon_obj_roundtrip_{}.obj", std::process::id()));

        write_obj(&path, &points, &indices).unwrap();
        let (points2, indices2) = read_obj(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(points.len(), points2.len());
        for (a, b) in points.iter().zip(&points2) {
            assert_eq!(a, b, "vertex round-trip must be bit-exact");
        }
        assert_eq!(indices, indices2);
    }

    #[test]
    fn reader_handles_slashes_fans_and_junk() {
        let path =
            std::env::temp_dir().join(format!("shannon_obj_reader_{}.obj", std::process::id()));
        std::fs::write(
            &path,
            "# comment\no thing\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvn 0 0 1\nf 1/1 2/2/1 3//1 4\ns off\n",
        )
        .unwrap();
        let (points, indices) = read_obj(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(points.len(), 4);
        // One quad → fan → two triangles (0,1,2) and (0,2,3).
        assert_eq!(indices, vec![0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn reader_rejects_bad_indices() {
        let path = std::env::temp_dir().join(format!("shannon_obj_bad_{}.obj", std::process::id()));
        std::fs::write(&path, "v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 9\n").unwrap();
        assert!(read_obj(&path).is_err(), "out-of-range index must error");
        std::fs::write(&path, "v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 -1\n").unwrap();
        assert!(read_obj(&path).is_err(), "negative index must error");
        std::fs::remove_file(&path).ok();
    }
}
