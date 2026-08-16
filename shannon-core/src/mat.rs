//! `Mat33` — row-major 3×3 matrix. Needed on Day 3 (W4 argument shapes).

use crate::vec::Vec3;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat33 {
    /// Row-major: `m[row][col]`. 36 bytes.
    pub m: [[f32; 3]; 3],
}

impl Mat33 {
    pub const IDENTITY: Self = Self {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    pub const ZERO: Self = Self { m: [[0.0; 3]; 3] };

    #[inline(always)]
    pub fn from_rows(r0: Vec3, r1: Vec3, r2: Vec3) -> Self {
        Self {
            m: [[r0.x, r0.y, r0.z], [r1.x, r1.y, r1.z], [r2.x, r2.y, r2.z]],
        }
    }

    #[inline(always)]
    pub fn transpose(self) -> Self {
        let m = self.m;
        Self {
            m: [
                [m[0][0], m[1][0], m[2][0]],
                [m[0][1], m[1][1], m[2][1]],
                [m[0][2], m[1][2], m[2][2]],
            ],
        }
    }

    #[inline(always)]
    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        Vec3::new(self.row(0).dot(v), self.row(1).dot(v), self.row(2).dot(v))
    }

    #[inline(always)]
    pub fn mul_mat(self, o: Self) -> Self {
        let mut out = Self::ZERO;
        // Row-major triple loop; unrolled fully by the compiler at 3×3.
        let mut r = 0;
        while r < 3 {
            let mut c = 0;
            while c < 3 {
                out.m[r][c] =
                    self.m[r][0] * o.m[0][c] + self.m[r][1] * o.m[1][c] + self.m[r][2] * o.m[2][c];
                c += 1;
            }
            r += 1;
        }
        out
    }

    #[inline(always)]
    pub fn determinant(self) -> f32 {
        let m = self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    /// Inverse via the adjugate. Returns ZERO when `|det| < EPS` — no panic,
    /// because panics are stripped on device.
    #[inline(always)]
    pub fn inverse(self) -> Self {
        let det = self.determinant();
        if crate::math::abs(det) < crate::EPS {
            return Self::ZERO;
        }
        let inv = 1.0 / det;
        let m = self.m;
        Self {
            m: [
                [
                    (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
                    (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
                    (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
                ],
                [
                    (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
                    (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
                    (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
                ],
                [
                    (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
                    (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
                    (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
                ],
            ],
        }
    }

    /// Returns `Vec3::ZERO` out of range — no panic on device.
    #[inline(always)]
    pub fn row(self, i: usize) -> Vec3 {
        if i < 3 {
            Vec3::new(self.m[i][0], self.m[i][1], self.m[i][2])
        } else {
            Vec3::ZERO
        }
    }

    /// Returns `Vec3::ZERO` out of range — no panic on device.
    #[inline(always)]
    pub fn col(self, i: usize) -> Vec3 {
        if i < 3 {
            Vec3::new(self.m[0][i], self.m[1][i], self.m[2][i])
        } else {
            Vec3::ZERO
        }
    }
}
