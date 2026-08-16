//! `Vec2`, `Vec3`, `Vec4` — plain `Copy` structs, componentwise operators.
//!
//! `#[repr(C)]` is REQUIRED, not stylistic — Day-6 adjoint code accumulates
//! component-wise via pointer arithmetic assuming fields at their declared
//! byte offsets. `#[repr(Rust)]` permits field reordering. Week-1 plan §9.2.

use crate::math;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

// ─────────────────────────────────────────────────────────────────────────────
// Vec3 — the workhorse
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    #[inline(always)]
    pub const fn splat(v: f32) -> Self {
        Self::new(v, v, v)
    }

    #[inline(always)]
    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    #[inline(always)]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }
    #[inline(always)]
    pub fn length(self) -> f32 {
        math::sqrt(self.length_sq())
    }

    #[inline(always)]
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// Guarded against the zero vector — returns ZERO rather than NaN.
    /// The unguarded form is a documented source of silent NaN propagation.
    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > crate::EPS {
            self * (1.0 / len)
        } else {
            Self::ZERO
        }
    }

    #[inline(always)]
    pub fn cw_min(self, o: Self) -> Self {
        Self::new(
            math::min(self.x, o.x),
            math::min(self.y, o.y),
            math::min(self.z, o.z),
        )
    }
    #[inline(always)]
    pub fn cw_max(self, o: Self) -> Self {
        Self::new(
            math::max(self.x, o.x),
            math::max(self.y, o.y),
            math::max(self.z, o.z),
        )
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::new(math::abs(self.x), math::abs(self.y), math::abs(self.z))
    }

    /// Runtime component access. Returns 0.0 out of range — no panic, because
    /// panics are stripped on device and would become a GPU trap.
    #[inline(always)]
    pub fn component(self, i: usize) -> f32 {
        match i {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => 0.0,
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Neg for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}
impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}
impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: f32) -> Self {
        Self::new(self.x / s, self.y / s, self.z / s)
    }
}
impl AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl SubAssign for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

/// View a Vec3 slice as a flat f32 slice — the host optimizer's view of Vec3
/// parameters (Day-6 plan §5.5).
///
/// SAFETY: #[repr(C)] Vec3 is exactly 3 consecutive f32s (12 bytes, align 4) —
/// the same layout fact HostGradVec3 relies on (shannon-rt/src/grad.rs).
pub fn vec3s_as_f32s(v: &[Vec3]) -> &[f32] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() * 3) }
}

/// Mutable form of [`vec3s_as_f32s`]. Same SAFETY argument; the exclusive
/// borrow is consumed for the lifetime of the returned slice.
pub fn vec3s_as_f32s_mut(v: &mut [Vec3]) -> &mut [f32] {
    unsafe { core::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut f32, v.len() * 3) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vec2
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };

    #[inline(always)]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    #[inline(always)]
    pub const fn splat(v: f32) -> Self {
        Self::new(v, v)
    }
    #[inline(always)]
    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y
    }
    #[inline(always)]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }
    #[inline(always)]
    pub fn length(self) -> f32 {
        math::sqrt(self.length_sq())
    }
    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > crate::EPS {
            self * (1.0 / len)
        } else {
            Self::ZERO
        }
    }
}

impl Add for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y)
    }
}
impl Sub for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y)
    }
}
impl Neg for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}
impl Mul<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }
}
impl Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline(always)]
    fn mul(self, v: Vec2) -> Vec2 {
        v * self
    }
}
impl Div<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: f32) -> Self {
        Self::new(self.x / s, self.y / s)
    }
}
impl AddAssign for Vec2 {
    #[inline(always)]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl SubAssign for Vec2 {
    #[inline(always)]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vec4
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
        w: 1.0,
    };

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
    #[inline(always)]
    pub const fn splat(v: f32) -> Self {
        Self::new(v, v, v, v)
    }
    #[inline(always)]
    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w
    }
    #[inline(always)]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }
    #[inline(always)]
    pub fn length(self) -> f32 {
        math::sqrt(self.length_sq())
    }
    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > crate::EPS {
            self * (1.0 / len)
        } else {
            Self::ZERO
        }
    }
    /// The xyz part — convenient for plane equations (`Vec4` as `(n, d)`).
    #[inline(always)]
    pub fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

impl Add for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z, self.w + o.w)
    }
}
impl Sub for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z, self.w - o.w)
    }
}
impl Neg for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, -self.w)
    }
}
impl Mul<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s, self.w * s)
    }
}
impl Mul<Vec4> for f32 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, v: Vec4) -> Vec4 {
        v * self
    }
}
impl Div<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: f32) -> Self {
        Self::new(self.x / s, self.y / s, self.z / s, self.w / s)
    }
}
impl AddAssign for Vec4 {
    #[inline(always)]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl SubAssign for Vec4 {
    #[inline(always)]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

#[cfg(test)]
mod cast_tests {
    use super::*;

    #[test]
    fn vec3_f32_casts_round_trip() {
        let mut v = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)];
        assert_eq!(vec3s_as_f32s(&v), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let f = vec3s_as_f32s_mut(&mut v);
        f[4] = 50.0; // element 1, component y
        assert_eq!(v[1], Vec3::new(4.0, 50.0, 6.0));
    }
}
