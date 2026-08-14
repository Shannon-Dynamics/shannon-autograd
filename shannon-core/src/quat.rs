//! Quaternions — `(x, y, z, w)` with `w` the scalar part, matching Warp's layout.

use crate::math;
use crate::vec::Vec3;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    #[inline(always)]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let half = angle * 0.5;
        let s = math::sin(half);
        let a = axis.normalize();
        Self::new(a.x * s, a.y * s, a.z * s, math::cos(half))
    }

    /// Roll (about x), pitch (about y), yaw (about z) — ZYX composition.
    #[inline(always)]
    pub fn from_rpy(roll: f32, pitch: f32, yaw: f32) -> Self {
        let (sr, cr) = (math::sin(roll * 0.5), math::cos(roll * 0.5));
        let (sp, cp) = (math::sin(pitch * 0.5), math::cos(pitch * 0.5));
        let (sy, cy) = (math::sin(yaw * 0.5), math::cos(yaw * 0.5));
        Self::new(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    }

    /// Rotate `v` by this (unit) quaternion.
    ///
    /// Shuffle-free form: `t = 2·(q_v × v); v' = v + w·t + q_v × t` — fewer ops
    /// than building a matrix. This is the hot path in Day 2's ray marcher.
    #[inline(always)]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let t = qv.cross(v) * 2.0;
        v + t * self.w + qv.cross(t)
    }

    /// Rotate by the inverse — for a unit quaternion, the conjugate.
    #[inline(always)]
    pub fn rotate_inv(self, v: Vec3) -> Vec3 {
        self.conjugate().rotate(v)
    }

    /// Hamilton product `self * o` (apply `o` first, then `self`).
    #[allow(clippy::should_implement_trait)] // explicit method reads better in kernels
    #[inline(always)]
    pub fn mul(self, o: Self) -> Self {
        Self::new(
            self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
            self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
        )
    }

    #[inline(always)]
    pub fn conjugate(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    /// Inverse. For unit quaternions this equals the conjugate; the general
    /// form divides by the squared norm, guarded at zero.
    #[inline(always)]
    pub fn inverse(self) -> Self {
        let n = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        if n > crate::EPS {
            let inv = 1.0 / n;
            Self::new(-self.x * inv, -self.y * inv, -self.z * inv, self.w * inv)
        } else {
            Self::IDENTITY
        }
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = math::sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w);
        if len > crate::EPS {
            let inv = 1.0 / len;
            Self::new(self.x * inv, self.y * inv, self.z * inv, self.w * inv)
        } else {
            Self::IDENTITY
        }
    }
}
