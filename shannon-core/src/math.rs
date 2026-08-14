//! Scalar math over `libm`.
//!
//! Free functions, not an extension trait: `core` gained some inherent `f32`
//! methods in recent releases, and a trait method with the same name would be
//! silently shadowed by the inherent one. Free functions are unambiguous.
//!
//! On GPU these lower to libdevice (`__nv_sqrtf`, ...) — see the Day-1 plan §4.2.

/// NOT `libm::sqrtf`: libm routes sqrt through an arch-specific inline-asm
/// wrapper that rustc MIR-inlines before cuda-oxide's call-site interception
/// can fire — and the fallback path then resolves to the WRONG WIDTH (f64).
/// `core::f32::math::sqrt` is literally `intrinsics::sqrtf32`, which lowers to
/// `__nv_sqrtf` on device. Day-2 plan §3.1.
#[inline(always)]
pub fn sqrt(x: f32) -> f32 {
    core::f32::math::sqrt(x)
}
#[inline(always)]
pub fn sin(x: f32) -> f32 {
    libm::sinf(x)
}
#[inline(always)]
pub fn cos(x: f32) -> f32 {
    libm::cosf(x)
}
#[inline(always)]
pub fn tan(x: f32) -> f32 {
    libm::tanf(x)
}
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    libm::expf(x)
}
#[inline(always)]
pub fn ln(x: f32) -> f32 {
    libm::logf(x)
}
#[inline(always)]
pub fn pow(x: f32, n: f32) -> f32 {
    libm::powf(x, n)
}
#[inline(always)]
pub fn abs(x: f32) -> f32 {
    libm::fabsf(x)
}
#[inline(always)]
pub fn floor(x: f32) -> f32 {
    libm::floorf(x)
}
#[inline(always)]
pub fn ceil(x: f32) -> f32 {
    libm::ceilf(x)
}
#[inline(always)]
pub fn atan2(y: f32, x: f32) -> f32 {
    libm::atan2f(y, x)
}

// `min`/`max` are hand-written rather than `libm::fminf`: the NaN behaviour is
// predictable and it avoids a call on the hottest path in BVH traversal (Day 4).
#[inline(always)]
pub fn min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}
#[inline(always)]
pub fn max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}
#[inline(always)]
pub fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    min(max(x, lo), hi)
}
#[inline(always)]
pub fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}
#[inline(always)]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
#[inline(always)]
pub fn rsqrt(x: f32) -> f32 {
    1.0 / sqrt(x)
}
