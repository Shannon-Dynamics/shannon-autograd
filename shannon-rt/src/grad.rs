//! Host-side `GradSink` implementations. 📦 Shipped.
//!
//! Rust has no stable `AtomicF32`, so accumulation is a compare-and-swap loop
//! over the bit pattern. Rayon-safe, which keeps the CPU path a faithful
//! reference for the GPU path.

use core::sync::atomic::{AtomicU32, Ordering};
use shannon_core::{GradSink, Vec3};

/// CAS-loop scatter-add over an `&mut [f32]` gradient buffer.
pub struct HostGradF32<'a> {
    buf: &'a [AtomicU32],
}

impl<'a> HostGradF32<'a> {
    pub fn new(buf: &'a mut [f32]) -> Self {
        // SAFETY: AtomicU32 and f32 are both 4 bytes with equal alignment;
        // AtomicU32 is repr(transparent) over UnsafeCell<u32>. The &mut borrow
        // is consumed into this struct, so no non-atomic access can coexist.
        let atomics =
            unsafe { core::slice::from_raw_parts(buf.as_ptr() as *const AtomicU32, buf.len()) };
        Self { buf: atomics }
    }
}

impl GradSink<f32> for HostGradF32<'_> {
    fn accumulate(&self, i: usize, g: f32) {
        let cell = &self.buf[i];
        let mut old = cell.load(Ordering::Relaxed);
        loop {
            let new = (f32::from_bits(old) + g).to_bits();
            match cell.compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(x) => old = x,
            }
        }
    }
}

/// CAS-loop scatter-add over an `&mut [Vec3]` gradient buffer — three
/// component loops per accumulate. Relies on `#[repr(C)]` for x/y/z at
/// consecutive 4-byte offsets (shannon-core guarantees it).
pub struct HostGradVec3<'a> {
    // Length is 3 × the Vec3 count — one AtomicU32 per component.
    buf: &'a [AtomicU32],
}

impl<'a> HostGradVec3<'a> {
    pub fn new(buf: &'a mut [Vec3]) -> Self {
        // SAFETY: #[repr(C)] Vec3 is exactly 3 consecutive f32s (12 bytes,
        // align 4); reinterpreting n Vec3s as 3n AtomicU32s is layout-exact.
        let atomics =
            unsafe { core::slice::from_raw_parts(buf.as_ptr() as *const AtomicU32, buf.len() * 3) };
        Self { buf: atomics }
    }

    fn add_component(&self, slot: usize, g: f32) {
        let cell = &self.buf[slot];
        let mut old = cell.load(Ordering::Relaxed);
        loop {
            let new = (f32::from_bits(old) + g).to_bits();
            match cell.compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(x) => old = x,
            }
        }
    }
}

impl GradSink<Vec3> for HostGradVec3<'_> {
    fn accumulate(&self, i: usize, g: Vec3) {
        self.add_component(i * 3, g.x);
        self.add_component(i * 3 + 1, g.y);
        self.add_component(i * 3 + 2, g.z);
    }
}
