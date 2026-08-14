#![no_std]
//! shannon-device — device-side `GradSink` implementations. 📦 Shipped.
//!
//! Third-party adjoint kernels import these; they cannot live in an example
//! crate. The GPU side of the one backend-specific operation (Day-1 plan §5.4).
//!
//! Gradient buffers arrive as `&[T]`, NOT `DisjointSlice`: scatter-add is the
//! opposite of one-thread-one-element, and must opt out of the disjointness
//! proof. See CUDA-OXIDE-AUTODIFF-REFERENCE.md §2.

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use shannon_core::{GradSink, Vec3};

/// Scatter-add sink over an `&[f32]` gradient buffer — hardware `atom.add.f32`.
pub struct DeviceGradF32<'a>(pub &'a [f32]);

impl GradSink<f32> for DeviceGradF32<'_> {
    #[inline(always)]
    fn accumulate(&self, i: usize, g: f32) {
        if i >= self.0.len() {
            return;
        }
        // SAFETY: DeviceAtomicF32 is repr(transparent) over UnsafeCell<f32>;
        // the buffer outlives the launch; every access to this element is
        // atomic. `Relaxed` is correct: gradient accumulation needs atomicity,
        // not ordering — results are read only after synchronize().
        unsafe {
            let p = &*(self.0.as_ptr().add(i) as *const DeviceAtomicF32);
            p.fetch_add(g, AtomicOrdering::Relaxed);
        }
    }
}

/// Scatter-add sink over an `&[Vec3]` gradient buffer.
///
/// Vec3 has no atomic instruction — accumulate component-wise. Relies on
/// `#[repr(C)]` putting x/y/z at byte offsets 0/4/8 (shannon-core guarantees it).
pub struct DeviceGradVec3<'a>(pub &'a [Vec3]);

impl GradSink<Vec3> for DeviceGradVec3<'_> {
    #[inline(always)]
    fn accumulate(&self, i: usize, g: Vec3) {
        if i >= self.0.len() {
            return;
        }
        // SAFETY: as above; #[repr(C)] fixes the component offsets.
        unsafe {
            let base = self.0.as_ptr().add(i) as *const f32;
            (*(base.add(0) as *const DeviceAtomicF32)).fetch_add(g.x, AtomicOrdering::Relaxed);
            (*(base.add(1) as *const DeviceAtomicF32)).fetch_add(g.y, AtomicOrdering::Relaxed);
            (*(base.add(2) as *const DeviceAtomicF32)).fetch_add(g.z, AtomicOrdering::Relaxed);
        }
    }
}
