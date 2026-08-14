//! The one backend-specific operation, abstracted.

/// Accumulates gradient contributions into a buffer.
///
/// This is the ONE operation that cannot be shared between backends:
///   GPU → hardware `atom.add.f32` via `DeviceAtomicF32::fetch_add`  (shannon-device)
///   CPU → compare-and-swap loop over `AtomicU32`                    (shannon-rt)
///
/// Generic over the element type so Vec4/Mat33 can be added later without
/// touching either implementation.
///
/// Reverse-mode AD is a SCATTER-ADD: several threads may target the same
/// index. Implementations MUST be atomic. See CUDA-OXIDE-AUTODIFF-REFERENCE §1.
pub trait GradSink<T: Copy> {
    fn accumulate(&self, index: usize, grad: T);
}
