//! The `launch!` macro and its argument-conversion trait.

use crate::array::Array;
use cuda_core::{DeviceBuffer, DeviceCopy};

/// Converts a call-site argument into what the `#[cuda_module]`-generated
/// launch method expects:
///
/// | call site        | generated method takes     |
/// |------------------|----------------------------|
/// | `&Array<T>`      | `&DeviceBuffer<T>`         |
/// | `&mut Array<T>`  | `&mut DeviceBuffer<T>`     |
/// | scalar by value  | the scalar, unchanged      |
///
/// Invoked via UFCS from `launch!` — callers never import it explicitly.
pub trait AsKernelArg {
    type Out;
    #[allow(clippy::wrong_self_convention)] // deliberately consumes the reference
    fn as_karg(self) -> Self::Out;
}

impl<'a, T: DeviceCopy> AsKernelArg for &'a Array<T> {
    type Out = &'a DeviceBuffer<T>;
    #[inline(always)]
    fn as_karg(self) -> &'a DeviceBuffer<T> {
        self.__buf()
    }
}

impl<'a, T: DeviceCopy> AsKernelArg for &'a mut Array<T> {
    type Out = &'a mut DeviceBuffer<T>;
    #[inline(always)]
    fn as_karg(self) -> &'a mut DeviceBuffer<T> {
        self.__buf_mut()
    }
}

macro_rules! scalar_karg {
    ($($t:ty),* $(,)?) => {
        $(
            impl AsKernelArg for $t {
                type Out = $t;
                #[inline(always)]
                fn as_karg(self) -> $t { self }
            }
        )*
    };
}

// Primitive scalars plus shannon-core value types (kernel params by value).
scalar_karg!(f32, f64, i32, i64, u32, u64, usize, bool);
scalar_karg!(
    shannon_core::Vec2,
    shannon_core::Vec3,
    shannon_core::Vec4,
    shannon_core::Quat,
    shannon_core::Mat33,
);

// W4 bench structs. KNOWN WART: the shipped runtime naming example-crate types
// is a dependency-direction smell — legal only because shannon-rt already
// depends on shannon-kernels for the module cache. The week-2 kernel registry
// (week-plan §15 item 6) dissolves both couplings at once. A blanket
// `impl<T: DeviceCopy> AsKernelArg for T` is NOT an alternative: it conflicts
// with the `&Array<T>` impls under coherence's future-compat rules (E0119).
scalar_karg!(
    shannon_kernels::bench::BenchS0,
    shannon_kernels::bench::BenchSf,
    shannon_kernels::bench::BenchSv,
    shannon_kernels::bench::BenchSm,
    shannon_kernels::bench::BenchSa,
    shannon_kernels::bench::BenchSz,
);

/// Launch a kernel from the W0 module on the default device.
///
/// Resolves the cached module, unwraps `Array` handles, and dispatches.
/// Deliberately minimal for Day 1 — no tape integration (that is Day 6) and no
/// runtime backend selection (deferred, week-1 plan §8.5).
///
/// Usage:  `launch!(affine, dim = n, (&a, scale, bias, &mut y))?;`
#[macro_export]
macro_rules! launch {
    ($kernel:ident, dim = $n:expr, ($($arg:expr),* $(,)?)) => {{
        (|| -> $crate::__anyhow::Result<()> {
            let __dev = $crate::Device::default()?;
            let __m = $crate::__module(__dev)?;
            let __cfg = $crate::__cuda_core::LaunchConfig::for_num_elems($n as u32);
            // SAFETY: 1-D launch; the kernel guards its index against the
            // buffer length, and the caller passes buffers covering `dim`.
            unsafe { __m.$kernel(__dev.stream(), __cfg, $($crate::AsKernelArg::as_karg($arg)),*) }
                .map_err(|e| $crate::__anyhow::anyhow!(
                    "launching kernel `{}`: {e:?}", stringify!($kernel)
                ))
        })()
    }};
}
