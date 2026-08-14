#![no_std]
#![feature(core_float_math)] // sqrt via core::intrinsics::sqrtf32 — Day-2 plan §3.1
//! shannon-core — numerical primitives shared by the CPU and GPU backends.
//!
//! INVARIANT: every CODE PATH in this crate uses only `core` and `libm`. It must
//! never *call* into `std`, `cuda-device`, or any backend API. Both backends
//! are thin adapters that call into here.
//!
//! The optional `cuda` feature (default OFF) adds `DeviceCopy` marker impls and
//! nothing else — no calls, no code. `cargo test -p shannon-core` builds without
//! it and remains the guard.
//!
//! On device, `libm` calls are intercepted by cuda-oxide's mir-importer and
//! lowered to libdevice intrinsics (`__nv_sinf`, ...) — the software-float
//! bodies are never translated. See Day-1 plan §4.2. Exception: `sqrt` — see
//! `math::sqrt` and Day-2 plan §3.1.

pub mod adjoint;
pub mod bvh;
pub mod conveyor;
#[cfg(feature = "cuda")]
mod device_copy;
pub mod grad;
mod kernel_macros;
pub mod loss;
pub mod mat;
pub mod math;
pub mod mesh;
pub mod quat;
pub mod scene;
pub mod scene_arm7;
pub mod scene_shannon;
pub mod sdf;
pub mod vec;

pub use bvh::BvhNode;
pub use grad::GradSink;
pub use mat::Mat33;
pub use mesh::{MeshQuery, Particle};
pub use quat::Quat;
pub use vec::{Vec2, Vec3, Vec4};

/// Guard threshold for operations singular at zero (`normalize`, `length` adjoint).
pub const EPS: f32 = 1.0e-8;
