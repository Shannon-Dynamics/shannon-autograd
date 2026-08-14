//! `DeviceCopy` marker impls, behind the `cuda` feature.
//!
//! These are EMPTY impls of a method-less marker trait. They add no callable
//! item, so they are invisible to cuda-oxide's device collector — see the
//! Day-2 plan §3.3.
//!
//! SAFETY for every impl below: each type is `#[repr(C)]` and contains only
//! `f32` (for `BvhNode`, also `i32`) fields, so it is plain-old-data — no
//! pointers, no padding that could be uninitialised, and the all-zero bit
//! pattern is a valid value (required because `DeviceBuffer::zeroed`
//! initialises with zero bytes).

use crate::{BvhNode, Mat33, MeshQuery, Particle, Quat, Vec2, Vec3, Vec4};

unsafe impl cuda_core::DeviceCopy for Vec2 {}
unsafe impl cuda_core::DeviceCopy for Vec3 {}
unsafe impl cuda_core::DeviceCopy for Vec4 {}
unsafe impl cuda_core::DeviceCopy for Quat {}
unsafe impl cuda_core::DeviceCopy for Mat33 {}
unsafe impl cuda_core::DeviceCopy for BvhNode {}
unsafe impl cuda_core::DeviceCopy for MeshQuery {}
unsafe impl cuda_core::DeviceCopy for Particle {}
