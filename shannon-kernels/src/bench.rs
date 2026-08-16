//! W4 benchmark argument structs — the struct-wrapped halves of Warp's
//! `benchmark_launches.py` shapes (S0/Sf/Sv/Sm/Sa/Sz). 🧪 Example.
//!
//! MARSHALLING PAYLOADS ONLY: the benchmark kernels never read them.
//!
//! ⚠️ THE POINTER FIELDS MUST BE **DEVICE** POINTERS (`cu_deviceptr()`),
//! NEVER HOST SLICE REFERENCES. A `&[f32]` field would embed a host address
//! into the byval parameter — slices are only scalarized to device pointers
//! as TOP-LEVEL kernel parameters, not as struct fields — and dereferencing
//! it on device is garbage (HMM is unreliable under WSL2). Day-3 plan §4.A.
//!
//! `derive(DeviceCopy)` works on these local types; it rejects enums
//! (Day-2 plan §3.4). `BenchSa`/`BenchSz` are `!Send` via the raw pointers —
//! deliberate; the bench binary is single-threaded and no impl is added.

use shannon_core::{Mat33, Vec3};

/// ZST — exercises cuda-oxide's param-drop path (ZST `.param`s are removed
/// and the host push is skipped; verified Day 1).
#[repr(C)]
#[derive(Clone, Copy, cuda_core::DeviceCopy)]
pub struct BenchS0 {}

#[repr(C)]
#[derive(Clone, Copy, cuda_core::DeviceCopy)]
pub struct BenchSf {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Clone, Copy, cuda_core::DeviceCopy)]
pub struct BenchSv {
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, cuda_core::DeviceCopy)]
pub struct BenchSm {
    pub m: Mat33,
    pub n: Mat33,
    pub o: Mat33,
}

#[repr(C)]
#[derive(Clone, Copy, cuda_core::DeviceCopy)]
pub struct BenchSa {
    pub a: *const f32,
    pub a_len: u64,
    pub b: *const f32,
    pub b_len: u64,
    pub c: *const f32,
    pub c_len: u64,
}

#[repr(C)]
#[derive(Clone, Copy, cuda_core::DeviceCopy)]
pub struct BenchSz {
    pub a: *const f32,
    pub a_len: u64,
    pub b: *const f32,
    pub b_len: u64,
    pub c: *const f32,
    pub c_len: u64,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
}

// AsKernelArg (identity, by value) for the bench structs. Lives HERE, not in
// shannon-rt: the shipped runtime must not name example-crate types — this
// was LIMITATIONS row 8 until the module-cache inversion was fixed. Exactly
// what any third-party crate writes for its own POD kernel params.
shannon_rt::impl_kernel_arg!(BenchS0, BenchSf, BenchSv, BenchSm, BenchSa, BenchSz);
