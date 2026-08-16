//! shannon-rt — the host runtime. 📦 Shipped.
//!
//! `Device`, `Array<T>`, the `launch!` macro, and the host-side `GradSink`
//! implementations. Matches Warp's call-site contract (Day-1 plan §5.6):
//!
//! | Warp property                     | Mechanism                              |
//! |-----------------------------------|----------------------------------------|
//! | Zero-ceremony init                | `Device::default()` — OnceLock         |
//! | Device belongs to the data        | `Array<T>` carries its `Device`        |
//! | Arrays passed whole               | `launch!` takes `&Array<T>`            |
//! | No module/context in user code    | `define_module_cache!` OnceLock per kernel crate |
//! | Readback implicitly syncs         | `to_vec()` → `to_host_vec`             |

mod array;
mod device;
mod grad;
mod launch;
mod timer;

pub use array::Array;
pub use device::Device;
pub use grad::{HostGradF32, HostGradVec3};
pub use launch::AsKernelArg;
pub use timer::{GpuTimer, ScopedTimer};

// Macro plumbing — `launch!` references these through `$crate::…` so callers
// need neither `anyhow` nor `cuda-core` in scope for the macro to expand.
#[doc(hidden)]
pub use anyhow as __anyhow;
#[doc(hidden)]
pub use cuda_core as __cuda_core;
