//! Cached handle to the W0 example's PTX module — users never see loading.
//!
//! Day-1 scope: single default device, one module (`shannon-kernels`). A
//! per-device keyed cache and a cross-crate kernel registry are week-2 items
//! (week-1 plan §8.5).

use crate::device::Device;
use anyhow::anyhow;
use shannon_kernels::kernels::LoadedModule;
use std::sync::OnceLock;

static MODULE: OnceLock<LoadedModule> = OnceLock::new();

/// Resolve (and on first use, load) the embedded PTX module for `dev`.
/// `#[doc(hidden)]` — referenced only by `launch!`.
#[doc(hidden)]
pub fn __module(dev: &Device) -> anyhow::Result<&'static LoadedModule> {
    if let Some(m) = MODULE.get() {
        return Ok(m);
    }
    let m = shannon_kernels::kernels::load(dev.ctx())
        .map_err(|e| anyhow!("loading embedded PTX module 'shannon-kernels': {e:?}"))?;
    Ok(MODULE.get_or_init(|| m))
}
