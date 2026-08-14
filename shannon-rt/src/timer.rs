//! Timing utilities. 📦 Shipped.
//!
//! `ScopedTimer` mirrors Warp's: named, wall-clock, prints on drop. That is
//! deliberate for launch benchmarks — the W4 methodology times host dispatch
//! (enqueue), not device execution (Day-3 plan §3). `GpuTimer` wraps CUDA
//! events for the occasions when device-side duration *is* the question.

use crate::device::Device;
use anyhow::anyhow;
use std::time::Instant;

/// Named wall-clock timer. Prints `name: X.XXX ms` on drop unless constructed
/// with [`ScopedTimer::quiet`]. Nestable by construction — just create more.
pub struct ScopedTimer {
    name: &'static str,
    start: Instant,
    print_on_drop: bool,
}

impl ScopedTimer {
    pub fn new(name: &'static str) -> Self {
        Self { name, start: Instant::now(), print_on_drop: true }
    }

    /// For benchmark loops that aggregate externally — never prints.
    pub fn quiet(name: &'static str) -> Self {
        Self { name, start: Instant::now(), print_on_drop: false }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1e3
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        if self.print_on_drop {
            println!("{}: {:.3} ms", self.name, self.elapsed_ms());
        }
    }
}

/// Device-side duration between two recorded events.
///
/// API verified against `cuda-core/src/event.rs` (`new_event` :65,
/// `record` :99, `synchronize` :106, `elapsed_ms` :134).
///
/// ⚠️ Events MUST be created with `CU_EVENT_DEFAULT`: `new_event(None)`
/// defaults to `CU_EVENT_DISABLE_TIMING`, and `elapsed_ms` on such an event
/// fails at runtime.
pub struct GpuTimer {
    start: cuda_core::CudaEvent,
    end: cuda_core::CudaEvent,
}

impl GpuTimer {
    pub fn new(dev: &Device) -> anyhow::Result<Self> {
        let flags = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
        let start = dev
            .ctx()
            .new_event(flags)
            .map_err(|e| anyhow!("creating start event: {e:?}"))?;
        let end = dev
            .ctx()
            .new_event(flags)
            .map_err(|e| anyhow!("creating end event: {e:?}"))?;
        Ok(Self { start, end })
    }

    /// Record the start event on the device's stream.
    pub fn start(&self, dev: &Device) -> anyhow::Result<()> {
        self.start
            .record(dev.stream())
            .map_err(|e| anyhow!("recording start event: {e:?}"))
    }

    /// Record the end event on the device's stream.
    pub fn stop(&self, dev: &Device) -> anyhow::Result<()> {
        self.end
            .record(dev.stream())
            .map_err(|e| anyhow!("recording end event: {e:?}"))
    }

    /// Synchronizes the end event, then returns start→end in milliseconds.
    pub fn elapsed_ms(&self) -> anyhow::Result<f32> {
        self.end
            .synchronize()
            .map_err(|e| anyhow!("synchronizing end event: {e:?}"))?;
        self.start
            .elapsed_ms(&self.end)
            .map_err(|e| anyhow!("event elapsed query: {e:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_is_monotone_and_quiet_never_prints() {
        // quiet(): construction + drop must not print (visual check under
        // --nocapture; the functional assertion is monotonicity).
        let t = ScopedTimer::quiet("test");
        let a = t.elapsed_ms();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = t.elapsed_ms();
        assert!(b >= a, "elapsed went backwards: {a} → {b}");
        assert!(b >= 2.0, "slept 2 ms but elapsed only {b} ms");
    }
}
