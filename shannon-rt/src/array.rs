//! `Array<T>` — a device buffer that knows where it lives, with an optional
//! `.grad` shadow buffer.

use crate::device::Device;
use anyhow::anyhow;
use cuda_core::{DeviceBuffer, DeviceCopy};

pub struct Array<T> {
    device: Device, // ← the array knows where it lives (cheap clone: two Arcs)
    buf: DeviceBuffer<T>,
    grad: Option<Box<Array<T>>>, // lazily allocated shadow buffer
    len: usize,
    requires_grad: bool,
}

impl<T: DeviceCopy + Default> Array<T> {
    // ── Default-device constructors — what user code calls. ────────────────

    pub fn zeros(n: usize) -> anyhow::Result<Self> {
        Self::zeros_on(Device::default()?, n)
    }

    pub fn from_slice(data: &[T]) -> anyhow::Result<Self> {
        Self::from_slice_on(Device::default()?, data)
    }

    // ── Explicit-device forms, for when it matters. ────────────────────────

    pub fn zeros_on(dev: &Device, n: usize) -> anyhow::Result<Self> {
        let buf = DeviceBuffer::<T>::zeroed(dev.stream(), n)
            .map_err(|e| anyhow!("allocating zeroed device buffer ({n} elems): {e:?}"))?;
        Ok(Self { device: dev.clone(), buf, grad: None, len: n, requires_grad: false })
    }

    pub fn from_slice_on(dev: &Device, data: &[T]) -> anyhow::Result<Self> {
        let buf = DeviceBuffer::from_host(dev.stream(), data)
            .map_err(|e| anyhow!("uploading {} elems to device: {e:?}", data.len()))?;
        Ok(Self { device: dev.clone(), buf, grad: None, len: data.len(), requires_grad: false })
    }

    /// Read back to the host. Implicitly synchronizes — `to_host_vec` already
    /// does (cuda-core/src/device_buffer.rs:564), matching Warp's `.numpy()`.
    pub fn to_vec(&self) -> anyhow::Result<Vec<T>> {
        self.buf
            .to_host_vec(self.device.stream())
            .map_err(|e| anyhow!("downloading {} elems from device: {e:?}", self.len))
    }

    /// Overwrite the buffer from a host slice. Length-checked; keeps the
    /// device pointer stable (no realloc) — `Mesh::refit` re-uploads nodes
    /// through this every frame, and the Day-6 tape will hold references into
    /// buffers updated this way.
    pub fn copy_from_slice(&mut self, data: &[T]) -> anyhow::Result<()> {
        anyhow::ensure!(
            data.len() == self.len,
            "copy_from_slice: length mismatch ({} vs {})",
            data.len(),
            self.len
        );
        self.buf
            .copy_from_host(self.device.stream(), data)
            .map_err(|e| anyhow!("uploading {} elems: {e:?}", self.len))
    }

    /// Lazily allocates the `.grad` shadow buffer, zeroed.
    pub fn requires_grad_(&mut self, on: bool) -> anyhow::Result<()> {
        if on && self.grad.is_none() {
            self.grad = Some(Box::new(Array::zeros_on(&self.device, self.len)?));
        }
        if !on {
            self.grad = None;
        }
        self.requires_grad = on;
        Ok(())
    }

    pub fn grad(&self) -> Option<&Array<T>> {
        self.grad.as_deref()
    }

    /// Mutable access to the grad shadow — the seed path (Day-6 plan §4.2):
    /// mutation-phase code does `loss.grad_mut()…copy_from_slice(&[1.0])`
    /// BEFORE anything records, because at backward time the tape's records
    /// hold shared borrows and this accessor is unreachable.
    pub fn grad_mut(&mut self) -> Option<&mut Array<T>> {
        self.grad.as_deref_mut()
    }

    /// MUST be called between optimizer iterations — adjoints ACCUMULATE.
    /// Forgetting this is the most common tape bug; it makes the loss still
    /// decrease, just wrongly. See CUDA-OXIDE-AUTODIFF-REFERENCE §7.
    pub fn zero_grad(&mut self) -> anyhow::Result<()> {
        if let Some(g) = self.grad.as_deref_mut() {
            g.fill_default()?;
        }
        Ok(())
    }

    /// Overwrite every element with `T::default()` (zero for numeric types).
    /// Keeps the device pointer stable, unlike a realloc — the Day-6 tape
    /// will hold references into this buffer.
    fn fill_default(&mut self) -> anyhow::Result<()> {
        let zeros = vec![T::default(); self.len];
        self.buf
            .copy_from_host(self.device.stream(), &zeros)
            .map_err(|e| anyhow!("zeroing {} elems: {e:?}", self.len))
    }

}

/// Bound-free accessors — `launch!`'s argument conversion must not require
/// `Default`, and these need no bounds at all.
impl<T> Array<T> {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    // ── Plumbing — used only by `launch!`. Not user-facing. ────────────────

    #[doc(hidden)]
    pub fn __buf(&self) -> &DeviceBuffer<T> {
        &self.buf
    }

    #[doc(hidden)]
    pub fn __buf_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    /// `grad_mut` on a grad-less array is `None`; after `requires_grad_` the
    /// seed path (grad_mut → copy_from_slice) round-trips. GPU-skipping like
    /// its sibling below.
    #[test]
    fn grad_mut_gates_on_requires_grad_and_seeds() {
        if Device::default().is_err() {
            eprintln!("skipped: no CUDA device available");
            return;
        }
        let mut loss = Array::from_slice(&[0.0f32]).unwrap();
        assert!(loss.grad_mut().is_none(), "no grad before requires_grad_");
        loss.requires_grad_(true).unwrap();
        loss.grad_mut().unwrap().copy_from_slice(&[1.0]).unwrap();
        assert_eq!(loss.grad().unwrap().to_vec().unwrap(), vec![1.0]);
        loss.zero_grad().unwrap();
        assert_eq!(loss.grad().unwrap().to_vec().unwrap(), vec![0.0]);
    }

    /// Round-trips through the device. Skips gracefully when no CUDA device is
    /// available so `cargo test -p shannon-rt` stays host-safe (the Day-4
    /// convention: host tests carry no GPU *requirement*).
    #[test]
    fn copy_from_slice_round_trips_and_checks_length() {
        if Device::default().is_err() {
            eprintln!("skipped: no CUDA device available");
            return;
        }
        let mut a = Array::from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
        a.copy_from_slice(&[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(a.to_vec().unwrap(), vec![4.0, 5.0, 6.0]);
        assert!(a.copy_from_slice(&[1.0]).is_err(), "length mismatch must error");
    }
}
