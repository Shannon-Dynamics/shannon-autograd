//! CUDA device + stream, with a zero-ceremony process-wide default.

use anyhow::anyhow;
use cuda_core::{CudaContext, CudaStream};
use std::sync::{Arc, OnceLock};

/// A CUDA device binding: context + default stream. Cheap to clone (two Arcs).
#[derive(Clone)]
pub struct Device {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

static DEFAULT: OnceLock<Device> = OnceLock::new();

impl Device {
    /// Zero-ceremony default: CUDA device 0, initialized on first use and
    /// cached process-wide. The user never calls this explicitly — `Array`
    /// constructors and `launch!` resolve it internally.
    ///
    /// (A CPU fallback variant lands with the backend enum — week-1 plan §8.5.)
    #[allow(clippy::should_implement_trait)] // deliberate: Warp-style default device
    pub fn default() -> anyhow::Result<&'static Device> {
        if let Some(d) = DEFAULT.get() {
            return Ok(d);
        }
        // Benign race: if two threads initialize concurrently, one Device is
        // dropped — primary-context refcounting makes that harmless.
        let d = Device::cuda(0)?;
        Ok(DEFAULT.get_or_init(|| d))
    }

    /// Bind to a specific CUDA device ordinal.
    pub fn cuda(ordinal: usize) -> anyhow::Result<Self> {
        let ctx = CudaContext::new(ordinal)
            .map_err(|e| anyhow!("creating CUDA context on device {ordinal}: {e:?}"))?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream })
    }

    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    pub fn synchronize(&self) -> anyhow::Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("stream synchronize: {e:?}"))
    }
}
