use std::num::NonZeroU64;
use std::sync::Arc;
use std::task::{Context, Poll};

use jxl_gpu_formats::{ImageLayout, PixelFormat};
use jxl_wgpu::{MemoryBudget, MemoryBudgetSnapshot, SubmissionPoller, WgpuBackend};

use crate::{
    BackendError, EncodeError, EncodeSession, EncoderCapabilities, FrameEncodeRequest,
    FrameSubmission, GpuFrameArtifacts, SessionDescriptor, UnsupportedFeature,
};

#[derive(Clone)]
pub struct WgpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    poller: SubmissionPoller,
    memory_budget: MemoryBudget,
    direct_mapping: bool,
}

const DEFAULT_ENCODER_IN_FLIGHT_MEMORY_BYTES: u64 = 256 * 1024 * 1024;

impl WgpuContext {
    /// Creates an encoder context with one bounded native completion worker.
    ///
    /// # Errors
    ///
    /// Returns a backend error if the native worker thread cannot be created.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Result<Self, EncodeError> {
        Self::with_memory_budget(
            device,
            queue,
            NonZeroU64::new(DEFAULT_ENCODER_IN_FLIGHT_MEMORY_BYTES)
                .expect("the default encoder memory budget is non-zero"),
        )
    }

    /// Creates a context with an application-selected aggregate in-flight byte limit.
    pub fn with_memory_budget(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        memory_budget_bytes: NonZeroU64,
    ) -> Result<Self, EncodeError> {
        let poller = SubmissionPoller::new(device.as_ref().clone())
            .map_err(BackendError::PollWorkerStart)?;
        Ok(Self {
            direct_mapping: device
                .features()
                .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            device,
            queue,
            poller,
            memory_budget: MemoryBudget::new(memory_budget_bytes),
        })
    }

    /// Shares the device, queue, and bounded completion worker of an existing render backend.
    #[must_use]
    pub fn from_backend(backend: &WgpuBackend) -> Self {
        Self {
            device: Arc::new(backend.device().clone()),
            queue: Arc::new(backend.queue().clone()),
            poller: backend.submission_poller().clone(),
            memory_budget: backend.transient_memory_budget().clone(),
            direct_mapping: backend.direct_readback_enabled(),
        }
    }

    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) const fn submission_poller(&self) -> &SubmissionPoller {
        &self.poller
    }

    pub(crate) const fn memory_budget(&self) -> &MemoryBudget {
        &self.memory_budget
    }

    pub(crate) const fn direct_mapping_enabled(&self) -> bool {
        self.direct_mapping
    }

    /// Reports bytes reserved by all live encoder jobs sharing this context.
    #[must_use]
    pub fn memory_stats(&self) -> MemoryBudgetSnapshot {
        self.memory_budget.snapshot()
    }
}

#[derive(Clone, Debug)]
pub struct BufferImageSource {
    pub buffer: Arc<wgpu::Buffer>,
    pub layout: ImageLayout,
}

impl BufferImageSource {
    pub fn new(buffer: Arc<wgpu::Buffer>, layout: ImageLayout) -> Result<Self, EncodeError> {
        if buffer.size() < layout.logical_size {
            return Err(EncodeError::InvalidSource(
                "GPU buffer is smaller than the declared pitch-linear image layout",
            ));
        }
        Ok(Self { buffer, layout })
    }
}

/// One directly sampleable `wgpu` texture. Multi-planar and packed formats use
/// [`BufferImageSource`] until portable multi-plane texture support exists.
#[derive(Clone, Debug)]
pub struct TextureImageSource {
    pub texture: Arc<wgpu::Texture>,
    pub texture_format: wgpu::TextureFormat,
    pub pixel_format: PixelFormat,
    pub mip_level: u32,
    pub array_layer: u32,
}

impl TextureImageSource {
    pub fn new(
        texture: Arc<wgpu::Texture>,
        texture_format: wgpu::TextureFormat,
        pixel_format: PixelFormat,
        mip_level: u32,
        array_layer: u32,
    ) -> Result<Self, EncodeError> {
        let size = texture.size();
        if mip_level >= texture.mip_level_count() {
            return Err(EncodeError::InvalidSource(
                "texture mip level is out of range",
            ));
        }
        if array_layer >= size.depth_or_array_layers {
            return Err(EncodeError::InvalidSource(
                "texture array layer is out of range",
            ));
        }
        if pixel_format.planes.len() != 1 {
            return Err(EncodeError::InvalidSource(
                "multi-planar formats must use a pitch-linear GPU buffer",
            ));
        }
        Ok(Self {
            texture,
            texture_format,
            pixel_format,
            mip_level,
            array_layer,
        })
    }
}

#[derive(Clone, Debug)]
pub enum GpuFrameSource {
    Buffer(BufferImageSource),
    Texture(TextureImageSource),
}

impl GpuFrameSource {
    #[must_use]
    pub fn pixel_format(&self) -> &PixelFormat {
        match self {
            Self::Buffer(source) => &source.layout.format,
            Self::Texture(source) => &source.pixel_format,
        }
    }
}

/// Runtime-neutral completion object returned by a GPU backend.
///
/// `wait` may call `Device::poll` internally. `poll_complete` must register the
/// supplied waker when completion will happen later. Neither method may run a
/// CPU image/transform/quantization fallback.
#[cfg(not(target_arch = "wasm32"))]
pub trait GpuEncodeJob: Send + Unpin + 'static {
    fn poll_complete(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<GpuFrameArtifacts, EncodeError>>;

    fn wait(self) -> Result<GpuFrameArtifacts, EncodeError>;
}

/// Browser WebGPU handles are main-thread-local, so the portable completion
/// contract does not require `Send` on `wasm32`.
#[cfg(target_arch = "wasm32")]
pub trait GpuEncodeJob: Unpin + 'static {
    fn poll_complete(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<GpuFrameArtifacts, EncodeError>>;

    fn wait(self) -> Result<GpuFrameArtifacts, EncodeError>;
}

/// Contract between orchestration and concrete JPEG XL compute kernels.
///
/// Implementations must record all pixel, coefficient, predictor,
/// quantization, tokenization, and histogram work through the supplied `wgpu`
/// context. The returned CPU-visible artifacts are already entropy-ready group
/// packets and serialized frame-header fields.
#[cfg(not(target_arch = "wasm32"))]
pub trait GpuEncodeBackend: Send + Sync + 'static {
    type Job: GpuEncodeJob;

    fn capabilities(&self) -> &EncoderCapabilities;

    fn supports_input(&self, source: &GpuFrameSource) -> bool;

    fn submit(
        &self,
        context: &WgpuContext,
        source: GpuFrameSource,
        request: &FrameEncodeRequest,
    ) -> Result<Self::Job, EncodeError>;
}

/// Browser WebGPU resources are main-thread-local, so a browser backend is not
/// required to implement native thread-transfer traits.
#[cfg(target_arch = "wasm32")]
pub trait GpuEncodeBackend: 'static {
    type Job: GpuEncodeJob;

    fn capabilities(&self) -> &EncoderCapabilities;

    fn supports_input(&self, source: &GpuFrameSource) -> bool;

    fn submit(
        &self,
        context: &WgpuContext,
        source: GpuFrameSource,
        request: &FrameEncodeRequest,
    ) -> Result<Self::Job, EncodeError>;
}

pub struct GpuEncoder<B> {
    context: WgpuContext,
    backend: Arc<B>,
}

impl<B> Clone for GpuEncoder<B> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            backend: Arc::clone(&self.backend),
        }
    }
}

impl<B: GpuEncodeBackend> GpuEncoder<B> {
    #[must_use]
    pub fn new(context: WgpuContext, backend: B) -> Self {
        Self {
            context,
            backend: Arc::new(backend),
        }
    }

    #[must_use]
    pub fn capabilities(&self) -> &EncoderCapabilities {
        self.backend.capabilities()
    }

    /// Reports aggregate byte-weighted memory admission for live jobs from this context.
    #[must_use]
    pub fn memory_stats(&self) -> MemoryBudgetSnapshot {
        self.context.memory_stats()
    }

    /// Returns the concrete backend so profile-specific limits and memory
    /// plans can be queried before a submission is admitted.
    #[must_use]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn submit_frame(
        &self,
        source: GpuFrameSource,
        request: FrameEncodeRequest,
    ) -> Result<FrameSubmission<B::Job>, EncodeError> {
        self.backend.capabilities().negotiate(&request)?;
        if !self.backend.supports_input(&source) {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        let expected_index = request.frame_index;
        let expected_last = request.is_last;
        let job = self.backend.submit(&self.context, source, &request)?;
        Ok(FrameSubmission::new(job, expected_index, expected_last))
    }

    pub fn begin_session(
        &self,
        descriptor: SessionDescriptor,
    ) -> Result<EncodeSession<B>, EncodeError> {
        if descriptor.canvas_width == 0 || descriptor.canvas_height == 0 {
            return Err(EncodeError::InvalidConfiguration(
                "the JPEG XL session canvas must be non-empty",
            ));
        }
        if descriptor.animation.is_animation() && !self.capabilities().animation {
            return Err(UnsupportedFeature::Animation.into());
        }
        Ok(EncodeSession::new(self.clone(), descriptor))
    }
}
