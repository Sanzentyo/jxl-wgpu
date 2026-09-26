use std::num::NonZeroU64;
use std::sync::Arc;
use std::task::{Context, Poll};

use jxl_gpu_formats::{ImageLayout, PixelFormat};
use jxl_wgpu::{KernelPolicy, MemoryBudget, MemoryBudgetSnapshot, SubmissionPoller, WgpuBackend};

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
    kernel_policy: KernelPolicy,
    direct_mapping: bool,
    yuv_pipeline: Arc<std::sync::OnceLock<Arc<wgpu::ComputePipeline>>>,
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
            kernel_policy: KernelPolicy::Default,
            yuv_pipeline: Arc::default(),
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
            kernel_policy: backend.kernel_policy().clone(),
            direct_mapping: backend.direct_readback_enabled(),
            yuv_pipeline: Arc::default(),
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

    /// Returns the workgroup policy inherited from the render backend.
    ///
    /// Standalone contexts created with [`Self::new`] or [`Self::with_memory_budget`] use the
    /// built-in defaults. Construct a [`WgpuBackend`] with an adapter-validated autotune profile
    /// and pass it to [`Self::from_backend`] to share tuned choices with the encoder.
    #[must_use]
    pub const fn kernel_policy(&self) -> &KernelPolicy {
        &self.kernel_policy
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

    pub(crate) fn yuv_pipeline(&self) -> &Arc<wgpu::ComputePipeline> {
        self.yuv_pipeline
            .get_or_init(|| Arc::new(crate::yuv_input::pipeline(&self.device)))
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
    pub(crate) extra_channels: Vec<BufferImageSource>,
    pub(crate) cmyk_encoding: crate::CmykSampleEncoding,
}

impl BufferImageSource {
    pub fn new(buffer: Arc<wgpu::Buffer>, layout: ImageLayout) -> Result<Self, EncodeError> {
        if buffer.size() < layout.logical_size {
            return Err(EncodeError::InvalidSource(
                "GPU buffer is smaller than the declared pitch-linear image layout",
            ));
        }
        Ok(Self {
            buffer,
            layout,
            extra_channels: Vec::new(),
            cmyk_encoding: Default::default(),
        })
    }

    /// Select the CMYK sample convention. ICC device storage defaults to integer ink amounts.
    /// Floating CMYK requires `Complemented` to retain every supplied component word.
    /// Alpha and independently attached scalar inputs keep their own sample conventions.
    pub fn with_cmyk_encoding(
        mut self,
        encoding: crate::CmykSampleEncoding,
    ) -> Result<Self, EncodeError> {
        if !matches!(&self.layout.format.color_spec,
            jxl_gpu_formats::ColorSpecification::Icc(profile) if profile.header().device_space.0 == *b"CMYK")
            || self.layout.format.model != jxl_gpu_formats::ColorModel::IccDevice
        {
            return Err(EncodeError::InvalidSource(
                "CMYK sample convention requires a CMYK ICC input",
            ));
        }
        self.cmyk_encoding = encoding;
        Ok(self)
    }

    #[must_use]
    pub fn cmyk_encoding(&self) -> crate::CmykSampleEncoding {
        self.cmyk_encoding
    }

    /// Attach scalar sources in `VarDctConfig::extra_channels` order (after any packed alpha).
    /// Precision and extents (including intrinsic shifts and the requested per-frame
    /// upsampling factors) are checked against that declaration before admission.
    pub fn with_extra_channels(mut self, channels: Vec<Self>) -> Result<Self, EncodeError> {
        if channels.len() > crate::extra_channel::MAX_EXTRA_CHANNELS
            || channels
                .iter()
                .any(|channel| !channel.extra_channels.is_empty())
        {
            return Err(EncodeError::InvalidSource(
                "extra sources must be flat and within the JPEG XL channel count",
            ));
        }
        self.extra_channels = channels;
        Ok(self)
    }

    #[must_use]
    pub fn extra_channels(&self) -> &[Self] {
        &self.extra_channels
    }
}

/// One mip/layer of a copyable, uncompressed 2D color texture.
///
/// `pixel_format` describes the raw copied texel bytes, including channel order and precision.
/// No sampler, normalization, sRGB conversion, or alpha conversion is applied. The texture
/// must have `COPY_SRC` usage. Multi-planar textures use [`BufferImageSource`] instead.
#[derive(Clone, Debug)]
pub struct TextureImageSource {
    pub texture: Arc<wgpu::Texture>,
    pub texture_format: wgpu::TextureFormat,
    pub pixel_format: PixelFormat,
    pub mip_level: u32,
    pub array_layer: u32,
    pub(crate) extra_channels: Vec<BufferImageSource>,
    pub(crate) cmyk_encoding: crate::CmykSampleEncoding,
}

impl TextureImageSource {
    pub fn new(
        texture: Arc<wgpu::Texture>,
        texture_format: wgpu::TextureFormat,
        pixel_format: PixelFormat,
        mip_level: u32,
        array_layer: u32,
    ) -> Result<Self, EncodeError> {
        let source = Self {
            texture,
            texture_format,
            pixel_format,
            mip_level,
            array_layer,
            extra_channels: Vec::new(),
            cmyk_encoding: Default::default(),
        };
        crate::source_input::FrameInputPlan::new(source.clone().into())?;
        Ok(source)
    }

    /// Attach independent scalar GPU buffers in the image's declared extra-channel order.
    pub fn with_extra_channels(
        mut self,
        channels: Vec<BufferImageSource>,
    ) -> Result<Self, EncodeError> {
        if channels.len() > crate::extra_channel::MAX_EXTRA_CHANNELS
            || channels
                .iter()
                .any(|channel| !channel.extra_channels().is_empty())
        {
            return Err(EncodeError::InvalidSource(
                "extra sources must be flat and within the JPEG XL channel count",
            ));
        }
        self.extra_channels = channels;
        Ok(self)
    }

    /// Select the meaning of CMYK device words, with the same contract as buffer inputs.
    pub fn with_cmyk_encoding(
        mut self,
        encoding: crate::CmykSampleEncoding,
    ) -> Result<Self, EncodeError> {
        if !matches!(&self.pixel_format.color_spec,
            jxl_gpu_formats::ColorSpecification::Icc(profile) if profile.header().device_space.0 == *b"CMYK")
            || self.pixel_format.model != jxl_gpu_formats::ColorModel::IccDevice
        {
            return Err(EncodeError::InvalidSource(
                "CMYK sample convention requires a CMYK ICC input",
            ));
        }
        self.cmyk_encoding = encoding;
        Ok(self)
    }

    #[must_use]
    pub fn extra_channels(&self) -> &[BufferImageSource] {
        &self.extra_channels
    }

    #[must_use]
    pub fn cmyk_encoding(&self) -> crate::CmykSampleEncoding {
        self.cmyk_encoding
    }
}

impl From<BufferImageSource> for GpuFrameSource {
    fn from(source: BufferImageSource) -> Self {
        Self::Buffer(source)
    }
}

impl From<TextureImageSource> for GpuFrameSource {
    fn from(source: TextureImageSource) -> Self {
        Self::Texture(source)
    }
}

impl From<crate::YuvImageSource> for GpuFrameSource {
    fn from(source: crate::YuvImageSource) -> Self {
        Self::Yuv(source)
    }
}

impl From<&crate::YuvImageSource> for GpuFrameSource {
    fn from(source: &crate::YuvImageSource) -> Self {
        source.clone().into()
    }
}

impl From<&BufferImageSource> for GpuFrameSource {
    fn from(source: &BufferImageSource) -> Self {
        source.clone().into()
    }
}

impl From<&TextureImageSource> for GpuFrameSource {
    fn from(source: &TextureImageSource) -> Self {
        source.clone().into()
    }
}

impl From<&GpuFrameSource> for GpuFrameSource {
    fn from(source: &GpuFrameSource) -> Self {
        source.clone()
    }
}

#[derive(Clone, Debug)]
pub enum GpuFrameSource {
    Buffer(BufferImageSource),
    Texture(TextureImageSource),
    Yuv(crate::YuvImageSource),
}

impl GpuFrameSource {
    #[must_use]
    pub fn pixel_format(&self) -> &PixelFormat {
        match self {
            Self::Buffer(source) => &source.layout.format,
            Self::Texture(source) => &source.pixel_format,
            Self::Yuv(source) => source.pixel_format(),
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

    /// Source-family preflight; request-dependent geometry and resource checks belong in submit.
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

    /// Source-family preflight; request-dependent geometry and resource checks belong in submit.
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

    pub(crate) fn memory_budget(&self) -> &MemoryBudget {
        self.context.memory_budget()
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
