//! Caller-selected Modular/VarDCT frames under one checked original color/alpha image contract.

use std::task::{Context, Poll};

use crate::session::FrameCoding;
use crate::{
    BufferImageSource, CodestreamAssembler, Determinism, EncodeError, EncodeProfile, EncodeSession,
    EncoderBufferPoolStats, EncoderCapabilities, FrameEncodeRequest, FrameIndex, FrameOptions,
    FrameSubmission, GpuEncodeBackend, GpuEncodeJob, GpuEncoder, GpuFrameArtifacts, GpuFrameSource,
    ImageSequenceDescriptor, LosslessModularBackend, LosslessModularConfig, LosslessModularJob,
    LosslessModularMemoryPlan, ProfileCapability, ProgressivePlan, SessionDescriptor,
    UnsupportedFeature, VarDctBackend, VarDctColorTransform, VarDctConfig, VarDctJob,
    VarDctMemoryPlan, VarDctStrategy, VarDctStrategyMap, WgpuContext,
};

/// Explicit physical-frame coding choice. This is not a content-adaptive selection policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixedModeFrameEncoding {
    Modular,
    VarDct,
}

/// Source geometry and transform policy for the VarDCT frames of a mixed sequence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum VarDctTransformSelection {
    /// Independent DCT8 blocks with per-frame source extents and replicated edges.
    #[default]
    TiledDct8,
    /// Every VarDCT source has exactly this transform's extent.
    Single(VarDctStrategy),
    /// Every VarDCT source has exactly this checked map's extent.
    Map(VarDctStrategyMap),
}

/// Fixed policies for both frame codecs. Both use `vardct`'s stream-wide source
/// color/alpha channels, precision, source color and image color options.
///
/// The default VarDCT domain is `Original`. An explicit XYB configuration is rejected:
/// the image-wide XYB flag cannot change between physical frames, and the Modular backend
/// encodes original components. Arbitrary extra channels, other source formats and
/// pre-transform references require a broader common image contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedModeConfig {
    pub modular: LosslessModularConfig,
    pub vardct: VarDctConfig,
    pub vardct_transform: VarDctTransformSelection,
}

impl Default for MixedModeConfig {
    fn default() -> Self {
        Self {
            modular: Default::default(),
            vardct: VarDctConfig {
                color_transform: VarDctColorTransform::Original,
                ..Default::default()
            },
            vardct_transform: Default::default(),
        }
    }
}

/// Exact codec-specific resources for one physical frame. No extra GPU buffers are
/// introduced by mode selection; both variants draw from the same context budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixedModeMemoryPlan {
    Modular(LosslessModularMemoryPlan),
    VarDct(VarDctMemoryPlan),
}

impl MixedModeMemoryPlan {
    /// Peak owned bytes for this job, including the selected codec's retained readback.
    /// Streamed Modular jobs reserve one live batch at a time.
    #[must_use]
    pub const fn owned_bytes_per_job(self) -> u64 {
        match self {
            Self::Modular(plan) => plan.owned_bytes_per_job,
            Self::VarDct(plan) => plan.owned_bytes_per_job,
        }
    }
}

struct MixedModeBackend {
    modular: LosslessModularBackend,
    vardct: VarDctBackend,
    modular_coding: FrameCoding,
    vardct_coding: FrameCoding,
    capabilities: EncoderCapabilities,
}

impl MixedModeBackend {
    fn new(context: &WgpuContext, config: MixedModeConfig) -> Result<Self, EncodeError> {
        if !config.vardct.extra_channels.is_empty() {
            return Err(EncodeError::InvalidConfiguration(
                "mixed sequences require Modular support for independently declared extra sources",
            ));
        }
        if config.vardct.color_transform != VarDctColorTransform::Original {
            return Err(EncodeError::InvalidConfiguration(
                "mixed Modular/VarDCT sequences require original-component coding",
            ));
        }
        let samples = config.vardct.sample_format;
        let modular_coding = FrameCoding {
            profile: EncodeProfile::ModularLossless {
                sample_bit_depth: samples.bit_depth(),
            },
            progressive: ProgressivePlan::single(),
        };
        let vardct_coding = FrameCoding {
            profile: EncodeProfile::VarDct {
                quantization: config.vardct.quantization,
            },
            progressive: config.vardct.progressive.clone(),
        };
        let vardct = match config.vardct_transform {
            VarDctTransformSelection::TiledDct8 => {
                VarDctBackend::new_tiled_dct8_with_config(context, config.vardct)?
            }
            VarDctTransformSelection::Single(strategy) => {
                VarDctBackend::new_with_config(context, strategy, config.vardct)?
            }
            VarDctTransformSelection::Map(map) => {
                VarDctBackend::new_with_strategy_map(context, map, config.vardct)?
            }
        };
        let modular = LosslessModularBackend::with_config(context, config.modular);
        let mut capabilities = vardct.capabilities().clone();
        capabilities
            .profiles
            .push(ProfileCapability::ModularLossless {
                min_bits_per_sample: samples.bits_per_sample(),
                max_bits_per_sample: samples.bits_per_sample(),
                exponent_bits_per_sample: samples.exponent_bits(),
            });
        capabilities.determinism = capabilities
            .determinism
            .min(modular.capabilities().determinism);
        for &stage in &modular.capabilities().implemented_stages {
            if !capabilities.implemented_stages.contains(&stage) {
                capabilities.implemented_stages.push(stage);
            }
        }
        Ok(Self {
            modular,
            vardct,
            modular_coding,
            vardct_coding,
            capabilities,
        })
    }

    fn coding(&self, encoding: MixedModeFrameEncoding) -> &FrameCoding {
        match encoding {
            MixedModeFrameEncoding::Modular => &self.modular_coding,
            MixedModeFrameEncoding::VarDct => &self.vardct_coding,
        }
    }

    fn validate_request(
        &self,
        source: &BufferImageSource,
        request: &FrameEncodeRequest,
    ) -> Result<(), EncodeError> {
        if !self.vardct.matches_source_format(&source.layout.format) {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        if request.options.save_before_color_transform {
            return Err(EncodeError::InvalidConfiguration(
                "mixed sequences support post-color-transform references only",
            ));
        }
        self.capabilities.negotiate(request)?;
        match request.profile {
            EncodeProfile::ModularLossless { .. } => {
                self.modular.capabilities().negotiate(request)?
            }
            EncodeProfile::VarDct { .. } => self.vardct.capabilities().negotiate(request)?,
        }
        Ok(())
    }

    fn memory_plan(
        &self,
        source: &BufferImageSource,
        request: &FrameEncodeRequest,
    ) -> Result<MixedModeMemoryPlan, EncodeError> {
        self.validate_request(source, request)?;
        match request.profile {
            EncodeProfile::ModularLossless { .. } => self
                .modular
                .memory_plan_for_request(source, request)
                .map(MixedModeMemoryPlan::Modular),
            EncodeProfile::VarDct { .. } => self
                .vardct
                .memory_plan_for_request(source, request)
                .map(MixedModeMemoryPlan::VarDct),
        }
    }
}

impl GpuEncodeBackend for MixedModeBackend {
    type Job = MixedModeJob;

    fn capabilities(&self) -> &EncoderCapabilities {
        &self.capabilities
    }

    fn supports_input(&self, source: &GpuFrameSource) -> bool {
        let GpuFrameSource::Buffer(buffer) = source else {
            return false;
        };
        self.vardct.matches_source_format(&buffer.layout.format)
            && (self.modular.supports_input(source) || self.vardct.supports_input(source))
    }

    fn submit(
        &self,
        context: &WgpuContext,
        source: GpuFrameSource,
        request: &FrameEncodeRequest,
    ) -> Result<Self::Job, EncodeError> {
        let GpuFrameSource::Buffer(buffer) = &source else {
            return Err(UnsupportedFeature::InputFormat.into());
        };
        self.validate_request(buffer, request)?;
        let state = match request.profile {
            EncodeProfile::ModularLossless { .. } => {
                MixedModeJobState::Modular(self.modular.submit(context, source, request)?)
            }
            EncodeProfile::VarDct { .. } => {
                MixedModeJobState::VarDct(Box::new(self.vardct.submit(context, source, request)?))
            }
        };
        Ok(MixedModeJob { state })
    }
}

enum MixedModeJobState {
    Modular(LosslessModularJob),
    VarDct(Box<VarDctJob>),
}

/// A job retaining the chosen codec's completion state, resources and memory permit.
pub struct MixedModeJob {
    state: MixedModeJobState,
}

impl GpuEncodeJob for MixedModeJob {
    fn poll_complete(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<GpuFrameArtifacts, EncodeError>> {
        match &mut self.state {
            MixedModeJobState::Modular(job) => job.poll_complete(cx),
            MixedModeJobState::VarDct(job) => job.poll_complete(cx),
        }
    }

    fn wait(self) -> Result<GpuFrameArtifacts, EncodeError> {
        match self.state {
            MixedModeJobState::Modular(job) => job.wait(),
            MixedModeJobState::VarDct(job) => (*job).wait(),
        }
    }
}

/// Reusable GPU encoders for explicit per-frame Modular/VarDCT selection.
/// The selected VarDCT transform policy constrains only VarDCT source extents.
///
/// A Modular frame preserves its supplied source words exactly. A sequence containing
/// quantized VarDCT frames does not promise lossless presentations.
///
/// ```no_run
/// # use jxl_wgpu_encode::{AnimationHeader, BufferImageSource, EncodeError, FrameCrop,
/// #     FrameOptions, MixedModeConfig, MixedModeEncoder, MixedModeFrameEncoding,
/// #     ImageSequenceDescriptor, WgpuContext};
/// # fn layers(context: WgpuContext, background: BufferImageSource,
/// #     patch: BufferImageSource) -> Result<Vec<u8>, EncodeError> {
/// let encoder = MixedModeEncoder::new(context, MixedModeConfig::default())?;
/// let mut sequence = encoder.begin_sequence(ImageSequenceDescriptor::new(
///     640, 480, AnimationHeader::Still,
/// )?)?;
/// let base = sequence.submit_frame(
///     background, MixedModeFrameEncoding::VarDct, FrameOptions::default(),
/// )?; // hidden background in reference slot zero
/// let final_layer = sequence.submit_last_frame(
///     patch, MixedModeFrameEncoding::Modular,
///     FrameOptions {
///         crop: Some(FrameCrop::new(100, 100, 64, 64)?),
///         ..FrameOptions::default()
///     },
/// )?;
/// sequence.insert(final_layer.wait()?)?;
/// sequence.insert(base.wait()?)?;
/// sequence.finish_indexed_container(Default::default(), Default::default())
/// # }
/// ```
pub struct MixedModeEncoder {
    encoder: GpuEncoder<MixedModeBackend>,
}

impl MixedModeEncoder {
    pub fn new(context: WgpuContext, config: MixedModeConfig) -> Result<Self, EncodeError> {
        let backend = MixedModeBackend::new(&context, config)?;
        Ok(Self {
            encoder: GpuEncoder::new(context, backend),
        })
    }

    #[must_use]
    pub fn capabilities(&self) -> &EncoderCapabilities {
        self.encoder.capabilities()
    }

    #[must_use]
    pub fn in_flight_memory_stats(&self) -> jxl_wgpu::MemoryBudgetSnapshot {
        self.encoder.memory_stats()
    }

    /// Reusable idle allocations of the Modular backend; active reservations are separate.
    #[must_use]
    pub fn buffer_pool_stats(&self) -> EncoderBufferPoolStats {
        self.encoder.backend().modular.buffer_pool_stats()
    }

    pub fn set_buffer_pool_limit(&self, bytes: u64) {
        self.encoder.backend().modular.set_buffer_pool_limit(bytes);
    }

    pub fn clear_buffer_pool(&self) {
        self.encoder.backend().modular.clear_buffer_pool();
    }

    /// Common color channels and precision for both frame codecs; alpha shares this precision.
    #[must_use]
    pub fn sample_format(&self) -> crate::ColorSampleFormat {
        self.encoder.backend().vardct.sample_format()
    }

    /// Common alpha association for both frame codecs.
    #[must_use]
    pub fn alpha_association(&self) -> Option<crate::AlphaAssociation> {
        self.encoder.backend().vardct.alpha_association()
    }

    pub fn begin_sequence(
        &self,
        descriptor: ImageSequenceDescriptor,
    ) -> Result<MixedModeSequenceSession, EncodeError> {
        let backend = self.encoder.backend();
        let (header, metadata_permit) = backend
            .vardct
            .sequence_header(&descriptor)?
            .finish(self.encoder.memory_budget())?;
        let assembler = CodestreamAssembler::new(header)?;
        let session = self.encoder.begin_session(SessionDescriptor {
            profile: backend.modular_coding.profile,
            progressive: backend.modular_coding.progressive.clone(),
            minimum_determinism: Determinism::SameDevice,
            animation: descriptor.animation(),
            canvas_width: descriptor.canvas_width(),
            canvas_height: descriptor.canvas_height(),
        })?;
        Ok(MixedModeSequenceSession {
            metadata_permit,
            session,
            assembler,
            descriptor,
        })
    }
}

/// One ordered sequence using the common image contract and existing session state machine.
pub struct MixedModeSequenceSession {
    metadata_permit: Option<jxl_wgpu::MemoryPermit>,
    session: EncodeSession<MixedModeBackend>,
    assembler: CodestreamAssembler,
    descriptor: ImageSequenceDescriptor,
}

impl MixedModeSequenceSession {
    #[must_use]
    pub fn descriptor(&self) -> &ImageSequenceDescriptor {
        &self.descriptor
    }

    #[must_use]
    pub fn next_frame_index(&self) -> FrameIndex {
        self.session.next_frame_index()
    }

    /// Checks the next physical frame without reserving memory or advancing sequence state.
    pub fn memory_plan(
        &self,
        source: &BufferImageSource,
        encoding: MixedModeFrameEncoding,
        options: FrameOptions,
        is_last: bool,
    ) -> Result<MixedModeMemoryPlan, EncodeError> {
        let backend = self.session.encoder().backend();
        let request = self
            .session
            .request_for(options, is_last, backend.coding(encoding))?;
        backend.memory_plan(source, &request)
    }

    pub fn submit_frame(
        &mut self,
        source: BufferImageSource,
        encoding: MixedModeFrameEncoding,
        options: FrameOptions,
    ) -> Result<FrameSubmission<MixedModeJob>, EncodeError> {
        self.submit(source, encoding, options, false)
    }

    pub fn submit_last_frame(
        &mut self,
        source: BufferImageSource,
        encoding: MixedModeFrameEncoding,
        options: FrameOptions,
    ) -> Result<FrameSubmission<MixedModeJob>, EncodeError> {
        self.submit(source, encoding, options, true)
    }

    fn submit(
        &mut self,
        source: BufferImageSource,
        encoding: MixedModeFrameEncoding,
        options: FrameOptions,
        last: bool,
    ) -> Result<FrameSubmission<MixedModeJob>, EncodeError> {
        let coding = self.session.encoder().backend().coding(encoding).clone();
        self.session
            .submit_with_coding(GpuFrameSource::Buffer(source), options, last, &coding)
    }

    /// Inserts validated GPU packets; completion and insertion order may differ from frame order.
    pub fn insert(&mut self, frame: GpuFrameArtifacts) -> Result<(), EncodeError> {
        self.assembler.insert(frame).map_err(Into::into)
    }

    pub fn finish_raw(self) -> Result<Vec<u8>, EncodeError> {
        let _metadata_permit = self.metadata_permit;
        self.session.ensure_closed()?;
        self.assembler.finish_raw().map_err(Into::into)
    }

    pub fn finish_container(self) -> Result<Vec<u8>, EncodeError> {
        let _metadata_permit = self.metadata_permit;
        self.session.ensure_closed()?;
        self.assembler.finish_container()
    }

    pub fn finish_indexed_container(
        self,
        inventory_limits: jxl_gpu_bitstream::InventoryLimits,
        index_limits: jxl_gpu_bitstream::FrameIndexLimits,
    ) -> Result<Vec<u8>, EncodeError> {
        let _metadata_permit = self.metadata_permit;
        self.session.ensure_closed()?;
        self.assembler
            .finish_indexed_container(inventory_limits, index_limits)
    }
}
