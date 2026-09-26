use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use jxl_gpu_bitstream::{
    ACCELERATION_INDEX_BOX_TYPE, BitWriter, ContainerBox, Gray8AccelerationIndex,
    write_container_with_boxes,
};

use super::color::ModularImageMetadata;
use super::dispatch::{LosslessModularBackend, ModularGroupPlan};
use super::entropy::{EncodedGroup, EntropyCode};
use super::grid::{LosslessModularGroup, LosslessModularGroupGrid};
use super::lz77::LosslessModularLz77;
use super::memory::{LosslessModularMemoryLimits, LosslessModularMemoryPlan};
use super::predictor::{LosslessModularPredictor, LosslessModularWeightedPredictor};
use super::source::lossless_modular_source_spec;
use super::streaming::LosslessModularJob;
use super::transform::{
    ModularTransformPlan, PlannedPalette, PlannedRct, TransformOperation, ValidatedPaletteCounts,
};
use super::types::{
    AlphaAssociation, LosslessModularFormat, LosslessModularTreeMode, ModularArtifactHeader,
    ModularEvent, modular_sample_depth,
};
use crate::ImageOptions;
use crate::frame_header::FrameHeaderPlan;
use crate::prefix::{LZ77_SYMBOLS, PrefixCode, RAW_SYMBOLS, RawPrefixCode};
use crate::source_color::SourceColorEncoding;
use crate::source_color::icc::{DEFAULT_PROFILE_LIMIT, PreparedImageHeader};
use crate::{
    AnimationHeader, BackendError, BitFragment, CodestreamAssembler, Determinism, EncodeError,
    EncodeProfile, EncodeSession, EncoderBufferPoolStats, EncoderCapabilities, FrameEncodeRequest,
    FrameGroupLayout, FrameIndex, FrameOptions, FramePacketSet, FrameSubmission,
    GpuAccelerationArtifact, GpuEncoder, GpuFrameArtifacts, GpuFrameSource, GroupPacket,
    GroupPacketKind, ProgressivePlan, SessionDescriptor, WgpuContext, assemble_frame,
};

/// Convenience API that produces a complete raw codestream or deterministic
/// `jxlc` container from a GPU-resident Gray, GrayAlpha, RGB, or RGBA integer/IEEE floating buffer.
pub struct LosslessModularEncoder {
    encoder: GpuEncoder<LosslessModularBackend>,
    image_options: ImageOptions,
    alpha_association: AlphaAssociation,
    max_icc_profile_bytes: u64,
}

impl LosslessModularEncoder {
    #[must_use]
    pub fn new(context: WgpuContext) -> Self {
        Self::with_config(context, Default::default())
    }

    /// Creates an encoder with explicit group geometry, MA-tree placement, RCT and prediction.
    #[must_use]
    pub fn with_config(context: WgpuContext, config: super::types::LosslessModularConfig) -> Self {
        let backend = LosslessModularBackend::with_config(&context, config);
        Self {
            encoder: GpuEncoder::new(context, backend),
            image_options: ImageOptions::default(),
            alpha_association: AlphaAssociation::default(),
            max_icc_profile_bytes: DEFAULT_PROFILE_LIMIT,
        }
    }

    /// Creates an encoder with an explicit multi-group MA-tree placement policy.
    #[must_use]
    pub fn with_tree_mode(context: WgpuContext, tree_mode: LosslessModularTreeMode) -> Self {
        Self::with_config(
            context,
            super::types::LosslessModularConfig {
                tree_mode,
                ..Default::default()
            },
        )
    }

    /// Creates an encoder with an application-selected idle buffer retention limit.
    ///
    /// The limit is independent of the context's live-job [`jxl_wgpu::MemoryBudget`]. A value of
    /// zero creates buffers on demand and drops them immediately after each mapping callback.
    #[must_use]
    pub fn with_buffer_pool_limit(context: WgpuContext, limit_bytes: u64) -> Self {
        let backend = LosslessModularBackend::new(&context);
        backend.set_buffer_pool_limit(limit_bytes);
        Self {
            encoder: GpuEncoder::new(context, backend),
            image_options: ImageOptions::default(),
            alpha_association: AlphaAssociation::default(),
            max_icc_profile_bytes: DEFAULT_PROFILE_LIMIT,
        }
    }

    /// Selects the declaration shared by subsequent stills and animations.
    /// Source primaries/white/transfer or ICC bytes continue to come from each source format.
    /// ICC input requires the selected intent to match the profile header.
    pub fn with_image_options(mut self, options: ImageOptions) -> Result<Self, EncodeError> {
        options.validate()?;
        self.image_options = options;
        Ok(self)
    }

    /// Declares how source color is associated with alpha; no source samples are changed.
    /// Associated input requires GrayAlpha or RGBA, and is checked before job admission.
    #[must_use]
    pub fn with_alpha_association(mut self, association: AlphaAssociation) -> Self {
        self.alpha_association = association;
        self
    }

    /// Bounds the original embedded ICC profile before header allocation. The default is
    /// 16 MiB; zero disables ICC input. JPEG XL's metadata limits also remain enforced.
    #[must_use]
    pub fn with_max_icc_profile_bytes(mut self, max_bytes: u64) -> Self {
        self.max_icc_profile_bytes = max_bytes;
        self
    }

    #[must_use]
    pub fn capabilities(&self) -> &EncoderCapabilities {
        self.encoder.capabilities()
    }

    #[must_use]
    pub fn config(&self) -> super::types::LosslessModularConfig {
        self.encoder.backend().config()
    }

    /// Reports aggregate owned bytes retained by currently live encode jobs.
    #[must_use]
    pub fn in_flight_memory_stats(&self) -> jxl_wgpu::MemoryBudgetSnapshot {
        self.encoder.memory_stats()
    }

    /// Computes source, ICC header, artifact, and readback bytes before a still submission.
    /// Animations retain one separately admitted image header across all frame jobs.
    pub fn memory_plan(
        &self,
        source: impl Into<GpuFrameSource>,
    ) -> Result<LosslessModularMemoryPlan, EncodeError> {
        let source = crate::source_input::FrameInputPlan::new(source.into())?;
        let mut plan = self.encoder.backend().memory_plan(&source)?;
        self.alpha_association.validate(plan.format)?;
        source.validate_alpha_association(self.alpha_association)?;
        let spec = lossless_modular_source_spec(&source.layout.format)?;
        let header = image_header(
            source.layout.extent.width,
            source.layout.extent.height,
            spec.packing.format,
            spec.packing.bits_per_sample,
            spec.packing.exponent_bits_per_sample,
            AnimationHeader::Still,
            ModularImageMetadata::new(
                spec.color.clone(),
                self.image_options,
                self.alpha_association,
                self.max_icc_profile_bytes,
            )
            .with_inputs(&self.config()),
        )?;
        plan.icc_profile_bytes = header.icc_profile_bytes;
        plan.icc_storage_bytes = header.icc_storage_bytes;
        plan.extra_channel_metadata_bytes = header.extra_storage_bytes;
        plan.owned_bytes_per_job = plan
            .owned_bytes_per_job
            .checked_add(header.icc_storage_bytes)
            .and_then(|bytes| bytes.checked_add(header.extra_storage_bytes))
            .ok_or(EncodeError::InvalidConfiguration("ICC job size overflow"))?;
        plan.addressed_bytes_per_job = plan
            .addressed_bytes_per_job
            .checked_add(header.icc_storage_bytes)
            .and_then(|bytes| bytes.checked_add(header.extra_storage_bytes))
            .ok_or(EncodeError::InvalidConfiguration(
                "ICC addressed size overflow",
            ))?;
        Ok(plan)
    }

    #[must_use]
    pub fn memory_limits(&self) -> LosslessModularMemoryLimits {
        self.encoder.backend().memory_limits()
    }

    /// Reports reusable encoder-owned GPU buffers and cumulative reuse counters.
    #[must_use]
    pub fn buffer_pool_stats(&self) -> EncoderBufferPoolStats {
        self.encoder.backend().buffer_pool_stats()
    }

    /// Changes the maximum idle allocation bytes retained for later submissions.
    pub fn set_buffer_pool_limit(&self, limit_bytes: u64) {
        self.encoder.backend().set_buffer_pool_limit(limit_bytes);
    }

    /// Clears idle buffers; in-flight sets from before the clear are discarded on completion.
    pub fn clear_buffer_pool(&self) {
        self.encoder.backend().clear_buffer_pool();
    }

    pub fn submit(
        &self,
        source: impl Into<GpuFrameSource>,
    ) -> Result<LosslessModularSubmission, EncodeError> {
        let source = source.into();
        self.memory_plan(&source)?;
        self.submit_inner(source, false)
    }

    pub fn submit_container(
        &self,
        source: impl Into<GpuFrameSource>,
    ) -> Result<LosslessModularSubmission, EncodeError> {
        let source = source.into();
        self.memory_plan(&source)?;
        self.submit_inner(source, true)
    }

    pub fn encode(&self, source: impl Into<GpuFrameSource>) -> Result<Vec<u8>, EncodeError> {
        self.submit(source)?.wait()
    }

    pub fn encode_container(
        &self,
        source: impl Into<GpuFrameSource>,
    ) -> Result<Vec<u8>, EncodeError> {
        self.submit_container(source)?.wait()
    }

    /// Starts a timed animation. For a layered still use [`Self::begin_sequence`].
    pub fn begin_animation(
        &self,
        descriptor: LosslessModularAnimationDescriptor,
    ) -> Result<LosslessModularAnimationSession, EncodeError> {
        if !descriptor.animation.is_animation() {
            return Err(EncodeError::InvalidConfiguration(
                "begin_animation requires an animation timebase",
            ));
        }
        self.begin_sequence(descriptor)
    }

    /// Starts a reusable layered-still or animation sequence.
    ///
    /// Every returned frame submission supports both [`Future`] and blocking
    /// [`FrameSubmission::wait`]. Submissions do not borrow this session, so multiple GPU frames
    /// can remain in flight and their completed artifacts may be inserted in any order.
    pub fn begin_sequence(
        &self,
        descriptor: LosslessModularSequenceDescriptor,
    ) -> Result<LosslessModularSequenceSession, EncodeError> {
        self.config()
            .color_transform
            .resolve(descriptor.format, descriptor.exponent_bits_per_sample)?;
        let (codestream_header, metadata_permit) = image_header_with_plan(
            &descriptor.header,
            descriptor.format,
            descriptor.bits_per_sample,
            descriptor.exponent_bits_per_sample,
            ModularImageMetadata::new(
                descriptor.color.clone(),
                self.image_options,
                self.alpha_association,
                self.max_icc_profile_bytes,
            )
            .with_inputs(&self.config()),
        )?
        .finish(self.encoder.memory_budget())?;
        let session = self.encoder.begin_session(SessionDescriptor {
            profile: EncodeProfile::ModularLossless {
                sample_bit_depth: descriptor.sample_bit_depth(),
            },
            progressive: ProgressivePlan::single(),
            // A sequence can contain explicitly converted YUV sources. Word-preserving
            // inputs retain the backend's stronger guarantee without requiring it here.
            minimum_determinism: Determinism::SameDevice,
            animation: descriptor.animation,
            canvas_width: descriptor.canvas_width,
            canvas_height: descriptor.canvas_height,
        })?;
        Ok(LosslessModularSequenceSession {
            alpha_association: self.alpha_association,
            session,
            assembler: CodestreamAssembler::new(codestream_header)?
                .with_preview(descriptor.preview()),
            descriptor,
            metadata_permit,
        })
    }

    fn submit_inner(
        &self,
        source: impl Into<GpuFrameSource>,
        container: bool,
    ) -> Result<LosslessModularSubmission, EncodeError> {
        // Preserve typed address/device-limit failures before the generic
        // backend admission predicate maps unsupported inputs to InputFormat.
        let source = crate::source_input::FrameInputPlan::new(source.into())?;
        self.encoder.backend().memory_plan(&source)?;
        source.validate_alpha_association(self.alpha_association)?;
        let width = source.layout.extent.width;
        let height = source.layout.extent.height;
        let source_spec = lossless_modular_source_spec(&source.layout.format)?;
        let format = source_spec.packing.format;
        let group_grid =
            LosslessModularGroupGrid::for_extent(width, height, self.config().group_size)?;
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::ModularLossless {
                sample_bit_depth: modular_sample_depth(
                    source_spec.packing.bits_per_sample,
                    source_spec.packing.exponent_bits_per_sample,
                ),
            },
            progressive: ProgressivePlan::single(),
            minimum_determinism: source.determinism(),
            animation: AnimationHeader::Still,
            canvas_width: width,
            canvas_height: height,
            options: FrameOptions::default(),
        };
        // Finish validating image metadata before any GPU admission or submission.
        let (codestream_header, metadata_permit) = image_header(
            width,
            height,
            format,
            source_spec.packing.bits_per_sample,
            source_spec.packing.exponent_bits_per_sample,
            AnimationHeader::Still,
            ModularImageMetadata::new(
                source_spec.color.clone(),
                self.image_options,
                self.alpha_association,
                self.max_icc_profile_bytes,
            )
            .with_inputs(&self.config()),
        )?
        .finish(self.encoder.memory_budget())?;
        let frame = self.encoder.submit_frame(source.into_source(), request)?;
        Ok(LosslessModularSubmission {
            frame: Some(frame),
            codestream_header: Some(codestream_header),
            metadata_permit,
            default_color: source_spec.color == SourceColorEncoding::default()
                && self.image_options == ImageOptions::default(),
            container,
            group_grid,
            format,
            bits_per_sample: source_spec.packing.bits_per_sample,
            exponent_bits_per_sample: source_spec.packing.exponent_bits_per_sample,
        })
    }
}

/// Compatibility name for [`LosslessModularSequenceDescriptor`].
pub type LosslessModularAnimationDescriptor = LosslessModularSequenceDescriptor;

/// Compatibility name for [`LosslessModularSequenceSession`].
pub type LosslessModularAnimationSession = LosslessModularSequenceSession;

/// Stream-wide contract for one lossless Modular layered still or animation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LosslessModularSequenceDescriptor {
    canvas_width: u32,
    canvas_height: u32,
    format: LosslessModularFormat,
    bits_per_sample: u8,
    exponent_bits_per_sample: u8,
    animation: AnimationHeader,
    color: SourceColorEncoding,
    header: crate::image_sequence::ImageHeaderPlan,
}

impl LosslessModularSequenceDescriptor {
    /// Declares an independently supplied GPU preview in the same image color/precision contract.
    #[must_use]
    pub fn with_preview(mut self, size: crate::PreviewSize) -> Self {
        self.header = self.header.with_preview(size);
        self
    }

    #[must_use]
    pub const fn preview(&self) -> Option<crate::PreviewSize> {
        self.header.preview()
    }
    /// Infers stream components, precision and color from a supported source format.
    ///
    /// Frame storage may differ, but every submitted frame must have the same encoded color
    /// declaration (custom xy rounded to 1e-6 and gamma to 1e-7, or identical original ICC
    /// bytes). Image white and rendering intent come from the encoder's
    /// [`ImageOptions`]; the intent must agree with an embedded profile.
    pub fn from_pixel_format(
        canvas_width: u32,
        canvas_height: u32,
        format: &jxl_gpu_formats::PixelFormat,
        animation: AnimationHeader,
    ) -> Result<Self, EncodeError> {
        let spec = lossless_modular_source_spec(format)?;
        let mut descriptor = Self::with_precision(
            canvas_width,
            canvas_height,
            spec.packing.format,
            spec.packing.bits_per_sample,
            spec.packing.exponent_bits_per_sample,
            animation,
        )?;
        descriptor.color = spec.color;
        Ok(descriptor)
    }

    pub fn new(
        canvas_width: u32,
        canvas_height: u32,
        format: LosslessModularFormat,
        bits_per_sample: u8,
        animation: AnimationHeader,
    ) -> Result<Self, EncodeError> {
        Self::with_precision(
            canvas_width,
            canvas_height,
            format,
            bits_per_sample,
            0,
            animation,
        )
    }

    /// Describes native IEEE binary16 or binary32 source components.
    pub fn new_float(
        canvas_width: u32,
        canvas_height: u32,
        format: LosslessModularFormat,
        bits_per_sample: u8,
        animation: AnimationHeader,
    ) -> Result<Self, EncodeError> {
        Self::from_pixel_format(
            canvas_width,
            canvas_height,
            &format.float_pixel_format(bits_per_sample)?,
            animation,
        )
    }

    fn with_precision(
        canvas_width: u32,
        canvas_height: u32,
        format: LosslessModularFormat,
        bits_per_sample: u8,
        exponent_bits_per_sample: u8,
        animation: AnimationHeader,
    ) -> Result<Self, EncodeError> {
        // The header uses the same checked precision as still-image serialization.
        let header =
            crate::image_sequence::ImageHeaderPlan::new(canvas_width, canvas_height, animation)?;
        image_header_with_plan(
            &header,
            format,
            bits_per_sample,
            exponent_bits_per_sample,
            ModularImageMetadata::default(),
        )?;
        Ok(Self {
            canvas_width,
            canvas_height,
            format,
            bits_per_sample,
            exponent_bits_per_sample,
            animation,
            color: SourceColorEncoding::default(),
            header,
        })
    }

    #[must_use]
    pub const fn canvas_width(&self) -> u32 {
        self.canvas_width
    }

    #[must_use]
    pub const fn canvas_height(&self) -> u32 {
        self.canvas_height
    }

    #[must_use]
    pub const fn format(&self) -> LosslessModularFormat {
        self.format
    }

    #[must_use]
    pub const fn bits_per_sample(&self) -> u8 {
        self.bits_per_sample
    }

    #[must_use]
    pub const fn sample_bit_depth(&self) -> jxl_gpu_bitstream::SampleBitDepth {
        modular_sample_depth(self.bits_per_sample, self.exponent_bits_per_sample)
    }

    #[must_use]
    pub const fn animation(&self) -> AnimationHeader {
        self.animation
    }
}

/// Multi-frame assembly state for a lossless Modular layered still or animation.
pub struct LosslessModularSequenceSession {
    alpha_association: AlphaAssociation,
    session: EncodeSession<LosslessModularBackend>,
    assembler: CodestreamAssembler,
    descriptor: LosslessModularSequenceDescriptor,
    metadata_permit: Option<jxl_wgpu::MemoryPermit>,
}

impl LosslessModularSequenceSession {
    /// GPU job footprint. Completed preview storage is admitted separately at its actual size.
    pub fn preview_memory_plan(
        &self,
        source: impl Into<GpuFrameSource>,
        options: FrameOptions,
    ) -> Result<LosslessModularMemoryPlan, EncodeError> {
        let source = crate::source_input::FrameInputPlan::new(source.into())?;
        self.validate_source(&source)?;
        let request = self.session.preview_request(
            &self.assembler,
            options,
            &self.session.default_coding(),
        )?;
        self.session
            .encoder()
            .backend()
            .memory_plan_for_request(&source, &request)
    }
    /// Submits the declared preview without advancing or closing the main frame sequence.
    pub fn submit_preview(
        &mut self,
        source: impl Into<GpuFrameSource>,
        options: FrameOptions,
    ) -> Result<crate::PreviewSubmission<LosslessModularJob>, EncodeError> {
        let source = crate::source_input::FrameInputPlan::new(source.into())?;
        self.validate_source(&source)?;
        self.session.submit_preview(
            &mut self.assembler,
            source.into_source(),
            options,
            &self.session.default_coding(),
        )
    }

    pub fn insert_preview(&mut self, preview: crate::EncodedPreview) -> Result<(), EncodeError> {
        Ok(self.assembler.insert_preview(preview)?)
    }
    #[must_use]
    pub const fn descriptor(&self) -> &LosslessModularSequenceDescriptor {
        &self.descriptor
    }

    #[must_use]
    pub const fn next_frame_index(&self) -> FrameIndex {
        self.session.next_frame_index()
    }

    pub fn submit_frame(
        &mut self,
        source: impl Into<GpuFrameSource>,
        options: FrameOptions,
    ) -> Result<FrameSubmission<LosslessModularJob>, EncodeError> {
        let source = crate::source_input::FrameInputPlan::new(source.into())?;
        self.validate_source(&source)?;
        self.session.submit_frame(source.into_source(), options)
    }

    pub fn submit_last_frame(
        &mut self,
        source: impl Into<GpuFrameSource>,
        options: FrameOptions,
    ) -> Result<FrameSubmission<LosslessModularJob>, EncodeError> {
        let source = crate::source_input::FrameInputPlan::new(source.into())?;
        self.validate_source(&source)?;
        self.session
            .submit_last_frame(source.into_source(), options)
    }

    /// Inserts one completed GPU frame. Completion order need not match frame order.
    pub fn insert(&mut self, frame: GpuFrameArtifacts) -> Result<(), EncodeError> {
        self.assembler.insert(frame)?;
        Ok(())
    }

    pub fn finish_raw(self) -> Result<Vec<u8>, EncodeError> {
        let _metadata_permit = self.metadata_permit;
        self.session.ensure_closed()?;
        Ok(self.assembler.finish_raw()?)
    }

    pub fn finish_container(self) -> Result<Vec<u8>, EncodeError> {
        let _metadata_permit = self.metadata_permit;
        self.session.ensure_closed()?;
        self.assembler.finish_container()
    }

    /// Finishes with a header-validated `jxli` index. See
    /// [`CodestreamAssembler::finish_indexed_container`] for limits and restart semantics.
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

    fn validate_source(
        &self,
        source: &crate::source_input::FrameInputPlan,
    ) -> Result<(), EncodeError> {
        source.validate_alpha_association(self.alpha_association)?;
        let spec = lossless_modular_source_spec(&source.layout.format)?;
        if spec.packing.format != self.descriptor.format
            || spec.packing.bits_per_sample != self.descriptor.bits_per_sample
            || spec.packing.exponent_bits_per_sample != self.descriptor.exponent_bits_per_sample
            || spec.color != self.descriptor.color
        {
            return Err(EncodeError::InvalidConfiguration(
                "every frame must match the stream format, sample precision and color",
            ));
        }
        Ok(())
    }
}

/// A `Future` with an executor-independent blocking counterpart.
pub struct LosslessModularSubmission {
    frame: Option<FrameSubmission<LosslessModularJob>>,
    codestream_header: Option<BitFragment>,
    metadata_permit: Option<jxl_wgpu::MemoryPermit>,
    container: bool,
    group_grid: LosslessModularGroupGrid,
    format: LosslessModularFormat,
    bits_per_sample: u8,
    exponent_bits_per_sample: u8,
    default_color: bool,
}

impl LosslessModularSubmission {
    #[must_use]
    pub const fn format(&self) -> LosslessModularFormat {
        self.format
    }

    /// Total encoded bits per component, including floating exponent and sign bits.
    #[must_use]
    pub const fn bits_per_sample(&self) -> u8 {
        self.bits_per_sample
    }

    #[must_use]
    pub const fn sample_bit_depth(&self) -> jxl_gpu_bitstream::SampleBitDepth {
        modular_sample_depth(self.bits_per_sample, self.exponent_bits_per_sample)
    }
    /// Exact row-major group grid dispatched by this submission.
    #[must_use]
    pub const fn group_grid(&self) -> LosslessModularGroupGrid {
        self.group_grid
    }

    /// Canonical descriptors for the independently executed GPU workgroups.
    pub fn ordered_groups(&self) -> impl ExactSizeIterator<Item = LosslessModularGroup> {
        self.group_grid.ordered_groups()
    }

    pub fn wait(mut self) -> Result<Vec<u8>, EncodeError> {
        let frame = self
            .frame
            .take()
            .expect("a lossless submission can only complete once")
            .wait()?;
        self.assemble(frame)
    }

    fn assemble(&mut self, frame: GpuFrameArtifacts) -> Result<Vec<u8>, EncodeError> {
        // The private Gray8 shortcut remains restricted to its original default color contract.
        let acceleration = frame.acceleration.filter(|_| self.default_color);
        let fused_group_size = acceleration
            .as_ref()
            .map(|_| {
                frame
                    .packets
                    .packets()
                    .first()
                    .ok_or_else(|| EncodeError::Backend("gray8 frame has no group packet".into()))
                    .map(|packet| packet.payload.len())
            })
            .transpose()?;
        let encoded_frame = assemble_frame(frame.packets)?;
        let header = self
            .codestream_header
            .take()
            .expect("unassembled image header");
        let header_bytes = header.bytes().len();
        let mut codestream = header.into_bytes();
        codestream
            .try_reserve_exact(encoded_frame.bytes().len())
            .map_err(|_| crate::PacketError::SizeOverflow)?;
        codestream.extend_from_slice(encoded_frame.bytes());
        if !self.container {
            return Ok(codestream);
        }

        let Some(acceleration) = acceleration else {
            // The current private acceleration-index schema describes one contiguous token span.
            // Multi-group output remains a fully standard deterministic `jxlc` container, without
            // inventing an incompatible extension record.
            return Ok(write_container_with_boxes(&codestream, &[])?);
        };
        let group_size = fused_group_size.ok_or_else(|| {
            EncodeError::Backend("gray8 acceleration metadata requires a fused group".into())
        })?;
        let bytes_before_group = encoded_frame
            .bytes()
            .len()
            .checked_sub(group_size)
            .ok_or_else(|| EncodeError::Backend("gray8 group size exceeds frame size".into()))?;
        let group_start = header_bytes
            .checked_add(bytes_before_group)
            .ok_or_else(|| EncodeError::Backend("gray8 codestream size overflow".into()))?;

        let GpuAccelerationArtifact::Gray8Prefix {
            width,
            height,
            token_bit_offset_in_group,
            token_bit_len,
            raw_prefix,
            lz77_prefix,
        } = acceleration;
        let group_start_bits = u64::try_from(group_start)
            .ok()
            .and_then(|value| value.checked_mul(8))
            .ok_or_else(|| EncodeError::Backend("gray8 token offset overflow".into()))?;
        let token_bit_offset = group_start_bits
            .checked_add(token_bit_offset_in_group)
            .ok_or_else(|| EncodeError::Backend("gray8 token offset overflow".into()))?;
        let index = Gray8AccelerationIndex::new(
            &codestream,
            width,
            height,
            token_bit_offset,
            token_bit_len,
            raw_prefix,
            lz77_prefix,
        )?;
        let payload = index.serialize();
        Ok(write_container_with_boxes(
            &codestream,
            &[ContainerBox {
                box_type: ACCELERATION_INDEX_BOX_TYPE,
                payload: &payload,
            }],
        )?)
    }
}

impl Future for LosslessModularSubmission {
    type Output = Result<Vec<u8>, EncodeError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let submission = self.get_mut();
        let frame = submission
            .frame
            .as_mut()
            .expect("a lossless submission must not be polled after completion");
        match Pin::new(frame).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                submission.frame.take();
                let result = result.and_then(|frame| submission.assemble(frame));
                submission.codestream_header.take();
                submission.metadata_permit.take();
                Poll::Ready(result)
            }
        }
    }
}

pub(super) struct PacketBuildInput<'a> {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) group_grid: LosslessModularGroupGrid,
    pub(super) format: LosslessModularFormat,
    pub(super) bits_per_sample: u8,
    pub(super) exponent_bits_per_sample: u8,
    pub(super) tree_mode: LosslessModularTreeMode,
    pub(super) transforms: Arc<ModularTransformPlan>,
    pub(super) predictor: LosslessModularPredictor,
    pub(super) weighted_predictor: LosslessModularWeightedPredictor,
    pub(super) lz77: LosslessModularLz77,
    pub(super) frame: &'a FrameHeaderPlan,
    pub(super) group_plans: &'a [ModularGroupPlan],
    pub(super) bytes: &'a [u8],
}

type RawHistograms = [[u64; RAW_SYMBOLS]; 4];
type Lz77Histograms = [[u64; LZ77_SYMBOLS]; 4];
pub(super) type DistanceCode = RawPrefixCode<RAW_SYMBOLS>;

pub(super) fn accumulate_artifact_histograms(
    channel: usize,
    artifact: &ValidatedModularArtifact<'_>,
    aggregate_raw: &mut RawHistograms,
    aggregate_lz77: &mut Lz77Histograms,
    aggregate_distance: &mut [u64; RAW_SYMBOLS],
) -> Result<(), EncodeError> {
    // The fixed four-leaf MA tree selects channels 0/1/2 separately and all later channels
    // together. Squeeze's additional residual planes therefore share the final distribution.
    let channel = channel.min(3);
    for (total, count) in aggregate_raw[channel]
        .iter_mut()
        .zip(artifact.header.raw_counts)
    {
        *total = total
            .checked_add(u64::from(count))
            .ok_or_else(|| invalid_gpu_artifact("aggregate raw histogram overflow"))?;
    }
    for (total, count) in aggregate_lz77[channel]
        .iter_mut()
        .zip(artifact.header.lz77_counts)
    {
        *total = total
            .checked_add(u64::from(count))
            .ok_or_else(|| invalid_gpu_artifact("aggregate LZ77 histogram overflow"))?;
    }
    for (total, count) in aggregate_distance
        .iter_mut()
        .zip(artifact.header.distance_counts)
    {
        *total = total
            .checked_add(u64::from(count))
            .ok_or_else(|| invalid_gpu_artifact("aggregate distance histogram overflow"))?;
    }
    Ok(())
}

pub(super) fn build_distance_code(
    mode: LosslessModularLz77,
    counts: &[u64; RAW_SYMBOLS],
) -> Result<Option<DistanceCode>, EncodeError> {
    if mode == LosslessModularLz77::ZeroRuns {
        if counts.iter().any(|&count| count != 0) {
            return Err(invalid_gpu_artifact(
                "zero-run mode contains an explicit distance histogram",
            ));
        }
        return Ok(None);
    }
    let mut frequencies = [0; RAW_SYMBOLS];
    for (frequency, &count) in frequencies.iter_mut().zip(counts) {
        *frequency = count
            .checked_mul(256)
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| invalid_gpu_artifact("distance histogram scaling overflow"))?;
    }
    // 33 symbols fit in six bits; eight leaves room to favour observed distances while
    // bounding both the metadata code lengths and host prefix-table construction.
    DistanceCode::from_counts_bounded(&frequencies, 8).map(Some)
}

pub(super) fn build_prefix_codes(
    format: LosslessModularFormat,
    bits_per_sample: u8,
    predictor: LosslessModularPredictor,
    transforms: &ModularTransformPlan,
    aggregate_raw: &RawHistograms,
    aggregate_lz77: &Lz77Histograms,
) -> Result<[PrefixCode; 4], EncodeError> {
    let channels = transforms.prefix_channels;
    let unused = PrefixCode::fixed_unused_channel();
    let mut codes = [unused.clone(), unused.clone(), unused.clone(), unused];
    for channel in 0..channels {
        let transformed_extra_token = u8::from(format.color_channel_count() == 3);
        // Other predictors may overshoot the sample range, including with custom WP
        // coefficients. Their wrapping residuals use the full integer alphabet.
        let wide_samples = bits_per_sample > 14
            || predictor != LosslessModularPredictor::Gradient
            || transforms.extended_prediction_domain;
        let max_raw_token = if predictor != LosslessModularPredictor::Gradient
            || transforms.extended_prediction_domain
        {
            RAW_SYMBOLS - 1
        } else if (15..=16).contains(&bits_per_sample) {
            18
        } else {
            usize::from(
                bits_per_sample
                    .saturating_add(1)
                    .saturating_add(transformed_extra_token)
                    .min((RAW_SYMBOLS - 1) as u8),
            )
        };
        codes[channel] = PrefixCode::from_aggregated_counts(
            &aggregate_raw[channel],
            &aggregate_lz77[channel],
            max_raw_token,
            wide_samples,
        )?;
    }
    Ok(codes)
}

pub(super) struct ModularPacketAssembler {
    width: u32,
    height: u32,
    group_grid: LosslessModularGroupGrid,
    format: LosslessModularFormat,
    bits_per_sample: u8,
    exponent_bits_per_sample: u8,
    tree_mode: LosslessModularTreeMode,
    transforms: Arc<ModularTransformPlan>,
    predictor: LosslessModularPredictor,
    weighted_predictor: LosslessModularWeightedPredictor,
    lz77: LosslessModularLz77,
    frame: FrameHeaderPlan,
    entropy: Arc<EntropyCode>,
    packets: Vec<GroupPacket>,
    single_group: Option<BitWriter>,
    token_bit_offset_in_group: u64,
    next_group: u32,
}

pub(super) struct ModularPacketConfig {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) group_grid: LosslessModularGroupGrid,
    pub(super) format: LosslessModularFormat,
    pub(super) bits_per_sample: u8,
    pub(super) exponent_bits_per_sample: u8,
    pub(super) tree_mode: LosslessModularTreeMode,
    pub(super) transforms: Arc<ModularTransformPlan>,
    pub(super) predictor: LosslessModularPredictor,
    pub(super) weighted_predictor: LosslessModularWeightedPredictor,
    pub(super) lz77: LosslessModularLz77,
    pub(super) frame: FrameHeaderPlan,
}

impl ModularPacketAssembler {
    pub(super) fn new(
        config: ModularPacketConfig,
        entropy: Arc<EntropyCode>,
    ) -> Result<Self, EncodeError> {
        let ModularPacketConfig {
            width,
            height,
            group_grid,
            format,
            bits_per_sample,
            exponent_bits_per_sample,
            tree_mode,
            transforms,
            predictor,
            weighted_predictor,
            lz77,
            frame,
        } = config;
        if !entropy.matches_lz77(lz77) {
            return Err(invalid_gpu_artifact(
                "distance code does not match LZ77 policy",
            ));
        }
        let (packets, single_group, token_bit_offset_in_group) = if group_grid.groups == 1 {
            // The exact palette size is a validated GPU result, available when this group arrives.
            (Vec::new(), Some(BitWriter::new()), 0)
        } else {
            let layout = FrameGroupLayout::new(group_grid.lf_groups, group_grid.groups, 1)?;
            let mut packets = Vec::with_capacity(layout.toc_entries());
            let mut dc_global = BitWriter::new();
            write_dc_global(
                &mut dc_global,
                &entropy,
                TransformHeader {
                    operations: &transforms.global_operations,
                    palette_counts: None,
                },
                predictor,
                weighted_predictor,
            )?;
            entropy.write_empty_stream(&mut dc_global)?;
            dc_global.align_to_byte()?;
            packets.push(GroupPacket::new(
                GroupPacketKind::DcGlobal,
                dc_global.into_bytes(),
            ));
            if !transforms
                .streams
                .iter()
                .any(|stream| stream.route == crate::extra_channel::input::ScalarRoute::Lf)
            {
                for group in 0..group_grid.lf_groups {
                    packets.push(GroupPacket::new(
                        GroupPacketKind::DcGroup(group),
                        Vec::new(),
                    ));
                }
            }
            // Lossless Modular has no VarDCT HF-global payload.
            packets.push(GroupPacket::new(GroupPacketKind::AcGlobal, Vec::new()));
            (packets, None, 0)
        };
        Ok(Self {
            width,
            height,
            group_grid,
            format,
            bits_per_sample,
            exponent_bits_per_sample,
            tree_mode,
            transforms,
            predictor,
            weighted_predictor,
            lz77,
            frame,
            entropy,
            packets,
            single_group,
            token_bit_offset_in_group,
            next_group: 0,
        })
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn entropy(&self) -> &EntropyCode {
        &self.entropy
    }

    pub(super) fn push_group(
        &mut self,
        group_index: u32,
        artifacts: &[ValidatedModularArtifact<'_>],
        encoded: Option<EncodedGroup<'_>>,
    ) -> Result<(), EncodeError> {
        if group_index != self.next_group {
            return Err(EncodeError::Backend(
                "GPU artifact groups are not in canonical order".into(),
            ));
        }
        let stream =
            self.transforms
                .streams
                .get(group_index as usize)
                .ok_or(BackendError::Invariant(
                    "Modular stream index exceeds frame plan",
                ))?;
        let topology = self.transforms.stream(group_index)?;
        let channels = topology.channels.len();
        if artifacts.len() != channels {
            return Err(EncodeError::Backend(
                "GPU group does not contain every Modular channel".into(),
            ));
        }
        let palette_counts = artifacts[0].palette_counts;
        if palette_counts.is_some() != topology.palette.is_some() {
            return Err(invalid_gpu_artifact(
                "palette metadata does not match the group policy",
            ));
        }
        if let Some(group) = &mut self.single_group {
            write_dc_global(
                group,
                &self.entropy,
                TransformHeader {
                    operations: &topology.operations,
                    palette_counts,
                },
                self.predictor,
                self.weighted_predictor,
            )?;
            self.token_bit_offset_in_group = u64::try_from(group.bit_len())
                .map_err(|_| EncodeError::Backend("gray8 token offset overflow".into()))?;
            self.entropy.write_stream(group, artifacts, encoded)?;
        } else {
            let mut pass_group = BitWriter::new();
            let use_global_tree = self.tree_mode == LosslessModularTreeMode::SharedGlobal;
            // Each local Modular stream declares its own WP coefficients and transform list.
            pass_group.write_bits(u64::from(use_global_tree), 1)?;
            write_weighted_predictor(&mut pass_group, self.weighted_predictor)?;
            write_transforms(
                &mut pass_group,
                TransformHeader {
                    operations: &topology.operations,
                    palette_counts,
                },
            )?;
            if !use_global_tree {
                write_ma_config(&mut pass_group, &self.entropy, self.predictor)?;
            }
            self.entropy
                .write_stream(&mut pass_group, artifacts, encoded)?;
            pass_group.align_to_byte()?;
            self.packets.push(GroupPacket::new(
                if stream.route == crate::extra_channel::input::ScalarRoute::Lf {
                    GroupPacketKind::DcGroup(stream.region.index)
                } else {
                    GroupPacketKind::AcGroup {
                        pass: 0,
                        group: stream.region.index,
                    }
                },
                pass_group.into_bytes(),
            ));
        }
        self.next_group = self
            .next_group
            .checked_add(1)
            .ok_or_else(|| EncodeError::Backend("Modular group index overflow".into()))?;
        Ok(())
    }

    pub(super) fn finish(
        mut self,
    ) -> Result<(FramePacketSet, Option<GpuAccelerationArtifact>), EncodeError> {
        if self.next_group as usize != self.transforms.streams.len() {
            return Err(EncodeError::Backend(
                "GPU artifact stream ended before every Modular group".into(),
            ));
        }
        if let Some(mut group) = self.single_group.take() {
            let token_bit_end = u64::try_from(group.bit_len())
                .map_err(|_| EncodeError::Backend("gray8 token length overflow".into()))?;
            let token_bit_len = token_bit_end
                .checked_sub(self.token_bit_offset_in_group)
                .ok_or_else(|| EncodeError::Backend("gray8 token length underflow".into()))?;
            group.align_to_byte()?;
            let packets = FramePacketSet::new(
                frame_header(&self.frame, self.group_grid.group_size)?,
                FrameGroupLayout::new(1, 1, 1)?,
                [GroupPacket::new(
                    GroupPacketKind::Single,
                    group.into_bytes(),
                )],
            )?;
            let acceleration = if let EntropyCode::Prefix { codes, .. } = &*self.entropy {
                (self.format == LosslessModularFormat::Gray
                    && self.bits_per_sample == 8
                    && self.exponent_bits_per_sample == 0
                    && self.predictor == LosslessModularPredictor::Gradient
                    && !self.transforms.extended_prediction_domain
                    && self.lz77 == LosslessModularLz77::ZeroRuns
                    && self.frame.sampling().is_unscaled())
                .then(|| GpuAccelerationArtifact::Gray8Prefix {
                    width: self.width,
                    height: self.height,
                    token_bit_offset_in_group: self.token_bit_offset_in_group,
                    token_bit_len,
                    raw_prefix: std::array::from_fn(|index| codes[0].raw_entries()[index]),
                    lz77_prefix: codes[0].lz77_entries(),
                })
            } else {
                None
            };
            return Ok((packets, acceleration));
        }
        let layout = FrameGroupLayout::new(self.group_grid.lf_groups, self.group_grid.groups, 1)?;
        Ok((
            FramePacketSet::new(
                frame_header(&self.frame, self.group_grid.group_size)?,
                layout,
                self.packets,
            )?,
            None,
        ))
    }
}

pub(super) fn build_packets(
    input: PacketBuildInput<'_>,
) -> Result<(FramePacketSet, Option<GpuAccelerationArtifact>), EncodeError> {
    let PacketBuildInput {
        width,
        height,
        group_grid,
        format,
        bits_per_sample,
        exponent_bits_per_sample,
        tree_mode,
        transforms,
        predictor,
        weighted_predictor,
        lz77,
        frame,
        group_plans,
        bytes,
    } = input;
    let expected_artifacts = transforms.dispatches as usize;
    if group_plans.len() != expected_artifacts {
        return Err(EncodeError::Backend(
            "GPU group plan does not match the frame grid".into(),
        ));
    }
    let mut artifacts = Vec::with_capacity(group_plans.len());
    let mut aggregate_raw = [[0u64; RAW_SYMBOLS]; 4];
    let mut aggregate_lz77 = [[0u64; LZ77_SYMBOLS]; 4];
    let mut aggregate_distance = [0u64; RAW_SYMBOLS];
    for plan in group_plans {
        let start = usize::try_from(plan.artifact_byte_offset)
            .map_err(|_| EncodeError::Backend("GPU artifact offset overflow".into()))?;
        let end = plan
            .artifact_byte_offset
            .checked_add(plan.output_size)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| EncodeError::Backend("GPU artifact range overflow".into()))?;
        let artifact_bytes = bytes
            .get(start..end)
            .ok_or_else(|| EncodeError::Backend("GPU group artifact is truncated".into()))?;
        let artifact = parse_planned_artifact(plan, artifact_bytes)?;
        accumulate_artifact_histograms(
            plan.channel as usize,
            &artifact,
            &mut aggregate_raw,
            &mut aggregate_lz77,
            &mut aggregate_distance,
        )?;
        artifacts.push(artifact);
    }

    let codes = build_prefix_codes(
        format,
        bits_per_sample,
        predictor,
        &transforms,
        &aggregate_raw,
        &aggregate_lz77,
    )?;
    let mut assembler = ModularPacketAssembler::new(
        ModularPacketConfig {
            width,
            height,
            group_grid,
            format,
            bits_per_sample,
            exponent_bits_per_sample,
            tree_mode,
            transforms: Arc::clone(&transforms),
            predictor,
            weighted_predictor,
            lz77,
            frame: frame.clone(),
        },
        Arc::new(EntropyCode::Prefix {
            codes: Box::new(codes),
            distance: build_distance_code(lz77, &aggregate_distance)?,
        }),
    )?;
    let mut start = 0;
    for group_index in 0..transforms.streams.len() as u32 {
        let channels = transforms.stream(group_index)?.channels.len();
        let end = start + channels;
        for (channel, plan) in group_plans[start..end].iter().enumerate() {
            if plan.group_index != group_index || plan.channel != channel as u32 {
                return Err(BackendError::Invariant(
                    "GPU group plan channel order is not canonical",
                )
                .into());
            }
        }
        assembler.push_group(group_index, &artifacts[start..end], None)?;
        start = end;
    }
    assembler.finish()
}

#[derive(Clone, Copy)]
pub(super) struct ValidatedModularArtifact<'a> {
    pub(super) header: ModularArtifactHeader,
    pub(super) events: &'a [ModularEvent],
    pub(super) palette_counts: Option<ValidatedPaletteCounts>,
}

pub(super) fn parse_planned_artifact<'a>(
    plan: &ModularGroupPlan,
    bytes: &'a [u8],
) -> Result<ValidatedModularArtifact<'a>, EncodeError> {
    // Check status before reading dynamic dimensions or granting token authority.
    parse_group_artifact_header(plan.max_events, bytes)?;
    let palette_counts = if let Some(palette) = plan.palette {
        let offset = palette.counts_byte_offset;
        let offset = usize::try_from(offset)
            .map_err(|_| invalid_gpu_artifact("palette metadata offset overflow"))?;
        let end = offset
            .checked_add(8)
            .ok_or_else(|| invalid_gpu_artifact("palette metadata offset overflow"))?;
        let value = bytes
            .get(offset..end)
            .ok_or_else(|| invalid_gpu_artifact("palette metadata is truncated"))?;
        let count = u32::from_le_bytes(value[..4].try_into().expect("four-byte palette count"));
        let deltas = u32::from_le_bytes(value[4..].try_into().expect("four-byte delta count"));
        if palette.capacity.entries() != plan.width {
            return Err(invalid_gpu_artifact(
                "palette capacity disagrees with channel geometry",
            ));
        }
        Some(palette.capacity.validate(count, deltas)?)
    } else {
        None
    };
    let mut artifact = parse_group_artifact(
        palette_counts.map_or(plan.width, ValidatedPaletteCounts::entries),
        plan.height,
        plan.max_events,
        bytes,
    )?;
    artifact.palette_counts = palette_counts;
    Ok(artifact)
}

pub(super) fn parse_group_artifact_header(
    max_events: usize,
    bytes: &[u8],
) -> Result<ModularArtifactHeader, EncodeError> {
    let header_bytes = bytes
        .get(..std::mem::size_of::<ModularArtifactHeader>())
        .ok_or_else(|| EncodeError::Backend("GPU artifact header is truncated".into()))?;
    let header = bytemuck::try_cast_slice::<u8, ModularArtifactHeader>(header_bytes)
        .map_err(|_| EncodeError::Backend("GPU artifact header has an invalid ABI layout".into()))?
        .first()
        .copied()
        .ok_or_else(|| EncodeError::Backend("GPU artifact header is truncated".into()))?;
    let event_count = usize::try_from(header.event_count)
        .map_err(|_| EncodeError::Backend("GPU event count overflow".into()))?;
    if header.event_count == u32::MAX - 1 {
        return Err(BackendError::ModularSqueezeOverflow.into());
    }
    if header.event_count == u32::MAX - 2 {
        return Err(BackendError::ModularPaletteOverflow.into());
    }
    if header.event_count == u32::MAX - 3 {
        return Err(invalid_gpu_artifact("GPU palette lookup failed"));
    }
    if event_count > max_events {
        return Err(EncodeError::Backend(
            "GPU emitted more token events than the output allocation".into(),
        ));
    }
    let required_bytes = event_count
        .checked_mul(std::mem::size_of::<ModularEvent>())
        .and_then(|event_bytes| {
            std::mem::size_of::<ModularArtifactHeader>().checked_add(event_bytes)
        })
        .ok_or_else(|| EncodeError::Backend("GPU event count overflow".into()))?;
    if bytes.len() < required_bytes {
        return Err(EncodeError::Backend("GPU event stream is truncated".into()));
    }
    Ok(header)
}

pub(super) fn parse_group_artifact<'a>(
    width: u32,
    height: u32,
    max_events: usize,
    bytes: &'a [u8],
) -> Result<ValidatedModularArtifact<'a>, EncodeError> {
    let header = parse_group_artifact_header(max_events, bytes)?;
    let event_count = usize::try_from(header.event_count)
        .map_err(|_| EncodeError::Backend("GPU event count overflow".into()))?;
    let event_bytes = event_count
        .checked_mul(std::mem::size_of::<ModularEvent>())
        .ok_or_else(|| EncodeError::Backend("GPU event count overflow".into()))?;
    let required_bytes = std::mem::size_of::<ModularArtifactHeader>()
        .checked_add(event_bytes)
        .ok_or_else(|| EncodeError::Backend("GPU event count overflow".into()))?;
    let events = bytes
        .get(std::mem::size_of::<ModularArtifactHeader>()..required_bytes)
        .ok_or_else(|| EncodeError::Backend("GPU event stream is truncated".into()))?;
    let events = bytemuck::try_cast_slice::<u8, ModularEvent>(events)
        .map_err(|_| EncodeError::Backend("GPU event stream has an invalid ABI layout".into()))?;

    validate_gpu_artifacts(width, height, &header, events)?;
    Ok(ValidatedModularArtifact {
        header,
        events,
        palette_counts: None,
    })
}

pub(super) fn write_events(
    output: &mut BitWriter,
    code: &PrefixCode,
    distance_code: Option<&DistanceCode>,
    events: &[ModularEvent],
) -> Result<(), EncodeError> {
    for event in events {
        match (event.kind, distance_code) {
            (0, _) => {
                code.write_raw(output, event.token, event.extra_bit_count, event.extra_bits)?
            }
            (1, None) => {
                code.write_run(output, event.token, event.extra_bit_count, event.extra_bits)?
            }
            (2, Some(_)) => {
                code.write_match(output, event.token, event.extra_bit_count, event.extra_bits)?
            }
            (3, Some(distance_code)) => distance_code.write_raw(
                output,
                event.token,
                event.extra_bit_count,
                event.extra_bits,
            )?,
            _ => {
                return Err(EncodeError::Backend(
                    "GPU emitted an unknown token kind".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_gpu_artifacts(
    width: u32,
    height: u32,
    header: &ModularArtifactHeader,
    events: &[ModularEvent],
) -> Result<(), EncodeError> {
    let mut raw_counts = [0u32; RAW_SYMBOLS];
    let mut lz77_counts = [0u32; LZ77_SYMBOLS];
    let mut distance_counts = [0u32; RAW_SYMBOLS];
    let mut sample_count = 0u64;
    let mut pending_distance = None;

    for event in events {
        if pending_distance.is_some() && event.kind != 3 {
            return Err(invalid_gpu_artifact("LZ77 match is missing its distance"));
        }
        match event.kind {
            0 => {
                let token = usize::try_from(event.token)
                    .map_err(|_| invalid_gpu_artifact("raw token overflow"))?;
                if token >= RAW_SYMBOLS {
                    return Err(invalid_gpu_artifact("impossible raw token"));
                }
                let expected_nbits = event.token.saturating_sub(1);
                if event.extra_bit_count != expected_nbits
                    || !canonical_extra_bits(event.extra_bit_count, event.extra_bits)
                {
                    return Err(invalid_gpu_artifact("non-canonical raw token"));
                }
                raw_counts[token] = raw_counts[token]
                    .checked_add(1)
                    .ok_or_else(|| invalid_gpu_artifact("raw histogram overflow"))?;
                sample_count = sample_count
                    .checked_add(1)
                    .ok_or_else(|| invalid_gpu_artifact("sample count overflow"))?;
            }
            1 | 2 => {
                let token = usize::try_from(event.token)
                    .map_err(|_| invalid_gpu_artifact("LZ77 token overflow"))?;
                if token > crate::prefix::MAX_LZ77_TOKEN {
                    return Err(invalid_gpu_artifact("impossible LZ77 token"));
                }
                let expected_nbits = if event.token < 16 {
                    0
                } else {
                    event.token - 12
                };
                if event.extra_bit_count != expected_nbits
                    || !canonical_extra_bits(event.extra_bit_count, event.extra_bits)
                {
                    return Err(invalid_gpu_artifact("non-canonical LZ77 token"));
                }
                if event.kind == 1 {
                    raw_counts[0] = raw_counts[0]
                        .checked_add(1)
                        .ok_or_else(|| invalid_gpu_artifact("raw histogram overflow"))?;
                } else {
                    pending_distance = Some(sample_count);
                }
                lz77_counts[token] = lz77_counts[token]
                    .checked_add(1)
                    .ok_or_else(|| invalid_gpu_artifact("LZ77 histogram overflow"))?;
                let encoded_value = if event.token < 16 {
                    u64::from(event.token)
                } else {
                    (1u64 << event.extra_bit_count) + u64::from(event.extra_bits)
                };
                sample_count = sample_count
                    .checked_add(encoded_value + 7 + u64::from(event.kind == 1))
                    .ok_or_else(|| invalid_gpu_artifact("sample count overflow"))?;
            }
            3 => {
                let available = pending_distance
                    .take()
                    .ok_or_else(|| invalid_gpu_artifact("distance without an LZ77 match"))?;
                let token = usize::try_from(event.token)
                    .map_err(|_| invalid_gpu_artifact("distance token overflow"))?;
                if token >= RAW_SYMBOLS
                    || event.extra_bit_count != event.token.saturating_sub(1)
                    || !canonical_extra_bits(event.extra_bit_count, event.extra_bits)
                {
                    return Err(invalid_gpu_artifact("non-canonical distance token"));
                }
                let coded = if event.token == 0 {
                    0
                } else {
                    (1u64 << event.extra_bit_count) + u64::from(event.extra_bits)
                };
                let distance = coded
                    .checked_sub(119)
                    .filter(|&distance| {
                        distance != 0 && distance <= (1 << 20) && distance <= available
                    })
                    .ok_or_else(|| {
                        invalid_gpu_artifact("LZ77 distance exceeds the channel history")
                    })?;
                debug_assert!(distance <= available);
                distance_counts[token] = distance_counts[token]
                    .checked_add(1)
                    .ok_or_else(|| invalid_gpu_artifact("distance histogram overflow"))?;
            }
            _ => return Err(invalid_gpu_artifact("unknown token kind")),
        }
    }

    if pending_distance.is_some() {
        return Err(invalid_gpu_artifact("LZ77 match is missing its distance"));
    }
    if raw_counts != header.raw_counts
        || lz77_counts != header.lz77_counts
        || distance_counts != header.distance_counts
    {
        return Err(invalid_gpu_artifact(
            "token histograms do not match the event stream",
        ));
    }
    let expected_samples = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| invalid_gpu_artifact("image sample count overflow"))?;
    if sample_count != expected_samples {
        return Err(invalid_gpu_artifact(
            "event stream does not cover the image exactly",
        ));
    }
    Ok(())
}

fn canonical_extra_bits(nbits: u32, bits: u32) -> bool {
    match nbits {
        0 => bits == 0,
        1..=31 => bits < (1u32 << nbits),
        _ => false,
    }
}

fn invalid_gpu_artifact(reason: &'static str) -> EncodeError {
    BackendError::InvalidArtifact(reason).into()
}

struct TransformHeader<'a> {
    operations: &'a [TransformOperation],
    palette_counts: Option<ValidatedPaletteCounts>,
}

fn write_dc_global(
    output: &mut BitWriter,
    entropy: &EntropyCode,
    transforms: TransformHeader<'_>,
    predictor: LosslessModularPredictor,
    weighted_predictor: LosslessModularWeightedPredictor,
) -> Result<(), EncodeError> {
    // Handcrafted Modular metadata adapted from zune-jpegxl 0.5.2. See this crate's
    // `THIRD_PARTY.md` and `LICENSES/zune-jpegxl-MIT.txt`.
    output.write_bits(1, 1)?; // default LF-channel dequantization
    output.write_bits(1, 1)?; // GlobalModular is present
    write_ma_config(output, entropy, predictor)?;
    output.write_bits(1, 1)?;
    write_weighted_predictor(output, weighted_predictor)?;
    write_transforms(output, transforms)
}

fn write_weighted_predictor(
    output: &mut BitWriter,
    predictor: LosslessModularWeightedPredictor,
) -> Result<(), EncodeError> {
    let default = predictor == LosslessModularWeightedPredictor::default();
    output.write_bits(u64::from(default), 1)?;
    if !default {
        for coefficient in predictor.coefficients() {
            output.write_bits(u64::from(coefficient), 5)?;
        }
        for weight in predictor.max_weights() {
            output.write_bits(u64::from(weight), 4)?;
        }
    }
    Ok(())
}

fn write_transforms(
    output: &mut BitWriter,
    transforms: TransformHeader<'_>,
) -> Result<(), EncodeError> {
    let has_palette = transforms
        .operations
        .iter()
        .any(|operation| matches!(operation, TransformOperation::Palette(_)));
    if has_palette != transforms.palette_counts.is_some() {
        return Err(invalid_gpu_artifact(
            "palette metadata does not match the transform plan",
        ));
    }
    let count = transforms.operations.len();
    if count > 273 {
        return Err(EncodeError::InvalidModularTransformCount { count });
    } else if count >= 18 {
        output.write_bits(3, 2)?;
        output.write_bits((count - 18) as u64, 8)?;
    } else if count >= 2 {
        output.write_bits(2, 2)?;
        output.write_bits((count - 2) as u64, 4)?;
    } else {
        output.write_bits(count as u64, 2)?;
    }
    for operation in transforms.operations {
        match operation {
            TransformOperation::Rct(rct) => write_rct_parameters(output, *rct)?,
            TransformOperation::Palette(palette) => {
                let counts = transforms
                    .palette_counts
                    .ok_or_else(|| invalid_gpu_artifact("missing palette counts"))?;
                palette
                    .capacity
                    .validate(counts.entries(), counts.deltas())?;
                write_palette_parameters(output, *palette, counts)?;
            }
            TransformOperation::Squeeze(steps) => {
                output.write_bits(2, 2)?;
                let count = steps.len();
                let (selector, base, bits) = if count <= 16 {
                    (1, 1, 4)
                } else if count <= 72 {
                    (2, 9, 6)
                } else {
                    (3, 41, 8)
                };
                output.write_bits(selector, 2)?;
                output.write_bits((count - base) as u64, bits)?;
                for step in steps {
                    output.write_bits(u64::from(step.horizontal), 1)?;
                    output.write_bits(u64::from(step.in_place), 1)?;
                    write_transform_begin(output, step.range.begin)?;
                    if step.range.count <= 3 {
                        output.write_bits(u64::from(step.range.count - 1), 2)?;
                    } else {
                        output.write_bits(3, 2)?;
                        output.write_bits(u64::from(step.range.count - 4), 4)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn write_palette_parameters(
    output: &mut BitWriter,
    palette: PlannedPalette,
    counts: ValidatedPaletteCounts,
) -> Result<(), EncodeError> {
    output.write_bits(1, 2)?; // Palette
    output.write_bits(0, 2)?; // begin channel U32 selector
    output.write_bits(u64::from(palette.range.begin), 3)?;
    match palette.range.count {
        1 => output.write_bits(0, 2)?,
        3 => output.write_bits(1, 2)?,
        4 => output.write_bits(2, 2)?,
        _ => {
            output.write_bits(3, 2)?;
            output.write_bits(u64::from(palette.range.count - 1), 13)?;
        }
    }
    let colors = counts.entries() - counts.deltas();
    let (selector, base, bits) = if colors < 256 {
        (0, 0, 8)
    } else if colors < 1280 {
        (1, 256, 10)
    } else if colors < 5376 {
        (2, 1280, 12)
    } else {
        (3, 5376, 16)
    };
    output.write_bits(selector, 2)?;
    output.write_bits(u64::from(colors - base), bits)?;
    if counts.deltas() != 0 {
        let (selector, base, bits) = if counts.deltas() < 257 {
            (1, 1, 8)
        } else if counts.deltas() < 1281 {
            (2, 257, 10)
        } else {
            (3, 1281, 16)
        };
        output.write_bits(selector, 2)?;
        output.write_bits(u64::from(counts.deltas() - base), bits)?;
    } else {
        output.write_bits(0, 2)?; // no delta entries
    }
    output.write_bits(
        u64::from(
            palette
                .delta_predictor
                .map_or(0, LosslessModularPredictor::value),
        ),
        4,
    )?;
    Ok(())
}

fn write_rct_parameters(output: &mut BitWriter, rct: PlannedRct) -> Result<(), EncodeError> {
    output.write_bits(0, 2)?; // reversible color transform
    write_transform_begin(output, rct.begin)?;
    let value = rct.rct_type.value();
    // U32(Val(6), Bits(2), BitsOffset(4, 2), BitsOffset(6, 10)).
    if value == 6 {
        output.write_bits(0, 2)?;
    } else if value < 4 {
        output.write_bits(1, 2)?;
        output.write_bits(u64::from(value), 2)?;
    } else if value < 18 {
        output.write_bits(2, 2)?;
        output.write_bits(u64::from(value - 2), 4)?;
    } else {
        output.write_bits(3, 2)?;
        output.write_bits(u64::from(value - 10), 6)?;
    }
    Ok(())
}

fn write_transform_begin(output: &mut BitWriter, begin: u32) -> Result<(), EncodeError> {
    let (selector, base, bits) = match begin {
        0..8 => (0, 0, 3),
        8..72 => (1, 8, 6),
        72..1096 => (2, 72, 10),
        1096..=9287 => (3, 1096, 13),
        _ => {
            return Err(
                BackendError::Invariant("planned transform begin exceeds wire range").into(),
            );
        }
    };
    output.write_bits(selector, 2)?;
    output.write_bits(u64::from(begin - base), bits)?;
    Ok(())
}

pub(super) fn write_ma_config(
    output: &mut BitWriter,
    entropy: &EntropyCode,
    predictor: LosslessModularPredictor,
) -> Result<(), EncodeError> {
    let gradient = predictor == LosslessModularPredictor::Gradient;
    output.write_bits(0, 1)?;
    output.write_bits(1, 1)?;
    output.write_bits(u64::from(!gradient), 2)?;
    if !gradient {
        // The predictor context has a single-symbol distribution. All other tree
        // fields keep the original split=0, four-symbol prefix distribution.
        for context in [0, 0, 1, 0, 0, 0] {
            output.write_bits(context, 1)?;
        }
    }
    output.write_bits(1, 1)?;
    output.write_bits(0, 4)?;
    if !gradient {
        output.write_bits(15, 4)?; // literal predictor, without hybrid extra bits
    }
    output.write_bits(0b100011, 6)?;
    if !gradient {
        let symbol = predictor.value();
        output.write_bits(u64::from(symbol != 0), 1)?;
        if symbol != 0 {
            let exponent = symbol.ilog2();
            output.write_bits(u64::from(exponent), 4)?;
            output.write_bits(u64::from(symbol - (1 << exponent)), exponent as u8)?;
        }
    }
    output.write_bits(1, 2)?;
    output.write_bits(3, 2)?;
    for symbol in 0..4 {
        output.write_bits(symbol, 2)?;
    }
    output.write_bits(0, 1)?;
    if !gradient && predictor.value() != 0 {
        output.write_bits(1, 2)?; // simple prefix tree
        output.write_bits(0, 2)?; // one symbol
        output.write_bits(
            u64::from(predictor.value()),
            (predictor.value() + 1).next_power_of_two().ilog2() as u8,
        )?;
    }

    const TREE_INDICES: [usize; 26] = [
        1, 2, 1, 4, 1, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0,
    ];
    const SYMBOL_BITS: [u64; 6] = [0b00, 0b10, 0b001, 0b101, 0b0011, 0b0111];
    const SYMBOL_NBITS: [u8; 6] = [2, 2, 3, 3, 4, 4];
    for index in TREE_INDICES {
        if gradient || index != 5 {
            output.write_bits(SYMBOL_BITS[index], SYMBOL_NBITS[index])?;
        }
    }

    entropy.write_lz77_config(output)?;
    entropy.write_context_map(output)?;
    let EntropyCode::Prefix { codes, distance } = entropy else {
        return entropy
            .ans()
            .expect("ANS entropy variant")
            .write_histograms(output);
    };
    let distance_code = distance.as_ref();
    output.write_bits(1, 1)?;
    output.write_bits(0, 4)?;
    for _ in 0..4 {
        output.write_bits(0, 4)?;
    }
    if distance_code.is_some() {
        output.write_bits(1, 1)?;
        output.write_bits(5, 4)?;
        output.write_bits(0, 5)?; // 1 + 2^5 = 33 distance symbols
    } else {
        output.write_bits(1, 5)?; // two-symbol distance alphabet
    }
    for _ in 0..4 {
        output.write_bits(1, 1)?;
        output.write_bits(8, 4)?;
        // libjxl's U32 selector stores the low eight bits of 256 here.
        output.write_bits(0, 8)?;
    }
    if let Some(code) = distance_code {
        code.write_raw_tree(output)?;
    } else {
        output.write_bits(1, 2)?;
        output.write_bits(0, 2)?;
        output.write_bits(1, 1)?;
    }
    for code in codes.iter() {
        code.write_tree(output)?;
    }
    Ok(())
}

pub(super) fn image_header(
    width: u32,
    height: u32,
    format: LosslessModularFormat,
    bits_per_sample: u8,
    exponent_bits_per_sample: u8,
    animation: AnimationHeader,
    color: ModularImageMetadata,
) -> Result<PreparedImageHeader, EncodeError> {
    image_header_with_plan(
        &crate::image_sequence::ImageHeaderPlan::new(width, height, animation)?,
        format,
        bits_per_sample,
        exponent_bits_per_sample,
        color,
    )
}

fn image_header_with_plan(
    header: &crate::image_sequence::ImageHeaderPlan,
    format: LosslessModularFormat,
    bits_per_sample: u8,
    exponent_bits_per_sample: u8,
    color: ModularImageMetadata,
) -> Result<PreparedImageHeader, EncodeError> {
    color.alpha.validate(format)?;
    let samples = if exponent_bits_per_sample == 0 {
        crate::ColorSampleFormat::integer(format.color_channels(), bits_per_sample)?
    } else {
        crate::ColorSampleFormat::float(
            format.color_channels(),
            bits_per_sample,
            exponent_bits_per_sample,
        )?
    };
    let samples = crate::sample_format::ImageSamplePlan::new(
        samples,
        format.has_alpha().then_some(color.alpha),
    )
    .with_cmyk(color.encoding.is_cmyk())?
    .with_extra_channels(
        &color.extra_channels,
        color.max_extra_channel_metadata_bytes,
    )?;
    header.encode(
        crate::image_sequence::ImageCoding::Modular,
        &samples,
        &color.encoding,
        color.options,
        color.max_icc_profile_bytes,
    )
}

pub(super) fn frame_header(
    frame: &FrameHeaderPlan,
    group_size: super::types::LosslessModularGroupSize,
) -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0, 1)?; // non-default frame header
    frame.write_kind(&mut output)?;
    output.write_bits(1, 1)?; // Modular encoding
    output.write_bits(0, 2)?; // zero frame flags
    output.write_bits(0, 1)?; // no YCbCr transform
    frame.sampling().write(&mut output)?;
    output.write_bits(u64::from(group_size.size_shift()), 2)?;
    if frame.has_passes() {
        output.write_bits(0, 2)?; // one pass when present
    }

    frame.append_to(&mut output)?;
    let bit_len = output.bit_len();
    BitFragment::new(output.into_bytes(), bit_len).map_err(Into::into)
}

#[cfg(test)]
mod rct_wire_tests {
    use super::*;
    use crate::LosslessModularRctType;
    use jxl_bitstream::{Bitstream, U};

    #[test]
    fn independent_wire_reader_recovers_every_rct_type_and_no_transform() {
        for value in (0..42).map(Some).chain([None]) {
            let mut writer = BitWriter::new();
            let operations: Vec<_> = value
                .map(|value| {
                    TransformOperation::Rct(PlannedRct::source(
                        LosslessModularRctType::new(value).unwrap(),
                    ))
                })
                .into_iter()
                .collect();
            write_transforms(
                &mut writer,
                TransformHeader {
                    operations: &operations,
                    palette_counts: None,
                },
            )
            .unwrap();
            let bytes = writer.into_bytes();
            let mut reader = Bitstream::new(&bytes);
            assert_eq!(
                reader.read_u32(0, 1, 2 + U(4), 18 + U(8)).unwrap(),
                u32::from(value.is_some())
            );
            if let Some(value) = value {
                assert_eq!(reader.read_bits(2).unwrap(), 0);
                assert_eq!(
                    reader
                        .read_u32(U(3), 8 + U(6), 72 + U(10), 1096 + U(13))
                        .unwrap(),
                    0
                );
                assert_eq!(
                    reader.read_u32(6, U(2), 2 + U(4), 10 + U(6)).unwrap(),
                    value
                );
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod predictor_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod lz77_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod palette_tests;
