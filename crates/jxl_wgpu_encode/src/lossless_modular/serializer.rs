use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use jxl_gpu_bitstream::{
    ACCELERATION_INDEX_BOX_TYPE, BitWriter, ContainerBox, Gray8AccelerationIndex,
    write_container_with_boxes,
};

use super::dispatch::frame_covers_canvas;
use super::dispatch::{LosslessModularBackend, ModularGroupPlan};
use super::grid::{LosslessModularGroup, LosslessModularGroupGrid};
use super::memory::{LosslessModularMemoryLimits, LosslessModularMemoryPlan};
use super::streaming::LosslessModularJob;
use super::types::{
    LosslessModularFormat, LosslessModularTreeMode, ModularArtifactHeader, ModularEvent,
    lossless_modular_source_spec,
};
use crate::prefix::{LZ77_SYMBOLS, PrefixCode, RAW_SYMBOLS};
use crate::{
    AnimationHeader, BackendError, BitFragment, BlendMode, CodestreamAssembler, Determinism,
    EncodeError, EncodeProfile, EncodeSession, EncoderBufferPoolStats, EncoderCapabilities,
    FrameBlend, FrameEncodeRequest, FrameGroupLayout, FrameIndex, FrameOptions, FramePacketSet,
    FrameSubmission, GpuAccelerationArtifact, GpuEncoder, GpuFrameArtifacts, GpuFrameSource,
    GroupPacket, GroupPacketKind, ProgressivePlan, SessionDescriptor, WgpuContext, assemble_frame,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ModularFrameHeader {
    pub(super) animation: AnimationHeader,
    pub(super) canvas_width: u32,
    pub(super) canvas_height: u32,
    pub(super) options: FrameOptions,
    pub(super) is_last: bool,
}
/// Convenience API that produces a complete raw codestream or deterministic
/// `jxlc` container from a GPU-resident packed Gray, RGB, or RGBA integer buffer.
pub struct LosslessModularEncoder {
    encoder: GpuEncoder<LosslessModularBackend>,
}

impl LosslessModularEncoder {
    #[must_use]
    pub fn new(context: WgpuContext) -> Self {
        let backend = LosslessModularBackend::new(&context);
        Self {
            encoder: GpuEncoder::new(context, backend),
        }
    }

    /// Creates an encoder with an explicit multi-group MA-tree placement policy.
    #[must_use]
    pub fn with_tree_mode(context: WgpuContext, tree_mode: LosslessModularTreeMode) -> Self {
        let backend = LosslessModularBackend::with_tree_mode(&context, tree_mode);
        Self {
            encoder: GpuEncoder::new(context, backend),
        }
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
        }
    }

    #[must_use]
    pub fn capabilities(&self) -> &EncoderCapabilities {
        self.encoder.capabilities()
    }

    /// Reports aggregate owned bytes retained by currently live encode jobs.
    #[must_use]
    pub fn in_flight_memory_stats(&self) -> jxl_wgpu::MemoryBudgetSnapshot {
        self.encoder.memory_stats()
    }

    /// Computes all source, artifact, and readback bytes before submission.
    pub fn memory_plan(
        &self,
        source: &crate::BufferImageSource,
    ) -> Result<LosslessModularMemoryPlan, EncodeError> {
        self.encoder.backend().memory_plan(source)
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
        source: crate::BufferImageSource,
    ) -> Result<LosslessModularSubmission, EncodeError> {
        self.memory_plan(&source)?;
        self.submit_inner(source, false)
    }

    pub fn submit_container(
        &self,
        source: crate::BufferImageSource,
    ) -> Result<LosslessModularSubmission, EncodeError> {
        self.memory_plan(&source)?;
        self.submit_inner(source, true)
    }

    pub fn encode(&self, source: crate::BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit(source)?.wait()
    }

    pub fn encode_container(
        &self,
        source: crate::BufferImageSource,
    ) -> Result<Vec<u8>, EncodeError> {
        self.submit_container(source)?.wait()
    }

    /// Starts a reusable multi-frame animation session.
    ///
    /// Every returned frame submission supports both [`Future`] and blocking
    /// [`FrameSubmission::wait`]. Submissions do not borrow this session, so multiple GPU frames
    /// can remain in flight and their completed artifacts may be inserted in any order.
    pub fn begin_animation(
        &self,
        descriptor: LosslessModularAnimationDescriptor,
    ) -> Result<LosslessModularAnimationSession, EncodeError> {
        let codestream_header = image_header(
            descriptor.canvas_width,
            descriptor.canvas_height,
            descriptor.format,
            descriptor.bits_per_sample,
            descriptor.animation,
        )?;
        let session = self.encoder.begin_session(SessionDescriptor {
            profile: EncodeProfile::ModularLossless {
                bits_per_sample: descriptor.bits_per_sample,
            },
            progressive: ProgressivePlan::single(),
            minimum_determinism: Determinism::CrossDevice,
            animation: descriptor.animation,
            canvas_width: descriptor.canvas_width,
            canvas_height: descriptor.canvas_height,
        })?;
        Ok(LosslessModularAnimationSession {
            session,
            assembler: CodestreamAssembler::new(codestream_header)?,
            descriptor,
        })
    }

    fn submit_inner(
        &self,
        source: crate::BufferImageSource,
        container: bool,
    ) -> Result<LosslessModularSubmission, EncodeError> {
        // Preserve typed address/device-limit failures before the generic
        // backend admission predicate maps unsupported inputs to InputFormat.
        self.encoder.backend().memory_plan(&source)?;
        let width = source.layout.extent.width;
        let height = source.layout.extent.height;
        let source_spec = lossless_modular_source_spec(&source.layout.format)?;
        let format = source_spec.format;
        let group_grid = LosslessModularGroupGrid::for_extent(width, height)?;
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::ModularLossless {
                bits_per_sample: source_spec.bits_per_sample,
            },
            progressive: ProgressivePlan::single(),
            minimum_determinism: Determinism::CrossDevice,
            animation: AnimationHeader::Still,
            canvas_width: width,
            canvas_height: height,
            options: FrameOptions::default(),
        };
        let frame = self
            .encoder
            .submit_frame(GpuFrameSource::Buffer(source), request)?;
        Ok(LosslessModularSubmission {
            frame: Some(frame),
            codestream_header: image_header(
                width,
                height,
                format,
                source_spec.bits_per_sample,
                AnimationHeader::Still,
            )?,
            container,
            group_grid,
            format,
            bits_per_sample: source_spec.bits_per_sample,
        })
    }
}

/// Stream-wide contract for one lossless Modular animation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularAnimationDescriptor {
    canvas_width: u32,
    canvas_height: u32,
    format: LosslessModularFormat,
    bits_per_sample: u8,
    animation: AnimationHeader,
}

impl LosslessModularAnimationDescriptor {
    pub fn new(
        canvas_width: u32,
        canvas_height: u32,
        format: LosslessModularFormat,
        bits_per_sample: u8,
        animation: AnimationHeader,
    ) -> Result<Self, EncodeError> {
        if !animation.is_animation() {
            return Err(EncodeError::InvalidConfiguration(
                "a Modular animation descriptor requires an animation timebase",
            ));
        }
        format.pixel_format(bits_per_sample)?;
        image_header(
            canvas_width,
            canvas_height,
            format,
            bits_per_sample,
            animation,
        )?;
        Ok(Self {
            canvas_width,
            canvas_height,
            format,
            bits_per_sample,
            animation,
        })
    }

    #[must_use]
    pub const fn canvas_width(self) -> u32 {
        self.canvas_width
    }

    #[must_use]
    pub const fn canvas_height(self) -> u32 {
        self.canvas_height
    }

    #[must_use]
    pub const fn format(self) -> LosslessModularFormat {
        self.format
    }

    #[must_use]
    pub const fn bits_per_sample(self) -> u8 {
        self.bits_per_sample
    }

    #[must_use]
    pub const fn animation(self) -> AnimationHeader {
        self.animation
    }
}

/// Multi-frame assembly state for a standard lossless Modular animation.
pub struct LosslessModularAnimationSession {
    session: EncodeSession<LosslessModularBackend>,
    assembler: CodestreamAssembler,
    descriptor: LosslessModularAnimationDescriptor,
}

impl LosslessModularAnimationSession {
    #[must_use]
    pub const fn descriptor(&self) -> LosslessModularAnimationDescriptor {
        self.descriptor
    }

    #[must_use]
    pub const fn next_frame_index(&self) -> FrameIndex {
        self.session.next_frame_index()
    }

    pub fn submit_frame(
        &mut self,
        source: crate::BufferImageSource,
        options: FrameOptions,
    ) -> Result<FrameSubmission<LosslessModularJob>, EncodeError> {
        self.validate_source(&source)?;
        self.session
            .submit_frame(GpuFrameSource::Buffer(source), options)
    }

    pub fn submit_last_frame(
        &mut self,
        source: crate::BufferImageSource,
        options: FrameOptions,
    ) -> Result<FrameSubmission<LosslessModularJob>, EncodeError> {
        self.validate_source(&source)?;
        self.session
            .submit_last_frame(GpuFrameSource::Buffer(source), options)
    }

    /// Inserts one completed GPU frame. Completion order need not match frame order.
    pub fn insert(&mut self, frame: GpuFrameArtifacts) -> Result<(), EncodeError> {
        self.assembler.insert(frame)?;
        Ok(())
    }

    pub fn finish_raw(self) -> Result<Vec<u8>, EncodeError> {
        self.session.ensure_closed()?;
        Ok(self.assembler.finish_raw()?)
    }

    pub fn finish_container(self) -> Result<Vec<u8>, EncodeError> {
        self.session.ensure_closed()?;
        self.assembler.finish_container()
    }

    fn validate_source(&self, source: &crate::BufferImageSource) -> Result<(), EncodeError> {
        let spec = lossless_modular_source_spec(&source.layout.format)?;
        if spec.format != self.descriptor.format
            || spec.bits_per_sample != self.descriptor.bits_per_sample
        {
            return Err(EncodeError::InvalidConfiguration(
                "every animation frame must match the stream format and integer depth",
            ));
        }
        Ok(())
    }
}

/// A `Future` with an executor-independent blocking counterpart.
pub struct LosslessModularSubmission {
    frame: Option<FrameSubmission<LosslessModularJob>>,
    codestream_header: BitFragment,
    container: bool,
    group_grid: LosslessModularGroupGrid,
    format: LosslessModularFormat,
    bits_per_sample: u8,
}

impl LosslessModularSubmission {
    #[must_use]
    pub const fn format(&self) -> LosslessModularFormat {
        self.format
    }

    /// Valid low bits encoded for every integer component.
    #[must_use]
    pub const fn bits_per_sample(&self) -> u8 {
        self.bits_per_sample
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

    fn assemble(&self, frame: GpuFrameArtifacts) -> Result<Vec<u8>, EncodeError> {
        let acceleration = frame.acceleration;
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
        let mut codestream = self.codestream_header.bytes().to_vec();
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
        let group_start = self
            .codestream_header
            .bytes()
            .len()
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
                Poll::Ready(result.and_then(|frame| submission.assemble(frame)))
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
    pub(super) tree_mode: LosslessModularTreeMode,
    pub(super) frame: &'a ModularFrameHeader,
    pub(super) group_plans: &'a [ModularGroupPlan],
    pub(super) bytes: &'a [u8],
}

type RawHistograms = [[u64; RAW_SYMBOLS]; 4];
type Lz77Histograms = [[u64; LZ77_SYMBOLS]; 4];

pub(super) fn accumulate_artifact_histograms(
    channel: usize,
    artifact: &ValidatedModularArtifact<'_>,
    aggregate_raw: &mut RawHistograms,
    aggregate_lz77: &mut Lz77Histograms,
) -> Result<(), EncodeError> {
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
    Ok(())
}

pub(super) fn build_prefix_codes(
    format: LosslessModularFormat,
    bits_per_sample: u8,
    aggregate_raw: &RawHistograms,
    aggregate_lz77: &Lz77Histograms,
) -> Result<[PrefixCode; 4], EncodeError> {
    let channels = usize::try_from(format.channel_count())
        .map_err(|_| EncodeError::Backend("Modular channel count overflow".into()))?;
    let unused = PrefixCode::fixed_unused_channel();
    let mut codes = [unused.clone(), unused.clone(), unused.clone(), unused];
    for channel in 0..channels {
        let transformed_extra_token = u8::from(format != LosslessModularFormat::Gray);
        let wide_samples = bits_per_sample > 14;
        let max_raw_token = if wide_samples {
            RAW_SYMBOLS - 1
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
    tree_mode: LosslessModularTreeMode,
    frame: ModularFrameHeader,
    codes: [PrefixCode; 4],
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
    pub(super) tree_mode: LosslessModularTreeMode,
    pub(super) frame: ModularFrameHeader,
}

impl ModularPacketAssembler {
    pub(super) fn new(
        config: ModularPacketConfig,
        codes: [PrefixCode; 4],
    ) -> Result<Self, EncodeError> {
        let ModularPacketConfig {
            width,
            height,
            group_grid,
            format,
            bits_per_sample,
            tree_mode,
            frame,
        } = config;
        let (packets, single_group, token_bit_offset_in_group) = if group_grid.groups == 1 {
            let mut group = BitWriter::new();
            write_dc_global(&mut group, &codes, format)?;
            let token_bit_offset = u64::try_from(group.bit_len())
                .map_err(|_| EncodeError::Backend("gray8 token offset overflow".into()))?;
            (Vec::new(), Some(group), token_bit_offset)
        } else {
            let layout = FrameGroupLayout::new(group_grid.lf_groups, group_grid.groups, 1)?;
            let mut packets = Vec::with_capacity(layout.toc_entries());
            let mut dc_global = BitWriter::new();
            write_dc_global(&mut dc_global, &codes, format)?;
            dc_global.align_to_byte()?;
            packets.push(GroupPacket::new(
                GroupPacketKind::DcGlobal,
                dc_global.into_bytes(),
            ));
            for group in 0..group_grid.lf_groups {
                packets.push(GroupPacket::new(
                    GroupPacketKind::DcGroup(group),
                    Vec::new(),
                ));
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
            tree_mode,
            frame,
            codes,
            packets,
            single_group,
            token_bit_offset_in_group,
            next_group: 0,
        })
    }

    pub(super) fn push_group(
        &mut self,
        group_index: u32,
        artifacts: &[ValidatedModularArtifact<'_>],
    ) -> Result<(), EncodeError> {
        if group_index != self.next_group {
            return Err(EncodeError::Backend(
                "GPU artifact groups are not in canonical order".into(),
            ));
        }
        let channels = usize::try_from(self.format.channel_count())
            .map_err(|_| EncodeError::Backend("Modular channel count overflow".into()))?;
        if artifacts.len() != channels {
            return Err(EncodeError::Backend(
                "GPU group does not contain every Modular channel".into(),
            ));
        }
        if let Some(group) = &mut self.single_group {
            for (channel, artifact) in artifacts.iter().enumerate() {
                write_events(group, &self.codes[channel], artifact.events)?;
            }
        } else {
            let mut pass_group = BitWriter::new();
            let use_global_tree = self.tree_mode == LosslessModularTreeMode::SharedGlobal;
            // GroupHeader: selected tree, default weighted predictor, no local transforms.
            pass_group.write_bits(u64::from(use_global_tree), 1)?;
            pass_group.write_bits(1, 1)?;
            pass_group.write_bits(0, 2)?;
            if !use_global_tree {
                write_ma_config(&mut pass_group, &self.codes)?;
            }
            for (channel, artifact) in artifacts.iter().enumerate() {
                write_events(&mut pass_group, &self.codes[channel], artifact.events)?;
            }
            pass_group.align_to_byte()?;
            self.packets.push(GroupPacket::new(
                GroupPacketKind::AcGroup {
                    pass: 0,
                    group: group_index,
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
        if self.next_group != self.group_grid.groups {
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
                frame_header(self.format, &self.frame)?,
                FrameGroupLayout::new(1, 1, 1)?,
                [GroupPacket::new(
                    GroupPacketKind::Single,
                    group.into_bytes(),
                )],
            )?;
            let acceleration = (self.format == LosslessModularFormat::Gray
                && self.bits_per_sample == 8)
                .then(|| GpuAccelerationArtifact::Gray8Prefix {
                    width: self.width,
                    height: self.height,
                    token_bit_offset_in_group: self.token_bit_offset_in_group,
                    token_bit_len,
                    raw_prefix: self.codes[0].raw_entries(),
                    lz77_prefix: self.codes[0].lz77_entries(),
                });
            return Ok((packets, acceleration));
        }
        let layout = FrameGroupLayout::new(self.group_grid.lf_groups, self.group_grid.groups, 1)?;
        Ok((
            FramePacketSet::new(
                frame_header(self.format, &self.frame)?,
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
        tree_mode,
        frame,
        group_plans,
        bytes,
    } = input;
    let channels = usize::try_from(format.channel_count())
        .map_err(|_| EncodeError::Backend("Modular channel count overflow".into()))?;
    let expected_artifacts = usize::try_from(group_grid.groups)
        .ok()
        .and_then(|groups| groups.checked_mul(channels))
        .ok_or_else(|| EncodeError::Backend("GPU group plan count overflow".into()))?;
    if group_plans.len() != expected_artifacts {
        return Err(EncodeError::Backend(
            "GPU group plan does not match the frame grid".into(),
        ));
    }
    let mut artifacts = Vec::with_capacity(group_plans.len());
    let mut aggregate_raw = [[0u64; RAW_SYMBOLS]; 4];
    let mut aggregate_lz77 = [[0u64; LZ77_SYMBOLS]; 4];
    for (artifact_index, plan) in group_plans.iter().enumerate() {
        let channel = artifact_index % channels;
        if plan.channel != channel as u32 {
            return Err(EncodeError::Backend(
                "GPU group plan channel order is not canonical".into(),
            ));
        }
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
        let artifact =
            parse_group_artifact(plan.width, plan.height, plan.max_events, artifact_bytes)?;
        accumulate_artifact_histograms(
            channel,
            &artifact,
            &mut aggregate_raw,
            &mut aggregate_lz77,
        )?;
        artifacts.push(artifact);
    }

    let codes = build_prefix_codes(format, bits_per_sample, &aggregate_raw, &aggregate_lz77)?;
    let mut assembler = ModularPacketAssembler::new(
        ModularPacketConfig {
            width,
            height,
            group_grid,
            format,
            bits_per_sample,
            tree_mode,
            frame: frame.clone(),
        },
        codes,
    )?;
    for group in 0..group_grid.groups {
        let start = usize::try_from(group)
            .ok()
            .and_then(|group| group.checked_mul(channels))
            .ok_or_else(|| EncodeError::Backend("Modular group index overflow".into()))?;
        let end = start
            .checked_add(channels)
            .ok_or_else(|| EncodeError::Backend("Modular group index overflow".into()))?;
        assembler.push_group(group, &artifacts[start..end])?;
    }
    assembler.finish()
}

#[derive(Clone, Copy)]
pub(super) struct ValidatedModularArtifact<'a> {
    pub(super) header: ModularArtifactHeader,
    pub(super) events: &'a [ModularEvent],
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
    Ok(ValidatedModularArtifact { header, events })
}

pub(super) fn write_events(
    output: &mut BitWriter,
    code: &PrefixCode,
    events: &[ModularEvent],
) -> Result<(), EncodeError> {
    for event in events {
        match event.kind {
            0 => code.write_raw(output, event.token, event.extra_bit_count, event.extra_bits)?,
            1 => code.write_run(output, event.token, event.extra_bit_count, event.extra_bits)?,
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
    let mut sample_count = 0u64;

    for event in events {
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
            1 => {
                let token = usize::try_from(event.token)
                    .map_err(|_| invalid_gpu_artifact("LZ77 token overflow"))?;
                if token > 27 {
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
                raw_counts[0] = raw_counts[0]
                    .checked_add(1)
                    .ok_or_else(|| invalid_gpu_artifact("raw histogram overflow"))?;
                lz77_counts[token] = lz77_counts[token]
                    .checked_add(1)
                    .ok_or_else(|| invalid_gpu_artifact("LZ77 histogram overflow"))?;
                let encoded_value = if event.token < 16 {
                    u64::from(event.token)
                } else {
                    (1u64 << event.extra_bit_count) + u64::from(event.extra_bits)
                };
                sample_count = sample_count
                    .checked_add(encoded_value + 8)
                    .ok_or_else(|| invalid_gpu_artifact("sample count overflow"))?;
            }
            _ => return Err(invalid_gpu_artifact("unknown token kind")),
        }
    }

    if raw_counts != header.raw_counts || lz77_counts != header.lz77_counts {
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

fn write_dc_global(
    output: &mut BitWriter,
    codes: &[PrefixCode; 4],
    format: LosslessModularFormat,
) -> Result<(), EncodeError> {
    // Handcrafted Modular metadata adapted from zune-jpegxl 0.5.2. See this crate's
    // `THIRD_PARTY.md` and `LICENSES/zune-jpegxl-MIT.txt`.
    output.write_bits(1, 1)?; // default LF-channel dequantization
    output.write_bits(1, 1)?; // GlobalModular is present
    write_ma_config(output, codes)?;
    output.write_bits(1, 1)?;
    output.write_bits(1, 1)?;
    if format.channel_count() > 2 {
        output.write_bits(1, 2)?; // one transform
        output.write_bits(0, 2)?; // reversible color transform
        output.write_bits(0, 5)?; // begin channel 0
        output.write_bits(0, 2)?; // YCoCg transform type 0
    } else {
        output.write_bits(0, 2)?; // no transforms
    }
    Ok(())
}

fn write_ma_config(output: &mut BitWriter, codes: &[PrefixCode; 4]) -> Result<(), EncodeError> {
    output.write_bits(0, 1)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 4)?;
    output.write_bits(0b100011, 6)?;
    output.write_bits(1, 2)?;
    output.write_bits(3, 2)?;
    for symbol in 0..4 {
        output.write_bits(symbol, 2)?;
    }
    output.write_bits(0, 1)?;

    const TREE_INDICES: [usize; 26] = [
        1, 2, 1, 4, 1, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0,
    ];
    const SYMBOL_BITS: [u64; 6] = [0b00, 0b10, 0b001, 0b101, 0b0011, 0b0111];
    const SYMBOL_NBITS: [u8; 6] = [2, 2, 3, 3, 4, 4];
    for index in TREE_INDICES {
        output.write_bits(SYMBOL_BITS[index], SYMBOL_NBITS[index])?;
    }

    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0b1010, 4)?;
    output.write_bits(4, 4)?;
    output.write_bits(0, 3)?;
    output.write_bits(0, 3)?;
    output.write_bits(1, 1)?;
    output.write_bits(3, 2)?;
    for context in [4, 3, 2, 1, 0] {
        output.write_bits(context, 3)?;
    }
    output.write_bits(1, 1)?;
    output.write_bits(0, 4)?;
    for _ in 0..4 {
        output.write_bits(0, 4)?;
    }
    output.write_bits(1, 5)?;
    for _ in 0..4 {
        output.write_bits(1, 1)?;
        output.write_bits(8, 4)?;
        // libjxl's U32 selector stores the low eight bits of 256 here.
        output.write_bits(0, 8)?;
    }
    output.write_bits(1, 2)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    for code in codes {
        code.write_tree(output)?;
    }
    Ok(())
}

pub(super) fn image_header(
    width: u32,
    height: u32,
    format: LosslessModularFormat,
    bits_per_sample: u8,
    animation: AnimationHeader,
) -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0x0aff, 16)?;
    output.write_bits(0, 1)?;
    write_size(&mut output, height, true)?;
    write_size(&mut output, width, false)?;
    output.write_bits(0, 1)?;
    output.write_bits(u64::from(animation.is_animation()), 1)?;
    if animation.is_animation() {
        output.write_bits(0, 3)?; // identity orientation minus one
        output.write_bits(0, 1)?; // no intrinsic size
        output.write_bits(0, 1)?; // no preview
        output.write_bits(1, 1)?; // animation metadata follows
        write_animation_header(&mut output, animation)?;
    }
    write_integer_bit_depth(&mut output, bits_per_sample)?;
    output.write_bits(u64::from(bits_per_sample <= 14), 1)?;
    if format.has_alpha() {
        output.write_bits(1, 2)?; // one alpha extra channel
        if bits_per_sample == 8 {
            output.write_bits(1, 1)?; // default 8-bit, unassociated alpha metadata
        } else {
            output.write_bits(0, 1)?; // explicit alpha metadata
            output.write_bits(0, 2)?; // alpha extra-channel type
            write_integer_bit_depth(&mut output, bits_per_sample)?;
            output.write_bits(0, 2)?; // full-resolution dim_shift
            output.write_bits(0, 2)?; // empty name
            output.write_bits(0, 1)?; // unassociated alpha
        }
    } else {
        output.write_bits(0, 2)?;
    }
    output.write_bits(0, 1)?;
    if format.channel_count() > 2 {
        output.write_bits(1, 1)?; // default sRGB color encoding
    } else {
        output.write_bits(0, 1)?;
        output.write_bits(0, 1)?;
        output.write_bits(1, 2)?;
        output.write_bits(1, 2)?;
        output.write_bits(0, 1)?;
        output.write_bits(0b10, 2)?;
        output.write_bits(11, 4)?;
        output.write_bits(1, 2)?;
    }
    if animation.is_animation() {
        output.write_bits(1, 1)?; // all-default SDR tone mapping
    }
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.align_to_byte()?;
    Ok(BitFragment::byte_aligned(output.into_bytes())?)
}

pub(super) fn write_animation_header(
    output: &mut BitWriter,
    animation: AnimationHeader,
) -> Result<(), EncodeError> {
    let AnimationHeader::Animation {
        ticks_per_second_numerator,
        ticks_per_second_denominator,
        num_loops,
        have_timecodes,
    } = animation
    else {
        return Err(EncodeError::InvalidConfiguration(
            "animation metadata requires an animation header",
        ));
    };
    let numerator = ticks_per_second_numerator.get();
    match numerator {
        100 => output.write_bits(0, 2)?,
        1000 => output.write_bits(1, 2)?,
        1..=1024 => {
            output.write_bits(2, 2)?;
            output.write_bits(u64::from(numerator - 1), 10)?;
        }
        1025..=1_073_741_824 => {
            output.write_bits(3, 2)?;
            output.write_bits(u64::from(numerator - 1), 30)?;
        }
        _ => {
            return Err(EncodeError::InvalidConfiguration(
                "animation ticks-per-second numerator exceeds the JPEG XL limit",
            ));
        }
    }
    let denominator = ticks_per_second_denominator.get();
    match denominator {
        1 => output.write_bits(0, 2)?,
        1001 => output.write_bits(1, 2)?,
        2..=256 => {
            output.write_bits(2, 2)?;
            output.write_bits(u64::from(denominator - 1), 8)?;
        }
        257..=1024 => {
            output.write_bits(3, 2)?;
            output.write_bits(u64::from(denominator - 1), 10)?;
        }
        _ => {
            return Err(EncodeError::InvalidConfiguration(
                "animation ticks-per-second denominator exceeds the JPEG XL limit",
            ));
        }
    }
    match num_loops {
        0 => output.write_bits(0, 2)?,
        1..=7 => {
            output.write_bits(1, 2)?;
            output.write_bits(u64::from(num_loops), 3)?;
        }
        8..=65_535 => {
            output.write_bits(2, 2)?;
            output.write_bits(u64::from(num_loops), 16)?;
        }
        _ => {
            output.write_bits(3, 2)?;
            output.write_bits(u64::from(num_loops), 32)?;
        }
    }
    output.write_bits(u64::from(have_timecodes), 1)?;
    Ok(())
}

fn write_integer_bit_depth(output: &mut BitWriter, bits_per_sample: u8) -> Result<(), EncodeError> {
    if !(1..=16).contains(&bits_per_sample) {
        return Err(EncodeError::InvalidConfiguration(
            "lossless Modular integer depth must be in 1..=16",
        ));
    }
    output.write_bits(0, 1)?; // integer samples
    match bits_per_sample {
        8 => output.write_bits(0, 2)?,
        10 => output.write_bits(1, 2)?,
        12 => output.write_bits(2, 2)?,
        bits => {
            output.write_bits(3, 2)?;
            output.write_bits(u64::from(bits - 1), 6)?;
        }
    }
    Ok(())
}

fn write_size(output: &mut BitWriter, size: u32, ratio: bool) -> Result<(), EncodeError> {
    if !(1..(1 << 30)).contains(&size) {
        return Err(EncodeError::InvalidConfiguration(
            "Modular dimensions must be in 1..2^30",
        ));
    }
    let value = size - 1;
    let (selector, bits) = if value < 1 << 9 {
        (0, 9)
    } else if value < 1 << 13 {
        (1, 13)
    } else if value < 1 << 18 {
        (2, 18)
    } else {
        (3, 30)
    };
    output.write_bits(selector, 2)?;
    output.write_bits(u64::from(value), bits)?;
    if ratio {
        output.write_bits(0, 3)?;
    }
    Ok(())
}

pub(super) fn frame_header(
    format: LosslessModularFormat,
    frame: &ModularFrameHeader,
) -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0, 1)?; // non-default frame header
    output.write_bits(0, 2)?; // regular frame
    output.write_bits(1, 1)?; // Modular encoding
    output.write_bits(0, 2)?; // zero frame flags
    output.write_bits(0, 1)?; // no YCbCr transform
    output.write_bits(0, 2)?; // color upsampling factor one
    if format.has_alpha() {
        output.write_bits(0, 2)?; // alpha upsampling factor one
    }
    output.write_bits(1, 2)?; // 256x256 Modular groups
    output.write_bits(0, 2)?; // one pass

    let have_crop = frame.options.crop.is_some();
    output.write_bits(u64::from(have_crop), 1)?;
    if let Some(crop) = frame.options.crop {
        write_frame_dimension(&mut output, pack_signed(crop.x()))?;
        write_frame_dimension(&mut output, pack_signed(crop.y()))?;
        write_frame_dimension(&mut output, crop.width())?;
        write_frame_dimension(&mut output, crop.height())?;
    }

    let full_frame =
        frame_covers_canvas(frame.options.crop, frame.canvas_width, frame.canvas_height);
    write_blending_info(
        &mut output,
        frame.options.color_blend,
        format.has_alpha(),
        frame.options.color_blend.mode == BlendMode::Replace && full_frame,
    )?;
    if format.has_alpha() {
        let alpha_blend = frame
            .options
            .extra_channel_blends
            .first()
            .copied()
            .unwrap_or_default();
        write_blending_info(
            &mut output,
            alpha_blend,
            true,
            frame.options.color_blend.mode == BlendMode::Replace && full_frame,
        )?;
    }

    if let AnimationHeader::Animation { have_timecodes, .. } = frame.animation {
        write_frame_duration(&mut output, frame.options.timing.duration_ticks)?;
        if have_timecodes {
            output.write_bits(
                u64::from(frame.options.timing.timecode.ok_or(
                    EncodeError::InvalidConfiguration(
                        "animated frame is missing its declared timecode",
                    ),
                )?),
                32,
            )?;
        }
    }
    output.write_bits(u64::from(frame.is_last), 1)?;
    if !frame.is_last {
        output.write_bits(u64::from(frame.options.save_as_reference.get()), 2)?;
        let can_be_referenced =
            frame.options.timing.duration_ticks == 0 || frame.options.save_as_reference.get() != 0;
        if frame.options.color_blend.mode == BlendMode::Replace && full_frame && can_be_referenced {
            output.write_bits(u64::from(frame.options.save_before_color_transform), 1)?;
        }
    }

    output.write_bits(0, 2)?; // empty frame name
    output.write_bits(0, 1)?; // non-default restoration filter
    output.write_bits(0, 1)?; // no Gaborish
    output.write_bits(0, 2)?; // no EPF iterations
    output.write_bits(0, 2)?; // no restoration-filter extensions
    output.write_bits(0, 2)?; // no frame extensions
    let bit_len = output.bit_len();
    BitFragment::new(output.into_bytes(), bit_len).map_err(Into::into)
}

fn write_blending_info(
    output: &mut BitWriter,
    blend: FrameBlend,
    has_alpha: bool,
    resets_canvas: bool,
) -> Result<(), EncodeError> {
    write_blend_mode(output, blend.mode)?;
    let uses_alpha = matches!(blend.mode, BlendMode::Blend | BlendMode::MultiplyAdd);
    if has_alpha && uses_alpha {
        output.write_bits(0, 2)?; // alpha extra-channel index zero
    }
    if (has_alpha && uses_alpha) || blend.mode == BlendMode::Multiply {
        output.write_bits(u64::from(blend.clamp), 1)?;
    } else if blend.clamp {
        return Err(EncodeError::InvalidConfiguration(
            "the selected JPEG XL blend mode has no clamp field",
        ));
    }
    if !resets_canvas {
        output.write_bits(u64::from(blend.source_reference.get()), 2)?;
    }
    Ok(())
}

fn write_blend_mode(output: &mut BitWriter, mode: BlendMode) -> Result<(), EncodeError> {
    match mode {
        BlendMode::Replace => output.write_bits(0, 2)?,
        BlendMode::Add => output.write_bits(1, 2)?,
        BlendMode::Blend => output.write_bits(2, 2)?,
        BlendMode::MultiplyAdd => {
            output.write_bits(3, 2)?;
            output.write_bits(0, 2)?;
        }
        BlendMode::Multiply => {
            output.write_bits(3, 2)?;
            output.write_bits(1, 2)?;
        }
    }
    Ok(())
}

pub(super) fn pack_signed(value: i32) -> u32 {
    if value >= 0 {
        (value as u32) << 1
    } else {
        (u32::try_from(-i64::from(value)).expect("an i32 magnitude fits u32") << 1).wrapping_sub(1)
    }
}

fn write_frame_dimension(output: &mut BitWriter, value: u32) -> Result<(), EncodeError> {
    let (selector, offset, bits) = if value < 256 {
        (0, 0, 8)
    } else if value < 2_304 {
        (1, 256, 11)
    } else if value < 18_688 {
        (2, 2_304, 14)
    } else if value < 18_688 + (1 << 30) {
        (3, 18_688, 30)
    } else {
        return Err(EncodeError::InvalidConfiguration(
            "animation frame crop coordinate exceeds the JPEG XL limit",
        ));
    };
    output.write_bits(selector, 2)?;
    output.write_bits(u64::from(value - offset), bits)?;
    Ok(())
}

fn write_frame_duration(output: &mut BitWriter, duration: u32) -> Result<(), EncodeError> {
    match duration {
        0 => output.write_bits(0, 2)?,
        1 => output.write_bits(1, 2)?,
        2..=255 => {
            output.write_bits(2, 2)?;
            output.write_bits(u64::from(duration), 8)?;
        }
        _ => {
            output.write_bits(3, 2)?;
            output.write_bits(u64::from(duration), 32)?;
        }
    }
    Ok(())
}
