// The JPEG XL header and fast-lossless control-plane construction in this module is derived
// from the permissively licensed zune-jpegxl 0.5.2 encoder. See `THIRD_PARTY.md` and
// `LICENSES/zune-jpegxl-MIT.txt` in this crate.

use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use jxl_gpu_bitstream::{
    ACCELERATION_INDEX_BOX_TYPE, BitWriter, ContainerBox, Gray8AccelerationIndex,
    write_container_with_boxes,
};
#[cfg(test)]
use jxl_gpu_formats::RgbChannelOrder;
use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, PackingField,
    PackingFieldKind, PackingWord, PixelFormat, PlaneFormat, PlaneSampling, SampleKind, Swizzle,
};
use jxl_wgpu::MemoryPermit;
#[cfg(test)]
use wgpu::util::DeviceExt;

use crate::buffer_pool::EncoderBufferPool;
use crate::prefix::{LZ77_SYMBOLS, PrefixCode, RAW_SYMBOLS};
use crate::{
    AnimationHeader, BitFragment, DEFAULT_ENCODER_BUFFER_POOL_BYTES, Determinism, EncodeError,
    EncodeProfile, EncoderBufferPoolStats, EncoderCapabilities, FrameEncodeRequest,
    FrameGroupLayout, FrameIndex, FrameOptions, FramePacketSet, FrameSubmission,
    GpuAccelerationArtifact, GpuEncodeBackend, GpuEncodeJob, GpuEncoder, GpuFrameArtifacts,
    GpuFrameSource, GroupPacket, GroupPacketKind, KernelStage, ProfileCapability, ProgressivePlan,
    UnsupportedFeature, WgpuContext, assemble_frame,
};

/// JPEG XL's default Modular pass-group edge length.
pub const LOSSLESS_MODULAR_GROUP_DIMENSION: u32 = 256;
const LOSSLESS_MODULAR_LF_GROUP_DIMENSION: u32 = LOSSLESS_MODULAR_GROUP_DIMENSION * 8;
const SHADER: &str = include_str!("lossless_modular.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ModularParams {
    width: u32,
    height: u32,
    row_stride: u32,
    byte_offset: u32,
    output_word_offset: u32,
    channel: u32,
    channels: u32,
    bytes_per_sample: u32,
    sample_mask: u32,
}

/// Fixed storage-buffer header written by `lossless_modular.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ModularArtifactHeader {
    event_count: u32,
    raw_counts: [u32; RAW_SYMBOLS],
    lz77_counts: [u32; LZ77_SYMBOLS],
}

/// Fixed storage-buffer event written after [`ModularArtifactHeader`].
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ModularEvent {
    kind: u32,
    token: u32,
    extra_bit_count: u32,
    extra_bits: u32,
}

const OUTPUT_HEADER_WORDS: usize = std::mem::size_of::<ModularArtifactHeader>() / 4;
const EVENT_WORDS: usize = std::mem::size_of::<ModularEvent>() / 4;

const _: () = {
    assert!(std::mem::size_of::<ModularParams>() == 36);
    assert!(std::mem::align_of::<ModularParams>() == 4);
    assert!(std::mem::size_of::<ModularArtifactHeader>() == 53 * 4);
    assert!(std::mem::align_of::<ModularArtifactHeader>() == 4);
    assert!(std::mem::size_of::<ModularEvent>() == 16);
    assert!(std::mem::align_of::<ModularEvent>() == 4);
};

/// Standard lossless Modular input profile selected from a pitch-linear source descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LosslessModularFormat {
    Gray,
    Rgb,
    Rgba,
}

impl LosslessModularFormat {
    #[must_use]
    pub const fn channel_count(self) -> u32 {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::Rgba)
    }

    /// Constructs the canonical pitch-linear source format for an unsigned integer depth.
    ///
    /// Depths `1..=8` use one native-endian `u8` word per component. Depths `9..=16` use one
    /// native-endian `u16` word per component. Sub-byte and sub-16-bit values occupy the low bits;
    /// the high padding bits are outside the valid sample and are ignored by the encoder.
    pub fn pixel_format(self, bits_per_sample: u8) -> Result<PixelFormat, EncodeError> {
        if !(1..=16).contains(&bits_per_sample) {
            return Err(EncodeError::InvalidConfiguration(
                "lossless Modular integer depth must be in 1..=16",
            ));
        }
        let storage_bits = if bits_per_sample <= 8 { 8 } else { 16 };
        let (model, color_spec, swizzle, channels): (_, _, _, &[Channel]) = match self {
            Self::Gray => (
                ColorModel::NonColor,
                ColorSpecification::Undefined,
                Swizzle::X000,
                &[Channel::X],
            ),
            Self::Rgb => (
                ColorModel::Rgb,
                ColorSpecification::Default,
                Swizzle::XYZ1,
                &[Channel::X, Channel::Y, Channel::Z],
            ),
            Self::Rgba => (
                ColorModel::Rgb,
                ColorSpecification::Default,
                Swizzle::XYZW,
                &[Channel::X, Channel::Y, Channel::Z, Channel::W],
            ),
        };
        let words = channels
            .iter()
            .copied()
            .map(|channel| {
                let mut fields = Vec::with_capacity(2);
                if bits_per_sample < storage_bits {
                    fields.push(PackingField::padding(storage_bits - bits_per_sample));
                }
                fields.push(PackingField::channel(channel, bits_per_sample));
                PackingWord { fields }
            })
            .collect();
        Ok(PixelFormat {
            model,
            color_spec,
            chroma_subsampling: ChromaSubsampling::None,
            sample_kind: SampleKind::Unsigned,
            byte_order: ByteOrder::Native,
            swizzle,
            planes: vec![PlaneFormat {
                sampling: PlaneSampling::FULL,
                pixels_per_element: 1,
                words,
            }],
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LosslessModularSourceSpec {
    format: LosslessModularFormat,
    bits_per_sample: u8,
    bytes_per_sample: u8,
}

fn lossless_modular_source_spec(
    format: &PixelFormat,
) -> Result<LosslessModularSourceSpec, EncodeError> {
    if format.sample_kind != SampleKind::Unsigned
        || format.byte_order != ByteOrder::Native
        || format.chroma_subsampling != ChromaSubsampling::None
        || format.planes.len() != 1
    {
        return Err(UnsupportedFeature::InputFormat.into());
    }
    let logical_format = match (format.model, format.swizzle, format.color_spec) {
        (ColorModel::NonColor, Swizzle::X000, ColorSpecification::Undefined) => {
            LosslessModularFormat::Gray
        }
        (
            ColorModel::Rgb,
            Swizzle::XYZ1,
            ColorSpecification::Default | ColorSpecification::Undefined,
        ) => LosslessModularFormat::Rgb,
        (
            ColorModel::Rgb,
            Swizzle::XYZW,
            ColorSpecification::Default | ColorSpecification::Undefined,
        ) => LosslessModularFormat::Rgba,
        _ => return Err(UnsupportedFeature::InputFormat.into()),
    };
    let plane = &format.planes[0];
    if plane.sampling != PlaneSampling::FULL
        || plane.pixels_per_element != 1
        || plane.words.len() != logical_format.channel_count() as usize
    {
        return Err(UnsupportedFeature::InputFormat.into());
    }
    let expected_channels = [Channel::X, Channel::Y, Channel::Z, Channel::W];
    let mut bits_per_sample = None;
    let mut storage_bits = None;
    for (word, expected_channel) in plane
        .words
        .iter()
        .zip(&expected_channels[..plane.words.len()])
    {
        let (padding, channel_bits, channel) = match word.fields.as_slice() {
            [field] => match field.kind {
                PackingFieldKind::Channel(channel) => (0, field.bits, channel),
                PackingFieldKind::Padding => return Err(UnsupportedFeature::InputFormat.into()),
            },
            [padding, sample] => match (padding.kind, sample.kind) {
                (PackingFieldKind::Padding, PackingFieldKind::Channel(channel)) => {
                    (padding.bits, sample.bits, channel)
                }
                _ => return Err(UnsupportedFeature::InputFormat.into()),
            },
            _ => return Err(UnsupportedFeature::InputFormat.into()),
        };
        let word_bits = padding
            .checked_add(channel_bits)
            .ok_or(UnsupportedFeature::InputFormat)?;
        let expected_storage_bits = if channel_bits <= 8 { 8 } else { 16 };
        if channel != *expected_channel
            || !(1..=16).contains(&channel_bits)
            || word_bits != expected_storage_bits
            || bits_per_sample.is_some_and(|bits| bits != channel_bits)
            || storage_bits.is_some_and(|bits| bits != word_bits)
        {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        bits_per_sample = Some(channel_bits);
        storage_bits = Some(word_bits);
    }
    let bits_per_sample = bits_per_sample.ok_or(UnsupportedFeature::InputFormat)?;
    let storage_bits = storage_bits.ok_or(UnsupportedFeature::InputFormat)?;
    Ok(LosslessModularSourceSpec {
        format: logical_format,
        bits_per_sample,
        bytes_per_sample: storage_bits / 8,
    })
}

/// Row-major JPEG XL pass-group grid used by one Modular frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularGroupGrid {
    pub width: u32,
    pub height: u32,
    pub columns: u32,
    pub rows: u32,
    pub groups: u32,
    pub lf_columns: u32,
    pub lf_rows: u32,
    pub lf_groups: u32,
}

impl LosslessModularGroupGrid {
    fn for_extent(width: u32, height: u32) -> Result<Self, EncodeError> {
        if width == 0 || height == 0 || width >= (1 << 30) || height >= (1 << 30) {
            return Err(EncodeError::InvalidConfiguration(
                "Modular dimensions must be in 1..2^30",
            ));
        }
        let columns = width.div_ceil(LOSSLESS_MODULAR_GROUP_DIMENSION);
        let rows = height.div_ceil(LOSSLESS_MODULAR_GROUP_DIMENSION);
        let groups = columns
            .checked_mul(rows)
            .ok_or(EncodeError::InvalidSource("Modular group count overflow"))?;
        let lf_columns = width.div_ceil(LOSSLESS_MODULAR_LF_GROUP_DIMENSION);
        let lf_rows = height.div_ceil(LOSSLESS_MODULAR_LF_GROUP_DIMENSION);
        let lf_groups = lf_columns
            .checked_mul(lf_rows)
            .ok_or(EncodeError::InvalidSource(
                "Modular LF group count overflow",
            ))?;
        // FrameGroupLayout performs the normative TOC-entry bound as well. Do it here so an
        // impossible grid is rejected before any driver allocation or queue interaction.
        FrameGroupLayout::new(lf_groups, groups, 1)?;
        Ok(Self {
            width,
            height,
            columns,
            rows,
            groups,
            lf_columns,
            lf_rows,
            lf_groups,
        })
    }

    /// Resolves a canonical row-major pass-group index to its exact pixel rectangle.
    #[must_use]
    pub fn group(self, index: u32) -> Option<LosslessModularGroup> {
        if index >= self.groups {
            return None;
        }
        let column = index % self.columns;
        let row = index / self.columns;
        let x = column.checked_mul(LOSSLESS_MODULAR_GROUP_DIMENSION)?;
        let y = row.checked_mul(LOSSLESS_MODULAR_GROUP_DIMENSION)?;
        Some(LosslessModularGroup {
            index,
            column,
            row,
            x,
            y,
            width: (self.width - x).min(LOSSLESS_MODULAR_GROUP_DIMENSION),
            height: (self.height - y).min(LOSSLESS_MODULAR_GROUP_DIMENSION),
        })
    }

    /// Iterates the standard JPEG XL TOC PassGroup order.
    pub fn ordered_groups(self) -> impl ExactSizeIterator<Item = LosslessModularGroup> {
        (0..self.groups).map(move |index| {
            self.group(index)
                .expect("an index from the checked group range is valid")
        })
    }
}

/// One GPU workgroup and its standard row-major JPEG XL PassGroup destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularGroup {
    pub index: u32,
    pub column: u32,
    pub row: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Checked memory accounting for one concrete Modular submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularMemoryPlan {
    pub group_grid: LosslessModularGroupGrid,
    pub format: LosslessModularFormat,
    /// Valid low bits in every unsigned integer component (`1..=16`).
    pub bits_per_sample: u8,
    /// Native storage bytes occupied by every component (`1` or `2`).
    pub bytes_per_sample: u8,
    /// Number of independently tokenized Modular channels (1, 3, or 4).
    pub channel_count: u32,
    pub source_binding_bytes: u64,
    pub parameter_storage_bytes: u64,
    pub artifact_storage_bytes: u64,
    pub readback_bytes: u64,
    pub owned_bytes_per_job: u64,
    pub addressed_bytes_per_job: u64,
}

/// Total memory exposure for a caller-selected maximum number of in-flight jobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularInFlightMemory {
    pub max_in_flight_jobs: u32,
    pub total_owned_bytes: u64,
    pub total_addressed_bytes: u64,
}

/// Device limits that bound concrete Modular source and artifact bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularMemoryLimits {
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub max_compute_workgroups_per_dimension: u32,
}

impl LosslessModularMemoryPlan {
    pub fn for_in_flight(
        self,
        max_in_flight_jobs: u32,
    ) -> Result<LosslessModularInFlightMemory, EncodeError> {
        if max_in_flight_jobs == 0 {
            return Err(EncodeError::InvalidConfiguration(
                "max in-flight job count must be non-zero",
            ));
        }
        let jobs = u64::from(max_in_flight_jobs);
        let total_owned_bytes =
            self.owned_bytes_per_job
                .checked_mul(jobs)
                .ok_or(EncodeError::InvalidConfiguration(
                    "in-flight encoder memory size overflow",
                ))?;
        let total_addressed_bytes = self.addressed_bytes_per_job.checked_mul(jobs).ok_or(
            EncodeError::InvalidConfiguration("in-flight encoder memory size overflow"),
        )?;
        Ok(LosslessModularInFlightMemory {
            max_in_flight_jobs,
            total_owned_bytes,
            total_addressed_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct ModularGroupPlan {
    width: u32,
    height: u32,
    channel: u32,
    artifact_byte_offset: u64,
    output_size: u64,
    max_events: usize,
}

#[derive(Clone, Debug)]
struct ModularDispatchPlan {
    width: u32,
    height: u32,
    group_grid: LosslessModularGroupGrid,
    format: LosslessModularFormat,
    bits_per_sample: u8,
    parameters: Vec<ModularParams>,
    groups: Vec<ModularGroupPlan>,
    source_binding_offset: u64,
    source_binding_size: NonZeroU64,
    output_size: u64,
    memory: LosslessModularMemoryPlan,
}

/// GPU lossless 1-16-bit integer Modular encoding with row-major 256x256 pass groups.
///
/// It never reads source pixels on the CPU. The source buffer may contain packed Gray, RGB, or
/// RGBA unsigned samples in canonical native `u8`/`u16` storage. RGB samples use the normative
/// reversible YCoCg transform in WGSL before prediction. The GPU emits predictor
/// residual tokens and histograms; the host only serializes those artifacts.
pub struct LosslessModularBackend {
    pipeline: Arc<wgpu::ComputePipeline>,
    buffer_pool: Arc<EncoderBufferPool>,
    capabilities: EncoderCapabilities,
    max_storage_binding_size: u64,
    max_buffer_size: u64,
    storage_offset_alignment: u64,
    max_compute_workgroups_per_dimension: u32,
}

impl LosslessModularBackend {
    #[must_use]
    pub fn new(context: &WgpuContext) -> Self {
        let module = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("jxl-wgpu lossless modular token kernel"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = Arc::new(context.device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu lossless modular token pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("encode"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));
        let limits = context.device().limits();
        Self {
            pipeline,
            buffer_pool: EncoderBufferPool::new(DEFAULT_ENCODER_BUFFER_POOL_BYTES),
            capabilities: EncoderCapabilities {
                profiles: vec![ProfileCapability::ModularLossless {
                    min_bits_per_sample: 1,
                    max_bits_per_sample: 16,
                }],
                max_progressive_passes: 1,
                animation: false,
                determinism: Determinism::CrossDevice,
                implemented_stages: vec![
                    KernelStage::ColorTransform,
                    KernelStage::ModularTransform,
                    KernelStage::ModularPrediction,
                    KernelStage::ModularResidualTokenization,
                    KernelStage::HistogramReduction,
                ],
            },
            max_storage_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            storage_offset_alignment: u64::from(limits.min_storage_buffer_offset_alignment),
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }

    pub fn memory_plan(
        &self,
        source: &crate::BufferImageSource,
    ) -> Result<LosslessModularMemoryPlan, EncodeError> {
        Ok(self.dispatch_plan(source)?.memory)
    }

    #[must_use]
    pub fn memory_limits(&self) -> LosslessModularMemoryLimits {
        LosslessModularMemoryLimits {
            max_storage_buffer_binding_size: self.max_storage_binding_size,
            max_buffer_size: self.max_buffer_size,
            min_storage_buffer_offset_alignment: self.storage_offset_alignment,
            max_compute_workgroups_per_dimension: self.max_compute_workgroups_per_dimension,
        }
    }

    /// Reports encoder-owned parameter, artifact, and readback buffers retained for reuse.
    #[must_use]
    pub fn buffer_pool_stats(&self) -> EncoderBufferPoolStats {
        self.buffer_pool.stats()
    }

    /// Changes the maximum idle allocation bytes retained by this backend.
    ///
    /// A value of zero disables retention. Reducing the limit immediately evicts oldest idle
    /// sets; resources already leased to GPU jobs follow the new limit when they complete.
    pub fn set_buffer_pool_limit(&self, limit_bytes: u64) {
        self.buffer_pool.set_limit(limit_bytes);
    }

    /// Drops all idle buffers and prevents currently leased sets from re-entering the pool.
    ///
    /// In-flight buffers remain exclusively owned by their submissions until their GPU mapping
    /// callback finishes, then are discarded instead of being retained.
    pub fn clear_buffer_pool(&self) {
        self.buffer_pool.clear();
    }

    fn dispatch_plan(
        &self,
        source: &crate::BufferImageSource,
    ) -> Result<ModularDispatchPlan, EncodeError> {
        let extent = source.layout.extent;
        let group_grid = LosslessModularGroupGrid::for_extent(extent.width, extent.height)?;
        let source_spec = lossless_modular_source_spec(&source.layout.format)?;
        let format = source_spec.format;
        let channels = format.channel_count();
        let dispatches =
            group_grid
                .groups
                .checked_mul(channels)
                .ok_or(EncodeError::InvalidSource(
                    "Modular dispatch count overflow",
                ))?;
        if dispatches > self.max_compute_workgroups_per_dimension {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_compute_workgroups_per_dimension",
                required: u64::from(dispatches),
                available: u64::from(self.max_compute_workgroups_per_dimension),
            }
            .into());
        }
        if source.layout.planes.len() != 1
            || !source.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            || !source.buffer.size().is_multiple_of(4)
        {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        let plane = source
            .layout
            .plane(0)
            .ok_or(EncodeError::InvalidSource("missing Modular plane"))?;
        let row_stride = u32::try_from(plane.row_stride).map_err(|_| {
            EncodeError::InvalidSource("row stride exceeds the Modular profile limit")
        })?;
        let row_bytes = u64::from(extent.width)
            .checked_mul(u64::from(channels))
            .and_then(|value| value.checked_mul(u64::from(source_spec.bytes_per_sample)))
            .ok_or(EncodeError::InvalidSource("source row size overflow"))?;
        if plane.row_stride < row_bytes {
            return Err(EncodeError::InvalidSource(
                "row stride is smaller than the packed Modular row width",
            ));
        }
        let preceding_rows = plane
            .row_stride
            .checked_mul(u64::from(extent.height - 1))
            .ok_or(EncodeError::InvalidSource(
                "source address arithmetic overflow",
            ))?;
        let sample_end = plane
            .offset
            .checked_add(preceding_rows)
            .and_then(|value| value.checked_add(row_bytes))
            .ok_or(EncodeError::InvalidSource(
                "source address arithmetic overflow",
            ))?;
        let _full_stride_end = plane
            .row_stride
            .checked_mul(u64::from(extent.height))
            .and_then(|value| plane.offset.checked_add(value))
            .ok_or(EncodeError::InvalidSource(
                "source address arithmetic overflow",
            ))?;
        let binding_end = align_up(sample_end, 4)
            .ok_or(EncodeError::InvalidSource("source binding size overflow"))?;
        if binding_end > source.buffer.size() {
            return Err(EncodeError::InvalidSource(
                "source binding does not contain the final addressable sample word",
            ));
        }
        // A storage array of u32 also needs a word-aligned base even on a
        // hypothetical device reporting a smaller dynamic-offset alignment.
        let alignment = self.storage_offset_alignment.max(4);
        let source_binding_offset = plane.offset - plane.offset % alignment;
        if !source_binding_offset.is_multiple_of(alignment) {
            return Err(EncodeError::InvalidSource(
                "source storage binding offset is not device-aligned",
            ));
        }
        let source_binding_bytes = binding_end
            .checked_sub(source_binding_offset)
            .ok_or(EncodeError::InvalidSource("source binding range underflow"))?;
        if !source_binding_bytes.is_multiple_of(4) {
            return Err(EncodeError::InvalidSource(
                "source storage binding size is not word-aligned",
            ));
        }
        if source_binding_bytes > self.max_storage_binding_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffer_binding_size",
                required: source_binding_bytes,
                available: self.max_storage_binding_size,
            }
            .into());
        }
        let source_binding_size = NonZeroU64::new(source_binding_bytes)
            .ok_or(EncodeError::InvalidSource("source binding is empty"))?;
        let shader_last_byte = sample_end
            .checked_sub(source_binding_offset)
            .and_then(|value| value.checked_sub(1))
            .ok_or(EncodeError::InvalidSource(
                "source address arithmetic underflow",
            ))?;
        u32::try_from(shader_last_byte).map_err(|_| {
            EncodeError::InvalidSource("source address exceeds the WGSL u32 address space")
        })?;

        let dispatch_count = usize::try_from(dispatches)
            .map_err(|_| EncodeError::InvalidSource("Modular dispatch count overflow"))?;
        let mut parameters = Vec::with_capacity(dispatch_count);
        let mut groups = Vec::with_capacity(dispatch_count);
        let mut output_size = 0u64;
        for group in group_grid.ordered_groups() {
            let x = group.x;
            let y = group.y;
            let width = group.width;
            let height = group.height;
            let pixel_count = usize::try_from(u64::from(width) * u64::from(height))
                .map_err(|_| EncodeError::InvalidSource("group dimensions overflow"))?;
            let max_events = event_capacity(pixel_count)?;
            let output_words = OUTPUT_HEADER_WORDS
                .checked_add(
                    max_events
                        .checked_mul(EVENT_WORDS)
                        .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?,
                )
                .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?;
            let group_output_size = u64::try_from(output_words)
                .ok()
                .and_then(|words| words.checked_mul(4))
                .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?;
            let tile_byte_offset = plane
                .offset
                .checked_add(plane.row_stride.checked_mul(u64::from(y)).ok_or(
                    EncodeError::InvalidSource("source address arithmetic overflow"),
                )?)
                .and_then(|value| {
                    u64::from(x)
                        .checked_mul(u64::from(channels))
                        .and_then(|value| {
                            value.checked_mul(u64::from(source_spec.bytes_per_sample))
                        })
                        .and_then(|x_offset| value.checked_add(x_offset))
                })
                .and_then(|value| value.checked_sub(source_binding_offset))
                .ok_or(EncodeError::InvalidSource(
                    "source address arithmetic overflow",
                ))?;
            let byte_offset = u32::try_from(tile_byte_offset).map_err(|_| {
                EncodeError::InvalidSource("source address exceeds the WGSL u32 address space")
            })?;
            for channel in 0..channels {
                let output_word_offset = u32::try_from(output_size / 4).map_err(|_| {
                    EncodeError::InvalidSource("artifact buffer exceeds WGSL u32 indexing")
                })?;
                parameters.push(ModularParams {
                    width,
                    height,
                    row_stride,
                    byte_offset,
                    output_word_offset,
                    channel,
                    channels,
                    bytes_per_sample: u32::from(source_spec.bytes_per_sample),
                    sample_mask: (1u32 << source_spec.bits_per_sample) - 1,
                });
                groups.push(ModularGroupPlan {
                    width,
                    height,
                    channel,
                    artifact_byte_offset: output_size,
                    output_size: group_output_size,
                    max_events,
                });
                output_size = output_size
                    .checked_add(group_output_size)
                    .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?;
            }
        }
        let output_words = output_size / 4;
        if output_words > u64::from(u32::MAX) + 1 {
            return Err(EncodeError::InvalidSource(
                "artifact buffer exceeds WGSL u32 indexing",
            ));
        }
        if output_size > self.max_storage_binding_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffer_binding_size",
                required: output_size,
                available: self.max_storage_binding_size,
            }
            .into());
        }
        if output_size > self.max_buffer_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_buffer_size",
                required: output_size,
                available: self.max_buffer_size,
            }
            .into());
        }
        let parameter_storage_bytes = u64::try_from(parameters.len())
            .ok()
            .and_then(|count| {
                count.checked_mul(u64::try_from(std::mem::size_of::<ModularParams>()).ok()?)
            })
            .ok_or(EncodeError::InvalidSource(
                "group parameter storage size overflow",
            ))?;
        if parameter_storage_bytes > self.max_storage_binding_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffer_binding_size",
                required: parameter_storage_bytes,
                available: self.max_storage_binding_size,
            }
            .into());
        }
        if parameter_storage_bytes > self.max_buffer_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_buffer_size",
                required: parameter_storage_bytes,
                available: self.max_buffer_size,
            }
            .into());
        }
        let owned_bytes_per_job = output_size
            .checked_mul(2)
            .and_then(|value| value.checked_add(parameter_storage_bytes))
            .ok_or(EncodeError::InvalidSource("per-job memory size overflow"))?;
        let addressed_bytes_per_job = owned_bytes_per_job
            .checked_add(source_binding_bytes)
            .ok_or(EncodeError::InvalidSource("per-job memory size overflow"))?;
        let memory = LosslessModularMemoryPlan {
            group_grid,
            format,
            bits_per_sample: source_spec.bits_per_sample,
            bytes_per_sample: source_spec.bytes_per_sample,
            channel_count: channels,
            source_binding_bytes,
            parameter_storage_bytes,
            artifact_storage_bytes: output_size,
            readback_bytes: output_size,
            owned_bytes_per_job,
            addressed_bytes_per_job,
        };
        Ok(ModularDispatchPlan {
            width: extent.width,
            height: extent.height,
            group_grid,
            format,
            bits_per_sample: source_spec.bits_per_sample,
            parameters,
            groups,
            source_binding_offset,
            source_binding_size,
            output_size,
            memory,
        })
    }
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let adjustment = alignment.checked_sub(1)?;
    value
        .checked_add(adjustment)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}

fn event_capacity(pixel_count: usize) -> Result<usize, EncodeError> {
    pixel_count
        .checked_add(pixel_count.div_ceil(8))
        .and_then(|value| value.checked_add(1))
        .ok_or(EncodeError::InvalidSource("event buffer size overflow"))
}

impl GpuEncodeBackend for LosslessModularBackend {
    type Job = LosslessModularJob;

    fn capabilities(&self) -> &EncoderCapabilities {
        &self.capabilities
    }

    fn supports_input(&self, source: &GpuFrameSource) -> bool {
        let GpuFrameSource::Buffer(source) = source else {
            return false;
        };
        self.dispatch_plan(source).is_ok()
    }

    fn submit(
        &self,
        context: &WgpuContext,
        source: GpuFrameSource,
        request: &FrameEncodeRequest,
    ) -> Result<Self::Job, EncodeError> {
        if request.animation != AnimationHeader::Still
            || request.frame_index != FrameIndex::new(0)
            || !request.is_last
        {
            return Err(UnsupportedFeature::Animation.into());
        }
        if request.options != FrameOptions::default() {
            return Err(EncodeError::InvalidConfiguration(
                "the lossless Modular profile only supports default still-frame options",
            ));
        }
        let GpuFrameSource::Buffer(source) = source else {
            return Err(UnsupportedFeature::InputFormat.into());
        };
        let plan = self.dispatch_plan(&source)?;
        if request.profile
            != (EncodeProfile::ModularLossless {
                bits_per_sample: plan.bits_per_sample,
            })
        {
            return Err(EncodeError::InvalidConfiguration(
                "requested Modular depth does not match the source valid bits",
            ));
        }
        let memory_permit = context
            .memory_budget()
            .try_reserve(plan.memory.owned_bytes_per_job)?;

        let buffer_lease = self.buffer_pool.checkout(
            context.device(),
            plan.memory.parameter_storage_bytes,
            plan.output_size,
        );
        let buffers = buffer_lease.buffers();
        context.queue().write_buffer(
            &buffers.parameters,
            0,
            bytemuck::cast_slice(&plan.parameters),
        );
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu lossless modular bindings"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &source.buffer,
                            offset: plan.source_binding_offset,
                            size: Some(plan.source_binding_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.artifact.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.parameters.as_entire_binding(),
                    },
                ],
            });
        let mut commands =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu lossless modular encode"),
                });
        commands.clear_buffer(&buffers.artifact, 0, None);
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu lossless modular tokenization"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(plan.group_grid.groups * plan.format.channel_count(), 1, 1);
        }
        commands.copy_buffer_to_buffer(
            &buffers.artifact,
            0,
            &buffers.readback,
            0,
            plan.output_size,
        );

        let completion = Arc::new(MapCompletion::default());
        let callback_completion = Arc::clone(&completion);
        let readback_for_map = Arc::clone(&buffers.readback);
        let lifetime = Arc::new(EncodeJobLifetime {
            buffer_lease,
            _memory_permit: memory_permit,
            mapped: AtomicBool::new(false),
        });
        let callback_lifetime = Arc::clone(&lifetime);
        commands.map_buffer_on_submit(
            &readback_for_map,
            wgpu::MapMode::Read,
            0..plan.output_size,
            move |result| {
                if result.is_ok() {
                    callback_lifetime.mapped.store(true, Ordering::Release);
                }
                callback_completion.complete(
                    result.map_err(|error| format!("GPU artifact mapping failed: {error}")),
                );
                drop(callback_lifetime);
            },
        );
        let poll_permit = context.submission_poller().try_reserve()?;
        let submission_index = context.queue().submit([commands.finish()]);
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll_permit.register(submission_index, move |error| {
            poll_completion.complete(Err(error));
        }) {
            completion.complete(Err(format!("GPU poll registration failed: {error}")));
        }

        Ok(LosslessModularJob {
            lifetime: Some(lifetime),
            completion,
            output_size: plan.output_size,
            group_grid: plan.group_grid,
            groups: plan.groups,
            format: plan.format,
            bits_per_sample: plan.bits_per_sample,
            width: plan.width,
            height: plan.height,
            frame_index: request.frame_index,
            is_last: request.is_last,
        })
    }
}

#[derive(Default)]
struct MapCompletion {
    state: Mutex<MapState>,
    condition: Condvar,
}

#[derive(Default)]
struct MapState {
    result: Option<Result<(), String>>,
    waker: Option<Waker>,
}

impl MapCompletion {
    fn complete(&self, result: Result<(), String>) {
        let waker = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.result.is_some() {
                return;
            }
            state.result = Some(result);
            state.waker.take()
        };
        self.condition.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn poll(&self, cx: &Context<'_>) -> Option<Result<(), String>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.result.is_none() {
            state.waker = Some(cx.waker().clone());
        }
        state.result.take()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.result.is_none() {
            state = self
                .condition
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state
            .result
            .take()
            .expect("map completion was checked as present")
    }
}

/// Runtime-neutral completion for the concrete GPU lossless profile.
pub struct LosslessModularJob {
    lifetime: Option<Arc<EncodeJobLifetime>>,
    completion: Arc<MapCompletion>,
    output_size: u64,
    group_grid: LosslessModularGroupGrid,
    groups: Vec<ModularGroupPlan>,
    format: LosslessModularFormat,
    bits_per_sample: u8,
    width: u32,
    height: u32,
    frame_index: FrameIndex,
    is_last: bool,
}

struct EncodeJobLifetime {
    buffer_lease: crate::buffer_pool::EncoderBufferLease,
    _memory_permit: MemoryPermit,
    mapped: AtomicBool,
}

impl Drop for EncodeJobLifetime {
    fn drop(&mut self) {
        if self.mapped.swap(false, Ordering::AcqRel) {
            self.buffer_lease.buffers().readback.unmap();
        }
    }
}

impl LosslessModularJob {
    fn finish(&mut self, mapping: Result<(), String>) -> Result<GpuFrameArtifacts, EncodeError> {
        let lifetime = self
            .lifetime
            .take()
            .ok_or_else(|| EncodeError::Backend("GPU job was already consumed".into()))?;
        mapping.map_err(EncodeError::Backend)?;
        let readback = &lifetime.buffer_lease.buffers().readback;
        let mapped = readback
            .slice(0..self.output_size)
            .get_mapped_range()
            .map_err(|error| {
                EncodeError::Backend(format!("invalid mapped artifact range: {error}"))
            })?;
        let expected = usize::try_from(self.output_size)
            .map_err(|_| EncodeError::Backend("mapped artifact size overflow".into()))?;
        let bytes = mapped
            .get(..expected)
            .ok_or_else(|| EncodeError::Backend("mapped artifact buffer was truncated".into()))?;
        let result = build_packets(
            self.width,
            self.height,
            self.group_grid,
            self.format,
            self.bits_per_sample,
            &self.groups,
            bytes,
        );
        drop(mapped);
        readback.unmap();
        lifetime.mapped.store(false, Ordering::Release);
        drop(lifetime);
        let (packets, acceleration) = result?;
        Ok(GpuFrameArtifacts {
            frame_index: self.frame_index,
            is_last: self.is_last,
            packets,
            acceleration,
        })
    }
}

impl GpuEncodeJob for LosslessModularJob {
    fn poll_complete(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<GpuFrameArtifacts, EncodeError>> {
        match self.completion.poll(cx) {
            Some(result) => Poll::Ready(self.finish(result)),
            None => Poll::Pending,
        }
    }

    fn wait(self) -> Result<GpuFrameArtifacts, EncodeError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut job = self;
            let result = job.completion.wait();
            job.finish(result)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(EncodeError::Backend(
                "blocking GPU waits are unavailable on browser WebGPU; await the submission".into(),
            ))
        }
    }
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
            options: FrameOptions::default(),
        };
        let frame = self
            .encoder
            .submit_frame(GpuFrameSource::Buffer(source), request)?;
        Ok(LosslessModularSubmission {
            frame: Some(frame),
            codestream_header: image_header(width, height, format, source_spec.bits_per_sample)?,
            container,
            group_grid,
            format,
            bits_per_sample: source_spec.bits_per_sample,
        })
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

fn build_packets(
    width: u32,
    height: u32,
    group_grid: LosslessModularGroupGrid,
    format: LosslessModularFormat,
    bits_per_sample: u8,
    group_plans: &[ModularGroupPlan],
    bytes: &[u8],
) -> Result<(FramePacketSet, Option<GpuAccelerationArtifact>), EncodeError> {
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
        artifacts.push(artifact);
    }

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
    if group_grid.groups == 1 {
        let mut group = BitWriter::new();
        write_dc_global(&mut group, &codes, format)?;
        let token_bit_offset_in_group = u64::try_from(group.bit_len())
            .map_err(|_| EncodeError::Backend("gray8 token offset overflow".into()))?;
        for channel in 0..channels {
            write_events(&mut group, &codes[channel], artifacts[channel].events)?;
        }
        let token_bit_end = u64::try_from(group.bit_len())
            .map_err(|_| EncodeError::Backend("gray8 token length overflow".into()))?;
        let token_bit_len = token_bit_end
            .checked_sub(token_bit_offset_in_group)
            .ok_or_else(|| EncodeError::Backend("gray8 token length underflow".into()))?;
        group.align_to_byte()?;
        let packets = FramePacketSet::new(
            frame_header(format)?,
            FrameGroupLayout::new(1, 1, 1)?,
            [GroupPacket::new(
                GroupPacketKind::Single,
                group.into_bytes(),
            )],
        )?;
        let acceleration =
            (format == LosslessModularFormat::Gray && bits_per_sample == 8).then(|| {
                GpuAccelerationArtifact::Gray8Prefix {
                    width,
                    height,
                    token_bit_offset_in_group,
                    token_bit_len,
                    raw_prefix: codes[0].raw_entries(),
                    lz77_prefix: codes[0].lz77_entries(),
                }
            });
        return Ok((packets, acceleration));
    }

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
    for group in 0..usize::try_from(group_grid.groups)
        .map_err(|_| EncodeError::Backend("Modular group index overflow".into()))?
    {
        let mut pass_group = BitWriter::new();
        // GroupHeader: use the LF-global tree, default weighted predictor, no transforms.
        pass_group.write_bits(1, 1)?;
        pass_group.write_bits(1, 1)?;
        pass_group.write_bits(0, 2)?;
        for channel in 0..channels {
            let artifact = &artifacts[group * channels + channel];
            write_events(&mut pass_group, &codes[channel], artifact.events)?;
        }
        pass_group.align_to_byte()?;
        packets.push(GroupPacket::new(
            GroupPacketKind::AcGroup {
                pass: 0,
                group: u32::try_from(group)
                    .map_err(|_| EncodeError::Backend("Modular group index overflow".into()))?,
            },
            pass_group.into_bytes(),
        ));
    }
    Ok((
        FramePacketSet::new(frame_header(format)?, layout, packets)?,
        None,
    ))
}

#[derive(Clone, Copy)]
struct ValidatedModularArtifact<'a> {
    header: ModularArtifactHeader,
    events: &'a [ModularEvent],
}

fn parse_group_artifact<'a>(
    width: u32,
    height: u32,
    max_events: usize,
    bytes: &'a [u8],
) -> Result<ValidatedModularArtifact<'a>, EncodeError> {
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

fn write_events(
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
    EncodeError::Backend(format!("invalid GPU artifact: {reason}"))
}

fn write_dc_global(
    output: &mut BitWriter,
    codes: &[PrefixCode; 4],
    format: LosslessModularFormat,
) -> Result<(), EncodeError> {
    // Handcrafted Modular metadata adapted from zune-jpegxl 0.5.2. See this crate's
    // `THIRD_PARTY.md` and `LICENSES/zune-jpegxl-MIT.txt`.
    output.write_bits(1, 1)?;
    output.write_bits(1, 1)?;
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

fn image_header(
    width: u32,
    height: u32,
    format: LosslessModularFormat,
    bits_per_sample: u8,
) -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0x0aff, 16)?;
    output.write_bits(0, 1)?;
    write_size(&mut output, height, true)?;
    write_size(&mut output, width, false)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 1)?;
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
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.align_to_byte()?;
    Ok(BitFragment::byte_aligned(output.into_bytes())?)
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

fn frame_header(format: LosslessModularFormat) -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    if format.has_alpha() {
        output.write_bits(0, 2)?;
    }
    output.write_bits(1, 2)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    if format.has_alpha() {
        output.write_bits(0, 2)?;
    }
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 2)?;
    let bit_len = output.bit_len();
    BitFragment::new(output.into_bytes(), bit_len).map_err(Into::into)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    use super::*;
    use jxl::api::{
        Endianness, JxlBitDepth, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions,
        JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
    };
    use jxl_gpu_formats::{ImageLayout, PitchLinearPlaneLayout};
    use jxl_gpu_protocol::Extent2d;
    use std::process::Command;

    fn checked_in_gpu_gray8_lossless() -> Vec<u8> {
        fn nibble(byte: u8) -> u8 {
            match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => panic!("invalid checked-in fixture hex digit"),
            }
        }

        let digits = include_str!("../test-data/gpu_gray8_lossless.jxl.hex")
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        assert_eq!(digits.len() % 2, 0, "fixture hex must contain whole bytes");
        digits
            .chunks_exact(2)
            .map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1]))
            .collect()
    }

    struct ReentrantWake {
        completion: Arc<MapCompletion>,
        observed_unlocked: Arc<AtomicBool>,
    }

    impl std::task::Wake for ReentrantWake {
        fn wake(self: Arc<Self>) {
            let _guard = self
                .completion
                .state
                .try_lock()
                .expect("completion mutex must be unlocked before invoking a waker");
            self.observed_unlocked.store(true, Ordering::Release);
        }
    }

    #[test]
    fn completion_wakes_after_releasing_its_mutex() {
        let completion = Arc::new(MapCompletion::default());
        let observed_unlocked = Arc::new(AtomicBool::new(false));
        let waker = std::task::Waker::from(Arc::new(ReentrantWake {
            completion: Arc::clone(&completion),
            observed_unlocked: Arc::clone(&observed_unlocked),
        }));
        let context = Context::from_waker(&waker);
        assert!(completion.poll(&context).is_none());
        completion.complete(Ok(()));
        assert!(observed_unlocked.load(Ordering::Acquire));
    }

    #[test]
    fn modular_params_abi_matches_wgsl_storage_array() {
        assert_eq!(std::mem::size_of::<ModularParams>(), 36);
        assert_eq!(std::mem::align_of::<ModularParams>(), 4);
        let params = ModularParams {
            width: 1,
            height: 2,
            row_stride: 3,
            byte_offset: 4,
            output_word_offset: 5,
            channel: 6,
            channels: 7,
            bytes_per_sample: 8,
            sample_mask: 9,
        };
        assert_eq!(
            bytemuck::cast::<ModularParams, [u32; 9]>(params),
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        );
        assert!(SHADER.contains(
            "struct Params {\n    width: u32,\n    height: u32,\n    row_stride: u32,\n    byte_offset: u32,\n    output_word_offset: u32,\n    channel: u32,\n    channels: u32,\n    bytes_per_sample: u32,\n    sample_mask: u32,\n}"
        ));
        assert!(SHADER.contains("var<storage, read> group_params: array<Params>;"));
    }

    #[test]
    fn modular_artifact_abi_matches_wgsl_word_schema() {
        assert_eq!(std::mem::size_of::<ModularArtifactHeader>(), 53 * 4);
        assert_eq!(std::mem::align_of::<ModularArtifactHeader>(), 4);
        assert_eq!(std::mem::size_of::<ModularEvent>(), 4 * 4);
        assert_eq!(std::mem::align_of::<ModularEvent>(), 4);

        let header = ModularArtifactHeader {
            event_count: 7,
            raw_counts: std::array::from_fn(|index| 100 + index as u32),
            lz77_counts: std::array::from_fn(|index| 200 + index as u32),
        };
        let words = bytemuck::cast::<ModularArtifactHeader, [u32; 53]>(header);
        assert_eq!(words[0], 7);
        assert_eq!(words[1..20], header.raw_counts);
        assert_eq!(words[20..53], header.lz77_counts);

        let event = ModularEvent {
            kind: 1,
            token: 2,
            extra_bit_count: 3,
            extra_bits: 4,
        };
        assert_eq!(
            bytemuck::cast::<ModularEvent, [u32; 4]>(event),
            [1, 2, 3, 4]
        );
        assert!(SHADER.contains("Word 0 is the event count, words 1..20 are raw-token counts"));
        assert!(SHADER.contains("// (kind, token, extra-bit count, extra bits)."));
        assert!(SHADER.contains("const OUTPUT_HEADER_WORDS: u32 = 53u;"));
        assert!(SHADER.contains("const EVENT_WORDS: u32 = 4u;"));
    }

    #[test]
    fn modular_input_contract_is_explicit_and_does_not_relabel_defined_color() {
        for format in [
            LosslessModularFormat::Gray,
            LosslessModularFormat::Rgb,
            LosslessModularFormat::Rgba,
        ] {
            for bits_per_sample in 1..=16 {
                let pixel_format = format.pixel_format(bits_per_sample).unwrap();
                let spec = lossless_modular_source_spec(&pixel_format).unwrap();
                assert_eq!(spec.format, format);
                assert_eq!(spec.bits_per_sample, bits_per_sample);
                assert_eq!(spec.bytes_per_sample, u8::from(bits_per_sample > 8) + 1);
                let word = &pixel_format.planes[0].words[0];
                assert_eq!(word.bits(), u32::from(spec.bytes_per_sample) * 8);
                assert!(matches!(
                    word.fields.last().map(|field| field.kind),
                    Some(PackingFieldKind::Channel(Channel::X))
                ));
            }
        }
        assert!(LosslessModularFormat::Gray.pixel_format(0).is_err());
        assert!(LosslessModularFormat::Gray.pixel_format(17).is_err());
        let undefined = ColorSpecification::Undefined;
        assert!(
            lossless_modular_source_spec(
                &PixelFormat::rgb8(RgbChannelOrder::Rgb, true, undefined,)
            )
            .is_err()
        );
        for order in [RgbChannelOrder::Bgr, RgbChannelOrder::Bgra] {
            assert!(
                lossless_modular_source_spec(&PixelFormat::rgb8(order, false, undefined)).is_err()
            );
        }
        let defined = ColorSpecification::Defined(jxl_gpu_formats::ColorSpec::bt709(
            jxl_gpu_formats::ColorRange::Full,
            jxl_gpu_formats::ChromaLocation2d::CENTER,
        ));
        assert!(
            lossless_modular_source_spec(&PixelFormat::rgb8(RgbChannelOrder::Rgb, false, defined,))
                .is_err()
        );
    }

    #[test]
    fn group_grid_is_row_major_and_covers_edge_tiles_exactly() {
        let grid = LosslessModularGroupGrid::for_extent(513, 257).unwrap();
        assert_eq!(
            grid,
            LosslessModularGroupGrid {
                width: 513,
                height: 257,
                columns: 3,
                rows: 2,
                groups: 6,
                lf_columns: 1,
                lf_rows: 1,
                lf_groups: 1,
            }
        );
        let groups = grid.ordered_groups().collect::<Vec<_>>();
        assert_eq!(groups.len(), 6);
        assert_eq!(
            groups[0],
            LosslessModularGroup {
                index: 0,
                column: 0,
                row: 0,
                x: 0,
                y: 0,
                width: 256,
                height: 256,
            }
        );
        assert_eq!(groups[2].x, 512);
        assert_eq!(groups[2].width, 1);
        assert_eq!(groups[3].y, 256);
        assert_eq!(groups[3].height, 1);
        assert_eq!((groups[5].x, groups[5].y), (512, 256));
        assert!(grid.group(6).is_none());

        assert_eq!(
            LosslessModularGroupGrid::for_extent(1, 1).unwrap().groups,
            1
        );
        assert!(LosslessModularGroupGrid::for_extent(0, 1).is_err());
        assert!(LosslessModularGroupGrid::for_extent(1, 0).is_err());
    }

    fn artifact_bytes(header: ModularArtifactHeader, events: &[ModularEvent]) -> Vec<u8> {
        let mut bytes = bytemuck::bytes_of(&header).to_vec();
        bytes.extend_from_slice(bytemuck::cast_slice(events));
        bytes
    }

    #[test]
    fn packet_builder_rejects_impossible_histogram_bins() {
        let mut header = ModularArtifactHeader {
            event_count: 1,
            raw_counts: [0; RAW_SYMBOLS],
            lz77_counts: [0; LZ77_SYMBOLS],
        };
        header.raw_counts[0] = 1;
        header.raw_counts[12] = 1;
        let bytes = artifact_bytes(
            header,
            &[ModularEvent {
                kind: 0,
                token: 0,
                extra_bit_count: 0,
                extra_bits: 0,
            }],
        );
        assert!(parse_group_artifact(1, 1, 1, &bytes).is_err());
    }

    #[test]
    fn packet_builder_rejects_noncanonical_events_and_histogram_mismatches() {
        let mut header = ModularArtifactHeader {
            event_count: 1,
            raw_counts: [0; RAW_SYMBOLS],
            lz77_counts: [0; LZ77_SYMBOLS],
        };
        header.raw_counts[2] = 1;
        let malformed = artifact_bytes(
            header,
            &[ModularEvent {
                kind: 0,
                token: 2,
                extra_bit_count: 0,
                extra_bits: 0,
            }],
        );
        assert!(parse_group_artifact(1, 1, 1, &malformed).is_err());

        header.raw_counts = [0; RAW_SYMBOLS];
        header.raw_counts[1] = 1;
        let mismatched = artifact_bytes(
            header,
            &[ModularEvent {
                kind: 0,
                token: 0,
                extra_bit_count: 0,
                extra_bits: 0,
            }],
        );
        assert!(parse_group_artifact(1, 1, 1, &mismatched).is_err());
    }

    #[test]
    fn packet_builder_rejects_event_streams_with_the_wrong_sample_count() {
        let mut header = ModularArtifactHeader {
            event_count: 1,
            raw_counts: [0; RAW_SYMBOLS],
            lz77_counts: [0; LZ77_SYMBOLS],
        };
        header.raw_counts[0] = 1;
        let bytes = artifact_bytes(
            header,
            &[ModularEvent {
                kind: 0,
                token: 0,
                extra_bit_count: 0,
                extra_bits: 0,
            }],
        );
        assert!(parse_group_artifact(2, 1, 1, &bytes).is_err());
    }

    /// Mirrors only the event-admission control flow in `encode` WGSL. A
    /// `true` sample is a zero packed residual; the actual token value is
    /// irrelevant to the number of four-word event records.
    fn simulated_shader_event_count(
        width: usize,
        height: usize,
        is_zero: impl Fn(usize) -> bool,
    ) -> usize {
        let mut run = 0usize;
        let mut events = 0usize;
        for y in 0..height {
            for chunk_x in (0..width).step_by(8) {
                let count = 8.min(width - chunk_x);
                let mut prefix = 0usize;
                while prefix < count && is_zero(y * width + chunk_x + prefix) {
                    prefix += 1;
                }
                if prefix == count && (run > 0 || prefix > 7) {
                    run += prefix;
                } else if prefix + run > 7 {
                    events += usize::from(run + prefix > 0);
                    events += count - prefix;
                    run = 0;
                } else {
                    events += count;
                }
            }
        }
        events + usize::from(run > 0)
    }

    #[test]
    fn event_allocation_bounds_every_shader_write() {
        // Exhaust every zero/non-zero residual stream up to 16 samples and
        // vary row boundaries because a run is intentionally frame-global.
        for width in 1usize..=16 {
            for height in 1usize..=(16 / width) {
                let pixels = width * height;
                let capacity = event_capacity(pixels).expect("small capacity is representable");
                for mask in 0u32..(1u32 << pixels) {
                    let events = simulated_shader_event_count(width, height, |index| {
                        mask & (1u32 << index) != 0
                    });
                    assert!(events <= capacity, "{width}x{height}, mask={mask:#x}");
                }
            }
        }

        let pixels = usize::try_from(
            u64::from(LOSSLESS_MODULAR_GROUP_DIMENSION)
                * u64::from(LOSSLESS_MODULAR_GROUP_DIMENSION),
        )
        .expect("maximum Modular profile dimensions fit usize");
        let capacity = event_capacity(pixels).expect("maximum event capacity fits usize");
        for events in [
            simulated_shader_event_count(pixels, 1, |_| false),
            simulated_shader_event_count(pixels, 1, |_| true),
            simulated_shader_event_count(pixels, 1, |index| index % 2 == 0),
            simulated_shader_event_count(pixels, 1, |index| index % 17 < 8),
        ] {
            assert!(events <= capacity);
        }

        let words = OUTPUT_HEADER_WORDS + capacity * EVENT_WORDS;
        let last_event_word = OUTPUT_HEADER_WORDS + (capacity - 1) * EVENT_WORDS + 3;
        assert!(last_event_word < words);
        assert_eq!(words * 4 % wgpu::COPY_BUFFER_ALIGNMENT as usize, 0);
    }

    fn test_context() -> Option<WgpuContext> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jxl-wgpu lossless encoder test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        WgpuContext::new(Arc::new(device), Arc::new(queue)).ok()
    }

    fn packed_test_source(
        context: &WgpuContext,
        width: u32,
        height: u32,
    ) -> crate::BufferImageSource {
        let extent = Extent2d::new(width, height);
        let layout = ImageLayout::packed(
            extent,
            PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        )
        .unwrap();
        let pixels = packed_test_pixels(width, height);
        let buffer = Arc::new(context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu encoder pool test source"),
                contents: &pixels,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        crate::BufferImageSource::new(buffer, layout).unwrap()
    }

    fn packed_test_pixels(width: u32, height: u32) -> Vec<u8> {
        let byte_count = usize::try_from(u64::from(width) * u64::from(height))
            .expect("test source size fits usize");
        (0..byte_count)
            .map(|index| ((index * 29 + index / 7) & 255) as u8)
            .collect()
    }

    fn packed_color_test_source(
        context: &WgpuContext,
        width: u32,
        height: u32,
        format: LosslessModularFormat,
    ) -> (crate::BufferImageSource, Vec<u8>) {
        let channels = format.channel_count();
        assert!(matches!(channels, 3 | 4));
        let extent = Extent2d::new(width, height);
        let row_bytes = u64::from(width) * u64::from(channels);
        let row_stride = row_bytes + 5;
        let offset = 4u64;
        let allocation_size = align_up(offset + row_stride * u64::from(height), 4).unwrap();
        let mut allocation = vec![0xa5; allocation_size as usize];
        let mut expected = Vec::with_capacity((width * height * channels) as usize);
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    let value = ((x * 37 + y * 71 + channel * 53 + (x * y + channel * y) % 251)
                        & 255) as u8;
                    let address =
                        offset + u64::from(y) * row_stride + u64::from(x * channels + channel);
                    allocation[address as usize] = value;
                    expected.push(value);
                }
            }
        }
        let order = match format {
            LosslessModularFormat::Rgb => RgbChannelOrder::Rgb,
            LosslessModularFormat::Rgba => RgbChannelOrder::Rgba,
            LosslessModularFormat::Gray => unreachable!(),
        };
        let pixel_format = PixelFormat::rgb8(order, false, ColorSpecification::Undefined);
        let layout = ImageLayout::from_planes(
            extent,
            pixel_format,
            vec![PitchLinearPlaneLayout {
                plane_index: 0,
                offset,
                row_stride,
                sample_extent: extent,
                row_bytes,
            }],
        )
        .unwrap();
        let buffer = Arc::new(context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu packed color test source"),
                contents: &allocation,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        (
            crate::BufferImageSource::new(buffer, layout).unwrap(),
            expected,
        )
    }

    fn modular_integer_test_source(
        context: &WgpuContext,
        width: u32,
        height: u32,
        format: LosslessModularFormat,
        bits_per_sample: u8,
    ) -> (crate::BufferImageSource, Vec<u16>) {
        let channels = format.channel_count();
        let bytes_per_sample = if bits_per_sample <= 8 { 1u64 } else { 2u64 };
        let max_value = (1u32 << bits_per_sample) - 1;
        let extent = Extent2d::new(width, height);
        let row_bytes = u64::from(width) * u64::from(channels) * bytes_per_sample;
        let row_stride = row_bytes + 5;
        let offset = 5u64;
        let allocation_size = align_up(offset + row_stride * u64::from(height), 4).unwrap();
        let mut allocation = vec![0xa5; allocation_size as usize];
        let mut expected = Vec::with_capacity((width * height * channels) as usize);
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    let selector = (x + y * 3 + channel * 5) % 7;
                    let generated = x * 37 + y * 71 + channel * 53 + (x * y + channel * y) % 251;
                    let value = match selector {
                        0 => 0,
                        1 => max_value,
                        2 => 1.min(max_value),
                        3 => max_value.saturating_sub(1),
                        _ => generated & max_value,
                    } as u16;
                    let sample_index = u64::from(x * channels + channel);
                    let address =
                        offset + u64::from(y) * row_stride + sample_index * bytes_per_sample;
                    if bytes_per_sample == 1 {
                        let padding = !max_value as u8;
                        allocation[address as usize] = value as u8 | padding;
                    } else {
                        let storage = value | (!max_value as u16);
                        allocation[address as usize..address as usize + 2]
                            .copy_from_slice(&storage.to_le_bytes());
                    }
                    expected.push(value);
                }
            }
        }
        let layout = ImageLayout::from_planes(
            extent,
            format.pixel_format(bits_per_sample).unwrap(),
            vec![PitchLinearPlaneLayout {
                plane_index: 0,
                offset,
                row_stride,
                sample_extent: extent,
                row_bytes,
            }],
        )
        .unwrap();
        let buffer = Arc::new(context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu packed integer Modular test source"),
                contents: &allocation,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        (
            crate::BufferImageSource::new(buffer, layout).unwrap(),
            expected,
        )
    }

    fn expected_artifact_storage_bytes(width: u32, height: u32) -> u64 {
        LosslessModularGroupGrid::for_extent(width, height)
            .unwrap()
            .ordered_groups()
            .map(|group| {
                let pixels =
                    usize::try_from(u64::from(group.width) * u64::from(group.height)).unwrap();
                let words = OUTPUT_HEADER_WORDS + event_capacity(pixels).unwrap() * EVENT_WORDS;
                u64::try_from(words).unwrap() * 4
            })
            .sum()
    }

    #[test]
    fn pool_exclusively_leases_real_gpu_buffer_sets_and_clear_invalidates_live_returns() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU encoder pool lease test: no wgpu adapter");
            return;
        };
        let pool = EncoderBufferPool::new(64 * 1024);
        let first = pool.checkout(context.device(), 16, 1024);
        let first_artifact = Arc::clone(&first.buffers().artifact);
        let second = pool.checkout(context.device(), 16, 1024);
        assert!(!Arc::ptr_eq(&first_artifact, &second.buffers().artifact));
        assert_eq!(pool.stats().leased_buffer_sets, 2);
        assert_eq!(pool.stats().allocation_misses, 2);

        drop(first);
        let third = pool.checkout(context.device(), 16, 1024);
        assert!(Arc::ptr_eq(&first_artifact, &third.buffers().artifact));
        assert_eq!(pool.stats().reuse_hits, 1);
        assert_eq!(pool.stats().leased_buffer_sets, 2);

        pool.clear();
        drop(second);
        drop(third);
        let stats = pool.stats();
        assert_eq!(stats.leased_buffer_sets, 0);
        assert_eq!(stats.idle_buffer_sets, 0);
        assert_eq!(stats.idle_buffers, 0);
        assert_eq!(stats.evicted_buffer_sets, 2);
        assert_eq!(stats.evicted_buffers, 6);
    }

    #[test]
    fn sequential_gpu_jobs_reuse_exact_buffer_sets_and_enforce_the_idle_limit() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU encoder reuse test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 17, 13);
        let encoder = LosslessModularEncoder::with_buffer_pool_limit(context, 8 * 1024 * 1024);
        let allocation_bytes = encoder.memory_plan(&source).unwrap().owned_bytes_per_job;

        encoder.submit(source.clone()).unwrap().wait().unwrap();
        let first = encoder.buffer_pool_stats();
        assert_eq!(first.allocation_misses, 1);
        assert_eq!(first.reuse_hits, 0);
        assert_eq!(first.idle_buffer_sets, 1);
        assert_eq!(first.idle_buffers, 3);
        assert_eq!(first.idle_bytes, allocation_bytes);

        encoder.submit(source).unwrap().wait().unwrap();
        let reused = encoder.buffer_pool_stats();
        assert_eq!(reused.allocation_misses, 1);
        assert_eq!(reused.reuse_hits, 1);
        assert_eq!(reused.idle_buffer_sets, 1);

        encoder.set_buffer_pool_limit(allocation_bytes - 1);
        let evicted = encoder.buffer_pool_stats();
        assert_eq!(evicted.limit_bytes, allocation_bytes - 1);
        assert_eq!(evicted.idle_bytes, 0);
        assert_eq!(evicted.idle_buffer_sets, 0);
        assert_eq!(evicted.evicted_buffer_sets, 1);
        assert_eq!(evicted.evicted_buffers, 3);
        assert_eq!(evicted.evicted_bytes, allocation_bytes);
    }

    #[test]
    fn abandoned_gpu_future_returns_buffers_and_live_memory_after_mapping() {
        let Some(context) = test_context() else {
            eprintln!("skipping abandoned GPU encoder reuse test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 71, 121);
        let encoder = LosslessModularEncoder::with_buffer_pool_limit(
            context.clone(),
            DEFAULT_ENCODER_BUFFER_POOL_BYTES,
        );
        let dropped = encoder.submit(source.clone()).unwrap();
        assert_eq!(encoder.buffer_pool_stats().leased_buffer_sets, 1);
        assert_eq!(encoder.buffer_pool_stats().idle_buffer_sets, 0);
        assert!(encoder.in_flight_memory_stats().reserved_bytes > 0);
        drop(dropped);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let pool = encoder.buffer_pool_stats();
            let memory = encoder.in_flight_memory_stats();
            if pool.leased_buffer_sets == 0
                && pool.idle_buffer_sets == 1
                && memory.reserved_bytes == 0
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "abandoned submission did not release resources: pool={pool:?}, memory={memory:?}"
            );
            let _ = context.device().poll(wgpu::PollType::Poll);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        encoder.submit(source).unwrap().wait().unwrap();
        let stats = encoder.buffer_pool_stats();
        assert_eq!(stats.allocation_misses, 1);
        assert_eq!(stats.reuse_hits, 1);
        assert_eq!(stats.leased_buffer_sets, 0);
    }

    #[test]
    fn concurrent_real_gpu_submissions_reuse_only_completed_buffer_sets() {
        let Some(context) = test_context() else {
            eprintln!("skipping concurrent GPU encoder reuse test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 71, 121);
        let encoder = LosslessModularEncoder::with_buffer_pool_limit(context, 32 * 1024 * 1024);
        let per_job = encoder.memory_plan(&source).unwrap().owned_bytes_per_job;
        let jobs = (0..8)
            .map(|_| encoder.submit(source.clone()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, per_job * 8);
        let first_outputs = jobs
            .into_iter()
            .map(LosslessModularSubmission::wait)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(first_outputs.windows(2).all(|pair| pair[0] == pair[1]));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);

        let first_stats = encoder.buffer_pool_stats();
        assert_eq!(first_stats.reuse_hits + first_stats.allocation_misses, 8);
        assert_eq!(first_stats.leased_buffer_sets, 0);
        assert!(first_stats.idle_buffer_sets >= 1);
        let guaranteed_hits = first_stats.idle_buffer_sets;

        let jobs = (0..8)
            .map(|_| encoder.submit(source.clone()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let second_outputs = jobs
            .into_iter()
            .map(LosslessModularSubmission::wait)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            second_outputs
                .iter()
                .all(|encoded| encoded == &first_outputs[0])
        );
        let second_stats = encoder.buffer_pool_stats();
        assert!(second_stats.reuse_hits >= first_stats.reuse_hits + guaranteed_hits);
        assert_eq!(second_stats.reuse_hits + second_stats.allocation_misses, 16);
        assert_eq!(second_stats.leased_buffer_sets, 0);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }

    #[test]
    fn poll_admission_failure_happens_before_submit_and_returns_the_pool_lease() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU encoder poll admission test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 17, 13);
        let encoder = LosslessModularEncoder::new(context.clone());
        let permits = (0..jxl_wgpu::SUBMISSION_POLLER_CAPACITY)
            .map(|_| context.submission_poller().try_reserve().unwrap())
            .collect::<Vec<_>>();

        let error = match encoder.submit(source.clone()) {
            Ok(_) => panic!("saturated poll admission must reject before queue submission"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            EncodeError::PollBackpressure(jxl_wgpu::SubmissionPollerError::Full { .. })
        ));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        let rejected = encoder.buffer_pool_stats();
        assert_eq!(rejected.leased_buffer_sets, 0);
        assert_eq!(rejected.idle_buffer_sets, 1);
        assert_eq!(rejected.allocation_misses, 1);

        drop(permits);
        encoder.submit(source).unwrap().wait().unwrap();
        let recovered = encoder.buffer_pool_stats();
        assert_eq!(recovered.reuse_hits, 1);
        assert_eq!(recovered.leased_buffer_sets, 0);
    }

    #[test]
    fn concurrent_encoder_jobs_use_owned_byte_backpressure() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU encoder backpressure test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 2, 2);
        let plan = LosslessModularBackend::new(&context)
            .memory_plan(&source)
            .unwrap();
        let limited = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(plan.owned_bytes_per_job).unwrap(),
        )
        .unwrap();
        let encoder = LosslessModularEncoder::new(limited);

        let first = encoder.submit(source.clone()).unwrap();
        assert_eq!(
            encoder.in_flight_memory_stats().reserved_bytes,
            plan.owned_bytes_per_job
        );
        assert!(matches!(
            encoder.submit(source.clone()),
            Err(EncodeError::MemoryBackpressure(
                jxl_wgpu::MemoryBudgetError::Exhausted { .. }
            ))
        ));
        first.wait().unwrap();
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        encoder.submit(source).unwrap().wait().unwrap();
    }

    fn decode_gray8(encoded: &[u8]) -> Result<((usize, usize), Vec<u8>), String> {
        let mut input = encoded;
        let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut decoder = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before image info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let size = decoder.basic_info().size;
        decoder.set_pixel_format(JxlPixelFormat {
            color_type: JxlColorType::Grayscale,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format: Vec::new(),
        });
        let mut frame = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before frame info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let mut pixels = vec![0u8; size.0 * size.1];
        {
            let mut buffers = [JxlOutputBuffer::new(&mut pixels, size.1, size.0)];
            loop {
                match frame
                    .process(&mut input, &mut buffers, None)
                    .map_err(|error| error.to_string())?
                {
                    ProcessingResult::Complete { .. } => break,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        if input.is_empty() {
                            return Err("decoder needed more input while rendering".into());
                        }
                        frame = fallback;
                    }
                }
            }
        }
        Ok((size, pixels))
    }

    fn decode_color8(
        encoded: &[u8],
        format: LosslessModularFormat,
    ) -> Result<((usize, usize), Vec<u8>), String> {
        let mut input = encoded;
        let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut decoder = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before image info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let size = decoder.basic_info().size;
        let pixel_format = match format {
            LosslessModularFormat::Rgb => JxlPixelFormat::rgb8(0),
            LosslessModularFormat::Rgba => JxlPixelFormat::rgba8(1),
            LosslessModularFormat::Gray => {
                return Err("color decoder helper requires RGB or RGBA".into());
            }
        };
        decoder.set_pixel_format(pixel_format);
        let mut frame = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before frame info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let channels = usize::try_from(format.channel_count())
            .map_err(|_| "channel count overflow".to_string())?;
        let mut pixels = vec![0u8; size.0 * size.1 * channels];
        {
            let mut buffers = [JxlOutputBuffer::new(&mut pixels, size.1, size.0 * channels)];
            loop {
                match frame
                    .process(&mut input, &mut buffers, None)
                    .map_err(|error| error.to_string())?
                {
                    ProcessingResult::Complete { .. } => break,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        if input.is_empty() {
                            return Err("decoder needed more input while rendering".into());
                        }
                        frame = fallback;
                    }
                }
            }
        }
        Ok((size, pixels))
    }

    fn decode_integer(
        encoded: &[u8],
        format: LosslessModularFormat,
        bits_per_sample: u8,
    ) -> Result<((usize, usize), Vec<u16>), String> {
        let mut input = encoded;
        let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut decoder = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before image info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let basic_info = decoder.basic_info();
        let size = basic_info.size;
        if basic_info.bit_depth
            != (JxlBitDepth::Int {
                bits_per_sample: u32::from(bits_per_sample),
            })
        {
            return Err(format!(
                "codestream depth is {:?}, expected {bits_per_sample}-bit integer",
                basic_info.bit_depth
            ));
        }
        let data_format = if bits_per_sample <= 8 {
            JxlDataFormat::U8 {
                bit_depth: bits_per_sample,
            }
        } else {
            JxlDataFormat::U16 {
                endianness: Endianness::LittleEndian,
                bit_depth: bits_per_sample,
            }
        };
        let color_type = match format {
            LosslessModularFormat::Gray => JxlColorType::Grayscale,
            LosslessModularFormat::Rgb => JxlColorType::Rgb,
            LosslessModularFormat::Rgba => JxlColorType::Rgba,
        };
        decoder.set_pixel_format(JxlPixelFormat {
            color_type,
            color_data_format: Some(data_format),
            extra_channel_format: vec![None; usize::from(format.has_alpha())],
        });
        let mut frame = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before frame info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let channels = usize::try_from(format.channel_count())
            .map_err(|_| "channel count overflow".to_string())?;
        let bytes_per_sample = data_format.bytes_per_sample();
        let row_bytes = size
            .0
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(bytes_per_sample))
            .ok_or_else(|| "decoder output row size overflow".to_string())?;
        let mut bytes = vec![0u8; row_bytes * size.1];
        {
            let mut buffers = [JxlOutputBuffer::new(&mut bytes, size.1, row_bytes)];
            loop {
                match frame
                    .process(&mut input, &mut buffers, None)
                    .map_err(|error| error.to_string())?
                {
                    ProcessingResult::Complete { .. } => break,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        if input.is_empty() {
                            return Err("decoder needed more input while rendering".into());
                        }
                        frame = fallback;
                    }
                }
            }
        }
        let pixels = if bytes_per_sample == 1 {
            bytes.into_iter().map(u16::from).collect()
        } else {
            bytes
                .chunks_exact(2)
                .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
                .collect()
        };
        Ok((size, pixels))
    }

    fn decode_integer_with_djxl_if_available(
        encoded: &[u8],
        format: LosslessModularFormat,
        bits_per_sample: u8,
    ) -> Option<Result<Vec<u16>, String>> {
        static DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);
        let djxl = "/opt/homebrew/bin/djxl";
        if Command::new(djxl).arg("-V").output().is_err() {
            return None;
        }
        let id = DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("jxl-wgpu-integer-{}-{id}", std::process::id()));
        if let Err(error) = std::fs::create_dir(&directory) {
            return Some(Err(format!(
                "could not create djxl test directory: {error}"
            )));
        }
        let input = directory.join("gpu.jxl");
        let output = directory.join("gpu.pam");
        let result = (|| {
            std::fs::write(&input, encoded)
                .map_err(|error| format!("could not write djxl input: {error}"))?;
            let command = Command::new(djxl)
                .arg(&input)
                .arg(&output)
                .arg("--quiet")
                .arg(format!("--bits_per_sample={bits_per_sample}"))
                .output()
                .map_err(|error| format!("could not execute djxl: {error}"))?;
            if !command.status.success() {
                return Err(format!(
                    "djxl rejected GPU integer codestream: {}",
                    String::from_utf8_lossy(&command.stderr)
                ));
            }
            let pam = std::fs::read(&output)
                .map_err(|error| format!("could not read djxl PAM: {error}"))?;
            parse_integer_pam(&pam, format, bits_per_sample)
        })();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_dir(directory);
        Some(result)
    }

    fn parse_integer_pam(
        bytes: &[u8],
        format: LosslessModularFormat,
        bits_per_sample: u8,
    ) -> Result<Vec<u16>, String> {
        let marker = b"ENDHDR\n";
        let header_end = bytes
            .windows(marker.len())
            .position(|window| window == marker)
            .map(|position| position + marker.len())
            .ok_or_else(|| "djxl PAM is missing ENDHDR".to_string())?;
        let header = std::str::from_utf8(&bytes[..header_end])
            .map_err(|error| format!("djxl PAM header is not UTF-8: {error}"))?;
        if !header.starts_with("P7\n") {
            return Err("djxl did not emit a PAM P7 image".into());
        }
        let value = |key: &str| -> Result<usize, String> {
            header
                .lines()
                .find_map(|line| line.strip_prefix(key))
                .ok_or_else(|| format!("djxl PAM is missing {key}"))?
                .trim()
                .parse::<usize>()
                .map_err(|error| format!("invalid djxl PAM {key}: {error}"))
        };
        let width = value("WIDTH")?;
        let height = value("HEIGHT")?;
        let depth = value("DEPTH")?;
        let max_value = value("MAXVAL")?;
        let expected_depth = usize::try_from(format.channel_count())
            .map_err(|_| "PAM channel count overflow".to_string())?;
        let expected_max = (1usize << bits_per_sample) - 1;
        if depth != expected_depth || max_value != expected_max {
            return Err(format!(
                "djxl PAM has depth/maxval {depth}/{max_value}, expected {expected_depth}/{expected_max}"
            ));
        }
        let pixels = bytes
            .get(header_end..)
            .ok_or_else(|| "djxl PAM pixels are truncated".to_string())?;
        let samples = width
            .checked_mul(height)
            .and_then(|value| value.checked_mul(depth))
            .ok_or_else(|| "djxl PAM dimensions overflow".to_string())?;
        let bytes_per_sample = usize::from(bits_per_sample > 8) + 1;
        if pixels.len() != samples * bytes_per_sample {
            return Err(format!(
                "djxl PAM has {} bytes, expected {}",
                pixels.len(),
                samples * bytes_per_sample
            ));
        }
        Ok(if bytes_per_sample == 1 {
            pixels.iter().copied().map(u16::from).collect()
        } else {
            pixels
                .chunks_exact(2)
                .map(|sample| u16::from_be_bytes([sample[0], sample[1]]))
                .collect()
        })
    }

    fn decode_with_djxl_if_available(encoded: &[u8]) -> Option<Result<Vec<u8>, String>> {
        static DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);
        if Command::new("djxl").arg("-V").output().is_err() {
            return None;
        }
        let id = DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("jxl-wgpu-gray8-{}-{id}", std::process::id()));
        if let Err(error) = std::fs::create_dir(&directory) {
            return Some(Err(format!(
                "could not create djxl test directory: {error}"
            )));
        }
        let input = directory.join("gpu.jxl");
        let output = directory.join("gpu.pgm");
        let result = (|| {
            std::fs::write(&input, encoded)
                .map_err(|error| format!("could not write djxl input: {error}"))?;
            let command = Command::new("djxl")
                .arg(&input)
                .arg(&output)
                .arg("--quiet")
                .output()
                .map_err(|error| format!("could not execute djxl: {error}"))?;
            if !command.status.success() {
                return Err(format!(
                    "djxl rejected GPU codestream: {}",
                    String::from_utf8_lossy(&command.stderr)
                ));
            }
            let pgm = std::fs::read(&output)
                .map_err(|error| format!("could not read djxl PGM: {error}"))?;
            parse_pgm(&pgm)
        })();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_dir(directory);
        Some(result)
    }

    fn decode_color_with_djxl_if_available(
        encoded: &[u8],
        format: LosslessModularFormat,
    ) -> Option<Result<Vec<u8>, String>> {
        static DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);
        if Command::new("djxl").arg("-V").output().is_err() {
            return None;
        }
        let id = DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("jxl-wgpu-color8-{}-{id}", std::process::id()));
        if let Err(error) = std::fs::create_dir(&directory) {
            return Some(Err(format!(
                "could not create djxl test directory: {error}"
            )));
        }
        let input = directory.join("gpu.jxl");
        let output = directory.join("gpu.pam");
        let result = (|| {
            std::fs::write(&input, encoded)
                .map_err(|error| format!("could not write djxl input: {error}"))?;
            let command = Command::new("djxl")
                .arg(&input)
                .arg(&output)
                .arg("--quiet")
                .output()
                .map_err(|error| format!("could not execute djxl: {error}"))?;
            if !command.status.success() {
                return Err(format!(
                    "djxl rejected GPU color codestream: {}",
                    String::from_utf8_lossy(&command.stderr)
                ));
            }
            let pam = std::fs::read(&output)
                .map_err(|error| format!("could not read djxl PAM: {error}"))?;
            parse_pam(&pam, format)
        })();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_dir(directory);
        Some(result)
    }

    fn parse_pam(bytes: &[u8], format: LosslessModularFormat) -> Result<Vec<u8>, String> {
        let marker = b"ENDHDR\n";
        let header_end = bytes
            .windows(marker.len())
            .position(|window| window == marker)
            .map(|position| position + marker.len())
            .ok_or_else(|| "djxl PAM is missing ENDHDR".to_string())?;
        let header = std::str::from_utf8(&bytes[..header_end])
            .map_err(|error| format!("djxl PAM header is not UTF-8: {error}"))?;
        if !header.starts_with("P7\n") {
            return Err("djxl did not emit a PAM P7 image".into());
        }
        let value = |key: &str| -> Result<usize, String> {
            header
                .lines()
                .find_map(|line| line.strip_prefix(key))
                .ok_or_else(|| format!("djxl PAM is missing {key}"))?
                .trim()
                .parse::<usize>()
                .map_err(|error| format!("invalid djxl PAM {key}: {error}"))
        };
        let width = value("WIDTH")?;
        let height = value("HEIGHT")?;
        let depth = value("DEPTH")?;
        let max_value = value("MAXVAL")?;
        let expected_depth = usize::try_from(format.channel_count())
            .map_err(|_| "PAM channel count overflow".to_string())?;
        if depth != expected_depth || max_value != 255 {
            return Err(format!(
                "djxl PAM has depth/maxval {depth}/{max_value}, expected {expected_depth}/255"
            ));
        }
        let pixels = bytes
            .get(header_end..)
            .ok_or_else(|| "djxl PAM pixels are truncated".to_string())?;
        let expected_bytes = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(depth))
            .ok_or_else(|| "djxl PAM dimensions overflow".to_string())?;
        if pixels.len() != expected_bytes {
            return Err(format!(
                "djxl PAM has {} bytes, expected {expected_bytes}",
                pixels.len()
            ));
        }
        Ok(pixels.to_vec())
    }

    fn parse_pgm(bytes: &[u8]) -> Result<Vec<u8>, String> {
        let mut cursor = 0usize;
        let mut token = || -> Result<&[u8], String> {
            loop {
                while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                    cursor += 1;
                }
                if bytes.get(cursor) == Some(&b'#') {
                    while bytes.get(cursor).is_some_and(|byte| *byte != b'\n') {
                        cursor += 1;
                    }
                    continue;
                }
                break;
            }
            let start = cursor;
            while bytes
                .get(cursor)
                .is_some_and(|byte| !byte.is_ascii_whitespace())
            {
                cursor += 1;
            }
            bytes
                .get(start..cursor)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| "truncated PGM header".into())
        };
        if token()? != b"P5" {
            return Err("djxl did not emit a binary grayscale PGM".into());
        }
        let width = std::str::from_utf8(token()?)
            .map_err(|error| error.to_string())?
            .parse::<usize>()
            .map_err(|error| error.to_string())?;
        let height = std::str::from_utf8(token()?)
            .map_err(|error| error.to_string())?
            .parse::<usize>()
            .map_err(|error| error.to_string())?;
        if token()? != b"255" {
            return Err("djxl PGM did not contain 8-bit samples".into());
        }
        while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        let pixels = bytes
            .get(cursor..)
            .ok_or_else(|| "truncated PGM pixels".to_string())?;
        if pixels.len() != width * height {
            return Err(format!(
                "djxl PGM has {} samples, expected {}",
                pixels.len(),
                width * height
            ));
        }
        Ok(pixels.to_vec())
    }

    #[test]
    fn gpu_groups_cover_safe_boundary_extents_and_decode_exactly() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU multi-group lossless encode test: no wgpu adapter");
            return;
        };
        let encoder = LosslessModularEncoder::new(context.clone());
        let extents = [
            (1, 1),
            (17, 13),
            (257, 1),
            (1, 257),
            (257, 257),
            (513, 3),
            (3, 513),
            (513, 513),
            (4_097, 1),
            (1, 4_097),
        ];
        for (width, height) in extents {
            let expected = packed_test_pixels(width, height);
            let source = packed_test_source(&context, width, height);
            let memory = encoder.memory_plan(&source).unwrap();
            assert_eq!(
                memory.parameter_storage_bytes,
                u64::from(memory.group_grid.groups) * std::mem::size_of::<ModularParams>() as u64
            );
            assert_eq!(
                memory.artifact_storage_bytes,
                expected_artifact_storage_bytes(width, height)
            );
            assert_eq!(memory.readback_bytes, memory.artifact_storage_bytes);
            assert_eq!(
                memory.owned_bytes_per_job,
                memory.parameter_storage_bytes + memory.artifact_storage_bytes * 2
            );
            assert_eq!(
                memory.addressed_bytes_per_job,
                memory.owned_bytes_per_job + memory.source_binding_bytes
            );

            let submission = encoder.submit(source.clone()).unwrap();
            assert_eq!(
                submission.ordered_groups().collect::<Vec<_>>(),
                memory.group_grid.ordered_groups().collect::<Vec<_>>()
            );
            let encoded = submission.wait().unwrap();
            let (size, decoded) = decode_gray8(&encoded)
                .unwrap_or_else(|error| panic!("Rust jxl rejected {width}x{height}: {error}"));
            assert_eq!(size, (width as usize, height as usize));
            assert_eq!(decoded, expected, "Rust jxl mismatch for {width}x{height}");
            let container = encoder.encode_container(source).unwrap();
            let parsed =
                jxl_gpu_bitstream::parse(&container, jxl_gpu_bitstream::ParseLimits::default())
                    .unwrap();
            assert_eq!(parsed.codestream(), encoded);
            let (_, container_decoded) = decode_gray8(&container).unwrap_or_else(|error| {
                panic!("Rust jxl rejected {width}x{height} container: {error}")
            });
            assert_eq!(container_decoded, expected);
            if let Some(decoded) = decode_with_djxl_if_available(&container) {
                assert_eq!(
                    decoded.unwrap_or_else(|error| {
                        panic!("djxl rejected {width}x{height} container: {error}")
                    }),
                    expected,
                    "djxl mismatch for {width}x{height}"
                );
            }
        }
    }

    #[test]
    fn packed_rgb8_and_rgba8_roundtrip_across_aspect_ratios_with_both_decoders() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU color encode test: no wgpu adapter");
            return;
        };
        let encoder = LosslessModularEncoder::new(context.clone());
        assert!(
            encoder
                .capabilities()
                .has_stage(KernelStage::ColorTransform)
        );
        assert!(
            encoder
                .capabilities()
                .has_stage(KernelStage::ModularTransform)
        );
        for format in [LosslessModularFormat::Rgb, LosslessModularFormat::Rgba] {
            for (width, height) in [(1, 513), (513, 1), (257, 3), (17, 13)] {
                let (source, expected) = packed_color_test_source(&context, width, height, format);
                let memory = encoder.memory_plan(&source).unwrap();
                assert_eq!(memory.format, format);
                assert_eq!(memory.channel_count, format.channel_count());
                assert_eq!(
                    memory.parameter_storage_bytes,
                    u64::from(memory.group_grid.groups)
                        * u64::from(format.channel_count())
                        * std::mem::size_of::<ModularParams>() as u64
                );
                let encoded = encoder.encode(source.clone()).unwrap();
                let (size, decoded) = decode_color8(&encoded, format).unwrap_or_else(|error| {
                    panic!("Rust decoder rejected {format:?} {width}x{height}: {error}")
                });
                assert_eq!(size, (width as usize, height as usize));
                assert_eq!(decoded, expected, "{format:?} {width}x{height}");
                if let Some(decoded) = decode_color_with_djxl_if_available(&encoded, format) {
                    assert_eq!(
                        decoded.unwrap_or_else(|error| {
                            panic!("djxl rejected {format:?} {width}x{height}: {error}")
                        }),
                        expected,
                        "{format:?} {width}x{height}"
                    );
                }
                if (width, height) == (17, 13) {
                    let async_encoded = pollster::block_on(encoder.submit(source).unwrap())
                        .expect("runtime-neutral color submission succeeds");
                    assert_eq!(async_encoded, encoded);
                }
            }
        }
    }

    #[test]
    fn packed_integer_depths_roundtrip_at_group_boundaries_with_both_decoders() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU integer encode test: no wgpu adapter");
            return;
        };
        let encoder = LosslessModularEncoder::new(context.clone());
        let formats = [
            LosslessModularFormat::Gray,
            LosslessModularFormat::Rgb,
            LosslessModularFormat::Rgba,
        ];
        let depths = 1u8..=16;
        let extents = [(1, 257), (255, 3), (256, 2), (257, 1)];
        for (format_index, format) in formats.into_iter().enumerate() {
            for (depth_index, bits_per_sample) in depths.clone().enumerate() {
                let (width, height) = extents[(format_index + depth_index) % extents.len()];
                let (source, expected) =
                    modular_integer_test_source(&context, width, height, format, bits_per_sample);
                let memory = encoder.memory_plan(&source).unwrap();
                assert_eq!(memory.format, format);
                assert_eq!(memory.bits_per_sample, bits_per_sample);
                assert_eq!(memory.bytes_per_sample, u8::from(bits_per_sample > 8) + 1);
                assert_eq!(memory.channel_count, format.channel_count());
                let submission = encoder.submit(source.clone()).unwrap();
                assert_eq!(submission.bits_per_sample(), bits_per_sample);
                let encoded = submission.wait().unwrap();
                if let Some(decoded) =
                    decode_integer_with_djxl_if_available(&encoded, format, bits_per_sample)
                {
                    assert_eq!(
                        decoded.unwrap_or_else(|error| {
                            panic!(
                                "djxl rejected {format:?} {bits_per_sample}-bit {width}x{height}: {error}"
                            )
                        }),
                        expected,
                        "djxl mismatch for {format:?} {bits_per_sample}-bit {width}x{height}"
                    );
                }
                let (size, decoded) = decode_integer(&encoded, format, bits_per_sample)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Rust jxl rejected {format:?} {bits_per_sample}-bit {width}x{height}: {error}"
                        )
                    });
                assert_eq!(size, (width as usize, height as usize));
                assert_eq!(
                    decoded, expected,
                    "Rust jxl mismatch for {format:?} {bits_per_sample}-bit {width}x{height}"
                );
                if format == LosslessModularFormat::Rgba && bits_per_sample == 12 {
                    let async_encoded = pollster::block_on(encoder.submit(source).unwrap())
                        .expect("runtime-neutral 12-bit RGBA submission succeeds");
                    assert_eq!(async_encoded, encoded);
                }
            }
        }
    }

    #[test]
    fn multi_group_container_and_runtime_neutral_future_are_deterministic() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU multi-group container test: no wgpu adapter");
            return;
        };
        let width = 257;
        let height = 257;
        let expected = packed_test_pixels(width, height);
        let source = packed_test_source(&context, width, height);
        let encoder = LosslessModularEncoder::new(context);
        let raw = encoder.encode(source.clone()).unwrap();
        let async_raw = pollster::block_on(encoder.submit(source.clone()).unwrap()).unwrap();
        assert_eq!(async_raw, raw);

        let container = encoder.encode_container(source.clone()).unwrap();
        let parsed =
            jxl_gpu_bitstream::parse(&container, jxl_gpu_bitstream::ParseLimits::default())
                .unwrap();
        assert_eq!(parsed.codestream(), raw);
        assert_eq!(
            parsed.boxes_of_type(ACCELERATION_INDEX_BOX_TYPE).count(),
            0,
            "the current private acceleration index is intentionally single-group"
        );
        let (size, decoded) = decode_gray8(&container).unwrap();
        assert_eq!(size, (width as usize, height as usize));
        assert_eq!(decoded, expected);
        if let Some(decoded) = decode_with_djxl_if_available(&container) {
            assert_eq!(decoded.unwrap(), expected);
        }
        assert_eq!(encoder.encode_container(source).unwrap(), container);
    }

    #[test]
    fn gpu_tokens_form_a_reference_decodable_lossless_codestream() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU lossless encode test: no wgpu adapter");
            return;
        };
        let width = 17u32;
        let height = 13u32;
        let row_stride = 20u64;
        let binding_alignment = u64::from(
            context
                .device()
                .limits()
                .min_storage_buffer_offset_alignment,
        )
        .max(4);
        let offset = binding_alignment + 4;
        let allocation_size = align_up(offset + row_stride * u64::from(height), 4)
            .expect("test allocation size is representable");
        let mut allocation = vec![0u8; allocation_size as usize];
        let mut expected = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let value = if y < 3 {
                    0
                } else {
                    ((x * 17 + y * 31 + (x * y) % 19) & 255) as u8
                };
                allocation[(offset + u64::from(y) * row_stride + u64::from(x)) as usize] = value;
                expected.push(value);
            }
        }
        let buffer = Arc::new(context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu gray8 test source"),
                contents: &allocation,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        let extent = Extent2d::new(width, height);
        let format = PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]);
        let layout = ImageLayout::from_planes(
            extent,
            format,
            vec![PitchLinearPlaneLayout {
                plane_index: 0,
                offset,
                row_stride,
                sample_extent: extent,
                row_bytes: u64::from(width),
            }],
        )
        .expect("test image layout is valid");
        let source = crate::BufferImageSource::new(buffer, layout).expect("test source is valid");
        let encoder = LosslessModularEncoder::new(context);
        let memory = encoder
            .memory_plan(&source)
            .expect("test source has a checked memory plan");
        let pixel_count = usize::try_from(width * height).expect("test dimensions fit usize");
        let expected_output_words = OUTPUT_HEADER_WORDS
            + event_capacity(pixel_count).expect("test event capacity") * EVENT_WORDS;
        let expected_output_bytes =
            u64::try_from(expected_output_words * 4).expect("test artifact size fits u64");
        assert_eq!(memory.group_grid.groups, 1);
        let parameter_bytes = std::mem::size_of::<ModularParams>() as u64;
        assert_eq!(memory.parameter_storage_bytes, parameter_bytes);
        assert_eq!(memory.artifact_storage_bytes, expected_output_bytes);
        assert_eq!(memory.readback_bytes, expected_output_bytes);
        assert_eq!(
            memory.owned_bytes_per_job,
            parameter_bytes + expected_output_bytes * 2
        );
        assert_eq!(
            memory.addressed_bytes_per_job,
            memory.owned_bytes_per_job + memory.source_binding_bytes
        );
        let in_flight = memory
            .for_in_flight(4)
            .expect("four-job memory total is representable");
        assert_eq!(in_flight.max_in_flight_jobs, 4);
        assert_eq!(in_flight.total_owned_bytes, memory.owned_bytes_per_job * 4);
        assert_eq!(
            in_flight.total_addressed_bytes,
            memory.addressed_bytes_per_job * 4
        );
        let limits = encoder.memory_limits();
        assert_eq!(
            limits.min_storage_buffer_offset_alignment.max(4),
            binding_alignment
        );
        let encoded = encoder
            .encode(source.clone())
            .expect("GPU lossless encode succeeds");
        let (size, decoded) = decode_gray8(&encoded).expect("jxl reference decoder accepts output");
        assert_eq!(size, (width as usize, height as usize));
        assert_eq!(decoded, expected);
        if let Some(decoded) = decode_with_djxl_if_available(&encoded) {
            assert_eq!(decoded.expect("djxl accepts GPU codestream"), expected);
        }
        let submission = encoder
            .submit(source.clone())
            .expect("runtime-neutral Future submission succeeds");
        let async_encoded =
            pollster::block_on(submission).expect("runtime-neutral Future encode succeeds");
        assert_eq!(async_encoded, encoded);

        let container = encoder
            .encode_container(source.clone())
            .expect("GPU lossless container encode succeeds");
        let parsed =
            jxl_gpu_bitstream::parse(&container, jxl_gpu_bitstream::ParseLimits::default())
                .expect("container is structurally valid");
        assert_eq!(parsed.codestream(), encoded);
        let boxes = parsed
            .boxes_of_type(ACCELERATION_INDEX_BOX_TYPE)
            .collect::<Vec<_>>();
        assert_eq!(boxes.len(), 1);
        let index = Gray8AccelerationIndex::parse_bound(boxes[0].payload, parsed.codestream())
            .expect("jwgp index is bound to the exact codestream");
        assert_eq!(index.width(), width);
        assert_eq!(index.height(), height);
        assert_eq!(index.sample_count(), width * height);
        let (_, decoded) =
            decode_gray8(&container).expect("jxl reference decoder ignores the private box");
        assert_eq!(decoded, expected);
        if let Some(decoded) = decode_with_djxl_if_available(&container) {
            assert_eq!(
                decoded.expect("djxl ignores jwgp and decodes jxlc"),
                expected
            );
        }
        let second = encoder
            .encode_container(source)
            .expect("second deterministic container encode succeeds");
        assert_eq!(container, second);
        assert_eq!(container, checked_in_gpu_gray8_lossless());
        if let Some(path) = std::env::var_os("JXL_WGPU_WRITE_FIXTURE") {
            std::fs::write(path, &container).expect("requested fixture path is writable");
        }
    }
}
