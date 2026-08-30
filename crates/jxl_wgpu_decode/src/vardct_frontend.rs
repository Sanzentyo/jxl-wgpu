//! Strict standard-codestream boundary for the first GPU VarDCT frontend.
//!
//! This module deliberately stops at physical section packets. It never invokes a CPU entropy,
//! coefficient, dequantization, color, or pixel decoder. The accepted profile is narrow enough
//! that every accepted entropy stream can be handed to the GPU implementation without a hidden
//! fallback.

use jxl_bitstream::{Bitstream as MetadataBitstream, U};
use jxl_gpu_bitstream::{
    BitRange, CodestreamInventory, ColourEncodingInventory, ColourSpaceInventory, FrameBlendMode,
    FrameEncoding, FrameInventory, FrameSectionKind, FrameType, SampleBitDepth,
};
use thiserror::Error;

const MAX_CODESTREAM_BYTES: u64 = 1 << 28;
const MAX_DIMENSION: u32 = 1 << 18;
const MAX_PIXELS: u64 = 1 << 32;
const MAX_GROUPS: u64 = 1 << 16;

/// Capability represented by [`StandardVarDctProfile`].
///
/// This is intentionally not named a full-color decode capability. It covers GPU entropy
/// reconstruction, dequantization, chroma correlation, and inverse VarDCT into XYB planes. Color
/// conversion, restoration, and output packing have separate capability checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VarDctFrontendCapability {
    SinglePassXybEntropyPackets,
}

/// Feature that prevents a codestream from entering the initial GPU VarDCT path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnsupportedVarDctFeature {
    CodestreamSize,
    ImageDimensions,
    FloatingPointSamples,
    NonXybImage,
    GrayscaleImage,
    EmbeddedIcc,
    ExtraChannels,
    Preview,
    Animation,
    MultipleFrames,
    NonRegularFrame,
    ModularFrame,
    FrameFeatures,
    Ycbcr,
    JpegSubsampling,
    Upsampling,
    Cropping,
    Blending,
    FrameReferences,
    ProgressivePasses,
    PermutedToc,
    SectionLayout,
}

/// Typed failure from profile negotiation or section packet construction.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum VarDctFrontendError {
    #[error("standard GPU VarDCT profile does not support {feature:?}")]
    Unsupported { feature: UnsupportedVarDctFeature },
    #[error("frame section bit range overflows the codestream address space")]
    SectionRangeOverflow,
    #[error("frame section {range:?} exceeds the {codestream_bits}-bit contiguous codestream")]
    SectionOutsideCodestream {
        range: BitRange,
        codestream_bits: u64,
    },
    #[error("duplicate {kind} section for logical index {index}")]
    DuplicateSection { kind: &'static str, index: u64 },
    #[error("missing {kind} section for logical index {index}")]
    MissingSection { kind: &'static str, index: u64 },
    #[error("pass-group section uses pass {pass_index}; the negotiated profile requires pass 0")]
    UnexpectedPass { pass_index: u32 },
    #[error("section group index {index} exceeds the declared {group_count} groups")]
    GroupIndexOutOfRange { index: u64, group_count: u64 },
}

/// Failure while parsing bounded, non-entropy LF-global metadata.
#[derive(Debug, Error)]
pub enum VarDctPacketError {
    #[error("failed to parse {stage} at the VarDCT metadata boundary")]
    MetadataBitstream {
        stage: &'static str,
        #[source]
        source: jxl_bitstream::Error,
    },
    #[error("failed to parse the HF block-context map: {0}")]
    BlockContextMap(#[from] jxl_coding::Error),
    #[error("the initial GPU VarDCT profile requires default {field}")]
    NonDefaultMetadata { field: &'static str },
    #[error("VarDCT metadata cursor {cursor} exceeds packet end {packet_end}")]
    PacketBoundary { cursor: u64, packet_end: u64 },
    #[error("VarDCT packet bit range overflows")]
    PacketRangeOverflow,
    #[error("HF block-context metadata exceeds the bounded frontend profile: {field}")]
    BlockContextLimit { field: &'static str },
    #[error("HF preset {preset_count} exceeds the frame's {group_count} groups")]
    HfPresetCount { preset_count: u32, group_count: u32 },
    #[error("VarDCT group geometry overflows the bounded GPU address space")]
    GeometryOverflow,
    #[error("{field} value {value} exceeds the portable WGSL u32 address space")]
    GpuAddressSpace { field: &'static str, value: u64 },
}

/// Entropy context metadata used by the HF coefficient GPU decoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfBlockContextIr {
    pub qf_thresholds: Vec<u32>,
    pub lf_thresholds: [Vec<i32>; 3],
    pub block_context_map: Vec<u8>,
    pub num_block_clusters: u32,
}

/// Scalar LF-global metadata preceding the entropy-coded MA tree.
///
/// `ma_tree_bit_offset` points at the first bit of the MA-tree entropy descriptor. No tree or
/// sample symbol has been expanded while constructing this value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LfGlobalPrefix {
    pub global_scale: u32,
    pub quant_lf: u32,
    pub hf_block_context: HfBlockContextIr,
    pub ma_tree_bit_offset: u64,
}

/// Fixed HF-global metadata preceding an optional entropy-coded coefficient-order permutation.
///
/// If `used_orders` is zero, `order_entropy_bit_offset` is also the beginning of the HF
/// coefficient entropy descriptor. Otherwise it points to the order decoder descriptor; its
/// permutation symbols and resulting cursor are consumed on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HfGlobalPrefix {
    pub num_hf_presets: u32,
    pub used_orders: u16,
    pub order_entropy_bit_offset: u64,
}

impl HfGlobalPrefix {
    /// Parse the default-matrix flag, preset count, and order mask without expanding an entropy
    /// symbol. Custom dequant matrices are rejected until their Modular raw-matrix path is GPU
    /// lowered.
    pub fn parse(
        codestream: &[u8],
        packet: BitRange,
        group_count: u32,
    ) -> Result<Self, VarDctPacketError> {
        let packet_end = packet.end().ok_or(VarDctPacketError::PacketRangeOverflow)?;
        validate_packet_end(codestream, packet_end)?;
        let packet_offset =
            usize::try_from(packet.offset).map_err(|_| VarDctPacketError::PacketRangeOverflow)?;
        let mut reader = MetadataBitstream::new(codestream);
        metadata_at(&mut reader, "HF-global packet offset", |reader| {
            reader.skip_bits(packet_offset)
        })?;
        metadata_require_default(&mut reader, "HF dequantization matrices")?;

        let preset_bits = if group_count <= 1 {
            0
        } else {
            group_count.next_power_of_two().trailing_zeros() as usize
        };
        let num_hf_presets = metadata_at(&mut reader, "HF preset count", |reader| {
            reader.read_bits(preset_bits).map(|value| value + 1)
        })?;
        if num_hf_presets > group_count {
            return Err(VarDctPacketError::HfPresetCount {
                preset_count: num_hf_presets,
                group_count,
            });
        }
        let used_orders = metadata_at(&mut reader, "HF coefficient-order mask", |reader| {
            reader.read_u32(0x5f, 0x13, 0, U(13))
        })?;
        let used_orders =
            u16::try_from(used_orders).map_err(|_| VarDctPacketError::PacketRangeOverflow)?;
        let cursor = u64::try_from(reader.num_read_bits())
            .map_err(|_| VarDctPacketError::PacketRangeOverflow)?;
        if cursor > packet_end {
            return Err(VarDctPacketError::PacketBoundary { cursor, packet_end });
        }
        Ok(Self {
            num_hf_presets,
            used_orders,
            order_entropy_bit_offset: cursor,
        })
    }
}

/// One channel reconstructed by a standard Modular subimage on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModularChannelPlan {
    pub width: u32,
    pub height: u32,
}

/// A bounded Modular residual stream whose samples remain encoded in the codestream buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModularStreamPlan {
    pub token_bit_offset: u64,
    pub token_bit_end: u64,
    pub stream_index: u32,
    pub channels: Vec<ModularChannelPlan>,
}

impl ModularStreamPlan {
    #[must_use]
    pub fn sample_count(&self) -> Option<u64> {
        self.channels.iter().try_fold(0u64, |total, channel| {
            total.checked_add(u64::from(channel.width).checked_mul(u64::from(channel.height))?)
        })
    }
}

/// Fixed LF-group prefix before the GPU-decoded quantized LF samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LfGroupPrefix {
    pub extra_precision: u8,
    pub lf_quant: ModularStreamPlan,
}

impl LfGroupPrefix {
    pub fn parse(
        codestream: &[u8],
        packet: BitRange,
        lf_width: u32,
        lf_height: u32,
        stream_index: u32,
    ) -> Result<Self, VarDctPacketError> {
        let packet_end = packet.end().ok_or(VarDctPacketError::PacketRangeOverflow)?;
        validate_packet_end(codestream, packet_end)?;
        let packet_offset =
            usize::try_from(packet.offset).map_err(|_| VarDctPacketError::PacketRangeOverflow)?;
        let mut reader = MetadataBitstream::new(codestream);
        metadata_at(&mut reader, "LF-group packet offset", |reader| {
            reader.skip_bits(packet_offset)
        })?;
        let extra_precision = metadata_at(&mut reader, "LF extra precision", |reader| {
            reader.read_bits(2)
        })?;
        let extra_precision =
            u8::try_from(extra_precision).map_err(|_| VarDctPacketError::GeometryOverflow)?;
        parse_default_modular_header(&mut reader, "LF quantization")?;
        let token_bit_offset = metadata_cursor(&reader)?;
        if token_bit_offset >= packet_end {
            return Err(VarDctPacketError::PacketBoundary {
                cursor: token_bit_offset,
                packet_end,
            });
        }
        let width = lf_width.div_ceil(8);
        let height = lf_height.div_ceil(8);
        let channels = vec![ModularChannelPlan { width, height }; 3];
        let lf_quant = ModularStreamPlan {
            token_bit_offset,
            token_bit_end: packet_end,
            stream_index,
            channels,
        };
        validate_modular_stream(&lf_quant)?;
        Ok(Self {
            extra_precision,
            lf_quant,
        })
    }
}

/// Fixed HF-metadata prefix following GPU reconstruction of the LF quantized image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfMetadataPrefix {
    pub block_width: u32,
    pub block_height: u32,
    pub block_count: u32,
    pub metadata: ModularStreamPlan,
}

impl HfMetadataPrefix {
    pub fn parse(
        codestream: &[u8],
        token_bit_offset: u64,
        packet_end: u64,
        lf_width: u32,
        lf_height: u32,
        stream_index: u32,
    ) -> Result<Self, VarDctPacketError> {
        validate_packet_end(codestream, packet_end)?;
        if token_bit_offset >= packet_end {
            return Err(VarDctPacketError::PacketBoundary {
                cursor: token_bit_offset,
                packet_end,
            });
        }
        let offset = usize::try_from(token_bit_offset)
            .map_err(|_| VarDctPacketError::PacketRangeOverflow)?;
        let mut reader = MetadataBitstream::new(codestream);
        metadata_at(&mut reader, "HF metadata packet offset", |reader| {
            reader.skip_bits(offset)
        })?;
        let block_width = lf_width.div_ceil(8);
        let block_height = lf_height.div_ceil(8);
        let block_area = block_width
            .checked_mul(block_height)
            .ok_or(VarDctPacketError::GeometryOverflow)?;
        if block_area == 0 {
            return Err(VarDctPacketError::GeometryOverflow);
        }
        let count_bits = block_area
            .checked_next_power_of_two()
            .ok_or(VarDctPacketError::GeometryOverflow)?
            .trailing_zeros();
        let count_bits =
            usize::try_from(count_bits).map_err(|_| VarDctPacketError::GeometryOverflow)?;
        let block_count = metadata_at(&mut reader, "HF varblock count", |reader| {
            reader.read_bits(count_bits).map(|value| value + 1)
        })?;
        if block_count > block_area {
            return Err(VarDctPacketError::GeometryOverflow);
        }
        parse_default_modular_header(&mut reader, "HF metadata")?;
        let metadata_token_offset = metadata_cursor(&reader)?;
        if metadata_token_offset >= packet_end {
            return Err(VarDctPacketError::PacketBoundary {
                cursor: metadata_token_offset,
                packet_end,
            });
        }
        let correlation_width = lf_width.div_ceil(64);
        let correlation_height = lf_height.div_ceil(64);
        let channels = vec![
            ModularChannelPlan {
                width: correlation_width,
                height: correlation_height,
            },
            ModularChannelPlan {
                width: correlation_width,
                height: correlation_height,
            },
            ModularChannelPlan {
                width: block_count,
                height: 2,
            },
            ModularChannelPlan {
                width: block_width,
                height: block_height,
            },
        ];
        let metadata = ModularStreamPlan {
            token_bit_offset: metadata_token_offset,
            token_bit_end: packet_end,
            stream_index,
            channels,
        };
        validate_modular_stream(&metadata)?;
        Ok(Self {
            block_width,
            block_height,
            block_count,
            metadata,
        })
    }
}

fn parse_default_modular_header(
    reader: &mut MetadataBitstream<'_>,
    stage: &'static str,
) -> Result<(), VarDctPacketError> {
    let use_global_tree = metadata_at(reader, stage, |reader| reader.read_bool())?;
    if !use_global_tree {
        return Err(VarDctPacketError::NonDefaultMetadata {
            field: "local Modular MA tree",
        });
    }
    metadata_require_default(reader, "local Modular weighted predictor")?;
    let transform_count = metadata_at(reader, stage, |reader| {
        reader.read_u32(0, 1, 2 + U(4), 18 + U(8))
    })?;
    if transform_count != 0 {
        return Err(VarDctPacketError::NonDefaultMetadata {
            field: "local Modular transforms",
        });
    }
    Ok(())
}

fn metadata_cursor(reader: &MetadataBitstream<'_>) -> Result<u64, VarDctPacketError> {
    u64::try_from(reader.num_read_bits()).map_err(|_| VarDctPacketError::PacketRangeOverflow)
}

fn validate_packet_end(codestream: &[u8], packet_end: u64) -> Result<(), VarDctPacketError> {
    let codestream_end = u64::try_from(codestream.len())
        .ok()
        .and_then(|bytes| bytes.checked_mul(8))
        .ok_or(VarDctPacketError::PacketRangeOverflow)?;
    if packet_end > codestream_end {
        return Err(VarDctPacketError::PacketBoundary {
            cursor: packet_end,
            packet_end: codestream_end,
        });
    }
    validate_gpu_u32("packet bit end", packet_end)
}

fn validate_modular_stream(stream: &ModularStreamPlan) -> Result<(), VarDctPacketError> {
    validate_gpu_u32("Modular token bit offset", stream.token_bit_offset)?;
    validate_gpu_u32("Modular token bit end", stream.token_bit_end)?;
    let sample_count = stream
        .sample_count()
        .ok_or(VarDctPacketError::GeometryOverflow)?;
    validate_gpu_u32("Modular sample count", sample_count)
}

fn validate_gpu_u32(field: &'static str, value: u64) -> Result<(), VarDctPacketError> {
    u32::try_from(value)
        .map(|_| ())
        .map_err(|_| VarDctPacketError::GpuAddressSpace { field, value })
}

impl LfGlobalPrefix {
    /// Parse only the fixed/default dequant, block-context, and correlation headers.
    ///
    /// The initial profile deliberately requires the standard LF dequantization, HF block
    /// context, and LF correlation defaults. Their numerical values are supplied by the GPU
    /// implementation, not materialized by a CPU dequantization path.
    pub fn parse(codestream: &[u8], packet: BitRange) -> Result<Self, VarDctPacketError> {
        let packet_end = packet.end().ok_or(VarDctPacketError::PacketRangeOverflow)?;
        validate_packet_end(codestream, packet_end)?;
        let packet_offset =
            usize::try_from(packet.offset).map_err(|_| VarDctPacketError::PacketRangeOverflow)?;
        let mut reader = MetadataBitstream::new(codestream);
        metadata_at(&mut reader, "LF-global packet offset", |reader| {
            reader.skip_bits(packet_offset)
        })?;
        metadata_require_default(&mut reader, "LF channel dequantization")?;
        let global_scale = metadata_at(&mut reader, "global quantizer scale", |reader| {
            reader.read_u32(1 + U(11), 2049 + U(11), 4097 + U(12), 8193 + U(16))
        })?;
        let quant_lf = metadata_at(&mut reader, "LF quantizer", |reader| {
            reader.read_u32(16, 1 + U(5), 1 + U(8), 1 + U(16))
        })?;
        let hf_block_context = parse_hf_block_context(&mut reader)?;
        metadata_require_default(&mut reader, "LF channel correlation")?;
        let has_global_ma_tree = metadata_at(&mut reader, "global MA-tree flag", |reader| {
            reader.read_bool()
        })?;
        if !has_global_ma_tree {
            return Err(VarDctPacketError::NonDefaultMetadata {
                field: "global MA tree presence",
            });
        }
        let cursor = u64::try_from(reader.num_read_bits())
            .map_err(|_| VarDctPacketError::PacketRangeOverflow)?;
        if cursor > packet_end {
            return Err(VarDctPacketError::PacketBoundary { cursor, packet_end });
        }
        Ok(Self {
            global_scale,
            quant_lf,
            hf_block_context,
            ma_tree_bit_offset: cursor,
        })
    }
}

fn parse_hf_block_context(
    reader: &mut MetadataBitstream<'_>,
) -> Result<HfBlockContextIr, VarDctPacketError> {
    if metadata_at(reader, "HF block-context default flag", |reader| {
        reader.read_bool()
    })? {
        return Ok(HfBlockContextIr {
            qf_thresholds: Vec::new(),
            lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
            block_context_map: vec![
                0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14,
                14, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
            ],
            num_block_clusters: 15,
        });
    }

    let mut lf_thresholds = [Vec::new(), Vec::new(), Vec::new()];
    let mut block_context_count = 1u32;
    for thresholds in &mut lf_thresholds {
        let count = metadata_at(reader, "LF block-context threshold count", |reader| {
            reader.read_bits(4)
        })?;
        block_context_count = block_context_count.checked_mul(count + 1).ok_or(
            VarDctPacketError::BlockContextLimit {
                field: "context count",
            },
        )?;
        thresholds.try_reserve_exact(count as usize).map_err(|_| {
            VarDctPacketError::BlockContextLimit {
                field: "LF thresholds",
            }
        })?;
        for _ in 0..count {
            let packed = metadata_at(reader, "LF block-context threshold", |reader| {
                reader.read_u32(U(4), 16 + U(8), 272 + U(16), 65_808 + U(32))
            })?;
            thresholds.push(jxl_bitstream::unpack_signed(packed));
        }
    }

    let qf_count = metadata_at(reader, "quant-field threshold count", |reader| {
        reader.read_bits(4)
    })?;
    block_context_count = block_context_count.checked_mul(qf_count + 1).ok_or(
        VarDctPacketError::BlockContextLimit {
            field: "context count",
        },
    )?;
    // Section 7.2 bounds this product to 64. Keeping the standard bound here prevents an
    // attacker-controlled context-map allocation from escaping the negotiated profile.
    if block_context_count > 64 {
        return Err(VarDctPacketError::BlockContextLimit {
            field: "more than 64 block contexts",
        });
    }
    let mut qf_thresholds = Vec::new();
    qf_thresholds
        .try_reserve_exact(qf_count as usize)
        .map_err(|_| VarDctPacketError::BlockContextLimit {
            field: "quant-field thresholds",
        })?;
    for _ in 0..qf_count {
        let threshold = metadata_at(reader, "quant-field threshold", |reader| {
            reader.read_u32(U(2), 4 + U(3), 12 + U(5), 44 + U(8))
        })?;
        qf_thresholds.push(threshold + 1);
    }

    let distribution_count =
        block_context_count
            .checked_mul(39)
            .ok_or(VarDctPacketError::BlockContextLimit {
                field: "distribution count",
            })?;
    let (num_block_clusters, block_context_map) =
        jxl_coding::read_clusters(reader, distribution_count)?;
    if num_block_clusters > 16 {
        return Err(VarDctPacketError::BlockContextLimit {
            field: "more than 16 HF block clusters",
        });
    }
    Ok(HfBlockContextIr {
        qf_thresholds,
        lf_thresholds,
        block_context_map,
        num_block_clusters,
    })
}

/// Exact packet layout that the GPU entropy frontend consumes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VarDctSectionLayout {
    /// JPEG XL's one-entry TOC. Logical stream boundaries are discovered by GPU decoders and
    /// carried forward as checked bit cursors; the host does not entropy-decode the entry.
    Single { packet: BitRange },
    /// Independently addressable LF-global, LF-group, HF-global, and pass-group packets.
    Sections {
        lf_global: BitRange,
        lf_groups: Vec<BitRange>,
        hf_global: BitRange,
        pass_groups: Vec<BitRange>,
    },
}

/// Validated first-stage VarDCT profile. Construction is the capability check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandardVarDctProfile {
    pub capability: VarDctFrontendCapability,
    pub width: u32,
    pub height: u32,
    pub bits_per_sample: u32,
    pub group_dimension: u32,
    pub group_count: u64,
    pub low_frequency_group_count: u64,
    /// Whether section F.2's adaptive smoothing pass is required after LF dequantization and
    /// chroma-from-luma reconstruction.
    pub adaptive_lf_smoothing: bool,
    pub sections: VarDctSectionLayout,
}

/// Pixel-space rectangle owned by one LF or pass group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctGroupRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl StandardVarDctProfile {
    /// Negotiate the strict single-frame/single-pass XYB VarDCT transform profile.
    pub fn negotiate(inventory: &CodestreamInventory) -> Result<Self, VarDctFrontendError> {
        validate_image(inventory)?;
        let frame = validate_frame(inventory)?;
        let sections = collect_sections(inventory, frame)?;
        let bits_per_sample = match inventory.image_header.bit_depth {
            SampleBitDepth::Integer { bits_per_sample } => bits_per_sample,
            SampleBitDepth::Float { .. } => {
                return unsupported(UnsupportedVarDctFeature::FloatingPointSamples);
            }
        };
        Ok(Self {
            capability: VarDctFrontendCapability::SinglePassXybEntropyPackets,
            width: frame.width,
            height: frame.height,
            bits_per_sample,
            group_dimension: 128u32 << frame.group_size_shift,
            group_count: frame.group_count,
            low_frequency_group_count: frame.low_frequency_group_count,
            adaptive_lf_smoothing: frame.flags & 0x80 == 0,
            sections,
        })
    }

    /// Meta-adaptive property-1 stream index for an LF quantization subimage.
    pub fn lf_quant_stream_index(&self, index: u64) -> Result<u32, VarDctFrontendError> {
        checked_stream_index(index, self.low_frequency_group_count, 1, 0)
    }

    /// Meta-adaptive property-1 stream index for an LF group's HF metadata subimage.
    pub fn hf_metadata_stream_index(&self, index: u64) -> Result<u32, VarDctFrontendError> {
        checked_stream_index(index, self.low_frequency_group_count, 1, 2)
    }

    pub fn low_frequency_group_rect(
        &self,
        index: u64,
    ) -> Result<VarDctGroupRect, VarDctFrontendError> {
        let dimension =
            self.group_dimension
                .checked_mul(8)
                .ok_or(VarDctFrontendError::Unsupported {
                    feature: UnsupportedVarDctFeature::ImageDimensions,
                })?;
        group_rect(
            self.width,
            self.height,
            dimension,
            index,
            self.low_frequency_group_count,
        )
    }

    pub fn pass_group_rect(&self, index: u64) -> Result<VarDctGroupRect, VarDctFrontendError> {
        group_rect(
            self.width,
            self.height,
            self.group_dimension,
            index,
            self.group_count,
        )
    }
}

fn group_rect(
    width: u32,
    height: u32,
    dimension: u32,
    index: u64,
    group_count: u64,
) -> Result<VarDctGroupRect, VarDctFrontendError> {
    if index >= group_count || dimension == 0 || width == 0 || height == 0 {
        return Err(VarDctFrontendError::GroupIndexOutOfRange { index, group_count });
    }
    let columns = width.div_ceil(dimension);
    let index = u32::try_from(index)
        .map_err(|_| VarDctFrontendError::GroupIndexOutOfRange { index, group_count })?;
    let column = index % columns;
    let row = index / columns;
    let x = column
        .checked_mul(dimension)
        .ok_or(VarDctFrontendError::Unsupported {
            feature: UnsupportedVarDctFeature::ImageDimensions,
        })?;
    let y = row
        .checked_mul(dimension)
        .ok_or(VarDctFrontendError::Unsupported {
            feature: UnsupportedVarDctFeature::ImageDimensions,
        })?;
    Ok(VarDctGroupRect {
        x,
        y,
        width: width.saturating_sub(x).min(dimension),
        height: height.saturating_sub(y).min(dimension),
    })
}

fn checked_stream_index(
    index: u64,
    group_count: u64,
    base: u64,
    group_count_multiplier: u64,
) -> Result<u32, VarDctFrontendError> {
    if index >= group_count {
        return Err(VarDctFrontendError::GroupIndexOutOfRange { index, group_count });
    }
    let stream_index = group_count
        .checked_mul(group_count_multiplier)
        .and_then(|offset| base.checked_add(offset))
        .and_then(|offset| offset.checked_add(index))
        .ok_or(VarDctFrontendError::Unsupported {
            feature: UnsupportedVarDctFeature::ImageDimensions,
        })?;
    u32::try_from(stream_index).map_err(|_| VarDctFrontendError::Unsupported {
        feature: UnsupportedVarDctFeature::ImageDimensions,
    })
}

fn validate_image(inventory: &CodestreamInventory) -> Result<(), VarDctFrontendError> {
    let image = &inventory.image_header;
    if inventory.codestream_bytes > MAX_CODESTREAM_BYTES {
        return unsupported(UnsupportedVarDctFeature::CodestreamSize);
    }
    let area = u64::from(image.width)
        .checked_mul(u64::from(image.height))
        .ok_or(VarDctFrontendError::Unsupported {
            feature: UnsupportedVarDctFeature::ImageDimensions,
        })?;
    if image.width == 0
        || image.height == 0
        || image.width > MAX_DIMENSION
        || image.height > MAX_DIMENSION
        || area > MAX_PIXELS
    {
        return unsupported(UnsupportedVarDctFeature::ImageDimensions);
    }
    if !image.xyb_encoded {
        return unsupported(UnsupportedVarDctFeature::NonXybImage);
    }
    if image.grayscale {
        return unsupported(UnsupportedVarDctFeature::GrayscaleImage);
    }
    if !matches!(
        image.colour_encoding,
        ColourEncodingInventory::Enumerated {
            colour_space: ColourSpaceInventory::Rgb,
            ..
        }
    ) {
        return unsupported(UnsupportedVarDctFeature::EmbeddedIcc);
    }
    if image.embedded_icc.is_some() {
        return unsupported(UnsupportedVarDctFeature::EmbeddedIcc);
    }
    if image.extra_channel_count != 0 || !image.extra_channels.is_empty() {
        return unsupported(UnsupportedVarDctFeature::ExtraChannels);
    }
    if image.preview_size.is_some() {
        return unsupported(UnsupportedVarDctFeature::Preview);
    }
    if image.animation.is_some() {
        return unsupported(UnsupportedVarDctFeature::Animation);
    }
    if inventory.frames.len() != 1 {
        return unsupported(UnsupportedVarDctFeature::MultipleFrames);
    }
    Ok(())
}

fn validate_frame(inventory: &CodestreamInventory) -> Result<&FrameInventory, VarDctFrontendError> {
    let frame = &inventory.frames[0];
    if frame.frame_type != FrameType::Regular || frame.is_preview || !frame.is_last {
        return unsupported(UnsupportedVarDctFeature::NonRegularFrame);
    }
    if frame.encoding != FrameEncoding::VarDct {
        return unsupported(UnsupportedVarDctFeature::ModularFrame);
    }
    // Noise, patches, splines, LF-frame reuse, and unknown frame extensions remain outside the
    // initial transform capability. Skip-adaptive-LF-smoothing is a supported rendering choice.
    const SUPPORTED_FLAGS: u64 = 0x80;
    if frame.flags & !SUPPORTED_FLAGS != 0 {
        return unsupported(UnsupportedVarDctFeature::FrameFeatures);
    }
    if frame.do_ycbcr {
        return unsupported(UnsupportedVarDctFeature::Ycbcr);
    }
    if frame.jpeg_upsampling != [0; 3] {
        return unsupported(UnsupportedVarDctFeature::JpegSubsampling);
    }
    if frame.upsampling != 1
        || frame
            .extra_channel_upsampling
            .iter()
            .any(|&value| value != 1)
    {
        return unsupported(UnsupportedVarDctFeature::Upsampling);
    }
    if frame.have_crop
        || frame.x0 != 0
        || frame.y0 != 0
        || frame.width != inventory.image_header.width
        || frame.height != inventory.image_header.height
    {
        return unsupported(UnsupportedVarDctFeature::Cropping);
    }
    if frame.color_blend.mode != FrameBlendMode::Replace
        || frame.color_blend.source != 0
        || frame.color_blend.alpha_channel.is_some()
        || frame.color_blend.clamp
        || !frame.extra_channel_blends.is_empty()
    {
        return unsupported(UnsupportedVarDctFeature::Blending);
    }
    if frame.save_as_reference != 0 || frame.save_before_color_transform {
        return unsupported(UnsupportedVarDctFeature::FrameReferences);
    }
    if frame.num_passes != 1
        || !frame.progressive_passes.shifts.is_empty()
        || !frame.progressive_passes.downsampling.is_empty()
        || !frame.progressive_passes.last_pass.is_empty()
    {
        return unsupported(UnsupportedVarDctFeature::ProgressivePasses);
    }
    if frame.toc_permuted {
        return unsupported(UnsupportedVarDctFeature::PermutedToc);
    }
    if frame.group_count == 0
        || frame.low_frequency_group_count == 0
        || frame.group_count > MAX_GROUPS
        || frame.low_frequency_group_count > MAX_GROUPS
    {
        return unsupported(UnsupportedVarDctFeature::ImageDimensions);
    }
    Ok(frame)
}

fn collect_sections(
    inventory: &CodestreamInventory,
    frame: &FrameInventory,
) -> Result<VarDctSectionLayout, VarDctFrontendError> {
    let codestream_bits = inventory
        .codestream_bytes
        .checked_mul(8)
        .ok_or(VarDctFrontendError::SectionRangeOverflow)?;
    for section in &frame.sections {
        validate_range(section.bits, codestream_bits)?;
    }
    if let [section] = frame.sections.as_slice() {
        if section.kind != FrameSectionKind::Single {
            return unsupported(UnsupportedVarDctFeature::SectionLayout);
        }
        return Ok(VarDctSectionLayout::Single {
            packet: section.bits,
        });
    }
    if frame
        .sections
        .iter()
        .any(|section| section.kind == FrameSectionKind::Single)
    {
        return unsupported(UnsupportedVarDctFeature::SectionLayout);
    }

    let mut lf_global = None;
    let mut hf_global = None;
    let mut lf_groups = vec![None; host_count(frame.low_frequency_group_count)?];
    let mut pass_groups = vec![None; host_count(frame.group_count)?];
    for section in &frame.sections {
        match section.kind {
            FrameSectionKind::LowFrequencyGlobal => {
                assign_once(&mut lf_global, section.bits, "LF-global", 0)?;
            }
            FrameSectionKind::HighFrequencyGlobal => {
                assign_once(&mut hf_global, section.bits, "HF-global", 0)?;
            }
            FrameSectionKind::LowFrequencyGroup { group_index } => {
                let slot =
                    group_slot(&mut lf_groups, group_index, frame.low_frequency_group_count)?;
                assign_once(slot, section.bits, "LF-group", group_index)?;
            }
            FrameSectionKind::PassGroup {
                pass_index,
                group_index,
            } => {
                if pass_index != 0 {
                    return Err(VarDctFrontendError::UnexpectedPass { pass_index });
                }
                let slot = group_slot(&mut pass_groups, group_index, frame.group_count)?;
                assign_once(slot, section.bits, "pass-group", group_index)?;
            }
            FrameSectionKind::Single => unreachable!("single entries rejected above"),
        }
    }
    Ok(VarDctSectionLayout::Sections {
        lf_global: lf_global.ok_or(VarDctFrontendError::MissingSection {
            kind: "LF-global",
            index: 0,
        })?,
        lf_groups: collect_required(lf_groups, "LF-group")?,
        hf_global: hf_global.ok_or(VarDctFrontendError::MissingSection {
            kind: "HF-global",
            index: 0,
        })?,
        pass_groups: collect_required(pass_groups, "pass-group")?,
    })
}

fn validate_range(range: BitRange, codestream_bits: u64) -> Result<(), VarDctFrontendError> {
    if range.end().is_none() {
        return Err(VarDctFrontendError::SectionRangeOverflow);
    }
    if range.end().is_some_and(|end| end > codestream_bits) {
        return Err(VarDctFrontendError::SectionOutsideCodestream {
            range,
            codestream_bits,
        });
    }
    Ok(())
}

fn host_count(count: u64) -> Result<usize, VarDctFrontendError> {
    usize::try_from(count).map_err(|_| VarDctFrontendError::Unsupported {
        feature: UnsupportedVarDctFeature::ImageDimensions,
    })
}

fn group_slot<T>(
    groups: &mut [T],
    index: u64,
    group_count: u64,
) -> Result<&mut T, VarDctFrontendError> {
    let index_usize = usize::try_from(index)
        .map_err(|_| VarDctFrontendError::GroupIndexOutOfRange { index, group_count })?;
    groups
        .get_mut(index_usize)
        .ok_or(VarDctFrontendError::GroupIndexOutOfRange { index, group_count })
}

fn assign_once(
    slot: &mut Option<BitRange>,
    range: BitRange,
    kind: &'static str,
    index: u64,
) -> Result<(), VarDctFrontendError> {
    if slot.replace(range).is_some() {
        return Err(VarDctFrontendError::DuplicateSection { kind, index });
    }
    Ok(())
}

fn collect_required(
    groups: Vec<Option<BitRange>>,
    kind: &'static str,
) -> Result<Vec<BitRange>, VarDctFrontendError> {
    groups
        .into_iter()
        .enumerate()
        .map(|(index, range)| {
            range.ok_or(VarDctFrontendError::MissingSection {
                kind,
                index: index as u64,
            })
        })
        .collect()
}

fn unsupported<T>(feature: UnsupportedVarDctFeature) -> Result<T, VarDctFrontendError> {
    Err(VarDctFrontendError::Unsupported { feature })
}

fn metadata_require_default(
    reader: &mut MetadataBitstream<'_>,
    field: &'static str,
) -> Result<(), VarDctPacketError> {
    let is_default = metadata_at(reader, field, |reader| reader.read_bool())?;
    if !is_default {
        return Err(VarDctPacketError::NonDefaultMetadata { field });
    }
    Ok(())
}

fn metadata_at<T>(
    reader: &mut MetadataBitstream<'_>,
    stage: &'static str,
    operation: impl FnOnce(&mut MetadataBitstream<'_>) -> Result<T, jxl_bitstream::Error>,
) -> Result<T, VarDctPacketError> {
    operation(reader).map_err(|source| VarDctPacketError::MetadataBitstream { stage, source })
}
