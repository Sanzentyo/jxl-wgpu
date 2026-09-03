//! Strict standard-codestream boundary for the first GPU VarDCT frontend.
//!
//! This module deliberately stops at physical section packets. It never invokes a CPU entropy,
//! coefficient, dequantization, color, or pixel decoder. The accepted profile is narrow enough
//! that every accepted entropy stream can be handed to the GPU implementation without a hidden
//! fallback.

use jxl_gpu_bitstream::{
    BitRange, BitReader, CodestreamInventory, ColourEncodingInventory, ColourSpaceInventory,
    FrameBlendMode, FrameEncoding, FrameInventory, FrameSectionKind, FrameType, SampleBitDepth,
};
use thiserror::Error;

use crate::modular_tree::{BitInput, MaTreeLimits, read_clusters};

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
    SinglePassEntropyPackets,
}

/// Effective horizontal and vertical JPEG component subsampling relative to the largest
/// component grid. JPEG XL permits only zero or one bit of shift on each axis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VarDctChannelShift {
    pub horizontal: u32,
    pub vertical: u32,
}

impl VarDctChannelShift {
    #[must_use]
    pub const fn is_subsampled(self) -> bool {
        self.horizontal != 0 || self.vertical != 0
    }

    pub(crate) fn shifted_extent(self, width: u32, height: u32) -> Option<[u32; 2]> {
        let horizontal = 1u32.checked_shl(self.horizontal)?;
        let vertical = 1u32.checked_shl(self.vertical)?;
        Some([width.div_ceil(horizontal), height.div_ceil(vertical)])
    }
}

/// Color-domain contract carried by resident VarDCT planes before output conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VarDctColorTransform {
    Xyb,
    Ycbcr,
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
    SubsampledAdaptiveLf,
    Upsampling,
    Cropping,
    Blending,
    FrameReferences,
    ProgressivePasses,
    SectionLayout,
}

/// Execution plan for Adaptive LF smoothing resolved during frontend negotiation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AdaptiveLfPlan {
    /// Adaptive LF smoothing is skipped (not signaled, or disabled by progressive DC).
    Skip,
    /// Adaptive LF smoothing executes on packed 4:4:4 XYB or YCbCr planes.
    ExecutePacked444,
    /// Adaptive LF smoothing executes on subsampled chroma channels with aligned sampling.
    #[allow(dead_code)]
    ExecuteSubsampled {
        channel_shifts: [VarDctChannelShift; 3],
    },
}

impl AdaptiveLfPlan {
    #[must_use]
    pub const fn executes(self) -> bool {
        matches!(self, Self::ExecutePacked444 | Self::ExecuteSubsampled { .. })
    }
}

/// Typed failure from profile negotiation or section packet construction.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum VarDctFrontendError {
    #[error("standard GPU VarDCT profile does not support {feature:?}")]
    Unsupported { feature: UnsupportedVarDctFeature },
    #[error("the JPEG XL VarDCT frame name is not valid UTF-8")]
    InvalidFrameName,
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
    #[error("failed to parse {stage} at the VarDCT span-reader boundary")]
    MetadataReader {
        stage: &'static str,
        #[source]
        source: VarDctMetadataReaderError,
    },
    #[error("failed to parse the HF block-context map: {0}")]
    BlockContextMap(#[from] jxl_coding::Error),
    #[error("the initial GPU VarDCT profile requires default {field}")]
    NonDefaultMetadata { field: &'static str },
    #[error("LF dequantization multiplier for {channel} is too small: {value}")]
    LfDequantizationTooSmall { channel: &'static str, value: f32 },
    #[error("base channel correlation for {channel} is outside [-4, 4]: {value}")]
    BaseCorrelationOutOfRange { channel: &'static str, value: f32 },
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

/// Non-recursive errors that can cross the span-backed metadata reader boundary.
#[derive(Debug, Error)]
pub enum VarDctMetadataReaderError {
    #[error("codestream bit reader failed: {0}")]
    Bitstream(#[from] jxl_gpu_bitstream::Error),
    #[error("Modular metadata parser failed: {0}")]
    ModularTree(#[from] crate::ModularTreeError),
    #[error("codestream metadata reader failed: {0}")]
    Other(&'static str),
}

/// Entropy context metadata used by the HF coefficient GPU decoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfBlockContextIr {
    pub qf_thresholds: Vec<u32>,
    pub lf_thresholds: [Vec<i32>; 3],
    pub block_context_map: Vec<u8>,
    pub num_block_clusters: u32,
}

/// Per-channel LF dequantization multipliers in JPEG XL X/Y/B order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LfChannelDequantization {
    pub multipliers: [f32; 3],
}

impl Default for LfChannelDequantization {
    fn default() -> Self {
        Self {
            multipliers: [1.0 / 32.0, 1.0 / 4.0, 1.0 / 2.0],
        }
    }
}

/// Global chroma-from-luma parameters shared by LF and HF reconstruction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LfChannelCorrelation {
    pub colour_factor: u32,
    pub base: [f32; 2],
    pub lf_factors: [i32; 2],
}

impl LfChannelCorrelation {
    /// Returns the final LF Y-to-X and Y-to-B correlation slopes.
    #[must_use]
    pub fn lf_slopes(self) -> [f32; 2] {
        let inverse_colour_factor = 1.0 / self.colour_factor as f32;
        [
            self.base[0] + self.lf_factors[0] as f32 * inverse_colour_factor,
            self.base[1] + self.lf_factors[1] as f32 * inverse_colour_factor,
        ]
    }

    /// Returns base X/B correlation and reciprocal color factor for HF map lowering.
    #[must_use]
    pub fn hf_params(self) -> [f32; 3] {
        [self.base[0], self.base[1], 1.0 / self.colour_factor as f32]
    }
}

impl Default for LfChannelCorrelation {
    fn default() -> Self {
        Self {
            colour_factor: 84,
            base: [0.0, 1.0],
            lf_factors: [0, 0],
        }
    }
}

/// Scalar LF-global metadata preceding an optional entropy-coded global MA tree.
///
/// `global_ma_tree_bit_offset` is `Some` only when the LF-global packet declares a global tree.
/// No tree or sample symbol has been expanded while constructing this value.
#[derive(Clone, Debug, PartialEq)]
pub struct LfGlobalPrefix {
    pub lf_dequantization: LfChannelDequantization,
    pub global_scale: u32,
    pub quant_lf: u32,
    pub hf_block_context: HfBlockContextIr,
    pub lf_correlation: LfChannelCorrelation,
    /// The bit after the global-tree presence flag. This is either the global tree descriptor or,
    /// when no global tree exists, the following LF-group payload.
    pub suffix_bit_offset: u64,
    pub global_ma_tree_bit_offset: Option<u64>,
}

/// Parsed JPEG XL Modular group header and the bit immediately following it.
///
/// When `use_global_tree` is false, `tree_or_token_bit_offset` points at the local MA-tree
/// descriptor. Otherwise it points directly at the first image-entropy descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularHeaderPrefix {
    pub use_global_tree: bool,
    pub tree_or_token_bit_offset: u64,
}

/// LF-group scalar prefix before its MA tree or quantized-LF entropy descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LfGroupHeaderPrefix {
    pub extra_precision: u8,
    pub modular: ModularHeaderPrefix,
}

/// HF-metadata scalar prefix before its MA tree or image-entropy descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HfMetadataHeaderPrefix {
    pub block_width: u32,
    pub block_height: u32,
    pub block_count: u32,
    pub modular: ModularHeaderPrefix,
}

/// Fixed HF-global metadata preceding an optional entropy-coded coefficient-order permutation.
///
/// If `used_orders` is zero, `order_entropy_bit_offset` is also the beginning of the HF
/// coefficient entropy descriptor. Otherwise it points to the order decoder descriptor; the
/// bounded frontend expands this small metadata permutation before preserving coefficient symbols
/// for the GPU pass-group executor.
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
        let mut reader = BitReader::new(codestream);
        reader
            .skip_bits(packet.offset)
            .map_err(|error| metadata_error("HF-global packet offset", error.into()))?;
        Self::parse_reader(&mut reader, packet_end, group_count)
    }

    /// Parses this prefix from a reader positioned at the packet's absolute bit offset.
    pub(crate) fn parse_reader(
        reader: &mut impl BitInput,
        packet_end: u64,
        group_count: u32,
    ) -> Result<Self, VarDctPacketError> {
        let mut reader = BoundedBitInput::new(reader, packet_end);
        metadata_require_default(&mut reader, "HF dequantization matrices")?;
        Self::parse_after_dequant_reader(&mut reader, packet_end, group_count)
    }

    /// Parses the suffix after a caller has consumed the complete dequant-matrix set.
    pub(crate) fn parse_after_dequant_reader(
        reader: &mut impl BitInput,
        packet_end: u64,
        group_count: u32,
    ) -> Result<Self, VarDctPacketError> {
        let mut reader = BoundedBitInput::new(reader, packet_end);

        let preset_bits = if group_count <= 1 {
            0
        } else {
            group_count.next_power_of_two().trailing_zeros() as usize
        };
        let preset_bits =
            u8::try_from(preset_bits).map_err(|_| VarDctPacketError::PacketRangeOverflow)?;
        let num_hf_presets =
            u32::try_from(metadata_bits(&mut reader, "HF preset count", preset_bits)?)
                .map_err(|_| VarDctPacketError::PacketRangeOverflow)?
                + 1;
        if num_hf_presets > group_count {
            return Err(VarDctPacketError::HfPresetCount {
                preset_count: num_hf_presets,
                group_count,
            });
        }
        let used_orders = metadata_u32(
            &mut reader,
            "HF coefficient-order mask",
            [
                MetadataU32Part::constant(0x5f),
                MetadataU32Part::constant(0x13),
                MetadataU32Part::constant(0),
                MetadataU32Part::bits(0, 13),
            ],
        )?;
        let used_orders =
            u16::try_from(used_orders).map_err(|_| VarDctPacketError::PacketRangeOverflow)?;
        let cursor = metadata_cursor(&reader)?;
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
        let mut reader = BitReader::new(codestream);
        reader
            .skip_bits(packet.offset)
            .map_err(|error| metadata_error("LF-group packet offset", error.into()))?;
        Self::parse_reader(&mut reader, packet_end, lf_width, lf_height, stream_index)
    }

    /// Parses this prefix from a reader positioned at the packet's absolute bit offset.
    pub(crate) fn parse_reader(
        reader: &mut impl BitInput,
        packet_end: u64,
        lf_width: u32,
        lf_height: u32,
        stream_index: u32,
    ) -> Result<Self, VarDctPacketError> {
        let mut reader = BoundedBitInput::new(reader, packet_end);
        let extra_precision = metadata_bits(&mut reader, "LF extra precision", 2)?;
        let extra_precision =
            u8::try_from(extra_precision).map_err(|_| VarDctPacketError::GeometryOverflow)?;
        let modular = parse_modular_header(&mut reader, "LF quantization")?;
        if !modular.use_global_tree {
            return Err(VarDctPacketError::NonDefaultMetadata {
                field: "local Modular MA tree",
            });
        }
        let token_bit_offset = modular.tree_or_token_bit_offset;
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
        let mut reader = BitReader::new(codestream);
        reader
            .skip_bits(token_bit_offset)
            .map_err(|error| metadata_error("HF metadata packet offset", error.into()))?;
        Self::parse_reader(&mut reader, packet_end, lf_width, lf_height, stream_index)
    }

    /// Parses this prefix from a reader positioned at the supplied token bit offset.
    pub(crate) fn parse_reader(
        reader: &mut impl BitInput,
        packet_end: u64,
        lf_width: u32,
        lf_height: u32,
        stream_index: u32,
    ) -> Result<Self, VarDctPacketError> {
        let mut reader = BoundedBitInput::new(reader, packet_end);
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
            u8::try_from(count_bits).map_err(|_| VarDctPacketError::GeometryOverflow)?;
        let block_count =
            u32::try_from(metadata_bits(&mut reader, "HF varblock count", count_bits)?)
                .map_err(|_| VarDctPacketError::PacketRangeOverflow)?
                + 1;
        if block_count > block_area {
            return Err(VarDctPacketError::GeometryOverflow);
        }
        let modular = parse_modular_header(&mut reader, "HF metadata")?;
        if !modular.use_global_tree {
            return Err(VarDctPacketError::NonDefaultMetadata {
                field: "local Modular MA tree",
            });
        }
        let metadata_token_offset = modular.tree_or_token_bit_offset;
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

fn parse_modular_header(
    reader: &mut impl BitInput,
    stage: &'static str,
) -> Result<ModularHeaderPrefix, VarDctPacketError> {
    let use_global_tree = metadata_bool(reader, stage)?;
    metadata_require_default(reader, "local Modular weighted predictor")?;
    let transform_count = metadata_u32(
        reader,
        stage,
        [
            MetadataU32Part::constant(0),
            MetadataU32Part::constant(1),
            MetadataU32Part::bits(2, 4),
            MetadataU32Part::bits(18, 8),
        ],
    )?;
    if transform_count != 0 {
        return Err(VarDctPacketError::NonDefaultMetadata {
            field: "local Modular transforms",
        });
    }
    Ok(ModularHeaderPrefix {
        use_global_tree,
        tree_or_token_bit_offset: metadata_cursor(reader)?,
    })
}

pub(crate) fn parse_lf_group_header_reader(
    reader: &mut impl BitInput,
    packet_end: u64,
) -> Result<LfGroupHeaderPrefix, VarDctPacketError> {
    let mut reader = BoundedBitInput::new(reader, packet_end);
    let extra_precision = metadata_bits(&mut reader, "LF extra precision", 2)?;
    let extra_precision =
        u8::try_from(extra_precision).map_err(|_| VarDctPacketError::GeometryOverflow)?;
    let modular = parse_modular_header(&mut reader, "LF quantization")?;
    if modular.tree_or_token_bit_offset >= packet_end {
        return Err(VarDctPacketError::PacketBoundary {
            cursor: modular.tree_or_token_bit_offset,
            packet_end,
        });
    }
    Ok(LfGroupHeaderPrefix {
        extra_precision,
        modular,
    })
}

pub(crate) fn parse_hf_metadata_header_reader(
    reader: &mut impl BitInput,
    packet_end: u64,
    lf_width: u32,
    lf_height: u32,
) -> Result<HfMetadataHeaderPrefix, VarDctPacketError> {
    let mut reader = BoundedBitInput::new(reader, packet_end);
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
    let count_bits = u8::try_from(count_bits).map_err(|_| VarDctPacketError::GeometryOverflow)?;
    let block_count = u32::try_from(metadata_bits(&mut reader, "HF varblock count", count_bits)?)
        .map_err(|_| VarDctPacketError::PacketRangeOverflow)?
        + 1;
    if block_count > block_area {
        return Err(VarDctPacketError::GeometryOverflow);
    }
    let modular = parse_modular_header(&mut reader, "HF metadata")?;
    if modular.tree_or_token_bit_offset >= packet_end {
        return Err(VarDctPacketError::PacketBoundary {
            cursor: modular.tree_or_token_bit_offset,
            packet_end,
        });
    }
    Ok(HfMetadataHeaderPrefix {
        block_width,
        block_height,
        block_count,
        modular,
    })
}

fn metadata_cursor(reader: &impl BitInput) -> Result<u64, VarDctPacketError> {
    Ok(reader.bit_offset())
}

fn validate_packet_end(codestream: &[u8], packet_end: u64) -> Result<(), VarDctPacketError> {
    let codestream_end = u64::try_from(codestream.len())
        .ok()
        .and_then(|bytes| bytes.checked_mul(8))
        .ok_or(VarDctPacketError::PacketRangeOverflow)?;
    validate_packet_end_bits(codestream_end, packet_end)
}

pub(crate) fn validate_packet_end_bits(
    codestream_bits: u64,
    packet_end: u64,
) -> Result<(), VarDctPacketError> {
    if packet_end > codestream_bits {
        return Err(VarDctPacketError::PacketBoundary {
            cursor: packet_end,
            packet_end: codestream_bits,
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
    /// Parses LF dequantization, block-context, and channel-correlation headers without expanding
    /// image entropy on the host.
    pub fn parse(codestream: &[u8], packet: BitRange) -> Result<Self, VarDctPacketError> {
        let packet_end = packet.end().ok_or(VarDctPacketError::PacketRangeOverflow)?;
        validate_packet_end(codestream, packet_end)?;
        let mut reader = BitReader::new(codestream);
        reader
            .skip_bits(packet.offset)
            .map_err(|error| metadata_error("LF-global packet offset", error.into()))?;
        Self::parse_reader(&mut reader, packet_end)
    }

    /// Parses this prefix from a reader positioned at the packet's absolute bit offset.
    pub(crate) fn parse_reader(
        reader: &mut impl BitInput,
        packet_end: u64,
    ) -> Result<Self, VarDctPacketError> {
        let mut reader = BoundedBitInput::new(reader, packet_end);
        let lf_dequantization = parse_lf_channel_dequantization(&mut reader)?;
        let global_scale = metadata_u32(
            &mut reader,
            "global quantizer scale",
            [
                MetadataU32Part::bits(1, 11),
                MetadataU32Part::bits(2049, 11),
                MetadataU32Part::bits(4097, 12),
                MetadataU32Part::bits(8193, 16),
            ],
        )?;
        let quant_lf = metadata_u32(
            &mut reader,
            "LF quantizer",
            [
                MetadataU32Part::constant(16),
                MetadataU32Part::bits(1, 5),
                MetadataU32Part::bits(1, 8),
                MetadataU32Part::bits(1, 16),
            ],
        )?;
        let hf_block_context = parse_hf_block_context(&mut reader)?;
        let lf_correlation = parse_lf_channel_correlation(&mut reader)?;
        let has_global_ma_tree = metadata_bool(&mut reader, "global MA-tree flag")?;
        let cursor = metadata_cursor(&reader)?;
        if cursor > packet_end {
            return Err(VarDctPacketError::PacketBoundary { cursor, packet_end });
        }
        Ok(Self {
            lf_dequantization,
            global_scale,
            quant_lf,
            hf_block_context,
            lf_correlation,
            suffix_bit_offset: cursor,
            global_ma_tree_bit_offset: has_global_ma_tree.then_some(cursor),
        })
    }
}

fn parse_lf_channel_dequantization(
    reader: &mut impl BitInput,
) -> Result<LfChannelDequantization, VarDctPacketError> {
    if metadata_bool(reader, "LF channel dequantization default flag")? {
        return Ok(LfChannelDequantization::default());
    }

    let mut multipliers = [0.0; 3];
    for (channel, multiplier) in ["X", "Y", "B"].into_iter().zip(&mut multipliers) {
        *multiplier = metadata_f16(reader, "LF channel dequantization multiplier")?;
        if *multiplier / 128.0 < 1.0e-8 {
            return Err(VarDctPacketError::LfDequantizationTooSmall {
                channel,
                value: *multiplier,
            });
        }
    }
    Ok(LfChannelDequantization { multipliers })
}

fn parse_lf_channel_correlation(
    reader: &mut impl BitInput,
) -> Result<LfChannelCorrelation, VarDctPacketError> {
    if metadata_bool(reader, "LF channel correlation default flag")? {
        return Ok(LfChannelCorrelation::default());
    }

    let colour_factor = metadata_u32(
        reader,
        "channel correlation colour factor",
        [
            MetadataU32Part::constant(84),
            MetadataU32Part::constant(256),
            MetadataU32Part::bits(2, 8),
            MetadataU32Part::bits(258, 16),
        ],
    )?;
    let mut base = [0.0; 2];
    for (channel, correlation) in ["X", "B"].into_iter().zip(&mut base) {
        *correlation = metadata_f16(reader, "base channel correlation")?;
        if correlation.abs() > 4.0 {
            return Err(VarDctPacketError::BaseCorrelationOutOfRange {
                channel,
                value: *correlation,
            });
        }
    }
    let mut lf_factors = [0; 2];
    for factor in &mut lf_factors {
        *factor = metadata_bits(reader, "LF channel correlation factor", 8)? as i32 - 128;
    }
    Ok(LfChannelCorrelation {
        colour_factor,
        base,
        lf_factors,
    })
}

fn parse_hf_block_context(
    reader: &mut impl BitInput,
) -> Result<HfBlockContextIr, VarDctPacketError> {
    if metadata_bool(reader, "HF block-context default flag")? {
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
        let count = metadata_bits(reader, "LF block-context threshold count", 4)? as u32;
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
            let packed = metadata_u32(
                reader,
                "LF block-context threshold",
                [
                    MetadataU32Part::bits(0, 4),
                    MetadataU32Part::bits(16, 8),
                    MetadataU32Part::bits(272, 16),
                    MetadataU32Part::bits(65_808, 32),
                ],
            )?;
            thresholds.push(unpack_signed(packed));
        }
    }

    let qf_count = metadata_bits(reader, "quant-field threshold count", 4)? as u32;
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
        let threshold = metadata_u32(
            reader,
            "quant-field threshold",
            [
                MetadataU32Part::bits(0, 2),
                MetadataU32Part::bits(4, 3),
                MetadataU32Part::bits(12, 5),
                MetadataU32Part::bits(44, 8),
            ],
        )?;
        qf_thresholds.push(threshold + 1);
    }

    let distribution_count =
        block_context_count
            .checked_mul(39)
            .ok_or(VarDctPacketError::BlockContextLimit {
                field: "distribution count",
            })?;
    let block_context_map = read_clusters(
        reader,
        usize::try_from(distribution_count).map_err(|_| VarDctPacketError::BlockContextLimit {
            field: "distribution count",
        })?,
        MaTreeLimits::default(),
    )
    .map_err(|source| metadata_error("HF block-context map", source))?;
    let num_block_clusters = block_context_map
        .iter()
        .copied()
        .max()
        .map_or(0, |maximum| u32::from(maximum) + 1);
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
    /// Group vectors are normalized to logical group order; each range still addresses the
    /// section's original physical location in the codestream.
    Sections {
        lf_global: BitRange,
        lf_groups: Vec<BitRange>,
        hf_global: BitRange,
        pass_groups: Vec<BitRange>,
    },
}

/// Precomputed, validated geometry and stream metadata for one LF group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidatedLfGroup {
    pub index: u32,
    pub section: BitRange,
    pub rect: VarDctGroupRect,
    pub padded_block_extent: [u32; 2],
    pub lf_stream_index: u32,
    pub hf_stream_index: u32,
}

/// Precomputed section location for one pass group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidatedPassGroup {
    pub group_index: u32,
    pub pass_index: u32,
    pub section: BitRange,
}

/// Validated first-stage VarDCT profile. Construction is the capability check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandardVarDctProfile {
    capability: VarDctFrontendCapability,
    /// Encoded color-sample width before frame upsampling.
    width: u32,
    /// Encoded color-sample height before frame upsampling.
    height: u32,
    /// Final color-frame width after frame upsampling.
    presentation_width: u32,
    /// Final color-frame height after frame upsampling.
    presentation_height: u32,
    /// Frame upsampling factor applied after restoration and before color conversion.
    upsampling: u32,
    bits_per_sample: u32,
    color_transform: VarDctColorTransform,
    /// Resident channel order is Cb/X, Y, Cr/B. XYB uses three zero shifts.
    channel_shifts: [VarDctChannelShift; 3],
    /// Raw JPEG component sampling selectors in Cb/X, Y, Cr/B order.
    jpeg_upsampling: [u32; 3],
    /// Axis padding required by the JPEG component sampling factors before channel shifts.
    jpeg_block_alignment: [u32; 2],
    group_dimension: u32,
    group_count: u64,
    low_frequency_group_count: u64,
    /// Resolved Adaptive LF smoothing execution plan.
    adaptive_lf: AdaptiveLfPlan,
    /// Earlier progressive-DC frame supplies the LF image instead of this frame's LF entropy.
    uses_lf_frame: bool,
    /// Progressive-DC level represented by this frame; zero is the final image level.
    lf_level: u32,
    /// Validated UTF-8 frame name preserved in authoritative [`crate::FrameMetadata`].
    frame_name: String,
    sections: VarDctSectionLayout,
    lf_groups: Box<[ValidatedLfGroup]>,
    pass_groups: Box<[ValidatedPassGroup]>,
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
    /// Negotiate the strict single-frame/single-pass XYB or JPEG-reconstruction VarDCT profile.
    pub fn negotiate(inventory: &CodestreamInventory) -> Result<Self, VarDctFrontendError> {
        Self::negotiate_for_role(inventory, VarDctFrameRole::Presentation)
    }

    pub(crate) fn negotiate_progressive_dc(
        inventory: &CodestreamInventory,
        is_final: bool,
    ) -> Result<Self, VarDctFrontendError> {
        Self::negotiate_for_role(
            inventory,
            if is_final {
                VarDctFrameRole::ProgressiveDcFinal
            } else {
                VarDctFrameRole::ProgressiveDcRefinement
            },
        )
    }

    fn negotiate_for_role(
        inventory: &CodestreamInventory,
        role: VarDctFrameRole,
    ) -> Result<Self, VarDctFrontendError> {
        validate_image(inventory)?;
        let frame = validate_frame(inventory, role)?;
        let sections = collect_sections(inventory, frame)?;
        let frame_name = String::from_utf8(frame.name_bytes.clone())
            .map_err(|_| VarDctFrontendError::InvalidFrameName)?;
        let bits_per_sample = match inventory.image_header.bit_depth {
            SampleBitDepth::Integer { bits_per_sample } => bits_per_sample,
            SampleBitDepth::Float { .. } => {
                return unsupported(UnsupportedVarDctFeature::FloatingPointSamples);
            }
        };
        let (width, height) =
            frame
                .color_sample_extent()
                .ok_or(VarDctFrontendError::Unsupported {
                    feature: UnsupportedVarDctFeature::ImageDimensions,
                })?;
        let is_presentation = role != VarDctFrameRole::ProgressiveDcRefinement;
        let (presentation_width, presentation_height, upsampling) = if is_presentation {
            (frame.width, frame.height, frame.upsampling)
        } else {
            (width, height, 1)
        };
        let group_dimension = 128u32 << frame.group_size_shift;
        let jpeg_shifts = jpeg_channel_shifts(frame.jpeg_upsampling);
        let jpeg_alignment = jpeg_block_alignment(frame.jpeg_upsampling);
        let adaptive_lf = if frame.flags & 0x80 == 0 && !frame.uses_lf_frame() {
            AdaptiveLfPlan::ExecutePacked444
        } else {
            AdaptiveLfPlan::Skip
        };
        let uses_lf_frame = frame.uses_lf_frame();
        let lf_level = frame.lf_level;
        let low_frequency_group_count = frame.low_frequency_group_count;
        let group_count = frame.group_count;
        let color_transform = if frame.do_ycbcr {
            VarDctColorTransform::Ycbcr
        } else {
            VarDctColorTransform::Xyb
        };

        let lf_dimension = group_dimension
            .checked_mul(8)
            .ok_or(VarDctFrontendError::Unsupported {
                feature: UnsupportedVarDctFeature::ImageDimensions,
            })?;
        let lf_group_sections: &[BitRange] = match &sections {
            VarDctSectionLayout::Single { packet } => std::slice::from_ref(packet),
            VarDctSectionLayout::Sections { lf_groups, .. } => lf_groups.as_slice(),
        };
        let mut validated_lf_groups = Vec::with_capacity(lf_group_sections.len());
        for (index_usize, &section) in lf_group_sections.iter().enumerate() {
            let index = u32::try_from(index_usize).map_err(|_| VarDctFrontendError::Unsupported {
                feature: UnsupportedVarDctFeature::ImageDimensions,
            })?;
            let rect = group_rect(
                width,
                height,
                lf_dimension,
                u64::from(index),
                low_frequency_group_count,
            )?;
            let padded_block_extent = [
                align_up_power_of_two(rect.width.div_ceil(8), jpeg_alignment[0])?,
                align_up_power_of_two(rect.height.div_ceil(8), jpeg_alignment[1])?,
            ];
            let lf_stream_index = checked_stream_index(
                u64::from(index),
                low_frequency_group_count,
                1,
                0,
            )?;
            let hf_stream_index = checked_stream_index(
                u64::from(index),
                low_frequency_group_count,
                1,
                2,
            )?;
            validated_lf_groups.push(ValidatedLfGroup {
                index,
                section,
                rect,
                padded_block_extent,
                lf_stream_index,
                hf_stream_index,
            });
        }
        let lf_groups: Box<[ValidatedLfGroup]> = validated_lf_groups.into_boxed_slice();

        let pass_group_sections: &[BitRange] = match &sections {
            VarDctSectionLayout::Single { .. } => &[][..],
            VarDctSectionLayout::Sections { pass_groups, .. } => pass_groups.as_slice(),
        };
        let mut validated_pass_groups = Vec::with_capacity(pass_group_sections.len());
        for (group_index_usize, &section) in pass_group_sections.iter().enumerate() {
            let group_index = u32::try_from(group_index_usize).map_err(|_| {
                VarDctFrontendError::Unsupported {
                    feature: UnsupportedVarDctFeature::ImageDimensions,
                }
            })?;
            validated_pass_groups.push(ValidatedPassGroup {
                group_index,
                pass_index: 0,
                section,
            });
        }
        let pass_groups: Box<[ValidatedPassGroup]> = validated_pass_groups.into_boxed_slice();

        Ok(Self {
            capability: VarDctFrontendCapability::SinglePassEntropyPackets,
            width,
            height,
            presentation_width,
            presentation_height,
            upsampling,
            bits_per_sample,
            color_transform,
            channel_shifts: jpeg_shifts,
            jpeg_upsampling: frame.jpeg_upsampling,
            jpeg_block_alignment: jpeg_alignment,
            group_dimension,
            group_count,
            low_frequency_group_count,
            adaptive_lf,
            uses_lf_frame,
            lf_level,
            frame_name,
            sections,
            lf_groups,
            pass_groups,
        })
    }

    #[must_use]
    pub const fn capability(&self) -> VarDctFrontendCapability {
        self.capability
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub const fn presentation_width(&self) -> u32 {
        self.presentation_width
    }

    #[must_use]
    pub const fn presentation_height(&self) -> u32 {
        self.presentation_height
    }

    #[must_use]
    pub const fn upsampling(&self) -> u32 {
        self.upsampling
    }

    #[must_use]
    pub const fn bits_per_sample(&self) -> u32 {
        self.bits_per_sample
    }

    #[must_use]
    pub const fn color_transform(&self) -> VarDctColorTransform {
        self.color_transform
    }

    #[must_use]
    pub const fn channel_shifts(&self) -> [VarDctChannelShift; 3] {
        self.channel_shifts
    }

    #[must_use]
    pub fn is_subsampled(&self) -> bool {
        self.channel_shifts != [VarDctChannelShift::default(); 3]
    }

    #[must_use]
    pub const fn jpeg_upsampling(&self) -> [u32; 3] {
        self.jpeg_upsampling
    }

    #[must_use]
    pub const fn jpeg_block_alignment(&self) -> [u32; 2] {
        self.jpeg_block_alignment
    }

    #[must_use]
    pub const fn group_dimension(&self) -> u32 {
        self.group_dimension
    }

    #[must_use]
    pub const fn group_count(&self) -> u64 {
        self.group_count
    }

    #[must_use]
    pub const fn low_frequency_group_count(&self) -> u64 {
        self.low_frequency_group_count
    }

    #[must_use]
    pub const fn uses_lf_frame(&self) -> bool {
        self.uses_lf_frame
    }

    #[must_use]
    pub const fn lf_level(&self) -> u32 {
        self.lf_level
    }

    #[must_use]
    pub fn frame_name(&self) -> &str {
        &self.frame_name
    }

    #[must_use]
    pub fn sections(&self) -> &VarDctSectionLayout {
        &self.sections
    }

    #[must_use]
    pub const fn adaptive_lf_smoothing(&self) -> bool {
        self.adaptive_lf.executes()
    }

    #[must_use]
    pub(crate) const fn adaptive_lf(&self) -> AdaptiveLfPlan {
        self.adaptive_lf
    }

    #[must_use]
    pub fn lf_groups(&self) -> &[ValidatedLfGroup] {
        &self.lf_groups
    }

    #[must_use]
    pub fn pass_groups(&self) -> &[ValidatedPassGroup] {
        &self.pass_groups
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

    pub fn low_frequency_group_index_for_pass_group(
        &self,
        index: u64,
    ) -> Result<u32, VarDctFrontendError> {
        let pass = self.pass_group_rect(index)?;
        let lf_dimension =
            self.group_dimension
                .checked_mul(8)
                .ok_or(VarDctFrontendError::Unsupported {
                    feature: UnsupportedVarDctFeature::ImageDimensions,
                })?;
        let lf_groups_per_row = self.width.div_ceil(lf_dimension);
        let lf_index = (pass.y / lf_dimension)
            .checked_mul(lf_groups_per_row)
            .and_then(|row| row.checked_add(pass.x / lf_dimension))
            .ok_or(VarDctFrontendError::Unsupported {
                feature: UnsupportedVarDctFeature::ImageDimensions,
            })?;
        if u64::from(lf_index) >= self.low_frequency_group_count {
            return Err(VarDctFrontendError::GroupIndexOutOfRange {
                index: u64::from(lf_index),
                group_count: self.low_frequency_group_count,
            });
        }
        Ok(lf_index)
    }

    pub(crate) fn padded_group_block_extent(
        &self,
        rect: VarDctGroupRect,
    ) -> Result<[u32; 2], VarDctFrontendError> {
        let width = align_up_power_of_two(rect.width.div_ceil(8), self.jpeg_block_alignment[0])?;
        let height = align_up_power_of_two(rect.height.div_ceil(8), self.jpeg_block_alignment[1])?;
        Ok([width, height])
    }

    pub(crate) fn channel_block_extent(
        &self,
        rect: VarDctGroupRect,
        channel: usize,
    ) -> Result<[u32; 2], VarDctFrontendError> {
        let [width, height] = self.padded_group_block_extent(rect)?;
        self.channel_shifts
            .get(channel)
            .and_then(|shift| shift.shifted_extent(width, height))
            .ok_or(VarDctFrontendError::Unsupported {
                feature: UnsupportedVarDctFeature::ImageDimensions,
            })
    }

    pub(crate) fn lf_entropy_channel_block_extents(
        &self,
        rect: VarDctGroupRect,
    ) -> Result<[[u32; 2]; 3], VarDctFrontendError> {
        Ok([
            self.channel_block_extent(rect, 1)?,
            self.channel_block_extent(rect, 0)?,
            self.channel_block_extent(rect, 2)?,
        ])
    }

    pub(crate) fn uses_chroma_from_luma(&self) -> bool {
        self.jpeg_upsampling == [0; 3]
    }
}

fn align_up_power_of_two(value: u32, shift: u32) -> Result<u32, VarDctFrontendError> {
    let alignment = 1u32
        .checked_shl(shift)
        .ok_or(VarDctFrontendError::Unsupported {
            feature: UnsupportedVarDctFeature::ImageDimensions,
        })?;
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or(VarDctFrontendError::Unsupported {
            feature: UnsupportedVarDctFeature::ImageDimensions,
        })
}

fn jpeg_block_alignment(jpeg_upsampling: [u32; 3]) -> [u32; 2] {
    const HORIZONTAL: [u32; 4] = [0, 1, 1, 0];
    const VERTICAL: [u32; 4] = [0, 1, 0, 1];
    jpeg_upsampling.into_iter().fold([0, 0], |maximum, value| {
        let index = value as usize;
        [
            maximum[0].max(HORIZONTAL[index]),
            maximum[1].max(VERTICAL[index]),
        ]
    })
}

fn jpeg_channel_shifts(jpeg_upsampling: [u32; 3]) -> [VarDctChannelShift; 3] {
    const HORIZONTAL: [u32; 4] = [0, 1, 1, 0];
    const VERTICAL: [u32; 4] = [0, 1, 0, 1];
    let maximum = jpeg_block_alignment(jpeg_upsampling);
    jpeg_upsampling.map(|value| {
        let index = value as usize;
        VarDctChannelShift {
            horizontal: maximum[0] - HORIZONTAL[index],
            vertical: maximum[1] - VERTICAL[index],
        }
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VarDctFrameRole {
    Presentation,
    ProgressiveDcRefinement,
    ProgressiveDcFinal,
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

fn validate_frame(
    inventory: &CodestreamInventory,
    role: VarDctFrameRole,
) -> Result<&FrameInventory, VarDctFrontendError> {
    let frame = &inventory.frames[0];
    let role_is_invalid = match role {
        VarDctFrameRole::Presentation => {
            frame.frame_type != FrameType::Regular
                || !frame.is_last
                || frame.uses_lf_frame()
                || frame.lf_level != 0
        }
        VarDctFrameRole::ProgressiveDcRefinement => {
            frame.frame_type != FrameType::LowFrequency
                || frame.is_last
                || !frame.uses_lf_frame()
                || frame.lf_level == 0
                || !frame.save_before_color_transform
        }
        VarDctFrameRole::ProgressiveDcFinal => {
            frame.frame_type != FrameType::Regular
                || !frame.is_last
                || !frame.uses_lf_frame()
                || frame.lf_level != 0
        }
    };
    if role_is_invalid || frame.is_preview {
        return unsupported(UnsupportedVarDctFeature::NonRegularFrame);
    }
    if frame.encoding != FrameEncoding::VarDct {
        return unsupported(UnsupportedVarDctFeature::ModularFrame);
    }
    // Noise, patches, splines, and unknown frame extensions remain outside the transform
    // capability. LF-frame reuse is accepted only by the recursive progressive-DC entry point.
    let supported_flags = if role == VarDctFrameRole::Presentation {
        0x80
    } else {
        0x20 | 0x80
    };
    if frame.flags & !supported_flags != 0 {
        return unsupported(UnsupportedVarDctFeature::FrameFeatures);
    }
    if !inventory.image_header.xyb_encoded && !frame.do_ycbcr {
        return unsupported(UnsupportedVarDctFeature::NonXybImage);
    }
    if inventory.image_header.xyb_encoded && frame.do_ycbcr {
        return unsupported(UnsupportedVarDctFeature::Ycbcr);
    }
    if frame.jpeg_upsampling.into_iter().any(|value| value > 3) {
        return unsupported(UnsupportedVarDctFeature::JpegSubsampling);
    }
    if !frame.do_ycbcr && frame.jpeg_upsampling != [0; 3] {
        return unsupported(UnsupportedVarDctFeature::JpegSubsampling);
    }
    let channel_shifts = jpeg_channel_shifts(frame.jpeg_upsampling);
    let adaptive_lf_signaled = frame.flags & 0x80 == 0;
    if adaptive_lf_signaled
        && !frame.uses_lf_frame()
        && channel_shifts.iter().any(|shift| shift.is_subsampled())
    {
        return unsupported(UnsupportedVarDctFeature::SubsampledAdaptiveLf);
    }
    if frame
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
    if frame.save_as_reference != 0
        || (role != VarDctFrameRole::ProgressiveDcRefinement && frame.save_before_color_transform)
    {
        return unsupported(UnsupportedVarDctFeature::FrameReferences);
    }
    if frame.num_passes != 1
        || !frame.progressive_passes.shifts.is_empty()
        || !frame.progressive_passes.downsampling.is_empty()
        || !frame.progressive_passes.last_pass.is_empty()
    {
        return unsupported(UnsupportedVarDctFeature::ProgressivePasses);
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
    reader: &mut impl BitInput,
    field: &'static str,
) -> Result<(), VarDctPacketError> {
    let is_default = metadata_bool(reader, field)?;
    if !is_default {
        return Err(VarDctPacketError::NonDefaultMetadata { field });
    }
    Ok(())
}

/// A view of a [`BitInput`] that refuses reads crossing a physical packet boundary.
pub(crate) struct BoundedBitInput<'a> {
    reader: &'a mut dyn BitInput,
    packet_end: u64,
}

impl<'a> BoundedBitInput<'a> {
    pub(crate) fn new<R: BitInput>(reader: &'a mut R, packet_end: u64) -> Self {
        Self { reader, packet_end }
    }
}

impl BitInput for BoundedBitInput<'_> {
    fn bit_offset(&self) -> u64 {
        self.reader.bit_offset()
    }

    fn read_bits(&mut self, count: u8) -> crate::Result<u64> {
        let end = self
            .bit_offset()
            .checked_add(u64::from(count))
            .ok_or_else(|| crate::Error::backend("metadata bit offset overflow"))?;
        if end > self.packet_end {
            return Err(jxl_gpu_bitstream::Error::UnexpectedEndOfBits.into());
        }
        self.reader.read_bits(count)
    }
}

#[derive(Clone, Copy)]
struct MetadataU32Part {
    offset: u32,
    bit_count: Option<u8>,
}

impl MetadataU32Part {
    const fn constant(value: u32) -> Self {
        Self {
            offset: value,
            bit_count: None,
        }
    }

    const fn bits(offset: u32, bit_count: u8) -> Self {
        Self {
            offset,
            bit_count: Some(bit_count),
        }
    }
}

pub(crate) fn metadata_bits(
    reader: &mut impl BitInput,
    stage: &'static str,
    count: u8,
) -> Result<u64, VarDctPacketError> {
    reader
        .read_bits(count)
        .map_err(|source| metadata_error(stage, source))
}

pub(crate) fn metadata_bool(
    reader: &mut impl BitInput,
    stage: &'static str,
) -> Result<bool, VarDctPacketError> {
    Ok(metadata_bits(reader, stage, 1)? != 0)
}

fn metadata_u32(
    reader: &mut impl BitInput,
    stage: &'static str,
    parts: [MetadataU32Part; 4],
) -> Result<u32, VarDctPacketError> {
    let selector = metadata_bits(reader, stage, 2)? as usize;
    let part = parts[selector];
    let value = part.bit_count.map_or(Ok(u64::from(part.offset)), |count| {
        metadata_bits(reader, stage, count).map(|value| value.wrapping_add(u64::from(part.offset)))
    })?;
    u32::try_from(value).map_err(|_| VarDctPacketError::PacketRangeOverflow)
}

pub(crate) fn metadata_f16(
    reader: &mut impl BitInput,
    stage: &'static str,
) -> Result<f32, VarDctPacketError> {
    let value = metadata_bits(reader, stage, 16)? as u32;
    let neg_bit = (value & 0x8000) << 16;
    if value & 0x7fff == 0 {
        return Ok(f32::from_bits(neg_bit));
    }
    let mantissa = value & 0x3ff;
    let exponent = (value >> 10) & 0x1f;
    if exponent == 0x1f {
        return Err(VarDctPacketError::MetadataBitstream {
            stage,
            source: jxl_bitstream::Error::InvalidFloat,
        });
    }
    if exponent == 0 {
        let value = (1.0 / 16384.0) * (mantissa as f32 / 1024.0);
        Ok(if neg_bit != 0 { -value } else { value })
    } else {
        let mantissa = mantissa << 13;
        let exponent = exponent + 112;
        let bitpattern = mantissa | (exponent << 23) | neg_bit;
        Ok(f32::from_bits(bitpattern))
    }
}

fn metadata_error(stage: &'static str, source: crate::Error) -> VarDctPacketError {
    match source {
        crate::Error::Bitstream(source) => VarDctPacketError::MetadataBitstream {
            stage,
            source: map_gpu_bitstream_error(source),
        },
        source => VarDctPacketError::MetadataReader {
            stage,
            source: map_metadata_reader_error(source),
        },
    }
}

pub(crate) fn map_metadata_reader_error(source: crate::Error) -> VarDctMetadataReaderError {
    match source {
        crate::Error::Bitstream(source) => VarDctMetadataReaderError::Bitstream(source),
        crate::Error::ModularTree(source) => VarDctMetadataReaderError::ModularTree(source),
        _ => VarDctMetadataReaderError::Other("metadata reader failed"),
    }
}

pub(crate) fn map_gpu_bitstream_error(source: jxl_gpu_bitstream::Error) -> jxl_bitstream::Error {
    match source {
        jxl_gpu_bitstream::Error::UnexpectedEndOfBits => {
            jxl_bitstream::Error::Io(std::io::ErrorKind::UnexpectedEof.into())
        }
        jxl_gpu_bitstream::Error::NonZeroPadding => jxl_bitstream::Error::NonZeroPadding,
        jxl_gpu_bitstream::Error::InvalidBitCount(_) => {
            jxl_bitstream::Error::ValidationFailed("invalid bit count")
        }
        _ => jxl_bitstream::Error::ValidationFailed("codestream bit reader failed"),
    }
}

fn unpack_signed(value: u32) -> i32 {
    if value & 1 == 0 {
        (value >> 1) as i32
    } else {
        let magnitude = (value >> 1) + 1;
        if magnitude == 1u32 << 31 {
            i32::MIN
        } else {
            -i32::try_from(magnitude).expect("signed mapping magnitude is below i32::MAX")
        }
    }
}

#[cfg(test)]
mod tests {
    use jxl_gpu_bitstream::{InventoryLimits, ParseLimits};

    use super::*;

    fn fixture(input: &str) -> Vec<u8> {
        fn nibble(byte: u8) -> u8 {
            match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => panic!("invalid checked-in fixture hex digit"),
            }
        }

        let digits = input
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        assert_eq!(digits.len() % 2, 0, "fixture hex must contain whole bytes");
        digits
            .chunks_exact(2)
            .map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1]))
            .collect()
    }

    #[test]
    fn strict_profile_preserves_utf8_frame_names_and_rejects_invalid_bytes() {
        let encoded = fixture(include_str!("../test-data/basic.jxl.hex"));
        let parsed = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default()).unwrap();
        let mut inventory = parsed
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        assert!(inventory.frames[0].name_bytes.is_empty());

        inventory.frames[0]
            .name_bytes
            .extend_from_slice(b"named frame");
        assert_eq!(
            StandardVarDctProfile::negotiate(&inventory)
                .unwrap()
                .frame_name,
            "named frame"
        );
        inventory.frames[0].name_bytes = vec![0xff];
        assert_eq!(
            StandardVarDctProfile::negotiate(&inventory).unwrap_err(),
            VarDctFrontendError::InvalidFrameName
        );
    }
}
