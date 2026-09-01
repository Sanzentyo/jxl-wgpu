//! GPU entropy frontend for the bounded standard VarDCT packet profile.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{BitRange, BitReader, CodestreamInventory};
use jxl_gpu_protocol::TransformKind;
use jxl_oxide_common::Bundle;
use jxl_vardct::{DequantMatrixSet, DequantMatrixSetParams};
use thiserror::Error;

use crate::GpuCodestream;
use crate::codestream_data::CodestreamBitReader;
use crate::entropy::EntropyStreamParams;
use crate::entropy_window::GroupStreamSegment;
use crate::modular_tree::{
    BitInput, EntropyDecoderIr, MaConfigIr, MaTreeLimits, MaTreeNodeIr, MetadataEntropyCursor,
    PackedModularMetadata, parse_ma_config,
};
use crate::vardct_frontend::{
    BoundedBitInput, HfBlockContextIr, HfGlobalPrefix, LfChannelCorrelation,
    LfChannelDequantization, LfGlobalPrefix, StandardVarDctProfile, VarDctFrontendError,
    VarDctGroupRect, VarDctMetadataReaderError, VarDctPacketError, VarDctSectionLayout,
    map_metadata_reader_error, metadata_bits, metadata_bool, metadata_f16,
    parse_hf_metadata_header_reader, parse_lf_group_header_reader, validate_packet_end_bits,
};

const SHADER_TEMPLATE: &str = include_str!("vardct_packet.wgsl");
const MODULAR_ENTROPY_ABI: &str = include_str!("modular_entropy_abi.wgsl");
const MODULAR_ENTROPY: &str = include_str!("modular_entropy.wgsl");
const MODULAR_RECONSTRUCT: &str = include_str!("modular_reconstruct.wgsl");
const ENTROPY_ABI_MARKER: &str = "/*__JXL_MODULAR_ENTROPY_ABI__*/";
const ENTROPY_MARKER: &str = "/*__JXL_MODULAR_ENTROPY__*/";
const RECONSTRUCT_MARKER: &str = "/*__JXL_MODULAR_RECONSTRUCT__*/";

const ZERO_AC_HF_GLOBAL: u32 = 0x2495;
const PACKET_WINDOW_ENABLED: u32 = 1 << 2;
pub(crate) const GENERIC_PACKET_EXECUTION_STATE_BYTES: u64 = 64;
pub(crate) const WEIGHTED_PACKET_EXECUTION_STATE_BYTES: u64 = 128;

/// A standard feature excluded from the deliberately bounded VarDCT packet profile.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum UnsupportedVarDctPacketFeature {
    #[error("the combined one-entry VarDCT packet cannot address multiple LF groups")]
    CombinedPacketMultipleLfGroups,
    #[error("the bounded VarDCT decoder currently accepts 8-bit samples")]
    BitDepth,
    #[error("the one-entry packet extent is not one implemented VarDCT transform")]
    TransformExtent,
    #[error(
        "the MA tree uses previous-channel property {property}; the heterogeneous VarDCT metadata layout is not implemented"
    )]
    PreviousChannelMaProperty { property: u32 },
}

/// Host-side failure before image entropy is submitted to the GPU.
#[derive(Debug, Error)]
pub enum BoundedVarDctPacketError {
    #[error(transparent)]
    Frontend(#[from] VarDctFrontendError),
    #[error(transparent)]
    Packet(#[from] VarDctPacketError),
    #[error(transparent)]
    Unsupported(#[from] UnsupportedVarDctPacketFeature),
    #[error("failed to read bounded VarDCT metadata: {0}")]
    Bitstream(#[from] jxl_gpu_bitstream::Error),
    #[error("failed to read bounded VarDCT metadata from codestream spans: {0}")]
    CodestreamReader(#[source] VarDctMetadataReaderError),
    #[error("failed to parse the bounded MA-tree descriptor: {0}")]
    ModularTree(String),
    #[error("failed to position the HF coefficient-order reader: {0}")]
    CoefficientOrderBitstream(#[source] jxl_bitstream::Error),
    #[error("failed to parse the HF coefficient-order span reader: {0}")]
    CoefficientOrderReader(#[source] VarDctMetadataReaderError),
    #[error("failed to decode the HF coefficient-order permutation: {0}")]
    CoefficientOrderCoding(#[source] jxl_coding::Error),
    #[error("the packed MA-tree metadata ABI is malformed")]
    PackedMetadata,
    #[error("{stage} requests a global MA tree, but LF-global did not provide one")]
    MissingGlobalMaTree { stage: &'static str },
    #[error("HF continuation cursor {cursor} precedes LF entropy start {lf_start}")]
    HfContinuationBeforeLf { cursor: u32, lf_start: u32 },
    #[error("VarDCT packet arithmetic overflowed while computing {field}")]
    ArithmeticOverflow { field: &'static str },
    #[error("HF-global entropy metadata leaves {bits} non-padding bits")]
    HfGlobalTrailingBits { bits: u64 },
    #[error("HF block-context map has {actual} entries; expected {expected}")]
    HfBlockContextMapLength { expected: usize, actual: usize },
    #[error("HF dequantization matrix {matrix} uses raw Modular encoding")]
    RawHfDequantMatrix { matrix: usize },
    #[error("HF dequantization matrix {matrix} uses invalid encoding mode {encoding}")]
    HfDequantMatrixEncoding { matrix: usize, encoding: u8 },
    #[error("HF dequantization matrix {matrix} is invalid: {reason}")]
    HfDequantMatrixValue { matrix: usize, reason: &'static str },
}

/// GPU-reported validation failure. No output is authoritative after this error.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum GpuVarDctPacketError {
    #[error(
        "GPU VarDCT packet entropy failed with status {code} at bit {cursor}/{end} after LF/HF {lf_decoded}/{hf_decoded} symbols (detail {detail})"
    )]
    Entropy {
        code: u32,
        cursor: u32,
        end: u32,
        lf_decoded: u32,
        hf_decoded: u32,
        detail: u32,
    },
    #[error("GPU VarDCT packet has invalid LF local header")]
    LfHeader,
    #[error("GPU VarDCT packet does not declare the negotiated bounded first-block count")]
    FirstBlock,
    #[error("GPU VarDCT packet has invalid HF metadata local header")]
    HfHeader,
    #[error("GPU VarDCT packet selects strategy {actual}, expected {expected}")]
    Strategy { actual: u32, expected: u32 },
    #[error("GPU VarDCT packet selects invalid EPF sharpness {value}")]
    Sharpness { value: u32 },
    #[error("GPU VarDCT packet does not contain the standard zero-AC HF-global bundle")]
    HfGlobal,
    #[error("GPU VarDCT packet returned unknown status {code}")]
    Unknown { code: u32 },
}

/// Parsed host metadata and untouched image entropy for one strict packet.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundedVarDctPacketPlan {
    pub profile: StandardVarDctProfile,
    /// A transform enforced for the single-entry packet form. Sectioned packets carry a
    /// GPU-decoded mixed-strategy topology and therefore have no host-assumed transform.
    pub uniform_transform: Option<TransformKind>,
    /// LF-global packet containing the scalar quantizer fields and global MA descriptor.
    pub lf_global: BitRange,
    /// Separate HF-global packet, or `None` when all three packets share a single TOC entry.
    pub hf_global: Option<BitRange>,
    /// Descriptor end used as the LF-group start by the one-entry TOC form.
    pub entropy_bit_offset: u32,
    /// Packed global MA metadata when LF-global declared it. Local-tree streams carry their own
    /// descriptor in each group plan instead.
    pub modular_metadata: Vec<u32>,
    /// Whether the shared MA tree requires the weighted self-correcting predictor workspace.
    pub needs_self_correcting: bool,
    pub lf_dequantization: LfChannelDequantization,
    pub global_scale: u32,
    pub quant_lf: u32,
    pub lf_correlation: LfChannelCorrelation,
    /// One independently bounded LF quantization/HF-metadata packet in logical raster order.
    pub groups: Vec<BoundedVarDctGroupPlan>,
    /// Descriptor-only HF coefficient entropy plan. Coefficient symbols remain in pass-group
    /// packets and are never expanded on the host.
    pub hf_coefficients: Option<HfCoefficientEntropyPlan>,
    hf_block_context: HfBlockContextIr,
    global_ma_config: Option<MaConfigIr>,
}

/// One host-packed MA tree/histogram bundle and its exact GPU reconstruction requirements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundedModularEntropyPlan {
    pub metadata: Vec<u32>,
    pub needs_self_correcting: bool,
    pub lz77_window_words: u32,
}

/// Host metadata discovered after the first GPU stage returns the LF entropy cursor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundedHfMetadataContinuation {
    pub token_bit_offset: u32,
    pub block_count: u32,
    pub modular: BoundedModularEntropyPlan,
}

/// One LF group's packet geometry and persistent decode contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundedVarDctGroupPlan {
    pub index: u32,
    pub rect: VarDctGroupRect,
    /// Maximum number of non-overlapping first blocks reconstructed from this group's HF
    /// metadata. The actual count is decoded and reported by the GPU packet frontend.
    pub task_capacity: u32,
    coefficient_words: u32,
    /// Packet containing this group's quantized LF and HF-metadata Modular streams.
    pub lf_group: BitRange,
    pub lf_stream_index: u32,
    pub hf_stream_index: u32,
    /// First LF image-entropy bit, after either the selected global tree header or this group's
    /// local MA descriptor.
    pub lf_entropy_bit_offset: u32,
    pub lf_modular: BoundedModularEntropyPlan,
    /// Physical power-of-two history ring reused by the group's two sequential Modular streams.
    pub lz77_window_words: u32,
    pub extra_precision: u8,
    /// HF metadata parsed directly when a progressive-DC producer replaces this group's LF stream.
    pub external_lf_hf: Option<BoundedHfMetadataContinuation>,
}

/// Host-packed entropy tables and untouched pass-group packets for one VarDCT AC pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfCoefficientEntropyPlan {
    pub num_hf_presets: u32,
    pub num_block_clusters: u32,
    pub metadata: Vec<u32>,
    /// One packed entropy-cluster index per JPEG XL HF coefficient context. The optional final
    /// LZ77 distance context remains internal to `metadata`.
    pub context_map: Vec<u32>,
    /// Default channel/order-to-block-cluster map used before coefficient contexts.
    pub block_context_map: Vec<u32>,
    /// Quant-field thresholds used to select the HF block-context map segment.
    pub qf_thresholds: Vec<u32>,
    /// Quantized LF thresholds in X, Y, B channel order.
    pub lf_thresholds: [Vec<i32>; 3],
    /// Thirty-nine channel/order descriptors followed by packed `(x, y)` coordinate tables.
    /// Natural orders share one table across their three channels; custom orders retain one table
    /// per channel.
    pub order_words: Vec<u32>,
    pub order_coordinate_offset_words: u32,
    pub pass_groups: Vec<BitRange>,
    /// Per-pass-group power-of-two history capacity for the common GPU entropy executor.
    pub lz77_window_words: u32,
    /// Complete matrix resource region as F32 bit patterns when HF-global overrides defaults.
    pub dequant_matrix_words: Option<Vec<[u32; 4]>>,
}

fn parse_ma_config_at_reader(
    reader: &mut impl BitInput,
    packet_end: u64,
) -> Result<(MaConfigIr, u64), BoundedVarDctPacketError> {
    let mut reader = BoundedBitInput::new(reader, packet_end);
    let config =
        parse_ma_config(&mut reader, MaTreeLimits::default()).map_err(map_modular_reader_error)?;
    let descriptor_end = reader.bit_offset();
    if descriptor_end > packet_end {
        return Err(VarDctPacketError::PacketBoundary {
            cursor: descriptor_end,
            packet_end,
        }
        .into());
    }
    validate_ma_config(&config)?;
    Ok((config, descriptor_end))
}

fn map_modular_reader_error(source: crate::Error) -> BoundedVarDctPacketError {
    match source {
        crate::Error::Bitstream(source) => BoundedVarDctPacketError::Bitstream(source),
        crate::Error::ModularTree(source) => {
            BoundedVarDctPacketError::ModularTree(source.to_string())
        }
        source => BoundedVarDctPacketError::CodestreamReader(map_metadata_reader_error(source)),
    }
}

#[derive(Clone, Copy)]
enum PacketSource<'source> {
    Slice(&'source [u8]),
    Spans(&'source GpuCodestream),
}

impl<'source> PacketSource<'source> {
    fn logical_bits(self) -> Result<u64, BoundedVarDctPacketError> {
        match self {
            Self::Slice(bytes) => u64::try_from(bytes.len())
                .ok()
                .and_then(|length| length.checked_mul(8))
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "codestream bit length",
                }),
            Self::Spans(source) => source.logical_bits().map_err(map_modular_reader_error),
        }
    }

    fn reader_at(
        self,
        bit_offset: u64,
    ) -> Result<PacketBitReader<'source>, BoundedVarDctPacketError> {
        let mut reader = match self {
            Self::Slice(bytes) => PacketBitReader::Slice(BitReader::new(bytes)),
            Self::Spans(source) => PacketBitReader::Spans(source.reader()),
        };
        reader
            .skip_bits(bit_offset)
            .map_err(map_modular_reader_error)?;
        Ok(reader)
    }
}

enum PacketBitReader<'source> {
    Slice(BitReader<'source>),
    Spans(CodestreamBitReader<'source>),
}

impl PacketBitReader<'_> {
    fn skip_bits(&mut self, count: u64) -> crate::Result<()> {
        match self {
            Self::Slice(reader) => reader.skip_bits(count).map_err(Into::into),
            Self::Spans(reader) => reader.skip_bits(count),
        }
    }
}

impl BitInput for PacketBitReader<'_> {
    fn bit_offset(&self) -> u64 {
        match self {
            Self::Slice(reader) => reader.bit_offset(),
            Self::Spans(reader) => reader.bit_offset(),
        }
    }

    fn read_bits(&mut self, count: u8) -> crate::Result<u64> {
        match self {
            Self::Slice(reader) => reader.read_bits(count).map_err(Into::into),
            Self::Spans(reader) => reader.read_bits(count),
        }
    }
}

fn source_reader_at<'source>(
    source: PacketSource<'source>,
    bit_offset: u64,
) -> Result<PacketBitReader<'source>, BoundedVarDctPacketError> {
    source.reader_at(bit_offset)
}

fn validate_source_packet_end(
    source: PacketSource<'_>,
    packet_end: u64,
) -> Result<(), BoundedVarDctPacketError> {
    let codestream_bits = source.logical_bits()?;
    validate_packet_end_bits(codestream_bits, packet_end).map_err(Into::into)
}

fn validate_ma_config(config: &MaConfigIr) -> Result<(), BoundedVarDctPacketError> {
    for node in &config.nodes {
        if let MaTreeNodeIr::Decision { property, .. } = *node
            && property >= 16
        {
            return Err(
                UnsupportedVarDctPacketFeature::PreviousChannelMaProperty { property }.into(),
            );
        }
    }
    Ok(())
}

fn pack_ma_metadata(config: &MaConfigIr) -> Result<Vec<u32>, BoundedVarDctPacketError> {
    let PackedModularMetadata { words } = config
        .pack_gpu_metadata()
        .map_err(|error| BoundedVarDctPacketError::ModularTree(error.to_string()))?;
    if words.len() <= 9 {
        return Err(BoundedVarDctPacketError::PackedMetadata);
    }
    Ok(words)
}

fn pack_modular_plan(
    config: &MaConfigIr,
    distance_multiplier: u32,
    decoded_symbol_limit: u32,
) -> Result<BoundedModularEntropyPlan, BoundedVarDctPacketError> {
    Ok(BoundedModularEntropyPlan {
        metadata: pack_ma_metadata(config)?,
        needs_self_correcting: config.needs_self_correcting(),
        lz77_window_words: config
            .entropy
            .lz77_window_words(distance_multiplier, decoded_symbol_limit)
            .map_err(|error| BoundedVarDctPacketError::ModularTree(error.to_string()))?,
    })
}

impl BoundedVarDctPacketPlan {
    /// Parses bounded scalar metadata only. Image symbols remain encoded for the GPU.
    pub fn parse(
        codestream: &[u8],
        inventory: &CodestreamInventory,
    ) -> Result<Self, BoundedVarDctPacketError> {
        Self::parse_inner(PacketSource::Slice(codestream), inventory, None)
    }

    /// Parses bounded metadata from a logically contiguous, potentially multi-span codestream.
    pub(crate) fn parse_source(
        source: &GpuCodestream,
        inventory: &CodestreamInventory,
    ) -> Result<Self, BoundedVarDctPacketError> {
        Self::parse_inner(PacketSource::Spans(source), inventory, None)
    }

    pub(crate) fn parse_progressive_dc_source(
        source: &GpuCodestream,
        inventory: &CodestreamInventory,
        is_final: bool,
    ) -> Result<Self, BoundedVarDctPacketError> {
        Self::parse_inner(PacketSource::Spans(source), inventory, Some(is_final))
    }

    fn parse_inner(
        source: PacketSource<'_>,
        inventory: &CodestreamInventory,
        progressive_dc_final: Option<bool>,
    ) -> Result<Self, BoundedVarDctPacketError> {
        let profile = progressive_dc_final.map_or_else(
            || StandardVarDctProfile::negotiate(inventory),
            |is_final| StandardVarDctProfile::negotiate_progressive_dc(inventory, is_final),
        )?;
        if profile.bits_per_sample != 8 {
            return Err(UnsupportedVarDctPacketFeature::BitDepth.into());
        }
        let (uniform_transform, lf_global_packet, lf_group_packets, hf_global, pass_groups) =
            match &profile.sections {
                VarDctSectionLayout::Single { packet } => {
                    if profile.low_frequency_group_count != 1 {
                        return Err(
                            UnsupportedVarDctPacketFeature::CombinedPacketMultipleLfGroups.into(),
                        );
                    }
                    (
                        if profile.uses_lf_frame {
                            None
                        } else {
                            Some(
                                transform_for_extent(profile.width, profile.height)
                                    .ok_or(UnsupportedVarDctPacketFeature::TransformExtent)?,
                            )
                        },
                        *packet,
                        vec![*packet],
                        None,
                        Vec::new(),
                    )
                }
                VarDctSectionLayout::Sections {
                    lf_global,
                    lf_groups,
                    hf_global,
                    pass_groups,
                } => (
                    None,
                    *lf_global,
                    lf_groups.clone(),
                    Some(*hf_global),
                    pass_groups.clone(),
                ),
            };
        let lf_global_end =
            lf_global_packet
                .end()
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "LF-global end",
                })?;
        validate_source_packet_end(source, lf_global_end)?;
        let mut lf_global_reader = source_reader_at(source, lf_global_packet.offset)?;
        let lf_global = LfGlobalPrefix::parse_reader(&mut lf_global_reader, lf_global_end)?;
        let (global_ma_config, descriptor_end) =
            if let Some(tree_offset) = lf_global.global_ma_tree_bit_offset {
                let mut tree_reader = source_reader_at(source, tree_offset)?;
                let (config, end) = parse_ma_config_at_reader(&mut tree_reader, lf_global_end)?;
                (Some(config), end)
            } else {
                (None, lf_global.suffix_bit_offset)
            };
        let lf_global_end =
            lf_global_packet
                .end()
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "LF-global end",
                })?;
        if descriptor_end > lf_global_end {
            return Err(VarDctPacketError::PacketBoundary {
                cursor: descriptor_end,
                packet_end: lf_global_end,
            }
            .into());
        }
        let words = global_ma_config
            .as_ref()
            .map(pack_ma_metadata)
            .transpose()?
            .unwrap_or_default();
        let entropy_bit_offset = u32::try_from(descriptor_end).map_err(|_| {
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "entropy bit offset",
            }
        })?;
        let groups = lf_group_packets
            .iter()
            .copied()
            .enumerate()
            .map(|(index, lf_group)| {
                let index = u32::try_from(index).map_err(|_| {
                    BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "LF-group index",
                    }
                })?;
                let rect = profile.low_frequency_group_rect(u64::from(index))?;
                let blocks_x = rect.width.div_ceil(8);
                let blocks_y = rect.height.div_ceil(8);
                let block_count = blocks_x.checked_mul(blocks_y).ok_or(
                    BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "LF-group block count",
                    },
                )?;
                let coefficient_words = block_count.checked_mul(8 * 8 * 3).ok_or(
                    BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "LF-group coefficient words",
                    },
                )?;
                let correlation_samples = rect
                    .width
                    .div_ceil(64)
                    .checked_mul(rect.height.div_ceil(64))
                    .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "LF-group correlation sample count",
                    })?;
                let hf_decoded_symbol_limit = block_count
                    .checked_mul(4)
                    .and_then(|samples| {
                        block_count
                            .checked_mul(2)
                            .and_then(|tasks| samples.checked_add(tasks))
                    })
                    .and_then(|samples| {
                        correlation_samples
                            .checked_mul(2)
                            .and_then(|correlations| samples.checked_add(correlations))
                    })
                    .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "LF-group decoded symbol limit",
                    })?;
                let lf_group_start = if hf_global.is_some() {
                    lf_group.offset
                } else {
                    descriptor_end
                };
                let lf_group_end =
                    lf_group
                        .end()
                        .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                            field: "LF-group end",
                        })?;
                validate_source_packet_end(source, lf_group_end)?;
                let lf_decoded_symbol_limit = block_count.checked_mul(3).ok_or(
                    BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "LF-group LF decoded symbol limit",
                    },
                )?;
                let (
                    lf_token_bit_offset,
                    lf_modular,
                    lz77_window_words,
                    extra_precision,
                    external_lf_hf,
                ) = if profile.uses_lf_frame {
                    let mut hf_reader = source_reader_at(source, lf_group_start)?;
                    let hf_header = parse_hf_metadata_header_reader(
                        &mut hf_reader,
                        lf_group_end,
                        rect.width,
                        rect.height,
                    )?;
                    let (hf_config, hf_token_bit_offset) = if hf_header.modular.use_global_tree {
                        (
                            global_ma_config.clone().ok_or(
                                BoundedVarDctPacketError::MissingGlobalMaTree {
                                    stage: "HF metadata",
                                },
                            )?,
                            hf_header.modular.tree_or_token_bit_offset,
                        )
                    } else {
                        let mut tree_reader =
                            source_reader_at(source, hf_header.modular.tree_or_token_bit_offset)?;
                        parse_ma_config_at_reader(&mut tree_reader, lf_group_end)?
                    };
                    let hf_modular = pack_modular_plan(
                        &hf_config,
                        block_count.max(blocks_x).max(1),
                        hf_decoded_symbol_limit,
                    )?;
                    let continuation = BoundedHfMetadataContinuation {
                        token_bit_offset: u32::try_from(hf_token_bit_offset).map_err(|_| {
                            BoundedVarDctPacketError::ArithmeticOverflow {
                                field: "HF metadata token bit offset",
                            }
                        })?,
                        block_count: hf_header.block_count,
                        modular: hf_modular.clone(),
                    };
                    (
                        hf_token_bit_offset,
                        hf_modular.clone(),
                        hf_modular.lz77_window_words,
                        0,
                        Some(continuation),
                    )
                } else {
                    let mut lf_reader = source_reader_at(source, lf_group_start)?;
                    let lf_header = parse_lf_group_header_reader(&mut lf_reader, lf_group_end)?;
                    let (lf_config, lf_token_bit_offset) = if lf_header.modular.use_global_tree {
                        (
                            global_ma_config.clone().ok_or(
                                BoundedVarDctPacketError::MissingGlobalMaTree {
                                    stage: "LF quantization",
                                },
                            )?,
                            lf_header.modular.tree_or_token_bit_offset,
                        )
                    } else {
                        let mut tree_reader =
                            source_reader_at(source, lf_header.modular.tree_or_token_bit_offset)?;
                        parse_ma_config_at_reader(&mut tree_reader, lf_group_end)?
                    };
                    let lf_modular =
                        pack_modular_plan(&lf_config, blocks_x.max(1), lf_decoded_symbol_limit)?;
                    let hf_window_words = if let Some(config) = &global_ma_config {
                        config
                            .entropy
                            .lz77_window_words(
                                block_count.max(blocks_x).max(1),
                                hf_decoded_symbol_limit,
                            )
                            .map_err(|error| {
                                BoundedVarDctPacketError::ModularTree(error.to_string())
                            })?
                    } else {
                        hf_decoded_symbol_limit.checked_next_power_of_two().ok_or(
                            BoundedVarDctPacketError::ArithmeticOverflow {
                                field: "LF-group deferred HF LZ77 window",
                            },
                        )?
                    };
                    (
                        lf_token_bit_offset,
                        lf_modular.clone(),
                        lf_modular.lz77_window_words.max(hf_window_words),
                        lf_header.extra_precision,
                        None,
                    )
                };
                Ok(BoundedVarDctGroupPlan {
                    index,
                    rect,
                    task_capacity: block_count,
                    coefficient_words,
                    lf_group,
                    lf_stream_index: profile.lf_quant_stream_index(u64::from(index))?,
                    hf_stream_index: profile.hf_metadata_stream_index(u64::from(index))?,
                    lf_entropy_bit_offset: u32::try_from(lf_token_bit_offset).map_err(|_| {
                        BoundedVarDctPacketError::ArithmeticOverflow {
                            field: "LF image-entropy bit offset",
                        }
                    })?,
                    lf_modular,
                    lz77_window_words,
                    extra_precision,
                    external_lf_hf,
                })
            })
            .collect::<Result<Vec<_>, BoundedVarDctPacketError>>()?;
        let hf_coefficients = hf_global
            .map(|packet| {
                let max_group_blocks = profile
                    .group_dimension
                    .div_ceil(8)
                    .checked_mul(profile.group_dimension.div_ceil(8))
                    .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "pass-group block count",
                    })?;
                let decoded_symbol_limit = max_group_blocks.checked_mul(3 * 64).ok_or(
                    BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "pass-group coefficient symbol limit",
                    },
                )?;
                HfCoefficientEntropyPlan::parse_inner(
                    source,
                    packet,
                    u32::try_from(profile.group_count).map_err(|_| {
                        BoundedVarDctPacketError::ArithmeticOverflow {
                            field: "pass-group count",
                        }
                    })?,
                    &lf_global.hf_block_context,
                    pass_groups,
                    decoded_symbol_limit,
                )
            })
            .transpose()?;
        let needs_self_correcting = if profile.uses_lf_frame {
            groups
                .iter()
                .any(|group| group.lf_modular.needs_self_correcting)
        } else {
            global_ma_config
                .as_ref()
                .is_some_and(MaConfigIr::needs_self_correcting)
        };
        Ok(Self {
            profile,
            uniform_transform,
            lf_global: lf_global_packet,
            hf_global,
            entropy_bit_offset,
            modular_metadata: words,
            needs_self_correcting,
            lf_dequantization: lf_global.lf_dequantization,
            global_scale: lf_global.global_scale,
            quant_lf: lf_global.quant_lf,
            lf_correlation: lf_global.lf_correlation,
            groups,
            hf_coefficients,
            hf_block_context: lf_global.hf_block_context,
            global_ma_config,
        })
    }

    #[must_use]
    pub fn block_extent(&self) -> [u32; 2] {
        [
            self.profile.width.div_ceil(8),
            self.profile.height.div_ceil(8),
        ]
    }

    pub fn total_task_capacity(&self) -> Result<u32, BoundedVarDctPacketError> {
        self.groups.iter().try_fold(0u32, |total, group| {
            total.checked_add(group.task_capacity).ok_or(
                BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "total LF-group task capacity",
                },
            )
        })
    }

    pub fn total_coefficient_words(&self) -> Result<u32, BoundedVarDctPacketError> {
        self.groups.iter().try_fold(0u32, |total, group| {
            total.checked_add(group.coefficient_words()).ok_or(
                BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "total LF-group coefficient words",
                },
            )
        })
    }

    /// Whether the HF metadata boundary must be discovered by a first GPU LF-only submission.
    #[must_use]
    pub const fn requires_local_tree_staging(&self) -> bool {
        !self.profile.uses_lf_frame && self.global_ma_config.is_none()
    }

    /// Whether HF metadata must stop at a GPU-discovered HF-global boundary in a single packet.
    ///
    /// A fused TOC entry does not expose the end of LF or HF-metadata entropy to the host.  The
    /// decoder therefore stages every single-entry packet instead of assuming the historical
    /// zero-AC HF-global bit pattern. This also covers ordinary frames without a progressive-DC
    /// dependency and allows the same continuation parser to accept general coefficient entropy.
    #[must_use]
    pub const fn requires_hf_global_staging(&self) -> bool {
        matches!(&self.profile.sections, VarDctSectionLayout::Single { .. })
    }

    /// Parses only the HF scalar header and its selected MA descriptor after the GPU reports the
    /// LF entropy end cursor. No image symbol is decoded on the host.
    pub fn parse_hf_continuation(
        &self,
        codestream: &[u8],
        group: &BoundedVarDctGroupPlan,
        lf_entropy_end: u32,
    ) -> Result<BoundedHfMetadataContinuation, BoundedVarDctPacketError> {
        self.parse_hf_continuation_inner(PacketSource::Slice(codestream), group, lf_entropy_end)
    }

    /// Parses the resumed HF metadata prefix from a multi-span codestream.
    pub(crate) fn parse_hf_continuation_source(
        &self,
        source: &GpuCodestream,
        group: &BoundedVarDctGroupPlan,
        lf_entropy_end: u32,
    ) -> Result<BoundedHfMetadataContinuation, BoundedVarDctPacketError> {
        self.parse_hf_continuation_inner(PacketSource::Spans(source), group, lf_entropy_end)
    }

    fn parse_hf_continuation_inner(
        &self,
        source: PacketSource<'_>,
        group: &BoundedVarDctGroupPlan,
        lf_entropy_end: u32,
    ) -> Result<BoundedHfMetadataContinuation, BoundedVarDctPacketError> {
        if lf_entropy_end < group.lf_entropy_bit_offset {
            return Err(BoundedVarDctPacketError::HfContinuationBeforeLf {
                cursor: lf_entropy_end,
                lf_start: group.lf_entropy_bit_offset,
            });
        }
        let packet_end =
            group
                .lf_group
                .end()
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "LF-group end",
                })?;
        validate_source_packet_end(source, packet_end)?;
        let mut prefix_reader = source_reader_at(source, u64::from(lf_entropy_end))?;
        let prefix = parse_hf_metadata_header_reader(
            &mut prefix_reader,
            packet_end,
            group.rect.width,
            group.rect.height,
        )?;
        let (config, token_bit_offset) = if prefix.modular.use_global_tree {
            (
                self.global_ma_config.clone().ok_or(
                    BoundedVarDctPacketError::MissingGlobalMaTree {
                        stage: "HF metadata",
                    },
                )?,
                prefix.modular.tree_or_token_bit_offset,
            )
        } else {
            let mut tree_reader =
                source_reader_at(source, prefix.modular.tree_or_token_bit_offset)?;
            parse_ma_config_at_reader(&mut tree_reader, packet_end)?
        };
        let correlation_samples = group.correlation_samples()?;
        let block_count = prefix.block_width.checked_mul(prefix.block_height).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF metadata block count",
            },
        )?;
        let decoded_symbol_limit = correlation_samples
            .checked_mul(2)
            .and_then(|samples| {
                prefix
                    .block_count
                    .checked_mul(2)
                    .and_then(|tasks| samples.checked_add(tasks))
            })
            .and_then(|samples| samples.checked_add(block_count))
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF metadata decoded symbol limit",
            })?;
        let distance_multiplier = prefix
            .block_width
            .max(prefix.block_count)
            .max(group.rect.width.div_ceil(64))
            .max(1);
        Ok(BoundedHfMetadataContinuation {
            token_bit_offset: u32::try_from(token_bit_offset).map_err(|_| {
                BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "HF metadata entropy bit offset",
                }
            })?,
            block_count: prefix.block_count,
            modular: pack_modular_plan(&config, distance_multiplier, decoded_symbol_limit)?,
        })
    }

    /// Parses the general HF-global descriptor and the sole pass-group tail of a single-entry
    /// packet after the GPU reports the HF-metadata entropy cursor.
    pub(crate) fn parse_single_hf_global_continuation_source(
        &self,
        source: &GpuCodestream,
        hf_metadata_end: u32,
    ) -> Result<HfCoefficientEntropyPlan, BoundedVarDctPacketError> {
        if self.groups.len() != 1 {
            return Err(BoundedVarDctPacketError::PackedMetadata);
        }
        let VarDctSectionLayout::Single { packet } = &self.profile.sections else {
            return Err(BoundedVarDctPacketError::PackedMetadata);
        };
        let packet_end = packet
            .end()
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "single-entry packet end",
            })?;
        if u64::from(hf_metadata_end) > packet_end {
            return Err(VarDctPacketError::PacketBoundary {
                cursor: u64::from(hf_metadata_end),
                packet_end,
            }
            .into());
        }
        let max_group_blocks = self
            .profile
            .group_dimension
            .div_ceil(8)
            .checked_mul(self.profile.group_dimension.div_ceil(8))
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "single-entry pass-group block count",
            })?;
        let decoded_symbol_limit = max_group_blocks.checked_mul(3 * 64).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "single-entry pass-group coefficient symbol limit",
            },
        )?;
        HfCoefficientEntropyPlan::parse_single_tail_inner(
            PacketSource::Spans(source),
            BitRange {
                offset: u64::from(hf_metadata_end),
                length: packet_end - u64::from(hf_metadata_end),
            },
            u32::try_from(self.profile.group_count).map_err(|_| {
                BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "single-entry pass-group count",
                }
            })?,
            &self.hf_block_context,
            decoded_symbol_limit,
        )
    }
}

impl BoundedVarDctGroupPlan {
    #[must_use]
    pub const fn coefficient_words(&self) -> u32 {
        self.coefficient_words
    }

    #[must_use]
    pub fn block_extent(&self) -> [u32; 2] {
        [self.rect.width.div_ceil(8), self.rect.height.div_ceil(8)]
    }

    /// U32 words retaining LF samples, weighted-predictor rows, the LZ history ring, and one
    /// aligned packet resume record. The resume record is shared by the sequential LF/HF packet
    /// consumers because their executions never overlap.
    pub fn reconstructed_words(
        &self,
        needs_self_correcting: bool,
    ) -> Result<u32, BoundedVarDctPacketError> {
        let state_offset = self.packet_execution_state_offset_words(needs_self_correcting)?;
        let state_words = u32::try_from(packet_execution_state_bytes(needs_self_correcting) / 4)
            .map_err(|_| BoundedVarDctPacketError::ArithmeticOverflow {
                field: "VarDCT packet execution-state words",
            })?;
        state_offset
            .checked_add(state_words)
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "VarDCT reconstruction and packet execution state",
            })
    }

    /// Word offset of the 16-byte-aligned packet resume record within `reconstructed`.
    pub fn packet_execution_state_offset_words(
        &self,
        needs_self_correcting: bool,
    ) -> Result<u32, BoundedVarDctPacketError> {
        let [blocks_x, blocks_y] = self.block_extent();
        let blocks =
            blocks_x
                .checked_mul(blocks_y)
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "block count",
                })?;
        let correlations = self.correlation_samples()?;
        let hf_samples = self
            .task_capacity
            .checked_mul(2)
            .and_then(|tasks| blocks.checked_add(tasks))
            .and_then(|samples| {
                correlations
                    .checked_mul(2)
                    .and_then(|cfl| samples.checked_add(cfl))
            });
        let samples = blocks
            .checked_mul(3)
            .zip(hf_samples)
            .map(|(lf, hf)| lf.max(hf))
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "VarDCT reconstruction samples",
            })?;
        let predictor_words = if needs_self_correcting {
            self.predictor_width_capacity()?.checked_mul(5).ok_or(
                BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "VarDCT weighted-predictor rows",
                },
            )?
        } else {
            0
        };
        let working_words = samples
            .checked_add(predictor_words)
            .and_then(|words| words.checked_add(self.lz77_window_words))
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "VarDCT reconstruction scratch",
            })?;
        working_words.checked_add(3).map(|words| words & !3).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "VarDCT packet execution-state alignment",
            },
        )
    }

    pub fn packet_control(
        &self,
        packet: &BoundedVarDctPacketPlan,
    ) -> Result<VarDctPacketControl, BoundedVarDctPacketError> {
        let range = |value: u64, field: &'static str| {
            u32::try_from(value).map_err(|_| BoundedVarDctPacketError::ArithmeticOverflow { field })
        };
        let lf_group_end =
            self.lf_group
                .end()
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "LF-group end",
                })?;
        let (lf_start, lf_end, hf_start, hf_end, separate_sections) = if let Some(hf_global) =
            packet.hf_global
        {
            let hf_end = hf_global
                .end()
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "HF-global end",
                })?;
            (
                range(self.lf_group.offset, "LF-group start")?,
                range(lf_group_end, "LF-group end")?,
                range(hf_global.offset, "HF-global start")?,
                range(hf_end, "HF-global end")?,
                1,
            )
        } else {
            (
                packet.entropy_bit_offset,
                range(lf_group_end, "combined packet end")?,
                0,
                range(lf_group_end, "combined packet end")?,
                0,
            )
        };
        let [blocks_x, blocks_y] = self.block_extent();
        let block_count =
            blocks_x
                .checked_mul(blocks_y)
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "block count",
                })?;
        let correlation_samples = self.correlation_samples()?;
        let strategy_offset = correlation_samples.checked_mul(2).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "strategy offset",
            },
        )?;
        let hf_mul_offset = strategy_offset.checked_add(self.task_capacity).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF multiplier offset",
            },
        )?;
        let sharpness_offset = hf_mul_offset.checked_add(self.task_capacity).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "sharpness offset",
            },
        )?;
        let raw_capacity = sharpness_offset.checked_add(block_count).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "raw metadata capacity",
            },
        )?;
        Ok(VarDctPacketControl {
            section_bits: [lf_start, lf_end, hf_start, hf_end],
            geometry: [self.rect.width, self.rect.height, blocks_x, blocks_y],
            offsets: [0, correlation_samples, strategy_offset, hf_mul_offset],
            capacities: [
                self.coefficient_words,
                raw_capacity,
                block_count
                    .checked_next_power_of_two()
                    .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "first-block field width",
                    })?
                    .trailing_zeros(),
                self.task_capacity,
            ],
            expected: [
                packet.uniform_transform.map_or(0, transform_id),
                u32::from(packet.uniform_transform.is_some()),
                ZERO_AC_HF_GLOBAL,
                sharpness_offset,
            ],
            quantization: [
                packet.global_scale,
                packet.quant_lf,
                u32::from(self.extra_precision),
                0,
            ],
            streams: [
                self.lf_stream_index,
                self.hf_stream_index,
                separate_sections,
                u32::try_from(packet.profile.group_count).map_err(|_| {
                    BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "pass-group count",
                    }
                })?,
            ],
            scratch: [self.predictor_width_capacity()?, 0, 0, 0],
        })
    }

    pub fn lf_stage_control(
        &self,
        packet: &BoundedVarDctPacketPlan,
    ) -> Result<VarDctPacketControl, BoundedVarDctPacketError> {
        let mut control = self.packet_control(packet)?;
        control.section_bits[0] = self.lf_entropy_bit_offset;
        control.quantization[3] = 0;
        Ok(control)
    }

    pub fn hf_stage_control(
        &self,
        packet: &BoundedVarDctPacketPlan,
        continuation: &BoundedHfMetadataContinuation,
    ) -> Result<VarDctPacketControl, BoundedVarDctPacketError> {
        let mut control = self.packet_control(packet)?;
        control.section_bits[2] = continuation.token_bit_offset;
        control.quantization[3] = continuation.block_count;
        Ok(control)
    }

    fn predictor_width_capacity(&self) -> Result<u32, BoundedVarDctPacketError> {
        let [blocks_x, _] = self.block_extent();
        Ok(blocks_x
            .max(self.task_capacity)
            .max(self.rect.width.div_ceil(64)))
    }

    fn correlation_samples(&self) -> Result<u32, BoundedVarDctPacketError> {
        self.rect
            .width
            .div_ceil(64)
            .checked_mul(self.rect.height.div_ceil(64))
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "correlation sample count",
            })
    }
}

#[derive(Clone, Debug)]
enum HfDequantMatrixEncoding {
    Default,
    Hornuss([[f32; 3]; 3]),
    Dct2([[f32; 6]; 3]),
    Dct4 {
        params: [[f32; 2]; 3],
        dct_params: [Vec<f32>; 3],
    },
    Dct4x8 {
        params: [[f32; 1]; 3],
        dct_params: [Vec<f32>; 3],
    },
    Afv {
        params: [[f32; 9]; 3],
        dct_params: [Vec<f32>; 3],
        dct4x4_params: [Vec<f32>; 3],
    },
    Dct([Vec<f32>; 3]),
}

fn parse_hf_dequant_matrices(
    reader: &mut impl BitInput,
) -> Result<Option<Vec<[u32; 4]>>, BoundedVarDctPacketError> {
    if metadata_bool(reader, "HF dequantization matrix defaults")? {
        return Ok(None);
    }

    fn read_fixed<const N: usize>(
        reader: &mut impl BitInput,
        matrix: usize,
    ) -> Result<[[f32; N]; 3], BoundedVarDctPacketError> {
        let mut output = [[0.0; N]; 3];
        for value in output.iter_mut().flatten() {
            *value = metadata_f16(reader, "HF dequantization matrix parameter")?;
            if !value.is_finite() {
                return Err(BoundedVarDctPacketError::HfDequantMatrixValue {
                    matrix,
                    reason: "parameter is not finite",
                });
            }
        }
        Ok(output)
    }

    fn read_dct_params(
        reader: &mut impl BitInput,
        matrix: usize,
    ) -> Result<[Vec<f32>; 3], BoundedVarDctPacketError> {
        let count = usize::try_from(metadata_bits(reader, "HF dequantization band count", 4)?)
            .map_err(|_| BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF dequantization band count",
            })?
            .checked_add(1)
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF dequantization band count",
            })?;
        let mut params = std::array::from_fn(|_| vec![0.0; count]);
        for value in params.iter_mut().flatten() {
            *value = metadata_f16(reader, "HF dequantization DCT band")?;
            if !value.is_finite() {
                return Err(BoundedVarDctPacketError::HfDequantMatrixValue {
                    matrix,
                    reason: "DCT band is not finite",
                });
            }
        }
        for first in params.iter_mut().filter_map(|values| values.first_mut()) {
            *first *= 64.0;
        }
        Ok(params)
    }

    let mut encodings = Vec::with_capacity(17);
    for matrix in 0..17 {
        let encoding = u8::try_from(metadata_bits(
            reader,
            "HF dequantization matrix encoding",
            3,
        )?)
        .map_err(|_| BoundedVarDctPacketError::ArithmeticOverflow {
            field: "HF dequantization matrix encoding",
        })?;
        if (1..=5).contains(&encoding) && !matches!(matrix, 0 | 1 | 2 | 3 | 9 | 10) {
            return Err(BoundedVarDctPacketError::HfDequantMatrixEncoding { matrix, encoding });
        }
        let encoding = match encoding {
            0 => HfDequantMatrixEncoding::Default,
            1 => HfDequantMatrixEncoding::Hornuss(read_fixed(reader, matrix)?),
            2 => HfDequantMatrixEncoding::Dct2(read_fixed(reader, matrix)?),
            3 => HfDequantMatrixEncoding::Dct4 {
                params: read_fixed(reader, matrix)?,
                dct_params: read_dct_params(reader, matrix)?,
            },
            4 => HfDequantMatrixEncoding::Dct4x8 {
                params: read_fixed(reader, matrix)?,
                dct_params: read_dct_params(reader, matrix)?,
            },
            5 => {
                let mut params = read_fixed::<9>(reader, matrix)?;
                for channel in &mut params {
                    for value in &mut channel[..6] {
                        *value *= 64.0;
                    }
                }
                HfDequantMatrixEncoding::Afv {
                    params,
                    dct_params: read_dct_params(reader, matrix)?,
                    dct4x4_params: read_dct_params(reader, matrix)?,
                }
            }
            6 => HfDequantMatrixEncoding::Dct(read_dct_params(reader, matrix)?),
            7 => return Err(BoundedVarDctPacketError::RawHfDequantMatrix { matrix }),
            _ => unreachable!("three-bit encoding is in 0..=7"),
        };
        encodings.push(encoding);
    }
    expand_hf_dequant_matrices(&encodings).map(Some)
}

fn expand_hf_dequant_matrices(
    encodings: &[HfDequantMatrixEncoding],
) -> Result<Vec<[u32; 4]>, BoundedVarDctPacketError> {
    fn matrix_param_index(transform: TransformKind) -> usize {
        match transform {
            TransformKind::Dct8 => 0,
            TransformKind::Hornuss => 1,
            TransformKind::Dct2x2 => 2,
            TransformKind::Dct4x4 => 3,
            TransformKind::Dct16x16 => 4,
            TransformKind::Dct32x32 => 5,
            TransformKind::Dct16x8 | TransformKind::Dct8x16 => 6,
            TransformKind::Dct32x8 | TransformKind::Dct8x32 => 7,
            TransformKind::Dct32x16 | TransformKind::Dct16x32 => 8,
            TransformKind::Dct4x8 | TransformKind::Dct8x4 => 9,
            TransformKind::Afv0
            | TransformKind::Afv1
            | TransformKind::Afv2
            | TransformKind::Afv3 => 10,
            TransformKind::Dct64x64 => 11,
            TransformKind::Dct64x32 | TransformKind::Dct32x64 => 12,
            TransformKind::Dct128x128 => 13,
            TransformKind::Dct128x64 | TransformKind::Dct64x128 => 14,
            TransformKind::Dct256x256 => 15,
            TransformKind::Dct256x128 | TransformKind::Dct128x256 => 16,
        }
    }

    fn representative(index: usize) -> TransformKind {
        [
            TransformKind::Dct8,
            TransformKind::Hornuss,
            TransformKind::Dct2x2,
            TransformKind::Dct4x4,
            TransformKind::Dct16x16,
            TransformKind::Dct32x32,
            TransformKind::Dct8x16,
            TransformKind::Dct8x32,
            TransformKind::Dct16x32,
            TransformKind::Dct4x8,
            TransformKind::Afv0,
            TransformKind::Dct64x64,
            TransformKind::Dct32x64,
            TransformKind::Dct128x128,
            TransformKind::Dct64x128,
            TransformKind::Dct256x256,
            TransformKind::Dct128x256,
        ][index]
    }

    fn interpolate(pos: f32, max: f32, bands: &[f32]) -> f32 {
        if let [value] = bands {
            return *value;
        }
        let scaled = pos * (bands.len() - 1) as f32 / max;
        let index = (scaled as usize).min(bands.len() - 2);
        let fraction = scaled - index as f32;
        let left = bands[index];
        let right = bands[index + 1];
        left * (right / left).powf(fraction)
    }

    fn multiplier(value: f32) -> f32 {
        if value > 0.0 {
            1.0 + value
        } else {
            1.0 / (1.0 - value)
        }
    }

    fn dct_weights(
        params: &[f32],
        width: u32,
        height: u32,
        matrix: usize,
    ) -> Result<Vec<f32>, BoundedVarDctPacketError> {
        let mut bands = Vec::with_capacity(params.len());
        let mut last = *params
            .first()
            .ok_or(BoundedVarDctPacketError::HfDequantMatrixValue {
                matrix,
                reason: "DCT matrix has no bands",
            })?;
        bands.push(last);
        for &value in &params[1..] {
            last *= multiplier(value);
            if !last.is_finite() || last <= 0.0 {
                return Err(BoundedVarDctPacketError::HfDequantMatrixValue {
                    matrix,
                    reason: "DCT band is non-positive or non-finite",
                });
            }
            bands.push(last);
        }
        let mut output = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 / (width - 1) as f32;
                let dy = y as f32 / (height - 1) as f32;
                output.push(interpolate(
                    (dx * dx + dy * dy).sqrt(),
                    std::f32::consts::SQRT_2 + 1e-6,
                    &bands,
                ));
            }
        }
        Ok(output)
    }

    fn expand(
        encoding: &HfDequantMatrixEncoding,
        transform: TransformKind,
        matrix: usize,
    ) -> Result<[Vec<f32>; 3], BoundedVarDctPacketError> {
        let output = match encoding {
            HfDequantMatrixEncoding::Default => unreachable!("defaults are expanded separately"),
            HfDequantMatrixEncoding::Dct(params) => {
                let extent = transform.pixel_extent();
                [
                    dct_weights(&params[0], extent.width, extent.height, matrix)?,
                    dct_weights(&params[1], extent.width, extent.height, matrix)?,
                    dct_weights(&params[2], extent.width, extent.height, matrix)?,
                ]
            }
            HfDequantMatrixEncoding::Hornuss(params) => params.map(|params| {
                let mut values = vec![params[0]; 64];
                values[0] = 1.0;
                values[1] = params[1];
                values[8] = params[1];
                values[9] = params[2];
                values
            }),
            HfDequantMatrixEncoding::Dct2(params) => params.map(|params| {
                let mut values = vec![0.0; 64];
                values[0] = 1.0;
                for (index, value) in params.into_iter().enumerate() {
                    let shift = index / 2;
                    let dimension = 1_usize << shift;
                    if index % 2 == 0 {
                        for y in 0..dimension {
                            for x in dimension..dimension * 2 {
                                values[y * 8 + x] = value;
                                values[x * 8 + y] = value;
                            }
                        }
                    } else {
                        for y in dimension..dimension * 2 {
                            for x in dimension..dimension * 2 {
                                values[y * 8 + x] = value;
                            }
                        }
                    }
                }
                values
            }),
            HfDequantMatrixEncoding::Dct4 { params, dct_params } => {
                let mut output = [Vec::new(), Vec::new(), Vec::new()];
                for (output, (params, dct)) in output.iter_mut().zip(params.iter().zip(dct_params))
                {
                    let matrix = dct_weights(dct, 4, 4, matrix)?;
                    *output = vec![0.0; 64];
                    for y in 0..4 {
                        for x in 0..4 {
                            output[y * 16 + x * 2] = matrix[y * 4 + x];
                            output[y * 16 + x * 2 + 1] = matrix[y * 4 + x];
                            output[(y * 2 + 1) * 8 + x * 2] = matrix[y * 4 + x];
                            output[(y * 2 + 1) * 8 + x * 2 + 1] = matrix[y * 4 + x];
                        }
                    }
                    output[1] /= params[0];
                    output[8] /= params[0];
                    output[9] /= params[1];
                }
                output
            }
            HfDequantMatrixEncoding::Dct4x8 { params, dct_params } => {
                let mut output = [Vec::new(), Vec::new(), Vec::new()];
                for (output, (params, dct)) in output.iter_mut().zip(params.iter().zip(dct_params))
                {
                    let matrix = dct_weights(dct, 8, 4, matrix)?;
                    *output = matrix
                        .chunks_exact(8)
                        .flat_map(|row| [row, row])
                        .flatten()
                        .copied()
                        .collect();
                    output[8] /= params[0];
                }
                output
            }
            HfDequantMatrixEncoding::Afv {
                params,
                dct_params,
                dct4x4_params,
            } => {
                const FREQUENCIES: [f32; 16] = [
                    0.0, 0.0, 0.8517779, 5.3777843, 0.0, 0.0, 4.734748, 5.4492455, 1.659827, 4.0,
                    7.275749, 10.423227, 2.6629324, 7.6306577, 8.962389, 12.971662,
                ];
                let mut output = [Vec::new(), Vec::new(), Vec::new()];
                for (output, ((params, dct), dct4)) in output
                    .iter_mut()
                    .zip(params.iter().zip(dct_params).zip(dct4x4_params))
                {
                    let weights_4x8 = dct_weights(dct, 8, 4, matrix)?;
                    let weights_4x4 = dct_weights(dct4, 4, 4, matrix)?;
                    let mut bands = [params[5], 0.0, 0.0, 0.0];
                    for index in 1..4 {
                        bands[index] = bands[index - 1] * multiplier(params[index + 5]);
                    }
                    *output = vec![0.0; 64];
                    for y in 0..4 {
                        for x in 0..4 {
                            output[16 * y + 2 * x] = match (x, y) {
                                (0, 0) => 1.0,
                                (0, 1) => params[2],
                                (1, 0) => params[3],
                                (1, 1) => params[4],
                                _ => interpolate(
                                    FREQUENCIES[y * 4 + x] - FREQUENCIES[2],
                                    FREQUENCIES[15] - FREQUENCIES[2] + 1e-6,
                                    &bands,
                                ),
                            };
                        }
                    }
                    for (y, ((rows, weights_8), weights_4)) in output
                        .chunks_exact_mut(16)
                        .zip(weights_4x8.chunks_exact(8))
                        .zip(weights_4x4.chunks_exact(4))
                        .enumerate()
                    {
                        let (row0, row1) = rows.split_at_mut(8);
                        for (x, (value, &weight)) in row1.iter_mut().zip(weights_8).enumerate() {
                            *value = if y == 0 && x == 0 { params[0] } else { weight };
                        }
                        for (x, (pair, &weight)) in
                            row0.chunks_exact_mut(2).zip(weights_4).enumerate()
                        {
                            pair[1] = if y == 0 && x == 0 { params[1] } else { weight };
                        }
                    }
                }
                output
            }
        };
        let mut output = output;
        for value in output.iter_mut().flatten() {
            *value = 1.0 / *value;
            if !value.is_finite() || *value <= 0.0 || *value >= 1e8 {
                return Err(BoundedVarDctPacketError::HfDequantMatrixValue {
                    matrix,
                    reason: "expanded value is non-positive, non-finite, or too large",
                });
            }
        }
        Ok(output)
    }

    fn transpose(channels: &[Vec<f32>; 3], width: u32, height: u32) -> [Vec<f32>; 3] {
        std::array::from_fn(|channel| {
            let mut output = vec![0.0; channels[channel].len()];
            for y in 0..height {
                for x in 0..width {
                    output[(x * height + y) as usize] = channels[channel][(y * width + x) as usize];
                }
            }
            output
        })
    }

    let encoded_default = [1_u8];
    let mut default_bits = jxl_bitstream::Bitstream::new(&encoded_default);
    let pool = jxl_threadpool::JxlThreadPool::none();
    let defaults = DequantMatrixSet::parse(
        &mut default_bits,
        DequantMatrixSetParams::new(8, 1, None, None, &pool),
    )
    .map_err(|_| BoundedVarDctPacketError::HfDequantMatrixValue {
        matrix: 0,
        reason: "normative defaults could not be constructed",
    })?;
    let mut packed = Vec::new();
    for transform in TransformKind::ALL {
        let index = matrix_param_index(transform);
        let extent = transform.pixel_extent();
        let channels = match &encodings[index] {
            HfDequantMatrixEncoding::Default => {
                let transform_type = crate::vardct_resource::vardct_transform_type(transform);
                std::array::from_fn(|channel| {
                    if transform.needs_transpose() {
                        defaults.get_transposed(channel, transform_type).to_vec()
                    } else {
                        defaults.get(channel, transform_type).to_vec()
                    }
                })
            }
            encoding => {
                let representative = representative(index);
                let channels = expand(encoding, representative, index)?;
                if transform.needs_transpose() {
                    let representative_extent = representative.pixel_extent();
                    transpose(
                        &channels,
                        representative_extent.width,
                        representative_extent.height,
                    )
                } else {
                    channels
                }
            }
        };
        let matrix_len = usize::try_from(extent.width.checked_mul(extent.height).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF dequantization matrix area",
            },
        )?)
        .map_err(|_| BoundedVarDctPacketError::ArithmeticOverflow {
            field: "HF dequantization matrix area",
        })?;
        if channels.iter().any(|channel| channel.len() != matrix_len) {
            return Err(BoundedVarDctPacketError::HfDequantMatrixValue {
                matrix: index,
                reason: "expanded matrix dimensions do not match the transform",
            });
        }
        let base = packed.len();
        packed.resize(
            base.checked_add(matrix_len)
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "HF dequantization matrix packing",
                })?,
            [0; 4],
        );
        for frequency_y in 0..extent.height {
            for frequency_x in 0..extent.width {
                let raster = (frequency_y * extent.width + frequency_x) as usize;
                let backend_index = if transform.is_special() || extent.height < extent.width {
                    frequency_y * extent.width + frequency_x
                } else {
                    frequency_x * extent.height + frequency_y
                } as usize;
                packed[base + backend_index] = [
                    channels[0][raster].to_bits(),
                    channels[1][raster].to_bits(),
                    channels[2][raster].to_bits(),
                    0,
                ];
            }
        }
    }
    Ok(packed)
}

impl HfCoefficientEntropyPlan {
    /// Parses HF-global tables without consuming a coefficient symbol. Pass-group ranges remain
    /// exact views into the caller-owned codestream for the GPU executor.
    #[cfg(test)]
    fn parse(
        codestream: &[u8],
        packet: BitRange,
        group_count: u32,
        block_context: &HfBlockContextIr,
        pass_groups: Vec<BitRange>,
        decoded_symbol_limit: u32,
    ) -> Result<Self, BoundedVarDctPacketError> {
        Self::parse_inner(
            PacketSource::Slice(codestream),
            packet,
            group_count,
            block_context,
            pass_groups,
            decoded_symbol_limit,
        )
    }

    fn parse_inner(
        source: PacketSource<'_>,
        packet: BitRange,
        group_count: u32,
        block_context: &HfBlockContextIr,
        pass_groups: Vec<BitRange>,
        decoded_symbol_limit: u32,
    ) -> Result<Self, BoundedVarDctPacketError> {
        Self::parse_inner_with_tail(
            source,
            packet,
            group_count,
            block_context,
            pass_groups,
            decoded_symbol_limit,
            false,
        )
    }

    fn parse_single_tail_inner(
        source: PacketSource<'_>,
        packet: BitRange,
        group_count: u32,
        block_context: &HfBlockContextIr,
        decoded_symbol_limit: u32,
    ) -> Result<Self, BoundedVarDctPacketError> {
        Self::parse_inner_with_tail(
            source,
            packet,
            group_count,
            block_context,
            Vec::new(),
            decoded_symbol_limit,
            true,
        )
    }

    fn parse_inner_with_tail(
        source: PacketSource<'_>,
        packet: BitRange,
        group_count: u32,
        block_context: &HfBlockContextIr,
        mut pass_groups: Vec<BitRange>,
        decoded_symbol_limit: u32,
        trailing_pass_group: bool,
    ) -> Result<Self, BoundedVarDctPacketError> {
        let packet_end = packet
            .end()
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF-global packet end",
            })?;
        validate_source_packet_end(source, packet_end)?;
        let mut reader = source_reader_at(source, packet.offset)?;
        let mut reader = BoundedBitInput::new(&mut reader, packet_end);
        let dequant_matrix_words = parse_hf_dequant_matrices(&mut reader)?;
        let prefix =
            HfGlobalPrefix::parse_after_dequant_reader(&mut reader, packet_end, group_count)?;
        let (coefficient_entropy_bit_offset, order_words, order_coordinate_offset_words) =
            parse_coefficient_orders_reader(&mut reader, prefix, packet_end)?;
        let lf_context_count = block_context
            .lf_thresholds
            .iter()
            .try_fold(1usize, |count, thresholds| {
                count.checked_mul(thresholds.len().checked_add(1)?)
            });
        let expected_block_contexts = lf_context_count
            .and_then(|count| count.checked_mul(block_context.qf_thresholds.len().checked_add(1)?))
            .and_then(|count| count.checked_mul(3 * 13))
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF block-context map length",
            })?;
        if block_context.block_context_map.len() != expected_block_contexts {
            return Err(BoundedVarDctPacketError::HfBlockContextMapLength {
                expected: expected_block_contexts,
                actual: block_context.block_context_map.len(),
            });
        }
        let block_cluster_count = block_context.num_block_clusters;
        let context_count = 495u32
            .checked_mul(prefix.num_hf_presets)
            .and_then(|count| count.checked_mul(block_cluster_count))
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF coefficient context count",
            })?;
        let context_count = usize::try_from(context_count).map_err(|_| {
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF coefficient context count",
            }
        })?;
        if coefficient_entropy_bit_offset > packet_end {
            return Err(VarDctPacketError::PacketBoundary {
                cursor: coefficient_entropy_bit_offset,
                packet_end,
            }
            .into());
        }
        let mut descriptor_reader = source_reader_at(source, coefficient_entropy_bit_offset)?;
        let mut descriptor_reader = BoundedBitInput::new(&mut descriptor_reader, packet_end);
        let descriptor = EntropyDecoderIr::parse(
            &mut descriptor_reader,
            context_count,
            MaTreeLimits::default(),
        )
        .map_err(map_modular_reader_error)?;
        let descriptor_end = descriptor_reader.bit_offset();
        if descriptor_end > packet_end {
            return Err(VarDctPacketError::PacketBoundary {
                cursor: descriptor_end,
                packet_end,
            }
            .into());
        }
        let remaining = packet_end - descriptor_end;
        if trailing_pass_group {
            pass_groups.push(BitRange {
                offset: descriptor_end,
                length: remaining,
            });
        } else if remaining > 7
            || descriptor_reader
                .read_bits(remaining as u8)
                .map_err(map_modular_reader_error)?
                != 0
        {
            return Err(BoundedVarDctPacketError::HfGlobalTrailingBits { bits: remaining });
        }
        let lz77_window_words = descriptor
            .lz77_window_words(0, decoded_symbol_limit)
            .map_err(|error| BoundedVarDctPacketError::ModularTree(error.to_string()))?;
        let context_map = descriptor
            .context_to_cluster
            .get(..context_count)
            .ok_or(BoundedVarDctPacketError::PackedMetadata)?
            .iter()
            .map(|&cluster| u32::from(cluster))
            .collect();
        let block_context_map = block_context
            .block_context_map
            .iter()
            .map(|&cluster| u32::from(cluster))
            .collect();
        let PackedModularMetadata { words } = descriptor
            .pack_gpu_metadata()
            .map_err(|error| BoundedVarDctPacketError::ModularTree(error.to_string()))?;
        Ok(Self {
            num_hf_presets: prefix.num_hf_presets,
            num_block_clusters: block_cluster_count,
            metadata: words,
            context_map,
            block_context_map,
            qf_thresholds: block_context.qf_thresholds.clone(),
            lf_thresholds: block_context.lf_thresholds.clone(),
            order_words,
            order_coordinate_offset_words,
            pass_groups,
            lz77_window_words,
            dequant_matrix_words,
        })
    }
}

fn parse_coefficient_orders_reader(
    reader: &mut impl BitInput,
    prefix: HfGlobalPrefix,
    packet_end: u64,
) -> Result<(u64, Vec<u32>, u32), BoundedVarDctPacketError> {
    use crate::vardct_artifact::{
        GpuHfOrderDescriptor, HF_ORDER_CHANNELS, HF_ORDER_COUNT, HF_ORDER_EXTENTS,
    };

    const DESCRIPTOR_WORDS: u32 = (HF_ORDER_COUNT * HF_ORDER_CHANNELS * 4) as u32;

    let mut bitstream = BoundedBitInput::new(reader, packet_end);
    let decoder = (prefix.used_orders != 0)
        .then(|| {
            EntropyDecoderIr::parse(&mut bitstream, 8, MaTreeLimits::default())
                .map_err(coefficient_reader_error)
        })
        .transpose()?;
    let mut decoder_cursor = decoder.as_ref().map(|decoder| {
        MetadataEntropyCursor::new(decoder, MaTreeLimits::default().metadata_symbol_limit)
    });
    if let Some(cursor) = decoder_cursor.as_mut() {
        cursor
            .begin(&mut bitstream)
            .map_err(coefficient_reader_error)?;
    }

    let mut descriptors = [GpuHfOrderDescriptor::zeroed(); HF_ORDER_COUNT * HF_ORDER_CHANNELS];
    let mut coordinates = Vec::new();
    for (order_id, [width, height]) in HF_ORDER_EXTENTS.into_iter().enumerate() {
        let natural = natural_coefficient_order(width, height)?;
        let len = u32::try_from(natural.len()).map_err(|_| {
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF order length",
            }
        })?;
        if prefix.used_orders & (1 << order_id) == 0 {
            let offset = u32::try_from(coordinates.len()).map_err(|_| {
                BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "HF order coordinate offset",
                }
            })?;
            coordinates.extend_from_slice(&natural);
            let descriptor = GpuHfOrderDescriptor {
                offset,
                len,
                width,
                height,
            };
            descriptors[order_id * HF_ORDER_CHANNELS..(order_id + 1) * HF_ORDER_CHANNELS]
                .fill(descriptor);
            continue;
        }

        let cursor = decoder_cursor
            .as_mut()
            .ok_or(BoundedVarDctPacketError::PackedMetadata)?;
        let skip = len / 64;
        for channel in 0..HF_ORDER_CHANNELS {
            let end = cursor
                .read_varint(&mut bitstream, coefficient_order_context(len), 0)
                .map_err(coefficient_reader_error)?;
            let permutation = if end > len - skip {
                return Err(BoundedVarDctPacketError::CoefficientOrderCoding(
                    jxl_coding::Error::InvalidPermutation,
                ));
            } else {
                let mut lehmer = Vec::with_capacity(end as usize);
                let mut previous = 0u32;
                for index in 0..end {
                    let value = cursor
                        .read_varint(&mut bitstream, coefficient_order_context(previous), 0)
                        .map_err(coefficient_reader_error)?;
                    if value >= len - skip - index {
                        return Err(BoundedVarDctPacketError::CoefficientOrderCoding(
                            jxl_coding::Error::InvalidPermutation,
                        ));
                    }
                    previous = value;
                    lehmer.push(value);
                }
                let mut temp = (skip as usize..len as usize).collect::<Vec<_>>();
                let mut permutation = (0..skip as usize).collect::<Vec<_>>();
                for index in lehmer {
                    permutation.push(temp.remove(index as usize));
                }
                permutation.extend(temp);
                permutation
            };
            let offset = u32::try_from(coordinates.len()).map_err(|_| {
                BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "HF order coordinate offset",
                }
            })?;
            coordinates.extend(permutation.into_iter().map(|index| natural[index]));
            descriptors[order_id * HF_ORDER_CHANNELS + channel] = GpuHfOrderDescriptor {
                offset,
                len,
                width,
                height,
            };
        }
    }
    if let Some(cursor) = &decoder_cursor {
        cursor.finalize().map_err(coefficient_reader_error)?;
    }
    let coefficient_entropy_bit_offset = bitstream.bit_offset();

    let descriptor_words = bytemuck::cast_slice::<GpuHfOrderDescriptor, u32>(&descriptors);
    let mut order_words = Vec::with_capacity(descriptor_words.len() + coordinates.len());
    order_words.extend_from_slice(descriptor_words);
    order_words.extend_from_slice(&coordinates);
    Ok((
        coefficient_entropy_bit_offset,
        order_words,
        DESCRIPTOR_WORDS,
    ))
}

fn coefficient_order_context(value: u32) -> usize {
    let bits = if value >= 0x8000_0000 {
        32
    } else {
        (value + 1).next_power_of_two().trailing_zeros()
    };
    usize::try_from(bits.min(7)).unwrap_or(7)
}

fn coefficient_reader_error(source: crate::Error) -> BoundedVarDctPacketError {
    match source {
        crate::Error::Bitstream(source) => BoundedVarDctPacketError::CoefficientOrderBitstream(
            crate::vardct_frontend::map_gpu_bitstream_error(source),
        ),
        source => {
            BoundedVarDctPacketError::CoefficientOrderReader(map_metadata_reader_error(source))
        }
    }
}

fn natural_coefficient_order(
    width: u32,
    height: u32,
) -> Result<Vec<u32>, BoundedVarDctPacketError> {
    let area = width
        .checked_mul(height)
        .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
            field: "natural HF order area",
        })?;
    let capacity =
        usize::try_from(area).map_err(|_| BoundedVarDctPacketError::ArithmeticOverflow {
            field: "natural HF order capacity",
        })?;
    let y_scale = width / height;
    let low_width = width / 8;
    let low_height = height / 8;
    let mut coordinates = Vec::with_capacity(capacity);
    for index in 0..low_width * low_height {
        coordinates.push((index % low_width) | ((index / low_width) << 16));
    }
    for distance in 1..2 * width {
        let margin = distance.saturating_sub(width);
        for order in margin..distance - margin {
            let (x, y) = if distance & 1 == 1 {
                (order, distance - 1 - order)
            } else {
                (distance - 1 - order, order)
            };
            if x < low_width && y < low_width || y % y_scale != 0 {
                continue;
            }
            coordinates.push(x | ((y / y_scale) << 16));
        }
    }
    if coordinates.len() != capacity {
        return Err(BoundedVarDctPacketError::PackedMetadata);
    }
    Ok(coordinates)
}

/// Exact 128-byte uniform consumed by `vardct_packet.wgsl`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VarDctPacketControl {
    pub section_bits: [u32; 4],
    pub geometry: [u32; 4],
    pub offsets: [u32; 4],
    pub capacities: [u32; 4],
    pub expected: [u32; 4],
    pub quantization: [u32; 4],
    pub streams: [u32; 4],
    /// Maximum predictor row width followed by reserved words.
    pub scratch: [u32; 4],
}

/// Generic Modular parameter ABI retained by the composable entropy/reconstruction fragments.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VarDctModularParams {
    entropy: EntropyStreamParams,
    window_logical_start: u32,
    window_upload_start: u32,
    stream_token_end: u32,
    window_yield_end: u32,
    window_flags: u32,
    entropy_state_offset: u32,
    stream_base_bit: u32,
    consumer_words: [u32; 49],
    _reserved: u32,
}

/// Common 56-byte prefix of both packet resume records.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PacketExecutionStatePrefix {
    entropy: [u32; 8],
    packet_phase: u32,
    lf_decoded: u32,
    hf_decoded: u32,
    first_blocks: u32,
    extra_precision: u32,
    predictor_prev_grad: i32,
}

/// Exact 64-byte packet state for generic MA prediction without SelfCorrecting rows.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GenericPacketExecutionState {
    prefix: PacketExecutionStatePrefix,
    _reserved: [u32; 2],
}

/// Exact 128-byte packet state including every SelfCorrecting predictor accumulator.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct WeightedPacketExecutionState {
    prefix: PacketExecutionStatePrefix,
    wp_true_errors: [i32; 4],
    wp_subpred_nw_ww: [u32; 4],
    wp_subpred_n_w: [u32; 4],
    wp_subpred_ne: [u32; 4],
    _reserved: [u32; 2],
}

#[must_use]
pub(crate) const fn packet_execution_state_bytes(needs_self_correcting: bool) -> u64 {
    if needs_self_correcting {
        WEIGHTED_PACKET_EXECUTION_STATE_BYTES
    } else {
        GENERIC_PACKET_EXECUTION_STATE_BYTES
    }
}

impl Default for VarDctModularParams {
    fn default() -> Self {
        let mut consumer_words = [0; 49];
        consumer_words[38] = 16;
        consumer_words[39] = 10;
        consumer_words[40] = 7;
        consumer_words[41] = 7;
        consumer_words[42] = 7;
        consumer_words[45] = 13;
        consumer_words[46] = 12;
        consumer_words[47] = 12;
        consumer_words[48] = 12;
        Self {
            entropy: EntropyStreamParams::default(),
            window_logical_start: 0,
            window_upload_start: 0,
            stream_token_end: 0,
            window_yield_end: 0,
            window_flags: 0,
            entropy_state_offset: 0,
            stream_base_bit: 0,
            consumer_words,
            _reserved: 0,
        }
    }
}

impl VarDctModularParams {
    /// Sets the exact power-of-two LZ ring represented by the packed entropy descriptor.
    pub fn with_lz77_window(mut self, words: u32) -> Self {
        self.entropy.lz77_window_mask = words.saturating_sub(1);
        self
    }

    /// Enables the weighted self-correcting predictor declared by the MA tree.
    pub fn with_self_correcting(mut self, enabled: bool) -> Self {
        self.consumer_words[9] = u32::from(enabled);
        self
    }

    /// Rebinds the packet entropy consumer to one relative, overlapping upload segment.
    #[must_use]
    pub(crate) fn with_stream_segment(
        mut self,
        segment: GroupStreamSegment,
        stream_base_bit: u32,
        entropy_state_offset: u32,
    ) -> Self {
        self.entropy.token_start = 0;
        self.entropy.token_end = segment.available_token_end;
        self.window_logical_start = segment.window_logical_start;
        self.window_upload_start = segment.window_upload_start;
        self.stream_token_end = segment.stream_token_end;
        self.window_yield_end = segment.window_yield_end;
        self.window_flags = segment.flags | PACKET_WINDOW_ENABLED;
        self.entropy_state_offset = entropy_state_offset;
        self.stream_base_bit = stream_base_bit;
        self
    }

    #[cfg(test)]
    fn window_contract(&self) -> [u32; 7] {
        [
            self.window_logical_start,
            self.window_upload_start,
            self.stream_token_end,
            self.window_yield_end,
            self.window_flags,
            self.entropy_state_offset,
            self.stream_base_bit,
        ]
    }
}

/// Exact 64-byte status written once by the serial packet parser.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuVarDctPacketStatus {
    pub code: u32,
    pub cursor: u32,
    pub expected_end: u32,
    pub lf_decoded: u32,
    pub hf_decoded: u32,
    pub strategy: u32,
    pub hf_mul: u32,
    pub coefficient_words: u32,
    pub detail: u32,
    pub global_scale: u32,
    pub quant_lf: u32,
    pub first_blocks: u32,
    pub extra_precision: u32,
    pub _reserved: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidatedVarDctPacket {
    pub first_blocks: u32,
}

/// Host-known values used to validate the authoritative GPU packet status.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VarDctPacketValidation {
    pub expected_strategy: Option<TransformKind>,
    pub expected_lf_samples: u32,
    pub block_count: u32,
    pub correlation_samples: u32,
    pub task_capacity: u32,
    pub expected_global_scale: u32,
    pub expected_quant_lf: u32,
    pub expected_extra_precision: u8,
}

impl GpuVarDctPacketStatus {
    /// Validates the synchronization record produced by the LF-only staging entry point.
    pub fn validate_lf_stage(
        self,
        expected_lf_samples: u32,
        expected_global_scale: u32,
        expected_quant_lf: u32,
        expected_extra_precision: u8,
    ) -> Result<u32, GpuVarDctPacketError> {
        match self.code {
            30 if self.cursor < self.expected_end
                && self.lf_decoded == expected_lf_samples
                && self.hf_decoded == 0
                && self.global_scale == expected_global_scale
                && self.quant_lf == expected_quant_lf
                && self.extra_precision == u32::from(expected_extra_precision) =>
            {
                Ok(self.cursor)
            }
            30 => Err(GpuVarDctPacketError::Entropy {
                code: self.code,
                cursor: self.cursor,
                end: self.expected_end,
                lf_decoded: self.lf_decoded,
                hf_decoded: self.hf_decoded,
                detail: self.detail,
            }),
            2..=13 => Err(GpuVarDctPacketError::Entropy {
                code: self.code,
                cursor: self.cursor,
                end: self.expected_end,
                lf_decoded: self.lf_decoded,
                hf_decoded: self.hf_decoded,
                detail: self.detail,
            }),
            code => Err(GpuVarDctPacketError::Unknown { code }),
        }
    }

    /// Validates an HF-metadata stage and returns its following HF-global cursor.
    pub fn validate_hf_metadata_stage(
        self,
        expected: VarDctPacketValidation,
    ) -> Result<u32, GpuVarDctPacketError> {
        let expected_hf_samples = self
            .first_blocks
            .checked_mul(2)
            .and_then(|tasks| expected.block_count.checked_add(tasks))
            .and_then(|samples| {
                expected
                    .correlation_samples
                    .checked_mul(2)
                    .and_then(|cfl| samples.checked_add(cfl))
            });
        let strategy_matches = expected
            .expected_strategy
            .map(|strategy| self.strategy == transform_id(strategy))
            .unwrap_or(true);
        match self.code {
            31 if self.cursor < self.expected_end
                && self.lf_decoded == expected.expected_lf_samples
                && Some(self.hf_decoded) == expected_hf_samples
                && self.first_blocks != 0
                && self.first_blocks <= expected.task_capacity
                && strategy_matches
                && self.hf_mul > 0
                && self.global_scale == expected.expected_global_scale
                && self.quant_lf == expected.expected_quant_lf
                && self.extra_precision == u32::from(expected.expected_extra_precision) =>
            {
                Ok(self.cursor)
            }
            31 | 2..=13 => Err(GpuVarDctPacketError::Entropy {
                code: self.code,
                cursor: self.cursor,
                end: self.expected_end,
                lf_decoded: self.lf_decoded,
                hf_decoded: self.hf_decoded,
                detail: self.detail,
            }),
            21 => Err(GpuVarDctPacketError::FirstBlock),
            24 => Err(GpuVarDctPacketError::Strategy {
                actual: self.detail,
                expected: expected.expected_strategy.map_or(u32::MAX, transform_id),
            }),
            25 => Err(GpuVarDctPacketError::Sharpness { value: self.detail }),
            code => Err(GpuVarDctPacketError::Unknown { code }),
        }
    }

    pub fn validate(
        self,
        expected: VarDctPacketValidation,
    ) -> Result<ValidatedVarDctPacket, GpuVarDctPacketError> {
        let expected_hf_samples = self
            .first_blocks
            .checked_mul(2)
            .and_then(|tasks| expected.block_count.checked_add(tasks))
            .and_then(|samples| {
                expected
                    .correlation_samples
                    .checked_mul(2)
                    .and_then(|cfl| samples.checked_add(cfl))
            });
        let strategy_matches = expected
            .expected_strategy
            .map(|strategy| self.strategy == transform_id(strategy))
            .unwrap_or(true);
        match self.code {
            1 if self.cursor == self.expected_end
                && self.lf_decoded == expected.expected_lf_samples
                && Some(self.hf_decoded) == expected_hf_samples
                && self.first_blocks != 0
                && self.first_blocks <= expected.task_capacity
                && strategy_matches
                && self.hf_mul > 0
                && self.global_scale == expected.expected_global_scale
                && self.quant_lf == expected.expected_quant_lf
                && self.extra_precision == u32::from(expected.expected_extra_precision) =>
            {
                Ok(ValidatedVarDctPacket {
                    first_blocks: self.first_blocks,
                })
            }
            1 => Err(GpuVarDctPacketError::Entropy {
                code: self.code,
                cursor: self.cursor,
                end: self.expected_end,
                lf_decoded: self.lf_decoded,
                hf_decoded: self.hf_decoded,
                detail: self.detail,
            }),
            20 => Err(GpuVarDctPacketError::LfHeader),
            21 => Err(GpuVarDctPacketError::FirstBlock),
            22 => Err(GpuVarDctPacketError::HfHeader),
            24 => Err(GpuVarDctPacketError::Strategy {
                actual: self.detail,
                expected: expected.expected_strategy.map_or(u32::MAX, transform_id),
            }),
            25 => Err(GpuVarDctPacketError::Sharpness { value: self.detail }),
            27 => Err(GpuVarDctPacketError::HfGlobal),
            2..=13 => Err(GpuVarDctPacketError::Entropy {
                code: self.code,
                cursor: self.cursor,
                end: self.expected_end,
                lf_decoded: self.lf_decoded,
                hf_decoded: self.hf_decoded,
                detail: self.detail,
            }),
            code => Err(GpuVarDctPacketError::Unknown { code }),
        }
    }
}

/// Buffers supplied to [`VarDctPacketPipeline::encode`].
pub struct VarDctPacketBuffers<'a> {
    pub codestream: &'a wgpu::Buffer,
    pub modular_metadata: &'a wgpu::Buffer,
    pub reconstructed_lf: &'a wgpu::Buffer,
    pub raw_hf_metadata: &'a wgpu::Buffer,
    pub coefficients: &'a wgpu::Buffer,
    pub status: &'a wgpu::Buffer,
    pub control: &'a wgpu::Buffer,
    pub modular_params: &'a wgpu::Buffer,
}

/// Reusable serial control-plane decoder. Image entropy is decoded in WGSL.
pub struct VarDctPacketPipeline {
    layout: wgpu::BindGroupLayout,
    combined: wgpu::ComputePipeline,
    lf: wgpu::ComputePipeline,
    hf: wgpu::ComputePipeline,
    hf_metadata: wgpu::ComputePipeline,
}

impl VarDctPacketPipeline {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let source = shader_source();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu bounded VarDCT packet frontend"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jxl-wgpu VarDCT packet bindings"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                storage(3, false),
                storage(4, false),
                storage(5, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(7, true),
            ],
        });
        let staged_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jxl-wgpu staged VarDCT packet layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&staged_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        Self {
            layout,
            combined: pipeline(
                "jxl-wgpu bounded VarDCT combined packet frontend",
                "decode_vardct_packet",
            ),
            lf: pipeline("jxl-wgpu bounded VarDCT LF frontend", "decode_vardct_lf"),
            hf: pipeline("jxl-wgpu bounded VarDCT HF frontend", "decode_vardct_hf"),
            hf_metadata: pipeline(
                "jxl-wgpu bounded VarDCT HF-metadata frontend",
                "decode_vardct_hf_metadata",
            ),
        }
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffers: VarDctPacketBuffers<'_>,
    ) {
        self.encode_stage(device, encoder, buffers, &self.combined);
    }

    /// Records only the LF Modular stream. Its status cursor is the host metadata boundary for a
    /// subsequent [`Self::encode_hf`] dispatch.
    pub fn encode_lf(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffers: VarDctPacketBuffers<'_>,
    ) {
        self.encode_stage(device, encoder, buffers, &self.lf);
    }

    /// Continues from host-packed HF metadata while retaining the resident LF reconstruction.
    pub fn encode_hf(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffers: VarDctPacketBuffers<'_>,
    ) {
        self.encode_stage(device, encoder, buffers, &self.hf);
    }

    /// Decodes and validates HF metadata, then stops at the following HF-global boundary.
    pub fn encode_hf_metadata(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffers: VarDctPacketBuffers<'_>,
    ) {
        self.encode_stage(device, encoder, buffers, &self.hf_metadata);
    }

    fn encode_stage(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffers: VarDctPacketBuffers<'_>,
        pipeline: &wgpu::ComputePipeline,
    ) {
        let resources = [
            buffers.codestream,
            buffers.modular_metadata,
            buffers.reconstructed_lf,
            buffers.raw_hf_metadata,
            buffers.coefficients,
            buffers.status,
            buffers.control,
            buffers.modular_params,
        ];
        let entries = resources
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu bounded VarDCT packet bindings"),
            layout: &self.layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu bounded VarDCT packet frontend"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

#[must_use]
pub fn vardct_packet_shader_source() -> String {
    shader_source()
}

fn shader_source() -> String {
    SHADER_TEMPLATE
        .replace(ENTROPY_ABI_MARKER, MODULAR_ENTROPY_ABI)
        .replace(ENTROPY_MARKER, MODULAR_ENTROPY)
        .replace(RECONSTRUCT_MARKER, MODULAR_RECONSTRUCT)
}

fn transform_for_extent(width: u32, height: u32) -> Option<TransformKind> {
    [
        TransformKind::Dct8,
        TransformKind::Dct16x16,
        TransformKind::Dct32x32,
        TransformKind::Dct16x8,
        TransformKind::Dct8x16,
        TransformKind::Dct32x8,
        TransformKind::Dct8x32,
        TransformKind::Dct32x16,
        TransformKind::Dct16x32,
    ]
    .into_iter()
    .find(|transform| {
        let extent = transform.pixel_extent();
        extent.width == width && extent.height == height
    })
}

const fn transform_id(transform: TransformKind) -> u32 {
    let mut index = 0;
    while index < TransformKind::ALL.len() {
        if TransformKind::ALL[index] as u8 == transform as u8 {
            return index as u32;
        }
        index += 1;
    }
    u32::MAX
}

const _: () = {
    assert!(std::mem::size_of::<VarDctPacketControl>() == 128);
    assert!(std::mem::align_of::<VarDctPacketControl>() == 16);
    assert!(std::mem::size_of::<VarDctModularParams>() == 240);
    assert!(std::mem::align_of::<VarDctModularParams>() == 16);
    assert!(std::mem::size_of::<PacketExecutionStatePrefix>() == 56);
    assert!(std::mem::align_of::<PacketExecutionStatePrefix>() == 4);
    assert!(std::mem::offset_of!(PacketExecutionStatePrefix, packet_phase) == 32);
    assert!(std::mem::offset_of!(PacketExecutionStatePrefix, lf_decoded) == 36);
    assert!(std::mem::offset_of!(PacketExecutionStatePrefix, hf_decoded) == 40);
    assert!(std::mem::offset_of!(PacketExecutionStatePrefix, first_blocks) == 44);
    assert!(std::mem::offset_of!(PacketExecutionStatePrefix, extra_precision) == 48);
    assert!(std::mem::offset_of!(PacketExecutionStatePrefix, predictor_prev_grad) == 52);
    assert!(std::mem::size_of::<GenericPacketExecutionState>() == 64);
    assert!(std::mem::align_of::<GenericPacketExecutionState>() == 16);
    assert!(std::mem::size_of::<WeightedPacketExecutionState>() == 128);
    assert!(std::mem::align_of::<WeightedPacketExecutionState>() == 16);
    assert!(GENERIC_PACKET_EXECUTION_STATE_BYTES == 64);
    assert!(WEIGHTED_PACKET_EXECUTION_STATE_BYTES == 128);
    assert!(std::mem::size_of::<GpuVarDctPacketStatus>() == 64);
};

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use jxl_gpu_bitstream::{BitWriter, StreamSlice};

    use super::*;

    fn decode_hex(source: &str) -> Vec<u8> {
        let digits = source
            .bytes()
            .filter(u8::is_ascii_hexdigit)
            .collect::<Vec<_>>();
        digits
            .chunks_exact(2)
            .map(|pair| {
                let high = (pair[0] as char).to_digit(16).unwrap();
                let low = (pair[1] as char).to_digit(16).unwrap();
                ((high << 4) | low) as u8
            })
            .collect()
    }

    fn legacy_coefficient_orders(
        codestream: &[u8],
        prefix: HfGlobalPrefix,
    ) -> (u64, Vec<u32>, u32) {
        use crate::vardct_artifact::{
            GpuHfOrderDescriptor, HF_ORDER_CHANNELS, HF_ORDER_COUNT, HF_ORDER_EXTENTS,
        };

        const DESCRIPTOR_WORDS: u32 = (HF_ORDER_COUNT * HF_ORDER_CHANNELS * 4) as u32;

        let mut bitstream = jxl_bitstream::Bitstream::new(codestream);
        bitstream
            .skip_bits(usize::try_from(prefix.order_entropy_bit_offset).unwrap())
            .unwrap();
        let mut decoder = (prefix.used_orders != 0)
            .then(|| jxl_coding::Decoder::parse(&mut bitstream, 8))
            .transpose()
            .unwrap();
        let mut descriptors = [GpuHfOrderDescriptor::zeroed(); HF_ORDER_COUNT * HF_ORDER_CHANNELS];
        let mut coordinates = Vec::new();
        for (order_id, [width, height]) in HF_ORDER_EXTENTS.into_iter().enumerate() {
            let natural = natural_coefficient_order(width, height).unwrap();
            let len = u32::try_from(natural.len()).unwrap();
            if prefix.used_orders & (1 << order_id) == 0 {
                let offset = u32::try_from(coordinates.len()).unwrap();
                coordinates.extend_from_slice(&natural);
                let descriptor = GpuHfOrderDescriptor {
                    offset,
                    len,
                    width,
                    height,
                };
                descriptors[order_id * HF_ORDER_CHANNELS..(order_id + 1) * HF_ORDER_CHANNELS]
                    .fill(descriptor);
                continue;
            }
            let decoder = decoder.as_mut().unwrap();
            let skip = len / 64;
            for channel in 0..HF_ORDER_CHANNELS {
                let permutation =
                    jxl_coding::read_permutation(&mut bitstream, decoder, len, skip).unwrap();
                let offset = u32::try_from(coordinates.len()).unwrap();
                coordinates.extend(permutation.into_iter().map(|index| natural[index]));
                descriptors[order_id * HF_ORDER_CHANNELS + channel] = GpuHfOrderDescriptor {
                    offset,
                    len,
                    width,
                    height,
                };
            }
        }
        if let Some(decoder) = &mut decoder {
            decoder.finalize().unwrap();
        }
        let coefficient_entropy_bit_offset = u64::try_from(bitstream.num_read_bits()).unwrap();
        let descriptor_words = bytemuck::cast_slice::<GpuHfOrderDescriptor, u32>(&descriptors);
        let mut order_words = Vec::with_capacity(descriptor_words.len() + coordinates.len());
        order_words.extend_from_slice(descriptor_words);
        order_words.extend_from_slice(&coordinates);
        (
            coefficient_entropy_bit_offset,
            order_words,
            DESCRIPTOR_WORDS,
        )
    }

    fn split_source(storage: Arc<[u8]>, split: usize) -> GpuCodestream {
        GpuCodestream::from_spans([
            (
                0,
                StreamSlice::from_shared_range(Arc::clone(&storage), 0..split).unwrap(),
            ),
            (
                split as u64,
                StreamSlice::from_shared_range(Arc::clone(&storage), split..storage.len()).unwrap(),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn packet_shader_is_portable_wgsl() {
        let module = naga::front::wgsl::parse_str(&shader_source()).unwrap();
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator.validate(&module).unwrap();
    }

    #[test]
    fn every_parametric_hf_matrix_mode_matches_the_jxl_vardct_oracle() {
        fn write_f16_ones(writer: &mut BitWriter, count: usize) {
            for _ in 0..count {
                writer.write_bits(0x3c00, 16).unwrap();
            }
        }

        fn write_dct_params(writer: &mut BitWriter) {
            writer.write_bits(0, 4).unwrap(); // one band per channel
            write_f16_ones(writer, 3);
        }

        let modes = [1, 2, 3, 4, 6, 0, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6];
        let mut writer = BitWriter::new();
        writer.write_bits(0, 1).unwrap(); // custom matrix set
        for mode in modes {
            writer.write_bits(mode, 3).unwrap();
            match mode {
                0 => {}
                1 => write_f16_ones(&mut writer, 3 * 3),
                2 => write_f16_ones(&mut writer, 3 * 6),
                3 => {
                    write_f16_ones(&mut writer, 3 * 2);
                    write_dct_params(&mut writer);
                }
                4 => {
                    write_f16_ones(&mut writer, 3);
                    write_dct_params(&mut writer);
                }
                5 => {
                    write_f16_ones(&mut writer, 3 * 9);
                    write_dct_params(&mut writer);
                    write_dct_params(&mut writer);
                }
                6 => write_dct_params(&mut writer),
                _ => unreachable!("test matrix mode is in 0..=6"),
            }
        }
        let bytes = writer.into_bytes();
        let mut reader = jxl_gpu_bitstream::BitReader::new(&bytes);

        let words = parse_hf_dequant_matrices(&mut reader)
            .unwrap()
            .expect("custom set produces an explicit resource payload");
        let layout = crate::vardct_resource::VarDctResourceLayout::new(1, 1, 1).unwrap();
        layout.validate_dequant_matrix_words(&words).unwrap();

        let mut oracle_bits = jxl_bitstream::Bitstream::new(&bytes);
        let pool = jxl_threadpool::JxlThreadPool::none();
        let oracle = DequantMatrixSet::parse(
            &mut oracle_bits,
            DequantMatrixSetParams::new(8, 1, None, None, &pool),
        )
        .unwrap();
        let mut expected = Vec::with_capacity(words.len());
        for transform in TransformKind::ALL {
            let transform_type = crate::vardct_resource::vardct_transform_type(transform);
            let channels: [&[f32]; 3] = std::array::from_fn(|channel| {
                if transform.needs_transpose() {
                    oracle.get_transposed(channel, transform_type)
                } else {
                    oracle.get(channel, transform_type)
                }
            });
            let extent = transform.pixel_extent();
            let area = usize::try_from(extent.width * extent.height).unwrap();
            let base = expected.len();
            expected.resize(base + area, [0; 4]);
            for y in 0..extent.height {
                for x in 0..extent.width {
                    let raster = (y * extent.width + x) as usize;
                    let backend_index = if transform.is_special() || extent.height < extent.width {
                        y * extent.width + x
                    } else {
                        x * extent.height + y
                    } as usize;
                    expected[base + backend_index] = [
                        channels[0][raster].to_bits(),
                        channels[1][raster].to_bits(),
                        channels[2][raster].to_bits(),
                        0,
                    ];
                }
            }
        }
        assert_eq!(words, expected);
    }

    #[test]
    fn raw_and_transform_incompatible_hf_matrices_have_typed_errors() {
        let mut raw = BitWriter::new();
        raw.write_bits(0, 1).unwrap();
        raw.write_bits(7, 3).unwrap();
        let raw_bytes = raw.into_bytes();
        let mut raw_reader = jxl_gpu_bitstream::BitReader::new(&raw_bytes);
        assert!(matches!(
            parse_hf_dequant_matrices(&mut raw_reader),
            Err(BoundedVarDctPacketError::RawHfDequantMatrix { matrix: 0 })
        ));

        let mut incompatible = BitWriter::new();
        incompatible.write_bits(0, 1).unwrap();
        for _ in 0..4 {
            incompatible.write_bits(0, 3).unwrap();
        }
        incompatible.write_bits(1, 3).unwrap();
        let incompatible_bytes = incompatible.into_bytes();
        let mut incompatible_reader = jxl_gpu_bitstream::BitReader::new(&incompatible_bytes);
        assert!(matches!(
            parse_hf_dequant_matrices(&mut incompatible_reader),
            Err(BoundedVarDctPacketError::HfDequantMatrixEncoding {
                matrix: 4,
                encoding: 1
            })
        ));
    }

    #[test]
    fn modular_parameter_record_preserves_the_consumer_word_layout() {
        let params = VarDctModularParams::default()
            .with_lz77_window(8)
            .with_self_correcting(true);
        let words = bytemuck::cast::<VarDctModularParams, [u32; 60]>(params);
        let mut expected = [0; 60];
        expected[2] = 7;
        expected[19] = 1;
        expected[48] = 16;
        expected[49] = 10;
        expected[50..=52].fill(7);
        expected[55] = 13;
        expected[56..=58].fill(12);
        assert_eq!(words, expected);
    }

    #[test]
    fn modular_parameter_record_rebases_one_packet_window() {
        let segment = GroupStreamSegment {
            group_index: 0,
            input_start: 0,
            input_end: 8,
            upload_offset: 0,
            window_logical_start: 16,
            window_upload_start: 3,
            available_token_end: 72,
            stream_token_end: 96,
            window_yield_end: 64,
            flags: GroupStreamSegment::FINAL,
        };
        let params = VarDctModularParams::default().with_stream_segment(segment, 41, 128);
        assert_eq!(params.entropy.token_start, 0);
        assert_eq!(params.entropy.token_end, 72);
        assert_eq!(params.window_contract(), [16, 3, 96, 64, 6, 128, 41]);
    }

    #[test]
    fn intermediate_packet_window_keeps_bounded_mode_without_boundary_flags() {
        let segment = GroupStreamSegment {
            group_index: 0,
            input_start: 8,
            input_end: 16,
            upload_offset: 0,
            window_logical_start: 64,
            window_upload_start: 0,
            available_token_end: 128,
            stream_token_end: 192,
            window_yield_end: 112,
            flags: 0,
        };
        let params = VarDctModularParams::default().with_stream_segment(segment, 41, 128);

        assert_eq!(params.window_contract(), [64, 0, 192, 112, 4, 128, 41]);
    }

    #[test]
    fn coefficient_entropy_plan_expands_all_custom_orders_before_coefficient_symbols() {
        let codestream = decode_hex(include_str!(
            "../../jxl_gpu_bitstream/test-data/green_queen_vardct_e3.jxl.hex"
        ));
        let parsed = jxl_gpu_bitstream::parse(&codestream, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let profile = StandardVarDctProfile::negotiate(&inventory).unwrap();
        let VarDctSectionLayout::Sections {
            lf_global,
            hf_global,
            ..
        } = profile.sections
        else {
            panic!("fixture has physical VarDCT sections")
        };
        let lf = LfGlobalPrefix::parse(&codestream, lf_global).unwrap();
        let prefix = HfGlobalPrefix::parse(
            &codestream,
            hf_global,
            u32::try_from(profile.group_count).unwrap(),
        )
        .unwrap();
        let mut order_reader = jxl_gpu_bitstream::BitReader::new(&codestream);
        order_reader
            .skip_bits(prefix.order_entropy_bit_offset)
            .unwrap();
        let actual_orders =
            parse_coefficient_orders_reader(&mut order_reader, prefix, hf_global.end().unwrap())
                .unwrap();
        assert_eq!(
            actual_orders,
            legacy_coefficient_orders(&codestream, prefix)
        );
        let plan = HfCoefficientEntropyPlan::parse(
            &codestream,
            hf_global,
            u32::try_from(profile.group_count).unwrap(),
            &lf.hf_block_context,
            Vec::new(),
            32 * 32 * 3 * 64,
        )
        .unwrap();
        assert_eq!(plan.order_coordinate_offset_words, 13 * 3 * 4);
        let descriptors = bytemuck::cast_slice::<u32, crate::vardct_artifact::GpuHfOrderDescriptor>(
            &plan.order_words[..plan.order_coordinate_offset_words as usize],
        );
        assert_eq!(descriptors.len(), 13 * 3);
        assert_eq!(descriptors[0].len, 64);
        assert_eq!([descriptors[0].width, descriptors[0].height], [8, 8]);
        assert_ne!(descriptors[0].offset, descriptors[1].offset);
        assert_ne!(descriptors[1].offset, descriptors[2].offset);
        assert_eq!(descriptors[3].offset, descriptors[4].offset);
        assert_eq!(descriptors[4].offset, descriptors[5].offset);
        assert_eq!(
            plan.order_words.len(),
            plan.order_coordinate_offset_words as usize
                + crate::vardct_artifact::HF_ORDER_EXTENTS
                    .iter()
                    .map(|[width, height]| (width * height) as usize)
                    .sum::<usize>()
                + 2 * 64
        );
    }

    #[test]
    fn packet_plan_is_identical_across_every_fixture_chunk_split() {
        let codestream = decode_hex(include_str!(
            "../test-data/green_queen_crop_vardct_epf2.jxl.hex"
        ));
        let parsed = jxl_gpu_bitstream::parse(&codestream, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let expected = BoundedVarDctPacketPlan::parse(&codestream, &inventory).unwrap();
        let storage: Arc<[u8]> = codestream.into();

        for split in 0..=storage.len() {
            let source = split_source(Arc::clone(&storage), split);
            assert_eq!(
                BoundedVarDctPacketPlan::parse_source(&source, &inventory).unwrap(),
                expected,
                "chunk split {split} changed the VarDCT packet plan"
            );
        }
    }

    #[test]
    fn custom_order_packet_and_hf_continuation_cross_byte_sized_spans() {
        let codestream = decode_hex(include_str!(
            "../../jxl_gpu_bitstream/test-data/green_queen_vardct_e3.jxl.hex"
        ));
        let parsed = jxl_gpu_bitstream::parse(&codestream, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let expected = BoundedVarDctPacketPlan::parse(&codestream, &inventory).unwrap();
        let expected_continuation = expected
            .parse_hf_continuation(&codestream, &expected.groups[0], 67_171)
            .unwrap();
        let storage: Arc<[u8]> = codestream.into();
        let spans = (0..storage.len()).map(|offset| {
            (
                offset as u64,
                StreamSlice::from_shared_range(Arc::clone(&storage), offset..offset + 1).unwrap(),
            )
        });
        let source = GpuCodestream::from_spans(spans).unwrap();
        let actual = BoundedVarDctPacketPlan::parse_source(&source, &inventory).unwrap();
        let actual_continuation = actual
            .parse_hf_continuation_source(&source, &actual.groups[0], 67_171)
            .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(actual_continuation, expected_continuation);
    }

    #[test]
    fn natural_orders_cover_each_transform_extent_once() {
        for [width, height] in crate::vardct_artifact::HF_ORDER_EXTENTS {
            let order = natural_coefficient_order(width, height).unwrap();
            let mut sorted = order
                .iter()
                .map(|packed| (packed & 0xffff, packed >> 16))
                .collect::<Vec<_>>();
            sorted.sort_unstable();
            let mut expected = (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .collect::<Vec<_>>();
            expected.sort_unstable();
            assert_eq!(sorted, expected, "{width}x{height}");
        }
    }

    #[test]
    fn tiled_profile_uses_the_standard_dct8_strategy_id() {
        assert_eq!(transform_id(TransformKind::Dct8), 0);
        assert_eq!(transform_for_extent(16, 32), Some(TransformKind::Dct32x16));
    }
}
