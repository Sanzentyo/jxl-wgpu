//! GPU entropy frontend for the bounded standard VarDCT packet profile.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{BitRange, BitReader, CodestreamInventory};
use jxl_gpu_protocol::TransformKind;
use thiserror::Error;

use crate::entropy::EntropyStreamParams;
use crate::modular_tree::{
    EntropyDecoderIr, MaConfigIr, MaTreeLimits, MaTreeNodeIr, PackedModularMetadata,
    parse_ma_config,
};
use crate::vardct_frontend::{
    HfBlockContextIr, HfGlobalPrefix, LfChannelCorrelation, LfChannelDequantization,
    LfGlobalPrefix, StandardVarDctProfile, VarDctFrontendError, VarDctGroupRect, VarDctPacketError,
    VarDctSectionLayout, parse_hf_metadata_header_prefix, parse_lf_group_header_prefix,
};

const SHADER_TEMPLATE: &str = include_str!("vardct_packet.wgsl");
const MODULAR_ENTROPY_ABI: &str = include_str!("modular_entropy_abi.wgsl");
const MODULAR_ENTROPY: &str = include_str!("modular_entropy.wgsl");
const MODULAR_RECONSTRUCT: &str = include_str!("modular_reconstruct.wgsl");
const ENTROPY_ABI_MARKER: &str = "/*__JXL_MODULAR_ENTROPY_ABI__*/";
const ENTROPY_MARKER: &str = "/*__JXL_MODULAR_ENTROPY__*/";
const RECONSTRUCT_MARKER: &str = "/*__JXL_MODULAR_RECONSTRUCT__*/";

const ZERO_AC_HF_GLOBAL: u32 = 0x2495;

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
    #[error("failed to parse the bounded MA-tree descriptor: {0}")]
    ModularTree(String),
    #[error("failed to position the HF coefficient-order reader: {0}")]
    CoefficientOrderBitstream(#[source] jxl_bitstream::Error),
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
}

/// GPU-reported validation failure. No output is authoritative after this error.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum GpuVarDctPacketError {
    #[error("GPU VarDCT packet entropy failed with status {code} at bit {cursor}/{end}")]
    Entropy { code: u32, cursor: u32, end: u32 },
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
#[derive(Clone, Debug)]
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
    global_ma_config: Option<MaConfigIr>,
}

/// One host-packed MA tree/histogram bundle and its exact GPU reconstruction requirements.
#[derive(Clone, Debug)]
pub struct BoundedModularEntropyPlan {
    pub metadata: Vec<u32>,
    pub needs_self_correcting: bool,
    pub lz77_window_words: u32,
}

/// Host metadata discovered after the first GPU stage returns the LF entropy cursor.
#[derive(Clone, Debug)]
pub struct BoundedHfMetadataContinuation {
    pub token_bit_offset: u32,
    pub block_count: u32,
    pub modular: BoundedModularEntropyPlan,
}

/// One LF group's packet geometry and persistent decode contract.
#[derive(Clone, Debug)]
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
}

/// Host-packed entropy tables and untouched pass-group packets for one VarDCT AC pass.
#[derive(Clone, Debug)]
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
}

fn parse_ma_config_at(
    codestream: &[u8],
    bit_offset: u64,
    packet_end: u64,
) -> Result<(MaConfigIr, u64), BoundedVarDctPacketError> {
    let mut reader = BitReader::new(codestream);
    reader.skip_bits(bit_offset)?;
    let config = parse_ma_config(&mut reader, MaTreeLimits::default())
        .map_err(|error| BoundedVarDctPacketError::ModularTree(error.to_string()))?;
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
        let profile = StandardVarDctProfile::negotiate(inventory)?;
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
                        Some(
                            transform_for_extent(profile.width, profile.height)
                                .ok_or(UnsupportedVarDctPacketFeature::TransformExtent)?,
                        ),
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
        let lf_global = LfGlobalPrefix::parse(codestream, lf_global_packet)?;
        let (global_ma_config, descriptor_end) =
            if let Some(tree_offset) = lf_global.global_ma_tree_bit_offset {
                let (config, end) = parse_ma_config_at(
                    codestream,
                    tree_offset,
                    lf_global_packet
                        .end()
                        .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                            field: "LF-global end",
                        })?,
                )?;
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
                let lf_header = parse_lf_group_header_prefix(
                    codestream,
                    BitRange {
                        offset: lf_group_start,
                        length: lf_group_end.checked_sub(lf_group_start).ok_or(
                            BoundedVarDctPacketError::ArithmeticOverflow {
                                field: "LF-group range",
                            },
                        )?,
                    },
                )?;
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
                    parse_ma_config_at(
                        codestream,
                        lf_header.modular.tree_or_token_bit_offset,
                        lf_group_end,
                    )?
                };
                let lf_decoded_symbol_limit = block_count.checked_mul(3).ok_or(
                    BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "LF-group LF decoded symbol limit",
                    },
                )?;
                let lf_modular =
                    pack_modular_plan(&lf_config, blocks_x.max(1), lf_decoded_symbol_limit)?;
                let hf_window_words = if let Some(config) = &global_ma_config {
                    config
                        .entropy
                        .lz77_window_words(
                            block_count.max(blocks_x).max(1),
                            hf_decoded_symbol_limit,
                        )
                        .map_err(|error| BoundedVarDctPacketError::ModularTree(error.to_string()))?
                } else {
                    hf_decoded_symbol_limit.checked_next_power_of_two().ok_or(
                        BoundedVarDctPacketError::ArithmeticOverflow {
                            field: "LF-group deferred HF LZ77 window",
                        },
                    )?
                };
                let lz77_window_words = lf_modular.lz77_window_words.max(hf_window_words);
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
                    extra_precision: lf_header.extra_precision,
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
                HfCoefficientEntropyPlan::parse(
                    codestream,
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
        Ok(Self {
            profile,
            uniform_transform,
            lf_global: lf_global_packet,
            hf_global,
            entropy_bit_offset,
            modular_metadata: words,
            needs_self_correcting: global_ma_config
                .as_ref()
                .is_some_and(MaConfigIr::needs_self_correcting),
            lf_dequantization: lf_global.lf_dequantization,
            global_scale: lf_global.global_scale,
            quant_lf: lf_global.quant_lf,
            lf_correlation: lf_global.lf_correlation,
            groups,
            hf_coefficients,
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
        self.global_ma_config.is_none()
    }

    /// Parses only the HF scalar header and its selected MA descriptor after the GPU reports the
    /// LF entropy end cursor. No image symbol is decoded on the host.
    pub fn parse_hf_continuation(
        &self,
        codestream: &[u8],
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
        let prefix = parse_hf_metadata_header_prefix(
            codestream,
            u64::from(lf_entropy_end),
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
            parse_ma_config_at(
                codestream,
                prefix.modular.tree_or_token_bit_offset,
                packet_end,
            )?
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

    /// U32 scratch words retaining LF samples, weighted-predictor rows, and the LZ history ring.
    pub fn reconstructed_words(
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
        samples
            .checked_add(predictor_words)
            .and_then(|words| words.checked_add(self.lz77_window_words))
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "VarDCT reconstruction scratch",
            })
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

impl HfCoefficientEntropyPlan {
    /// Parses HF-global tables without consuming a coefficient symbol. Pass-group ranges remain
    /// exact views into the caller-owned codestream for the GPU executor.
    fn parse(
        codestream: &[u8],
        packet: BitRange,
        group_count: u32,
        block_context: &HfBlockContextIr,
        pass_groups: Vec<BitRange>,
        decoded_symbol_limit: u32,
    ) -> Result<Self, BoundedVarDctPacketError> {
        let prefix = HfGlobalPrefix::parse(codestream, packet, group_count)?;
        let (coefficient_entropy_bit_offset, order_words, order_coordinate_offset_words) =
            parse_coefficient_orders(codestream, prefix)?;
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
        let packet_end = packet
            .end()
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF-global packet end",
            })?;
        if coefficient_entropy_bit_offset > packet_end {
            return Err(VarDctPacketError::PacketBoundary {
                cursor: coefficient_entropy_bit_offset,
                packet_end,
            }
            .into());
        }
        let mut reader = BitReader::new(codestream);
        reader.skip_bits(coefficient_entropy_bit_offset)?;
        let descriptor =
            EntropyDecoderIr::parse(&mut reader, context_count, MaTreeLimits::default())
                .map_err(|error| BoundedVarDctPacketError::ModularTree(error.to_string()))?;
        let descriptor_end = reader.bit_offset();
        if descriptor_end > packet_end {
            return Err(VarDctPacketError::PacketBoundary {
                cursor: descriptor_end,
                packet_end,
            }
            .into());
        }
        let remaining = packet_end - descriptor_end;
        if remaining > 7 || reader.read_bits(remaining as u8)? != 0 {
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
        })
    }
}

fn parse_coefficient_orders(
    codestream: &[u8],
    prefix: HfGlobalPrefix,
) -> Result<(u64, Vec<u32>, u32), BoundedVarDctPacketError> {
    use crate::vardct_artifact::{
        GpuHfOrderDescriptor, HF_ORDER_CHANNELS, HF_ORDER_COUNT, HF_ORDER_EXTENTS,
    };

    const DESCRIPTOR_WORDS: u32 = (HF_ORDER_COUNT * HF_ORDER_CHANNELS * 4) as u32;

    let mut bitstream = jxl_bitstream::Bitstream::new(codestream);
    bitstream
        .skip_bits(
            usize::try_from(prefix.order_entropy_bit_offset).map_err(|_| {
                BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "HF coefficient-order bit offset",
                }
            })?,
        )
        .map_err(BoundedVarDctPacketError::CoefficientOrderBitstream)?;
    let mut decoder = (prefix.used_orders != 0)
        .then(|| jxl_coding::Decoder::parse(&mut bitstream, 8))
        .transpose()
        .map_err(BoundedVarDctPacketError::CoefficientOrderCoding)?;

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

        let decoder = decoder
            .as_mut()
            .ok_or(BoundedVarDctPacketError::PackedMetadata)?;
        let skip = len / 64;
        for channel in 0..HF_ORDER_CHANNELS {
            let permutation = jxl_coding::read_permutation(&mut bitstream, decoder, len, skip)
                .map_err(BoundedVarDctPacketError::CoefficientOrderCoding)?;
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
    if let Some(decoder) = &mut decoder {
        decoder
            .finalize()
            .map_err(BoundedVarDctPacketError::CoefficientOrderCoding)?;
    }
    let coefficient_entropy_bit_offset =
        u64::try_from(bitstream.num_read_bits()).map_err(|_| {
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF coefficient entropy bit offset",
            }
        })?;

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
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VarDctModularParams {
    entropy: EntropyStreamParams,
    consumer_words: [u32; 49],
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
            consumer_words,
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
            }),
            2..=13 => Err(GpuVarDctPacketError::Entropy {
                code: self.code,
                cursor: self.cursor,
                end: self.expected_end,
            }),
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
    assert!(std::mem::size_of::<VarDctModularParams>() == 208);
    assert!(std::mem::align_of::<VarDctModularParams>() == 4);
    assert!(std::mem::size_of::<GpuVarDctPacketStatus>() == 64);
};

#[cfg(test)]
mod tests {
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
    fn modular_parameter_record_preserves_the_consumer_word_layout() {
        let params = VarDctModularParams::default()
            .with_lz77_window(8)
            .with_self_correcting(true);
        let words = bytemuck::cast::<VarDctModularParams, [u32; 52]>(params);
        let mut expected = [0; 52];
        expected[2] = 7;
        expected[12] = 1;
        expected[41] = 16;
        expected[42] = 10;
        expected[43..=45].fill(7);
        expected[48] = 13;
        expected[49..=51].fill(12);
        assert_eq!(words, expected);
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
