//! GPU entropy frontend for the bounded standard regular-VarDCT packet profiles.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{BitRange, BitReader, CodestreamInventory};
use jxl_gpu_protocol::TransformKind;
use thiserror::Error;

use crate::entropy::EntropyStreamParams;
use crate::modular_tree::{
    EntropyDecoderIr, MaTreeLimits, MaTreeNodeIr, PackedModularMetadata, parse_ma_config,
};
use crate::vardct_frontend::{
    HfBlockContextIr, HfGlobalPrefix, LfGlobalPrefix, StandardVarDctProfile, VarDctFrontendError,
    VarDctPacketError, VarDctSectionLayout,
};

const SHADER_TEMPLATE: &str = include_str!("vardct_packet.wgsl");
const MODULAR_ENTROPY_ABI: &str = include_str!("modular_entropy_abi.wgsl");
const MODULAR_ENTROPY: &str = include_str!("modular_entropy.wgsl");
const MODULAR_RECONSTRUCT: &str = include_str!("modular_reconstruct.wgsl");
const ENTROPY_ABI_MARKER: &str = "/*__JXL_MODULAR_ENTROPY_ABI__*/";
const ENTROPY_MARKER: &str = "/*__JXL_MODULAR_ENTROPY__*/";
const RECONSTRUCT_MARKER: &str = "/*__JXL_MODULAR_RECONSTRUCT__*/";

const ZERO_AC_HF_GLOBAL: u32 = 0x2495;

/// A standard feature excluded from the deliberately bounded regular-VarDCT packet profile.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum UnsupportedVarDctPacketFeature {
    #[error("the bounded tiled-VarDCT decoder requires exactly one LF group")]
    MultipleLfGroups,
    #[error("the DCT8 coefficient executor cannot use coefficient-order mask {used_orders:#x}")]
    CustomCoefficientOrders { used_orders: u16 },
    #[error("the bounded regular-VarDCT decoder currently accepts 8-bit samples")]
    BitDepth,
    #[error("the one-entry packet extent is not one implemented regular VarDCT transform")]
    TransformExtent,
    #[error("the standard packet requests skip-adaptive-LF smoothing")]
    SkipAdaptiveLfSmoothing,
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
    #[error("failed to position the bounded MA-tree parser: {0}")]
    Bitstream(#[from] jxl_gpu_bitstream::Error),
    #[error("failed to parse the bounded MA-tree descriptor: {0}")]
    ModularTree(String),
    #[error("failed to position the HF coefficient-order reader: {0}")]
    CoefficientOrderBitstream(#[source] jxl_bitstream::Error),
    #[error("failed to decode the HF coefficient-order permutation: {0}")]
    CoefficientOrderCoding(#[source] jxl_coding::Error),
    #[error("the packed MA-tree metadata ABI is malformed")]
    PackedMetadata,
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
    #[error("GPU VarDCT packet uses nonzero HF chroma correlation {value}")]
    Correlation { value: u32 },
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
    /// Regular transform shared by every first block in this bounded packet.
    pub transform: TransformKind,
    /// Number of non-overlapping first blocks reconstructed from HF metadata.
    pub task_count: u32,
    coefficient_words: u32,
    /// LF-global packet containing the scalar quantizer fields and global MA descriptor.
    pub lf_global: BitRange,
    /// LF-group packet containing quantized LF and HF-metadata Modular streams.
    pub lf_group: BitRange,
    /// Separate HF-global packet, or `None` when all three packets share a single TOC entry.
    pub hf_global: Option<BitRange>,
    /// Descriptor end used as the LF-group start by the one-entry TOC form.
    pub entropy_bit_offset: u32,
    pub lf_stream_index: u32,
    pub hf_stream_index: u32,
    pub modular_metadata: Vec<u32>,
    /// Physical power-of-two history ring used by both sequential Modular streams.
    pub lz77_window_words: u32,
    /// Whether the shared MA tree requires the weighted self-correcting predictor workspace.
    pub needs_self_correcting: bool,
    pub global_scale: u32,
    pub quant_lf: u32,
    /// Descriptor-only HF coefficient entropy plan. Coefficient symbols remain in pass-group
    /// packets and are never expanded on the host.
    pub hf_coefficients: Option<HfCoefficientEntropyPlan>,
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
    /// Three DCT8 order descriptors followed by their packed `(x, y)` coordinate tables.
    pub order_words: Vec<u32>,
    pub order_coordinate_offset_words: u32,
    pub pass_groups: Vec<BitRange>,
    /// Per-pass-group power-of-two history capacity for the common GPU entropy executor.
    pub lz77_window_words: u32,
}

impl BoundedVarDctPacketPlan {
    /// Parses bounded scalar metadata only. Image symbols remain encoded for the GPU.
    pub fn parse(
        codestream: &[u8],
        inventory: &CodestreamInventory,
    ) -> Result<Self, BoundedVarDctPacketError> {
        let profile = StandardVarDctProfile::negotiate(inventory)?;
        if profile.low_frequency_group_count != 1 {
            return Err(UnsupportedVarDctPacketFeature::MultipleLfGroups.into());
        }
        if profile.bits_per_sample != 8 {
            return Err(UnsupportedVarDctPacketFeature::BitDepth.into());
        }
        if !profile.adaptive_lf_smoothing {
            return Err(UnsupportedVarDctPacketFeature::SkipAdaptiveLfSmoothing.into());
        }
        let (transform, task_count, lf_global_packet, lf_group, hf_global, pass_groups) =
            match &profile.sections {
                VarDctSectionLayout::Single { packet } => (
                    transform_for_extent(profile.width, profile.height)
                        .ok_or(UnsupportedVarDctPacketFeature::TransformExtent)?,
                    1,
                    *packet,
                    *packet,
                    None,
                    Vec::new(),
                ),
                VarDctSectionLayout::Sections {
                    lf_global,
                    lf_groups,
                    hf_global,
                    pass_groups,
                } => {
                    let lf_group = *lf_groups
                        .first()
                        .ok_or(UnsupportedVarDctPacketFeature::MultipleLfGroups)?;
                    let blocks = profile
                        .width
                        .div_ceil(8)
                        .checked_mul(profile.height.div_ceil(8))
                        .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                            field: "tiled DCT8 task count",
                        })?;
                    (
                        TransformKind::Dct8,
                        blocks,
                        *lf_global,
                        lf_group,
                        Some(*hf_global),
                        pass_groups.clone(),
                    )
                }
            };
        let lf_global = LfGlobalPrefix::parse(codestream, lf_global_packet)?;
        let mut reader = BitReader::new(codestream);
        reader.skip_bits(lf_global.ma_tree_bit_offset)?;
        let ma_config = parse_ma_config(&mut reader, MaTreeLimits::default())
            .map_err(|error| BoundedVarDctPacketError::ModularTree(error.to_string()))?;
        let descriptor_end = reader.bit_offset();
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
        for node in &ma_config.nodes {
            if let MaTreeNodeIr::Decision { property, .. } = *node
                && property >= 16
            {
                return Err(
                    UnsupportedVarDctPacketFeature::PreviousChannelMaProperty { property }.into(),
                );
            }
        }
        let PackedModularMetadata { words } = ma_config
            .pack_gpu_metadata()
            .map_err(|error| BoundedVarDctPacketError::ModularTree(error.to_string()))?;
        if words.len() <= 9 {
            return Err(BoundedVarDctPacketError::PackedMetadata);
        }
        let entropy_bit_offset = u32::try_from(descriptor_end).map_err(|_| {
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "entropy bit offset",
            }
        })?;
        let blocks_x = profile.width.div_ceil(8);
        let blocks_y = profile.height.div_ceil(8);
        let block_count =
            blocks_x
                .checked_mul(blocks_y)
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "block count",
                })?;
        let coefficient_words = block_count.checked_mul(8 * 8 * 3).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "padded coefficient words",
            },
        )?;
        let correlation_samples = profile
            .width
            .div_ceil(64)
            .checked_mul(profile.height.div_ceil(64))
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "correlation sample count",
            })?;
        let decoded_symbol_limit = block_count
            .checked_mul(4)
            .and_then(|samples| {
                task_count
                    .checked_mul(2)
                    .and_then(|tasks| samples.checked_add(tasks))
            })
            .and_then(|samples| {
                correlation_samples
                    .checked_mul(2)
                    .and_then(|correlations| samples.checked_add(correlations))
            })
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "decoded symbol limit",
            })?;
        let lz77_window_words = ma_config
            .entropy
            .lz77_window_words(blocks_x.max(1), decoded_symbol_limit)
            .map_err(|error| BoundedVarDctPacketError::ModularTree(error.to_string()))?;
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
            lf_stream_index: profile.lf_quant_stream_index(0)?,
            hf_stream_index: profile.hf_metadata_stream_index(0)?,
            profile,
            transform,
            task_count,
            coefficient_words,
            lf_global: lf_global_packet,
            lf_group,
            hf_global,
            entropy_bit_offset,
            modular_metadata: words,
            lz77_window_words,
            needs_self_correcting: ma_config.needs_self_correcting(),
            global_scale: lf_global.global_scale,
            quant_lf: lf_global.quant_lf,
            hf_coefficients,
        })
    }

    #[must_use]
    pub fn coefficient_words(&self) -> u32 {
        self.coefficient_words
    }

    #[must_use]
    pub fn block_extent(&self) -> [u32; 2] {
        [
            self.profile.width.div_ceil(8),
            self.profile.height.div_ceil(8),
        ]
    }

    /// U32 scratch words retaining LF samples, weighted-predictor rows, and the LZ history ring.
    pub fn reconstructed_words(&self) -> Result<u32, BoundedVarDctPacketError> {
        let [blocks_x, blocks_y] = self.block_extent();
        let blocks =
            blocks_x
                .checked_mul(blocks_y)
                .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                    field: "block count",
                })?;
        let correlations = self.correlation_samples()?;
        let hf_samples = self
            .task_count
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
        let predictor_words = if self.needs_self_correcting {
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

    pub fn packet_control(&self) -> Result<VarDctPacketControl, BoundedVarDctPacketError> {
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
            self.hf_global
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
                self.entropy_bit_offset,
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
        let hf_mul_offset = strategy_offset.checked_add(self.task_count).ok_or(
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF multiplier offset",
            },
        )?;
        let sharpness_offset = hf_mul_offset.checked_add(self.task_count).ok_or(
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
            geometry: [self.profile.width, self.profile.height, blocks_x, blocks_y],
            offsets: [0, correlation_samples, strategy_offset, hf_mul_offset],
            capacities: [
                self.coefficient_words(),
                raw_capacity,
                block_count
                    .checked_next_power_of_two()
                    .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "first-block field width",
                    })?
                    .trailing_zeros(),
                self.task_count,
            ],
            expected: [
                transform_id(self.transform),
                0,
                ZERO_AC_HF_GLOBAL,
                sharpness_offset,
            ],
            quantization: [self.global_scale, self.quant_lf, 0, 0],
            streams: [
                self.lf_stream_index,
                self.hf_stream_index,
                separate_sections,
                u32::try_from(self.profile.group_count).map_err(|_| {
                    BoundedVarDctPacketError::ArithmeticOverflow {
                        field: "pass-group count",
                    }
                })?,
            ],
            scratch: [self.predictor_width_capacity()?, 0, 0, 0],
        })
    }

    fn predictor_width_capacity(&self) -> Result<u32, BoundedVarDctPacketError> {
        let [blocks_x, _] = self.block_extent();
        Ok(blocks_x
            .max(self.task_count)
            .max(self.profile.width.div_ceil(64)))
    }

    fn correlation_samples(&self) -> Result<u32, BoundedVarDctPacketError> {
        self.profile
            .width
            .div_ceil(64)
            .checked_mul(self.profile.height.div_ceil(64))
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
        if prefix.used_orders & !1 != 0 {
            return Err(UnsupportedVarDctPacketFeature::CustomCoefficientOrders {
                used_orders: prefix.used_orders,
            }
            .into());
        }
        let (coefficient_entropy_bit_offset, order_words, order_coordinate_offset_words) =
            parse_dct8_coefficient_orders(codestream, prefix)?;
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

fn parse_dct8_coefficient_orders(
    codestream: &[u8],
    prefix: HfGlobalPrefix,
) -> Result<(u64, Vec<u32>, u32), BoundedVarDctPacketError> {
    use crate::vardct_artifact::NaturalDct8OrderTable;

    const DESCRIPTOR_WORDS: u32 = 3 * 4;

    let natural = NaturalDct8OrderTable::new();
    if prefix.used_orders == 0 {
        return Ok((
            prefix.order_entropy_bit_offset,
            natural.packed_words(),
            DESCRIPTOR_WORDS,
        ));
    }

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
    let mut decoder = jxl_coding::Decoder::parse(&mut bitstream, 8)
        .map_err(BoundedVarDctPacketError::CoefficientOrderCoding)?;
    let permutations = (0..3)
        .map(|_| {
            jxl_coding::read_permutation(&mut bitstream, &mut decoder, 64, 1)
                .map_err(BoundedVarDctPacketError::CoefficientOrderCoding)
        })
        .collect::<Result<Vec<_>, _>>()?;
    decoder
        .finalize()
        .map_err(BoundedVarDctPacketError::CoefficientOrderCoding)?;
    let coefficient_entropy_bit_offset =
        u64::try_from(bitstream.num_read_bits()).map_err(|_| {
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "HF coefficient entropy bit offset",
            }
        })?;

    let mut order_words = Vec::with_capacity(DESCRIPTOR_WORDS as usize + 3 * 64);
    for channel in 0..3u32 {
        order_words.extend_from_slice(&[channel * 64, 64, 8, 8]);
    }
    for permutation in permutations {
        order_words.extend(
            permutation
                .into_iter()
                .map(|index| natural.coordinates[index]),
        );
    }
    Ok((
        coefficient_entropy_bit_offset,
        order_words,
        DESCRIPTOR_WORDS,
    ))
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
    pub _reserved: [u32; 5],
}

impl GpuVarDctPacketStatus {
    pub fn validate(
        self,
        expected_strategy: TransformKind,
        expected_lf_samples: u32,
        expected_hf_samples: u32,
        expected_global_scale: u32,
        expected_quant_lf: u32,
    ) -> Result<(), GpuVarDctPacketError> {
        match self.code {
            1 if self.cursor == self.expected_end
                && self.lf_decoded == expected_lf_samples
                && self.hf_decoded == expected_hf_samples
                && self.strategy == transform_id(expected_strategy)
                && self.hf_mul > 0
                && self.global_scale == expected_global_scale
                && self.quant_lf == expected_quant_lf =>
            {
                Ok(())
            }
            1 => Err(GpuVarDctPacketError::Entropy {
                code: self.code,
                cursor: self.cursor,
                end: self.expected_end,
            }),
            20 => Err(GpuVarDctPacketError::LfHeader),
            21 => Err(GpuVarDctPacketError::FirstBlock),
            22 => Err(GpuVarDctPacketError::HfHeader),
            23 => Err(GpuVarDctPacketError::Correlation { value: self.detail }),
            24 => Err(GpuVarDctPacketError::Strategy {
                actual: self.detail,
                expected: transform_id(expected_strategy),
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
    pipeline: wgpu::ComputePipeline,
}

impl VarDctPacketPipeline {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let source = shader_source();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu bounded VarDCT packet frontend"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu bounded VarDCT packet frontend"),
            layout: None,
            module: &module,
            entry_point: Some("decode_vardct_packet"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline }
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffers: VarDctPacketBuffers<'_>,
    ) {
        let layout = self.pipeline.get_bind_group_layout(0);
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
            layout: &layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu bounded VarDCT packet frontend"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
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
    fn coefficient_entropy_plan_expands_dct8_custom_orders_before_coefficient_symbols() {
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
        assert_eq!(plan.order_coordinate_offset_words, 12);
        assert_eq!(plan.order_words.len(), 12 + 3 * 64);
        assert_eq!(
            &plan.order_words[..12],
            &[0, 64, 8, 8, 64, 64, 8, 8, 128, 64, 8, 8]
        );
    }

    #[test]
    fn tiled_profile_uses_the_standard_dct8_strategy_id() {
        assert_eq!(transform_id(TransformKind::Dct8), 0);
        assert_eq!(transform_for_extent(16, 32), Some(TransformKind::Dct32x16));
    }
}
