//! Resident Modular side images embedded in VarDCT metadata.
//!
//! JPEG XL mode-7 HF dequantization matrices are not scalar metadata: each one is a complete
//! three-channel Modular image whose entropy cursor determines where the following HF-global
//! metadata begins. This module lowers that local header into the same topology, MA-tree,
//! previous-channel reference, and inverse-transform contracts used by the main Modular decoder.

use crate::modular_inverse::{ModularInversePlan, plan_modular_inverse};
use crate::modular_transform::{
    GpuModularChannelLayout, ModularChannelTopology, ModularTransformLimits,
    PackedModularChannelMetadata, parse_modular_transforms,
};
use crate::modular_tree::{
    BitInput, MaConfigIr, MaTreeLimits, PackedModularMetadata, WpHeaderIr, parse_ma_config,
};
use crate::vardct_frontend::{metadata_bool, metadata_f16};
use crate::vardct_packet::BoundedVarDctPacketError;

const RAW_MATRIX_COUNT: usize = 17;

/// Host-known execution contract for one raw HF dequantization matrix side image.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RawHfDequantSideImagePlan {
    pub matrix_index: usize,
    pub bit_depth: u32,
    pub denominator: f32,
    pub stream_index: u32,
    pub token_bit_offset: u32,
    pub wp_header: WpHeaderIr,
    pub metadata: Vec<u32>,
    pub needs_self_correcting: bool,
    pub channel_metadata: PackedModularChannelMetadata,
    pub inverse_plan: ModularInversePlan,
    pub final_planes: [GpuModularChannelLayout; 3],
    pub decoded_words: u32,
    pub maximum_width: u32,
    pub lz77_window_words: u32,
}

impl RawHfDequantSideImagePlan {
    pub(crate) fn parse(
        reader: &mut impl BitInput,
        matrix_index: usize,
        bit_depth: u32,
        low_frequency_group_count: u64,
        global_ma_config: Option<&MaConfigIr>,
    ) -> Result<Self, BoundedVarDctPacketError> {
        let [width, height] = raw_matrix_extent(matrix_index).ok_or(
            BoundedVarDctPacketError::HfDequantMatrixValue {
                matrix: matrix_index,
                reason: "matrix index is outside the normative set",
            },
        )?;
        let denominator = metadata_f16(reader, "raw HF dequantization matrix denominator")?;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(BoundedVarDctPacketError::HfDequantMatrixValue {
                matrix: matrix_index,
                reason: "raw denominator must be finite and positive",
            });
        }

        let use_global_tree = metadata_bool(reader, "raw HF dequantization matrix MA tree")?;
        let wp_header = WpHeaderIr::parse(reader).map_err(map_modular_error)?;
        let limits = ModularTransformLimits::default();
        let initial_topology =
            ModularChannelTopology::full_resolution(width, height, bit_depth, 3, limits)
                .map_err(map_modular_error)?;
        let transform_plan = parse_modular_transforms(reader, initial_topology, limits)
            .map_err(map_modular_error)?;
        let ma_config = if use_global_tree {
            global_ma_config
                .ok_or(BoundedVarDctPacketError::MissingGlobalMaTree {
                    stage: "raw HF dequantization matrix",
                })?
                .clone()
        } else {
            parse_ma_config(reader, MaTreeLimits::default()).map_err(map_modular_error)?
        };
        let token_bit_offset = u32::try_from(reader.bit_offset()).map_err(|_| {
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "raw HF dequantization matrix entropy bit offset",
            }
        })?;
        let channel_metadata = transform_plan
            .topology
            .gpu_entropy_channels(ma_config.maximum_tree_property())
            .map_err(map_modular_error)?;
        let decoded_words = channel_metadata
            .channels
            .last()
            .map_or(0, |channel| channel.decoded_end);
        let maximum_width = channel_metadata
            .channels
            .iter()
            .map(|channel| channel.width)
            .max()
            .ok_or(BoundedVarDctPacketError::HfDequantMatrixValue {
                matrix: matrix_index,
                reason: "raw Modular topology has no entropy channels",
            })?;
        let inverse_plan = plan_modular_inverse(&transform_plan).map_err(map_modular_error)?;
        let final_planes: [GpuModularChannelLayout; 3] = inverse_plan
            .final_gpu_layouts()
            .try_into()
            .map_err(|_| BoundedVarDctPacketError::HfDequantMatrixValue {
                matrix: matrix_index,
                reason: "raw Modular inverse does not produce three channels",
            })?;
        if final_planes.iter().any(|plane| {
            plane.width != width || plane.height != height || plane.hshift != 0 || plane.vshift != 0
        }) {
            return Err(BoundedVarDctPacketError::HfDequantMatrixValue {
                matrix: matrix_index,
                reason: "raw Modular inverse does not restore the matrix extent",
            });
        }
        let lz77_window_words = ma_config
            .entropy
            .lz77_window_words(maximum_width, decoded_words)
            .map_err(map_modular_error)?;
        let PackedModularMetadata { words: metadata } =
            ma_config.pack_gpu_metadata().map_err(map_modular_error)?;
        let low_frequency_group_count = u32::try_from(low_frequency_group_count).map_err(|_| {
            BoundedVarDctPacketError::ArithmeticOverflow {
                field: "raw HF dequantization matrix LF-group count",
            }
        })?;
        let stream_index = low_frequency_group_count
            .checked_mul(3)
            .and_then(|index| index.checked_add(1))
            .and_then(|index| index.checked_add(matrix_index as u32))
            .ok_or(BoundedVarDctPacketError::ArithmeticOverflow {
                field: "raw HF dequantization matrix stream index",
            })?;

        Ok(Self {
            matrix_index,
            bit_depth,
            denominator,
            stream_index,
            token_bit_offset,
            wp_header,
            metadata,
            needs_self_correcting: ma_config.needs_self_correcting(),
            channel_metadata,
            inverse_plan,
            final_planes,
            decoded_words,
            maximum_width,
            lz77_window_words,
        })
    }
}

fn map_modular_error(error: crate::Error) -> BoundedVarDctPacketError {
    match error {
        crate::Error::Bitstream(source) => BoundedVarDctPacketError::Bitstream(source),
        source => BoundedVarDctPacketError::ModularTree(source.to_string()),
    }
}

#[must_use]
pub(crate) const fn raw_matrix_extent(matrix_index: usize) -> Option<[u32; 2]> {
    if matrix_index >= RAW_MATRIX_COUNT {
        return None;
    }
    Some(match matrix_index {
        0 | 1 | 2 | 3 | 9 | 10 => [8, 8],
        4 => [16, 16],
        5 => [32, 32],
        6 => [16, 8],
        7 => [32, 8],
        8 => [32, 16],
        11 => [64, 64],
        12 => [64, 32],
        13 => [128, 128],
        14 => [128, 64],
        15 => [256, 256],
        16 => [256, 128],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use jxl_gpu_bitstream::{BitReader, PrefixCodeEntry};

    use super::*;
    use crate::modular_tree::{
        EntropyCoderIr, EntropyDecoderIr, HybridIntegerConfigIr, MaTreeNodeIr, PrefixHistogramIr,
    };

    fn single_zero_config() -> MaConfigIr {
        MaConfigIr {
            nodes: vec![MaTreeNodeIr::Leaf {
                cluster: 0,
                predictor: 0,
                offset: 0,
                multiplier: 1,
            }],
            max_depth: 0,
            entropy: EntropyDecoderIr {
                lz77: None,
                context_to_cluster: vec![0],
                configs: vec![HybridIntegerConfigIr {
                    split_exponent: 0,
                    msb_in_token: 0,
                    lsb_in_token: 0,
                }],
                coder: EntropyCoderIr::Prefix(vec![PrefixHistogramIr {
                    entries: vec![PrefixCodeEntry::EMPTY],
                    single_symbol: Some(0),
                }]),
            },
        }
    }

    #[test]
    fn raw_matrix_extents_match_the_normative_representatives() {
        assert_eq!(raw_matrix_extent(0), Some([8, 8]));
        assert_eq!(raw_matrix_extent(6), Some([16, 8]));
        assert_eq!(raw_matrix_extent(14), Some([128, 64]));
        assert_eq!(raw_matrix_extent(16), Some([256, 128]));
        assert_eq!(raw_matrix_extent(17), None);
    }

    #[test]
    fn raw_matrix_header_lowers_through_the_general_modular_contract() {
        // f16(1/2040), global tree, default weighted predictor, zero transforms.
        let bytes = [0x04, 0x18, 0x03];
        let mut reader = BitReader::new(&bytes);
        let plan =
            RawHfDequantSideImagePlan::parse(&mut reader, 6, 8, 5, Some(&single_zero_config()))
                .unwrap();

        assert_eq!(plan.matrix_index, 6);
        assert_eq!(plan.stream_index, 22);
        assert_eq!(plan.token_bit_offset, 20);
        assert_eq!(
            plan.final_planes.map(|plane| [plane.width, plane.height]),
            [[16, 8]; 3]
        );
        assert_eq!(plan.decoded_words, 16 * 8 * 3);
        assert_eq!(plan.maximum_width, 16);
        assert_eq!(plan.inverse_plan.jobs(), &[]);
        assert!(!plan.needs_self_correcting);
    }

    #[test]
    fn raw_matrix_requires_a_positive_denominator_and_available_global_tree() {
        let mut zero = BitReader::new(&[0, 0, 0]);
        assert!(matches!(
            RawHfDequantSideImagePlan::parse(&mut zero, 0, 8, 1, Some(&single_zero_config())),
            Err(BoundedVarDctPacketError::HfDequantMatrixValue {
                matrix: 0,
                reason: "raw denominator must be finite and positive"
            })
        ));

        let mut missing = BitReader::new(&[0x00, 0x3c, 0x03]);
        assert!(matches!(
            RawHfDequantSideImagePlan::parse(&mut missing, 0, 8, 1, None),
            Err(BoundedVarDctPacketError::MissingGlobalMaTree {
                stage: "raw HF dequantization matrix"
            })
        ));
    }
}
