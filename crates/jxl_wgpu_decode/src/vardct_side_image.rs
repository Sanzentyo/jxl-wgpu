//! Resident Modular side images embedded in VarDCT metadata.
//!
//! JPEG XL mode-7 HF dequantization matrices are not scalar metadata: each one is a complete
//! three-channel Modular image whose entropy cursor determines where the following HF-global
//! metadata begins. This module lowers that local header into the same topology, MA-tree,
//! previous-channel reference, and inverse-transform contracts used by the main Modular decoder.

use crate::modular_side_image::ModularSideImagePlan;
use crate::modular_transform::{ModularChannelTopology, ModularTransformLimits};
use crate::modular_tree::{BitInput, MaConfigIr};
use crate::vardct_frontend::metadata_f16;
use crate::vardct_packet::BoundedVarDctPacketError;

const RAW_MATRIX_COUNT: usize = 17;

/// Host-known execution contract for one raw HF dequantization matrix side image.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RawHfDequantSideImagePlan {
    pub matrix_index: usize,
    pub denominator: f32,
    pub image: ModularSideImagePlan,
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

        let topology = ModularChannelTopology::full_resolution(
            width,
            height,
            bit_depth,
            3,
            ModularTransformLimits::default(),
        )
        .map_err(map_modular_error)?;
        let image = ModularSideImagePlan::parse(
            reader,
            topology,
            bit_depth,
            stream_index,
            global_ma_config,
        )
        .map_err(map_modular_error)?;
        if image.final_planes.len() != 3
            || image.final_planes.iter().any(|plane| {
                plane.width != width
                    || plane.height != height
                    || plane.hshift != 0
                    || plane.vshift != 0
            })
        {
            return Err(BoundedVarDctPacketError::HfDequantMatrixValue {
                matrix: matrix_index,
                reason: "raw Modular inverse does not restore the matrix extent",
            });
        }
        Ok(Self {
            matrix_index,
            denominator,
            image,
        })
    }
}

fn map_modular_error(error: crate::Error) -> BoundedVarDctPacketError {
    match error {
        crate::Error::MissingGlobalMaTree { .. } => BoundedVarDctPacketError::MissingGlobalMaTree {
            stage: "raw HF dequantization matrix",
        },
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
        assert_eq!(plan.image.stream_index, 22);
        assert_eq!(plan.image.token_bit_offset, 20);
        assert_eq!(
            plan.image
                .final_planes
                .iter()
                .map(|plane| [plane.width, plane.height])
                .collect::<Vec<_>>(),
            [[16, 8]; 3]
        );
        assert_eq!(plan.image.decoded_words, 16 * 8 * 3);
        assert_eq!(plan.image.maximum_width, 16);
        assert_eq!(plan.image.inverse_plan.jobs(), &[]);
        assert!(!plan.image.needs_self_correcting);
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
