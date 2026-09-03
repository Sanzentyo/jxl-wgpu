use jxl_gpu_bitstream::{
    EdgePreservingFilterInventory, GaborishInventory, RestorationFilterInventory,
};
use jxl_gpu_protocol::EpfPass;
use jxl_wgpu::{ResidentEpfParameters, ResidentGaborishWeights};

use crate::vardct_epf::EpfSigmaConfig;
use crate::vardct_packet::BoundedVarDctPacketPlan;

use super::source::VarDctGroupSource;
use super::types::VarDctDecodeError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct VarDctEpfHeader {
    pub(super) iterations: u32,
    pub(super) sharp_lut: [f32; 8],
    pub(super) channel_scale: [f32; 3],
    pub(super) quant_mul: f32,
    pub(super) pass0_sigma_scale: f32,
    pub(super) pass2_sigma_scale: f32,
    pub(super) border_sad_mul: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VarDctEpfPlan {
    pub(crate) sigma_groups: Vec<EpfSigmaConfig>,
    pub(crate) passes: Vec<ResidentEpfParameters>,
}

impl VarDctEpfHeader {
    pub(super) fn plan(
        self,
        packet: &BoundedVarDctPacketPlan,
        groups: &[VarDctGroupSource],
        global_scale: u32,
    ) -> Result<VarDctEpfPlan, VarDctDecodeError> {
        let mut passes = Vec::with_capacity(self.iterations as usize);
        if self.iterations >= 3 {
            passes.push(ResidentEpfParameters {
                pass: EpfPass::Pass0,
                sigma_scale: self.pass0_sigma_scale,
                border_sad_mul: self.border_sad_mul,
                channel_scale: self.channel_scale,
            });
        }
        if self.iterations >= 1 {
            passes.push(ResidentEpfParameters {
                pass: EpfPass::Pass1,
                sigma_scale: 1.0,
                border_sad_mul: self.border_sad_mul,
                channel_scale: self.channel_scale,
            });
        }
        if self.iterations >= 2 {
            passes.push(ResidentEpfParameters {
                pass: EpfPass::Pass2,
                sigma_scale: self.pass2_sigma_scale,
                border_sad_mul: self.border_sad_mul,
                channel_scale: self.channel_scale,
            });
        }
        debug_assert_eq!(passes.len(), self.iterations as usize);
        let [output_blocks_x, output_blocks_y] = packet.block_extent();
        let sigma_groups = packet
            .groups
            .iter()
            .zip(groups)
            .map(|(packet_group, group)| {
                let [blocks_x, blocks_y] = packet_group.block_extent();
                Ok(EpfSigmaConfig {
                    blocks_x,
                    blocks_y,
                    output_blocks_x,
                    output_blocks_y,
                    output_origin: [packet_group.rect.x / 8, packet_group.rect.y / 8],
                    task_count: packet_group.task_capacity,
                    sharpness_offset_words: group.control.expected[3],
                    artifact_status_offset_words: group.artifact_layout.status_offset_words,
                    task_metadata_offset_words: group.artifact_layout.task_metadata_offset_words,
                    global_scale,
                    quant_mul: self.quant_mul,
                    sharp_lut: self.sharp_lut,
                })
            })
            .collect::<Result<Vec<_>, VarDctDecodeError>>()?;
        Ok(VarDctEpfPlan {
            sigma_groups,
            passes,
        })
    }
}

pub(super) fn dequant_matrix_multiplier(
    channel: &'static str,
    scale: u32,
) -> Result<f32, VarDctDecodeError> {
    // JPEG XL 3-bit X/B quant-matrix scale: (1 / 1.25)^(scale - 2).
    const MULTIPLIERS: [f32; 8] = [1.5625, 1.25, 1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768];
    MULTIPLIERS
        .get(scale as usize)
        .copied()
        .ok_or(VarDctDecodeError::InvalidQuantMatrixScale { channel, scale })
}

pub(super) fn restoration_config(
    restoration: RestorationFilterInventory,
) -> Result<(Option<ResidentGaborishWeights>, Option<VarDctEpfHeader>), VarDctDecodeError> {
    let (gaborish, epf) = match restoration {
        RestorationFilterInventory::Default => (
            GaborishInventory::Default,
            EdgePreservingFilterInventory::default(),
        ),
        RestorationFilterInventory::Custom { gaborish, epf } => (gaborish, epf),
    };
    let gaborish = match gaborish {
        GaborishInventory::Disabled => None,
        GaborishInventory::Default => Some(ResidentGaborishWeights::DEFAULT),
        GaborishInventory::Custom { weights } => Some(ResidentGaborishWeights {
            x: weights[0].map(|value| value.to_f32()),
            y: weights[1].map(|value| value.to_f32()),
            b: weights[2].map(|value| value.to_f32()),
        }),
    };
    let epf = match epf {
        EdgePreservingFilterInventory::Disabled => None,
        EdgePreservingFilterInventory::Enabled {
            iterations,
            sharp_lut,
            weights,
            sigma,
            sigma_for_modular: _,
        } => {
            if !(1..=3).contains(&iterations) {
                return Err(VarDctDecodeError::InvalidEpfIterations { iterations });
            }
            let sharp_lut = sharp_lut.map_or(
                [
                    0.0,
                    1.0 / 7.0,
                    2.0 / 7.0,
                    3.0 / 7.0,
                    4.0 / 7.0,
                    5.0 / 7.0,
                    6.0 / 7.0,
                    1.0,
                ],
                |values| values.map(|value| value.to_f32()),
            );
            let channel_scale = weights.map_or([40.0, 5.0, 3.5], |weights| {
                weights.channel_scale.map(|value| value.to_f32())
            });
            let (quant_mul, pass0_sigma_scale, pass2_sigma_scale, border_sad_mul) =
                sigma.map_or((0.46, 0.9, 6.5, 2.0 / 3.0), |sigma| {
                    (
                        sigma.quant_mul.map_or(0.46, |value| value.to_f32()),
                        sigma.pass0_sigma_scale.to_f32(),
                        sigma.pass2_sigma_scale.to_f32(),
                        sigma.border_sad_mul.to_f32(),
                    )
                });
            Some(VarDctEpfHeader {
                iterations,
                sharp_lut,
                channel_scale,
                quant_mul,
                pass0_sigma_scale,
                pass2_sigma_scale,
                border_sad_mul,
            })
        }
    };
    Ok((gaborish, epf))
}

pub(super) struct RestorationCursor<'a> {
    pub(super) image: &'a [wgpu::Buffer; 3],
    pub(super) scratch: &'a [wgpu::Buffer; 3],
    pub(super) current_is_scratch: bool,
}

impl<'a> RestorationCursor<'a> {
    pub(super) fn new(image: &'a [wgpu::Buffer; 3], scratch: &'a [wgpu::Buffer; 3]) -> Self {
        Self {
            image,
            scratch,
            current_is_scratch: false,
        }
    }

    pub(super) fn advance(&mut self) -> (&'a [wgpu::Buffer; 3], &'a [wgpu::Buffer; 3]) {
        let pair = if self.current_is_scratch {
            (self.scratch, self.image)
        } else {
            (self.image, self.scratch)
        };
        self.current_is_scratch = !self.current_is_scratch;
        pair
    }

    pub(super) fn current(&self) -> &'a [wgpu::Buffer; 3] {
        if self.current_is_scratch {
            self.scratch
        } else {
            self.image
        }
    }
}
