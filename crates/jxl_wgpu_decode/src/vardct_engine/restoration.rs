use crate::restoration::EpfConfig;
use jxl_wgpu::ResidentEpfParameters;

use crate::vardct_epf::EpfSigmaConfig;
use crate::vardct_packet::BoundedVarDctPacketPlan;

use super::source::VarDctGroupSource;
use super::types::VarDctDecodeError;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct VarDctEpfPlan {
    pub(super) sigma_groups: Vec<EpfSigmaConfig>,
    pub(super) passes: Vec<ResidentEpfParameters>,
}

impl EpfConfig {
    pub(super) fn plan(
        self,
        packet: &BoundedVarDctPacketPlan,
        groups: &[VarDctGroupSource],
        global_scale: u32,
    ) -> Result<VarDctEpfPlan, VarDctDecodeError> {
        let passes = self.passes();
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
