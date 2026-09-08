//! Shared lowering of JPEG XL restoration metadata for both coding modes.

use jxl_gpu_bitstream::{
    EdgePreservingFilterInventory, GaborishInventory, RestorationFilterInventory,
};
use jxl_gpu_protocol::EpfPass;
use jxl_wgpu::{ResidentEpfParameters, ResidentGaborishWeights};

#[derive(Clone, Debug, thiserror::Error)]
pub enum RestorationError {
    #[error("invalid EPF iteration count {iterations}")]
    InvalidEpfIterations { iterations: u32 },
    #[error("Modular EPF sigma {value} must be at least 1e-8")]
    InvalidModularSigma { value: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EpfConfig {
    pub(crate) iterations: u32,
    pub(crate) sharp_lut: [f32; 8],
    pub(crate) channel_scale: [f32; 3],
    pub(crate) quant_mul: f32,
    pub(crate) pass0_sigma_scale: f32,
    pub(crate) pass2_sigma_scale: f32,
    pub(crate) border_sad_mul: f32,
}

impl EpfConfig {
    pub(crate) fn passes(self) -> Vec<ResidentEpfParameters> {
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
        passes
    }
}

pub(crate) fn restoration_config(
    restoration: RestorationFilterInventory,
) -> Result<(Option<ResidentGaborishWeights>, Option<EpfConfig>), RestorationError> {
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
                return Err(RestorationError::InvalidEpfIterations { iterations });
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
            Some(EpfConfig {
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
