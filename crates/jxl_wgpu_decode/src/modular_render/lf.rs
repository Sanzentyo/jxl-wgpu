//! LF dependencies use the same reconstruction as presentation frames, before color conversion.

use jxl_gpu_bitstream::UpsamplingWeightsInventory;
use jxl_wgpu::{
    GpuBufferLease, MemoryPermit, ResidentStorageBinding, ResidentUpsampleKernel,
    ResidentUpsampleWeights,
};

use super::color::{
    ModularColorConfig, ReconstructionBuffers, ReconstructionInputs, ReconstructionPipeline,
    ReconstructionPlan,
};
use super::{ModularOutputPlane, Result, invalid, require, upsample_kernel};
use crate::progressive_dc::ProgressiveDcXybPlanes;

pub(crate) struct ModularLfPipelines<'a> {
    pub(crate) color: &'a ReconstructionPipeline,
    pub(crate) extras: Option<&'a super::ModularRenderPipeline>,
}

#[derive(Debug)]
pub(crate) struct ModularLfPlan {
    reconstruction: ReconstructionPlan,
    kernel: Option<ResidentUpsampleKernel>,
    extras: Option<super::ModularRenderPlan>,
}

impl ModularLfPlan {
    pub(crate) fn new(
        config: ModularColorConfig,
        sources: &[ModularOutputPlane],
        resampling: &[crate::frame_resampling::ChannelResampling],
        weights: &UpsamplingWeightsInventory,
        limits: &wgpu::Limits,
    ) -> Result<Self> {
        if sources.len() < 3 || sources.len() != resampling.len() {
            return invalid("LF reconstruction requires three color planes and matching factors");
        }
        if resampling[..3]
            .iter()
            .any(|channel| *channel != resampling[0])
        {
            return invalid("LF color resampling grids differ");
        }
        let crate::frame_resampling::ChannelResampling { extent, factor } = resampling[0];
        let reconstruction =
            ReconstructionPlan::new(config, extent, &sources[..3], factor, limits)?;
        let extras = (sources.len() > 3)
            .then(|| {
                super::ModularRenderPlan::new(
                    sources[3..].to_vec(),
                    resampling[3..].to_vec(),
                    weights,
                    limits,
                )
            })
            .transpose()?;
        let kernel = (factor != 1)
            .then(|| upsample_kernel(weights, factor))
            .transpose()?;
        if let Some(kernel) = &kernel {
            require(
                "LF upsampling weights",
                kernel.weight_bytes(),
                limits
                    .max_buffer_size
                    .min(limits.max_storage_buffer_binding_size),
            )?;
        }
        Ok(Self {
            reconstruction,
            kernel,
            extras,
        })
    }

    pub(crate) fn has_extras(&self) -> bool {
        self.extras.is_some()
    }

    pub(crate) fn plane_bytes(&self) -> u64 {
        let extent = self.reconstruction.output_extent;
        u64::from(extent.width) * u64::from(extent.height) * 4 * 3
            + self.extras.as_ref().map_or(0, |plan| plan.output_bytes)
    }

    pub(crate) fn uniform_bytes(&self) -> u64 {
        self.reconstruction.uniform_bytes
            + self.extras.as_ref().map_or(0, |plan| plan.uniform_bytes)
    }

    pub(crate) fn total_bytes(&self) -> u64 {
        self.reconstruction.storage_bytes
            + self.reconstruction.uniform_bytes
            + self.extras.as_ref().map_or(0, |plan| plan.total_bytes())
            + self
                .kernel
                .as_ref()
                .map_or(0, ResidentUpsampleKernel::weight_bytes)
    }

    pub(crate) fn allocate(
        &self,
        device: &wgpu::Device,
        permit: &mut MemoryPermit,
    ) -> crate::Result<(ModularLfBuffers, crate::progressive_dc::ProgressiveDcOutput)> {
        let reconstruction = self.reconstruction.allocate(device);
        let output = self.reconstruction.output_buffers(&reconstruction);
        let reservations = [
            permit
                .split_off(output[0].size())
                .map_err(crate::progressive_dc::ProgressiveDcGpuError::from)?,
            permit
                .split_off(output[1].size())
                .map_err(crate::progressive_dc::ProgressiveDcGpuError::from)?,
            permit
                .split_off(output[2].size())
                .map_err(crate::progressive_dc::ProgressiveDcGpuError::from)?,
        ];
        let mut reservations = reservations.into_iter();
        let extent = self.reconstruction.output_extent;
        let planes = ProgressiveDcXybPlanes::from_leases(
            output.clone().map(|buffer| {
                GpuBufferLease::from_tracked(
                    buffer,
                    reservations.next().expect("three LF plane reservations"),
                )
            }),
            extent.width,
            extent.height,
            extent.width,
        )?;
        let extra_buffers = self
            .extras
            .as_ref()
            .map(|plan| plan.allocate(device))
            .transpose()?;
        let extras = self
            .extras
            .as_ref()
            .zip(extra_buffers.as_ref())
            .map(|(plan, buffers)| {
                crate::progressive_dc::ProgressiveDcExtras::retain(plan, buffers, permit)
            })
            .transpose()?;
        Ok((
            ModularLfBuffers {
                extras: extra_buffers,
                reconstruction,
                weights: self
                    .kernel
                    .as_ref()
                    .map(|kernel| kernel.upload(device))
                    .transpose()
                    .map_err(super::ModularRenderError::from)?,
            },
            crate::progressive_dc::ProgressiveDcOutput {
                xyb: planes,
                extras,
            },
        ))
    }

    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: ModularLfPipelines<'_>,
        buffers: &ModularLfBuffers,
        source: ResidentStorageBinding<'_>,
        sources: &[ModularOutputPlane],
    ) -> Result<Vec<wgpu::Buffer>> {
        let mut uniforms = pipelines.color.encode(
            device,
            encoder,
            ReconstructionInputs {
                plan: &self.reconstruction,
                buffers: &buffers.reconstruction,
                source,
                sources: &sources[..3],
                weights: buffers.weights.as_ref(),
            },
        )?;
        if let Some(plan) = &self.extras {
            let pipeline = pipelines.extras.ok_or(super::ModularRenderError::Invalid {
                reason: "LF extra pipeline is missing",
            })?;
            let buffers = buffers
                .extras
                .as_ref()
                .ok_or(super::ModularRenderError::Invalid {
                    reason: "LF extra buffers are missing",
                })?;
            uniforms.extend(pipeline.encode(
                device,
                encoder,
                plan,
                buffers,
                source,
                &sources[3..],
            )?);
        }
        Ok(uniforms)
    }
}

pub(crate) struct ModularLfBuffers {
    extras: Option<super::ModularRenderBuffers>,
    reconstruction: ReconstructionBuffers,
    weights: Option<ResidentUpsampleWeights>,
}
