//! LF dependencies use the same reconstruction as presentation frames, before color conversion.

use jxl_gpu_bitstream::UpsamplingWeightsInventory;
use jxl_gpu_protocol::Extent2d;
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

#[derive(Debug)]
pub(crate) struct ModularLfPlan {
    reconstruction: ReconstructionPlan,
    kernel: Option<ResidentUpsampleKernel>,
}

impl ModularLfPlan {
    pub(crate) fn new(
        config: ModularColorConfig,
        extent: Extent2d,
        sources: &[ModularOutputPlane],
        factor: u32,
        weights: &UpsamplingWeightsInventory,
        limits: &wgpu::Limits,
    ) -> Result<Self> {
        if sources.len() != 3 {
            return invalid("LF reconstruction requires exactly three color planes");
        }
        let reconstruction = ReconstructionPlan::new(config, extent, sources, factor, limits)?;
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
        })
    }

    pub(crate) fn plane_bytes(&self) -> u64 {
        let extent = self.reconstruction.output_extent;
        u64::from(extent.width) * u64::from(extent.height) * 4 * 3
    }

    pub(crate) fn uniform_bytes(&self) -> u64 {
        self.reconstruction.uniform_bytes
    }

    pub(crate) fn total_bytes(&self) -> u64 {
        self.reconstruction.storage_bytes
            + self.uniform_bytes()
            + self
                .kernel
                .as_ref()
                .map_or(0, ResidentUpsampleKernel::weight_bytes)
    }

    pub(crate) fn allocate(
        &self,
        device: &wgpu::Device,
        permit: &mut MemoryPermit,
    ) -> crate::Result<(ModularLfBuffers, ProgressiveDcXybPlanes)> {
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
        Ok((
            ModularLfBuffers {
                reconstruction,
                weights: self
                    .kernel
                    .as_ref()
                    .map(|kernel| kernel.upload(device))
                    .transpose()
                    .map_err(super::ModularRenderError::from)?,
            },
            planes,
        ))
    }

    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &ReconstructionPipeline,
        buffers: &ModularLfBuffers,
        source: ResidentStorageBinding<'_>,
        sources: &[ModularOutputPlane],
    ) -> Result<Vec<wgpu::Buffer>> {
        pipeline.encode(
            device,
            encoder,
            ReconstructionInputs {
                plan: &self.reconstruction,
                buffers: &buffers.reconstruction,
                source,
                sources,
                weights: buffers.weights.as_ref(),
            },
        )
    }
}

pub(crate) struct ModularLfBuffers {
    reconstruction: ReconstructionBuffers,
    weights: Option<ResidentUpsampleWeights>,
}
