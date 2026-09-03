//! GPU ABI for JPEG XL adaptive LF smoothing.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

pub const ADAPTIVE_LF_SHADER: &str = include_str!("vardct_lf.wgsl");
pub const ADAPTIVE_LF_TILE: u32 = 16;

/// One 32-byte, 16-byte-aligned uniform shared with `vardct_lf.wgsl`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AdaptiveLfParams {
    pub extent_and_offsets: [u32; 4],
    pub lf_scale: [f32; 4],
}

impl AdaptiveLfParams {
    pub fn new(
        width: u32,
        height: u32,
        input_offset: u32,
        output_offset: u32,
        lf_scale: [f32; 3],
    ) -> Self {
        Self {
            extent_and_offsets: [width, height, input_offset, output_offset],
            lf_scale: [lf_scale[0], lf_scale[1], lf_scale[2], 0.0],
        }
    }

    pub fn dispatch(self) -> [u32; 2] {
        [
            self.extent_and_offsets[0].div_ceil(ADAPTIVE_LF_TILE),
            self.extent_and_offsets[1].div_ceil(ADAPTIVE_LF_TILE),
        ]
    }
}

/// Layout and boundary calculation for adaptive LF smoothing on subsampled chroma channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubsampledAdaptiveLfLayout {
    pub base_blocks_x: u32,
    pub base_blocks_y: u32,
    pub channel_shifts: [crate::vardct_frontend::VarDctChannelShift; 3],
}

impl SubsampledAdaptiveLfLayout {
    /// Constructs layout for the given base LF grid dimensions and channel shifts.
    pub fn new(
        base_blocks_x: u32,
        base_blocks_y: u32,
        channel_shifts: [crate::vardct_frontend::VarDctChannelShift; 3],
    ) -> Self {
        Self {
            base_blocks_x,
            base_blocks_y,
            channel_shifts,
        }
    }

    /// Computes the block extent for a specific channel (0 = X, 1 = Y, 2 = B).
    pub fn channel_extent(&self, channel: usize) -> jxl_gpu_protocol::Extent2d {
        let shift = self.channel_shifts[channel];
        let width = (self.base_blocks_x >> shift.horizontal).max(1);
        let height = (self.base_blocks_y >> shift.vertical).max(1);
        jxl_gpu_protocol::Extent2d::new(width, height)
    }

    /// Builds [`AdaptiveLfParams`] for one channel with proper boundary alignment.
    pub fn channel_params(
        &self,
        channel: usize,
        input_offset: u32,
        output_offset: u32,
        lf_scale: [f32; 3],
    ) -> AdaptiveLfParams {
        let extent = self.channel_extent(channel);
        AdaptiveLfParams::new(
            extent.width,
            extent.height,
            input_offset,
            output_offset,
            lf_scale,
        )
    }
}

/// GPU buffers consumed by [`AdaptiveLfPipeline::encode`].
pub struct AdaptiveLfBuffers<'a> {
    pub input: &'a wgpu::Buffer,
    pub output: &'a wgpu::Buffer,
}

/// Reusable adaptive LF smoothing pipeline.
pub struct AdaptiveLfPipeline {
    pipeline: wgpu::ComputePipeline,
}

impl AdaptiveLfPipeline {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu VarDCT adaptive LF smoothing"),
            source: wgpu::ShaderSource::Wgsl(ADAPTIVE_LF_SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu VarDCT adaptive LF smoothing"),
            layout: None,
            module: &module,
            entry_point: Some("adaptive_lf_smoothing"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline }
    }

    /// Records the standard adaptive smoothing pass and returns its 32-byte uniform.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffers: AdaptiveLfBuffers<'_>,
        params: AdaptiveLfParams,
    ) -> wgpu::Buffer {
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu VarDCT adaptive LF params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu VarDCT adaptive LF bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, buffers.input),
                entry(1, buffers.output),
                entry(2, &uniform),
            ],
        });
        let [x, y] = params.dispatch();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu VarDCT adaptive LF smoothing"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
        drop(pass);
        uniform
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

const _: () = {
    assert!(std::mem::size_of::<AdaptiveLfParams>() == 32);
    assert!(std::mem::align_of::<AdaptiveLfParams>() == 16);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vardct_frontend::VarDctChannelShift;
    use jxl_gpu_protocol::Extent2d;

    #[test]
    fn subsampled_adaptive_lf_layout_channel_extents() {
        // 4:2:0 subsampling: ch0(X) and ch2(B) shifted by (1, 1), ch1(Y) unshifted (0, 0)
        let shifts_420 = [
            VarDctChannelShift { horizontal: 1, vertical: 1 },
            VarDctChannelShift { horizontal: 0, vertical: 0 },
            VarDctChannelShift { horizontal: 1, vertical: 1 },
        ];
        let layout_420 = SubsampledAdaptiveLfLayout::new(16, 16, shifts_420);
        assert_eq!(layout_420.channel_extent(0), Extent2d::new(8, 8));
        assert_eq!(layout_420.channel_extent(1), Extent2d::new(16, 16));
        assert_eq!(layout_420.channel_extent(2), Extent2d::new(8, 8));

        // 4:2:2 subsampling: ch0 and ch2 shifted horizontally only (1, 0)
        let shifts_422 = [
            VarDctChannelShift { horizontal: 1, vertical: 0 },
            VarDctChannelShift { horizontal: 0, vertical: 0 },
            VarDctChannelShift { horizontal: 1, vertical: 0 },
        ];
        let layout_422 = SubsampledAdaptiveLfLayout::new(16, 16, shifts_422);
        assert_eq!(layout_422.channel_extent(0), Extent2d::new(8, 16));
        assert_eq!(layout_422.channel_extent(1), Extent2d::new(16, 16));
        assert_eq!(layout_422.channel_extent(2), Extent2d::new(8, 16));

        // 4:4:0 subsampling: ch0 and ch2 shifted vertically only (0, 1)
        let shifts_440 = [
            VarDctChannelShift { horizontal: 0, vertical: 1 },
            VarDctChannelShift { horizontal: 0, vertical: 0 },
            VarDctChannelShift { horizontal: 0, vertical: 1 },
        ];
        let layout_440 = SubsampledAdaptiveLfLayout::new(16, 16, shifts_440);
        assert_eq!(layout_440.channel_extent(0), Extent2d::new(16, 8));
        assert_eq!(layout_440.channel_extent(1), Extent2d::new(16, 16));
        assert_eq!(layout_440.channel_extent(2), Extent2d::new(16, 8));
    }

    #[test]
    fn subsampled_adaptive_lf_layout_channel_params() {
        let shifts_420 = [
            VarDctChannelShift { horizontal: 1, vertical: 1 },
            VarDctChannelShift { horizontal: 0, vertical: 0 },
            VarDctChannelShift { horizontal: 1, vertical: 1 },
        ];
        let layout = SubsampledAdaptiveLfLayout::new(32, 32, shifts_420);

        let lf_scale = [0.5, 1.2, 0.8];
        let params_x = layout.channel_params(0, 0, 100, lf_scale);
        assert_eq!(params_x.extent_and_offsets[0], 16);
        assert_eq!(params_x.extent_and_offsets[1], 16);
        assert_eq!(params_x.extent_and_offsets[2], 0);
        assert_eq!(params_x.extent_and_offsets[3], 100);
        assert_eq!(params_x.lf_scale, [0.5, 1.2, 0.8, 0.0]);
        assert_eq!(params_x.dispatch(), [1, 1]);

        let params_y = layout.channel_params(1, 200, 300, lf_scale);
        assert_eq!(params_y.extent_and_offsets[0], 32);
        assert_eq!(params_y.extent_and_offsets[1], 32);
        assert_eq!(params_y.extent_and_offsets[2], 200);
        assert_eq!(params_y.extent_and_offsets[3], 300);
        assert_eq!(params_y.lf_scale, [0.5, 1.2, 0.8, 0.0]);
        assert_eq!(params_y.dispatch(), [2, 2]);
    }
}
