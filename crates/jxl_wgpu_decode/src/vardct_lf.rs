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
