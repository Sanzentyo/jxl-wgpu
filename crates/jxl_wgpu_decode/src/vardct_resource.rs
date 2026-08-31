//! Default resource table and LF dequantization for the strict zero-AC VarDCT profile.

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

const RESOURCE_SHADER: &str = include_str!("vardct_resource.wgsl");
const GLOBAL_SCALE: f32 = 8_813.0;
const QUANT_LF: f32 = 10.0;
const HF_MUL: f32 = 6.0;

/// Vec4-indexed resource table consumed by the resident VarDCT renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctResourceLayout {
    pub quant_offset: u32,
    pub correlation_offset: u32,
    pub lf_offset: u32,
    pub matrix_offset: u32,
    pub vector_count: u32,
    pub block_count: u32,
    pub correlation_count: u32,
    pub transform_area: u32,
}

impl VarDctResourceLayout {
    pub fn new(
        blocks_x: u32,
        blocks_y: u32,
        transform_area: u32,
    ) -> Result<Self, VarDctResourceError> {
        let block_count =
            blocks_x
                .checked_mul(blocks_y)
                .ok_or(VarDctResourceError::ArithmeticOverflow {
                    field: "LF block count",
                })?;
        let correlation_count = blocks_x
            .div_ceil(8)
            .checked_mul(blocks_y.div_ceil(8))
            .ok_or(VarDctResourceError::ArithmeticOverflow {
                field: "correlation cell count",
            })?;
        let quant_offset = 0_u32;
        let correlation_offset = 1_u32;
        let lf_offset = correlation_offset.checked_add(correlation_count).ok_or(
            VarDctResourceError::ArithmeticOverflow {
                field: "LF resource offset",
            },
        )?;
        let matrix_offset =
            lf_offset
                .checked_add(block_count)
                .ok_or(VarDctResourceError::ArithmeticOverflow {
                    field: "matrix offset",
                })?;
        let vector_count = matrix_offset.checked_add(transform_area).ok_or(
            VarDctResourceError::ArithmeticOverflow {
                field: "resource vector count",
            },
        )?;
        Ok(Self {
            quant_offset,
            correlation_offset,
            lf_offset,
            matrix_offset,
            vector_count,
            block_count,
            correlation_count,
            transform_area,
        })
    }

    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.vector_count as u64 * 16
    }

    /// Builds immutable quant/correlation values and the semantically unused zero-AC matrix.
    ///
    /// The accepted packet's GPU status proves every AC coefficient is zero before this table is
    /// authoritative. Matrix entries therefore cannot affect any accepted output; finite ones are
    /// retained solely so the common resident shader has a complete in-bounds resource range.
    #[must_use]
    pub fn initial_values(self) -> Vec<[f32; 4]> {
        let mut values = vec![[0.0; 4]; self.vector_count as usize];
        values[self.quant_offset as usize] = [
            0.8 * 65_536.0 / (GLOBAL_SCALE * HF_MUL),
            65_536.0 / (GLOBAL_SCALE * HF_MUL),
            0.8 * 65_536.0 / (GLOBAL_SCALE * HF_MUL),
            0.0,
        ];
        let correlation_end = self.correlation_offset + self.correlation_count;
        values[self.correlation_offset as usize..correlation_end as usize]
            .fill([0.0, 1.0, 0.0, 0.0]);
        values[self.matrix_offset as usize..].fill([1.0, 1.0, 1.0, 0.0]);
        values
    }
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum VarDctResourceError {
    #[error("VarDCT resource arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
}

/// Exact 64-byte LF preparation uniform.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VarDctResourceParams {
    pub geometry: [u32; 4],
    pub offsets: [u32; 4],
    pub scales: [f32; 4],
    pub _reserved: [u32; 4],
}

impl VarDctResourceParams {
    pub fn new(blocks_x: u32, blocks_y: u32) -> Result<Self, VarDctResourceError> {
        let blocks =
            blocks_x
                .checked_mul(blocks_y)
                .ok_or(VarDctResourceError::ArithmeticOverflow {
                    field: "LF preparation block count",
                })?;
        let denominator = GLOBAL_SCALE * QUANT_LF;
        Ok(Self {
            geometry: [blocks_x, blocks_y, blocks, 0],
            offsets: [0, blocks, 2 * blocks, 0],
            scales: [
                16.0 / denominator,
                128.0 / denominator,
                256.0 / denominator,
                1.0,
            ],
            _reserved: [0; 4],
        })
    }

    #[must_use]
    pub const fn dispatch(self) -> u32 {
        self.geometry[2].div_ceil(64)
    }

    #[must_use]
    pub fn smoothing_thresholds(self) -> [f32; 3] {
        let denominator = GLOBAL_SCALE * QUANT_LF;
        [16.0 / denominator, 128.0 / denominator, 256.0 / denominator]
    }
}

pub struct VarDctResourceBuffers<'a> {
    pub quantized_lf: &'a wgpu::Buffer,
    pub dequantized_lf: &'a wgpu::Buffer,
}

pub struct VarDctResourcePipeline {
    pipeline: wgpu::ComputePipeline,
}

impl VarDctResourcePipeline {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu VarDCT LF resource preparation"),
            source: wgpu::ShaderSource::Wgsl(RESOURCE_SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu VarDCT LF resource preparation"),
            layout: None,
            module: &module,
            entry_point: Some("prepare_lf"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline }
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffers: VarDctResourceBuffers<'_>,
        params: VarDctResourceParams,
    ) -> wgpu::Buffer {
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu VarDCT LF resource params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu VarDCT LF resource bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, buffers.quantized_lf),
                entry(1, buffers.dequantized_lf),
                entry(2, &uniform),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu VarDCT LF resource preparation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(params.dispatch(), 1, 1);
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
    assert!(std::mem::size_of::<VarDctResourceParams>() == 64);
    assert!(std::mem::align_of::<VarDctResourceParams>() == 16);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_and_shader_are_bounded() {
        let layout = VarDctResourceLayout::new(4, 2, 256).unwrap();
        assert_eq!(layout.correlation_count, 1);
        assert_eq!(layout.lf_offset, 2);
        assert_eq!(layout.matrix_offset, 10);
        assert_eq!(layout.vector_count, 266);
        assert_eq!(layout.bytes(), 4_256);
        assert_eq!(layout.initial_values().len(), 266);
        let module = naga::front::wgsl::parse_str(RESOURCE_SHADER).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn correlation_grid_scales_past_one_frequency_cell() {
        let layout = VarDctResourceLayout::new(17, 9, 64).unwrap();
        assert_eq!(layout.correlation_count, 6);
        assert_eq!(layout.correlation_offset, 1);
        assert_eq!(layout.lf_offset, 7);
        let values = layout.initial_values();
        assert_eq!(&values[1..7], &[[0.0, 1.0, 0.0, 0.0]; 6]);
    }
}
