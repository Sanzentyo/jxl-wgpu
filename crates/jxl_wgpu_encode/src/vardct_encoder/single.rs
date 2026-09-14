//! Resident normalization, forward transform, quantization and AC serialization.

use jxl_wgpu::{
    ForwardVarDctInputs, ForwardVarDctPipeline, ForwardVarDctScratch, KernelVariant,
    ResidentF32Plane, ResidentStorageBinding,
};
use wgpu::util::DeviceExt;

use super::dispatch::shader_source;
use super::types::{VarDctStrategy, VarDctTransformMemoryPlan};
use crate::EncodeError;

pub(super) struct Pipeline {
    normalize: wgpu::ComputePipeline,
    quantize_ac: wgpu::ComputePipeline,
    quantize_lf: wgpu::ComputePipeline,
    serialize_ac: wgpu::ComputePipeline,
    serialize_lf: wgpu::ComputePipeline,
    forward: ForwardVarDctPipeline,
    variant: KernelVariant,
}

pub(super) struct Inputs<'a> {
    pub(super) strategy: VarDctStrategy,
    pub(super) source: wgpu::BufferBinding<'a>,
    pub(super) parameters: &'a wgpu::Buffer,
    pub(super) artifact: &'a wgpu::Buffer,
}

/// These handles keep every charged resident allocation alive until completion.
pub(super) struct Scratch {
    _buffers: [wgpu::Buffer; 5],
    _forward: ForwardVarDctScratch,
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl Pipeline {
    pub(super) fn new(device: &wgpu::Device, variant: KernelVariant) -> Result<Self, EncodeError> {
        let forward = ForwardVarDctPipeline::new(device, variant)?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu general VarDCT encoder"),
            source: wgpu::ShaderSource::Wgsl(shader_source(include_str!("single.wgsl")).into()),
        });
        let constants = [("wg_x", f64::from(variant.workgroup_size().0))];
        let make = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        };
        Ok(Self {
            normalize: make("normalize_single"),
            quantize_ac: make("quantize_single_ac"),
            quantize_lf: make("quantize_single_lf"),
            serialize_ac: make("serialize_single_ac"),
            serialize_lf: make("serialize_control"),
            forward,
            variant,
        })
    }

    pub(super) fn encode(
        &self,
        device: &wgpu::Device,
        commands: &mut wgpu::CommandEncoder,
        inputs: Inputs<'_>,
    ) -> Result<Scratch, EncodeError> {
        let memory = VarDctTransformMemoryPlan::new(inputs.strategy);
        let transform = inputs.strategy;
        let extent = transform.pixel_extent();
        let area = extent.width * extent.height;
        let lf = transform.lf_extent();
        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let xyb = buffer("resident encoder XYB", memory.xyb_bytes);
        let coefficients = buffer(
            "resident encoder forward coefficients",
            memory.coefficient_bytes,
        );
        let low_frequency = buffer("resident encoder forward LF", memory.lf_bytes);
        let quantized = buffer(
            "resident encoder quantized coefficients",
            memory.quantized_bytes,
        );
        let order = transform.natural_order();
        let metadata = transform
            .default_dequant_matrix()
            .scales
            .into_iter()
            .zip(order)
            .map(|([x, y, b], order)| [x.to_bits(), y.to_bits(), b.to_bits(), order])
            .collect::<Vec<_>>();
        let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("encoder quantization and order metadata"),
            contents: bytemuck::cast_slice(&metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lanes = self.variant.workgroup_size().0;
        let workgroups_x = area
            .div_ceil(lanes)
            .min(device.limits().max_compute_workgroups_per_dimension);
        let dispatch = |commands: &mut wgpu::CommandEncoder,
                        pipeline: &wgpu::ComputePipeline,
                        entries: &[wgpu::BindGroupEntry<'_>],
                        count: u32,
                        scalar: bool| {
            let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("single-transform encoder stage bindings"),
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            });
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &binding, &[]);
            if scalar {
                pass.dispatch_workgroups(1, 1, 1);
            } else {
                let groups = count.div_ceil(lanes);
                pass.dispatch_workgroups(
                    groups.min(workgroups_x),
                    groups.div_ceil(workgroups_x),
                    1,
                );
            }
        };
        dispatch(
            commands,
            &self.normalize,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(inputs.source),
                },
                entry(1, inputs.parameters),
                entry(5, &xyb),
            ],
            area,
            false,
        );
        let channel_bytes = memory.xyb_bytes / 3;
        let forward = self.forward.encode(
            device,
            commands,
            ForwardVarDctInputs {
                transform,
                sources: std::array::from_fn(|channel| ResidentF32Plane {
                    storage: ResidentStorageBinding {
                        buffer: &xyb,
                        offset: channel as u64 * channel_bytes,
                        size: std::num::NonZeroU64::new(channel_bytes).unwrap(),
                    },
                    width: extent.width,
                    height: extent.height,
                    stride: 0,
                }),
                origins: [0; 3],
                coefficients: ResidentStorageBinding::entire(&coefficients)
                    .map_err(jxl_wgpu::ForwardVarDctError::from)?,
                low_frequency: ResidentStorageBinding::entire(&low_frequency)
                    .map_err(jxl_wgpu::ForwardVarDctError::from)?,
            },
        )?;
        dispatch(
            commands,
            &self.quantize_ac,
            &[
                entry(1, inputs.parameters),
                entry(3, &coefficients),
                entry(6, &quantized),
                entry(7, &metadata),
            ],
            area,
            false,
        );
        dispatch(
            commands,
            &self.quantize_lf,
            &[
                entry(1, inputs.parameters),
                entry(2, inputs.artifact),
                entry(4, &low_frequency),
            ],
            lf.width * lf.height,
            false,
        );
        dispatch(
            commands,
            &self.serialize_ac,
            &[
                entry(1, inputs.parameters),
                entry(2, inputs.artifact),
                entry(6, &quantized),
                entry(7, &metadata),
            ],
            1,
            true,
        );
        dispatch(
            commands,
            &self.serialize_lf,
            &[entry(1, inputs.parameters), entry(2, inputs.artifact)],
            1,
            true,
        );
        Ok(Scratch {
            _buffers: [xyb, coefficients, low_frequency, quantized, metadata],
            _forward: forward,
        })
    }
}
