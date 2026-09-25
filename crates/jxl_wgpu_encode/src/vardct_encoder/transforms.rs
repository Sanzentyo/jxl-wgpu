//! Resident normalization, forward transform, quantization and AC serialization.

use jxl_wgpu::{
    ForwardVarDctBatchInputs, ForwardVarDctPipeline, ForwardVarDctScratch, KernelVariant,
    ResidentF32Plane, ResidentStorageBinding,
};
use wgpu::util::DeviceExt;

use super::dispatch::shader_source;
use super::strategy_map::TransformPlan;
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
    pub(super) plan: &'a TransformPlan,
    pub(super) source: wgpu::BufferBinding<'a>,
    pub(super) parameters: &'a wgpu::Buffer,
    pub(super) artifact: &'a wgpu::Buffer,
}

/// These handles keep every charged resident allocation alive until completion.
pub(super) struct Scratch {
    _buffers: [wgpu::Buffer; 6],
    _forward: Vec<ForwardVarDctScratch>,
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
            source: wgpu::ShaderSource::Wgsl(shader_source(include_str!("transforms.wgsl")).into()),
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
            normalize: make("normalize_image"),
            quantize_ac: make("quantize_transforms_ac"),
            quantize_lf: make("quantize_transforms_lf"),
            serialize_ac: make("serialize_transforms_ac"),
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
        let memory = inputs.plan.memory;
        let extent = inputs.plan.map.extent();
        let width = extent.width.div_ceil(8) * 8;
        let height = extent.height.div_ceil(8) * 8;
        let area = width * height;
        let count = inputs.plan.tasks.len() as u32;
        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let components = buffer("resident encoder color components", memory.xyb_bytes);
        let coefficients = buffer(
            "resident encoder forward coefficients",
            memory.coefficient_bytes,
        );
        let low_frequency = buffer("resident encoder forward LF", memory.lf_bytes);
        let quantized = buffer(
            "resident encoder quantized coefficients",
            memory.quantized_bytes,
        );
        let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("encoder quantization and order metadata"),
            contents: bytemuck::cast_slice(&inputs.plan.metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let tasks = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("encoder transform tasks"),
            contents: bytemuck::cast_slice(&inputs.plan.tasks),
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
                        per_transform: bool| {
            let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mapped-transform encoder stage bindings"),
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            });
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &binding, &[]);
            {
                let groups = if per_transform {
                    count
                } else {
                    count.div_ceil(lanes)
                };
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
                entry(2, inputs.artifact),
                entry(5, &components),
            ],
            area,
            false,
        );
        let component_storage = ResidentStorageBinding::entire(&components)
            .map_err(jxl_wgpu::ForwardVarDctError::from)?;
        let mut forward = Vec::with_capacity(inputs.plan.batches.len());
        for batch in &inputs.plan.batches {
            forward.push(
                self.forward.encode_batch(
                    device,
                    commands,
                    ForwardVarDctBatchInputs {
                        transform: batch.strategy,
                        sources: std::array::from_fn(|_| ResidentF32Plane {
                            storage: component_storage,
                            width,
                            height,
                            stride: 0,
                        }),
                        tasks: &batch.tasks,
                        coefficients: ResidentStorageBinding::entire(&coefficients)
                            .map_err(jxl_wgpu::ForwardVarDctError::from)?,
                        low_frequency: ResidentStorageBinding::entire(&low_frequency)
                            .map_err(jxl_wgpu::ForwardVarDctError::from)?,
                    },
                )?,
            );
        }
        dispatch(
            commands,
            &self.quantize_ac,
            &[
                entry(1, inputs.parameters),
                entry(2, inputs.artifact),
                entry(3, &coefficients),
                entry(6, &quantized),
                entry(7, &metadata),
                entry(8, &tasks),
            ],
            count,
            true,
        );
        dispatch(
            commands,
            &self.quantize_lf,
            &[
                entry(1, inputs.parameters),
                entry(2, inputs.artifact),
                entry(4, &low_frequency),
                entry(8, &tasks),
            ],
            count,
            true,
        );
        dispatch(
            commands,
            &self.serialize_ac,
            &[
                entry(1, inputs.parameters),
                entry(2, inputs.artifact),
                entry(6, &quantized),
                entry(7, &metadata),
                entry(8, &tasks),
            ],
            count,
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
            _buffers: [
                components,
                coefficients,
                low_frequency,
                quantized,
                metadata,
                tasks,
            ],
            _forward: forward,
        })
    }
}
