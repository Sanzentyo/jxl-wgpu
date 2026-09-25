//! Checked ICC input normalization, resident color conversion and completion-owned storage.
//! Relative linear BT.709/Gray is the XYB working connection; the original profile remains metadata.

use std::{num::NonZeroU64, sync::Arc};

use jxl_gpu_protocol::{Extent2d, icc::IccTransform};
use jxl_wgpu::{
    KernelVariant, ResidentIccDispatch, ResidentIccInputs, ResidentIccMemoryPlan,
    ResidentIccPipeline, ResidentIccPlane, ResidentIccProgram, ResidentIccSampleEncoding,
    ResidentStorageBinding,
};
use wgpu::util::DeviceExt;

use super::{dispatch::SOURCE_BINDINGS, types::VarDctKernelParams};
use crate::{EncodeError, UnsupportedFeature};

/// Allocations added by one XYB ICC source conversion. Included in the job's total reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctIccMemoryPlan {
    /// Padded normalized device planes, one for Gray or three for RGB.
    pub input_bytes: u64,
    /// Padded linear Gray/BT.709 planes, consumed directly by the forward-transform loader.
    pub linear_bytes: u64,
    pub program_bytes: u64,
    /// Original source-layout parameters plus the resident ICC dispatch record.
    pub parameter_bytes: u64,
    pub total_bytes: u64,
}

pub(super) struct Pipeline {
    transform: Arc<IccTransform>,
    memory: ResidentIccMemoryPlan,
    normalize: wgpu::ComputePipeline,
    color: ResidentIccPipeline,
    workgroup: (u32, u32),
    max_workgroups: u32,
    channels: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    original: VarDctKernelParams,
    extent: Extent2d,
    pub(super) memory: VarDctIccMemoryPlan,
}

pub(super) struct Scratch {
    pub(super) linear: wgpu::Buffer,
    pub(super) parameters: wgpu::Buffer,
    _input: wgpu::Buffer,
    _program: ResidentIccProgram,
    _dispatch: ResidentIccDispatch,
}

impl Pipeline {
    pub(super) fn new(
        device: &wgpu::Device,
        transform: Arc<IccTransform>,
        variant: KernelVariant,
    ) -> Result<Self, EncodeError> {
        let memory = ResidentIccMemoryPlan::new(&transform, &device.limits())?;
        let channels = transform.program().input_channels();
        if !matches!(channels, 1 | 3) || transform.program().output_channels() != channels {
            return Err(EncodeError::InvalidConfiguration(
                "ICC input and working channel plans disagree",
            ));
        }
        // A relative working connection has no black compensation or GPU-derived metadata.
        // Keep that invariant explicit if the shared program lowering changes in the future.
        if memory.validation_bytes != 0 {
            return Err(EncodeError::InvalidConfiguration(
                "relative ICC input unexpectedly requires a dynamic connection",
            ));
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VarDCT ICC source normalization"),
            source: wgpu::ShaderSource::Wgsl(
                super::dispatch::shader_source(include_str!("icc_input.wgsl")).into(),
            ),
        });
        let normalize = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VarDCT ICC device samples"),
            layout: None,
            module: &module,
            entry_point: Some("normalize_icc_input"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    ("wg_x", f64::from(variant.workgroup_size().0)),
                    ("icc_channels", channels as f64),
                ],
                ..Default::default()
            },
            cache: None,
        });
        Ok(Self {
            transform,
            memory,
            normalize,
            color: ResidentIccPipeline::with_variant(device, variant)?,
            workgroup: variant.workgroup_size(),
            max_workgroups: device.limits().max_compute_workgroups_per_dimension,
            channels,
        })
    }

    pub(super) fn plan(
        &self,
        params: &mut VarDctKernelParams,
        limit: u64,
    ) -> Result<Plan, EncodeError> {
        let extent = Extent2d::new(params.blocks_x * 8, params.blocks_y * 8);
        let required = [
            params.workgroups_x,
            params
                .source_validation_groups
                .div_ceil(params.workgroups_x),
            extent.width.div_ceil(self.workgroup.0),
            extent.height.div_ceil(self.workgroup.1),
        ]
        .into_iter()
        .max()
        .unwrap();
        if required > self.max_workgroups {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "ICC input workgroups",
                required: u64::from(required),
                available: u64::from(self.max_workgroups),
            }
            .into());
        }
        let plane_bytes = u64::from(extent.width) * u64::from(extent.height) * 4;
        let image_bytes = plane_bytes * self.channels as u64;
        if image_bytes > limit.min(u64::from(u32::MAX)) {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "ICC input storage bytes",
                required: image_bytes,
                available: limit.min(u64::from(u32::MAX)),
            }
            .into());
        }
        let parameter_bytes =
            std::mem::size_of::<VarDctKernelParams>() as u64 + self.memory.dispatch_bytes;
        let memory = VarDctIccMemoryPlan {
            input_bytes: image_bytes,
            linear_bytes: image_bytes,
            program_bytes: self.memory.program_bytes,
            parameter_bytes,
            total_bytes: image_bytes * 2 + self.memory.program_bytes + parameter_bytes,
        };
        let mut original = *params;
        original.color_normalization = 1; // validation names nonfinite original device samples
        params.source_sample_mask = u32::MAX;
        params.source_exponent_bits = 8;
        params.source_big_endian = 0; // GPU storage F32 words, independent of host/source byte order
        params.sources = std::array::from_fn(|channel| crate::source::SourceParams {
            row_stride: extent.width * 4,
            byte_offset: (channel % self.channels) as u32 * plane_bytes as u32,
            pixel_stride: 4,
            word_bytes: 4,
            bit_shift: 0,
            plane: 0,
        });
        Ok(Plan {
            original,
            extent,
            memory,
        })
    }

    pub(super) fn encode(
        &self,
        device: &wgpu::Device,
        commands: &mut wgpu::CommandEncoder,
        plan: Plan,
        sources: [wgpu::BindGroupEntry<'_>; 4],
        artifact: &wgpu::Buffer,
    ) -> Result<Scratch, EncodeError> {
        let storage = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let input = storage("ICC normalized device input", plan.memory.input_bytes);
        let linear = storage("ICC linear input for VarDCT", plan.memory.linear_bytes);
        let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ICC original source layout"),
            contents: bytemuck::bytes_of(&plan.original),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ICC input normalization"),
            layout: &self.normalize.get_bind_group_layout(0),
            entries: &[
                sources[0].clone(),
                sources[1].clone(),
                sources[2].clone(),
                sources[3].clone(),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parameters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: artifact.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ICC source validation and normalization"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.normalize);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(
                plan.original.workgroups_x,
                plan.original
                    .source_validation_groups
                    .div_ceil(plan.original.workgroups_x),
                1,
            );
        }
        let program = ResidentIccProgram::new(device, &self.transform)?;
        let planes: [_; 3] = std::array::from_fn(|channel| ResidentIccPlane {
            offset: channel as u32 * plan.extent.width * plan.extent.height,
            stride: plan.extent.width,
        });
        let dispatch = self.color.encode(
            device,
            commands,
            &program,
            ResidentIccInputs {
                input: ResidentStorageBinding {
                    buffer: &input,
                    offset: 0,
                    size: NonZeroU64::new(plan.memory.input_bytes).unwrap(),
                },
                output: ResidentStorageBinding {
                    buffer: &linear,
                    offset: 0,
                    size: NonZeroU64::new(plan.memory.linear_bytes).unwrap(),
                },
                extent: plan.extent,
                input_planes: &planes[..program.input_channels()],
                output_planes: &planes[..program.output_channels()],
                input_encoding: ResidentIccSampleEncoding::Direct,
                output_encoding: ResidentIccSampleEncoding::Direct,
            },
        )?;
        Ok(Scratch {
            linear,
            parameters,
            _input: input,
            _program: program,
            _dispatch: dispatch,
        })
    }
}

impl Scratch {
    pub(super) fn source_entries(&self) -> [wgpu::BindGroupEntry<'_>; 4] {
        SOURCE_BINDINGS.map(|binding| wgpu::BindGroupEntry {
            binding,
            resource: self.linear.as_entire_binding(),
        })
    }
}
