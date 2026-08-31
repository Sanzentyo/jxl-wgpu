//! EPF filtering directly between GPU-resident planar XYB buffers.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::EpfPass;
use wgpu::util::DeviceExt;

use crate::{KernelVariant, ResidentF32Plane};

/// Normative parameters for one EPF pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidentEpfParameters {
    pub pass: EpfPass,
    pub sigma_scale: f32,
    pub border_sad_mul: f32,
    pub channel_scale: [f32; 3],
}

/// Non-aliasing resident XYB inputs/outputs and a per-8x8-block inverse-sigma plane.
#[derive(Clone, Copy, Debug)]
pub struct ResidentEpfInputs<'a> {
    pub inputs: [ResidentF32Plane<'a>; 3],
    pub outputs: [ResidentF32Plane<'a>; 3],
    pub sigma: ResidentF32Plane<'a>,
    pub parameters: ResidentEpfParameters,
}

/// Exact explicit allocation made while recording one resident EPF dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentEpfMemoryPlan {
    pub uniform_bytes: u64,
}

impl ResidentEpfMemoryPlan {
    pub const UNIFORM_BYTES: u64 = std::mem::size_of::<ResidentEpfUniform>() as u64;

    #[must_use]
    pub const fn new() -> Self {
        Self {
            uniform_bytes: Self::UNIFORM_BYTES,
        }
    }
}

impl Default for ResidentEpfMemoryPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResidentEpfError {
    #[error("resident EPF requires a tiled workgroup, got {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    #[error("resident EPF workgroup variant {variant:?} exceeds device limits")]
    WorkgroupVariant { variant: KernelVariant },
    #[error("resident EPF plane {plane} has invalid {width}x{height} stride {stride}")]
    PlaneGeometry {
        plane: usize,
        width: u32,
        height: u32,
        stride: u32,
    },
    #[error("resident EPF image plane {plane} geometry differs from plane zero")]
    PlaneExtent { plane: usize },
    #[error(
        "resident EPF sigma plane {width}x{height} does not cover the required {required_width}x{required_height} block grid"
    )]
    SigmaExtent {
        width: u32,
        height: u32,
        required_width: u32,
        required_height: u32,
    },
    #[error("resident EPF plane {plane} buffer is missing STORAGE usage")]
    MissingStorageUsage { plane: usize },
    #[error("resident EPF plane {plane} offset {offset} is not aligned to {alignment}")]
    BindingAlignment {
        plane: usize,
        offset: u64,
        alignment: u64,
    },
    #[error("resident EPF plane {plane} range {offset}..{end} exceeds buffer size {available}")]
    BindingRange {
        plane: usize,
        offset: u64,
        end: u64,
        available: u64,
    },
    #[error("resident EPF plane {plane} needs {required} bytes, binding has {available}")]
    BindingSize {
        plane: usize,
        required: u64,
        available: u64,
    },
    #[error(
        "resident EPF plane {plane} binding needs {required} bytes, device permits {available}"
    )]
    StorageBindingLimit {
        plane: usize,
        required: u64,
        available: u64,
    },
    #[error("resident EPF parameters contain a non-finite value")]
    NonFiniteParameters,
    #[error("resident EPF dispatch {axis} count {required} exceeds device limit {available}")]
    WorkgroupCount {
        axis: &'static str,
        required: u32,
        available: u32,
    },
    #[error("resident EPF arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
}

/// Reusable three-pass EPF pipeline for already-resident XYB planes.
pub struct ResidentEpfPipeline {
    pass0: wgpu::ComputePipeline,
    pass1: wgpu::ComputePipeline,
    pass2: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl ResidentEpfPipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, ResidentEpfError> {
        Self::with_variant(device, KernelVariant::Tile16x16)
    }

    pub fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, ResidentEpfError> {
        if variant.is_linear() {
            return Err(ResidentEpfError::WorkgroupShape { variant });
        }
        variant
            .validate_for("resident_epf", &device.limits(), 0)
            .map_err(|_| ResidentEpfError::WorkgroupVariant { variant })?;
        let module = device.create_shader_module(wgpu::include_wgsl!("../shaders/epf.wgsl"));
        let pipeline = |label, entry_point| {
            let (workgroup_x, workgroup_y) = variant.workgroup_size();
            let constants = [
                ("wg_x", f64::from(workgroup_x)),
                ("wg_y", f64::from(workgroup_y)),
            ];
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        };
        Ok(Self {
            pass0: pipeline("jxl-wgpu resident EPF pass 0", "epf0"),
            pass1: pipeline("jxl-wgpu resident EPF pass 1", "epf1"),
            pass2: pipeline("jxl-wgpu resident EPF pass 2", "epf2"),
            variant,
        })
    }

    /// Validates every binding, records one pass, and returns its uniform buffer.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: ResidentEpfInputs<'_>,
    ) -> Result<wgpu::Buffer, ResidentEpfError> {
        let uniform_data = validate_inputs(device, inputs)?;
        let (workgroup_x, workgroup_y) = self.variant.workgroup_size();
        let dispatch_x = uniform_data.width.div_ceil(workgroup_x);
        let dispatch_y = uniform_data.height.div_ceil(workgroup_y);
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        for (axis, required) in [("x", dispatch_x), ("y", dispatch_y)] {
            if required > maximum {
                return Err(ResidentEpfError::WorkgroupCount {
                    axis,
                    required,
                    available: maximum,
                });
            }
        }
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu resident EPF params"),
            contents: bytemuck::bytes_of(&uniform_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let pipeline = match inputs.parameters.pass {
            EpfPass::Pass0 => &self.pass0,
            EpfPass::Pass1 => &self.pass1,
            EpfPass::Pass2 => &self.pass2,
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu resident EPF bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                storage_entry(0, inputs.inputs[0]),
                storage_entry(1, inputs.inputs[1]),
                storage_entry(2, inputs.inputs[2]),
                storage_entry(3, inputs.sigma),
                storage_entry(4, inputs.outputs[0]),
                storage_entry(5, inputs.outputs[1]),
                storage_entry(6, inputs.outputs[2]),
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu resident EPF"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        drop(pass);
        Ok(uniform)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ResidentEpfUniform {
    width: u32,
    height: u32,
    input_stride_x: u32,
    input_stride_y: u32,
    input_stride_b: u32,
    output_stride_x: u32,
    output_stride_y: u32,
    output_stride_b: u32,
    sigma_width: u32,
    sigma_height: u32,
    sigma_stride: u32,
    sigma_is_plane: u32,
    sigma_scale: f32,
    border_sad_mul: f32,
    channel_scale_x: f32,
    channel_scale_y: f32,
    channel_scale_b: f32,
    min_sigma: f32,
    _padding: [u32; 2],
}

fn validate_inputs(
    device: &wgpu::Device,
    inputs: ResidentEpfInputs<'_>,
) -> Result<ResidentEpfUniform, ResidentEpfError> {
    let reference = inputs.inputs[0];
    for (plane, candidate) in inputs.inputs.into_iter().chain(inputs.outputs).enumerate() {
        validate_plane(device, plane, candidate)?;
        if candidate.width != reference.width || candidate.height != reference.height {
            return Err(ResidentEpfError::PlaneExtent { plane });
        }
    }
    validate_plane(device, 6, inputs.sigma)?;
    let required_width = reference.width.div_ceil(8);
    let required_height = reference.height.div_ceil(8);
    if inputs.sigma.width < required_width || inputs.sigma.height < required_height {
        return Err(ResidentEpfError::SigmaExtent {
            width: inputs.sigma.width,
            height: inputs.sigma.height,
            required_width,
            required_height,
        });
    }
    let parameters = inputs.parameters;
    if [parameters.sigma_scale, parameters.border_sad_mul]
        .into_iter()
        .chain(parameters.channel_scale)
        .any(|value| !value.is_finite())
    {
        return Err(ResidentEpfError::NonFiniteParameters);
    }
    Ok(ResidentEpfUniform {
        width: reference.width,
        height: reference.height,
        input_stride_x: inputs.inputs[0].effective_stride(),
        input_stride_y: inputs.inputs[1].effective_stride(),
        input_stride_b: inputs.inputs[2].effective_stride(),
        output_stride_x: inputs.outputs[0].effective_stride(),
        output_stride_y: inputs.outputs[1].effective_stride(),
        output_stride_b: inputs.outputs[2].effective_stride(),
        sigma_width: inputs.sigma.width,
        sigma_height: inputs.sigma.height,
        sigma_stride: inputs.sigma.effective_stride(),
        sigma_is_plane: 1,
        sigma_scale: parameters.sigma_scale,
        border_sad_mul: parameters.border_sad_mul,
        channel_scale_x: parameters.channel_scale[0],
        channel_scale_y: parameters.channel_scale[1],
        channel_scale_b: parameters.channel_scale[2],
        min_sigma: -3.905_243,
        _padding: [0; 2],
    })
}

fn validate_plane(
    device: &wgpu::Device,
    plane: usize,
    candidate: ResidentF32Plane<'_>,
) -> Result<(), ResidentEpfError> {
    let stride = candidate.effective_stride();
    if candidate.width == 0 || candidate.height == 0 || stride < candidate.width {
        return Err(ResidentEpfError::PlaneGeometry {
            plane,
            width: candidate.width,
            height: candidate.height,
            stride,
        });
    }
    if !candidate
        .storage
        .buffer
        .usage()
        .contains(wgpu::BufferUsages::STORAGE)
    {
        return Err(ResidentEpfError::MissingStorageUsage { plane });
    }
    let alignment = u64::from(device.limits().min_storage_buffer_offset_alignment).max(4);
    if !candidate.storage.offset.is_multiple_of(alignment) {
        return Err(ResidentEpfError::BindingAlignment {
            plane,
            offset: candidate.storage.offset,
            alignment,
        });
    }
    let end = candidate
        .storage
        .offset
        .checked_add(candidate.storage.size.get())
        .ok_or(ResidentEpfError::ArithmeticOverflow {
            field: "storage binding range",
        })?;
    if end > candidate.storage.buffer.size() {
        return Err(ResidentEpfError::BindingRange {
            plane,
            offset: candidate.storage.offset,
            end,
            available: candidate.storage.buffer.size(),
        });
    }
    let required = u64::from(candidate.height - 1)
        .checked_mul(u64::from(stride))
        .and_then(|value| value.checked_add(u64::from(candidate.width)))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or(ResidentEpfError::ArithmeticOverflow {
            field: "plane byte size",
        })?;
    if required > candidate.storage.size.get() {
        return Err(ResidentEpfError::BindingSize {
            plane,
            required,
            available: candidate.storage.size.get(),
        });
    }
    let maximum = device.limits().max_storage_buffer_binding_size;
    if candidate.storage.size.get() > maximum {
        return Err(ResidentEpfError::StorageBindingLimit {
            plane,
            required: candidate.storage.size.get(),
            available: maximum,
        });
    }
    Ok(())
}

fn storage_entry(binding: u32, plane: ResidentF32Plane<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: plane.storage.buffer,
            offset: plane.storage.offset,
            size: Some(plane.storage.size),
        }),
    }
}

const _: () = {
    assert!(std::mem::size_of::<ResidentEpfUniform>() == 80);
    assert!(std::mem::align_of::<ResidentEpfUniform>() == 16);
    assert!(std::mem::offset_of!(ResidentEpfUniform, width) == 0);
    assert!(std::mem::offset_of!(ResidentEpfUniform, input_stride_x) == 8);
    assert!(std::mem::offset_of!(ResidentEpfUniform, output_stride_x) == 20);
    assert!(std::mem::offset_of!(ResidentEpfUniform, sigma_width) == 32);
    assert!(std::mem::offset_of!(ResidentEpfUniform, sigma_scale) == 48);
    assert!(std::mem::offset_of!(ResidentEpfUniform, channel_scale_x) == 56);
    assert!(std::mem::offset_of!(ResidentEpfUniform, min_sigma) == 68);
    assert!(std::mem::offset_of!(ResidentEpfUniform, _padding) == 72);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_and_uniform_abi_are_portable() {
        let module = naga::front::wgsl::parse_str(include_str!("../shaders/epf.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(ResidentEpfMemoryPlan::new().uniform_bytes, 80);
    }
}
