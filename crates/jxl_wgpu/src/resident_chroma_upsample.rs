//! JPEG XL quarter/three-quarter chroma interpolation between GPU-resident F32 planes.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{KernelVariant, ResidentF32Plane};

/// Axes on which one resident component is sampled at half resolution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentChromaShift {
    /// Expand one input sample into two output columns.
    pub horizontal: bool,
    /// Expand one input sample into two output rows.
    pub vertical: bool,
}

impl ResidentChromaShift {
    #[must_use]
    pub const fn is_subsampled(self) -> bool {
        self.horizontal || self.vertical
    }
}

/// One non-aliasing resident source/destination pair and its sampling contract.
#[derive(Clone, Copy, Debug)]
pub struct ResidentChromaUpsampleInputs<'a> {
    /// Compact component plane before interpolation.
    pub input: ResidentF32Plane<'a>,
    /// Full-resolution, non-aliasing destination plane.
    pub output: ResidentF32Plane<'a>,
    /// Axes encoded at half resolution.
    pub shift: ResidentChromaShift,
}

/// Exact explicit allocation retained for one recorded interpolation dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentChromaUpsampleMemoryPlan {
    /// Uniform retained until the recorded dispatch completes.
    pub uniform_bytes: u64,
}

impl ResidentChromaUpsampleMemoryPlan {
    pub const UNIFORM_BYTES: u64 = std::mem::size_of::<ResidentChromaUpsampleParams>() as u64;

    #[must_use]
    pub const fn new() -> Self {
        Self {
            uniform_bytes: Self::UNIFORM_BYTES,
        }
    }
}

impl Default for ResidentChromaUpsampleMemoryPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResidentChromaUpsampleError {
    #[error("resident chroma upsampling requires a tiled workgroup, got {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    #[error("resident chroma upsampling workgroup variant {variant:?} exceeds device limits")]
    WorkgroupVariant { variant: KernelVariant },
    #[error("resident chroma upsampling requires at least one shifted axis")]
    NoShiftedAxis,
    #[error(
        "resident chroma upsampling geometry {input_width}x{input_height} -> {output_width}x{output_height} does not match shift {horizontal}x{vertical}"
    )]
    Extent {
        input_width: u32,
        input_height: u32,
        output_width: u32,
        output_height: u32,
        horizontal: bool,
        vertical: bool,
    },
    #[error("resident chroma {role} plane has invalid {width}x{height} stride {stride}")]
    PlaneGeometry {
        role: &'static str,
        width: u32,
        height: u32,
        stride: u32,
    },
    #[error("resident chroma {role} plane is missing STORAGE usage")]
    MissingStorageUsage { role: &'static str },
    #[error("resident chroma {role} plane offset {offset} is not aligned to {alignment}")]
    BindingAlignment {
        role: &'static str,
        offset: u64,
        alignment: u64,
    },
    #[error("resident chroma {role} range {offset}..{end} exceeds buffer size {available}")]
    BindingRange {
        role: &'static str,
        offset: u64,
        end: u64,
        available: u64,
    },
    #[error("resident chroma {role} plane needs {required} bytes, binding has {available}")]
    BindingSize {
        role: &'static str,
        required: u64,
        available: u64,
    },
    #[error("resident chroma {role} binding needs {required} bytes, device permits {available}")]
    StorageBindingLimit {
        role: &'static str,
        required: u64,
        available: u64,
    },
    #[error("resident chroma dispatch {axis} count {required} exceeds device limit {available}")]
    WorkgroupCount {
        axis: &'static str,
        required: u32,
        available: u32,
    },
    #[error("resident chroma arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
}

/// Reusable resident interpolation pipelines. Two-axis sampling uses the scheduler's fused kernel.
pub struct ResidentChromaUpsamplePipeline {
    axis_pipeline: wgpu::ComputePipeline,
    two_dimensional_pipeline: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl ResidentChromaUpsamplePipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, ResidentChromaUpsampleError> {
        Self::with_variant(device, KernelVariant::Tile16x16)
    }

    pub fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, ResidentChromaUpsampleError> {
        if variant.is_linear() {
            return Err(ResidentChromaUpsampleError::WorkgroupShape { variant });
        }
        variant
            .validate_for("resident_chroma_upsample", &device.limits(), 0)
            .map_err(|_| ResidentChromaUpsampleError::WorkgroupVariant { variant })?;
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        let constants = [
            ("wg_x", f64::from(workgroup_x)),
            ("wg_y", f64::from(workgroup_y)),
        ];
        let create = |label, module: wgpu::ShaderModule| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        };
        let axis_pipeline = create(
            "jxl-wgpu resident chroma axis upsample",
            device.create_shader_module(wgpu::include_wgsl!("../shaders/chroma_upsample.wgsl")),
        );
        let two_dimensional_pipeline = create(
            "jxl-wgpu resident chroma 2D upsample",
            device.create_shader_module(wgpu::include_wgsl!("../shaders/chroma_2d.wgsl")),
        );
        Ok(Self {
            axis_pipeline,
            two_dimensional_pipeline,
            variant,
        })
    }

    /// Validates both ranges and records one interpolation dispatch.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: ResidentChromaUpsampleInputs<'_>,
    ) -> Result<wgpu::Buffer, ResidentChromaUpsampleError> {
        let params = validate_inputs(device, inputs)?;
        let pipeline = if inputs.shift.horizontal && inputs.shift.vertical {
            &self.two_dimensional_pipeline
        } else {
            &self.axis_pipeline
        };
        let (workgroup_x, workgroup_y) = self.variant.workgroup_size();
        let dispatch_x = inputs.output.width.div_ceil(workgroup_x);
        let dispatch_y = inputs.output.height.div_ceil(workgroup_y);
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        for (axis, required) in [("x", dispatch_x), ("y", dispatch_y)] {
            if required > maximum {
                return Err(ResidentChromaUpsampleError::WorkgroupCount {
                    axis,
                    required,
                    available: maximum,
                });
            }
        }
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu resident chroma upsample params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu resident chroma upsample bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                storage_entry(0, inputs.input),
                storage_entry(1, inputs.output),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu resident chroma upsample"),
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
struct ResidentChromaUpsampleParams {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    input_stride: u32,
    output_stride: u32,
    axis: u32,
    _padding: u32,
}

fn validate_inputs(
    device: &wgpu::Device,
    inputs: ResidentChromaUpsampleInputs<'_>,
) -> Result<ResidentChromaUpsampleParams, ResidentChromaUpsampleError> {
    if !inputs.shift.is_subsampled() {
        return Err(ResidentChromaUpsampleError::NoShiftedAxis);
    }
    validate_plane(device, "input", inputs.input)?;
    validate_plane(device, "output", inputs.output)?;
    let width_matches = if inputs.shift.horizontal {
        inputs.output.width.div_ceil(2) == inputs.input.width
    } else {
        inputs.output.width == inputs.input.width
    };
    let height_matches = if inputs.shift.vertical {
        inputs.output.height.div_ceil(2) == inputs.input.height
    } else {
        inputs.output.height == inputs.input.height
    };
    if !width_matches || !height_matches {
        return Err(ResidentChromaUpsampleError::Extent {
            input_width: inputs.input.width,
            input_height: inputs.input.height,
            output_width: inputs.output.width,
            output_height: inputs.output.height,
            horizontal: inputs.shift.horizontal,
            vertical: inputs.shift.vertical,
        });
    }
    Ok(ResidentChromaUpsampleParams {
        input_width: inputs.input.width,
        input_height: inputs.input.height,
        output_width: inputs.output.width,
        output_height: inputs.output.height,
        input_stride: inputs.input.effective_stride(),
        output_stride: inputs.output.effective_stride(),
        axis: u32::from(inputs.shift.vertical),
        _padding: 0,
    })
}

fn validate_plane(
    device: &wgpu::Device,
    role: &'static str,
    plane: ResidentF32Plane<'_>,
) -> Result<(), ResidentChromaUpsampleError> {
    let stride = plane.effective_stride();
    if plane.width == 0 || plane.height == 0 || stride < plane.width {
        return Err(ResidentChromaUpsampleError::PlaneGeometry {
            role,
            width: plane.width,
            height: plane.height,
            stride,
        });
    }
    if !plane
        .storage
        .buffer
        .usage()
        .contains(wgpu::BufferUsages::STORAGE)
    {
        return Err(ResidentChromaUpsampleError::MissingStorageUsage { role });
    }
    let alignment = u64::from(device.limits().min_storage_buffer_offset_alignment).max(4);
    if !plane.storage.offset.is_multiple_of(alignment) {
        return Err(ResidentChromaUpsampleError::BindingAlignment {
            role,
            offset: plane.storage.offset,
            alignment,
        });
    }
    let end = plane
        .storage
        .offset
        .checked_add(plane.storage.size.get())
        .ok_or(ResidentChromaUpsampleError::ArithmeticOverflow {
            field: "storage binding range",
        })?;
    if end > plane.storage.buffer.size() {
        return Err(ResidentChromaUpsampleError::BindingRange {
            role,
            offset: plane.storage.offset,
            end,
            available: plane.storage.buffer.size(),
        });
    }
    let required = u64::from(plane.height - 1)
        .checked_mul(u64::from(stride))
        .and_then(|value| value.checked_add(u64::from(plane.width)))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or(ResidentChromaUpsampleError::ArithmeticOverflow {
            field: "plane byte size",
        })?;
    if required > plane.storage.size.get() {
        return Err(ResidentChromaUpsampleError::BindingSize {
            role,
            required,
            available: plane.storage.size.get(),
        });
    }
    let maximum = device.limits().max_storage_buffer_binding_size;
    if plane.storage.size.get() > maximum {
        return Err(ResidentChromaUpsampleError::StorageBindingLimit {
            role,
            required: plane.storage.size.get(),
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
    assert!(std::mem::size_of::<ResidentChromaUpsampleParams>() == 32);
    assert!(std::mem::align_of::<ResidentChromaUpsampleParams>() == 16);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_and_uniform_abis_are_portable() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<ResidentChromaUpsampleParams>();
        assert_eq!(ResidentChromaUpsampleMemoryPlan::new().uniform_bytes, 32);
        for shader in [
            include_str!("../shaders/chroma_upsample.wgsl"),
            include_str!("../shaders/chroma_2d.wgsl"),
        ] {
            let module = naga::front::wgsl::parse_str(shader).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}
