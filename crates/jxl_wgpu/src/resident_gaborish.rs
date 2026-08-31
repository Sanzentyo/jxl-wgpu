//! Gaborish filtering directly between GPU-resident planar XYB buffers.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{KernelVariant, ResidentF32Plane};

/// The two serialized, unnormalized Gaborish neighbor weights for each XYB channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidentGaborishWeights {
    pub x: [f32; 2],
    pub y: [f32; 2],
    pub b: [f32; 2],
}

impl ResidentGaborishWeights {
    /// Standard JPEG XL Gaborish weights.
    pub const DEFAULT: Self = Self {
        x: [0.115_169_525, 0.061_248_592],
        y: [0.115_169_525, 0.061_248_592],
        b: [0.115_169_525, 0.061_248_592],
    };

    fn normalized(self) -> Result<[[f32; 4]; 3], ResidentGaborishError> {
        Ok([
            normalize_weights(0, self.x)?,
            normalize_weights(1, self.y)?,
            normalize_weights(2, self.b)?,
        ])
    }
}

impl Default for ResidentGaborishWeights {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Six non-aliasing F32 planes consumed and produced by one fused XYB dispatch.
#[derive(Clone, Copy, Debug)]
pub struct ResidentGaborishInputs<'a> {
    pub inputs: [ResidentF32Plane<'a>; 3],
    pub outputs: [ResidentF32Plane<'a>; 3],
    pub weights: ResidentGaborishWeights,
}

/// Exact explicit allocation made while recording one resident Gaborish dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentGaborishMemoryPlan {
    pub uniform_bytes: u64,
}

impl ResidentGaborishMemoryPlan {
    pub const UNIFORM_BYTES: u64 = std::mem::size_of::<ResidentGaborishParams>() as u64;

    #[must_use]
    pub const fn new() -> Self {
        Self {
            uniform_bytes: Self::UNIFORM_BYTES,
        }
    }
}

impl Default for ResidentGaborishMemoryPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResidentGaborishError {
    #[error("resident Gaborish requires a tiled workgroup, got {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    #[error("resident Gaborish workgroup variant {variant:?} exceeds device limits")]
    WorkgroupVariant { variant: KernelVariant },
    #[error("resident Gaborish plane {plane} has invalid {width}x{height} stride {stride}")]
    PlaneGeometry {
        plane: usize,
        width: u32,
        height: u32,
        stride: u32,
    },
    #[error("resident Gaborish plane {plane} geometry differs from plane zero")]
    PlaneExtent { plane: usize },
    #[error("resident Gaborish plane {plane} buffer is missing STORAGE usage")]
    MissingStorageUsage { plane: usize },
    #[error("resident Gaborish plane {plane} offset {offset} is not aligned to {alignment}")]
    BindingAlignment {
        plane: usize,
        offset: u64,
        alignment: u64,
    },
    #[error(
        "resident Gaborish plane {plane} range {offset}..{end} exceeds buffer size {available}"
    )]
    BindingRange {
        plane: usize,
        offset: u64,
        end: u64,
        available: u64,
    },
    #[error("resident Gaborish plane {plane} needs {required} bytes, binding has {available}")]
    BindingSize {
        plane: usize,
        required: u64,
        available: u64,
    },
    #[error(
        "resident Gaborish plane {plane} binding needs {required} bytes, device permits {available}"
    )]
    StorageBindingLimit {
        plane: usize,
        required: u64,
        available: u64,
    },
    #[error("resident Gaborish channel {channel} has invalid serialized weights")]
    InvalidWeights { channel: usize },
    #[error("resident Gaborish dispatch {axis} count {required} exceeds device limit {available}")]
    WorkgroupCount {
        axis: &'static str,
        required: u32,
        available: u32,
    },
    #[error("resident Gaborish arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
}

/// Reusable fused XYB Gaborish pipeline for already-resident planes.
pub struct ResidentGaborishPipeline {
    pipeline: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl ResidentGaborishPipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, ResidentGaborishError> {
        Self::with_variant(device, KernelVariant::Tile16x16)
    }

    pub fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, ResidentGaborishError> {
        if variant.is_linear() {
            return Err(ResidentGaborishError::WorkgroupShape { variant });
        }
        variant
            .validate_for("resident_gaborish", &device.limits(), 0)
            .map_err(|_| ResidentGaborishError::WorkgroupVariant { variant })?;
        let module =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/gaborish_rgb.wgsl"));
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        let constants = [
            ("wg_x", f64::from(workgroup_x)),
            ("wg_y", f64::from(workgroup_y)),
        ];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu resident Gaborish RGB"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
        Ok(Self { pipeline, variant })
    }

    /// Validates all plane ranges, records one fused dispatch, and returns its uniform buffer.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: ResidentGaborishInputs<'_>,
    ) -> Result<wgpu::Buffer, ResidentGaborishError> {
        let params = validate_inputs(device, inputs)?;
        let (workgroup_x, workgroup_y) = self.variant.workgroup_size();
        let dispatch_x = params.width.div_ceil(workgroup_x);
        let dispatch_y = params.height.div_ceil(workgroup_y);
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        for (axis, required) in [("x", dispatch_x), ("y", dispatch_y)] {
            if required > maximum {
                return Err(ResidentGaborishError::WorkgroupCount {
                    axis,
                    required,
                    available: maximum,
                });
            }
        }
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu resident Gaborish params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu resident Gaborish bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                storage_entry(0, inputs.inputs[0]),
                storage_entry(1, inputs.inputs[1]),
                storage_entry(2, inputs.inputs[2]),
                storage_entry(3, inputs.outputs[0]),
                storage_entry(4, inputs.outputs[1]),
                storage_entry(5, inputs.outputs[2]),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu resident Gaborish RGB"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        drop(pass);
        Ok(uniform)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ResidentGaborishParams {
    width: u32,
    height: u32,
    input_stride_x: u32,
    input_stride_y: u32,
    input_stride_b: u32,
    output_stride_x: u32,
    output_stride_y: u32,
    output_stride_b: u32,
    weights_x: [f32; 4],
    weights_y: [f32; 4],
    weights_b: [f32; 4],
}

fn normalize_weights(channel: usize, weights: [f32; 2]) -> Result<[f32; 4], ResidentGaborishError> {
    let denominator = 1.0 + 4.0 * (weights[0] + weights[1]);
    if weights.into_iter().any(|weight| !weight.is_finite())
        || !denominator.is_finite()
        || denominator.abs() < f32::EPSILON
    {
        return Err(ResidentGaborishError::InvalidWeights { channel });
    }
    Ok([
        denominator.recip(),
        weights[0] / denominator,
        weights[1] / denominator,
        0.0,
    ])
}

fn validate_inputs(
    device: &wgpu::Device,
    inputs: ResidentGaborishInputs<'_>,
) -> Result<ResidentGaborishParams, ResidentGaborishError> {
    let reference = inputs.inputs[0];
    for (plane, candidate) in inputs.inputs.into_iter().chain(inputs.outputs).enumerate() {
        validate_plane(device, plane, candidate)?;
        if candidate.width != reference.width || candidate.height != reference.height {
            return Err(ResidentGaborishError::PlaneExtent { plane });
        }
    }
    let weights = inputs.weights.normalized()?;
    Ok(ResidentGaborishParams {
        width: reference.width,
        height: reference.height,
        input_stride_x: inputs.inputs[0].effective_stride(),
        input_stride_y: inputs.inputs[1].effective_stride(),
        input_stride_b: inputs.inputs[2].effective_stride(),
        output_stride_x: inputs.outputs[0].effective_stride(),
        output_stride_y: inputs.outputs[1].effective_stride(),
        output_stride_b: inputs.outputs[2].effective_stride(),
        weights_x: weights[0],
        weights_y: weights[1],
        weights_b: weights[2],
    })
}

fn validate_plane(
    device: &wgpu::Device,
    plane: usize,
    candidate: ResidentF32Plane<'_>,
) -> Result<(), ResidentGaborishError> {
    let stride = candidate.effective_stride();
    if candidate.width == 0 || candidate.height == 0 || stride < candidate.width {
        return Err(ResidentGaborishError::PlaneGeometry {
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
        return Err(ResidentGaborishError::MissingStorageUsage { plane });
    }
    let alignment = u64::from(device.limits().min_storage_buffer_offset_alignment).max(4);
    if !candidate.storage.offset.is_multiple_of(alignment) {
        return Err(ResidentGaborishError::BindingAlignment {
            plane,
            offset: candidate.storage.offset,
            alignment,
        });
    }
    let end = candidate
        .storage
        .offset
        .checked_add(candidate.storage.size.get())
        .ok_or(ResidentGaborishError::ArithmeticOverflow {
            field: "storage binding range",
        })?;
    if end > candidate.storage.buffer.size() {
        return Err(ResidentGaborishError::BindingRange {
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
        .ok_or(ResidentGaborishError::ArithmeticOverflow {
            field: "plane byte size",
        })?;
    if required > candidate.storage.size.get() {
        return Err(ResidentGaborishError::BindingSize {
            plane,
            required,
            available: candidate.storage.size.get(),
        });
    }
    let maximum = device.limits().max_storage_buffer_binding_size;
    if candidate.storage.size.get() > maximum {
        return Err(ResidentGaborishError::StorageBindingLimit {
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
    assert!(std::mem::size_of::<ResidentGaborishParams>() == 80);
    assert!(std::mem::align_of::<ResidentGaborishParams>() == 16);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_weights_match_the_normative_normalized_kernel() {
        let weights = ResidentGaborishWeights::DEFAULT.normalized().unwrap();
        for channel in weights {
            assert!((channel[0] + 4.0 * channel[1] + 4.0 * channel[2] - 1.0).abs() < 1.0e-7);
            assert_eq!(channel[3], 0.0);
        }
    }

    #[test]
    fn invalid_weights_report_the_exact_channel() {
        let error = ResidentGaborishWeights {
            x: ResidentGaborishWeights::DEFAULT.x,
            y: [f32::INFINITY, 0.0],
            b: ResidentGaborishWeights::DEFAULT.b,
        }
        .normalized()
        .unwrap_err();
        assert_eq!(error, ResidentGaborishError::InvalidWeights { channel: 1 });

        let error = normalize_weights(2, [-0.25, 0.0]).unwrap_err();
        assert_eq!(error, ResidentGaborishError::InvalidWeights { channel: 2 });
    }

    #[test]
    fn shader_and_uniform_abi_are_portable() {
        let module =
            naga::front::wgsl::parse_str(include_str!("../shaders/gaborish_rgb.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(ResidentGaborishMemoryPlan::new().uniform_bytes, 80);
    }
}
