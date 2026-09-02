//! JPEG XL 2x/4x/8x interpolation across three GPU-resident F32 planes.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{KernelVariant, ResidentF32Plane};

/// Expanded phase-major JPEG XL interpolation kernels shared by all three planes.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidentImageUpsampleWeights {
    factor: u32,
    phase_major: Arc<[f32]>,
}

impl ResidentImageUpsampleWeights {
    /// Expands the serialized symmetric triangle into `factor²` row-major 5x5 kernels.
    pub fn new(factor: u32, compact: &[f32]) -> Result<Self, ResidentImageUpsampleError> {
        let half = validated_factor(factor)? / 2;
        let triangle_side = usize::try_from(half * 5).map_err(|_| {
            ResidentImageUpsampleError::ArithmeticOverflow {
                field: "upsampling weight triangle side",
            }
        })?;
        let expected = triangle_side
            .checked_mul(triangle_side + 1)
            .and_then(|value| value.checked_div(2))
            .ok_or(ResidentImageUpsampleError::ArithmeticOverflow {
                field: "upsampling compact weight count",
            })?;
        if compact.len() != expected {
            return Err(ResidentImageUpsampleError::WeightCount {
                factor,
                actual: compact.len(),
                expected,
            });
        }
        if let Some(index) = compact.iter().position(|value| !value.is_finite()) {
            return Err(ResidentImageUpsampleError::NonFiniteWeight { index });
        }
        let phase_count = usize::try_from(factor)
            .ok()
            .and_then(|factor| factor.checked_mul(factor))
            .ok_or(ResidentImageUpsampleError::ArithmeticOverflow {
                field: "upsampling phase count",
            })?;
        let mut phase_major = vec![0.0; phase_count * 25];
        let half_usize = half as usize;
        let factor_usize = factor as usize;
        for phase_y in 0..half_usize {
            for phase_x in 0..half_usize {
                let destinations = [
                    (phase_y, phase_x, false, false),
                    (phase_y, factor_usize - 1 - phase_x, false, true),
                    (factor_usize - 1 - phase_y, phase_x, true, false),
                    (
                        factor_usize - 1 - phase_y,
                        factor_usize - 1 - phase_x,
                        true,
                        true,
                    ),
                ];
                for kernel_y in 0..5 {
                    for kernel_x in 0..5 {
                        let triangle_y = 5 * phase_y + kernel_y;
                        let triangle_x = 5 * phase_x + kernel_x;
                        let minimum = triangle_y.min(triangle_x);
                        let maximum = triangle_y.max(triangle_x);
                        let compact_index = triangle_side * minimum
                            - minimum * minimum.saturating_sub(1) / 2
                            + maximum
                            - minimum;
                        let weight = compact[compact_index];
                        for &(destination_y, destination_x, flip_y, flip_x) in &destinations {
                            let y = if flip_y { 4 - kernel_y } else { kernel_y };
                            let x = if flip_x { 4 - kernel_x } else { kernel_x };
                            let phase = destination_y * factor_usize + destination_x;
                            phase_major[phase * 25 + y * 5 + x] = weight;
                        }
                    }
                }
            }
        }
        Ok(Self {
            factor,
            phase_major: phase_major.into(),
        })
    }

    #[must_use]
    pub const fn factor(&self) -> u32 {
        self.factor
    }

    #[must_use]
    pub fn storage_bytes(&self) -> u64 {
        self.phase_major.len() as u64 * std::mem::size_of::<f32>() as u64
    }
}

/// Three same-extent input planes, distinct output planes, and their interpolation kernels.
#[derive(Clone, Copy, Debug)]
pub struct ResidentImageUpsampleInputs<'a> {
    pub inputs: [ResidentF32Plane<'a>; 3],
    pub outputs: [ResidentF32Plane<'a>; 3],
    pub weights: &'a ResidentImageUpsampleWeights,
}

/// Buffers created while recording one fused three-plane dispatch.
pub struct ResidentImageUpsampleResources {
    _weights: wgpu::Buffer,
    _uniform: wgpu::Buffer,
}

/// Exact retained allocation for one fused image interpolation dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentImageUpsampleMemoryPlan {
    pub weight_bytes: u64,
    pub uniform_bytes: u64,
    pub total_bytes: u64,
}

impl ResidentImageUpsampleMemoryPlan {
    pub const UNIFORM_BYTES: u64 = std::mem::size_of::<ResidentImageUpsampleParams>() as u64;

    pub fn new(weights: &ResidentImageUpsampleWeights) -> Result<Self, ResidentImageUpsampleError> {
        let weight_bytes = weights.storage_bytes();
        let total_bytes = weight_bytes.checked_add(Self::UNIFORM_BYTES).ok_or(
            ResidentImageUpsampleError::ArithmeticOverflow {
                field: "upsampling retained bytes",
            },
        )?;
        Ok(Self {
            weight_bytes,
            uniform_bytes: Self::UNIFORM_BYTES,
            total_bytes,
        })
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResidentImageUpsampleError {
    #[error("resident image upsampling factor {factor} is not 2, 4, or 8")]
    InvalidFactor { factor: u32 },
    #[error("{factor}x image upsampling has {actual} compact weights, expected {expected}")]
    WeightCount {
        factor: u32,
        actual: usize,
        expected: usize,
    },
    #[error("image upsampling compact weight {index} is not finite")]
    NonFiniteWeight { index: usize },
    #[error("resident image upsampling requires a tiled workgroup, got {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    #[error("resident image upsampling workgroup variant {variant:?} exceeds device limits")]
    WorkgroupVariant { variant: KernelVariant },
    #[error(
        "resident image upsampling requires seven storage bindings, device permits {available}"
    )]
    StorageBindingCount { available: u32 },
    #[error("resident image upsampling plane {plane} has invalid {width}x{height} stride {stride}")]
    PlaneGeometry {
        plane: usize,
        width: u32,
        height: u32,
        stride: u32,
    },
    #[error("resident image upsampling plane {plane} geometry differs from plane zero")]
    PlaneExtent { plane: usize },
    #[error(
        "resident image upsampling extent {input_width}x{input_height} -> {output_width}x{output_height} does not match factor {factor}"
    )]
    Extent {
        input_width: u32,
        input_height: u32,
        output_width: u32,
        output_height: u32,
        factor: u32,
    },
    #[error("resident image upsampling plane {plane} is missing STORAGE usage")]
    MissingStorageUsage { plane: usize },
    #[error(
        "resident image upsampling plane {plane} offset {offset} is not aligned to {alignment}"
    )]
    BindingAlignment {
        plane: usize,
        offset: u64,
        alignment: u64,
    },
    #[error(
        "resident image upsampling plane {plane} range {offset}..{end} exceeds buffer size {available}"
    )]
    BindingRange {
        plane: usize,
        offset: u64,
        end: u64,
        available: u64,
    },
    #[error(
        "resident image upsampling plane {plane} needs {required} bytes, binding has {available}"
    )]
    BindingSize {
        plane: usize,
        required: u64,
        available: u64,
    },
    #[error(
        "resident image upsampling {role} binding needs {required} bytes, device permits {available}"
    )]
    StorageBindingLimit {
        role: &'static str,
        required: u64,
        available: u64,
    },
    #[error(
        "resident image upsampling dispatch {axis} count {required} exceeds device limit {available}"
    )]
    WorkgroupCount {
        axis: &'static str,
        required: u32,
        available: u32,
    },
    #[error("resident image upsampling arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
}

/// Reusable fused three-plane 2x/4x/8x image interpolation pipeline.
pub struct ResidentImageUpsamplePipeline {
    pipeline: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl ResidentImageUpsamplePipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, ResidentImageUpsampleError> {
        Self::with_variant(device, KernelVariant::Tile16x16)
    }

    pub fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, ResidentImageUpsampleError> {
        if variant.is_linear() {
            return Err(ResidentImageUpsampleError::WorkgroupShape { variant });
        }
        variant
            .validate_for("resident_image_upsample", &device.limits(), 0)
            .map_err(|_| ResidentImageUpsampleError::WorkgroupVariant { variant })?;
        if device.limits().max_storage_buffers_per_shader_stage < 7 {
            return Err(ResidentImageUpsampleError::StorageBindingCount {
                available: device.limits().max_storage_buffers_per_shader_stage,
            });
        }
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        let module =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/upsample_rgb.wgsl"));
        let constants = [
            ("wg_x", f64::from(workgroup_x)),
            ("wg_y", f64::from(workgroup_y)),
        ];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu resident image upsample"),
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

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: ResidentImageUpsampleInputs<'_>,
    ) -> Result<ResidentImageUpsampleResources, ResidentImageUpsampleError> {
        let params = validate_inputs(device, inputs)?;
        let plan = ResidentImageUpsampleMemoryPlan::new(inputs.weights)?;
        let maximum_binding = device.limits().max_storage_buffer_binding_size;
        if plan.weight_bytes > maximum_binding {
            return Err(ResidentImageUpsampleError::StorageBindingLimit {
                role: "weight",
                required: plan.weight_bytes,
                available: maximum_binding,
            });
        }
        let weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu resident image upsample weights"),
            contents: bytemuck::cast_slice(&inputs.weights.phase_major),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu resident image upsample params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu resident image upsample bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                storage_entry(0, inputs.inputs[0]),
                storage_entry(1, inputs.inputs[1]),
                storage_entry(2, inputs.inputs[2]),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: weights.as_entire_binding(),
                },
                storage_entry(4, inputs.outputs[0]),
                storage_entry(5, inputs.outputs[1]),
                storage_entry(6, inputs.outputs[2]),
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let (workgroup_x, workgroup_y) = self.variant.workgroup_size();
        let dispatch_x = inputs.outputs[0].width.div_ceil(workgroup_x);
        let dispatch_y = inputs.outputs[0].height.div_ceil(workgroup_y);
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        for (axis, required) in [("x", dispatch_x), ("y", dispatch_y)] {
            if required > maximum {
                return Err(ResidentImageUpsampleError::WorkgroupCount {
                    axis,
                    required,
                    available: maximum,
                });
            }
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu resident image upsample"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        drop(pass);
        Ok(ResidentImageUpsampleResources {
            _weights: weights,
            _uniform: uniform,
        })
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ResidentImageUpsampleParams {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    input_stride_x: u32,
    input_stride_y: u32,
    input_stride_b: u32,
    factor: u32,
    output_stride_x: u32,
    output_stride_y: u32,
    output_stride_b: u32,
    _padding: u32,
}

fn validated_factor(factor: u32) -> Result<u32, ResidentImageUpsampleError> {
    if matches!(factor, 2 | 4 | 8) {
        Ok(factor)
    } else {
        Err(ResidentImageUpsampleError::InvalidFactor { factor })
    }
}

fn validate_inputs(
    device: &wgpu::Device,
    inputs: ResidentImageUpsampleInputs<'_>,
) -> Result<ResidentImageUpsampleParams, ResidentImageUpsampleError> {
    let factor = validated_factor(inputs.weights.factor)?;
    let input = inputs.inputs[0];
    let output = inputs.outputs[0];
    for (plane, candidate) in inputs.inputs.into_iter().chain(inputs.outputs).enumerate() {
        validate_plane(device, plane, candidate)?;
        let reference = if plane < 3 { input } else { output };
        if candidate.width != reference.width || candidate.height != reference.height {
            return Err(ResidentImageUpsampleError::PlaneExtent { plane });
        }
    }
    if output.width.div_ceil(factor) != input.width
        || output.height.div_ceil(factor) != input.height
    {
        return Err(ResidentImageUpsampleError::Extent {
            input_width: input.width,
            input_height: input.height,
            output_width: output.width,
            output_height: output.height,
            factor,
        });
    }
    Ok(ResidentImageUpsampleParams {
        input_width: input.width,
        input_height: input.height,
        output_width: output.width,
        output_height: output.height,
        input_stride_x: inputs.inputs[0].effective_stride(),
        input_stride_y: inputs.inputs[1].effective_stride(),
        input_stride_b: inputs.inputs[2].effective_stride(),
        factor,
        output_stride_x: inputs.outputs[0].effective_stride(),
        output_stride_y: inputs.outputs[1].effective_stride(),
        output_stride_b: inputs.outputs[2].effective_stride(),
        _padding: 0,
    })
}

fn validate_plane(
    device: &wgpu::Device,
    plane: usize,
    candidate: ResidentF32Plane<'_>,
) -> Result<(), ResidentImageUpsampleError> {
    let stride = candidate.effective_stride();
    if candidate.width == 0 || candidate.height == 0 || stride < candidate.width {
        return Err(ResidentImageUpsampleError::PlaneGeometry {
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
        return Err(ResidentImageUpsampleError::MissingStorageUsage { plane });
    }
    let alignment = u64::from(device.limits().min_storage_buffer_offset_alignment).max(4);
    if !candidate.storage.offset.is_multiple_of(alignment) {
        return Err(ResidentImageUpsampleError::BindingAlignment {
            plane,
            offset: candidate.storage.offset,
            alignment,
        });
    }
    let end = candidate
        .storage
        .offset
        .checked_add(candidate.storage.size.get())
        .ok_or(ResidentImageUpsampleError::ArithmeticOverflow {
            field: "storage binding range",
        })?;
    if end > candidate.storage.buffer.size() {
        return Err(ResidentImageUpsampleError::BindingRange {
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
        .ok_or(ResidentImageUpsampleError::ArithmeticOverflow {
            field: "plane byte size",
        })?;
    if required > candidate.storage.size.get() {
        return Err(ResidentImageUpsampleError::BindingSize {
            plane,
            required,
            available: candidate.storage.size.get(),
        });
    }
    let maximum = device.limits().max_storage_buffer_binding_size;
    if candidate.storage.size.get() > maximum {
        return Err(ResidentImageUpsampleError::StorageBindingLimit {
            role: "plane",
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
    assert!(std::mem::size_of::<ResidentImageUpsampleParams>() == 48);
    assert!(std::mem::align_of::<ResidentImageUpsampleParams>() == 16);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_weight_shapes_expand_to_every_phase() {
        for (factor, compact_len) in [(2, 15), (4, 55), (8, 210)] {
            let compact = (0..compact_len)
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            let weights = ResidentImageUpsampleWeights::new(factor, &compact).unwrap();
            assert_eq!(
                weights.phase_major.len(),
                factor as usize * factor as usize * 25
            );
            assert_eq!(weights.storage_bytes(), u64::from(factor * factor * 25 * 4));
        }
    }

    #[test]
    fn shader_and_uniform_abi_are_semantically_valid() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<ResidentImageUpsampleParams>();
        let module =
            naga::front::wgsl::parse_str(include_str!("../shaders/upsample_rgb.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}
