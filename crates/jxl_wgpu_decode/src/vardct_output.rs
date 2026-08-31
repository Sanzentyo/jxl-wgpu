//! Fused GPU-resident XYB inverse, sRGB transfer, and packed RGB8 output.
//!
//! The kernel assigns one invocation to each output `u32`. That ownership rule
//! removes byte-level read/modify/write races while retaining the externally
//! visible tightly packed `RGBRGB...` byte layout. The allocation may contain
//! up to three zero padding bytes after the logical image payload.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::XybParams;
use jxl_wgpu::{KernelVariant, ResidentStorageBinding};
use wgpu::util::DeviceExt;

const OUTPUT_WORD_BYTES: u64 = std::mem::size_of::<u32>() as u64;
#[cfg(test)]
const WORKGROUP_SIZE: u32 = 256;
const DEFAULT_VARIANT: KernelVariant = KernelVariant::Lanes256;

/// WGSL source for the fused VarDCT output kernel.
pub const VAR_DCT_OUTPUT_SHADER: &str = include_str!("vardct_output.wgsl");

/// One GPU-resident F32 XYB plane.
#[derive(Clone, Copy, Debug)]
pub struct VarDctOutputPlane<'a> {
    /// Checked storage-buffer subrange containing row-major F32 samples.
    pub storage: ResidentStorageBinding<'a>,
    /// Row stride in F32 scalars. Zero selects the configured image width.
    pub stride: u32,
}

impl VarDctOutputPlane<'_> {
    const fn effective_stride(self, width: u32) -> u32 {
        if self.stride == 0 { width } else { self.stride }
    }
}

/// JPEG XL inverse-opsin fields used by the fused output kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarDctInverseOpsin {
    /// Per-LMS opsin biases from the codestream image metadata.
    pub opsin_bias: [f32; 3],
    /// Row-major matrix mapping reconstructed LMS into linear RGB.
    pub inverse_opsin_matrix: [[f32; 3]; 3],
    /// JPEG XL intensity target in nits.
    pub intensity_target: f32,
}

impl From<&XybParams> for VarDctInverseOpsin {
    fn from(value: &XybParams) -> Self {
        Self {
            opsin_bias: value.opsin_bias,
            inverse_opsin_matrix: value.inverse_opsin_matrix,
            intensity_target: value.intensity_target,
        }
    }
}

/// Host-known geometry and inverse-opsin metadata for one packed output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarDctOutputConfig {
    /// Logical pixel width.
    pub width: u32,
    /// Logical pixel height.
    pub height: u32,
    /// Inverse-opsin parameters with the same semantics as
    /// [`jxl_gpu_protocol::XybParams`].
    pub inverse_opsin: VarDctInverseOpsin,
}

/// GPU bindings consumed by [`VarDctOutputPacker::encode`].
#[derive(Clone, Copy, Debug)]
pub struct VarDctOutputInputs<'a> {
    /// X, Y, and B F32 planes, in that order.
    pub planes: [VarDctOutputPlane<'a>; 3],
    /// Packed output storage. Its logical bytes are tightly interleaved RGB8;
    /// its allocated/bound length is rounded up to four bytes.
    pub output: ResidentStorageBinding<'a>,
    /// Output geometry and inverse-opsin metadata.
    pub config: VarDctOutputConfig,
}

/// Exact byte counts for one fused output operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctOutputMemoryPlan {
    /// Externally visible RGB byte count, exactly `width * height * 3`.
    pub logical_output_bytes: u64,
    /// GPU storage allocation/binding requirement, rounded up to a `u32`.
    pub output_storage_bytes: u64,
    /// Fixed uniform allocation retained until submission.
    pub uniform_bytes: u64,
    /// All transient GPU-buffer bytes allocated by the packer.
    pub transient_bytes: u64,
    /// Output storage plus packer-owned transient storage.
    pub total_bytes: u64,
}

impl VarDctOutputMemoryPlan {
    /// Computes exact output and transient buffer bytes without a device.
    ///
    /// # Errors
    ///
    /// Returns a typed error for empty geometry, arithmetic overflow, or an
    /// image whose packed addressing cannot be represented by WGSL `u32`.
    pub fn new(width: u32, height: u32) -> Result<Self, VarDctOutputError> {
        let (pixel_count, logical_output_bytes, output_storage_bytes) =
            packed_geometry(width, height)?;
        debug_assert!(pixel_count != 0);
        let uniform_bytes = std::mem::size_of::<VarDctOutputParams>() as u64;
        let transient_bytes = uniform_bytes;
        let total_bytes = output_storage_bytes.checked_add(transient_bytes).ok_or(
            VarDctOutputError::ArithmeticOverflow {
                field: "VarDCT output total bytes",
            },
        )?;
        Ok(Self {
            logical_output_bytes,
            output_storage_bytes,
            uniform_bytes,
            transient_bytes,
            total_bytes,
        })
    }
}

/// Linear work distribution and byte accounting validated against one device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctOutputPlan {
    /// Exact byte accounting.
    pub memory: VarDctOutputMemoryPlan,
    /// Number of `u32` records written by the kernel.
    pub output_words: u32,
    /// Number of X workgroups.
    pub workgroups_x: u32,
    /// Number of Y workgroups.
    pub workgroups_y: u32,
    /// Number of invocations in each logical dispatch row.
    pub dispatch_width: u32,
}

impl VarDctOutputPlan {
    /// Plans a portable two-dimensional dispatch from device limits.
    ///
    /// # Errors
    ///
    /// Returns a typed limit or arithmetic error if the output cannot be
    /// represented by a single WebGPU storage binding and dispatch.
    pub fn for_limits(
        width: u32,
        height: u32,
        limits: &wgpu::Limits,
    ) -> Result<Self, VarDctOutputError> {
        Self::for_limits_with_variant(width, height, limits, DEFAULT_VARIANT)
    }

    /// Plans a dispatch using the selected 1D workgroup variant.
    pub fn for_limits_with_variant(
        width: u32,
        height: u32,
        limits: &wgpu::Limits,
        variant: KernelVariant,
    ) -> Result<Self, VarDctOutputError> {
        let memory = VarDctOutputMemoryPlan::new(width, height)?;
        validate_required_buffer(
            "packed RGB8 output",
            memory.output_storage_bytes,
            limits,
            true,
        )?;
        if memory.uniform_bytes > limits.max_uniform_buffer_binding_size {
            return Err(VarDctOutputError::UniformBindingLimit {
                required: memory.uniform_bytes,
                available: limits.max_uniform_buffer_binding_size,
            });
        }
        validate_workgroup_variant(variant, limits)?;

        let output_words_u64 = memory.output_storage_bytes / OUTPUT_WORD_BYTES;
        let output_words =
            u32::try_from(output_words_u64).map_err(|_| VarDctOutputError::ShaderAddressSpace {
                field: "packed RGB8 words",
                required: output_words_u64,
                available: u64::from(u32::MAX),
            })?;
        let limit = limits.max_compute_workgroups_per_dimension;
        if limit == 0 {
            return Err(VarDctOutputError::DispatchLimit {
                required_y: 1,
                available: 0,
            });
        }
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        let required_x = output_words.div_ceil(workgroup_x);
        let workgroups_x = required_x.min(limit);
        let dispatch_width =
            workgroups_x
                .checked_mul(workgroup_x)
                .ok_or(VarDctOutputError::ArithmeticOverflow {
                    field: "VarDCT output dispatch width",
                })?;
        let required_y = output_words.div_ceil(dispatch_width);
        let workgroups_y = required_y.div_ceil(workgroup_y);
        if workgroups_y > limit {
            return Err(VarDctOutputError::DispatchLimit {
                required_y: workgroups_y,
                available: limit,
            });
        }
        Ok(Self {
            memory,
            output_words,
            workgroups_x,
            workgroups_y,
            dispatch_width,
        })
    }
}

fn validate_workgroup_variant(
    variant: KernelVariant,
    limits: &wgpu::Limits,
) -> Result<(), VarDctOutputError> {
    if !variant.is_linear() {
        return Err(VarDctOutputError::WorkgroupShape { variant });
    }
    variant
        .validate_for("vardct_output", limits, 0)
        .map_err(|_| VarDctOutputError::WorkgroupSizeLimit {
            required: variant.invocations(),
            max_invocations: limits.max_compute_invocations_per_workgroup,
            max_size_x: limits.max_compute_workgroup_size_x,
        })
}

/// Uniform allocation that must remain live through command submission.
#[derive(Debug)]
pub struct VarDctOutputScratch {
    /// The fixed 128-byte parameter buffer.
    pub uniform: wgpu::Buffer,
    /// Exact output/transient accounting and dispatch geometry.
    pub plan: VarDctOutputPlan,
}

/// Typed validation errors for GPU-resident VarDCT RGB8 output.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum VarDctOutputError {
    /// The logical image has a zero dimension.
    #[error("VarDCT RGB8 output extent must be nonzero, got {width}x{height}")]
    EmptyExtent { width: u32, height: u32 },
    /// Checked size arithmetic overflowed.
    #[error("VarDCT RGB8 output arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
    /// A value cannot be addressed by WGSL's `u32` indices.
    #[error(
        "VarDCT RGB8 output {field} needs {required} addressable values, WGSL permits {available}"
    )]
    ShaderAddressSpace {
        field: &'static str,
        required: u64,
        available: u64,
    },
    /// One F32 plane has a row stride shorter than its width.
    #[error("VarDCT RGB8 input plane {plane} stride {stride} is shorter than width {width}")]
    InputStride {
        plane: usize,
        stride: u32,
        width: u32,
    },
    /// An inverse-opsin field is non-finite.
    #[error("VarDCT RGB8 inverse-opsin field {field} must be finite")]
    NonFiniteParameter { field: &'static str },
    /// The intensity target is finite but not positive.
    #[error("VarDCT RGB8 intensity target must be positive")]
    InvalidIntensityTarget,
    /// A buffer does not carry STORAGE usage.
    #[error("VarDCT RGB8 {role} buffer is missing STORAGE usage")]
    MissingStorageUsage { role: &'static str },
    /// A binding starts at an invalid device-specific offset.
    #[error("VarDCT RGB8 {role} offset {offset} is not aligned to {alignment}")]
    BindingOffsetAlignment {
        role: &'static str,
        offset: u64,
        alignment: u64,
    },
    /// A typed array binding does not end at a whole 32-bit word.
    #[error("VarDCT RGB8 {role} binding size {size} is not four-byte aligned")]
    BindingSizeAlignment { role: &'static str, size: u64 },
    /// A subrange exceeds its backing buffer.
    #[error("VarDCT RGB8 {role} range {offset}..{end} exceeds buffer size {available}")]
    BindingRange {
        role: &'static str,
        offset: u64,
        end: u64,
        available: u64,
    },
    /// A subrange is smaller than its image geometry requires.
    #[error("VarDCT RGB8 {role} binding needs {required} bytes, has {available}")]
    BindingSize {
        role: &'static str,
        required: u64,
        available: u64,
    },
    /// A required allocation exceeds a device buffer limit.
    #[error("VarDCT RGB8 {role} needs {required} bytes, device buffer limit is {available}")]
    BufferLimit {
        role: &'static str,
        required: u64,
        available: u64,
    },
    /// A required storage binding exceeds the device binding limit.
    #[error("VarDCT RGB8 {role} needs {required} bytes, storage binding limit is {available}")]
    StorageBindingLimit {
        role: &'static str,
        required: u64,
        available: u64,
    },
    /// The 128-byte uniform exceeds an unusual device limit.
    #[error("VarDCT RGB8 uniform needs {required} bytes, uniform binding limit is {available}")]
    UniformBindingLimit { required: u64, available: u64 },
    /// Output packing requires a one-dimensional workgroup.
    #[error("VarDCT RGB8 output requires a linear workgroup, got {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    /// The selected output workgroup cannot run on the device.
    #[error(
        "VarDCT RGB8 workgroup needs {required} X invocations, device permits {max_invocations} total and {max_size_x} in X"
    )]
    WorkgroupSizeLimit {
        required: u32,
        max_invocations: u32,
        max_size_x: u32,
    },
    /// A two-dimensional linearization still exceeds the device's Y limit.
    #[error("VarDCT RGB8 dispatch needs {required_y} Y workgroups, device permits {available}")]
    DispatchLimit { required_y: u32, available: u32 },
}

/// Reusable fused XYB-to-packed-sRGB8 compute pipeline.
pub struct VarDctOutputPacker {
    pipeline: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl VarDctOutputPacker {
    /// Compiles the output shader with the portable default workgroup.
    pub fn new(device: &wgpu::Device) -> Result<Self, VarDctOutputError> {
        Self::with_variant(device, DEFAULT_VARIANT)
    }

    /// Compiles the output shader with a selected workgroup variant.
    pub fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, VarDctOutputError> {
        validate_workgroup_variant(variant, &device.limits())?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu decode VarDCT packed RGB8"),
            source: wgpu::ShaderSource::Wgsl(VAR_DCT_OUTPUT_SHADER.into()),
        });
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        let constants = [
            ("wg_x", f64::from(workgroup_x)),
            ("wg_y", f64::from(workgroup_y)),
        ];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu decode VarDCT packed RGB8"),
            layout: None,
            module: &module,
            entry_point: Some("pack_rgb8"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
        Ok(Self { pipeline, variant })
    }

    /// Validates resident bindings and records one fused output dispatch.
    ///
    /// No pixel crosses the CPU. The caller owns the output allocation and can
    /// retain it for zero-copy display or copy/map it through the shared
    /// readback path. Keep the returned scratch value alive through submission.
    ///
    /// # Errors
    ///
    /// Returns a typed geometry, metadata, binding, arithmetic, or device-limit
    /// error before recording the dispatch.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: VarDctOutputInputs<'_>,
    ) -> Result<VarDctOutputScratch, VarDctOutputError> {
        let (params, plan) = validate_inputs(device, inputs, self.variant)?;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu decode VarDCT packed RGB8 params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu decode VarDCT packed RGB8 bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                binding_entry(0, inputs.planes[0].storage),
                binding_entry(1, inputs.planes[1].storage),
                binding_entry(2, inputs.planes[2].storage),
                binding_entry(3, inputs.output),
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu decode VarDCT packed RGB8"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(plan.workgroups_x, plan.workgroups_y, 1);
        drop(pass);
        Ok(VarDctOutputScratch { uniform, plan })
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct VarDctOutputParams {
    width: u32,
    height: u32,
    input_stride_x: u32,
    input_stride_y: u32,
    input_stride_b: u32,
    pixel_count: u32,
    output_word_count: u32,
    dispatch_width: u32,
    matrix_r: [f32; 4],
    matrix_g: [f32; 4],
    matrix_b: [f32; 4],
    bias_cbrt: [f32; 4],
    scaled_bias: [f32; 4],
    intensity_scale: f32,
    _padding: [u32; 3],
}

fn validate_inputs(
    device: &wgpu::Device,
    inputs: VarDctOutputInputs<'_>,
    variant: KernelVariant,
) -> Result<(VarDctOutputParams, VarDctOutputPlan), VarDctOutputError> {
    validate_inverse_opsin(inputs.config.inverse_opsin)?;
    let plan = VarDctOutputPlan::for_limits_with_variant(
        inputs.config.width,
        inputs.config.height,
        &device.limits(),
        variant,
    )?;
    let pixel_count_u64 = u64::from(inputs.config.width)
        .checked_mul(u64::from(inputs.config.height))
        .ok_or(VarDctOutputError::ArithmeticOverflow {
            field: "VarDCT output pixel count",
        })?;
    let pixel_count =
        u32::try_from(pixel_count_u64).map_err(|_| VarDctOutputError::ShaderAddressSpace {
            field: "pixels",
            required: pixel_count_u64,
            available: u64::from(u32::MAX),
        })?;

    let mut strides = [0; 3];
    for (plane, input) in inputs.planes.into_iter().enumerate() {
        let stride = input.effective_stride(inputs.config.width);
        if stride < inputs.config.width {
            return Err(VarDctOutputError::InputStride {
                plane,
                stride,
                width: inputs.config.width,
            });
        }
        let required_scalars = u64::from(inputs.config.height - 1)
            .checked_mul(u64::from(stride))
            .and_then(|value| value.checked_add(u64::from(inputs.config.width)))
            .ok_or(VarDctOutputError::ArithmeticOverflow {
                field: "VarDCT input plane scalars",
            })?;
        if required_scalars > u64::from(u32::MAX) {
            return Err(VarDctOutputError::ShaderAddressSpace {
                field: "input plane scalars",
                required: required_scalars,
                available: u64::from(u32::MAX),
            });
        }
        let required_bytes =
            required_scalars
                .checked_mul(4)
                .ok_or(VarDctOutputError::ArithmeticOverflow {
                    field: "VarDCT input plane bytes",
                })?;
        let role = match plane {
            0 => "X input",
            1 => "Y input",
            _ => "B input",
        };
        validate_binding(device, role, input.storage, required_bytes)?;
        strides[plane] = stride;
    }
    validate_binding(
        device,
        "packed RGB8 output",
        inputs.output,
        plan.memory.output_storage_bytes,
    )?;

    let inverse = inputs.config.inverse_opsin;
    let intensity_scale = 255.0 / inverse.intensity_target;
    let bias_cbrt = inverse.opsin_bias.map(f32::cbrt);
    let scaled_bias = inverse.opsin_bias.map(|value| value * intensity_scale);
    Ok((
        VarDctOutputParams {
            width: inputs.config.width,
            height: inputs.config.height,
            input_stride_x: strides[0],
            input_stride_y: strides[1],
            input_stride_b: strides[2],
            pixel_count,
            output_word_count: plan.output_words,
            dispatch_width: plan.dispatch_width,
            matrix_r: matrix_row(inverse.inverse_opsin_matrix[0]),
            matrix_g: matrix_row(inverse.inverse_opsin_matrix[1]),
            matrix_b: matrix_row(inverse.inverse_opsin_matrix[2]),
            bias_cbrt: [bias_cbrt[0], bias_cbrt[1], bias_cbrt[2], 0.0],
            scaled_bias: [scaled_bias[0], scaled_bias[1], scaled_bias[2], 0.0],
            intensity_scale,
            _padding: [0; 3],
        },
        plan,
    ))
}

fn validate_inverse_opsin(inverse: VarDctInverseOpsin) -> Result<(), VarDctOutputError> {
    if !inverse.intensity_target.is_finite() {
        return Err(VarDctOutputError::NonFiniteParameter {
            field: "intensity_target",
        });
    }
    if inverse.intensity_target <= 0.0 {
        return Err(VarDctOutputError::InvalidIntensityTarget);
    }
    for (index, value) in inverse.opsin_bias.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(VarDctOutputError::NonFiniteParameter {
                field: ["opsin_bias[0]", "opsin_bias[1]", "opsin_bias[2]"][index],
            });
        }
    }
    for (row, matrix_row) in inverse.inverse_opsin_matrix.into_iter().enumerate() {
        for (column, value) in matrix_row.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(VarDctOutputError::NonFiniteParameter {
                    field: [
                        ["matrix[0][0]", "matrix[0][1]", "matrix[0][2]"],
                        ["matrix[1][0]", "matrix[1][1]", "matrix[1][2]"],
                        ["matrix[2][0]", "matrix[2][1]", "matrix[2][2]"],
                    ][row][column],
                });
            }
        }
    }
    Ok(())
}

fn packed_geometry(width: u32, height: u32) -> Result<(u32, u64, u64), VarDctOutputError> {
    if width == 0 || height == 0 {
        return Err(VarDctOutputError::EmptyExtent { width, height });
    }
    let pixel_count_u64 = u64::from(width).checked_mul(u64::from(height)).ok_or(
        VarDctOutputError::ArithmeticOverflow {
            field: "VarDCT output pixel count",
        },
    )?;
    let pixel_count =
        u32::try_from(pixel_count_u64).map_err(|_| VarDctOutputError::ShaderAddressSpace {
            field: "pixels",
            required: pixel_count_u64,
            available: u64::from(u32::MAX),
        })?;
    let logical_output_bytes =
        pixel_count_u64
            .checked_mul(3)
            .ok_or(VarDctOutputError::ArithmeticOverflow {
                field: "logical RGB8 output bytes",
            })?;
    if logical_output_bytes > u64::from(u32::MAX) {
        return Err(VarDctOutputError::ShaderAddressSpace {
            field: "logical RGB8 output bytes",
            required: logical_output_bytes,
            available: u64::from(u32::MAX),
        });
    }
    let output_storage_bytes = logical_output_bytes
        .checked_add(OUTPUT_WORD_BYTES - 1)
        .map(|value| value / OUTPUT_WORD_BYTES * OUTPUT_WORD_BYTES)
        .ok_or(VarDctOutputError::ArithmeticOverflow {
            field: "aligned RGB8 output bytes",
        })?;
    Ok((pixel_count, logical_output_bytes, output_storage_bytes))
}

fn validate_required_buffer(
    role: &'static str,
    required: u64,
    limits: &wgpu::Limits,
    storage: bool,
) -> Result<(), VarDctOutputError> {
    if required > limits.max_buffer_size {
        return Err(VarDctOutputError::BufferLimit {
            role,
            required,
            available: limits.max_buffer_size,
        });
    }
    if storage && required > limits.max_storage_buffer_binding_size {
        return Err(VarDctOutputError::StorageBindingLimit {
            role,
            required,
            available: limits.max_storage_buffer_binding_size,
        });
    }
    Ok(())
}

fn validate_binding(
    device: &wgpu::Device,
    role: &'static str,
    binding: ResidentStorageBinding<'_>,
    required: u64,
) -> Result<(), VarDctOutputError> {
    if !binding.buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
        return Err(VarDctOutputError::MissingStorageUsage { role });
    }
    let limits = device.limits();
    let alignment = u64::from(limits.min_storage_buffer_offset_alignment).max(4);
    if !binding.offset.is_multiple_of(alignment) {
        return Err(VarDctOutputError::BindingOffsetAlignment {
            role,
            offset: binding.offset,
            alignment,
        });
    }
    let size = binding.size.get();
    if !size.is_multiple_of(OUTPUT_WORD_BYTES) {
        return Err(VarDctOutputError::BindingSizeAlignment { role, size });
    }
    let end = binding
        .offset
        .checked_add(size)
        .ok_or(VarDctOutputError::ArithmeticOverflow {
            field: "VarDCT storage binding range",
        })?;
    if end > binding.buffer.size() {
        return Err(VarDctOutputError::BindingRange {
            role,
            offset: binding.offset,
            end,
            available: binding.buffer.size(),
        });
    }
    if size < required {
        return Err(VarDctOutputError::BindingSize {
            role,
            required,
            available: size,
        });
    }
    validate_required_buffer(role, size, &limits, true)
}

fn binding_entry(binding: u32, storage: ResidentStorageBinding<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: storage.buffer,
            offset: storage.offset,
            size: Some(storage.size),
        }),
    }
}

const fn matrix_row(row: [f32; 3]) -> [f32; 4] {
    [row[0], row[1], row[2], 0.0]
}

const _: () = {
    assert!(std::mem::size_of::<VarDctOutputParams>() == 128);
    assert!(std::mem::align_of::<VarDctOutputParams>() == 16);
};

#[cfg(test)]
mod tests {
    use super::*;

    fn generous_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_buffer_size: 2 * 1024 * 1024 * 1024,
            max_storage_buffer_binding_size: 2 * 1024 * 1024 * 1024,
            max_uniform_buffer_binding_size: 64 * 1024,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroups_per_dimension: 65_535,
            ..wgpu::Limits::default()
        }
    }

    fn inverse_opsin() -> VarDctInverseOpsin {
        VarDctInverseOpsin {
            opsin_bias: [-0.003_793_073_4; 3],
            inverse_opsin_matrix: [
                [11.031_567, -9.866_944, -0.164_622_99],
                [-3.254_147_3, 4.418_770_3, -0.164_622_99],
                [-3.658_851_4, 2.712_923, 1.945_928_2],
            ],
            intensity_target: 255.0,
        }
    }

    #[test]
    fn wgsl_and_uniform_abi_validate() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<VarDctOutputParams>();
        assert_eq!(std::mem::size_of::<VarDctOutputParams>(), 128);
        assert_eq!(std::mem::align_of::<VarDctOutputParams>(), 16);

        let module =
            naga::front::wgsl::parse_str(VAR_DCT_OUTPUT_SHADER).expect("VarDCT output WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("VarDCT output WGSL validates with portable capabilities");
    }

    #[test]
    fn memory_plan_separates_logical_storage_and_transient_bytes() {
        let memory = VarDctOutputMemoryPlan::new(5, 3).unwrap();
        assert_eq!(memory.logical_output_bytes, 45);
        assert_eq!(memory.output_storage_bytes, 48);
        assert_eq!(memory.uniform_bytes, 128);
        assert_eq!(memory.transient_bytes, 128);
        assert_eq!(memory.total_bytes, 176);

        let plan = VarDctOutputPlan::for_limits(5, 3, &generous_limits()).unwrap();
        assert_eq!(plan.output_words, 12);
        assert_eq!((plan.workgroups_x, plan.workgroups_y), (1, 1));
        assert_eq!(plan.dispatch_width, WORKGROUP_SIZE);
    }

    #[test]
    fn sixteen_k_output_is_split_across_dispatch_rows() {
        let plan = VarDctOutputPlan::for_limits(16_384, 16_384, &generous_limits()).unwrap();
        assert_eq!(plan.memory.logical_output_bytes, 805_306_368);
        assert_eq!(plan.output_words, 201_326_592);
        assert_eq!(plan.workgroups_x, 65_535);
        assert_eq!(plan.dispatch_width, 65_535 * WORKGROUP_SIZE);
        assert_eq!(plan.workgroups_y, 13);
        assert!(
            u64::from(plan.workgroups_x) * u64::from(plan.workgroups_y) * u64::from(WORKGROUP_SIZE)
                >= u64::from(plan.output_words)
        );
    }

    #[test]
    fn invalid_geometry_and_opsin_have_stable_typed_errors() {
        assert_eq!(
            VarDctOutputMemoryPlan::new(0, 7).unwrap_err(),
            VarDctOutputError::EmptyExtent {
                width: 0,
                height: 7
            }
        );
        let mut invalid = inverse_opsin();
        invalid.intensity_target = 0.0;
        assert_eq!(
            validate_inverse_opsin(invalid).unwrap_err(),
            VarDctOutputError::InvalidIntensityTarget
        );
        assert_eq!(
            VarDctOutputPlan::for_limits_with_variant(
                1,
                1,
                &wgpu::Limits::default(),
                KernelVariant::Tile8x8,
            )
            .unwrap_err(),
            VarDctOutputError::WorkgroupShape {
                variant: KernelVariant::Tile8x8,
            }
        );
        invalid = inverse_opsin();
        invalid.inverse_opsin_matrix[2][1] = f32::NAN;
        assert_eq!(
            validate_inverse_opsin(invalid).unwrap_err(),
            VarDctOutputError::NonFiniteParameter {
                field: "matrix[2][1]"
            }
        );
    }

    #[test]
    fn existing_xyb_contract_maps_without_reinterpretation() {
        let inverse = inverse_opsin();
        let protocol = XybParams {
            opsin_bias: inverse.opsin_bias,
            inverse_opsin_matrix: inverse.inverse_opsin_matrix,
            intensity_target: inverse.intensity_target,
        };
        assert_eq!(VarDctInverseOpsin::from(&protocol), inverse);
        assert_eq!(255.0 / inverse.intensity_target, 1.0);
        let cube_root = inverse.opsin_bias[0].cbrt();
        assert!((cube_root * cube_root * cube_root - inverse.opsin_bias[0]).abs() < 1.0e-8);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_packer_owns_words_and_zeroes_tail_padding() {
        use std::num::NonZeroU64;
        use std::sync::mpsc;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            }))
        else {
            eprintln!("skipping VarDCT RGB8 packer GPU test: no adapter");
            return;
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("jxl-wgpu VarDCT RGB8 packer test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))
        else {
            eprintln!("skipping VarDCT RGB8 packer GPU test: device request failed");
            return;
        };
        let storage_plane = |label, samples: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(samples),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let x = storage_plane("VarDCT RGB8 test X", &[0.028_100_073, -0.015_386_105, 0.0]);
        let y = storage_plane(
            "VarDCT RGB8 test Y",
            &[0.488_188_2, 0.714_781_34, 0.278_128_2],
        );
        let b = storage_plane(
            "VarDCT RGB8 test B",
            &[0.471_659, 0.437_076_93, 0.666_139_84],
        );
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VarDCT RGB8 test output"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VarDCT RGB8 test staging"),
            size: 12,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        fn binding(buffer: &wgpu::Buffer) -> ResidentStorageBinding<'_> {
            ResidentStorageBinding {
                buffer,
                offset: 0,
                size: NonZeroU64::new(buffer.size()).unwrap(),
            }
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("VarDCT RGB8 test commands"),
        });
        let packer = VarDctOutputPacker::new(&device).unwrap();
        let scratch = packer
            .encode(
                &device,
                &mut encoder,
                VarDctOutputInputs {
                    planes: [
                        VarDctOutputPlane {
                            storage: binding(&x),
                            stride: 3,
                        },
                        VarDctOutputPlane {
                            storage: binding(&y),
                            stride: 3,
                        },
                        VarDctOutputPlane {
                            storage: binding(&b),
                            stride: 3,
                        },
                    ],
                    output: binding(&output),
                    config: VarDctOutputConfig {
                        width: 3,
                        height: 1,
                        inverse_opsin: inverse_opsin(),
                    },
                },
            )
            .expect("record fused VarDCT RGB8 output");
        assert_eq!(scratch.plan.memory.logical_output_bytes, 9);
        assert_eq!(scratch.plan.memory.output_storage_bytes, 12);
        assert_eq!(scratch.uniform.size(), 128);
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 12);
        let submission = queue.submit([encoder.finish()]);
        let (sender, receiver) = mpsc::sync_channel(1);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("poll fused VarDCT RGB8 output");
        receiver
            .recv()
            .expect("VarDCT RGB8 map callback")
            .expect("map VarDCT RGB8 output");
        let mapped = staging
            .slice(..)
            .get_mapped_range()
            .expect("mapped VarDCT RGB8 output");
        assert_eq!(
            &*mapped,
            &[255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0],
            "RGB primaries exercise all three word phases and tail padding"
        );
    }
}
