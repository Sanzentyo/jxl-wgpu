//! Fused GPU-resident XYB/YCbCr reconstruction and shared pitch-linear color output.
//!
//! The kernel assigns one invocation to each output `u32`. That ownership rule
//! removes byte-level read/modify/write races across all planes and pixel layouts. The allocation may contain
//! up to three zero padding bytes after the logical image payload.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_formats::{ColorSpecification, ImageLayout, TransferFunction};
use jxl_gpu_protocol::{Extent2d, OutputOrientation, RgbColorEncoding, XybParams};
use jxl_wgpu::{ImageOutputParams, ImageOutputSource, KernelVariant, ResidentStorageBinding};
use wgpu::util::DeviceExt;

use crate::vardct_frontend::VarDctChannelShift;

const OUTPUT_WORD_BYTES: u64 = std::mem::size_of::<u32>() as u64;
#[cfg(test)]
const WORKGROUP_SIZE: u32 = 256;
const DEFAULT_VARIANT: KernelVariant = KernelVariant::Lanes256;

/// WGSL source for the fused VarDCT output kernel.
pub fn vardct_output_shader() -> String {
    format!(
        "{}\n{}",
        jxl_wgpu::IMAGE_OUTPUT_SHADER,
        include_str!("vardct_output.wgsl")
    )
}

/// One GPU-resident F32 XYB or JPEG component plane.
#[derive(Clone, Copy, Debug)]
pub struct VarDctOutputPlane<'a> {
    /// Checked storage-buffer subrange containing row-major F32 samples.
    pub storage: ResidentStorageBinding<'a>,
    /// Available logical samples in each row.
    pub width: u32,
    /// Available logical rows.
    pub height: u32,
    /// Row stride in F32 scalars. Zero selects the configured image width.
    pub stride: u32,
}

impl VarDctOutputPlane<'_> {
    const fn effective_stride(self) -> u32 {
        if self.stride == 0 {
            self.width
        } else {
            self.stride
        }
    }
}

/// Color transform fused into the final packed color kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VarDctOutputTransform {
    /// JPEG XL XYB inverse into linear BT.709; the shared output applies the requested transfer.
    Xyb(VarDctInverseOpsin),
    /// JPEG reconstruction's encoded YCbCr, including component upsampling.
    Ycbcr {
        channel_shifts: [VarDctChannelShift; 3],
    },
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
    /// Logical image extent after resampling, before orientation.
    pub extent: Extent2d,
    /// Maps input coordinates into the display-oriented packed output.
    pub orientation: OutputOrientation,
    pub transform: VarDctOutputTransform,
}

impl VarDctOutputConfig {
    /// Extent of the packed output, including transposition for orientations 5–8.
    pub const fn output_extent(self) -> Extent2d {
        self.orientation.map_extent(self.extent)
    }

    /// Validates target color, geometry, and packing before allocation or submission.
    pub fn validate_layout(self, layout: &ImageLayout) -> Result<(), VarDctOutputError> {
        self.image_params(layout, [self.extent.width; 3], 1)
            .map(|_| ())
    }

    fn image_params(
        self,
        layout: &ImageLayout,
        strides: [u32; 3],
        dispatch_width: u32,
    ) -> Result<ImageOutputParams, VarDctOutputError> {
        if matches!(layout.format.color_spec, ColorSpecification::Defined(color) if matches!(color.transfer, TransferFunction::Pq | TransferFunction::Hlg))
        {
            return Err(VarDctOutputError::HdrLuminanceMappingRequired);
        }
        Ok(ImageOutputParams::new(
            layout,
            ImageOutputSource {
                extent: self.extent,
                orientation: self.orientation,
                strides,
                encoding: match self.transform {
                    VarDctOutputTransform::Xyb(_) => RgbColorEncoding::LINEAR_BT709,
                    VarDctOutputTransform::Ycbcr { .. } => RgbColorEncoding::SRGB_BT709,
                },
            },
            dispatch_width,
        )?)
    }
}

/// GPU bindings consumed by [`VarDctOutputPacker::encode`].
#[derive(Clone, Copy, Debug)]
pub struct VarDctOutputInputs<'a> {
    /// X, Y, and B F32 planes, in that order.
    pub planes: [VarDctOutputPlane<'a>; 3],
    /// Optional full-resolution, unassociated integer alpha reconstructed by Modular.
    pub alpha: Option<VarDctOutputAlpha<'a>>,
    /// Output storage for the requested pitch-linear layout;
    /// its allocated/bound length is rounded up to four bytes.
    pub output: ResidentStorageBinding<'a>,
    /// Exact target layout, including oriented extent, plane offsets, and row pitches.
    pub layout: &'a ImageLayout,
    /// Output geometry and inverse-opsin metadata.
    pub config: VarDctOutputConfig,
}

/// An opacity plane in a resident Modular or normalized F32 arena, one word per sample.
#[derive(Clone, Copy, Debug)]
pub struct VarDctOutputAlpha<'a> {
    pub domain: crate::ModularSampleDomain,
    pub storage: ResidentStorageBinding<'a>,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub word_offset: u32,
    pub bits_per_sample: u32,
}

/// Exact byte counts for one fused output operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctOutputMemoryPlan {
    /// Externally visible bytes, including row padding and alignment between planes.
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
    /// Independent shared-output and codec-source uniform bindings.
    pub const UNIFORM_BINDING_BYTES: [u64; 2] = [
        std::mem::size_of::<ImageOutputParams>() as u64,
        std::mem::size_of::<VarDctSourceParams>() as u64,
    ];

    /// Computes exact output and transient buffer bytes without a device.
    ///
    /// # Errors
    ///
    /// Returns a typed error for empty geometry, arithmetic overflow, or an
    /// image whose packed addressing cannot be represented by WGSL `u32`.
    pub fn new(layout: &ImageLayout) -> Result<Self, VarDctOutputError> {
        let (logical_output_bytes, output_storage_bytes) = packed_geometry(layout)?;
        let uniform_bytes = Self::UNIFORM_BINDING_BYTES.into_iter().sum();
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
        layout: &ImageLayout,
        limits: &wgpu::Limits,
    ) -> Result<Self, VarDctOutputError> {
        Self::for_limits_with_variant(layout, limits, DEFAULT_VARIANT)
    }

    /// Plans a dispatch using the selected 1D workgroup variant.
    pub fn for_limits_with_variant(
        layout: &ImageLayout,
        limits: &wgpu::Limits,
        variant: KernelVariant,
    ) -> Result<Self, VarDctOutputError> {
        validate_storage_bindings(limits)?;
        let memory = VarDctOutputMemoryPlan::new(layout)?;
        validate_required_buffer(
            "packed color output",
            memory.output_storage_bytes,
            limits,
            true,
        )?;
        let [output_uniform, source_uniform] = VarDctOutputMemoryPlan::UNIFORM_BINDING_BYTES;
        let largest_uniform = output_uniform.max(source_uniform);
        if largest_uniform > limits.max_uniform_buffer_binding_size {
            return Err(VarDctOutputError::UniformBindingLimit {
                required: largest_uniform,
                available: limits.max_uniform_buffer_binding_size,
            });
        }
        validate_workgroup_variant(variant, limits)?;

        let output_words_u64 = memory.output_storage_bytes / OUTPUT_WORD_BYTES;
        let output_words =
            u32::try_from(output_words_u64).map_err(|_| VarDctOutputError::ShaderAddressSpace {
                field: "packed color words",
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

fn validate_storage_bindings(limits: &wgpu::Limits) -> Result<(), VarDctOutputError> {
    if limits.max_storage_buffers_per_shader_stage < 5 {
        return Err(VarDctOutputError::StorageBindingCount {
            available: limits.max_storage_buffers_per_shader_stage,
        });
    }
    Ok(())
}

/// Uniform allocation that must remain live through command submission.
#[derive(Debug)]
pub struct VarDctOutputScratch {
    /// The shared 176-byte color/layout parameter buffer.
    pub uniform: wgpu::Buffer,
    /// The 160-byte inverse-opsin/JPEG and alpha source parameter buffer.
    pub source_uniform: wgpu::Buffer,
    /// Exact output/transient accounting and dispatch geometry.
    pub plan: VarDctOutputPlan,
}

/// Typed validation errors for GPU-resident VarDCT color output.
#[derive(Debug, thiserror::Error)]
pub enum VarDctOutputError {
    #[error("VarDCT output needs five storage bindings, device permits {available}")]
    StorageBindingCount { available: u32 },
    #[error("VarDCT alpha requires 1–16-bit integer samples, got {bits}")]
    InvalidAlphaBitDepth { bits: u32 },
    /// The common output contract rejected color or layout metadata.
    #[error(transparent)]
    ImageOutput(#[from] jxl_wgpu::Error),
    /// Relative SDR values cannot be relabeled as absolute PQ or scene-linear HLG.
    #[error("VarDCT HDR output requires an explicit luminance mapping")]
    HdrLuminanceMappingRequired,
    /// Checked size arithmetic overflowed.
    #[error("VarDCT color output arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
    /// A value cannot be addressed by WGSL's `u32` indices.
    #[error(
        "VarDCT color output {field} needs {required} addressable values, WGSL permits {available}"
    )]
    ShaderAddressSpace {
        field: &'static str,
        required: u64,
        available: u64,
    },
    /// One F32 plane has a row stride shorter than its width.
    #[error("VarDCT color input plane {plane} stride {stride} is shorter than width {width}")]
    InputStride {
        plane: usize,
        stride: u32,
        width: u32,
    },
    /// One input plane does not cover the component extent required by the color transform.
    #[error(
        "VarDCT color input plane {plane} extent {width}x{height} is smaller than required {required_width}x{required_height}"
    )]
    InputExtent {
        plane: usize,
        width: u32,
        height: u32,
        required_width: u32,
        required_height: u32,
    },
    /// JPEG component shifts are limited to the one-bit factors defined by the codestream.
    #[error("VarDCT color JPEG channel {channel} has invalid shift {horizontal}x{vertical}")]
    InvalidJpegShift {
        channel: usize,
        horizontal: u32,
        vertical: u32,
    },
    /// An inverse-opsin field is non-finite.
    #[error("VarDCT color inverse-opsin field {field} must be finite")]
    NonFiniteParameter { field: &'static str },
    /// The intensity target is finite but not positive.
    #[error("VarDCT color intensity target must be positive")]
    InvalidIntensityTarget,
    /// A buffer does not carry STORAGE usage.
    #[error("VarDCT color {role} buffer is missing STORAGE usage")]
    MissingStorageUsage { role: &'static str },
    /// A binding starts at an invalid device-specific offset.
    #[error("VarDCT color {role} offset {offset} is not aligned to {alignment}")]
    BindingOffsetAlignment {
        role: &'static str,
        offset: u64,
        alignment: u64,
    },
    /// A typed array binding does not end at a whole 32-bit word.
    #[error("VarDCT color {role} binding size {size} is not four-byte aligned")]
    BindingSizeAlignment { role: &'static str, size: u64 },
    /// A subrange exceeds its backing buffer.
    #[error("VarDCT color {role} range {offset}..{end} exceeds buffer size {available}")]
    BindingRange {
        role: &'static str,
        offset: u64,
        end: u64,
        available: u64,
    },
    /// A subrange is smaller than its image geometry requires.
    #[error("VarDCT color {role} binding needs {required} bytes, has {available}")]
    BindingSize {
        role: &'static str,
        required: u64,
        available: u64,
    },
    /// A required allocation exceeds a device buffer limit.
    #[error("VarDCT color {role} needs {required} bytes, device buffer limit is {available}")]
    BufferLimit {
        role: &'static str,
        required: u64,
        available: u64,
    },
    /// A required storage binding exceeds the device binding limit.
    #[error("VarDCT color {role} needs {required} bytes, storage binding limit is {available}")]
    StorageBindingLimit {
        role: &'static str,
        required: u64,
        available: u64,
    },
    /// The 176-byte uniform exceeds an unusual device limit.
    #[error("VarDCT color uniform needs {required} bytes, uniform binding limit is {available}")]
    UniformBindingLimit { required: u64, available: u64 },
    /// Output packing requires a one-dimensional workgroup.
    #[error("VarDCT color output requires a linear workgroup, got {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    /// The selected output workgroup cannot run on the device.
    #[error(
        "VarDCT color workgroup needs {required} X invocations, device permits {max_invocations} total and {max_size_x} in X"
    )]
    WorkgroupSizeLimit {
        required: u32,
        max_invocations: u32,
        max_size_x: u32,
    },
    /// A two-dimensional linearization still exceeds the device's Y limit.
    #[error("VarDCT color dispatch needs {required_y} Y workgroups, device permits {available}")]
    DispatchLimit { required_y: u32, available: u32 },
}

/// Reusable fused XYB/YCbCr reconstruction and pitch-linear color output pipeline.
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
        validate_storage_bindings(&device.limits())?;
        validate_workgroup_variant(variant, &device.limits())?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu decode VarDCT packed color"),
            source: wgpu::ShaderSource::Wgsl(vardct_output_shader().into()),
        });
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        let constants = [
            ("wg_x", f64::from(workgroup_x)),
            ("wg_y", f64::from(workgroup_y)),
        ];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu decode VarDCT packed color"),
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
        let (source_params, params, plan) = validate_inputs(device, inputs, self.variant)?;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu decode VarDCT packed color params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let source_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu decode VarDCT source params"),
            contents: bytemuck::bytes_of(&source_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu decode VarDCT packed color bindings"),
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
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: source_uniform.as_entire_binding(),
                },
                binding_entry(
                    6,
                    inputs
                        .alpha
                        .map_or(inputs.planes[0].storage, |alpha| alpha.storage),
                ),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu decode VarDCT packed color"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(plan.workgroups_x, plan.workgroups_y, 1);
        drop(pass);
        Ok(VarDctOutputScratch {
            uniform,
            source_uniform,
            plan,
        })
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct VarDctSourceParams {
    plane_geometry: [[u32; 4]; 3],
    alpha_geometry: [u32; 4],
    matrix_r: [f32; 4],
    matrix_g: [f32; 4],
    matrix_b: [f32; 4],
    bias_cbrt: [f32; 4],
    scaled_bias: [f32; 4],
    intensity_scale: f32,
    mode: u32,
    _padding: [u32; 2],
}

fn validate_inputs(
    device: &wgpu::Device,
    inputs: VarDctOutputInputs<'_>,
    variant: KernelVariant,
) -> Result<(VarDctSourceParams, ImageOutputParams, VarDctOutputPlan), VarDctOutputError> {
    if let VarDctOutputTransform::Xyb(inverse) = inputs.config.transform {
        validate_inverse_opsin(inverse)?;
    }
    let plan = VarDctOutputPlan::for_limits_with_variant(inputs.layout, &device.limits(), variant)?;
    let output_params = inputs.config.image_params(
        inputs.layout,
        inputs.planes.map(VarDctOutputPlane::effective_stride),
        plan.dispatch_width,
    )?;

    let required_extents = match inputs.config.transform {
        VarDctOutputTransform::Xyb(_) => {
            [[inputs.config.extent.width, inputs.config.extent.height]; 3]
        }
        VarDctOutputTransform::Ycbcr { channel_shifts } => {
            let mut extents = [[0; 2]; 3];
            for (channel, shift) in channel_shifts.into_iter().enumerate() {
                if shift.horizontal > 1 || shift.vertical > 1 {
                    return Err(VarDctOutputError::InvalidJpegShift {
                        channel,
                        horizontal: shift.horizontal,
                        vertical: shift.vertical,
                    });
                }
                extents[channel] = [
                    inputs.config.extent.width.div_ceil(1 << shift.horizontal),
                    inputs.config.extent.height.div_ceil(1 << shift.vertical),
                ];
            }
            extents
        }
    };
    let mut plane_geometry = [[0; 4]; 3];
    for (plane, input) in inputs.planes.into_iter().enumerate() {
        let [required_width, required_height] = required_extents[plane];
        if input.width < required_width || input.height < required_height {
            return Err(VarDctOutputError::InputExtent {
                plane,
                width: input.width,
                height: input.height,
                required_width,
                required_height,
            });
        }
        let stride = input.effective_stride();
        if stride < input.width {
            return Err(VarDctOutputError::InputStride {
                plane,
                stride,
                width: input.width,
            });
        }
        let required_scalars = u64::from(required_height - 1)
            .checked_mul(u64::from(stride))
            .and_then(|value| value.checked_add(u64::from(required_width)))
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
        let shift = match inputs.config.transform {
            VarDctOutputTransform::Xyb(_) => VarDctChannelShift::default(),
            VarDctOutputTransform::Ycbcr { channel_shifts } => channel_shifts[plane],
        };
        plane_geometry[plane] = [
            stride,
            required_width,
            required_height,
            shift.horizontal | (shift.vertical << 1),
        ];
    }
    validate_binding(
        device,
        "packed color output",
        inputs.output,
        plan.memory.output_storage_bytes,
    )?;

    let alpha_geometry = if let Some(alpha) = inputs.alpha {
        if !(1..=16).contains(&alpha.bits_per_sample) {
            return Err(VarDctOutputError::InvalidAlphaBitDepth {
                bits: alpha.bits_per_sample,
            });
        }
        if alpha.width < inputs.config.extent.width || alpha.height < inputs.config.extent.height {
            return Err(VarDctOutputError::InputExtent {
                plane: 3,
                width: alpha.width,
                height: alpha.height,
                required_width: inputs.config.extent.width,
                required_height: inputs.config.extent.height,
            });
        }
        if alpha.stride < alpha.width {
            return Err(VarDctOutputError::InputStride {
                plane: 3,
                stride: alpha.stride,
                width: alpha.width,
            });
        }
        let end = u64::from(inputs.config.extent.height - 1)
            .checked_mul(u64::from(alpha.stride))
            .and_then(|value| value.checked_add(u64::from(alpha.word_offset)))
            .and_then(|value| value.checked_add(u64::from(inputs.config.extent.width)))
            .ok_or(VarDctOutputError::ArithmeticOverflow {
                field: "alpha addressing",
            })?;
        if end > u64::from(u32::MAX) {
            return Err(VarDctOutputError::ShaderAddressSpace {
                field: "alpha plane scalars",
                required: end,
                available: u64::from(u32::MAX),
            });
        }
        validate_binding(device, "alpha input", alpha.storage, end * 4)?;
        [
            alpha.word_offset,
            alpha.stride,
            (1 << alpha.bits_per_sample) - 1,
            1 + alpha.domain as u32,
        ]
    } else {
        [0; 4]
    };

    let (mode, matrix, bias_cbrt, scaled_bias, intensity_scale) = match inputs.config.transform {
        VarDctOutputTransform::Xyb(inverse) => {
            let intensity_scale = 255.0 / inverse.intensity_target;
            (
                0,
                inverse.inverse_opsin_matrix,
                inverse.opsin_bias.map(f32::cbrt),
                inverse.opsin_bias.map(|value| value * intensity_scale),
                intensity_scale,
            )
        }
        VarDctOutputTransform::Ycbcr { .. } => (1, [[0.0; 3]; 3], [0.0; 3], [0.0; 3], 0.0),
    };
    Ok((
        VarDctSourceParams {
            plane_geometry,
            alpha_geometry,
            matrix_r: matrix_row(matrix[0]),
            matrix_g: matrix_row(matrix[1]),
            matrix_b: matrix_row(matrix[2]),
            bias_cbrt: [bias_cbrt[0], bias_cbrt[1], bias_cbrt[2], 0.0],
            scaled_bias: [scaled_bias[0], scaled_bias[1], scaled_bias[2], 0.0],
            intensity_scale,
            mode,
            _padding: [0; 2],
        },
        output_params,
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

fn packed_geometry(layout: &ImageLayout) -> Result<(u64, u64), VarDctOutputError> {
    let validated =
        ImageLayout::from_planes(layout.extent, layout.format.clone(), layout.planes.clone())
            .map_err(jxl_wgpu::Error::from)?;
    if validated.logical_size != layout.logical_size {
        return Err(jxl_wgpu::Error::InvalidPayload(
            "VarDCT output logical size disagrees with its planes".into(),
        )
        .into());
    }
    let logical_output_bytes = layout.logical_size;
    if logical_output_bytes > u64::from(u32::MAX) {
        return Err(VarDctOutputError::ShaderAddressSpace {
            field: "logical output bytes",
            required: logical_output_bytes,
            available: u64::from(u32::MAX),
        });
    }
    Ok((
        logical_output_bytes,
        logical_output_bytes.div_ceil(OUTPUT_WORD_BYTES) * OUTPUT_WORD_BYTES,
    ))
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
    assert!(std::mem::size_of::<VarDctSourceParams>() == 160);
    assert!(std::mem::align_of::<VarDctSourceParams>() == 16);
    assert!(std::mem::offset_of!(VarDctSourceParams, mode) == 148);
    assert!(std::mem::offset_of!(VarDctSourceParams, plane_geometry) == 0);
};

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_layout(width: u32, height: u32) -> ImageLayout {
        ImageLayout::packed(
            Extent2d::new(width, height),
            crate::vardct_engine::vardct_rgb8_format(),
        )
        .unwrap()
    }

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
        assert_pod::<VarDctSourceParams>();
        assert_eq!(std::mem::size_of::<VarDctSourceParams>(), 160);
        assert_eq!(std::mem::align_of::<VarDctSourceParams>(), 16);

        let module = naga::front::wgsl::parse_str(&vardct_output_shader())
            .expect("VarDCT output WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("VarDCT output WGSL validates with portable capabilities");
        assert_eq!(
            module
                .global_variables
                .iter()
                .filter(|(_, variable)| matches!(
                    variable.space,
                    naga::AddressSpace::Storage { .. }
                ))
                .count(),
            5
        );
        let mut limits = generous_limits();
        limits.max_storage_buffers_per_shader_stage = 4;
        assert!(matches!(
            VarDctOutputPlan::for_limits(&rgb_layout(5, 3), &limits),
            Err(VarDctOutputError::StorageBindingCount { available: 4 })
        ));
    }

    #[test]
    fn memory_plan_separates_logical_storage_and_transient_bytes() {
        let memory = VarDctOutputMemoryPlan::new(&rgb_layout(5, 3)).unwrap();
        assert_eq!(memory.logical_output_bytes, 45);
        assert_eq!(memory.output_storage_bytes, 48);
        assert_eq!(memory.uniform_bytes, 336);
        assert_eq!(memory.transient_bytes, 336);
        assert_eq!(memory.total_bytes, 384);

        let plan = VarDctOutputPlan::for_limits(&rgb_layout(5, 3), &generous_limits()).unwrap();
        assert_eq!(plan.output_words, 12);
        assert_eq!((plan.workgroups_x, plan.workgroups_y), (1, 1));
        assert_eq!(plan.dispatch_width, WORKGROUP_SIZE);
    }

    #[test]
    fn sixteen_k_output_is_split_across_dispatch_rows() {
        let plan =
            VarDctOutputPlan::for_limits(&rgb_layout(16_384, 16_384), &generous_limits()).unwrap();
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
        let mut layout = rgb_layout(1, 7);
        layout.extent.width = 0;
        assert!(matches!(
            VarDctOutputMemoryPlan::new(&layout),
            Err(VarDctOutputError::ImageOutput(
                jxl_wgpu::Error::ImageLayout(jxl_gpu_formats::LayoutError::EmptyExtent)
            ))
        ));
        let mut invalid = inverse_opsin();
        invalid.intensity_target = 0.0;
        assert!(matches!(
            validate_inverse_opsin(invalid).unwrap_err(),
            VarDctOutputError::InvalidIntensityTarget
        ));
        assert!(matches!(
            VarDctOutputPlan::for_limits_with_variant(
                &rgb_layout(1, 1),
                &wgpu::Limits::default(),
                KernelVariant::Tile8x8,
            )
            .unwrap_err(),
            VarDctOutputError::WorkgroupShape {
                variant: KernelVariant::Tile8x8,
            }
        ));
        invalid = inverse_opsin();
        invalid.inverse_opsin_matrix[2][1] = f32::NAN;
        assert!(matches!(
            validate_inverse_opsin(invalid).unwrap_err(),
            VarDctOutputError::NonFiniteParameter {
                field: "matrix[2][1]"
            }
        ));
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

    #[test]
    fn output_contract_rejects_inconsistent_layout_and_unmapped_hdr() {
        let config = VarDctOutputConfig {
            extent: Extent2d::new(5, 3),
            orientation: OutputOrientation::from_exif_value(6).unwrap(),
            transform: VarDctOutputTransform::Xyb(inverse_opsin()),
        };
        let layout = rgb_layout(3, 5);
        config.validate_layout(&layout).unwrap();
        assert!(matches!(
            config.validate_layout(&rgb_layout(5, 3)),
            Err(VarDctOutputError::ImageOutput(
                jxl_wgpu::Error::InvalidPayload(_)
            ))
        ));
        let mut wrong_size = layout.clone();
        wrong_size.logical_size -= 1;
        assert!(matches!(
            config.validate_layout(&wrong_size),
            Err(VarDctOutputError::ImageOutput(
                jxl_wgpu::Error::InvalidPayload(_)
            ))
        ));
        for transfer in [TransferFunction::Pq, TransferFunction::Hlg] {
            let mut hdr = layout.clone();
            let ColorSpecification::Defined(ref mut color) = hdr.format.color_spec else {
                unreachable!()
            };
            color.transfer = transfer;
            assert!(matches!(
                config.validate_layout(&hdr),
                Err(VarDctOutputError::HdrLuminanceMappingRequired)
            ));
        }
        let mut limited_rgb = layout;
        let ColorSpecification::Defined(ref mut color) = limited_rgb.format.color_spec else {
            unreachable!()
        };
        color.range = jxl_gpu_formats::ColorRange::Limited;
        assert!(matches!(
            config.validate_layout(&limited_rgb),
            Err(VarDctOutputError::ImageOutput(
                jxl_wgpu::Error::Unsupported(_)
            ))
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_packer_orients_one_pixel_axes_and_zeroes_tail_padding() {
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
            eprintln!("skipping VarDCT color packer GPU test: no adapter");
            return;
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("jxl-wgpu VarDCT color packer test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))
        else {
            eprintln!("skipping VarDCT color packer GPU test: device request failed");
            return;
        };
        let storage_plane = |label, samples: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(samples),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let x = storage_plane("VarDCT color test X", &[0.028_100_073, -0.015_386_105, 0.0]);
        let y = storage_plane(
            "VarDCT color test Y",
            &[0.488_188_2, 0.714_781_34, 0.278_128_2],
        );
        let b = storage_plane(
            "VarDCT color test B",
            &[0.471_659, 0.437_076_93, 0.666_139_84],
        );
        // Modular reconstruction is signed and can overshoot the declared sample range.
        // F32 output preserves it; integer output clamps during final quantization.
        let alpha = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VarDCT signed alpha with prefix"),
            contents: bytemuck::cast_slice(&[99_i32, 99, -7, 17, 40]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VarDCT color test output"),
            size: 128,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VarDCT color test staging"),
            size: 128,
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
        let packer = VarDctOutputPacker::new(&device).unwrap();
        for extent in [Extent2d::new(3, 1), Extent2d::new(1, 3)] {
            for value in 1..=8 {
                let orientation = OutputOrientation::from_exif_value(value).unwrap();
                for layout_kind in 0..5 {
                    let oriented = orientation.map_extent(extent);
                    let layout = if layout_kind == 0 {
                        rgb_layout(oriented.width, oriented.height)
                    } else {
                        let order = if layout_kind == 1 || layout_kind == 3 {
                            jxl_gpu_formats::RgbChannelOrder::Rgb
                        } else {
                            jxl_gpu_formats::RgbChannelOrder::Rgba
                        };
                        let constructor = if layout_kind >= 3 {
                            jxl_gpu_formats::PixelFormat::rgb_f32
                        } else {
                            jxl_gpu_formats::PixelFormat::rgb8
                        };
                        let format = constructor(
                            order,
                            true,
                            crate::vardct_engine::vardct_rgb8_format().color_spec,
                        );
                        let mut layout = ImageLayout::packed(oriented, format).unwrap();
                        let mut offset = 5;
                        for plane in &mut layout.planes {
                            plane.offset = offset;
                            plane.row_stride = plane.row_bytes + 3;
                            offset = plane.end_offset().unwrap();
                        }
                        ImageLayout::from_planes(oriented, layout.format, layout.planes).unwrap()
                    };
                    queue.write_buffer(&output, 0, &[0xa5; 128]);
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("VarDCT color test commands"),
                        });
                    let scratch = packer
                        .encode(
                            &device,
                            &mut encoder,
                            VarDctOutputInputs {
                                alpha: matches!(layout_kind, 2 | 4).then_some(VarDctOutputAlpha {
                                    domain: crate::ModularSampleDomain::SignedInteger,
                                    storage: binding(&alpha),
                                    width: extent.width,
                                    height: extent.height,
                                    stride: extent.width,
                                    word_offset: 2,
                                    bits_per_sample: 5,
                                }),
                                planes: [
                                    VarDctOutputPlane {
                                        storage: binding(&x),
                                        width: extent.width,
                                        height: extent.height,
                                        stride: extent.width,
                                    },
                                    VarDctOutputPlane {
                                        storage: binding(&y),
                                        width: extent.width,
                                        height: extent.height,
                                        stride: extent.width,
                                    },
                                    VarDctOutputPlane {
                                        storage: binding(&b),
                                        width: extent.width,
                                        height: extent.height,
                                        stride: extent.width,
                                    },
                                ],
                                output: binding(&output),
                                layout: &layout,
                                config: VarDctOutputConfig {
                                    extent,
                                    orientation,
                                    transform: VarDctOutputTransform::Xyb(inverse_opsin()),
                                },
                            },
                        )
                        .expect("record fused VarDCT color output");
                    assert_eq!(
                        scratch.plan.memory.logical_output_bytes,
                        layout.logical_size
                    );
                    assert_eq!(
                        scratch.plan.memory.output_storage_bytes,
                        layout.logical_size.div_ceil(4) * 4
                    );
                    assert_eq!(scratch.uniform.size(), 176);
                    assert_eq!(scratch.source_uniform.size(), 160);
                    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 128);
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
                        .expect("poll fused VarDCT color output");
                    receiver
                        .recv()
                        .expect("VarDCT color map callback")
                        .expect("map VarDCT color output");
                    let mapped = staging
                        .slice(..)
                        .get_mapped_range()
                        .expect("mapped VarDCT color output");
                    // These gold sequences use the physical row/column direction of each orientation,
                    // independently of the shader's inverse-coordinate mapping.
                    let reversed = if extent.width == 3 {
                        matches!(value, 2 | 3 | 7 | 8)
                    } else {
                        matches!(value, 3 | 4 | 6 | 7)
                    };
                    let expected = if reversed {
                        [0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 0, 0]
                    } else {
                        [255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0]
                    };
                    let mut stored = vec![0xa5; 128];
                    stored[..scratch.plan.memory.output_storage_bytes as usize].fill(0);
                    if layout_kind == 0 {
                        stored[..9].copy_from_slice(&expected[..9]);
                    } else {
                        for (channel, plane) in layout.planes.iter().enumerate() {
                            for pixel in 0..3 {
                                let x = pixel % oriented.width as usize;
                                let y = pixel / oriented.width as usize;
                                let sample_bytes = if layout_kind >= 3 { 4 } else { 1 };
                                let offset = plane.offset as usize
                                    + y * plane.row_stride as usize
                                    + x * sample_bytes;
                                let alpha_value = [-7.0_f32, 17.0, 40.0]
                                    [if reversed { 2 - pixel } else { pixel }]
                                    / 31.0;
                                let expected_code = if channel == 3 {
                                    (alpha_value.clamp(0.0, 1.0) * 255.0).round() as u8
                                } else {
                                    expected[pixel * 3 + channel]
                                };
                                if sample_bytes == 4 {
                                    let actual = f32::from_le_bytes(
                                        mapped[offset..offset + 4].try_into().unwrap(),
                                    );
                                    let expected = if channel == 3 {
                                        alpha_value
                                    } else {
                                        f32::from(expected_code) / 255.0
                                    };
                                    assert!(
                                        (actual - expected).abs() < 2e-5,
                                        "float primary sample {actual} != {expected}"
                                    );
                                    stored[offset..offset + 4]
                                        .copy_from_slice(&mapped[offset..offset + 4]);
                                } else {
                                    stored[offset] = expected_code;
                                }
                            }
                        }
                    }
                    assert_eq!(
                        &*mapped, stored,
                        "{extent:?}, orientation {value}, layout {layout_kind}"
                    );
                    drop(mapped);
                    staging.unmap();
                }
            }
        }
    }
}
