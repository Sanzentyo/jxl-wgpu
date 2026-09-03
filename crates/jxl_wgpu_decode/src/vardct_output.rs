//! Fused GPU-resident XYB/YCbCr reconstruction and packed 8-bit output.
//!
//! Supports RGB8, RGBA8, BGR8, and BGRA8 packings. For 3-byte formats (RGB8/BGR8), the kernel
//! assigns one invocation to each output `u32`. That ownership rule removes byte-level
//! read/modify/write races while retaining the externally visible tightly packed byte layout.
//! The allocation may contain up to three zero padding bytes after the logical image payload.
//! For 4-byte formats (RGBA8/BGRA8), each invocation writes one full word (alpha is filled
//! with 255 for opaque output when no decoded alpha source is connected).

use bytemuck::{Pod, Zeroable};
use jxl_gpu_formats::{
    ChromaLocation2d, ColorRange, ColorSpace, ColorSpec, ColorSpecification, PixelFormat,
    RgbChannelOrder, Swizzle, TransferFunction, YcbcrEncoding,
};
use jxl_gpu_protocol::XybParams;
use jxl_wgpu::{KernelVariant, ResidentStorageBinding};
use wgpu::util::DeviceExt;

use crate::vardct_frontend::VarDctChannelShift;

const OUTPUT_WORD_BYTES: u64 = std::mem::size_of::<u32>() as u64;
#[cfg(test)]
const WORKGROUP_SIZE: u32 = 256;
const DEFAULT_VARIANT: KernelVariant = KernelVariant::Lanes256;

/// WGSL source for the fused VarDCT output kernel.
pub const VAR_DCT_OUTPUT_SHADER: &str = include_str!("vardct_output.wgsl");

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

/// Color transform fused into the final packed 8-bit kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VarDctOutputTransform {
    /// JPEG XL XYB inverse followed by the sRGB transfer function.
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

/// Canonical packed 8-bit output format supported by the fast output packer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub(crate) enum PackedU8Format {
    Rgb = 0,
    Rgba = 1,
    Bgr = 2,
    Bgra = 3,
}

impl PackedU8Format {
    /// Number of bytes per pixel for this format.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba | Self::Bgra => 4,
        }
    }

    /// Canonical [`PixelFormat`] representation for this VarDCT packed format.
    #[must_use]
    pub fn pixel_format(self) -> PixelFormat {
        let order = match self {
            Self::Rgb => RgbChannelOrder::Rgb,
            Self::Rgba => RgbChannelOrder::Rgba,
            Self::Bgr => RgbChannelOrder::Bgr,
            Self::Bgra => RgbChannelOrder::Bgra,
        };
        PixelFormat::rgb8(
            order,
            false,
            ColorSpecification::Defined(ColorSpec {
                space: ColorSpace::Bt709,
                encoding: YcbcrEncoding::Undefined,
                transfer: TransferFunction::Srgb,
                range: ColorRange::Full,
                chroma_location: ChromaLocation2d::BOTH,
            }),
        )
    }

    /// Recognizes a canonical packed-u8 pixel format for the fast path, if supported.
    #[must_use]
    pub fn recognize(format: &PixelFormat) -> Option<Self> {
        Self::try_from(format).ok()
    }
}

impl TryFrom<&PixelFormat> for PackedU8Format {
    type Error = crate::vardct_engine::VarDctDecodeError;

    fn try_from(format: &PixelFormat) -> Result<Self, Self::Error> {
        let candidate = match format.swizzle {
            Swizzle::XYZ1 => Self::Rgb,
            Swizzle::XYZW => Self::Rgba,
            Swizzle::ZYX1 => Self::Bgr,
            Swizzle::ZYXW => Self::Bgra,
            _ => return Err(crate::vardct_engine::VarDctDecodeError::UnsupportedOutput),
        };

        if format == &candidate.pixel_format() {
            Ok(candidate)
        } else {
            Err(crate::vardct_engine::VarDctDecodeError::UnsupportedOutput)
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
    pub(crate) format: PackedU8Format,
    pub transform: VarDctOutputTransform,
}

/// GPU bindings consumed by [`VarDctOutputPacker::encode`].
#[derive(Clone, Copy, Debug)]
pub struct VarDctOutputInputs<'a> {
    /// X, Y, and B F32 planes, in that order.
    pub planes: [VarDctOutputPlane<'a>; 3],
    /// Packed output storage. For 3-byte formats (RGB8/BGR8), logical bytes are tightly
    /// interleaved and allocation length is rounded up to four bytes; for 4-byte formats
    /// (RGBA8/BGRA8), storage is word-aligned with opaque alpha (255).
    pub output: ResidentStorageBinding<'a>,
    /// Output geometry and inverse-opsin metadata.
    pub config: VarDctOutputConfig,
}

/// Exact byte counts for one fused output operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctOutputMemoryPlan {
    /// Color and packing format.
    pub(crate) format: PackedU8Format,
    /// Externally visible byte count.
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
    pub(crate) fn new(
        width: u32,
        height: u32,
        format: PackedU8Format,
    ) -> Result<Self, VarDctOutputError> {
        let (pixel_count, logical_output_bytes, output_storage_bytes) =
            packed_geometry(width, height, format)?;
        debug_assert!(pixel_count != 0);
        let uniform_bytes = std::mem::size_of::<VarDctOutputParams>() as u64;
        let transient_bytes = uniform_bytes;
        let total_bytes = output_storage_bytes.checked_add(transient_bytes).ok_or(
            VarDctOutputError::ArithmeticOverflow {
                field: "VarDCT output total bytes",
            },
        )?;
        Ok(Self {
            format,
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
    #[cfg(test)]
    pub(crate) fn for_limits(
        width: u32,
        height: u32,
        format: PackedU8Format,
        limits: &wgpu::Limits,
    ) -> Result<Self, VarDctOutputError> {
        Self::for_limits_with_variant(width, height, format, limits, DEFAULT_VARIANT)
    }

    /// Plans a dispatch using the selected 1D workgroup variant.
    pub(crate) fn for_limits_with_variant(
        width: u32,
        height: u32,
        format: PackedU8Format,
        limits: &wgpu::Limits,
        variant: KernelVariant,
    ) -> Result<Self, VarDctOutputError> {
        let memory = VarDctOutputMemoryPlan::new(width, height, format)?;
        validate_required_buffer("packed output", memory.output_storage_bytes, limits, true)?;
        if memory.uniform_bytes > limits.max_uniform_buffer_binding_size {
            return Err(VarDctOutputError::UniformBindingLimit {
                required: memory.uniform_bytes,
                available: limits.max_uniform_buffer_binding_size,
            });
        }
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        if workgroup_y != 1 {
            return Err(VarDctOutputError::WorkgroupShape { variant });
        }
        variant
            .validate_for("vardct_output", limits, 0)
            .map_err(|_| VarDctOutputError::WorkgroupSizeLimit {
                required: workgroup_x,
                max_invocations: limits.max_compute_invocations_per_workgroup,
                max_size_x: limits.max_compute_workgroup_size_x,
            })?;
        let output_words_u64 = match format {
            PackedU8Format::Rgba | PackedU8Format::Bgra => u64::from(width)
                .checked_mul(u64::from(height))
                .ok_or(VarDctOutputError::ArithmeticOverflow {
                    field: "output pixels",
                })?,
            PackedU8Format::Rgb | PackedU8Format::Bgr => {
                memory.output_storage_bytes / OUTPUT_WORD_BYTES
            }
        };
        let output_words =
            u32::try_from(output_words_u64).map_err(|_| VarDctOutputError::ShaderAddressSpace {
                field: "packed output words",
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
    /// The fixed 176-byte parameter buffer.
    pub uniform: wgpu::Buffer,
    /// Exact output/transient accounting and dispatch geometry.
    pub plan: VarDctOutputPlan,
}

/// Typed validation errors for GPU-resident VarDCT packed 8-bit output.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum VarDctOutputError {
    /// The logical image has a zero dimension.
    #[error("VarDCT packed output extent must be nonzero, got {width}x{height}")]
    EmptyExtent { width: u32, height: u32 },
    /// Checked size arithmetic overflowed.
    #[error("VarDCT packed output arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
    /// A value cannot be addressed by WGSL's `u32` indices.
    #[error(
        "VarDCT packed output {field} needs {required} addressable values, WGSL permits {available}"
    )]
    ShaderAddressSpace {
        field: &'static str,
        required: u64,
        available: u64,
    },
    /// One F32 plane has a row stride shorter than its width.
    #[error(
        "VarDCT packed output input plane {plane} stride {stride} is shorter than width {width}"
    )]
    InputStride {
        plane: usize,
        stride: u32,
        width: u32,
    },
    /// One input plane does not cover the component extent required by the color transform.
    #[error(
        "VarDCT packed output input plane {plane} extent {width}x{height} is smaller than required {required_width}x{required_height}"
    )]
    InputExtent {
        plane: usize,
        width: u32,
        height: u32,
        required_width: u32,
        required_height: u32,
    },
    /// JPEG component shifts are limited to the one-bit factors defined by the codestream.
    #[error(
        "VarDCT packed output JPEG channel {channel} has invalid shift {horizontal}x{vertical}"
    )]
    InvalidJpegShift {
        channel: usize,
        horizontal: u32,
        vertical: u32,
    },
    /// An inverse-opsin field is non-finite.
    #[error("VarDCT packed output inverse-opsin field {field} must be finite")]
    NonFiniteParameter { field: &'static str },
    /// The intensity target is finite but not positive.
    #[error("VarDCT packed output intensity target must be positive")]
    InvalidIntensityTarget,
    /// A buffer does not carry STORAGE usage.
    #[error("VarDCT packed output {role} buffer is missing STORAGE usage")]
    MissingStorageUsage { role: &'static str },
    /// A binding starts at an invalid device-specific offset.
    #[error("VarDCT packed output {role} offset {offset} is not aligned to {alignment}")]
    BindingOffsetAlignment {
        role: &'static str,
        offset: u64,
        alignment: u64,
    },
    /// A typed array binding does not end at a whole 32-bit word.
    #[error("VarDCT packed output {role} binding size {size} is not four-byte aligned")]
    BindingSizeAlignment { role: &'static str, size: u64 },
    /// A subrange exceeds its backing buffer.
    #[error("VarDCT packed output {role} range {offset}..{end} exceeds buffer size {available}")]
    BindingRange {
        role: &'static str,
        offset: u64,
        end: u64,
        available: u64,
    },
    /// A subrange is smaller than its image geometry requires.
    #[error("VarDCT packed output {role} binding needs {required} bytes, has {available}")]
    BindingSize {
        role: &'static str,
        required: u64,
        available: u64,
    },
    /// A required allocation exceeds a device buffer limit.
    #[error(
        "VarDCT packed output {role} needs {required} bytes, device buffer limit is {available}"
    )]
    BufferLimit {
        role: &'static str,
        required: u64,
        available: u64,
    },
    /// A required storage binding exceeds the device binding limit.
    #[error(
        "VarDCT packed output {role} needs {required} bytes, storage binding limit is {available}"
    )]
    StorageBindingLimit {
        role: &'static str,
        required: u64,
        available: u64,
    },
    /// The 176-byte uniform exceeds an unusual device limit.
    #[error(
        "VarDCT packed output uniform needs {required} bytes, uniform binding limit is {available}"
    )]
    UniformBindingLimit { required: u64, available: u64 },
    /// Output packing requires a one-dimensional workgroup.
    #[error("VarDCT packed output requires a linear workgroup, got {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    /// The selected output workgroup cannot run on the device.
    #[error(
        "VarDCT packed output workgroup needs {required} X invocations, device permits {max_invocations} total and {max_size_x} in X"
    )]
    WorkgroupSizeLimit {
        required: u32,
        max_invocations: u32,
        max_size_x: u32,
    },
    /// A two-dimensional linearization still exceeds the device's Y limit.
    #[error(
        "VarDCT packed output dispatch needs {required_y} Y workgroups, device permits {available}"
    )]
    DispatchLimit { required_y: u32, available: u32 },
}

/// Reusable fused XYB/YCbCr-to-packed-RGB8 compute pipeline.
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
            label: Some("jxl-wgpu decode VarDCT packed output"),
            source: wgpu::ShaderSource::Wgsl(VAR_DCT_OUTPUT_SHADER.into()),
        });
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        let constants = [
            ("wg_x", f64::from(workgroup_x)),
            ("wg_y", f64::from(workgroup_y)),
        ];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu decode VarDCT packed output"),
            layout: None,
            module: &module,
            entry_point: Some("pack_packed_u8"),
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
            label: Some("jxl-wgpu decode VarDCT packed u8 params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu decode VarDCT packed u8 bindings"),
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
            label: Some("jxl-wgpu decode VarDCT packed u8"),
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
    image: [u32; 4],
    dispatch: [u32; 4],
    plane_geometry: [[u32; 4]; 3],
    matrix_r: [f32; 4],
    matrix_g: [f32; 4],
    matrix_b: [f32; 4],
    bias_cbrt: [f32; 4],
    scaled_bias: [f32; 4],
    intensity_scale: f32,
    format_selector: u32,
    _padding: [u32; 2],
}

fn validate_inputs(
    device: &wgpu::Device,
    inputs: VarDctOutputInputs<'_>,
    variant: KernelVariant,
) -> Result<(VarDctOutputParams, VarDctOutputPlan), VarDctOutputError> {
    if let VarDctOutputTransform::Xyb(inverse) = inputs.config.transform {
        validate_inverse_opsin(inverse)?;
    }
    let plan = VarDctOutputPlan::for_limits_with_variant(
        inputs.config.width,
        inputs.config.height,
        inputs.config.format,
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

    let required_extents = match inputs.config.transform {
        VarDctOutputTransform::Xyb(_) => [[inputs.config.width, inputs.config.height]; 3],
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
                    inputs.config.width.div_ceil(1 << shift.horizontal),
                    inputs.config.height.div_ceil(1 << shift.vertical),
                ];
            }
            extents
        }
    };
    let mut plane_geometry = [[0; 4]; 3];
    for (plane, input) in inputs.planes.into_iter().enumerate() {
        let [required_width, required_height] = required_extents[plane];
        if input.width != required_width || input.height != required_height {
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
        "packed output",
        inputs.output,
        plan.memory.output_storage_bytes,
    )?;

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
        VarDctOutputParams {
            image: [
                inputs.config.width,
                inputs.config.height,
                pixel_count,
                plan.output_words,
            ],
            dispatch: [plan.dispatch_width, mode, 0, 0],
            plane_geometry,
            matrix_r: matrix_row(matrix[0]),
            matrix_g: matrix_row(matrix[1]),
            matrix_b: matrix_row(matrix[2]),
            bias_cbrt: [bias_cbrt[0], bias_cbrt[1], bias_cbrt[2], 0.0],
            scaled_bias: [scaled_bias[0], scaled_bias[1], scaled_bias[2], 0.0],
            intensity_scale,
            format_selector: inputs.config.format as u32,
            _padding: [0; 2],
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

fn packed_geometry(
    width: u32,
    height: u32,
    format: PackedU8Format,
) -> Result<(u32, u64, u64), VarDctOutputError> {
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
    let bpp = u64::from(format.bytes_per_pixel());
    let logical_output_bytes =
        pixel_count_u64
            .checked_mul(bpp)
            .ok_or(VarDctOutputError::ArithmeticOverflow {
                field: "logical output bytes",
            })?;
    if logical_output_bytes > u64::from(u32::MAX) {
        return Err(VarDctOutputError::ShaderAddressSpace {
            field: "logical output bytes",
            required: logical_output_bytes,
            available: u64::from(u32::MAX),
        });
    }
    let output_storage_bytes = logical_output_bytes
        .checked_add(OUTPUT_WORD_BYTES - 1)
        .map(|value| value / OUTPUT_WORD_BYTES * OUTPUT_WORD_BYTES)
        .ok_or(VarDctOutputError::ArithmeticOverflow {
            field: "aligned output bytes",
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
    assert!(std::mem::size_of::<VarDctOutputParams>() == 176);
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
        assert_eq!(std::mem::size_of::<VarDctOutputParams>(), 176);
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
        let memory = VarDctOutputMemoryPlan::new(5, 3, PackedU8Format::Rgb).unwrap();
        assert_eq!(memory.logical_output_bytes, 45);
        assert_eq!(memory.output_storage_bytes, 48);
        assert_eq!(memory.uniform_bytes, 176);
        assert_eq!(memory.transient_bytes, 176);
        assert_eq!(memory.total_bytes, 224);

        let rgba_memory = VarDctOutputMemoryPlan::new(5, 3, PackedU8Format::Rgba).unwrap();
        assert_eq!(rgba_memory.logical_output_bytes, 60);
        assert_eq!(rgba_memory.output_storage_bytes, 60);
        assert_eq!(rgba_memory.total_bytes, 236);

        let plan =
            VarDctOutputPlan::for_limits(5, 3, PackedU8Format::Rgb, &generous_limits()).unwrap();
        assert_eq!(plan.output_words, 12);
        assert_eq!((plan.workgroups_x, plan.workgroups_y), (1, 1));
        assert_eq!(plan.dispatch_width, WORKGROUP_SIZE);

        let rgba_plan =
            VarDctOutputPlan::for_limits(5, 3, PackedU8Format::Rgba, &generous_limits()).unwrap();
        assert_eq!(rgba_plan.output_words, 15);
    }

    #[test]
    fn sixteen_k_output_is_split_across_dispatch_rows() {
        let plan =
            VarDctOutputPlan::for_limits(16_384, 16_384, PackedU8Format::Rgb, &generous_limits())
                .unwrap();
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
            VarDctOutputMemoryPlan::new(0, 7, PackedU8Format::Rgb).unwrap_err(),
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
                PackedU8Format::Rgb,
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
    fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jxl-wgpu VarDCT output packer test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_packer_owns_words_and_zeroes_tail_padding() {
        use std::num::NonZeroU64;
        use std::sync::mpsc;

        let Some((device, queue)) = test_device() else {
            eprintln!("skipping VarDCT RGB8 packer GPU test: device unavailable");
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
                            width: 3,
                            height: 1,
                            stride: 3,
                        },
                        VarDctOutputPlane {
                            storage: binding(&y),
                            width: 3,
                            height: 1,
                            stride: 3,
                        },
                        VarDctOutputPlane {
                            storage: binding(&b),
                            width: 3,
                            height: 1,
                            stride: 3,
                        },
                    ],
                    output: binding(&output),
                    config: VarDctOutputConfig {
                        width: 3,
                        height: 1,
                        format: PackedU8Format::Rgb,
                        transform: VarDctOutputTransform::Xyb(inverse_opsin()),
                    },
                },
            )
            .expect("record fused VarDCT RGB8 output");
        assert_eq!(scratch.plan.memory.logical_output_bytes, 9);
        assert_eq!(scratch.plan.memory.output_storage_bytes, 12);
        assert_eq!(scratch.uniform.size(), 176);
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

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn fused_vardct_output_packer_matches_scalar_oracle_across_all_formats_and_dimensions() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let packer = VarDctOutputPacker::new(&device).unwrap();

        // Scalar oracle for YCbCr 4:4:4 identity-scale color transform and clamping
        fn scalar_pack(
            width: u32,
            height: u32,
            y_plane: &[f32],
            cb_plane: &[f32],
            cr_plane: &[f32],
            format: PackedU8Format,
        ) -> Vec<u8> {
            let capacity =
                (width as usize) * (height as usize) * (format.bytes_per_pixel() as usize);
            let mut out = Vec::with_capacity(capacity);
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) as usize;
                    let luma = y_plane[idx] + 128.0 / 255.0;
                    let cb = cb_plane[idx];
                    let cr = cr_plane[idx];
                    // Standard JPEG YCbCr to linear RGB in WGSL
                    let r = luma + 1.402 * cr;
                    let g = luma - (0.114 * 1.772 / 0.587) * cb - (0.299 * 1.402 / 0.587) * cr;
                    let b = luma + 1.772 * cb;
                    let r_u8 = (r * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8;
                    let g_u8 = (g * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8;
                    let b_u8 = (b * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8;
                    match format {
                        PackedU8Format::Rgb => {
                            out.extend_from_slice(&[r_u8, g_u8, b_u8]);
                        }
                        PackedU8Format::Rgba => {
                            out.extend_from_slice(&[r_u8, g_u8, b_u8, 255]);
                        }
                        PackedU8Format::Bgr => {
                            out.extend_from_slice(&[b_u8, g_u8, r_u8]);
                        }
                        PackedU8Format::Bgra => {
                            out.extend_from_slice(&[b_u8, g_u8, r_u8, 255]);
                        }
                    }
                }
            }
            out
        }

        let test_dimensions = [
            (1_u32, 1_u32), // 1 pixel
            (2_u32, 1_u32), // 2 pixels: 6 bytes RGB (non-multiple of 4)
            (5_u32, 3_u32), // 15 pixels: odd width, non-multiple rows
            (8_u32, 4_u32), // 32 pixels
        ];

        let formats = [
            PackedU8Format::Rgb,
            PackedU8Format::Rgba,
            PackedU8Format::Bgr,
            PackedU8Format::Bgra,
        ];

        use std::num::NonZeroU64;
        use std::sync::mpsc;

        for &(width, height) in &test_dimensions {
            let pixel_count = (width * height) as usize;
            // Samples including clamp edges: negative values and > 1.0 values
            let y_data: Vec<f32> = (0..pixel_count)
                .map(|i| {
                    if i % 4 == 0 {
                        -0.2
                    } else if i % 4 == 1 {
                        1.3
                    } else {
                        (i as f32) / (pixel_count as f32)
                    }
                })
                .collect();
            let cb_data: Vec<f32> = (0..pixel_count).map(|i| (i as f32 * 0.1) - 0.05).collect();
            let cr_data: Vec<f32> = (0..pixel_count).map(|i| 0.05 - (i as f32 * 0.08)).collect();

            let y_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test Y"),
                contents: bytemuck::cast_slice(&y_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let cb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test Cb"),
                contents: bytemuck::cast_slice(&cb_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let cr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test Cr"),
                contents: bytemuck::cast_slice(&cr_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

            fn binding(buffer: &wgpu::Buffer) -> ResidentStorageBinding<'_> {
                ResidentStorageBinding {
                    buffer,
                    offset: 0,
                    size: NonZeroU64::new(buffer.size()).unwrap(),
                }
            }

            for &format in &formats {
                let bytes_per_px = format.bytes_per_pixel() as u64;
                let logical_bytes = (pixel_count as u64) * bytes_per_px;
                let storage_bytes = logical_bytes.div_ceil(4) * 4;

                let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("test out"),
                    size: storage_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("test staging"),
                    size: storage_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                packer
                    .encode(
                        &device,
                        &mut encoder,
                        VarDctOutputInputs {
                            planes: [
                                VarDctOutputPlane {
                                    storage: binding(&cb_buf),
                                    width,
                                    height,
                                    stride: width,
                                },
                                VarDctOutputPlane {
                                    storage: binding(&y_buf),
                                    width,
                                    height,
                                    stride: width,
                                },
                                VarDctOutputPlane {
                                    storage: binding(&cr_buf),
                                    width,
                                    height,
                                    stride: width,
                                },
                            ],
                            output: binding(&output_buf),
                            config: VarDctOutputConfig {
                                width,
                                height,
                                format,
                                transform: VarDctOutputTransform::Ycbcr {
                                    channel_shifts: [VarDctChannelShift::default(); 3],
                                },
                            },
                        },
                    )
                    .unwrap();

                encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, storage_bytes);
                let submission = queue.submit([encoder.finish()]);

                let (sender, receiver) = mpsc::sync_channel(1);
                staging
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |res| {
                        let _ = sender.send(res);
                    });
                device
                    .poll(wgpu::PollType::Wait {
                        submission_index: Some(submission),
                        timeout: None,
                    })
                    .unwrap();
                receiver.recv().unwrap().unwrap();

                let view = staging.slice(..).get_mapped_range().expect("mapped output");
                let actual = &(&*view)[..logical_bytes as usize];
                let expected = scalar_pack(width, height, &y_data, &cb_data, &cr_data, format);

                for (idx, (&act, &exp)) in actual.iter().zip(&expected).enumerate() {
                    let diff = (act as i16 - exp as i16).abs();
                    assert!(
                        diff <= 1,
                        "mismatch at byte {idx} for geom {width}x{height} format {format:?}: actual={act}, expected={exp}"
                    );
                }
            }
        }
    }

    #[test]
    fn packed_u8_format_strictly_rejects_non_canonical_descriptors() {
        use jxl_gpu_formats::{
            Channel, ColorRange, ColorSpace, ColorSpecification, PackingField, PackingWord,
            TransferFunction,
        };

        let canonical_rgb = PackedU8Format::Rgb.pixel_format();
        assert_eq!(
            PackedU8Format::try_from(&canonical_rgb).unwrap(),
            PackedU8Format::Rgb
        );

        let canonical_rgba = PackedU8Format::Rgba.pixel_format();
        assert_eq!(
            PackedU8Format::try_from(&canonical_rgba).unwrap(),
            PackedU8Format::Rgba
        );

        let canonical_bgr = PackedU8Format::Bgr.pixel_format();
        assert_eq!(
            PackedU8Format::try_from(&canonical_bgr).unwrap(),
            PackedU8Format::Bgr
        );

        let canonical_bgra = PackedU8Format::Bgra.pixel_format();
        assert_eq!(
            PackedU8Format::try_from(&canonical_bgra).unwrap(),
            PackedU8Format::Bgra
        );

        // 1. Display-P3 RGB8
        let p3 = PixelFormat::rgb8(
            RgbChannelOrder::Rgb,
            false,
            ColorSpecification::Defined(ColorSpec {
                space: ColorSpace::DisplayP3,
                encoding: YcbcrEncoding::Undefined,
                transfer: TransferFunction::Srgb,
                range: ColorRange::Full,
                chroma_location: ChromaLocation2d::BOTH,
            }),
        );
        assert!(PackedU8Format::try_from(&p3).is_err());

        // 2. BT.2020 RGB8
        let bt2020 = PixelFormat::rgb8(
            RgbChannelOrder::Rgb,
            false,
            ColorSpecification::Defined(ColorSpec {
                space: ColorSpace::Bt2020,
                encoding: YcbcrEncoding::Undefined,
                transfer: TransferFunction::Srgb,
                range: ColorRange::Full,
                chroma_location: ChromaLocation2d::BOTH,
            }),
        );
        assert!(PackedU8Format::try_from(&bt2020).is_err());

        // 3. Linear RGB8
        let linear = PixelFormat::rgb8(
            RgbChannelOrder::Rgb,
            false,
            ColorSpecification::Defined(ColorSpec {
                space: ColorSpace::Bt709,
                encoding: YcbcrEncoding::Undefined,
                transfer: TransferFunction::Linear,
                range: ColorRange::Full,
                chroma_location: ChromaLocation2d::BOTH,
            }),
        );
        assert!(PackedU8Format::try_from(&linear).is_err());

        // 4. Limited-range RGB8
        let limited = PixelFormat::rgb8(
            RgbChannelOrder::Rgb,
            false,
            ColorSpecification::Defined(ColorSpec {
                space: ColorSpace::Bt709,
                encoding: YcbcrEncoding::Undefined,
                transfer: TransferFunction::Srgb,
                range: ColorRange::Limited,
                chroma_location: ChromaLocation2d::BOTH,
            }),
        );
        assert!(PackedU8Format::try_from(&limited).is_err());

        // 5. ColorSpecification::Default
        let mut default_spec = canonical_rgb.clone();
        default_spec.color_spec = ColorSpecification::Default;
        assert!(PackedU8Format::try_from(&default_spec).is_err());

        // 6. ColorSpecification::Undefined
        let mut undefined_spec = canonical_rgb.clone();
        undefined_spec.color_spec = ColorSpecification::Undefined;
        assert!(PackedU8Format::try_from(&undefined_spec).is_err());

        // 7. Planar RGB8
        let planar = PixelFormat::rgb8(
            RgbChannelOrder::Rgb,
            true,
            ColorSpecification::Defined(ColorSpec {
                space: ColorSpace::Bt709,
                encoding: YcbcrEncoding::Undefined,
                transfer: TransferFunction::Srgb,
                range: ColorRange::Full,
                chroma_location: ChromaLocation2d::BOTH,
            }),
        );
        assert!(PackedU8Format::try_from(&planar).is_err());

        // 8. Channel field is X/X/X in 3-word format
        let mut xxx = canonical_rgb.clone();
        xxx.planes[0].words = vec![
            PackingWord::channel(Channel::X, 8),
            PackingWord::channel(Channel::X, 8),
            PackingWord::channel(Channel::X, 8),
        ];
        assert!(PackedU8Format::try_from(&xxx).is_err());

        // 9. Padding in 8-bit word
        let mut padding = canonical_rgb.clone();
        padding.planes[0].words = vec![
            PackingWord::channel(Channel::X, 8),
            PackingWord::channel(Channel::Y, 8),
            PackingWord {
                fields: vec![PackingField::padding(8)],
            },
        ];
        assert!(PackedU8Format::try_from(&padding).is_err());

        // 10. Channel order in words does not match swizzle (e.g. ZYX words with XYZ1 swizzle)
        let mut mismatched_words = canonical_rgb.clone();
        mismatched_words.planes[0].words = vec![
            PackingWord::channel(Channel::Z, 8),
            PackingWord::channel(Channel::Y, 8),
            PackingWord::channel(Channel::X, 8),
        ];
        assert!(PackedU8Format::try_from(&mismatched_words).is_err());
    }
}
