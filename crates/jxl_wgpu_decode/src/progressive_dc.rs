//! GPU-resident conversion of a Modular progressive-DC frame into VarDCT LF resources.
//!
//! Modular final planes are stored as signed i32 words in `[Y, X, B]` order.  Progressive-DC
//! VarDCT consumes floating-point XYB values in `[X, Y, B]` order, and its LF resource table is
//! an array of `[X, Y, B, 0]` vec4 values.  This module keeps both operations on the GPU and owns
//! the intermediate planes so they can be retained across command submissions.

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use jxl_wgpu::{KernelPolicy, KernelVariant, ResidentStorageBinding};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::modular_transform::GpuModularChannelLayout;

const PROGRESSIVE_DC_SHADER: &str = include_str!("progressive_dc.wgsl");
const F32_BYTES: u64 = std::mem::size_of::<f32>() as u64;
const RESOURCE_VEC4_BYTES: u64 = std::mem::size_of::<[f32; 4]>() as u64;

/// Stable kernel-policy key for the conversion and LF-resource packing passes.
pub(crate) const PROGRESSIVE_DC_KERNEL_KEY: &str = "progressive_dc";

/// Built-in linear variant used when the adapter policy has no tuned entry.
pub(crate) const DEFAULT_PROGRESSIVE_DC_VARIANT: KernelVariant = KernelVariant::Lanes64;

/// One owned GPU-resident F32 XYB plane.
#[derive(Clone, Debug)]
pub(crate) struct ProgressiveDcXybPlane {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) stride: u32,
}

impl ProgressiveDcXybPlane {
    /// Returns the exclusive scalar count needed by this plane.
    pub(crate) fn required_scalars(&self) -> Result<u64, ProgressiveDcGpuError> {
        required_plane_scalars(self.width, self.height, self.stride, "XYB plane scalars")
    }
}

/// Three owned GPU-resident planar F32 buffers in `[X, Y, B]` order.
///
/// `wgpu::Buffer` handles are cloneable references to the same GPU allocation, so cloning this
/// value is inexpensive and preserves the ability to retain the planes across submissions.
#[derive(Clone, Debug)]
pub(crate) struct ProgressiveDcXybPlanes {
    pub(crate) planes: [ProgressiveDcXybPlane; 3],
}

impl ProgressiveDcXybPlanes {
    /// Allocates three storage-capable XYB planes with one common geometry.
    ///
    /// A zero stride selects a tightly packed row stride.  The actual stride is retained in the
    /// owned representation so later consumers can use the buffers without reconstructing it.
    pub(crate) fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        stride: u32,
    ) -> Result<Self, ProgressiveDcGpuError> {
        let stride = normalized_stride(width, height, stride, "XYB output")?;
        let required_scalars = required_plane_scalars(width, height, stride, "XYB output scalars")?;
        let required_bytes = required_scalars.checked_mul(F32_BYTES).ok_or(
            ProgressiveDcGpuError::ArithmeticOverflow {
                field: "XYB output bytes",
            },
        )?;
        validate_allocated_buffer_size(device, "XYB output", required_bytes)?;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let labels = [
            "jxl-wgpu progressive-DC X plane",
            "jxl-wgpu progressive-DC Y plane",
            "jxl-wgpu progressive-DC B plane",
        ];
        let planes = std::array::from_fn(|index| ProgressiveDcXybPlane {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(labels[index]),
                size: required_bytes,
                usage,
                mapped_at_creation: false,
            }),
            width,
            height,
            stride,
        });
        Ok(Self { planes })
    }

    /// Wraps three already-created storage buffers in the owned XYB representation.
    ///
    /// Buffer usage, range, and device-limit checks are deferred to [`ProgressiveDcPipeline`] so
    /// this constructor can remain independent of a device handle while still validating all
    /// geometry and arithmetic that is intrinsic to the representation.
    pub(crate) fn from_buffers(
        buffers: [wgpu::Buffer; 3],
        width: u32,
        height: u32,
        stride: u32,
    ) -> Result<Self, ProgressiveDcGpuError> {
        let stride = normalized_stride(width, height, stride, "XYB output")?;
        Ok(Self {
            planes: buffers.map(|buffer| ProgressiveDcXybPlane {
                buffer,
                width,
                height,
                stride,
            }),
        })
    }

    /// Returns the common plane width.
    #[must_use]
    pub(crate) const fn width(&self) -> u32 {
        self.planes[0].width
    }

    /// Returns the common plane height.
    #[must_use]
    pub(crate) const fn height(&self) -> u32 {
        self.planes[0].height
    }
}

/// Inputs to the Modular-i32 to progressive-DC XYB conversion pass.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ProgressiveDcConvertInputs<'a> {
    /// Resident Modular frame arena containing the source words.
    pub(crate) arena: ResidentStorageBinding<'a>,
    /// Final source planes in the fixed `[Y, X, B]` order.
    pub(crate) source_planes: [GpuModularChannelLayout; 3],
    /// Owned `[X, Y, B]` F32 destination planes.
    pub(crate) outputs: &'a ProgressiveDcXybPlanes,
    /// JPEG XL LF multipliers in `[X, Y, B]` order.  The shader divides each by 128.
    pub(crate) multipliers: [f32; 3],
}

/// Inputs to the progressive-DC LF resource packing pass.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ProgressiveDcPackInputs<'a> {
    /// Resident `[X, Y, B]` dependency planes to pack.
    pub(crate) planes: &'a ProgressiveDcXybPlanes,
    /// Existing VarDCT resource table, interpreted as `array<vec4<f32>>`.
    pub(crate) resources: ResidentStorageBinding<'a>,
    /// Destination LF index in vec4 elements, relative to `resources`.
    pub(crate) lf_offset: u32,
    /// Destination LF row stride in vec4 elements.
    pub(crate) lf_stride: u32,
}

/// Exact host-shareable uniform for [`convert_modular`](ProgressiveDcPipeline::encode_convert).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ProgressiveDcConvertParams {
    /// Width, height, pixel count, and a reserved zero word.
    pub(crate) geometry: [u32; 4],
    /// Word offset, row stride, width, and height for source Y.
    pub(crate) source_y: [u32; 4],
    /// Word offset, row stride, width, and height for source X.
    pub(crate) source_x: [u32; 4],
    /// Word offset, row stride, width, and height for source B.
    pub(crate) source_b: [u32; 4],
    /// F32 output row strides for X, Y, and B, plus a reserved zero word.
    pub(crate) output_strides: [u32; 4],
    /// Multipliers for X, Y, and B, plus a reserved zero word.
    pub(crate) multipliers: [f32; 4],
}

/// Exact host-shareable uniform for [`pack_lf`](ProgressiveDcPipeline::encode_pack).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ProgressiveDcPackParams {
    /// Width, height, pixel count, and a reserved zero word.
    pub(crate) geometry: [u32; 4],
    /// F32 input row strides for X, Y, and B, plus a reserved zero word.
    pub(crate) input_strides: [u32; 4],
    /// LF vec4 offset, LF vec4 row stride, and two reserved zero words.
    pub(crate) destination: [u32; 4],
}

/// Typed errors raised before a progressive-DC dispatch is recorded.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ProgressiveDcGpuError {
    #[error("progressive-DC {role} has an empty {axis} extent")]
    EmptyExtent {
        role: &'static str,
        axis: &'static str,
    },
    #[error("progressive-DC {role} stride {stride} is smaller than width {width}")]
    InvalidStride {
        role: &'static str,
        stride: u32,
        width: u32,
    },
    #[error(
        "progressive-DC plane {plane} has geometry {actual_width}x{actual_height}, expected {expected_width}x{expected_height}"
    )]
    PlaneExtent {
        plane: usize,
        actual_width: u32,
        actual_height: u32,
        expected_width: u32,
        expected_height: u32,
    },
    #[error("progressive-DC source plane {plane} has invalid final-plane metadata: {reason}")]
    InvalidSourceLayout { plane: usize, reason: &'static str },
    #[error("progressive-DC source plane {plane} address range exceeds WGSL's u32 word space")]
    SourceAddressSpace { plane: usize },
    #[error(
        "progressive-DC source plane {plane} needs {required} bytes, arena binding has {available}"
    )]
    SourceBindingSize {
        plane: usize,
        required: u64,
        available: u64,
    },
    #[error("progressive-DC {role} buffer is empty")]
    EmptyBuffer { role: &'static str },
    #[error("progressive-DC {role} buffer is missing STORAGE usage")]
    MissingStorageUsage { role: &'static str },
    #[error("progressive-DC {role} binding offset {offset} is not aligned to {alignment}")]
    BindingAlignment {
        role: &'static str,
        offset: u64,
        alignment: u64,
    },
    #[error("progressive-DC {role} binding range {offset}..{end} exceeds buffer size {available}")]
    BindingRange {
        role: &'static str,
        offset: u64,
        end: u64,
        available: u64,
    },
    #[error("progressive-DC {role} binding size {size} is not aligned to {alignment} bytes")]
    BindingSizeAlignment {
        role: &'static str,
        size: u64,
        alignment: u64,
    },
    #[error("progressive-DC {role} needs {required} bytes, binding has {available}")]
    BindingSize {
        role: &'static str,
        required: u64,
        available: u64,
    },
    #[error("progressive-DC {role} binding needs {required} bytes, device permits {available}")]
    StorageBindingLimit {
        role: &'static str,
        required: u64,
        available: u64,
    },
    #[error("progressive-DC {role} needs {required} bytes, device permits {available}")]
    BufferLimit {
        role: &'static str,
        required: u64,
        available: u64,
    },
    #[error("progressive-DC uniform needs {required} bytes, device permits {available}")]
    UniformBindingLimit { required: u64, available: u64 },
    #[error("progressive-DC resource LF stride {stride} is smaller than width {width}")]
    InvalidLfStride { stride: u32, width: u32 },
    #[error("progressive-DC LF resource address range exceeds WGSL's u32 vec4 index space")]
    ResourceAddressSpace,
    #[error("progressive-DC multiplier for channel {channel} is not finite")]
    NonFiniteMultiplier { channel: usize },
    #[error("progressive-DC arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
    #[error("progressive-DC requires a linear workgroup, got {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    #[error("progressive-DC workgroup variant {variant:?} exceeds device limits")]
    WorkgroupVariant { variant: KernelVariant },
    #[error("progressive-DC dispatch needs {required} workgroups, device permits {available}")]
    WorkgroupCount { required: u32, available: u32 },
    #[error("progressive-DC kernel policy failed: {0}")]
    KernelPolicy(String),
}

/// Reusable conversion and LF-resource packing pipelines for progressive-DC frames.
pub(crate) struct ProgressiveDcPipeline {
    convert: wgpu::ComputePipeline,
    pack: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl ProgressiveDcPipeline {
    /// Compiles both entry points with a selected linear [`KernelVariant`].
    pub(crate) fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, ProgressiveDcGpuError> {
        validate_variant(variant, &device.limits())?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu progressive-DC conversion and LF packing"),
            source: wgpu::ShaderSource::Wgsl(PROGRESSIVE_DC_SHADER.into()),
        });
        let constants = [("wg_x", f64::from(variant.workgroup_size().0))];
        let make_pipeline = |label, entry_point| {
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
            convert: make_pipeline("jxl-wgpu progressive-DC Modular to XYB", "convert_modular"),
            pack: make_pipeline("jxl-wgpu progressive-DC LF resource packing", "pack_lf"),
            variant,
        })
    }

    /// Selects a linear variant using the shared adapter policy.
    pub(crate) fn with_policy(
        device: &wgpu::Device,
        policy: &KernelPolicy,
    ) -> Result<Self, ProgressiveDcGpuError> {
        let variant = policy
            .variant_for(PROGRESSIVE_DC_KERNEL_KEY, DEFAULT_PROGRESSIVE_DC_VARIANT)
            .map_err(|error| ProgressiveDcGpuError::KernelPolicy(error.to_string()))?;
        Self::with_variant(device, variant)
    }

    /// Records conversion of final Modular i32 planes into owned F32 XYB planes.
    ///
    /// Source planes are read in `[Y, X, B]` order.  The B source is first added to Y with i32
    /// saturation, then the output planes are scaled in `[X, Y, B]` order by `m / 128`.
    pub(crate) fn encode_convert(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: ProgressiveDcConvertInputs<'_>,
    ) -> Result<wgpu::Buffer, ProgressiveDcGpuError> {
        let params = validate_convert_inputs(device, inputs, self.variant)?;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu progressive-DC conversion params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let output_bindings = inputs
            .outputs
            .planes
            .iter()
            .map(|plane| entire_storage_binding(&plane.buffer))
            .collect::<Result<Vec<_>, _>>()?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu progressive-DC conversion bindings"),
            layout: &self.convert.get_bind_group_layout(0),
            entries: &[
                storage_entry(0, inputs.arena),
                storage_entry(1, output_bindings[0]),
                storage_entry(2, output_bindings[1]),
                storage_entry(3, output_bindings[2]),
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu progressive-DC Modular to XYB"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.convert);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            params.geometry[2].div_ceil(self.variant.workgroup_size().0),
            1,
            1,
        );
        drop(pass);
        Ok(uniform)
    }

    /// Records packing of XYB dependency planes into an existing VarDCT LF resource table.
    ///
    /// The destination is addressed as vec4 elements and receives the renderer convention
    /// `[X, Y, B, 0]` at `lf_offset + y * lf_stride + x`.
    pub(crate) fn encode_pack(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: ProgressiveDcPackInputs<'_>,
    ) -> Result<wgpu::Buffer, ProgressiveDcGpuError> {
        let params = validate_pack_inputs(device, inputs, self.variant)?;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu progressive-DC LF packing params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let input_bindings = inputs
            .planes
            .planes
            .iter()
            .map(|plane| entire_storage_binding(&plane.buffer))
            .collect::<Result<Vec<_>, _>>()?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu progressive-DC LF packing bindings"),
            layout: &self.pack.get_bind_group_layout(0),
            entries: &[
                storage_entry(5, input_bindings[0]),
                storage_entry(6, input_bindings[1]),
                storage_entry(7, input_bindings[2]),
                storage_entry(8, inputs.resources),
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu progressive-DC LF resource packing"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pack);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            params.geometry[2].div_ceil(self.variant.workgroup_size().0),
            1,
            1,
        );
        drop(pass);
        Ok(uniform)
    }
}

fn validate_variant(
    variant: KernelVariant,
    limits: &wgpu::Limits,
) -> Result<(), ProgressiveDcGpuError> {
    if !variant.is_linear() {
        return Err(ProgressiveDcGpuError::WorkgroupShape { variant });
    }
    variant
        .validate_for(PROGRESSIVE_DC_KERNEL_KEY, limits, 0)
        .map_err(|_| ProgressiveDcGpuError::WorkgroupVariant { variant })
}

fn validate_convert_inputs(
    device: &wgpu::Device,
    inputs: ProgressiveDcConvertInputs<'_>,
    variant: KernelVariant,
) -> Result<ProgressiveDcConvertParams, ProgressiveDcGpuError> {
    validate_variant(variant, &device.limits())?;
    let source = validate_source_planes(inputs.source_planes)?;
    validate_storage_binding(
        device,
        "Modular frame arena",
        inputs.arena,
        source.required_bytes,
        4,
    )?;
    let output_strides = validate_xyb_outputs(device, inputs.outputs, source.width, source.height)?;
    validate_multipliers(inputs.multipliers)?;
    validate_uniform_limit::<ProgressiveDcConvertParams>(device)?;
    let workgroups = validate_dispatch(device, source.pixel_count, variant)?;
    debug_assert!(workgroups != 0);
    Ok(ProgressiveDcConvertParams {
        geometry: [source.width, source.height, source.pixel_count, 0],
        source_y: [
            source.offsets[0],
            source.strides[0],
            source.width,
            source.height,
        ],
        source_x: [
            source.offsets[1],
            source.strides[1],
            source.width,
            source.height,
        ],
        source_b: [
            source.offsets[2],
            source.strides[2],
            source.width,
            source.height,
        ],
        output_strides: [output_strides[0], output_strides[1], output_strides[2], 0],
        multipliers: [
            inputs.multipliers[0],
            inputs.multipliers[1],
            inputs.multipliers[2],
            0.0,
        ],
    })
}

fn validate_pack_inputs(
    device: &wgpu::Device,
    inputs: ProgressiveDcPackInputs<'_>,
    variant: KernelVariant,
) -> Result<ProgressiveDcPackParams, ProgressiveDcGpuError> {
    validate_variant(variant, &device.limits())?;
    let width = inputs.planes.width();
    let height = inputs.planes.height();
    let pixel_count =
        width
            .checked_mul(height)
            .ok_or(ProgressiveDcGpuError::ArithmeticOverflow {
                field: "LF pack pixel count",
            })?;
    if pixel_count == 0 {
        return Err(ProgressiveDcGpuError::EmptyExtent {
            role: "LF pack",
            axis: "pixel",
        });
    }
    let source_strides = validate_xyb_outputs(device, inputs.planes, width, height)?;
    if inputs.lf_stride < width {
        return Err(ProgressiveDcGpuError::InvalidLfStride {
            stride: inputs.lf_stride,
            width,
        });
    }
    let resource_vectors = u64::from(inputs.lf_offset)
        .checked_add(
            u64::from(height - 1)
                .checked_mul(u64::from(inputs.lf_stride))
                .ok_or(ProgressiveDcGpuError::ArithmeticOverflow {
                    field: "LF resource row range",
                })?,
        )
        .and_then(|value| value.checked_add(u64::from(width)))
        .ok_or(ProgressiveDcGpuError::ArithmeticOverflow {
            field: "LF resource range",
        })?;
    if resource_vectors > u64::from(u32::MAX) {
        return Err(ProgressiveDcGpuError::ResourceAddressSpace);
    }
    let resource_bytes = resource_vectors.checked_mul(RESOURCE_VEC4_BYTES).ok_or(
        ProgressiveDcGpuError::ArithmeticOverflow {
            field: "LF resource bytes",
        },
    )?;
    validate_storage_binding(
        device,
        "VarDCT LF resources",
        inputs.resources,
        resource_bytes,
        RESOURCE_VEC4_BYTES,
    )?;
    validate_uniform_limit::<ProgressiveDcPackParams>(device)?;
    let workgroups = validate_dispatch(device, pixel_count, variant)?;
    debug_assert!(workgroups != 0);
    Ok(ProgressiveDcPackParams {
        geometry: [width, height, pixel_count, 0],
        input_strides: [source_strides[0], source_strides[1], source_strides[2], 0],
        destination: [inputs.lf_offset, inputs.lf_stride, 0, 0],
    })
}

struct ValidatedSourcePlanes {
    width: u32,
    height: u32,
    pixel_count: u32,
    offsets: [u32; 3],
    strides: [u32; 3],
    required_bytes: u64,
}

fn validate_source_planes(
    source_planes: [GpuModularChannelLayout; 3],
) -> Result<ValidatedSourcePlanes, ProgressiveDcGpuError> {
    let first = source_planes[0];
    if first.width == 0 {
        return Err(ProgressiveDcGpuError::EmptyExtent {
            role: "source plane 0",
            axis: "width",
        });
    }
    if first.height == 0 {
        return Err(ProgressiveDcGpuError::EmptyExtent {
            role: "source plane 0",
            axis: "height",
        });
    }
    let pixel_count =
        first
            .width
            .checked_mul(first.height)
            .ok_or(ProgressiveDcGpuError::ArithmeticOverflow {
                field: "source pixel count",
            })?;
    let mut offsets = [0u32; 3];
    let mut strides = [0u32; 3];
    let mut required_bytes = 0u64;
    for (plane, layout) in source_planes.into_iter().enumerate() {
        if layout.width == 0 {
            return Err(ProgressiveDcGpuError::EmptyExtent {
                role: "source plane",
                axis: "width",
            });
        }
        if layout.height == 0 {
            return Err(ProgressiveDcGpuError::EmptyExtent {
                role: "source plane",
                axis: "height",
            });
        }
        if layout.width != first.width || layout.height != first.height {
            return Err(ProgressiveDcGpuError::PlaneExtent {
                plane,
                actual_width: layout.width,
                actual_height: layout.height,
                expected_width: first.width,
                expected_height: first.height,
            });
        }
        if layout.row_stride_words < layout.width {
            return Err(ProgressiveDcGpuError::InvalidStride {
                role: "source plane",
                stride: layout.row_stride_words,
                width: layout.width,
            });
        }
        if layout.hshift != 0 || layout.vshift != 0 {
            return Err(ProgressiveDcGpuError::InvalidSourceLayout {
                plane,
                reason: "final planes must have zero horizontal and vertical shifts",
            });
        }
        if layout.reserved != 0 {
            return Err(ProgressiveDcGpuError::InvalidSourceLayout {
                plane,
                reason: "reserved metadata must be zero",
            });
        }
        let required_words = required_plane_scalars(
            layout.width,
            layout.height,
            layout.row_stride_words,
            "source plane words",
        )?;
        let absolute_end_words = u64::from(layout.word_offset)
            .checked_add(required_words)
            .ok_or(ProgressiveDcGpuError::ArithmeticOverflow {
                field: "source plane address range",
            })?;
        if absolute_end_words > u64::from(u32::MAX) {
            return Err(ProgressiveDcGpuError::SourceAddressSpace { plane });
        }
        let plane_bytes =
            required_words
                .checked_mul(4)
                .ok_or(ProgressiveDcGpuError::ArithmeticOverflow {
                    field: "source plane bytes",
                })?;
        required_bytes = required_bytes.max(
            u64::from(layout.word_offset)
                .checked_mul(4)
                .and_then(|offset| offset.checked_add(plane_bytes))
                .ok_or(ProgressiveDcGpuError::ArithmeticOverflow {
                    field: "source plane binding range",
                })?,
        );
        offsets[plane] = layout.word_offset;
        strides[plane] = layout.row_stride_words;
    }
    Ok(ValidatedSourcePlanes {
        width: first.width,
        height: first.height,
        pixel_count,
        offsets,
        strides,
        required_bytes,
    })
}

fn validate_xyb_outputs(
    device: &wgpu::Device,
    outputs: &ProgressiveDcXybPlanes,
    width: u32,
    height: u32,
) -> Result<[u32; 3], ProgressiveDcGpuError> {
    let mut strides = [0u32; 3];
    for (plane, output) in outputs.planes.iter().enumerate() {
        if output.width == 0 {
            return Err(ProgressiveDcGpuError::EmptyExtent {
                role: "XYB output",
                axis: "width",
            });
        }
        if output.height == 0 {
            return Err(ProgressiveDcGpuError::EmptyExtent {
                role: "XYB output",
                axis: "height",
            });
        }
        if output.width != width || output.height != height {
            return Err(ProgressiveDcGpuError::PlaneExtent {
                plane,
                actual_width: output.width,
                actual_height: output.height,
                expected_width: width,
                expected_height: height,
            });
        }
        if output.stride < width {
            return Err(ProgressiveDcGpuError::InvalidStride {
                role: "XYB output",
                stride: output.stride,
                width,
            });
        }
        let required_scalars = output.required_scalars()?;
        let required_bytes = required_scalars.checked_mul(F32_BYTES).ok_or(
            ProgressiveDcGpuError::ArithmeticOverflow {
                field: "XYB output bytes",
            },
        )?;
        let binding = entire_storage_binding(&output.buffer)?;
        validate_storage_binding(device, "XYB output", binding, required_bytes, F32_BYTES)?;
        strides[plane] = output.stride;
    }
    Ok(strides)
}

fn validate_multipliers(multipliers: [f32; 3]) -> Result<(), ProgressiveDcGpuError> {
    for (channel, value) in multipliers.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(ProgressiveDcGpuError::NonFiniteMultiplier { channel });
        }
    }
    Ok(())
}

fn validate_dispatch(
    device: &wgpu::Device,
    pixel_count: u32,
    variant: KernelVariant,
) -> Result<u32, ProgressiveDcGpuError> {
    let workgroups = pixel_count.div_ceil(variant.workgroup_size().0);
    let available = device.limits().max_compute_workgroups_per_dimension;
    if workgroups > available {
        return Err(ProgressiveDcGpuError::WorkgroupCount {
            required: workgroups,
            available,
        });
    }
    Ok(workgroups)
}

fn required_plane_scalars(
    width: u32,
    height: u32,
    stride: u32,
    field: &'static str,
) -> Result<u64, ProgressiveDcGpuError> {
    if width == 0 {
        return Err(ProgressiveDcGpuError::EmptyExtent {
            role: "plane",
            axis: "width",
        });
    }
    if height == 0 {
        return Err(ProgressiveDcGpuError::EmptyExtent {
            role: "plane",
            axis: "height",
        });
    }
    u64::from(height - 1)
        .checked_mul(u64::from(stride))
        .and_then(|value| value.checked_add(u64::from(width)))
        .ok_or(ProgressiveDcGpuError::ArithmeticOverflow { field })
}

fn normalized_stride(
    width: u32,
    height: u32,
    stride: u32,
    role: &'static str,
) -> Result<u32, ProgressiveDcGpuError> {
    if width == 0 {
        return Err(ProgressiveDcGpuError::EmptyExtent {
            role,
            axis: "width",
        });
    }
    if height == 0 {
        return Err(ProgressiveDcGpuError::EmptyExtent {
            role,
            axis: "height",
        });
    }
    let stride = if stride == 0 { width } else { stride };
    if stride < width {
        return Err(ProgressiveDcGpuError::InvalidStride {
            role,
            stride,
            width,
        });
    }
    required_plane_scalars(width, height, stride, "plane scalar range")?;
    Ok(stride)
}

fn validate_allocated_buffer_size(
    device: &wgpu::Device,
    role: &'static str,
    required: u64,
) -> Result<(), ProgressiveDcGpuError> {
    if required == 0 {
        return Err(ProgressiveDcGpuError::EmptyBuffer { role });
    }
    let limits = device.limits();
    if required > limits.max_buffer_size {
        return Err(ProgressiveDcGpuError::BufferLimit {
            role,
            required,
            available: limits.max_buffer_size,
        });
    }
    if required > limits.max_storage_buffer_binding_size {
        return Err(ProgressiveDcGpuError::StorageBindingLimit {
            role,
            required,
            available: limits.max_storage_buffer_binding_size,
        });
    }
    Ok(())
}

fn validate_storage_binding(
    device: &wgpu::Device,
    role: &'static str,
    binding: ResidentStorageBinding<'_>,
    required: u64,
    element_alignment: u64,
) -> Result<(), ProgressiveDcGpuError> {
    if binding.size.get() == 0 {
        return Err(ProgressiveDcGpuError::EmptyBuffer { role });
    }
    if !binding.buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
        return Err(ProgressiveDcGpuError::MissingStorageUsage { role });
    }
    let limits = device.limits();
    let offset_alignment = u64::from(limits.min_storage_buffer_offset_alignment).max(4);
    if !binding.offset.is_multiple_of(offset_alignment) {
        return Err(ProgressiveDcGpuError::BindingAlignment {
            role,
            offset: binding.offset,
            alignment: offset_alignment,
        });
    }
    let end = binding.offset.checked_add(binding.size.get()).ok_or(
        ProgressiveDcGpuError::ArithmeticOverflow {
            field: "storage binding range",
        },
    )?;
    if end > binding.buffer.size() {
        return Err(ProgressiveDcGpuError::BindingRange {
            role,
            offset: binding.offset,
            end,
            available: binding.buffer.size(),
        });
    }
    if !binding.size.get().is_multiple_of(element_alignment) {
        return Err(ProgressiveDcGpuError::BindingSizeAlignment {
            role,
            size: binding.size.get(),
            alignment: element_alignment,
        });
    }
    if binding.size.get() < required {
        return Err(ProgressiveDcGpuError::BindingSize {
            role,
            required,
            available: binding.size.get(),
        });
    }
    if binding.size.get() > limits.max_storage_buffer_binding_size {
        return Err(ProgressiveDcGpuError::StorageBindingLimit {
            role,
            required: binding.size.get(),
            available: limits.max_storage_buffer_binding_size,
        });
    }
    Ok(())
}

fn validate_uniform_limit<T>(device: &wgpu::Device) -> Result<(), ProgressiveDcGpuError> {
    let required = std::mem::size_of::<T>() as u64;
    let available = device.limits().max_uniform_buffer_binding_size;
    if required > available {
        return Err(ProgressiveDcGpuError::UniformBindingLimit {
            required,
            available,
        });
    }
    Ok(())
}

fn entire_storage_binding(
    buffer: &wgpu::Buffer,
) -> Result<ResidentStorageBinding<'_>, ProgressiveDcGpuError> {
    let size = NonZeroU64::new(buffer.size())
        .ok_or(ProgressiveDcGpuError::EmptyBuffer { role: "storage" })?;
    Ok(ResidentStorageBinding {
        buffer,
        offset: 0,
        size,
    })
}

fn storage_entry<'a>(
    binding: u32,
    storage: ResidentStorageBinding<'a>,
) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: storage.buffer,
            offset: storage.offset,
            size: Some(storage.size),
        }),
    }
}

const _: () = {
    assert!(std::mem::size_of::<ProgressiveDcConvertParams>() == 96);
    assert!(std::mem::align_of::<ProgressiveDcConvertParams>() == 16);
    assert!(std::mem::size_of::<ProgressiveDcPackParams>() == 48);
    assert!(std::mem::align_of::<ProgressiveDcPackParams>() == 16);
    assert!(std::mem::offset_of!(ProgressiveDcConvertParams, geometry) == 0);
    assert!(std::mem::offset_of!(ProgressiveDcConvertParams, source_y) == 16);
    assert!(std::mem::offset_of!(ProgressiveDcConvertParams, source_x) == 32);
    assert!(std::mem::offset_of!(ProgressiveDcConvertParams, source_b) == 48);
    assert!(std::mem::offset_of!(ProgressiveDcConvertParams, output_strides) == 64);
    assert!(std::mem::offset_of!(ProgressiveDcConvertParams, multipliers) == 80);
    assert!(std::mem::offset_of!(ProgressiveDcPackParams, geometry) == 0);
    assert!(std::mem::offset_of!(ProgressiveDcPackParams, input_strides) == 16);
    assert!(std::mem::offset_of!(ProgressiveDcPackParams, destination) == 32);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_parse_and_validate_semantically() {
        let module = naga::front::wgsl::parse_str(PROGRESSIVE_DC_SHADER)
            .expect("progressive-DC WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("progressive-DC WGSL validates");
        let entry_points = module
            .entry_points
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(entry_points, ["convert_modular", "pack_lf"]);
    }

    #[test]
    fn parameter_abi_is_pod_and_wgsl_aligned() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<ProgressiveDcConvertParams>();
        assert_pod::<ProgressiveDcPackParams>();
        assert_eq!(std::mem::size_of::<ProgressiveDcConvertParams>(), 96);
        assert_eq!(std::mem::align_of::<ProgressiveDcConvertParams>(), 16);
        assert_eq!(std::mem::size_of::<ProgressiveDcPackParams>(), 48);
        assert_eq!(std::mem::align_of::<ProgressiveDcPackParams>(), 16);
        assert_eq!(
            bytemuck::bytes_of(&ProgressiveDcConvertParams::zeroed()).len(),
            96
        );
        assert_eq!(
            bytemuck::bytes_of(&ProgressiveDcPackParams::zeroed()).len(),
            48
        );
    }

    #[test]
    fn source_range_arithmetic_tracks_offsets_and_padding() {
        let layout = GpuModularChannelLayout {
            word_offset: 7,
            row_stride_words: 5,
            width: 3,
            height: 2,
            hshift: 0,
            vshift: 0,
            bit_depth: 16,
            reserved: 0,
        };
        let source = validate_source_planes([layout; 3]).expect("valid source layout");
        assert_eq!(source.pixel_count, 6);
        assert_eq!(source.offsets, [7; 3]);
        assert_eq!(source.strides, [5; 3]);
        assert_eq!(source.required_bytes, 60);
    }
}
