//! Owned pre-color-transform LF dependency planes and GPU packing into VarDCT LF resources.
//!
//! Producers reconstruct and retain `[X, Y, B]` F32 planes; this module packs them into
//! the consumer's `[X, Y, B, 0]` vec4 resource table without host pixel transfers.

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use jxl_wgpu::{
    GpuBufferLease, KernelPolicy, KernelVariant, MemoryPermitSplitError, ResidentStorageBinding,
};
use thiserror::Error;
use wgpu::util::DeviceExt;

const PROGRESSIVE_DC_SHADER: &str = include_str!("progressive_dc.wgsl");
const F32_BYTES: u64 = std::mem::size_of::<f32>() as u64;
const RESOURCE_VEC4_BYTES: u64 = std::mem::size_of::<[f32; 4]>() as u64;

/// Stable kernel-policy key for the LF-resource packing pass.
pub(crate) const PROGRESSIVE_DC_KERNEL_KEY: &str = "progressive_dc";

/// Built-in linear variant used when the adapter policy has no tuned entry.
pub(crate) const DEFAULT_PROGRESSIVE_DC_VARIANT: KernelVariant = KernelVariant::Lanes64;

/// One owned GPU-resident F32 XYB plane.
#[derive(Clone, Debug)]
pub(crate) struct ProgressiveDcXybPlane {
    pub(crate) buffer: GpuBufferLease,
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
/// Cloning retains both the allocation and its byte reservation across submissions. Producer
/// scratch can be released as soon as validation completes, independently of LF consumers.
#[derive(Clone, Debug)]
pub(crate) struct ProgressiveDcXybPlanes {
    pub(crate) planes: [ProgressiveDcXybPlane; 3],
}

impl ProgressiveDcXybPlanes {
    pub(crate) fn validate_extent(
        &self,
        [expected_width, expected_height]: [u32; 2],
    ) -> Result<(), ProgressiveDcGpuError> {
        for (plane, actual) in self.planes.iter().enumerate() {
            if actual.width != expected_width || actual.height != expected_height {
                return Err(ProgressiveDcGpuError::PlaneExtent {
                    plane,
                    actual_width: actual.width,
                    actual_height: actual.height,
                    expected_width,
                    expected_height,
                });
            }
        }
        Ok(())
    }

    /// Wraps three already-created storage buffers in the owned XYB representation.
    ///
    /// Buffer usage, range, and device-limit checks are deferred to [`ProgressiveDcPipeline`] so
    /// this constructor can remain independent of a device handle while still validating all
    /// geometry and arithmetic that is intrinsic to the representation.
    pub(crate) fn from_leases(
        buffers: [GpuBufferLease; 3],
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
    #[error("progressive-DC plane reservation: {0}")]
    MemoryPartition(#[from] MemoryPermitSplitError),
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
    #[error("progressive-DC uniform needs {required} bytes, device permits {available}")]
    UniformBindingLimit { required: u64, available: u64 },
    #[error("progressive-DC resource LF stride {stride} is smaller than width {width}")]
    InvalidLfStride { stride: u32, width: u32 },
    #[error("progressive-DC LF resource address range exceeds WGSL's u32 vec4 index space")]
    ResourceAddressSpace,
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

/// Reusable LF-resource packing pipeline for progressive-DC frames.
pub(crate) struct ProgressiveDcPipeline {
    pack: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl ProgressiveDcPipeline {
    /// Compiles the packing entry point with a selected linear [`KernelVariant`].
    pub(crate) fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, ProgressiveDcGpuError> {
        validate_variant(variant, &device.limits())?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu progressive-DC LF packing"),
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
            .map(|plane| entire_storage_binding(plane.buffer.as_wgpu_buffer()))
            .collect::<Result<Vec<_>, _>>()?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu progressive-DC LF packing bindings"),
            layout: &self.pack.get_bind_group_layout(0),
            entries: &[
                storage_entry(0, input_bindings[0]),
                storage_entry(1, input_bindings[1]),
                storage_entry(2, input_bindings[2]),
                storage_entry(3, inputs.resources),
                wgpu::BindGroupEntry {
                    binding: 4,
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
        let binding = entire_storage_binding(output.buffer.as_wgpu_buffer())?;
        validate_storage_binding(device, "XYB output", binding, required_bytes, F32_BYTES)?;
        strides[plane] = output.stride;
    }
    Ok(strides)
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
    assert!(std::mem::size_of::<ProgressiveDcPackParams>() == 48);
    assert!(std::mem::align_of::<ProgressiveDcPackParams>() == 16);
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
        assert_eq!(entry_points, ["pack_lf"]);
    }

    #[test]
    fn parameter_abi_is_pod_and_wgsl_aligned() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<ProgressiveDcPackParams>();
        assert_eq!(std::mem::size_of::<ProgressiveDcPackParams>(), 48);
        assert_eq!(std::mem::align_of::<ProgressiveDcPackParams>(), 16);
        assert_eq!(
            bytemuck::bytes_of(&ProgressiveDcPackParams::zeroed()).len(),
            48
        );
    }
}
