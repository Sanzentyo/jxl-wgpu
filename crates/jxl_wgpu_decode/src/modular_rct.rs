//! GPU-resident inverse JPEG XL Modular reversible color transform.
//!
//! A Modular RCT is an in-place operation on three equal-size i32 planes.  The three planes are
//! views into one storage arena so that a caller can keep every transformed channel resident
//! between inverse Modular passes.  Each invocation loads all three input words before it writes
//! any result; this is what makes an in-place permutation safe as well as keeping the storage
//! binding unambiguous to WebGPU.

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use jxl_wgpu::{KernelPolicy, KernelVariant, ResidentStorageBinding};
use thiserror::Error;
use wgpu::util::DeviceExt;

/// Number of legal JPEG XL Modular RCT types.
pub const MODULAR_RCT_TYPE_COUNT: u32 = 42;

/// Stable kernel-policy key for this pass.
pub const MODULAR_RCT_KERNEL_KEY: &str = "modular_rct";

/// Built-in linear workgroup variant used by [`ModularRctPipeline::new`].
pub const DEFAULT_MODULAR_RCT_VARIANT: KernelVariant = KernelVariant::Lanes64;

/// Linear variants accepted by this pass and by its kernel policy.
pub const MODULAR_RCT_VARIANTS: [KernelVariant; 5] = [
    KernelVariant::Scalar,
    KernelVariant::Lanes32,
    KernelVariant::Lanes64,
    KernelVariant::Lanes128,
    KernelVariant::Lanes256,
];

/// WGSL for [`ModularRctPipeline`].
pub const MODULAR_RCT_SHADER: &str = r#"
override wg_x: u32 = 64u;

struct Params {
    // width, height, row stride (in i32 words), and offset (in words)
    first: vec4<u32>,
    second: vec4<u32>,
    third: vec4<u32>,
    // operation.x is the RCT type; the remaining words are reserved.
    operation: vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> arena: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

struct RctValues {
    first: i32,
    second: i32,
    third: i32,
};

fn add_wrap(left: i32, right: i32) -> i32 {
    return bitcast<i32>(bitcast<u32>(left) + bitcast<u32>(right));
}

fn sub_wrap(left: i32, right: i32) -> i32 {
    return bitcast<i32>(bitcast<u32>(left) - bitcast<u32>(right));
}

// Arithmetic right shift by one, expressed on the bit pattern so this remains exact even on
// implementations where signed overflow behavior is not useful as a portability primitive.
fn arithmetic_shr_one(value: i32) -> i32 {
    let bits = bitcast<u32>(value);
    let sign = select(0u, 0x80000000u, (bits & 0x80000000u) != 0u);
    return bitcast<i32>((bits >> 1u) | sign);
}

// This is jxl 0.6's rct_impl.  The host validates operation.x in 0..42, so the final return is
// only a defensive value for malformed data supplied after command recording.
fn rct_transform(v0: i32, v1: i32, v2: i32, rct_type: u32) -> RctValues {
    let operation = rct_type % 7u;
    if operation == 0u {
        return RctValues(v0, v1, v2);
    }
    if operation == 1u {
        return RctValues(v0, v1, add_wrap(v2, v0));
    }
    if operation == 2u {
        return RctValues(v0, add_wrap(v1, v0), v2);
    }
    if operation == 3u {
        return RctValues(v0, add_wrap(v1, v0), add_wrap(v2, v0));
    }
    if operation == 4u {
        let average = arithmetic_shr_one(add_wrap(v0, v2));
        return RctValues(v0, add_wrap(v1, average), v2);
    }
    if operation == 5u {
        let third = add_wrap(v0, v2);
        let average = arithmetic_shr_one(add_wrap(v0, third));
        return RctValues(v0, add_wrap(v1, average), third);
    }
    if operation == 6u {
        let y0 = sub_wrap(v0, arithmetic_shr_one(v2));
        let green = add_wrap(v2, y0);
        let y = sub_wrap(y0, arithmetic_shr_one(v1));
        let red = add_wrap(y, v1);
        return RctValues(red, green, y);
    }
    return RctValues(v0, v1, v2);
}

fn load_word(plane: vec4<u32>, x: u32, y: u32) -> i32 {
    return bitcast<i32>(arena[plane.w + y * plane.z + x]);
}

fn store_word(plane: vec4<u32>, x: u32, y: u32, value: i32) {
    arena[plane.w + y * plane.z + x] = bitcast<u32>(value);
}

fn inverse_rct_at(x: u32, y: u32) {
    // Keep these three loads before every store.  The three views may be the only live copies of
    // their channels, and the permutation below can write any view first.
    let v0 = load_word(params.first, x, y);
    let v1 = load_word(params.second, x, y);
    let v2 = load_word(params.third, x, y);
    let values = rct_transform(v0, v1, v2, params.operation.x);
    let permutation = params.operation.x / 7u;
    if permutation == 0u {
        store_word(params.first, x, y, values.first);
        store_word(params.second, x, y, values.second);
        store_word(params.third, x, y, values.third);
    } else if permutation == 1u {
        store_word(params.first, x, y, values.third);
        store_word(params.second, x, y, values.first);
        store_word(params.third, x, y, values.second);
    } else if permutation == 2u {
        store_word(params.first, x, y, values.second);
        store_word(params.second, x, y, values.third);
        store_word(params.third, x, y, values.first);
    } else if permutation == 3u {
        store_word(params.first, x, y, values.first);
        store_word(params.second, x, y, values.third);
        store_word(params.third, x, y, values.second);
    } else if permutation == 4u {
        store_word(params.first, x, y, values.second);
        store_word(params.second, x, y, values.first);
        store_word(params.third, x, y, values.third);
    } else {
        store_word(params.first, x, y, values.third);
        store_word(params.second, x, y, values.second);
        store_word(params.third, x, y, values.first);
    }
}

@compute @workgroup_size(wg_x, 1, 1)
fn inverse_rct(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let pixel_count = params.first.x * params.first.y;
    if invocation.x >= pixel_count {
        return;
    }
    let y = invocation.x / params.first.x;
    let x = invocation.x - y * params.first.x;
    inverse_rct_at(x, y);
}
"#;

/// A planar i32 storage view.  `stride` and `offset_words` are measured in i32/u32 words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModularRctPlane {
    pub width: u32,
    pub height: u32,
    /// Row stride in words; it may be larger than `width` for a subview.
    pub stride: u32,
    /// Word offset relative to the corresponding [`ResidentStorageBinding`].
    pub offset_words: u32,
}

impl ModularRctPlane {
    /// Creates a tightly packed plane view.
    #[must_use]
    pub const fn tight(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            stride: width,
            offset_words: 0,
        }
    }

    /// Returns the exclusive word offset needed to cover this view.
    fn required_words(self) -> Option<u64> {
        if self.width == 0 || self.height == 0 {
            return Some(u64::from(self.offset_words));
        }
        u64::from(self.offset_words)
            .checked_add(u64::from(self.height - 1) * u64::from(self.stride))?
            .checked_add(u64::from(self.width))
    }
}

/// Exact 64-byte uniform consumed by [`MODULAR_RCT_SHADER`].
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct ModularRctParams {
    /// Width, height, row stride, and offset for the first input/output plane.
    pub first: [u32; 4],
    /// Width, height, row stride, and offset for the second input/output plane.
    pub second: [u32; 4],
    /// Width, height, row stride, and offset for the third input/output plane.
    pub third: [u32; 4],
    /// RCT type in word zero (0 through 41); remaining words are reserved zero.
    pub operation: [u32; 4],
}

impl ModularRctParams {
    /// Builds the wire uniform from three equal-geometry plane views and an RCT type.
    #[must_use]
    pub const fn new(
        rct_type: u32,
        first: ModularRctPlane,
        second: ModularRctPlane,
        third: ModularRctPlane,
    ) -> Self {
        Self {
            first: [first.width, first.height, first.stride, first.offset_words],
            second: [
                second.width,
                second.height,
                second.stride,
                second.offset_words,
            ],
            third: [third.width, third.height, third.stride, third.offset_words],
            operation: [rct_type, 0, 0, 0],
        }
    }

    /// Returns the encoded RCT type.
    #[must_use]
    pub const fn rct_type(self) -> u32 {
        self.operation[0]
    }

    #[must_use]
    pub const fn first_plane(self) -> ModularRctPlane {
        ModularRctPlane {
            width: self.first[0],
            height: self.first[1],
            stride: self.first[2],
            offset_words: self.first[3],
        }
    }

    #[must_use]
    pub const fn second_plane(self) -> ModularRctPlane {
        ModularRctPlane {
            width: self.second[0],
            height: self.second[1],
            stride: self.second[2],
            offset_words: self.second[3],
        }
    }

    #[must_use]
    pub const fn third_plane(self) -> ModularRctPlane {
        ModularRctPlane {
            width: self.third[0],
            height: self.third[1],
            stride: self.third[2],
            offset_words: self.third[3],
        }
    }

    #[must_use]
    pub const fn planes(self) -> [ModularRctPlane; 3] {
        [self.first_plane(), self.second_plane(), self.third_plane()]
    }
}

/// Storage arena supplied to the in-place inverse RCT pass.
#[derive(Clone, Copy, Debug)]
pub struct ModularRctArena<'a> {
    pub storage: ResidentStorageBinding<'a>,
}

impl<'a> ModularRctArena<'a> {
    /// Uses one complete non-empty storage buffer as the arena.
    pub fn entire(buffer: &'a wgpu::Buffer) -> Result<Self, ModularRctError> {
        let size = NonZeroU64::new(buffer.size()).ok_or(ModularRctError::EmptyBinding)?;
        Ok(Self {
            storage: ResidentStorageBinding {
                buffer,
                offset: 0,
                size,
            },
        })
    }

    /// Wraps a checked subrange of a storage buffer as the arena binding.
    #[must_use]
    pub const fn from_storage(storage: ResidentStorageBinding<'a>) -> Self {
        Self { storage }
    }
}

/// Host-side dispatch geometry after checking one inverse RCT operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModularRctPlan {
    pub params: ModularRctParams,
    pub pixel_count: u32,
    pub workgroups: u32,
    pub variant: KernelVariant,
}

/// Typed planning and recording failures for [`ModularRctPipeline`].
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ModularRctError {
    #[error("Modular RCT arena binding is empty")]
    EmptyBinding,
    #[error("Modular RCT arena buffer is missing STORAGE usage")]
    MissingStorageUsage,
    #[error("Modular RCT arena binding offset {offset} is not aligned to {alignment}")]
    BindingAlignment { offset: u64, alignment: u64 },
    #[error("Modular RCT arena binding range {offset}..{end} exceeds buffer size {available}")]
    BindingRange {
        offset: u64,
        end: u64,
        available: u64,
    },
    #[error("Modular RCT arena binding size {size} is not a multiple of four bytes")]
    BindingSizeAlignment { size: u64 },
    #[error("Modular RCT plane {plane} needs {required} bytes, binding has {available}")]
    BindingSize {
        plane: &'static str,
        required: u64,
        available: u64,
    },
    #[error("Modular RCT arena binding needs {required} bytes, device permits {available}")]
    StorageBindingLimit { required: u64, available: u64 },
    #[error("Modular RCT uniform needs {required} bytes, device permits {available}")]
    UniformBindingLimit { required: u64, available: u64 },
    #[error("Modular RCT {plane} has a zero {axis} extent")]
    ZeroExtent {
        plane: &'static str,
        axis: &'static str,
    },
    #[error("Modular RCT {plane} stride {stride} is smaller than width {width}")]
    InvalidStride {
        plane: &'static str,
        stride: u32,
        width: u32,
    },
    #[error(
        "Modular RCT {plane} has geometry {actual_width}x{actual_height}, expected {expected_width}x{expected_height}"
    )]
    UnequalGeometry {
        plane: &'static str,
        actual_width: u32,
        actual_height: u32,
        expected_width: u32,
        expected_height: u32,
    },
    #[error("Modular RCT {first} and {second} plane footprints overlap")]
    PlaneOverlap {
        first: &'static str,
        second: &'static str,
    },
    #[error("Modular RCT type {rct_type} is invalid; valid types are 0 through 41")]
    InvalidRctType { rct_type: u32 },
    #[error("Modular RCT reserved parameter word {word} must be zero")]
    NonZeroReservedParameter { word: usize },
    #[error("Modular RCT address range for {plane} exceeds WGSL's u32 word address space")]
    ShaderAddressSpace { plane: &'static str },
    #[error("Modular RCT arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
    #[error("Modular RCT requires a linear workgroup, got {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    #[error("Modular RCT workgroup variant {variant:?} exceeds device limits")]
    WorkgroupVariant { variant: KernelVariant },
    #[error("Modular RCT dispatch needs {required} workgroups, device permits {available}")]
    WorkgroupCount { required: u32, available: u32 },
    #[error("Modular RCT kernel policy failed: {0}")]
    KernelPolicy(String),
}

/// Reusable compute pipeline for one GPU-resident inverse RCT operation.
pub struct ModularRctPipeline {
    pipeline: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl ModularRctPipeline {
    /// Compiles the default linear kernel variant.
    pub fn new(device: &wgpu::Device) -> Result<Self, ModularRctError> {
        Self::with_variant(device, DEFAULT_MODULAR_RCT_VARIANT)
    }

    /// Compiles the kernel with a linear [`KernelVariant`].
    pub fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, ModularRctError> {
        validate_variant(variant, &device.limits())?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu decode Modular inverse RCT"),
            source: wgpu::ShaderSource::Wgsl(MODULAR_RCT_SHADER.into()),
        });
        let constants = [("wg_x", f64::from(variant.workgroup_size().0))];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu decode Modular inverse RCT"),
            layout: None,
            module: &module,
            entry_point: Some("inverse_rct"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
        Ok(Self { pipeline, variant })
    }

    /// Selects a linear variant using the shared adapter policy.
    pub fn with_policy(
        device: &wgpu::Device,
        policy: &KernelPolicy,
    ) -> Result<Self, ModularRctError> {
        let variant = policy
            .variant_for(MODULAR_RCT_KERNEL_KEY, DEFAULT_MODULAR_RCT_VARIANT)
            .map_err(|error| ModularRctError::KernelPolicy(error.to_string()))?;
        Self::with_variant(device, variant)
    }

    /// Records one in-place inverse RCT operation and returns its uniform allocation.
    ///
    /// All samples stay in the storage arena owned by the caller.  The returned uniform must be
    /// retained until command submission, matching the lifetime contract of the other standalone
    /// resident decode pipelines.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        arena: ModularRctArena<'_>,
        params: ModularRctParams,
    ) -> Result<wgpu::Buffer, ModularRctError> {
        let plan = plan_for_device(device, arena, params, self.variant)?;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu decode Modular inverse RCT params"),
            contents: bytemuck::bytes_of(&plan.params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu decode Modular inverse RCT bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                binding_entry(0, arena.storage),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu decode Modular inverse RCT"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(plan.workgroups, 1, 1);
        drop(pass);
        Ok(uniform)
    }

    /// Returns the selected workgroup variant.
    #[must_use]
    pub const fn variant(&self) -> KernelVariant {
        self.variant
    }
}

fn binding_entry<'a>(
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

fn validate_variant(variant: KernelVariant, limits: &wgpu::Limits) -> Result<(), ModularRctError> {
    if !variant.is_linear() {
        return Err(ModularRctError::WorkgroupShape { variant });
    }
    variant
        .validate_for(MODULAR_RCT_KERNEL_KEY, limits, 0)
        .map_err(|_| ModularRctError::WorkgroupVariant { variant })
}

fn validate_params(params: ModularRctParams) -> Result<u32, ModularRctError> {
    if params.rct_type() >= MODULAR_RCT_TYPE_COUNT {
        return Err(ModularRctError::InvalidRctType {
            rct_type: params.rct_type(),
        });
    }
    for (word, value) in params.operation[1..].iter().copied().enumerate() {
        if value != 0 {
            return Err(ModularRctError::NonZeroReservedParameter { word: word + 1 });
        }
    }

    let planes = [
        (params.first_plane(), "first"),
        (params.second_plane(), "second"),
        (params.third_plane(), "third"),
    ];
    let reference = planes[0].0;
    for (plane, name) in planes {
        if plane.width == 0 {
            return Err(ModularRctError::ZeroExtent {
                plane: name,
                axis: "width",
            });
        }
        if plane.height == 0 {
            return Err(ModularRctError::ZeroExtent {
                plane: name,
                axis: "height",
            });
        }
        if plane.stride < plane.width {
            return Err(ModularRctError::InvalidStride {
                plane: name,
                stride: plane.stride,
                width: plane.width,
            });
        }
        if plane.width != reference.width || plane.height != reference.height {
            return Err(ModularRctError::UnequalGeometry {
                plane: name,
                actual_width: plane.width,
                actual_height: plane.height,
                expected_width: reference.width,
                expected_height: reference.height,
            });
        }
    }

    let pixel_count = reference.width.checked_mul(reference.height).ok_or(
        ModularRctError::ArithmeticOverflow {
            field: "pixel count",
        },
    )?;
    for (plane, name) in planes {
        let required_words = plane
            .required_words()
            .ok_or(ModularRctError::ArithmeticOverflow {
                field: "plane word range",
            })?;
        if required_words > u64::from(u32::MAX) {
            return Err(ModularRctError::ShaderAddressSpace { plane: name });
        }
    }
    validate_non_overlapping_planes(planes)?;
    Ok(pixel_count)
}

fn validate_non_overlapping_planes(
    planes: [(ModularRctPlane, &'static str); 3],
) -> Result<(), ModularRctError> {
    for first_index in 0..planes.len() {
        for second_index in (first_index + 1)..planes.len() {
            let (first, first_name) = planes[first_index];
            let (second, second_name) = planes[second_index];
            let first_end = first
                .required_words()
                .ok_or(ModularRctError::ArithmeticOverflow {
                    field: "plane overlap range",
                })?;
            let second_end =
                second
                    .required_words()
                    .ok_or(ModularRctError::ArithmeticOverflow {
                        field: "plane overlap range",
                    })?;
            if u64::from(first.offset_words) < second_end
                && u64::from(second.offset_words) < first_end
            {
                return Err(ModularRctError::PlaneOverlap {
                    first: first_name,
                    second: second_name,
                });
            }
        }
    }
    Ok(())
}

fn validate_binding(
    binding: ResidentStorageBinding<'_>,
    limits: &wgpu::Limits,
) -> Result<(), ModularRctError> {
    if binding.size.get() == 0 {
        return Err(ModularRctError::EmptyBinding);
    }
    if !binding.buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
        return Err(ModularRctError::MissingStorageUsage);
    }
    let alignment = u64::from(limits.min_storage_buffer_offset_alignment).max(4);
    if !binding.offset.is_multiple_of(alignment) {
        return Err(ModularRctError::BindingAlignment {
            offset: binding.offset,
            alignment,
        });
    }
    if !binding.offset.is_multiple_of(4) {
        return Err(ModularRctError::BindingAlignment {
            offset: binding.offset,
            alignment: 4,
        });
    }
    let end = binding.offset.checked_add(binding.size.get()).ok_or(
        ModularRctError::ArithmeticOverflow {
            field: "storage binding range",
        },
    )?;
    if end > binding.buffer.size() {
        return Err(ModularRctError::BindingRange {
            offset: binding.offset,
            end,
            available: binding.buffer.size(),
        });
    }
    if !binding.size.get().is_multiple_of(4) {
        return Err(ModularRctError::BindingSizeAlignment {
            size: binding.size.get(),
        });
    }
    if binding.size.get() > limits.max_storage_buffer_binding_size {
        return Err(ModularRctError::StorageBindingLimit {
            required: binding.size.get(),
            available: limits.max_storage_buffer_binding_size,
        });
    }
    Ok(())
}

fn plan_for_device(
    device: &wgpu::Device,
    arena: ModularRctArena<'_>,
    params: ModularRctParams,
    variant: KernelVariant,
) -> Result<ModularRctPlan, ModularRctError> {
    validate_variant(variant, &device.limits())?;
    let pixel_count = validate_params(params)?;
    validate_binding(arena.storage, &device.limits())?;
    for (plane, name) in [
        (params.first_plane(), "first"),
        (params.second_plane(), "second"),
        (params.third_plane(), "third"),
    ] {
        let required = plane
            .required_words()
            .ok_or(ModularRctError::ArithmeticOverflow {
                field: "plane byte range",
            })?
            .checked_mul(4)
            .ok_or(ModularRctError::ArithmeticOverflow {
                field: "plane byte range",
            })?;
        if required > arena.storage.size.get() {
            return Err(ModularRctError::BindingSize {
                plane: name,
                required,
                available: arena.storage.size.get(),
            });
        }
    }
    let uniform_bytes = std::mem::size_of::<ModularRctParams>() as u64;
    if uniform_bytes > device.limits().max_uniform_buffer_binding_size {
        return Err(ModularRctError::UniformBindingLimit {
            required: uniform_bytes,
            available: device.limits().max_uniform_buffer_binding_size,
        });
    }
    let workgroups = pixel_count.div_ceil(variant.workgroup_size().0);
    let maximum = device.limits().max_compute_workgroups_per_dimension;
    if workgroups > maximum {
        return Err(ModularRctError::WorkgroupCount {
            required: workgroups,
            available: maximum,
        });
    }
    Ok(ModularRctPlan {
        params,
        pixel_count,
        workgroups,
        variant,
    })
}

const _: () = {
    assert!(std::mem::size_of::<ModularRctParams>() == 64);
    assert!(std::mem::align_of::<ModularRctParams>() == 16);
    assert!(std::mem::offset_of!(ModularRctParams, first) == 0);
    assert!(std::mem::offset_of!(ModularRctParams, second) == 16);
    assert!(std::mem::offset_of!(ModularRctParams, third) == 32);
    assert!(std::mem::offset_of!(ModularRctParams, operation) == 48);
};

#[cfg(test)]
mod tests {
    use super::*;

    fn wrap_add(left: i32, right: i32) -> i32 {
        left.wrapping_add(right)
    }

    fn wrap_sub(left: i32, right: i32) -> i32 {
        left.wrapping_sub(right)
    }

    fn scalar_transform(first: i32, second: i32, third: i32, rct_type: u32) -> [i32; 3] {
        let operation = rct_type % 7;
        let values = match operation {
            0 => [first, second, third],
            1 => [first, second, wrap_add(third, first)],
            2 => [first, wrap_add(second, first), third],
            3 => [first, wrap_add(second, first), wrap_add(third, first)],
            4 => {
                let average = wrap_add(first, third) >> 1;
                [first, wrap_add(second, average), third]
            }
            5 => {
                let third = wrap_add(first, third);
                let average = wrap_add(first, third) >> 1;
                [first, wrap_add(second, average), third]
            }
            6 => {
                let y0 = wrap_sub(first, third >> 1);
                let green = wrap_add(third, y0);
                let y = wrap_sub(y0, second >> 1);
                let red = wrap_add(y, second);
                [red, green, y]
            }
            _ => unreachable!(),
        };
        match rct_type / 7 {
            0 => values,
            1 => [values[2], values[0], values[1]],
            2 => [values[1], values[2], values[0]],
            3 => [values[0], values[2], values[1]],
            4 => [values[1], values[0], values[2]],
            5 => [values[2], values[1], values[0]],
            _ => unreachable!(),
        }
    }

    fn pattern(plane: usize, x: u32, y: u32) -> i32 {
        match (usize::try_from(x + 5 * y).unwrap() + plane) % 13 {
            0 => i32::MIN,
            1 => i32::MAX,
            2 => -1,
            3 => 0,
            4 => 1,
            5 => 0x4000_0000,
            6 => -0x4000_0000,
            7 => 0x5555_5555,
            8 => -0x5555_5555,
            9 => 0x7fff_fffe,
            10 => i32::MIN + 1,
            11 => 0x1234_5678,
            _ => -0x1234_5678,
        }
    }

    #[test]
    fn shader_and_uniform_abi_are_semantically_valid() {
        fn assert_pod<T: Pod>() {}
        assert_pod::<ModularRctParams>();
        assert_eq!(std::mem::size_of::<ModularRctParams>(), 64);
        assert_eq!(std::mem::align_of::<ModularRctParams>(), 16);
        let module =
            naga::front::wgsl::parse_str(MODULAR_RCT_SHADER).expect("Modular RCT WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("Modular RCT WGSL validates with portable capabilities");
    }

    #[test]
    fn scalar_oracle_fixes_every_operation_and_permutation() {
        assert_eq!(scalar_transform(8, 4, 2, 0), [8, 4, 2]);
        assert_eq!(scalar_transform(8, 4, 2, 1), [8, 4, 10]);
        assert_eq!(scalar_transform(8, 4, 2, 2), [8, 12, 2]);
        assert_eq!(scalar_transform(8, 4, 2, 3), [8, 12, 10]);
        assert_eq!(scalar_transform(8, 4, 2, 4), [8, 9, 2]);
        assert_eq!(scalar_transform(8, 4, 2, 5), [8, 13, 10]);
        assert_eq!(scalar_transform(8, 4, 2, 6), [9, 9, 5]);

        assert_eq!(scalar_transform(7, -3, 11, 0), [7, -3, 11]);
        assert_eq!(scalar_transform(7, -3, 11, 7), [11, 7, -3]);
        assert_eq!(scalar_transform(7, -3, 11, 14), [-3, 11, 7]);
        assert_eq!(scalar_transform(7, -3, 11, 21), [7, 11, -3]);
        assert_eq!(scalar_transform(7, -3, 11, 28), [-3, 7, 11]);
        assert_eq!(scalar_transform(7, -3, 11, 35), [11, -3, 7]);
    }

    #[test]
    fn parameters_require_equal_non_overlapping_planes_and_valid_type() {
        let first = ModularRctPlane {
            width: 5,
            height: 3,
            stride: 7,
            offset_words: 3,
        };
        let second = ModularRctPlane {
            width: 5,
            height: 3,
            stride: 9,
            offset_words: 32,
        };
        let third = ModularRctPlane {
            width: 5,
            height: 3,
            stride: 11,
            offset_words: 64,
        };
        let params = ModularRctParams::new(41, first, second, third);
        assert_eq!(validate_params(params).unwrap(), 15);
        assert_eq!(params.planes(), [first, second, third]);

        let mut invalid = params;
        invalid.operation[0] = 42;
        assert_eq!(
            validate_params(invalid).unwrap_err(),
            ModularRctError::InvalidRctType { rct_type: 42 }
        );

        let mut invalid = params;
        invalid.first[0] = 4;
        assert!(matches!(
            validate_params(invalid),
            Err(ModularRctError::UnequalGeometry {
                plane: "second",
                ..
            })
        ));

        let mut invalid = params;
        invalid.third[3] = 20;
        assert_eq!(
            validate_params(invalid).unwrap_err(),
            ModularRctError::PlaneOverlap {
                first: "first",
                second: "third"
            }
        );
    }

    #[test]
    fn linear_kernel_variants_are_the_supported_policy_domain() {
        let limits = wgpu::Limits::default();
        for variant in MODULAR_RCT_VARIANTS {
            assert!(validate_variant(variant, &limits).is_ok());
        }
        assert_eq!(
            validate_variant(KernelVariant::Tile8x8, &limits).unwrap_err(),
            ModularRctError::WorkgroupShape {
                variant: KernelVariant::Tile8x8
            }
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn actual_adapter_matches_scalar_oracle_for_all_42_types() {
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
            eprintln!("skipping Modular RCT GPU test: no adapter");
            return;
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("jxl-wgpu Modular RCT test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))
        else {
            eprintln!("skipping Modular RCT GPU test: device request failed");
            return;
        };

        let variant = if validate_variant(KernelVariant::Lanes64, &device.limits()).is_ok() {
            KernelVariant::Lanes64
        } else {
            KernelVariant::Scalar
        };
        let pipeline = ModularRctPipeline::with_variant(&device, variant).unwrap();
        let planes = [
            ModularRctPlane {
                width: 5,
                height: 3,
                stride: 7,
                offset_words: 3,
            },
            ModularRctPlane {
                width: 5,
                height: 3,
                stride: 9,
                offset_words: 32,
            },
            ModularRctPlane {
                width: 5,
                height: 3,
                stride: 11,
                offset_words: 64,
            },
        ];
        let arena_words = 128usize;
        let arena_bytes = (arena_words * std::mem::size_of::<u32>()) as u64;

        for rct_type in 0..MODULAR_RCT_TYPE_COUNT {
            let mut words = vec![0u32; arena_words];
            for (plane_index, plane) in planes.into_iter().enumerate() {
                for y in 0..plane.height {
                    for x in 0..plane.width {
                        let index =
                            usize::try_from(plane.offset_words + y * plane.stride + x).unwrap();
                        words[index] = pattern(plane_index, x, y) as u32;
                    }
                }
            }
            let expected = (0..planes[0].height)
                .flat_map(|y| {
                    (0..planes[0].width).map(move |x| {
                        scalar_transform(
                            pattern(0, x, y),
                            pattern(1, x, y),
                            pattern(2, x, y),
                            rct_type,
                        )
                    })
                })
                .collect::<Vec<_>>();

            let arena_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Modular RCT GPU arena"),
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Modular RCT GPU staging"),
                size: arena_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let arena = ModularRctArena::entire(&arena_buffer).unwrap();
            let params = ModularRctParams::new(rct_type, planes[0], planes[1], planes[2]);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Modular RCT GPU encoder"),
            });
            let uniform = pipeline
                .encode(&device, &mut encoder, arena, params)
                .unwrap();
            encoder.copy_buffer_to_buffer(&arena_buffer, 0, &staging, 0, arena_bytes);
            queue.submit(Some(encoder.finish()));
            drop(uniform);
            let slice = staging.slice(..arena_bytes);
            let (sender, receiver) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            receiver.recv().unwrap().unwrap();
            let mapped = slice.get_mapped_range().expect("mapped Modular RCT output");
            let actual: &[u32] = bytemuck::cast_slice(&mapped);
            for (plane_index, plane) in planes.into_iter().enumerate() {
                for y in 0..plane.height {
                    for x in 0..plane.width {
                        let index =
                            usize::try_from(plane.offset_words + y * plane.stride + x).unwrap();
                        let pixel = usize::try_from(y * plane.width + x).unwrap();
                        assert_eq!(
                            actual[index] as i32, expected[pixel][plane_index],
                            "RCT type {rct_type}, plane {plane_index}, x {x}, y {y}",
                        );
                    }
                }
            }
            drop(mapped);
            staging.unmap();
        }
    }
}
