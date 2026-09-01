//! GPU-resident inverse JPEG XL Modular Squeeze.
//!
//! A Squeeze inverse consumes one average channel and one residual channel and writes the
//! reconstructed channel into a third, non-overlapping view of one GPU-resident storage arena.
//! Using one read-write binding avoids WebGPU resource-alias ambiguity when a lifetime allocator
//! reuses different regions of the same buffer between inverse passes.
//!
//! The kernel follows the scalar inverse in `jxl 0.6`/libjxl.  Its smooth tendency is evaluated
//! with enough precision for the full signed i32 input domain, while reconstructed samples are
//! narrowed with two's-complement (wrapping) i32 semantics.  Horizontal and vertical passes use
//! one invocation per row or column, respectively, because each pair depends on the preceding
//! reconstructed residual in that line.

use bytemuck::{Pod, Zeroable};
use jxl_wgpu::{KernelVariant, ResidentStorageBinding};
use thiserror::Error;
use wgpu::util::DeviceExt;

/// WGSL for [`ModularSqueezePipeline`].
pub const MODULAR_SQUEEZE_SHADER: &str = r#"
override wg_x: u32 = 64u;

struct Params {
    // width, height, row stride (in i32 words), and offset (in words)
    average: vec4<u32>,
    residual: vec4<u32>,
    output: vec4<u32>,
    // x is 0 for horizontal and 1 for vertical.  The remaining words are reserved.
    operation: vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> arena: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

// WGSL has no portable i64.  These helpers hold an unsigned 64-bit magnitude in little-endian
// words.  The largest intermediate in smooth tendency is below 2^35, but retaining the full
// pair keeps all additions and boundary clamps explicit and auditable.
struct U64 {
    lo: u32,
    hi: u32,
};

struct S64 {
    magnitude: U64,
    negative: bool,
};

struct Pair {
    first: i32,
    second: i32,
};

fn u64_zero() -> U64 {
    return U64(0u, 0u);
}

fn u64_is_zero(value: U64) -> bool {
    return value.lo == 0u && value.hi == 0u;
}

fn u64_add(left: U64, right: U64) -> U64 {
    let lo = left.lo + right.lo;
    let carry = select(0u, 1u, lo < left.lo);
    return U64(lo, left.hi + right.hi + carry);
}

// The caller only supplies left >= right.
fn u64_sub(left: U64, right: U64) -> U64 {
    let borrow = select(0u, 1u, left.lo < right.lo);
    return U64(left.lo - right.lo, left.hi - right.hi - borrow);
}

fn u64_gt(left: U64, right: U64) -> bool {
    return left.hi > right.hi || (left.hi == right.hi && left.lo > right.lo);
}

fn u64_mul_small(value: u32, factor: u32) -> U64 {
    // Two 16-bit limbs keep every product below 2^32 before the carry is extracted.
    let low_limb = (value & 0xffffu) * factor;
    let high_limb = (value >> 16u) * factor + (low_limb >> 16u);
    return U64(
        (low_limb & 0xffffu) | ((high_limb & 0xffffu) << 16u),
        high_limb >> 16u,
    );
}

fn u64_div_small(value: U64, divisor: u32) -> U64 {
    // Long division in base 2^16.  The running remainder is below divisor, so the temporary
    // (remainder * 65536 + limb) is bounded for the divisor 12 used below.
    var remainder = 0u;
    let digit3 = value.hi >> 16u;
    let quotient3 = (remainder * 65536u + digit3) / divisor;
    remainder = remainder * 65536u + digit3 - quotient3 * divisor;
    let digit2 = value.hi & 0xffffu;
    let quotient2 = (remainder * 65536u + digit2) / divisor;
    remainder = remainder * 65536u + digit2 - quotient2 * divisor;
    let digit1 = value.lo >> 16u;
    let quotient1 = (remainder * 65536u + digit1) / divisor;
    remainder = remainder * 65536u + digit1 - quotient1 * divisor;
    let digit0 = value.lo & 0xffffu;
    let quotient0 = (remainder * 65536u + digit0) / divisor;
    return U64(
        (quotient1 << 16u) | quotient0,
        (quotient3 << 16u) | quotient2,
    );
}

fn u64_shift_right_one(value: U64) -> U64 {
    return U64((value.lo >> 1u) | (value.hi << 31u), value.hi >> 1u);
}

fn u64_odd(value: U64) -> U64 {
    return U64(value.lo & 1u, 0u);
}

fn s64_normalize(magnitude: U64, negative: bool) -> S64 {
    return S64(magnitude, negative && !u64_is_zero(magnitude));
}

fn s64_from_i32(value: i32) -> S64 {
    let bits = bitcast<u32>(value);
    if value < 0i {
        return S64(U64(0u - bits, 0u), true);
    }
    return S64(U64(bits, 0u), false);
}

fn s64_add(left: S64, right: S64) -> S64 {
    if left.negative == right.negative {
        return s64_normalize(u64_add(left.magnitude, right.magnitude), left.negative);
    }
    if u64_gt(left.magnitude, right.magnitude) {
        return s64_normalize(u64_sub(left.magnitude, right.magnitude), left.negative);
    }
    return s64_normalize(u64_sub(right.magnitude, left.magnitude), right.negative);
}

fn s64_sub(left: S64, right: S64) -> S64 {
    return s64_add(left, s64_normalize(right.magnitude, !right.negative));
}

fn s64_div_two(value: S64) -> S64 {
    return s64_normalize(u64_shift_right_one(value.magnitude), value.negative);
}

fn s64_low_i32(value: S64) -> i32 {
    var bits = value.magnitude.lo;
    if value.negative {
        bits = 0u - bits;
    }
    return bitcast<i32>(bits);
}

// This is jxl 0.6's smooth_tendency_scalar, rewritten in terms of non-negative differences.
// For a monotone decreasing triplet, d0 = prev - avg and d1 = avg - next, so
//   4*prev - 3*next - avg + 6 = 4*d0 + 3*d1 + 6.
// For an increasing triplet the corresponding expression is the negative of that quantity.
// This algebra avoids signed overflow while retaining the scalar's truncation-toward-zero
// division and exact clamp inequalities.
fn smooth_tendency(prev: i32, avg: i32, next_avg: i32) -> S64 {
    if prev >= avg && avg >= next_avg {
        let first_delta = bitcast<u32>(prev) - bitcast<u32>(avg);
        let second_delta = bitcast<u32>(avg) - bitcast<u32>(next_avg);
        var tendency = u64_div_small(
            u64_add(
                u64_add(
                    u64_mul_small(first_delta, 4u),
                    u64_mul_small(second_delta, 3u),
                ),
                U64(6u, 0u),
            ),
            12u,
        );
        let odd = u64_odd(tendency);
        let first_limit = u64_mul_small(first_delta, 2u);
        if u64_gt(u64_sub(tendency, odd), first_limit) {
            tendency = u64_add(first_limit, U64(1u, 0u));
        }
        let second_limit = u64_mul_small(second_delta, 2u);
        if u64_gt(u64_add(tendency, odd), second_limit) {
            tendency = second_limit;
        }
        return S64(tendency, false);
    }
    if prev <= avg && avg <= next_avg {
        let first_delta = bitcast<u32>(avg) - bitcast<u32>(prev);
        let second_delta = bitcast<u32>(next_avg) - bitcast<u32>(avg);
        var magnitude = u64_div_small(
            u64_add(
                u64_add(
                    u64_mul_small(first_delta, 4u),
                    u64_mul_small(second_delta, 3u),
                ),
                U64(6u, 0u),
            ),
            12u,
        );
        let odd = u64_odd(magnitude);
        let first_limit = u64_mul_small(first_delta, 2u);
        // (-x + (x & 1)) < -2*d0  is equivalent to x - (x & 1) > 2*d0.
        if u64_gt(u64_sub(magnitude, odd), first_limit) {
            magnitude = u64_add(first_limit, U64(1u, 0u));
        }
        let second_limit = u64_mul_small(second_delta, 2u);
        // (-x - (x & 1)) < -2*d1  is equivalent to x + (x & 1) > 2*d1.
        if u64_gt(u64_add(magnitude, odd), second_limit) {
            magnitude = second_limit;
        }
        return S64(magnitude, true);
    }
    return S64(u64_zero(), false);
}

fn unsqueeze(avg: i32, residual: i32, next_avg: i32, previous: i32) -> Pair {
    let tendency = smooth_tendency(previous, avg, next_avg);
    let difference = s64_add(s64_from_i32(residual), tendency);
    let first_wide = s64_add(s64_from_i32(avg), s64_div_two(difference));
    let second_wide = s64_sub(first_wide, difference);
    return Pair(s64_low_i32(first_wide), s64_low_i32(second_wide));
}

fn load_average(index: u32) -> i32 {
    return bitcast<i32>(arena[index]);
}

fn load_residual(index: u32) -> i32 {
    return bitcast<i32>(arena[index]);
}

fn store_output(index: u32, value: i32) {
    arena[index] = bitcast<u32>(value);
}

fn inverse_horizontal(line: u32) {
    let average_width = params.average.x;
    let residual_width = params.residual.x;
    let average_base = params.average.w + line * params.average.z;
    let residual_base = params.residual.w + line * params.residual.z;
    let output_base = params.output.w + line * params.output.z;
    var previous = load_average(average_base);
    var x = 0u;
    loop {
        if x >= residual_width {
            break;
        }
        let average = load_average(average_base + x);
        var next_average = average;
        if x + 1u < average_width {
            next_average = load_average(average_base + x + 1u);
        } else if (params.output.x & 1u) != 0u {
            next_average = load_average(average_base + residual_width);
        }
        let pair = unsqueeze(average, load_residual(residual_base + x), next_average, previous);
        store_output(output_base + 2u * x, pair.first);
        store_output(output_base + 2u * x + 1u, pair.second);
        previous = pair.second;
        x += 1u;
    }
    if (params.output.x & 1u) != 0u {
        store_output(output_base + params.output.x - 1u, load_average(average_base + average_width - 1u));
    }
}

fn inverse_vertical(line: u32) {
    let average_height = params.average.y;
    let residual_height = params.residual.y;
    var previous = load_average(params.average.w + line);
    var y = 0u;
    loop {
        if y >= residual_height {
            break;
        }
        let average_base = params.average.w + y * params.average.z + line;
        let residual_base = params.residual.w + y * params.residual.z + line;
        let output_base = params.output.w + (2u * y) * params.output.z + line;
        let average = load_average(average_base);
        var next_average = average;
        if y + 1u < average_height {
            next_average = load_average(params.average.w + (y + 1u) * params.average.z + line);
        } else if (params.output.y & 1u) != 0u {
            next_average = load_average(params.average.w + residual_height * params.average.z + line);
        }
        let pair = unsqueeze(average, load_residual(residual_base), next_average, previous);
        store_output(output_base, pair.first);
        store_output(output_base + params.output.z, pair.second);
        previous = pair.second;
        y += 1u;
    }
    if (params.output.y & 1u) != 0u {
        store_output(
            params.output.w + (params.output.y - 1u) * params.output.z + line,
            load_average(params.average.w + (average_height - 1u) * params.average.z + line),
        );
    }
}

@compute @workgroup_size(wg_x, 1, 1)
fn inverse_squeeze(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let line = invocation.x;
    if params.operation.x == 0u {
        if line < params.average.y {
            inverse_horizontal(line);
        }
    } else if line < params.average.x {
        inverse_vertical(line);
    }
}
"#;

/// Direction of the inverse Squeeze pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModularSqueezeDirection {
    /// Reconstruct pairs along each row.
    #[default]
    Horizontal,
    /// Reconstruct pairs along each column.
    Vertical,
}

impl ModularSqueezeDirection {
    const fn word(self) -> u32 {
        match self {
            Self::Horizontal => 0,
            Self::Vertical => 1,
        }
    }
}

/// A planar i32 storage view.  `stride` and `offset_words` are measured in i32/u32 words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModularSqueezePlane {
    pub width: u32,
    pub height: u32,
    /// Row stride in words; it may be larger than `width` for a subview.
    pub stride: u32,
    /// Word offset relative to the corresponding [`ResidentStorageBinding`].
    pub offset_words: u32,
}

impl ModularSqueezePlane {
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
            return Some(self.offset_words as u64);
        }
        (self.offset_words as u64)
            .checked_add((self.height - 1) as u64 * self.stride as u64)?
            .checked_add(self.width as u64)
    }
}

/// Exact 64-byte uniform consumed by [`MODULAR_SQUEEZE_SHADER`].
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct ModularSqueezeParams {
    /// width, height, row stride, offset for the average plane.
    pub average: [u32; 4],
    /// width, height, row stride, offset for the residual plane.
    pub residual: [u32; 4],
    /// width, height, row stride, offset for the destination plane.
    pub output: [u32; 4],
    /// direction in word zero (0 horizontal, 1 vertical); remaining words are reserved zero.
    pub operation: [u32; 4],
}

impl ModularSqueezeParams {
    /// Builds the wire uniform from three plane views.
    #[must_use]
    pub const fn new(
        direction: ModularSqueezeDirection,
        average: ModularSqueezePlane,
        residual: ModularSqueezePlane,
        output: ModularSqueezePlane,
    ) -> Self {
        Self {
            average: [
                average.width,
                average.height,
                average.stride,
                average.offset_words,
            ],
            residual: [
                residual.width,
                residual.height,
                residual.stride,
                residual.offset_words,
            ],
            output: [
                output.width,
                output.height,
                output.stride,
                output.offset_words,
            ],
            operation: [direction.word(), 0, 0, 0],
        }
    }

    #[must_use]
    pub const fn direction(self) -> ModularSqueezeDirection {
        if self.operation[0] == 0 {
            ModularSqueezeDirection::Horizontal
        } else {
            ModularSqueezeDirection::Vertical
        }
    }

    #[must_use]
    pub const fn average_plane(self) -> ModularSqueezePlane {
        ModularSqueezePlane {
            width: self.average[0],
            height: self.average[1],
            stride: self.average[2],
            offset_words: self.average[3],
        }
    }

    #[must_use]
    pub const fn residual_plane(self) -> ModularSqueezePlane {
        ModularSqueezePlane {
            width: self.residual[0],
            height: self.residual[1],
            stride: self.residual[2],
            offset_words: self.residual[3],
        }
    }

    #[must_use]
    pub const fn output_plane(self) -> ModularSqueezePlane {
        ModularSqueezePlane {
            width: self.output[0],
            height: self.output[1],
            stride: self.output[2],
            offset_words: self.output[3],
        }
    }
}

/// Storage arena supplied to the inverse Squeeze pass.
///
/// All plane offsets are relative to this binding. Average, residual, and output footprints must
/// be pairwise disjoint for the duration of the dispatch.
#[derive(Clone, Copy, Debug)]
pub struct ModularSqueezeArena<'a> {
    pub storage: ResidentStorageBinding<'a>,
}

impl<'a> ModularSqueezeArena<'a> {
    /// Uses one complete non-empty storage buffer as the arena.
    pub fn entire(buffer: &'a wgpu::Buffer) -> Result<Self, ModularSqueezeError> {
        let size =
            std::num::NonZeroU64::new(buffer.size()).ok_or(ModularSqueezeError::EmptyBinding)?;
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

/// Host-side dispatch geometry after checking one inverse Squeeze operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModularSqueezePlan {
    pub params: ModularSqueezeParams,
    pub line_count: u32,
    pub workgroups: u32,
    pub variant: KernelVariant,
}

/// Typed planning and recording failures for [`ModularSqueezePipeline`].
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ModularSqueezeError {
    #[error("Modular Squeeze arena binding is empty")]
    EmptyBinding,
    #[error("Modular Squeeze arena buffer is missing STORAGE usage")]
    MissingStorageUsage,
    #[error("Modular Squeeze arena binding offset {offset} is not aligned to {alignment}")]
    BindingAlignment { offset: u64, alignment: u64 },
    #[error("Modular Squeeze arena binding range {offset}..{end} exceeds buffer size {available}")]
    BindingRange {
        offset: u64,
        end: u64,
        available: u64,
    },
    #[error("Modular Squeeze arena binding size {size} is not a multiple of four bytes")]
    BindingSizeAlignment { size: u64 },
    #[error("Modular Squeeze {plane} needs {required} bytes, binding has {available}")]
    BindingSize {
        plane: &'static str,
        required: u64,
        available: u64,
    },
    #[error("Modular Squeeze arena binding needs {required} bytes, device permits {available}")]
    StorageBindingLimit { required: u64, available: u64 },
    #[error("Modular Squeeze uniform needs {required} bytes, device permits {available}")]
    UniformBindingLimit { required: u64, available: u64 },
    #[error("Modular Squeeze {plane} has a zero {axis} extent")]
    ZeroExtent {
        plane: &'static str,
        axis: &'static str,
    },
    #[error("Modular Squeeze {plane} stride {stride} is smaller than width {width}")]
    InvalidStride {
        plane: &'static str,
        stride: u32,
        width: u32,
    },
    #[error("Modular Squeeze average and residual planes have incompatible geometry")]
    IncompatibleGeometry,
    #[error(
        "Modular Squeeze output plane has geometry {actual_width}x{actual_height}, expected {expected_width}x{expected_height}"
    )]
    OutputGeometry {
        actual_width: u32,
        actual_height: u32,
        expected_width: u32,
        expected_height: u32,
    },
    #[error("Modular Squeeze {plane} address range exceeds WGSL's u32 word address space")]
    ShaderAddressSpace { plane: &'static str },
    #[error("Modular Squeeze {first} and {second} plane footprints overlap")]
    PlaneOverlap {
        first: &'static str,
        second: &'static str,
    },
    #[error("Modular Squeeze arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
    #[error("Modular Squeeze requires a linear workgroup, got {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    #[error("Modular Squeeze workgroup variant {variant:?} exceeds device limits")]
    WorkgroupVariant { variant: KernelVariant },
    #[error("Modular Squeeze dispatch needs {required} workgroups, device permits {available}")]
    WorkgroupCount { required: u32, available: u32 },
    #[error("Modular Squeeze direction word {direction} is invalid")]
    InvalidDirection { direction: u32 },
    #[error("Modular Squeeze reserved parameter word {word} must be zero")]
    NonZeroReservedParameter { word: usize },
}

/// Reusable compute pipeline for one GPU-resident inverse Squeeze operation.
pub struct ModularSqueezePipeline {
    pipeline: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl ModularSqueezePipeline {
    /// Compiles the default 64-lane linear kernel.
    pub fn new(device: &wgpu::Device) -> Result<Self, ModularSqueezeError> {
        Self::with_variant(device, KernelVariant::Lanes64)
    }

    /// Compiles the kernel with a linear [`KernelVariant`] (Scalar/Lanes32/64/128/256).
    pub fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, ModularSqueezeError> {
        validate_variant(variant, &device.limits())?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu decode Modular inverse Squeeze"),
            source: wgpu::ShaderSource::Wgsl(MODULAR_SQUEEZE_SHADER.into()),
        });
        let constants = [("wg_x", f64::from(variant.workgroup_size().0))];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu decode Modular inverse Squeeze"),
            layout: None,
            module: &module,
            entry_point: Some("inverse_squeeze"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
        Ok(Self { pipeline, variant })
    }

    /// Records one horizontal or vertical inverse operation and returns its uniform allocation.
    ///
    /// All samples stay in storage buffers owned by the caller.  The returned uniform must be
    /// retained until command submission, matching the lifetime contract of the other standalone
    /// resident decode pipelines.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        arena: ModularSqueezeArena<'_>,
        params: ModularSqueezeParams,
    ) -> Result<wgpu::Buffer, ModularSqueezeError> {
        let plan = plan_for_device(device, arena, params, self.variant)?;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu decode Modular inverse Squeeze params"),
            contents: bytemuck::bytes_of(&plan.params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu decode Modular inverse Squeeze bindings"),
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
            label: Some("jxl-wgpu decode Modular inverse Squeeze"),
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

fn validate_variant(
    variant: KernelVariant,
    limits: &wgpu::Limits,
) -> Result<(), ModularSqueezeError> {
    if !variant.is_linear() {
        return Err(ModularSqueezeError::WorkgroupShape { variant });
    }
    variant
        .validate_for("modular_squeeze", limits, 0)
        .map_err(|_| ModularSqueezeError::WorkgroupVariant { variant })
}

fn validate_params(params: ModularSqueezeParams) -> Result<(u32, u32), ModularSqueezeError> {
    if params.operation[0] > 1 {
        return Err(ModularSqueezeError::InvalidDirection {
            direction: params.operation[0],
        });
    }
    for (word, value) in params.operation[1..].iter().copied().enumerate() {
        if value != 0 {
            return Err(ModularSqueezeError::NonZeroReservedParameter { word: word + 1 });
        }
    }
    let average = params.average_plane();
    let residual = params.residual_plane();
    let output = params.output_plane();
    if average.width == 0 {
        return Err(ModularSqueezeError::ZeroExtent {
            plane: "average",
            axis: "width",
        });
    }
    if average.height == 0 {
        return Err(ModularSqueezeError::ZeroExtent {
            plane: "average",
            axis: "height",
        });
    }
    for (plane, name) in [
        (average, "average"),
        (residual, "residual"),
        (output, "output"),
    ] {
        if plane.stride < plane.width {
            return Err(ModularSqueezeError::InvalidStride {
                plane: name,
                stride: plane.stride,
                width: plane.width,
            });
        }
    }
    let direction = params.direction();
    let (line_count, expected_width, expected_height) = match direction {
        ModularSqueezeDirection::Horizontal => {
            if average.height != residual.height || residual.width > average.width {
                return Err(ModularSqueezeError::IncompatibleGeometry);
            }
            let expected_width = average.width.checked_add(residual.width).ok_or(
                ModularSqueezeError::ArithmeticOverflow {
                    field: "horizontal output width",
                },
            )?;
            if average.width - residual.width > 1 {
                return Err(ModularSqueezeError::IncompatibleGeometry);
            }
            (average.height, expected_width, average.height)
        }
        ModularSqueezeDirection::Vertical => {
            if average.width != residual.width || residual.height > average.height {
                return Err(ModularSqueezeError::IncompatibleGeometry);
            }
            let expected_height = average.height.checked_add(residual.height).ok_or(
                ModularSqueezeError::ArithmeticOverflow {
                    field: "vertical output height",
                },
            )?;
            if average.height - residual.height > 1 {
                return Err(ModularSqueezeError::IncompatibleGeometry);
            }
            (average.width, average.width, expected_height)
        }
    };
    if output.width != expected_width || output.height != expected_height {
        return Err(ModularSqueezeError::OutputGeometry {
            actual_width: output.width,
            actual_height: output.height,
            expected_width,
            expected_height,
        });
    }
    for (plane, name) in [
        (average, "average"),
        (residual, "residual"),
        (output, "output"),
    ] {
        let Some(required_words) = plane.required_words() else {
            return Err(ModularSqueezeError::ArithmeticOverflow {
                field: "plane word range",
            });
        };
        if required_words > u64::from(u32::MAX) {
            return Err(ModularSqueezeError::ShaderAddressSpace { plane: name });
        }
    }
    Ok((line_count, direction.word()))
}

fn validate_non_overlapping_planes(
    params: ModularSqueezeParams,
) -> Result<(), ModularSqueezeError> {
    let planes = [
        ("average", params.average_plane()),
        ("residual", params.residual_plane()),
        ("output", params.output_plane()),
    ];
    for first_index in 0..planes.len() {
        for second_index in (first_index + 1)..planes.len() {
            let (first_name, first) = planes[first_index];
            let (second_name, second) = planes[second_index];
            let first_end =
                first
                    .required_words()
                    .ok_or(ModularSqueezeError::ArithmeticOverflow {
                        field: "plane overlap range",
                    })?;
            let second_end =
                second
                    .required_words()
                    .ok_or(ModularSqueezeError::ArithmeticOverflow {
                        field: "plane overlap range",
                    })?;
            if u64::from(first.offset_words) < second_end
                && u64::from(second.offset_words) < first_end
            {
                return Err(ModularSqueezeError::PlaneOverlap {
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
) -> Result<(), ModularSqueezeError> {
    if binding.size.get() == 0 {
        return Err(ModularSqueezeError::EmptyBinding);
    }
    if !binding.buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
        return Err(ModularSqueezeError::MissingStorageUsage);
    }
    let alignment = u64::from(limits.min_storage_buffer_offset_alignment).max(4);
    if !binding.offset.is_multiple_of(alignment) {
        return Err(ModularSqueezeError::BindingAlignment {
            offset: binding.offset,
            alignment,
        });
    }
    if !binding.offset.is_multiple_of(4) {
        return Err(ModularSqueezeError::BindingAlignment {
            offset: binding.offset,
            alignment: 4,
        });
    }
    let end = binding.offset.checked_add(binding.size.get()).ok_or(
        ModularSqueezeError::ArithmeticOverflow {
            field: "storage binding range",
        },
    )?;
    if end > binding.buffer.size() {
        return Err(ModularSqueezeError::BindingRange {
            offset: binding.offset,
            end,
            available: binding.buffer.size(),
        });
    }
    if !binding.size.get().is_multiple_of(4) {
        return Err(ModularSqueezeError::BindingSizeAlignment {
            size: binding.size.get(),
        });
    }
    if binding.size.get() > limits.max_storage_buffer_binding_size {
        return Err(ModularSqueezeError::StorageBindingLimit {
            required: binding.size.get(),
            available: limits.max_storage_buffer_binding_size,
        });
    }
    Ok(())
}

fn plan_for_device(
    device: &wgpu::Device,
    arena: ModularSqueezeArena<'_>,
    params: ModularSqueezeParams,
    variant: KernelVariant,
) -> Result<ModularSqueezePlan, ModularSqueezeError> {
    validate_variant(variant, &device.limits())?;
    let (line_count, _) = validate_params(params)?;
    validate_binding(arena.storage, &device.limits())?;
    for (plane, name) in [
        (params.average_plane(), "average"),
        (params.residual_plane(), "residual"),
        (params.output_plane(), "output"),
    ] {
        let required = plane
            .required_words()
            .ok_or(ModularSqueezeError::ArithmeticOverflow {
                field: "plane byte range",
            })?
            .checked_mul(4)
            .ok_or(ModularSqueezeError::ArithmeticOverflow {
                field: "plane byte range",
            })?;
        if required > arena.storage.size.get() {
            return Err(ModularSqueezeError::BindingSize {
                plane: name,
                required,
                available: arena.storage.size.get(),
            });
        }
    }
    validate_non_overlapping_planes(params)?;
    let uniform_bytes = std::mem::size_of::<ModularSqueezeParams>() as u64;
    if uniform_bytes > device.limits().max_uniform_buffer_binding_size {
        return Err(ModularSqueezeError::UniformBindingLimit {
            required: uniform_bytes,
            available: device.limits().max_uniform_buffer_binding_size,
        });
    }
    let workgroups = line_count.div_ceil(variant.workgroup_size().0);
    let maximum = device.limits().max_compute_workgroups_per_dimension;
    if workgroups > maximum {
        return Err(ModularSqueezeError::WorkgroupCount {
            required: workgroups,
            available: maximum,
        });
    }
    Ok(ModularSqueezePlan {
        params,
        line_count,
        workgroups,
        variant,
    })
}

const _: () = {
    assert!(std::mem::size_of::<ModularSqueezeParams>() == 64);
    assert!(std::mem::align_of::<ModularSqueezeParams>() == 16);
};

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_tendency(prev: i32, avg: i32, next: i32) -> i64 {
        let (prev, avg, next) = (i64::from(prev), i64::from(avg), i64::from(next));
        let mut tendency = 0;
        if prev >= avg && avg >= next {
            tendency = (4 * prev - 3 * next - avg + 6) / 12;
            if tendency - (tendency & 1) > 2 * (prev - avg) {
                tendency = 2 * (prev - avg) + 1;
            }
            if tendency + (tendency & 1) > 2 * (avg - next) {
                tendency = 2 * (avg - next);
            }
        } else if prev <= avg && avg <= next {
            tendency = (4 * prev - 3 * next - avg - 6) / 12;
            if tendency + (tendency & 1) < 2 * (prev - avg) {
                tendency = 2 * (prev - avg) - 1;
            }
            if tendency - (tendency & 1) < 2 * (avg - next) {
                tendency = 2 * (avg - next);
            }
        }
        tendency
    }

    fn wrap_i32(value: i64) -> i32 {
        value as i32
    }

    fn scalar_pair(avg: i32, residual: i32, next: i32, previous: i32) -> (i32, i32) {
        let difference = i64::from(residual) + scalar_tendency(previous, avg, next);
        let first = i64::from(avg) + difference / 2;
        let second = first - difference;
        (wrap_i32(first), wrap_i32(second))
    }

    fn scalar_horizontal(average: &[i32], residual: &[i32], width: u32, height: u32) -> Vec<i32> {
        let average_width = width.div_ceil(2);
        let residual_width = width / 2;
        let mut output = vec![0; (width * height) as usize];
        for y in 0..height {
            let average_base = (y * average_width) as usize;
            let residual_base = (y * residual_width) as usize;
            let output_base = (y * width) as usize;
            let mut previous = average[average_base];
            for x in 0..residual_width {
                let next = if x + 1 < average_width {
                    average[average_base + x as usize + 1]
                } else if width % 2 == 1 {
                    average[average_base + residual_width as usize]
                } else {
                    average[average_base + x as usize]
                };
                let (first, second) = scalar_pair(
                    average[average_base + x as usize],
                    residual[residual_base + x as usize],
                    next,
                    previous,
                );
                output[output_base + 2 * x as usize] = first;
                output[output_base + 2 * x as usize + 1] = second;
                previous = second;
            }
            if width % 2 == 1 {
                output[output_base + width as usize - 1] =
                    average[average_base + average_width as usize - 1];
            }
        }
        output
    }

    fn scalar_vertical(average: &[i32], residual: &[i32], width: u32, height: u32) -> Vec<i32> {
        let average_height = height.div_ceil(2);
        let residual_height = height / 2;
        let mut output = vec![0; (width * height) as usize];
        for x in 0..width {
            let mut previous = average[x as usize];
            for y in 0..residual_height {
                let average_index = (y * width + x) as usize;
                let residual_index = (y * width + x) as usize;
                let next = if y + 1 < average_height {
                    average[((y + 1) * width + x) as usize]
                } else if height % 2 == 1 {
                    average[(residual_height * width + x) as usize]
                } else {
                    average[average_index]
                };
                let (first, second) = scalar_pair(
                    average[average_index],
                    residual[residual_index],
                    next,
                    previous,
                );
                output[(2 * y * width + x) as usize] = first;
                output[((2 * y + 1) * width + x) as usize] = second;
                previous = second;
            }
            if height % 2 == 1 {
                output[((height - 1) * width + x) as usize] =
                    average[((average_height - 1) * width + x) as usize];
            }
        }
        output
    }

    fn edge_values(length: usize, seed: i32) -> Vec<i32> {
        (0..length)
            .map(|index| match index % 7 {
                0 => i32::MIN,
                1 => i32::MAX,
                2 => seed.wrapping_mul(31),
                3 => seed.wrapping_add(index as i32),
                4 => -seed,
                5 => 0,
                _ => 1,
            })
            .collect()
    }

    #[test]
    fn abi_and_semantic_wgsl_validation() {
        assert_eq!(std::mem::size_of::<ModularSqueezeParams>(), 64);
        assert_eq!(std::mem::align_of::<ModularSqueezeParams>(), 16);
        fn assert_pod<T: Pod>() {}
        assert_pod::<ModularSqueezeParams>();

        let module = naga::front::wgsl::parse_str(MODULAR_SQUEEZE_SHADER)
            .expect("Modular Squeeze WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("Modular Squeeze WGSL validates with portable capabilities");
    }

    #[test]
    fn geometry_planning_covers_odd_even_and_one_dimensional_cases() {
        let horizontal = ModularSqueezeParams::new(
            ModularSqueezeDirection::Horizontal,
            ModularSqueezePlane::tight(3, 1),
            ModularSqueezePlane::tight(2, 1),
            ModularSqueezePlane::tight(5, 1),
        );
        assert_eq!(validate_params(horizontal).unwrap(), (1, 0));

        let vertical = ModularSqueezeParams::new(
            ModularSqueezeDirection::Vertical,
            ModularSqueezePlane::tight(1, 3),
            ModularSqueezePlane::tight(1, 2),
            ModularSqueezePlane::tight(1, 5),
        );
        assert_eq!(validate_params(vertical).unwrap(), (1, 1));

        let one_pixel = ModularSqueezeParams::new(
            ModularSqueezeDirection::Horizontal,
            ModularSqueezePlane::tight(1, 7),
            ModularSqueezePlane::tight(0, 7),
            ModularSqueezePlane::tight(1, 7),
        );
        assert_eq!(validate_params(one_pixel).unwrap(), (7, 0));
    }

    #[test]
    fn scalar_oracle_exercises_smooth_edges_extremes_wrapping_and_odd_shapes() {
        for &(width, height) in &[
            (1u32, 1u32),
            (2, 1),
            (3, 1),
            (7, 3),
            (8, 2),
            (9, 5),
            (17, 4),
        ] {
            let average_width = width.div_ceil(2) as usize;
            let residual_width = (width / 2) as usize;
            let average = edge_values(average_width * height as usize, width as i32);
            let residual = edge_values(residual_width * height as usize, -(height as i32));
            let output = scalar_horizontal(&average, &residual, width, height);
            assert_eq!(output.len(), (width * height) as usize);

            let average_height = height.div_ceil(2) as usize;
            let residual_height = (height / 2) as usize;
            let average = edge_values(average_height * width as usize, height as i32);
            let residual = edge_values(residual_height * width as usize, -(width as i32));
            let output = scalar_vertical(&average, &residual, width, height);
            assert_eq!(output.len(), (width * height) as usize);
        }

        for &(prev, avg, next) in &[
            (0, 0, 0),
            (10, 5, 2),
            (-10, -5, -2),
            (i32::MIN, i32::MIN, i32::MIN),
            (i32::MAX, i32::MAX, i32::MAX),
            (i32::MAX, 0, i32::MIN),
            (i32::MIN, 0, i32::MAX),
        ] {
            let tendency = scalar_tendency(prev, avg, next);
            assert!(tendency >= i64::from(i32::MIN) - (1_i64 << 34));
            let _ = scalar_pair(avg, 0, next, prev);
        }
    }

    #[test]
    fn malformed_parameters_have_typed_failures() {
        let mut params = ModularSqueezeParams::new(
            ModularSqueezeDirection::Horizontal,
            ModularSqueezePlane::tight(3, 1),
            ModularSqueezePlane::tight(1, 1),
            ModularSqueezePlane::tight(4, 1),
        );
        params.operation[0] = 2;
        assert_eq!(
            validate_params(params).unwrap_err(),
            ModularSqueezeError::InvalidDirection { direction: 2 }
        );

        let mut params = ModularSqueezeParams::new(
            ModularSqueezeDirection::Horizontal,
            ModularSqueezePlane::tight(3, 1),
            ModularSqueezePlane::tight(1, 1),
            ModularSqueezePlane::tight(4, 1),
        );
        params.operation[1] = 1;
        assert_eq!(
            validate_params(params).unwrap_err(),
            ModularSqueezeError::NonZeroReservedParameter { word: 1 }
        );

        let overlapping = ModularSqueezeParams::new(
            ModularSqueezeDirection::Horizontal,
            ModularSqueezePlane::tight(3, 1),
            ModularSqueezePlane {
                width: 2,
                height: 1,
                stride: 2,
                offset_words: 2,
            },
            ModularSqueezePlane {
                width: 5,
                height: 1,
                stride: 5,
                offset_words: 5,
            },
        );
        assert_eq!(
            validate_non_overlapping_planes(overlapping).unwrap_err(),
            ModularSqueezeError::PlaneOverlap {
                first: "average",
                second: "residual",
            }
        );
    }

    #[test]
    fn linear_kernel_variants_are_the_supported_policy_domain() {
        let limits = wgpu::Limits::default();
        for variant in [
            KernelVariant::Scalar,
            KernelVariant::Lanes32,
            KernelVariant::Lanes64,
            KernelVariant::Lanes128,
            KernelVariant::Lanes256,
        ] {
            assert!(validate_variant(variant, &limits).is_ok());
        }
        assert_eq!(
            validate_variant(KernelVariant::Tile8x8, &limits).unwrap_err(),
            ModularSqueezeError::WorkgroupShape {
                variant: KernelVariant::Tile8x8
            }
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn actual_adapter_matches_scalar_oracle_for_horizontal_and_vertical() {
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
            eprintln!("skipping Modular Squeeze GPU test: no adapter");
            return;
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("jxl-wgpu Modular Squeeze test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))
        else {
            eprintln!("skipping Modular Squeeze GPU test: device request failed");
            return;
        };

        let run = |direction: ModularSqueezeDirection, width: u32, height: u32| {
            let average_width = if direction == ModularSqueezeDirection::Horizontal {
                width.div_ceil(2)
            } else {
                width
            };
            let average_height = if direction == ModularSqueezeDirection::Vertical {
                height.div_ceil(2)
            } else {
                height
            };
            let residual_width = if direction == ModularSqueezeDirection::Horizontal {
                width / 2
            } else {
                width
            };
            let residual_height = if direction == ModularSqueezeDirection::Vertical {
                height / 2
            } else {
                height
            };
            let average = edge_values((average_width * average_height) as usize, 19);
            let residual = edge_values((residual_width * residual_height) as usize, -37);
            let expected = if direction == ModularSqueezeDirection::Horizontal {
                scalar_horizontal(&average, &residual, width, height)
            } else {
                scalar_vertical(&average, &residual, width, height)
            };
            let output_len = (width * height) as usize;
            let residual_offset = average.len();
            let output_offset = residual_offset + residual.len();
            let mut arena_words = Vec::with_capacity(output_offset + output_len);
            arena_words.extend_from_slice(&average);
            arena_words.extend_from_slice(&residual);
            arena_words.resize(output_offset + output_len, 0);
            let arena_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Modular Squeeze GPU arena"),
                contents: bytemuck::cast_slice(&arena_words),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Modular Squeeze GPU staging"),
                size: (output_len.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let arena = ModularSqueezeArena::entire(&arena_buffer).unwrap();
            let params = ModularSqueezeParams::new(
                direction,
                ModularSqueezePlane {
                    offset_words: 0,
                    ..ModularSqueezePlane::tight(average_width, average_height)
                },
                ModularSqueezePlane {
                    offset_words: residual_offset as u32,
                    ..ModularSqueezePlane::tight(residual_width, residual_height)
                },
                ModularSqueezePlane {
                    offset_words: output_offset as u32,
                    ..ModularSqueezePlane::tight(width, height)
                },
            );
            let pipeline =
                ModularSqueezePipeline::with_variant(&device, KernelVariant::Lanes64).unwrap();
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Modular Squeeze GPU encoder"),
            });
            let uniform = pipeline
                .encode(&device, &mut encoder, arena, params)
                .unwrap();
            encoder.copy_buffer_to_buffer(
                &arena_buffer,
                (output_offset * 4) as u64,
                &staging,
                0,
                (output_len * 4) as u64,
            );
            queue.submit(Some(encoder.finish()));
            drop(uniform);
            let slice = staging.slice(..(output_len * 4) as u64);
            let (sender, receiver) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            receiver.recv().unwrap().unwrap();
            let mapped = slice
                .get_mapped_range()
                .expect("mapped Modular Squeeze output");
            let actual: Vec<i32> = bytemuck::cast_slice(&mapped).to_vec();
            drop(mapped);
            staging.unmap();
            assert_eq!(actual, expected);
        };

        run(ModularSqueezeDirection::Horizontal, 9, 5);
        run(ModularSqueezeDirection::Vertical, 7, 9);
        run(ModularSqueezeDirection::Horizontal, 1, 7);
        run(ModularSqueezeDirection::Vertical, 7, 1);
    }
}
