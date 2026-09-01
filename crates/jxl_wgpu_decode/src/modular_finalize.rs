//! Final packing from GPU-resident inverse-Modular source planes.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{KernelVariant, ResidentStorageBinding};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::modular_transform::GpuModularChannelLayout;

const SHADER: &str = include_str!("modular_finalize.wgsl");
const F64_BINDING_MARKER: &str = "/*__JXL_F64_BINDING__*/";
const F64_OUTPUT_MARKER: &str = "/*__JXL_F64_OUTPUT__*/";
const F64_NATIVE_BINDING: &str =
    "@group(0) @binding(4) var<storage, read_write> output_f64: array<f64>;";
const F64_EXACT_OUTPUT: &str = r#"
                let words = widen_normalized_f32_to_f64_words(normalized_bits);
                write_word(offset, words.x);
                write_word(offset + 4u, words.y);
"#;
const F64_NATIVE_OUTPUT: &str = r#"
                if (offset & 7u) == 0u && offset <= params.bounds.x
                    && params.bounds.x - offset >= 8u {
                    output_f64[offset >> 3u] = f64(sample) / 255.0;
                }
"#;

pub(crate) const MODULAR_FINALIZE_KERNEL_KEY: &str = "modular_finalize";
pub(crate) const DEFAULT_MODULAR_FINALIZE_VARIANT: KernelVariant = KernelVariant::Lanes64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModularFinalizeF64Path {
    ExactF32Widening,
    NativeArithmetic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularFinalizeOutput {
    pub kind: u32,
    pub transfer: u32,
    pub limited_range: bool,
    pub channels: u32,
    pub order: u32,
    pub bits: u32,
    pub storage_bits: u32,
    pub numeric_mapping: u32,
    pub plane_offsets: [u32; 4],
    pub plane_strides: [u32; 4],
    pub logical_size: u32,
    pub chroma_extent: Extent2d,
}

/// Uniform shared with `modular_finalize.wgsl`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct ModularFinalizeParams {
    extent: [u32; 4],
    source_offsets: [u32; 4],
    source_strides: [u32; 4],
    output: [u32; 4],
    format: [u32; 4],
    plane01: [u32; 4],
    plane23: [u32; 4],
    bounds: [u32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<ModularFinalizeParams>() == 128);
    assert!(std::mem::align_of::<ModularFinalizeParams>() == 16);
};

impl ModularFinalizeParams {
    pub(crate) fn new(
        extent: Extent2d,
        source_bits: u8,
        source_planes: &[GpuModularChannelLayout],
        arena_words: u32,
        output: ModularFinalizeOutput,
    ) -> Result<Self, ModularFinalizeError> {
        if extent.width == 0 || extent.height == 0 {
            return Err(ModularFinalizeError::InvalidParams {
                reason: "source extent is empty",
            });
        }
        let source_channels = u32::try_from(source_planes.len()).map_err(|_| {
            ModularFinalizeError::InvalidParams {
                reason: "source channel count exceeds u32",
            }
        })?;
        if !matches!(source_channels, 1 | 3 | 4) || !(1..=16).contains(&source_bits) {
            return Err(ModularFinalizeError::InvalidParams {
                reason: "source channel count or bit depth is unsupported",
            });
        }
        let mut source_offsets = [0u32; 4];
        let mut source_strides = [0u32; 4];
        for (index, plane) in source_planes.iter().copied().enumerate() {
            if plane.width != extent.width
                || plane.height != extent.height
                || plane.hshift != 0
                || plane.vshift != 0
                || plane.bit_depth != u32::from(source_bits)
                || plane.row_stride_words < plane.width
                || plane.reserved != 0
            {
                return Err(ModularFinalizeError::InvalidParams {
                    reason: "source plane does not match the final image geometry",
                });
            }
            let end = plane
                .height
                .checked_sub(1)
                .and_then(|rows| rows.checked_mul(plane.row_stride_words))
                .and_then(|words| words.checked_add(plane.word_offset))
                .and_then(|words| words.checked_add(plane.width))
                .ok_or(ModularFinalizeError::InvalidParams {
                    reason: "source plane address overflows u32",
                })?;
            if end > arena_words {
                return Err(ModularFinalizeError::InvalidParams {
                    reason: "source plane exceeds the resident arena",
                });
            }
            source_offsets[index] = plane.word_offset;
            source_strides[index] = plane.row_stride_words;
        }
        validate_output(extent, source_channels, source_bits, output)?;
        extent
            .width
            .checked_mul(extent.height)
            .ok_or(ModularFinalizeError::InvalidParams {
                reason: "source sample count exceeds WGSL u32",
            })?;
        Ok(Self {
            extent: [
                extent.width,
                extent.height,
                source_channels,
                u32::from(source_bits),
            ],
            source_offsets,
            source_strides,
            output: [
                output.kind,
                output.transfer,
                u32::from(output.limited_range),
                output.channels,
            ],
            format: [
                output.order,
                output.bits,
                output.storage_bits,
                output.numeric_mapping,
            ],
            plane01: [
                output.plane_offsets[0],
                output.plane_strides[0],
                output.plane_offsets[1],
                output.plane_strides[1],
            ],
            plane23: [
                output.plane_offsets[2],
                output.plane_strides[2],
                output.plane_offsets[3],
                output.plane_strides[3],
            ],
            bounds: [
                output.logical_size,
                output.chroma_extent.width,
                output.chroma_extent.height,
                0,
            ],
        })
    }

    const fn logical_size(self) -> u32 {
        self.bounds[0]
    }

    fn required_arena_words(self) -> u32 {
        let mut required = 0u32;
        for channel in 0..self.extent[2] as usize {
            let end = self.source_offsets[channel]
                + (self.extent[1] - 1) * self.source_strides[channel]
                + self.extent[0];
            required = required.max(end);
        }
        required
    }
}

fn validate_output(
    extent: Extent2d,
    source_channels: u32,
    source_bits: u8,
    output: ModularFinalizeOutput,
) -> Result<(), ModularFinalizeError> {
    if output.logical_size == 0 || output.channels == 0 || !output.storage_bits.is_multiple_of(8) {
        return Err(ModularFinalizeError::InvalidParams {
            reason: "output layout is empty or not byte-addressable",
        });
    }
    if source_channels != 1 {
        if output.kind != 9
            || output.channels != source_channels
            || output.bits != u32::from(source_bits)
            || output.numeric_mapping != 3
            || !matches!(output.storage_bits, 8 | 16)
        {
            return Err(ModularFinalizeError::InvalidParams {
                reason: "multichannel sources require matching native Modular output",
            });
        }
        return validate_output_planes(extent, source_channels, output);
    }
    if output.kind == 9 {
        if output.channels != 1
            || output.bits != u32::from(source_bits)
            || output.numeric_mapping != 3
            || !matches!(output.storage_bits, 8 | 16)
        {
            return Err(ModularFinalizeError::InvalidParams {
                reason: "native Gray output does not match its source",
            });
        }
        return validate_output_planes(extent, source_channels, output);
    }
    if source_bits != 8 {
        return Err(ModularFinalizeError::InvalidParams {
            reason: "converted output requires an 8-bit Gray source",
        });
    }
    match output.kind {
        0 | 7
            if output.numeric_mapping == 1
                && matches!(output.bits, 8 | 16 | 32)
                && output.storage_bits == output.bits => {}
        8 if output.storage_bits == output.bits
            && ((output.bits == 32 && output.numeric_mapping == 1)
                || (output.bits == 64 && matches!(output.numeric_mapping, 1 | 2))) => {}
        1..=3 if matches!(output.bits, 8 | 16) && output.storage_bits == output.bits => {}
        4 if output.bits == 8 && output.storage_bits == 8 => {}
        5 | 6 if output.bits == 8 && output.storage_bits == 8 => {}
        _ => {
            return Err(ModularFinalizeError::InvalidParams {
                reason: "converted output contract is unsupported",
            });
        }
    }
    validate_output_planes(extent, source_channels, output)
}

fn validate_output_planes(
    extent: Extent2d,
    source_channels: u32,
    output: ModularFinalizeOutput,
) -> Result<(), ModularFinalizeError> {
    let bytes_per_storage = output.storage_bits / 8;
    let mut planes = [(0u32, 0u32); 4];
    match output.kind {
        0 | 7 | 8 => {
            planes[0] = (
                extent.height,
                extent
                    .width
                    .checked_mul(output.channels)
                    .and_then(|samples| samples.checked_mul(output.bits / 8))
                    .ok_or(ModularFinalizeError::InvalidParams {
                        reason: "numeric output row size overflows u32",
                    })?,
            );
        }
        1 => {
            planes[0] = (
                extent.height,
                checked_product(&[extent.width, bytes_per_storage])?,
            );
        }
        2 => {
            planes[0] = (
                extent.height,
                checked_product(&[extent.width, bytes_per_storage])?,
            );
            planes[1] = (
                output.chroma_extent.height,
                checked_product(&[output.chroma_extent.width, 2, bytes_per_storage])?,
            );
        }
        3 => {
            planes[0] = (
                extent.height,
                checked_product(&[extent.width, bytes_per_storage])?,
            );
            planes[1] = (
                output.chroma_extent.height,
                checked_product(&[output.chroma_extent.width, bytes_per_storage])?,
            );
            planes[2] = planes[1];
        }
        4 => {
            planes[0] = (
                extent.height,
                checked_product(&[extent.width.div_ceil(2), 4])?,
            );
        }
        5 => {
            planes[0] = (
                extent.height,
                checked_product(&[extent.width, output.channels])?,
            );
        }
        6 => {
            for plane in planes.iter_mut().take(output.channels as usize) {
                *plane = (extent.height, extent.width);
            }
        }
        9 => {
            planes[0] = (
                extent.height,
                extent
                    .width
                    .checked_mul(source_channels)
                    .and_then(|samples| samples.checked_mul(bytes_per_storage))
                    .ok_or(ModularFinalizeError::InvalidParams {
                        reason: "native output row size overflows u32",
                    })?,
            );
        }
        _ => {
            return Err(ModularFinalizeError::InvalidParams {
                reason: "output kind is invalid",
            });
        }
    }
    for (index, (height, row_bytes)) in planes.into_iter().enumerate() {
        if height == 0 {
            continue;
        }
        let stride = output.plane_strides[index];
        if stride < row_bytes {
            return Err(ModularFinalizeError::InvalidParams {
                reason: "output plane stride is shorter than its row",
            });
        }
        let end = height
            .checked_sub(1)
            .and_then(|rows| rows.checked_mul(stride))
            .and_then(|bytes| bytes.checked_add(output.plane_offsets[index]))
            .and_then(|bytes| bytes.checked_add(row_bytes))
            .ok_or(ModularFinalizeError::InvalidParams {
                reason: "output plane address overflows u32",
            })?;
        if end > output.logical_size {
            return Err(ModularFinalizeError::InvalidParams {
                reason: "output plane exceeds the logical allocation",
            });
        }
    }
    Ok(())
}

fn checked_product(values: &[u32]) -> Result<u32, ModularFinalizeError> {
    values.iter().try_fold(1u32, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(ModularFinalizeError::InvalidParams {
                reason: "output row size overflows u32",
            })
    })
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ModularFinalizeBindings<'a> {
    pub arena: ResidentStorageBinding<'a>,
    pub output_words: ResidentStorageBinding<'a>,
    pub status: ResidentStorageBinding<'a>,
    pub output_f64: Option<ResidentStorageBinding<'a>>,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ModularFinalizeError {
    #[error("invalid resident Modular finalizer parameters: {reason}")]
    InvalidParams { reason: &'static str },
    #[error("resident Modular finalizer {binding} binding lacks STORAGE usage")]
    MissingStorageUsage { binding: &'static str },
    #[error("resident Modular finalizer {binding} offset {offset} is not aligned to {alignment}")]
    BindingAlignment {
        binding: &'static str,
        offset: u64,
        alignment: u64,
    },
    #[error("resident Modular finalizer {binding} range {offset}..{end} exceeds {available} bytes")]
    BindingRange {
        binding: &'static str,
        offset: u64,
        end: u64,
        available: u64,
    },
    #[error("resident Modular finalizer {binding} size {size} is not four-byte aligned")]
    BindingSizeAlignment { binding: &'static str, size: u64 },
    #[error(
        "resident Modular finalizer {binding} requires {required} bytes, binding has {available}"
    )]
    BindingSize {
        binding: &'static str,
        required: u64,
        available: u64,
    },
    #[error(
        "resident Modular finalizer storage binding needs {required} bytes, device permits {available}"
    )]
    StorageBindingLimit { required: u64, available: u64 },
    #[error(
        "resident Modular finalizer uniform needs {required} bytes, device permits {available}"
    )]
    UniformBindingLimit { required: u64, available: u64 },
    #[error("resident Modular finalizer requires a linear workgroup variant, received {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    #[error("resident Modular finalizer workgroup variant {variant:?} exceeds device limits")]
    WorkgroupVariant { variant: KernelVariant },
    #[error(
        "resident Modular finalizer requires {required} workgroups, device permits {available}"
    )]
    WorkgroupCount { required: u32, available: u32 },
}

pub(crate) struct ModularFinalizePipeline {
    pipeline: wgpu::ComputePipeline,
    variant: KernelVariant,
    f64_path: ModularFinalizeF64Path,
}

impl ModularFinalizePipeline {
    pub(crate) fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
        f64_path: ModularFinalizeF64Path,
    ) -> Result<Self, ModularFinalizeError> {
        validate_variant(variant, &device.limits())?;
        let shader = shader_source(f64_path);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu decode resident Modular finalizer"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        let constants = [("wg_x", f64::from(variant.workgroup_size().0))];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu decode resident Modular finalizer"),
            layout: None,
            module: &module,
            entry_point: Some("finalize"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
        Ok(Self {
            pipeline,
            variant,
            f64_path,
        })
    }

    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bindings: ModularFinalizeBindings<'_>,
        params: ModularFinalizeParams,
    ) -> Result<wgpu::Buffer, ModularFinalizeError> {
        let workgroups =
            validate_for_device(device, bindings, params, self.variant, self.f64_path)?;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu decode resident Modular finalizer params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut entries = vec![
            binding_entry(0, bindings.arena),
            binding_entry(1, bindings.output_words),
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            },
            binding_entry(3, bindings.status),
        ];
        if let Some(output_f64) = bindings.output_f64 {
            entries.push(binding_entry(4, output_f64));
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu decode resident Modular finalizer bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu resident Modular final output"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups[0], workgroups[1], 1);
        drop(pass);
        Ok(uniform)
    }
}

fn shader_source(path: ModularFinalizeF64Path) -> String {
    let (binding, output) = match path {
        ModularFinalizeF64Path::ExactF32Widening => ("", F64_EXACT_OUTPUT),
        ModularFinalizeF64Path::NativeArithmetic => (F64_NATIVE_BINDING, F64_NATIVE_OUTPUT),
    };
    SHADER
        .replace(F64_BINDING_MARKER, binding)
        .replace(F64_OUTPUT_MARKER, output)
}

fn validate_variant(
    variant: KernelVariant,
    limits: &wgpu::Limits,
) -> Result<(), ModularFinalizeError> {
    if !variant.is_linear() {
        return Err(ModularFinalizeError::WorkgroupShape { variant });
    }
    variant
        .validate_for(MODULAR_FINALIZE_KERNEL_KEY, limits, 0)
        .map_err(|_| ModularFinalizeError::WorkgroupVariant { variant })
}

fn validate_for_device(
    device: &wgpu::Device,
    bindings: ModularFinalizeBindings<'_>,
    params: ModularFinalizeParams,
    variant: KernelVariant,
    path: ModularFinalizeF64Path,
) -> Result<[u32; 2], ModularFinalizeError> {
    let limits = device.limits();
    validate_variant(variant, &limits)?;
    validate_binding(bindings.arena, "arena", &limits)?;
    validate_binding(bindings.output_words, "word output", &limits)?;
    validate_binding(bindings.status, "status", &limits)?;
    if bindings.status.size.get() < 4 {
        return Err(ModularFinalizeError::BindingSize {
            binding: "status",
            required: 4,
            available: bindings.status.size.get(),
        });
    }
    let output_binding = match (path, bindings.output_f64) {
        (ModularFinalizeF64Path::NativeArithmetic, Some(output)) => {
            if params.output[0] != 8 || params.format[1] != 64 || params.format[3] != 2 {
                return Err(ModularFinalizeError::InvalidParams {
                    reason: "native F64 pipeline received a non-native output contract",
                });
            }
            validate_binding(output, "F64 output", &limits)?;
            output
        }
        (ModularFinalizeF64Path::NativeArithmetic, None) => {
            return Err(ModularFinalizeError::InvalidParams {
                reason: "native F64 pipeline is missing its F64 output binding",
            });
        }
        (ModularFinalizeF64Path::ExactF32Widening, None) => bindings.output_words,
        (ModularFinalizeF64Path::ExactF32Widening, Some(_)) => {
            return Err(ModularFinalizeError::InvalidParams {
                reason: "portable finalizer received an unexpected F64 output binding",
            });
        }
    };
    let required_arena_bytes = u64::from(params.required_arena_words()) * 4;
    if bindings.arena.size.get() < required_arena_bytes {
        return Err(ModularFinalizeError::BindingSize {
            binding: "arena",
            required: required_arena_bytes,
            available: bindings.arena.size.get(),
        });
    }
    if path == ModularFinalizeF64Path::ExactF32Widening && params.format[3] == 2 {
        return Err(ModularFinalizeError::InvalidParams {
            reason: "portable finalizer received a native F64 output contract",
        });
    }
    if output_binding.size.get() < u64::from(params.logical_size()) {
        return Err(ModularFinalizeError::BindingSize {
            binding: "output",
            required: u64::from(params.logical_size()),
            available: output_binding.size.get(),
        });
    }
    let uniform_bytes = std::mem::size_of::<ModularFinalizeParams>() as u64;
    if uniform_bytes > limits.max_uniform_buffer_binding_size {
        return Err(ModularFinalizeError::UniformBindingLimit {
            required: uniform_bytes,
            available: limits.max_uniform_buffer_binding_size,
        });
    }
    let workgroups = dispatch_shape(params, variant);
    let required = workgroups[0].max(workgroups[1]);
    if required > limits.max_compute_workgroups_per_dimension {
        return Err(ModularFinalizeError::WorkgroupCount {
            required,
            available: limits.max_compute_workgroups_per_dimension,
        });
    }
    Ok(workgroups)
}

fn dispatch_shape(params: ModularFinalizeParams, variant: KernelVariant) -> [u32; 2] {
    [
        params.extent[0].div_ceil(variant.workgroup_size().0),
        params.extent[1],
    ]
}

fn validate_binding(
    binding: ResidentStorageBinding<'_>,
    name: &'static str,
    limits: &wgpu::Limits,
) -> Result<(), ModularFinalizeError> {
    if !binding.buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
        return Err(ModularFinalizeError::MissingStorageUsage { binding: name });
    }
    let alignment = u64::from(limits.min_storage_buffer_offset_alignment);
    if !binding.offset.is_multiple_of(alignment) {
        return Err(ModularFinalizeError::BindingAlignment {
            binding: name,
            offset: binding.offset,
            alignment,
        });
    }
    if !binding.size.get().is_multiple_of(4) {
        return Err(ModularFinalizeError::BindingSizeAlignment {
            binding: name,
            size: binding.size.get(),
        });
    }
    let end = binding.offset.checked_add(binding.size.get()).ok_or(
        ModularFinalizeError::BindingRange {
            binding: name,
            offset: binding.offset,
            end: u64::MAX,
            available: binding.buffer.size(),
        },
    )?;
    if end > binding.buffer.size() {
        return Err(ModularFinalizeError::BindingRange {
            binding: name,
            offset: binding.offset,
            end,
            available: binding.buffer.size(),
        });
    }
    if binding.size.get() > limits.max_storage_buffer_binding_size {
        return Err(ModularFinalizeError::StorageBindingLimit {
            required: binding.size.get(),
            available: limits.max_storage_buffer_binding_size,
        });
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_params(arena_words: u32) -> Result<ModularFinalizeParams, ModularFinalizeError> {
        ModularFinalizeParams::new(
            Extent2d::new(7, 3),
            8,
            &[GpuModularChannelLayout {
                word_offset: 5,
                row_stride_words: 9,
                width: 7,
                height: 3,
                hshift: 0,
                vshift: 0,
                bit_depth: 8,
                reserved: 0,
            }],
            arena_words,
            ModularFinalizeOutput {
                kind: 9,
                transfer: 0,
                limited_range: false,
                channels: 1,
                order: 0,
                bits: 8,
                storage_bits: 8,
                numeric_mapping: 3,
                plane_offsets: [0; 4],
                plane_strides: [7, 0, 0, 0],
                logical_size: 21,
                chroma_extent: Extent2d::new(0, 0),
            },
        )
    }

    #[test]
    fn uniform_and_wgsl_abis_validate_semantically() {
        assert_eq!(std::mem::size_of::<ModularFinalizeParams>(), 128);
        assert_eq!(std::mem::align_of::<ModularFinalizeParams>(), 16);
        fn assert_pod<T: Pod>() {}
        assert_pod::<ModularFinalizeParams>();
        for (source, capabilities) in [
            (
                shader_source(ModularFinalizeF64Path::ExactF32Widening),
                naga::valid::Capabilities::empty(),
            ),
            (
                shader_source(ModularFinalizeF64Path::NativeArithmetic),
                naga::valid::Capabilities::FLOAT64,
            ),
        ] {
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
                .validate(&module)
                .unwrap();
        }
    }

    #[test]
    fn final_plane_footprints_are_checked_against_the_arena() {
        let params = gray_params(30).unwrap();
        assert_eq!(params.extent, [7, 3, 1, 8]);
        assert_eq!(params.source_offsets, [5, 0, 0, 0]);
        assert_eq!(params.source_strides, [9, 0, 0, 0]);
        assert_eq!(params.extent[0] * params.extent[1], 21);
        assert!(matches!(
            gray_params(29),
            Err(ModularFinalizeError::InvalidParams {
                reason: "source plane exceeds the resident arena"
            })
        ));
    }

    #[test]
    fn every_linear_variant_dispatches_sixteen_k_within_portable_dimensions() {
        let extent = Extent2d::new(16_384, 16_384);
        let words = extent.width * extent.height;
        let params = ModularFinalizeParams::new(
            extent,
            8,
            &[GpuModularChannelLayout {
                word_offset: 0,
                row_stride_words: extent.width,
                width: extent.width,
                height: extent.height,
                hshift: 0,
                vshift: 0,
                bit_depth: 8,
                reserved: 0,
            }],
            words,
            ModularFinalizeOutput {
                kind: 9,
                transfer: 0,
                limited_range: false,
                channels: 1,
                order: 0,
                bits: 8,
                storage_bits: 8,
                numeric_mapping: 3,
                plane_offsets: [0; 4],
                plane_strides: [extent.width, 0, 0, 0],
                logical_size: words,
                chroma_extent: Extent2d::new(0, 0),
            },
        )
        .unwrap();
        for variant in [
            KernelVariant::Scalar,
            KernelVariant::Lanes32,
            KernelVariant::Lanes64,
            KernelVariant::Lanes128,
            KernelVariant::Lanes256,
        ] {
            let shape = dispatch_shape(params, variant);
            assert!(shape[0] <= 65_535, "{variant:?} x dimension");
            assert!(shape[1] <= 65_535, "{variant:?} y dimension");
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn actual_adapter_packs_native_nv12_and_f64_outputs_in_one_submission() {
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
            eprintln!("skipping Modular finalizer GPU test: no adapter");
            return;
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("jxl-wgpu Modular finalizer test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))
        else {
            eprintln!("skipping Modular finalizer GPU test: device request failed");
            return;
        };
        let variant = if validate_variant(KernelVariant::Lanes128, &device.limits()).is_ok() {
            KernelVariant::Lanes128
        } else {
            KernelVariant::Scalar
        };
        let pipeline = ModularFinalizePipeline::with_variant(
            &device,
            variant,
            ModularFinalizeF64Path::ExactF32Widening,
        )
        .unwrap();

        let mut arena_words = vec![0u32; 128];
        let gray_plane = GpuModularChannelLayout {
            word_offset: 5,
            row_stride_words: 9,
            width: 7,
            height: 3,
            hshift: 0,
            vshift: 0,
            bit_depth: 8,
            reserved: 0,
        };
        let gray_expected = (0..gray_plane.height)
            .flat_map(|y| (0..gray_plane.width).map(move |x| ((x * 29 + y * 71 + 3) & 255) as u8))
            .collect::<Vec<_>>();
        for y in 0..gray_plane.height {
            for x in 0..gray_plane.width {
                arena_words
                    [(gray_plane.word_offset + y * gray_plane.row_stride_words + x) as usize] =
                    u32::from(gray_expected[(y * gray_plane.width + x) as usize]);
            }
        }

        let rgb_planes = [40u32, 64, 88].map(|word_offset| GpuModularChannelLayout {
            word_offset,
            row_stride_words: 7,
            width: 5,
            height: 3,
            hshift: 0,
            vshift: 0,
            bit_depth: 8,
            reserved: 0,
        });
        let mut rgb_expected = Vec::with_capacity(5 * 3 * 3);
        for y in 0..3u32 {
            for x in 0..5u32 {
                for (channel, plane) in rgb_planes.iter().copied().enumerate() {
                    let value = ((channel as u32 * 83 + x * 17 + y * 43 + 11) & 255) as u8;
                    arena_words[(plane.word_offset + y * plane.row_stride_words + x) as usize] =
                        u32::from(value);
                    rgb_expected.push(value);
                }
            }
        }

        let arena = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Modular finalizer padded source arena"),
            contents: bytemuck::cast_slice(&arena_words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut invalid_arena_words = arena_words.clone();
        invalid_arena_words[gray_plane.word_offset as usize] = u32::MAX;
        let invalid_arena = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Modular finalizer out-of-range source arena"),
            contents: bytemuck::cast_slice(&invalid_arena_words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let status = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Modular finalizer successful status"),
            contents: bytemuck::cast_slice(&[1u32, 0, 0, 0]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let invalid_status = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Modular finalizer out-of-range status"),
            contents: bytemuck::cast_slice(&[1u32, 0, 0, 0]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let gray_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modular finalizer Gray8 output"),
            size: 24,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rgb_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modular finalizer RGB8 output"),
            size: 48,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let nv12_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modular finalizer NV12 output"),
            size: 40,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let f64_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modular finalizer F64 output"),
            size: 168,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let invalid_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modular finalizer rejected output"),
            size: 24,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modular finalizer aggregate staging"),
            size: 296,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let arena_binding = ResidentStorageBinding {
            buffer: &arena,
            offset: 0,
            size: NonZeroU64::new(arena.size()).unwrap(),
        };
        let status_binding = ResidentStorageBinding {
            buffer: &status,
            offset: 0,
            size: NonZeroU64::new(status.size()).unwrap(),
        };
        let native_output = |channels, stride, logical_size| ModularFinalizeOutput {
            kind: 9,
            transfer: 0,
            limited_range: false,
            channels,
            order: 0,
            bits: 8,
            storage_bits: 8,
            numeric_mapping: 3,
            plane_offsets: [0; 4],
            plane_strides: [stride, 0, 0, 0],
            logical_size,
            chroma_extent: Extent2d::new(0, 0),
        };
        let gray_params = ModularFinalizeParams::new(
            Extent2d::new(7, 3),
            8,
            &[gray_plane],
            arena_words.len() as u32,
            native_output(1, 7, 21),
        )
        .unwrap();
        let rgb_params = ModularFinalizeParams::new(
            Extent2d::new(5, 3),
            8,
            &rgb_planes,
            arena_words.len() as u32,
            native_output(3, 15, 45),
        )
        .unwrap();
        let nv12_params = ModularFinalizeParams::new(
            Extent2d::new(7, 3),
            8,
            &[gray_plane],
            arena_words.len() as u32,
            ModularFinalizeOutput {
                kind: 2,
                transfer: 0,
                limited_range: false,
                channels: 3,
                order: 0,
                bits: 8,
                storage_bits: 8,
                numeric_mapping: 0,
                plane_offsets: [0, 24, 0, 0],
                plane_strides: [7, 8, 0, 0],
                logical_size: 40,
                chroma_extent: Extent2d::new(4, 2),
            },
        )
        .unwrap();
        let f64_params = ModularFinalizeParams::new(
            Extent2d::new(7, 3),
            8,
            &[gray_plane],
            arena_words.len() as u32,
            ModularFinalizeOutput {
                kind: 8,
                transfer: 0,
                limited_range: false,
                channels: 1,
                order: 0,
                bits: 64,
                storage_bits: 64,
                numeric_mapping: 1,
                plane_offsets: [0; 4],
                plane_strides: [56, 0, 0, 0],
                logical_size: 168,
                chroma_extent: Extent2d::new(0, 0),
            },
        )
        .unwrap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Modular finalizer aggregate encoder"),
        });
        encoder.clear_buffer(&gray_output, 0, None);
        encoder.clear_buffer(&rgb_output, 0, None);
        encoder.clear_buffer(&nv12_output, 0, None);
        encoder.clear_buffer(&f64_output, 0, None);
        let gray_uniform = pipeline
            .encode(
                &device,
                &mut encoder,
                ModularFinalizeBindings {
                    arena: arena_binding,
                    output_words: ResidentStorageBinding {
                        buffer: &gray_output,
                        offset: 0,
                        size: NonZeroU64::new(gray_output.size()).unwrap(),
                    },
                    status: status_binding,
                    output_f64: None,
                },
                gray_params,
            )
            .unwrap();
        let rgb_uniform = pipeline
            .encode(
                &device,
                &mut encoder,
                ModularFinalizeBindings {
                    arena: arena_binding,
                    output_words: ResidentStorageBinding {
                        buffer: &rgb_output,
                        offset: 0,
                        size: NonZeroU64::new(rgb_output.size()).unwrap(),
                    },
                    status: status_binding,
                    output_f64: None,
                },
                rgb_params,
            )
            .unwrap();
        let nv12_uniform = pipeline
            .encode(
                &device,
                &mut encoder,
                ModularFinalizeBindings {
                    arena: arena_binding,
                    output_words: ResidentStorageBinding {
                        buffer: &nv12_output,
                        offset: 0,
                        size: NonZeroU64::new(nv12_output.size()).unwrap(),
                    },
                    status: status_binding,
                    output_f64: None,
                },
                nv12_params,
            )
            .unwrap();
        let f64_uniform = pipeline
            .encode(
                &device,
                &mut encoder,
                ModularFinalizeBindings {
                    arena: arena_binding,
                    output_words: ResidentStorageBinding {
                        buffer: &f64_output,
                        offset: 0,
                        size: NonZeroU64::new(f64_output.size()).unwrap(),
                    },
                    status: status_binding,
                    output_f64: None,
                },
                f64_params,
            )
            .unwrap();
        let invalid_uniform = pipeline
            .encode(
                &device,
                &mut encoder,
                ModularFinalizeBindings {
                    arena: ResidentStorageBinding {
                        buffer: &invalid_arena,
                        offset: 0,
                        size: NonZeroU64::new(invalid_arena.size()).unwrap(),
                    },
                    output_words: ResidentStorageBinding {
                        buffer: &invalid_output,
                        offset: 0,
                        size: NonZeroU64::new(invalid_output.size()).unwrap(),
                    },
                    status: ResidentStorageBinding {
                        buffer: &invalid_status,
                        offset: 0,
                        size: NonZeroU64::new(invalid_status.size()).unwrap(),
                    },
                    output_f64: None,
                },
                gray_params,
            )
            .unwrap();
        encoder.copy_buffer_to_buffer(&gray_output, 0, &staging, 0, 24);
        encoder.copy_buffer_to_buffer(&rgb_output, 0, &staging, 24, 48);
        encoder.copy_buffer_to_buffer(&nv12_output, 0, &staging, 72, 40);
        encoder.copy_buffer_to_buffer(&f64_output, 0, &staging, 112, 168);
        encoder.copy_buffer_to_buffer(&invalid_status, 0, &staging, 280, 16);
        queue.submit([encoder.finish()]);
        drop((
            gray_uniform,
            rgb_uniform,
            nv12_uniform,
            f64_uniform,
            invalid_uniform,
        ));

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range().unwrap();
        assert_eq!(&mapped[..gray_expected.len()], gray_expected);
        assert_eq!(&mapped[24..24 + rgb_expected.len()], rgb_expected);
        let mut nv12_expected = vec![0u8; 40];
        nv12_expected[..gray_expected.len()].copy_from_slice(&gray_expected);
        nv12_expected[24..].fill(128);
        assert_eq!(&mapped[72..112], nv12_expected);
        let expected_f64 = gray_expected
            .iter()
            .flat_map(|sample| f64::from(f32::from(*sample) / 255.0).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(&mapped[112..280], expected_f64);
        assert_eq!(u32::from_le_bytes(mapped[280..284].try_into().unwrap()), 9);
        drop(mapped);
        staging.unmap();
    }
}
