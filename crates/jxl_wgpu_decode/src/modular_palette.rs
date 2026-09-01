//! GPU-resident inverse JPEG XL Modular Palette.

use bytemuck::{Pod, Zeroable};
use jxl_wgpu::{KernelVariant, ResidentStorageBinding};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::modular_tree::WpHeaderIr;

const SHADER: &str = include_str!("modular_palette.wgsl");

pub(crate) const MODULAR_PALETTE_KERNEL_KEY: &str = "modular_palette";
pub(crate) const DEFAULT_MODULAR_PALETTE_VARIANT: KernelVariant = KernelVariant::Lanes64;
pub(crate) const MODULAR_PALETTE_STATE_WORDS: u32 = 20;
pub(crate) const MODULAR_PALETTE_SERIAL_CHUNK_SAMPLES: u32 = 1 << 18;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularPalettePlane {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub offset_words: u32,
}

impl ModularPalettePlane {
    fn required_words(self) -> Option<u32> {
        if self.width == 0 || self.height == 0 {
            return Some(self.offset_words);
        }
        (self.height - 1)
            .checked_mul(self.stride)?
            .checked_add(self.offset_words)?
            .checked_add(self.width)
    }

    fn span(self) -> Result<(u32, u32), ModularPaletteError> {
        let end = self
            .required_words()
            .ok_or(ModularPaletteError::InvalidParams {
                reason: "plane address overflows WGSL u32",
            })?;
        Ok((self.offset_words, end))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularPaletteMetadata {
    pub palette_channel: u32,
    pub color_count: u32,
    pub delta_count: u32,
    pub predictor: u32,
    pub bit_depth: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularPaletteScratch {
    pub offset_words: u32,
    pub words: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularPaletteJob {
    pub palette: ModularPalettePlane,
    pub indices: ModularPalettePlane,
    pub output: ModularPalettePlane,
    pub palette_channel: u32,
    pub color_count: u32,
    pub delta_count: u32,
    pub predictor: u32,
    pub bit_depth: u32,
    pub scratch_offset_words: u32,
    pub scratch_words: u32,
    sample_count: u32,
}

impl ModularPaletteJob {
    pub(crate) fn new(
        palette: ModularPalettePlane,
        indices: ModularPalettePlane,
        output: ModularPalettePlane,
        metadata: ModularPaletteMetadata,
        scratch: ModularPaletteScratch,
    ) -> Result<Self, ModularPaletteError> {
        let sample_count = indices.width.checked_mul(indices.height).ok_or(
            ModularPaletteError::InvalidParams {
                reason: "palette sample count overflows WGSL u32",
            },
        )?;
        let job = Self {
            palette,
            indices,
            output,
            palette_channel: metadata.palette_channel,
            color_count: metadata.color_count,
            delta_count: metadata.delta_count,
            predictor: metadata.predictor,
            bit_depth: metadata.bit_depth,
            scratch_offset_words: scratch.offset_words,
            scratch_words: scratch.words,
            sample_count,
        };
        job.validate()?;
        Ok(job)
    }

    pub(crate) fn dispatch_count(self) -> u32 {
        if self.predictor == 0 {
            1
        } else {
            self.sample_count()
                .div_ceil(MODULAR_PALETTE_SERIAL_CHUNK_SAMPLES)
        }
    }

    pub(crate) fn uniform_bytes(self) -> u64 {
        u64::from(self.dispatch_count()) * std::mem::size_of::<ModularPaletteParams>() as u64
    }

    const fn sample_count(self) -> u32 {
        self.sample_count
    }

    fn required_words(self) -> Result<u32, ModularPaletteError> {
        let scratch_end = self
            .scratch_offset_words
            .checked_add(self.scratch_words)
            .ok_or(ModularPaletteError::InvalidParams {
                reason: "scratch address overflows WGSL u32",
            })?;
        Ok(self
            .palette
            .required_words()
            .and_then(|required| self.indices.required_words().map(|v| required.max(v)))
            .and_then(|required| self.output.required_words().map(|v| required.max(v)))
            .ok_or(ModularPaletteError::InvalidParams {
                reason: "plane address overflows WGSL u32",
            })?
            .max(scratch_end))
    }

    fn validate(self) -> Result<(), ModularPaletteError> {
        if self.predictor >= 14 || !(1..=24).contains(&self.bit_depth) {
            return Err(ModularPaletteError::InvalidParams {
                reason: "predictor or bit depth is outside the JPEG XL domain",
            });
        }
        let palette_width = self.color_count.checked_add(self.delta_count).ok_or(
            ModularPaletteError::InvalidParams {
                reason: "palette entry count overflows WGSL u32",
            },
        )?;
        if palette_width > i32::MAX as u32 {
            return Err(ModularPaletteError::InvalidParams {
                reason: "palette entry count exceeds the signed JPEG XL index domain",
            });
        }
        if self.palette.width != palette_width
            || self.palette_channel >= self.palette.height
            || self.palette.stride < self.palette.width
            || self.indices.width == 0
            || self.indices.height == 0
            || self.indices.stride < self.indices.width
            || self.output.width != self.indices.width
            || self.output.height != self.indices.height
            || self.output.stride < self.output.width
        {
            return Err(ModularPaletteError::InvalidParams {
                reason: "palette, index, or output geometry is inconsistent",
            });
        }
        let required_scratch = palette_scratch_words(self.indices.width, self.predictor)?;
        if self.scratch_words < required_scratch {
            return Err(ModularPaletteError::InvalidParams {
                reason: "palette predictor scratch is too small",
            });
        }
        let palette_span = self.palette.span()?;
        let index_span = self.indices.span()?;
        let output_span = self.output.span()?;
        let scratch_end = self
            .scratch_offset_words
            .checked_add(self.scratch_words)
            .ok_or(ModularPaletteError::InvalidParams {
                reason: "scratch address overflows WGSL u32",
            })?;
        if spans_overlap(palette_span, index_span)
            || spans_overlap(index_span, output_span)
            || spans_overlap(palette_span, output_span)
            || (self.scratch_words != 0
                && [palette_span, index_span, output_span]
                    .into_iter()
                    .any(|span| spans_overlap(span, (self.scratch_offset_words, scratch_end))))
        {
            return Err(ModularPaletteError::InvalidParams {
                reason: "palette input, output, and predictor scratch must not overlap",
            });
        }
        Ok(())
    }
}

fn spans_overlap(left: (u32, u32), right: (u32, u32)) -> bool {
    left.0 < right.1 && right.0 < left.1
}

pub(crate) fn palette_scratch_words(
    width: u32,
    predictor: u32,
) -> Result<u32, ModularPaletteError> {
    match predictor {
        0 => Ok(0),
        6 => width
            .checked_mul(5)
            .and_then(|words| words.checked_add(MODULAR_PALETTE_STATE_WORDS))
            .ok_or(ModularPaletteError::InvalidParams {
                reason: "weighted palette scratch overflows WGSL u32",
            }),
        1..=5 | 7..=13 => Ok(0),
        _ => Err(ModularPaletteError::InvalidParams {
            reason: "palette predictor is outside the JPEG XL domain",
        }),
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct ModularPaletteWeightedParams {
    first: [u32; 4],
    second: [u32; 4],
    third: [u32; 4],
}

impl From<WpHeaderIr> for ModularPaletteWeightedParams {
    fn from(header: WpHeaderIr) -> Self {
        Self {
            first: [header.p1, header.p2, header.p3a, header.p3b],
            second: [header.p3c, header.p3d, header.p3e, header.w0],
            third: [header.w1, header.w2, header.w3, 0],
        }
    }
}

/// Exact 128-byte uniform consumed by `modular_palette.wgsl`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct ModularPaletteParams {
    palette: [u32; 4],
    indices: [u32; 4],
    output: [u32; 4],
    info: [u32; 4],
    range: [u32; 4],
    wp_first: [u32; 4],
    wp_second: [u32; 4],
    wp_third: [u32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<ModularPaletteWeightedParams>() == 48);
    assert!(std::mem::align_of::<ModularPaletteWeightedParams>() == 16);
    assert!(std::mem::size_of::<ModularPaletteParams>() == 128);
    assert!(std::mem::align_of::<ModularPaletteParams>() == 16);
};

impl ModularPaletteParams {
    fn new(
        job: ModularPaletteJob,
        weighted: ModularPaletteWeightedParams,
        start: u32,
        end: u32,
    ) -> Result<Self, ModularPaletteError> {
        job.validate()?;
        if start > end || end > job.sample_count() {
            return Err(ModularPaletteError::InvalidParams {
                reason: "palette chunk lies outside the index plane",
            });
        }
        Ok(Self {
            palette: [
                job.palette.width,
                job.palette.height,
                job.palette.stride,
                job.palette.offset_words,
            ],
            indices: [
                job.indices.width,
                job.indices.height,
                job.indices.stride,
                job.indices.offset_words,
            ],
            output: [
                job.output.width,
                job.output.height,
                job.output.stride,
                job.output.offset_words,
            ],
            info: [
                job.palette_channel,
                job.color_count,
                job.delta_count,
                job.predictor,
            ],
            range: [start, end, job.bit_depth, job.scratch_offset_words],
            wp_first: weighted.first,
            wp_second: weighted.second,
            wp_third: weighted.third,
        })
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ModularPaletteError {
    #[error("invalid resident Modular Palette parameters: {reason}")]
    InvalidParams { reason: &'static str },
    #[error("resident Modular Palette arena lacks STORAGE usage")]
    MissingStorageUsage,
    #[error("resident Modular Palette arena offset {offset} is not aligned to {alignment}")]
    BindingAlignment { offset: u64, alignment: u64 },
    #[error("resident Modular Palette arena range {offset}..{end} exceeds {available} bytes")]
    BindingRange {
        offset: u64,
        end: u64,
        available: u64,
    },
    #[error("resident Modular Palette arena size {size} is not four-byte aligned")]
    BindingSizeAlignment { size: u64 },
    #[error("resident Modular Palette needs {required} arena bytes, binding has {available}")]
    BindingSize { required: u64, available: u64 },
    #[error(
        "resident Modular Palette storage binding needs {required} bytes, device permits {available}"
    )]
    StorageBindingLimit { required: u64, available: u64 },
    #[error("resident Modular Palette uniform needs {required} bytes, device permits {available}")]
    UniformBindingLimit { required: u64, available: u64 },
    #[error("resident Modular Palette requires a linear workgroup variant, received {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    #[error("resident Modular Palette workgroup variant {variant:?} exceeds device limits")]
    WorkgroupVariant { variant: KernelVariant },
    #[error("resident Modular Palette requires {required} workgroups, device permits {available}")]
    WorkgroupCount { required: u32, available: u32 },
}

pub(crate) struct ModularPalettePipeline {
    pipeline: wgpu::ComputePipeline,
    variant: KernelVariant,
}

impl ModularPalettePipeline {
    pub(crate) fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, ModularPaletteError> {
        validate_variant(variant, &device.limits())?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu decode resident Modular Palette"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let constants = [("wg_x", f64::from(variant.workgroup_size().0))];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu decode resident Modular Palette"),
            layout: None,
            module: &module,
            entry_point: Some("inverse_palette"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
        Ok(Self { pipeline, variant })
    }

    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        arena: ResidentStorageBinding<'_>,
        job: ModularPaletteJob,
        weighted: ModularPaletteWeightedParams,
    ) -> Result<Vec<wgpu::Buffer>, ModularPaletteError> {
        self.encode_with_chunk_samples(
            device,
            encoder,
            arena,
            job,
            weighted,
            MODULAR_PALETTE_SERIAL_CHUNK_SAMPLES,
        )
    }

    fn encode_with_chunk_samples(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        arena: ResidentStorageBinding<'_>,
        job: ModularPaletteJob,
        weighted: ModularPaletteWeightedParams,
        chunk_samples: u32,
    ) -> Result<Vec<wgpu::Buffer>, ModularPaletteError> {
        validate_for_device(device, arena, job, self.variant)?;
        if chunk_samples == 0 {
            return Err(ModularPaletteError::InvalidParams {
                reason: "palette serial chunk size is zero",
            });
        }
        let serial = job.predictor != 0;
        let sample_count = job.sample_count();
        let mut start = 0u32;
        let dispatch_count = if serial {
            sample_count.div_ceil(chunk_samples)
        } else {
            1
        };
        let mut uniforms = Vec::with_capacity(dispatch_count as usize);
        loop {
            let end = if serial {
                start.saturating_add(chunk_samples).min(sample_count)
            } else {
                sample_count
            };
            let params = ModularPaletteParams::new(job, weighted, start, end)?;
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu decode resident Modular Palette params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu decode resident Modular Palette bindings"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: arena.buffer,
                            offset: arena.offset,
                            size: Some(arena.size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu resident Modular Palette inverse"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            if serial {
                pass.dispatch_workgroups(1, 1, 1);
            } else {
                pass.dispatch_workgroups(
                    job.indices.width.div_ceil(self.variant.workgroup_size().0),
                    job.indices.height,
                    1,
                );
            }
            drop(pass);
            uniforms.push(uniform);
            if end == sample_count {
                break;
            }
            start = end;
        }
        Ok(uniforms)
    }
}

fn validate_variant(
    variant: KernelVariant,
    limits: &wgpu::Limits,
) -> Result<(), ModularPaletteError> {
    if !variant.is_linear() {
        return Err(ModularPaletteError::WorkgroupShape { variant });
    }
    variant
        .validate_for(MODULAR_PALETTE_KERNEL_KEY, limits, 0)
        .map_err(|_| ModularPaletteError::WorkgroupVariant { variant })
}

fn validate_for_device(
    device: &wgpu::Device,
    arena: ResidentStorageBinding<'_>,
    job: ModularPaletteJob,
    variant: KernelVariant,
) -> Result<(), ModularPaletteError> {
    job.validate()?;
    let limits = device.limits();
    validate_variant(variant, &limits)?;
    if !arena.buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
        return Err(ModularPaletteError::MissingStorageUsage);
    }
    let alignment = u64::from(limits.min_storage_buffer_offset_alignment);
    if !arena.offset.is_multiple_of(alignment) {
        return Err(ModularPaletteError::BindingAlignment {
            offset: arena.offset,
            alignment,
        });
    }
    if !arena.size.get().is_multiple_of(4) {
        return Err(ModularPaletteError::BindingSizeAlignment {
            size: arena.size.get(),
        });
    }
    let end =
        arena
            .offset
            .checked_add(arena.size.get())
            .ok_or(ModularPaletteError::BindingRange {
                offset: arena.offset,
                end: u64::MAX,
                available: arena.buffer.size(),
            })?;
    if end > arena.buffer.size() {
        return Err(ModularPaletteError::BindingRange {
            offset: arena.offset,
            end,
            available: arena.buffer.size(),
        });
    }
    let required = u64::from(job.required_words()?) * 4;
    if required > arena.size.get() {
        return Err(ModularPaletteError::BindingSize {
            required,
            available: arena.size.get(),
        });
    }
    if arena.size.get() > limits.max_storage_buffer_binding_size {
        return Err(ModularPaletteError::StorageBindingLimit {
            required: arena.size.get(),
            available: limits.max_storage_buffer_binding_size,
        });
    }
    let uniform_bytes = std::mem::size_of::<ModularPaletteParams>() as u64;
    if uniform_bytes > limits.max_uniform_buffer_binding_size {
        return Err(ModularPaletteError::UniformBindingLimit {
            required: uniform_bytes,
            available: limits.max_uniform_buffer_binding_size,
        });
    }
    let required_workgroups = job
        .indices
        .width
        .div_ceil(variant.workgroup_size().0)
        .max(job.indices.height);
    if required_workgroups > limits.max_compute_workgroups_per_dimension {
        return Err(ModularPaletteError::WorkgroupCount {
            required: required_workgroups,
            available: limits.max_compute_workgroups_per_dimension,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_predict(predictor: u32, output: &[i32], width: usize, x: usize, y: usize) -> i32 {
        let at = |x: usize, y: usize| output[y * width + x];
        let mut w = 0;
        if x != 0 {
            w = at(x - 1, y);
        } else if y != 0 {
            w = at(x, y - 1);
        }
        let mut n = w;
        let mut nw = w;
        if y != 0 {
            n = at(x, y - 1);
            nw = if x == 0 { n } else { at(x - 1, y - 1) };
        }
        let ne = if y != 0 && x + 1 < width {
            at(x + 1, y - 1)
        } else {
            n
        };
        let nee = if y != 0 && x + 2 < width {
            at(x + 2, y - 1)
        } else {
            ne
        };
        let nn = if y >= 2 { at(x, y - 2) } else { n };
        let ww = if x >= 2 { at(x - 2, y) } else { w };
        match predictor {
            0 => 0,
            1 => w,
            2 => n,
            3 => ((i64::from(w) + i64::from(n)) / 2) as i32,
            4 => {
                if n.abs_diff(nw) < w.abs_diff(nw) {
                    w
                } else {
                    n
                }
            }
            5 => (i64::from(n) + i64::from(w) - i64::from(nw))
                .clamp(i64::from(n.min(w)), i64::from(n.max(w))) as i32,
            7 => ne,
            8 => nw,
            9 => ww,
            10 => ((i64::from(w) + i64::from(nw)) / 2) as i32,
            11 => ((i64::from(n) + i64::from(nw)) / 2) as i32,
            12 => ((i64::from(n) + i64::from(ne)) / 2) as i32,
            13 => {
                ((6 * i64::from(n) - 2 * i64::from(nn)
                    + 7 * i64::from(w)
                    + i64::from(ww)
                    + i64::from(nee)
                    + 3 * i64::from(ne)
                    + 8)
                    / 16) as i32
            }
            _ => panic!("scalar non-weighted Palette oracle received predictor {predictor}"),
        }
    }

    fn scalar_explicit_palette(
        palette: &[i32],
        indices: &[i32],
        width: usize,
        delta_count: i32,
        predictor: u32,
    ) -> Vec<i32> {
        let mut output = vec![0; indices.len()];
        for (cursor, &index) in indices.iter().enumerate() {
            let prediction =
                scalar_predict(predictor, &output, width, cursor % width, cursor / width);
            let entry = palette[index as usize];
            output[cursor] = if index < delta_count {
                entry.wrapping_add(prediction)
            } else {
                entry
            };
        }
        output
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn test_device(label: &'static str) -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()
    }

    #[test]
    fn uniform_and_shader_are_semantically_valid() {
        assert_eq!(std::mem::size_of::<ModularPaletteParams>(), 128);
        assert_eq!(std::mem::align_of::<ModularPaletteParams>(), 16);
        fn assert_pod<T: Pod>() {}
        assert_pod::<ModularPaletteParams>();
        let module = naga::front::wgsl::parse_str(SHADER).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn jobs_validate_disjoint_views_and_bounded_weighted_scratch() {
        let palette = ModularPalettePlane {
            width: 5,
            height: 3,
            stride: 5,
            offset_words: 0,
        };
        let indices = ModularPalettePlane {
            width: 9,
            height: 5,
            stride: 9,
            offset_words: 15,
        };
        let output = ModularPalettePlane {
            width: 9,
            height: 5,
            stride: 9,
            offset_words: 60,
        };
        let scratch_words = palette_scratch_words(9, 6).unwrap();
        let metadata = ModularPaletteMetadata {
            palette_channel: 2,
            color_count: 3,
            delta_count: 2,
            predictor: 6,
            bit_depth: 8,
        };
        let job = ModularPaletteJob::new(
            palette,
            indices,
            output,
            metadata,
            ModularPaletteScratch {
                offset_words: 105,
                words: scratch_words,
            },
        )
        .unwrap();
        assert_eq!(job.dispatch_count(), 1);
        assert_eq!(job.required_words().unwrap(), 170);
        assert!(matches!(
            ModularPaletteJob::new(
                palette,
                indices,
                output,
                metadata,
                ModularPaletteScratch {
                    offset_words: 100,
                    words: scratch_words,
                },
            ),
            Err(ModularPaletteError::InvalidParams {
                reason: "palette input, output, and predictor scratch must not overlap"
            })
        ));
    }

    #[test]
    fn sixteen_k_zero_predictor_uses_portable_two_dimensional_dispatch() {
        let extent = 16_384;
        let samples = extent * extent;
        let job = ModularPaletteJob::new(
            ModularPalettePlane {
                width: 0,
                height: 1,
                stride: 0,
                offset_words: 0,
            },
            ModularPalettePlane {
                width: extent,
                height: extent,
                stride: extent,
                offset_words: 0,
            },
            ModularPalettePlane {
                width: extent,
                height: extent,
                stride: extent,
                offset_words: samples,
            },
            ModularPaletteMetadata {
                palette_channel: 0,
                color_count: 0,
                delta_count: 0,
                predictor: 0,
                bit_depth: 8,
            },
            ModularPaletteScratch {
                offset_words: 0,
                words: 0,
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
            assert!(extent.div_ceil(variant.workgroup_size().0) <= 65_535);
            assert!(extent <= 65_535);
        }
        assert_eq!(job.dispatch_count(), 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn actual_adapter_matches_every_non_weighted_predictor_and_implicit_palette() {
        use std::num::NonZeroU64;
        use std::sync::mpsc;

        use wgpu::util::DeviceExt;

        let Some((device, queue)) = test_device("jxl-wgpu Modular Palette predictor test") else {
            eprintln!("skipping Modular Palette GPU test: no adapter");
            return;
        };
        let variant = if validate_variant(KernelVariant::Lanes128, &device.limits()).is_ok() {
            KernelVariant::Lanes128
        } else {
            KernelVariant::Scalar
        };
        let pipeline = ModularPalettePipeline::with_variant(&device, variant).unwrap();
        let predictors = [0u32, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13];
        let width = 9u32;
        let height = 5u32;
        let samples = width * height;
        let palette_values = [-3i32, 5, 40, 180];
        let indices = (0..samples)
            .map(|index| i32::try_from((index * 7 + index / width * 3) % 4).unwrap())
            .collect::<Vec<_>>();
        let palette_offset = 0u32;
        let index_offset = palette_values.len() as u32;
        let output_start = index_offset + samples;
        let implicit_indices = [-1i32, -2, -3, -4, 0, 3, 63, 64, 68];
        let implicit_index_offset = output_start + samples * predictors.len() as u32;
        let implicit_output_offset = implicit_index_offset + implicit_indices.len() as u32;
        let arena_words = implicit_output_offset + implicit_indices.len() as u32;
        let mut initial = vec![0i32; arena_words as usize];
        initial[..palette_values.len()].copy_from_slice(&palette_values);
        initial[index_offset as usize..(index_offset + samples) as usize].copy_from_slice(&indices);
        initial[implicit_index_offset as usize
            ..implicit_index_offset as usize + implicit_indices.len()]
            .copy_from_slice(&implicit_indices);

        let arena = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Modular Palette predictor arena"),
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modular Palette predictor readback"),
            size: arena.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let binding = ResidentStorageBinding {
            buffer: &arena,
            offset: 0,
            size: NonZeroU64::new(arena.size()).unwrap(),
        };
        let palette_plane = ModularPalettePlane {
            width: 4,
            height: 1,
            stride: 4,
            offset_words: palette_offset,
        };
        let index_plane = ModularPalettePlane {
            width,
            height,
            stride: width,
            offset_words: index_offset,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Modular Palette predictor encoder"),
        });
        let mut uniforms = Vec::new();
        for (job_index, predictor) in predictors.into_iter().enumerate() {
            let output_offset = output_start + job_index as u32 * samples;
            uniforms.extend(
                pipeline
                    .encode(
                        &device,
                        &mut encoder,
                        binding,
                        ModularPaletteJob::new(
                            palette_plane,
                            index_plane,
                            ModularPalettePlane {
                                width,
                                height,
                                stride: width,
                                offset_words: output_offset,
                            },
                            ModularPaletteMetadata {
                                palette_channel: 0,
                                color_count: 2,
                                delta_count: 2,
                                predictor,
                                bit_depth: 8,
                            },
                            ModularPaletteScratch {
                                offset_words: 0,
                                words: 0,
                            },
                        )
                        .unwrap(),
                        ModularPaletteWeightedParams::from(WpHeaderIr::default()),
                    )
                    .unwrap(),
            );
        }
        uniforms.extend(
            pipeline
                .encode(
                    &device,
                    &mut encoder,
                    binding,
                    ModularPaletteJob::new(
                        ModularPalettePlane {
                            width: 0,
                            height: 1,
                            stride: 0,
                            offset_words: 0,
                        },
                        ModularPalettePlane {
                            width: implicit_indices.len() as u32,
                            height: 1,
                            stride: implicit_indices.len() as u32,
                            offset_words: implicit_index_offset,
                        },
                        ModularPalettePlane {
                            width: implicit_indices.len() as u32,
                            height: 1,
                            stride: implicit_indices.len() as u32,
                            offset_words: implicit_output_offset,
                        },
                        ModularPaletteMetadata {
                            palette_channel: 0,
                            color_count: 0,
                            delta_count: 0,
                            predictor: 0,
                            bit_depth: 8,
                        },
                        ModularPaletteScratch {
                            offset_words: 0,
                            words: 0,
                        },
                    )
                    .unwrap(),
                    ModularPaletteWeightedParams::from(WpHeaderIr::default()),
                )
                .unwrap(),
        );
        encoder.copy_buffer_to_buffer(&arena, 0, &staging, 0, arena.size());
        queue.submit([encoder.finish()]);
        drop(uniforms);

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range().unwrap();
        let actual = bytemuck::cast_slice::<u8, i32>(&mapped);
        for (job_index, predictor) in predictors.into_iter().enumerate() {
            let expected =
                scalar_explicit_palette(&palette_values, &indices, width as usize, 2, predictor);
            let start = (output_start + job_index as u32 * samples) as usize;
            assert_eq!(&actual[start..start + samples as usize], expected);
        }
        assert_eq!(
            &actual[implicit_output_offset as usize
                ..implicit_output_offset as usize + implicit_indices.len()],
            &[0, 4, -4, 11, 32, 223, 223, 0, 255]
        );
        drop(mapped);
        staging.unmap();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn actual_adapter_weighted_chunks_match_one_resident_dispatch() {
        use std::num::NonZeroU64;
        use std::sync::mpsc;

        use wgpu::util::DeviceExt;

        let Some((device, queue)) = test_device("jxl-wgpu Modular Palette weighted chunk test")
        else {
            eprintln!("skipping Modular Palette weighted GPU test: no adapter");
            return;
        };
        let pipeline =
            ModularPalettePipeline::with_variant(&device, KernelVariant::Lanes64).unwrap();
        let width = 1_025u32;
        let height = 257u32;
        let samples = width * height;
        assert!(samples > MODULAR_PALETTE_SERIAL_CHUNK_SAMPLES);
        let palette_offset = 0u32;
        let index_offset = 2u32;
        let reference_offset = index_offset + samples;
        let chunked_offset = reference_offset + samples;
        let scratch_words = palette_scratch_words(width, 6).unwrap();
        let reference_scratch = chunked_offset + samples;
        let chunked_scratch = reference_scratch + scratch_words;
        let arena_words = chunked_scratch + scratch_words;
        let mut initial = vec![0u32; arena_words as usize];
        initial[palette_offset as usize] = 0;
        initial[palette_offset as usize + 1] = 100;
        initial[index_offset as usize] = 1;
        let arena = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Modular Palette weighted chunk arena"),
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modular Palette weighted chunk readback"),
            size: u64::from(samples) * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let binding = ResidentStorageBinding {
            buffer: &arena,
            offset: 0,
            size: NonZeroU64::new(arena.size()).unwrap(),
        };
        let make_job = |output_offset, scratch_offset| {
            ModularPaletteJob::new(
                ModularPalettePlane {
                    width: 2,
                    height: 1,
                    stride: 2,
                    offset_words: palette_offset,
                },
                ModularPalettePlane {
                    width,
                    height,
                    stride: width,
                    offset_words: index_offset,
                },
                ModularPalettePlane {
                    width,
                    height,
                    stride: width,
                    offset_words: output_offset,
                },
                ModularPaletteMetadata {
                    palette_channel: 0,
                    color_count: 1,
                    delta_count: 1,
                    predictor: 6,
                    bit_depth: 8,
                },
                ModularPaletteScratch {
                    offset_words: scratch_offset,
                    words: scratch_words,
                },
            )
            .unwrap()
        };
        let weighted = ModularPaletteWeightedParams::from(WpHeaderIr::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Modular Palette weighted chunk encoder"),
        });
        let one_uniform = pipeline
            .encode_with_chunk_samples(
                &device,
                &mut encoder,
                binding,
                make_job(reference_offset, reference_scratch),
                weighted,
                samples,
            )
            .unwrap();
        let chunked_uniforms = pipeline
            .encode(
                &device,
                &mut encoder,
                binding,
                make_job(chunked_offset, chunked_scratch),
                weighted,
            )
            .unwrap();
        assert_eq!(one_uniform.len(), 1);
        assert_eq!(chunked_uniforms.len(), 2);
        encoder.copy_buffer_to_buffer(
            &arena,
            u64::from(reference_offset) * 4,
            &staging,
            0,
            u64::from(samples) * 4,
        );
        encoder.copy_buffer_to_buffer(
            &arena,
            u64::from(chunked_offset) * 4,
            &staging,
            u64::from(samples) * 4,
            u64::from(samples) * 4,
        );
        queue.submit([encoder.finish()]);
        drop((one_uniform, chunked_uniforms));

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range().unwrap();
        let actual = bytemuck::cast_slice::<u8, i32>(&mapped);
        let (reference, chunked) = actual.split_at(samples as usize);
        assert_eq!(reference, chunked);
        assert_eq!(reference[0], 100);
        assert!(reference[1..].iter().any(|&sample| sample != 0));
        drop(mapped);
        staging.unmap();
    }
}
