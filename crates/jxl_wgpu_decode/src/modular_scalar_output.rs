//! Scalar delivery from resident signed Modular working samples, without a color surface.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_formats::{ImageLayout, PixelFormatClass, SampleKind, classify_pixel_format};
use jxl_gpu_protocol::{Extent2d, OutputOrientation};
use jxl_wgpu::{KernelVariant, ResidentStorageBinding};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::modular_sample::ModularSampleEncoding;
use crate::modular_transform::GpuModularChannelLayout;
use crate::{ModularChannels, NumericSampleMapping};

type Result<T> = std::result::Result<T, ModularScalarOutputError>;

#[derive(Debug, Error)]
pub enum ModularScalarOutputError {
    #[error("invalid resident Modular scalar output: {reason}")]
    Invalid { reason: &'static str },
    #[error("resident Modular scalar output {resource} requires {required}, available {available}")]
    Limit {
        resource: &'static str,
        required: u64,
        available: u64,
    },
    #[error("resident Modular scalar output {binding} binding is invalid: {reason}")]
    Binding {
        binding: &'static str,
        reason: &'static str,
    },
    #[error("resident Modular scalar output kernel is invalid: {message}")]
    Kernel { message: String },
    #[error("resident Modular scalar sample cannot be represented at the requested unsigned depth")]
    SampleOutOfRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularScalarOutputConfig {
    pub extent: Extent2d,
    pub orientation: OutputOrientation,
    pub encoding: ModularSampleEncoding,
    pub mapping: NumericSampleMapping,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScalarParams {
    source: [u32; 4],      // width, height, word stride, word offset
    destination: [u32; 4], // width, height, byte stride, byte offset
    encoding: [u32; 4],    // source encoding, component bytes, floating output, orientation
    bounds: [u32; 4],      // logical bytes, output words, dispatch width, reserved
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularScalarOutputPlan {
    pub config: ModularScalarOutputConfig,
    pub storage_bytes: u64,
    logical_bytes: u32,
    destination: [u32; 4],
    component_bytes: u32,
    dispatch: [u32; 2],
    dispatch_width: u32,
    variant: KernelVariant,
}

impl ModularScalarOutputPlan {
    pub(crate) const UNIFORM_BYTES: u64 = std::mem::size_of::<ScalarParams>() as u64;
    pub(crate) const STATUS_BYTES: u64 = 4;

    pub(crate) fn new(
        config: ModularScalarOutputConfig,
        layout: &ImageLayout,
        limits: &wgpu::Limits,
        variant: KernelVariant,
    ) -> Result<Self> {
        if config.extent.width == 0
            || config.extent.height == 0
            || config.orientation.map_extent(config.extent) != layout.extent
            || layout.planes.len() != 1
        {
            return invalid("source extent, precision or output geometry");
        }
        let component_bytes = match config.mapping {
            NumericSampleMapping::NativeUnsigned if !config.encoding.is_float() => {
                let native = crate::model::native_modular_format(&layout.format)
                    .filter(|native| {
                        native.channels == ModularChannels::Gray
                            && native.bits_per_sample == config.encoding.bits()
                    })
                    .ok_or(ModularScalarOutputError::Invalid {
                        reason: "native scalar depth must match the extra-channel declaration",
                    })?;
                u32::from(native.bits_per_sample).div_ceil(8)
            }
            NumericSampleMapping::NormalizedUnsigned if !config.encoding.is_float() => {
                if !matches!(classify_pixel_format(&layout.format),
                    Ok(PixelFormatClass::Numeric(n)) if n.components == 1
                        && n.sample_kind == SampleKind::Float && n.bits_per_component == 32)
                {
                    return invalid("normalized scalar output requires F32 storage");
                }
                4
            }
            NumericSampleMapping::NativeFloat if config.encoding.is_float() => {
                if !matches!(classify_pixel_format(&layout.format),
                    Ok(PixelFormatClass::Numeric(n)) if n.components == 1
                        && n.sample_kind == SampleKind::Float && n.bits_per_component == 32)
                {
                    return invalid("floating source output requires scalar F32 storage");
                }
                4
            }
            _ => return invalid("unsupported scalar sample mapping"),
        };
        let plane = &layout.planes[0];
        let row_bytes = u64::from(layout.extent.width) * u64::from(component_bytes);
        let required = u64::from(layout.extent.height - 1)
            .checked_mul(plane.row_stride)
            .and_then(|bytes| plane.offset.checked_add(bytes))
            .and_then(|end| end.checked_add(row_bytes))
            .ok_or(ModularScalarOutputError::Invalid {
                reason: "output address overflow",
            })?;
        if plane.row_stride < row_bytes || required > layout.logical_size {
            return invalid("output rows exceed the declared layout");
        }
        let logical_bytes =
            u32::try_from(layout.logical_size).map_err(|_| ModularScalarOutputError::Invalid {
                reason: "output exceeds WGSL byte addressing",
            })?;
        let storage_bytes = u64::from(logical_bytes).div_ceil(4) * 4;
        require_limit(
            "output storage bytes",
            storage_bytes,
            limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size),
        )?;
        require_limit(
            "uniform bytes",
            Self::UNIFORM_BYTES,
            limits.max_uniform_buffer_binding_size,
        )?;
        require_limit(
            "storage bindings",
            3,
            u64::from(limits.max_storage_buffers_per_shader_stage),
        )?;
        variant
            .validate_for("modular_scalar_output", limits, 0)
            .map_err(|error| ModularScalarOutputError::Kernel {
                message: error.to_string(),
            })?;
        let (wg_x, wg_y) = variant.workgroup_size();
        if wg_y != 1 {
            return invalid("scalar packing requires a linear workgroup");
        }
        let groups = (storage_bytes / 4).div_ceil(u64::from(wg_x));
        let x = groups.min(u64::from(limits.max_compute_workgroups_per_dimension));
        if x == 0 {
            return invalid("empty dispatch limit");
        }
        let y = groups.div_ceil(x);
        require_limit(
            "dispatch rows",
            y,
            u64::from(limits.max_compute_workgroups_per_dimension),
        )?;
        Ok(Self {
            config,
            storage_bytes,
            logical_bytes,
            destination: [
                layout.extent.width,
                layout.extent.height,
                u32::try_from(plane.row_stride).map_err(|_| ModularScalarOutputError::Invalid {
                    reason: "row stride exceeds WGSL u32",
                })?,
                u32::try_from(plane.offset).map_err(|_| ModularScalarOutputError::Invalid {
                    reason: "plane offset exceeds WGSL u32",
                })?,
            ],
            component_bytes,
            dispatch: [x as u32, y as u32],
            dispatch_width: (x as u32).checked_mul(wg_x).ok_or(
                ModularScalarOutputError::Invalid {
                    reason: "dispatch width overflow",
                },
            )?,
            variant,
        })
    }
}

pub(crate) struct ModularScalarOutputPipeline {
    pipeline: wgpu::ComputePipeline,
    variant: KernelVariant,
}

pub(crate) struct ModularScalarOutputScratch {
    _uniform: wgpu::Buffer,
    pub status: wgpu::Buffer,
}

pub(crate) struct ModularScalarOutputInputs<'a> {
    pub plane: GpuModularChannelLayout,
    pub domain: crate::ModularSampleDomain,
    pub arena: ResidentStorageBinding<'a>,
    pub output: ResidentStorageBinding<'a>,
}

impl ModularScalarOutputScratch {
    pub(crate) fn validate_status(bytes: &[u8]) -> Result<()> {
        match bytes.try_into().map(u32::from_le_bytes) {
            Ok(0) => Ok(()),
            Ok(1) => Err(ModularScalarOutputError::SampleOutOfRange),
            _ => invalid("scalar status ABI"),
        }
    }
}

impl ModularScalarOutputPipeline {
    pub(crate) fn new(device: &wgpu::Device, variant: KernelVariant) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu resident Modular scalar output"),
            source: wgpu::ShaderSource::Wgsl(shader().into()),
        });
        Self {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu resident Modular scalar output"),
                layout: None,
                module: &module,
                entry_point: Some("pack"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("wg_x", f64::from(variant.workgroup_size().0))],
                    ..Default::default()
                },
                cache: None,
            }),
            variant,
        }
    }

    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        plan: ModularScalarOutputPlan,
        inputs: ModularScalarOutputInputs<'_>,
    ) -> Result<ModularScalarOutputScratch> {
        let ModularScalarOutputInputs {
            plane,
            domain,
            arena,
            output,
        } = inputs;
        if self.variant != plan.variant
            || plane.width != plan.config.extent.width
            || plane.height != plan.config.extent.height
            || plane.row_stride_words < plane.width
            || plane.hshift != 0
            || plane.vshift != 0
            || plane.reserved != 0
        {
            return invalid("scalar plan differs from its resident plane or pipeline");
        }
        let words = u64::from(plane.word_offset)
            + u64::from(plane.height - 1) * u64::from(plane.row_stride_words)
            + u64::from(plane.width);
        require_limit("source word addresses", words, u64::from(u32::MAX))?;
        validate_binding(arena, "source", words * 4, &device.limits())?;
        validate_binding(output, "output", plan.storage_bytes, &device.limits())?;
        let params = ScalarParams {
            source: [
                plane.width,
                plane.height,
                plane.row_stride_words,
                plane.word_offset,
            ],
            destination: plan.destination,
            encoding: [
                plan.config.encoding.packed(),
                plan.component_bytes,
                u32::from(plan.config.mapping != NumericSampleMapping::NativeUnsigned),
                plan.config.orientation.to_exif_value() - 1,
            ],
            bounds: [
                plan.logical_bytes,
                (plan.storage_bytes / 4) as u32,
                plan.dispatch_width,
                domain as u32,
            ],
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu Modular scalar output parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu Modular scalar output status"),
            size: ModularScalarOutputPlan::STATUS_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.clear_buffer(&status, 0, None);
        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu Modular scalar output bindings"),
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
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: output.buffer,
                        offset: output.offset,
                        size: Some(output.size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: status.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &binding, &[]);
        pass.dispatch_workgroups(plan.dispatch[0], plan.dispatch[1], 1);
        Ok(ModularScalarOutputScratch {
            _uniform: uniform,
            status,
        })
    }
}

fn shader() -> String {
    format!(
        "{}\n{}",
        jxl_wgpu::IMAGE_ORIENTATION_SHADER,
        crate::modular_sample::shader(include_str!("modular_scalar_output.wgsl"))
    )
}

fn invalid<T>(reason: &'static str) -> Result<T> {
    Err(ModularScalarOutputError::Invalid { reason })
}

fn require_limit(resource: &'static str, required: u64, available: u64) -> Result<()> {
    if required > available {
        Err(ModularScalarOutputError::Limit {
            resource,
            required,
            available,
        })
    } else {
        Ok(())
    }
}

fn validate_binding(
    binding: ResidentStorageBinding<'_>,
    name: &'static str,
    required: u64,
    limits: &wgpu::Limits,
) -> Result<()> {
    if !binding.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
        || !binding
            .offset
            .is_multiple_of(u64::from(limits.min_storage_buffer_offset_alignment))
        || !binding.size.get().is_multiple_of(4)
        || binding
            .offset
            .checked_add(binding.size.get())
            .is_none_or(|end| end > binding.buffer.size())
    {
        return Err(ModularScalarOutputError::Binding {
            binding: name,
            reason: "usage, alignment or buffer range",
        });
    }
    require_limit(name, required, binding.size.get())?;
    require_limit(
        "storage binding bytes",
        binding.size.get(),
        limits.max_storage_buffer_binding_size,
    )
}

#[cfg(test)]
mod tests;
