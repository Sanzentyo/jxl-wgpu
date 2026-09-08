//! Decode selected Modular sample representations and reconstruct presentation resolution on GPU.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::UpsamplingWeightsInventory;
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{
    ResidentF32Plane, ResidentStorageBinding, ResidentUpsampleError, ResidentUpsampleInputs,
    ResidentUpsampleKernel, ResidentUpsamplePipeline, ResidentUpsampleWeights,
};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::modular_sample::ModularOutputPlane;
use crate::modular_transform::GpuModularChannelLayout;

/// Interpretation of resident Modular output words after inverse transforms or resampling.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModularSampleDomain {
    /// Integer working words containing the original declared sample representation.
    Encoded = 0,
    /// Decoded binary32 values after sample conversion and optional filtering.
    DecodedF32 = 1,
}

#[derive(Clone, Debug, Error)]
pub enum ModularRenderError {
    #[error("invalid Modular render plane: {reason}")]
    Invalid { reason: &'static str },
    #[error("Modular render {resource} requires {required}, available {available}")]
    Limit {
        resource: &'static str,
        required: u64,
        available: u64,
    },
    #[error(transparent)]
    Upsample(#[from] ResidentUpsampleError),
}

type Result<T> = std::result::Result<T, ModularRenderError>;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NormalizeParams {
    source: [u32; 4],  // width, height, stride, word offset
    mapping: [u32; 4], // source sample encoding, output stride, reserved
}
const _: () = assert!(std::mem::size_of::<NormalizeParams>() == 32);

#[derive(Debug)]
pub(crate) struct ModularRenderPlan {
    pub extent: Extent2d,
    sources: Vec<ModularOutputPlane>,
    factors: Vec<u32>,
    planes: Vec<ModularOutputPlane>,
    kernels: Vec<ResidentUpsampleKernel>,
    pub output_bytes: u64,
    pub scratch_bytes: u64,
    pub weight_bytes: u64,
    pub uniform_bytes: u64,
}

impl ModularRenderPlan {
    pub(crate) fn new(
        extent: Extent2d,
        sources: Vec<ModularOutputPlane>,
        factors: Vec<u32>,
        weights: &UpsamplingWeightsInventory,
        limits: &wgpu::Limits,
    ) -> Result<Self> {
        if extent.width == 0
            || extent.height == 0
            || extent.width > i32::MAX as u32 / 2
            || extent.height > i32::MAX as u32 / 2
            || sources.is_empty()
            || u32::try_from(sources.len()).is_err()
            || sources.len() != factors.len()
        {
            return invalid("extent or selected channel count");
        }
        jxl_wgpu::KernelVariant::Tile16x16
            .validate_for("modular_render", limits, 0)
            .map_err(|_| ResidentUpsampleError::WorkgroupVariant {
                variant: jxl_wgpu::KernelVariant::Tile16x16,
            })?;
        let pixels = u64::from(extent.width) * u64::from(extent.height);
        let alignment = u64::from(limits.min_storage_buffer_offset_alignment).max(4);
        let overflow = || ModularRenderError::Invalid {
            reason: "render allocation size overflow",
        };
        let plane_bytes = pixels
            .checked_mul(4)
            .ok_or_else(overflow)?
            .div_ceil(alignment)
            .checked_mul(alignment)
            .ok_or_else(overflow)?;
        let output_bytes = plane_bytes
            .checked_mul(sources.len() as u64)
            .ok_or_else(overflow)?;
        let storage_limit = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size)
            .min(u64::from(u32::MAX) * 4);
        require("output arena bytes", output_bytes, storage_limit)?;
        require(
            "uniform bytes",
            std::mem::size_of::<NormalizeParams>() as u64,
            limits.max_uniform_buffer_binding_size,
        )?;
        require(
            "X workgroups",
            u64::from(extent.width.div_ceil(16)),
            u64::from(limits.max_compute_workgroups_per_dimension),
        )?;
        require(
            "Y workgroups",
            u64::from(extent.height.div_ceil(16)),
            u64::from(limits.max_compute_workgroups_per_dimension),
        )?;
        let mut scratch_bytes = 0;
        let mut kernels = Vec::<ResidentUpsampleKernel>::new();
        let mut planes = Vec::new();
        let mut uniform_bytes = 0;
        for (index, (&source_info, &factor)) in sources.iter().zip(&factors).enumerate() {
            let source = source_info.layout;
            if !matches!(factor, 1 | 2 | 4 | 8)
                || source.width != extent.width.div_ceil(factor)
                || source.height != extent.height.div_ceil(factor)
                || source.row_stride_words < source.width
                || source.reserved != 0
                || source.hshift < 0
                || source.vshift < 0
            {
                return invalid("source geometry, factor or precision");
            }
            let source_words = u64::from(source.word_offset)
                + u64::from(source.height - 1) * u64::from(source.row_stride_words)
                + u64::from(source.width);
            require("source address words", source_words, u64::from(u32::MAX))?;
            uniform_bytes += std::mem::size_of::<NormalizeParams>() as u64;
            if factor != 1 {
                scratch_bytes =
                    scratch_bytes.max(u64::from(source.width) * u64::from(source.height) * 4);
                uniform_bytes += ResidentUpsamplePipeline::UNIFORM_BYTES;
                if !kernels.iter().any(|kernel| kernel.factor() == factor) {
                    kernels.push(upsample_kernel(weights, factor)?);
                }
            }
            planes.push(ModularOutputPlane::new(
                GpuModularChannelLayout {
                    width: extent.width,
                    height: extent.height,
                    row_stride_words: extent.width,
                    word_offset: (plane_bytes / 4 * index as u64) as u32,
                    hshift: 0,
                    vshift: 0,
                    bit_depth: source.bit_depth,
                    reserved: 0,
                },
                source_info.encoding,
            ));
        }
        require("normalization scratch bytes", scratch_bytes, storage_limit)?;
        let weight_bytes = kernels
            .iter()
            .map(ResidentUpsampleKernel::weight_bytes)
            .sum();
        for kernel in &kernels {
            require("weight bytes", kernel.weight_bytes(), storage_limit)?;
        }
        Ok(Self {
            extent,
            sources,
            factors,
            planes,
            kernels,
            output_bytes,
            scratch_bytes,
            weight_bytes,
            uniform_bytes,
        })
    }

    pub(crate) fn total_bytes(&self) -> u64 {
        self.output_bytes + self.scratch_bytes + self.weight_bytes + self.uniform_bytes
    }

    pub(crate) fn planes(&self) -> &[ModularOutputPlane] {
        &self.planes
    }

    pub(crate) fn allocate(&self, device: &wgpu::Device) -> Result<ModularRenderBuffers> {
        let create = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        Ok(ModularRenderBuffers {
            output: create(
                "jxl-wgpu reconstructed Modular render planes",
                self.output_bytes,
            ),
            scratch: (self.scratch_bytes != 0)
                .then(|| create("jxl-wgpu Modular normalization scratch", self.scratch_bytes)),
            weights: self
                .kernels
                .iter()
                .map(|kernel| kernel.upload(device))
                .collect::<std::result::Result<_, _>>()?,
        })
    }
}

pub(crate) fn upsample_kernel(
    weights: &UpsamplingWeightsInventory,
    factor: u32,
) -> Result<ResidentUpsampleKernel> {
    let compact: Vec<f32> = match factor {
        2 => weights.up2.iter().map(|v| v.to_f32()).collect(),
        4 => weights.up4.iter().map(|v| v.to_f32()).collect(),
        8 => weights.up8.iter().map(|v| v.to_f32()).collect(),
        _ => return invalid("upsampling kernel factor"),
    };
    Ok(ResidentUpsampleKernel::from_compact(factor, &compact)?)
}

pub(crate) struct ModularRenderBuffers {
    pub output: wgpu::Buffer,
    scratch: Option<wgpu::Buffer>,
    weights: Vec<ResidentUpsampleWeights>,
}

pub(crate) struct ModularRenderPipeline {
    normalize: wgpu::ComputePipeline,
    upsample: ResidentUpsamplePipeline,
}

impl ModularRenderPipeline {
    pub(crate) fn new(device: &wgpu::Device) -> Result<Self> {
        // Validate the shared 16×16 workgroup before creating either compute pipeline.
        let upsample = ResidentUpsamplePipeline::new(device)?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu Modular sample decoding"),
            source: wgpu::ShaderSource::Wgsl(
                crate::modular_sample::shader(include_str!("modular_render.wgsl")).into(),
            ),
        });
        Ok(Self {
            normalize: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu Modular sample normalization"),
                layout: None,
                module: &shader,
                entry_point: Some("normalize"),
                compilation_options: Default::default(),
                cache: None,
            }),
            upsample,
        })
    }

    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        plan: &ModularRenderPlan,
        buffers: &ModularRenderBuffers,
        input: ResidentStorageBinding<'_>,
        sources: &[ModularOutputPlane],
    ) -> Result<Vec<wgpu::Buffer>> {
        if !input.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            || !input.size.get().is_multiple_of(4)
            || !input.offset.is_multiple_of(u64::from(
                device.limits().min_storage_buffer_offset_alignment,
            ))
            || input
                .offset
                .checked_add(input.size.get())
                .is_none_or(|end| end > input.buffer.size())
        {
            return invalid("source storage binding");
        }
        require(
            "source storage binding bytes",
            input.size.get(),
            device.limits().max_storage_buffer_binding_size,
        )?;
        if sources.len() != plan.sources.len() {
            return invalid("selected source count changed");
        }
        let mut uniforms = Vec::new();
        for (index, (&source_info, &factor)) in sources.iter().zip(&plan.factors).enumerate() {
            let source = source_info.layout;
            let expected = plan.sources[index].layout;
            if source.width != expected.width
                || source.height != expected.height
                || source_info.encoding != plan.sources[index].encoding
                || source.row_stride_words < source.width
                || source.reserved != 0
            {
                return invalid("selected source geometry changed");
            }
            let source_bytes = (u64::from(source.word_offset)
                + u64::from(source.height - 1) * u64::from(source.row_stride_words)
                + u64::from(source.width))
                * 4;
            require(
                "source address words",
                source_bytes / 4,
                u64::from(u32::MAX),
            )?;
            require("source binding bytes", source_bytes, input.size.get())?;
            let plane = plan.planes[index].layout;
            let output_binding = ResidentStorageBinding {
                buffer: &buffers.output,
                offset: u64::from(plane.word_offset) * 4,
                size: std::num::NonZeroU64::new(
                    u64::from(plane.width) * u64::from(plane.height) * 4,
                )
                .ok_or(ModularRenderError::Invalid {
                    reason: "empty render plane",
                })?,
            };
            let normalized = if factor == 1 {
                output_binding
            } else {
                binding(
                    buffers
                        .scratch
                        .as_ref()
                        .ok_or(ModularRenderError::Invalid {
                            reason: "missing normalization scratch",
                        })?,
                )?
            };
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu Modular normalization params"),
                contents: bytemuck::bytes_of(&NormalizeParams {
                    source: [
                        source.width,
                        source.height,
                        source.row_stride_words,
                        source.word_offset,
                    ],
                    mapping: [source_info.encoding.packed(), source.width, 0, 0],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let resources = [
                resource(input),
                resource(normalized),
                uniform.as_entire_binding(),
            ];
            let entries: Vec<_> = resources
                .into_iter()
                .enumerate()
                .map(|(index, resource)| wgpu::BindGroupEntry {
                    binding: index as u32,
                    resource,
                })
                .collect();
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu Modular normalization bindings"),
                layout: &self.normalize.get_bind_group_layout(0),
                entries: &entries,
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.normalize);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(source.width.div_ceil(16), source.height.div_ceil(16), 1);
            }
            uniforms.push(uniform);
            if factor != 1 {
                let kernel_index = plan
                    .kernels
                    .iter()
                    .position(|kernel| kernel.factor() == factor)
                    .ok_or(ModularRenderError::Invalid {
                        reason: "missing upsampling weights",
                    })?;
                uniforms.push(self.upsample.encode(
                    device,
                    encoder,
                    ResidentUpsampleInputs {
                        input: ResidentF32Plane {
                            storage: normalized,
                            width: source.width,
                            height: source.height,
                            stride: source.width,
                        },
                        output: ResidentF32Plane {
                            storage: output_binding,
                            width: plane.width,
                            height: plane.height,
                            stride: plane.row_stride_words,
                        },
                        weights: &buffers.weights[kernel_index],
                    },
                )?);
            }
        }
        Ok(uniforms)
    }
}

fn binding(buffer: &wgpu::Buffer) -> Result<ResidentStorageBinding<'_>> {
    ResidentStorageBinding::entire(buffer).map_err(|_| ModularRenderError::Invalid {
        reason: "empty render buffer",
    })
}
fn resource(binding: ResidentStorageBinding<'_>) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: binding.buffer,
        offset: binding.offset,
        size: Some(binding.size),
    })
}
fn invalid<T>(reason: &'static str) -> Result<T> {
    Err(ModularRenderError::Invalid { reason })
}
fn require(resource: &'static str, required: u64, available: u64) -> Result<()> {
    if required > available {
        Err(ModularRenderError::Limit {
            resource,
            required,
            available,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights() -> UpsamplingWeightsInventory {
        let value = jxl_gpu_bitstream::FiniteF32::from_f32(0.03125).unwrap();
        UpsamplingWeightsInventory {
            up2: [value; 15],
            up4: [value; 55],
            up8: [value; 210],
        }
    }

    fn plane(extent: Extent2d, factor: u32, bits: u32) -> ModularOutputPlane {
        ModularOutputPlane::new(
            GpuModularChannelLayout {
                width: extent.width.div_ceil(factor),
                height: extent.height.div_ceil(factor),
                row_stride_words: extent.width.div_ceil(factor) + 2,
                word_offset: 3,
                hshift: factor.ilog2() as i32,
                vshift: factor.ilog2() as i32,
                bit_depth: bits,
                reserved: 0,
            },
            crate::modular_sample::ModularSampleEncoding::integer(bits).unwrap(),
        )
    }

    #[test]
    fn independent_grids_share_one_scratch_and_deduplicated_kernels() {
        let extent = Extent2d::new(17, 9);
        let limits = wgpu::Limits::default();
        let factors = vec![1, 2, 4, 8];
        let sources = factors
            .iter()
            .zip([8, 5, 12, 16])
            .map(|(&factor, bits)| plane(extent, factor, bits))
            .collect();
        let plan = ModularRenderPlan::new(extent, sources, factors, &weights(), &limits).unwrap();
        let alignment = u64::from(limits.min_storage_buffer_offset_alignment).max(4);
        assert_eq!(
            plan.output_bytes,
            (17 * 9 * 4u64).div_ceil(alignment) * alignment * 4
        );
        assert_eq!(plan.scratch_bytes, 9 * 5 * 4);
        assert_eq!(plan.weight_bytes, (4 + 16 + 64) * 25 * 4);
        assert_eq!(plan.uniform_bytes, 4 * 32 + 3 * 32);
        for (index, plane) in plan.planes().iter().enumerate() {
            assert_eq!(
                (
                    plane.layout.width,
                    plane.layout.height,
                    plane.layout.hshift,
                    plane.layout.vshift
                ),
                (17, 9, 0, 0)
            );
            assert!(u64::from(plane.layout.word_offset * 4).is_multiple_of(alignment));
            if index > 0 {
                assert!(
                    plan.planes()[index - 1].layout.word_offset + 17 * 9
                        <= plane.layout.word_offset
                );
            }
        }
        let duplicate = ModularRenderPlan::new(
            extent,
            vec![plane(extent, 2, 8); 2],
            vec![2, 2],
            &weights(),
            &limits,
        )
        .unwrap();
        assert_eq!(duplicate.weight_bytes, 2 * 2 * 25 * 4);
        assert_eq!(duplicate.scratch_bytes, 9 * 5 * 4);
    }

    #[test]
    fn render_limits_and_geometry_fail_before_gpu_allocation() {
        let extent = Extent2d::new(17, 9);
        let source = plane(extent, 2, 8);
        let mut limits = wgpu::Limits {
            max_storage_buffer_binding_size: 100,
            ..Default::default()
        };
        assert!(matches!(
            ModularRenderPlan::new(extent, vec![source], vec![2], &weights(), &limits),
            Err(ModularRenderError::Limit {
                resource: "output arena bytes",
                ..
            })
        ));
        limits = wgpu::Limits::default();
        let small_workgroup = wgpu::Limits {
            max_compute_invocations_per_workgroup: 128,
            ..limits.clone()
        };
        assert!(matches!(
            ModularRenderPlan::new(extent, vec![source], vec![2], &weights(), &small_workgroup),
            Err(ModularRenderError::Upsample(
                ResidentUpsampleError::WorkgroupVariant { .. }
            ))
        ));
        for (extent, source, factor) in [
            (Extent2d::new(u32::MAX, u32::MAX), source, 2),
            (
                extent,
                ModularOutputPlane {
                    layout: GpuModularChannelLayout {
                        width: 8,
                        ..source.layout
                    },
                    ..source
                },
                2,
            ),
            (
                extent,
                ModularOutputPlane {
                    layout: GpuModularChannelLayout {
                        reserved: 1,
                        ..source.layout
                    },
                    ..source
                },
                2,
            ),
            (extent, source, 3),
        ] {
            assert!(matches!(
                ModularRenderPlan::new(extent, vec![source], vec![factor], &weights(), &limits),
                Err(ModularRenderError::Invalid { .. })
            ));
        }
        assert!(matches!(
            ModularRenderPlan::new(
                extent,
                vec![ModularOutputPlane {
                    layout: GpuModularChannelLayout {
                        word_offset: u32::MAX,
                        ..source.layout
                    },
                    ..source
                }],
                vec![2],
                &weights(),
                &limits
            ),
            Err(ModularRenderError::Limit {
                resource: "source address words",
                ..
            })
        ));
    }
}
