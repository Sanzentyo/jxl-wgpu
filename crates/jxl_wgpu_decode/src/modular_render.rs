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

use crate::frame_resampling::ChannelResampling;
use crate::modular_sample::ModularOutputPlane;
use crate::modular_transform::GpuModularChannelLayout;

mod color;
mod lf;
pub(crate) use color::ModularColorConfig;
pub(crate) use color::ReconstructionPipeline as ModularReconstructionPipeline;
pub(crate) use lf::{ModularLfBuffers, ModularLfPipelines, ModularLfPlan};

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
    #[error(transparent)]
    Restoration(#[from] crate::RestorationError),
    #[error(transparent)]
    Gaborish(#[from] jxl_wgpu::ResidentGaborishError),
    #[error(transparent)]
    Epf(#[from] jxl_wgpu::ResidentEpfError),
    #[error(transparent)]
    Noise(#[from] jxl_wgpu::ResidentNoiseError),
    #[error(transparent)]
    ColorOutput(std::sync::Arc<crate::color_output::ColorOutputError>),
    #[error(transparent)]
    Layout(#[from] jxl_gpu_formats::LayoutError),
}

type Result<T> = std::result::Result<T, ModularRenderError>;

impl From<crate::color_output::ColorOutputError> for ModularRenderError {
    fn from(value: crate::color_output::ColorOutputError) -> Self {
        Self::ColorOutput(std::sync::Arc::new(value))
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NormalizeParams {
    source: [u32; 4],  // width, height, stride, word offset
    mapping: [u32; 4], // source sample encoding, output stride, reserved
}
const _: () = assert!(std::mem::size_of::<NormalizeParams>() == 32);

#[derive(Debug)]
pub(crate) struct ModularRenderPlan {
    color: Option<color::ColorPlan>,
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
        sources: Vec<ModularOutputPlane>,
        resampling: Vec<ChannelResampling>,
        weights: &UpsamplingWeightsInventory,
        limits: &wgpu::Limits,
    ) -> Result<Self> {
        if sources.is_empty()
            || u32::try_from(sources.len()).is_err()
            || sources.len() != resampling.len()
        {
            return invalid("extent or selected channel count");
        }
        jxl_wgpu::KernelVariant::Tile16x16
            .validate_for("modular_render", limits, 0)
            .map_err(|_| ResidentUpsampleError::WorkgroupVariant {
                variant: jxl_wgpu::KernelVariant::Tile16x16,
            })?;
        let alignment = u64::from(limits.min_storage_buffer_offset_alignment).max(4);
        let overflow = || ModularRenderError::Invalid {
            reason: "render allocation size overflow",
        };
        let storage_limit = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size)
            .min(u64::from(u32::MAX) * 4);
        require(
            "uniform bytes",
            std::mem::size_of::<NormalizeParams>() as u64,
            limits.max_uniform_buffer_binding_size,
        )?;
        let mut output_bytes = 0_u64;
        let mut offsets = Vec::with_capacity(resampling.len());
        for &ChannelResampling { extent, .. } in &resampling {
            if extent.width == 0
                || extent.height == 0
                || extent.width > i32::MAX as u32 / 2
                || extent.height > i32::MAX as u32 / 2
            {
                return invalid("channel output extent");
            }
            for (resource, dimension) in [
                ("X workgroups", extent.width),
                ("Y workgroups", extent.height),
            ] {
                require(
                    resource,
                    u64::from(dimension.div_ceil(16)),
                    u64::from(limits.max_compute_workgroups_per_dimension),
                )?;
            }
            offsets.push((output_bytes / 4) as u32);
            let bytes = (u64::from(extent.width) * u64::from(extent.height))
                .checked_mul(4)
                .ok_or_else(overflow)?
                .div_ceil(alignment)
                .checked_mul(alignment)
                .ok_or_else(overflow)?;
            output_bytes = output_bytes.checked_add(bytes).ok_or_else(overflow)?;
            require("output arena bytes", output_bytes, storage_limit)?;
        }
        let mut scratch_bytes = 0;
        let mut kernels = Vec::<ResidentUpsampleKernel>::new();
        let mut planes = Vec::new();
        let mut uniform_bytes = 0;
        for (index, (&source_info, &ChannelResampling { extent, factor })) in
            sources.iter().zip(&resampling).enumerate()
        {
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
                    word_offset: offsets[index],
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
            color: None,
            sources,
            factors: resampling.iter().map(|channel| channel.factor).collect(),
            planes,
            kernels,
            output_bytes,
            scratch_bytes,
            weight_bytes,
            uniform_bytes,
        })
    }

    pub(crate) fn total_bytes(&self) -> u64 {
        self.output_bytes
            + self.scratch_bytes
            + self.weight_bytes
            + self.uniform_bytes
            + self.color.as_ref().map_or(0, |color| color.storage_bytes)
    }

    pub(crate) fn with_color(
        mut self,
        config: ModularColorConfig,
        target: jxl_gpu_formats::ColorSpecification,
        limits: &wgpu::Limits,
    ) -> Result<Self> {
        let color = color::ColorPlan::new(
            config,
            self.output_extent(),
            &self.sources,
            &self.factors,
            &self.planes,
            target,
            limits,
        )?;
        self.uniform_bytes -= 3 * std::mem::size_of::<NormalizeParams>() as u64
            + if self.factors[0] == 1 {
                0
            } else {
                3 * ResidentUpsamplePipeline::UNIFORM_BYTES
            };
        self.uniform_bytes += color.uniform_bytes;
        self.scratch_bytes = self
            .sources
            .iter()
            .zip(&self.factors)
            .skip(3)
            .filter(|(_, factor)| **factor != 1)
            .map(|(plane, _)| u64::from(plane.layout.width) * u64::from(plane.layout.height) * 4)
            .max()
            .unwrap_or(0);
        self.color = Some(color);
        Ok(self)
    }

    pub(crate) fn color_converted(&self) -> bool {
        self.color.is_some()
    }

    pub(crate) fn planes(&self) -> &[ModularOutputPlane] {
        &self.planes
    }

    pub(crate) fn output_extent(&self) -> Extent2d {
        let plane = self.planes[0].layout;
        Extent2d::new(plane.width, plane.height)
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
            color: self.color.as_ref().map(|plan| plan.allocate(device)),
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
    color: Option<color::ReconstructionBuffers>,
    pub output: wgpu::Buffer,
    scratch: Option<wgpu::Buffer>,
    weights: Vec<ResidentUpsampleWeights>,
}

pub(crate) struct ModularRenderPipeline {
    color: std::sync::OnceLock<Result<color::ColorPipeline>>,
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
            color: std::sync::OnceLock::new(),
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
        if let Some(color) = &plan.color {
            let pipeline = self
                .color
                .get_or_init(|| color::ColorPipeline::new(device))
                .as_ref()
                .map_err(Clone::clone)?;
            let buffers_color = buffers.color.as_ref().ok_or(ModularRenderError::Invalid {
                reason: "missing color render buffers",
            })?;
            let weights = plan
                .kernels
                .iter()
                .position(|kernel| kernel.factor() == plan.factors[0])
                .map(|index| &buffers.weights[index]);
            uniforms.extend(pipeline.encode(
                device,
                encoder,
                color::ColorInputs {
                    plan: color,
                    buffers: buffers_color,
                    source: input,
                    sources,
                    output: binding(&buffers.output)?,
                    weights,
                },
            )?);
        }
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
            if plan.color.is_some() && index < 3 {
                continue;
            }
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
                uniforms.push(
                    self.upsample.encode(
                        device,
                        encoder,
                        ResidentUpsampleInputs {
                            input: ResidentF32Plane {
                                storage: normalized,
                                width: source.width,
                                height: source.height,
                                stride: source.width,
                            }
                            .into(),
                            output: ResidentF32Plane {
                                storage: output_binding,
                                width: plane.width,
                                height: plane.height,
                                stride: plane.row_stride_words,
                            },
                            weights: &buffers.weights[kernel_index],
                        },
                    )?,
                );
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
        let plan = ModularRenderPlan::new(
            sources,
            factors
                .into_iter()
                .map(|factor| ChannelResampling { extent, factor })
                .collect(),
            &weights(),
            &limits,
        )
        .unwrap();
        let alignment = u64::from(limits.min_storage_buffer_offset_alignment).max(4);
        assert_eq!(
            plan.output_bytes,
            (17 * 9 * 4u64).div_ceil(alignment) * alignment * 4
        );
        assert_eq!(plan.scratch_bytes, 9 * 5 * 4);
        assert_eq!(plan.weight_bytes, (4 + 16 + 64) * 25 * 4);
        assert_eq!(plan.uniform_bytes, 4 * 32 + 3 * 48);
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
            vec![plane(extent, 2, 8); 2],
            vec![ChannelResampling { extent, factor: 2 }; 2],
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
            ModularRenderPlan::new(
                vec![source],
                vec![ChannelResampling { extent, factor: 2 }],
                &weights(),
                &limits
            ),
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
            ModularRenderPlan::new(
                vec![source],
                vec![ChannelResampling { extent, factor: 2 }],
                &weights(),
                &small_workgroup
            ),
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
                ModularRenderPlan::new(
                    vec![source],
                    vec![ChannelResampling { extent, factor }],
                    &weights(),
                    &limits
                ),
                Err(ModularRenderError::Invalid { .. })
            ));
        }
        assert!(matches!(
            ModularRenderPlan::new(
                vec![ModularOutputPlane {
                    layout: GpuModularChannelLayout {
                        word_offset: u32::MAX,
                        ..source.layout
                    },
                    ..source
                }],
                vec![ChannelResampling { extent, factor: 2 }],
                &weights(),
                &limits
            ),
            Err(ModularRenderError::Limit {
                resource: "source address words",
                ..
            })
        ));
    }

    #[test]
    fn early_extra_planes_do_not_inherit_the_coded_color_extent() {
        let coded = Extent2d::new(13, 9);
        let full = Extent2d::new(25, 17);
        let limits = wgpu::Limits::default();
        let resampling = crate::frame_resampling::FrameResampling::new(full, 2, &[2, 8]);
        let stage = crate::frame_surface::FrameRenderStage::BeforeFeatures;
        let plan = ModularRenderPlan::new(
            vec![plane(coded, 1, 8), plane(full, 2, 16), plane(full, 8, 8)],
            vec![
                resampling.color(stage),
                resampling.extra(2, stage),
                resampling.extra(8, stage),
            ],
            &weights(),
            &limits,
        )
        .unwrap();
        assert_eq!(plan.factors, [1, 2, 8]);
        let layouts: Vec<_> = plan.planes().iter().map(|plane| plane.layout).collect();
        assert_eq!(
            (layouts[0].width, layouts[0].height, layouts[0].word_offset),
            (13, 9, 0)
        );
        assert_eq!(
            (layouts[1].width, layouts[1].height, layouts[1].word_offset),
            (25, 17, 128)
        );
        assert_eq!(
            (layouts[2].width, layouts[2].height, layouts[2].word_offset),
            (25, 17, 576)
        );
        assert_eq!(plan.output_bytes, 4096);
        assert_eq!(plan.scratch_bytes, 13 * 9 * 4);
        assert_eq!(plan.uniform_bytes, 3 * 32 + 2 * 48);
    }
}
