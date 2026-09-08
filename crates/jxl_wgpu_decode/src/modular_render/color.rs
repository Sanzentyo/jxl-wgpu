//! Frame-wide Modular color reconstruction, restoration, and conversion before composition.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{
    EdgePreservingFilterInventory, FrameInventory, ImageHeaderInventory, RestorationFilterInventory,
};
use jxl_gpu_formats::{ColorSpecification, ImageLayout, PixelFormat, RgbChannelOrder};
use jxl_gpu_protocol::{Extent2d, OutputOrientation, RgbColorEncoding};
use jxl_wgpu::{
    ResidentEpfInputs, ResidentEpfParameters, ResidentEpfPipeline, ResidentEpfSigma,
    ResidentF32Plane, ResidentGaborishInputs, ResidentGaborishPipeline, ResidentGaborishWeights,
    ResidentStorageBinding, ResidentUpsampleInputs, ResidentUpsamplePipeline,
    ResidentUpsampleWeights,
};
use wgpu::util::DeviceExt;

use super::{ModularOutputPlane, ModularRenderError, Result, binding, invalid, require, resource};
use crate::color_output::{
    ColorOutputConfig, ColorOutputInputs, ColorOutputPacker, ColorOutputPlan, ColorOutputPlane,
    ColorOutputTransform, InverseOpsin,
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ModularColorConfig {
    xyb: Option<([f32; 3], InverseOpsin)>,
    gaborish: Option<ResidentGaborishWeights>,
    epf: Vec<ResidentEpfParameters>,
    inverse_sigma: f32,
}

impl ModularColorConfig {
    pub(crate) fn new(
        image: &ImageHeaderInventory,
        frame: &FrameInventory,
        lf: [f32; 3],
    ) -> Result<Option<Self>> {
        let (gaborish, epf) = crate::restoration::restoration_config(frame.restoration_filter)?;
        let sigma = match frame.restoration_filter {
            RestorationFilterInventory::Custom {
                epf:
                    EdgePreservingFilterInventory::Enabled {
                        sigma_for_modular, ..
                    },
                ..
            } => sigma_for_modular.map_or(1.0, |value| value.to_f32()),
            _ => 1.0,
        };
        if epf.is_some() && sigma < 1e-8 {
            return Err(crate::RestorationError::InvalidModularSigma { value: sigma }.into());
        }
        let xyb = if image.xyb_encoded {
            Some((
                lf.map(|value| value / 128.0),
                InverseOpsin::from_image(image).ok_or(ModularRenderError::Invalid {
                    reason: "XYB inverse opsin metadata is missing",
                })?,
            ))
        } else {
            None
        };
        Ok(
            (xyb.is_some() || gaborish.is_some() || epf.is_some()).then(|| Self {
                xyb,
                gaborish,
                epf: epf.map_or_else(Vec::new, |epf| epf.passes()),
                inverse_sigma: -1.171_572_9 / sigma,
            }),
        )
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NormalizeColorParams {
    sources: [[u32; 4]; 3], // width, height, stride, word offset
    encodings: [u32; 4],    // each source encoding, XYB flag
    multipliers: [f32; 4],  // LF dequantization in X/Y/B order
}
const _: () = assert!(std::mem::size_of::<NormalizeColorParams>() == 80);

#[derive(Debug)]
pub(super) struct ColorPlan {
    reconstruction: ReconstructionPlan,
    layout: ImageLayout,
    output_config: ColorOutputConfig,
    pub storage_bytes: u64,
    pub uniform_bytes: u64,
}

impl ColorPlan {
    pub(super) fn new(
        config: ModularColorConfig,
        extent: Extent2d,
        sources: &[ModularOutputPlane],
        factors: &[u32],
        planes: &[ModularOutputPlane],
        target: ColorSpecification,
        limits: &wgpu::Limits,
    ) -> Result<Self> {
        if sources.len() < 3 || factors.len() != sources.len() || planes.len() != sources.len() {
            return invalid("color reconstruction requires three selected color planes");
        }
        if factors[..3].iter().any(|&factor| factor != factors[0]) {
            return invalid("Modular color reconstruction requires equal color grids");
        }
        let output_config = ColorOutputConfig {
            extent,
            orientation: OutputOrientation::Identity,
            transform: config.xyb.map_or(
                ColorOutputTransform::Rgb(RgbColorEncoding::SRGB_BT709),
                |(_, inverse)| ColorOutputTransform::Xyb(inverse),
            ),
            alpha_conversion: jxl_wgpu::AlphaConversion::Preserve,
        };
        let reconstruction = ReconstructionPlan::new(config, extent, sources, factors[0], limits)?;
        let format = PixelFormat::rgb_f32(RgbChannelOrder::Rgb, true, target);
        let packed = ImageLayout::packed(extent, format.clone())?;
        let layouts = packed
            .planes
            .into_iter()
            .zip(planes)
            .map(|(mut plane, source)| {
                plane.offset = u64::from(source.layout.word_offset) * 4;
                plane.row_stride = u64::from(source.layout.row_stride_words) * 4;
                plane
            })
            .collect();
        let layout = ImageLayout::from_planes(extent, format, layouts)?;
        output_config.validate_layout(&layout)?;
        let packing = ColorOutputPlan::for_limits(&layout, limits)?;
        Ok(Self {
            storage_bytes: reconstruction.storage_bytes,
            uniform_bytes: reconstruction.uniform_bytes + packing.memory.uniform_bytes,
            reconstruction,
            layout,
            output_config,
        })
    }

    pub(super) fn allocate(&self, device: &wgpu::Device) -> ReconstructionBuffers {
        self.reconstruction.allocate(device)
    }
}

/// Shared pre-color-transform reconstruction for presentation and LF dependency frames.
#[derive(Debug)]
pub(crate) struct ReconstructionPlan {
    config: ModularColorConfig,
    source_extent: Extent2d,
    pub(super) output_extent: Extent2d,
    factor: u32,
    source_plane_bytes: u64,
    upsample_plane_bytes: u64,
    pub(super) storage_bytes: u64,
    pub(super) uniform_bytes: u64,
}

impl ReconstructionPlan {
    pub(super) fn new(
        config: ModularColorConfig,
        extent: Extent2d,
        sources: &[ModularOutputPlane],
        factor: u32,
        limits: &wgpu::Limits,
    ) -> Result<Self> {
        if sources.len() < 3
            || extent.width == 0
            || extent.height == 0
            || !matches!(factor, 1 | 2 | 4 | 8)
            || extent.width > i32::MAX as u32 / 2
            || extent.height > i32::MAX as u32 / 2
        {
            return invalid("color reconstruction extent, factor or plane count");
        }
        let source = sources[0].layout;
        if sources[..3].iter().any(|plane| {
            let layout = plane.layout;
            layout.width != extent.width.div_ceil(factor)
                || layout.height != extent.height.div_ceil(factor)
                || layout.row_stride_words < layout.width
                || layout.reserved != 0
        }) {
            return invalid("Modular color reconstruction requires equal color grids");
        }
        for plane in &sources[..3] {
            let layout = plane.layout;
            require(
                "color source address words",
                u64::from(layout.word_offset)
                    + u64::from(layout.height - 1) * u64::from(layout.row_stride_words)
                    + u64::from(layout.width),
                u64::from(u32::MAX),
            )?;
        }
        jxl_wgpu::KernelVariant::Tile16x16
            .validate_for("modular_render", limits, 0)
            .map_err(|_| jxl_wgpu::ResidentUpsampleError::WorkgroupVariant {
                variant: jxl_wgpu::KernelVariant::Tile16x16,
            })?;
        for dimension in [extent.width, extent.height] {
            require(
                "color workgroups",
                u64::from(dimension.div_ceil(16)),
                u64::from(limits.max_compute_workgroups_per_dimension),
            )?;
        }
        require(
            "color uniform bytes",
            std::mem::size_of::<NormalizeColorParams>() as u64,
            limits.max_uniform_buffer_binding_size,
        )?;
        let source_plane_bytes = u64::from(source.width) * u64::from(source.height) * 4;
        let upsample_plane_bytes = if factor == 1 {
            0
        } else {
            u64::from(extent.width) * u64::from(extent.height) * 4
        };
        let limit = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size)
            .min(u64::from(u32::MAX) * 4);
        require("color source plane", source_plane_bytes, limit)?;
        require("color upsampling plane", upsample_plane_bytes, limit)?;
        let storage_bytes = source_plane_bytes
            * if config.gaborish.is_some() || !config.epf.is_empty() {
                6
            } else {
                3
            }
            + upsample_plane_bytes * 3;
        let uniform_bytes = std::mem::size_of::<NormalizeColorParams>() as u64
            + config
                .gaborish
                .map_or(0, |_| jxl_wgpu::ResidentGaborishMemoryPlan::UNIFORM_BYTES)
            + config.epf.len() as u64 * jxl_wgpu::ResidentEpfMemoryPlan::UNIFORM_BYTES
            + if factor == 1 {
                0
            } else {
                3 * ResidentUpsamplePipeline::UNIFORM_BYTES
            };
        Ok(Self {
            config,
            source_extent: Extent2d::new(source.width, source.height),
            output_extent: extent,
            factor,
            source_plane_bytes,
            upsample_plane_bytes,
            storage_bytes,
            uniform_bytes,
        })
    }

    /// The final allocation is determined before submission, so its reservation can outlive scratch.
    pub(super) fn output_buffers<'a>(
        &self,
        buffers: &'a ReconstructionBuffers,
    ) -> &'a [wgpu::Buffer; 3] {
        if let Some(upsampled) = &buffers.upsampled {
            upsampled
        } else if (usize::from(self.config.gaborish.is_some()) + self.config.epf.len()) % 2 == 1 {
            buffers
                .scratch
                .as_ref()
                .expect("restoration scratch was planned")
        } else {
            &buffers.normalized
        }
    }
    pub(super) fn allocate(&self, device: &wgpu::Device) -> ReconstructionBuffers {
        let create = |label, size| {
            std::array::from_fn(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            })
        };
        ReconstructionBuffers {
            normalized: create("jxl-wgpu Modular decoded color", self.source_plane_bytes),
            scratch: (self.config.gaborish.is_some() || !self.config.epf.is_empty()).then(|| {
                create(
                    "jxl-wgpu Modular restoration scratch",
                    self.source_plane_bytes,
                )
            }),
            upsampled: (self.factor != 1).then(|| {
                create(
                    "jxl-wgpu Modular upsampled color",
                    self.upsample_plane_bytes,
                )
            }),
        }
    }
}
pub(crate) struct ReconstructionBuffers {
    normalized: [wgpu::Buffer; 3],
    scratch: Option<[wgpu::Buffer; 3]>,
    upsampled: Option<[wgpu::Buffer; 3]>,
}

pub(super) struct ColorPipeline {
    reconstruction: ReconstructionPipeline,
    output: ColorOutputPacker,
}

impl ColorPipeline {
    pub(super) fn new(device: &wgpu::Device) -> Result<Self> {
        Ok(Self {
            reconstruction: ReconstructionPipeline::new(device)?,
            output: ColorOutputPacker::new(device)?,
        })
    }

    pub(super) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: ColorInputs<'_>,
    ) -> Result<Vec<wgpu::Buffer>> {
        let ColorInputs {
            plan,
            buffers,
            source,
            sources,
            output,
            weights,
        } = inputs;
        let mut uniforms = self.reconstruction.encode(
            device,
            encoder,
            ReconstructionInputs {
                plan: &plan.reconstruction,
                buffers,
                source,
                sources,
                weights,
            },
        )?;
        let packed = self.output.encode(
            device,
            encoder,
            ColorOutputInputs {
                planes: resident_planes(
                    plan.reconstruction.output_buffers(buffers),
                    plan.reconstruction.output_extent,
                )?
                .map(|plane| ColorOutputPlane {
                    storage: plane.storage,
                    width: plane.width,
                    height: plane.height,
                    stride: plane.stride,
                }),
                alpha: None,
                output,
                layout: &plan.layout,
                config: plan.output_config,
            },
        )?;
        uniforms.extend([packed.uniform, packed.source_uniform]);
        Ok(uniforms)
    }
}

pub(crate) struct ReconstructionPipeline {
    normalize: wgpu::ComputePipeline,
    gaborish: ResidentGaborishPipeline,
    epf: ResidentEpfPipeline,
    upsample: ResidentUpsamplePipeline,
}

impl ReconstructionPipeline {
    pub(crate) fn new(device: &wgpu::Device) -> Result<Self> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu Modular color normalization"),
            source: wgpu::ShaderSource::Wgsl(
                crate::modular_sample::shader(include_str!("color.wgsl")).into(),
            ),
        });
        Ok(Self {
            normalize: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu Modular color normalization"),
                layout: None,
                module: &module,
                entry_point: Some("normalize_color"),
                compilation_options: Default::default(),
                cache: None,
            }),
            gaborish: ResidentGaborishPipeline::new(device)?,
            epf: ResidentEpfPipeline::new(device)?,
            upsample: ResidentUpsamplePipeline::new(device)?,
        })
    }

    pub(super) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: ReconstructionInputs<'_>,
    ) -> Result<Vec<wgpu::Buffer>> {
        let ReconstructionInputs {
            plan,
            buffers,
            source,
            sources,
            weights,
        } = inputs;
        if sources.len() < 3 {
            return invalid("missing color source planes");
        }
        for plane in &sources[..3] {
            let layout = plane.layout;
            if layout.width != plan.source_extent.width
                || layout.height != plan.source_extent.height
                || layout.row_stride_words < layout.width
                || layout.reserved != 0
            {
                return invalid("color source geometry changed");
            }
            require(
                "color source address words",
                u64::from(layout.word_offset)
                    + u64::from(layout.height - 1) * u64::from(layout.row_stride_words)
                    + u64::from(layout.width),
                u64::from(u32::MAX),
            )?;
            require(
                "color source binding",
                (u64::from(layout.word_offset)
                    + u64::from(layout.height - 1) * u64::from(layout.row_stride_words)
                    + u64::from(layout.width))
                    * 4,
                source.size.get(),
            )?;
        }
        let params = NormalizeColorParams {
            sources: std::array::from_fn(|index| {
                let source = sources[index].layout;
                [
                    source.width,
                    source.height,
                    source.row_stride_words,
                    source.word_offset,
                ]
            }),
            encodings: [
                sources[0].encoding.packed(),
                sources[1].encoding.packed(),
                sources[2].encoding.packed(),
                u32::from(plan.config.xyb.is_some()),
            ],
            multipliers: plan.config.xyb.map_or([0.0; 4], |(values, _)| {
                [values[0], values[1], values[2], 0.0]
            }),
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu Modular color normalization parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let resources = [
            resource(source),
            buffers.normalized[0].as_entire_binding(),
            buffers.normalized[1].as_entire_binding(),
            buffers.normalized[2].as_entire_binding(),
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
            label: Some("jxl-wgpu Modular color normalization bindings"),
            layout: &self.normalize.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.normalize);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(
                plan.source_extent.width.div_ceil(16),
                plan.source_extent.height.div_ceil(16),
                1,
            );
        }
        let mut uniforms = vec![uniform];
        let mut current = &buffers.normalized;
        if let Some(scratch) = &buffers.scratch {
            let mut destination = scratch;
            if let Some(weights) = plan.config.gaborish {
                uniforms.push(self.gaborish.encode(
                    device,
                    encoder,
                    ResidentGaborishInputs {
                        inputs: resident_planes(current, plan.source_extent)?,
                        outputs: resident_planes(destination, plan.source_extent)?,
                        weights,
                    },
                )?);
                std::mem::swap(&mut current, &mut destination);
            }
            for &parameters in &plan.config.epf {
                uniforms.push(self.epf.encode(
                    device,
                    encoder,
                    ResidentEpfInputs {
                        inputs: resident_planes(current, plan.source_extent)?,
                        outputs: resident_planes(destination, plan.source_extent)?,
                        sigma: ResidentEpfSigma::Constant(plan.config.inverse_sigma),
                        parameters,
                    },
                )?);
                std::mem::swap(&mut current, &mut destination);
            }
        }
        if let Some(upsampled) = &buffers.upsampled {
            let weights = weights.ok_or(ModularRenderError::Invalid {
                reason: "missing color upsampling weights",
            })?;
            for (input, output) in resident_planes(current, plan.source_extent)?
                .into_iter()
                .zip(resident_planes(upsampled, plan.output_extent)?)
            {
                uniforms.push(self.upsample.encode(
                    device,
                    encoder,
                    ResidentUpsampleInputs {
                        input,
                        output,
                        weights,
                    },
                )?);
            }
        }
        Ok(uniforms)
    }
}

pub(super) struct ColorInputs<'a> {
    pub plan: &'a ColorPlan,
    pub buffers: &'a ReconstructionBuffers,
    pub source: ResidentStorageBinding<'a>,
    pub sources: &'a [ModularOutputPlane],
    pub output: ResidentStorageBinding<'a>,
    pub weights: Option<&'a ResidentUpsampleWeights>,
}

pub(super) struct ReconstructionInputs<'a> {
    pub plan: &'a ReconstructionPlan,
    pub buffers: &'a ReconstructionBuffers,
    pub source: ResidentStorageBinding<'a>,
    pub sources: &'a [ModularOutputPlane],
    pub weights: Option<&'a ResidentUpsampleWeights>,
}

fn resident_planes(
    buffers: &[wgpu::Buffer; 3],
    extent: Extent2d,
) -> Result<[ResidentF32Plane<'_>; 3]> {
    let plane = |buffer| -> Result<_> {
        Ok(ResidentF32Plane {
            storage: binding(buffer)?,
            width: extent.width,
            height: extent.height,
            stride: extent.width,
        })
    };
    Ok([
        plane(&buffers[0])?,
        plane(&buffers[1])?,
        plane(&buffers[2])?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular_sample::ModularSampleEncoding;
    use crate::modular_transform::GpuModularChannelLayout;

    #[test]
    fn lf_reconstruction_retains_only_final_plane_reservations() {
        use crate::modular_render::ModularLfPlan;
        use jxl_gpu_bitstream::{FiniteF32, GaborishInventory, UpsamplingWeightsInventory};
        use jxl_wgpu::{MemoryBudget, WgpuBackend, WgpuBackendConfig};
        let Ok(backend) = pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
            enable_timestamps: false,
            ..Default::default()
        })) else {
            return;
        };
        let extent = Extent2d::new(37, 17);
        let weight = FiniteF32::from_f32(0.03125).unwrap();
        let weights = UpsamplingWeightsInventory {
            up2: [weight; 15],
            up4: [weight; 55],
            up8: [weight; 210],
        };
        for factor in [1, 2, 4, 8] {
            let source = ModularOutputPlane::new(
                GpuModularChannelLayout {
                    width: extent.width.div_ceil(factor),
                    height: extent.height.div_ceil(factor),
                    row_stride_words: extent.width.div_ceil(factor) + 3,
                    word_offset: 7,
                    hshift: 0,
                    vshift: 0,
                    bit_depth: 8,
                    reserved: 0,
                },
                ModularSampleEncoding::integer(8).unwrap(),
            );
            for gaborish in [false, true] {
                for iterations in 0..=3 {
                    let config = ModularColorConfig {
                        xyb: None,
                        gaborish: gaborish.then_some(ResidentGaborishWeights::DEFAULT),
                        epf: if iterations == 0 {
                            Vec::new()
                        } else {
                            crate::restoration::restoration_config(
                                RestorationFilterInventory::Custom {
                                    gaborish: GaborishInventory::Disabled,
                                    epf: EdgePreservingFilterInventory::Enabled {
                                        iterations,
                                        sharp_lut: None,
                                        weights: None,
                                        sigma: None,
                                        sigma_for_modular: None,
                                    },
                                },
                            )
                            .unwrap()
                            .1
                            .unwrap()
                            .passes()
                        },
                        inverse_sigma: -1.171_572_9,
                    };
                    let plan = ModularLfPlan::new(
                        config,
                        extent,
                        &[source; 3],
                        factor,
                        &weights,
                        &backend.device().limits(),
                    )
                    .unwrap();
                    let budget =
                        MemoryBudget::new(std::num::NonZeroU64::new(plan.total_bytes()).unwrap());
                    let mut permit = budget.try_reserve(plan.total_bytes()).unwrap();
                    let (buffers, planes) = plan.allocate(backend.device(), &mut permit).unwrap();
                    assert_eq!((planes.width(), planes.height()), (37, 17));
                    assert_eq!(plan.plane_bytes(), 37 * 17 * 4 * 3);
                    assert_eq!(permit.bytes(), plan.total_bytes() - plan.plane_bytes());
                    assert_eq!(budget.snapshot().reserved_bytes, plan.total_bytes());
                    drop(buffers);
                    drop(permit);
                    assert_eq!(budget.snapshot().reserved_bytes, plan.plane_bytes());
                    let shared = planes.clone();
                    drop(planes);
                    assert_eq!(budget.snapshot().reserved_bytes, plan.plane_bytes());
                    drop(shared);
                    assert_eq!(budget.snapshot().reserved_bytes, 0);
                }
            }
        }
    }

    #[test]
    fn color_plan_accounts_restoration_and_upsampling_and_checks_contracts() {
        let extent = Extent2d::new(37, 17);
        let layout = GpuModularChannelLayout {
            width: 19,
            height: 9,
            row_stride_words: 21,
            word_offset: 3,
            hshift: 0,
            vshift: 0,
            bit_depth: 8,
            reserved: 0,
        };
        let encoding = ModularSampleEncoding::integer(8).unwrap();
        let source = ModularOutputPlane::new(layout, encoding);
        let planes: Vec<_> = (0..3)
            .map(|index| {
                ModularOutputPlane::new(
                    GpuModularChannelLayout {
                        width: 37,
                        height: 17,
                        row_stride_words: 37,
                        word_offset: index * 640,
                        ..layout
                    },
                    encoding,
                )
            })
            .collect();
        let config = ModularColorConfig {
            xyb: None,
            gaborish: Some(ResidentGaborishWeights::DEFAULT),
            epf: crate::restoration::restoration_config(RestorationFilterInventory::Default)
                .unwrap()
                .1
                .unwrap()
                .passes(),
            inverse_sigma: -1.171_572_9,
        };
        let limits = wgpu::Limits::default();
        let target = crate::vardct_rgb8_format().color_spec;
        let plan = ColorPlan::new(
            config.clone(),
            extent,
            &[source; 3],
            &[2; 3],
            &planes,
            target,
            &limits,
        )
        .unwrap();
        assert_eq!(plan.storage_bytes, 6 * 19 * 9 * 4 + 3 * 37 * 17 * 4);
        assert_eq!(plan.uniform_bytes, 80 + 80 + 2 * 80 + 3 * 32 + 352);
        assert_eq!(
            plan.layout
                .planes
                .iter()
                .map(|plane| plane.offset)
                .collect::<Vec<_>>(),
            vec![0, 2560, 5120]
        );
        assert!(matches!(
            ColorPlan::new(
                config.clone(),
                extent,
                &[source; 3],
                &[2, 4, 2],
                &planes,
                target,
                &limits
            ),
            Err(ModularRenderError::Invalid { .. })
        ));
        let limited = wgpu::Limits {
            max_storage_buffer_binding_size: 2500,
            ..limits
        };
        assert!(matches!(
            ColorPlan::new(
                config,
                extent,
                &[source; 3],
                &[2; 3],
                &planes,
                target,
                &limited
            ),
            Err(ModularRenderError::Limit {
                resource: "color upsampling plane",
                required: 2516,
                available: 2500
            })
        ));
    }
}
