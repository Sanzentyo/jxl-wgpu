//! Presentation of validated LF dependencies, independently of their retained prediction planes.

use std::num::NonZeroU64;
use std::sync::Arc;

use jxl_gpu_bitstream::ImageHeaderInventory;
use jxl_gpu_formats::ImageLayout;
use jxl_gpu_protocol::{Extent2d, OutputOrientation};
use jxl_wgpu::{
    GpuBufferLease, ResidentF32Plane, ResidentStorageBinding, ResidentUpsampleInputs,
    ResidentUpsampleKernel, ResidentUpsamplePipeline, WgpuBackend,
};

use super::submission::{GpuWork, completion_fence_bytes, submit_recorded, validate_size};
use crate::color_output::{
    ColorOutputConfig, ColorOutputInputs, ColorOutputPacker, ColorOutputPlan, ColorOutputPlane,
    ColorOutputTransform, InverseOpsin,
};
use crate::progressive_dc::ProgressiveDcXybPlanes;
use crate::{Error, GpuOutputRequest, Result};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

#[derive(Clone)]
pub(super) struct LfPreview {
    backend: WgpuBackend,
    config: ColorOutputConfig,
    pub(super) layout: ImageLayout,
    pub(super) surface: Option<crate::frame_surface::FrameSurfaceLayout>,
    pub(super) compositor: Option<Arc<super::gpu::Compositor>>,
    extra_count: usize,
    output_plan: ColorOutputPlan,
    output_storage_bytes: u64,
    kernel: Arc<ResidentUpsampleKernel>,
    upsample: Arc<ResidentUpsamplePipeline>,
    packer: Arc<ColorOutputPacker>,
}

impl std::fmt::Debug for LfPreview {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LfPreview")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl LfPreview {
    pub(super) fn new(
        backend: WgpuBackend,
        image: &ImageHeaderInventory,
        request: &GpuOutputRequest,
    ) -> Result<Self> {
        let canonical =
            !image.extra_channels.is_empty() || request.mapping() != crate::GpuOutputMapping::Color;
        let compositor = canonical
            .then(|| {
                super::gpu::Compositor::new(
                    backend.clone(),
                    Extent2d::new(image.width, image.height),
                    &image.extra_channels,
                    image.grayscale,
                    image.bit_depth,
                    OutputOrientation::from_exif_value(image.orientation).ok_or(
                        Error::InvalidImageOrientation {
                            value: image.orientation,
                        },
                    )?,
                    request,
                )
            })
            .transpose()?
            .map(Arc::new);
        let working = request
            .clone()
            .for_frame_surface(crate::frame_surface::FrameSurfaceEncoding::Linear);
        let render_request = if canonical { &working } else { request };
        let config = ColorOutputConfig {
            extent: Extent2d::new(image.width, image.height),
            orientation: render_request.orientation_policy().resolve(
                OutputOrientation::from_exif_value(image.orientation).ok_or(
                    Error::InvalidImageOrientation {
                        value: image.orientation,
                    },
                )?,
            ),
            transform: ColorOutputTransform::Xyb(InverseOpsin::from_image(image).ok_or(
                Error::EngineContract("LF presentation has no inverse opsin metadata"),
            )?),
            alpha_conversion: render_request.alpha_conversion(&image.extra_channels),
        };
        let layout = ImageLayout::packed(config.output_extent(), render_request.format().clone())?;
        config.validate_layout(&layout)?;
        let device = backend.device();
        let output_plan = ColorOutputPlan::for_limits(&layout, &device.limits())?;
        let kernel = ResidentUpsampleKernel::from_compact(
            8,
            &image
                .upsampling_weights
                .up8
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
        )?;
        let preview = Self {
            surface: None,
            compositor,
            extra_count: image.extra_channels.len(),
            config,
            layout,
            output_plan,
            output_storage_bytes: output_plan.memory.output_storage_bytes,
            kernel: Arc::new(kernel),
            upsample: Arc::new(ResidentUpsamplePipeline::new(device)?),
            packer: Arc::new(ColorOutputPacker::new(device)?),
            backend,
        };
        if canonical {
            preview.for_surface(
                config.extent,
                crate::frame_surface::FrameSurfaceEncoding::Linear,
            )
        } else {
            Ok(preview)
        }
    }

    /// Reuse the image's renderer for a physical layer in the compositor's canonical storage.
    pub(super) fn for_surface(
        &self,
        extent: Extent2d,
        encoding: crate::frame_surface::FrameSurfaceEncoding,
    ) -> Result<Self> {
        let surface = crate::frame_surface::FrameSurfaceLayout::with_encoding(
            extent,
            self.extra_count,
            encoding,
            &self.backend.device().limits(),
        )?;
        let config = ColorOutputConfig {
            extent,
            orientation: OutputOrientation::Identity,
            alpha_conversion: jxl_wgpu::AlphaConversion::Preserve,
            ..self.config
        };
        config.validate_layout(&surface.color)?;
        let output_plan =
            ColorOutputPlan::for_limits(&surface.color, &self.backend.device().limits())?;
        Ok(Self {
            config,
            layout: surface.color.clone(),
            surface: Some(surface.clone()),
            output_plan,
            output_storage_bytes: surface.storage_bytes,
            ..self.clone()
        })
    }

    #[cfg(test)]
    pub(super) fn submit(&self, planes: &ProgressiveDcXybPlanes, level: u8) -> Result<GpuWork> {
        self.submit_with_extras(planes, None, level)
    }

    pub(super) fn submit_with_extras(
        &self,
        planes: &ProgressiveDcXybPlanes,
        extras: Option<&crate::progressive_dc::ProgressiveDcExtras>,
        level: u8,
    ) -> Result<GpuWork> {
        if !(1..=4).contains(&level) {
            return Err(Error::EngineContract(
                "LF presentation level is outside 1 through 4",
            ));
        }
        let extent_at = |level| {
            let divisor = 1_u32 << (3 * level);
            Extent2d::new(
                self.config.extent.width.div_ceil(divisor),
                self.config.extent.height.div_ceil(divisor),
            )
        };
        let input_extent = extent_at(level);
        let planes = planes
            .clone()
            .into_extent([input_extent.width, input_extent.height])?;
        // Each recursive dependency is reconstructed with the image's Up8 kernel. Clip to the
        // exact grid before the next stage so odd borders never sample padding as image data.
        if extras.map_or(0, |extras| extras.planes.len()) != self.extra_count {
            return Err(Error::EngineContract(
                "LF presentation extra channel count mismatch",
            ));
        }
        if extras.is_some_and(|extras| {
            extras.planes.iter().any(|plane| {
                plane.width < input_extent.width
                    || plane.height < input_extent.height
                    || plane.row_stride_words < plane.width
            })
        }) {
            return Err(Error::EngineContract(
                "LF presentation extra channel extent mismatch",
            ));
        }
        let channels = 3 + self.extra_count;
        let extents = (0..level).rev().map(extent_at).collect::<Vec<_>>();
        let device = self.backend.device();
        let sizes = extents
            .iter()
            .map(|extent| {
                let size = u64::from(extent.width) * u64::from(extent.height) * 4;
                validate_size(device, size)?;
                Ok(size)
            })
            .collect::<Result<Vec<_>>>()?;
        let transient_bytes = sizes.iter().sum::<u64>() * channels as u64
            + u64::from(level) * channels as u64 * ResidentUpsamplePipeline::UNIFORM_BYTES
            + self.kernel.weight_bytes()
            + self.output_plan.memory.transient_bytes
            + completion_fence_bytes();
        let poll = self.backend.submission_poller().try_reserve()?;
        let mut permit = self
            .backend
            .transient_memory_budget()
            .try_reserve(transient_bytes + self.output_storage_bytes)?;
        let output_permit = permit
            .split_off(self.output_storage_bytes)
            .map_err(crate::progressive_dc::ProgressiveDcGpuError::from)?;
        let output = GpuBufferLease::from_tracked(
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("JPEG XL LF intermediate output"),
                size: self.output_storage_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            output_permit,
        );
        let stages = sizes
            .iter()
            .map(|&size| {
                (0..channels)
                    .map(|_| {
                        device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("JPEG XL LF intermediate XYB"),
                            size,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let weights = self.kernel.upload(device)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("JPEG XL LF intermediate rendering"),
        });
        let mut uniforms = Vec::with_capacity(usize::from(level) * channels);
        for (index, (stage, extent)) in stages.iter().zip(&extents).enumerate() {
            for (channel, output) in stage.iter().enumerate() {
                let input = if index == 0 && channel >= 3 {
                    let extras = extras.expect("validated LF extras");
                    let plane = extras.planes[channel - 3];
                    ResidentF32Plane {
                        storage: ResidentStorageBinding {
                            buffer: extras.buffer.as_wgpu_buffer(),
                            offset: u64::from(plane.word_offset) * 4,
                            size: NonZeroU64::new(
                                (u64::from(input_extent.height - 1)
                                    * u64::from(plane.row_stride_words)
                                    + u64::from(input_extent.width))
                                    * 4,
                            )
                            .expect("nonempty extra plane"),
                        },
                        width: input_extent.width,
                        height: input_extent.height,
                        stride: plane.row_stride_words,
                    }
                } else if index == 0 {
                    let plane = &planes.planes[channel];
                    ResidentF32Plane {
                        storage: binding(plane.buffer.as_wgpu_buffer()),
                        width: plane.width,
                        height: plane.height,
                        stride: plane.stride,
                    }
                } else {
                    let prior = extents[index - 1];
                    ResidentF32Plane {
                        storage: binding(&stages[index - 1][channel]),
                        width: prior.width,
                        height: prior.height,
                        stride: prior.width,
                    }
                };
                uniforms.push(self.upsample.encode(
                    device,
                    &mut encoder,
                    ResidentUpsampleInputs {
                        input: input.into(),
                        output: ResidentF32Plane {
                            storage: binding(output),
                            width: extent.width,
                            height: extent.height,
                            stride: extent.width,
                        },
                        weights: &weights,
                    },
                )?);
            }
        }
        let scratch = self.packer.encode(
            device,
            &mut encoder,
            ColorOutputInputs {
                planes: std::array::from_fn(|channel| ColorOutputPlane {
                    storage: binding(&stages.last().expect("nonzero LF level")[channel]),
                    width: self.config.extent.width,
                    height: self.config.extent.height,
                    stride: self.config.extent.width,
                }),
                alpha: None,
                output: binding(output.as_wgpu_buffer()),
                layout: &self.layout,
                config: self.config,
            },
        )?;
        if let Some(surface) = &self.surface {
            for (channel, layout) in surface.extras.iter().enumerate() {
                encoder.copy_buffer_to_buffer(
                    &stages.last().expect("nonzero LF level")[3 + channel],
                    0,
                    output.as_wgpu_buffer(),
                    layout.planes[0].offset,
                    u64::from(self.config.extent.width) * u64::from(self.config.extent.height) * 4,
                );
            }
        }
        submit_recorded(
            &self.backend,
            encoder,
            output,
            planes
                .planes
                .iter()
                .map(|plane| plane.buffer.clone())
                .chain(extras.map(|extra| extra.buffer.clone()))
                .collect(),
            (stages, weights, uniforms, scratch),
            permit,
            poll,
        )
    }
}

fn binding(buffer: &wgpu::Buffer) -> ResidentStorageBinding<'_> {
    ResidentStorageBinding {
        buffer,
        offset: 0,
        size: NonZeroU64::new(buffer.size()).expect("validated nonempty buffer"),
    }
}
