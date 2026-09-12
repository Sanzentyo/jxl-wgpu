//! Complete splines, deferred upsampling and noise on fresh, accounted component planes.

use std::num::NonZeroU64;
use std::sync::Arc;

use jxl_gpu_bitstream::{FrameInventory, ImageHeaderInventory};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{
    GpuBufferLease, MemoryPermit, ResidentF32Plane, ResidentNoiseParameters, ResidentNoisePlan,
    ResidentStorageBinding, ResidentUpsampleInputs, ResidentUpsampleKernel,
    ResidentUpsamplePipeline, WgpuBackend,
};

use super::gpu::Surface;
use super::splines;
use super::submission::{GpuWork, completion_fence_bytes, submit_recorded};
use crate::frame_surface::{FrameSurfaceEncoding, FrameSurfaceLayout};
use crate::progressive_dc::{
    ProgressiveDcExtras, ProgressiveDcGpuError, ProgressiveDcOutput, ProgressiveDcXybPlanes,
};
use crate::{Error, Result};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

#[derive(Debug)]
pub(super) struct Plan {
    input_extent: Extent2d,
    output_extent: Extent2d,
    extra_count: usize,
    upsample: Option<ResidentUpsampleKernel>,
    noise: Option<ResidentNoisePlan>,
    splines: Option<Arc<splines::Cache>>,
}

impl Plan {
    pub(super) fn new(
        image: &ImageHeaderInventory,
        frame: &FrameInventory,
        noise: Option<ResidentNoiseParameters>,
        splines: Option<Arc<splines::Cache>>,
        limits: &wgpu::Limits,
    ) -> Result<Option<Arc<Self>>> {
        if frame.upsampling == 1 && noise.is_none() && splines.is_none() {
            return Ok(None);
        }
        let divisor = 1_u32 << (3 * frame.lf_level);
        let output_extent = Extent2d::new(
            frame.width.div_ceil(divisor),
            frame.height.div_ceil(divisor),
        );
        let input_extent = Extent2d::new(
            output_extent.width.div_ceil(frame.upsampling),
            output_extent.height.div_ceil(frame.upsampling),
        );
        let upsample = (frame.upsampling != 1)
            .then(|| {
                crate::modular_render::upsample_kernel(&image.upsampling_weights, frame.upsampling)
            })
            .transpose()?;
        let noise = noise
            .map(|parameters| ResidentNoisePlan::new(output_extent, parameters, limits))
            .transpose()?;
        Ok(Some(Arc::new(Self {
            input_extent,
            output_extent,
            extra_count: image.extra_channels.len(),
            upsample,
            noise,
            splines,
        })))
    }

    fn scratch_bytes(&self) -> u64 {
        completion_fence_bytes()
            + self.upsample.as_ref().map_or(0, |kernel| {
                kernel.weight_bytes()
                    + (3 + self.extra_count) as u64 * ResidentUpsamplePipeline::UNIFORM_BYTES
            })
            + self
                .noise
                .as_ref()
                .map_or(0, ResidentNoisePlan::total_bytes)
            + self.splines.as_ref().map_or(0, |cache| {
                cache.scratch_bytes()
                    + if self.upsample.is_some() {
                        u64::from(self.input_extent.width)
                            * u64::from(self.input_extent.height)
                            * 12
                    } else {
                        0
                    }
            })
    }

    fn layout(&self, backend: &WgpuBackend) -> Result<FrameSurfaceLayout> {
        Ok(FrameSurfaceLayout::with_encoding(
            self.output_extent,
            self.extra_count,
            FrameSurfaceEncoding::Encoded,
            &backend.device().limits(),
        )?)
    }

    pub(super) fn render(
        &self,
        backend: &WgpuBackend,
        source: &Surface,
    ) -> Result<GpuWork<Surface>> {
        if source.encoding != FrameSurfaceEncoding::Encoded || source.extent != self.input_extent {
            return Err(Error::EngineContract(
                "deferred features require coded component geometry",
            ));
        }
        let layout = self.layout(backend)?;
        let poll = backend.submission_poller().try_reserve()?;
        let mut permit = backend
            .transient_memory_budget()
            .try_reserve(layout.storage_bytes + self.scratch_bytes())?;
        let output = Surface {
            buffer: retained(backend.device(), &mut permit, layout.storage_bytes)?,
            extent: self.output_extent,
            plane_words: (layout.plane_bytes / 4) as u32,
            encoding: FrameSurfaceEncoding::Encoded,
        };
        let input_planes = surface_planes(source, 3 + self.extra_count);
        let output_planes = surface_planes(&output, 3 + self.extra_count);
        let mut encoder = backend.device().create_command_encoder(&Default::default());
        let resources = self.record(backend, &mut encoder, &input_planes, &output_planes)?;
        submit_recorded(
            backend,
            encoder,
            output,
            vec![source.buffer.clone()],
            resources,
            permit,
            poll,
        )
    }

    pub(super) fn render_lf(
        &self,
        backend: &WgpuBackend,
        source: &ProgressiveDcOutput,
        retain_extras: bool,
    ) -> Result<GpuWork<ProgressiveDcOutput>> {
        if source.xyb.width() != self.input_extent.width
            || source.xyb.height() != self.input_extent.height
            || source.extras.as_ref().map_or(0, |extra| extra.planes.len()) != self.extra_count
        {
            return Err(Error::EngineContract("deferred LF feature plane geometry"));
        }
        let layout = self.layout(backend)?;
        let color_bytes =
            u64::from(self.output_extent.width) * u64::from(self.output_extent.height) * 4;
        // Every extra participates in late upsampling. Final-only completion drops its separate
        // output after the submission, rather than keeping it charged through an LF prediction.
        let extra_bytes = layout.plane_bytes * self.extra_count as u64;
        let poll = backend.submission_poller().try_reserve()?;
        let mut permit = backend
            .transient_memory_budget()
            .try_reserve(color_bytes * 3 + extra_bytes + self.scratch_bytes())?;
        let [x, y, b] = [(); 3].map(|()| retained(backend.device(), &mut permit, color_bytes));
        let output = ProgressiveDcOutput {
            xyb: ProgressiveDcXybPlanes::from_leases(
                [x?, y?, b?],
                self.output_extent.width,
                self.output_extent.height,
                self.output_extent.width,
            )?,
            extras: source
                .extras
                .as_ref()
                .map(|extra| {
                    Ok::<_, Error>(ProgressiveDcExtras {
                        buffer: retained(backend.device(), &mut permit, extra_bytes)?,
                        planes: extra
                            .planes
                            .iter()
                            .enumerate()
                            .map(|(index, plane)| {
                                crate::modular_transform::GpuModularChannelLayout {
                                    width: self.output_extent.width,
                                    height: self.output_extent.height,
                                    row_stride_words: self.output_extent.width,
                                    word_offset: index as u32 * (layout.plane_bytes / 4) as u32,
                                    ..*plane
                                }
                            })
                            .collect(),
                    })
                })
                .transpose()?,
        };
        let input_planes = lf_planes(source);
        let output_planes = lf_planes(&output);
        let mut encoder = backend.device().create_command_encoder(&Default::default());
        let resources = self.record(backend, &mut encoder, &input_planes, &output_planes)?;
        let mut result = output.clone();
        if !retain_extras {
            result.extras = None;
        }
        submit_recorded(
            backend,
            encoder,
            result,
            source
                .xyb
                .planes
                .iter()
                .map(|plane| plane.buffer.clone())
                .chain(source.extras.as_ref().map(|extra| extra.buffer.clone()))
                .collect(),
            (resources, output),
            permit,
            poll,
        )
    }

    fn record(
        &self,
        backend: &WgpuBackend,
        encoder: &mut wgpu::CommandEncoder,
        inputs: &[ResidentF32Plane<'_>],
        outputs: &[ResidentF32Plane<'_>],
    ) -> Result<Resources> {
        let device = backend.device();
        if inputs.len() != 3 + self.extra_count || outputs.len() != inputs.len() {
            return Err(Error::EngineContract("deferred feature channel count"));
        }
        for (planes, extent, usage) in [
            (inputs, self.input_extent, wgpu::BufferUsages::COPY_SRC),
            (outputs, self.output_extent, wgpu::BufferUsages::COPY_DST),
        ] {
            for plane in planes {
                let storage = plane.storage;
                let bytes = u64::from(plane.height.saturating_sub(1))
                    .checked_mul(u64::from(plane.stride))
                    .and_then(|words| words.checked_add(u64::from(plane.width)))
                    .and_then(|words| words.checked_mul(4));
                if plane.width != extent.width
                    || plane.height != extent.height
                    || plane.width == 0
                    || plane.height == 0
                    || plane.stride < plane.width
                    || bytes.is_none_or(|bytes| bytes > storage.size.get())
                    || storage
                        .offset
                        .checked_add(storage.size.get())
                        .is_none_or(|end| end > storage.buffer.size())
                    || !storage.offset.is_multiple_of(
                        u64::from(device.limits().min_storage_buffer_offset_alignment).max(4),
                    )
                    || !storage
                        .buffer
                        .usage()
                        .contains(usage | wgpu::BufferUsages::STORAGE)
                {
                    return Err(Error::EngineContract(
                        "deferred feature plane range or usage",
                    ));
                }
            }
        }
        let weights = self
            .upsample
            .as_ref()
            .map(|kernel| kernel.upload(device))
            .transpose()?;
        let mut uniforms = Vec::new();
        let coded_planes: Vec<_> = if self.splines.is_some() && weights.is_some() {
            (0..3)
                .map(|_| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("JPEG XL spline components before upsampling"),
                        size: u64::from(self.input_extent.width)
                            * u64::from(self.input_extent.height)
                            * 4,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut rendered_inputs = inputs.to_vec();
        if let Some(cache) = self.splines.as_ref().filter(|_| weights.is_some()) {
            for (index, buffer) in coded_planes.iter().enumerate() {
                let plane = ResidentF32Plane {
                    storage: ResidentStorageBinding {
                        buffer,
                        offset: 0,
                        size: NonZeroU64::new(buffer.size()).expect("nonempty spline plane"),
                    },
                    width: self.input_extent.width,
                    height: self.input_extent.height,
                    stride: self.input_extent.width,
                };
                copy_plane(encoder, inputs[index], plane);
                rendered_inputs[index] = plane;
            }
            uniforms.extend(cache.record(
                device,
                encoder,
                [rendered_inputs[0], rendered_inputs[1], rendered_inputs[2]],
            )?);
        }
        if let Some(weights) = &weights {
            let pipeline = ResidentUpsamplePipeline::new(device)?;
            for (&input, &output) in rendered_inputs.iter().zip(outputs) {
                uniforms.push(pipeline.encode(
                    device,
                    encoder,
                    ResidentUpsampleInputs {
                        input: input.into(),
                        output,
                        weights,
                    },
                )?);
            }
        } else {
            for (&input, &output) in inputs.iter().zip(outputs) {
                if input.width != self.output_extent.width
                    || input.height != self.output_extent.height
                {
                    return Err(Error::EngineContract("deferred feature copy geometry"));
                }
                copy_plane(encoder, input, output);
            }
            if let Some(cache) = &self.splines {
                uniforms.extend(cache.record(
                    device,
                    encoder,
                    [outputs[0], outputs[1], outputs[2]],
                )?);
            }
        }
        let noise = self.noise.as_ref().map(|plan| plan.allocate(device));
        if let Some((plan, scratch)) = self.noise.as_ref().zip(noise.as_ref()) {
            uniforms.push(jxl_wgpu::ResidentNoisePipeline::new(device)?.encode(
                device,
                encoder,
                jxl_wgpu::ResidentNoiseInputs {
                    plan,
                    scratch,
                    planes: [outputs[0], outputs[1], outputs[2]],
                },
            )?);
        }
        Ok(Resources {
            _weights: weights,
            _uniforms: uniforms,
            _noise: noise,
            _splines: self.splines.clone(),
            _coded_planes: coded_planes,
        })
    }
}

struct Resources {
    _weights: Option<jxl_wgpu::ResidentUpsampleWeights>,
    _uniforms: Vec<wgpu::Buffer>,
    _noise: Option<wgpu::Buffer>,
    _splines: Option<Arc<splines::Cache>>,
    _coded_planes: Vec<wgpu::Buffer>,
}

fn copy_plane(
    encoder: &mut wgpu::CommandEncoder,
    input: ResidentF32Plane<'_>,
    output: ResidentF32Plane<'_>,
) {
    for row in 0..input.height {
        encoder.copy_buffer_to_buffer(
            input.storage.buffer,
            input.storage.offset + u64::from(row) * u64::from(input.stride) * 4,
            output.storage.buffer,
            output.storage.offset + u64::from(row) * u64::from(output.stride) * 4,
            u64::from(input.width) * 4,
        );
    }
}

fn retained(
    device: &wgpu::Device,
    permit: &mut MemoryPermit,
    bytes: u64,
) -> Result<GpuBufferLease> {
    let output = permit
        .split_off(bytes)
        .map_err(ProgressiveDcGpuError::from)?;
    Ok(GpuBufferLease::from_tracked(
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("JPEG XL completed frame features"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        output,
    ))
}

fn surface_planes(surface: &Surface, count: usize) -> Vec<ResidentF32Plane<'_>> {
    (0..count)
        .map(|index| ResidentF32Plane {
            storage: ResidentStorageBinding {
                buffer: surface.buffer.as_wgpu_buffer(),
                offset: index as u64 * u64::from(surface.plane_words) * 4,
                size: NonZeroU64::new(u64::from(surface.plane_words) * 4)
                    .expect("nonempty surface"),
            },
            width: surface.extent.width,
            height: surface.extent.height,
            stride: surface.extent.width,
        })
        .collect()
}

fn lf_planes(source: &ProgressiveDcOutput) -> Vec<ResidentF32Plane<'_>> {
    source
        .xyb
        .planes
        .iter()
        .map(|plane| ResidentF32Plane {
            storage: ResidentStorageBinding {
                buffer: plane.buffer.as_wgpu_buffer(),
                offset: 0,
                size: NonZeroU64::new(plane.buffer.size()).expect("nonempty LF plane"),
            },
            width: plane.width,
            height: plane.height,
            stride: plane.stride,
        })
        .chain(source.extras.iter().flat_map(|extra| {
            extra.planes.iter().map(|plane| ResidentF32Plane {
                storage: ResidentStorageBinding {
                    buffer: extra.buffer.as_wgpu_buffer(),
                    offset: u64::from(plane.word_offset) * 4,
                    size: NonZeroU64::new(
                        (u64::from(plane.height - 1) * u64::from(plane.row_stride_words)
                            + u64::from(plane.width))
                            * 4,
                    )
                    .expect("nonempty LF extra"),
                },
                width: plane.width,
                height: plane.height,
                stride: plane.row_stride_words,
            })
        }))
        .collect()
}
