//! Gain application fused with the existing word-owned output kernel.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{ImageHeaderInventory, gain_map::GainMapMetadata};
use jxl_gpu_formats::ImageLayout;
use jxl_gpu_protocol::{
    ChangedRegions, DisplayIntensity, Extent2d, OutputId, Region, RgbColorEncoding, SubmissionToken,
};
use jxl_wgpu::{
    AlphaConversion, GpuBufferLease, GpuImageFrame, GpuImageOutput, ImageOutputParams,
    ImageOutputSource, WgpuBackend,
};
use wgpu::util::DeviceExt;

use crate::color_output::ColorOutputPlan;
use crate::gpu_submission::{GpuWork, completion_fence_bytes, submit_recorded, validate_size};
use crate::{AlphaOutputPolicy, GpuOutputRequest, Result};

use super::GainMapDecodeError;

/// 176 bytes: plane geometry, five channel vectors and explicit gain/luminance application.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    base_offsets: [u32; 4],
    base_strides: [u32; 4],
    map_planes: [[u32; 4]; 3],
    minimum: [f32; 4],
    maximum: [f32; 4],
    inverse_gamma: [f32; 4],
    base_offset: [f32; 4],
    alternate_offset: [f32; 4],
    application: Application,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Application {
    weight: f32,
    base_to_reference: f32,
    reference_to_base: f32,
    // Uniform structures require a 16-byte stride.
    padding: u32,
}

pub(super) struct Plan {
    output: ImageLayout,
    output_plan: ColorOutputPlan,
    output_params: ImageOutputParams,
    params: Params,
}

impl Plan {
    pub(super) fn layout(&self) -> &ImageLayout {
        &self.output
    }

    pub(super) fn new(
        backend: &WgpuBackend,
        request: &GpuOutputRequest,
        image: &ImageHeaderInventory,
        working: RgbColorEncoding,
        metadata: &GainMapMetadata,
        weight: f32,
        reference_white: DisplayIntensity,
    ) -> Result<Self> {
        let extent = Extent2d::new(image.width, image.height);
        let orientation = request.orientation_policy().resolve(
            jxl_gpu_protocol::OutputOrientation::from_exif_value(image.orientation)
                .ok_or(GainMapDecodeError::Contract("invalid primary orientation"))?,
        );
        let output = ImageLayout::packed(orientation.map_extent(extent), request.format().clone())?;
        let output_plan = ColorOutputPlan::for_limits(&output, &backend.device().limits())?;
        let output_params = ImageOutputParams::new_with_intensity_target(
            &output,
            ImageOutputSource {
                extent,
                orientation,
                strides: [image.width; 3],
                encoding: working,
            },
            output_plan.dispatch_width,
            request.white_point_adaptation(),
            image.tone_mapping.intensity_target.to_f32(),
        )?
        .with_gamut_mapping(request.gamut_mapping())?
        .with_alpha_conversion(
            if request.alpha_output_policy() == AlphaOutputPolicy::Associated
                || (request.alpha_output_policy() == AlphaOutputPolicy::Preserve
                    && super::associated_alpha(image))
            {
                AlphaConversion::Premultiply
            } else {
                AlphaConversion::Preserve
            },
        );
        let mut params = Params::zeroed();
        let ratio = f64::from(image.tone_mapping.intensity_target.to_f32())
            / f64::from(reference_white.nits());
        let base_to_reference = ratio as f32;
        let reference_to_base = ratio.recip() as f32;
        if !base_to_reference.is_normal() || !reference_to_base.is_normal() {
            return Err(GainMapDecodeError::Unsupported(
                "reference-white scaling exceeds portable F32 range",
            )
            .into());
        }
        params.application = Application {
            weight,
            base_to_reference,
            reference_to_base,
            padding: 0,
        };
        for (i, c) in metadata.channels.iter().enumerate() {
            if [c.min.value(), c.max.value()]
                .into_iter()
                .any(|v| (v * f64::from(weight)).abs() > 120.0)
            {
                return Err(GainMapDecodeError::Unsupported(
                    "gain exponent exceeds portable F32 range",
                )
                .into());
            }
            params.minimum[i] = c.min.value() as f32;
            params.maximum[i] = c.max.value() as f32;
            params.inverse_gamma[i] = (1.0 / c.gamma.value()) as f32;
            params.base_offset[i] = c.base_offset.value() as f32;
            params.alternate_offset[i] = c.alternate_offset.value() as f32;
        }
        Ok(Self {
            output,
            output_plan,
            output_params,
            params,
        })
    }

    pub(super) fn submit(
        &self,
        backend: &WgpuBackend,
        base: &GpuImageFrame,
        map: &GpuImageFrame,
    ) -> Result<GpuWork> {
        let base = only_output(base)?;
        let map = only_output(map)?;
        if base.layout.planes.len() != 4 || !matches!(map.layout.planes.len(), 1 | 3) {
            return Err(GainMapDecodeError::Contract(
                "expected planar RGBA baseline and Gray/RGB map",
            )
            .into());
        }
        let device = backend.device();
        validate_size(device, base.buffer.size())?;
        validate_size(device, map.buffer.size())?;
        let mut params = self.params;
        for i in 0..4 {
            let [offset, stride, _, _] = plane(base, i)?;
            params.base_offsets[i] = offset;
            params.base_strides[i] = stride;
        }
        for i in 0..3 {
            params.map_planes[i] = plane(map, if map.layout.planes.len() == 1 { 0 } else { i })?;
        }
        let poll = backend.submission_poller().try_reserve()?;
        let memory = backend.transient_memory_budget();
        let output_permit = memory.try_reserve(self.output_plan.memory.output_storage_bytes)?;
        let transient = memory.try_reserve(
            (size_of::<Params>() + size_of::<ImageOutputParams>()) as u64
                + completion_fence_bytes(),
        )?;
        let output = GpuBufferLease::from_tracked(
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("JPEG XL alternate image"),
                size: self.output_plan.memory.output_storage_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            output_permit,
        );
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JPEG XL gain-map parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let output_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JPEG XL alternate output parameters"),
            contents: bytemuck::bytes_of(&self.output_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("JPEG XL gain-map application"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!(
                "{}\n{}",
                jxl_wgpu::IMAGE_OUTPUT_SHADER,
                include_str!("render.wgsl")
            ))),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("JPEG XL gain-map application"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let entries = [
            base.buffer.as_wgpu_buffer(),
            base.buffer.as_wgpu_buffer(),
            base.buffer.as_wgpu_buffer(),
            output.as_wgpu_buffer(),
            &output_params,
            &params,
            map.buffer.as_wgpu_buffer(),
        ]
        .into_iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect::<Vec<_>>();
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JPEG XL gain-map bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(
                self.output_plan.workgroups_x,
                self.output_plan.workgroups_y,
                1,
            );
        }
        submit_recorded(
            backend,
            encoder,
            output,
            vec![base.buffer.clone(), map.buffer.clone()],
            (params, output_params),
            transient,
            poll,
        )
    }

    pub(super) fn frame(&self, token: SubmissionToken, buffer: GpuBufferLease) -> GpuImageFrame {
        let id = OutputId(0);
        let mut changed = ChangedRegions::default();
        changed.outputs.insert(
            id,
            vec![Region::new(
                0,
                0,
                self.output.extent.width,
                self.output.extent.height,
            )],
        );
        GpuImageFrame {
            token,
            outputs: vec![GpuImageOutput {
                id,
                layout: self.output.clone(),
                buffer,
            }],
            changed,
        }
    }
}

fn only_output(frame: &GpuImageFrame) -> Result<&GpuImageOutput> {
    let [output] = frame.outputs.as_slice() else {
        return Err(GainMapDecodeError::Contract("expected one image output").into());
    };
    Ok(output)
}

fn plane(output: &GpuImageOutput, i: usize) -> Result<[u32; 4]> {
    let p = &output.layout.planes[i];
    if !p.offset.is_multiple_of(4)
        || !p.row_stride.is_multiple_of(4)
        || p.end_offset()? > output.buffer.size()
    {
        return Err(GainMapDecodeError::Contract("invalid resident plane bounds").into());
    }
    Ok([
        u32::try_from(p.offset / 4).map_err(|_| GainMapDecodeError::Contract("plane offset"))?,
        u32::try_from(p.row_stride / 4)
            .map_err(|_| GainMapDecodeError::Contract("plane stride"))?,
        output.layout.extent.width,
        output.layout.extent.height,
    ])
}

#[cfg(test)]
mod tests;
