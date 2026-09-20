//! Exact integer quantization capture before the shared side-image arena is released.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::vardct_side_image::RawHfDequantSideImagePlan;
use crate::{Error, Result};

pub(crate) const JPEG_QUANTIZATION_WORDS: u32 = 192;
const SHADER: &str = include_str!("jpeg.wgsl");

/// Both destinations belong to the frame, not to the temporary Modular job.
#[derive(Clone, Copy)]
pub(crate) struct JpegQuantizationCapture<'a> {
    pub(crate) output: &'a wgpu::Buffer,
    pub(crate) status: &'a wgpu::Buffer,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CaptureParams {
    geometry: [u32; 4],
    offsets: [u32; 4],
    strides: [u32; 4],
}

pub(crate) const JPEG_QUANTIZATION_CAPTURE_BYTES: u64 = size_of::<CaptureParams>() as u64;
const _: () = {
    assert!(size_of::<CaptureParams>() == 48);
    assert!(align_of::<CaptureParams>() == 16);
};

pub(super) struct JpegQuantizationCapturePipeline {
    pipeline: wgpu::ComputePipeline,
}

impl JpegQuantizationCapturePipeline {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu exact JPEG quantization capture"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        Self {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu exact JPEG quantization capture"),
                layout: None,
                module: &module,
                entry_point: Some("capture"),
                compilation_options: Default::default(),
                cache: None,
            }),
        }
    }

    pub(super) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        arena: &wgpu::Buffer,
        source_status: &wgpu::Buffer,
        plan: &RawHfDequantSideImagePlan,
        destination: JpegQuantizationCapture<'_>,
    ) -> Result<wgpu::Buffer> {
        if plan.matrix_index != 0
            || (plan.denominator - 1.0 / (8.0 * 255.0)).abs() > 1e-8
            || !plan.denominator.is_finite()
            || plan.image.final_planes.len() != 3
            || plan
                .image
                .final_planes
                .iter()
                .any(|plane| plane.width != 8 || plane.height != 8)
            || destination.output.size() < u64::from(JPEG_QUANTIZATION_WORDS) * 4
            || destination.status.size() < 16
        {
            return Err(Error::EngineContract(
                "invalid JPEG raw quantization capture binding",
            ));
        }
        let params = CaptureParams {
            geometry: [8, 8, plan.image.decoded_words, JPEG_QUANTIZATION_WORDS],
            offsets: std::array::from_fn(|channel| {
                plan.image
                    .final_planes
                    .get(channel)
                    .map_or(0, |p| p.word_offset)
            }),
            strides: std::array::from_fn(|channel| {
                plan.image
                    .final_planes
                    .get(channel)
                    .map_or(0, |p| p.row_stride_words)
            }),
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu JPEG quantization capture parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu JPEG quantization capture bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                super::entry(0, arena),
                super::entry(1, source_status),
                super::entry(2, destination.output),
                super::entry(3, destination.status),
                super::entry(4, &uniform),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu JPEG quantization capture"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &binding, &[]);
        pass.dispatch_workgroups(JPEG_QUANTIZATION_WORDS.div_ceil(64), 1, 1);
        drop(pass);
        Ok(uniform)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_capture_abi_and_shader_validate() {
        assert_eq!(std::mem::offset_of!(CaptureParams, geometry), 0);
        assert_eq!(std::mem::offset_of!(CaptureParams, offsets), 16);
        assert_eq!(std::mem::offset_of!(CaptureParams, strides), 32);
        let module = naga::front::wgsl::parse_str(SHADER).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}
