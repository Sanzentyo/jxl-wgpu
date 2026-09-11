//! The inverse codec color transform consumes an explicitly tagged component surface.

use std::num::NonZeroU64;

use jxl_gpu_bitstream::{FrameInventory, ImageHeaderInventory};
use jxl_gpu_protocol::{OutputOrientation, RgbColorEncoding};
use jxl_wgpu::{GpuBufferLease, ResidentStorageBinding, WgpuBackend};

use super::gpu::Surface;
use super::submission::{GpuWork, completion_fence_bytes, submit_recorded};
use crate::color_output::{
    ColorOutputConfig, ColorOutputInputs, ColorOutputPacker, ColorOutputPlan, ColorOutputPlane,
    ColorOutputTransform, InverseOpsin,
};
use crate::frame_surface::{FrameSurfaceEncoding, FrameSurfaceLayout};
use crate::{Error, Result};

pub(super) fn convert(
    backend: &WgpuBackend,
    source: &Surface,
    image: &ImageHeaderInventory,
    frame: &FrameInventory,
    encoding: FrameSurfaceEncoding,
) -> Result<GpuWork> {
    if source.encoding != FrameSurfaceEncoding::Encoded {
        return Err(Error::EngineContract(
            "inverse color transform requires codec components",
        ));
    }
    let device = backend.device();
    let layout = FrameSurfaceLayout::with_encoding(
        source.extent,
        image.extra_channels.len(),
        encoding,
        &device.limits(),
    )?;
    let plan = ColorOutputPlan::for_limits(&layout.color, &device.limits())?;
    let poll = backend.submission_poller().try_reserve()?;
    let output_permit = backend
        .transient_memory_budget()
        .try_reserve(layout.storage_bytes)?;
    let permit = backend
        .transient_memory_budget()
        .try_reserve(plan.memory.uniform_bytes + completion_fence_bytes())?;
    let output = GpuBufferLease::from_tracked(
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("JPEG XL color-transformed frame"),
            size: layout.storage_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        output_permit,
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let planes = [0u64, 1, 2].map(|channel| ColorOutputPlane {
        storage: ResidentStorageBinding {
            buffer: source.buffer.as_wgpu_buffer(),
            offset: channel * u64::from(source.plane_words) * 4,
            size: NonZeroU64::new(u64::from(source.plane_words) * 4).expect("nonempty component"),
        },
        width: source.extent.width,
        height: source.extent.height,
        stride: source.extent.width,
    });
    let config = ColorOutputConfig {
        extent: source.extent,
        orientation: OutputOrientation::Identity,
        transform: if image.xyb_encoded {
            ColorOutputTransform::Xyb(
                InverseOpsin::from_image(image)
                    .ok_or(Error::EngineContract("encoded XYB lacks inverse opsin"))?,
            )
        } else if frame.do_ycbcr {
            ColorOutputTransform::Ycbcr {
                channel_shifts: Default::default(),
            }
        } else {
            ColorOutputTransform::Rgb(RgbColorEncoding::SRGB_BT709)
        },
        alpha_conversion: jxl_wgpu::AlphaConversion::Preserve,
    };
    let scratch = ColorOutputPacker::new(device)?.encode(
        device,
        &mut encoder,
        ColorOutputInputs {
            planes,
            alpha: None,
            output: ResidentStorageBinding {
                buffer: output.as_wgpu_buffer(),
                offset: 0,
                size: NonZeroU64::new(layout.storage_bytes).expect("nonempty surface"),
            },
            layout: &layout.color,
            config,
        },
    )?;
    for extra in &layout.extras {
        let plane = &extra.planes[0];
        encoder.copy_buffer_to_buffer(
            source.buffer.as_wgpu_buffer(),
            plane.offset,
            output.as_wgpu_buffer(),
            plane.offset,
            u64::from(source.extent.width) * u64::from(source.extent.height) * 4,
        );
    }
    submit_recorded(
        backend,
        encoder,
        output,
        vec![source.buffer.clone()],
        scratch,
        permit,
        poll,
    )
}
