//! The inverse codec color transform consumes an explicitly tagged component surface.

use std::num::NonZeroU64;
use std::sync::Arc;

use jxl_gpu_bitstream::{FrameInventory, ImageHeaderInventory};
use jxl_gpu_protocol::OutputOrientation;
use jxl_wgpu::{GpuBufferLease, ResidentStorageBinding, WgpuBackend};

use super::gpu::{Compositor, Surface};
use super::icc_transform::ColorBinding;
use super::submission::{GpuWork, completion_fence_bytes, submit_recorded};
use crate::color_output::{
    ColorOutputConfig, ColorOutputEncoding, ColorOutputInputs, ColorOutputPacker, ColorOutputPlan,
    ColorOutputPlane, ColorOutputTransform, InverseOpsin,
};
use crate::frame_surface::{FrameSurfaceEncoding, FrameSurfaceLayout};
use crate::{Error, Result};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub(super) fn convert(
    backend: &WgpuBackend,
    source: &Surface,
    image: &ImageHeaderInventory,
    frame: &FrameInventory,
    compositor: &Compositor,
    encoding: FrameSurfaceEncoding,
) -> Result<GpuWork<Surface>> {
    let original = &compositor.original;
    if source.encoding != FrameSurfaceEncoding::Encoded {
        return Err(Error::EngineContract(
            "inverse color transform requires codec components",
        ));
    }
    let plane_words = source.uniform_plane_words()?;
    if source.layout.extras.len() != image.extra_channels.len() {
        return Err(Error::EngineContract(
            "inverse color transform extra channel count",
        ));
    }
    if matches!(original, FrameSurfaceEncoding::Icc(_)) && !image.xyb_encoded {
        if encoding != *original {
            return Err(Error::EngineContract(
                "ICC component reconstruction requires the original device profile",
            ));
        }
        if !frame.do_ycbcr {
            return copy_device(backend, source, encoding);
        }
    }
    let device = backend.device();
    let layout = FrameSurfaceLayout::with_encoding(
        source.extent(),
        image.extra_channels.len(),
        encoding.clone(),
        &device.limits(),
    )?;
    let connection = if image.xyb_encoded && matches!(encoding, FrameSurfaceEncoding::Icc(_)) {
        if encoding != *original {
            return Err(Error::EngineContract(
                "XYB reconstruction target is not its original profile",
            ));
        }
        Some(
            compositor
                .reconstruction
                .as_ref()
                .ok_or(Error::EngineContract(
                    "unplanned original ICC reconstruction",
                ))?,
        )
    } else {
        None
    };
    let linear_layout = connection
        .map(|_| {
            FrameSurfaceLayout::with_encoding(
                source.extent(),
                0,
                compositor.linear_encoding(),
                &device.limits(),
            )
        })
        .transpose()?;
    let rendered = linear_layout.as_ref().unwrap_or(&layout);
    let plan = ColorOutputPlan::for_limits(&rendered.color, &device.limits())?;
    let poll = backend.submission_poller().try_reserve()?;
    let output_permit = backend
        .transient_memory_budget()
        .try_reserve(layout.storage_bytes)?;
    let permit = backend.transient_memory_budget().try_reserve(
        plan.memory.uniform_bytes
            + completion_fence_bytes()
            + linear_layout
                .as_ref()
                .map_or(0, |layout| layout.storage_bytes)
            + connection.map_or(0, |connection| connection.memory.dispatch_uniform_bytes),
    )?;
    let program = connection
        .map(|connection| connection.resident(backend))
        .transpose()?;
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
    let linear = linear_layout.as_ref().map(|layout| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("JPEG XL linear XYB reconstruction before original ICC"),
            size: layout.storage_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let planes = [0u64, 1, 2].map(|channel| ColorOutputPlane {
        storage: ResidentStorageBinding {
            buffer: source.buffer.as_wgpu_buffer(),
            offset: channel * u64::from(plane_words) * 4,
            size: NonZeroU64::new(u64::from(plane_words) * 4).expect("nonempty component"),
        },
        width: source.extent().width,
        height: source.extent().height,
        stride: source.extent().width,
    });
    let config = ColorOutputConfig {
        linear_black_threshold: if image.xyb_encoded {
            original.rgb_encoding().and_then(|original| {
                crate::image_color::reconstruction_black_threshold(
                    original,
                    &rendered.color.format.color_spec,
                )
            })
        } else {
            None
        },
        white_point_adaptation: jxl_gpu_protocol::WhitePointAdaptation::Bradford,
        extent: source.extent(),
        orientation: OutputOrientation::Identity,
        transform: if image.xyb_encoded {
            ColorOutputTransform::Xyb(
                InverseOpsin::from_image(image)
                    .ok_or(Error::EngineContract("encoded XYB lacks inverse opsin"))?,
            )
        } else {
            let encoding = match original {
                FrameSurfaceEncoding::Rgb(encoding) => ColorOutputEncoding::Rgb(*encoding),
                FrameSurfaceEncoding::Icc(profile) => ColorOutputEncoding::Icc(profile.clone()),
                FrameSurfaceEncoding::Encoded => {
                    return Err(Error::EngineContract(
                        "original color interpretation cannot be codec components",
                    ));
                }
            };
            if frame.do_ycbcr {
                ColorOutputTransform::Ycbcr {
                    channel_shifts: Default::default(),
                    encoding,
                }
            } else if let ColorOutputEncoding::Rgb(encoding) = encoding {
                ColorOutputTransform::Rgb(encoding)
            } else {
                return Err(Error::EngineContract(
                    "original ICC components must use the device copy path",
                ));
            }
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
                buffer: linear.as_ref().unwrap_or(output.as_wgpu_buffer()),
                offset: 0,
                size: NonZeroU64::new(rendered.storage_bytes).expect("nonempty surface"),
            },
            layout: &rendered.color,
            config: &config,
        },
    )?;
    let icc_uniform = if let (Some(connection), Some(program), Some(linear)) =
        (connection, &program, &linear)
    {
        Some(connection.encode(
            backend,
            &mut encoder,
            program,
            ColorBinding {
                storage: ResidentStorageBinding {
                    buffer: linear,
                    offset: 0,
                    size: NonZeroU64::new(rendered.storage_bytes).expect("nonempty linear surface"),
                },
                layout: &rendered.color,
            },
            ColorBinding {
                storage: ResidentStorageBinding {
                    buffer: output.as_wgpu_buffer(),
                    offset: 0,
                    size: NonZeroU64::new(layout.storage_bytes).expect("nonempty original surface"),
                },
                layout: &layout.color,
            },
        )?)
    } else {
        None
    };
    source.copy_extras(
        &mut encoder,
        ResidentStorageBinding {
            buffer: output.as_wgpu_buffer(),
            offset: 0,
            size: NonZeroU64::new(layout.storage_bytes).expect("nonempty surface"),
        },
        &layout,
    )?;
    submit_recorded(
        backend,
        encoder,
        Surface {
            buffer: output,
            layout: Arc::new(layout),
            encoding,
        },
        vec![source.buffer.clone()],
        (scratch, linear, icc_uniform, program),
        permit,
        poll,
    )
}

fn copy_device(
    backend: &WgpuBackend,
    source: &Surface,
    encoding: FrameSurfaceEncoding,
) -> Result<GpuWork<Surface>> {
    let device = backend.device();
    let layout = FrameSurfaceLayout::with_encoding(
        source.extent(),
        source.layout.extras.len(),
        encoding.clone(),
        &device.limits(),
    )?;
    let poll = backend.submission_poller().try_reserve()?;
    let output_permit = backend
        .transient_memory_budget()
        .try_reserve(layout.storage_bytes)?;
    let permit = backend
        .transient_memory_budget()
        .try_reserve(completion_fence_bytes())?;
    let buffer = GpuBufferLease::from_tracked(
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("JPEG XL original ICC device surface"),
            size: layout.storage_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        output_permit,
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let output = ResidentStorageBinding {
        buffer: buffer.as_wgpu_buffer(),
        offset: 0,
        size: NonZeroU64::new(layout.storage_bytes).expect("nonempty device surface"),
    };
    let planes = source
        .layout
        .color
        .planes
        .iter()
        .take(layout.color.planes.len())
        .map(|plane| {
            Ok(jxl_wgpu::ResidentF32Plane {
                storage: ResidentStorageBinding {
                    buffer: source.buffer.as_wgpu_buffer(),
                    offset: plane.offset,
                    size: NonZeroU64::new(plane.end_offset()? - plane.offset)
                        .expect("nonempty component"),
                },
                width: source.extent().width,
                height: source.extent().height,
                stride: (plane.row_stride / 4) as u32,
            })
        })
        .collect::<std::result::Result<Vec<_>, jxl_gpu_formats::LayoutError>>()?;
    crate::frame_surface::copy::planes(&mut encoder, &planes, output, &layout.color)?;
    source.copy_extras(&mut encoder, output, &layout)?;
    submit_recorded(
        backend,
        encoder,
        Surface {
            buffer,
            layout: Arc::new(layout),
            encoding,
        },
        vec![source.buffer.clone()],
        (),
        permit,
        poll,
    )
}
