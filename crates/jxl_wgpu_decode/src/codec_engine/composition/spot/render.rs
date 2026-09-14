//! A presentation-only copy in the source color domain, before an ICC connection.

use std::num::NonZeroU64;

use jxl_gpu_bitstream::ExtraChannelInventory;
use jxl_wgpu::{ResidentStorageBinding, WgpuBackend};
use wgpu::util::DeviceExt;

use super::super::gpu::{Surface, dispatch, pipeline};
use super::super::submission::validate_size;
use super::{SpotColor, spot_colors};
use crate::Result;
use crate::frame_surface::FrameSurfaceLayout;

#[derive(Debug)]
pub(in crate::codec_engine::composition) struct Rendering {
    layout: FrameSurfaceLayout,
    colors: Vec<SpotColor>,
    pipeline: wgpu::ComputePipeline,
    dispatch: [u32; 2],
}

pub(in crate::codec_engine::composition) struct Rendered {
    pub buffer: wgpu::Buffer,
    _metadata: wgpu::Buffer,
}

impl Rendering {
    pub(in crate::codec_engine::composition) fn new(
        backend: &WgpuBackend,
        layout: &FrameSurfaceLayout,
        extras: &[ExtraChannelInventory],
    ) -> Result<Self> {
        let colors = spot_colors(extras, &layout.extras);
        validate_size(
            backend.device(),
            std::mem::size_of_val(colors.as_slice()) as u64,
        )?;
        let pixels = u64::from(layout.color.extent.width) * u64::from(layout.color.extent.height);
        let dispatch = dispatch(backend.device(), pixels)?;
        let shader = format!(
            "{}\n{}",
            include_str!("../spot.wgsl"),
            include_str!("render.wgsl")
        );
        let pipeline = pipeline(
            backend.device(),
            "JPEG XL spots before ICC conversion",
            &shader,
            &[
                ("surface_plane_words", (layout.color_plane_bytes / 4) as f64),
                ("surface_color_channels", layout.color.planes.len() as f64),
                ("surface_pixels", pixels as f64),
                ("dispatch_width", f64::from(dispatch[0] * 64)),
            ],
        );
        Ok(Self {
            layout: layout.clone(),
            colors,
            pipeline,
            dispatch,
        })
    }

    pub(in crate::codec_engine::composition) fn memory_bytes(&self) -> u64 {
        self.layout.storage_bytes + std::mem::size_of_val(self.colors.as_slice()) as u64
    }

    pub(in crate::codec_engine::composition) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &Surface,
    ) -> Result<Rendered> {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("JPEG XL spot-rendered presentation source"),
            size: self.layout.storage_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        source.copy_extras(
            encoder,
            ResidentStorageBinding {
                buffer: &buffer,
                offset: 0,
                size: NonZeroU64::new(self.layout.storage_bytes).expect("nonempty spot surface"),
            },
            &self.layout,
        )?;
        let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JPEG XL spot inks before ICC conversion"),
            contents: bytemuck::cast_slice(&self.colors),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JPEG XL spot presentation bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source.buffer.as_wgpu_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: metadata.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(self.dispatch[0], self.dispatch[1], 1);
        }
        Ok(Rendered {
            buffer,
            _metadata: metadata,
        })
    }
}
