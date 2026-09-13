use super::*;
use jxl_gpu_formats::{Channel, PixelFormat};
use jxl_gpu_protocol::{Extent2d, OutputId};
use jxl_wgpu::{GpuBufferLease, GpuImageOutput, WgpuBackend};
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

#[test]
fn component_copies_preserve_ieee_words_and_independent_plane_guards() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let device = backend.device();
    let extent = Extent2d::new(5, 3);
    let patterns = [
        0,
        0x8000_0000,
        1,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1234,
        0xffc0_4321,
    ];
    for count in [1, 3] {
        let buffers: Vec<_> = (0..count)
            .map(|channel| {
                let mut words = [0x5555_5555_u32; 32];
                for y in 0..3 {
                    for x in 0..5 {
                        words[3 + y * 7 + x] = patterns[(y * 5 + x + channel) % patterns.len()];
                    }
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("component copy test source"),
                    contents: bytemuck::cast_slice(&words),
                    usage: wgpu::BufferUsages::COPY_SRC,
                })
            })
            .collect();
        let inputs: Vec<_> = buffers
            .iter()
            .map(|buffer| ResidentF32Plane {
                storage: ResidentStorageBinding {
                    buffer,
                    offset: 12,
                    size: NonZeroU64::new(buffer.size() - 12).unwrap(),
                },
                width: 5,
                height: 3,
                stride: 7,
            })
            .collect();
        let format = if count == 1 {
            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X])
        } else {
            crate::frame_surface::FrameSurfaceEncoding::Encoded.format()
        };
        let packed = ImageLayout::packed(extent, format.clone()).unwrap();
        let layouts = packed
            .planes
            .into_iter()
            .enumerate()
            .map(|(index, mut plane)| {
                plane.offset = 16 + index as u64 * 128;
                plane.row_stride = 28;
                plane
            })
            .collect();
        let layout = ImageLayout::from_planes(extent, format, layouts).unwrap();
        let mut expected = vec![0xaaaa_aaaa_u32; 104];
        let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("component copy guarded destination"),
            contents: bytemuck::cast_slice(&expected),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let binding = ResidentStorageBinding {
            buffer: &output,
            offset: 8,
            size: NonZeroU64::new(output.size() - 8).unwrap(),
        };
        let mut commands = device.create_command_encoder(&Default::default());
        let mut invalid = inputs.clone();
        invalid[count - 1].width = 4;
        assert!(matches!(planes(&mut commands, &invalid, binding, &layout),
            Err(FrameSurfaceError::Copy { role: "input", plane, .. }) if plane == count - 1));
        backend.queue().submit([commands.finish()]);
        let permit = backend
            .transient_memory_budget()
            .try_reserve(output.size())
            .unwrap();
        let view = GpuImageOutput {
            id: OutputId(0),
            layout: ImageLayout::packed(
                Extent2d::new(expected.len() as u32, 1),
                PixelFormat::non_color(SampleKind::Unsigned, 32, &[Channel::X]),
            )
            .unwrap(),
            buffer: GpuBufferLease::from_tracked(output.clone(), permit),
        };
        assert_eq!(
            jxl_test_support::gpu::planes::read(&backend, &view),
            expected
        );
        let mut commands = device.create_command_encoder(&Default::default());
        planes(&mut commands, &inputs, binding, &layout).unwrap();
        backend.queue().submit([commands.finish()]);
        for (channel, plane) in layout.planes.iter().enumerate() {
            for y in 0..3 {
                for x in 0..5 {
                    let index = (8 + plane.offset + y as u64 * plane.row_stride) as usize / 4 + x;
                    expected[index] = patterns[(y * 5 + x + channel) % patterns.len()];
                }
            }
        }
        assert_eq!(
            jxl_test_support::gpu::planes::read(&backend, &view),
            expected
        );
        drop(view);
        assert_eq!(
            backend.transient_memory_budget().snapshot().reserved_bytes,
            0
        );
    }
}
