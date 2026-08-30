// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![cfg(not(target_arch = "wasm32"))]

use std::sync::{Arc, mpsc};

use jxl_gpu_protocol::{Extent2d, OutputId, OutputLayout, SampleType};
use jxl_wgpu::{
    ChromaLocation2d, ChromaOrder, ChromaSubsampling, ColorRange, ColorSpace, ColorSpec,
    ColorSpecification, DirectReadbackPolicy, DisplayPipeline, DisplayTexture,
    DisplayTextureDescriptor, GpuImageOutput, GpuOutputBuffer, Packed422Order, PixelFormat,
    RgbChannelOrder, TransferFunction, WgpuBackend, WgpuBackendConfig, YcbcrEncoding,
};
use wgpu::util::DeviceExt;

fn test_backend() -> Option<WgpuBackend> {
    match pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
        enable_timestamps: false,
        direct_readback_policy: DirectReadbackPolicy::Disabled,
        ..WgpuBackendConfig::default()
    })) {
        Ok(backend) => Some(backend),
        Err(jxl_wgpu::Error::NoAdapter) => {
            eprintln!("skipping display test: no compatible adapter");
            None
        }
        Err(error) => panic!("request display test backend: {error}"),
    }
}

fn read_texture(backend: &WgpuBackend, texture: &DisplayTexture) -> Vec<u8> {
    let bytes_per_row = texture.extent.width.checked_mul(4).unwrap().div_ceil(256) * 256;
    let size = u64::from(bytes_per_row) * u64::from(texture.extent.height);
    let staging = backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu display test readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = backend
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu display test dependent copy"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: texture.texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: texture.extent.width,
            height: texture.extent.height,
            depth_or_array_layers: 1,
        },
    );
    // This is submitted after the non-blocking display conversion, without waiting for it first.
    let submission = backend.queue().submit([encoder.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("poll display readback");
    receiver
        .recv()
        .expect("display mapping callback")
        .expect("map display texture readback");
    let mapped = staging
        .slice(..)
        .get_mapped_range()
        .expect("mapped display range");
    let mut packed = Vec::with_capacity(
        usize::try_from(texture.extent.width * texture.extent.height * 4).unwrap(),
    );
    let row_bytes = usize::try_from(texture.extent.width * 4).unwrap();
    for y in 0..texture.extent.height {
        let offset = usize::try_from(y * bytes_per_row).unwrap();
        packed.extend_from_slice(&mapped[offset..offset + row_bytes]);
    }
    drop(mapped);
    staging.unmap();
    packed
}

#[test]
fn rgb_and_nv12_become_queue_ordered_display_textures() {
    let Some(backend) = test_backend() else {
        return;
    };
    let display = DisplayPipeline::new(&backend);

    let extent = Extent2d::new(2, 1);
    let rgb = [1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5];
    let rgb_buffer = Arc::new(backend.device().create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu display RGB input"),
            contents: bytemuck::cast_slice(&rgb),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        },
    ));
    let rgb_output = GpuOutputBuffer {
        id: OutputId(0),
        extent,
        sample_type: SampleType::F32,
        channels: 4,
        layout: OutputLayout::Interleaved,
        logical_size: rgb_buffer.size(),
        buffer: rgb_buffer,
    };
    let submitted = display
        .submit_rgb(&rgb_output, DisplayTextureDescriptor::default())
        .expect("submit RGB display conversion");
    drop(rgb_output);
    let rgba = read_texture(&backend, &submitted.texture);
    assert_eq!(&rgba[..4], &[255, 0, 0, 255]);
    assert_eq!(&rgba[4..], &[0, 255, 0, 128]);
    assert_eq!(display.cache_stats().pipelines, 1);

    let extent = Extent2d::new(3, 3);
    let grey = vec![0.5f32; extent.area().unwrap()];
    let color = ColorSpecification::Defined(ColorSpec {
        space: ColorSpace::Bt709,
        encoding: YcbcrEncoding::Bt709,
        transfer: TransferFunction::Bt709,
        range: ColorRange::Limited,
        chroma_location: ChromaLocation2d::CENTER,
    });
    let format =
        PixelFormat::yuv_semiplanar(ChromaSubsampling::Cs420, 8, 8, ChromaOrder::CbCr, color)
            .unwrap();
    let converted =
        jxl_gpu_formats::convert_rgb_f32([&grey, &grey, &grey], extent, &format).unwrap();
    let layout = converted.layout;
    let mut yuv = converted.bytes;
    let padded_size = layout.logical_size.div_ceil(4) * 4;
    yuv.resize(usize::try_from(padded_size).unwrap(), 0);
    let yuv_buffer = Arc::new(backend.device().create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu display NV12 input"),
            contents: &yuv,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        },
    ));
    let yuv_output = GpuImageOutput {
        id: OutputId(1),
        layout,
        buffer: yuv_buffer,
    };
    let submitted = display
        .submit_image(&yuv_output, DisplayTextureDescriptor::default())
        .expect("submit NV12 display conversion");
    drop(yuv_output);
    let rgba = read_texture(&backend, &submitted.texture);
    for pixel in rgba.chunks_exact(4) {
        assert!(pixel[0].abs_diff(128) <= 2, "red channel: {pixel:?}");
        assert!(pixel[1].abs_diff(128) <= 2, "green channel: {pixel:?}");
        assert!(pixel[2].abs_diff(128) <= 2, "blue channel: {pixel:?}");
        assert_eq!(pixel[3], 255);
    }
    assert_eq!(display.cache_stats().pipelines, 2);
}

#[test]
fn rgba8_copy_validates_row_alignment_and_copies_when_aligned() {
    let Some(backend) = test_backend() else {
        return;
    };
    let display = DisplayPipeline::new(&backend);
    let extent = Extent2d::new(64, 2);
    let mut rgba = vec![0u8; extent.area().unwrap() * 4];
    rgba[..4].copy_from_slice(&[11, 22, 33, 44]);
    rgba[256..260].copy_from_slice(&[55, 66, 77, 88]);
    let buffer = Arc::new(
        backend
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu RGBA8 copy input"),
                contents: &rgba,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }),
    );
    let output = GpuOutputBuffer {
        id: OutputId(2),
        extent,
        sample_type: SampleType::U8,
        channels: 4,
        layout: OutputLayout::Interleaved,
        logical_size: buffer.size(),
        buffer,
    };
    let submitted = display
        .submit_rgba8_copy(&output, DisplayTextureDescriptor::default())
        .expect("submit aligned RGBA8 copy");
    let copied = read_texture(&backend, &submitted.texture);
    assert_eq!(&copied[..4], &[11, 22, 33, 44]);
    assert_eq!(&copied[256..260], &[55, 66, 77, 88]);

    let unaligned_extent = Extent2d::new(3, 2);
    let unaligned = vec![0u8; unaligned_extent.area().unwrap() * 4];
    let buffer = Arc::new(
        backend
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu unaligned RGBA8 copy input"),
                contents: &unaligned,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }),
    );
    let output = GpuOutputBuffer {
        id: OutputId(3),
        extent: unaligned_extent,
        sample_type: SampleType::U8,
        channels: 4,
        layout: OutputLayout::Interleaved,
        logical_size: buffer.size(),
        buffer,
    };
    assert!(matches!(
        display.submit_rgba8_copy(&output, DisplayTextureDescriptor::default()),
        Err(jxl_wgpu::Error::TextureCopyRowAlignment {
            bytes_per_row: 12,
            required_alignment: 256
        })
    ));
}

#[test]
fn generic_image_display_supports_high_depth_packed_and_rgb_layouts() {
    let Some(backend) = test_backend() else {
        return;
    };
    let display = DisplayPipeline::new(&backend);
    let extent = Extent2d::new(3, 3);
    let grey = vec![0.5f32; extent.area().unwrap()];
    let color = ColorSpecification::Defined(ColorSpec {
        space: ColorSpace::Bt709,
        encoding: YcbcrEncoding::Bt709,
        transfer: TransferFunction::Bt709,
        range: ColorRange::Limited,
        chroma_location: ChromaLocation2d::CENTER,
    });
    let formats = [
        PixelFormat::luma(16, color),
        PixelFormat::p010(color),
        PixelFormat::i420(12, 16, color).unwrap(),
        PixelFormat::nv42(color),
        PixelFormat::packed_yuv4228(Packed422Order::Uyvy, color),
        PixelFormat::rgb8(RgbChannelOrder::Bgra, true, color),
    ];
    let format_count = formats.len();

    for (index, format) in formats.into_iter().enumerate() {
        let converted = jxl_gpu_formats::convert_rgb_f32([&grey, &grey, &grey], extent, &format)
            .expect("convert generic display input");
        let mut bytes = converted.bytes;
        bytes.resize(bytes.len().div_ceil(4) * 4, 0);
        let buffer = Arc::new(backend.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu generic display input"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        ));
        let output = GpuImageOutput {
            id: OutputId(u32::try_from(index).unwrap()),
            layout: converted.layout,
            buffer,
        };
        let submitted = display
            .submit_image(&output, DisplayTextureDescriptor::default())
            .expect("submit generic display conversion");
        let rgba = read_texture(&backend, &submitted.texture);
        for pixel in rgba.chunks_exact(4) {
            assert!(pixel[0].abs_diff(128) <= 2, "red {format:?}: {pixel:?}");
            assert!(pixel[1].abs_diff(128) <= 2, "green {format:?}: {pixel:?}");
            assert!(pixel[2].abs_diff(128) <= 2, "blue {format:?}: {pixel:?}");
            assert_eq!(pixel[3], 255);
        }
    }
    assert_eq!(display.cache_stats().pipelines, format_count);
}
