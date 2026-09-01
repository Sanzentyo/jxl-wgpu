// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![cfg(not(target_arch = "wasm32"))]

use std::sync::{Arc, mpsc};

use jxl_gpu_protocol::{Extent2d, OutputId, OutputLayout, SampleType};
use jxl_wgpu::{
    ChromaLocation2d, ChromaOrder, ChromaSubsampling, ColorRange, ColorSpace, ColorSpec,
    ColorSpecification, DirectReadbackPolicy, DisplayColorEncoding, DisplayLuminanceEncoding,
    DisplayPipeline, DisplayTexture, DisplayTextureDescriptor, GpuImageOutput, GpuOutputBuffer,
    NumericDisplayChannels, NumericDisplayClamp, NumericDisplayContract, NumericDisplayPrecision,
    NumericDisplaySource, NumericDisplayTransfer, NumericNonFinitePolicy, Packed422Order,
    PixelFormat, RgbChannelOrder, SampleKind, TransferFunction, WgpuBackend, WgpuBackendConfig,
    YcbcrEncoding,
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
    let bytes_per_pixel = match texture.format {
        wgpu::TextureFormat::Rgba8Unorm => 4,
        wgpu::TextureFormat::Rgba16Float => 8,
        unsupported => panic!("unsupported display test texture {unsupported:?}"),
    };
    let bytes_per_row = texture
        .extent
        .width
        .checked_mul(bytes_per_pixel)
        .unwrap()
        .div_ceil(256)
        * 256;
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
        usize::try_from(texture.extent.width * texture.extent.height * bytes_per_pixel).unwrap(),
    );
    let row_bytes = usize::try_from(texture.extent.width * bytes_per_pixel).unwrap();
    for y in 0..texture.extent.height {
        let offset = usize::try_from(y * bytes_per_row).unwrap();
        packed.extend_from_slice(&mapped[offset..offset + row_bytes]);
    }
    drop(mapped);
    staging.unmap();
    packed
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    match exponent {
        0 => sign * f32::from(fraction) * 2.0f32.powi(-24),
        0x1f if fraction == 0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => sign * (1.0 + f32::from(fraction) / 1024.0) * 2.0f32.powi(i32::from(exponent) - 15),
    }
}

fn rgba16f(bytes: &[u8]) -> [f32; 4] {
    std::array::from_fn(|channel| {
        let offset = channel * 2;
        f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
    })
}

fn gpu_image(
    backend: &WgpuBackend,
    layout: jxl_wgpu::ImageLayout,
    mut bytes: Vec<u8>,
) -> GpuImageOutput {
    bytes.resize(bytes.len().div_ceil(4) * 4, 0);
    let buffer = backend
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu color display source"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    GpuImageOutput {
        id: OutputId(99),
        layout,
        buffer: jxl_wgpu::GpuBufferLease::from_external(buffer),
    }
}

fn signed_map(value: f32, map: impl FnOnce(f32) -> f32) -> f32 {
    map(value.abs()).copysign(value)
}

fn display_to_linear(value: f32, transfer: TransferFunction) -> f32 {
    signed_map(value, |value| match transfer {
        TransferFunction::Linear => value,
        TransferFunction::Srgb | TransferFunction::Sycc => {
            if value <= 0.040_45 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        }
        TransferFunction::Bt709 => {
            if value < 0.081 {
                value / 4.5
            } else {
                ((value + 0.099) / 1.099).powf(1.0 / 0.45)
            }
        }
        TransferFunction::Pq => {
            let m1 = 2610.0 / 16384.0;
            let m2 = (2523.0 / 4096.0) * 128.0;
            let c1 = 3424.0 / 4096.0;
            let c2 = (2413.0 / 4096.0) * 32.0;
            let c3 = (2392.0 / 4096.0) * 32.0;
            let powered = value.powf(1.0 / m2);
            ((powered - c1).max(0.0) / (c2 - c3 * powered).max(1e-10)).powf(1.0 / m1)
        }
        TransferFunction::Hlg => {
            let a = 0.178_832_77;
            let b = 1.0 - 4.0 * a;
            let c = 0.559_910_7;
            if value <= 0.5 {
                value * value / 3.0
            } else {
                (((value - c) / a).exp() + b) / 12.0
            }
        }
        TransferFunction::Bt2020 => {
            let alpha = 1.099_296_8;
            let beta = 0.018_053_97;
            if value < 4.5 * beta {
                value / 4.5
            } else {
                ((value + alpha - 1.0) / alpha).powf(1.0 / 0.45)
            }
        }
        TransferFunction::Undefined | TransferFunction::Smpte240M => {
            panic!("unsupported display oracle transfer")
        }
    })
}

fn bt2020_from_linear(value: f32) -> f32 {
    signed_map(value, |value| {
        let alpha = 1.099_296_8;
        let beta = 0.018_053_97;
        if value < beta {
            4.5 * value
        } else {
            alpha * value.powf(0.45) - (alpha - 1.0)
        }
    })
}

fn to_linear_bt709(space: ColorSpace, linear: [f32; 3]) -> [f32; 3] {
    let source_to_xyz = match space {
        ColorSpace::Bt709 => [
            [0.412_456_4, 0.357_576_1, 0.180_437_5],
            [0.212_672_9, 0.715_152_2, 0.072_175],
            [0.019_333_9, 0.119_192, 0.950_304_1],
        ],
        ColorSpace::Bt2020 => [
            [0.636_958, 0.144_616_9, 0.168_881],
            [0.262_700_2, 0.677_998_1, 0.059_301_7],
            [0.0, 0.028_072_7, 1.060_985_1],
        ],
        ColorSpace::DisplayP3 => [
            [0.486_570_95, 0.265_667_7, 0.198_217_29],
            [0.228_974_57, 0.691_738_55, 0.079_286_91],
            [0.0, 0.045_113_38, 1.043_944_4],
        ],
        unsupported => panic!("unsupported display oracle primaries {unsupported:?}"),
    };
    let xyz = source_to_xyz.map(|row| row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2]);
    [
        [3.240_454_2, -1.537_138_5, -0.498_531_4],
        [-0.969_266, 1.876_010_8, 0.041_556],
        [0.055_643_4, -0.204_025_9, 1.057_225_2],
    ]
    .map(|row| row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2])
}

fn bt709_linear_code(encoded: f32) -> u8 {
    let linear = if encoded < 0.081 {
        encoded / 4.5
    } else {
        ((encoded + 0.099) / 1.099).powf(1.0 / 0.45)
    };
    (linear * 255.0).round() as u8
}

fn unorm8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
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
        buffer: jxl_wgpu::GpuBufferLease::from_external(rgb_buffer.as_ref().clone()),
    };
    let submitted = display
        .submit_rgb(&rgb_output, DisplayTextureDescriptor::default())
        .expect("submit RGB display conversion");
    assert_eq!(
        submitted.texture.color_encoding,
        DisplayColorEncoding::LinearBt709
    );
    assert_eq!(
        submitted.texture.luminance_encoding,
        DisplayLuminanceEncoding::Relative
    );
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
        buffer: jxl_wgpu::GpuBufferLease::from_external(yuv_buffer.as_ref().clone()),
    };
    let submitted = display
        .submit_image(&yuv_output, DisplayTextureDescriptor::default())
        .expect("submit NV12 display conversion");
    drop(yuv_output);
    let rgba = read_texture(&backend, &submitted.texture);
    let expected_gray = bt709_linear_code(0.5);
    for pixel in rgba.chunks_exact(4) {
        assert!(
            pixel[0].abs_diff(expected_gray) <= 2,
            "red channel: {pixel:?}"
        );
        assert!(
            pixel[1].abs_diff(expected_gray) <= 2,
            "green channel: {pixel:?}"
        );
        assert!(
            pixel[2].abs_diff(expected_gray) <= 2,
            "blue channel: {pixel:?}"
        );
        assert_eq!(pixel[3], 255);
    }
    assert_eq!(display.cache_stats().pipelines, 2);
}

#[test]
fn wide_gamut_and_hdr_images_become_linear_float_textures() {
    let Some(backend) = test_backend() else {
        return;
    };
    let display = DisplayPipeline::new(&backend);
    let extent = Extent2d::new(1, 1);
    let cases = [
        (
            "BT.2020 PQ",
            ColorSpace::Bt2020,
            TransferFunction::Pq,
            DisplayLuminanceEncoding::PqNormalized10000Nits,
            [128, 164, 201, 127],
        ),
        (
            "Display-P3 HLG",
            ColorSpace::DisplayP3,
            TransferFunction::Hlg,
            DisplayLuminanceEncoding::HlgScene,
            [100, 151, 220, 255],
        ),
        (
            "BT.2020 OETF",
            ColorSpace::Bt2020,
            TransferFunction::Bt2020,
            DisplayLuminanceEncoding::Relative,
            [51, 127, 230, 204],
        ),
    ];

    for (name, space, transfer, luminance_encoding, stored) in cases {
        let color = ColorSpecification::Defined(ColorSpec {
            space,
            encoding: YcbcrEncoding::Undefined,
            transfer,
            range: ColorRange::Full,
            chroma_location: ChromaLocation2d::BOTH,
        });
        let format = PixelFormat::rgb8(RgbChannelOrder::Rgba, false, color);
        let layout = jxl_wgpu::ImageLayout::packed(extent, format).expect("valid RGB8 layout");
        let source = gpu_image(&backend, layout, stored.to_vec());
        assert!(matches!(
            display.submit_image(&source, DisplayTextureDescriptor::default()),
            Err(jxl_wgpu::Error::Unsupported(_))
        ));
        let submission = display
            .submit_image(&source, DisplayTextureDescriptor::linear_bt709_hdr())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(submission.texture.format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(
            submission.texture.color_encoding,
            DisplayColorEncoding::LinearBt709
        );
        assert_eq!(submission.texture.luminance_encoding, luminance_encoding);
        let actual = rgba16f(&read_texture(&backend, &submission.texture));
        let source_linear = std::array::from_fn(|channel| {
            display_to_linear(f32::from(stored[channel]) / 255.0, transfer)
        });
        let expected_rgb = to_linear_bt709(space, source_linear);
        for channel in 0..3 {
            assert!(
                (actual[channel] - expected_rgb[channel]).abs() <= 0.002,
                "{name} channel {channel}: GPU {}, scalar {}",
                actual[channel],
                expected_rgb[channel]
            );
        }
        assert!((actual[3] - f32::from(stored[3]) / 255.0).abs() <= 0.001);
    }
}

#[test]
fn bt2020_constant_luminance_displays_through_its_normative_inverse() {
    let Some(backend) = test_backend() else {
        return;
    };
    let display = DisplayPipeline::new(&backend);
    let extent = Extent2d::new(1, 1);
    let source_linear = [0.18, 0.43, 0.72];
    let encoded = source_linear.map(bt2020_from_linear);
    let kr = 0.2627;
    let kb = 0.0593;
    let kg = 1.0 - kr - kb;
    let y =
        bt2020_from_linear(kr * source_linear[0] + kg * source_linear[1] + kb * source_linear[2]);
    let cb_divisor = if encoded[2] > y { 1.5816 } else { 1.9404 };
    let cr_divisor = if encoded[0] > y { 0.9936 } else { 1.7184 };
    let codes = [
        unorm8(y),
        unorm8((encoded[2] - y) / cb_divisor + 0.5),
        unorm8((encoded[0] - y) / cr_divisor + 0.5),
    ];
    let color = ColorSpecification::Defined(ColorSpec {
        space: ColorSpace::Bt2020,
        encoding: YcbcrEncoding::Bt2020ConstantLuminance,
        transfer: TransferFunction::Bt2020,
        range: ColorRange::Full,
        chroma_location: ChromaLocation2d::BOTH,
    });
    let format = PixelFormat::i444(8, 8, color).expect("valid BT.2020 CL I444");
    let layout = jxl_wgpu::ImageLayout::packed(extent, format).expect("valid I444 layout");
    let mut bytes = vec![0; usize::try_from(layout.logical_size).unwrap()];
    for (plane, code) in codes.into_iter().enumerate() {
        bytes[usize::try_from(layout.planes[plane].offset).unwrap()] = code;
    }
    let source = gpu_image(&backend, layout, bytes);
    let submission = display
        .submit_image(&source, DisplayTextureDescriptor::linear_bt709_hdr())
        .expect("submit BT.2020 constant-luminance display");
    let actual = rgba16f(&read_texture(&backend, &submission.texture));

    let y_encoded = f32::from(codes[0]) / 255.0;
    let cb = (f32::from(codes[1]) - 128.0) / 255.0;
    let cr = (f32::from(codes[2]) - 128.0) / 255.0;
    let b_encoded = y_encoded + cb * if cb > 0.0 { 1.5816 } else { 1.9404 };
    let r_encoded = y_encoded + cr * if cr > 0.0 { 0.9936 } else { 1.7184 };
    let y_linear = display_to_linear(y_encoded, TransferFunction::Bt2020);
    let r = display_to_linear(r_encoded, TransferFunction::Bt2020);
    let b = display_to_linear(b_encoded, TransferFunction::Bt2020);
    let g = (y_linear - kr * r - kb * b) / kg;
    let expected = to_linear_bt709(ColorSpace::Bt2020, [r, g, b]);
    for channel in 0..3 {
        assert!(
            (actual[channel] - expected[channel]).abs() <= 0.003,
            "channel {channel}: GPU {}, scalar {}",
            actual[channel],
            expected[channel]
        );
    }
    assert_eq!(actual[3], 1.0);

    // A luma-only layout has no chroma planes. Its matrix tag must not make the shader enter the
    // constant-luminance chroma reconstruction path.
    let luma_layout = jxl_wgpu::ImageLayout::packed(extent, PixelFormat::luma(8, color))
        .expect("valid BT.2020 luma layout");
    let luma_source = gpu_image(&backend, luma_layout, vec![codes[0]]);
    let luma_submission = display
        .submit_image(&luma_source, DisplayTextureDescriptor::linear_bt709_hdr())
        .expect("submit BT.2020 luma display");
    let luma_actual = rgba16f(&read_texture(&backend, &luma_submission.texture));
    let luma_linear = display_to_linear(y_encoded, TransferFunction::Bt2020);
    let luma_expected = to_linear_bt709(ColorSpace::Bt2020, [luma_linear; 3]);
    for channel in 0..3 {
        assert!(
            (luma_actual[channel] - luma_expected[channel]).abs() <= 0.002,
            "luma channel {channel}: GPU {}, scalar {}",
            luma_actual[channel],
            luma_expected[channel]
        );
    }
    assert_eq!(luma_actual[3], 1.0);
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
        buffer: jxl_wgpu::GpuBufferLease::from_external(buffer.as_ref().clone()),
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
        buffer: jxl_wgpu::GpuBufferLease::from_external(buffer.as_ref().clone()),
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
            buffer: jxl_wgpu::GpuBufferLease::from_external(buffer.as_ref().clone()),
        };
        let submitted = display
            .submit_image(&output, DisplayTextureDescriptor::default())
            .expect("submit generic display conversion");
        let rgba = read_texture(&backend, &submitted.texture);
        let expected_gray = bt709_linear_code(0.5);
        for pixel in rgba.chunks_exact(4) {
            assert!(
                pixel[0].abs_diff(expected_gray) <= 4,
                "red {format:?}: {pixel:?}"
            );
            assert!(
                pixel[1].abs_diff(expected_gray) <= 4,
                "green {format:?}: {pixel:?}"
            );
            assert!(
                pixel[2].abs_diff(expected_gray) <= 4,
                "blue {format:?}: {pixel:?}"
            );
            assert_eq!(pixel[3], 255);
        }
    }
    assert_eq!(display.cache_stats().pipelines, format_count);
}

#[test]
fn odd_width_packed_422_preserves_color_and_tail_luma_order() {
    let Some(backend) = test_backend() else {
        return;
    };
    let display = DisplayPipeline::new(&backend);
    let extent = Extent2d::new(3, 1);
    let encoded_rgb = [0.72f32, 0.31, 0.12];
    let red = vec![encoded_rgb[0]; extent.area().unwrap()];
    let green = vec![encoded_rgb[1]; extent.area().unwrap()];
    let blue = vec![encoded_rgb[2]; extent.area().unwrap()];
    let color = ColorSpecification::Defined(ColorSpec {
        space: ColorSpace::Bt709,
        encoding: YcbcrEncoding::Bt709,
        transfer: TransferFunction::Bt709,
        range: ColorRange::Full,
        chroma_location: ChromaLocation2d::CENTER,
    });

    for (index, order) in [Packed422Order::Yuyv, Packed422Order::Uyvy]
        .into_iter()
        .enumerate()
    {
        let format = PixelFormat::packed_yuv4228(order, color);
        let converted = jxl_gpu_formats::convert_rgb_f32([&red, &green, &blue], extent, &format)
            .expect("convert packed 4:2:2 display input");
        let mut bytes = converted.bytes;
        bytes.resize(bytes.len().div_ceil(4) * 4, 0);
        let buffer = Arc::new(backend.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu packed 4:2:2 color input"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        ));
        let output = GpuImageOutput {
            id: OutputId(u32::try_from(index).unwrap()),
            layout: converted.layout,
            buffer: jxl_wgpu::GpuBufferLease::from_external(buffer.as_ref().clone()),
        };
        let submitted = display
            .submit_image(&output, DisplayTextureDescriptor::default())
            .expect("submit packed 4:2:2 display conversion");
        let rgba = read_texture(&backend, &submitted.texture);
        let expected = encoded_rgb.map(bt709_linear_code);
        for pixel in rgba.chunks_exact(4) {
            for channel in 0..3 {
                assert!(
                    pixel[channel].abs_diff(expected[channel]) <= 7,
                    "{order:?} channel {channel}: got {pixel:?}, expected {expected:?}"
                );
            }
            assert_eq!(pixel[3], 255);
        }
    }
}

#[test]
fn all_vpi_numeric_formats_use_explicit_gpu_display_contracts() {
    use jxl_gpu_formats::vpi::VpiPitchLinearFormat as Vpi;

    let Some(backend) = test_backend() else {
        return;
    };
    let display = DisplayPipeline::new(&backend);
    let extent = Extent2d::new(4, 1);
    let formats = [
        Vpi::U8,
        Vpi::S8,
        Vpi::U16,
        Vpi::U32,
        Vpi::S32,
        Vpi::S16,
        Vpi::TwoS16,
        Vpi::F32,
        Vpi::F64,
        Vpi::TwoF32,
    ];

    for (index, predefined) in formats.into_iter().enumerate() {
        let (bytes, values) = numeric_source(predefined);
        let format = predefined.pixel_format();
        let contract = numeric_contract(predefined, format.sample_kind);
        let layout = jxl_gpu_formats::ImageLayout::packed(extent, format).unwrap();
        assert_eq!(
            bytes.len() as u64,
            layout.logical_size,
            "{}",
            predefined.name()
        );
        let mut padded = bytes;
        padded.resize(padded.len().div_ceil(4) * 4, 0);
        let buffer = Arc::new(backend.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu VPI numeric display source"),
                contents: &padded,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        ));
        let source = GpuImageOutput {
            id: OutputId(u32::try_from(index).unwrap()),
            layout,
            buffer: jxl_wgpu::GpuBufferLease::from_external(buffer.as_ref().clone()),
        };
        let submission = display
            .submit_numeric_image(&source, contract, DisplayTextureDescriptor::default())
            .unwrap_or_else(|error| panic!("{} display submission: {error}", predefined.name()));
        let precision = submission
            .texture
            .numeric_precision
            .expect("numeric display reports its arithmetic path");
        let actual = read_texture(&backend, &submission.texture);
        let expected = numeric_display_oracle(predefined, &values, contract, precision);
        assert_eq!(actual, expected, "{} ({precision:?})", predefined.name());
    }
}

fn numeric_source(format: jxl_gpu_formats::vpi::VpiPitchLinearFormat) -> (Vec<u8>, Vec<[f64; 2]>) {
    use jxl_gpu_formats::vpi::VpiPitchLinearFormat as Vpi;

    let mut bytes = Vec::new();
    let values = match format {
        Vpi::U8 => vec![[0.0, 0.0], [64.0, 0.0], [128.0, 0.0], [255.0, 0.0]],
        Vpi::S8 => vec![[-128.0, 0.0], [-64.0, 0.0], [0.0, 0.0], [127.0, 0.0]],
        Vpi::U16 => vec![[0.0, 0.0], [16384.0, 0.0], [32768.0, 0.0], [65535.0, 0.0]],
        Vpi::U32 => vec![
            [0.0, 0.0],
            [f64::from(1_u32 << 30), 0.0],
            [f64::from(1_u32 << 31), 0.0],
            [f64::from(u32::MAX), 0.0],
        ],
        Vpi::S32 => vec![
            [f64::from(i32::MIN), 0.0],
            [f64::from(-(1_i32 << 30)), 0.0],
            [0.0, 0.0],
            [f64::from(i32::MAX), 0.0],
        ],
        Vpi::S16 => vec![[-32768.0, 0.0], [-16384.0, 0.0], [0.0, 0.0], [32767.0, 0.0]],
        Vpi::TwoS16 => vec![
            [-32768.0, 32767.0],
            [-16384.0, 16384.0],
            [0.0, 0.0],
            [32767.0, -32768.0],
        ],
        Vpi::F32 | Vpi::F64 => vec![
            [f64::NAN, 0.0],
            [f64::NEG_INFINITY, 0.0],
            [f64::INFINITY, 0.0],
            [0.5, 0.0],
        ],
        Vpi::TwoF32 => vec![[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [1.0, 0.0]],
        _ => unreachable!("numeric test inventory contains only numeric formats"),
    };
    for value in &values {
        match format {
            Vpi::U8 => bytes.push(value[0] as u8),
            Vpi::S8 => bytes.push((value[0] as i8) as u8),
            Vpi::U16 => bytes.extend_from_slice(&(value[0] as u16).to_le_bytes()),
            Vpi::U32 => bytes.extend_from_slice(&(value[0] as u32).to_le_bytes()),
            Vpi::S32 => bytes.extend_from_slice(&(value[0] as i32).to_le_bytes()),
            Vpi::S16 => bytes.extend_from_slice(&(value[0] as i16).to_le_bytes()),
            Vpi::TwoS16 => {
                bytes.extend_from_slice(&(value[0] as i16).to_le_bytes());
                bytes.extend_from_slice(&(value[1] as i16).to_le_bytes());
            }
            Vpi::F32 => bytes.extend_from_slice(&(value[0] as f32).to_le_bytes()),
            Vpi::F64 => bytes.extend_from_slice(&value[0].to_le_bytes()),
            Vpi::TwoF32 => {
                bytes.extend_from_slice(&(value[0] as f32).to_le_bytes());
                bytes.extend_from_slice(&(value[1] as f32).to_le_bytes());
            }
            _ => unreachable!("numeric test inventory contains only numeric formats"),
        }
    }
    (bytes, values)
}

fn numeric_contract(
    format: jxl_gpu_formats::vpi::VpiPitchLinearFormat,
    sample_kind: SampleKind,
) -> NumericDisplayContract {
    use jxl_gpu_formats::vpi::VpiPitchLinearFormat as Vpi;

    let (scale, bias) = match format {
        Vpi::U8 => (1.0 / f32::from(u8::MAX), 0.0),
        Vpi::U16 => (1.0 / f32::from(u16::MAX), 0.0),
        Vpi::U32 => (1.0 / u32::MAX as f32, 0.0),
        Vpi::S8 => {
            let scale = 1.0 / f32::from(u8::MAX);
            (scale, -f32::from(i8::MIN) * scale)
        }
        Vpi::S16 | Vpi::TwoS16 => {
            let scale = 1.0 / f32::from(u16::MAX);
            (scale, -f32::from(i16::MIN) * scale)
        }
        Vpi::S32 => {
            let scale = 1.0 / u32::MAX as f32;
            (scale, -(i32::MIN as f32) * scale)
        }
        Vpi::F32 | Vpi::F64 | Vpi::TwoF32 => (1.0, 0.0),
        _ => unreachable!("numeric test inventory contains only numeric formats"),
    };
    NumericDisplayContract {
        source: match sample_kind {
            SampleKind::Unsigned => NumericDisplaySource::Unsigned,
            SampleKind::Signed => NumericDisplaySource::Signed,
            SampleKind::Float => NumericDisplaySource::Floating {
                non_finite: NumericNonFinitePolicy::Saturate,
            },
        },
        scale,
        bias,
        channels: match format {
            Vpi::TwoS16 => NumericDisplayChannels::LumaAlpha,
            Vpi::TwoF32 => NumericDisplayChannels::RedGreen,
            _ => NumericDisplayChannels::Luma,
        },
        transfer: if format == Vpi::F32 {
            NumericDisplayTransfer::Srgb
        } else {
            NumericDisplayTransfer::Linear
        },
        clamp: NumericDisplayClamp::Unit,
    }
}

fn numeric_display_oracle(
    format: jxl_gpu_formats::vpi::VpiPitchLinearFormat,
    values: &[[f64; 2]],
    contract: NumericDisplayContract,
    precision: NumericDisplayPrecision,
) -> Vec<u8> {
    let normalize = |value: f64| {
        let value = if format == jxl_gpu_formats::vpi::VpiPitchLinearFormat::F64
            && precision == NumericDisplayPrecision::NativeF64
        {
            (value * f64::from(contract.scale) + f64::from(contract.bias)) as f32
        } else {
            (value as f32) * contract.scale + contract.bias
        };
        let finite = if value.is_nan() {
            0.0
        } else if value.is_infinite() {
            match contract.source {
                NumericDisplaySource::Floating {
                    non_finite: NumericNonFinitePolicy::Saturate,
                } if value.is_sign_positive() => 1.0,
                _ => 0.0,
            }
        } else {
            value
        };
        finite.clamp(0.0, 1.0)
    };
    values
        .iter()
        .flat_map(|value| {
            let x = normalize(value[0]);
            let y = normalize(value[1]);
            let mut rgba = match contract.channels {
                NumericDisplayChannels::Luma => [x, x, x, 1.0],
                NumericDisplayChannels::LumaAlpha => [x, x, x, y],
                NumericDisplayChannels::RedGreen => [x, y, 0.0, 1.0],
            };
            if contract.transfer == NumericDisplayTransfer::Srgb {
                for channel in &mut rgba[..3] {
                    *channel = if *channel <= 0.04045 {
                        *channel / 12.92
                    } else {
                        ((*channel + 0.055) / 1.055).powf(2.4)
                    };
                }
            }
            rgba.map(unorm8)
        })
        .collect()
}
