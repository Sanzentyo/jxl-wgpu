// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![cfg(not(target_arch = "wasm32"))]

use std::sync::{Arc, mpsc};

use jxl_gpu_protocol::{Extent2d, OutputId, OutputLayout, SampleType};
use jxl_wgpu::{
    ChromaLocation2d, ChromaOrder, ChromaSubsampling, ColorRange, ColorSpace, ColorSpec,
    ColorSpecification, DirectReadbackPolicy, DisplayColorEncoding, DisplayPipeline,
    DisplayTexture, DisplayTextureDescriptor, GpuImageOutput, GpuOutputBuffer,
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
