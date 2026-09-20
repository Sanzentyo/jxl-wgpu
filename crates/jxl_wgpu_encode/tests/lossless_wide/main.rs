#![cfg(not(target_arch = "wasm32"))]

mod animation;
mod lifetime;

use std::num::NonZeroU64;
use std::sync::Arc;

use jxl_gpu_bitstream::SampleBitDepth;
use jxl_gpu_formats::{ImageLayout, PitchLinearPlaneLayout};
use jxl_gpu_protocol::Extent2d;
use jxl_test_support::gpu::planes::{open_fragmented, read_bytes};
use jxl_test_support::oracles::{extra_channels, modular_integer};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, NumericSampleMapping, WgpuDecodeEngine};
use jxl_wgpu_encode::{
    BufferImageSource, LosslessModularEncoder, LosslessModularFormat, LosslessModularTreeMode,
    WgpuContext,
};
use wgpu::util::DeviceExt;

fn source(
    context: &WgpuContext,
    extent: Extent2d,
    format: LosslessModularFormat,
    bits: u8,
    phase: u32,
) -> (BufferImageSource, Vec<u32>) {
    let maximum = (1u32 << bits) - 1;
    let channels = format.channel_count();
    let offset = 5;
    let row_bytes = u64::from(extent.width) * u64::from(channels) * 4;
    let row_stride = row_bytes + 5;
    let mut bytes =
        vec![0xa5; (offset + row_stride * u64::from(extent.height)).div_ceil(4) as usize * 4];
    let mut expected = Vec::new();
    for y in 0..extent.height {
        for x in 0..extent.width {
            for channel in 0..channels {
                // Deliberately include adjacent YCoCg differences spanning both signed
                // endpoints, low bits beyond F32 precision, and long exact zero runs.
                let extremes = [maximum, 0, 1, maximum - 1, maximum / 2, maximum / 2 + 1];
                let sample = if (32..64).contains(&(x % 256)) {
                    0
                } else if x % 256 < 16 {
                    extremes[((x + y + channel * 2 + phase) % 6) as usize]
                } else {
                    let hash = x.wrapping_mul(0x9e37_79b9)
                        ^ y.wrapping_mul(0x85eb_ca6b)
                        ^ channel.wrapping_mul(0xc2b2_ae35)
                        ^ phase.wrapping_mul(0x27d4_eb2d);
                    (hash ^ (hash >> 16)).wrapping_mul(0x7feb_352d) & maximum
                };
                let address =
                    offset + u64::from(y) * row_stride + u64::from(x * channels + channel) * 4;
                bytes[address as usize..address as usize + 4]
                    .copy_from_slice(&(sample | !maximum).to_le_bytes());
                expected.push(sample);
            }
        }
    }
    let layout = ImageLayout::from_planes(
        extent,
        format.pixel_format(bits).unwrap(),
        vec![PitchLinearPlaneLayout {
            plane_index: 0,
            offset,
            row_stride,
            sample_extent: extent,
            row_bytes,
        }],
    )
    .unwrap();
    let buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wide integer encoder source with poisoned padding"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    (
        BufferImageSource::new(Arc::new(buffer), layout).unwrap(),
        expected,
    )
}

fn check_oracles(encoded: &[u8], expected: &[u32], format: LosslessModularFormat, bits: u8) {
    check_frame_oracles(encoded, &[expected], format, bits);
}

fn check_frame_oracles(encoded: &[u8], frames: &[&[u32]], format: LosslessModularFormat, bits: u8) {
    let channels = format.channel_count() as usize;
    let pixels = frames[0].len() / channels;
    let frame_words = pixels * (4 + usize::from(format.has_alpha()));
    let native = extra_channels::libjxl_output(encoded, &["--original"])
        .expect("required libjxl 0.12.0 original-component oracle");
    assert_eq!(native.len(), frames.len() * frame_words);
    let maximum = f64::from((1u32 << bits) - 1);
    for (frame_index, expected) in frames.iter().enumerate() {
        assert_eq!(expected.len(), pixels * channels);
        let planes = modular_integer::original_planes(encoded, frame_index);
        assert_eq!(planes.len(), channels);
        for (channel, plane) in planes.iter().enumerate() {
            assert_eq!(plane.len(), pixels);
            for (pixel, &actual) in plane.iter().enumerate() {
                assert_eq!(
                    actual,
                    expected[pixel * channels + channel] as i32,
                    "jxl-oxide exact {format:?}/{bits}: frame {frame_index}, pixel {pixel}, channel {channel}"
                );
            }
        }
        let native = &native[frame_index * frame_words..(frame_index + 1) * frame_words];
        for (pixel, sample) in expected.chunks_exact(channels).enumerate() {
            for channel in 0..4 {
                let expected = if channel == 3 && !format.has_alpha() {
                    1.0
                } else {
                    f64::from(
                        sample[if format == LosslessModularFormat::Gray {
                            0
                        } else {
                            channel
                        }],
                    ) / maximum
                };
                let actual = f64::from(native[pixel * 4 + channel]);
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= 2e-7,
                    "libjxl normalized {format:?}/{bits}: frame {frame_index}, pixel {pixel}, channel {channel}: {actual} != {expected}"
                );
            }
            if format.has_alpha() {
                let actual = f64::from(native[pixels * 4 + pixel]);
                assert!((actual - f64::from(sample[3]) / maximum).abs() <= 2e-7);
            }
        }
    }
}

#[test]
fn every_wide_integer_depth_roundtrips_exactly_with_independent_oracles() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let whole = GpuDecoder::wgpu(backend.clone()).unwrap();
    let bounded = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    for tree_mode in [
        LosslessModularTreeMode::SharedGlobal,
        LosslessModularTreeMode::LocalPerGroup,
    ] {
        let encoder = LosslessModularEncoder::with_tree_mode(context.clone(), tree_mode);
        for (format_index, format) in [
            LosslessModularFormat::Gray,
            LosslessModularFormat::Rgb,
            LosslessModularFormat::Rgba,
        ]
        .into_iter()
        .enumerate()
        {
            for bits in 17..=31 {
                let (width, height) = [(1, 257), (255, 3), (256, 2), (257, 9)]
                    [(format_index + usize::from(bits - 17)) % 4];
                let (source, expected) =
                    source(&context, Extent2d::new(width, height), format, bits, 0);
                let memory = encoder.memory_plan(&source).unwrap();
                assert_eq!(memory.bits_per_sample, bits);
                assert_eq!(memory.bytes_per_sample, 4);
                let encoded =
                    pollster::block_on(encoder.submit_container(source).unwrap()).unwrap();
                assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
                let inventory = jxl_gpu_bitstream::parse(&encoded, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap();
                assert_eq!(
                    inventory.image_header.bit_depth,
                    SampleBitDepth::Integer {
                        bits_per_sample: u32::from(bits)
                    }
                );
                assert_eq!(
                    inventory.image_header.extra_channels.len(),
                    usize::from(format.has_alpha())
                );
                assert!(
                    inventory
                        .image_header
                        .extra_channels
                        .iter()
                        .all(|extra| extra.bit_depth == inventory.image_header.bit_depth)
                );
                check_oracles(&encoded, &expected, format, bits);
                for (decoder, fragmented) in [(&whole, false), (&bounded, true)] {
                    let pixel_format = format.pixel_format(bits).unwrap();
                    let request = if format == LosslessModularFormat::Gray {
                        GpuOutputRequest::numeric(
                            pixel_format,
                            NumericSampleMapping::NativeUnsigned,
                        )
                    } else {
                        GpuOutputRequest::color(pixel_format)
                    }
                    .unwrap();
                    let mut session = if fragmented {
                        open_fragmented(decoder, &encoded, request)
                    } else {
                        decoder.open(&encoded, request).unwrap()
                    };
                    let frame = pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap();
                    assert!(
                        pollster::block_on(session.next_frame_async())
                            .unwrap()
                            .is_none()
                    );
                    drop(session);
                    let actual = read_bytes(&backend, &frame.output().outputs[0]);
                    let words: Vec<_> = actual
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|v| u32::from_le_bytes(*v))
                        .collect();
                    assert_eq!(
                        words, expected,
                        "GPU {format:?}/{bits}, {tree_mode:?}, bounded={fragmented}"
                    );
                    drop(frame);
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                }
            }
        }
    }
}
