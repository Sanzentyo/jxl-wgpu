#![cfg(not(target_arch = "wasm32"))]

mod animation;
mod lifetime;

use std::num::NonZeroU64;
use std::sync::Arc;

use jxl_gpu_bitstream::SampleBitDepth;
use jxl_gpu_formats::{
    Channel, ImageLayout, PitchLinearPlaneLayout, PixelFormat, RgbChannelOrder, SampleKind,
};
use jxl_gpu_protocol::Extent2d;
use jxl_test_support::gpu::planes::{open_fragmented, read};
use jxl_test_support::oracles::{extra_channels, modular_integer};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, NumericSampleMapping, WgpuDecodeEngine,
};
use jxl_wgpu_encode::{
    BufferImageSource, LosslessModularEncoder, LosslessModularFormat, LosslessModularTreeMode,
    WgpuContext,
};
use wgpu::util::DeviceExt;

fn depth(bits: u8) -> SampleBitDepth {
    SampleBitDepth::Float {
        bits_per_sample: u32::from(bits),
        exponent_bits_per_sample: if bits == 16 { 5 } else { 8 },
    }
}

use jxl_test_support::oracles::sample_bits::binary32;

fn source(
    context: &WgpuContext,
    extent: Extent2d,
    format: LosslessModularFormat,
    bits: u8,
    phase: u32,
) -> (BufferImageSource, Vec<u32>) {
    let channels = format.channel_count();
    let component_bytes = u64::from(bits / 8);
    let row_bytes = u64::from(extent.width) * u64::from(channels) * component_bytes;
    let offset = 3;
    let row_stride = row_bytes + 5;
    let mut bytes =
        vec![0xa5; (offset + u64::from(extent.height) * row_stride).div_ceil(4) as usize * 4];
    let special: &[u32] = if bits == 16 {
        &[
            0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x7bff, 0xfbff, 0x7c00, 0xfc00,
            0x7c01, 0xfc01, 0x7fff, 0xffff, 0x3c00, 0xbc00, 0x3800, 0xb800,
        ]
    } else {
        &[
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x807f_ffff,
            0x0080_0000,
            0x8080_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7f80_0001,
            0xff80_0001,
            0x7fff_ffff,
            0xffff_ffff,
            0x3f80_0000,
            0xbf80_0000,
            0x3f00_0000,
            0xbf00_0000,
        ]
    };
    let mut expected = Vec::new();
    for y in 0..extent.height {
        for x in 0..extent.width {
            for channel in 0..channels {
                let index = (y * extent.width + x) * channels + channel + phase;
                let word = if extent == Extent2d::new(256, 16) && bits == 32 {
                    let fractions = [
                        0, 1, 0x3ff, 0x400, 0x3f_ffff, 0x40_0000, 0x40_0001, 0x7f_ffff,
                    ];
                    ((y & 1) << 31) | (x << 23) | fractions[(y / 2) as usize]
                } else if extent == Extent2d::new(256, 256) && bits == 16 {
                    index & 0xffff
                } else if x % 256 < 32 {
                    special[index as usize % special.len()]
                } else if x % 256 < 64 {
                    0 // Long zero runs exercise LZ77 as well as full-width residuals.
                } else {
                    index.wrapping_mul(0x9e37_79b9).rotate_left(11) & (u32::MAX >> (32 - bits))
                };
                let address = (offset
                    + u64::from(y) * row_stride
                    + u64::from(x * channels + channel) * component_bytes)
                    as usize;
                bytes[address..address + component_bytes as usize]
                    .copy_from_slice(&word.to_le_bytes()[..component_bytes as usize]);
                expected.push(word);
            }
        }
    }
    let layout = ImageLayout::from_planes(
        extent,
        format.float_pixel_format(bits).unwrap(),
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
            label: Some("floating encoder source with poisoned padding"),
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
    let native = extra_channels::libjxl_output(
        encoded,
        &["--original", "--preserve-alpha", "--keep-orientation"],
    )
    .expect("required libjxl 0.12.0 floating original-component oracle");
    assert_eq!(native.len(), frames.len() * frame_words);
    for (frame_index, expected) in frames.iter().enumerate() {
        assert_eq!(expected.len(), pixels * channels);
        let rust = modular_integer::original_planes(encoded, frame_index);
        assert_eq!(rust.len(), channels);
        for (channel, values) in rust.iter().enumerate() {
            assert_eq!(values.len(), pixels);
            for (pixel, &word) in values.iter().enumerate() {
                assert_eq!(
                    word as u32,
                    expected[pixel * channels + channel],
                    "jxl-oxide original words {format:?}/{bits}/{frame_index}/{pixel}/{channel}"
                );
            }
        }
        let native = &native[frame_index * frame_words..(frame_index + 1) * frame_words];
        let rgba = expected_output(expected, format, bits, None);
        for (sample, &reference) in rgba.iter().enumerate() {
            assert_eq!(
                native[sample].to_bits(),
                reference,
                "libjxl original {format:?}/{bits}/{frame_index}/{sample}"
            );
        }
        if format.has_alpha() {
            for (pixel, sample) in expected.chunks_exact(channels).enumerate() {
                assert_eq!(
                    native[pixels * 4 + pixel].to_bits(),
                    binary32(sample[3], bits)
                );
            }
        }
    }
}

fn decoders(backend: &WgpuBackend) -> [GpuDecoder<WgpuDecodeEngine>; 2] {
    [
        GpuDecoder::wgpu(backend.clone()).unwrap(),
        GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
        ),
    ]
}

fn requests(format: LosslessModularFormat) -> Vec<(Option<u32>, GpuOutputRequest)> {
    let mut requests: Vec<_> = (0..format.channel_count())
        .map(|channel| {
            let request = GpuOutputRequest::numeric(
                PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                NumericSampleMapping::NativeFloat,
            )
            .unwrap();
            let request = if channel == 3 {
                request.with_extra_channel(0).unwrap()
            } else {
                request.with_color_channel(channel).unwrap()
            };
            (Some(channel), request)
        })
        .collect();
    requests.push((
        None,
        GpuOutputRequest::color(PixelFormat::rgb_f32(
            RgbChannelOrder::Rgba,
            false,
            jxl_wgpu_decode::vardct_rgb8_format().color_spec,
        ))
        .unwrap()
        .with_alpha_output_policy(AlphaOutputPolicy::Preserve),
    ));
    requests
}

fn expected_output(
    words: &[u32],
    format: LosslessModularFormat,
    bits: u8,
    selected: Option<u32>,
) -> Vec<u32> {
    let channels = format.channel_count() as usize;
    if let Some(channel) = selected {
        return words
            .iter()
            .skip(channel as usize)
            .step_by(channels)
            .map(|&word| binary32(word, bits))
            .collect();
    }
    words
        .chunks_exact(channels)
        .flat_map(|pixel| {
            let rgb = if format == LosslessModularFormat::Gray {
                [pixel[0]; 3]
            } else {
                [pixel[0], pixel[1], pixel[2]]
            };
            let [r, g, b] = rgb.map(|word| binary32(word, bits));
            [
                r,
                g,
                b,
                if format.has_alpha() {
                    binary32(pixel[3], bits)
                } else {
                    1.0f32.to_bits()
                },
            ]
        })
        .collect()
}

fn check_gpu(
    backend: &WgpuBackend,
    decoders: &[GpuDecoder<WgpuDecodeEngine>; 2],
    encoded: &[u8],
    expected: &[u32],
    format: LosslessModularFormat,
    bits: u8,
) {
    for (decoder, fragmented) in [(&decoders[0], false), (&decoders[1], true)] {
        for (channel, request) in requests(format) {
            let mut session = if fragmented {
                open_fragmented(decoder, encoded, request)
            } else {
                decoder.open(encoded, request).unwrap()
            };
            let frame = pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap();
            let actual = read(backend, &frame.output().outputs[0]);
            let reference = expected_output(expected, format, bits, channel);
            assert_eq!(actual.len(), reference.len());
            for (index, (&actual, &reference)) in actual.iter().zip(&reference).enumerate() {
                assert_eq!(
                    actual, reference,
                    "GPU {format:?}/{bits}/{channel:?}/{fragmented}, sample {index}"
                );
            }
            assert!(
                pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .is_none()
            );
            drop((frame, session));
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn floating_source_bits_roundtrip_through_independent_oracles_and_gpu() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let decoders = decoders(&backend);
    let mut checked = 0;
    for tree_mode in [
        LosslessModularTreeMode::SharedGlobal,
        LosslessModularTreeMode::LocalPerGroup,
    ] {
        let encoder = LosslessModularEncoder::with_tree_mode(context.clone(), tree_mode);
        for format in [
            LosslessModularFormat::Gray,
            LosslessModularFormat::Rgb,
            LosslessModularFormat::Rgba,
        ] {
            for bits in [16, 32] {
                for (width, height) in [(1, 1), (1, 257), (257, 3), (2051, 1)] {
                    let (source, expected) =
                        source(&context, Extent2d::new(width, height), format, bits, 0);
                    assert_eq!(
                        encoder.memory_plan(&source).unwrap().sample_bit_depth(),
                        depth(bits)
                    );
                    let submission = encoder.submit_container(source).unwrap();
                    assert_eq!(submission.sample_bit_depth(), depth(bits));
                    let encoded = pollster::block_on(submission).unwrap();
                    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
                    let inventory = jxl_gpu_bitstream::parse(&encoded, Default::default())
                        .unwrap()
                        .codestream_inventory(Default::default())
                        .unwrap();
                    assert_eq!(inventory.image_header.bit_depth, depth(bits));
                    assert!(!inventory.image_header.modular_16bit_buffers);
                    assert_eq!(
                        inventory.image_header.extra_channels.len(),
                        usize::from(format.has_alpha())
                    );
                    assert!(
                        inventory
                            .image_header
                            .extra_channels
                            .iter()
                            .all(|extra| extra.bit_depth == depth(bits))
                    );
                    check_oracles(&encoded, &expected, format, bits);
                    check_gpu(&backend, &decoders, &encoded, &expected, format, bits);
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 48);
}

#[test]
fn every_binary16_word_and_binary32_exponent_retains_its_bits() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let decoders = decoders(&backend);
    for tree_mode in [
        LosslessModularTreeMode::SharedGlobal,
        LosslessModularTreeMode::LocalPerGroup,
    ] {
        let encoder = LosslessModularEncoder::with_tree_mode(context.clone(), tree_mode);
        for (bits, extent) in [(16, Extent2d::new(256, 256)), (32, Extent2d::new(256, 16))] {
            let format = LosslessModularFormat::Gray;
            let (source, expected) = source(&context, extent, format, bits, 0);
            let encoded = encoder.encode(source).unwrap();
            check_oracles(&encoded, &expected, format, bits);
            check_gpu(&backend, &decoders, &encoded, &expected, format, bits);
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}
