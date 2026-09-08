#![cfg(not(target_arch = "wasm32"))]

use jxl_gpu_bitstream::SampleBitDepth;
use jxl_gpu_formats::{Channel, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_wgpu::{GpuImageOutput, WgpuBackend};
use jxl_wgpu_decode::{
    AlphaOutputPolicy, DecodeProfile, GpuDecodeSession, GpuDecoder, GpuOutputRequest,
    NumericSampleMapping, SpotColorPolicy, WgpuDecodeEngine, WgpuDecodeSubmissionSession,
};
use std::num::NonZeroU64;
use std::sync::Arc;

fn text(name: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test-data/floating")
            .join(name),
    )
    .unwrap()
}

fn encoded(name: &str) -> Vec<u8> {
    let hex = text(&format!("{name}.jxl.hex"))
        .split_whitespace()
        .collect::<String>();
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn reference(name: &str, suffix: &str) -> Vec<u32> {
    text(&format!("{name}.{suffix}"))
        .split_whitespace()
        .map(|word| u32::from_str_radix(word, 16).unwrap())
        .collect()
}

fn fixture(bits: u32, exponent: u32) -> (Vec<u8>, Vec<u32>) {
    let name = format!("{bits}-{exponent}");
    (encoded(&name), reference(&name, "f32.hex"))
}

fn open_fragmented(
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    data: &[u8],
    request: GpuOutputRequest,
) -> GpuDecodeSession<WgpuDecodeSubmissionSession> {
    let mut stream = decoder.stream(request).unwrap();
    let mut transport =
        jxl_gpu_bitstream::ContainerStreamScanner::new(decoder.container_stream_limits());
    for chunk in data.chunks(43) {
        for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
            stream.push_transport_event(&event).unwrap();
        }
    }
    for event in transport.finish_input().unwrap() {
        stream.push_transport_event(&event).unwrap();
    }
    stream.finish().unwrap()
}

fn numeric(mapping: NumericSampleMapping) -> GpuOutputRequest {
    GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
        mapping,
    )
    .unwrap()
}

fn color() -> GpuOutputRequest {
    GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_spot_color_policy(SpotColorPolicy::Preserve)
}

fn read_bytes(backend: &WgpuBackend, output: &GpuImageOutput) -> Vec<u8> {
    let device = backend.device();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("floating source test readback"),
        size: output.buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut commands = device.create_command_encoder(&Default::default());
    commands.copy_buffer_to_buffer(output.buffer.as_wgpu_buffer(), 0, &buffer, 0, buffer.size());
    backend.queue().submit([commands.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    let bytes = mapped[..output.layout.logical_size as usize].to_vec();
    drop(mapped);
    buffer.unmap();
    bytes
}

fn read(backend: &WgpuBackend, output: &GpuImageOutput) -> Vec<u32> {
    read_bytes(backend, output)
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

#[test]
fn all_floating_precisions_preserve_binary32_bits_through_gpu_decode() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let bounded = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    let numeric = numeric(NumericSampleMapping::NativeFloat);
    let color = GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgb,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap();
    for exponent in 2..=8 {
        for mantissa in 2..=23 {
            let bits = 1 + exponent + mantissa;
            let (data, expected) = fixture(bits, exponent);
            assert_eq!(expected.len(), 120);
            let depth = SampleBitDepth::Float {
                bits_per_sample: bits,
                exponent_bits_per_sample: exponent,
            };
            for (engine, request, components, fragmented) in [
                (&decoder, &numeric, 1, false),
                (&bounded, &numeric, 1, true),
                (&decoder, &color, 3, false),
            ] {
                let mut session = if fragmented {
                    open_fragmented(engine, &data, request.clone())
                } else {
                    engine
                        .open(&data, request.clone())
                        .unwrap_or_else(|error| panic!("{bits}/{exponent}: open: {error}"))
                };
                assert!(
                    matches!(session.profile(),DecodeProfile::Modular { sample_bit_depth, .. } if sample_bit_depth==depth)
                );
                let frame = pollster::block_on(session.next_frame_async())
                    .unwrap_or_else(|error| panic!("{bits}/{exponent}: decode: {error}"))
                    .unwrap();
                let actual = read(&backend, &frame.output().outputs[0]);
                let expected: Vec<_> = expected
                    .iter()
                    .flat_map(|&word| std::iter::repeat_n(word, components))
                    .collect();
                assert_eq!(
                    actual, expected,
                    "{bits}/{exponent}, {components} components"
                );
            }
        }
    }
}

fn rendering_fixtures() -> Vec<String> {
    let mut names = Vec::new();
    for prefix in ["extras", "vardct_extras"] {
        for suffix in [
            "rgb",
            "gray",
            "associated",
            "resampled",
            "resampled4",
            "resampled8",
            "distributed",
            "squeeze",
            "integer_rgb",
            "integer_gray",
        ] {
            names.push(format!("{prefix}_float_{suffix}"));
        }
    }
    names.push("vardct_extras_float_progressive_dc".into());
    names.push("extras_float_global".into());
    for suffix in ["rgb", "gray", "vardct", "resampled", "vardct_resampled"] {
        names.push(format!("composition_extras_float_{suffix}"));
    }
    names
}

fn compare(name: &str, actual: &[u32], expected: &[u32], color: bool, unpremultiplied: bool) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    let tolerance = if color {
        if name.contains("vardct") { 0.003 } else { 2e-5 }
    } else {
        2e-6
    };
    for (index, (&actual_word, &expected_word)) in actual.iter().zip(expected).enumerate() {
        let (a, b) = (f32::from_bits(actual_word), f32::from_bits(expected_word));
        // Compare reconstructed color before output unpremultiplication, matching the existing
        // VarDCT oracle contract when a tiny alpha amplifies IDCT rounding differences.
        let scale = if unpremultiplied && index % 4 != 3 {
            f32::from_bits(expected[index / 4 * 4 + 3]).max(1.0 / 67108864.0)
        } else {
            1.0
        };
        let limit = if color && index % 4 == 3 {
            2e-6
        } else {
            tolerance
        };
        assert!(
            a.is_finite() && (a - b).abs() * scale <= limit * (1.0 + (b * scale).abs()),
            "{name}/{index}: {a} vs {b}, difference {}",
            (a - b).abs()
        );
    }
}

#[test]
fn floating_channels_resample_compose_and_render_against_libjxl() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let whole = GpuDecoder::wgpu(backend.clone()).unwrap();
    let bounded = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    for name in rendering_fixtures() {
        let data = encoded(&name);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let pixels = image.width as usize * image.height as usize;
        let channels = image.extra_channels.len();
        let base = reference(&name, "f32.hex");
        let frame_words = pixels * (4 + channels);
        let frames = base.len() / frame_words;
        assert!(frames > 0 && base.len().is_multiple_of(frame_words));
        if name.contains("progressive_dc") {
            assert!(inventory.frames.len() > 1);
        }
        for selected in 0..channels + 3 {
            let color_output = selected >= channels;
            let request = if !color_output {
                numeric(
                    if matches!(
                        image.extra_channels[selected].bit_depth,
                        SampleBitDepth::Float { .. }
                    ) {
                        NumericSampleMapping::NativeFloat
                    } else {
                        NumericSampleMapping::NormalizedUnsigned
                    },
                )
                .with_extra_channel(selected as u32)
                .unwrap()
            } else if selected == channels + 1 {
                color().with_spot_color_policy(SpotColorPolicy::Render)
            } else if selected == channels + 2 {
                color().with_alpha_output_policy(AlphaOutputPolicy::Preserve)
            } else {
                color()
            };
            let expected: Vec<_> = if selected == channels + 1 {
                reference(&name, "spots.f32.hex")
            } else if selected == channels + 2 {
                reference(&name, "associated.f32.hex")
            } else {
                base.chunks_exact(frame_words)
                    .flat_map(|frame| {
                        let range = if color_output {
                            0..pixels * 4
                        } else {
                            pixels * (4 + selected)..pixels * (5 + selected)
                        };
                        frame[range].iter().copied()
                    })
                    .collect()
            };
            let mut first = Vec::new();
            for (decoder, fragmented) in [(&whole, false), (&bounded, true)] {
                let mut session = if fragmented {
                    open_fragmented(decoder, &data, request.clone())
                } else {
                    decoder
                        .open(&data, request.clone())
                        .unwrap_or_else(|e| panic!("{name}/{selected}: open: {e}"))
                };
                assert_eq!(session.metadata().extra_channels, image.extra_channels);
                let mut actual = Vec::new();
                let mut count = 0;
                while let Some(frame) = pollster::block_on(session.next_frame_async())
                    .unwrap_or_else(|e| panic!("{name}/{selected}: decode: {e}"))
                {
                    actual.extend(read(&backend, &frame.output().outputs[0]));
                    count += 1;
                }
                assert_eq!(count, frames, "{name}/{selected}");
                let associated = image
                    .extra_channels
                    .iter()
                    .find_map(|extra| match extra.channel_type {
                        jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha { associated } => {
                            Some(associated)
                        }
                        _ => None,
                    })
                    .unwrap_or(false);
                compare(
                    &format!("{name}/{selected}"),
                    &actual,
                    &expected,
                    color_output,
                    color_output && associated && selected != channels + 2,
                );
                if fragmented {
                    assert_eq!(
                        actual, first,
                        "{name}/{selected}: bounded input changes words"
                    );
                } else {
                    first = actual;
                }
                drop(session);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}

#[test]
fn floating_color_quantizes_only_at_the_requested_integer_output() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for name in [
        "extras_float_rgb",
        "extras_float_gray",
        "extras_float_associated",
        "extras_float_resampled",
        "composition_extras_float_rgb",
        "vardct_extras_float_rgb",
        "vardct_extras_float_progressive_dc",
        "extras_float_integer_rgb",
        "vardct_extras_float_integer_rgb",
    ] {
        let data = encoded(name);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = inventory.image_header;
        let pixels = image.width as usize * image.height as usize;
        for (order, channels) in [(RgbChannelOrder::Rgb, 3), (RgbChannelOrder::Rgba, 4)] {
            let request = GpuOutputRequest::color(PixelFormat::rgb8(
                order,
                false,
                jxl_wgpu_decode::vardct_rgb8_format().color_spec,
            ))
            .unwrap()
            .with_spot_color_policy(SpotColorPolicy::Preserve);
            let mut session = decoder
                .open(&data, request)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut actual = Vec::new();
            while let Some(frame) = pollster::block_on(session.next_frame_async()).unwrap() {
                actual.extend(read_bytes(&backend, &frame.output().outputs[0]));
            }
            let expected: Vec<_> = reference(name, "f32.hex")
                .chunks_exact(pixels * (4 + image.extra_channels.len()))
                .flat_map(|frame| {
                    frame[..pixels * 4].chunks_exact(4).flat_map(|pixel| {
                        pixel[..channels].iter().map(|&word| {
                            (f32::from_bits(word).clamp(0.0, 1.0) * 255.0).round() as u8
                        })
                    })
                })
                .collect();
            assert_eq!(actual.len(), expected.len(), "{name}");
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    actual.abs_diff(*expected) <= 1,
                    "{name}/{index}: {actual} vs {expected}"
                );
            }
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn numeric_mapping_rejects_mismatched_source_types_before_submission() {
    for (kind, bits, channels) in [
        (SampleKind::Unsigned, 16, &[Channel::X][..]),
        (SampleKind::Float, 64, &[Channel::X][..]),
        (SampleKind::Float, 32, &[Channel::X, Channel::Y][..]),
    ] {
        assert!(matches!(
            GpuOutputRequest::numeric(
                PixelFormat::non_color(kind, bits, channels),
                NumericSampleMapping::NativeFloat
            ),
            Err(jxl_wgpu_decode::Error::UnsupportedOutputFormat(_))
        ));
    }
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend).unwrap();
    for (name, request) in [
        ("32-8", numeric(NumericSampleMapping::NormalizedUnsigned)),
        (
            "32-8",
            GpuOutputRequest::numeric(
                PixelFormat::non_color(SampleKind::Unsigned, 16, &[Channel::X]),
                NumericSampleMapping::NativeUnsigned,
            )
            .unwrap(),
        ),
        (
            "extras_float_rgb",
            numeric(NumericSampleMapping::NormalizedUnsigned)
                .with_extra_channel(0)
                .unwrap(),
        ),
        (
            "extras_float_rgb",
            numeric(NumericSampleMapping::NativeFloat)
                .with_extra_channel(1)
                .unwrap(),
        ),
        (
            "vardct_extras_float_rgb",
            numeric(NumericSampleMapping::NormalizedUnsigned)
                .with_extra_channel(0)
                .unwrap(),
        ),
        (
            "vardct_extras_float_rgb",
            numeric(NumericSampleMapping::NativeFloat)
                .with_extra_channel(1)
                .unwrap(),
        ),
        (
            "composition_extras_float_rgb",
            numeric(NumericSampleMapping::NormalizedUnsigned)
                .with_extra_channel(0)
                .unwrap(),
        ),
        (
            "composition_extras_float_rgb",
            numeric(NumericSampleMapping::NativeFloat)
                .with_extra_channel(1)
                .unwrap(),
        ),
    ] {
        assert!(decoder.open(&encoded(name), request).is_err(), "{name}");
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}
