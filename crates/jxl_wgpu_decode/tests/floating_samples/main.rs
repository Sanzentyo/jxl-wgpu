#![cfg(not(target_arch = "wasm32"))]

use jxl_gpu_bitstream::SampleBitDepth;
use jxl_gpu_formats::{Channel, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_test_support::gpu::planes;
use jxl_test_support::gpu::rendering;
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    DecodeProfile, GpuDecoder, GpuOutputRequest, NumericSampleMapping, SpotColorPolicy,
    WgpuDecodeEngine,
};
use planes::{open_fragmented, read, read_bytes};
use std::num::NonZeroU64;

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
        .as_chunks::<2>()
        .0
        .iter()
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

fn numeric(mapping: NumericSampleMapping) -> GpuOutputRequest {
    GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
        mapping,
    )
    .unwrap()
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

#[test]
fn floating_channels_resample_compose_and_render_against_libjxl() {
    rendering::check_rendering("floating", rendering_fixtures());
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
                    frame[..pixels * 4]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .flat_map(|pixel| {
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
