#![cfg(not(target_arch = "wasm32"))]

use jxl_gpu_bitstream::SampleBitDepth;
use jxl_gpu_formats::{Channel, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    DecodeProfile, GpuDecoder, GpuOutputRequest, ModularChannels, NumericSampleMapping,
    SpotColorPolicy, WgpuDecodeEngine, native_modular_pixel_format,
};
use std::num::NonZeroU64;

use jxl_test_support::gpu::planes;
use jxl_test_support::gpu::rendering;
use planes::{open_fragmented, read, read_bytes};

fn text(name: &str, suffix: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test-data/integer")
            .join(format!("{name}.{suffix}")),
    )
    .unwrap()
}

fn encoded(name: &str) -> Vec<u8> {
    let digits = text(name, "jxl.hex").split_whitespace().collect::<String>();
    digits
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn words(name: &str, suffix: &str) -> Vec<u32> {
    text(name, suffix)
        .split_whitespace()
        .map(|word| u32::from_str_radix(word, 16).unwrap())
        .collect()
}

fn precisions() -> Vec<String> {
    let mut names: Vec<_> = std::fs::read_dir(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/integer"),
    )
    .unwrap()
    .map(|entry| entry.unwrap().file_name())
    .filter_map(|name| name.to_str()?.strip_suffix(".u32.hex").map(str::to_owned))
    .collect();
    names.sort();
    assert_eq!(names.len(), 42);
    names
}

fn depth(sample: SampleBitDepth) -> u8 {
    match sample {
        SampleBitDepth::Integer { bits_per_sample } => bits_per_sample.try_into().unwrap(),
        _ => panic!("expected integer source"),
    }
}

fn numeric(bits: u8) -> GpuOutputRequest {
    GpuOutputRequest::numeric(
        native_modular_pixel_format(ModularChannels::Gray, bits).unwrap(),
        NumericSampleMapping::NativeUnsigned,
    )
    .unwrap()
}

fn normalized() -> GpuOutputRequest {
    GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
        NumericSampleMapping::NormalizedUnsigned,
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

fn unpack(bytes: &[u8], bits: u8) -> Vec<u32> {
    let storage = bits.next_power_of_two().max(8) as usize / 8;
    assert!(bytes.len().is_multiple_of(storage));
    bytes
        .chunks_exact(storage)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0, |value, (i, &byte)| value | (u32::from(byte) << (8 * i)))
        })
        .collect()
}

#[test]
fn every_integer_precision_preserves_native_codes_and_independent_alpha() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let whole = GpuDecoder::wgpu(backend.clone()).unwrap();
    let bounded = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    for name in precisions() {
        let data = encoded(&name);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = inventory.image_header;
        let bits = depth(image.bit_depth);
        let colors = if image.grayscale { 1 } else { 3 };
        let channels = colors + image.extra_channels.len();
        let raw = words(&name, "u32.hex");
        assert_eq!(
            raw.len(),
            image.width as usize * image.height as usize * channels
        );
        let alpha_bits = image
            .extra_channels
            .first()
            .map(|extra| depth(extra.bit_depth));
        let request = if colors == 1 {
            numeric(bits)
        } else {
            GpuOutputRequest::color(
                native_modular_pixel_format(
                    if alpha_bits.is_some() {
                        ModularChannels::Rgba
                    } else {
                        ModularChannels::Rgb
                    },
                    bits,
                )
                .unwrap(),
            )
            .unwrap()
        };
        let expected: Vec<_> = raw
            .chunks_exact(channels)
            .flat_map(|pixel| {
                pixel.iter().enumerate().map(|(c, &code)| {
                    if c == colors {
                        let source = (1u64 << alpha_bits.unwrap()) - 1;
                        let target = (1u64 << bits) - 1;
                        ((u64::from(code) * target + source / 2) / source) as u32
                    } else {
                        code
                    }
                })
            })
            .collect();
        for (decoder, fragmented) in [(&whole, false), (&bounded, true)] {
            let mut session = if fragmented {
                open_fragmented(decoder, &data, request.clone())
            } else {
                decoder
                    .open(&data, request.clone())
                    .unwrap_or_else(|e| panic!("{name}: open: {e}"))
            };
            assert!(
                matches!(session.profile(), DecodeProfile::Modular { sample_bit_depth, .. }
                if sample_bit_depth == image.bit_depth)
            );
            let frame = pollster::block_on(session.next_frame_async())
                .unwrap_or_else(|e| panic!("{name}: decode: {e}"))
                .unwrap();
            let actual = unpack(&read_bytes(&backend, &frame.output().outputs[0]), bits);
            assert_eq!(actual, expected, "{name}, fragmented={fragmented}");
            drop(frame);
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            if let Some(bits) = alpha_bits {
                let mut session = decoder
                    .open(&data, numeric(bits).with_extra_channel(0).unwrap())
                    .unwrap();
                let frame = pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap();
                let actual = unpack(&read_bytes(&backend, &frame.output().outputs[0]), bits);
                let expected: Vec<_> = raw
                    .chunks_exact(channels)
                    .map(|pixel| pixel[colors])
                    .collect();
                assert_eq!(actual, expected, "{name}: selected alpha");
            }
        }
    }
}

#[test]
fn integer_f32_normalization_and_color_match_independent_libjxl() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for name in precisions() {
        let data = encoded(&name);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = inventory.image_header;
        let pixels = image.width as usize * image.height as usize;
        let oracle = words(&name, "f32.hex");
        let mut requests = vec![(color(), oracle[..pixels * 4].to_vec())];
        if image.grayscale {
            requests.push((
                normalized(),
                oracle[..pixels * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|pixel| pixel[0])
                    .collect(),
            ));
        }
        if !image.extra_channels.is_empty() {
            requests.push((
                normalized().with_extra_channel(0).unwrap(),
                oracle[pixels * 4..].to_vec(),
            ));
        }
        for (request, expected) in requests {
            let mut session = decoder
                .open(&data, request)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let frame = pollster::block_on(session.next_frame_async())
                .unwrap_or_else(|e| panic!("{name}: {e}"))
                .unwrap();
            let actual = read(&backend, &frame.output().outputs[0]);
            assert_eq!(actual.len(), expected.len());
            for (i, (a, b)) in actual.into_iter().zip(expected).enumerate() {
                // Portable GPU conversion/division may differ by one ULP from libjxl's accurate
                // double multiplication above 23 bits.
                assert!(
                    a.abs_diff(b) <= 1,
                    "{name}/{i}: {} vs {}",
                    f32::from_bits(a),
                    f32::from_bits(b)
                );
            }
        }
    }
}

#[test]
fn native_integer_layout_has_canonical_storage_and_checked_precision() {
    for bits in 1..=31 {
        let format = native_modular_pixel_format(ModularChannels::Gray, bits).unwrap();
        let layout =
            jxl_gpu_formats::ImageLayout::packed(jxl_gpu_protocol::Extent2d::new(3, 2), format)
                .unwrap();
        assert_eq!(
            layout.logical_size,
            6 * u64::from(bits.next_power_of_two().max(8) / 8)
        );
        assert!(numeric(bits).extra_channel().is_none());
    }
    for bits in [0, 32, u8::MAX] {
        assert!(matches!(
            native_modular_pixel_format(ModularChannels::Gray, bits),
            Err(jxl_wgpu_decode::Error::UnsupportedOutputFormat(_))
        ));
    }
}

#[test]
fn native_scalar_requests_match_declared_depth_before_submission() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend).unwrap();
    for (name, selected) in [
        ("31-1-0-33x5-p0-r0", None),
        ("composition_extras_integer_gray", None),
        ("extras_integer_rgb", Some(0)),
        ("vardct_extras_integer_rgb_extended31", Some(0)),
        ("composition_extras_integer_resampled_extended31", Some(0)),
    ] {
        let data = encoded(name);
        let mut request = numeric(16);
        if let Some(index) = selected {
            request = request.with_extra_channel(index).unwrap();
        }
        let Err(error) = decoder.open(&data, request) else {
            panic!("{name}: a mismatched integer declaration was accepted");
        };
        assert!(
            matches!(
                error,
                jxl_wgpu_decode::Error::UnsupportedOutputFormat(_)
                    | jxl_wgpu_decode::Error::VarDct(
                        jxl_wgpu_decode::VarDctDecodeError::ScalarOutput(
                            jxl_wgpu_decode::ModularScalarOutputError::Invalid { .. }
                        )
                    )
            ),
            "{name}: {error}"
        );
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        let image = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap()
            .image_header;
        let actual_depth = selected.map_or(image.bit_depth, |index| {
            image.extra_channels[index as usize].bit_depth
        });
        let mut request = numeric(depth(actual_depth));
        if let Some(index) = selected {
            request = request.with_extra_channel(index).unwrap();
        }
        drop(decoder.open(&data, request).unwrap());
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn wide_integer_color_quantizes_at_presentation_without_setting_padding_bits() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for name in [
        "31-1-0-33x5-p0-r0",
        "extras_integer_gray",
        "extras_integer_resampled",
        "extras_integer_resampled_extended31",
        "vardct_extras_integer_rgb_extended31",
        "vardct_extras_integer_progressive_dc_extended31",
        "composition_extras_integer_resampled_extended31",
        "composition_extras_integer_vardct_resampled_extended31",
    ] {
        let data = encoded(name);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = inventory.image_header;
        let pixels = image.width as usize * image.height as usize;
        let bits = depth(image.bit_depth);
        for (format, output_bits, channels) in [
            (
                PixelFormat::rgb8(RgbChannelOrder::Rgb, false, color().format().color_spec),
                8,
                3,
            ),
            (
                PixelFormat::rgb8(RgbChannelOrder::Rgba, false, color().format().color_spec),
                8,
                4,
            ),
            (
                native_modular_pixel_format(ModularChannels::Rgba, bits).unwrap(),
                bits,
                4,
            ),
        ] {
            let request = GpuOutputRequest::color(format)
                .unwrap()
                .with_spot_color_policy(SpotColorPolicy::Preserve);
            let mut session = decoder
                .open(&data, request)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let maximum = (1u32 << output_bits) - 1;
            let expected = words(name, "f32.hex");
            let mut frames = expected.chunks_exact(pixels * (4 + image.extra_channels.len()));
            while let Some(frame) = pollster::block_on(session.next_frame_async())
                .unwrap_or_else(|e| panic!("{name}: {e}"))
            {
                let actual = unpack(
                    &read_bytes(&backend, &frame.output().outputs[0]),
                    output_bits,
                );
                let expected = &frames.next().unwrap()[..pixels * 4];
                assert_eq!(actual.len(), pixels * channels);
                for (index, (a, b)) in actual
                    .iter()
                    .zip(
                        expected
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .flat_map(|pixel| &pixel[..channels]),
                    )
                    .enumerate()
                {
                    let value = f64::from(f32::from_bits(*b).clamp(0.0, 1.0));
                    let code = (value * f64::from(maximum)).round() as u32;
                    let tolerance = if output_bits == 8 {
                        1
                    } else {
                        // Wide color output retains the established reconstruction precision;
                        // valid source metadata does not imply a 31-bit accurate lossy IDCT.
                        (f64::from(maximum) * if name.contains("vardct") { 0.003 } else { 2e-5 })
                            .ceil() as u32
                    };
                    assert!(*a <= maximum, "{name}/{index}: high padding bits changed");
                    assert!(
                        a.abs_diff(code) <= tolerance,
                        "{name}/{index}: {a} vs {code} at {output_bits} bits"
                    );
                }
            }
            assert!(frames.next().is_none());
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn wide_integer_channels_resample_compose_and_render_against_libjxl() {
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
        ] {
            names.push(format!("{prefix}_integer_{suffix}"));
        }
    }
    names.push("vardct_extras_integer_progressive_dc".into());
    for suffix in ["rgb", "gray", "vardct", "resampled", "vardct_resampled"] {
        names.push(format!("composition_extras_integer_{suffix}"));
    }
    for bits in 18..=31 {
        names.push(format!("vardct_extras_integer_rgb_extended{bits}"));
    }
    for name in [
        "extras_integer_resampled",
        "extras_integer_distributed",
        "vardct_extras_integer_progressive_dc",
        "vardct_extras_integer_resampled",
        "composition_extras_integer_resampled",
        "composition_extras_integer_vardct_resampled",
    ] {
        names.push(format!("{name}_extended31"));
    }
    rendering::check_rendering("integer", names);
}
