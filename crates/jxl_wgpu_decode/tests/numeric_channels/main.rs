#![cfg(not(target_arch = "wasm32"))]

use std::num::NonZeroU64;

use jxl_gpu_bitstream::{CodestreamInventory, SampleBitDepth};
use jxl_gpu_formats::{Channel, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, DecodeProfile, Error, GpuDecoder, GpuOutputRequest, ModularChannels,
    NumericChannel, NumericSampleMapping, SpotColorPolicy, WgpuDecodeEngine,
    native_modular_pixel_format,
};

use jxl_test_support::gpu::planes;
use planes::{open_fragmented, read, read_bytes};

use jxl_test_support::oracles::extra_channels as oracle;
mod progression;

fn text(name: &str, suffix: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test-data")
            .join(format!("{name}.{suffix}")),
    )
    .unwrap()
}

fn fixture(name: &str) -> (Vec<u8>, CodestreamInventory) {
    let digits = text(name, "jxl.hex").split_whitespace().collect::<String>();
    let data: Vec<_> = digits
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
        .collect();
    let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    (data, inventory)
}

fn words(name: &str, suffix: &str) -> Vec<u32> {
    text(name, suffix)
        .split_whitespace()
        .map(|v| u32::from_str_radix(v, 16).unwrap())
        .collect()
}

fn scalar(mapping: NumericSampleMapping) -> GpuOutputRequest {
    GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
        mapping,
    )
    .unwrap()
}

fn native(bits: u8) -> GpuOutputRequest {
    GpuOutputRequest::numeric(
        native_modular_pixel_format(ModularChannels::Gray, bits).unwrap(),
        NumericSampleMapping::NativeUnsigned,
    )
    .unwrap()
}

fn unpack(bytes: &[u8], bits: u8) -> Vec<u32> {
    let storage = bits.next_power_of_two().max(8) as usize / 8;
    assert!(bytes.len().is_multiple_of(storage));
    bytes
        .chunks_exact(storage)
        .map(|bytes| {
            bytes
                .iter()
                .enumerate()
                .fold(0, |word, (i, &byte)| word | (u32::from(byte) << (8 * i)))
        })
        .collect()
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

#[test]
fn channel_selection_has_one_source_and_requires_scalar_sample_semantics() {
    let request = scalar(NumericSampleMapping::NativeFloat)
        .with_extra_channel(5)
        .unwrap()
        .with_color_channel(2)
        .unwrap();
    assert_eq!(request.numeric_channel(), Some(NumericChannel::Color(2)));
    assert_eq!(request.extra_channel(), None);
    let request = request
        .with_numeric_channel(NumericChannel::Extra(1))
        .unwrap();
    assert_eq!(request.color_channel(), None);
    assert_eq!(request.extra_channel(), Some(1));
    for request in [
        GpuOutputRequest::color(jxl_wgpu_decode::vardct_rgb8_format()).unwrap(),
        scalar(NumericSampleMapping::NormalizedGray8),
        GpuOutputRequest::numeric(
            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X, Channel::Y]),
            NumericSampleMapping::NormalizedUnsigned,
        )
        .unwrap(),
    ] {
        for channel in [NumericChannel::Color(0), NumericChannel::Extra(0)] {
            assert!(matches!(
                request.clone().with_numeric_channel(channel),
                Err(Error::UnsupportedOutputFormat(_))
            ));
        }
    }
}

#[test]
fn invalid_color_channels_and_source_mappings_fail_before_gpu_admission() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend).unwrap();
    for name in [
        "floating/extras_float_rgb",
        "floating/extras_float_gray",
        "floating/vardct_extras_float_rgb",
        "floating/vardct_extras_float_gray",
        "floating/composition_extras_float_vardct",
    ] {
        let (data, inventory) = fixture(name);
        let count = if inventory.image_header.grayscale {
            1
        } else {
            3
        };
        for index in [count, u32::MAX] {
            let error = decoder
                .open(
                    &data,
                    scalar(NumericSampleMapping::NativeFloat)
                        .with_color_channel(index)
                        .unwrap(),
                )
                .err()
                .expect("invalid color index was accepted");
            assert!(
                matches!(error, Error::ColorChannelIndex { index: i, count: c } if i == index && c == count)
            );
        }
        if count == 3 {
            assert!(matches!(
                decoder.open(&data, scalar(NumericSampleMapping::NativeFloat)),
                Err(Error::NumericColorChannelRequired)
            ));
        }
        for request in [scalar(NumericSampleMapping::NormalizedUnsigned), native(16)] {
            assert!(matches!(
                decoder.open(&data, request.with_color_channel(0).unwrap()),
                Err(Error::UnsupportedOutputFormat(_))
            ));
        }
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn selected_modular_rgb_preserves_wide_integer_codes_without_f32_roundtrip() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoders = decoders(&backend);
    for name in [
        "integer/17-3-31-33x5-p0-r0",
        "integer/31-3-5-33x5-p0-r0",
        "integer/31-3-24-33x5-p0-r0",
    ] {
        let (data, inventory) = fixture(name);
        let SampleBitDepth::Integer { bits_per_sample } = inventory.image_header.bit_depth else {
            unreachable!()
        };
        let expected = words(name, "u32.hex");
        let channels = 3 + inventory.image_header.extra_channels.len();
        for channel in 0..3 {
            for (bounded, decoder) in decoders.iter().enumerate() {
                let request = native(bits_per_sample as u8)
                    .with_extra_channel(0)
                    .unwrap()
                    .with_color_channel(channel)
                    .unwrap()
                    .with_alpha_output_policy(AlphaOutputPolicy::Associated);
                let mut session = if bounded == 0 {
                    decoder.open(&data, request).unwrap()
                } else {
                    open_fragmented(decoder, &data, request)
                };
                assert!(matches!(session.profile(), DecodeProfile::Modular { .. }));
                let frame = pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap();
                let actual = unpack(
                    &read_bytes(&backend, &frame.output().outputs[0]),
                    bits_per_sample as u8,
                );
                let expected: Vec<_> = expected
                    .chunks_exact(channels)
                    .map(|p| p[channel as usize])
                    .collect();
                assert_eq!(actual, expected, "{name}/{channel}/bounded{bounded}");
                drop((frame, session));
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}

#[test]
fn numeric_color_matches_original_associated_libjxl_components() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    eprintln!("numeric color GPU: {:?}", backend.adapter_info());
    let decoders = decoders(&backend);
    for name in [
        "floating/extras_float_rgb",
        "floating/extras_float_gray",
        "floating/extras_float_associated",
        "floating/vardct_extras_float_rgb",
        "floating/vardct_extras_float_gray",
        "floating/vardct_extras_float_associated",
        "floating/vardct_extras_float_resampled8",
        "floating/vardct_extras_float_progressive_dc",
        "floating/composition_extras_float_vardct",
        "integer/vardct_extras_integer_rgb",
        "integer/vardct_extras_integer_gray",
        "integer/vardct_extras_integer_resampled8",
        "integer/vardct_extras_integer_progressive_dc_extended31",
        "integer/composition_extras_integer_vardct_resampled_extended31",
    ] {
        let (data, inventory) = fixture(name);
        let image = &inventory.image_header;
        let rgba = words(name, "associated.f32.hex");
        let mut color = decoders[0]
            .open(
                &data,
                GpuOutputRequest::color(PixelFormat::rgb_f32(
                    RgbChannelOrder::Rgba,
                    false,
                    jxl_wgpu_decode::vardct_rgb8_format().color_spec,
                ))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                .with_spot_color_policy(SpotColorPolicy::Preserve),
            )
            .unwrap();
        let mut color_words = Vec::new();
        while let Some(frame) = color.next_frame().unwrap() {
            color_words.extend(read(&backend, &frame.output().outputs[0]));
        }
        drop(color);
        assert_eq!(color_words.len(), rgba.len());
        let channels = if image.grayscale { 1 } else { 3 };
        let mapping = match image.bit_depth {
            SampleBitDepth::Float { .. } => NumericSampleMapping::NativeFloat,
            SampleBitDepth::Integer { .. } => NumericSampleMapping::NormalizedUnsigned,
        };
        for channel in 0..channels {
            let expected: Vec<_> = rgba
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| f32::from_bits(p[channel as usize]))
                .collect();
            let mut whole = Vec::new();
            for (bounded, decoder) in decoders.iter().enumerate() {
                // Default gray selection and both alpha policies must preserve exactly the same
                // original association. Spot rendering is enabled deliberately and must be ignored.
                let request = if image.grayscale && bounded == 0 {
                    scalar(mapping)
                } else {
                    scalar(mapping).with_color_channel(channel).unwrap()
                }
                .with_alpha_output_policy(if bounded == 0 {
                    AlphaOutputPolicy::Associated
                } else {
                    AlphaOutputPolicy::Unassociated
                })
                .with_spot_color_policy(SpotColorPolicy::Render);
                let mut session = if bounded == 0 {
                    decoder
                        .open(&data, request)
                        .unwrap_or_else(|e| panic!("{name}: {e}"))
                } else {
                    open_fragmented(decoder, &data, request)
                };
                let mut actual = Vec::new();
                while let Some(frame) = pollster::block_on(session.next_frame_async()).unwrap() {
                    actual.extend(read(&backend, &frame.output().outputs[0]));
                }
                assert_eq!(actual.len(), expected.len(), "{name}/{channel}");
                // Use the existing precision corpus's native VarDCT tolerance; the additional
                // tight comparison below isolates scalar routing from reconstruction error.
                let limit = if name.contains("vardct") { 0.003 } else { 2e-5 };
                let mut max = 0_f32;
                let mut route_error = 0_f32;
                for (&word, &expected) in actual.iter().zip(&expected) {
                    let actual = f32::from_bits(word);
                    assert!(actual.is_finite() && expected.is_finite());
                    max = max.max((actual - expected).abs() / expected.abs().max(1.0));
                }
                for (&word, pixel) in actual.iter().zip(color_words.as_chunks::<4>().0.iter()) {
                    let color = f32::from_bits(pixel[channel as usize]);
                    assert!(color.is_finite());
                    route_error = route_error
                        .max((f32::from_bits(word) - color).abs() / color.abs().max(1.0));
                }
                assert!(
                    route_error < 5e-6,
                    "{name}/{channel}: scalar vs RGB route error {route_error}"
                );
                assert!(
                    max < limit,
                    "{name}/{channel}/bounded{bounded}: max error {max}"
                );
                if bounded == 0 {
                    eprintln!("{name}/{channel}: native error {max}, scalar vs RGB {route_error}");
                    whole = actual;
                } else {
                    assert_eq!(
                        actual, whole,
                        "{name}/{channel}: bounded input/alpha policy changes words"
                    );
                }
                drop(session);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}

// Independent exact-rational reference: binary32's significand is multiplied in u128 before
// division/rounding. F64 multiplication can itself lose a half-code decision at 31-bit depth.
fn quantize(value: f32, bits: u8) -> u32 {
    let maximum = (1_u128 << bits) - 1;
    if value <= 0.0 {
        return 0;
    }
    if value >= 1.0 {
        return maximum as u32;
    }
    assert!(value.is_finite());
    let word = value.to_bits();
    let exponent = (word >> 23) & 255;
    if exponent == 0 {
        return 0;
    }
    let denominator_bits = 150 - exponent;
    if denominator_bits >= 128 {
        return 0;
    }
    let numerator = u128::from((word & 0x7f_ffff) | 0x80_0000) * maximum;
    ((numerator + (1_u128 << (denominator_bits - 1))) >> denominator_bits) as u32
}

#[test]
fn vardct_native_color_clamps_and_rounds_only_at_the_declared_output_depth() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for name in [
        "integer/vardct_extras_integer_gray",
        "integer/vardct_extras_integer_rgb_extended31",
        "integer/composition_extras_integer_vardct_resampled_extended31",
    ] {
        let (data, inventory) = fixture(name);
        let SampleBitDepth::Integer { bits_per_sample } = inventory.image_header.bit_depth else {
            unreachable!()
        };
        let bits = bits_per_sample as u8;
        let channel = if inventory.image_header.grayscale {
            0
        } else {
            2
        };
        let mut floating = decoder
            .open(
                &data,
                scalar(NumericSampleMapping::NormalizedUnsigned)
                    .with_color_channel(channel)
                    .unwrap(),
            )
            .unwrap();
        let mut expected = Vec::new();
        while let Some(frame) = floating.next_frame().unwrap() {
            expected.extend(
                read(&backend, &frame.output().outputs[0])
                    .into_iter()
                    .map(|word| quantize(f32::from_bits(word), bits)),
            );
        }
        drop(floating);
        let mut session = decoder
            .open(&data, native(bits).with_color_channel(channel).unwrap())
            .unwrap();
        let mut actual = Vec::new();
        while let Some(frame) = session.next_frame().unwrap() {
            actual.extend(unpack(
                &read_bytes(&backend, &frame.output().outputs[0]),
                bits,
            ));
        }
        assert_eq!(actual, expected, "{name}");
        drop(session);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        for wrong in [
            native(if bits == 16 { 8 } else { 16 }),
            scalar(NumericSampleMapping::NativeFloat),
        ] {
            assert!(matches!(
                decoder.open(&data, wrong.with_color_channel(channel).unwrap()),
                Err(Error::UnsupportedOutputFormat(_))
            ));
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn original_rgb_and_jpeg_components_use_reconstructed_color_before_scalar_selection() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoders = decoders(&backend);
    for name in [
        "noise/vardct_rgb_float32_up4",
        "noise/vardct_rgb_gray16",
        "noise/vardct_rgb_frames",
        "noise/jpeg_422",
        "noise/jpeg_gray",
    ] {
        let (data, inventory) = fixture(name);
        let image = &inventory.image_header;
        assert!(!image.xyb_encoded);
        let Some(reference) = oracle::libjxl_output(&data, &["--preserve-alpha"]) else {
            eprintln!("skipping original-color numeric oracle: libjxl is unavailable");
            return;
        };
        let pixels = image.width as usize * image.height as usize;
        let frames = reference.chunks_exact(pixels * (4 + image.extra_channels.len()));
        assert!(frames.remainder().is_empty());
        let color: Vec<_> = frames
            .flat_map(|frame| frame[..pixels * 4].iter().copied())
            .collect();
        let mapping = match image.bit_depth {
            SampleBitDepth::Float { .. } => NumericSampleMapping::NativeFloat,
            SampleBitDepth::Integer { .. } => NumericSampleMapping::NormalizedUnsigned,
        };
        for channel in 0..if image.grayscale { 1 } else { 3 } {
            let request = scalar(mapping).with_color_channel(channel).unwrap();
            let mut whole = Vec::new();
            for (bounded, decoder) in decoders.iter().enumerate() {
                let mut session = if bounded == 0 {
                    decoder.open(&data, request.clone()).unwrap()
                } else {
                    open_fragmented(decoder, &data, request.clone())
                };
                let mut actual = Vec::new();
                while let Some(frame) = session.next_frame().unwrap() {
                    actual.extend(read(&backend, &frame.output().outputs[0]));
                }
                assert_eq!(actual.len() * 4, color.len());
                let mut max = 0_f32;
                for (&actual, pixel) in actual.iter().zip(color.as_chunks::<4>().0.iter()) {
                    let actual = f32::from_bits(actual);
                    let expected = pixel[channel as usize];
                    assert!(actual.is_finite() && expected.is_finite());
                    max = max.max((actual - expected).abs() / expected.abs().max(1.0));
                }
                assert!(max < 1e-4, "{name}/{channel}: {max}");
                if bounded == 0 {
                    whole = actual;
                } else {
                    assert_eq!(actual, whole);
                }
                drop(session);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}
