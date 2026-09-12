#![cfg(not(target_arch = "wasm32"))]

#[path = "support/patches.rs"]
mod fixtures;
#[allow(dead_code)]
#[path = "../examples/support/offline/hex.rs"]
mod hex;
#[path = "patches/lf.rs"]
mod lf;
#[path = "patches/lf_producers.rs"]
mod lf_producers;
#[path = "common/extra_channel_oracle.rs"]
#[allow(dead_code)]
mod oracle;
#[path = "support/planes.rs"]
mod planes;
#[path = "patches/progression.rs"]
mod progression;

use jxl_gpu_formats::{PixelFormat, RgbChannelOrder};
use jxl_wgpu::{WgpuBackend, WgpuBackendConfig};
use jxl_wgpu_decode::{
    AlphaOutputPolicy, Error, GpuDecoder, GpuOutputRequest, NumericSampleMapping, SpotColorPolicy,
    WgpuDecodeEngine,
};
use std::num::NonZeroU64;

fn encoded(name: &str) -> Vec<u8> {
    hex::unhex(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("test-data/patches/{name}.jxl.hex")),
        )
        .unwrap(),
    )
}
fn reference(name: &str) -> Vec<f32> {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("test-data/patches/{name}.f32.hex")),
    )
    .unwrap()
    .split_whitespace()
    .map(|word| f32::from_bits(u32::from_str_radix(word, 16).unwrap()))
    .collect()
}
fn request() -> GpuOutputRequest {
    GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
    .with_spot_color_policy(SpotColorPolicy::Preserve)
}

#[test]
fn patch_linear_color_matches_native_without_extended_srgb_approximations() {
    let backend = backend();
    for name in ["xyb_modular", "xyb_vardct", "float_vardct"] {
        let expected = reference(&format!("{name}.linear"));
        let mut format = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
        if let jxl_gpu_formats::ColorSpecification::Defined(ref mut color) = format {
            color.transfer = jxl_gpu_formats::TransferFunction::Linear;
        }
        let req =
            GpuOutputRequest::color(PixelFormat::rgb_f32(RgbChannelOrder::Rgba, false, format))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                .with_spot_color_policy(SpotColorPolicy::Preserve);
        let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
        let mut session = decoder.open(&encoded(name), req).unwrap();
        let frame = session.next_frame().unwrap().unwrap();
        let words = planes::read(&backend, &frame.output().outputs[0]);
        let worst = words
            .iter()
            .zip(&expected)
            .enumerate()
            .map(|(i, (&a, &b))| {
                let a = f32::from_bits(a);
                (i, a, b, (a - b).abs() / (1.0 + b.abs()))
            })
            .max_by(|a, b| a.3.total_cmp(&b.3))
            .unwrap();
        eprintln!("linear {name}: {worst:?}");
        assert!(worst.3 < if name == "xyb_modular" { 1e-4 } else { 0.003 });
    }
}
fn backend() -> WgpuBackend {
    pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
        enable_timestamps: false,
        ..Default::default()
    }))
    .unwrap()
}

#[test]
fn patch_color_matches_native_before_transform_references() {
    let backend = backend();
    for name in [
        "modular",
        "gray",
        "alpha",
        "xyb_modular",
        "xyb_vardct",
        "float",
        "float_vardct",
        "associated",
        "associated_vardct",
    ] {
        for suffix in ["_empty", ""] {
            let name = format!("{name}{suffix}");
            let data = encoded(&name);
            let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            let expected = if inventory.image_header.xyb_encoded {
                // Native fast sRGB is an approximation outside its nominal cube. Compare its
                // independently decoded linear samples through the analytic extended OETF.
                let mut values = reference(&format!("{name}.linear"));
                let color_words = inventory.image_header.width as usize
                    * inventory.image_header.height as usize
                    * 4;
                for (index, value) in values[..color_words].iter_mut().enumerate() {
                    if index % 4 != 3 {
                        let v = f64::from(*value);
                        *value = (v.signum()
                            * if v.abs() <= 0.0031308 {
                                v.abs() * 12.92
                            } else {
                                1.055 * v.abs().powf(1.0 / 2.4) - 0.055
                            }) as f32;
                    }
                }
                values
            } else {
                reference(&name)
            };
            let mut prior = None;
            for limit in [None, NonZeroU64::new(256)] {
                let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                if let Some(limit) = limit {
                    engine = engine.with_stream_window_limit(limit);
                }
                let decoder = GpuDecoder::new(engine);
                let mut session = if limit.is_some() {
                    planes::open_fragmented(&decoder, &data, request())
                } else {
                    decoder.open(&data, request()).unwrap()
                };
                let frame = if limit.is_some() {
                    pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap()
                } else {
                    session.next_frame().unwrap().unwrap()
                };
                let actual = planes::read(&backend, &frame.output().outputs[0]);
                assert_eq!(
                    actual.len(),
                    inventory.image_header.width as usize
                        * inventory.image_header.height as usize
                        * 4
                );
                let mut max_error = 0f32;
                for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                    let actual = f32::from_bits(actual);
                    let error = (actual - expected).abs() / (1.0 + expected.abs());
                    max_error = max_error.max(error);
                    let tolerance = if index % 4 == 3 {
                        2e-6
                    } else if name.contains("vardct") {
                        0.003
                    } else {
                        1e-4
                    };
                    assert!(
                        actual.is_finite() && error <= tolerance,
                        "{name}/{index}: {actual} vs {expected}, error {error}"
                    );
                }
                if let Some(prior) = &prior {
                    assert_eq!(&actual, prior, "bounded {name}");
                }
                eprintln!("{name}, window {limit:?}, max normalized error {max_error}");
                prior = Some(actual);
                assert!(session.next_frame().unwrap().is_none());
            }
        }
    }
}

fn source(name: &str) -> Vec<u8> {
    hex::unhex(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("test-data/{name}.jxl.hex")),
        )
        .unwrap(),
    )
}

fn inventory(bytes: &[u8]) -> jxl_gpu_bitstream::CodestreamInventory {
    jxl_gpu_bitstream::parse(bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
}

#[test]
fn patch_extra_channels_match_native_including_color_alpha_override() {
    let backend = backend();
    for name in [
        "alpha",
        "xyb_modular",
        "xyb_vardct",
        "float",
        "float_vardct",
        "associated",
        "associated_vardct",
    ] {
        let data = encoded(name);
        let image = inventory(&data).image_header;
        let pixels = image.width as usize * image.height as usize;
        let expected = reference(name);
        assert_eq!(expected.len(), pixels * (4 + image.extra_channels.len()));
        for (index, extra) in image.extra_channels.iter().enumerate() {
            let mapping = match extra.bit_depth {
                jxl_gpu_bitstream::SampleBitDepth::Integer { .. } => {
                    NumericSampleMapping::NormalizedUnsigned
                }
                jxl_gpu_bitstream::SampleBitDepth::Float { .. } => {
                    NumericSampleMapping::NativeFloat
                }
            };
            let request = GpuOutputRequest::numeric(
                PixelFormat::non_color(
                    jxl_gpu_formats::SampleKind::Float,
                    32,
                    &[jxl_gpu_formats::Channel::X],
                ),
                mapping,
            )
            .unwrap()
            .with_extra_channel(index as u32)
            .unwrap();
            let mut previous = None;
            for limit in [None, NonZeroU64::new(256)] {
                let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                if let Some(limit) = limit {
                    engine = engine.with_stream_window_limit(limit);
                }
                let decoder = GpuDecoder::new(engine);
                let mut session = planes::open_fragmented(&decoder, &data, request.clone());
                let frame = pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap();
                let actual = planes::read(&backend, &frame.output().outputs[0]);
                assert_eq!(actual.len(), pixels);
                for (&actual, &expected) in actual.iter().zip(&expected[(4 + index) * pixels..]) {
                    let actual = f32::from_bits(actual);
                    assert!(
                        actual.is_finite()
                            && (actual - expected).abs() <= 2e-6 * (1.0 + expected.abs()),
                        "{name}/extra{index}: {actual} vs {expected}"
                    );
                }
                if let Some(previous) = &previous {
                    assert_eq!(&actual, previous);
                }
                previous = Some(actual);
                drop((frame, session));
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}

#[test]
fn patch_batches_and_dictionary_continuations_preserve_order() {
    let backend = backend();
    let original = source("testsrc_modular_orientation_rgb_1");
    let original_inventory = inventory(&original);
    let frame = &original_inventory.frames[0];
    // More than one render batch and more than 4096 parser steps. Only replacement and no-op
    // keep repeated overlaps finite; alternating rectangles and offsets still make order matter.
    let count = 1100;
    let mut values = vec![1, 3, 0, 0, 2, 2, count - 1];
    let mut previous = [0i32; 2];
    for i in 0..count {
        let position = [
            (i * 3 % (frame.width - 2)) as i32,
            (i * 2 % (frame.height - 2)) as i32,
        ];
        for axis in 0..2 {
            let delta = position[axis] - previous[axis];
            values.push(if i == 0 {
                position[axis] as u32
            } else {
                ((delta << 1) ^ (delta >> 31)) as u32
            });
        }
        values.push(i % 2);
        previous = position;
    }
    let data = fixtures::assemble(&original, &values);
    let expected = oracle::libjxl_output(&data, &["--preserve-alpha"]);
    let mut previous = None;
    for limit in [None, NonZeroU64::new(40)] {
        let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        if let Some(limit) = limit {
            engine = engine.with_stream_window_limit(limit);
        }
        let decoder = GpuDecoder::new(engine);
        let mut session = planes::open_fragmented(&decoder, &data, request());
        let frame = pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap();
        let actual = planes::read(&backend, &frame.output().outputs[0]);
        if let Some(expected) = &expected {
            assert_eq!(actual.len(), expected.len());
            for (&actual, &expected) in actual.iter().zip(expected) {
                assert!((f32::from_bits(actual) - expected).abs() < 1e-6);
            }
        }
        if let Some(previous) = &previous {
            assert_eq!(&actual, previous);
        }
        previous = Some(actual);
        drop((frame, session));
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn invalid_patch_dictionaries_fail_gpu_validation_and_release_allocations() {
    let backend = backend();
    let original = source("testsrc_modular_orientation_rgb_1");
    let frame = inventory(&original).frames.remove(0);
    let valid = fixtures::values(&frame, 0, 1);
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for (index, value, code) in [
        (0, 1025 + frame.width * frame.height / 4, 12),
        (1, 4, 13),        // Invalid slot.
        (1, 2, 13),        // Missing reference.
        (2, u32::MAX, 14), // Source X outside the reference.
        (4, u32::MAX, 14), // Width plus one overflows.
        (5, frame.height, 14),
        (6, u32::MAX, 12),    // Position count plus one overflows.
        (7, frame.width, 14), // Destination X outside the frame.
        (8, frame.height, 14),
        (9, 8, 15), // Unknown blend mode.
    ] {
        let mut values = valid.clone();
        values[index] = value;
        let data = fixtures::assemble(&original, &values);
        let mut session = decoder.open(&data, request()).unwrap();
        let error = match pollster::block_on(session.next_frame_async()) {
            Ok(_) => panic!("accepted invalid patch field {index} = {value}"),
            Err(error) => error,
        };
        assert!(
            matches!(error, Error::PatchDictionary { code: actual } if actual == code),
            "field{index}={value}: {error:?}"
        );
        drop(session);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn patches_use_all_four_reference_slots_and_the_latest_completed_version() {
    let backend = backend();
    let original = source("testsrc_modular_orientation_rgb_1");
    let frame = inventory(&original).frames.remove(0);
    let dictionaries: Vec<_> = (0..4)
        .map(|slot| {
            let mut values = fixtures::values(&frame, 0, 16);
            values[1] = slot;
            values
        })
        .collect();
    let data = fixtures::assemble_frames(
        &original,
        &[
            (Some((0, true)), None),
            (Some((1, true)), Some(&dictionaries[0])),
            (Some((2, true)), Some(&dictionaries[1])),
            (Some((3, true)), Some(&dictionaries[2])),
            (Some((0, true)), Some(&dictionaries[3])),
            (None, Some(&dictionaries[0])),
        ],
    );
    let expected = oracle::libjxl_output(&data, &["--preserve-alpha"]);
    let mut previous = None;
    for limit in [None, NonZeroU64::new(40)] {
        let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        if let Some(limit) = limit {
            engine = engine.with_stream_window_limit(limit);
        }
        let decoder = GpuDecoder::new(engine);
        let mut session = planes::open_fragmented(&decoder, &data, request());
        let frame = pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap();
        let actual = planes::read(&backend, &frame.output().outputs[0]);
        if let Some(expected) = &expected {
            assert_eq!(actual.len(), expected.len());
            for (&actual, &expected) in actual.iter().zip(expected) {
                assert!((f32::from_bits(actual) - expected).abs() <= 2e-6 * (1.0 + expected.abs()));
            }
        }
        if let Some(previous) = &previous {
            assert_eq!(&actual, previous);
        }
        previous = Some(actual);
        assert!(session.next_frame().unwrap().is_none());
        drop((frame, session));
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
    let data = fixtures::assemble_frames(
        &original,
        &[(Some((0, false)), None), (None, Some(&dictionaries[0]))],
    );
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let mut session = decoder.open(&data, request()).unwrap();
    assert!(matches!(
        session.next_frame(),
        Err(Error::PatchDictionary { code: 13 })
    ));
    drop(session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}
