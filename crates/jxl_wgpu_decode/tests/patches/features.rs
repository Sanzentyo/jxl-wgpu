use super::progression::{color_request, compare, drain, srgb};
use super::*;
use jxl_wgpu_decode::{FrameProgression, OrientationPolicy};

use jxl_test_support::fixtures::patch_features as corpus;

#[test]
fn patch_features_match_native_and_preserve_progressive_images() {
    check_features(false);
}

#[test]
fn lf_patch_features_match_native_and_preserve_progressive_images() {
    check_features(true);
}

fn check_features(lf: bool) {
    let backend = backend();
    for family in corpus::FAMILIES.iter().filter(|family| family.lf == lf) {
        let original = inventory(&encoded(&format!("../{}", family.source)));
        let patched = reference(&format!("features/{}", family.name));
        let empty = reference(&format!("features/{}_empty", family.name));
        assert_ne!(patched, empty, "patches must affect {}", family.name);
        if family.zero_noise {
            assert_ne!(
                patched,
                reference(&format!("features/{}_zero", family.name))
            );
        }
        if family.inject_noise {
            assert_ne!(
                patched,
                reference(&format!("features/{}_noise", family.name))
            );
            assert_ne!(
                empty,
                reference(&format!("features/{}_noise_empty", family.name))
            );
        }
        if family.chain {
            assert_ne!(
                patched,
                reference(&format!("features/{}_chain", family.name))
            );
        }
        for suffix in family.suffixes() {
            let name = format!("features/{}{suffix}", family.name);
            let data = encoded(&name);
            let info = inventory(&data);
            assert_eq!(info.image_header, original.image_header);
            assert_eq!(
                info.frames
                    .iter()
                    .any(|frame| frame.flags & 2 != 0 && frame.lf_level != 0),
                family.lf
            );
            eprintln!("patch feature {name}");
            let pixels = info.image_header.width as usize * info.image_header.height as usize;
            let native = reference(&name);
            let mut linear_words: Option<Vec<u32>> = None;
            for linear in [true, false] {
                let mut expected = native[..pixels * 4].to_vec();
                if !linear {
                    // Native IDCT rounding is amplified near black by the nonlinear transfer.
                    // Check accuracy against native in linear light, then independently check
                    // that the sRGB request applies the analytic OETF to those validated values.
                    expected = linear_words
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|&word| f32::from_bits(word))
                        .collect();
                    for (index, value) in expected.iter_mut().enumerate() {
                        if index % 4 != 3 {
                            *value = srgb(*value);
                        }
                    }
                }
                let mut baseline = None;
                for limit in [None, NonZeroU64::new(256)] {
                    let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                    if let Some(limit) = limit {
                        engine = engine.with_stream_window_limit(limit);
                    }
                    let decoder = GpuDecoder::new(engine);
                    let request =
                        color_request(linear).with_orientation_policy(OrientationPolicy::Keep);
                    let mut session = if limit.is_some() {
                        planes::open_fragmented(&decoder, &data, request.clone())
                    } else {
                        decoder.open(&data, request.clone()).unwrap()
                    };
                    let mut images = Vec::new();
                    let mut snapshots = Vec::new();
                    while let Some(update) = if limit.is_some() {
                        pollster::block_on(session.next_update_async()).unwrap()
                    } else {
                        session.next_update().unwrap()
                    } {
                        if let Some(progression) = update.progression() {
                            let physical = info
                                .frames
                                .iter()
                                .find(|frame| {
                                    frame.frame_index == progression.physical_frame_index()
                                })
                                .unwrap();
                            match progression {
                                FrameProgression::LowFrequency { level, .. } => {
                                    assert_eq!(physical.lf_level, u32::from(level))
                                }
                                _ => assert_eq!(
                                    physical.frame_index,
                                    info.frames.last().unwrap().frame_index
                                ),
                            }
                        }
                        let words = planes::read(&backend, &update.output().outputs[0]);
                        assert!(words.iter().all(|word| f32::from_bits(*word).is_finite()));
                        snapshots.push((update.progression(), words));
                        images.push(update);
                    }
                    assert!(snapshots.last().unwrap().0.is_none());
                    let tolerance = if linear { 1.0 / 1024.0 } else { 2e-6 };
                    compare(
                        &snapshots.last().unwrap().1,
                        &expected,
                        tolerance,
                        &format!("{name}/linear{linear}/{limit:?}"),
                    );
                    if linear && linear_words.is_none() {
                        linear_words = Some(snapshots.last().unwrap().1.clone());
                    }
                    if let Some(baseline) = &baseline {
                        assert_eq!(&snapshots, baseline);
                    }
                    let mut final_only = decoder
                        .open(&data, request.with_progressive_output(false))
                        .unwrap();
                    let final_image = final_only.next_frame().unwrap().unwrap();
                    assert_eq!(
                        planes::read(&backend, &final_image.output().outputs[0]),
                        snapshots.last().unwrap().1
                    );
                    for (image, (_, words)) in images.iter().zip(&snapshots) {
                        assert_eq!(image.metadata, final_image.metadata);
                        assert_eq!(&planes::read(&backend, &image.output().outputs[0]), words);
                    }
                    baseline = Some(snapshots);
                    drop((images, session, final_only, final_image));
                    drain(&backend, 0);
                    assert_eq!(
                        decoder.incremental_input_budget().snapshot().reserved_bytes,
                        0
                    );
                }
            }
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
            );
            for channel in 0..info.image_header.extra_channels.len() {
                let request = GpuOutputRequest::numeric(
                    jxl_gpu_formats::vpi::VpiPitchLinearFormat::F32.pixel_format(),
                    NumericSampleMapping::NormalizedUnsigned,
                )
                .unwrap()
                .with_extra_channel(channel as u32)
                .unwrap()
                .with_orientation_policy(OrientationPolicy::Keep);
                let mut session = planes::open_fragmented(&decoder, &data, request);
                let image = pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap();
                let offset = (4 + channel) * pixels;
                compare(
                    &planes::read(&backend, &image.output().outputs[0]),
                    &native[offset..offset + pixels],
                    2e-6,
                    &format!("{name}/extra{channel}"),
                );
                drop((image, session));
                drain(&backend, 0);
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    0
                );
            }
        }
    }
}

#[test]
fn patch_extra_resampling_mismatch_is_malformed_before_gpu_admission() {
    let backend = backend();
    for (name, lf) in [
        ("extras_resampled_color", false),
        ("vardct_extras_resampled_color", false),
        ("lf_conformance/resampled_associated_modular", true),
        ("lf_conformance/resampled_associated_vardct", true),
    ] {
        let source = encoded(&format!("../{name}"));
        let data = if lf {
            fixtures::assemble_lf_producers(&source, &[Some(&[0])])
        } else {
            fixtures::assemble(&source, &[0])
        };
        let info = inventory(&data);
        let invalid = info
            .frames
            .iter()
            .find(|frame| frame.flags & 2 != 0)
            .unwrap();
        let expected_extra_factor = invalid.extra_channel_upsampling[0];
        assert_ne!(expected_extra_factor, invalid.upsampling);
        for limit in [None, NonZeroU64::new(256)] {
            let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            if let Some(limit) = limit {
                engine = engine.with_stream_window_limit(limit);
            }
            let decoder = GpuDecoder::new(engine);
            let Err(error) = decoder.open(&data, color_request(true)) else {
                panic!("mismatched patch extra resampling was admitted: {name}");
            };
            assert!(
                matches!(error, Error::PatchExtraUpsampling { frame_index, channel: 0, color_factor, extra_factor }
                if frame_index == invalid.frame_index && color_factor == invalid.upsampling && extra_factor == expected_extra_factor),
                "{name}: {error:?}"
            );
            assert_eq!(
                backend.transient_memory_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}

fn inventory(data: &[u8]) -> jxl_gpu_bitstream::CodestreamInventory {
    jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
}
