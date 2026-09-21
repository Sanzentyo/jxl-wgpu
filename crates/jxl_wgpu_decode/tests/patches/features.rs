use super::progression::{color_request, compare, drain, srgb};
use super::*;
use jxl_wgpu_decode::{FrameProgression, OrientationPolicy};

use jxl_test_support::fixtures::patch_features as corpus;
use jxl_test_support::fixtures::patch_references::ColorErrorScale;

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
    let decoders = decoders(&backend, NonZeroU64::new(256).unwrap());
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
            check_image(
                &backend,
                &decoders,
                &name,
                &data,
                ImageReferences {
                    linear: ImageReference {
                        samples: &reference(&name),
                        tolerance: 1.0 / 1024.0,
                    },
                    srgb: None,
                    scale: ColorErrorScale::Component,
                },
            );
        }
    }
}

pub(super) struct ImageReference<'a> {
    pub samples: &'a [f32],
    pub tolerance: f32,
}

pub(super) struct ImageReferences<'a> {
    pub linear: ImageReference<'a>,
    pub srgb: Option<ImageReference<'a>>,
    pub scale: ColorErrorScale,
}

fn compare_color(
    actual: &[u32],
    expected: &[f32],
    tolerance: f32,
    scale: ColorErrorScale,
    label: &str,
) {
    if scale == ColorErrorScale::Component {
        return compare(actual, expected, tolerance, label);
    }
    assert_eq!(actual.len(), expected.len(), "{label}");
    assert!(actual.len().is_multiple_of(4));
    let mut maximum = 0f32;
    for (pixel, (actual, expected)) in actual
        .as_chunks::<4>()
        .0
        .iter()
        .zip(expected.as_chunks::<4>().0)
        .enumerate()
    {
        let rgb_scale = 1.0 + expected[..3].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for channel in 0..4 {
            let value = f32::from_bits(actual[channel]);
            let error = (value - expected[channel]).abs()
                / if channel == 3 {
                    1.0 + expected[channel].abs()
                } else {
                    rgb_scale
                };
            let bound = if channel == 3 { 2e-6 } else { tolerance };
            assert!(
                value.is_finite() && expected[channel].is_finite() && error <= bound,
                "{label}/pixel{pixel}/channel{channel}: {value} vs {}, error {error}",
                expected[channel]
            );
            maximum = maximum.max(error);
        }
    }
    eprintln!("{label}: max RGB-vector/independent-alpha normalized error {maximum}");
}

pub(super) fn check_image(
    backend: &WgpuBackend,
    decoders: &[GpuDecoder<WgpuDecodeEngine>; 2],
    name: &str,
    data: &[u8],
    references: ImageReferences<'_>,
) {
    let info = inventory(data);
    eprintln!("patch feature {name}");
    let pixels = info.image_header.width as usize * info.image_header.height as usize;
    assert_eq!(
        references.linear.samples.len(),
        pixels * (4 + info.image_header.extra_channels.len())
    );
    if let Some(srgb) = &references.srgb {
        assert_eq!(srgb.samples.len(), references.linear.samples.len());
        assert_eq!(
            &srgb.samples[pixels * 4..],
            &references.linear.samples[pixels * 4..]
        );
    }
    let mut linear_words: Option<Vec<u32>> = None;
    for linear in [true, false] {
        let mut expected = references.linear.samples[..pixels * 4].to_vec();
        if !linear {
            if let Some(srgb) = &references.srgb {
                expected.copy_from_slice(&srgb.samples[..pixels * 4]);
            } else {
                // Reference IDCT rounding is amplified near black by the nonlinear transfer.
                // Check accuracy against the reference in linear light, then independently check
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
        }
        let mut baseline = None;
        for (limit, decoder) in [None, NonZeroU64::new(256)].into_iter().zip(decoders) {
            let request = color_request(linear).with_orientation_policy(OrientationPolicy::Keep);
            let mut session = if limit.is_some() {
                planes::open_fragmented(decoder, data, request.clone())
            } else {
                decoder.open(data, request.clone()).unwrap()
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
                        .find(|frame| frame.frame_index == progression.physical_frame_index())
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
                let words = planes::read(backend, &update.output().outputs[0]);
                assert!(words.iter().all(|word| f32::from_bits(*word).is_finite()));
                snapshots.push((update.progression(), words));
                images.push(update);
            }
            assert!(snapshots.last().unwrap().0.is_none());
            let tolerance = if linear {
                references.linear.tolerance
            } else {
                references
                    .srgb
                    .as_ref()
                    .map_or(2e-6, |reference| reference.tolerance)
            };
            compare_color(
                &snapshots.last().unwrap().1,
                &expected,
                tolerance,
                references.scale,
                &format!("{name}/linear{linear}/{limit:?}"),
            );
            if linear && linear_words.is_none() {
                linear_words = Some(snapshots.last().unwrap().1.clone());
            }
            if let Some(baseline) = &baseline {
                assert_eq!(&snapshots, baseline);
            }
            let mut final_only = decoder
                .open(data, request.with_progressive_output(false))
                .unwrap();
            let final_image = final_only.next_frame().unwrap().unwrap();
            assert_eq!(
                planes::read(backend, &final_image.output().outputs[0]),
                snapshots.last().unwrap().1
            );
            for (image, (_, words)) in images.iter().zip(&snapshots) {
                assert_eq!(image.metadata, final_image.metadata);
                assert_eq!(&planes::read(backend, &image.output().outputs[0]), words);
            }
            baseline = Some(snapshots);
            drop((images, session, final_only, final_image));
            drain(backend, 0);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
    let decoder = &decoders[1];
    for (channel, extra) in info.image_header.extra_channels.iter().enumerate() {
        let mapping = match extra.bit_depth {
            jxl_gpu_bitstream::SampleBitDepth::Integer { .. } => {
                NumericSampleMapping::NormalizedUnsigned
            }
            jxl_gpu_bitstream::SampleBitDepth::Float { .. } => NumericSampleMapping::NativeFloat,
        };
        let request = GpuOutputRequest::numeric(
            jxl_gpu_formats::vpi::VpiPitchLinearFormat::F32.pixel_format(),
            mapping,
        )
        .unwrap()
        .with_extra_channel(channel as u32)
        .unwrap()
        .with_orientation_policy(OrientationPolicy::Keep);
        let mut session = planes::open_fragmented(decoder, data, request);
        let image = pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap();
        let offset = (4 + channel) * pixels;
        compare(
            &planes::read(backend, &image.output().outputs[0]),
            &references.linear.samples[offset..offset + pixels],
            2e-6,
            &format!("{name}/extra{channel}"),
        );
        drop((image, session));
        drain(backend, 0);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
    }
}

#[test]
fn patch_extra_resampling_mismatch_is_malformed_before_gpu_admission() {
    let backend = backend();
    let decoders = decoders(&backend, NonZeroU64::new(256).unwrap());
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
        for decoder in &decoders {
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
