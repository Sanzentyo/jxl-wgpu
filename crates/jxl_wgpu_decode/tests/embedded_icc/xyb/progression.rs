use super::{corpus, inventory};
use crate::profile;
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction};
use jxl_test_support::{corpus as streams, gpu::planes};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, FrameProgression, GpuDecoder, GpuOutputRequest, OrientationPolicy,
    WgpuDecodeEngine,
};
use std::num::{NonZeroU64, NonZeroUsize};

#[derive(Debug, PartialEq)]
struct Update {
    progression: Option<FrameProgression>,
    complete: bool,
    words: Vec<u32>,
}

fn updates(backend: &WgpuBackend, data: &[u8], limit: Option<NonZeroU64>) -> Vec<Update> {
    let mut color = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
    let ColorSpecification::Defined(ref mut specification) = color else {
        unreachable!()
    };
    specification.transfer = TransferFunction::Linear;
    let format = PixelFormat::rgb_f32(RgbChannelOrder::Rgba, false, color);
    let request = GpuOutputRequest::color(format.clone())
        .unwrap()
        .with_progressive_output(true)
        .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
        .with_orientation_policy(OrientationPolicy::Keep)
        .with_max_frame_slots(NonZeroUsize::new(1).unwrap());
    let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    if let Some(limit) = limit {
        engine = engine.with_stream_window_limit(limit);
    }
    let decoder = GpuDecoder::new(engine);
    let mut session = if limit.is_some() {
        planes::open_fragmented(&decoder, data, request)
    } else {
        decoder.open(data, request).unwrap()
    };
    let mut held = Vec::new();
    while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
        let output = &update.output().outputs[0];
        assert_eq!(output.layout.format, format);
        let result = Update {
            progression: update.progression(),
            complete: update.is_complete(),
            words: planes::read(backend, output),
        };
        held.push((update, result));
    }
    assert_eq!(session.frames_submitted(), 1);
    drop(session);
    let result = held
        .into_iter()
        .map(|(update, result)| {
            assert_eq!(
                planes::read(backend, &update.output().outputs[0]),
                result.words
            );
            result
        })
        .collect();
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
    result
}

#[test]
fn icc_xyb_lf_and_coefficient_previews_preserve_dependency_pixels_and_update_identity() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let mut cases = vec![
        ("rgb", streams::vardct_progressive_dc_ac().to_vec()),
        ("gray", streams::vardct_gray("dc_ac")),
        (
            "custom",
            streams::with_custom_upsampling_weights(streams::vardct_progressive_dc_ac()),
        ),
    ];
    for name in [
        "vardct_gab0",
        "modular_gab1",
        "nested_vardct_gab1",
        "nested_modular_gab1",
    ] {
        let path = corpus::directory()
            .parent()
            .unwrap()
            .join(format!("patches/lf_producers/{name}.jxl.hex"));
        let data = jxl_test_support::offline::unhex(&std::fs::read_to_string(path).unwrap());
        cases.push((name, data));
    }
    for (name, original) in cases {
        let parsed = inventory(&original);
        let mut lf = Vec::new();
        let mut dependency = parsed.frames.last().unwrap().lf_source_frame;
        while let Some(index) = dependency {
            let frame = &parsed.frames[index as usize];
            lf.push((frame.frame_index, frame.lf_level as u8));
            dependency = frame.lf_source_frame;
        }
        lf.reverse();
        assert!(!lf.is_empty());
        let image = parsed.image_header;
        let donor = corpus::cases()
            .find(|case| case.xyb && case.gray == image.grayscale)
            .unwrap();
        let data = profile::replace(&original, &donor.bytes());
        let expected = updates(&backend, &original, None);
        assert_eq!(
            expected
                .iter()
                .filter_map(|update| match update.progression {
                    Some(FrameProgression::LowFrequency {
                        physical_frame_index,
                        level,
                    }) => Some((physical_frame_index, level)),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            lf
        );
        assert_eq!(expected.iter().filter(|u| u.complete).count(), 1);
        assert!(
            expected
                .iter()
                .any(|u| matches!(u.progression, Some(FrameProgression::Coefficients { .. })))
        );
        let mut baseline = None;
        for limit in [None, NonZeroU64::new(40)] {
            let actual = updates(&backend, &data, limit);
            assert_eq!(actual.len(), expected.len());
            for (step, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(actual.progression, expected.progression);
                assert_eq!(actual.complete, expected.complete);
                assert_eq!(actual.words.len(), expected.words.len());
                for (index, (&actual, &expected)) in
                    actual.words.iter().zip(&expected.words).enumerate()
                {
                    if index % 4 == 3 {
                        assert_eq!(actual, expected);
                    } else {
                        let actual = f32::from_bits(actual);
                        let expected = f32::from_bits(expected);
                        assert!(
                            actual.is_finite()
                                && expected.is_finite()
                                && (actual - expected).abs() <= 2e-5,
                            "{name} step {step} sample {index} {limit:?}: {actual} vs {expected}"
                        );
                    }
                }
            }
            if let Some(baseline) = &baseline {
                assert_eq!(&actual, baseline);
            }
            baseline = Some(actual);
        }
    }
}

#[test]
fn icc_xyb_patched_lf_producers_preserve_selected_alpha_and_depth() {
    use jxl_gpu_bitstream::SampleBitDepth;
    use jxl_gpu_formats::{Channel, SampleKind};
    use jxl_wgpu_decode::NumericSampleMapping;
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let donor = corpus::cases()
        .find(|case| case.xyb && !case.gray)
        .unwrap()
        .bytes();
    for name in [
        "vardct_gab0",
        "modular_gab1",
        "nested_vardct_gab1",
        "nested_modular_gab1",
    ] {
        let path = corpus::directory()
            .parent()
            .unwrap()
            .join(format!("patches/lf_producers/{name}.jxl.hex"));
        let original = jxl_test_support::offline::unhex(&std::fs::read_to_string(path).unwrap());
        let parsed = inventory(&original);
        assert!(
            parsed
                .frames
                .iter()
                .any(|frame| frame.lf_level != 0 && frame.flags & 2 != 0)
        );
        assert_eq!(parsed.image_header.extra_channels.len(), 2);
        let data = profile::replace(&original, &donor);
        for (index, extra) in parsed.image_header.extra_channels.iter().enumerate() {
            let request = GpuOutputRequest::numeric(
                PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                match extra.bit_depth {
                    SampleBitDepth::Float { .. } => NumericSampleMapping::NativeFloat,
                    SampleBitDepth::Integer { .. } => NumericSampleMapping::NormalizedUnsigned,
                },
            )
            .unwrap()
            .with_extra_channel(index as u32)
            .unwrap();
            let expected = crate::output::frames(&backend, &original, request.clone(), None);
            for limit in [None, NonZeroU64::new(40)] {
                assert_eq!(
                    crate::output::frames(&backend, &data, request.clone(), limit),
                    expected,
                    "{name} extra {index} {limit:?}"
                );
            }
        }
    }
}
