use super::*;
use jxl_test_support::gpu::planes::{open_fragmented, read_bytes};
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, NumericSampleMapping, WgpuDecodeEngine,
};

#[test]
fn mixed_frames_keep_packed_associated_alpha_before_independent_scalars() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoder = GpuDecoder::wgpu(gpu.clone()).unwrap();
    let extent = Extent2d::new(17, 13);
    let definitions = vec![definition(13, 1)];
    let (base, mut expected) = inputs(
        &context,
        extent,
        &definitions,
        extent,
        &[UpsamplingFactor::One],
    );
    let depth = expected.pop().unwrap();
    let values: Vec<_> = (0..extent.width * extent.height * 4)
        .map(|v| v * 13 % 256)
        .collect();
    let source = source(
        &context,
        extent,
        LosslessModularFormat::Rgba.pixel_format(8).unwrap(),
        &values,
    )
    .with_extra_channels(vec![base.extra_channels()[0].clone()])
    .unwrap();
    let mut expected: Vec<_> = (0..4)
        .map(|c| modular_integer::ExtraWords {
            width: extent.width,
            height: extent.height,
            words: values.iter().skip(c).step_by(4).copied().collect(),
        })
        .collect();
    expected.push(depth);
    let encoder = MixedModeEncoder::new(
        context.clone(),
        MixedModeConfig {
            vardct: VarDctConfig {
                alpha: Some(AlphaAssociation::Associated),
                color_transform: VarDctColorTransform::Original,
                extra_channels: definitions,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    let mut sequence = encoder
        .begin_sequence(
            ImageSequenceDescriptor::new(
                extent.width,
                extent.height,
                AnimationHeader::Animation {
                    ticks_per_second_numerator: 1.try_into().unwrap(),
                    ticks_per_second_denominator: 1.try_into().unwrap(),
                    num_loops: 0,
                    have_timecodes: false,
                },
            )
            .unwrap(),
        )
        .unwrap();
    let options = FrameOptions {
        timing: FrameTiming {
            duration_ticks: 1,
            timecode: None,
        },
        ..Default::default()
    };
    let first = sequence
        .submit_frame(
            source.clone(),
            MixedModeFrameEncoding::VarDct,
            options.clone(),
        )
        .unwrap();
    let last = sequence
        .submit_last_frame(source, MixedModeFrameEncoding::Modular, options)
        .unwrap();
    sequence.insert(last.wait().unwrap()).unwrap();
    sequence.insert(first.wait().unwrap()).unwrap();
    let bytes = sequence.finish_raw().unwrap();
    assert_eq!(
        modular_integer::vardct_extra_words(&bytes, 0),
        expected[3..]
    );
    assert_eq!(modular_integer::modular_channel_words(&bytes, 1), expected);
    let native =
        extra_channels::libjxl_output(&bytes, &["--original", "--preserve-alpha"]).unwrap();
    let pixels = (extent.width * extent.height) as usize;
    for selected in 0..2 {
        let request = GpuOutputRequest::numeric(
            SamplePrecision::float(32, 8).unwrap().pixel_format(),
            NumericSampleMapping::NormalizedUnsigned,
        )
        .unwrap()
        .with_extra_channel(selected)
        .unwrap();
        let mut session = open_fragmented(&decoder, &bytes, request);
        for frame_index in 0..2 {
            let frame = session.next_frame().unwrap().unwrap();
            let raw = read_bytes(&gpu, &frame.output().outputs[0]);
            let actual = extra_channels::floats(&raw);
            let start = frame_index * pixels * 6 + (4 + selected as usize) * pixels;
            for (&actual, &expected) in actual.iter().zip(&native[start..start + pixels]) {
                assert!((actual - expected).abs() <= 2e-6);
            }
        }
        assert!(session.next_frame().unwrap().is_none());
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
use std::num::NonZeroU64;

#[test]
fn mixed_sampling_reference_crop_and_blends_match_native_and_fragmented_gpu_output() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(gpu.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    let extent = Extent2d::new(33, 35);
    let definitions = vec![
        ExtraChannel::new(
            ExtraChannelKind::Alpha(AlphaAssociation::Unassociated),
            SamplePrecision::integer(8).unwrap(),
            0,
            b"independent alpha".to_vec(),
        )
        .unwrap(),
        definition(13, 1),
        definition(5, 3),
    ];
    let encoder = MixedModeEncoder::new(
        context.clone(),
        MixedModeConfig {
            vardct: VarDctConfig {
                color_transform: VarDctColorTransform::Original,
                extra_channels: definitions.clone(),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    for mode in [
        BlendMode::Replace,
        BlendMode::Add,
        BlendMode::Blend,
        BlendMode::MultiplyAdd,
        BlendMode::Multiply,
    ] {
        let mut sequence = encoder
            .begin_sequence(
                ImageSequenceDescriptor::new(
                    extent.width,
                    extent.height,
                    AnimationHeader::Animation {
                        ticks_per_second_numerator: 1000.try_into().unwrap(),
                        ticks_per_second_denominator: 1.try_into().unwrap(),
                        num_loops: 0,
                        have_timecodes: true,
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let mut expected = Vec::new();
        for index in 0..3 {
            let displayed = if index == 2 {
                Extent2d::new(29, 31)
            } else {
                extent
            };
            let factor = if index == 1 {
                UpsamplingFactor::Two
            } else {
                UpsamplingFactor::One
            };
            let factors = [factor, UpsamplingFactor::Four, UpsamplingFactor::One];
            let (source, words) = inputs(
                &context,
                factor.source_extent(displayed),
                &definitions,
                displayed,
                &factors,
            );
            expected.push(words);
            let blend = FrameBlend {
                mode,
                source_reference: ReferenceSlot::new(if index == 1 { 1 } else { 3 }).unwrap(),
                alpha_channel: 0,
                clamp: matches!(
                    mode,
                    BlendMode::Blend | BlendMode::MultiplyAdd | BlendMode::Multiply
                ),
            };
            let options = if index == 0 {
                FrameOptions {
                    kind: FrameKind::ReferenceOnly,
                    save_as_reference: ReferenceSlot::new(1).unwrap(),
                    extra_channel_upsampling: factors.to_vec(),
                    ..Default::default()
                }
            } else {
                FrameOptions {
                    upsampling: factor,
                    extra_channel_upsampling: factors.to_vec(),
                    crop: (index == 2).then(|| FrameCrop::new(3, -1, 29, 31).unwrap()),
                    color_blend: blend,
                    extra_channel_blends: vec![blend; 3],
                    timing: FrameTiming {
                        duration_ticks: 3,
                        timecode: Some(100 + index),
                    },
                    save_as_reference: ReferenceSlot::new(if index == 1 { 3 } else { 0 }).unwrap(),
                    ..Default::default()
                }
            };
            let encoding = if index == 1 {
                MixedModeFrameEncoding::VarDct
            } else {
                MixedModeFrameEncoding::Modular
            };
            assert!(
                sequence
                    .memory_plan(&source, encoding, options.clone(), index == 2)
                    .unwrap()
                    .owned_bytes_per_job()
                    > 0
            );
            let job = if index == 2 {
                sequence.submit_last_frame(source, encoding, options)
            } else {
                sequence.submit_frame(source, encoding, options)
            }
            .unwrap();
            sequence.insert(job.wait().unwrap()).unwrap();
        }
        let bytes = sequence.finish_raw().unwrap();
        for (index, expected) in expected.iter().enumerate() {
            if index == 1 {
                assert_eq!(
                    modular_integer::vardct_extra_words(&bytes, index),
                    expected[3..]
                );
            } else {
                assert_eq!(
                    &modular_integer::modular_channel_words(&bytes, index),
                    expected
                );
            }
        }
        let native =
            extra_channels::libjxl_output(&bytes, &["--original", "--preserve-alpha"]).unwrap();
        let pixels = (extent.width * extent.height) as usize;
        let stride = pixels * 7;
        assert_eq!(native.len(), stride * 2);
        for selected in [None, Some(0), Some(1), Some(2)] {
            let request = if let Some(index) = selected {
                GpuOutputRequest::numeric(
                    SamplePrecision::float(32, 8).unwrap().pixel_format(),
                    NumericSampleMapping::NormalizedUnsigned,
                )
                .unwrap()
                .with_extra_channel(index)
                .unwrap()
            } else {
                GpuOutputRequest::color(jxl_gpu_formats::PixelFormat::rgb_f32(
                    jxl_gpu_formats::RgbChannelOrder::Rgba,
                    false,
                    jxl_wgpu_decode::vardct_rgb8_format().color_spec,
                ))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
            };
            let mut whole = Vec::new();
            for fragmented in [false, true] {
                let mut session = if fragmented {
                    open_fragmented(&decoder, &bytes, request.clone())
                } else {
                    decoder.open(&bytes, request.clone()).unwrap()
                };
                for index in 0..2 {
                    let frame = session.next_frame().unwrap().unwrap();
                    assert_eq!(frame.metadata.timecode, Some(101 + index as u32));
                    let raw = read_bytes(&gpu, &frame.output().outputs[0]);
                    if fragmented {
                        assert_eq!(&raw, &whole[index]);
                    } else {
                        whole.push(raw.clone());
                    }
                    let actual = extra_channels::floats(&raw);
                    let (offset, count) = selected.map_or((0, pixels * 4), |channel| {
                        ((4 + channel as usize) * pixels, pixels)
                    });
                    assert_eq!(actual.len(), count);
                    for (sample, (&actual, &expected)) in actual
                        .iter()
                        .zip(&native[index * stride + offset..index * stride + offset + count])
                        .enumerate()
                    {
                        let tolerance = if selected.is_some() { 2e-6 } else { 2e-4 };
                        assert!(
                            (actual - expected).abs() <= tolerance,
                            "{mode:?}/{selected:?}/{index}/{sample}: {actual} vs {expected}"
                        );
                    }
                }
                assert!(session.next_frame().unwrap().is_none());
            }
        }
        assert_eq!(context.memory_stats().reserved_bytes, 0);
    }
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}
