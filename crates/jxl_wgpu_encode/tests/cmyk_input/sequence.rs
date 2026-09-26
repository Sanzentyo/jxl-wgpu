use super::*;
use jxl_test_support::{
    gpu::planes::{open_fragmented, read_bytes},
    oracles::{extra_channels, icc_profile::IccProfileOracle},
};
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, NumericSampleMapping, WgpuDecodeEngine};
use std::num::NonZeroU64;

fn numeric(channel: u32) -> GpuOutputRequest {
    let request = GpuOutputRequest::numeric(
        SamplePrecision::float(32, 8).unwrap().pixel_format(),
        NumericSampleMapping::NormalizedUnsigned,
    )
    .unwrap();
    if channel < 3 {
        request.with_color_channel(channel)
    } else {
        request.with_extra_channel(channel - 3)
    }
    .unwrap()
}

#[test]
fn modular_cmyk_descriptor_resolves_color_black_and_alpha_sampling_together() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let profile = profile("lut8_lab_4");
    let config = config(&profile, 8, 0, true);
    let displayed = Extent2d::new(35, 19);
    let encoder = LosslessModularEncoder::new(context.clone())
        .with_image_options(config.image_options)
        .unwrap();
    for factor in [
        UpsamplingFactor::One,
        UpsamplingFactor::Two,
        UpsamplingFactor::Four,
        UpsamplingFactor::Eight,
    ] {
        let extent = factor.source_extent(displayed);
        let words: Vec<_> = (0..extent.area().unwrap() * 5)
            .map(|v| v as u32 * 19 % 256)
            .collect();
        let source = input(
            &context,
            extent,
            &config,
            Storage::Split,
            CmykSampleEncoding::InkAmounts,
            &words,
        );
        let mut sequence = encoder
            .begin_sequence(
                LosslessModularSequenceDescriptor::from_pixel_format(
                    displayed.width,
                    displayed.height,
                    &config.pixel_format(),
                    AnimationHeader::Still,
                )
                .unwrap(),
            )
            .unwrap();
        let frame = sequence
            .submit_last_frame(
                source,
                FrameOptions {
                    upsampling: factor,
                    extra_channel_upsampling: vec![factor; 2],
                    ..Default::default()
                },
            )
            .unwrap()
            .wait()
            .unwrap();
        sequence.insert(frame).unwrap();
        let bytes = sequence.finish_raw().unwrap();
        assert_eq!(
            modular_words::channel_frames(&bytes)[0],
            expected(extent, &words, 8, true, CmykSampleEncoding::InkAmounts)
        );
        let headers = modular_words::sampling_headers(&bytes);
        assert_eq!(headers[0].extras, vec![factor.factor(); 2]);
        assert_eq!(headers[0].coded, [extent.width, extent.height]);
    }
}

#[test]
fn cmyk_mixed_sampling_crop_and_independent_black_references_match_native() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(gpu.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    let profile = profile("lut16_xyz_4");
    let profiles = IccProfileOracle::compile();
    let extent = Extent2d::new(33, 19);
    let depth = ExtraChannel::new(
        ExtraChannelKind::Depth,
        SamplePrecision::integer(13).unwrap(),
        1,
        b"depth".to_vec(),
    )
    .unwrap();
    let mut config = config(&profile, 8, 0, true);
    config.extra_channels = vec![depth.clone()];
    for reverse in [false, true] {
        config.alpha = Some(if reverse {
            AlphaAssociation::Associated
        } else {
            AlphaAssociation::Unassociated
        });
        let convention = if reverse {
            CmykSampleEncoding::Complemented
        } else {
            CmykSampleEncoding::InkAmounts
        };
        let encoder = MixedModeEncoder::new(
            context.clone(),
            MixedModeConfig {
                vardct: config.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        for mode in [BlendMode::Replace, BlendMode::Blend] {
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
            let mut physical = Vec::new();
            for index in 0..3 {
                let displayed = if index == 2 {
                    Extent2d::new(29, 17)
                } else {
                    extent
                };
                let factor = if index == 1 {
                    UpsamplingFactor::Two
                } else {
                    UpsamplingFactor::One
                };
                let coded = factor.source_extent(displayed);
                let words: Vec<_> = (0..coded.area().unwrap() * 5)
                    .map(|v| (v as u32 * 7 + index * 31) % 256)
                    .collect();
                let scalar_extent =
                    depth.source_extent_with_upsampling(displayed, UpsamplingFactor::Four);
                let scalar: Vec<_> = (0..scalar_extent.area().unwrap())
                    .map(|v| v as u32 * 37 % 8192)
                    .collect();
                let source = input(
                    &context,
                    coded,
                    &config,
                    Storage::Planar,
                    convention,
                    &words,
                )
                .with_extra_channels(vec![raw_source(
                    &context,
                    scalar_extent,
                    depth.precision().pixel_format(),
                    Storage::Packed,
                    &scalar,
                )])
                .unwrap();
                let mut expected = expected(coded, &words, 8, true, convention);
                expected.push(modular_integer::ExtraWords {
                    width: scalar_extent.width,
                    height: scalar_extent.height,
                    words: scalar,
                });
                let modular = (index != 1) != reverse;
                let encoding = if modular {
                    MixedModeFrameEncoding::Modular
                } else {
                    MixedModeFrameEncoding::VarDct
                };
                physical.push((modular, expected));
                let color = FrameBlend {
                    mode,
                    source_reference: ReferenceSlot::new(if index == 1 { 1 } else { 3 }).unwrap(),
                    alpha_channel: 0,
                    clamp: mode == BlendMode::Blend,
                };
                let black = FrameBlend {
                    source_reference: ReferenceSlot::new(1).unwrap(),
                    ..color
                };
                let options = if index == 0 {
                    FrameOptions {
                        kind: FrameKind::ReferenceOnly,
                        save_as_reference: ReferenceSlot::new(1).unwrap(),
                        extra_channel_upsampling: vec![factor, factor, UpsamplingFactor::Four],
                        ..Default::default()
                    }
                } else {
                    FrameOptions {
                        upsampling: factor,
                        extra_channel_upsampling: vec![factor, factor, UpsamplingFactor::Four],
                        crop: (index == 2).then(|| FrameCrop::new(3, -1, 29, 17).unwrap()),
                        color_blend: color,
                        extra_channel_blends: vec![color, black, color],
                        timing: FrameTiming {
                            duration_ticks: 3,
                            timecode: Some(100 + index),
                        },
                        save_as_reference: ReferenceSlot::new(if index == 1 { 3 } else { 0 })
                            .unwrap(),
                        ..Default::default()
                    }
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
            assert_eq!(profiles.read(&bytes).profile, profile.bytes().as_ref());
            for (index, (modular, reference)) in physical.iter().enumerate() {
                if *modular {
                    assert_eq!(
                        &modular_integer::modular_channel_words(&bytes, index),
                        reference
                    );
                } else {
                    assert_eq!(
                        modular_integer::vardct_extra_words(&bytes, index),
                        reference[3..]
                    );
                }
            }
            let headers = modular_words::sampling_headers(&bytes);
            assert_eq!(headers[1].extras, vec![2, 2, 8]);
            let native = extra_channels::libjxl_output(
                &bytes,
                &["--original-icc", "--no-cms", "--preserve-alpha"],
            )
            .unwrap();
            let pixels = extent.area().unwrap();
            assert_eq!(native.len(), 2 * pixels * 7);
            for channel in 0..6 {
                let mut whole = Vec::new();
                for fragmented in [false, true] {
                    let mut session = if fragmented {
                        open_fragmented(&decoder, &bytes, numeric(channel))
                    } else {
                        decoder.open(&bytes, numeric(channel)).unwrap()
                    };
                    for frame_index in 0..2 {
                        let frame = session.next_frame().unwrap().unwrap();
                        let raw = read_bytes(&gpu, &frame.output().outputs[0]);
                        if fragmented {
                            assert_eq!(raw, whole[frame_index]);
                        } else {
                            whole.push(raw.clone());
                        }
                        let actual = extra_channels::floats(&raw);
                        assert_eq!(actual.len(), pixels);
                        for (i, &value) in actual.iter().enumerate() {
                            let offset = if channel < 3 {
                                i * 4 + channel as usize
                            } else {
                                (channel as usize + 1) * pixels + i
                            };
                            let expected = native[frame_index * pixels * 7 + offset];
                            let tolerance = if channel < 3 { 2e-4 } else { 2e-6 };
                            assert!(
                                (value - expected).abs() <= tolerance,
                                "{reverse}/{mode:?}/{channel}/{frame_index}/{i}: {value} vs {expected}"
                            );
                        }
                    }
                    assert!(session.next_frame().unwrap().is_none());
                }
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn cmyk_previews_retain_the_shared_profile_alpha_and_black_contract() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoder = GpuDecoder::wgpu(gpu.clone()).unwrap();
    let profile = profile("ab_xyz_4");
    let config = config(&profile, 8, 0, true);
    let extent = Extent2d::new(9, 5);
    let words: Vec<_> = (0..extent.area().unwrap() * 5)
        .map(|v| v as u32 * 11 % 256)
        .collect();
    let encoder = MixedModeEncoder::new(
        context.clone(),
        MixedModeConfig {
            vardct: config.clone(),
            ..Default::default()
        },
    )
    .unwrap();
    for codec in [
        MixedModeFrameEncoding::Modular,
        MixedModeFrameEncoding::VarDct,
    ] {
        let mut sequence = encoder
            .begin_sequence(
                ImageSequenceDescriptor::new(extent.width, extent.height, AnimationHeader::Still)
                    .unwrap()
                    .with_preview(PreviewSize::new(extent.width, extent.height).unwrap()),
            )
            .unwrap();
        let source = input(
            &context,
            extent,
            &config,
            Storage::Planar,
            CmykSampleEncoding::InkAmounts,
            &words,
        );
        let baseline = context.memory_stats().reserved_bytes;
        let preview = sequence
            .submit_preview(source.clone(), codec, FrameOptions::default())
            .unwrap()
            .wait()
            .unwrap();
        assert!(preview.reserved_bytes() > 0);
        assert_eq!(
            context.memory_stats().reserved_bytes,
            baseline + preview.reserved_bytes()
        );
        sequence.insert_preview(preview).unwrap();
        let main = sequence
            .submit_last_frame(
                source,
                MixedModeFrameEncoding::Modular,
                FrameOptions::default(),
            )
            .unwrap()
            .wait()
            .unwrap();
        sequence.insert(main).unwrap();
        let bytes = sequence.finish_raw().unwrap();
        let native = extra_channels::libjxl_output(
            &bytes,
            &[
                "--preview",
                "--original-icc",
                "--no-cms",
                "--preserve-alpha",
            ],
        )
        .unwrap();
        assert_eq!(native.len(), extent.area().unwrap() * 4);
        for channel in 0..5 {
            let request =
                numeric(channel).with_image_selection(jxl_wgpu_decode::ImageSelection::Preview);
            let mut session = open_fragmented(&decoder, &bytes, request);
            let frame = session.next_frame().unwrap().unwrap();
            let actual = extra_channels::floats(&read_bytes(&gpu, &frame.output().outputs[0]));
            for (i, &value) in actual.iter().enumerate() {
                let expected = if channel == 4 {
                    (255 - words[i * 5 + 3]) as f32 / 255.0
                } else {
                    native[i * 4 + channel as usize]
                };
                assert!((value - expected).abs() <= if channel < 3 { 2e-4 } else { 2e-6 });
            }
            assert!(session.next_frame().unwrap().is_none());
        }
        assert_eq!(context.memory_stats().reserved_bytes, 0);
    }
}
