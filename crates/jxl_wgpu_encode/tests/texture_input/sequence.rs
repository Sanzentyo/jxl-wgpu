use super::*;
use jxl_test_support::{
    gpu::planes::{open_fragmented, read_bytes},
    oracles::extra_channels,
};
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, NumericSampleMapping};

fn inputs(
    context: &WgpuContext,
    displayed: Extent2d,
    factor: UpsamplingFactor,
    definition: &ExtraChannel,
) -> (
    TextureImageSource,
    BufferImageSource,
    TexturePlanesSource,
    Vec<modular_integer::ExtraWords>,
) {
    let extent = factor.source_extent(displayed);
    let config = VarDctConfig {
        alpha: Some(AlphaAssociation::Unassociated),
        ..Default::default()
    };
    let words: Vec<_> = (0..extent.area().unwrap() * 4)
        .map(|i| (i * 17 + 31) as u32 % 256)
        .collect();
    let raw = bytes(&words, 1);
    let scalar_extent = definition.source_extent_with_upsampling(displayed, UpsamplingFactor::Four);
    let scalar: Vec<_> = (0..scalar_extent.area().unwrap())
        .map(|i| (i * 71 + 331) as u32 % 8192)
        .collect();
    let extra = buffer(
        context,
        scalar_extent,
        definition.precision().pixel_format(),
        &bytes(&scalar, 2),
    );
    let input = texture(
        context,
        extent,
        config.pixel_format(),
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &raw,
    )
    .with_extra_channels(vec![extra.clone()])
    .unwrap();
    let planar = separate_planes::split(
        context,
        extent,
        config.pixel_format(),
        &raw,
        &[1, 1, 2],
        true,
    )
    .with_extra_channels(vec![extra.clone()])
    .unwrap();
    let canonical = buffer(context, extent, config.pixel_format(), &raw)
        .with_extra_channels(vec![extra])
        .unwrap();
    let mut expected = planes(extent, &words, 4);
    expected.push(modular_integer::ExtraWords {
        width: scalar_extent.width,
        height: scalar_extent.height,
        words: scalar,
    });
    (input, canonical, planar, expected)
}

#[test]
fn mixed_texture_sampling_crop_references_and_fragmented_decode_match_native() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoder = GpuDecoder::wgpu(gpu.clone()).unwrap();
    let extent = Extent2d::new(33, 19);
    let depth = ExtraChannel::new(
        ExtraChannelKind::Depth,
        SamplePrecision::integer(13).unwrap(),
        1,
        b"depth".to_vec(),
    )
    .unwrap();
    let encoder = MixedModeEncoder::new(
        context.clone(),
        MixedModeConfig {
            vardct: VarDctConfig {
                alpha: Some(AlphaAssociation::Unassociated),
                color_transform: VarDctColorTransform::Original,
                extra_channels: vec![depth.clone()],
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    for reverse in [false, true] {
        let mut encoded = Vec::new();
        let mut expected = Vec::new();
        for storage in 0..3 {
            let mut sequence = encoder
                .begin_sequence(
                    ImageSequenceDescriptor::new(
                        extent.width,
                        extent.height,
                        AnimationHeader::Animation {
                            ticks_per_second_numerator: 100.try_into().unwrap(),
                            ticks_per_second_denominator: 1.try_into().unwrap(),
                            num_loops: 0,
                            have_timecodes: false,
                        },
                    )
                    .unwrap(),
                )
                .unwrap();
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
                let (input, canonical, planar, words) = inputs(&context, displayed, factor, &depth);
                if storage == 0 {
                    expected.push(words);
                }
                let source: GpuFrameSource = match storage {
                    0 => canonical.into(),
                    1 => input.into(),
                    _ => planar.into(),
                };
                let modular = (index != 1) != reverse;
                let codec = if modular {
                    MixedModeFrameEncoding::Modular
                } else {
                    MixedModeFrameEncoding::VarDct
                };
                let blend = FrameBlend {
                    mode: BlendMode::Blend,
                    source_reference: ReferenceSlot::new(if index == 1 { 1 } else { 3 }).unwrap(),
                    alpha_channel: 0,
                    clamp: true,
                };
                let options = if index == 0 {
                    FrameOptions {
                        kind: FrameKind::ReferenceOnly,
                        save_as_reference: ReferenceSlot::new(1).unwrap(),
                        extra_channel_upsampling: vec![factor, UpsamplingFactor::Four],
                        ..Default::default()
                    }
                } else {
                    FrameOptions {
                        upsampling: factor,
                        extra_channel_upsampling: vec![factor, UpsamplingFactor::Four],
                        crop: (index == 2).then(|| FrameCrop::new(3, -1, 29, 17).unwrap()),
                        color_blend: blend,
                        extra_channel_blends: vec![blend; 2],
                        save_as_reference: ReferenceSlot::new(if index == 1 { 3 } else { 0 })
                            .unwrap(),
                        timing: FrameTiming {
                            duration_ticks: 1,
                            timecode: None,
                        },
                        ..Default::default()
                    }
                };
                sequence
                    .memory_plan(&source, codec, options.clone(), index == 2)
                    .unwrap();
                let job = if index == 2 {
                    sequence.submit_last_frame(source, codec, options)
                } else {
                    sequence.submit_frame(source, codec, options)
                }
                .unwrap();
                sequence.insert(job.wait().unwrap()).unwrap();
            }
            encoded.push(sequence.finish_raw().unwrap());
        }
        assert_eq!(encoded[0], encoded[1]);
        assert_eq!(encoded[0], encoded[2]);
        let encoded = &encoded[1];
        for (index, words) in expected.iter().enumerate() {
            if (index != 1) != reverse {
                assert_eq!(
                    &modular_integer::modular_channel_words(encoded, index),
                    words
                );
            } else {
                assert_eq!(
                    modular_integer::vardct_extra_words(encoded, index),
                    words[3..]
                );
            }
        }
        let native =
            extra_channels::libjxl_output(encoded, &["--original", "--preserve-alpha"]).unwrap();
        let pixels = extent.area().unwrap();
        assert_eq!(native.len(), 2 * pixels * 6);
        let request = GpuOutputRequest::numeric(
            SamplePrecision::float(32, 8).unwrap().pixel_format(),
            NumericSampleMapping::NormalizedUnsigned,
        )
        .unwrap()
        .with_extra_channel(1)
        .unwrap();
        let mut session = open_fragmented(&decoder, encoded, request);
        for index in 0..2 {
            let frame = session.next_frame().unwrap().unwrap();
            let actual = extra_channels::floats(&read_bytes(&gpu, &frame.output().outputs[0]));
            for (i, value) in actual.iter().enumerate() {
                assert!((value - native[index * pixels * 6 + pixels * 5 + i]).abs() <= 2e-6);
            }
        }
        assert!(session.next_frame().unwrap().is_none());
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn texture_previews_share_input_plans_and_release_copy_before_retaining_packets() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(9, 5);
    let words: Vec<_> = (0..45).map(|i| i * 11 % 256).collect();
    let raw = bytes(&words, 1);
    let source = || {
        texture(
            &context,
            extent,
            ColorSampleFormat::GRAY8.pixel_format(),
            wgpu::TextureFormat::R8Uint,
            &raw,
        )
    };
    let encoder = MixedModeEncoder::new(
        context.clone(),
        MixedModeConfig {
            vardct: VarDctConfig {
                sample_format: ColorSampleFormat::GRAY8,
                color_transform: VarDctColorTransform::Original,
                ..Default::default()
            },
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
                ImageSequenceDescriptor::new(9, 5, AnimationHeader::Still)
                    .unwrap()
                    .with_preview(PreviewSize::new(9, 5).unwrap()),
            )
            .unwrap();
        let input = source();
        let owner = Arc::downgrade(&input.texture);
        sequence
            .preview_memory_plan(&input, codec, FrameOptions::default())
            .unwrap();
        let baseline = context.memory_stats().reserved_bytes;
        let preview = sequence
            .submit_preview(input, codec, FrameOptions::default())
            .unwrap()
            .wait()
            .unwrap();
        assert!(owner.upgrade().is_none());
        assert_eq!(
            context.memory_stats().reserved_bytes,
            baseline + preview.reserved_bytes()
        );
        sequence.insert_preview(preview).unwrap();
        let main = sequence
            .submit_last_frame(
                source(),
                MixedModeFrameEncoding::Modular,
                FrameOptions::default(),
            )
            .unwrap()
            .wait()
            .unwrap();
        sequence.insert(main).unwrap();
        let encoded = sequence.finish_raw().unwrap();
        let native = extra_channels::libjxl_output(&encoded, &["--preview", "--original"]).unwrap();
        assert_eq!(native.len(), 45 * 4);
        if codec == MixedModeFrameEncoding::Modular {
            for (i, &word) in words.iter().enumerate() {
                assert!((native[i * 4] - word as f32 / 255.0).abs() <= 1e-6);
            }
        }
    }
    let modular = LosslessModularEncoder::new(context.clone());
    let descriptor = LosslessModularSequenceDescriptor::from_pixel_format(
        9,
        5,
        &ColorSampleFormat::GRAY8.pixel_format(),
        AnimationHeader::Still,
    )
    .unwrap()
    .with_preview(PreviewSize::new(9, 5).unwrap());
    let mut sequence = modular.begin_sequence(descriptor).unwrap();
    sequence
        .preview_memory_plan(source(), FrameOptions::default())
        .unwrap();
    let preview = sequence
        .submit_preview(source(), FrameOptions::default())
        .unwrap()
        .wait()
        .unwrap();
    sequence.insert_preview(preview).unwrap();
    let main = sequence
        .submit_last_frame(source(), FrameOptions::default())
        .unwrap()
        .wait()
        .unwrap();
    sequence.insert(main).unwrap();
    let encoded = sequence.finish_raw().unwrap();
    let native = extra_channels::libjxl_output(&encoded, &["--preview", "--original"]).unwrap();
    for (i, &word) in words.iter().enumerate() {
        assert!((native[i * 4] - word as f32 / 255.0).abs() <= 1e-6);
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
