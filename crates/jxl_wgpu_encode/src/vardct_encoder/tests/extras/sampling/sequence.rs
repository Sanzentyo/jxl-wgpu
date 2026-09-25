use super::*;
use crate::{
    AlphaAssociation, BlendMode, FrameBlend, FrameCrop, FrameKind, FrameTiming, ReferenceSlot,
};

#[test]
fn extra_sampling_changes_per_frame_before_reference_crop_and_all_blend_modes() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoder = GpuDecoder::wgpu(gpu.clone()).unwrap();
    let readback = ImageReadbackPipeline::new(&gpu);
    for association in [AlphaAssociation::Unassociated, AlphaAssociation::Associated] {
        let definitions: Vec<_> = [
            ExtraChannelKind::Alpha(association),
            ExtraChannelKind::Depth,
            ExtraChannelKind::Thermal,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, kind)| {
            ExtraChannel::new(
                kind,
                SamplePrecision::float(32, 8).unwrap(),
                u8::from(i == 1),
                Vec::new(),
            )
            .unwrap()
        })
        .collect();
        let encoder = TiledVarDctEncoder::new_with_config(
            context.clone(),
            VarDctConfig {
                color_transform: crate::VarDctColorTransform::Original,
                extra_channels: definitions.clone(),
                progressive: progressive::combined(),
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
            let mut session = encoder
                .begin_sequence(
                    ImageSequenceDescriptor::new(
                        65,
                        67,
                        AnimationHeader::Animation {
                            ticks_per_second_numerator: 1000.try_into().unwrap(),
                            ticks_per_second_denominator: 1.try_into().unwrap(),
                            num_loops: 2,
                            have_timecodes: true,
                        },
                    )
                    .unwrap(),
                )
                .unwrap();
            let factors = [
                [
                    ExtraChannelUpsampling::Two,
                    ExtraChannelUpsampling::Four,
                    ExtraChannelUpsampling::One,
                ],
                [
                    ExtraChannelUpsampling::Four,
                    ExtraChannelUpsampling::One,
                    ExtraChannelUpsampling::Two,
                ],
                [
                    ExtraChannelUpsampling::Eight,
                    ExtraChannelUpsampling::Two,
                    ExtraChannelUpsampling::Four,
                ],
            ];
            let mut physical = Vec::new();
            for (index, factors) in factors.iter().enumerate() {
                let extent = if index == 2 {
                    Extent2d::new(63, 61)
                } else {
                    Extent2d::new(65, 67)
                };
                let (source, words) = input(&context, extent, &definitions, factors, index);
                physical.push(words);
                let blend = |mode, reference| FrameBlend {
                    mode,
                    source_reference: ReferenceSlot::new(reference).unwrap(),
                    ..Default::default()
                };
                let mut options = if index == 0 {
                    FrameOptions {
                        kind: FrameKind::ReferenceOnly,
                        save_as_reference: ReferenceSlot::new(1).unwrap(),
                        ..Default::default()
                    }
                } else {
                    FrameOptions {
                        timing: FrameTiming {
                            duration_ticks: index as u32 + 2,
                            timecode: Some(100 + index as u32),
                        },
                        crop: (index == 2).then(|| FrameCrop::new(-1, 7, 63, 61).unwrap()),
                        color_blend: blend(
                            if index == 1 { BlendMode::Add } else { mode },
                            if index == 1 { 1 } else { 3 },
                        ),
                        extra_channel_blends: (0..3)
                            .map(|extra| {
                                blend(
                                    if index == 1 { BlendMode::Add } else { mode },
                                    if index == 1 || extra != 1 { 1 } else { 3 },
                                )
                            })
                            .collect(),
                        save_as_reference: ReferenceSlot::new(if index == 1 { 3 } else { 0 })
                            .unwrap(),
                        ..Default::default()
                    }
                };
                options.extra_channel_upsampling = factors.to_vec();
                let job = if index == 2 {
                    session.submit_last_frame(source, options)
                } else {
                    session.submit_frame(source, options)
                }
                .unwrap();
                session.insert(job.wait().unwrap()).unwrap();
            }
            let bytes = session.finish_raw().unwrap();
            for (index, words) in physical.iter().enumerate() {
                check_header(&bytes, index, &definitions, &factors[index]);
                assert_eq!(&modular_integer::vardct_extra_words(&bytes, index), words);
            }
            let native = extra_channels::libjxl_output(&bytes, &[]).unwrap();
            let pixels = 65 * 67;
            let stride = pixels * 7;
            assert_eq!(native.len(), 2 * stride);
            for selected in [None, Some(0), Some(1), Some(2)] {
                let request = if let Some(index) = selected {
                    GpuOutputRequest::numeric(
                        SamplePrecision::float(32, 8).unwrap().pixel_format(),
                        jxl_wgpu_decode::NumericSampleMapping::NativeFloat,
                    )
                    .unwrap()
                    .with_extra_channel(index)
                    .unwrap()
                } else {
                    GpuOutputRequest::color(jxl_gpu_formats::PixelFormat::rgb_f32(
                        jxl_gpu_formats::RgbChannelOrder::Rgba,
                        false,
                        vardct_rgb8_format().color_spec,
                    ))
                    .unwrap()
                };
                let mut session = decoder.open(&bytes, request).unwrap();
                for index in 0..2 {
                    let frame = session.next_frame().unwrap().unwrap();
                    assert_eq!(frame.metadata.timecode, Some(101 + index as u32));
                    let output = readback.submit(frame.output()).unwrap().wait().unwrap();
                    let actual = extra_channels::floats(&output.frame.outputs[0].bytes);
                    let (offset, count) = selected.map_or((0, 4 * pixels), |channel| {
                        ((4 + channel as usize) * pixels, pixels)
                    });
                    let start = index * stride + offset;
                    assert_eq!(actual.len(), count);
                    for (pixel, (&actual, &expected)) in
                        actual.iter().zip(&native[start..start + count]).enumerate()
                    {
                        let tolerance = if selected.is_some() { 2e-6 } else { 2e-4 };
                        assert!(
                            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
                            "{association:?}/{mode:?}/frame {index}/plane {selected:?}/pixel {pixel}: {actual} vs {expected}"
                        );
                    }
                }
                assert!(session.next_frame().unwrap().is_none());
                drop(session);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
            assert_eq!(context.memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn extra_sampling_all_topologies_share_packed_alpha_order_and_reject_implicit_resizing() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(8, 8);
    let config = VarDctConfig {
        alpha: Some(AlphaAssociation::Associated),
        extra_channels: vec![declaration(ExtraChannelKind::Depth, 13, 1)],
        ..Default::default()
    };
    let packed = alpha::input(extent, config.sample_format);
    let primary = alpha::upload(&context, extent, &config, &packed, Storage::Split, true);
    let scalar = scalar_source(
        &context,
        Extent2d::new(1, 1),
        config.extra_channels[0].precision(),
        &[719],
    );
    let source = primary.clone().with_extra_channels(vec![scalar]).unwrap();
    for topology in [
        layouts::Topology::Single,
        layouts::Topology::Map,
        layouts::Topology::Tiled,
    ] {
        let backend = topology.backend(&context, extent, &config);
        let encoder = GpuEncoder::new(context.clone(), backend);
        let mut request = layouts::request(extent, &config);
        for factors in [
            vec![ExtraChannelUpsampling::Four],
            vec![ExtraChannelUpsampling::Two, ExtraChannelUpsampling::Four],
        ] {
            request.options.extra_channel_upsampling = factors;
            assert!(matches!(
                encoder.submit_frame(GpuFrameSource::Buffer(source.clone()), request.clone()),
                Err(EncodeError::InvalidConfiguration(_))
            ));
            assert_eq!(context.memory_stats().reserved_bytes, 0);
        }
        request.options.extra_channel_upsampling =
            vec![ExtraChannelUpsampling::One, ExtraChannelUpsampling::Four];
        let artifacts = encoder
            .submit_frame(GpuFrameSource::Buffer(source.clone()), request)
            .unwrap()
            .wait()
            .unwrap();
        let color = VarDctColorPlan::new(&config).unwrap();
        let (header, permit) = color
            .image_header(&ImageSequenceDescriptor::new(8, 8, AnimationHeader::Still).unwrap())
            .unwrap()
            .finish(context.memory_budget())
            .unwrap();
        let mut bytes = header.bytes().to_vec();
        bytes.extend(assemble_frame(artifacts.packets).unwrap().into_bytes());
        drop(permit);
        let words = modular_integer::vardct_extra_words(&bytes, 0);
        assert_eq!(
            words[0].words,
            packed
                .as_chunks::<4>()
                .0
                .iter()
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>()
        );
        assert_eq!(
            words[1],
            modular_integer::ExtraWords {
                width: 1,
                height: 1,
                words: vec![719]
            }
        );
        let (_, native) = extra_channels::libjxl_planes(&bytes, 64, 2).unwrap();
        assert_numeric(&native[1], &[719; 64], config.extra_channels[0].precision());
    }
    let config = VarDctConfig {
        alpha: config.alpha,
        ..Default::default()
    };
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
    let baseline = encoder.encode(primary.clone()).unwrap();
    let mut session = encoder
        .begin_sequence(ImageSequenceDescriptor::new(8, 8, AnimationHeader::Still).unwrap())
        .unwrap();
    let result = session
        .submit_last_frame(
            primary.clone(),
            FrameOptions {
                extra_channel_upsampling: vec![ExtraChannelUpsampling::One],
                ..Default::default()
            },
        )
        .unwrap()
        .wait()
        .unwrap();
    session.insert(result).unwrap();
    assert_eq!(session.finish_raw().unwrap(), baseline);
    let modular = crate::LosslessModularBackend::new(&context);
    let mut request = layouts::request(extent, &VarDctConfig::default());
    request.profile = crate::EncodeProfile::ModularLossless {
        sample_bit_depth: SamplePrecision::integer(8).unwrap().bit_depth(),
    };
    request.options.extra_channel_upsampling = vec![ExtraChannelUpsampling::Two];
    assert!(matches!(
        modular.memory_plan_for_request(&primary, &request),
        Err(EncodeError::InvalidConfiguration(_))
    ));
    request.options.extra_channel_upsampling = vec![ExtraChannelUpsampling::One];
    modular.memory_plan_for_request(&primary, &request).unwrap();
    request.options.extra_channel_upsampling = vec![ExtraChannelUpsampling::One; 257];
    assert!(matches!(
        modular.memory_plan_for_request(&primary, &request),
        Err(EncodeError::InvalidConfiguration(_))
    ));
    request.options.extra_channel_upsampling = vec![ExtraChannelUpsampling::One];
    assert!(matches!(
        modular.memory_plan_for_request(&color_source(&context, extent), &request),
        Err(EncodeError::InvalidConfiguration(_))
    ));
    let mut mixed_config = crate::MixedModeConfig::default();
    mixed_config.vardct.alpha = Some(AlphaAssociation::Associated);
    let mixed = crate::MixedModeEncoder::new(context.clone(), mixed_config).unwrap();
    let mut mixed_session = mixed
        .begin_sequence(ImageSequenceDescriptor::new(8, 8, AnimationHeader::Still).unwrap())
        .unwrap();
    let retained = context.memory_stats().reserved_bytes;
    for encoding in [
        crate::MixedModeFrameEncoding::Modular,
        crate::MixedModeFrameEncoding::VarDct,
    ] {
        let options = FrameOptions {
            extra_channel_upsampling: vec![ExtraChannelUpsampling::Two],
            ..Default::default()
        };
        assert!(matches!(
            mixed_session.memory_plan(&primary, encoding, options.clone(), true),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            mixed_session.submit_last_frame(primary.clone(), encoding, options),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert_eq!(mixed_session.next_frame_index().get(), 0);
        assert_eq!(context.memory_stats().reserved_bytes, retained);
        mixed_session
            .memory_plan(
                &primary,
                encoding,
                FrameOptions {
                    extra_channel_upsampling: vec![ExtraChannelUpsampling::One],
                    ..Default::default()
                },
                true,
            )
            .unwrap();
    }
    drop(mixed_session);
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
