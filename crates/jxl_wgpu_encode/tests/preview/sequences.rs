use super::*;

#[test]
fn mixed_preview_keeps_animation_timebase_separate_from_main_frames_and_index() {
    let rig = Rig::new();
    let native = jxl_test_support::oracles::icc_profile::IccProfileOracle::compile();
    let main = Extent2d::new(33, 19);
    let preview = PreviewSize::new(17, 11).unwrap();
    let animation = AnimationHeader::Animation {
        ticks_per_second_numerator: 1000.try_into().unwrap(),
        ticks_per_second_denominator: 1.try_into().unwrap(),
        num_loops: 2,
        have_timecodes: true,
    };
    for preview_codec in [
        MixedModeFrameEncoding::Modular,
        MixedModeFrameEncoding::VarDct,
    ] {
        let orientation = OutputOrientation::from_exif_value(6).unwrap();
        let encoder = MixedModeEncoder::new(
            rig.context.clone(),
            MixedModeConfig {
                vardct: VarDctConfig {
                    color_transform: VarDctColorTransform::Original,
                    alpha: Some(AlphaAssociation::Associated),
                    image_options: ImageOptions {
                        orientation,
                        intrinsic_size: Some(IntrinsicSize::new(1 << 31, 1 << 30).unwrap()),
                        min_nits: jxl_gpu_bitstream::FiniteF16::from_bits(0x2c00).unwrap(),
                        linear_below: ToneMappingThreshold::DisplayFraction(
                            jxl_gpu_bitstream::FiniteF16::from_bits(0x3000).unwrap(),
                        ),
                        ..Default::default()
                    },
                    progressive: progression(),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let mut session = encoder
            .begin_sequence(
                ImageSequenceDescriptor::new(main.width, main.height, animation)
                    .unwrap()
                    .with_preview(preview),
            )
            .unwrap();
        let preview_job = session
            .submit_preview(
                source(&rig, preview.extent(), ColorSampleFormat::RGB8, true, 1),
                preview_codec,
                FrameOptions {
                    timing: FrameTiming {
                        duration_ticks: 77,
                        timecode: Some(900),
                    },
                    name: CodestreamName::new("preview").unwrap(),
                    ..Default::default()
                },
            )
            .unwrap();
        let mut jobs = Vec::new();
        for index in 0..3 {
            let options = if index == 0 {
                FrameOptions {
                    kind: FrameKind::ReferenceOnly,
                    save_as_reference: ReferenceSlot::new(1).unwrap(),
                    ..Default::default()
                }
            } else {
                FrameOptions {
                    timing: FrameTiming {
                        duration_ticks: index as u32,
                        timecode: Some(100 + index as u32),
                    },
                    color_blend: FrameBlend {
                        mode: BlendMode::Add,
                        source_reference: ReferenceSlot::new(1).unwrap(),
                        ..Default::default()
                    },
                    extra_channel_blends: vec![FrameBlend {
                        mode: BlendMode::Add,
                        source_reference: ReferenceSlot::new(1).unwrap(),
                        ..Default::default()
                    }],
                    ..Default::default()
                }
            };
            let encoding = if index == 1 {
                MixedModeFrameEncoding::VarDct
            } else {
                MixedModeFrameEncoding::Modular
            };
            let input = source(&rig, main, ColorSampleFormat::RGB8, true, index as u32 + 3);
            jobs.push(
                if index == 2 {
                    session.submit_last_frame(input, encoding, options)
                } else {
                    session.submit_frame(input, encoding, options)
                }
                .unwrap(),
            );
        }
        for job in jobs.into_iter().rev() {
            session.insert(job.wait().unwrap()).unwrap();
        }
        session.insert_preview(preview_job.wait().unwrap()).unwrap();
        let bytes = session
            .finish_indexed_container(Default::default(), Default::default())
            .unwrap();
        rig.check(&bytes, main, preview, orientation, 1);
        let parsed = jxl_gpu_bitstream::parse(&bytes, Default::default()).unwrap();
        let inventory = Arc::new(parsed.codestream_inventory(Default::default()).unwrap());
        let index = jxl_gpu_bitstream::FrameIndex::from_container(&parsed, Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            index.entries()[0].codestream_offset,
            inventory.frames[1].header_bits.offset / 8
        );
        let selected = inventory.select_image(ImageSelection::Main).unwrap();
        let metadata = native.image_info(&bytes);
        assert_eq!(metadata.intrinsic_size, (1 << 31, 1 << 30));
        assert_eq!(
            metadata.preview,
            Some((preview.extent().width, preview.extent().height))
        );
        assert!(metadata.animation && metadata.relative_to_max_display);
        assert_eq!(metadata.min_nits, 0.0625);
        assert_eq!(metadata.linear_below, 0.125);
        assert_eq!(
            selected.image_header.intrinsic_size,
            Some(metadata.intrinsic_size)
        );
        let selected_preview = inventory.select_image(ImageSelection::Preview).unwrap();
        assert_eq!(selected_preview.image_header.intrinsic_size, None);
        assert_eq!(selected_preview.image_header.animation, None);
        assert_eq!(
            selected_preview.image_header.tone_mapping,
            selected.image_header.tone_mapping
        );
        assert_eq!(selected.frames, inventory.frames[1..]);
        assert_eq!(selected.frames[0].frame_index, 1);
        let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
            jxl_gpu_formats::RgbChannelOrder::Rgba,
            false,
            jxl_wgpu_decode::vardct_rgb8_format().color_spec,
        ))
        .unwrap()
        .with_alpha_output_policy(AlphaOutputPolicy::Preserve);
        let mut sequential = rig.decoder.open(&bytes, request.clone()).unwrap();
        for target in 0..2 {
            let expected = sequential.next_frame().unwrap().unwrap();
            let mut seek = rig
                .decoder
                .open_seek(
                    &bytes,
                    request.clone(),
                    target,
                    Default::default(),
                    Default::default(),
                )
                .unwrap();
            let actual = seek.next_frame().unwrap().unwrap();
            assert_eq!(actual.metadata, expected.metadata);
            assert_eq!(
                read_bytes(&rig.gpu, &actual.output().outputs[0]),
                read_bytes(&rig.gpu, &expected.output().outputs[0])
            );
            assert!(seek.next_frame().unwrap().is_none());
        }
        drop(sequential);
        assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
    }
}
