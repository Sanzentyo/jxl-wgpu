use super::*;

#[test]
fn mixed_names_belong_to_physical_frames_and_orientation_follows_composition() {
    let rig = Rig::new();
    let canvas = Extent2d::new(33, 19);
    let animation = AnimationHeader::Animation {
        ticks_per_second_numerator: 1000.try_into().unwrap(),
        ticks_per_second_denominator: 1.try_into().unwrap(),
        num_loops: 2,
        have_timecodes: true,
    };
    for value in 1..=8 {
        let orientation = OutputOrientation::from_exif_value(value).unwrap();
        let encoder = MixedModeEncoder::new(
            rig.context.clone(),
            MixedModeConfig {
                vardct: VarDctConfig {
                    color_transform: VarDctColorTransform::Original,
                    alpha: Some(AlphaAssociation::Unassociated),
                    progressive: progression(),
                    image_options: ImageOptions {
                        orientation,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                modular: LosslessModularConfig {
                    entropy: LosslessModularEntropyCoding::Ans,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let mut session = encoder
            .begin_sequence(
                ImageSequenceDescriptor::new(canvas.width, canvas.height, animation).unwrap(),
            )
            .unwrap();
        let names = (0..4).map(|i| name(value as usize + i)).collect::<Vec<_>>();
        let mut jobs = Vec::new();
        for (index, name) in names.iter().enumerate() {
            let factor = [
                UpsamplingFactor::Two,
                UpsamplingFactor::One,
                UpsamplingFactor::Four,
                UpsamplingFactor::Two,
            ][index];
            let extent = if index == 3 {
                Extent2d::new(29, 13)
            } else {
                canvas
            };
            let (source, _) = source(
                &rig,
                factor.source_extent(extent),
                ColorSampleFormat::RGB8,
                true,
                index as u32,
            );
            let reference = ReferenceSlot::new(1).unwrap();
            let mut options = FrameOptions {
                name: name.clone(),
                upsampling: factor,
                timing: FrameTiming {
                    duration_ticks: index.saturating_sub(1) as u32,
                    timecode: Some(600 + index as u32),
                },
                ..Default::default()
            };
            if index == 0 {
                options.kind = FrameKind::ReferenceOnly;
                options.timing = Default::default();
                options.save_as_reference = reference;
            } else if index == 1 {
                options.save_as_reference = ReferenceSlot::new(2).unwrap();
            } else {
                options.color_blend = FrameBlend {
                    mode: if index == 2 {
                        BlendMode::Add
                    } else {
                        BlendMode::Replace
                    },
                    source_reference: reference,
                    ..Default::default()
                };
                options.extra_channel_blends = vec![options.color_blend];
                if index == 3 {
                    options.crop =
                        Some(FrameCrop::new(-3, 9, extent.width, extent.height).unwrap());
                }
            }
            let encoding = if (index + value as usize).is_multiple_of(2) {
                MixedModeFrameEncoding::Modular
            } else {
                MixedModeFrameEncoding::VarDct
            };
            jobs.push(
                if index == 3 {
                    session.submit_last_frame(source, encoding, options)
                } else {
                    session.submit_frame(source, encoding, options)
                }
                .unwrap(),
            );
        }
        // Reverse completion preserves metadata's physical-frame owner and finality.
        for job in jobs.into_iter().rev() {
            session.insert(job.wait().unwrap()).unwrap();
        }
        let bytes = session.finish_raw().unwrap();
        check_headers(&bytes, orientation, &names);
        rig.check_output(&bytes, canvas, orientation, true);
        assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
    }
}
