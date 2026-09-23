use super::*;
use jxl_wgpu_encode::{LosslessModularPalette as Palette, LosslessModularSqueezeStep as Step};

fn sequence(steps: &[(bool, u32, u32, bool)]) -> Squeeze {
    Squeeze::sequence(
        steps
            .iter()
            .map(|&(axis, begin, count, in_place)| Step::new(axis, begin, count, in_place).unwrap())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn pyramid(channels: u32, levels: usize, in_place: bool) -> Squeeze {
    sequence(
        &(0..levels)
            .flat_map(|_| {
                [
                    (true, 0, channels, in_place),
                    (false, 0, channels, in_place),
                ]
            })
            .collect::<Vec<_>>(),
    )
}

#[test]
fn explicit_squeeze_sequences_keep_repeated_axes_unequal_channels_and_empty_residuals() {
    let rig = Rig::new();
    let mut variant = 0;
    for format in [
        LosslessModularFormat::Gray,
        LosslessModularFormat::GrayAlpha,
        LosslessModularFormat::Rgb,
        LosslessModularFormat::Rgba,
    ] {
        let channels = format.channel_count();
        for in_place in [false, true] {
            let policy = pyramid(channels, 3, in_place);
            let config = selection::config(policy, variant);
            let mut case = case(format, 16, SampleKind::Unsigned);
            case.storage = [Storage::Packed, Storage::Planar, Storage::Split][variant % 3];
            for extent in [
                Extent2d::new(1, 1),
                Extent2d::new(1, 257),
                Extent2d::new(257, 1),
                Extent2d::new(257, 9),
            ] {
                selection::checked(
                    &rig,
                    config.clone(),
                    case,
                    extent,
                    &case.samples(extent),
                    channels * 7,
                );
            }
            variant += 1;
        }
    }
    let steps = [
        (true, 1, 2, false),
        (false, 0, 1, true),
        (true, 3, 2, true),
        (false, 6, 1, false),
    ];
    let case = case(LosslessModularFormat::Rgba, 8, SampleKind::Unsigned);
    let config = selection::config(sequence(&steps), 0);
    let extent = Extent2d::new(256, 9);
    let encoded = selection::checked(&rig, config, case, extent, &case.samples(extent), 10);
    let expected_steps: Vec<_> = steps
        .iter()
        .map(|&(axis, begin, count, place)| (axis, place, begin, count))
        .collect();
    assert_eq!(
        selection::local_steps(&encoded),
        [expected_steps.clone(), expected_steps]
    );
}

#[test]
fn explicit_squeeze_sequences_match_native_words_through_every_precision_and_rct() {
    let rig = Rig::new();
    for (kind, bits) in (1..=31)
        .map(|bits| (SampleKind::Unsigned, bits))
        .chain([(SampleKind::Float, 16), (SampleKind::Float, 32)])
    {
        let case = case(LosslessModularFormat::Rgba, bits, kind);
        let extent = Extent2d::new(19, 11);
        let mut expected = case.samples(extent);
        for (pixel, words) in expected.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            words[1] = if kind == SampleKind::Float {
                if bits == 32 { 0xffc0_0001 } else { 0xfe01 }
            } else {
                (u32::MAX >> (32 - bits)) - pixel as u32 % (1u32 << bits.min(8))
            };
        }
        for in_place in [false, true] {
            let steps = [
                (true, 1, 1, in_place),
                (true, 1, 1, in_place),
                (false, 1, 1, in_place),
                (false, 1, 1, in_place),
            ];
            selection::checked(
                &rig,
                selection::config(sequence(&steps), bits as usize),
                case,
                extent,
                &expected,
                8,
            );
        }
    }
    for value in 0..42 {
        let mut config = selection::config(pyramid(4, 2, value % 2 == 0), value as usize);
        config.color_transform = if value % 2 == 0 {
            Transform::GlobalRct(Rct::new(value).unwrap())
        } else {
            Transform::LocalRct(Rct::new(value).unwrap())
        };
        let case = case(LosslessModularFormat::Rgba, 16, SampleKind::Unsigned);
        let extent = Extent2d::new(config.group_size.dimension() + 1, 3);
        selection::checked(&rig, config, case, extent, &case.samples(extent), 20);
    }
}

#[test]
fn explicit_squeeze_sequences_compose_with_all_palette_policies_and_index_ranges() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Rgba, 16, SampleKind::Unsigned);
    let extent = Extent2d::new(257, 3);
    let mut expected = case.samples(extent);
    for words in expected.as_chunks_mut::<4>().0 {
        words[1] %= 3;
        words[2] %= 5;
    }
    for (index, palette) in [
        Palette::new(32).unwrap(),
        Palette::deltas(4096, Predictor::West).unwrap(),
        Palette::mixed(3, 4096, Predictor::Weighted).unwrap(),
        Palette::implicit(4096, Predictor::Gradient).unwrap(),
    ]
    .into_iter()
    .enumerate()
    {
        for in_place in [false, true] {
            // Original component 0, Palette index and component 3 keep independent lineages.
            let steps = [
                (true, 1, 1, in_place),
                (true, 0, 2, in_place),
                (false, 0, 2, in_place),
            ];
            let config = LosslessModularConfig {
                palette: Some(palette.with_components(1, 2).unwrap()),
                group_size: Size::Pixels256,
                ..selection::config(sequence(&steps), index)
            };
            let encoded = selection::checked(&rig, config, case, extent, &expected, 9);
            let expected_steps: Vec<_> = steps
                .iter()
                .map(|&(axis, begin, count, place)| (axis, place, begin + 1, count))
                .collect();
            assert_eq!(
                selection::local_steps(&encoded),
                [expected_steps.clone(), expected_steps]
            );
        }
    }
}

#[test]
fn explicit_squeeze_preserves_bytes_of_equivalent_separable_policies() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Rgba, 16, SampleKind::Unsigned);
    for (mode, axes) in [
        vec![true],
        vec![false],
        vec![true, false],
        vec![false, true],
    ]
    .into_iter()
    .enumerate()
    {
        for in_place in [false, true] {
            for extent in [Extent2d::new(17, 9), Extent2d::new(256, 3)] {
                let config = selection::config(MODES[mode].clone().with_in_place(in_place), 0);
                let encoder =
                    LosslessModularEncoder::with_config(rig.context.clone(), config.clone());
                let expected = case.samples(extent);
                let original = encoder
                    .encode_container(upload(&rig.context, &case, extent, &expected, 0))
                    .unwrap();
                let steps: Vec<_> = axes
                    .iter()
                    .enumerate()
                    .map(|(index, &axis)| (axis, 0, 4 << index, in_place))
                    .collect();
                let encoded = selection::checked(
                    &rig,
                    LosslessModularConfig {
                        squeeze: sequence(&steps),
                        ..config
                    },
                    case,
                    extent,
                    &expected,
                    4 << axes.len(),
                );
                assert_eq!(encoded, original);
            }
        }
    }
}

fn split_all(steps: &mut Vec<(bool, u32, u32, bool)>, horizontal: bool, channels: u32) {
    for begin in (0..channels).step_by(19) {
        steps.push((horizontal, begin, (channels - begin).min(19), false));
    }
}

#[test]
fn explicit_squeeze_wire_covers_parameter_count_and_channel_index_buckets() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Gray, 8, SampleKind::Unsigned);
    let extent = Extent2d::new(256, 1);
    for count in [1, 16, 17, 72, 73, 296] {
        let mut steps = Vec::new();
        if count == 1 {
            steps.push((true, 0, 1, false));
        } else {
            for level in 0..7 {
                split_all(&mut steps, true, 1 << level);
            }
            let seed = steps.len();
            while steps.len() < count {
                let index = steps.len() - seed;
                let begin = (index % 7) as u32 * 19;
                steps.push(((index / 7) % 2 == 0, begin, (128 - begin).min(19), false));
            }
        }
        let channels = 1 + steps.iter().map(|step| step.2).sum::<u32>();
        let config = selection::config(sequence(&steps), 0);
        let encoded =
            selection::checked(&rig, config, case, extent, &case.samples(extent), channels);
        let expected: Vec<_> = steps
            .iter()
            .map(|&(axis, begin, count, place)| (axis, place, begin, count))
            .collect();
        assert_eq!(
            selection::local_steps(&encoded),
            [expected.clone(), expected]
        );
    }
    let case = super::case(LosslessModularFormat::Rgba, 8, SampleKind::Unsigned);
    let extent = Extent2d::new(256, 3);
    let mut steps = Vec::new();
    for level in 0..7 {
        split_all(&mut steps, true, 4 << level);
    }
    split_all(&mut steps, false, 512);
    split_all(&mut steps, false, 512);
    steps.push((true, 1096, 19, true));
    let encoded = selection::checked(
        &rig,
        selection::config(sequence(&steps), 0),
        case,
        extent,
        &case.samples(extent),
        1555,
    );
    let expected: Vec<_> = steps
        .iter()
        .map(|&(axis, begin, count, place)| (axis, place, begin, count))
        .collect();
    assert_eq!(
        selection::local_steps(&encoded),
        [expected.clone(), expected]
    );
}

#[test]
fn explicit_squeeze_rejects_invalid_intermediate_topology_before_allocation() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Gray, 8, SampleKind::Unsigned);
    let extent = Extent2d::new(1, 1);
    let policies = [
        sequence(&[(true, 1, 1, false)]),
        sequence(&[(true, 0, 1, false), (false, 1, 1, false)]),
        sequence(&[(true, 0, 1, false); 32]),
    ];
    for (index, policy) in policies.into_iter().enumerate() {
        let encoder =
            LosslessModularEncoder::with_config(rig.context.clone(), selection::config(policy, 0));
        let input = upload(&rig.context, &case, extent, &[17], 0);
        for error in [
            encoder.memory_plan(&input).unwrap_err(),
            encoder.submit(input).err().unwrap(),
        ] {
            match index {
                0 => assert!(matches!(
                    error,
                    EncodeError::InvalidModularSqueezeChannels {
                        begin: 1,
                        count: 1,
                        channels: 1
                    }
                )),
                1 => assert!(matches!(
                    error,
                    EncodeError::EmptyModularSqueezeChannel {
                        step: 1,
                        channel: 1
                    }
                )),
                _ => assert!(matches!(
                    error,
                    EncodeError::ModularSqueezeShiftLimit {
                        step: 31,
                        channel: 0,
                        horizontal: 31,
                        vertical: 0
                    }
                )),
            }
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(encoder.buffer_pool_stats().allocation_misses, 0);
    }
    let encoder = LosslessModularEncoder::with_config(
        rig.context.clone(),
        selection::config(sequence(&[(true, 0, 1, false); 31]), 0),
    );
    let encoded = encoder
        .encode(upload(&rig.context, &case, extent, &[17], 0))
        .unwrap();
    palette::delta::check_frame_oracles(&encoded, &[&[17]], &case);
}

#[test]
fn later_squeeze_stage_overflow_never_publishes_resident_or_streamed_output() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::GrayAlpha, 32, SampleKind::Float);
    for in_place in [false, true] {
        for variant in [0, 1] {
            let encoder = LosslessModularEncoder::with_config(
                rig.context.clone(),
                selection::config(
                    sequence(&[(true, 1, 1, in_place), (true, 2, 1, in_place)]),
                    variant,
                ),
            );
            let edge = encoder.config().group_size.dimension();
            for extent in [Extent2d::new(4, 1), Extent2d::new(edge * 65 + 4, 1)] {
                let mut expected: Vec<_> = (0..extent.width)
                    .flat_map(|x| {
                        [
                            0x7fc0_0001,
                            if extent.width > 4 && x < edge * 64 {
                                0x3f80_0000
                            } else {
                                [0xc000_0000, 0x3fff_ffff, 0x3fff_ffff, 0xc000_0000]
                                    [(x % 4) as usize]
                            },
                        ]
                    })
                    .collect();
                let input = upload(&rig.context, &case, extent, &expected, 4099);
                let plan = encoder.memory_plan(&input).unwrap();
                assert_eq!(plan.streaming, extent.width > 4);
                assert!(plan.squeeze_scratch_bytes > 0);
                let source = Arc::downgrade(&input.buffer);
                let error =
                    pollster::block_on(encoder.submit_container(input).unwrap()).unwrap_err();
                assert!(
                    matches!(
                        error,
                        EncodeError::Backend(BackendError::ModularSqueezeOverflow)
                    ),
                    "{error:?}"
                );
                assert!(source.upgrade().is_none());
                assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
                assert_eq!(encoder.buffer_pool_stats().leased_buffer_sets, 0);
                for words in expected.as_chunks_mut::<2>().0 {
                    words[1] = 0x3f80_0000;
                }
                let encoded = encoder
                    .encode(upload(&rig.context, &case, extent, &expected, 0))
                    .unwrap();
                palette::delta::check_frame_oracles(&encoded, &[&expected], &case);
            }
        }
    }
}

#[test]
fn squeeze_sequence_arenas_obey_exact_admission_cancellation_and_pool_reuse() {
    fn samples(case: Case, extent: Extent2d, colors: u32) -> Vec<u32> {
        let entries = case.samples(Extent2d::new(colors, 1));
        (0..extent.area().unwrap())
            .flat_map(|pixel| {
                entries[pixel % colors as usize * 4..][..4]
                    .iter()
                    .map(|word| 0x4000_0000 | (word & 0xffff))
            })
            .collect()
    }
    let rig = Rig::new();
    for index in 0..4 {
        let config = LosslessModularConfig {
            palette: Some(
                Palette::mixed(3, 4096, Predictor::Weighted)
                    .unwrap()
                    .with_components(1, 2)
                    .unwrap(),
            ),
            color_transform: Transform::LocalRct(Rct::new(41).unwrap()),
            predictor: Predictor::Weighted,
            lz77: Lz77::Greedy,
            ..selection::config(pyramid(3, 2, index % 2 == 0), index)
        };
        palette::check_lifetime_with_samples(
            &rig,
            config,
            palette::delta::check_frame_oracles,
            samples,
        );
    }
}

#[test]
fn squeeze_sequences_keep_animation_words_and_cropped_references() {
    let rig = Rig::new();
    for in_place in [false, true] {
        let config = selection::config(pyramid(4, 2, in_place), usize::from(in_place));
        let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config.clone());
        groups::animation::check_animation_words_with_oracle(
            &rig,
            &encoder,
            config.group_size,
            LosslessModularFormat::Rgba,
            SampleKind::Unsigned,
            16,
            palette::delta::check_frame_oracles,
        );
        groups::animation::check_cropped_frames(&rig, &encoder, config.group_size);
    }
}
