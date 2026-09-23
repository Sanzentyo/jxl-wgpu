use super::*;
use jxl_test_support::oracles::modular_words;

const FORMATS: [LosslessModularFormat; 4] = [
    LosslessModularFormat::Gray,
    LosslessModularFormat::GrayAlpha,
    LosslessModularFormat::Rgb,
    LosslessModularFormat::Rgba,
];

fn policies(predictor: Predictor) -> [Palette; 4] {
    [
        Palette::new(32).unwrap(),
        Palette::deltas(4096, predictor).unwrap(),
        Palette::mixed(3, 4096, predictor).unwrap(),
        Palette::implicit(4096, predictor).unwrap(),
    ]
}

fn ranges(channels: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..channels).flat_map(move |begin| (1..=channels - begin).map(move |count| (begin, count)))
}

fn config(policy: Palette, begin: u32, count: u32, variant: usize) -> LosslessModularConfig {
    LosslessModularConfig {
        palette: Some(policy.with_components(begin, count).unwrap()),
        group_size: Size::ALL[variant % 4],
        tree_mode: TREES[variant % 2],
        color_transform: Transform::None,
        predictor: Predictor::ALL[(variant + 7) % 14],
        weighted_predictor: Weighted::new([31, 0, 17, 3, 11, 31, 1], [0, 15, 7, 12]).unwrap(),
        lz77: if variant.is_multiple_of(2) {
            Lz77::ZeroRuns
        } else {
            Lz77::Greedy
        },
        squeeze: SQUEEZES[variant % 5].clone(),
    }
}

#[test]
fn every_component_range_keeps_unselected_channels_with_all_palette_policies() {
    let rig = Rig::new();
    let mut variant = 0;
    for format in FORMATS {
        for (begin, count) in ranges(format.channel_count()) {
            for policy in policies(Predictor::ALL[variant % 14]) {
                let config = config(policy, begin, count, variant);
                let encoder =
                    LosslessModularEncoder::with_config(rig.context.clone(), config.clone());
                let mut case = case(format, 8, SampleKind::Unsigned);
                case.storage = [Storage::Packed, Storage::Planar, Storage::Split][variant % 3];
                let extent = Extent2d::new(config.group_size.dimension() + 1, 3);
                let mut expected = samples(case, extent, 17);
                for (pixel, words) in expected
                    .chunks_exact_mut(format.channel_count() as usize)
                    .enumerate()
                {
                    for (channel, word) in words.iter_mut().enumerate() {
                        if !(begin..begin + count).contains(&(channel as u32)) {
                            // More than 32 independent unselected values must not consume the table.
                            *word = (pixel as u32 * 73 + channel as u32 * 19) & 255;
                        }
                    }
                }
                let plan = encoder
                    .memory_plan(&upload(&rig.context, &case, extent, &expected, 0))
                    .unwrap();
                let bands = [1, 2, 2, 4, 4][variant % 5];
                assert_eq!(
                    plan.channel_count,
                    1 + (format.channel_count() - count + 1) * bands
                );
                let encoded = checked_stream(&rig, &encoder, case, extent, &expected);
                for header in mixed::local_headers(&encoded) {
                    assert_eq!((header.begin, header.components), (begin, count));
                }
                variant += 1;
            }
        }
    }
    assert_eq!(variant, 80);
}

#[test]
fn selected_and_unselected_components_keep_every_integer_and_ieee_precision() {
    let rig = Rig::new();
    for (kind, bits) in (1..=31)
        .map(|bits| (SampleKind::Unsigned, bits))
        .chain([(SampleKind::Float, 16), (SampleKind::Float, 32)])
    {
        for (index, policy) in policies(Predictor::ALL[bits as usize % 14])
            .into_iter()
            .enumerate()
        {
            let (begin, count) = [(1, 2), (3, 1), (0, 3), (2, 1)][index];
            let encoder = LosslessModularEncoder::with_config(
                rig.context.clone(),
                LosslessModularConfig {
                    squeeze: Squeeze::None,
                    ..config(policy, begin, count, bits as usize + index)
                },
            );
            let case = case(LosslessModularFormat::Rgba, bits, kind);
            let extent = Extent2d::new(17, 9);
            check(&rig, &encoder, case, extent, &samples(case, extent, 17));
        }
    }
}

// Keep wide low bits while ensuring every post-RCT/Squeeze residual is representable.
fn representable_samples(case: Case, extent: Extent2d, colors: u32) -> Vec<u32> {
    let mask = (u32::MAX >> (32 - case.bits)).min(0xffff);
    let base = if case.bits > 18 {
        1 << (case.bits - 2)
    } else {
        0
    };
    samples(case, extent, colors)
        .into_iter()
        .map(|word| base | (word & mask))
        .collect()
}

#[test]
fn component_ranges_compose_with_every_rct_and_single_pixel_squeeze_axes() {
    let rig = Rig::new();
    let selections: Vec<_> = ranges(4).collect();
    for value in 0..42 {
        let variant = value as usize;
        let (begin, count) = selections[variant % selections.len()];
        let mut config = config(
            policies(Predictor::ALL[variant % 14])[variant % 4],
            begin,
            count,
            variant,
        );
        config.color_transform = if value % 2 == 0 {
            Transform::GlobalRct(Rct::new(value).unwrap())
        } else {
            Transform::LocalRct(Rct::new(value).unwrap())
        };
        let edge = config.group_size.dimension();
        let extent = [
            Extent2d::new(edge + 1, 3),
            Extent2d::new(1, edge + 1),
            Extent2d::new(edge + 1, 1),
            Extent2d::new(17, 9),
            Extent2d::new(1, 1),
        ][variant % 5];
        let case = case(
            LosslessModularFormat::Rgba,
            (value % 31 + 1) as u8,
            SampleKind::Unsigned,
        );
        let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config);
        check(
            &rig,
            &encoder,
            case,
            extent,
            &representable_samples(case, extent, 17),
        );
    }
}

#[test]
fn implicit_entries_are_relative_to_the_selected_components_including_alpha() {
    let rig = Rig::new();
    for (signed, bits) in [(false, 8), (true, 31)] {
        let entries = modular_words::implicit_entries(bits);
        for format in FORMATS {
            for (begin, count) in ranges(format.channel_count()) {
                let case = case(format, bits, SampleKind::Unsigned);
                let extent = Extent2d::new(if signed { 286 } else { 189 }, 1);
                let mut expected = case.samples(extent);
                for (pixel, words) in expected
                    .chunks_exact_mut(format.channel_count() as usize)
                    .enumerate()
                {
                    for component in 0..count as usize {
                        words[begin as usize + component] = if signed {
                            let value = entries[pixel / 2][component];
                            (1 + if pixel % 2 == 0 {
                                (-value).max(0)
                            } else {
                                value.max(0)
                            }) as u32
                        } else {
                            entries[143 + pixel][component] as u32
                        };
                    }
                }
                let encoder = LosslessModularEncoder::with_config(
                    rig.context.clone(),
                    LosslessModularConfig {
                        palette: Some(
                            Palette::implicit(
                                if signed { 512 } else { 1 },
                                if signed {
                                    Predictor::West
                                } else {
                                    Predictor::Zero
                                },
                            )
                            .unwrap()
                            .with_components(begin, count)
                            .unwrap(),
                        ),
                        group_size: Size::Pixels1024,
                        color_transform: Transform::None,
                        predictor: Predictor::Weighted,
                        lz77: Lz77::Greedy,
                        ..Default::default()
                    },
                );
                let encoded = checked_stream(&rig, &encoder, case, extent, &expected);
                let [negative, cubes, explicit] = modular_words::palette_index_counts(&encoded);
                assert_eq!(negative + cubes + explicit, extent.width);
                if signed {
                    assert!(negative >= 143, "{format:?}: {begin}+{count}: {negative}");
                    assert_eq!(cubes, 0);
                } else {
                    assert_eq!(cubes, 189);
                }
            }
        }
    }
}

#[test]
fn selected_components_keep_animation_words_and_cropped_reference_composition() {
    let rig = Rig::new();
    for (index, policy) in policies(Predictor::Weighted).into_iter().enumerate() {
        let policy = if index == 0 {
            Palette::new(4096).unwrap()
        } else {
            policy
        };
        let config = LosslessModularConfig {
            squeeze: Squeeze::None,
            ..config(policy, 1, 1, index)
        };
        let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config.clone());
        for (format, kind, bits) in [
            (LosslessModularFormat::Rgba, SampleKind::Unsigned, 31),
            (LosslessModularFormat::GrayAlpha, SampleKind::Float, 32),
        ] {
            groups::animation::check_animation_words_with_oracle(
                &rig,
                &encoder,
                config.group_size,
                format,
                kind,
                bits,
                delta::check_frame_oracles,
            );
        }
        let composed = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                squeeze: SQUEEZES[index + 1].clone(),
                ..config
            },
        );
        groups::animation::check_cropped_frames(&rig, &composed, config.group_size);
    }
}

#[test]
fn selected_component_scratch_keeps_exact_admission_cancellation_and_pool_reuse() {
    let rig = Rig::new();
    for (index, policy) in policies(Predictor::Weighted).into_iter().enumerate() {
        let (begin, count) = [(0, 1), (1, 2), (2, 2), (3, 1)][index];
        check_lifetime_with_samples(
            &rig,
            LosslessModularConfig {
                squeeze: SQUEEZES[index + 1].clone(),
                predictor: Predictor::Weighted,
                lz77: Lz77::Greedy,
                color_transform: Transform::LocalRct(Rct::new(41).unwrap()),
                ..config(policy, begin, count, index)
            },
            delta::check_frame_oracles,
            representable_samples,
        );
    }
}

#[test]
fn selected_component_overflow_never_publishes_resident_or_late_streamed_output() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Rgba, 32, SampleKind::Float);
    for (index, policy) in [
        Palette::new(1).unwrap(),
        Palette::deltas(1, Predictor::Zero).unwrap(),
        Palette::mixed(1, 1, Predictor::Zero).unwrap(),
        Palette::implicit(1, Predictor::Zero).unwrap(),
    ]
    .into_iter()
    .enumerate()
    {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(policy.with_components(3, 1).unwrap()),
                color_transform: Transform::None,
                tree_mode: TREES[index % 2],
                ..Default::default()
            },
        );
        for extent in [Extent2d::new(7, 3), Extent2d::new(256 * 33 + 7, 3)] {
            let mut expected = case.samples(extent);
            for (pixel, words) in expected.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let x = pixel as u32 % extent.width;
                words[3] = if extent.width > 7 && x < 256 * 32 {
                    0
                } else {
                    [
                        0,
                        0x8000_0000,
                        0x7fc0_0001,
                        1,
                        0xffff_ffff,
                        0x3f80_0000,
                        0x7f80_0000,
                    ][pixel % 7]
                };
            }
            let input = upload(&rig.context, &case, extent, &expected, 4099);
            assert_eq!(
                encoder.memory_plan(&input).unwrap().streaming,
                extent.width > 7
            );
            let source = Arc::downgrade(&input.buffer);
            let error = pollster::block_on(encoder.submit_container(input).unwrap()).unwrap_err();
            assert!(
                matches!(
                    error,
                    EncodeError::Backend(BackendError::ModularPaletteOverflow)
                ),
                "{error:?}"
            );
            assert!(source.upgrade().is_none());
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(encoder.buffer_pool_stats().leased_buffer_sets, 0);
            for words in expected.as_chunks_mut::<4>().0 {
                words[3] = 0;
            }
            let encoded = encoder
                .encode(upload(&rig.context, &case, extent, &expected, 0))
                .unwrap();
            delta::check_frame_oracles(&encoded, &[&expected], &case);
        }
    }
}

#[test]
fn component_admission_and_unselected_squeeze_overflow_remain_typed() {
    let rig = Rig::new();
    for format in &FORMATS[..3] {
        let case = case(*format, 8, SampleKind::Unsigned);
        let extent = Extent2d::new(3, 1);
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            config(Palette::new(4).unwrap(), format.channel_count(), 1, 0),
        );
        let input = upload(&rig.context, &case, extent, &case.samples(extent), 0);
        assert!(matches!(
            encoder.memory_plan(&input),
            Err(EncodeError::InvalidModularPaletteComponents { .. })
        ));
        assert!(matches!(
            encoder.submit(input),
            Err(EncodeError::InvalidModularPaletteComponents { .. })
        ));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(encoder.buffer_pool_stats().allocation_misses, 0);
    }
    let case = case(LosslessModularFormat::Rgba, 32, SampleKind::Float);
    let extent = Extent2d::new(2, 1);
    for policy in policies(Predictor::Weighted) {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                squeeze: Squeeze::Horizontal,
                ..config(policy, 3, 1, 0)
            },
        );
        let expected = [0x8000_0000, 0, 0, 0, 0x7fff_ffff, 0, 0, 0];
        let error = encoder
            .encode(upload(&rig.context, &case, extent, &expected, 0))
            .unwrap_err();
        assert!(
            matches!(
                error,
                EncodeError::Backend(BackendError::ModularSqueezeOverflow)
            ),
            "{error:?}"
        );
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(encoder.buffer_pool_stats().leased_buffer_sets, 0);
    }
}
