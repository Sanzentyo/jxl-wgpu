use super::*;
use jxl_test_support::oracles::modular_words;

const FORMATS: [LosslessModularFormat; 4] = [
    LosslessModularFormat::Gray,
    LosslessModularFormat::GrayAlpha,
    LosslessModularFormat::Rgb,
    LosslessModularFormat::Rgba,
];

#[test]
fn implicit_cubes_cover_every_interoperable_depth_and_component_count() {
    let rig = Rig::new();
    for (kind, bits) in (1..=24)
        .map(|bits| (SampleKind::Unsigned, bits))
        .chain([(SampleKind::Float, 16)])
    {
        let entries = modular_words::implicit_entries(bits);
        for (index, format) in FORMATS.into_iter().enumerate() {
            let variant = usize::from(bits) + index;
            let group_size = Size::ALL[variant % 4];
            let delta_predictor = Predictor::ALL[variant % 14];
            let encoder = LosslessModularEncoder::with_config(
                rig.context.clone(),
                LosslessModularConfig {
                    entropy: Default::default(),
                    palette: Some(Palette::implicit(1, delta_predictor).unwrap()),
                    predictor: Predictor::ALL[(variant + 7) % 14],
                    weighted_predictor: Weighted::new([31, 0, 17, 3, 11, 31, 1], [0, 15, 7, 12])
                        .unwrap(),
                    local_transforms: SQUEEZES[variant % 5].clone().into(),
                    group_size,
                    tree_mode: TREES[variant % 2],
                    color_transform: Transform::None,
                    lz77: if variant % 2 == 0 {
                        Lz77::ZeroRuns
                    } else {
                        Lz77::Greedy
                    },
                },
            );
            let case = case(format, bits, kind);
            let extent = Extent2d::new(group_size.dimension() + 1, 3);
            let expected: Vec<_> = (0..extent.area().unwrap())
                .flat_map(|pixel| {
                    entries[143 + pixel % 189][..format.channel_count() as usize]
                        .iter()
                        .map(|&v| v as u32)
                })
                .collect();
            let encoded = checked_stream(&rig, &encoder, case, extent, &expected);
            assert!(
                mixed::local_counts(&encoded)
                    .iter()
                    .all(|&(colors, deltas, predictor)| (colors, deltas, predictor)
                        == (0, 1, delta_predictor as u32))
            );
            if encoder.config().local_transforms.squeeze_policy() == Some(&Squeeze::None) {
                assert_eq!(
                    modular_words::palette_index_counts(&encoded),
                    [0, extent.area().unwrap() as u32, 0]
                );
            }
        }
    }
}

#[test]
fn wide_words_use_exact_residuals_instead_of_native_capped_cubes() {
    let rig = Rig::new();
    let entries = modular_words::implicit_entries(24);
    for (kind, bits) in (25..=31)
        .map(|bits| (SampleKind::Unsigned, bits))
        .chain([(SampleKind::Float, 32)])
    {
        for format in FORMATS {
            let case = case(format, bits, kind);
            let extent = Extent2d::new(257, 1);
            // These words match the native depth-24 cube, but the normative cube at this
            // source depth differs. Neither native nor GPU reconstruction may alter them.
            let expected: Vec<_> = (0..257)
                .flat_map(|pixel| {
                    entries[143 + pixel % 189][..format.channel_count() as usize]
                        .iter()
                        .map(|&v| v as u32)
                })
                .collect();
            let encoder = LosslessModularEncoder::with_config(
                rig.context.clone(),
                LosslessModularConfig {
                    palette: Some(Palette::implicit(256, Predictor::Zero).unwrap()),
                    color_transform: Transform::None,
                    ..Default::default()
                },
            );
            let encoded = checked_stream(&rig, &encoder, case, extent, &expected);
            let [negative, cubes, explicit] = modular_words::palette_index_counts(&encoded);
            assert_eq!(cubes, 0);
            assert_eq!(negative + explicit, 257);
            assert!(explicit > 0);
        }
    }
}

#[test]
fn implicit_signed_entries_preserve_every_native_tuple_and_wide_original_words() {
    let rig = Rig::new();
    for (kind, bits) in [
        (SampleKind::Unsigned, 8),
        (SampleKind::Unsigned, 16),
        (SampleKind::Unsigned, 24),
        (SampleKind::Unsigned, 31),
        (SampleKind::Float, 16),
        (SampleKind::Float, 32),
    ] {
        let entries = modular_words::implicit_entries(bits);
        for format in FORMATS {
            let case = case(format, bits, kind);
            let extent = Extent2d::new(286, 1);
            let expected: Vec<_> = entries[..143]
                .iter()
                .flat_map(|delta| {
                    [false, true].into_iter().flat_map(move |second| {
                        delta[..format.channel_count() as usize]
                            .iter()
                            .map(move |&value| {
                                (1 + if second {
                                    value.max(0)
                                } else {
                                    (-value).max(0)
                                }) as u32
                            })
                    })
                })
                .collect();
            let encoder = LosslessModularEncoder::with_config(
                rig.context.clone(),
                LosslessModularConfig {
                    palette: Some(Palette::implicit(512, Predictor::West).unwrap()),
                    color_transform: Transform::None,
                    group_size: Size::Pixels1024,
                    predictor: Predictor::Weighted,
                    lz77: Lz77::Greedy,
                    ..Default::default()
                },
            );
            let encoded = checked_stream(&rig, &encoder, case, extent, &expected);
            let [negative, cubes, explicit] = modular_words::palette_index_counts(&encoded);
            assert!(negative >= 143, "{format:?}, {bits}: {negative}");
            assert_eq!(negative + cubes + explicit, 286);
            if bits > 24 {
                assert_eq!(cubes, 0);
            }
        }
    }
}

#[test]
fn implicit_policy_composes_rct_all_predictors_squeeze_and_ieee_special_words() {
    let rig = Rig::new();
    for value in 0..42 {
        let group_size = Size::ALL[value as usize % 4];
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                entropy: Default::default(),
                palette: Some(
                    Palette::implicit(4096, Predictor::ALL[value as usize % 14]).unwrap(),
                ),
                predictor: Predictor::ALL[(value as usize + 7) % 14],
                weighted_predictor: Weighted::new([31, 0, 17, 3, 11, 31, 1], [0, 15, 7, 12])
                    .unwrap(),
                local_transforms: SQUEEZES[value as usize % 5].clone().into(),
                group_size,
                tree_mode: TREES[value as usize % 2],
                color_transform: if value % 2 == 0 {
                    Transform::GlobalRct(Rct::new(value).unwrap())
                } else {
                    Transform::LocalRct(Rct::new(value).unwrap())
                },
                lz77: if value % 3 == 0 {
                    Lz77::ZeroRuns
                } else {
                    Lz77::Greedy
                },
            },
        );
        let mut case = case(
            LosslessModularFormat::Rgba,
            (value % 31 + 1) as u8,
            SampleKind::Unsigned,
        );
        case.storage = [Storage::Packed, Storage::Planar, Storage::Split][value as usize % 3];
        let extent = Extent2d::new(group_size.dimension() + 1, 3);
        check(&rig, &encoder, case, extent, &samples(case, extent, 17));
    }
    for (index, predictor) in Predictor::ALL.into_iter().enumerate() {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::implicit(1024, predictor).unwrap()),
                predictor: Predictor::Weighted,
                local_transforms: SQUEEZES[index % 5].clone().into(),
                tree_mode: TREES[index % 2],
                ..Default::default()
            },
        );
        for bits in [16, 32] {
            let case = case(FORMATS[index % 4], bits, SampleKind::Float);
            let extent = Extent2d::new(17, 9);
            check(&rig, &encoder, case, extent, &samples(case, extent, 17));
        }
    }
}

#[test]
fn implicit_palette_animation_preserves_exact_words_crops_and_reference_composition() {
    let rig = Rig::new();
    for (index, predictor) in [Predictor::Weighted, Predictor::AverageAll]
        .into_iter()
        .enumerate()
    {
        let group_size = Size::ALL[index];
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::implicit(4096, predictor).unwrap()),
                predictor: Predictor::Weighted,
                local_transforms: SQUEEZES[index + 3].clone().into(),
                group_size,
                tree_mode: TREES[index],
                ..Default::default()
            },
        );
        for (format, kind, bits) in [
            (LosslessModularFormat::Rgba, SampleKind::Unsigned, 31),
            (LosslessModularFormat::GrayAlpha, SampleKind::Float, 32),
        ] {
            groups::animation::check_animation_words_with_oracle(
                &rig,
                &encoder,
                group_size,
                format,
                kind,
                bits,
                delta::check_frame_oracles,
            );
        }
        groups::animation::check_cropped_frames(&rig, &encoder, group_size);
    }
}

#[test]
fn implicit_palette_capacity_failure_publishes_no_resident_or_streamed_output() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Gray, 32, SampleKind::Float);
    for (index, squeeze) in SQUEEZES.into_iter().enumerate() {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(
                    Palette::implicit(
                        1,
                        [
                            Predictor::Zero,
                            Predictor::Weighted,
                            Predictor::West,
                            Predictor::Gradient,
                            Predictor::AverageAll,
                        ][index],
                    )
                    .unwrap(),
                ),
                local_transforms: squeeze.into(),
                lz77: if index % 2 == 0 {
                    Lz77::ZeroRuns
                } else {
                    Lz77::Greedy
                },
                ..Default::default()
            },
        );
        for extent in [Extent2d::new(7, 3), Extent2d::new(256 * 33 + 7, 3)] {
            let expected: Vec<_> = (0..extent.height)
                .flat_map(|y| {
                    (0..extent.width).map(move |x| {
                        if extent.width > 7 && x < 256 * 32 {
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
                            ][((x + y) % 7) as usize]
                        }
                    })
                })
                .collect();
            let input = upload(&rig.context, &case, extent, &expected, 0);
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
            let valid = vec![0; extent.area().unwrap()];
            let encoded = encoder
                .encode(upload(&rig.context, &case, extent, &valid, 0))
                .unwrap();
            delta::check_frame_oracles(&encoded, &[&valid], &case);
        }
    }
}

#[test]
fn implicit_palette_hash_obeys_exact_admission_cancellation_and_pool_reuse() {
    let rig = Rig::new();
    for (index, group_size) in Size::ALL.into_iter().enumerate() {
        check_lifetime(
            &rig,
            LosslessModularConfig {
                entropy: Default::default(),
                palette: Some(Palette::implicit(4096, Predictor::Weighted).unwrap()),
                local_transforms: SQUEEZES[index + 1].clone().into(),
                group_size,
                tree_mode: TREES[index % 2],
                predictor: Predictor::Weighted,
                weighted_predictor: Weighted::new([31, 0, 17, 3, 11, 31, 1], [0, 15, 7, 12])
                    .unwrap(),
                lz77: Lz77::Greedy,
                color_transform: Transform::LocalRct(Rct::new(41).unwrap()),
            },
            delta::check_frame_oracles,
        );
    }
}
