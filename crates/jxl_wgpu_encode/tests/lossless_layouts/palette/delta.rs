use super::*;

#[test]
fn delta_palette_scratch_obeys_exact_admission_cancellation_and_pool_reuse() {
    let rig = Rig::new();
    for (index, group_size) in Size::ALL.into_iter().enumerate() {
        check_lifetime(
            &rig,
            LosslessModularConfig {
                palette: Some(Palette::deltas(4096, Predictor::Weighted).unwrap()),
                local_transforms: SQUEEZES[index + 1].clone().into(),
                group_size,
                tree_mode: TREES[index % 2],
                predictor: Predictor::Weighted,
                weighted_predictor: Weighted::new([31, 0, 17, 3, 11, 31, 1], [0, 15, 7, 12])
                    .unwrap(),
                lz77: Lz77::Greedy,
                color_transform: Transform::LocalRct(Rct::new(41).unwrap()),
            },
            check_frame_oracles,
        );
    }
}

pub(crate) fn check_frame_oracles(encoded: &[u8], expected: &[&[u32]], case: &Case) -> Vec<f32> {
    let original = jxl_test_support::oracles::modular_words::original_frames(encoded);
    assert_eq!(original.len(), expected.len());
    for (frame, expected) in original.iter().zip(expected) {
        assert_eq!(frame.bits, u32::from(case.bits));
        assert_eq!(
            frame.exponent_bits,
            if case.kind == SampleKind::Unsigned {
                0
            } else if case.bits == 16 {
                5
            } else {
                8
            }
        );
        assert_eq!(
            u64::from(frame.width)
                * u64::from(frame.height)
                * u64::from(case.format.channel_count()),
            expected.len() as u64
        );
    }
    let native = native(encoded);
    let planes: Vec<_> = original.into_iter().map(|frame| frame.planes).collect();
    check_frame_samples_with_planes(expected, case, &native, &planes);
    native
}

fn check_oracles(encoded: &[u8], expected: &[u32], case: &Case) -> Vec<f32> {
    check_frame_oracles(encoded, &[expected], case)
}

#[test]
fn delta_palette_composes_every_predictor_rct_precision_and_squeeze_policy() {
    let rig = Rig::new();
    for value in 0..42 {
        let group_size = Size::ALL[value as usize % 4];
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::deltas(4096, Predictor::ALL[value as usize % 14]).unwrap()),
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
            if value % 2 == 0 {
                LosslessModularFormat::Rgb
            } else {
                LosslessModularFormat::Rgba
            },
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
                palette: Some(Palette::deltas(1024, predictor).unwrap()),
                predictor: Predictor::Weighted,
                weighted_predictor: Weighted::new([31, 0, 17, 3, 11, 31, 1], [0, 15, 7, 12])
                    .unwrap(),
                local_transforms: SQUEEZES[index % 5].clone().into(),
                tree_mode: TREES[index % 2],
                ..Default::default()
            },
        );
        for bits in [16, 32] {
            let case = case(
                if index % 2 == 0 {
                    LosslessModularFormat::GrayAlpha
                } else {
                    LosslessModularFormat::Rgba
                },
                bits,
                SampleKind::Float,
            );
            let extent = Extent2d::new(17, 9);
            check(&rig, &encoder, case, extent, &samples(case, extent, 17));
        }
    }
}

#[test]
fn delta_palette_encodes_smooth_high_color_images_with_a_small_dictionary() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Gray, 31, SampleKind::Unsigned);
    let extent = Extent2d::new(257, 17);
    let expected: Vec<_> = (0..extent.height)
        .flat_map(|y| (0..extent.width).map(move |x| 1_900_000_000 + x * 1234 + y * 2345))
        .collect();
    let exact = LosslessModularEncoder::with_config(
        rig.context.clone(),
        LosslessModularConfig {
            palette: Some(Palette::new(4).unwrap()),
            ..Default::default()
        },
    );
    let error = exact
        .encode(upload(&rig.context, &case, extent, &expected, 0))
        .unwrap_err();
    assert!(matches!(
        error,
        EncodeError::Backend(BackendError::ModularPaletteOverflow)
    ));
    let delta = LosslessModularEncoder::with_config(
        rig.context.clone(),
        LosslessModularConfig {
            palette: Some(Palette::deltas(4, Predictor::Gradient).unwrap()),
            ..Default::default()
        },
    );
    let palette_bytes = check(&rig, &delta, case, extent, &expected);
    let plain = LosslessModularEncoder::new(rig.context.clone())
        .encode_container(upload(&rig.context, &case, extent, &expected, 0))
        .unwrap();
    check_oracles(&plain, &expected, &case);
    assert!(
        palette_bytes < plain.len(),
        "delta {palette_bytes}, plain {}",
        plain.len()
    );
}

#[test]
fn delta_palette_covers_every_count_bucket_through_the_maximum() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Gray, 31, SampleKind::Unsigned);
    for entries in [1, 256, 257, 1280, 1281, Palette::MAX_DELTAS] {
        let extent = Extent2d::new(entries.min(1024), entries.div_ceil(1024));
        let expected: Vec<_> = (0..extent.area().unwrap())
            .map(|index| 0x7fff_ffff - index as u32 % entries)
            .collect();
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::deltas(entries, Predictor::Zero).unwrap()),
                group_size: Size::ALL[3],
                ..Default::default()
            },
        );
        eprintln!("delta palette entry count {entries}");
        let encoded = encoder
            .encode_container(upload(&rig.context, &case, extent, &expected, 0))
            .unwrap();
        check_oracles(&encoded, &expected, &case);
        color::check_numeric(&rig, &encoded, &[expected], &case);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn delta_palette_animation_retains_ieee_words_and_independent_references() {
    let rig = Rig::new();
    for (index, predictor) in [Predictor::Weighted, Predictor::AverageAll]
        .into_iter()
        .enumerate()
    {
        let group_size = Size::ALL[index];
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::deltas(4096, predictor).unwrap()),
                predictor: Predictor::Weighted,
                local_transforms: SQUEEZES[index + 3].clone().into(),
                group_size,
                tree_mode: TREES[index],
                ..Default::default()
            },
        );
        groups::animation::check_animation_words_with_oracle(
            &rig,
            &encoder,
            group_size,
            LosslessModularFormat::Rgba,
            SampleKind::Unsigned,
            31,
            check_frame_oracles,
        );
        groups::animation::check_animation_words_with_oracle(
            &rig,
            &encoder,
            group_size,
            LosslessModularFormat::GrayAlpha,
            SampleKind::Float,
            32,
            check_frame_oracles,
        );
        groups::animation::check_cropped_frames(&rig, &encoder, group_size);
    }
}

#[test]
fn delta_palette_overflow_releases_resident_and_late_streamed_jobs() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Gray, 32, SampleKind::Float);
    for (index, squeeze) in SQUEEZES.into_iter().enumerate() {
        let predictor = [
            Predictor::Weighted,
            Predictor::Gradient,
            Predictor::West,
            Predictor::North,
            Predictor::AverageAll,
        ][index];
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::deltas(1, predictor).unwrap()),
                local_transforms: squeeze.into(),
                lz77: if index % 2 == 0 {
                    Lz77::ZeroRuns
                } else {
                    Lz77::Greedy
                },
                ..Default::default()
            },
        );
        for extent in [Extent2d::new(2, 2), Extent2d::new(256 * 33 + 1, 2)] {
            let expected: Vec<_> = (0..extent.height)
                .flat_map(|y| {
                    (0..extent.width).map(move |x| {
                        if extent.width > 2 && x < 256 * 32 || (x + y) % 2 == 0 {
                            0
                        } else {
                            0x8000_0000
                        }
                    })
                })
                .collect();
            let input = upload(&rig.context, &case, extent, &expected, 0);
            assert_eq!(
                encoder.memory_plan(&input).unwrap().streaming,
                extent.width > 2
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
            check_oracles(&encoded, &valid, &case);
        }
    }
}
