use super::*;
use jxl_bitstream::U;

// Only reads local transform declarations with independent frame/TOC and bit readers.
// Sample validation uses native original words, not jxl-oxide's delta inverse.
pub(super) fn local_counts(encoded: &[u8]) -> Vec<(u32, u32, u32)> {
    let image = jxl_oxide::JxlImage::read_with_defaults(encoded).unwrap();
    let frame = image.frame(0).unwrap();
    assert!(!frame.toc().is_single_entry());
    (0..frame.header().num_groups())
        .map(|group| {
            let mut stream = frame.pass_group_bitstream(0, group).unwrap().unwrap();
            assert!(!stream.partial);
            let bits = &mut stream.bitstream;
            let _global_tree = bits.read_bool().unwrap();
            if !bits.read_bool().unwrap() {
                for _ in 0..7 {
                    bits.read_bits(5).unwrap();
                }
                for _ in 0..4 {
                    bits.read_bits(4).unwrap();
                }
            }
            let count = bits.read_u32(0, 1, 2 + U(4), 18 + U(8)).unwrap();
            assert!((1..=3).contains(&count));
            for _ in 0..count {
                let transform = bits.read_bits(2).unwrap();
                assert_eq!(
                    bits.read_u32(U(3), 8 + U(6), 72 + U(10), 1096 + U(13))
                        .unwrap(),
                    0
                );
                if transform == 0 {
                    bits.read_u32(6, U(2), 2 + U(4), 10 + U(6)).unwrap();
                } else {
                    assert_eq!(transform, 1);
                    bits.read_u32(1, 3, 4, 1 + U(13)).unwrap();
                    let colors = bits
                        .read_u32(U(8), 256 + U(10), 1280 + U(12), 5376 + U(16))
                        .unwrap();
                    let deltas = bits
                        .read_u32(0, 1 + U(8), 257 + U(10), 1281 + U(16))
                        .unwrap();
                    let predictor = bits.read_bits(4).unwrap();
                    return (colors, deltas, predictor);
                }
            }
            panic!("missing palette");
        })
        .collect()
}

#[test]
fn mixed_palette_preserves_every_predictor_rct_precision_and_squeeze_order() {
    let rig = Rig::new();
    for value in 0..42 {
        let group_size = Size::ALL[value as usize % 4];
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(
                    Palette::mixed(3, 4096, Predictor::ALL[value as usize % 14]).unwrap(),
                ),
                predictor: Predictor::ALL[(value as usize + 7) % 14],
                weighted_predictor: Weighted::new([31, 0, 17, 3, 11, 31, 1], [0, 15, 7, 12])
                    .unwrap(),
                squeeze: SQUEEZES[value as usize % 5],
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
                palette: Some(Palette::mixed(2, 1024, predictor).unwrap()),
                predictor: Predictor::Weighted,
                weighted_predictor: Weighted::new([31, 0, 17, 3, 11, 31, 1], [0, 15, 7, 12])
                    .unwrap(),
                squeeze: SQUEEZES[index % 5],
                tree_mode: TREES[index % 2],
                ..Default::default()
            },
        );
        for bits in [16, 32] {
            let case = case(
                [
                    LosslessModularFormat::Gray,
                    LosslessModularFormat::GrayAlpha,
                    LosslessModularFormat::Rgba,
                ][index % 3],
                bits,
                SampleKind::Float,
            );
            let extent = Extent2d::new(17, 9);
            check(&rig, &encoder, case, extent, &samples(case, extent, 17));
        }
    }
}

#[test]
fn mixed_palette_separates_identical_color_and_delta_words_and_reuses_absolute_entries() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Gray, 8, SampleKind::Unsigned);
    let extent = Extent2d::new(257, 1);
    let expected: Vec<_> = (0..257)
        .map(|x| [7, 14, 7, 21, 7, 28, 7, 14][x % 8])
        .collect();
    for policy in [
        Palette::new(1).unwrap(),
        Palette::deltas(3, Predictor::West).unwrap(),
    ] {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(policy),
                ..Default::default()
            },
        );
        assert!(matches!(
            encoder.encode(upload(&rig.context, &case, extent, &expected, 0)),
            Err(EncodeError::Backend(BackendError::ModularPaletteOverflow))
        ));
    }
    for squeeze in SQUEEZES {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::mixed(1, 3, Predictor::West).unwrap()),
                squeeze,
                ..Default::default()
            },
        );
        let encoded = encoder
            .encode(upload(&rig.context, &case, extent, &expected, 0))
            .unwrap();
        assert_eq!(local_counts(&encoded), [(1, 3, 1), (1, 0, 1)]);
        delta::check_frame_oracles(&encoded, &[&expected], &case);
        color::check_numeric(&rig, &encoded, std::slice::from_ref(&expected), &case);
    }
}

#[test]
fn mixed_palette_covers_both_count_buckets_and_the_combined_maximum() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Gray, 31, SampleKind::Unsigned);
    for (colors, deltas) in [
        (1u32, 1u32),
        (255, 256),
        (256, 257),
        (1279, 1280),
        (1280, 1281),
        (5375, 1),
        (5376, 1),
        (Palette::MAX_COLORS, Palette::MAX_DELTAS),
    ] {
        let entries = colors + deltas;
        let extent = Extent2d::new(1025, entries.div_ceil(1024));
        let expected: Vec<_> = (0..extent.height)
            .flat_map(|y| {
                (0..extent.width).map(move |x| {
                    if x == 1024 {
                        0
                    } else {
                        0x7fff_ffff - (y * 1024 + x) % entries
                    }
                })
            })
            .collect();
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::mixed(colors, deltas, Predictor::Zero).unwrap()),
                group_size: Size::Pixels1024,
                ..Default::default()
            },
        );
        let encoded = encoder
            .encode_container(upload(&rig.context, &case, extent, &expected, 0))
            .unwrap();
        assert_eq!(local_counts(&encoded), [(colors, deltas, 0), (1, 0, 0)]);
        delta::check_frame_oracles(&encoded, &[&expected], &case);
        color::check_numeric(&rig, &encoded, &[expected], &case);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn mixed_palette_animation_retains_original_words_and_reference_composition() {
    let rig = Rig::new();
    for (index, predictor) in [Predictor::Weighted, Predictor::AverageAll]
        .into_iter()
        .enumerate()
    {
        let group_size = Size::ALL[index];
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::mixed(3, 4096, predictor).unwrap()),
                predictor: Predictor::Weighted,
                squeeze: SQUEEZES[index + 3],
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
fn mixed_palette_overflow_never_publishes_resident_or_late_streamed_output() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Gray, 32, SampleKind::Float);
    for (index, squeeze) in SQUEEZES.into_iter().enumerate() {
        let predictor = [
            Predictor::Zero,
            Predictor::Weighted,
            Predictor::West,
            Predictor::Gradient,
            Predictor::AverageAll,
        ][index];
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::mixed(1, 1, predictor).unwrap()),
                squeeze,
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
fn mixed_palette_scratch_obeys_admission_cancellation_and_reuse() {
    let rig = Rig::new();
    for (index, group_size) in Size::ALL.into_iter().enumerate() {
        check_lifetime(
            &rig,
            LosslessModularConfig {
                palette: Some(Palette::mixed(3, 4096, Predictor::Weighted).unwrap()),
                squeeze: SQUEEZES[index + 1],
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
