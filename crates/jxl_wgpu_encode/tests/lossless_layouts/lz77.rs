use super::*;
use jxl_wgpu_encode::{
    LosslessModularColorTransform as Transform, LosslessModularConfig,
    LosslessModularGroupSize as Size, LosslessModularLz77 as Lz77,
    LosslessModularPredictor as Predictor, LosslessModularRctType as Rct,
    LosslessModularWeightedPredictor as Weighted,
};

fn check(
    rig: &Rig,
    encoder: &LosslessModularEncoder,
    case: &Case,
    extent: Extent2d,
    expected: &[u32],
) -> usize {
    let input = upload(&rig.context, case, extent, expected, 4099);
    let plan = encoder.memory_plan(&input).unwrap();
    if !plan.streaming {
        let scratch: u64 = plan
            .group_grid
            .ordered_groups()
            .map(|group| {
                let pixels = group.width * group.height;
                4 * (2 * u64::from(pixels) + u64::from(pixels.next_power_of_two().min(1 << 16)))
                    * u64::from(plan.channel_count)
            })
            .sum();
        assert_eq!(
            plan.lz77_scratch_bytes,
            if encoder.config().lz77 == Lz77::Greedy {
                scratch
            } else {
                0
            }
        );
    }
    let encoded = pollster::block_on(encoder.submit_container(input).unwrap()).unwrap();
    assert_eq!(
        encoded,
        encoder
            .encode_container(upload(&rig.context, &case.canonical(), extent, expected, 0))
            .unwrap()
    );
    check_oracles(&encoded, expected, case);
    color::check_numeric(rig, &encoded, &[expected.to_vec()], case);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    encoded.len()
}

#[test]
fn greedy_lz77_preserves_every_predictor_with_integer_ieee_and_rct_words() {
    let rig = Rig::new();
    let formats = [
        LosslessModularFormat::Gray,
        LosslessModularFormat::GrayAlpha,
        LosslessModularFormat::Rgb,
        LosslessModularFormat::Rgba,
    ];
    for predictor in Predictor::ALL {
        for (tree, tree_mode) in TREES.into_iter().enumerate() {
            for (index, (kind, bits)) in [
                (SampleKind::Unsigned, 1),
                (SampleKind::Unsigned, 8),
                (SampleKind::Unsigned, 16),
                (SampleKind::Unsigned, 31),
                (SampleKind::Float, 16),
                (SampleKind::Float, 32),
            ]
            .into_iter()
            .enumerate()
            {
                let group_size = Size::ALL[(predictor.value() as usize + index) % 4];
                let format = formats[(index + tree * 2) % 4];
                let rct = Rct::new((predictor.value() * 3 + index as u32) % 42).unwrap();
                let encoder = LosslessModularEncoder::with_config(
                    rig.context.clone(),
                    LosslessModularConfig {
                        lz77: Lz77::Greedy,
                        predictor,
                        group_size,
                        tree_mode,
                        weighted_predictor: Weighted::new(
                            [31, 0, 3, 7, 11, 17, 31],
                            [0, 15, 7, 12],
                        )
                        .unwrap(),
                        color_transform: if format.color_channel_count() == 1 {
                            Transform::None
                        } else if tree == 0 {
                            Transform::GlobalRct(rct)
                        } else {
                            Transform::LocalRct(rct)
                        },
                    },
                );
                let case = Case {
                    format,
                    bits,
                    kind,
                    storage: [Storage::Packed, Storage::Planar, Storage::Split][index % 3],
                    reversed: true,
                    byte_order: ByteOrder::Big,
                    shifted: true,
                };
                let edge = group_size.dimension();
                let extent = match index {
                    0 => Extent2d::new(1, 7),
                    1 => Extent2d::new(edge + 1, 9),
                    2 => Extent2d::new(17, 1),
                    3 => Extent2d::new(edge + 1, 3),
                    4 => Extent2d::new(1, edge * 8 + 1),
                    _ => Extent2d::new(33, 9),
                };
                let mut expected = case.samples(extent);
                let period = 17 * format.channel_count() as usize;
                for index in period..expected.len() {
                    expected[index] = expected[index % period];
                }
                check(&rig, &encoder, &case, extent, &expected);
            }
        }
    }
}

#[test]
fn general_repetition_improves_on_zero_runs_and_preserves_short_tails() {
    let rig = Rig::new();
    let case = Case {
        format: LosslessModularFormat::Gray,
        bits: 8,
        kind: SampleKind::Unsigned,
        storage: Storage::Packed,
        reversed: false,
        byte_order: ByteOrder::Native,
        shifted: false,
    };
    let encoders = [Lz77::ZeroRuns, Lz77::Greedy].map(|lz77| {
        LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                lz77,
                predictor: Predictor::Zero,
                ..Default::default()
            },
        )
    });
    for length in 1..=18 {
        let extent = Extent2d::new(length, 1);
        check(&rig, &encoders[1], &case, extent, &vec![7; length as usize]);
    }
    let extent = Extent2d::new(127, 65);
    let expected: Vec<_> = (0..extent.width * extent.height)
        .map(|index| 1 + ((index % 37) * 193) % 255)
        .collect();
    let sizes = encoders
        .each_ref()
        .map(|encoder| check(&rig, encoder, &case, extent, &expected));
    assert!(sizes[1] * 2 < sizes[0], "{sizes:?}");
}

#[test]
fn greedy_lz77_uses_long_distances_and_maximum_group_match_lengths() {
    let rig = Rig::new();
    let encoder = LosslessModularEncoder::with_config(
        rig.context.clone(),
        LosslessModularConfig {
            lz77: Lz77::Greedy,
            predictor: Predictor::Zero,
            group_size: Size::Pixels1024,
            ..Default::default()
        },
    );
    let case = Case {
        format: LosslessModularFormat::Gray,
        bits: 8,
        kind: SampleKind::Unsigned,
        storage: Storage::Planar,
        reversed: false,
        byte_order: ByteOrder::Big,
        shifted: true,
    };
    let extent = Extent2d::new(1024, 1024);
    let mut expected = vec![0; 1024 * 1024];
    let end = expected.len() - 7;
    expected[..7].copy_from_slice(&[1, 3, 7, 15, 31, 63, 127]);
    expected.copy_within(..7, end);
    check(&rig, &encoder, &case, extent, &expected);
}

#[test]
fn greedy_lz77_keeps_streamed_admission_cancellation_and_reuse() {
    let rig = Rig::new();
    for (index, group_size) in Size::ALL.into_iter().enumerate() {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                lz77: Lz77::Greedy,
                group_size,
                tree_mode: TREES[index % 2],
                predictor: if index % 2 == 0 {
                    Predictor::Weighted
                } else {
                    Predictor::AverageAll
                },
                weighted_predictor: Weighted::new([31; 7], [15; 4]).unwrap(),
                color_transform: Transform::LocalRct(Rct::new(41).unwrap()),
            },
        );
        groups::lifetime::check_admission(
            &rig,
            &encoder,
            group_size,
            if index % 2 == 0 {
                SampleKind::Unsigned
            } else {
                SampleKind::Float
            },
        );
    }
}

#[test]
fn greedy_lz77_keeps_animation_words_crops_and_independent_references() {
    let rig = Rig::new();
    for (index, predictor) in [Predictor::Weighted, Predictor::AverageAll]
        .into_iter()
        .enumerate()
    {
        let group_size = Size::ALL[index];
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                lz77: Lz77::Greedy,
                group_size,
                predictor,
                tree_mode: TREES[index],
                weighted_predictor: Weighted::new([31; 7], [0, 15, 1, 8]).unwrap(),
                color_transform: Transform::LocalRct(Rct::new(41).unwrap()),
            },
        );
        groups::animation::check_animation_words(
            &rig,
            &encoder,
            group_size,
            LosslessModularFormat::Rgba,
            SampleKind::Float,
            32,
        );
        groups::animation::check_cropped_frames(&rig, &encoder, group_size);
    }
}
