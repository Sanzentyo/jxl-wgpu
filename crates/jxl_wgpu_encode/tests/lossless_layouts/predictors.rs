use super::*;
use jxl_wgpu_encode::{
    LosslessModularColorTransform as Transform, LosslessModularConfig,
    LosslessModularGroupSize as Size, LosslessModularPredictor as Predictor,
    LosslessModularRctType as Rct, LosslessModularWeightedPredictor as Weighted,
};

const FORMATS: [LosslessModularFormat; 4] = [
    LosslessModularFormat::Gray,
    LosslessModularFormat::GrayAlpha,
    LosslessModularFormat::Rgb,
    LosslessModularFormat::Rgba,
];

fn check(rig: &Rig, encoder: &LosslessModularEncoder, case: &Case, extent: Extent2d) {
    let expected = case.samples(extent);
    check_samples(rig, encoder, case, extent, &expected);
}

fn check_samples(
    rig: &Rig,
    encoder: &LosslessModularEncoder,
    case: &Case,
    extent: Extent2d,
    expected: &[u32],
) -> usize {
    let input = upload(&rig.context, case, extent, expected, 4099);
    let plan = encoder.memory_plan(&input).unwrap();
    let config = encoder.config();
    assert!(!plan.streaming);
    assert_eq!(
        plan.weighted_predictor_scratch_bytes,
        if config.predictor == Predictor::Weighted {
            plan.group_grid
                .ordered_groups()
                .map(|group| u64::from(group.width) * 20 * u64::from(plan.channel_count))
                .sum()
        } else {
            0
        }
    );
    let encoded = pollster::block_on(encoder.submit_container(input).unwrap()).unwrap();
    assert_eq!(
        encoded,
        encoder
            .encode_container(upload(&rig.context, &case.canonical(), extent, expected, 0))
            .unwrap()
    );
    for header in modular_integer::local_modular_headers(&encoded, 0) {
        assert_eq!(header.global_tree, config.tree_mode == TREES[0]);
        assert_eq!(
            header.coefficients,
            config.weighted_predictor.coefficients()
        );
        assert_eq!(header.max_weights, config.weighted_predictor.max_weights());
        assert_eq!(
            header.rct,
            match config.color_transform {
                Transform::LocalRct(rct) => Some(rct.value()),
                _ => None,
            }
        );
    }
    check_oracles(&encoded, expected, case);
    color::check_numeric(rig, &encoded, &[expected.to_vec()], case);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    encoded.len()
}

fn matrix(kind: SampleKind, depths: &[u8]) {
    let rig = Rig::new();
    for predictor in Predictor::ALL {
        for (tree, tree_mode) in TREES.into_iter().enumerate() {
            for (index, &bits) in depths.iter().enumerate() {
                let group_size = Size::ALL[(predictor.value() as usize + index) % 4];
                let format = FORMATS[(index + tree * 2) % 4];
                let value = (predictor.value() * 3 + index as u32) % 42;
                let color_transform = if format.color_channel_count() == 1 {
                    Transform::None
                } else if tree == 0 {
                    Transform::GlobalRct(Rct::new(value).unwrap())
                } else {
                    Transform::LocalRct(Rct::new(value).unwrap())
                };
                let encoder = LosslessModularEncoder::with_config(
                    rig.context.clone(),
                    LosslessModularConfig {
                        predictor,
                        group_size,
                        tree_mode,
                        color_transform,
                        ..Default::default()
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
                    0 => Extent2d::new(1, 17),
                    1 => Extent2d::new(edge + 1, 5),
                    2 => Extent2d::new(17, 1),
                    3 => Extent2d::new(33, 9),
                    4 => Extent2d::new(1, edge * 8 + 1),
                    _ => Extent2d::new(edge + 1, 3),
                };
                check(&rig, &encoder, &case, extent);
            }
        }
    }
}

#[test]
fn every_predictor_preserves_integer_words_across_group_and_rct_boundaries() {
    matrix(SampleKind::Unsigned, &[1, 8, 14, 16, 24, 31]);
}

#[test]
fn every_predictor_preserves_ieee_special_words() {
    matrix(SampleKind::Float, &[16, 32]);
}

#[test]
fn custom_weighted_parameters_preserve_full_words_and_every_integer_depth() {
    let rig = Rig::new();
    let mut parameters = vec![
        Weighted::default(),
        Weighted::new([0; 7], [0; 4]).unwrap(),
        Weighted::new([31; 7], [15; 4]).unwrap(),
    ];
    for index in 0..11 {
        let mut coefficients = [0; 7];
        let mut weights = [0; 4];
        if index < 7 {
            coefficients[index] = 31;
        } else {
            weights[index - 7] = 15;
        }
        parameters.push(Weighted::new(coefficients, weights).unwrap());
    }
    for (index, weighted_predictor) in parameters.into_iter().enumerate() {
        for (tree, tree_mode) in TREES.into_iter().enumerate() {
            let encoder = LosslessModularEncoder::with_config(
                rig.context.clone(),
                LosslessModularConfig {
                    predictor: Predictor::Weighted,
                    weighted_predictor,
                    group_size: Size::Pixels128,
                    tree_mode,
                    color_transform: Transform::LocalRct(
                        Rct::new((index as u32 * 3) % 42).unwrap(),
                    ),
                    ..Default::default()
                },
            );
            let case = Case {
                format: LosslessModularFormat::Rgba,
                bits: if tree == 0 { 31 } else { 32 },
                kind: if tree == 0 {
                    SampleKind::Unsigned
                } else {
                    SampleKind::Float
                },
                storage: Storage::Split,
                reversed: true,
                byte_order: ByteOrder::Big,
                shifted: true,
            };
            check(&rig, &encoder, &case, Extent2d::new(129, 9));
        }
    }
    for bits in 1..=31 {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                predictor: Predictor::Weighted,
                weighted_predictor: Weighted::new([31; 7], [15; 4]).unwrap(),
                ..Default::default()
            },
        );
        let case = Case {
            format: FORMATS[bits as usize % 4],
            bits,
            kind: SampleKind::Unsigned,
            storage: Storage::Planar,
            reversed: true,
            byte_order: ByteOrder::Big,
            shifted: true,
        };
        check(&rig, &encoder, &case, Extent2d::new(33, 17));
    }
}

#[test]
fn weighted_row_state_survives_streaming_and_retires_after_cancellation() {
    let rig = Rig::new();
    for (index, group_size) in Size::ALL.into_iter().enumerate() {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                predictor: Predictor::Weighted,
                weighted_predictor: Weighted::new([31, 0, 17, 3, 11, 31, 1], [0, 15, 7, 12])
                    .unwrap(),
                group_size,
                tree_mode: TREES[index % 2],
                color_transform: Transform::LocalRct(Rct::new(41).unwrap()),
                ..Default::default()
            },
        );
        let extent = Extent2d::new(group_size.dimension() * 17, 1);
        let case = Case {
            format: LosslessModularFormat::Rgba,
            bits: 31,
            kind: SampleKind::Unsigned,
            storage: Storage::Packed,
            reversed: false,
            byte_order: ByteOrder::Native,
            shifted: false,
        };
        let plan = encoder
            .memory_plan(upload(
                &rig.context,
                &case,
                extent,
                &case.samples(extent),
                0,
            ))
            .unwrap();
        assert!(plan.streaming);
        assert_eq!(
            plan.weighted_predictor_scratch_bytes,
            16 * 4 * 20 * u64::from(group_size.dimension())
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
fn selected_predictors_keep_animation_words_crops_and_references() {
    let rig = Rig::new();
    for (index, predictor) in [Predictor::Weighted, Predictor::AverageAll]
        .into_iter()
        .enumerate()
    {
        let group_size = Size::ALL[index];
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                predictor,
                group_size,
                tree_mode: TREES[index],
                weighted_predictor: Weighted::new([31; 7], [0, 15, 1, 8]).unwrap(),
                color_transform: Transform::LocalRct(Rct::new(41).unwrap()),
                ..Default::default()
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

#[test]
fn explicit_prediction_can_improve_compression_over_gradient() {
    let rig = Rig::new();
    let extent = Extent2d::new(127, 65);
    let case = Case {
        format: LosslessModularFormat::Gray,
        bits: 8,
        kind: SampleKind::Unsigned,
        storage: Storage::Packed,
        reversed: false,
        byte_order: ByteOrder::Native,
        shifted: false,
    };
    // Each row is the preceding row shifted left, with edge replication. Its
    // independent source formula makes NE prediction useful on a nonconstant image.
    let samples: Vec<_> = (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).map(move |x| {
                let index = (x + y).min(extent.width - 1);
                index.wrapping_mul(0x9e37_79b9).rotate_left(13) & 255
            })
        })
        .collect();
    let mut sizes = [0usize; 14];
    for predictor in Predictor::ALL {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                predictor,
                // Prove custom header retention even when the tree does not use WP.
                weighted_predictor: Weighted::new([0; 7], [15; 4]).unwrap(),
                ..Default::default()
            },
        );
        sizes[predictor.value() as usize] = check_samples(&rig, &encoder, &case, extent, &samples);
    }
    assert!(
        sizes[Predictor::NorthEast.value() as usize] < sizes[Predictor::Gradient.value() as usize],
        "{sizes:?}"
    );
}
