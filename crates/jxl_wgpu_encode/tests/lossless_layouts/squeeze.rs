use super::*;
mod program;
mod selection;
mod sequence;
use jxl_wgpu_encode::{
    BackendError, LosslessModularColorTransform as Transform, LosslessModularConfig,
    LosslessModularGroupSize as Size, LosslessModularLz77 as Lz77,
    LosslessModularPredictor as Predictor, LosslessModularRctType as Rct,
    LosslessModularSqueeze as Squeeze, LosslessModularWeightedPredictor as Weighted,
};

const MODES: [Squeeze; 4] = [
    Squeeze::Horizontal,
    Squeeze::Vertical,
    Squeeze::HorizontalThenVertical,
    Squeeze::VerticalThenHorizontal,
];

fn case(format: LosslessModularFormat, bits: u8, kind: SampleKind) -> Case {
    Case {
        format,
        bits,
        kind,
        storage: Storage::Split,
        reversed: true,
        byte_order: ByteOrder::Big,
        shifted: true,
    }
}

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
) {
    eprintln!("{:?}, {extent:?}, {case:?}", encoder.config());
    let source = upload(&rig.context, case, extent, expected, 4099);
    let plan = encoder.memory_plan(&source).unwrap();
    let factor = match (
        encoder
            .config()
            .local_transforms
            .squeeze_policy()
            .unwrap()
            .clone(),
        extent.width > 1,
        extent.height > 1,
    ) {
        (Squeeze::None, _, _)
        | (Squeeze::Horizontal, false, _)
        | (Squeeze::Vertical, _, false)
        | (_, false, false) => 1,
        (Squeeze::Horizontal | Squeeze::Vertical, _, _) | (_, false, _) | (_, _, false) => 2,
        _ => 4,
    };
    assert_eq!(plan.channel_count, factor * case.format.channel_count());
    let encoded = pollster::block_on(encoder.submit_container(source).unwrap()).unwrap();
    assert_eq!(
        encoded,
        encoder
            .encode_container(upload(&rig.context, &case.canonical(), extent, expected, 0))
            .unwrap()
    );
    check_oracles(&encoded, expected, case);
    color::check_numeric(rig, &encoded, &[expected.to_vec()], case);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn squeeze_composes_with_all_rcts_predictors_group_sizes_and_entropy_policies() {
    let rig = Rig::new();
    for value in 0..42 {
        let group_size = Size::ALL[value as usize % 4];
        let config = LosslessModularConfig {
            entropy: Default::default(),
            palette: None,
            local_transforms: MODES[value as usize % 4].clone().into(),
            group_size,
            tree_mode: TREES[value as usize % 2],
            color_transform: if value % 2 == 0 {
                Transform::GlobalRct(Rct::new(value).unwrap())
            } else {
                Transform::LocalRct(Rct::new(value).unwrap())
            },
            predictor: Predictor::ALL[value as usize % 14],
            weighted_predictor: Weighted::new([31, 0, 17, 3, 11, 31, 1], [0, 15, 7, 12]).unwrap(),
            lz77: if value % 3 == 0 {
                Lz77::ZeroRuns
            } else {
                Lz77::Greedy
            },
        };
        let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config);
        let extent = Extent2d::new(group_size.dimension() + 1, 5);
        let mut case = case(LosslessModularFormat::Rgba, 16, SampleKind::Unsigned);
        case.storage = [Storage::Packed, Storage::Planar, Storage::Split][value as usize % 3];
        check(&rig, &encoder, &case, extent);
    }
}

#[test]
fn squeeze_keeps_every_integer_precision_and_representable_ieee_words() {
    let rig = Rig::new();
    let encoders = MODES.map(|squeeze| {
        LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                local_transforms: squeeze.into(),
                color_transform: Transform::None,
                ..Default::default()
            },
        )
    });
    for bits in 1..=31 {
        let case = case(LosslessModularFormat::GrayAlpha, bits, SampleKind::Unsigned);
        let extent = Extent2d::new(19, 11);
        let maximum = u32::MAX >> (32 - bits);
        // Keep the full source precision while bounding differences, not intermediate sums.
        let samples: Vec<_> = (0..extent.area().unwrap() * 2)
            .map(|i| maximum - (i as u32 * 149).rotate_left(7) % (maximum.min(8191) + 1))
            .collect();
        check_samples(&rig, &encoders[bits as usize % 4], &case, extent, &samples);
    }
    for (index, encoder) in encoders.iter().enumerate() {
        check(
            &rig,
            encoder,
            &case(LosslessModularFormat::Rgba, 16, SampleKind::Float),
            Extent2d::new(17, 9),
        );
        let case = case(LosslessModularFormat::GrayAlpha, 32, SampleKind::Float);
        let extent = Extent2d::new(17, 9);
        let special = case.samples(Extent2d::new(8, 1));
        // Each component holds one special word. This exercises signed wide averages at
        // both i32 extremes without demanding an unrepresentable difference between them.
        for pair in special.as_chunks::<2>().0.iter().skip(index * 2).take(2) {
            let samples: Vec<_> = (0..extent.area().unwrap())
                .flat_map(|_| pair.iter().copied())
                .collect();
            check_samples(&rig, encoder, &case, extent, &samples);
        }
    }
}

#[test]
fn squeeze_animation_retains_crops_references_and_ieee_words() {
    let rig = Rig::new();
    for (index, squeeze) in [
        Squeeze::HorizontalThenVertical,
        Squeeze::VerticalThenHorizontal,
    ]
    .into_iter()
    .enumerate()
    {
        let group_size = Size::ALL[index];
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                local_transforms: squeeze.into(),
                group_size,
                tree_mode: TREES[index],
                ..Default::default()
            },
        );
        groups::animation::check_animation_words(
            &rig,
            &encoder,
            group_size,
            LosslessModularFormat::Rgba,
            SampleKind::Unsigned,
            16,
        );
        groups::animation::check_cropped_frames(&rig, &encoder, group_size);
    }
}

#[test]
fn squeeze_overflow_returns_no_stream_and_releases_the_job() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Gray, 32, SampleKind::Float);
    for (index, squeeze) in MODES.into_iter().enumerate() {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                local_transforms: squeeze.into(),
                lz77: if index % 2 == 0 {
                    Lz77::ZeroRuns
                } else {
                    Lz77::Greedy
                },
                ..Default::default()
            },
        );
        for extent in [Extent2d::new(2, 2), Extent2d::new(256 * 65 + 1, 2)] {
            let samples: Vec<_> = (0..extent.height)
                .flat_map(|y| {
                    (0..extent.width).map(move |x| {
                        if extent.width > 2 && x < 256 * 64 {
                            0x3f80_0000
                        } else if (x + y) % 2 == 0 {
                            0x8000_0000
                        } else {
                            0x7fff_ffff
                        }
                    })
                })
                .collect();
            let source = upload(&rig.context, &case, extent, &samples, 0);
            assert_eq!(
                encoder.memory_plan(&source).unwrap().streaming,
                extent.width > 2
            );
            let weak = Arc::downgrade(&source.buffer);
            let error = pollster::block_on(encoder.submit_container(source).unwrap()).unwrap_err();
            assert!(
                matches!(
                    error,
                    EncodeError::Backend(BackendError::ModularSqueezeOverflow)
                ),
                "{error:?}"
            );
            assert!(weak.upgrade().is_none());
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(encoder.buffer_pool_stats().leased_buffer_sets, 0);
            // The same allocation remains usable after the rejected GPU artifact.
            let valid = vec![0x3f80_0000; extent.area().unwrap()];
            let encoded = encoder
                .encode(upload(&rig.context, &case, extent, &valid, 0))
                .unwrap();
            check_oracles(&encoded, &valid, &case);
        }
    }
}

#[test]
fn squeeze_streaming_preserves_exact_admission_cancellation_and_pool_reuse() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Rgba, 16, SampleKind::Unsigned);
    for (index, group_size) in Size::ALL.into_iter().enumerate() {
        let config = LosslessModularConfig {
            local_transforms: MODES[index].clone().into(),
            group_size,
            tree_mode: TREES[index % 2],
            predictor: Predictor::Weighted,
            lz77: Lz77::Greedy,
            color_transform: Transform::LocalRct(Rct::new(41).unwrap()),
            ..Default::default()
        };
        let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config.clone());
        for extent in [
            Extent2d::new(group_size.dimension() + 1, 3),
            Extent2d::new(group_size.dimension() * 17 + 1, 3),
        ] {
            let expected = case.samples(extent);
            let input = upload(&rig.context, &case, extent, &expected, 4099);
            let plan = encoder.memory_plan(&input).unwrap();
            assert_eq!(plan.streaming, plan.group_grid.groups > 2);
            let limited = |bytes| {
                LosslessModularEncoder::with_config(
                    WgpuContext::with_memory_budget(
                        Arc::new(rig.context.device().clone()),
                        Arc::new(rig.context.queue().clone()),
                        NonZeroU64::new(bytes).unwrap(),
                    )
                    .unwrap(),
                    config.clone(),
                )
            };
            let short = limited(plan.owned_bytes_per_job - 1);
            let failure = match short.submit(input.clone()) {
                Ok(job) => pollster::block_on(job).unwrap_err(),
                Err(error) => error,
            };
            assert!(
                matches!(failure, EncodeError::MemoryBackpressure(_)),
                "{failure:?}"
            );
            assert_eq!(short.in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(short.buffer_pool_stats().allocation_misses, 0);
            let exact = limited(plan.owned_bytes_per_job);
            let mut abandoned = input.clone();
            abandoned.buffer = Arc::new(input.buffer.as_ref().clone());
            let source = Arc::downgrade(&abandoned.buffer);
            drop(exact.submit(abandoned).unwrap());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while source.upgrade().is_some()
                || exact.in_flight_memory_stats().reserved_bytes != 0
                || exact.buffer_pool_stats().leased_buffer_sets != 0
            {
                assert!(
                    std::time::Instant::now() < deadline,
                    "cancelled Squeeze source or GPU memory retained"
                );
                rig.context.device().poll(wgpu::PollType::Poll).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let encoded = exact.encode(input.clone()).unwrap();
            assert_eq!(
                encoded,
                pollster::block_on(exact.submit(input).unwrap()).unwrap()
            );
            assert_eq!(exact.in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(exact.buffer_pool_stats().leased_buffer_sets, 0);
            assert!(exact.buffer_pool_stats().reuse_hits > 0);
            check_oracles(&encoded, &expected, &case);
            color::check_numeric(&rig, &encoded, &[expected], &case);
        }
    }
}

#[test]
fn separable_squeeze_roundtrips_both_axes_and_single_pixel_edges() {
    let rig = Rig::new();
    for (index, squeeze) in MODES.into_iter().enumerate() {
        for tree_mode in TREES {
            let encoder = LosslessModularEncoder::with_config(
                rig.context.clone(),
                LosslessModularConfig {
                    local_transforms: squeeze.clone().into(),
                    tree_mode,
                    ..Default::default()
                },
            );
            if tree_mode == TREES[0] {
                check(
                    &rig,
                    &encoder,
                    &case(LosslessModularFormat::Gray, 8, SampleKind::Unsigned),
                    Extent2d::new(1, 1),
                );
                if index >= 2 {
                    check(
                        &rig,
                        &encoder,
                        &case(LosslessModularFormat::Gray, 8, SampleKind::Unsigned),
                        Extent2d::new(257, 257),
                    );
                }
            }
            for (channel, format) in [
                LosslessModularFormat::Gray,
                LosslessModularFormat::GrayAlpha,
                LosslessModularFormat::Rgb,
                LosslessModularFormat::Rgba,
            ]
            .into_iter()
            .enumerate()
            {
                let extent = [
                    Extent2d::new(1, 257),
                    Extent2d::new(257, 1),
                    Extent2d::new(17, 9),
                    Extent2d::new(257, 3),
                ][(index + channel) % 4];
                check(
                    &rig,
                    &encoder,
                    &case(format, 8, SampleKind::Unsigned),
                    extent,
                );
            }
        }
    }
}
