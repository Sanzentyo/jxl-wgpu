use super::*;
mod components;
pub(crate) mod delta;
mod implicit;
mod mixed;
use jxl_wgpu_encode::{
    BackendError, LosslessModularColorTransform as Transform, LosslessModularConfig,
    LosslessModularGroupSize as Size, LosslessModularLz77 as Lz77,
    LosslessModularPalette as Palette, LosslessModularPredictor as Predictor,
    LosslessModularRctType as Rct, LosslessModularSqueeze as Squeeze,
    LosslessModularWeightedPredictor as Weighted,
};

const SQUEEZES: [Squeeze; 5] = [
    Squeeze::None,
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

fn samples(case: Case, extent: Extent2d, colors: u32) -> Vec<u32> {
    let entries = case.samples(Extent2d::new(colors, 1));
    let channels = case.format.channel_count() as usize;
    (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).flat_map({
                let entries = &entries;
                move |x| {
                    let index = ((x * 13) ^ (y * 3) ^ (x / 7)) % colors;
                    entries[index as usize * channels..][..channels]
                        .iter()
                        .copied()
                }
            })
        })
        .collect()
}

fn checked_stream(
    rig: &Rig,
    encoder: &LosslessModularEncoder,
    case: Case,
    extent: Extent2d,
    expected: &[u32],
) -> Vec<u8> {
    eprintln!("{:?}, {case:?}, {extent:?}", encoder.config());
    let input = upload(&rig.context, &case, extent, expected, 4099);
    let plan = encoder.memory_plan(&input).unwrap();
    assert!(plan.palette_scratch_bytes > 0);
    let encoded = pollster::block_on(encoder.submit_container(input).unwrap()).unwrap();
    assert_eq!(
        encoded,
        encoder
            .encode_container(upload(&rig.context, &case.canonical(), extent, expected, 0))
            .unwrap()
    );
    if encoder
        .config()
        .palette
        .unwrap()
        .delta_predictor()
        .is_some()
    {
        delta::check_frame_oracles(&encoded, &[expected], &case);
    } else {
        check_oracles(&encoded, expected, &case);
    }
    color::check_numeric(rig, &encoded, &[expected.to_vec()], &case);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    encoded
}

#[test]
fn local_palette_preserves_every_component_and_composes_with_both_squeeze_orders() {
    let rig = Rig::new();
    for (index, squeeze) in SQUEEZES.into_iter().enumerate() {
        for tree_mode in TREES {
            let encoder = LosslessModularEncoder::with_config(
                rig.context.clone(),
                LosslessModularConfig {
                    palette: Some(Palette::new(32).unwrap()),
                    squeeze: squeeze.clone(),
                    tree_mode,
                    ..Default::default()
                },
            );
            for (channel, format) in [
                LosslessModularFormat::Gray,
                LosslessModularFormat::GrayAlpha,
                LosslessModularFormat::Rgb,
                LosslessModularFormat::Rgba,
            ]
            .into_iter()
            .enumerate()
            {
                let case = Case {
                    format,
                    bits: 8,
                    kind: SampleKind::Unsigned,
                    storage: Storage::Split,
                    reversed: true,
                    byte_order: ByteOrder::Big,
                    shifted: true,
                };
                let extent = [
                    Extent2d::new(1, 1),
                    Extent2d::new(17, 9),
                    Extent2d::new(257, 3),
                    Extent2d::new(1, 257),
                ][(index + channel) % 4];
                check(&rig, &encoder, case, extent, &samples(case, extent, 17));
            }
        }
    }
}

#[test]
fn palette_composes_with_all_rcts_predictors_and_full_precision_words() {
    let rig = Rig::new();
    for value in 0..42 {
        let group_size = Size::ALL[value as usize % 4];
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::new(32).unwrap()),
                squeeze: SQUEEZES[value as usize % 5].clone(),
                group_size,
                tree_mode: TREES[value as usize % 2],
                color_transform: if value % 2 == 0 {
                    Transform::GlobalRct(Rct::new(value).unwrap())
                } else {
                    Transform::LocalRct(Rct::new(value).unwrap())
                },
                predictor: Predictor::ALL[value as usize % 14],
                weighted_predictor: Weighted::new([31, 0, 17, 3, 11, 31, 1], [0, 15, 7, 12])
                    .unwrap(),
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
        let extent = Extent2d::new(group_size.dimension() + 1, 5);
        check(&rig, &encoder, case, extent, &samples(case, extent, 17));
    }
    for squeeze in SQUEEZES {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::new(32).unwrap()),
                squeeze,
                color_transform: Transform::None,
                ..Default::default()
            },
        );
        for bits in [16, 32] {
            // No arithmetic or numerical equality may merge signed zero or NaN payloads.
            let case = case(LosslessModularFormat::GrayAlpha, bits, SampleKind::Float);
            let extent = Extent2d::new(17, 9);
            check(&rig, &encoder, case, extent, &samples(case, extent, 17));
        }
    }
}

#[test]
fn palette_color_counts_cover_every_wire_bucket_and_the_maximum() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Gray, 31, SampleKind::Unsigned);
    for colors in [1, 255, 256, 1279, 1280, 5375, 5376, Palette::MAX_COLORS] {
        let extent = Extent2d::new(colors.min(1024), colors.div_ceil(1024));
        let expected: Vec<_> = (0..extent.area().unwrap())
            .map(|index| 0x7fff_ffff - index as u32 % colors)
            .collect();
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::new(colors).unwrap()),
                group_size: Size::ALL[3],
                ..Default::default()
            },
        );
        eprintln!("palette wire color count {colors}");
        let encoded = encoder
            .encode_container(upload(&rig.context, &case, extent, &expected, 0))
            .unwrap();
        check_oracles(&encoded, &expected, &case);
        color::check_numeric(&rig, &encoded, &[expected], &case);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn palette_animation_keeps_special_words_crops_and_reference_composition() {
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
                palette: Some(Palette::new(4096).unwrap()),
                squeeze,
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
            31,
        );
        groups::animation::check_animation_words(
            &rig,
            &encoder,
            group_size,
            LosslessModularFormat::GrayAlpha,
            SampleKind::Float,
            32,
        );
        groups::animation::check_cropped_frames(&rig, &encoder, group_size);
    }
}

#[test]
fn palette_overflow_returns_no_stream_and_releases_resident_and_streamed_jobs() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Gray, 32, SampleKind::Float);
    for (index, squeeze) in SQUEEZES.into_iter().enumerate() {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette: Some(Palette::new(1).unwrap()),
                squeeze,
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
                        // The first complete artifact batch is valid; overflow comes later.
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
            let valid = vec![0x8000_0000; extent.area().unwrap()];
            let encoded = encoder
                .encode(upload(&rig.context, &case, extent, &valid, 0))
                .unwrap();
            check_oracles(&encoded, &valid, &case);
        }
    }
}

#[test]
fn palette_scratch_obeys_exact_admission_cancellation_and_pool_reuse() {
    let rig = Rig::new();
    for (index, group_size) in Size::ALL.into_iter().enumerate() {
        let config = LosslessModularConfig {
            palette: Some(Palette::new(32).unwrap()),
            squeeze: SQUEEZES[index + 1].clone(),
            group_size,
            tree_mode: TREES[index % 2],
            predictor: Predictor::Weighted,
            lz77: Lz77::Greedy,
            color_transform: Transform::LocalRct(Rct::new(41).unwrap()),
            ..Default::default()
        };
        check_lifetime(&rig, config, check_frame_oracles);
    }
}

fn check_lifetime(rig: &Rig, config: LosslessModularConfig, oracle: FrameOracle) {
    check_lifetime_with_samples(rig, config, oracle, samples);
}

pub(super) fn check_lifetime_with_samples(
    rig: &Rig,
    config: LosslessModularConfig,
    oracle: FrameOracle,
    make_samples: fn(Case, Extent2d, u32) -> Vec<u32>,
) {
    let group_size = config.group_size;
    let case = case(LosslessModularFormat::Rgba, 31, SampleKind::Unsigned);
    let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config.clone());
    for extent in [
        Extent2d::new(group_size.dimension() + 1, 3),
        Extent2d::new(group_size.dimension() * 33 + 1, 3),
    ] {
        let expected = make_samples(case, extent, 17);
        let input = upload(&rig.context, &case, extent, &expected, 4099);
        let plan = encoder.memory_plan(&input).unwrap();
        assert!(plan.palette_scratch_bytes > 0);
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
        assert_eq!(
            short.buffer_pool_stats().allocation_misses,
            0,
            "{config:?}, {extent:?}, {plan:?}"
        );
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
                "cancelled Palette source or GPU memory retained"
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
        oracle(&encoded, &[&expected], &case);
        color::check_numeric(rig, &encoded, &[expected], &case);
    }
}

fn check(
    rig: &Rig,
    encoder: &LosslessModularEncoder,
    case: Case,
    extent: Extent2d,
    expected: &[u32],
) -> usize {
    checked_stream(rig, encoder, case, extent, expected).len()
}
