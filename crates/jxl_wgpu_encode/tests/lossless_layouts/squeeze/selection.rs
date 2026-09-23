use super::*;
use jxl_wgpu_encode::LosslessModularPalette as Palette;

fn ranges(channels: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..channels).flat_map(move |begin| (1..=channels - begin).map(move |count| (begin, count)))
}

fn policy(mode: usize, begin: u32, count: u32, in_place: bool) -> Squeeze {
    MODES[mode]
        .with_channels(begin, count)
        .unwrap()
        .with_in_place(in_place)
}

fn config(squeeze: Squeeze, variant: usize) -> LosslessModularConfig {
    LosslessModularConfig {
        squeeze,
        color_transform: Transform::None,
        group_size: Size::ALL[variant % 4],
        tree_mode: TREES[variant % 2],
        predictor: Predictor::ALL[variant % 14],
        weighted_predictor: Weighted::new([31, 0, 17, 3, 11, 31, 1], [0, 15, 7, 12]).unwrap(),
        lz77: if variant.is_multiple_of(2) {
            Lz77::ZeroRuns
        } else {
            Lz77::Greedy
        },
        ..Default::default()
    }
}

fn checked(
    rig: &Rig,
    config: LosslessModularConfig,
    case: Case,
    extent: Extent2d,
    expected: &[u32],
    channels: u32,
) -> Vec<u8> {
    eprintln!("{config:?}, {case:?}, {extent:?}");
    let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config);
    let input = upload(&rig.context, &case, extent, expected, 4099);
    assert_eq!(encoder.memory_plan(&input).unwrap().channel_count, channels);
    let encoded = pollster::block_on(encoder.submit_container(input).unwrap()).unwrap();
    assert_eq!(
        encoded,
        encoder
            .encode_container(upload(&rig.context, &case.canonical(), extent, expected, 0))
            .unwrap()
    );
    let native = palette::delta::check_frame_oracles(&encoded, &[expected], &case);
    if config
        .palette
        .is_none_or(|palette| palette.delta_predictor().is_none())
    {
        check_frame_samples(&encoded, &[expected], &case, &native);
    }
    color::check_numeric(rig, &encoded, &[expected.to_vec()], &case);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(encoder.buffer_pool_stats().leased_buffer_sets, 0);
    encoded
}

#[test]
fn selected_squeeze_covers_every_range_axis_order_and_residual_placement() {
    let rig = Rig::new();
    let mut variant = 0;
    for format in [
        LosslessModularFormat::Gray,
        LosslessModularFormat::GrayAlpha,
        LosslessModularFormat::Rgb,
        LosslessModularFormat::Rgba,
    ] {
        for (begin, count) in ranges(format.channel_count()) {
            for mode in 0..4 {
                for in_place in [false, true] {
                    let config = config(policy(mode, begin, count, in_place), variant);
                    let mut case = case(format, 16, SampleKind::Unsigned);
                    case.storage = [Storage::Packed, Storage::Planar, Storage::Split][variant % 3];
                    let extent = Extent2d::new(17, 9);
                    let channels = format.channel_count() + count * if mode < 2 { 1 } else { 3 };
                    checked(&rig, config, case, extent, &case.samples(extent), channels);
                    variant += 1;
                }
            }
        }
    }
    assert_eq!(variant, 160);
}

#[test]
fn selected_squeeze_composes_with_each_palette_policy_and_unselected_source() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Rgba, 16, SampleKind::Unsigned);
    let extent = Extent2d::new(17, 9);
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
        // Post-Palette image channels are [component 0, palette index, component 3].
        for (variant, (begin, count)) in ranges(3).enumerate() {
            for in_place in [false, true] {
                let mode = (index + variant) % 4;
                let config = LosslessModularConfig {
                    palette: Some(palette.with_components(1, 2).unwrap()),
                    ..config(policy(mode, begin, count, in_place), variant + index)
                };
                checked(
                    &rig,
                    config,
                    case,
                    extent,
                    &expected,
                    4 + count * if mode < 2 { 1 } else { 3 },
                );
            }
        }
    }
}

#[test]
fn selected_squeeze_keeps_wide_and_ieee_words_with_untransformed_extremes() {
    let rig = Rig::new();
    for (kind, bits) in (1..=31)
        .map(|bits| (SampleKind::Unsigned, bits))
        .chain([(SampleKind::Float, 16), (SampleKind::Float, 32)])
    {
        let case = case(LosslessModularFormat::Rgba, bits, kind);
        let extent = Extent2d::new(17, 9);
        let mut expected = case.samples(extent);
        for (pixel, words) in expected.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            // Bound selected residuals while keeping high bits; unselected IEEE words include
            // differences outside signed i32. Those must never enter Squeeze arithmetic.
            words[1] = if kind == SampleKind::Float {
                if bits == 32 { 0xffc0_0001 } else { 0xfe01 }
            } else {
                (u32::MAX >> (32 - bits)) - pixel as u32 % (1u32 << bits.min(8))
            };
        }
        for in_place in [false, true] {
            let mode = bits as usize % 4;
            checked(
                &rig,
                config(policy(mode, 1, 1, in_place), bits as usize),
                case,
                extent,
                &expected,
                if mode < 2 { 5 } else { 7 },
            );
        }
    }
}

#[test]
fn selected_squeeze_keeps_all_rcts_and_group_edge_axis_elision() {
    let rig = Rig::new();
    let selections: Vec<_> = ranges(4).collect();
    for value in 0..42 {
        let variant = value as usize;
        let (begin, count) = selections[variant % selections.len()];
        let mode = variant % 4;
        let mut config = config(policy(mode, begin, count, value % 2 == 0), variant);
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
            Extent2d::new(1, 1),
        ][variant / 4 % 4];
        let case = case(LosslessModularFormat::Rgba, 16, SampleKind::Unsigned);
        let stages =
            u32::from(mode != 1 && extent.width > 1) + u32::from(mode != 0 && extent.height > 1);
        checked(
            &rig,
            config,
            case,
            extent,
            &case.samples(extent),
            4 + count * ((1 << stages) - 1),
        );
    }
}

// Read the wire with independent libjxl-style bit fields, not the encoder's transform plan.
fn local_steps(encoded: &[u8]) -> Vec<Vec<(bool, bool, u32, u32)>> {
    use jxl_bitstream::U;
    let image = jxl_oxide::JxlImage::read_with_defaults(encoded).unwrap();
    let frame = image.frame(0).unwrap();
    assert!(!frame.toc().is_single_entry());
    (0..frame.header().num_groups())
        .map(|group| {
            let mut stream = frame.pass_group_bitstream(0, group).unwrap().unwrap();
            assert!(!stream.partial);
            let bits = &mut stream.bitstream;
            bits.read_bool().unwrap();
            if !bits.read_bool().unwrap() {
                for _ in 0..7 {
                    bits.read_bits(5).unwrap();
                }
                for _ in 0..4 {
                    bits.read_bits(4).unwrap();
                }
            }
            let count = bits.read_u32(0, 1, 2 + U(4), 18 + U(8)).unwrap();
            let mut steps = Vec::new();
            for _ in 0..count {
                match bits.read_bits(2).unwrap() {
                    0 => {
                        bits.read_u32(U(3), 8 + U(6), 72 + U(10), 1096 + U(13))
                            .unwrap();
                        bits.read_u32(6, U(2), 2 + U(4), 10 + U(6)).unwrap();
                    }
                    1 => {
                        bits.read_u32(U(3), 8 + U(6), 72 + U(10), 1096 + U(13))
                            .unwrap();
                        bits.read_u32(1, 3, 4, 1 + U(13)).unwrap();
                        bits.read_u32(U(8), 256 + U(10), 1280 + U(12), 5376 + U(16))
                            .unwrap();
                        bits.read_u32(0, 1 + U(8), 257 + U(10), 1281 + U(16))
                            .unwrap();
                        bits.read_bits(4).unwrap();
                    }
                    2 => {
                        let parameters = bits.read_u32(0, 1 + U(4), 9 + U(6), 41 + U(8)).unwrap();
                        for _ in 0..parameters {
                            let horizontal = bits.read_bool().unwrap();
                            let in_place = bits.read_bool().unwrap();
                            let begin = bits
                                .read_u32(U(3), 8 + U(6), 72 + U(10), 1096 + U(13))
                                .unwrap();
                            let count = bits.read_u32(1, 2, 3, 4 + U(4)).unwrap();
                            steps.push((horizontal, in_place, begin, count));
                        }
                    }
                    other => panic!("unexpected transform {other}"),
                }
            }
            steps
        })
        .collect()
}

#[test]
fn selected_squeeze_wire_steps_preserve_in_place_flags_ranges_and_axis_elision() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Rgba, 8, SampleKind::Unsigned);
    let extent = Extent2d::new(129, 3);
    let mut expected = case.samples(extent);
    for words in expected.as_chunks_mut::<4>().0 {
        words[1] %= 3;
        words[2] %= 5;
    }
    for palette in [
        None,
        Some(Palette::new(32).unwrap().with_components(1, 2).unwrap()),
    ] {
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
                let config = LosslessModularConfig {
                    palette,
                    group_size: Size::Pixels128,
                    ..config(policy(mode, 1, 1, in_place), mode)
                };
                let encoded = checked(
                    &rig,
                    config,
                    case,
                    extent,
                    &expected,
                    if mode < 2 { 5 } else { 7 },
                );
                let expected_steps: Vec<_> = [128, 1]
                    .map(|width| {
                        let axes: Vec<_> = axes
                            .iter()
                            .copied()
                            .filter(|horizontal| !horizontal || width > 1)
                            .collect();
                        let begin = 1 + u32::from(palette.is_some());
                        match axes.as_slice() {
                            [] => vec![],
                            &[first] => vec![(first, in_place, begin, 1)],
                            &[first, second] if in_place => {
                                vec![(first, true, begin, 1), (second, true, begin, 2)]
                            }
                            &[first, second] => vec![
                                (first, false, begin, 1),
                                (second, false, begin, 1),
                                (second, false, 4, 1),
                            ],
                            _ => unreachable!(),
                        }
                    })
                    .into_iter()
                    .collect();
                assert_eq!(local_steps(&encoded), expected_steps);
            }
        }
    }
}

#[test]
fn selected_squeeze_rejects_missing_channels_before_allocation() {
    let rig = Rig::new();
    for (format, palette, channels) in [
        (LosslessModularFormat::Gray, None, 1),
        (LosslessModularFormat::GrayAlpha, None, 2),
        (LosslessModularFormat::Rgb, None, 3),
        (
            LosslessModularFormat::Rgba,
            Some(Palette::new(4).unwrap()),
            1,
        ),
        (
            LosslessModularFormat::Rgba,
            Some(Palette::new(4).unwrap().with_components(1, 2).unwrap()),
            3,
        ),
    ] {
        let case = case(format, 8, SampleKind::Unsigned);
        let extent = Extent2d::new(1, 1); // invalid even when both axes would be skipped
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                palette,
                ..config(policy(2, channels, 1, true), 0)
            },
        );
        let input = upload(&rig.context, &case, extent, &case.samples(extent), 0);
        assert!(
            matches!(encoder.memory_plan(&input), Err(EncodeError::InvalidModularSqueezeChannels { channels: actual, .. }) if actual == channels)
        );
        assert!(matches!(
            encoder.submit(input),
            Err(EncodeError::InvalidModularSqueezeChannels { .. })
        ));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(encoder.buffer_pool_stats().allocation_misses, 0);
    }
}

#[test]
fn selected_squeeze_overflow_never_publishes_resident_or_late_streamed_output() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::GrayAlpha, 32, SampleKind::Float);
    for mode in 0..4 {
        for in_place in [false, true] {
            let encoder = LosslessModularEncoder::with_config(
                rig.context.clone(),
                config(policy(mode, 1, 1, in_place), mode),
            );
            let edge = encoder.config().group_size.dimension();
            for extent in [Extent2d::new(2, 2), Extent2d::new(edge * 65 + 2, 2)] {
                let mut expected: Vec<_> = (0..extent.height)
                    .flat_map(|y| {
                        (0..extent.width).flat_map(move |x| {
                            [
                                0x7fc0_0001,
                                if extent.width > 2 && x < edge * 64 {
                                    0x3f80_0000
                                } else if (x + y) % 2 == 0 {
                                    0x8000_0000
                                } else {
                                    0x7fff_ffff
                                },
                            ]
                        })
                    })
                    .collect();
                let input = upload(&rig.context, &case, extent, &expected, 4099);
                assert_eq!(
                    encoder.memory_plan(&input).unwrap().streaming,
                    extent.width > 2
                );
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
fn selected_squeeze_retains_cropped_animation_references() {
    let rig = Rig::new();
    for mode in [2, 3] {
        for in_place in [false, true] {
            let config = config(policy(mode, 1, 2, in_place), mode - 2);
            let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config);
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
}

#[test]
fn selected_squeeze_preserves_exact_budget_cancellation_and_pool_reuse() {
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
    for mode in 0..4 {
        for in_place in [false, true] {
            palette::check_lifetime_with_samples(
                &rig,
                LosslessModularConfig {
                    palette: Some(
                        Palette::mixed(3, 4096, Predictor::Weighted)
                            .unwrap()
                            .with_components(1, 2)
                            .unwrap(),
                    ),
                    predictor: Predictor::Weighted,
                    lz77: Lz77::Greedy,
                    color_transform: Transform::LocalRct(Rct::new(41).unwrap()),
                    ..config(policy(mode, 0, 2, in_place), mode)
                },
                palette::delta::check_frame_oracles,
                samples,
            );
        }
    }
}
