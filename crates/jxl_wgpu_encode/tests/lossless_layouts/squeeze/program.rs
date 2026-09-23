use super::*;
use jxl_wgpu_encode::{
    LosslessModularLocalTransforms as Program, LosslessModularPalette as Palette,
    LosslessModularSqueezeStep as Step, LosslessModularTransform as Op,
};

fn rct(begin_channel: u32, value: u32) -> Op {
    Op::Rct {
        begin_channel,
        rct_type: Rct::new(value).unwrap(),
    }
}

fn squeeze(horizontal: bool, begin: u32, count: u32, in_place: bool) -> Op {
    Op::Squeeze(Step::new(horizontal, begin, count, in_place).unwrap())
}

fn config(operations: Vec<Op>, variant: usize) -> LosslessModularConfig {
    LosslessModularConfig {
        local_transforms: Program::sequence(operations).unwrap(),
        ..selection::config(Squeeze::None, variant)
    }
}

#[test]
fn ordered_rct_squeeze_programs_cover_all_rct_types_and_empty_residuals() {
    let rig = Rig::new();
    for value in 0..42 {
        for in_place in [false, true] {
            let operations = vec![
                rct(0, value),
                squeeze(true, 0, 3, in_place),
                rct(0, (value + 13) % 42),
                rct(if in_place { 3 } else { 4 }, value),
                squeeze(false, 0, 3, in_place),
            ];
            let config = config(operations, value as usize);
            let mut case = case(LosslessModularFormat::Rgba, 16, SampleKind::Unsigned);
            case.storage = [Storage::Packed, Storage::Planar, Storage::Split][value as usize % 3];
            let extent = [
                Extent2d::new(1, 1),
                Extent2d::new(1, 257),
                Extent2d::new(257, 1),
                Extent2d::new(257, 9),
            ][value as usize % 4];
            selection::checked(&rig, config, case, extent, &case.samples(extent), 10);
        }
    }
}

#[test]
fn ordered_programs_preserve_every_precision_alpha_and_gray_channel_lineages() {
    let rig = Rig::new();
    for (kind, bits) in (1..=31)
        .map(|bits| (SampleKind::Unsigned, bits))
        .chain([(SampleKind::Float, 16), (SampleKind::Float, 32)])
    {
        let case = case(LosslessModularFormat::Rgba, bits, kind);
        let extent = Extent2d::new(19, 11);
        let expected = case.samples(extent);
        // RCT's arithmetic is wrapping even for IEEE words and includes the alpha word.
        selection::checked(
            &rig,
            config(
                vec![rct(1, u32::from(bits) % 42), rct(0, 41)],
                bits as usize,
            ),
            case,
            extent,
            &expected,
            4,
        );
        let mut bounded = expected;
        for (pixel, words) in bounded.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            for (component, word) in words.iter_mut().enumerate().take(3) {
                *word = if kind == SampleKind::Float {
                    if bits == 32 {
                        [0xffc0_0001, 0x7f80_0000, 0x8000_0000][component]
                    } else {
                        [0xfe01, 0x7c00, 0x8000][component]
                    }
                } else {
                    ((pixel * (component + 1)) as u32 & ((1 << bits.min(8)) - 1))
                        | if bits > 8 { 1 << (bits - 2) } else { 0 }
                };
            }
        }
        for in_place in [false, true] {
            selection::checked(
                &rig,
                config(
                    vec![
                        rct(0, 41),
                        squeeze(true, 0, 3, in_place),
                        rct(0, 13),
                        squeeze(false, 0, 3, in_place),
                    ],
                    bits as usize,
                ),
                case,
                extent,
                &bounded,
                10,
            );
        }
    }
    for format in [
        LosslessModularFormat::Gray,
        LosslessModularFormat::GrayAlpha,
    ] {
        let count = format.channel_count();
        let case = case(format, 16, SampleKind::Unsigned);
        let extent = Extent2d::new(16, 8);
        for in_place in [false, true] {
            let operations = vec![
                squeeze(true, 0, count, in_place),
                squeeze(false, 0, 2 * count, in_place),
                rct(1, 41),
                squeeze(true, 1, 3, in_place),
                rct(1, 6),
            ];
            selection::checked(
                &rig,
                config(operations, count as usize),
                case,
                extent,
                &case.samples(extent),
                count * 4 + 3,
            );
        }
    }
}

#[test]
fn ordered_programs_address_post_palette_images_for_every_palette_policy() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Rgba, 16, SampleKind::Unsigned);
    let extent = Extent2d::new(256, 8);
    let entries = case.samples(Extent2d::new(5, 1));
    let expected: Vec<_> = (0..extent.area().unwrap())
        .flat_map(|pixel| entries[pixel % 5 * 4..][..4].iter().copied())
        .collect();
    for (variant, palette) in [
        Palette::new(32).unwrap(),
        Palette::deltas(4096, Predictor::West).unwrap(),
        Palette::mixed(3, 4096, Predictor::Weighted).unwrap(),
        Palette::implicit(4096, Predictor::Gradient).unwrap(),
    ]
    .into_iter()
    .enumerate()
    {
        for count in 1..=4 {
            let channels = 5 - count;
            let in_place = count % 2 == 0;
            let config = LosslessModularConfig {
                palette: Some(palette.with_components(4 - count, count).unwrap()),
                color_transform: if variant % 2 == 0 {
                    Transform::GlobalRct(Rct::new(41).unwrap())
                } else {
                    Transform::LocalRct(Rct::new(13).unwrap())
                },
                group_size: Size::Pixels128,
                ..config(
                    vec![
                        squeeze(true, 0, channels, in_place),
                        squeeze(false, 0, 2 * channels, in_place),
                        rct(1, 41),
                        squeeze(true, 1, 3, in_place),
                        rct(1, 6),
                    ],
                    variant,
                )
            };
            let encoded = selection::checked(
                &rig,
                config.clone(),
                case,
                extent,
                &expected,
                1 + 4 * channels + 3,
            );
            let mut wire = Vec::new();
            if variant % 2 != 0 {
                wire.push(selection::WireTransform::Rct(0, 13));
            }
            wire.push(selection::WireTransform::Palette(4 - count, count));
            wire.extend(expected_wire(
                config.local_transforms.operations().unwrap(),
                1,
            ));
            for actual in selection::local_operations(&encoded) {
                assert_eq!(actual, wire);
            }
        }
    }
}

fn expected_wire(operations: &[Op], meta: u32) -> Vec<selection::WireTransform> {
    operations
        .iter()
        .map(|operation| match *operation {
            Op::Rct {
                begin_channel,
                rct_type,
            } => selection::WireTransform::Rct(meta + begin_channel, rct_type.value()),
            Op::Squeeze(step) => selection::WireTransform::Squeeze(vec![(
                step.horizontal(),
                step.in_place(),
                meta + step.channel_range().start,
                step.channel_range().len() as u32,
            )]),
        })
        .collect()
}

#[test]
fn ordered_program_wire_covers_transform_counts_and_rct_begin_buckets() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Rgba, 8, SampleKind::Unsigned);
    let extent = Extent2d::new(256, 1);
    for count in [1, 2, 17, 18, 273] {
        let operations: Vec<_> = (0..count)
            .map(|index| rct(index % 2, 7 * (1 + index % 5)))
            .collect();
        let expected_wire = expected_wire(&operations, 0);
        let mut config = config(operations, 0);
        if count == 273 {
            // A separate global header does not consume a local transform slot.
            config.color_transform = Transform::GlobalRct(Rct::YCOCG);
        }
        let encoded = selection::checked(&rig, config, case, extent, &case.samples(extent), 4);
        for actual in selection::local_operations(&encoded) {
            assert_eq!(actual, expected_wire);
        }
    }
    let extent = Extent2d::new(256, 3);
    let mut operations = Vec::new();
    let mut split_all = |horizontal, channels| {
        for begin in (0..channels).step_by(19) {
            operations.push(squeeze(
                horizontal,
                begin,
                (channels - begin).min(19),
                false,
            ));
        }
    };
    for level in 0..7 {
        split_all(true, 4 << level);
    }
    split_all(false, 512);
    split_all(false, 512);
    operations.extend([rct(0, 6), rct(8, 13), rct(72, 27), rct(1096, 41)]);
    let wire = expected_wire(&operations, 0);
    let encoded = selection::checked(
        &rig,
        config(operations, 0),
        case,
        extent,
        &case.samples(extent),
        1536,
    );
    for actual in selection::local_operations(&encoded) {
        assert_eq!(actual, wire);
    }
}

#[test]
fn ordered_programs_reject_invalid_topology_and_header_limits_before_admission() {
    for count in [0, 274] {
        assert!(matches!(Program::sequence(vec![rct(0, 0); count]),
            Err(EncodeError::InvalidModularTransformCount { count: actual }) if actual == count));
    }
    assert!(matches!(
        Program::sequence(vec![rct(9288, 0)]),
        Err(EncodeError::InvalidModularRctBegin { begin: 9288 })
    ));
    let rig = Rig::new();
    let mut excessive_header = config(vec![rct(0, 0); 273], 0);
    excessive_header.color_transform = Transform::LocalRct(Rct::YCOCG);
    let cases = [
        (
            LosslessModularFormat::Gray,
            Extent2d::new(2, 2),
            config(vec![rct(0, 0)], 0),
        ),
        (
            LosslessModularFormat::Rgba,
            Extent2d::new(2, 2),
            config(vec![rct(2, 0)], 0),
        ),
        (
            LosslessModularFormat::Rgba,
            Extent2d::new(2, 2),
            config(vec![squeeze(true, 0, 1, false), rct(0, 0)], 0),
        ),
        // Equal dimensions alone do not suffice: this changes channel zero's shift.
        (
            LosslessModularFormat::Rgba,
            Extent2d::new(1, 1),
            config(vec![squeeze(true, 0, 1, false), rct(0, 0)], 0),
        ),
        // The full group is valid; the final width-one group has unequal residual geometry.
        (
            LosslessModularFormat::GrayAlpha,
            Extent2d::new(129, 2),
            config(vec![squeeze(true, 0, 2, false), rct(1, 0)], 0),
        ),
        (
            LosslessModularFormat::Rgba,
            Extent2d::new(2, 2),
            excessive_header,
        ),
    ];
    for (index, (format, extent, config)) in cases.into_iter().enumerate() {
        let case = case(format, 8, SampleKind::Unsigned);
        let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config);
        let input = upload(&rig.context, &case, extent, &case.samples(extent), 0);
        for error in [
            encoder.memory_plan(&input).unwrap_err(),
            encoder.submit(input).err().unwrap(),
        ] {
            match index {
                0 | 1 => assert!(
                    matches!(error, EncodeError::InvalidModularRctChannels { .. }),
                    "{error:?}"
                ),
                2..=4 => assert!(
                    matches!(error, EncodeError::UnequalModularRctChannels { .. }),
                    "{error:?}"
                ),
                _ => assert!(
                    matches!(
                        error,
                        EncodeError::InvalidModularTransformCount { count: 274 }
                    ),
                    "{error:?}"
                ),
            }
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(encoder.buffer_pool_stats().allocation_misses, 0);
    }
}

#[test]
fn ordered_program_late_overflow_never_publishes_resident_or_streamed_output() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Rgba, 32, SampleKind::Float);
    for in_place in [false, true] {
        for variant in [0, 1] {
            let residual = if in_place { 3 } else { 4 };
            let encoder = LosslessModularEncoder::with_config(
                rig.context.clone(),
                config(
                    vec![
                        squeeze(true, 0, 3, in_place),
                        rct(residual, 6),
                        squeeze(true, residual, 3, in_place),
                    ],
                    variant,
                ),
            );
            let edge = encoder.config().group_size.dimension();
            for extent in [Extent2d::new(4, 1), Extent2d::new(edge * 65 + 4, 1)] {
                let mut expected: Vec<_> = (0..extent.width)
                    .flat_map(|x| {
                        let value = if extent.width > 4 && x < edge * 64 {
                            0x3f80_0000
                        } else {
                            [0xc000_0000, 0x3fff_ffff, 0x3fff_ffff, 0xc000_0000][(x % 4) as usize]
                        };
                        [value, value, value, 0x7fc0_0001]
                    })
                    .collect();
                let input = upload(&rig.context, &case, extent, &expected, 4099);
                let plan = encoder.memory_plan(&input).unwrap();
                assert_eq!(plan.streaming, extent.width > 4);
                assert!(plan.transform_scratch_bytes > 0);
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
                for words in expected.as_chunks_mut::<4>().0 {
                    words[..3].fill(0x3f80_0000);
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
fn ordered_program_arenas_obey_exact_admission_cancellation_and_pool_reuse() {
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
        let in_place = index % 2 == 0;
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
            ..config(
                vec![
                    rct(0, 6),
                    squeeze(true, 0, 3, in_place),
                    rct(0, 41),
                    squeeze(false, 0, 3, in_place),
                ],
                index,
            )
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
fn ordered_programs_keep_animation_words_and_cropped_references() {
    let rig = Rig::new();
    for in_place in [false, true] {
        let config = config(
            vec![
                squeeze(true, 0, 4, in_place),
                rct(0, 7),
                squeeze(false, 0, 3, in_place),
            ],
            usize::from(in_place),
        );
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
