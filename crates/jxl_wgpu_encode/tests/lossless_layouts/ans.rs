use super::*;
use jxl_wgpu_encode::{
    LosslessModularConfig, LosslessModularEntropyCoding, LosslessModularGroupSize,
    LosslessModularLz77,
};

fn config(lz77: LosslessModularLz77) -> LosslessModularConfig {
    LosslessModularConfig {
        entropy: LosslessModularEntropyCoding::Ans,
        lz77,
        group_size: LosslessModularGroupSize::Pixels128,
        ..Default::default()
    }
}

fn read_global_entropy(bits: &mut jxl_bitstream::Bitstream<'_>) -> jxl_coding::Decoder {
    assert!(bits.read_bool().unwrap()); // default LF dequantization
    assert!(bits.read_bool().unwrap()); // global Modular tree
    let mut tree = jxl_coding::Decoder::parse(bits, 6).unwrap();
    tree.begin(bits).unwrap();
    let mut nodes = 1;
    while nodes != 0 {
        nodes -= 1;
        if tree.read_varint(bits, 1).unwrap() == 0 {
            for context in 2..6 {
                tree.read_varint(bits, context).unwrap();
            }
        } else {
            tree.read_varint(bits, 0).unwrap();
            nodes += 2;
        }
    }
    tree.finalize().unwrap();
    jxl_coding::Decoder::parse(bits, 4).unwrap()
}

#[test]
fn ans_selects_shared_or_distinct_histograms_from_actual_gpu_populations() {
    use jxl_wgpu_encode::{LosslessModularColorTransform, LosslessModularPredictor};
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Rgba, 16, SampleKind::Unsigned);
    for tree_mode in TREES {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                tree_mode,
                predictor: LosslessModularPredictor::Zero,
                color_transform: LosslessModularColorTransform::None,
                ..config(LosslessModularLz77::ZeroRuns)
            },
        );
        for extent in [Extent2d::new(17, 3), Extent2d::new(129, 3)] {
            for shared in [true, false] {
                let pixel = if shared { [1; 4] } else { [1, 8, 64, 512] };
                let expected: Vec<_> = (0..extent.area().unwrap()).flat_map(|_| pixel).collect();
                let input = upload(&rig.context, &case, extent, &expected, 4099);
                let encoded = encoder.encode(input).unwrap();
                assert_eq!(
                    encoded,
                    encoder
                        .encode(upload(
                            &rig.context,
                            &case.canonical(),
                            extent,
                            &expected,
                            0
                        ))
                        .unwrap()
                );
                let parsed = jxl_gpu_bitstream::parse(&encoded, Default::default()).unwrap();
                let inventory = parsed.codestream_inventory(Default::default()).unwrap();
                let section = inventory.frames[0]
                    .sections
                    .iter()
                    .find(|section| {
                        matches!(
                            section.kind,
                            jxl_gpu_bitstream::FrameSectionKind::Single
                                | jxl_gpu_bitstream::FrameSectionKind::LowFrequencyGlobal
                        )
                    })
                    .unwrap();
                let start = section.bytes.offset as usize;
                let mut bits = jxl_bitstream::Bitstream::new(
                    &encoded[start..start + section.bytes.length as usize],
                );
                let descriptor = read_global_entropy(&mut bits);
                let map = descriptor.cluster_map();
                assert_eq!(map.len(), 5);
                assert_eq!(*map.iter().max().unwrap() + 1, if shared { 1 } else { 4 });
                let native = check_oracles(&encoded, &expected, &case);
                rig.check_gpu(&encoded, &expected, &case, &native);
            }
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(encoder.buffer_pool_stats().leased_buffer_sets, 0);
    }
}

#[test]
fn ans_single_and_multiple_groups_preserve_words_and_streaming_decode() {
    let rig = Rig::new();
    for (lz77, tree) in [LosslessModularLz77::ZeroRuns, LosslessModularLz77::Greedy]
        .into_iter()
        .zip(TREES)
    {
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                tree_mode: tree,
                ..config(lz77)
            },
        );
        for format in [
            LosslessModularFormat::Gray,
            LosslessModularFormat::GrayAlpha,
            LosslessModularFormat::Rgb,
            LosslessModularFormat::Rgba,
        ] {
            let case = Case {
                format,
                bits: 12,
                kind: SampleKind::Unsigned,
                storage: Storage::Planar,
                reversed: true,
                byte_order: ByteOrder::Big,
                shifted: true,
            };
            for extent in [Extent2d::new(17, 3), Extent2d::new(129, 3)] {
                check(&rig, &encoder, &case, extent);
            }
        }
    }
}

fn check(rig: &Rig, encoder: &LosslessModularEncoder, case: &Case, extent: Extent2d) {
    let expected = case.samples(extent);
    let input = upload(&rig.context, case, extent, &expected, 4099);
    let plan = encoder.memory_plan(&input).unwrap();
    assert!(plan.streaming);
    assert_eq!(plan.hybrid_histogram_bytes, 165_768);
    assert_eq!(plan.gpu_submission_count, 2 * plan.batch_count);
    let encoded = pollster::block_on(encoder.submit_container(input).unwrap()).unwrap();
    let canonical = upload(&rig.context, &case.canonical(), extent, &expected, 0);
    assert_eq!(encoded, encoder.encode_container(canonical).unwrap());
    let native = check_oracles(&encoded, &expected, case);
    if case.format == LosslessModularFormat::GrayAlpha {
        color::check_numeric(rig, &encoded, &[expected], case);
    } else {
        rig.check_gpu(&encoded, &expected, case, &native);
    }
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(encoder.buffer_pool_stats().leased_buffer_sets, 0);
}

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

#[test]
fn ans_preserves_all_integer_depths_and_ieee_special_words() {
    let rig = Rig::new();
    for mode in [LosslessModularLz77::ZeroRuns, LosslessModularLz77::Greedy] {
        let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config(mode));
        for (kind, bits) in (1..=31)
            .map(|bits| (SampleKind::Unsigned, bits))
            .chain([(SampleKind::Float, 16), (SampleKind::Float, 32)])
        {
            let format = [
                LosslessModularFormat::Gray,
                LosslessModularFormat::GrayAlpha,
                LosslessModularFormat::Rgb,
                LosslessModularFormat::Rgba,
            ][bits as usize % 4];
            let case = case(format, bits, kind);
            let extent = [
                Extent2d::new(1, 129),
                Extent2d::new(129, 1),
                Extent2d::new(17, 3),
            ][bits as usize % 3];
            check(&rig, &encoder, &case, extent);
        }
    }
}

#[test]
fn ans_composes_with_every_predictor_rct_and_group_size() {
    use jxl_wgpu_encode::{
        LosslessModularColorTransform as Color, LosslessModularPredictor as Predictor,
        LosslessModularRctType as Rct, LosslessModularWeightedPredictor as Weighted,
    };
    let rig = Rig::new();
    for value in 0..42 {
        let group_size = LosslessModularGroupSize::ALL[value as usize % 4];
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            LosslessModularConfig {
                predictor: Predictor::ALL[value as usize % 14],
                weighted_predictor: Weighted::new([31, 0, 1, 2, 3, 4, 5], [0, 15, 1, 2]).unwrap(),
                color_transform: if value % 2 == 0 {
                    Color::GlobalRct(Rct::new(value).unwrap())
                } else {
                    Color::LocalRct(Rct::new(value).unwrap())
                },
                group_size,
                tree_mode: TREES[value as usize % 2],
                ..config(if value % 3 == 0 {
                    LosslessModularLz77::ZeroRuns
                } else {
                    LosslessModularLz77::Greedy
                })
            },
        );
        check(
            &rig,
            &encoder,
            &case(LosslessModularFormat::Rgba, 31, SampleKind::Unsigned),
            Extent2d::new(group_size.dimension() + 1, 2),
        );
    }
}

#[test]
fn ans_palette_and_ordered_transforms_share_one_group_state_including_empty_planes() {
    use jxl_wgpu_encode::{
        LosslessModularLocalTransforms as Program, LosslessModularPalette as Palette,
        LosslessModularPredictor as Predictor, LosslessModularRctType as Rct,
        LosslessModularSqueezeStep as Step, LosslessModularTransform as Op,
    };
    let rig = Rig::new();
    for (variant, palette) in [
        Palette::new(32).unwrap(),
        Palette::deltas(4096, Predictor::West).unwrap(),
        Palette::mixed(3, 4096, Predictor::Weighted).unwrap(),
        Palette::implicit(4096, Predictor::Gradient).unwrap(),
    ]
    .into_iter()
    .enumerate()
    {
        let config = LosslessModularConfig {
            palette: Some(palette.with_components(1, 2).unwrap()),
            local_transforms: Program::sequence(vec![
                Op::Rct {
                    begin_channel: 0,
                    rct_type: Rct::new(41).unwrap(),
                },
                Op::Squeeze(Step::new(true, 0, 3, false).unwrap()),
                Op::Rct {
                    begin_channel: 3,
                    rct_type: Rct::new(13).unwrap(),
                },
                Op::Squeeze(Step::new(false, 0, 3, true).unwrap()),
            ])
            .unwrap(),
            tree_mode: TREES[variant % 2],
            ..config(if variant % 2 == 0 {
                LosslessModularLz77::ZeroRuns
            } else {
                LosslessModularLz77::Greedy
            })
        };
        let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config);
        let case = case(LosslessModularFormat::Rgba, 16, SampleKind::Unsigned);
        for extent in [Extent2d::new(1, 1), Extent2d::new(129, 3)] {
            let entries = case.samples(Extent2d::new(5, 1));
            let expected: Vec<_> = (0..extent.area().unwrap())
                .flat_map(|pixel| entries[pixel % 5 * 4..][..4].iter().copied())
                .collect();
            palette::checked_stream(&rig, &encoder, case, extent, &expected);
        }
    }
}

#[test]
fn ans_exact_budget_cancellation_and_pool_reuse_cover_one_and_many_batches() {
    let rig = Rig::new();
    let case = case(LosslessModularFormat::Rgba, 31, SampleKind::Unsigned);
    let config = config(LosslessModularLz77::Greedy);
    let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config.clone());
    for extent in [Extent2d::new(17, 3), Extent2d::new(16_384, 1)] {
        let expected = case.samples(extent);
        let input = upload(&rig.context, &case, extent, &expected, 4099);
        let plan = encoder.memory_plan(&input).unwrap();
        assert_eq!(plan.batch_count > 1, extent.width == 16_384);
        assert!(plan.ans_output_bytes > 0 && plan.ans_output_bytes < plan.artifact_storage_bytes);
        assert_eq!(plan.hybrid_histogram_bytes, 165_768);
        assert!(plan.ans_output_bytes + plan.hybrid_histogram_bytes < plan.artifact_storage_bytes);
        assert_eq!(plan.gpu_submission_count, 2 * plan.batch_count);
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
        let error = match short.submit(input.clone()) {
            Ok(job) => pollster::block_on(job).unwrap_err(),
            Err(error) => error,
        };
        assert!(
            matches!(error, EncodeError::MemoryBackpressure(_)),
            "{error:?}"
        );
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
                "ANS cancellation retained a source or lease"
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
        let native = check_oracles(&encoded, &expected, &case);
        rig.check_gpu(&encoded, &expected, &case, &native);
    }
}

#[test]
fn ans_animation_keeps_exact_physical_words_and_cropped_references() {
    let rig = Rig::new();
    for mode in [LosslessModularLz77::ZeroRuns, LosslessModularLz77::Greedy] {
        let config = config(mode);
        let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config.clone());
        groups::animation::check_animation_words_with_oracle(
            &rig,
            &encoder,
            config.group_size,
            LosslessModularFormat::Rgba,
            SampleKind::Unsigned,
            16,
            check_frame_oracles,
        );
        groups::animation::check_cropped_frames(&rig, &encoder, config.group_size);
    }
}

#[test]
fn empty_global_ans_state_and_padding_are_checked_by_native_and_gpu_decoders() {
    let rig = Rig::new();
    let encoder = LosslessModularEncoder::with_config(
        rig.context.clone(),
        LosslessModularConfig {
            weighted_predictor: jxl_wgpu_encode::LosslessModularWeightedPredictor::new(
                [0; 7], [0; 4],
            )
            .unwrap(),
            ..config(LosslessModularLz77::Greedy)
        },
    );
    let case = case(LosslessModularFormat::Gray, 8, SampleKind::Unsigned);
    let extent = Extent2d::new(129, 3);
    let expected = case.samples(extent);
    let encoded = encoder
        .encode(upload(&rig.context, &case, extent, &expected, 0))
        .unwrap();
    let parsed = jxl_gpu_bitstream::parse(&encoded, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let section = inventory.frames[0]
        .sections
        .iter()
        .find(|section| section.kind == jxl_gpu_bitstream::FrameSectionKind::LowFrequencyGlobal)
        .unwrap();
    let start = section.bytes.offset as usize;
    let end = start + section.bytes.length as usize;
    let mut bits = jxl_bitstream::Bitstream::new(&encoded[start..end]);
    let mut entropy = read_global_entropy(&mut bits);
    assert!(bits.read_bool().unwrap()); // shared tree
    assert!(!bits.read_bool().unwrap()); // explicit WP leaves an unaligned entropy start
    for _ in 0..7 {
        assert_eq!(bits.read_bits(5).unwrap(), 0);
    }
    for _ in 0..4 {
        assert_eq!(bits.read_bits(4).unwrap(), 0);
    }
    assert_eq!(bits.read_bits(2).unwrap(), 0); // no transforms on Gray
    let state_bit = start * 8 + bits.num_read_bits();
    entropy.begin(&mut bits).unwrap();
    entropy.finalize().unwrap();
    let padding_bit = start * 8 + bits.num_read_bits();
    assert!(padding_bit < end * 8 && end * 8 - padding_bit <= 7);
    palette::delta::check_frame_oracles(&encoded, &[&expected], &case);
    let binary =
        std::env::var_os("JXL_MODULAR_WORD_ORACLE").expect("required pinned native word oracle");
    let input =
        std::env::temp_dir().join(format!("jxl-ans-empty-state-{}.jxl", std::process::id()));
    for bit in [state_bit, state_bit + 31, padding_bit] {
        let mut corrupt = encoded.clone();
        corrupt[bit / 8] ^= 1 << (bit % 8);
        std::fs::write(&input, &corrupt).unwrap();
        let native = std::process::Command::new(&binary)
            .arg(&input)
            .output()
            .unwrap();
        assert!(!native.status.success() && native.stdout.is_empty());
        for (decoder, fragmented) in rig.decoders.iter().zip([false, true]) {
            let request = GpuOutputRequest::numeric(
                case.format.pixel_format(8).unwrap(),
                NumericSampleMapping::NativeUnsigned,
            )
            .unwrap();
            let mut session = if fragmented {
                open_fragmented(decoder, &corrupt, request)
            } else {
                decoder.open(&corrupt, request).unwrap()
            };
            let error = pollster::block_on(session.next_frame_async())
                .expect_err("damaged empty ANS must not publish a frame");
            assert!(
                matches!(error, jxl_wgpu_decode::Error::ModularEntropyRejected { .. }),
                "{error:?}"
            );
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
    std::fs::remove_file(input).unwrap();
}

#[test]
fn late_transform_failures_release_ans_jobs_without_publishing_a_codestream() {
    use jxl_wgpu_encode::{
        BackendError, LosslessModularLocalTransforms as Program, LosslessModularPalette as Palette,
        LosslessModularRctType as Rct, LosslessModularSqueezeStep as Step,
        LosslessModularTransform as Op,
    };
    let rig = Rig::new();
    let extent = Extent2d::new(128 * 65 + 4, 1);
    for palette in [true, false] {
        let case = case(LosslessModularFormat::Rgba, 32, SampleKind::Float);
        let config = LosslessModularConfig {
            palette: palette.then(|| Palette::new(1).unwrap()),
            local_transforms: if palette {
                Default::default()
            } else {
                Program::sequence(vec![
                    Op::Squeeze(Step::new(true, 0, 3, false).unwrap()),
                    Op::Rct {
                        begin_channel: 4,
                        rct_type: Rct::new(6).unwrap(),
                    },
                    Op::Squeeze(Step::new(true, 4, 3, false).unwrap()),
                ])
                .unwrap()
            },
            ..config(LosslessModularLz77::Greedy)
        };
        let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config);
        let mut expected: Vec<_> = (0..extent.width)
            .flat_map(|x| {
                let value = if x < 128 * 64 {
                    0x3f80_0000
                } else {
                    [0xc000_0000, 0x3fff_ffff, 0x3fff_ffff, 0xc000_0000][(x % 4) as usize]
                };
                [value, value, value, 0x7fc0_0001]
            })
            .collect();
        let input = upload(&rig.context, &case, extent, &expected, 4099);
        assert!(encoder.memory_plan(&input).unwrap().batch_count > 1);
        let source = Arc::downgrade(&input.buffer);
        let error = pollster::block_on(encoder.submit_container(input).unwrap()).unwrap_err();
        assert!(
            match error {
                EncodeError::Backend(BackendError::ModularPaletteOverflow) => palette,
                EncodeError::Backend(BackendError::ModularSqueezeOverflow) => !palette,
                _ => false,
            },
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
