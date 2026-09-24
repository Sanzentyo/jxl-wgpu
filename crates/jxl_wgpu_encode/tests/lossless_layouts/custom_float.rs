use super::*;
use jxl_gpu_formats::FloatPrecision;
use jxl_wgpu_encode::{
    AnimationHeader, EncodeProfile, FrameOptions, FrameTiming, LosslessModularAnimationDescriptor,
    LosslessModularColorTransform as Transform, LosslessModularConfig,
    LosslessModularEntropyCoding as Entropy, LosslessModularGroupSize as Size,
    LosslessModularLz77 as Lz77, LosslessModularPalette as Palette,
    LosslessModularPredictor as Predictor, LosslessModularRctType as Rct,
    LosslessModularSqueeze as Squeeze,
};

fn case(bits: u8, exponent: u8, index: usize) -> Case {
    Case {
        format: [
            LosslessModularFormat::Gray,
            LosslessModularFormat::GrayAlpha,
            LosslessModularFormat::Rgb,
            LosslessModularFormat::Rgba,
        ][index % 4],
        bits,
        kind: SampleKind::CustomFloat(FloatPrecision::new(bits, exponent).unwrap()),
        storage: if bits <= 7 {
            Storage::SharedWord
        } else if bits == 24 {
            Storage::ThreeBytes
        } else {
            [Storage::Planar, Storage::Split, Storage::Packed][index % 3]
        },
        reversed: true,
        byte_order: if index.is_multiple_of(2) {
            ByteOrder::Big
        } else {
            ByteOrder::Little
        },
        shifted: true,
    }
}

fn config(entropy: Entropy) -> LosslessModularConfig {
    LosslessModularConfig {
        entropy,
        group_size: Size::Pixels128,
        tree_mode: if entropy == Entropy::Ans {
            TREES[1]
        } else {
            TREES[0]
        },
        lz77: if entropy == Entropy::Ans {
            Lz77::Greedy
        } else {
            Lz77::ZeroRuns
        },
        ..Default::default()
    }
}

fn all_precisions(entropy: Entropy) {
    let rig = Rig::new();
    let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config(entropy));
    let mut count = 0;
    for exponent in 2..=8 {
        for fraction in 2..=23 {
            let bits = 1 + exponent + fraction;
            let case = case(bits, exponent, count);
            let profile = EncodeProfile::ModularLossless {
                sample_bit_depth: jxl_gpu_bitstream::SampleBitDepth::Float {
                    bits_per_sample: u32::from(bits),
                    exponent_bits_per_sample: u32::from(exponent),
                },
            };
            assert!(
                encoder
                    .capabilities()
                    .profiles
                    .iter()
                    .any(|cap| cap.supports(profile))
            );
            rig.check(&encoder, &case, Extent2d::new(23, 5));
            count += 1;
        }
    }
    assert_eq!(count, 154);
    for exponent in 0..=10 {
        for bits in 0..=34 {
            let profile = EncodeProfile::ModularLossless {
                sample_bit_depth: jxl_gpu_bitstream::SampleBitDepth::Float {
                    bits_per_sample: u32::from(bits),
                    exponent_bits_per_sample: u32::from(exponent),
                },
            };
            assert_eq!(
                encoder
                    .capabilities()
                    .profiles
                    .iter()
                    .any(|cap| cap.supports(profile)),
                FloatPrecision::new(bits, exponent).is_ok()
            );
        }
    }
}

#[test]
fn custom_float_prefix_preserves_all_154_precisions() {
    all_precisions(Entropy::Prefix);
}

#[test]
fn custom_float_ans_preserves_all_154_precisions() {
    all_precisions(Entropy::Ans);
}

#[test]
fn custom_float_precision_composes_with_rct_palette_squeeze_and_weighted_prediction() {
    let rig = Rig::new();
    for entropy in [Entropy::Prefix, Entropy::Ans] {
        for (index, (bits, exponent)) in
            [(5, 2), (16, 3), (16, 8), (24, 2), (24, 7), (31, 7), (32, 8)]
                .into_iter()
                .enumerate()
        {
            let case = Case {
                format: LosslessModularFormat::Rgba,
                ..case(bits, exponent, index)
            };
            let encoder = LosslessModularEncoder::with_config(
                rig.context.clone(),
                LosslessModularConfig {
                    palette: Some(Palette::new(64).unwrap()),
                    color_transform: if index.is_multiple_of(2) {
                        Transform::LocalRct(Rct::new(41).unwrap())
                    } else {
                        Transform::GlobalRct(Rct::new(6).unwrap())
                    },
                    predictor: Predictor::Weighted,
                    local_transforms: if index.is_multiple_of(2) {
                        Squeeze::HorizontalThenVertical.into()
                    } else {
                        Squeeze::VerticalThenHorizontal.into()
                    },
                    ..config(entropy)
                },
            );
            let extent = Extent2d::new(129, 5);
            let entries = case.samples(Extent2d::new(32, 1));
            let expected: Vec<_> = (0..extent.width * extent.height * 4)
                .map(|i| entries[i as usize % entries.len()])
                .collect();
            let encoded = palette::checked_stream(&rig, &encoder, case, extent, &expected);
            let frames = jxl_test_support::oracles::modular_words::original_frames(&encoded);
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].bits, u32::from(bits));
            assert_eq!(frames[0].exponent_bits, u32::from(exponent));
            for (channel, plane) in frames[0].planes.iter().enumerate() {
                assert_eq!(
                    plane.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                    expected
                        .iter()
                        .skip(channel)
                        .step_by(4)
                        .copied()
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn custom_float_ieee_aliases_keep_existing_bytes_and_reject_precision_mismatches() {
    let rig = Rig::new();
    let encoder = LosslessModularEncoder::new(rig.context.clone());
    for (bits, exponent) in [(16, 5), (32, 8)] {
        let case = case(bits, exponent, 3);
        let extent = Extent2d::new(129, 3);
        let expected = case.samples(extent);
        let input = upload(&rig.context, &case, extent, &expected, 259);
        let mut invalid = input.clone();
        invalid.layout.format.sample_kind =
            SampleKind::CustomFloat(FloatPrecision::new(bits - 1, exponent).unwrap());
        let allocations = encoder.buffer_pool_stats().allocation_misses;
        assert!(matches!(
            encoder.submit(invalid),
            Err(EncodeError::Unsupported(_))
        ));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(encoder.buffer_pool_stats().leased_buffer_sets, 0);
        assert_eq!(encoder.buffer_pool_stats().allocation_misses, allocations);
        let encoded = encoder.encode_container(input).unwrap();
        let native_case = Case {
            kind: SampleKind::Float,
            ..case
        };
        assert_eq!(
            encoded,
            encoder
                .encode_container(upload(&rig.context, &native_case, extent, &expected, 259))
                .unwrap()
        );
    }
}

#[test]
fn custom_float_animations_keep_exponent_identity_timing_and_retained_outputs() {
    let rig = Rig::new();
    for (index, (bits, exponent)) in [(5, 2), (16, 8), (24, 7), (31, 7)].into_iter().enumerate() {
        let case = Case {
            format: LosslessModularFormat::Rgba,
            ..case(bits, exponent, index)
        };
        let extent = Extent2d::new(129, 3);
        let encoder = LosslessModularEncoder::with_config(
            rig.context.clone(),
            config(if index.is_multiple_of(2) {
                Entropy::Prefix
            } else {
                Entropy::Ans
            }),
        );
        let timing = AnimationHeader::Animation {
            ticks_per_second_numerator: std::num::NonZeroU32::new(100).unwrap(),
            ticks_per_second_denominator: std::num::NonZeroU32::new(1).unwrap(),
            num_loops: 2,
            have_timecodes: true,
        };
        let descriptor = LosslessModularAnimationDescriptor::from_pixel_format(
            extent.width,
            extent.height,
            &case.pixel_format(),
            timing,
        )
        .unwrap();
        assert_eq!(
            descriptor.sample_bit_depth(),
            jxl_gpu_bitstream::SampleBitDepth::Float {
                bits_per_sample: u32::from(bits),
                exponent_bits_per_sample: u32::from(exponent),
            }
        );
        let mut assembly = encoder.begin_animation(descriptor).unwrap();
        // Same storage width but a different legal exponent must not join the stream.
        if bits > 5 {
            let wrong = Case {
                kind: SampleKind::CustomFloat(
                    FloatPrecision::new(bits, if exponent == 8 { 7 } else { exponent + 1 })
                        .unwrap(),
                ),
                ..case
            };
            let before = encoder.in_flight_memory_stats().reserved_bytes;
            assert!(matches!(
                assembly.submit_frame(
                    upload(&rig.context, &wrong, extent, &wrong.samples(extent), 0),
                    FrameOptions::default()
                ),
                Err(EncodeError::InvalidConfiguration(_))
            ));
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, before);
            assert_eq!(encoder.buffer_pool_stats().allocation_misses, 0);
        }
        let mut frames = Vec::new();
        let mut pending = Vec::new();
        for (frame, storage) in [Storage::Planar, Storage::Split, Storage::Packed]
            .into_iter()
            .enumerate()
        {
            let case = Case {
                storage,
                byte_order: if frame == 1 {
                    ByteOrder::Little
                } else {
                    ByteOrder::Big
                },
                ..case
            };
            let mut samples = case.samples(extent);
            samples.rotate_left(frame * 7);
            let input = upload(&rig.context, &case, extent, &samples, 4099);
            frames.push(samples);
            let options = FrameOptions {
                timing: FrameTiming {
                    duration_ticks: frame as u32 + 2,
                    timecode: Some(20 + frame as u32),
                },
                ..Default::default()
            };
            pending.push(
                if frame == 2 {
                    assembly.submit_last_frame(input, options)
                } else {
                    assembly.submit_frame(input, options)
                }
                .unwrap(),
            );
        }
        for job in pending.into_iter().rev() {
            assembly.insert(job.wait().unwrap()).unwrap();
        }
        let encoded = assembly.finish_container().unwrap();
        check_frame_oracles(
            &encoded,
            &frames.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &case,
        );
        color::check_numeric(&rig, &encoded, &frames, &case);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn custom_float_resident_and_streamed_jobs_keep_budget_and_cancellation_contracts() {
    let rig = Rig::new();
    for (index, (bits, exponent)) in [(5, 2), (24, 7)].into_iter().enumerate() {
        lifetime::check_case(
            &rig,
            Case {
                format: LosslessModularFormat::Rgba,
                storage: Storage::Planar,
                ..case(bits, exponent, index)
            },
            LosslessModularConfig {
                tree_mode: TREES[1],
                ..config(Entropy::Prefix)
            },
            Extent2d::new(17, 9),
        );
    }
}

#[test]
fn custom_float_ans_jobs_keep_budget_and_cancellation_for_one_and_many_batches() {
    let rig = Rig::new();
    for (bits, exponent) in [(24, 7), (32, 8)] {
        ans::check_lifetime(
            &rig,
            Case {
                format: LosslessModularFormat::Rgba,
                storage: Storage::Planar,
                ..case(bits, exponent, 3)
            },
        );
    }
}
