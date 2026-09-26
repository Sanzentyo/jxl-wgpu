use super::*;
use resampling::{Arithmetic, Plane, Sample};

#[test]
fn reduced_modular_sources_keep_exact_words_and_filter_before_presentation() {
    let rig = Rig::new();
    for (family, samples, alpha) in [
        (0, ColorSampleFormat::GRAY8, false),
        (1, ColorSampleFormat::RGB8, true),
        (
            2,
            ColorSampleFormat::integer(ColorChannels::Gray, 13).unwrap(),
            true,
        ),
        (
            3,
            ColorSampleFormat::float(ColorChannels::Rgb, 32, 8).unwrap(),
            false,
        ),
    ] {
        for (group, entropy) in [
            (
                LosslessModularGroupSize::Pixels128,
                LosslessModularEntropyCoding::Prefix,
            ),
            (
                LosslessModularGroupSize::Pixels256,
                LosslessModularEntropyCoding::Ans,
            ),
            (
                LosslessModularGroupSize::Pixels512,
                LosslessModularEntropyCoding::Prefix,
            ),
            (
                LosslessModularGroupSize::Pixels1024,
                LosslessModularEntropyCoding::Ans,
            ),
        ] {
            let encoder = LosslessModularEncoder::with_config(
                rig.context.clone(),
                LosslessModularConfig {
                    group_size: group,
                    entropy,
                    local_transforms: LosslessModularSqueeze::HorizontalThenVertical.into(),
                    predictor: LosslessModularPredictor::Weighted,
                    ..Default::default()
                },
            );
            let format = packed_format(samples, alpha);
            // A multi-group source for each standard group size; alternating one-pixel axes
            // probe native mirror padding and local-transform elision as well.
            let coded = Extent2d::new(group.dimension() + 1, if family == 0 { 1 } else { 3 });
            let input = words(coded, samples, alpha, family);
            let source = upload(&rig.context, coded, format.clone(), &input, family as usize);
            for factor in FACTORS {
                let extent = presented(coded, factor);
                let descriptor = LosslessModularSequenceDescriptor::from_pixel_format(
                    extent.width,
                    extent.height,
                    &format,
                    AnimationHeader::Still,
                )
                .unwrap();
                let mut session = encoder.begin_sequence(descriptor).unwrap();
                let encoded = session
                    .submit_last_frame(
                        source.clone(),
                        FrameOptions {
                            upsampling: factor,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .wait()
                    .unwrap();
                assert!(encoded.acceleration.is_none());
                session.insert(encoded).unwrap();
                let bytes = session.finish_raw().unwrap();
                check_header(&bytes, 0, extent, coded, factor);
                let raw = modular_words::original_frames(&bytes);
                assert_eq!(raw.len(), 1);
                assert_eq!((raw[0].width, raw[0].height), (coded.width, coded.height));
                let channels = samples.channels().count() as usize + usize::from(alpha);
                assert_eq!(raw[0].planes.len(), channels);
                for (channel, plane) in raw[0].planes.iter().enumerate() {
                    assert_eq!(
                        plane.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                        input
                            .iter()
                            .skip(channel)
                            .step_by(channels)
                            .copied()
                            .collect::<Vec<_>>()
                    );
                }
                let native = native(&bytes);
                let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap();
                let weights = &inventory.image_header.upsampling_weights;
                let pixels = extent.width as usize * extent.height as usize;
                for channel in 0..channels {
                    let request = GpuOutputRequest::numeric(
                        SamplePrecision::float(32, 8).unwrap().pixel_format(),
                        if samples.exponent_bits() == 0 {
                            NumericSampleMapping::NormalizedUnsigned
                        } else {
                            NumericSampleMapping::NativeFloat
                        },
                    )
                    .unwrap();
                    let is_alpha = alpha && channel == channels - 1;
                    let request = if is_alpha {
                        request.with_extra_channel(0)
                    } else {
                        request.with_color_channel(channel as u32)
                    }
                    .unwrap();
                    let actual = rig.render(&bytes, request.clone(), false);
                    // Whole and bounded fragmented output must be bit-identical.
                    if factor == UpsamplingFactor::Eight {
                        assert_eq!(actual, rig.render(&bytes, request, true));
                    }
                    for (arithmetic, output) in [
                        (Arithmetic::Wgsl, extra_channels::floats(&actual)),
                        (
                            Arithmetic::Native,
                            (0..pixels)
                                .map(|i| {
                                    if is_alpha {
                                        native[4 * pixels + i]
                                    } else {
                                        native[4 * i
                                            + if samples.channels() == ColorChannels::Gray {
                                                0
                                            } else {
                                                channel
                                            }]
                                    }
                                })
                                .collect(),
                        ),
                    ] {
                        let reference = Plane {
                            width: coded.width as usize,
                            height: coded.height as usize,
                            samples: input
                                .iter()
                                .skip(channel)
                                .step_by(channels)
                                .map(|&word| Sample::decoded(word, samples.bit_depth(), arithmetic))
                                .collect(),
                        }
                        .reconstruct(
                            factor.factor(),
                            extent.width as usize,
                            extent.height as usize,
                            weights,
                            arithmetic,
                        );
                        assert_eq!(output.len(), reference.samples.len());
                        for (i, (&sample, expected)) in
                            output.iter().zip(reference.samples).enumerate()
                        {
                            expected.check(
                                sample,
                                &format!("Modular {family}/{group:?}/{factor:?}/channel {channel}"),
                                i,
                            );
                        }
                    }
                }
                assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
            }
        }
    }
}
