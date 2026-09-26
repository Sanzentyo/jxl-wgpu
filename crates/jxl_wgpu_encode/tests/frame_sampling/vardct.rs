use super::*;
use resampling::{Arithmetic, Plane, Sample};

#[test]
fn vardct_sampling_keeps_coded_geometry_across_topologies_and_color_domains() {
    let rig = Rig::new();
    for samples in [
        ColorSampleFormat::RGB8,
        ColorSampleFormat::float(ColorChannels::Gray, 32, 8).unwrap(),
    ] {
        for color_transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
            let config = VarDctConfig {
                sample_format: samples,
                color_transform,
                ..Default::default()
            };
            for topology in 0..3 {
                let coded = if topology == 0 {
                    Extent2d::new(8, 8)
                } else {
                    Extent2d::new(33, 17)
                };
                let fixed = match topology {
                    0 => Some(
                        VarDctEncoder::new_with_config(
                            rig.context.clone(),
                            VarDctStrategy::Dct8,
                            config.clone(),
                        )
                        .unwrap(),
                    ),
                    1 => Some(
                        VarDctEncoder::new_with_strategy_map(
                            rig.context.clone(),
                            VarDctStrategyMap::new(
                                coded.width,
                                coded.height,
                                (0..coded.height.div_ceil(8))
                                    .flat_map(|y| {
                                        (0..coded.width.div_ceil(8)).map(move |x| {
                                            VarDctTransform::new(x, y, VarDctStrategy::Dct8)
                                        })
                                    })
                                    .collect(),
                            )
                            .unwrap(),
                            config.clone(),
                        )
                        .unwrap(),
                    ),
                    _ => None,
                };
                let tiled =
                    TiledVarDctEncoder::new_with_config(rig.context.clone(), config.clone())
                        .unwrap();
                let input = words(coded, samples, false, topology);
                let source = upload(
                    &rig.context,
                    coded,
                    packed_format(samples, false),
                    &input,
                    topology as usize,
                );
                let baseline = if let Some(encoder) = &fixed {
                    encoder.encode(source.clone())
                } else {
                    tiled.encode(source.clone())
                }
                .unwrap();
                for factor in FACTORS {
                    let extent = presented(coded, factor);
                    let descriptor = ImageSequenceDescriptor::new(
                        extent.width,
                        extent.height,
                        AnimationHeader::Still,
                    )
                    .unwrap();
                    let mut session = if let Some(encoder) = &fixed {
                        encoder.begin_sequence(descriptor)
                    } else {
                        tiled.begin_sequence(descriptor)
                    }
                    .unwrap();
                    let frame = session
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
                    session.insert(frame).unwrap();
                    let bytes = session.finish_raw().unwrap();
                    check_header(&bytes, 0, extent, coded, factor);
                    // Sampling changes metadata, not source normalization, transforms or tokens.
                    let coded_image =
                        jxl_oxide::JxlImage::read_with_defaults(&baseline[..]).unwrap();
                    let scaled_image = jxl_oxide::JxlImage::read_with_defaults(&bytes[..]).unwrap();
                    let coded_frame = coded_image.frame(0).unwrap();
                    let scaled_frame = scaled_image.frame(0).unwrap();
                    assert_eq!(
                        coded_frame
                            .toc()
                            .iter_bitstream_order()
                            .map(|group| (group.kind, group.size))
                            .collect::<Vec<_>>(),
                        scaled_frame
                            .toc()
                            .iter_bitstream_order()
                            .map(|group| (group.kind, group.size))
                            .collect::<Vec<_>>()
                    );
                    let a = jxl_gpu_bitstream::parse(&baseline, Default::default())
                        .unwrap()
                        .codestream_inventory(Default::default())
                        .unwrap();
                    let b = jxl_gpu_bitstream::parse(&bytes, Default::default())
                        .unwrap()
                        .codestream_inventory(Default::default())
                        .unwrap();
                    for (a, b) in a.frames[0].sections.iter().zip(&b.frames[0].sections) {
                        assert_eq!(
                            &baseline[a.bytes.offset as usize..a.bytes.end().unwrap() as usize],
                            &bytes[b.bytes.offset as usize..b.bytes.end().unwrap() as usize]
                        );
                    }
                    let expected = native(&bytes);
                    let actual = rig.render(&bytes, color_request(), false);
                    check_color(
                        &actual,
                        &expected,
                        &format!("{samples:?}/{color_transform:?}/topology {topology}/{factor:?}"),
                    );
                    if factor == UpsamplingFactor::Eight {
                        assert_eq!(actual, rig.render(&bytes, color_request(), true));
                    }
                    assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
                }
            }
        }
    }
}

#[test]
fn color_sampling_routes_all_legal_extra_factors_relative_to_the_coded_grid() {
    let rig = Rig::new();
    for color in FACTORS {
        let coded = Extent2d::new(
            if color == UpsamplingFactor::Two {
                2049
            } else {
                513
            },
            5,
        );
        let extent = presented(coded, color);
        let mut specifications = vec![(3, UpsamplingFactor::Eight), (0, color)];
        specifications.extend((0..=3).flat_map(|shift| {
            [
                UpsamplingFactor::One,
                UpsamplingFactor::Two,
                UpsamplingFactor::Four,
                UpsamplingFactor::Eight,
            ]
            .into_iter()
            .filter(move |factor| factor.factor() << shift >= color.factor())
            .map(move |factor| (shift, factor))
        }));
        let definitions: Vec<_> = specifications
            .iter()
            .enumerate()
            .map(|(index, &(shift, _))| {
                ExtraChannel::new(
                    if index % 2 == 0 {
                        ExtraChannelKind::Depth
                    } else {
                        ExtraChannelKind::Thermal
                    },
                    if index % 2 == 0 {
                        SamplePrecision::integer(13).unwrap()
                    } else {
                        SamplePrecision::float(32, 8).unwrap()
                    },
                    shift,
                    format!("sampling-{index}").into_bytes(),
                )
                .unwrap()
            })
            .collect();
        let expected: Vec<_> = definitions
            .iter()
            .zip(&specifications)
            .enumerate()
            .map(|(index, (definition, &(_, factor)))| {
                let size = definition.source_extent_with_upsampling(extent, factor);
                let format = definition.precision().color(ColorChannels::Gray);
                modular_integer::ExtraWords {
                    width: size.width,
                    height: size.height,
                    words: words(size, format, false, index as u32),
                }
            })
            .collect();
        let extras: Vec<_> = expected
            .iter()
            .zip(&definitions)
            .enumerate()
            .map(|(index, (values, definition))| {
                upload(
                    &rig.context,
                    Extent2d::new(values.width, values.height),
                    definition.precision().pixel_format(),
                    &values.words,
                    index,
                )
            })
            .collect();
        let source = upload(
            &rig.context,
            coded,
            ColorSampleFormat::RGB8.pixel_format(),
            &words(coded, ColorSampleFormat::RGB8, false, 17),
            0,
        )
        .with_extra_channels(extras)
        .unwrap();
        for progressive in [ProgressivePlan::single(), progression()] {
            let encoder = TiledVarDctEncoder::new_with_config(
                rig.context.clone(),
                VarDctConfig {
                    extra_channels: definitions.clone(),
                    progressive,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut session = encoder
                .begin_sequence(
                    ImageSequenceDescriptor::new(
                        extent.width,
                        extent.height,
                        AnimationHeader::Still,
                    )
                    .unwrap(),
                )
                .unwrap();
            let frame = session
                .submit_last_frame(
                    source.clone(),
                    FrameOptions {
                        upsampling: color,
                        extra_channel_upsampling: specifications
                            .iter()
                            .map(|&(_, factor)| factor)
                            .collect(),
                        ..Default::default()
                    },
                )
                .unwrap()
                .wait()
                .unwrap();
            session.insert(frame).unwrap();
            let bytes = session.finish_raw().unwrap();
            check_header(&bytes, 0, extent, coded, color);
            assert_eq!(modular_integer::vardct_extra_words(&bytes, 0), expected);
            let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            assert_eq!(
                inventory.frames[0].extra_channel_upsampling,
                specifications
                    .iter()
                    .map(|&(shift, f)| f.factor() << shift)
                    .collect::<Vec<_>>()
            );
            // Global prefix, unshifted pass and LF routing (relative shift >= 3).
            for selected in [0, 1, specifications.len() - 1] {
                let words = &expected[selected];
                let definition = &definitions[selected];
                let (_, factor) = specifications[selected];
                let reference = Plane {
                    width: words.width as usize,
                    height: words.height as usize,
                    samples: words
                        .words
                        .iter()
                        .map(|&word| {
                            Sample::decoded(
                                word,
                                definition.precision().bit_depth(),
                                Arithmetic::Wgsl,
                            )
                        })
                        .collect(),
                }
                .reconstruct(
                    factor.factor() << definition.dimension_shift(),
                    extent.width as usize,
                    extent.height as usize,
                    &inventory.image_header.upsampling_weights,
                    Arithmetic::Wgsl,
                );
                let mapping = if definition
                    .precision()
                    .color(ColorChannels::Gray)
                    .exponent_bits()
                    == 0
                {
                    NumericSampleMapping::NormalizedUnsigned
                } else {
                    NumericSampleMapping::NativeFloat
                };
                let request = GpuOutputRequest::numeric(
                    SamplePrecision::float(32, 8).unwrap().pixel_format(),
                    mapping,
                )
                .unwrap()
                .with_extra_channel(selected as u32)
                .unwrap();
                let actual = rig.render(&bytes, request.clone(), false);
                for (i, (&sample, expected)) in extra_channels::floats(&actual)
                    .iter()
                    .zip(reference.samples)
                    .enumerate()
                {
                    expected.check(sample, &format!("color {color:?}, extra {selected}"), i);
                }
                if selected == specifications.len() - 1 {
                    assert_eq!(actual, rig.render(&bytes, request, true));
                }
            }
        }
    }
}

#[test]
fn sampled_progressive_images_keep_native_precision_and_coded_group_boundaries() {
    use jxl_gpu_formats::{ColorSpecification, RgbChannelOrder, TransferFunction};
    use jxl_test_support::oracles::progressive::{
        native_original_updates, native_updates, scalar_linear_updates, scalar_original_updates,
    };
    let rig = Rig::new();
    for (case, (width, height)) in [(1, 1), (1, 17), (255, 3), (256, 3), (257, 3), (2049, 3)]
        .into_iter()
        .enumerate()
    {
        let coded = Extent2d::new(width, height);
        let samples = if case % 2 == 0 {
            ColorSampleFormat::RGB8
        } else {
            ColorSampleFormat::float(ColorChannels::Gray, 32, 8).unwrap()
        };
        let config = VarDctConfig {
            sample_format: samples,
            color_transform: if case % 2 == 0 {
                VarDctColorTransform::Xyb
            } else {
                VarDctColorTransform::Original
            },
            progressive: progression(),
            group_order: VarDctGroupOrder::center_first(),
            ..Default::default()
        };
        let encoder = TiledVarDctEncoder::new_with_config(rig.context.clone(), config).unwrap();
        let source = upload(
            &rig.context,
            coded,
            samples.pixel_format(),
            &words(coded, samples, false, case as u32),
            case,
        );
        for factor in std::iter::once(UpsamplingFactor::One).chain(FACTORS) {
            let extent = presented(coded, factor);
            let mut session = encoder
                .begin_sequence(
                    ImageSequenceDescriptor::new(
                        extent.width,
                        extent.height,
                        AnimationHeader::Still,
                    )
                    .unwrap(),
                )
                .unwrap();
            let frame = session
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
            session.insert(frame).unwrap();
            let bytes = session.finish_raw().unwrap();
            check_header(&bytes, 0, extent, coded, factor);
            let original = samples.channels() == ColorChannels::Gray;
            let (scalar, simd) = if original {
                // Native Gray CMS output selects one component; this request is RGB.
                // Retain all independently decoded RGBA components, verify the native
                // original profile, then apply independent F64 sRGB EOTF to each RGB
                // component. Both paths are compared in linear RGB at the same bounds.
                (
                    scalar_original_updates(&bytes),
                    native_original_updates(&bytes).expect("required native original oracle"),
                )
            } else {
                (
                    scalar_linear_updates(&bytes),
                    native_updates(&bytes, true).expect("required native progressive oracle"),
                )
            };
            let linear = |bytes: &[u8]| {
                extra_channels::floats(bytes)
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        if original && i % 4 != 3 {
                            jxl_test_support::oracles::color::to_linear(
                                f64::from(v),
                                TransferFunction::Srgb,
                            ) as f32
                        } else {
                            v
                        }
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(scalar.len(), 4);
            assert_eq!(simd.len(), scalar.len());
            let mut format = PixelFormat::rgb_f32(
                RgbChannelOrder::Rgba,
                false,
                jxl_wgpu_decode::vardct_rgb8_format().color_spec,
            );
            let ColorSpecification::Defined(ref mut color) = format.color_spec else {
                unreachable!()
            };
            color.transfer = TransferFunction::Linear;
            let request = GpuOutputRequest::color(format).unwrap();
            let final_only = rig.render(&bytes, request.clone(), false);
            let mut whole = Vec::new();
            for (decoder, fragmented) in rig.decoders.iter().zip([false, true]) {
                let request = request.clone().with_progressive_output(true);
                let mut session = if fragmented {
                    open_fragmented(decoder, &bytes, request)
                } else {
                    decoder.open(&bytes, request).unwrap()
                };
                let mut held = Vec::new();
                for (stage, (native, simd)) in scalar.iter().zip(&simd).enumerate() {
                    let frame = session.next_update().unwrap().unwrap();
                    assert_eq!(frame.is_complete(), native.complete);
                    if let Some(progress) = frame.progression() {
                        assert_eq!(progress.intended_downsampling(), native.ratio);
                    }
                    let actual = read_bytes(&rig.gpu, &frame.output().outputs[0]);
                    assert_eq!(actual.len(), native.pixels.len());
                    let bound = if stage == 0 {
                        1e-5
                    } else if frame.is_complete() {
                        1e-4
                    } else {
                        2e-4
                    };
                    let code = |linear: f32| {
                        let value = linear.clamp(0.0, 1.0);
                        let srgb = if value <= 0.0031308 {
                            value * 12.92
                        } else {
                            1.055 * value.powf(1.0 / 2.4) - 0.055
                        };
                        (srgb * 255.0).round() as u8
                    };
                    for (i, ((&a, &b), &c)) in extra_channels::floats(&actual)
                        .iter()
                        .zip(linear(&native.pixels).iter())
                        .zip(linear(&simd.pixels).iter())
                        .enumerate()
                    {
                        assert!(
                            a.is_finite() && b.is_finite() && (a - b).abs() < bound,
                            "{coded:?}/{factor:?}/stage {stage}/sample {i}: {a} vs {b}, bound {bound}"
                        );
                        assert!(code(a).abs_diff(code(b)) <= 1);
                        assert!(code(a).abs_diff(code(c)) <= 1);
                    }
                    if frame.is_complete() {
                        assert_eq!(actual, final_only);
                    }
                    if fragmented {
                        assert_eq!(actual, whole[stage]);
                    } else {
                        whole.push(actual);
                    }
                    held.push(frame);
                }
                assert!(session.next_update().unwrap().is_none());
                drop(session);
                for (frame, expected) in held.iter().zip(&whole) {
                    assert_eq!(&read_bytes(&rig.gpu, &frame.output().outputs[0]), expected);
                }
                drop(held);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}
