use super::*;

#[test]
fn modular_presentation_preserves_source_words_and_all_name_length_buckets() {
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
        let extent = if family == 0 {
            Extent2d::new(1, 7)
        } else {
            Extent2d::new(129, 3)
        };
        let (source, words) = source(&rig, extent, samples, alpha, family);
        let mut baseline = None;
        for value in 1..=8 {
            let orientation = OutputOrientation::from_exif_value(value).unwrap();
            let encoder = LosslessModularEncoder::with_config(
                rig.context.clone(),
                LosslessModularConfig {
                    group_size: LosslessModularGroupSize::Pixels128,
                    entropy: if family % 2 == 0 {
                        LosslessModularEntropyCoding::Prefix
                    } else {
                        LosslessModularEntropyCoding::Ans
                    },
                    ..Default::default()
                },
            )
            .with_image_options(ImageOptions {
                orientation,
                ..Default::default()
            })
            .unwrap();
            let descriptor = LosslessModularSequenceDescriptor::from_pixel_format(
                extent.width,
                extent.height,
                &source.layout.format,
                AnimationHeader::Still,
            )
            .unwrap();
            let mut session = encoder.begin_sequence(descriptor).unwrap();
            let name = name(value as usize - 1);
            let frame = session
                .submit_last_frame(
                    source.clone(),
                    FrameOptions {
                        name: name.clone(),
                        ..Default::default()
                    },
                )
                .unwrap()
                .wait()
                .unwrap();
            session.insert(frame).unwrap();
            let bytes = session.finish_raw().unwrap();
            check_headers(&bytes, orientation, &[name]);
            let raw = modular_words::original_frames(&bytes);
            assert_eq!(raw.len(), 1);
            assert_eq!((raw[0].width, raw[0].height), (extent.width, extent.height));
            let count = samples.channels().count() as usize + usize::from(alpha);
            assert_eq!(raw[0].planes.len(), count);
            for (c, plane) in raw[0].planes.iter().enumerate() {
                assert_eq!(
                    plane.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                    words
                        .iter()
                        .skip(c)
                        .step_by(count)
                        .copied()
                        .collect::<Vec<_>>()
                );
            }
            let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            assert_eq!(
                inventory.image_header.modular_16bit_buffers,
                samples.exponent_bits() == 0
            );
            let payload = inventory.frames[0]
                .sections
                .iter()
                .map(|section| {
                    bytes[section.bytes.offset as usize..section.bytes.end().unwrap() as usize]
                        .to_vec()
                })
                .collect::<Vec<_>>();
            if let Some(baseline) = &baseline {
                assert_eq!(&payload, baseline);
            } else {
                baseline = Some(payload);
            }
            rig.check_output(&bytes, extent, orientation, alpha);
            // The still convenience entry point must use the same image metadata owner.
            if value == 8 {
                let still = encoder.encode(source.clone()).unwrap();
                check_headers(&still, orientation, &[CodestreamName::default()]);
                rig.check_output(&still, extent, orientation, alpha);
            }
            assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn vardct_presentation_is_shared_by_single_map_and_tiled_encoders() {
    let rig = Rig::new();
    for color_transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
        for value in 1..=8 {
            let orientation = OutputOrientation::from_exif_value(value).unwrap();
            let topology = value % 3;
            let coded = match topology {
                0 => Extent2d::new(8, 8),
                1 => Extent2d::new(17, 9),
                _ => Extent2d::new(259, 3),
            };
            let config = VarDctConfig {
                color_transform,
                progressive: progression(),
                image_options: ImageOptions {
                    orientation,
                    ..Default::default()
                },
                ..Default::default()
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
            let tiled = TiledVarDctEncoder::new_with_config(rig.context.clone(), config).unwrap();
            let (source, _) = source(&rig, coded, ColorSampleFormat::RGB8, false, value);
            let extent = Extent2d::new(coded.width * 2 - 1, coded.height * 2 - 1);
            let descriptor =
                ImageSequenceDescriptor::new(extent.width, extent.height, AnimationHeader::Still)
                    .unwrap();
            let mut session = if let Some(encoder) = &fixed {
                encoder.begin_sequence(descriptor)
            } else {
                tiled.begin_sequence(descriptor)
            }
            .unwrap();
            let name = name(value as usize - 1);
            let frame = session
                .submit_last_frame(
                    source,
                    FrameOptions {
                        name: name.clone(),
                        upsampling: UpsamplingFactor::Two,
                        ..Default::default()
                    },
                )
                .unwrap()
                .wait()
                .unwrap();
            session.insert(frame).unwrap();
            let bytes = session.finish_raw().unwrap();
            check_headers(&bytes, orientation, &[name]);
            rig.check_output(&bytes, extent, orientation, false);
            assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn orientation_and_names_coexist_with_icc_and_tone_metadata() {
    use jxl_gpu_formats::ColorSpecification;
    use jxl_gpu_protocol::icc::IccProfile;
    use jxl_test_support::oracles::icc_profile::IccProfileOracle;

    let rig = Rig::new();
    let native_profile = IccProfileOracle::compile();
    let profile = IccProfile::parse(
        std::fs::read(jxl_test_support::fixtures::embedded_icc::directory().join("rgb.icc"))
            .unwrap()
            .into(),
        Default::default(),
    )
    .unwrap();
    let extent = Extent2d::new(17, 9);
    let (mut source, _) = source(&rig, extent, ColorSampleFormat::RGB8, false, 3);
    source.layout.format.color_spec = ColorSpecification::Icc(profile.clone());
    for value in [1, 6] {
        for intensity in [0x5bf8, 0x63d0] {
            let options = ImageOptions {
                orientation: OutputOrientation::from_exif_value(value).unwrap(),
                rendering_intent: profile.header().rendering_intent,
                intensity_target: jxl_gpu_bitstream::FiniteF16::from_bits(intensity).unwrap(),
                intrinsic_size: Some(IntrinsicSize::new(8193, 513).unwrap()),
                min_nits: display::half(0x2c00),
                linear_below: ToneMappingThreshold::DisplayFraction(display::half(0x3000)),
            };
            for modular in [true, false] {
                let name = name(7);
                let bytes = if modular {
                    let encoder = LosslessModularEncoder::new(rig.context.clone())
                        .with_image_options(options)
                        .unwrap();
                    let descriptor = LosslessModularSequenceDescriptor::from_pixel_format(
                        extent.width,
                        extent.height,
                        &source.layout.format,
                        AnimationHeader::Still,
                    )
                    .unwrap();
                    let mut session = encoder.begin_sequence(descriptor).unwrap();
                    let frame = session
                        .submit_last_frame(
                            source.clone(),
                            FrameOptions {
                                name: name.clone(),
                                ..Default::default()
                            },
                        )
                        .unwrap()
                        .wait()
                        .unwrap();
                    session.insert(frame).unwrap();
                    session.finish_raw().unwrap()
                } else {
                    let encoder = TiledVarDctEncoder::new_with_config(
                        rig.context.clone(),
                        VarDctConfig {
                            color_transform: VarDctColorTransform::Original,
                            source_color: source.layout.format.color_spec.clone(),
                            image_options: options,
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
                                name: name.clone(),
                                ..Default::default()
                            },
                        )
                        .unwrap()
                        .wait()
                        .unwrap();
                    session.insert(frame).unwrap();
                    session.finish_raw().unwrap()
                };
                let image = jxl_oxide::JxlImage::read_with_defaults(&bytes[..]).unwrap();
                assert_eq!(image.image_header().metadata.orientation, value);
                assert_eq!(
                    image.frame(0).unwrap().header().name.as_bytes(),
                    name.as_str().as_bytes()
                );
                let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap();
                assert_eq!(
                    inventory
                        .image_header
                        .tone_mapping
                        .intensity_target
                        .to_f32(),
                    options.intensity_target.to_f32()
                );
                assert_eq!(
                    inventory
                        .image_header
                        .embedded_icc
                        .as_ref()
                        .unwrap()
                        .profile
                        .as_ref(),
                    profile.bytes().as_ref()
                );
                assert_eq!(
                    native_profile.read(&bytes).profile,
                    profile.bytes().as_ref()
                );
                display::check_metadata(&bytes, &native_profile, extent, options);
                assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
            }
        }
    }
}
