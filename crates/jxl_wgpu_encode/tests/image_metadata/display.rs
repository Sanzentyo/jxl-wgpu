use super::*;
use jxl_gpu_bitstream::FiniteF16;
use jxl_oxide_common::Bundle;
use jxl_test_support::oracles::icc_profile::IccProfileOracle;

pub(super) fn half(bits: u16) -> FiniteF16 {
    FiniteF16::from_bits(bits).unwrap()
}

pub(super) fn check_metadata(
    bytes: &[u8],
    native: &IccProfileOracle,
    extent: Extent2d,
    options: ImageOptions,
) {
    let info = native.image_info(bytes);
    let parsed =
        jxl_image::ImageHeader::parse(&mut jxl_bitstream::Bitstream::new(bytes), ()).unwrap();
    let header = jxl_gpu_bitstream::parse(bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
        .image_header;
    let intrinsic = options
        .intrinsic_size
        .map(|size| (size.width(), size.height()));
    assert_eq!(info.size, (extent.width, extent.height));
    assert_eq!(info.orientation, options.orientation.to_exif_value());
    assert_eq!(
        info.intrinsic_size,
        intrinsic.unwrap_or((extent.width, extent.height))
    );
    assert_eq!(
        parsed
            .metadata
            .intrinsic_size
            .map(|size| (size.width, size.height)),
        intrinsic
    );
    assert_eq!(header.intrinsic_size, intrinsic);
    assert_eq!(info.preview, header.preview_size);
    assert_eq!(info.animation, header.animation.is_some());
    let (relative, threshold) = match options.linear_below {
        ToneMappingThreshold::AbsoluteNits(value) => (false, value),
        ToneMappingThreshold::DisplayFraction(value) => (true, value),
    };
    assert_eq!(info.relative_to_max_display, relative);
    assert_eq!(
        parsed.metadata.tone_mapping.relative_to_max_display,
        relative
    );
    assert_eq!(header.tone_mapping.relative_to_max_display, relative);
    for (actual, rust, inventory, expected) in [
        (
            info.intensity_target,
            parsed.metadata.tone_mapping.intensity_target,
            header.tone_mapping.intensity_target.to_f32(),
            options.intensity_target,
        ),
        (
            info.min_nits,
            parsed.metadata.tone_mapping.min_nits,
            header.tone_mapping.min_nits.to_f32(),
            options.min_nits,
        ),
        (
            info.linear_below,
            parsed.metadata.tone_mapping.linear_below,
            header.tone_mapping.linear_below.to_f32(),
            threshold,
        ),
    ] {
        for value in [actual, rust, inventory] {
            assert_eq!(value.to_bits(), expected.to_f32().to_bits());
        }
    }
}

fn sections(bytes: &[u8]) -> Vec<Vec<u8>> {
    jxl_gpu_bitstream::parse(bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
        .frames
        .iter()
        .flat_map(|frame| &frame.sections)
        .map(|section| {
            bytes[section.bytes.offset as usize..section.bytes.end().unwrap() as usize].to_vec()
        })
        .collect()
}

pub(super) fn encode(
    rig: &Rig,
    source: BufferImageSource,
    config: VarDctConfig,
    topology: usize,
) -> (Vec<u8>, u64) {
    if topology == 0 {
        let encoder = LosslessModularEncoder::new(rig.context.clone())
            .with_image_options(config.image_options)
            .unwrap();
        let owned = encoder.memory_plan(&source).unwrap().owned_bytes_per_job;
        (encoder.encode(source).unwrap(), owned)
    } else if topology == 3 {
        let encoder = TiledVarDctEncoder::new_with_config(rig.context.clone(), config).unwrap();
        let owned = encoder.memory_plan(&source).unwrap().owned_bytes_per_job;
        (encoder.encode(source).unwrap(), owned)
    } else {
        let encoder = if topology == 1 {
            VarDctEncoder::new_with_config(rig.context.clone(), VarDctStrategy::Dct8, config)
        } else {
            let extent = source.layout.extent;
            VarDctEncoder::new_with_strategy_map(
                rig.context.clone(),
                VarDctStrategyMap::new(
                    extent.width,
                    extent.height,
                    (0..extent.height.div_ceil(8))
                        .flat_map(|y| {
                            (0..extent.width.div_ceil(8))
                                .map(move |x| VarDctTransform::new(x, y, VarDctStrategy::Dct8))
                        })
                        .collect(),
                )
                .unwrap(),
                config,
            )
        }
        .unwrap();
        let owned = encoder.memory_plan(&source).unwrap().owned_bytes_per_job;
        (encoder.encode(source).unwrap(), owned)
    }
}

#[test]
fn intrinsic_size_and_tone_declarations_do_not_change_coded_geometry_or_payload() {
    let rig = Rig::new();
    let native = IccProfileOracle::compile();
    let sizes = [
        (1, 1),
        (512, 513),
        (513, 8192),
        (8192, 8193),
        (8193, 262144),
        (262144, 262145),
        (262145, 1 << 30),
        (1 << 31, 1 << 30),
    ];
    for topology in 0..4 {
        let extent = [
            Extent2d::new(17, 9),
            Extent2d::new(8, 8),
            Extent2d::new(17, 9),
            Extent2d::new(259, 3),
        ][topology];
        let (source, words) = source(&rig, extent, ColorSampleFormat::RGB8, false, 4);
        let config = VarDctConfig {
            color_transform: if topology == 1 {
                VarDctColorTransform::Xyb
            } else {
                VarDctColorTransform::Original
            },
            ..Default::default()
        };
        let (baseline, memory) = encode(&rig, source.clone(), config.clone(), topology);
        check_metadata(&baseline, &native, extent, ImageOptions::default());
        for (index, (width, height)) in sizes.into_iter().enumerate() {
            let options = ImageOptions {
                intrinsic_size: Some(IntrinsicSize::new(width, height).unwrap()),
                orientation: OutputOrientation::from_exif_value(index as u32 + 1).unwrap(),
                // Include intrinsic-only/default tone, absolute light, and fractional light.
                min_nits: half(if index % 3 == 0 { 0 } else { 0x2c00 }),
                linear_below: match index % 3 {
                    0 => ToneMappingThreshold::default(),
                    1 => ToneMappingThreshold::AbsoluteNits(half(0x4d00)), // 20 nit
                    _ => ToneMappingThreshold::DisplayFraction(half(0x3000)), // 1/8
                },
                ..Default::default()
            };
            let (bytes, owned) = encode(
                &rig,
                source.clone(),
                VarDctConfig {
                    image_options: options,
                    ..config.clone()
                },
                topology,
            );
            check_metadata(&bytes, &native, extent, options);
            assert_eq!(
                owned, memory,
                "display hint must not allocate image storage"
            );
            assert_eq!(sections(&bytes), sections(&baseline));
            if topology == 0 {
                let frames = modular_words::original_frames(&bytes);
                for (c, plane) in frames[0].planes.iter().enumerate() {
                    assert_eq!(
                        plane.iter().map(|&word| word as u32).collect::<Vec<_>>(),
                        words.iter().skip(c).step_by(3).copied().collect::<Vec<_>>()
                    );
                }
            }
            rig.check_output(&bytes, extent, options.orientation, false);
            assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn tone_binary16_edges_preserve_exact_declarations() {
    let rig = Rig::new();
    let native = IccProfileOracle::compile();
    let extent = Extent2d::new(3, 2);
    let (source, _) = source(&rig, extent, ColorSampleFormat::GRAY8, false, 1);
    let mut cases = Vec::new();
    for bits in [1, 0x0400, 0x5bf8, 0x7bff] {
        let value = half(bits);
        cases.push(ImageOptions {
            intensity_target: value,
            min_nits: value,
            linear_below: ToneMappingThreshold::AbsoluteNits(half(0x7bff)),
            ..Default::default()
        });
    }
    for bits in [0, 0x8000, 1, 0x3c00] {
        cases.push(ImageOptions {
            min_nits: half(0x8000),
            linear_below: ToneMappingThreshold::DisplayFraction(half(bits)),
            ..Default::default()
        });
    }
    cases.push(ImageOptions {
        linear_below: ToneMappingThreshold::AbsoluteNits(half(0x8000)),
        ..Default::default()
    });
    let mut payload = None;
    for options in cases {
        let bytes = LosslessModularEncoder::new(rig.context.clone())
            .with_image_options(options)
            .unwrap()
            .encode(source.clone())
            .unwrap();
        check_metadata(&bytes, &native, extent, options);
        let actual = sections(&bytes);
        if let Some(expected) = &payload {
            assert_eq!(&actual, expected);
        }
        payload = Some(actual);
    }
    assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
}

#[test]
fn invalid_light_declarations_reject_all_encoder_entry_points_before_admission() {
    let rig = Rig::new();
    for options in [
        ImageOptions {
            intensity_target: half(0),
            ..Default::default()
        },
        ImageOptions {
            intensity_target: half(0x8000),
            ..Default::default()
        },
        ImageOptions {
            intensity_target: half(0xbc00),
            ..Default::default()
        },
        ImageOptions {
            min_nits: half(0x8001),
            ..Default::default()
        },
        ImageOptions {
            min_nits: half(0x5bf9),
            ..Default::default()
        },
        ImageOptions {
            linear_below: ToneMappingThreshold::AbsoluteNits(half(0x8001)),
            ..Default::default()
        },
        ImageOptions {
            linear_below: ToneMappingThreshold::DisplayFraction(half(0x3c01)),
            ..Default::default()
        },
        ImageOptions {
            linear_below: ToneMappingThreshold::DisplayFraction(half(0x8001)),
            ..Default::default()
        },
    ] {
        assert!(matches!(
            LosslessModularEncoder::new(rig.context.clone()).with_image_options(options),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        let config = VarDctConfig {
            image_options: options,
            ..Default::default()
        };
        assert!(matches!(
            VarDctEncoder::new_with_config(
                rig.context.clone(),
                VarDctStrategy::Dct8,
                config.clone()
            ),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            TiledVarDctEncoder::new_with_config(rig.context.clone(), config.clone()),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            VarDctEncoder::new_with_strategy_map(
                rig.context.clone(),
                VarDctStrategyMap::new(
                    8,
                    8,
                    vec![VarDctTransform::new(0, 0, VarDctStrategy::Dct8)]
                )
                .unwrap(),
                config.clone()
            ),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            MixedModeEncoder::new(
                rig.context.clone(),
                MixedModeConfig {
                    vardct: config,
                    ..Default::default()
                }
            ),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
    }
}
