use super::*;

#[test]
fn cmyk_integer_and_explicit_complemented_float_words_keep_every_precision() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let profile = profile("lut16_xyz_4");
    let extent = Extent2d::new(7, 3);
    let precisions = (1..=31)
        .map(|bits| (bits, 0))
        .chain((2..=8).flat_map(|exponent| {
            (2..=23).map(move |fraction| (1 + exponent + fraction, exponent))
        }));
    let mut count = 0;
    for (bits, exponent) in precisions {
        let config = config(&profile, bits, exponent, true);
        let mask = u32::MAX >> (32 - bits);
        let encoding = if exponent == 0 {
            CmykSampleEncoding::InkAmounts
        } else {
            CmykSampleEncoding::Complemented
        };
        let words: Vec<_> = (0..extent.area().unwrap() * 5)
            .map(|i| {
                let special = if exponent == 0 {
                    [0, mask, 1, mask / 2]
                } else {
                    let fraction = bits - exponent - 1;
                    let infinity = ((1u32 << exponent) - 1) << fraction;
                    // Signed zero, subnormal, infinity and noncanonical NaN payloads are data.
                    [1u32 << (bits - 1), 1, infinity, infinity | 3]
                };
                if i % 7 < 4 {
                    special[i % 7]
                } else {
                    (i as u32).wrapping_mul(91_712_711) & mask
                }
            })
            .collect();
        let source = input(&context, extent, &config, Storage::Planar, encoding, &words);
        let reference = expected(extent, &words, bits, true, encoding);
        for entropy in [
            LosslessModularEntropyCoding::Prefix,
            LosslessModularEntropyCoding::Ans,
        ] {
            let encoder = LosslessModularEncoder::with_config(
                context.clone(),
                LosslessModularConfig {
                    entropy,
                    ..Default::default()
                },
            )
            .with_image_options(config.image_options)
            .unwrap();
            let bytes = encoder.encode(source.clone()).unwrap();
            assert_eq!(
                modular_words::channel_frames(&bytes)[0],
                reference,
                "native {bits}/{exponent}/{entropy:?}"
            );
            assert_eq!(
                modular_integer::modular_channel_words(&bytes, 0),
                reference,
                "Rust {bits}/{exponent}/{entropy:?}"
            );
        }
        // Lossy color remains finite while Black and alpha retain arbitrary raw words.
        let mut finite = words;
        for pixel in finite.as_chunks_mut::<5>().0 {
            pixel[..3].fill(if exponent == 0 {
                mask / 3
            } else {
                ((1 << (exponent - 1)) - 1) << (bits - exponent - 1)
            });
        }
        let source = input(&context, extent, &config, Storage::Split, encoding, &finite);
        let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
        let bytes = encoder.encode(source).unwrap();
        assert_eq!(
            modular_integer::vardct_extra_words(&bytes, 0),
            reference[3..],
            "VarDCT {bits}/{exponent}"
        );
        count += 1;
    }
    assert_eq!(count, 185);
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn primary_black_joins_transforms_groups_and_independent_scalar_routes() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let profile = profile("ab_lab_4");
    for (extent, storage, bits) in [
        (Extent2d::new(17, 9), Storage::SharedWord, 5),
        (Extent2d::new(19, 13), Storage::MixedWords, 7),
        (Extent2d::new(259, 5), Storage::ThreeBytes, 13),
        (Extent2d::new(2051, 9), Storage::Planar, 16),
    ] {
        let config = config(&profile, bits, 0, true);
        let words: Vec<_> = (0..extent.area().unwrap() * 5)
            .map(|v| v as u32 % 31)
            .collect();
        let definition = ExtraChannel::new(
            ExtraChannelKind::Depth,
            SamplePrecision::integer(13).unwrap(),
            3,
            b"depth".to_vec(),
        )
        .unwrap();
        let scalar_extent = definition.source_extent(extent);
        let scalar: Vec<_> = (0..scalar_extent.area().unwrap())
            .map(|v| v as u32 * 71 % 8192)
            .collect();
        let source = input(
            &context,
            extent,
            &config,
            storage,
            CmykSampleEncoding::InkAmounts,
            &words,
        )
        .with_extra_channels(vec![raw_source(
            &context,
            scalar_extent,
            definition.precision().pixel_format(),
            Storage::Packed,
            &scalar,
        )])
        .unwrap();
        let mut reference = expected(extent, &words, bits, true, CmykSampleEncoding::InkAmounts);
        reference.push(modular_integer::ExtraWords {
            width: scalar_extent.width,
            height: scalar_extent.height,
            words: scalar,
        });
        for entropy in [
            LosslessModularEntropyCoding::Prefix,
            LosslessModularEntropyCoding::Ans,
        ] {
            let encoder = LosslessModularEncoder::with_config(
                context.clone(),
                LosslessModularConfig {
                    entropy,
                    extra_channels: vec![definition.clone()],
                    group_size: LosslessModularGroupSize::Pixels128,
                    palette: Some(
                        LosslessModularPalette::new(128)
                            .unwrap()
                            .with_components(1, 2)
                            .unwrap(),
                    ),
                    local_transforms: LosslessModularSqueeze::HorizontalThenVertical
                        .with_channels(2, 2)
                        .unwrap()
                        .into(),
                    ..Default::default()
                },
            )
            .with_image_options(config.image_options)
            .unwrap();
            let bytes = encoder.encode(source.clone()).unwrap();
            assert_eq!(
                modular_words::channel_frames(&bytes)[0],
                reference,
                "native {extent:?}/{entropy:?}"
            );
            assert_eq!(modular_integer::modular_channel_words(&bytes, 0), reference);
        }
    }
}
