use super::*;

#[test]
fn iso_fraction_layouts_roundtrip_without_reduction() {
    for multi in [false, true] {
        for common in [false, true] {
            for backward in [false, true] {
                for base in [false, true] {
                    let mut metadata = GainMapMetadata {
                        backward_direction: backward,
                        use_base_color_space: base,
                        ..Default::default()
                    };
                    metadata.channels[0].min.numerator = -3;
                    metadata.channels = [metadata.channels[0]; 3];
                    if multi {
                        metadata.channels[1].max.numerator = 8;
                    }
                    if !common {
                        metadata.alternate_hdr_headroom.denominator = 7;
                    }
                    let bytes = metadata.encode().unwrap();
                    assert_eq!(
                        bytes.len(),
                        match (multi, common) {
                            (false, true) => 37,
                            (true, true) => 77,
                            (false, false) => 61,
                            (true, false) => 141,
                        }
                    );
                    assert_eq!(GainMapMetadata::parse(&bytes).unwrap(), metadata);
                    for length in 0..bytes.len() {
                        assert!(GainMapMetadata::parse(&bytes[..length]).is_err());
                    }
                    let mut trailing = bytes.clone();
                    trailing.push(0);
                    assert!(GainMapMetadata::parse(&trailing).is_err());
                }
            }
        }
    }
}

#[test]
fn iso_rejects_bad_versions_flags_fractions_and_exact_reversed_ranges() {
    let bytes = GainMapMetadata::default().encode().unwrap();
    for (index, value) in [(1, 1), (3, 1), (4, bytes[4] | 1), (4, bytes[4] | 0x10)] {
        let mut invalid = bytes.clone();
        invalid[index] = value;
        assert!(GainMapMetadata::parse(&invalid).is_err());
    }
    let mut invalid = bytes.clone();
    invalid[5..9].fill(0);
    assert!(GainMapMetadata::parse(&invalid).is_err());
    for channel in 0..3 {
        let mut m = GainMapMetadata::default();
        m.channels[channel].gamma.numerator = 0;
        assert!(m.encode().is_err());
        m.channels[channel].gamma.numerator = 1;
        m.channels[channel].min = SignedFraction {
            numerator: i32::MAX,
            denominator: u32::MAX - 1,
        };
        m.channels[channel].max = SignedFraction {
            numerator: i32::MAX,
            denominator: u32::MAX,
        };
        // These collapse to the same F32. Ordering must use the exact fractions.
        assert_eq!(
            m.channels[channel].min.value() as f32,
            m.channels[channel].max.value() as f32
        );
        assert!(m.encode().is_err());
    }
}

#[test]
fn jhgm_borrows_the_codestream_and_checks_every_length_before_use() {
    let limits = GainMapLimits::default();
    let code = [0xff, 0x0a, 7, 9];
    let bundle = GainMapBundle::new(Default::default(), &[1], &[], &code, limits).unwrap();
    let bytes = bundle.encode(limits).unwrap();
    let parsed = GainMapBundle::parse(&bytes, limits).unwrap();
    assert_eq!(parsed.codestream(), code);
    assert_eq!(
        parsed.codestream().as_ptr(),
        bytes[bytes.len() - code.len()..].as_ptr()
    );
    assert_eq!(parsed.encode(limits).unwrap(), bytes);
    assert!(matches!(
        parsed.alternate_color_encoding(),
        Some(ColourEncodingInventory::Enumerated { .. })
    ));
    for length in 0..bytes.len() - 2 {
        assert!(
            GainMapBundle::parse(&bytes[..length], limits).is_err(),
            "{length}"
        );
    }
    let mut version = bytes.clone();
    version[0] = 1;
    assert!(matches!(
        GainMapBundle::parse(&version, limits),
        Err(GainMapError::Version { .. })
    ));
    assert!(GainMapBundle::new(Default::default(), &[0x81], &[], &code, limits).is_err());
    assert!(GainMapBundle::new(Default::default(), &[1, 0], &[], &code, limits).is_err());
    assert!(GainMapBundle::new(Default::default(), &[], &[], b"container", limits).is_err());
    for smaller in [
        GainMapLimits {
            max_payload_bytes: bytes.len() as u64 - 1,
            ..limits
        },
        GainMapLimits {
            max_codestream_bytes: 3,
            ..limits
        },
    ] {
        assert!(matches!(
            GainMapBundle::parse(&bytes, smaller),
            Err(GainMapError::Limit { .. })
        ));
        assert!(matches!(
            parsed.encode(smaller),
            Err(GainMapError::Limit { .. })
        ));
    }
}

#[test]
fn alternate_icc_uses_bounded_metadata_reconstruction() {
    let raw = crate::test_fixtures::with_icc();
    let image = crate::parse(&raw, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
        .image_header;
    let icc = image.embedded_icc.unwrap();
    let start = icc.bit_range.offset;
    let mut bits = crate::BitWriter::new();
    for bit in start..icc.bit_range.end().unwrap() {
        bits.write_bits(u64::from((raw[bit as usize / 8] >> (bit % 8)) & 1), 1)
            .unwrap();
    }
    let compressed = bits.into_bytes();
    let limits = GainMapLimits::default();
    let bundle =
        GainMapBundle::new(Default::default(), &[], &compressed, &[0xff, 0x0a], limits).unwrap();
    assert_eq!(bundle.alternate_icc().unwrap(), &icc.profile);
    assert!(
        GainMapBundle::new(
            Default::default(),
            &[],
            &compressed,
            &[0xff, 0x0a],
            GainMapLimits {
                max_decoded_icc_bytes: icc.profile.len() as u64 - 1,
                ..limits
            }
        )
        .is_err()
    );
    assert!(
        GainMapBundle::new(
            Default::default(),
            &[],
            &compressed,
            &[0xff, 0x0a],
            GainMapLimits {
                max_transformed_icc_bytes: icc.encoded_byte_count - 1,
                ..limits
            }
        )
        .is_err()
    );
}
