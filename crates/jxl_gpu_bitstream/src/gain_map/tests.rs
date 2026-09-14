use super::*;

#[test]
fn iso_fraction_layouts_roundtrip_without_reduction() {
    for multi in [false, true] {
        for equal_denominators in [false, true] {
            for descending_headroom in [false, true] {
                for base in [false, true] {
                    let mut metadata = GainMapMetadata {
                        use_base_color_space: base,
                        ..Default::default()
                    };
                    if descending_headroom {
                        std::mem::swap(
                            &mut metadata.base_hdr_headroom,
                            &mut metadata.alternate_hdr_headroom,
                        );
                    }
                    metadata.channels[0].min.numerator = -3;
                    metadata.channels = [metadata.channels[0]; 3];
                    if multi {
                        metadata.channels[1].max.numerator = 8;
                    }
                    if !equal_denominators {
                        metadata.alternate_hdr_headroom.denominator = 7;
                    }
                    let bytes = metadata.encode().unwrap();
                    assert_eq!(bytes.len(), if multi { 141 } else { 61 });
                    assert_eq!(bytes[4], (u8::from(multi) << 7) | (u8::from(base) << 6));
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
    for (index, value) in [(0, 1), (1, 1)] {
        let mut invalid = bytes.clone();
        invalid[index] = value;
        assert!(GainMapMetadata::parse(&invalid).is_err());
    }
    // All six bits are reserved, including the removed draft direction/common-denominator bits.
    for bit in 0..6 {
        let mut invalid = bytes.clone();
        invalid[4] |= 1 << bit;
        assert!(GainMapMetadata::parse(&invalid).is_err());
    }
    let mut invalid = bytes.clone();
    invalid[9..13].fill(0);
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
fn compatible_writer_extensions_are_preserved_and_bounded_by_the_bundle_length_field() {
    for version in [1, 255, u16::MAX] {
        let mut bytes = GainMapMetadata::default().encode().unwrap();
        bytes[2..4].copy_from_slice(&version.to_be_bytes());
        for extension in [&[][..], &[13, 0, 255, 7][..]] {
            let mut record = bytes.clone();
            record.extend_from_slice(extension);
            let metadata = GainMapMetadata::parse(&record).unwrap();
            assert_eq!(metadata.writer_version, version);
            assert_eq!(metadata.extensions, extension);
            assert_eq!(metadata.encode().unwrap(), record);
            let bundle =
                GainMapBundle::new(metadata, &[], &[], &[0xff, 0x0a], Default::default()).unwrap();
            let payload = bundle.encode(Default::default()).unwrap();
            assert_eq!(
                GainMapBundle::parse(&payload, Default::default())
                    .unwrap()
                    .encode(Default::default())
                    .unwrap(),
                payload
            );
        }
    }
    let mut metadata = GainMapMetadata {
        writer_version: 1,
        extensions: vec![7; usize::from(u16::MAX) - 61],
        ..Default::default()
    };
    let record = metadata.encode().unwrap();
    assert_eq!(record.len(), usize::from(u16::MAX));
    assert_eq!(GainMapMetadata::parse(&record).unwrap(), metadata);
    metadata.extensions.push(0);
    assert!(matches!(metadata.encode(), Err(GainMapError::Limit { .. })));
    assert!(matches!(
        GainMapMetadata::parse(&vec![0; 65536]),
        Err(GainMapError::Limit { .. })
    ));
    metadata.writer_version = 0;
    assert!(matches!(metadata.validate(), Err(GainMapError::Invalid(_))));
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
