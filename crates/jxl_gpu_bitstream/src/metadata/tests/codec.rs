use super::*;

#[test]
fn brotli_boundaries_reject_truncation_trailing_data_and_extensions() {
    let limits = MetadataLimits::default();
    let input = (0..12_987)
        .map(|n| ((n * 117 + n / 11) % 251) as u8)
        .collect::<Vec<_>>();
    let encoded = item(JUMBF, &input, true);
    assert_eq!(
        encoded
            .decode(MetadataLimits {
                max_decoded_box_bytes: input.len() as u64,
                ..limits
            })
            .unwrap()
            .as_ref(),
        input
    );
    assert!(matches!(
        encoded.decode(MetadataLimits {
            max_decoded_box_bytes: input.len() as u64 - 1,
            ..limits
        }),
        Err(MetadataError::Limit {
            resource: MetadataResource::DecodedBoxBytes,
            ..
        })
    ));
    for end in 4..encoded.payload.len() {
        let truncated = MetadataBox::from_encoded(
            ContainerBoxRef {
                box_type: BROB,
                payload: &encoded.payload[..end],
            },
            limits,
        )
        .unwrap();
        assert!(truncated.decode(limits).is_err(), "prefix {end}");
    }
    for suffix in [&[0][..], &encoded.payload[4..]] {
        let mut extended = encoded.clone();
        extended.payload.extend_from_slice(suffix);
        assert_eq!(
            extended.decode(limits),
            Err(MetadataError::TrailingBrotliData)
        );
    }
    let mut extension = encoded.clone();
    extension.payload[4] = 0x11;
    assert_eq!(extension.decode(limits), Err(MetadataError::InvalidBrotli));
    assert!(matches!(
        encoded.decode(MetadataLimits {
            max_brotli_window_bits: 10,
            ..limits
        }),
        Err(MetadataError::Limit {
            resource: MetadataResource::BrotliWindowBits,
            ..
        })
    ));
}

#[test]
fn expansion_and_compression_limits_include_exact_empty_and_large_outputs() {
    let limits = MetadataLimits {
        max_expansion_ratio: u32::MAX,
        ..MetadataLimits::default()
    };
    let input = vec![42; 256 * 1024];
    let encoded = MetadataBox::new(
        XMP,
        &input,
        MetadataCompression::Brotli(BrotliOptions::default()),
        limits,
    )
    .unwrap();
    assert_eq!(encoded.decode(limits).unwrap().as_ref(), input);
    assert!(matches!(
        encoded.decode(MetadataLimits {
            max_expansion_ratio: 3,
            ..limits
        }),
        Err(MetadataError::ExpansionLimit { max_ratio: 3, .. })
    ));
    assert!(matches!(
        MetadataBox::new(
            XMP,
            &input,
            MetadataCompression::Brotli(BrotliOptions::default()),
            MetadataLimits {
                max_expansion_ratio: 3,
                ..limits
            }
        ),
        Err(MetadataError::ExpansionLimit { .. })
    ));
    let empty = item(XMP, &[], true);
    assert!(
        empty
            .decode(MetadataLimits {
                max_decoded_box_bytes: 0,
                max_expansion_ratio: 0,
                ..limits
            })
            .unwrap()
            .is_empty()
    );
    for maximum in [0, 3, 4, 5] {
        assert!(matches!(
            MetadataBox::new(
                XMP,
                &input,
                MetadataCompression::Brotli(BrotliOptions::default()),
                MetadataLimits {
                    max_encoded_box_bytes: maximum,
                    ..limits
                }
            ),
            Err(MetadataError::Limit {
                resource: MetadataResource::EncodedBoxBytes,
                ..
            })
        ));
    }
    assert!(BrotliOptions::new(12, 22).is_none());
    assert!(BrotliOptions::new(6, 9).is_none());
    assert!(BrotliOptions::new(6, 25).is_none());
    assert!(BrotliOptions::new(0, 10).is_some());
    assert!(BrotliOptions::new(11, 24).is_some());
}
