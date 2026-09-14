use super::*;

mod codec;
#[cfg(not(target_arch = "wasm32"))]
mod native;
mod stream;

fn item(box_type: [u8; 4], payload: &[u8], compressed: bool) -> MetadataBox {
    MetadataBox::new(
        box_type,
        payload,
        if compressed {
            MetadataCompression::Brotli(BrotliOptions::default())
        } else {
            MetadataCompression::None
        },
        MetadataLimits::default(),
    )
    .unwrap()
}

#[test]
fn opaque_boxes_preserve_duplicates_compression_and_atomic_replacement() {
    let limits = MetadataLimits::default();
    let mut metadata = Metadata::default();
    for entry in [
        item(EXIF, b"\0\0\0\x02binary TIFF offset", true),
        item(XMP, b"<xml/>", false),
        item(EXIF, b"second", false),
        item(*b"priv", b"\0\xff\x80unknown", true),
    ] {
        metadata.push(entry, limits).unwrap();
    }
    let raw = [0xff, 0x0a, 4, 5];
    let wrapped = metadata.write_container(&raw).unwrap();
    let parsed = crate::parse(&wrapped, crate::ParseLimits::default()).unwrap();
    assert_eq!(parsed.codestream(), raw);
    assert_eq!(
        parsed.metadata(&MetadataSelection::All, limits).unwrap(),
        metadata
    );
    assert_eq!(metadata.boxes_of_type(EXIF).count(), 2);
    assert!(matches!(
        metadata.boxes[1].decode(limits).unwrap(),
        Cow::Borrowed(_)
    ));
    let replacement = item(EXIF, b"replacement", true);
    let before = metadata.clone();
    assert!(
        metadata
            .replace(
                EXIF,
                Some(replacement.clone()),
                MetadataLimits {
                    max_retained_bytes: 0,
                    ..limits
                }
            )
            .is_err()
    );
    assert_eq!(metadata, before);
    assert_eq!(
        metadata.replace(EXIF, Some(item(XMP, b"wrong", false)), limits),
        Err(MetadataError::ReplacementType)
    );
    assert_eq!(metadata, before);
    metadata
        .replace(EXIF, Some(replacement.clone()), limits)
        .unwrap();
    assert_eq!(metadata.boxes[0], replacement);
    assert_eq!(metadata.boxes.len(), 3);
    metadata.replace(XMP, None, limits).unwrap();
    metadata
        .replace(JUMBF, Some(item(JUMBF, b"new", false)), limits)
        .unwrap();
    assert_eq!(
        metadata
            .boxes
            .iter()
            .map(MetadataBox::box_type)
            .collect::<Vec<_>>(),
        [EXIF, *b"priv", JUMBF]
    );
    assert_eq!(
        metadata.retained_bytes,
        metadata
            .boxes
            .iter()
            .map(|entry| entry.payload.len() as u64)
            .sum()
    );
}

#[test]
fn individual_aggregate_and_selection_limits_are_independent() {
    let limits = MetadataLimits::default();
    let mut metadata = Metadata::default();
    metadata.push(item(EXIF, &[1; 200], true), limits).unwrap();
    metadata.push(item(XMP, &[2; 300], false), limits).unwrap();
    assert_eq!(
        metadata
            .decode_all(MetadataLimits {
                max_total_decoded_bytes: 500,
                ..limits
            })
            .unwrap()
            .len(),
        2
    );
    assert!(matches!(
        metadata.decode_all(MetadataLimits {
            max_total_decoded_bytes: 499,
            ..limits
        }),
        Err(MetadataError::Limit {
            resource: MetadataResource::TotalDecodedBytes,
            ..
        })
    ));
    assert!(matches!(
        metadata.decode_all(MetadataLimits {
            max_decoded_box_bytes: 199,
            ..limits
        }),
        Err(MetadataError::Limit {
            resource: MetadataResource::DecodedBoxBytes,
            ..
        })
    ));
    let wrapped = metadata.write_container(&[0xff, 0x0a]).unwrap();
    let parsed = crate::parse(&wrapped, crate::ParseLimits::default()).unwrap();
    let zero = MetadataLimits {
        max_boxes: 0,
        max_retained_bytes: 0,
        max_encoded_box_bytes: 0,
        ..limits
    };
    assert_eq!(
        parsed.metadata(&MetadataSelection::None, zero).unwrap(),
        Metadata::default()
    );
    let selection = MetadataSelection::Types(vec![EXIF]);
    let exact = MetadataLimits {
        max_boxes: 1,
        max_retained_bytes: metadata.boxes[0].payload.len() as u64,
        ..limits
    };
    assert_eq!(parsed.metadata(&selection, exact).unwrap().boxes.len(), 1);
    assert!(matches!(
        parsed.metadata(
            &selection,
            MetadataLimits {
                max_retained_bytes: exact.max_retained_bytes - 1,
                ..exact
            }
        ),
        Err(MetadataError::Limit {
            resource: MetadataResource::RetainedBytes,
            ..
        })
    ));
    assert!(matches!(
        parsed.metadata(&MetadataSelection::All, exact),
        Err(MetadataError::Limit {
            resource: MetadataResource::BoxCount,
            ..
        })
    ));
}

#[test]
fn reserved_types_and_brob_prefixes_are_checked_without_decompressing() {
    let limits = MetadataLimits::default();
    for box_type in [*b"JXL ", *b"ftyp", *b"jxlc", *b"jxlp"] {
        assert_eq!(
            MetadataBox::from_encoded(
                ContainerBoxRef {
                    box_type,
                    payload: b"arbitrary"
                },
                limits
            )
            .unwrap_err(),
            MetadataError::ForbiddenType(box_type)
        );
    }
    for box_type in [
        *b"JXL ", *b"ftyp", *b"jxlc", *b"jxlp", *b"jxli", *b"jxll", *b"jxlz", *b"jbrd", BROB,
    ] {
        assert_eq!(
            MetadataBox::from_encoded(
                ContainerBoxRef {
                    box_type: BROB,
                    payload: &box_type
                },
                limits
            )
            .unwrap_err(),
            MetadataError::ForbiddenType(box_type)
        );
    }
    for length in 0..4 {
        assert_eq!(
            MetadataBox::from_encoded(
                ContainerBoxRef {
                    box_type: BROB,
                    payload: &EXIF[..length]
                },
                limits
            )
            .unwrap_err(),
            MetadataError::TruncatedBrotliType
        );
    }
    let opaque = MetadataBox::from_encoded(
        ContainerBoxRef {
            box_type: BROB,
            payload: &EXIF,
        },
        limits,
    )
    .unwrap();
    assert_eq!(opaque.box_type(), EXIF);
    assert_eq!(opaque.decode(limits), Err(MetadataError::TruncatedBrotli));
}
