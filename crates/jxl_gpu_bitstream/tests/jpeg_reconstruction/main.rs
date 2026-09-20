use std::{mem::size_of, ops::Range};

use jxl_gpu_bitstream::{
    ContainerBox, ParseLimits, ParsedJxl,
    jpeg_reconstruction::{
        AppMarker, HuffmanTable, JBRD, JpegComponent, JpegReconstructionEncodeOptions as Options,
        JpegReconstructionError as Error, JpegReconstructionLimits as Limits,
        JpegReconstructionMetadata as Metadata, JpegReconstructionResource as Resource, JpegScan,
        QuantizationTable, ScanComponent,
    },
    metadata::{BrotliOptions, MetadataError, MetadataResource},
    parse, write_container_with_boxes,
};

mod native;

use jxl_test_support::corpus::jpeg_reconstruction::CASES;

fn payload<'a>(parsed: &'a ParsedJxl<'_>) -> &'a [u8] {
    parsed
        .auxiliary_boxes()
        .iter()
        .find(|b| b.box_type == JBRD)
        .unwrap()
        .payload
}

fn replace(parsed: &ParsedJxl<'_>, replacement: &[u8]) -> Vec<u8> {
    let boxes: Vec<_> = parsed
        .auxiliary_boxes()
        .iter()
        .map(|b| ContainerBox {
            box_type: b.box_type,
            payload: if b.box_type == JBRD {
                replacement
            } else {
                b.payload
            },
        })
        .collect();
    write_container_with_boxes(parsed.codestream(), &boxes).unwrap()
}

fn assert_metadata(a: &Metadata, b: &Metadata) {
    assert_eq!(a.grayscale_hint(), b.grayscale_hint());
    assert_eq!(a.markers(), b.markers());
    assert_eq!(a.app_markers(), b.app_markers());
    assert_eq!(
        a.comments().collect::<Vec<_>>(),
        b.comments().collect::<Vec<_>>()
    );
    assert_eq!(a.quantization_tables(), b.quantization_tables());
    assert_eq!(a.components(), b.components());
    assert_eq!(a.huffman_tables(), b.huffman_tables());
    assert_eq!(a.scans(), b.scans());
    assert_eq!(a.restart_interval(), b.restart_interval());
    assert_eq!(
        a.intermarker_data().collect::<Vec<_>>(),
        b.intermarker_data().collect::<Vec<_>>()
    );
    assert_eq!(a.tail(), b.tail());
    assert_eq!(a.has_preserved_padding(), b.has_preserved_padding());
    assert_eq!(a.padding_bit_count(), b.padding_bit_count());
    assert_eq!(a.padding_bytes(), b.padding_bytes());
    assert_eq!(a.opaque_body(), b.opaque_body());
    assert_eq!(a.logical_owned_bytes(), b.logical_owned_bytes());
}

fn storage_cost(m: &Metadata) -> (u64, usize) {
    let counts = [
        (m.markers().len(), size_of::<u8>()),
        (m.app_markers().len(), size_of::<AppMarker>()),
        (m.comments().len(), size_of::<Range<usize>>()),
        (
            m.quantization_tables().len(),
            size_of::<QuantizationTable>(),
        ),
        (m.components().len(), size_of::<JpegComponent>()),
        (m.huffman_tables().len(), size_of::<HuffmanTable>()),
        (
            m.huffman_tables().iter().map(|h| h.values.len()).sum(),
            size_of::<u16>(),
        ),
        (m.scans().len(), size_of::<JpegScan>()),
        (
            m.scans().iter().map(|s| s.components.len()).sum(),
            size_of::<ScanComponent>(),
        ),
        (
            m.scans().iter().map(|s| s.resets.len()).sum(),
            size_of::<u32>(),
        ),
        (
            m.scans().iter().map(|s| s.extra_zeros.len()).sum(),
            size_of::<(u32, u8)>(),
        ),
        (m.intermarker_data().len(), size_of::<Range<usize>>()),
        (m.padding_bytes().len(), size_of::<u8>()),
    ];
    (
        (counts.iter().map(|(n, bytes)| n * bytes).sum::<usize>() + m.opaque_body().len()) as u64,
        counts.iter().map(|(n, _)| n).sum(),
    )
}

fn assert_limit<T: std::fmt::Debug>(result: Result<T, Error>, expected: Resource) {
    match result.unwrap_err() {
        Error::Limit {
            resource,
            required,
            limit,
        } => {
            assert_eq!(resource, expected);
            assert!(required > limit);
        }
        error => panic!("expected {expected:?} limit, got {error}"),
    }
}

#[test]
fn pinned_corpus_preserves_all_metadata_and_canonical_output() {
    let mut scan_count = 0;
    for case in CASES {
        case.validate();
        let parsed = parse(case.input, ParseLimits::default()).unwrap();
        let metadata = parsed
            .jpeg_reconstruction(Limits::default())
            .unwrap()
            .unwrap();
        assert_eq!(metadata.scans().len(), case.scans, "{}", case.name);
        assert_eq!(
            metadata.padding_bit_count(),
            case.padding_bits,
            "{}",
            case.name
        );
        scan_count += metadata.scans().len();
        let encoded = metadata.encode(Options::default()).unwrap();
        let reparsed = Metadata::parse(encoded.as_bytes(), Limits::default()).unwrap();
        assert_eq!(encoded.header_bytes(), reparsed.source_header_bytes());
        assert_eq!(
            encoded.header_bytes() + encoded.compressed_body_bytes(),
            encoded.as_bytes().len()
        );
        assert_metadata(&metadata, &reparsed);
        assert_eq!(
            encoded.as_bytes(),
            reparsed.encode(Options::default()).unwrap().as_bytes()
        );
        let rewritten = replace(&parsed, encoded.as_bytes());
        let checked = parse(&rewritten, ParseLimits::default()).unwrap();
        assert_eq!(checked.codestream(), parsed.codestream());
        let other_boxes = |input: &ParsedJxl<'_>| {
            input
                .auxiliary_boxes()
                .iter()
                .filter(|b| b.box_type != JBRD)
                .map(|b| (b.box_type, b.payload.to_vec()))
                .collect::<Vec<_>>()
        };
        assert_eq!(other_boxes(&checked), other_boxes(&parsed));
    }
    assert_eq!(CASES.len(), 36);
    assert_eq!(scan_count, 140);
}

#[test]
fn exact_limits_account_for_nested_records_and_simultaneous_emission() {
    for case in CASES {
        let parsed = parse(case.input, ParseLimits::default()).unwrap();
        let input = payload(&parsed);
        let metadata = Metadata::parse(input, Limits::default()).unwrap();
        let (owned, entries) = storage_cost(&metadata);
        assert_eq!(owned, metadata.logical_owned_bytes());
        let exact = Limits {
            max_encoded_bytes: input.len() as u64,
            max_owned_bytes: owned,
            max_markers: metadata.markers().len(),
            max_entries: entries,
            max_decoded_body_bytes: metadata.opaque_body().len() as u64,
            ..Limits::default()
        };
        Metadata::parse(input, exact).unwrap();
        for (limits, resource) in [
            (
                Limits {
                    max_encoded_bytes: exact.max_encoded_bytes - 1,
                    ..exact
                },
                Resource::EncodedBytes,
            ),
            (
                Limits {
                    max_owned_bytes: owned - 1,
                    ..exact
                },
                Resource::OwnedBytes,
            ),
            (
                Limits {
                    max_markers: exact.max_markers - 1,
                    ..exact
                },
                Resource::Markers,
            ),
            (
                Limits {
                    max_entries: entries - 1,
                    ..exact
                },
                Resource::Entries,
            ),
        ] {
            assert_limit(Metadata::parse(input, limits), resource);
        }
        if exact.max_decoded_body_bytes != 0 {
            assert_limit(
                Metadata::parse(
                    input,
                    Limits {
                        max_decoded_body_bytes: exact.max_decoded_body_bytes - 1,
                        ..exact
                    },
                ),
                Resource::DecodedBodyBytes,
            );
        }
        let encoded = metadata.encode(Options::default()).unwrap();
        assert_eq!(
            encoded.logical_peak_owned_bytes(),
            owned + encoded.as_bytes().len() as u64 + encoded.compressed_body_bytes() as u64
        );
        let exact = Options {
            max_encoded_bytes: encoded.as_bytes().len() as u64,
            max_owned_bytes: encoded.logical_peak_owned_bytes(),
            ..Options::default()
        };
        assert_eq!(
            metadata.encode(exact).unwrap().as_bytes(),
            encoded.as_bytes()
        );
        assert_limit(
            metadata.encode(Options {
                max_encoded_bytes: exact.max_encoded_bytes - 1,
                ..exact
            }),
            Resource::EncodedBytes,
        );
        assert_limit(
            metadata.encode(Options {
                max_owned_bytes: exact.max_owned_bytes - 1,
                ..exact
            }),
            Resource::OwnedBytes,
        );
        assert_limit(
            metadata.encode(Options {
                max_owned_bytes: owned - 1,
                ..exact
            }),
            Resource::OwnedBytes,
        );
        assert_limit(
            metadata.encode(Options {
                max_encoded_bytes: 0,
                ..exact
            }),
            Resource::EncodedBytes,
        );
        // A failed output admission leaves the immutable input usable for a valid retry.
        assert_eq!(
            metadata.encode(exact).unwrap().into_bytes(),
            encoded.into_bytes()
        );
    }
}

#[test]
fn window_and_expansion_limits_apply_to_parse_and_emission() {
    let parsed = parse(CASES[0].input, ParseLimits::default()).unwrap();
    let input = payload(&parsed);
    let metadata = Metadata::parse(input, Limits::default()).unwrap();
    assert!(!metadata.opaque_body().is_empty());
    for result in [
        Metadata::parse(
            input,
            Limits {
                max_brotli_window_bits: 0,
                ..Limits::default()
            },
        )
        .map(|_| ()),
        metadata
            .encode(Options {
                max_brotli_window_bits: 0,
                ..Options::default()
            })
            .map(|_| ()),
    ] {
        assert!(matches!(
            result,
            Err(Error::Metadata(MetadataError::Limit {
                resource: MetadataResource::BrotliWindowBits,
                ..
            }))
        ));
    }
    for result in [
        Metadata::parse(
            input,
            Limits {
                max_expansion_ratio: 0,
                ..Limits::default()
            },
        )
        .map(|_| ()),
        metadata
            .encode(Options {
                max_expansion_ratio: 0,
                ..Options::default()
            })
            .map(|_| ()),
    ] {
        assert!(matches!(
            result,
            Err(Error::Metadata(MetadataError::ExpansionLimit { .. }))
        ));
    }
}

#[test]
fn every_truncation_and_trailing_stream_byte_is_rejected() {
    for case in CASES {
        let parsed = parse(case.input, ParseLimits::default()).unwrap();
        let input = payload(&parsed);
        for end in 0..input.len() {
            assert!(
                Metadata::parse(&input[..end], Limits::default()).is_err(),
                "{} truncated at {end}",
                case.name
            );
        }
        for byte in [0, 0xa5] {
            let mut trailing = input.to_vec();
            trailing.push(byte);
            assert_eq!(
                Metadata::parse(&trailing, Limits::default()).unwrap_err(),
                Error::Metadata(MetadataError::TrailingBrotliData)
            );
        }
    }
}

#[test]
fn complete_transport_inventory_rejects_duplicate_and_wrapped_boxes_first() {
    let raw = [0xff, 0x0a];
    assert!(
        parse(&raw, ParseLimits::default())
            .unwrap()
            .jpeg_reconstruction(Limits::default())
            .unwrap()
            .is_none()
    );
    let bad = ContainerBox {
        box_type: JBRD,
        payload: &[],
    };
    let duplicate = write_container_with_boxes(&raw, &[bad, bad]).unwrap();
    assert_eq!(
        parse(&duplicate, ParseLimits::default())
            .unwrap()
            .jpeg_reconstruction(Limits::default())
            .unwrap_err(),
        Error::DuplicateBox
    );
    let wrapped = ContainerBox {
        box_type: *b"brob",
        payload: b"jbrd\0",
    };
    for boxes in [vec![wrapped], vec![bad, wrapped], vec![wrapped, bad]] {
        let input = write_container_with_boxes(&raw, &boxes).unwrap();
        assert_eq!(
            parse(&input, ParseLimits::default())
                .unwrap()
                .jpeg_reconstruction(Limits::default())
                .unwrap_err(),
            Error::WrappedBox
        );
    }
}

#[test]
fn native_jpeg_reconstruction_is_exact_for_original_and_reencoded_metadata() {
    let oracle = native::Oracle::compile();
    let mut reconstructions = 0;
    let mut negatives = 0;
    for case in CASES {
        case.validate();
        oracle.assert_exact(case.input, case.jpeg);
        reconstructions += 1;
        let parsed = parse(case.input, ParseLimits::default()).unwrap();
        let metadata = parsed
            .jpeg_reconstruction(Limits::default())
            .unwrap()
            .unwrap();
        for (quality, window) in [(0, 10), (6, 22), (11, 24)] {
            let encoded = metadata
                .encode(Options {
                    brotli: BrotliOptions::new(quality, window).unwrap(),
                    ..Options::default()
                })
                .unwrap();
            let reparsed = Metadata::parse(encoded.as_bytes(), Limits::default()).unwrap();
            assert_metadata(&metadata, &reparsed);
            oracle.assert_exact(&replace(&parsed, encoded.as_bytes()), case.jpeg);
            reconstructions += 1;
        }
        if case.official {
            let boxes: Vec<_> = parsed
                .auxiliary_boxes()
                .iter()
                .filter(|b| b.box_type != JBRD)
                .map(|b| ContainerBox {
                    box_type: b.box_type,
                    payload: b.payload,
                })
                .collect();
            oracle.assert_rejected(
                &write_container_with_boxes(parsed.codestream(), &boxes).unwrap(),
                case.jpeg.len(),
            );
            negatives += 1;
            let input = payload(&parsed);
            for invalid in [
                Vec::new(),
                input[..input.len() - 1].to_vec(),
                input[..input.len() - 2].to_vec(),
                [input, &[0]].concat(),
                [input, &[0xa5]].concat(),
            ] {
                assert!(Metadata::parse(&invalid, Limits::default()).is_err());
                oracle.assert_rejected(&replace(&parsed, &invalid), case.jpeg.len());
                negatives += 1;
            }
        }
    }
    assert_eq!(reconstructions, 144);
    assert_eq!(negatives, 18);
    eprintln!(
        "{reconstructions} native JPEG byte comparisons and {negatives} native negative cases passed"
    );
}
