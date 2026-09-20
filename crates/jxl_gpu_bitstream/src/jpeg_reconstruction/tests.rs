use super::*;

fn sample() -> JpegReconstructionMetadata {
    let input = include_bytes!("../../test-data/jpeg_reconstruction/rgb_sequential/input.jxl");
    crate::parse(input, crate::ParseLimits::default())
        .unwrap()
        .jpeg_reconstruction(JpegReconstructionLimits::default())
        .unwrap()
        .unwrap()
}

// Only this private test module can alter a parsed model. Encode deliberately malformed
// declarations to exercise rejection at the wire boundary; public callers cannot do this.
fn rejects(
    mut metadata: JpegReconstructionMetadata,
    change: impl FnOnce(&mut JpegReconstructionMetadata),
    expected: &str,
) {
    change(&mut metadata);
    let bytes = metadata
        .encode(JpegReconstructionEncodeOptions::default())
        .unwrap();
    assert_eq!(
        JpegReconstructionMetadata::parse(bytes.as_bytes(), JpegReconstructionLimits::default())
            .unwrap_err()
            .to_string(),
        format!("invalid JPEG reconstruction metadata: {expected}")
    );
}

#[test]
fn huffman_alphabet_and_prefix_space_are_checked() {
    rejects(
        sample(),
        |m| {
            let counts = &mut m.huffman[0].counts;
            let first = counts.iter().position(|&count| count != 0).unwrap();
            counts[first] -= 1;
            counts[0] = 1;
        },
        "zero-length Huffman code",
    );
    rejects(
        sample(),
        |m| {
            let h = &mut m.huffman[0];
            let count = h.values.len() as u16;
            h.counts = [0; 17];
            h.counts[1] = count;
        },
        "oversubscribed Huffman code",
    );
    rejects(
        sample(),
        |m| m.huffman[0].values[1] = m.huffman[0].values[0],
        "duplicate Huffman symbol",
    );
    rejects(
        sample(),
        |m| m.huffman[0].values[0] = 12,
        "DC Huffman symbol",
    );
    rejects(
        sample(),
        |m| {
            let h = &mut m.huffman[0];
            let last = h.values.len() - 1;
            h.values.swap(0, last);
        },
        "Huffman terminal symbol",
    );
}

#[test]
fn table_groups_and_use_before_definition_are_checked() {
    rejects(sample(), |m| m.quant[0].last = false, "DQT group count");
    rejects(
        sample(),
        |m| m.markers.retain(|&v| v != 0xdb),
        "unconsumed quantization table",
    );
    rejects(
        sample(),
        |m| m.huffman.last_mut().unwrap().last = false,
        "DHT group count",
    );
    rejects(
        sample(),
        |m| {
            let first = m.markers.iter().position(|&v| v == 0xc4).unwrap();
            let last = m.markers.iter().position(|&v| v == 0xda).unwrap();
            m.markers.swap(first, last);
        },
        "DC table used before definition",
    );
    rejects(
        sample(),
        |m| m.scans[0].components[0].ac = 3,
        "AC table used before definition",
    );
    rejects(
        sample(),
        |m| m.scans[0].components[0].dc = 3,
        "DC table used before definition",
    );
    rejects(
        sample(),
        |m| {
            m.huffman[0].last = false;
            m.huffman[1].counts = [0; 17];
            m.huffman[1].values.clear();
        },
        "empty DHT within nonempty group",
    );
}

#[test]
fn selectors_counts_and_native_block_endpoints_are_checked() {
    rejects(
        sample(),
        |m| m.quant.resize(4, m.quant[0]),
        "quantization table count",
    );
    rejects(
        sample(),
        |m| m.components[0].quant = 3,
        "component quantization selector",
    );
    rejects(
        sample(),
        |m| {
            m.quant.push(m.quant[0]);
            for c in &mut m.components {
                c.quant = 1;
            }
        },
        "unused first quantization table",
    );
    rejects(sample(), |m| m.components.truncate(2), "component count");
    rejects(
        sample(),
        |m| m.scans[0].components[0].component = 3,
        "scan component selector",
    );
    rejects(
        sample(),
        |m| {
            let component = m.scans[0].components[0];
            m.scans[0].components.resize(4, component);
        },
        "scan component count",
    );
    rejects(
        sample(),
        |m| m.scans[0].resets = vec![3 << 26],
        "reset block index",
    );
    rejects(
        sample(),
        |m| m.scans[0].extra_zeros = vec![((3 << 26) + 1, 1)],
        "extra zero block index",
    );
    rejects(
        sample(),
        |m| m.scans[0].extra_zeros = vec![(0, 5)],
        "extra zero run count",
    );
    let mut metadata = sample();
    metadata.scans[0].resets = vec![(3 << 26) - 1];
    metadata.scans[0].extra_zeros = vec![(3 << 26, 4)];
    let bytes = metadata
        .encode(JpegReconstructionEncodeOptions::default())
        .unwrap();
    // Grammar endpoints do not authorize these indices for this actual image.
    let parsed =
        JpegReconstructionMetadata::parse(bytes.as_bytes(), JpegReconstructionLimits::default())
            .unwrap();
    assert_eq!(parsed.scans()[0].extra_zeros, metadata.scans[0].extra_zeros);
    assert_eq!(parsed.scans()[0].resets, metadata.scans[0].resets);
}

#[test]
fn opaque_marker_lengths_and_declared_body_length_are_exact() {
    rejects(sample(), |m| m.apps[0].size = 2, "APP length");
    for (kind, minimum) in [
        (AppMarkerKind::Icc, 17),
        (AppMarkerKind::Exif, 9),
        (AppMarkerKind::Xmp, 32),
    ] {
        rejects(
            sample(),
            |m| {
                m.apps[0].kind = kind;
                m.apps[0].size = minimum - 1;
            },
            "APP length",
        );
    }
    rejects(
        sample(),
        |m| {
            let index = m.apps[0].body.as_ref().unwrap().start;
            m.body[index + 2] ^= 1;
        },
        "APP/COM encoded length",
    );
    rejects(
        sample(),
        |m| {
            m.body.pop();
        },
        "decompressed metadata length",
    );
    let mut metadata = sample();
    metadata.body.push(0);
    let bytes = metadata
        .encode(JpegReconstructionEncodeOptions::default())
        .unwrap();
    assert!(matches!(
        JpegReconstructionMetadata::parse(bytes.as_bytes(), JpegReconstructionLimits::default()),
        Err(JpegReconstructionError::Metadata(MetadataError::Limit {
            resource: crate::metadata::MetadataResource::DecodedBoxBytes,
            ..
        }))
    ));
}

#[test]
fn empty_body_and_non_authoritative_hints_are_preserved() {
    let mut metadata = sample();
    metadata
        .markers
        .retain(|marker| !(0xe0..=0xef).contains(marker));
    metadata.apps.clear();
    metadata.body.clear();
    metadata.tail = 0..0;
    assert!(metadata.comments.is_empty() && metadata.intermarker.is_empty());
    metadata.gray_hint = true; // The explicit three-component list supersedes this hint.
    metadata.scans[0].last_pass = 10;
    metadata.has_zero_padding = true;
    metadata.padding_bits = 0;
    metadata.padding.clear();
    let bytes = metadata
        .encode(JpegReconstructionEncodeOptions::default())
        .unwrap();
    let parsed = JpegReconstructionMetadata::parse(
        bytes.as_bytes(),
        JpegReconstructionLimits {
            max_decoded_body_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(parsed.opaque_body().is_empty());
    assert!(parsed.grayscale_hint());
    assert_eq!(parsed.components().len(), 3);
    assert_eq!(parsed.scans()[0].last_pass, 10);
    assert!(parsed.has_preserved_padding());
    assert_eq!(parsed.padding_bit_count(), 0);
}

#[test]
fn header_padding_marker_bound_and_empty_scan_inventory_are_checked() {
    rejects(sample(), |m| m.markers = vec![0xd9], "no scan");
    let mut metadata = sample();
    // Ensure at least one alignment bit by varying explicit padding count; these bits are
    // independent from the entropy-padding data, and libjxl requires them to be zero.
    let mut found = false;
    for count in 0..8 {
        metadata.has_zero_padding = true;
        metadata.padding_bits = count;
        metadata.padding = vec![0];
        let encoded = metadata
            .encode(JpegReconstructionEncodeOptions::default())
            .unwrap();
        let mut input = encoded.into_bytes();
        let valid =
            JpegReconstructionMetadata::parse(&input, JpegReconstructionLimits::default()).unwrap();
        input[valid.source_header_bytes() - 1] |= 0x80;
        if let Err(JpegReconstructionError::Invalid("nonzero header padding")) =
            JpegReconstructionMetadata::parse(&input, JpegReconstructionLimits::default())
        {
            found = true;
            break;
        }
    }
    assert!(found);
    let mut markers = crate::BitWriter::new();
    markers.write_bits(0, 1).unwrap();
    for _ in 0..16385 {
        markers.write_bits(0, 6).unwrap();
    }
    let input = markers.into_bytes();
    assert!(matches!(
        JpegReconstructionMetadata::parse(
            &input,
            JpegReconstructionLimits {
                max_markers: usize::MAX,
                ..Default::default()
            }
        ),
        Err(JpegReconstructionError::Limit {
            resource: JpegReconstructionResource::Markers,
            required: 16385,
            limit: 16384
        })
    ));
}
