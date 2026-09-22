use super::*;

fn golden() -> Vec<u8> {
    // Three entries: offsets 13/300/1000, durations 20/30/40 ms, spans 2/1/4 frames.
    vec![
        3, 0, 0, 0, 1, 0, 0, 3, 232, 13, 20, 2, 159, 2, 30, 1, 188, 5, 40, 4,
    ]
}

#[test]
fn independent_wire_vector_and_nonminimal_integers_roundtrip() {
    let index = FrameIndex::parse(&golden(), Default::default()).unwrap();
    assert_eq!(index.tick_numerator(), 1);
    assert_eq!(index.tick_denominator().get(), 1000);
    assert_eq!(
        index.entries(),
        &[
            FrameIndexEntry {
                codestream_offset: 13,
                duration_ticks: 20,
                frames: 2
            },
            FrameIndexEntry {
                codestream_offset: 300,
                duration_ticks: 30,
                frames: 1
            },
            FrameIndexEntry {
                codestream_offset: 1000,
                duration_ticks: 40,
                frames: 4
            },
        ]
    );
    assert_eq!(index.encode(Default::default()).unwrap(), golden());
    let mut padded = vec![131, 0];
    padded.extend_from_slice(&golden()[1..]);
    assert_eq!(
        FrameIndex::parse(&padded, Default::default()).unwrap(),
        index
    );
}

#[test]
fn truncation_overflow_duplicate_offsets_and_empty_intervals_are_errors() {
    let bytes = golden();
    for end in 0..bytes.len() {
        assert!(
            FrameIndex::parse(&bytes[..end], Default::default()).is_err(),
            "prefix {end}"
        );
    }
    let mut denominator = bytes.clone();
    denominator[5..9].fill(0);
    assert_eq!(
        FrameIndex::parse(&denominator, Default::default()),
        Err(FrameIndexError::ZeroDenominator)
    );
    let mut empty = bytes.clone();
    empty[11] = 0;
    assert_eq!(
        FrameIndex::parse(&empty, Default::default()),
        Err(FrameIndexError::EmptyFrameInterval)
    );
    let mut duplicate = bytes.clone();
    duplicate.splice(12..14, [0]);
    assert_eq!(
        FrameIndex::parse(&duplicate, Default::default()),
        Err(FrameIndexError::OffsetOrder)
    );
    let mut tail = bytes;
    tail.push(0);
    assert_eq!(
        FrameIndex::parse(&tail, Default::default()),
        Err(FrameIndexError::TrailingBytes)
    );
    assert_eq!(
        FrameIndex::parse(&[0], Default::default()),
        Err(FrameIndexError::Empty)
    );
    for last in [1, 127, 128, 255] {
        let mut over = vec![255; 9];
        over.push(last);
        assert_eq!(
            FrameIndex::parse(&over, Default::default()),
            Err(FrameIndexError::InvalidVarint)
        );
    }
}

#[test]
fn limits_precede_entry_allocation_and_are_rechecked_for_emission() {
    for (limits, expected) in [
        (
            FrameIndexLimits {
                max_entries: 2,
                ..Default::default()
            },
            FrameIndexError::EntryLimit,
        ),
        (
            FrameIndexLimits {
                max_frames: 6,
                ..Default::default()
            },
            FrameIndexError::FrameLimit,
        ),
        (
            FrameIndexLimits {
                max_payload_bytes: 19,
                ..Default::default()
            },
            FrameIndexError::PayloadLimit,
        ),
    ] {
        assert_eq!(FrameIndex::parse(&golden(), limits), Err(expected.clone()));
        let index = FrameIndex::parse(&golden(), Default::default()).unwrap();
        assert_eq!(index.encode(limits), Err(expected));
    }
    assert_eq!(
        FrameIndex::parse(
            &[255; 9],
            FrameIndexLimits {
                max_entries: 0,
                ..Default::default()
            }
        ),
        Err(FrameIndexError::Truncated)
    );
    let mut huge = vec![255; 8];
    huge.push(127);
    assert_eq!(
        FrameIndex::parse(&huge, Default::default()),
        Err(FrameIndexError::EntryLimit)
    );
    let exact = FrameIndexLimits {
        max_entries: 3,
        max_frames: 7,
        max_payload_bytes: 20,
    };
    FrameIndex::parse(&golden(), exact)
        .unwrap()
        .encode(exact)
        .unwrap();
}

#[test]
fn maximum_varints_and_cumulative_overflow_stay_bounded() {
    let limits = FrameIndexLimits {
        max_frames: u64::MAX,
        ..Default::default()
    };
    let record = FrameIndexEntry {
        codestream_offset: MAX_VARINT,
        duration_ticks: MAX_VARINT,
        frames: MAX_VARINT,
    };
    let index = FrameIndex::new(0, NonZeroU32::new(1).unwrap(), vec![record], limits).unwrap();
    assert_eq!(
        FrameIndex::parse(&index.encode(limits).unwrap(), limits).unwrap(),
        index
    );
    let records: Vec<_> = (0..3)
        .map(|i| FrameIndexEntry {
            codestream_offset: i,
            ..record
        })
        .collect();
    assert_eq!(
        FrameIndex::new(1, NonZeroU32::new(1).unwrap(), records, limits),
        Err(FrameIndexError::Overflow)
    );
}

#[test]
fn containers_have_zero_or_one_index_and_offsets_ignore_fragment_boxes() {
    use crate::{ContainerBox, FragmentedContainerWriter, parse, write_container_with_boxes};
    let payload = golden();
    let index_box = ContainerBox {
        box_type: FRAME_INDEX_BOX_TYPE,
        payload: &payload,
    };
    let raw = [255, 10, 1, 2, 3];
    assert!(
        FrameIndex::from_container(
            &parse(&raw, Default::default()).unwrap(),
            Default::default()
        )
        .unwrap()
        .is_none()
    );
    let duplicate = write_container_with_boxes(&raw, &[index_box, index_box]).unwrap();
    assert_eq!(
        FrameIndex::from_container(
            &parse(&duplicate, Default::default()).unwrap(),
            Default::default()
        ),
        Err(FrameIndexError::DuplicateBox)
    );
    let mut writer = FragmentedContainerWriter::new();
    writer.push_box(index_box).unwrap();
    writer.push_fragment(&raw[..2], false).unwrap();
    writer.push_fragment(&raw[2..], true).unwrap();
    let data = writer.finish().unwrap();
    let parsed = parse(&data, Default::default()).unwrap();
    assert_eq!(
        FrameIndex::from_container(&parsed, Default::default())
            .unwrap()
            .unwrap()
            .encode(Default::default())
            .unwrap(),
        payload
    );
    let compressed = write_container_with_boxes(
        &raw,
        &[ContainerBox {
            box_type: *b"brob",
            payload: b"jxli",
        }],
    )
    .unwrap();
    assert_eq!(
        FrameIndex::from_container(
            &parse(&compressed, Default::default()).unwrap(),
            Default::default()
        ),
        Err(FrameIndexError::CompressedBox)
    );
}
