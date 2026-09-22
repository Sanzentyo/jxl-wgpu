use std::sync::Arc;

use super::*;
use crate::{ContainerStreamScanner, StreamSlice};

const RAW: &[u8] = &[255, 10, 1, 2, 3];
const PAYLOAD: &[u8] = &[
    3, 0, 0, 0, 1, 0, 0, 3, 232, 13, 20, 2, 159, 2, 30, 1, 188, 5, 40, 4,
];

fn append_box(data: &mut Vec<u8>, kind: [u8; 4], payload: &[u8], size: u32) {
    let length = match size {
        0 | 1 => size,
        _ => payload.len() as u32 + 8,
    };
    data.extend_from_slice(&length.to_be_bytes());
    data.extend_from_slice(&kind);
    if size == 1 {
        data.extend_from_slice(&(payload.len() as u64 + 16).to_be_bytes());
    }
    data.extend_from_slice(payload);
}

fn boxed(payload: &[u8], size: u32) -> Vec<u8> {
    let mut data = crate::write_container(RAW).unwrap();
    append_box(&mut data, FRAME_INDEX_BOX_TYPE, payload, size);
    data
}

fn events(data: &[u8]) -> Vec<ContainerStreamEvent> {
    let mut scanner = ContainerStreamScanner::new(Default::default());
    let mut events = scanner.push_chunk(Arc::from(data)).unwrap();
    events.extend(scanner.finish_input().unwrap());
    events
}

fn cleared(collector: &FrameIndexCollector) {
    assert_eq!(
        collector.stats(),
        FrameIndexCollectorStats {
            failed: true,
            ..Default::default()
        }
    );
}

#[test]
fn every_split_and_byte_drip_preserve_index_after_authoritative_end_only() {
    let expected = FrameIndex::parse(PAYLOAD, Default::default()).unwrap();
    let mut variants: Vec<_> = [0, 1, 2].map(|size| boxed(PAYLOAD, size)).into();
    for reversed in [false, true] {
        let mut data = crate::CONTAINER_SIGNATURE_BOX.to_vec();
        let mut file_type = crate::CONTAINER_FILE_TYPE_BOX_V0;
        file_type[15] = u8::from(reversed);
        data.extend_from_slice(&file_type);
        let chunks = [(0u32, &RAW[..2]), (0x8000_0001, &RAW[2..])];
        for i in 0..2 {
            let (index, chunk) = chunks[if reversed { 1 - i } else { i }];
            let payload = [index.to_be_bytes().as_slice(), chunk].concat();
            append_box(&mut data, *b"jxlp", &payload, 2);
            if i == 0 {
                append_box(&mut data, FRAME_INDEX_BOX_TYPE, PAYLOAD, 2);
            }
        }
        variants.push(data);
    }
    for data in variants {
        for split in 0..=data.len() + 1 {
            let chunks: Vec<&[u8]> = if split > data.len() {
                data.chunks(1).collect()
            } else {
                vec![&data[..split], &data[split..]]
            };
            let mut scanner = ContainerStreamScanner::new(Default::default());
            let mut collector = FrameIndexCollector::new(Default::default());
            for chunk in chunks {
                let storage: Arc<[u8]> = Arc::from(chunk);
                let weak = Arc::downgrade(&storage);
                for event in scanner.push_chunk(storage).unwrap() {
                    collector.push_transport_event(&event).unwrap();
                }
                assert!(
                    weak.upgrade().is_none(),
                    "collector retained caller allocation"
                );
                assert!(!collector.stats().authoritative_end);
                assert!(collector.stats().retained_payload_bytes <= PAYLOAD.len() as u64);
            }
            for event in scanner.finish_input().unwrap() {
                collector.push_transport_event(&event).unwrap();
            }
            assert_eq!(
                collector.stats(),
                FrameIndexCollectorStats {
                    retained_entries: 3,
                    authoritative_end: true,
                    ..Default::default()
                }
            );
            assert_eq!(collector.finish().unwrap(), Some(expected.clone()));
        }
    }
    let mut collector = FrameIndexCollector::new(Default::default());
    for event in events(&boxed(PAYLOAD, 2))
        .iter()
        .filter(|event| !matches!(event, ContainerStreamEvent::End { .. }))
    {
        collector.push_transport_event(event).unwrap();
    }
    assert_eq!(
        collector.finish(),
        Err(FrameIndexError::IncompleteTransport)
    );
    let mut collector = FrameIndexCollector::new(Default::default());
    for event in events(RAW) {
        collector.push_transport_event(&event).unwrap();
    }
    assert_eq!(collector.finish().unwrap(), None);
}

#[test]
fn payload_entry_and_frame_limits_apply_before_handoff_and_release_on_failure() {
    for size in [0, 1, 2] {
        for (limits, expected) in [
            (
                FrameIndexLimits {
                    max_payload_bytes: 19,
                    ..Default::default()
                },
                FrameIndexError::PayloadLimit,
            ),
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
        ] {
            let mut collector = FrameIndexCollector::new(limits);
            let error = events(&boxed(PAYLOAD, size))
                .iter()
                .try_for_each(|event| collector.push_transport_event(event))
                .unwrap_err();
            assert_eq!(error, expected);
            cleared(&collector);
            assert_eq!(collector.finish(), Err(FrameIndexError::CollectorFailed));
        }
        let mut collector = FrameIndexCollector::new(FrameIndexLimits {
            max_payload_bytes: 20,
            max_entries: 3,
            max_frames: 7,
        });
        for event in events(&boxed(PAYLOAD, size)) {
            collector.push_transport_event(&event).unwrap();
        }
        assert_eq!(collector.finish().unwrap().unwrap().entries().len(), 3);
    }
    for end in 0..PAYLOAD.len() {
        let mut collector = FrameIndexCollector::new(Default::default());
        assert!(
            events(&boxed(&PAYLOAD[..end], 2))
                .iter()
                .try_for_each(|event| collector.push_transport_event(event))
                .is_err()
        );
        cleared(&collector);
    }
    // Known-length boxes reject at Start, before a byte of an oversized index is allocated.
    let mut collector = FrameIndexCollector::new(FrameIndexLimits {
        max_payload_bytes: 19,
        ..Default::default()
    });
    let start = events(&boxed(PAYLOAD, 2))
        .into_iter()
        .find(|event| matches!(event, ContainerStreamEvent::AuxiliaryBoxStart(_)))
        .unwrap();
    assert_eq!(
        collector.push_transport_event(&start),
        Err(FrameIndexError::PayloadLimit)
    );
    cleared(&collector);
}

#[test]
fn duplicate_compressed_and_misordered_events_poison_all_retained_metadata() {
    let mut duplicate = boxed(PAYLOAD, 2);
    append_box(&mut duplicate, FRAME_INDEX_BOX_TYPE, PAYLOAD, 2);
    let mut compressed = boxed(PAYLOAD, 2);
    append_box(&mut compressed, *b"brob", b"jxliANY compressed bytes", 2);
    for (data, expected) in [
        (duplicate, FrameIndexError::DuplicateBox),
        (compressed, FrameIndexError::CompressedBox),
    ] {
        let mut scanner = ContainerStreamScanner::new(Default::default());
        let mut collector = FrameIndexCollector::new(Default::default());
        let mut failure = None;
        for byte in data {
            for event in scanner.push_chunk(Arc::from([byte])).unwrap() {
                if let Err(error) = collector.push_transport_event(&event) {
                    failure = Some(error);
                    break;
                }
            }
            if failure.is_some() {
                break;
            }
        }
        assert_eq!(failure, Some(expected));
        cleared(&collector);
    }
    let all = events(&boxed(PAYLOAD, 2));
    let start = all
        .iter()
        .find(|event| matches!(event, ContainerStreamEvent::AuxiliaryBoxStart(_)))
        .unwrap();
    let end = ContainerStreamEvent::End {
        codestream_bytes: RAW.len() as u64,
        is_container: true,
    };
    let chunk = |kind, offset| ContainerStreamEvent::AuxiliaryBoxChunk {
        box_type: kind,
        payload_offset: offset,
        bytes: StreamSlice::from_shared(Arc::from(PAYLOAD)),
    };
    for invalid in [
        start.clone(),
        chunk(*b"jxli", 1),
        chunk(*b"Exif", 0),
        ContainerStreamEvent::AuxiliaryBoxEnd { box_type: *b"jxli" },
        ContainerStreamEvent::CodestreamChunk {
            logical_offset: 0,
            bytes: StreamSlice::from_shared(Arc::from(RAW)),
        },
        end.clone(),
    ] {
        let mut collector = FrameIndexCollector::new(Default::default());
        collector.push_transport_event(start).unwrap();
        assert!(collector.push_transport_event(&invalid).is_err());
        cleared(&collector);
        assert_eq!(
            collector.push_transport_event(&end),
            Err(FrameIndexError::CollectorFailed)
        );
    }
    let mut collector = FrameIndexCollector::new(Default::default());
    for event in all {
        collector.push_transport_event(&event).unwrap();
    }
    assert_eq!(
        collector.push_transport_event(&end),
        Err(FrameIndexError::CollectorFinished)
    );
    cleared(&collector);
}

#[test]
fn unrelated_boxes_retain_no_payload_even_with_tiny_index_budget() {
    let mut data = crate::write_container(RAW).unwrap();
    append_box(&mut data, *b"Exif", &[42; 4096], 2);
    append_box(
        &mut data,
        *b"brob",
        &[b"xml ".as_slice(), &[0; 4096]].concat(),
        2,
    );
    let mut collector = FrameIndexCollector::new(FrameIndexLimits {
        max_payload_bytes: 1,
        ..Default::default()
    });
    for event in events(&data) {
        collector.push_transport_event(&event).unwrap();
        assert_eq!(collector.stats().retained_payload_bytes, 0);
        assert_eq!(collector.stats().retained_entries, 0);
    }
    assert!(collector.finish().unwrap().is_none());
}
