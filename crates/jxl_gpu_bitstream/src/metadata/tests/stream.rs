use std::sync::Arc;

use crate::{ContainerStreamEvent, ContainerStreamLimits, ContainerStreamScanner, StreamSlice};

use super::*;

fn collect(
    input: &[u8],
    sizes: &[usize],
    selection: MetadataSelection,
    limits: MetadataLimits,
) -> Result<Metadata, MetadataError> {
    let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits::default());
    let mut collector = MetadataCollector::new(selection, limits);
    let mut offset = 0;
    for &size in sizes {
        let bytes: Arc<[u8]> = Arc::from(&input[offset..offset + size]);
        for event in scanner.push_chunk(Arc::clone(&bytes)).unwrap() {
            collector.push_transport_event(&event)?;
        }
        assert_eq!(
            Arc::strong_count(&bytes),
            1,
            "collector retained the caller allocation"
        );
        offset += size;
    }
    assert_eq!(offset, input.len());
    for event in scanner.finish_input().unwrap() {
        collector.push_transport_event(&event)?;
    }
    collector.finish()
}

fn fragment(output: &mut Vec<u8>, index: u32, bytes: &[u8]) {
    let mut payload = index.to_be_bytes().to_vec();
    payload.extend_from_slice(bytes);
    crate::append_box(output, crate::JXLP, &payload).unwrap();
}

#[test]
fn every_split_byte_drip_and_box_size_encoding_preserve_selected_payloads() {
    let limits = MetadataLimits::default();
    let mut expected = Metadata::default();
    for entry in [
        item(EXIF, b"\0\0\0\0II*\0opaque", true),
        item(XMP, b"<xmp orientation='8'/>", false),
        item(JUMBF, b"\0\xff\x80\x01", true),
    ] {
        expected.push(entry, limits).unwrap();
    }
    for version in [0, 1] {
        for extended in [false, true] {
            let mut input = crate::CONTAINER_SIGNATURE_BOX.to_vec();
            let mut ftyp = crate::CONTAINER_FILE_TYPE_BOX_V0;
            ftyp[15] = version;
            input.extend_from_slice(&ftyp);
            if version == 0 {
                fragment(&mut input, 0, &[0xff]);
            } else {
                fragment(&mut input, 1 | (1 << 31), &[0x0a, 7]);
            }
            for entry in &expected.boxes[..2] {
                let wire = entry.as_container_box();
                if extended {
                    input.extend_from_slice(&1_u32.to_be_bytes());
                    input.extend_from_slice(&wire.box_type);
                    input.extend_from_slice(&(wire.payload.len() as u64 + 16).to_be_bytes());
                    input.extend_from_slice(wire.payload);
                } else {
                    crate::append_box(&mut input, wire.box_type, wire.payload).unwrap();
                }
            }
            if version == 0 {
                fragment(&mut input, 1 | (1 << 31), &[0x0a, 7]);
            } else {
                fragment(&mut input, 0, &[0xff]);
            }
            let final_box = expected.boxes[2].as_container_box();
            input.extend_from_slice(&0_u32.to_be_bytes());
            input.extend_from_slice(&final_box.box_type);
            input.extend_from_slice(final_box.payload);
            for split in 0..=input.len() {
                assert_eq!(
                    collect(
                        &input,
                        &[split, input.len() - split],
                        MetadataSelection::All,
                        limits
                    )
                    .unwrap(),
                    expected
                );
            }
            assert_eq!(
                collect(
                    &input,
                    &vec![1; input.len()],
                    MetadataSelection::All,
                    limits
                )
                .unwrap(),
                expected
            );
            let selected = collect(
                &input,
                &vec![1; input.len()],
                MetadataSelection::Types(vec![JUMBF]),
                limits,
            )
            .unwrap();
            assert_eq!(selected.boxes, expected.boxes[2..]);
            assert_eq!(
                collect(
                    &input,
                    &vec![1; input.len()],
                    MetadataSelection::None,
                    MetadataLimits {
                        max_retained_bytes: 0,
                        max_encoded_box_bytes: 0,
                        max_boxes: 0,
                        ..limits
                    }
                )
                .unwrap(),
                Metadata::default()
            );
        }
    }
}

#[test]
fn collector_limits_poison_and_release_prior_and_partial_payloads() {
    let limits = MetadataLimits::default();
    let mut metadata = Metadata::default();
    metadata.push(item(EXIF, b"first", false), limits).unwrap();
    metadata
        .push(item(XMP, b"second payload", true), limits)
        .unwrap();
    let input = metadata.write_container(&[0xff, 0x0a]).unwrap();
    for limited in [
        MetadataLimits {
            max_boxes: 1,
            ..limits
        },
        MetadataLimits {
            max_encoded_box_bytes: 5,
            ..limits
        },
        MetadataLimits {
            max_retained_bytes: metadata.retained_bytes - 1,
            ..limits
        },
    ] {
        let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits::default());
        let mut collector = MetadataCollector::new(MetadataSelection::All, limited);
        let mut failed = false;
        for byte in &input {
            for event in scanner.push_chunk(Arc::from([*byte])).unwrap() {
                if collector.push_transport_event(&event).is_err() {
                    failed = true;
                    break;
                }
            }
            if failed {
                break;
            }
        }
        assert!(failed);
        assert_eq!(collector.retained_bytes(), 0);
        assert_eq!(
            collector.push_transport_event(&ContainerStreamEvent::End {
                codestream_bytes: 2,
                is_container: true
            }),
            Err(MetadataError::CollectorFailed)
        );
        assert_eq!(collector.finish(), Err(MetadataError::CollectorFailed));
    }
    assert_eq!(
        collect(
            &input,
            &vec![1; input.len()],
            MetadataSelection::All,
            MetadataLimits {
                max_retained_bytes: metadata.retained_bytes,
                max_boxes: 2,
                ..limits
            }
        )
        .unwrap(),
        metadata
    );
}

#[test]
fn collector_requires_authoritative_end_and_checks_event_contract() {
    let limits = MetadataLimits::default();
    assert_eq!(
        MetadataCollector::new(MetadataSelection::All, limits).finish(),
        Err(MetadataError::IncompleteTransport)
    );
    let encoded = crate::write_container_with_boxes(
        &[0xff, 0x0a],
        &[ContainerBox {
            box_type: EXIF,
            payload: b"123456",
        }],
    )
    .unwrap();
    let mut scanner = ContainerStreamScanner::new(ContainerStreamLimits::default());
    let events = scanner.push_chunk(Arc::from(encoded)).unwrap();
    let header = events
        .iter()
        .find(|event| matches!(event, ContainerStreamEvent::AuxiliaryBoxStart(_)))
        .unwrap();
    for event in [
        ContainerStreamEvent::AuxiliaryBoxChunk {
            box_type: XMP,
            payload_offset: 0,
            bytes: StreamSlice::from_shared(Arc::from([1])),
        },
        ContainerStreamEvent::AuxiliaryBoxChunk {
            box_type: EXIF,
            payload_offset: 1,
            bytes: StreamSlice::from_shared(Arc::from([1])),
        },
        ContainerStreamEvent::AuxiliaryBoxChunk {
            box_type: EXIF,
            payload_offset: 0,
            bytes: StreamSlice::from_shared(Arc::from([1; 7])),
        },
        ContainerStreamEvent::AuxiliaryBoxEnd { box_type: EXIF },
        header.clone(),
    ] {
        let mut collector = MetadataCollector::new(MetadataSelection::All, limits);
        collector.push_transport_event(header).unwrap();
        assert_eq!(
            collector.push_transport_event(&event),
            Err(MetadataError::EventContract)
        );
        assert_eq!(collector.retained_bytes(), 0);
    }
    let mut collector = MetadataCollector::new(MetadataSelection::All, limits);
    for event in &events {
        collector.push_transport_event(event).unwrap();
    }
    let end = scanner.finish_input().unwrap();
    for event in &end {
        collector.push_transport_event(event).unwrap();
    }
    assert_eq!(
        collector.push_transport_event(&end[0]),
        Err(MetadataError::CollectorFinished)
    );
    assert_eq!(collector.finish().unwrap().boxes.len(), 1);
}
