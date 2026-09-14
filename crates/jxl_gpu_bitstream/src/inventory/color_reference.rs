use super::*;
use crate::BitWriter;

struct Frame {
    kind: FrameType,
    last: bool,
    duration: u32,
    slot: u32,
    before: bool,
    add: bool,
    retained: bool,
}

// This writes frame syntax, including cases where save_before_ct is implicit.
// A deliberately absent TOC makes the rejection boundary observable.
fn header(frame: &Frame, xyb: bool, modular: bool) -> (Vec<u8>, u64, u64) {
    let mut w = BitWriter::new();
    let normal = matches!(frame.kind, FrameType::Regular | FrameType::SkipProgressive);
    let mut bits = |value, count| w.write_bits(value, count).unwrap();
    bits(0, 1);
    bits(frame.kind as u64, 2);
    bits(u64::from(modular), 1);
    bits(0, 2); // flags
    if !xyb {
        bits(0, 1); // no YCbCr
    }
    bits(0, 2); // upsampling
    if modular {
        bits(1, 2);
    } else if xyb {
        bits(3, 3);
        bits(2, 3);
    }
    if frame.kind != FrameType::ReferenceOnly {
        bits(0, 2); // one pass
    }
    if frame.kind == FrameType::LowFrequency {
        bits(0, 2); // LF level 1
    } else {
        bits(0, 1); // no crop
    }
    if normal {
        bits(u64::from(frame.add), 2);
        if frame.add {
            bits(0, 2); // background slot
        }
        bits(u64::from(frame.duration), 2);
        bits(u64::from(frame.last), 1);
    }
    if frame.kind != FrameType::LowFrequency && !frame.last {
        bits(u64::from(frame.slot), 2);
    }
    if frame.kind == FrameType::ReferenceOnly || (normal && frame.retained && !frame.add) {
        bits(u64::from(frame.before), 1);
    }
    let reference_end = w.bit_len() as u64;
    w.write_bits(0, 2).unwrap(); // name
    w.write_bits(1, 1).unwrap(); // default filters
    w.write_bits(0, 2).unwrap(); // extensions
    let end = w.bit_len() as u64;
    (w.into_bytes(), reference_end, end)
}

#[test]
fn reference_colour_validation_obeys_retention_and_stops_before_unrelated_header_fields() {
    let mut checked = 0;
    for xyb in [false, true] {
        for has_icc in [false, true] {
            for modular in [false, true] {
                for kind in [
                    FrameType::Regular,
                    FrameType::SkipProgressive,
                    FrameType::ReferenceOnly,
                ] {
                    for slot in 0..4 {
                        for before in [false, true] {
                            let frame = Frame {
                                kind,
                                last: false,
                                duration: 0,
                                slot,
                                before,
                                add: false,
                                retained: true,
                            };
                            let (bytes, decision, end) = header(&frame, xyb, modular);
                            let context = ImageContext {
                                width: 17,
                                height: 9,
                                preview_size: None,
                                xyb_encoded: xyb,
                                has_icc,
                                num_extra_channels: 0,
                                extra_channel_shifts: vec![],
                                have_animation: true,
                                have_timecodes: false,
                            };
                            let mut reader = BitReader::new(&bytes);
                            let result = parse_frame_header(
                                &mut reader,
                                context.frame_context(false).unwrap(),
                                false,
                                Default::default(),
                            );
                            if xyb && has_icc && !before {
                                assert!(
                                    matches!(result, Err(InventoryError::XybIccReference { slot: actual }) if actual == slot)
                                );
                                assert_eq!(reader.bit_offset(), decision);
                            } else {
                                let actual = result.unwrap();
                                assert_eq!(actual.frame_type, kind);
                                assert_eq!(actual.save_as_reference, slot);
                                assert_eq!(actual.save_before_color_transform, before);
                                assert_eq!(reader.bit_offset(), end);
                            }
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checked, 192);
}

#[test]
fn unused_reference_flags_and_lf_storage_do_not_forbid_icc_xyb_output() {
    let context = ImageContext {
        width: 17,
        height: 9,
        preview_size: Some((9, 7)),
        xyb_encoded: true,
        has_icc: true,
        num_extra_channels: 0,
        extra_channel_shifts: vec![],
        have_animation: true,
        have_timecodes: false,
    };
    for modular in [false, true] {
        for frame in [
            Frame {
                kind: FrameType::Regular,
                last: true,
                duration: 0,
                slot: 0,
                before: false,
                add: false,
                retained: false,
            },
            Frame {
                kind: FrameType::Regular,
                last: false,
                duration: 1,
                slot: 0,
                before: false,
                add: false,
                retained: false,
            },
            Frame {
                kind: FrameType::SkipProgressive,
                last: false,
                duration: 1,
                slot: 0,
                before: false,
                add: false,
                retained: false,
            },
            Frame {
                kind: FrameType::LowFrequency,
                last: false,
                duration: 0,
                slot: 0,
                before: true,
                add: false,
                retained: false,
            },
            Frame {
                kind: FrameType::Regular,
                last: false,
                duration: 1,
                slot: 2,
                before: false,
                add: true,
                retained: true,
            },
        ] {
            let (bytes, decision, end) = header(&frame, true, modular);
            let mut reader = BitReader::new(&bytes);
            let result = parse_frame_header(
                &mut reader,
                context.frame_context(false).unwrap(),
                false,
                Default::default(),
            );
            if frame.retained {
                assert!(matches!(
                    result,
                    Err(InventoryError::XybIccReference { slot: 2 })
                ));
                assert_eq!(reader.bit_offset(), decision);
            } else {
                assert_eq!(result.unwrap().save_before_color_transform, frame.before);
                assert_eq!(reader.bit_offset(), end);
            }
        }
    }
    // The all-default final-frame path, including embedded previews, has no retained reference.
    for preview in [false, true] {
        let frame = parse_frame_header(
            &mut BitReader::new(&[1]),
            context.frame_context(preview).unwrap(),
            preview,
            Default::default(),
        )
        .unwrap();
        assert!(frame.is_last);
        assert!(!frame.save_before_color_transform);
        assert_eq!(
            (frame.width, frame.height),
            if preview { (9, 7) } else { (17, 9) }
        );
    }
}
