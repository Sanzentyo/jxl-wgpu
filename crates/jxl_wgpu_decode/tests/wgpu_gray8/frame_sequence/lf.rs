use super::independent::{payloads, reassemble, retired};
use super::*;
use jxl_gpu_bitstream::{FrameEncoding, FrameType};
use jxl_wgpu_decode::Error;

#[test]
fn overwritten_unused_lf_roots_are_validated_in_both_sequence_paths() {
    let Some(backend) = backend() else {
        return;
    };
    for case in cases()
        .into_iter()
        .chain(composition_cases())
        .filter(|case| case.name == "vardct_dc")
    {
        let original = encoded(&case);
        let parsed = parse(&original, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let data = parsed.codestream();
        let first = &inventory.frames[0];
        assert_eq!(first.frame_type, FrameType::LowFrequency);
        let prefix = &data[..first.header_bits.offset as usize / 8];
        let duplicate = |truncate: bool| {
            let mut result = prefix.to_vec();
            let mut packets = payloads(data, first);
            if truncate {
                let packet = packets
                    .iter_mut()
                    .max_by_key(|packet| packet.len())
                    .unwrap();
                packet.truncate(packet.len() - 8);
            }
            result.extend(reassemble(data, first, packets));
            result.extend_from_slice(&data[prefix.len()..]);
            result
        };
        let valid = duplicate(false);
        let invalid = duplicate(true);
        let checked = parse(&valid, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let plan = FrameExecutionPlan::negotiate(&checked).unwrap();
        assert!(
            plan.nodes
                .iter()
                .all(|node| node.lf_source_frame != Some(0))
        );
        assert_eq!(checked.frames.len(), inventory.frames.len() + 1);
        assert_eq!(rust_frames(&case, &valid), rust_frames(&case, &original));
        assert_eq!(djxl_frames(&case, &valid), djxl_frames(&case, &original));
        let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
        let mut session = decoder
            .open(&invalid, request(&case).with_progressive_output(true))
            .unwrap();
        let result = session.next_update();
        assert!(
            matches!(result, Err(Error::ModularEntropyRejected { .. })),
            "{result:?}"
        );
    }
}

#[test]
fn lf_versions_are_reused_across_presentations_and_released_after_the_last_consumer() {
    let case = cases()
        .into_iter()
        .find(|case| case.name == "vardct_dc")
        .unwrap();
    let original = encoded(&case);
    let parsed = parse(&original, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let data = parsed.codestream();
    let first_color = inventory
        .frames
        .iter()
        .position(|frame| frame.frame_type != FrameType::LowFrequency)
        .unwrap();
    let mut reused = data[..inventory.frames[0].header_bits.offset as usize / 8].to_vec();
    for (index, frame) in inventory.frames.iter().enumerate() {
        if index < first_color || frame.frame_type != FrameType::LowFrequency {
            reused.extend(reassemble(data, frame, payloads(data, frame)));
        }
    }
    let inventory = parse(&reused, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
    assert!(plan.presentations.len() > 1);
    let source = first_color - 1;
    assert_eq!(inventory.frames[source].encoding, FrameEncoding::VarDct);
    assert_eq!(
        plan.nodes[source].lf_last_use,
        Some((plan.nodes.len() - 1) as u32)
    );
    assert!(
        plan.presentations[1..]
            .iter()
            .all(|frame| !frame.metadata.is_keyframe)
    );
    let expected = rust_frames(&case, &reused);
    let djxl = djxl_frames(&case, &reused);
    let Some(backend) = backend() else {
        return;
    };
    for (bounded, progressive) in [(false, false), (true, false), (false, true), (true, true)] {
        let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        let decoder = GpuDecoder::new(if bounded {
            engine.with_stream_window_limit(NonZeroU64::new(4096).unwrap())
        } else {
            engine
        });
        let mut session = if bounded {
            incremental(
                &decoder,
                &reused,
                request(&case).with_progressive_output(progressive),
            )
        } else {
            decoder
                .open(&reused, request(&case).with_progressive_output(progressive))
                .unwrap()
        };
        let progress = session.prefetch(NonZeroUsize::new(3).unwrap()).unwrap();
        assert_eq!(progress.queued, 1);
        assert_eq!(
            progress.backpressure,
            Some(PrefetchBackpressure::FrameDependency { index: 0 })
        );
        let (width, height) = inventory.frames[source].color_sample_extent().unwrap();
        let retained_bytes =
            u64::from(width.div_ceil(8) * 8) * u64::from(height.div_ceil(8) * 8) * 4 * 3;
        for (index, (_, oracle)) in expected.iter().enumerate() {
            let mut updates = 0;
            let mut coefficients = 0;
            let physical = &inventory.frames[plan.presentations[index].physical_frames.end - 1];
            let frame = loop {
                let frame = if progressive && bounded {
                    pollster::block_on(session.next_update_async())
                        .unwrap()
                        .unwrap()
                } else if progressive {
                    session.next_update().unwrap().unwrap()
                } else if bounded {
                    pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap()
                } else {
                    session.next_frame().unwrap().unwrap()
                };
                assert_eq!(frame.metadata, plan.presentations[index].metadata);
                if frame.is_complete() {
                    break frame;
                }
                match frame.progression().unwrap() {
                    jxl_wgpu_decode::FrameProgression::LowFrequency { .. } => updates += 1,
                    jxl_wgpu_decode::FrameProgression::Modular { .. } => {
                        panic!("the terminal producer is VarDCT")
                    }
                    jxl_wgpu_decode::FrameProgression::Coefficients {
                        physical_frame_index,
                        completed_passes,
                        total_passes,
                        ..
                    } => {
                        assert_eq!(physical_frame_index, physical.frame_index);
                        assert_eq!(u32::from(completed_passes), coefficients);
                        assert_eq!(u32::from(total_passes), physical.num_passes);
                        coefficients += 1;
                    }
                }
            };
            assert_eq!(
                updates,
                if progressive && index == 0 {
                    first_color
                } else {
                    0
                }
            );
            assert_eq!(
                coefficients,
                if progressive { physical.num_passes } else { 0 }
            );
            assert_eq!(frame.metadata, plan.presentations[index].metadata);
            let pixels = samples(
                &read_output(&backend, &frame.output().outputs[0]),
                case.bits,
            );
            assert_eq!(pixels.len(), oracle.len());
            assert!(pixels.iter().zip(oracle).all(|(a, b)| a.abs_diff(*b) <= 1));
            if let Some(djxl) = &djxl {
                assert_eq!(pixels.len(), djxl[index].len());
                assert!(
                    pixels
                        .iter()
                        .zip(&djxl[index])
                        .all(|(a, b)| a.abs_diff(*b) <= 1)
                );
            }
            drop(frame);
            retired(&backend);
            assert_eq!(
                decoder.engine().in_flight_memory_stats().reserved_bytes,
                if index + 1 == expected.len() {
                    0
                } else {
                    retained_bytes
                }
            );
            assert!(session.submission_session().submissions_per_frame() > 0);
        }
        assert!(session.next_frame().unwrap().is_none());
        drop(session);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
    }
}

#[test]
fn frame_plan_validates_lf_slot_versions_levels_modes_flags_and_extents() {
    let case = cases()
        .into_iter()
        .find(|case| case.name == "vardct_dc")
        .unwrap();
    let data = encoded(&case);
    let inventory = parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let original = FrameExecutionPlan::negotiate(&inventory).unwrap();
    for node in &original.nodes {
        let last = original
            .nodes
            .iter()
            .rev()
            .find(|consumer| consumer.lf_source_frame == Some(node.frame_index))
            .map(|consumer| consumer.frame_index);
        assert_eq!(node.lf_last_use, last);
    }
    let first_color = inventory
        .frames
        .iter()
        .position(|frame| frame.frame_type != FrameType::LowFrequency)
        .unwrap();
    for change in 0..8 {
        let mut invalid = inventory.clone();
        match change {
            0 => invalid.frames[0].lf_level = 0,
            1 => invalid.frames[0].lf_level = 5,
            2 => invalid.frames[first_color].lf_source_frame = Some(0),
            3 => invalid.frames[first_color].encoding = FrameEncoding::Modular,
            4 => invalid.frames[first_color].width *= 2,
            5 => invalid.frames[first_color].flags &= !32,
            6 => invalid.frames[first_color].lf_level = 1,
            7 => invalid.frames[first_color].height *= 2,
            _ => unreachable!(),
        }
        assert!(
            matches!(
                FrameExecutionPlan::negotiate(&invalid),
                Err(FramePlanError::InvalidFrame { .. })
            ),
            "mutation {change}"
        );
    }
}

#[test]
fn unused_lf_prefix_keeps_independent_final_coefficient_updates() {
    let color = Case {
        name: "unused_lf_progressive",
        hex: include_str!("../../../test-data/lf_vardct_root_16x2.jxl.hex"),
        format: LosslessModularFormat::Rgb,
        bits: 8,
        vardct: true,
    };
    let root = Case {
        name: "unused_lf",
        hex: include_str!("../../../test-data/lf_modular_root_2x1.jxl.hex"),
        format: LosslessModularFormat::Rgb,
        bits: 8,
        vardct: false,
    };
    let original = encoded(&color);
    let parsed = parse(&original, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let data = parsed.codestream();
    let start = inventory.frames[0].header_bits.offset as usize / 8;
    let mut modified = data[..start].to_vec();
    modified.extend(lf_root(&root, false, 0, 1, false, 1));
    modified.extend_from_slice(&data[start..]);
    let plan = FrameExecutionPlan::negotiate(
        &parse(&modified, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(plan.nodes.len(), 2);
    assert!(
        plan.nodes
            .iter()
            .all(|node| node.lf_source_frame.is_none() && node.lf_last_use.is_none())
    );
    assert_eq!(
        rust_frames(&color, &modified),
        rust_frames(&color, &original)
    );
    assert_eq!(
        djxl_frames(&color, &modified),
        djxl_frames(&color, &original)
    );
    let Some(backend) = backend() else {
        return;
    };
    for bounded in [false, true] {
        let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        let decoder = GpuDecoder::new(if bounded {
            engine.with_stream_window_limit(NonZeroU64::new(40).unwrap())
        } else {
            engine
        });
        let request = GpuOutputRequest::color(jxl_wgpu_decode::vardct_rgb8_format())
            .unwrap()
            .with_progressive_output(true);
        let mut session = if bounded {
            incremental(&decoder, &modified, request)
        } else {
            decoder.open(&modified, request).unwrap()
        };
        let dc = session.next_update().unwrap().unwrap();
        assert_eq!(
            dc.progression(),
            Some(jxl_wgpu_decode::FrameProgression::Coefficients {
                physical_frame_index: 1,
                completed_passes: 0,
                total_passes: 1,
                intended_downsampling: 8,
            })
        );
        let frame = pollster::block_on(session.next_update_async())
            .unwrap()
            .unwrap();
        assert!(frame.is_complete());
        assert_eq!(frame.metadata, dc.metadata);
        assert!(session.next_update().unwrap().is_none());
        drop(frame);
        drop(dc);
        drop(session);
        retired(&backend);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn cancelling_after_lf_validation_releases_tracked_planes_and_incremental_input() {
    use std::task::{Context, Poll, Waker};
    let Some(backend) = backend() else {
        return;
    };
    let case = cases()
        .into_iter()
        .find(|case| case.name == "vardct_dc")
        .unwrap();
    let data = encoded(&case);
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    for completed in [0, 1] {
        let mut session = incremental(&decoder, &data, request(&case));
        session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        let root_submissions = session.submission_session().submissions_per_frame();
        assert!(root_submissions > 0);
        let mut context = Context::from_waker(Waker::noop());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while completed != 0
            && session.submission_session().submissions_per_frame() <= root_submissions
        {
            assert!(std::time::Instant::now() < deadline);
            assert!(matches!(
                session.poll_next_frame(&mut context),
                Poll::Pending
            ));
            std::thread::yield_now();
        }
        assert!(decoder.engine().in_flight_memory_stats().reserved_bytes > 0);
        drop(session);
        retired(&backend);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
    }
}

/// Reframe an independent cjxl VarDCT packet at LF level 2. This preserves all entropy bytes;
/// only the normative physical header changes. The small image has restoration disabled.
fn vardct_root(gaborish: bool, upsampling: u32) -> Vec<u8> {
    let case = Case {
        name: "lf_root",
        hex: if upsampling == 1 {
            include_str!("../../../test-data/lf_vardct_root_16x2.jxl.hex")
        } else {
            include_str!("../../../test-data/lf_vardct_root_8x1.jxl.hex")
        },
        format: LosslessModularFormat::Rgb,
        bits: 8,
        vardct: true,
    };
    lf_root(&case, gaborish, 0, upsampling, false, 2)
}

fn lf_root(
    case: &Case,
    gaborish: bool,
    epf: u32,
    upsampling: u32,
    custom: bool,
    level: u32,
) -> Vec<u8> {
    use jxl_gpu_bitstream::{
        BitRange, BitWriter, EdgePreservingFilterInventory, GaborishInventory,
        RestorationFilterInventory,
    };
    let data = encoded(case);
    let parsed = parse(&data, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let mut frame = inventory.frames[0].clone();
    assert_eq!(
        frame.encoding,
        if case.vardct {
            FrameEncoding::VarDct
        } else {
            FrameEncoding::Modular
        }
    );
    assert!(inventory.image_header.xyb_encoded);
    assert!((1..=4).contains(&level));
    assert_eq!(frame.upsampling, 1);
    assert_eq!(frame.num_passes, 1);
    assert!(matches!(frame.flags, 0 | 128));
    assert_eq!(
        frame.restoration_filter,
        RestorationFilterInventory::Custom {
            gaborish: GaborishInventory::Disabled,
            epf: EdgePreservingFilterInventory::Disabled
        }
    );
    let mut header = BitWriter::new();
    header.write_bits(0, 1).unwrap(); // non-default physical header
    header.write_bits(1, 2).unwrap(); // LowFrequency
    header.write_bits(u64::from(!case.vardct), 1).unwrap(); // encoding
    if frame.flags == 0 {
        header.write_bits(0, 2).unwrap();
    } else {
        header.write_bits(2, 2).unwrap();
        header.write_bits(frame.flags - 17, 8).unwrap();
    }
    header
        .write_bits(u64::from(upsampling.trailing_zeros()), 2)
        .unwrap(); // frame upsampling
    if case.vardct {
        header.write_bits(u64::from(frame.x_qm_scale), 3).unwrap();
        header.write_bits(u64::from(frame.b_qm_scale), 3).unwrap();
    } else {
        header
            .write_bits(u64::from(frame.group_size_shift), 2)
            .unwrap();
    }
    header.write_bits(0, 2).unwrap(); // one pass
    header.write_bits(u64::from(level - 1), 2).unwrap();
    header.write_bits(0, 2).unwrap(); // empty name
    header.write_bits(0, 1).unwrap(); // custom restoration
    header.write_bits(u64::from(gaborish), 1).unwrap();
    if gaborish {
        header.write_bits(u64::from(custom), 1).unwrap();
        if custom {
            for (a, b) in [(0x2c00, 0x2800), (0x3000, 0x2800), (0x2800, 0x2400)] {
                header.write_bits(a, 16).unwrap();
                header.write_bits(b, 16).unwrap();
            }
        }
    }
    header.write_bits(u64::from(epf), 2).unwrap();
    if epf != 0 {
        assert!(!case.vardct);
        header.write_bits(0, 1).unwrap(); // default channel weights
        header.write_bits(u64::from(custom), 1).unwrap();
        if custom {
            for value in [0x3e00, 0x3a00, 0x3d00] {
                // pass0 1.5, pass2 .75, border 1.25
                header.write_bits(value, 16).unwrap();
            }
        }
        header
            .write_bits(if custom { 0x4000 } else { 0x3c00 }, 16)
            .unwrap(); // Modular sigma
    }
    header.write_bits(0, 2).unwrap(); // no restoration extensions
    header.write_bits(0, 2).unwrap(); // no frame extensions
    frame.header_bits = BitRange {
        offset: 0,
        length: header.bit_len() as u64,
    };
    reassemble(
        &header.into_bytes(),
        &frame,
        payloads(parsed.codestream(), &inventory.frames[0]),
    )
}

#[test]
fn vardct_lf_roots_and_skip_progressive_consumers_match_both_reference_decoders() {
    let Some(original) = crate::common::cjxl_progressive_dc_codestream(2) else {
        return;
    };
    let parsed = parse(&original, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let data = parsed.codestream();
    let start = inventory.frames[0].header_bits.offset as usize / 8;
    let end = inventory.frames[1].header_bits.offset as usize / 8;
    let case = Case {
        name: "vardct_lf_root",
        hex: "",
        format: LosslessModularFormat::Rgb,
        bits: 8,
        vardct: true,
    };
    let Some(backend) = backend() else {
        return;
    };
    for (gaborish, upsampling) in [(false, 1), (true, 1), (true, 2)] {
        let root = vardct_root(gaborish, upsampling);
        for unused in [false, true] {
            let mut modified = data[..start].to_vec();
            modified.extend_from_slice(&root);
            modified.extend_from_slice(&data[if unused { start } else { end }..]);
            let mut inventory = parse(&modified, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            let last = inventory.frames.last().unwrap();
            let offset = last.header_bits.offset as usize;
            assert_eq!((modified[offset / 8] >> (offset % 8)) & 1, 0);
            // SkipProgressive shares the regular physical header grammar and must consume LF too.
            for bit in [offset + 1, offset + 2] {
                modified[bit / 8] |= 1 << (bit % 8);
            }
            inventory = parse(&modified, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            assert_eq!(
                inventory.frames.last().unwrap().frame_type,
                FrameType::SkipProgressive
            );
            let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
            assert_eq!(plan.nodes[0].lf_last_use.is_none(), unused);
            let expected = rust_frames(&case, &modified);
            let djxl = djxl_frames(&case, &modified);
            for bounded in [false, true] {
                let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                let decoder = GpuDecoder::new(if bounded {
                    engine.with_stream_window_limit(NonZeroU64::new(256).unwrap())
                } else {
                    engine
                });
                let mut session = if bounded {
                    incremental(&decoder, &modified, request(&case))
                } else {
                    decoder.open(&modified, request(&case)).unwrap()
                };
                let frame = if bounded {
                    pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap()
                } else {
                    session.next_frame().unwrap().unwrap()
                };
                let pixels = samples(&read_output(&backend, &frame.output().outputs[0]), 8);
                assert_eq!(expected.len(), 1);
                assert_eq!(pixels.len(), expected[0].1.len());
                assert!(
                    pixels
                        .iter()
                        .zip(&expected[0].1)
                        .all(|(a, b)| a.abs_diff(*b) <= 1)
                );
                if let Some(djxl) = &djxl {
                    assert!(
                        pixels
                            .iter()
                            .zip(&djxl[0])
                            .all(|(a, b)| a.abs_diff(*b) <= 1)
                    );
                }
                drop(frame);
                drop(session);
                retired(&backend);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    0
                );
            }
        }
    }
}

#[test]
fn modular_lf_restoration_and_upsampling_match_both_reference_decoders() {
    let Some(original) = crate::common::cjxl_progressive_dc_codestream(2) else {
        return;
    };
    let Some(backend) = backend() else {
        return;
    };
    let parsed = parse(&original, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let data = parsed.codestream();
    let start = inventory.frames[0].header_bits.offset as usize / 8;
    let end = inventory.frames[1].header_bits.offset as usize / 8;
    let mut previous = None;
    for (gaborish, epf, upsampling, custom) in [
        (false, 0, 1, false),
        (true, 0, 1, false),
        (false, 1, 1, false),
        (false, 2, 1, false),
        (false, 3, 1, false),
        (true, 1, 1, true),
        (true, 2, 1, true),
        (true, 3, 1, true),
        (true, 3, 2, true),
        (true, 3, 4, true),
        (true, 3, 8, true),
    ] {
        let seed = Case {
            name: "modular_lf_root",
            format: LosslessModularFormat::Rgb,
            bits: 8,
            vardct: false,
            hex: match upsampling {
                1 => include_str!("../../../test-data/lf_modular_root_16x2.jxl.hex"),
                2 => include_str!("../../../test-data/lf_modular_root_8x1.jxl.hex"),
                4 => include_str!("../../../test-data/lf_modular_root_4x1.jxl.hex"),
                8 => include_str!("../../../test-data/lf_modular_root_2x1.jxl.hex"),
                _ => unreachable!(),
            },
        };
        let root = lf_root(&seed, gaborish, epf, upsampling, custom, 2);
        let mut modified = data[..start].to_vec();
        modified.extend_from_slice(&root);
        modified.extend_from_slice(&data[end..]);
        let case = Case {
            vardct: true,
            ..seed
        };
        let checked = parse(&modified, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert_eq!(checked.frames[0].encoding, FrameEncoding::Modular);
        assert_eq!(checked.frames[0].upsampling, upsampling);
        assert_eq!(checked.frames[0].lf_level, 2);
        assert_eq!(
            checked.frames[0].color_sample_extent(),
            Some((16_u32.div_ceil(upsampling), 2_u32.div_ceil(upsampling)))
        );
        FrameExecutionPlan::negotiate(&checked).unwrap();
        let expected = rust_frames(&case, &modified);
        let djxl = djxl_frames(&case, &modified);
        assert_eq!(expected.len(), 1);
        // Filtering must affect the independent reference; otherwise this would only test routing.
        if let Some(previous) = previous.as_ref().filter(|_| upsampling == 1) {
            assert!(
                &expected[0].1 != previous,
                "unchanged reference: Gaborish={gaborish}, EPF={epf}, upsampling={upsampling}, custom={custom}"
            );
        }
        previous = Some(expected[0].1.clone());
        for bounded in [false, true] {
            let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            let decoder = GpuDecoder::new(if bounded {
                engine.with_stream_window_limit(NonZeroU64::new(256).unwrap())
            } else {
                engine
            });
            let mut session = if bounded {
                incremental(&decoder, &modified, request(&case))
            } else {
                decoder.open(&modified, request(&case)).unwrap()
            };
            let frame = if bounded {
                pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap()
            } else {
                session.next_frame().unwrap().unwrap()
            };
            let pixels = samples(&read_output(&backend, &frame.output().outputs[0]), 8);
            for oracle in
                std::iter::once(&expected[0].1).chain(djxl.as_ref().map(|frames| &frames[0]))
            {
                assert_eq!(pixels.len(), oracle.len());
                let max_error = pixels
                    .iter()
                    .zip(oracle)
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap();
                assert!(
                    max_error <= 1,
                    "Gaborish={gaborish}, EPF={epf}, upsampling={upsampling}, custom={custom}, bounded={bounded}: {max_error}"
                );
            }
            drop(frame);
            assert!(session.next_frame().unwrap().is_none());
            drop(session);
            retired(&backend);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}
