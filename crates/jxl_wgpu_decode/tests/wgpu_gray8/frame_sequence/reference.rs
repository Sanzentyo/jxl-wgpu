//! Conformance-only header re-serialization. libjxl's public encoder cannot emit reference-only
//! frames or save slot 3. The entropy and image header come from its checked-in Modular fixture.
use super::*;
use jxl_gpu_bitstream::{BitWriter, FrameBlendInfo, FrameBlendMode, FrameEncoding, FrameType};

fn u32_field(writer: &mut BitWriter, value: u32, coding: [(u32, u8); 4]) {
    let (selector, (base, count)) = coding
        .into_iter()
        .enumerate()
        .find(|(_, (base, count))| value >= *base && u64::from(value - base) < (1u64 << count))
        .unwrap();
    writer.write_bits(selector as u64, 2).unwrap();
    writer.write_bits(u64::from(value - base), count).unwrap();
}

fn slot(value: u32) -> u32 {
    match value {
        1 => 3,
        3 => 1,
        other => other,
    }
}

fn blend(writer: &mut BitWriter, blend: FrameBlendInfo, full: bool, has_alpha: bool) {
    u32_field(writer, blend.mode as u32, [(0, 0), (1, 0), (2, 0), (3, 2)]);
    let alpha = matches!(
        blend.mode,
        FrameBlendMode::Blend | FrameBlendMode::MultiplyAdd
    ) && has_alpha;
    if alpha {
        u32_field(
            writer,
            blend.alpha_channel.unwrap(),
            [(0, 0), (1, 0), (2, 0), (3, 3)],
        );
    }
    if alpha || blend.mode == FrameBlendMode::Multiply {
        writer.write_bits(u64::from(blend.clamp), 1).unwrap();
    }
    if !full || blend.mode != FrameBlendMode::Replace {
        writer.write_bits(u64::from(slot(blend.source)), 2).unwrap();
    }
}

fn rewrite_headers(bytes: &[u8], before_transform: bool, single_crop: bool) -> Vec<u8> {
    let parsed = parse(bytes, Default::default()).unwrap();
    let inventory = parsed
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    let bytes = parsed.codestream();
    let image = &inventory.image_header;
    assert!(!image.xyb_encoded);
    let mut result = bytes[..inventory.frames[0].header_bits.offset as usize / 8].to_vec();
    let frames = if single_crop {
        &inventory.frames[1..2]
    } else {
        &inventory.frames
    };
    for frame in frames {
        assert_eq!(frame.encoding, FrameEncoding::Modular);
        assert_eq!(frame.flags, 0);
        assert!(!frame.do_ycbcr);
        assert_eq!(frame.num_passes, 1);
        assert_eq!(frame.upsampling, 1);
        assert!(!frame.toc_permuted);
        let reference = !single_crop && frame.frame_index == 0;
        let is_last = single_crop || frame.is_last;
        let mut writer = BitWriter::new();
        writer.write_bits(0, 1).unwrap(); // Non-default frame header.
        writer
            .write_bits(
                if reference {
                    2
                } else {
                    frame.frame_type as u64
                },
                2,
            )
            .unwrap();
        writer.write_bits(1, 1).unwrap(); // Modular.
        writer.write_bits(0, 2).unwrap(); // Flags U64(0).
        writer.write_bits(0, 1).unwrap(); // No YCbCr.
        writer.write_bits(0, 2).unwrap(); // Upsampling = 1.
        for &upsampling in &frame.extra_channel_upsampling {
            assert_eq!(upsampling, 1);
            writer.write_bits(0, 2).unwrap();
        }
        writer
            .write_bits(u64::from(frame.group_size_shift), 2)
            .unwrap();
        if !reference {
            writer.write_bits(0, 2).unwrap();
        } // One pass, absent for ReferenceOnly.
        writer.write_bits(u64::from(frame.have_crop), 1).unwrap();
        if frame.have_crop {
            let dimension = [(0, 8), (256, 11), (2304, 14), (18688, 30)];
            if !reference {
                for origin in [frame.x0, frame.y0] {
                    let signed = i64::from(origin);
                    u32_field(
                        &mut writer,
                        ((signed << 1) ^ (signed >> 63)) as u32,
                        dimension,
                    );
                }
            }
            u32_field(&mut writer, frame.width, dimension);
            u32_field(&mut writer, frame.height, dimension);
        }
        let full = frame.x0 <= 0
            && frame.y0 <= 0
            && i64::from(frame.x0) + i64::from(frame.width) >= i64::from(image.width)
            && i64::from(frame.y0) + i64::from(frame.height) >= i64::from(image.height);
        if !reference {
            blend(
                &mut writer,
                frame.color_blend,
                full,
                !image.extra_channels.is_empty(),
            );
            for &ec in &frame.extra_channel_blends {
                blend(&mut writer, ec, full, true);
            }
            if image.animation.is_some() {
                u32_field(
                    &mut writer,
                    frame.duration_ticks,
                    [(0, 0), (1, 0), (0, 8), (0, 32)],
                );
            }
            if let Some(timecode) = frame.timecode {
                writer.write_bits(u64::from(timecode), 32).unwrap();
            }
            writer.write_bits(u64::from(is_last), 1).unwrap();
        }
        if !is_last {
            writer
                .write_bits(u64::from(slot(frame.save_as_reference)), 2)
                .unwrap();
        }
        let can_reference = !is_last && (frame.duration_ticks == 0 || frame.save_as_reference != 0);
        if reference || (can_reference && full && frame.color_blend.mode == FrameBlendMode::Replace)
        {
            writer
                .write_bits(u64::from(reference && before_transform), 1)
                .unwrap();
        }
        writer.write_bits(0, 2).unwrap(); // Empty name.
        assert!(matches!(
            frame.restoration_filter,
            jxl_gpu_bitstream::RestorationFilterInventory::Custom {
                gaborish: jxl_gpu_bitstream::GaborishInventory::Disabled,
                epf: jxl_gpu_bitstream::EdgePreservingFilterInventory::Disabled,
            }
        ));
        writer.write_bits(0, 6).unwrap(); // Non-default filter, no Gaborish/EPF, no extensions.
        writer.write_bits(0, 2).unwrap(); // Frame extensions.
        writer.write_bits(0, 1).unwrap(); // No TOC permutation.
        writer.align_to_byte().unwrap();
        for section in &frame.sections {
            u32_field(
                &mut writer,
                section.bytes.length as u32,
                [(0, 10), (1024, 14), (17408, 22), (4211712, 30)],
            );
        }
        writer.align_to_byte().unwrap();
        result.extend_from_slice(writer.as_bytes());
        for section in &frame.sections {
            let start = section.bytes.offset as usize;
            result.extend_from_slice(&bytes[start..start + section.bytes.length as usize]);
        }
    }
    result
}

#[test]
fn reference_only_frames_save_slot_three_without_a_presentation() {
    let Some(backend) = backend() else {
        return;
    };
    for case in composition_cases()
        .into_iter()
        .filter(|case| matches!(case.name, "gray" | "rgba16"))
    {
        let bytes = rewrite_headers(&encoded(&case), false, false);
        let inventory = parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
        assert_eq!(inventory.frames[0].frame_type, FrameType::ReferenceOnly);
        assert_eq!(plan.nodes[0].save_reference, Some(3));
        assert_eq!(plan.nodes[1].references[3].unwrap().frame_index, 0);
        assert_eq!(plan.presentations[0].physical_frames, 0..3);
        let expected = rust_float_frames(&bytes, case.format);
        let djxl = djxl_frames(&case, &bytes);
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(4096).unwrap()),
        );
        let mut session = incremental(&decoder, &bytes, request(&case));
        for (index, oracle) in expected.iter().enumerate() {
            let frame = pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap();
            assert_eq!(frame.metadata, plan.presentations[index].metadata);
            let pixels = samples(
                &read_output(&backend, &frame.output().outputs[0]),
                case.bits,
            );
            let max = ((1u32 << case.bits) - 1) as f32;
            assert!(
                pixels
                    .iter()
                    .zip(oracle)
                    .all(|(a, b)| a.abs_diff((b.clamp(0.0, 1.0) * max).round() as u16) <= 1)
            );
            if let Some(djxl) = &djxl {
                assert!(
                    pixels
                        .iter()
                        .zip(&djxl[index])
                        .all(|(a, b)| a.abs_diff(*b) <= 1)
                );
            }
        }
        assert!(session.next_frame().unwrap().is_none());
        // The identical entropy with a pre-transform reference is invalid as an after-transform
        // blending background, and must be rejected before allocating or submitting anything.
        let invalid = rewrite_headers(&encoded(&case), true, false);
        assert!(matches!(
            decoder.open(&invalid, request(&case)),
            Err(jxl_wgpu_decode::Error::FramePlan(
                FramePlanError::InvalidFrame { frame_index: 1, .. }
            ))
        ));
    }
}

#[test]
fn single_cropped_still_uses_the_frame_compositor() {
    let Some(backend) = backend() else {
        return;
    };
    let case = composition_cases()
        .into_iter()
        .find(|case| case.name == "still")
        .unwrap();
    let bytes = rewrite_headers(&encoded(&case), false, true);
    let inventory = parse(&bytes, Default::default())
        .unwrap()
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    assert!(inventory.image_header.animation.is_none());
    assert_eq!(inventory.frames.len(), 1);
    assert!(inventory.frames[0].have_crop && inventory.frames[0].is_last);
    let expected = rust_frames(&case, &bytes);
    let djxl = djxl_frames(&case, &bytes);
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let mut session = decoder.open(&bytes, request(&case)).unwrap();
    assert!(matches!(
        session.profile(),
        DecodeProfile::FrameSequence {
            physical_frames: 1,
            presentation_frames: 1
        }
    ));
    let frame = session.next_frame().unwrap().unwrap();
    let pixels = samples(&read_output(&backend, &frame.output().outputs[0]), 8);
    assert!(pixels.contains(&0) && pixels.iter().any(|&value| value != 0));
    assert_eq!(pixels, expected[0].1);
    if let Some(djxl) = djxl {
        assert_eq!(pixels, djxl[0]);
    }
}
