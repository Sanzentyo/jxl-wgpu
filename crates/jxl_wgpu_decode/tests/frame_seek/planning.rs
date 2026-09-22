use super::*;

#[test]
fn generated_indexes_bind_exact_offsets_intervals_and_seek_bounds() {
    for name in [
        "sequence_modular_gray",
        "sequence_modular_many",
        "sequence_vardct_dc",
        "composition_mixed",
        "preview/animation_modular",
    ] {
        let data = source(name);
        let source = inventory(&data);
        let selected = SelectedImageInventory::new(source.clone(), ImageSelection::Main).unwrap();
        let execution = FrameExecutionPlan::negotiate(selected.reconstruction_inventory()).unwrap();
        let bound = BoundFrameIndex::new(source.clone(), None, Default::default()).unwrap();
        let payload = bound.index().encode(Default::default()).unwrap();
        let parsed = FrameIndex::parse(&payload, Default::default()).unwrap();
        BoundFrameIndex::new(source.clone(), Some(parsed), Default::default()).unwrap();
        for target in 0..execution.presentations.len() {
            let seek = bound.seek(target, Default::default()).unwrap();
            assert_eq!(seek.target(), &execution.presentations[target].metadata);
            let count = seek.physical_frames().len();
            assert!(matches!(
                bound.seek(
                    target,
                    FrameSeekLimits {
                        max_physical_frames: count - 1,
                        ..Default::default()
                    }
                ),
                Err(FrameSeekError::PhysicalLimit { .. })
            ));
            if seek.preroll_presentations() > 0 {
                assert!(matches!(
                    bound.seek(
                        target,
                        FrameSeekLimits {
                            max_preroll_presentations: seek.preroll_presentations() - 1,
                            ..Default::default()
                        }
                    ),
                    Err(FrameSeekError::PrerollLimit { .. })
                ));
            }
        }
        assert!(matches!(
            bound.seek(execution.presentations.len(), Default::default()),
            Err(FrameSeekError::Target { .. })
        ));
        for field in 0..3 {
            let mut entries = bound.index().entries().to_vec();
            match field {
                0 => entries[0].codestream_offset += 1,
                1 => entries[0].duration_ticks += 1,
                _ => entries.last_mut().unwrap().frames += 1,
            }
            let wrong = FrameIndex::new(
                bound.index().tick_numerator(),
                bound.index().tick_denominator(),
                entries,
                Default::default(),
            )
            .unwrap();
            assert!(BoundFrameIndex::new(source.clone(), Some(wrong), Default::default()).is_err());
        }
    }
}

#[test]
fn older_reference_versions_force_preroll_across_a_later_independent_anchor() {
    use jxl_gpu_bitstream::{FrameBlendMode, FrameType};
    let mut source = (*inventory(&source("sequence_modular_many"))).clone();
    // Header-only graph probe: three visible frames, an old slot, an unrelated keyframe,
    // and an Add consumer of the old slot. Entropy is not used by this planner test.
    source.frames.truncate(3);
    for (i, frame) in source.frames.iter_mut().enumerate() {
        frame.frame_type = FrameType::Regular;
        frame.duration_ticks = 1;
        frame.is_last = i == 2;
        frame.save_as_reference = if i == 0 { 1 } else { 0 };
        frame.save_before_color_transform = false;
        frame.flags = 0;
        frame.color_blend.mode = FrameBlendMode::Replace;
    }
    source.frames[2].color_blend.mode = FrameBlendMode::Add;
    source.frames[2].color_blend.source = 1;
    let bound = BoundFrameIndex::new(Arc::new(source), None, Default::default()).unwrap();
    assert_eq!(bound.index().entries().len(), 2);
    assert_eq!(
        bound
            .seek(1, Default::default())
            .unwrap()
            .restart_presentation(),
        1
    );
    let target = bound.seek(2, Default::default()).unwrap();
    assert_eq!(target.restart_presentation(), 0);
    assert_eq!(target.preroll_presentations(), 2);
}

#[test]
fn alpha_patch_and_overwritten_slot_dependencies_use_the_correct_producer() {
    use jxl_gpu_bitstream::{FrameBlendMode, FrameType};
    let base = inventory(&source("composition_rgba8"));
    for feature in ["alpha", "patch", "overwrite"] {
        // Header-only dependency probe. No entropy is manufactured or decoded by this test.
        let mut source = (*base).clone();
        let prototype = source.frames[0].clone();
        source.frames = (0..4)
            .map(|i| {
                let mut frame = prototype.clone();
                frame.frame_index = i;
                frame.frame_type = FrameType::Regular;
                frame.duration_ticks = 1;
                frame.is_last = i == 3;
                frame.is_preview = false;
                frame.x0 = 0;
                frame.y0 = 0;
                frame.width = source.image_header.width;
                frame.height = source.image_header.height;
                frame.have_crop = false;
                frame.save_as_reference = if i == 0 { 1 } else { 0 };
                frame.save_before_color_transform = false;
                frame.flags = 0;
                frame.header_bits.offset += u64::from(i) * 80;
                for blend in
                    std::iter::once(&mut frame.color_blend).chain(&mut frame.extra_channel_blends)
                {
                    blend.mode = FrameBlendMode::Replace;
                    blend.source = 0;
                    blend.alpha_channel = None;
                }
                frame
            })
            .collect();
        match feature {
            "alpha" => {
                // Color reads empty slot 2, but its alpha comes from old slot 1 despite Replace.
                source.frames[3].color_blend.mode = FrameBlendMode::Blend;
                source.frames[3].color_blend.source = 2;
                source.frames[3].color_blend.alpha_channel = Some(0);
                source.frames[3].extra_channel_blends[0].source = 1;
            }
            "patch" => source.frames[3].flags = 2,
            _ => {
                source.frames[2].save_as_reference = 1;
                source.frames[3].color_blend.mode = FrameBlendMode::Add;
                source.frames[3].color_blend.source = 1;
            }
        }
        let source = Arc::new(source);
        let bound = BoundFrameIndex::new(source.clone(), None, Default::default()).unwrap();
        assert_eq!(
            bound
                .seek(3, Default::default())
                .unwrap()
                .restart_presentation(),
            if feature == "overwrite" { 2 } else { 0 },
            "{feature}"
        );
        // Merely placing an index at the dependent target cannot make it independent.
        let mut entries = bound.index().entries().to_vec();
        entries.last_mut().unwrap().frames -= 1;
        entries.last_mut().unwrap().duration_ticks -= 1;
        entries.push(jxl_gpu_bitstream::FrameIndexEntry {
            codestream_offset: source.frames[3].header_bits.offset / 8,
            duration_ticks: 1,
            frames: 1,
        });
        let wrong = FrameIndex::new(
            bound.index().tick_numerator(),
            bound.index().tick_denominator(),
            entries,
            Default::default(),
        )
        .unwrap();
        assert!(
            matches!(
                BoundFrameIndex::new(source, Some(wrong), Default::default()),
                Err(FrameSeekError::DependentAnchor { .. })
            ),
            "{feature}"
        );
    }
}

#[test]
fn equivalent_tick_units_bind_without_rounding_and_limits_are_rechecked() {
    let source = inventory(&source("sequence_modular_many"));
    let bound = BoundFrameIndex::new(source.clone(), None, Default::default()).unwrap();
    let mut entries = bound.index().entries().to_vec();
    for entry in &mut entries {
        entry.duration_ticks *= 3;
    }
    let scaled = FrameIndex::new(
        bound.index().tick_numerator(),
        std::num::NonZeroU32::new(bound.index().tick_denominator().get() * 3).unwrap(),
        entries,
        Default::default(),
    )
    .unwrap();
    BoundFrameIndex::new(source.clone(), Some(scaled.clone()), Default::default()).unwrap();
    assert!(matches!(
        BoundFrameIndex::new(
            source,
            Some(scaled),
            jxl_gpu_bitstream::FrameIndexLimits {
                max_entries: 1,
                ..Default::default()
            }
        ),
        Err(FrameSeekError::Index(
            jxl_gpu_bitstream::FrameIndexError::EntryLimit
        ))
    ));
}
