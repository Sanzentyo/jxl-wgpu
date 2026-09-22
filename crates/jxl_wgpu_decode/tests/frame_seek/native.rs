use super::*;

#[test]
fn native_indexes_preserve_offsets_and_bad_native_clocks_cannot_authorize_seeking() {
    let oracle = jxl_test_support::oracles::frame_index::FrameIndexOracle::compile();
    let backend = backend();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for mode in ["still", "dense", "sparse"] {
        let data = oracle.encode(mode);
        let parsed = jxl_gpu_bitstream::parse(&data, Default::default()).unwrap();
        let index = FrameIndex::from_container(&parsed, Default::default())
            .unwrap()
            .unwrap();
        let source = inventory(&data);
        let plan = FrameExecutionPlan::negotiate(&source).unwrap();
        let mut presentation = 0;
        for entry in index.entries() {
            assert_eq!(
                entry.codestream_offset,
                source.frames[plan.presentations[presentation].physical_frames.start]
                    .header_bits
                    .offset
                    / 8
            );
            presentation += entry.frames as usize;
        }
        assert_eq!(presentation, plan.presentations.len());
        if mode == "still" {
            BoundFrameIndex::new(source.clone(), Some(index), Default::default()).unwrap();
            let stream = streaming::receive(&decoder, &data, request());
            let mut seek = stream.finish(0, Default::default()).unwrap();
            let frame = seek.next_frame().unwrap().unwrap();
            assert_eq!(frame.metadata.index, 0);
            assert!(frame.is_complete());
            drop((frame, seek));
            released(&backend);
        } else {
            // Unmodified 0.12.0 writes a zero first interval despite a nonzero frame duration.
            // Keep this as a rejection witness, not a compatibility relaxation or oracle repair.
            assert_eq!(index.entries()[0].duration_ticks, 0);
            assert!(plan.presentations[0].metadata.duration.ticks > 0);
            assert!(matches!(
                BoundFrameIndex::new(source.clone(), Some(index), Default::default()),
                Err(FrameSeekError::Duration { entry: 0 })
            ));
            assert!(matches!(
                decoder.open_seek(&data, request(), 0, Default::default(), Default::default()),
                Err(jxl_wgpu_decode::Error::FrameSeek(
                    FrameSeekError::Duration { entry: 0 }
                ))
            ));
            assert!(matches!(
                streaming::receive(&decoder, &data, request()).finish(0, Default::default()),
                Err(jxl_wgpu_decode::Error::FrameSeek(
                    FrameSeekError::Duration { entry: 0 }
                ))
            ));
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
            released(&backend);
        }
        let generated = BoundFrameIndex::new(source, None, Default::default()).unwrap();
        let output = indexed(&data, generated.index(), false);
        let native: Vec<_> = native_updates(&output, false)
            .expect("native decoder required")
            .into_iter()
            .filter(|frame| frame.complete)
            .collect();
        for (target, reference) in native.iter().enumerate() {
            let mut seek = decoder
                .open_seek(
                    &output,
                    request(),
                    target,
                    Default::default(),
                    Default::default(),
                )
                .unwrap();
            let frame = seek.next_frame().unwrap().unwrap();
            let pixels = planes::read_bytes(&backend, &frame.output().outputs[0]);
            let expected: Vec<_> = (0..17 * 9)
                .flat_map(|pixel| {
                    let mut rgb = [0; 4];
                    for (channel, value) in rgb[..3].iter_mut().enumerate() {
                        *value = ((3 * pixel + channel) * 17 + target * 29) as u8;
                    }
                    rgb[3] = 255;
                    rgb
                })
                .collect();
            assert!(pixels == expected, "{mode}/{target}: source words");
            assert_eq!(
                reference.duration,
                if mode == "still" {
                    0
                } else {
                    [3, 5, 7, 11][target]
                }
            );
            let oracle: Vec<_> = reference
                .pixels
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| (f32::from_le_bytes(*word) * 255.0).round() as u8)
                .collect();
            assert!(pixels == oracle, "{mode}/{target}: native words");
            drop((frame, seek));
            released(&backend);
        }
    }
}
