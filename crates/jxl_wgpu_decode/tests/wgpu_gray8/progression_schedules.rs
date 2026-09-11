use super::*;
use common::modular_passes::{reframe, samples, schedules};
use jxl_wgpu_decode::Error;

#[test]
fn every_modular_pass_count_and_saturated_boundaries_match_native_output() {
    let Some(backend) = backend() else { return };
    let decoder = GpuDecoder::new(WgpuSubmissionEngine::new(backend.clone()));
    for (name, squeeze, width, height, hex) in [
        (
            "plain",
            false,
            1025,
            3,
            include_str!("../../test-data/modular_passes/plain.jxl.hex"),
        ),
        (
            "squeeze",
            true,
            2051,
            17,
            include_str!("../../test-data/modular_passes/squeeze.jxl.hex"),
        ),
    ] {
        let original = hex_bytes(hex);
        let expected = samples(width, height);
        for schedule in schedules(squeeze) {
            eprintln!("{name} {schedule:?}");
            let data = reframe(&original, &schedule);
            let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            let frame = &inventory.frames[0];
            assert_eq!(frame.num_passes, u32::from(schedule.passes));
            assert_eq!(
                frame.progressive_passes.downsampling,
                schedule
                    .boundaries
                    .iter()
                    .map(|&(factor, _)| factor)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                frame.progressive_passes.last_pass,
                schedule
                    .boundaries
                    .iter()
                    .map(|&(_, pass)| pass)
                    .collect::<Vec<_>>()
            );
            // The current Rust oracle rejects saturated downsampling headers and a full image
            // completed before an empty final pass. Those cases use source/native checks below.
            if schedule.boundaries.len() < usize::from(schedule.passes)
                && schedule.source_passes.last() == Some(&(schedule.passes - 1))
            {
                assert_eq!(
                    rust_jxl_decode_gray8(&data).unwrap(),
                    ((width as usize, height as usize), expected.clone())
                );
            }
            let mut native = decoder
                .open(
                    &data,
                    GpuOutputRequest::numeric(
                        Vpi::U8.pixel_format(),
                        NumericSampleMapping::NativeUnsigned,
                    )
                    .unwrap(),
                )
                .unwrap();
            assert_eq!(
                read_output(
                    &backend,
                    &native.next_frame().unwrap().unwrap().output().outputs[0]
                ),
                expected
            );
            drop(native);

            let mut session = decoder.open(&data, rgba_request()).unwrap();
            let mut seen = Vec::new();
            let mut held = Vec::new();
            while let Some(update) = session.next_update().unwrap() {
                let (input, options) = match update.progression() {
                    Some(FrameProgression::Modular {
                        completed_passes,
                        total_passes,
                        intended_downsampling,
                        ..
                    }) => {
                        assert_eq!(total_passes, schedule.passes);
                        assert_eq!(
                            intended_downsampling,
                            schedule
                                .boundaries
                                .iter()
                                .filter(|&&(_, last)| u32::from(completed_passes) > last)
                                .fold(8, |target, &(factor, _)| target.min(factor))
                        );
                        seen.push(completed_passes);
                        (
                            &data[..prefix_end(frame, completed_passes)],
                            vec!["--prefix", "--keep-orientation"],
                        )
                    }
                    None => (&data[..], vec!["--keep-orientation"]),
                    other => panic!("unexpected boundary {other:?}"),
                };
                let actual = read_output(&backend, &update.output().outputs[0]);
                if let Some(reference) =
                    extra_channels::extra_channel_oracle::libjxl_output(input, &options)
                {
                    assert_eq!(actual.len(), reference.len() * 4);
                    let error = actual
                        .chunks_exact(4)
                        .zip(reference)
                        .map(|(word, reference)| {
                            let value = f32::from_le_bytes(word.try_into().unwrap());
                            assert!(value.is_finite() && reference.is_finite());
                            (value - reference).abs()
                        })
                        .fold(0_f32, f32::max);
                    assert!(
                        error < 2e-6,
                        "{name} {schedule:?} {:?}: {error}",
                        update.progression()
                    );
                }
                held.push((update, actual));
            }
            let expected_passes = (0..schedule.passes)
                .filter(|&completed| squeeze || completed > schedule.source_passes[0])
                .collect::<Vec<_>>();
            assert_eq!(
                seen, expected_passes,
                "empty leading passes contain no image samples"
            );
            drop(session);
            for (update, expected) in &held {
                assert_eq!(
                    read_output(&backend, &update.output().outputs[0]),
                    *expected
                );
            }
            drop(held);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn eleven_modular_passes_preserve_bounded_admission_cancellation_and_late_validation() {
    let Some(backend) = backend() else { return };
    let original = hex_bytes(include_str!(
        "../../test-data/modular_passes/squeeze.jxl.hex"
    ));
    let schedule = schedules(true).pop().unwrap();
    assert_eq!(schedule.passes, 11);
    assert_eq!(schedule.source_passes, [0, 7]);
    let data = reframe(&original, &schedule);
    let request = rgba_request();
    let whole = GpuDecoder::new(WgpuSubmissionEngine::new(backend.clone()));
    let mut reference = whole.open(&data, request.clone()).unwrap();
    let mut expected = Vec::new();
    while let Some(update) = reference.next_update().unwrap() {
        expected.push((
            update.progression(),
            read_output(&backend, &update.output().outputs[0]),
        ));
    }
    drop(reference);
    assert_eq!(expected.len(), 12);
    assert_eq!(whole.engine().in_flight_memory_stats().reserved_bytes, 0);

    let decoder = GpuDecoder::new(
        WgpuSubmissionEngine::new(backend.clone())
            .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
    );
    let stats = decoder
        .open(&data, request.clone())
        .unwrap()
        .submission_session()
        .memory_stats();
    assert_eq!(
        stats.intermediate_output_bytes,
        stats.output_lease_bytes * 11
    );
    assert_eq!(stats.submissions_per_frame, stats.stream_batch_count + 12);
    let budget = MemoryBudget::new(NonZeroU64::new(stats.per_frame_bytes).unwrap());
    let bounded = GpuDecoder::new(
        WgpuSubmissionEngine::with_memory_budget(backend.clone(), budget.clone())
            .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
    );
    let mut session = bounded.open(&data, request.clone()).unwrap();
    let blocker = budget.try_reserve(1).unwrap();
    let progress = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert!(progress.backpressure.is_some());
    drop(blocker);
    for (progression, pixels) in &expected {
        let update = pollster::block_on(session.next_update_async())
            .unwrap()
            .unwrap();
        assert_eq!(update.progression(), *progression);
        assert_eq!(read_output(&backend, &update.output().outputs[0]), *pixels);
    }
    assert!(session.next_update().unwrap().is_none());
    drop(session);
    assert_eq!(budget.snapshot().reserved_bytes, 0);

    for completed in [0, 1, 8, 11] {
        let mut session = decoder.open(&data, request.clone()).unwrap();
        session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        let mut held = None;
        for (progression, pixels) in &expected[..completed] {
            let update = session.next_update().unwrap().unwrap();
            assert_eq!(update.progression(), *progression);
            assert_eq!(read_output(&backend, &update.output().outputs[0]), *pixels);
            held = Some(update);
        }
        drop(session);
        lifecycle::drain(&backend);
        assert_eq!(
            decoder.engine().in_flight_memory_stats().reserved_bytes,
            if held.is_some() {
                stats.output_lease_bytes
            } else {
                0
            }
        );
        if let Some(update) = &held {
            assert_eq!(
                read_output(&backend, &update.output().outputs[0]),
                expected[completed - 1].1
            );
        }
        drop(held);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }

    let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let section = inventory.frames[0]
        .sections
        .iter()
        .filter(|section| {
            matches!(
                section.kind,
                jxl_gpu_bitstream::FrameSectionKind::PassGroup { pass_index: 7, .. }
            )
        })
        .max_by_key(|section| section.bytes.length)
        .unwrap();
    assert!(section.bytes.length > 32);
    let mut corrupt = data.clone();
    let end = section.bytes.end().unwrap() as usize;
    corrupt[end - 16..end].fill(0xff);
    let mut session = decoder.open(&corrupt, request.clone()).unwrap();
    let mut held = Vec::new();
    for (progression, pixels) in &expected[..8] {
        let update = session.next_update().unwrap().unwrap();
        assert_eq!(update.progression(), *progression);
        assert_eq!(read_output(&backend, &update.output().outputs[0]), *pixels);
        held.push(update);
    }
    assert!(matches!(
        session.next_update(),
        Err(Error::ModularEntropyRejected { .. })
    ));
    assert!(matches!(session.next_update(), Err(Error::SessionPoisoned)));
    drop(session);
    lifecycle::drain(&backend);
    assert_eq!(
        decoder.engine().in_flight_memory_stats().reserved_bytes,
        stats.output_lease_bytes * held.len() as u64
    );
    for (update, (_, pixels)) in held.iter().zip(&expected) {
        assert_eq!(read_output(&backend, &update.output().outputs[0]), *pixels);
    }
    drop(held);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let mut session = decoder.open(&data, request).unwrap();
    let first = session.next_update().unwrap().unwrap();
    let final_image = pollster::block_on(session.next_frame_async())
        .unwrap()
        .unwrap();
    assert_eq!(
        read_output(&backend, &final_image.output().outputs[0]),
        expected.last().unwrap().1
    );
    drop((first, final_image, session));
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}
