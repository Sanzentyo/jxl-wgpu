use super::*;

fn retire(backend: &WgpuBackend, decoder: &GpuDecoder<WgpuDecodeEngine>, expected: u64) {
    drain_gpu(backend, expected);
    assert_eq!(
        decoder.engine().in_flight_memory_stats().reserved_bytes,
        expected
    );
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        0
    );
}

#[test]
fn sequence_refinements_retry_admission_cancel_and_switch_to_final_completion() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    for (name, hex) in cases()
        .into_iter()
        .filter(|(name, _)| matches!(*name, "rgb" | "lf" | "composed" | "composed_lf"))
    {
        let data = encoded(hex);
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
        );
        let target = usize::from(name.starts_with("composed"));
        let mut reference = decoder
            .open(&data, request(false).with_progressive_output(false))
            .unwrap();
        let mut finals = Vec::new();
        while let Some(frame) = reference.next_frame().unwrap() {
            finals.push((frame.metadata.clone(), read(&backend, frame.output())));
        }
        drop(reference);
        for boundary in 0..=2 {
            for cancel in [true, false] {
                let mut session = open_incremental(&decoder, &data, request(false));
                let memory = backend.transient_memory_budget();
                let blocker = memory
                    .try_reserve(memory.snapshot().available_bytes)
                    .unwrap();
                for _ in 0..2 {
                    let progress = session.prefetch(session.resolved_frame_slots()).unwrap();
                    assert_eq!(progress.submitted, 0);
                    assert_eq!(session.frames_submitted(), 0);
                    assert!(matches!(
                        progress.backpressure,
                        Some(PrefetchBackpressure::Memory(_))
                    ));
                }
                drop(blocker);
                for _ in 0..target {
                    drop(session.next_frame().unwrap().unwrap());
                }
                session.prefetch(session.resolved_frame_slots()).unwrap();
                let mut held = None;
                let mut coefficients = 0;
                while coefficients < boundary {
                    let update = pollster::block_on(session.next_update_async())
                        .unwrap()
                        .unwrap();
                    assert_eq!(update.metadata, finals[target].0);
                    assert!(!update.is_complete());
                    if let Some(FrameProgression::Coefficients {
                        completed_passes, ..
                    }) = update.progression()
                    {
                        assert_eq!(usize::from(completed_passes), coefficients);
                        coefficients += 1;
                        held = Some((owned(update.output()), read(&backend, update.output())));
                    }
                }
                if !cancel {
                    let frame = if boundary == 1 {
                        session.next_frame().unwrap().unwrap()
                    } else {
                        pollster::block_on(session.next_frame_async())
                            .unwrap()
                            .unwrap()
                    };
                    assert_eq!(frame.metadata, finals[target].0);
                    assert_eq!(
                        read(&backend, frame.output()),
                        finals[target].1,
                        "{name} boundary{boundary}"
                    );
                    drop(frame);
                    let next = session.next_frame().unwrap().unwrap();
                    assert_eq!(next.metadata, finals[target + 1].0);
                    assert_eq!(read(&backend, next.output()), finals[target + 1].1);
                }
                drop(session);
                let bytes = held.as_ref().map_or(0, |(frame, _)| {
                    frame.outputs[0].buffer.as_wgpu_buffer().size()
                });
                retire(&backend, &decoder, bytes);
                if let Some((frame, expected)) = held.as_ref() {
                    assert_eq!(
                        read(&backend, frame),
                        *expected,
                        "{name} cancel{cancel} boundary{boundary}"
                    );
                }
                drop(held);
                retire(&backend, &decoder, 0);
            }
        }
    }
}

#[test]
fn sequence_late_entropy_and_composition_pressure_preserve_earlier_updates() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    for (name, hex) in cases()
        .into_iter()
        .filter(|(name, _)| matches!(*name, "rgb" | "lf" | "composed" | "composed_lf"))
    {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
        let target = usize::from(name.starts_with("composed"));
        let physical = &inventory.frames[plan.presentations[target].physical_frames.end - 1];
        let packet = physical.sections.iter().filter(|section| matches!(section.kind,
            jxl_gpu_bitstream::FrameSectionKind::PassGroup { pass_index, .. } if pass_index + 1 == physical.num_passes))
            .max_by_key(|section| section.bytes.length).unwrap();
        let mut damaged = data.clone();
        let start = packet.bytes.offset as usize;
        damaged[start..start + packet.bytes.length as usize].fill(0xff);
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
        );
        for pressure in [false, true] {
            if pressure && !name.starts_with("composed") {
                continue;
            }
            let mut session = open_incremental(
                &decoder,
                if pressure { &data } else { &damaged },
                request(false),
            );
            let mut held = Vec::new();
            for _ in 0..target {
                let frame = session.next_frame().unwrap().unwrap();
                held.push((owned(frame.output()), read(&backend, frame.output())));
            }
            loop {
                let update = session.next_update().unwrap().unwrap();
                assert!(!update.is_complete());
                if matches!(
                    update.progression(),
                    Some(FrameProgression::Coefficients {
                        completed_passes: 0,
                        ..
                    })
                ) {
                    held.push((owned(update.output()), read(&backend, update.output())));
                    break;
                }
            }
            let memory = backend.transient_memory_budget();
            let blocker = pressure.then(|| {
                memory
                    .try_reserve(memory.snapshot().available_bytes)
                    .unwrap()
            });
            loop {
                match pollster::block_on(session.next_update_async()) {
                    Ok(Some(update)) => {
                        assert!(!pressure && !update.is_complete());
                        held.push((owned(update.output()), read(&backend, update.output())));
                    }
                    Err(error) => {
                        if pressure {
                            assert!(
                                matches!(error, DecodeError::MemoryBackpressure(_)),
                                "{name}: {error:?}"
                            );
                        } else {
                            assert!(
                                matches!(
                                    error,
                                    DecodeError::VarDct(VarDctDecodeError::HfCoefficientGpu(_))
                                ),
                                "{name}: {error:?}"
                            );
                        }
                        break;
                    }
                    Ok(None) => panic!("{name}: damaged presentation completed"),
                }
            }
            assert!(matches!(
                session.next_update(),
                Err(DecodeError::SessionPoisoned)
            ));
            assert!(matches!(
                session.next_frame(),
                Err(DecodeError::SessionPoisoned)
            ));
            drop(blocker);
            drop(session);
            let bytes = held
                .iter()
                .map(|(frame, _)| frame.outputs[0].buffer.as_wgpu_buffer().size())
                .sum();
            retire(&backend, &decoder, bytes);
            for (frame, expected) in &held {
                assert_eq!(read(&backend, frame), *expected);
            }
            drop(held);
            retire(&backend, &decoder, 0);
        }
    }
}
