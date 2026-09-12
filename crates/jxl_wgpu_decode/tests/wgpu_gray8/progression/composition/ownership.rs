use super::*;
use jxl_wgpu_decode::{Error, VarDctDecodeError};

fn request(name: &str) -> GpuOutputRequest {
    if name != "modular_pass_rgb" {
        GpuOutputRequest::numeric(
            Vpi::F32.pixel_format(),
            if name == "modular_pass_float" {
                NumericSampleMapping::NativeFloat
            } else {
                NumericSampleMapping::NormalizedUnsigned
            },
        )
        .unwrap()
        .with_extra_channel(0)
        .unwrap()
        .with_progressive_output(true)
    } else {
        rgba_request()
    }
    .with_max_frame_slots(NonZeroUsize::new(1).unwrap())
}

fn retire(
    backend: &WgpuBackend,
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    held: &[(GpuImageFrame, Vec<u8>)],
) {
    lifecycle::drain(backend);
    let retained: u64 = held
        .iter()
        .map(|(frame, _)| frame.outputs[0].buffer.size())
        .sum();
    assert_eq!(
        decoder.engine().in_flight_memory_stats().reserved_bytes,
        retained
    );
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        0
    );
    for (frame, expected) in held {
        assert_eq!(read_output(backend, &frame.outputs[0]), *expected);
    }
}

#[test]
fn composed_updates_retry_admission_cancel_and_drain_to_unchanged_final_frames() {
    let Some(backend) = backend() else {
        return;
    };
    for (name, _, hex) in cases() {
        let data = hex_bytes(hex);
        let request = request(name);
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
        );
        let mut baseline = decoder
            .open(&data, request.clone().with_progressive_output(false))
            .unwrap();
        let mut finals = Vec::new();
        while let Some(frame) = baseline.next_frame().unwrap() {
            finals.push((
                frame.metadata.clone(),
                read_output(&backend, &frame.output().outputs[0]),
            ));
        }
        drop(baseline);
        let target = 2;
        for boundary in 0..=2 {
            for cancel in [false, true] {
                let mut session = frame_sequence::incremental(&decoder, &data, request.clone());
                let memory = backend.transient_memory_budget();
                let blocker = memory
                    .try_reserve(memory.snapshot().available_bytes)
                    .unwrap();
                for _ in 0..2 {
                    let progress = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
                    assert_eq!(progress.submitted, 0);
                    assert!(matches!(
                        progress.backpressure,
                        Some(PrefetchBackpressure::Memory(_))
                    ));
                }
                drop(blocker);
                for _ in 0..target {
                    drop(session.next_frame().unwrap().unwrap());
                }
                session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
                let mut held = Vec::new();
                for completed in 0..boundary {
                    let update = pollster::block_on(session.next_update_async())
                        .unwrap()
                        .unwrap();
                    assert_eq!(update.metadata, finals[target].0);
                    assert_eq!(
                        update
                            .progression()
                            .and_then(FrameProgression::completed_passes),
                        Some(completed)
                    );
                    held.push((
                        owned(update.output()),
                        read_output(&backend, &update.output().outputs[0]),
                    ));
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
                        read_output(&backend, &frame.output().outputs[0]),
                        finals[target].1
                    );
                    drop(frame);
                    let next = session.next_frame().unwrap().unwrap();
                    assert_eq!(next.metadata, finals[target + 1].0);
                    assert_eq!(
                        read_output(&backend, &next.output().outputs[0]),
                        finals[target + 1].1
                    );
                }
                drop(session);
                retire(&backend, &decoder, &held);
                drop(held);
                retire(&backend, &decoder, &[]);
                eprintln!("{name} boundary{boundary} cancel{cancel}: retired");
            }
        }
    }
}

#[test]
fn composed_late_pass_and_hidden_frame_failures_preserve_prior_images() {
    let Some(backend) = backend() else {
        return;
    };
    for (name, _, hex) in cases() {
        let transport = hex_bytes(hex);
        let data = jxl_gpu_bitstream::parse(&transport, Default::default())
            .unwrap()
            .codestream()
            .to_vec();
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
        let request = request(name);
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
        );
        for failure in ["pass", "hidden"] {
            let target = if failure == "hidden" { 3 } else { 2 };
            let physical = if failure == "hidden" {
                plan.presentations[target].physical_frames.start
            } else {
                plan.presentations[target].physical_frames.end - 1
            };
            let frame = &inventory.frames[physical];
            if failure == "hidden" {
                assert_eq!(frame.duration_ticks, 0);
            }
            let mut damaged = data.clone();
            {
                let section = frame.sections.iter().filter(|section| matches!(section.kind,
                    jxl_gpu_bitstream::FrameSectionKind::PassGroup { pass_index, .. } if pass_index + 1 == frame.num_passes))
                    .max_by_key(|section| section.bytes.length).unwrap();
                assert!(section.bytes.length > 32);
                let end = section.bytes.end().unwrap() as usize;
                damaged[end - 16..end].fill(0xff);
            }
            let mut session = frame_sequence::incremental(&decoder, &damaged, request.clone());
            let mut held = Vec::new();
            for _ in 0..target {
                let frame = session.next_frame().unwrap().unwrap();
                held.push((
                    owned(frame.output()),
                    read_output(&backend, &frame.output().outputs[0]),
                ));
            }
            if failure != "hidden" {
                let update = session.next_update().unwrap().unwrap();
                assert_eq!(
                    update
                        .progression()
                        .and_then(FrameProgression::completed_passes),
                    Some(0)
                );
                held.push((
                    owned(update.output()),
                    read_output(&backend, &update.output().outputs[0]),
                ));
            }
            loop {
                match pollster::block_on(session.next_update_async()) {
                    Ok(Some(update)) => {
                        assert_eq!(
                            failure, "pass",
                            "an invalid hidden frame cannot publish an image"
                        );
                        assert!(!update.is_complete());
                        held.push((
                            owned(update.output()),
                            read_output(&backend, &update.output().outputs[0]),
                        ));
                    }
                    Err(error) => {
                        match failure {
                            _ if name == "associated_vardct" => assert!(
                                matches!(
                                    error,
                                    Error::VarDct(VarDctDecodeError::HfCoefficientGpu(_))
                                ),
                                "{error:?}"
                            ),
                            _ => assert!(
                                matches!(error, Error::ModularEntropyRejected { .. }),
                                "{error:?}"
                            ),
                        }
                        break;
                    }
                    Ok(None) => panic!("damaged presentation completed"),
                }
            }
            assert!(matches!(session.next_update(), Err(Error::SessionPoisoned)));
            assert!(matches!(session.next_frame(), Err(Error::SessionPoisoned)));
            drop(session);
            retire(&backend, &decoder, &held);
            drop(held);
            retire(&backend, &decoder, &[]);
            eprintln!("{name} failure {failure}: retired");
        }
    }
}
