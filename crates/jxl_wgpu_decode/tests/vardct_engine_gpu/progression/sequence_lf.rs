use super::*;

fn fixture(deferred: bool) -> Vec<u8> {
    let data = encoded(include_str!(
        "../../../test-data/composition_vardct_dc.jxl.hex"
    ));
    if !deferred {
        return data;
    }
    let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(inventory.frames.len(), 15);
    // Move a real zero-duration layer between LF1 and its visible consumer. That hidden layer
    // writes slot1, which the consumer reads. Every header/entropy byte remains unchanged.
    let mut output = data[..inventory.frames[0].header_bits.offset as usize / 8].to_vec();
    for index in [0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14] {
        let start = inventory.frames[index].header_bits.offset as usize / 8;
        let end = inventory
            .frames
            .get(index + 1)
            .map_or(data.len(), |frame| frame.header_bits.offset as usize / 8);
        output.extend_from_slice(&data[start..end]);
    }
    output
}

#[test]
fn composed_lf_images_follow_exact_references_and_match_independent_layer_flushes() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    for deferred in [false, true] {
        let data = fixture(deferred);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        if deferred {
            let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
            assert_eq!(plan.nodes[8].lf_source_frame, Some(6));
            assert_eq!(plan.nodes[8].references[1].unwrap().frame_index, 7);
        }
        for keep in [false, true] {
            let Some((native, native_lf)) =
                sequence_oracle::composed_with_lf(&data, &inventory, "composed_lf", keep)
            else {
                eprintln!("native libjxl composition oracle unavailable");
                return;
            };
            assert_eq!(native_lf.len(), 3);
            let plan = FrameExecutionPlan::negotiate_with_orientation(
                &inventory,
                request(keep).orientation_policy(),
            )
            .unwrap();
            let expected_lf: Vec<_> = plan
                .presentations
                .iter()
                .enumerate()
                .flat_map(|(index, presentation)| {
                    let mut dependencies = Vec::new();
                    let mut source =
                        inventory.frames[presentation.physical_frames.end - 1].lf_source_frame;
                    while let Some(id) = source {
                        let frame = &inventory.frames[id as usize];
                        if id as usize >= presentation.physical_frames.start {
                            dependencies.push((index, id, frame.lf_level as u8));
                        }
                        source = frame.lf_source_frame;
                    }
                    dependencies.reverse();
                    dependencies
                })
                .collect();
            assert_eq!(expected_lf.len(), 6);
            let mut whole = None;
            for cap in [u64::MAX, 40] {
                let decoder = GpuDecoder::new(
                    WgpuDecodeEngine::new(backend.clone())
                        .unwrap()
                        .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
                );
                let mut session = if cap == u64::MAX {
                    decoder.open(&data, request(keep)).unwrap()
                } else {
                    open_incremental(&decoder, &data, request(keep))
                };
                let mut seen_lf = 0;
                let mut seen_lf1 = 0;
                let mut complete = 0;
                let mut held = Vec::new();
                let mut pixels = Vec::new();
                while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                    let actual = read(&backend, update.output());
                    if let Some(FrameProgression::LowFrequency {
                        physical_frame_index,
                        level,
                    }) = update.progression()
                    {
                        let (presentation, physical, expected_level) = expected_lf[seen_lf];
                        assert_eq!((physical_frame_index, level), (physical, expected_level));
                        assert_eq!(update.metadata, plan.presentations[presentation].metadata);
                        seen_lf += 1;
                        if level == 1 {
                            let expected = &native_lf[seen_lf1];
                            assert_eq!(
                                (expected.frame, expected.physical_frame_index),
                                (presentation, physical)
                            );
                            let error = relative_error(&actual, &expected.pixels);
                            eprintln!(
                                "composed LF1 deferred{deferred} keep{keep} cap{cap} presentation{presentation}: {error}"
                            );
                            assert!(error < 1e-3);
                            seen_lf1 += 1;
                        }
                    }
                    if update.is_complete() {
                        let expected = native
                            .iter()
                            .filter(|image| image.complete)
                            .nth(complete)
                            .unwrap();
                        assert!(relative_error(&actual, &expected.pixels) < 1e-3);
                        complete += 1;
                    }
                    held.push(owned(update.output()));
                    pixels.push(actual);
                }
                assert_eq!((seen_lf, seen_lf1, complete), (6, 3, 6));
                for (frame, expected) in held.iter().zip(&pixels) {
                    assert_eq!(read(&backend, frame), *expected);
                }
                if let Some(whole) = &whole {
                    assert_eq!(&pixels, whole);
                } else {
                    whole = Some(pixels);
                }
                drop(held);
                drop(session);
                drain_gpu(&backend, 0);
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
fn a_later_hidden_reference_error_prevents_publication_of_queued_lf_images() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let mut data = fixture(true);
    let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let hidden = &inventory.frames[7];
    assert_eq!(hidden.duration_ticks, 0);
    let packet = hidden.sections.iter().filter(|section| matches!(section.kind,
        jxl_gpu_bitstream::FrameSectionKind::PassGroup { pass_index, .. } if pass_index + 1 == hidden.num_passes))
        .max_by_key(|section| section.bytes.length).unwrap();
    let start = packet.bytes.offset as usize;
    data[start..start + packet.bytes.length as usize].fill(0xff);
    for cap in [u64::MAX, 40] {
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
        );
        let mut session = if cap == u64::MAX {
            decoder.open(&data, request(false)).unwrap()
        } else {
            open_incremental(&decoder, &data, request(false))
        };
        drop(session.next_frame().unwrap().unwrap());
        let frame = session.next_frame().unwrap().unwrap();
        let held = owned(frame.output());
        let pixels = read(&backend, &held);
        drop(frame);
        assert!(matches!(
            pollster::block_on(session.next_update_async()),
            Err(DecodeError::VarDct(VarDctDecodeError::HfCoefficientGpu(_)))
        ));
        assert!(matches!(
            session.next_frame(),
            Err(DecodeError::SessionPoisoned)
        ));
        drop(session);
        drain_gpu(&backend, held.outputs[0].buffer.as_wgpu_buffer().size());
        assert_eq!(read(&backend, &held), pixels);
        drop(held);
        drain_gpu(&backend, 0);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
    }
}

#[test]
fn composed_lf_publication_survives_cancellation_pressure_and_final_only_switching() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    for deferred in [false, true] {
        let data = fixture(deferred);
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
        );
        let mut baseline = decoder
            .open(&data, request(false).with_progressive_output(false))
            .unwrap();
        let mut expected = Vec::new();
        while let Some(frame) = baseline.next_frame().unwrap() {
            expected.push((frame.metadata.clone(), read(&backend, frame.output())));
        }
        drop(baseline);
        for boundary in 0..=2 {
            for finish in [false, true] {
                let mut session = open_incremental(&decoder, &data, request(false));
                for _ in 0..2 {
                    drop(session.next_frame().unwrap().unwrap());
                }
                session.prefetch(session.resolved_frame_slots()).unwrap();
                let mut held = Vec::new();
                for i in 0..boundary {
                    let frame = session.next_update().unwrap().unwrap();
                    assert_eq!(frame.metadata, expected[2].0);
                    assert!(
                        matches!(frame.progression(), Some(FrameProgression::LowFrequency { level, .. }) if level == 2 - i)
                    );
                    held.push((owned(frame.output()), read(&backend, frame.output())));
                }
                if finish {
                    let frame = if boundary == 1 {
                        session.next_frame().unwrap().unwrap()
                    } else {
                        pollster::block_on(session.next_frame_async())
                            .unwrap()
                            .unwrap()
                    };
                    assert_eq!(frame.metadata, expected[2].0);
                    assert_eq!(read(&backend, frame.output()), expected[2].1);
                    drop(frame);
                    let frame = session.next_frame().unwrap().unwrap();
                    assert_eq!(frame.metadata, expected[3].0);
                    assert_eq!(read(&backend, frame.output()), expected[3].1);
                }
                drop(session);
                let bytes = held
                    .iter()
                    .map(|(frame, _)| frame.outputs[0].buffer.as_wgpu_buffer().size())
                    .sum();
                drain_gpu(&backend, bytes);
                assert_eq!(
                    decoder.engine().in_flight_memory_stats().reserved_bytes,
                    bytes
                );
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    0
                );
                for (frame, pixels) in &held {
                    assert_eq!(read(&backend, frame), *pixels);
                }
                drop(held);
                drain_gpu(&backend, 0);
            }
        }
        for boundary in 1..=2 {
            let mut session = open_incremental(&decoder, &data, request(false));
            for _ in 0..2 {
                drop(session.next_frame().unwrap().unwrap());
            }
            let mut held = None;
            for _ in 0..boundary {
                held = session.next_update().unwrap();
            }
            let held = held.unwrap();
            let pixels = read(&backend, held.output());
            let memory = backend.transient_memory_budget();
            let blocker = memory
                .try_reserve(memory.snapshot().available_bytes)
                .unwrap();
            assert!(session.next_update().is_err());
            assert!(matches!(
                session.next_frame(),
                Err(DecodeError::SessionPoisoned)
            ));
            drop(blocker);
            drop(session);
            let bytes = held.output().outputs[0].buffer.as_wgpu_buffer().size();
            drain_gpu(&backend, bytes);
            assert_eq!(memory.snapshot().reserved_bytes, bytes);
            assert_eq!(read(&backend, held.output()), pixels);
            drop(held);
            drain_gpu(&backend, 0);
        }
    }
}
