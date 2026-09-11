use super::*;
use jxl_wgpu::{ImageReadbackPipeline, WgpuBackend};

fn data(deferred: bool) -> Vec<u8> {
    let text = include_str!("../../../test-data/composition_vardct_dc.jxl.hex")
        .split_whitespace()
        .collect::<String>();
    let data: Vec<_> = text
        .as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect();
    if !deferred {
        return data;
    }
    let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let mut result = data[..inventory.frames[0].header_bits.offset as usize / 8].to_vec();
    for index in [0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14] {
        let start = inventory.frames[index].header_bits.offset as usize / 8;
        let end = inventory
            .frames
            .get(index + 1)
            .map_or(data.len(), |frame| frame.header_bits.offset as usize / 8);
        result.extend_from_slice(&data[start..end]);
    }
    result
}

#[test]
fn lf_patch_flags_are_rejected_before_the_dependency_path_can_skip_rendering() {
    let data = data(false);
    let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let mut checked = 0;
    for (index, frame) in inventory.frames.iter().enumerate() {
        if frame.frame_type != FrameType::LowFrequency {
            continue;
        }
        let mut patched = inventory.clone();
        patched.frames[index].flags |= 2;
        let plan = FrameExecutionPlan::negotiate(&patched).unwrap();
        assert!(
            matches!(validate(&patched, &plan), Err(Error::UnsupportedProfile(error))
            if error.feature == UnsupportedCodestreamFeature::Patches)
        );
        checked += 1;
    }
    assert!(checked > 0);
}

fn read(backend: &WgpuBackend, frame: &GpuImageFrame) -> Vec<u8> {
    ImageReadbackPipeline::new(backend)
        .submit(frame)
        .unwrap()
        .wait()
        .unwrap()
        .frame
        .outputs[0]
        .bytes
        .clone()
}

fn decode(pending: &mut DependentPending) {
    let Some(Stage::Decode(mut decode)) = pending.stage.take() else {
        panic!("physical decode stage")
    };
    if pending.nodes[pending.physical - pending.first]
        .lf_last_use
        .is_some()
    {
        if let WgpuDecodePendingFrame::VarDct(producer) = decode.pending.as_mut() {
            producer.wait_until_dependency_submitted().unwrap();
        }
        decode.lf = Some(lf_output(&decode.pending).unwrap());
    }
    let frame = decode.pending.wait().unwrap();
    assert!(
        pending
            .decoded(frame, &decode.count, decode.lf, true)
            .unwrap()
            .is_none()
    );
}

#[test]
fn queued_and_submitted_lf_composition_stages_cancel_and_drain_without_reference_mutation() {
    let backend = match pollster::block_on(WgpuBackend::request_default(Default::default())) {
        Ok(backend) => backend,
        Err(jxl_wgpu::Error::NoAdapter) => return,
        Err(error) => panic!("{error:?}"),
    };
    for deferred in [false, true] {
        let data = data(deferred);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
        let source = Arc::new(
            GpuCodestream::from_shared(Arc::from(data.as_slice()), 0..data.len(), false).unwrap(),
        );
        let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        let request = GpuOutputRequest::color(crate::vardct_rgb8_format())
            .unwrap()
            .with_progressive_output(true);
        let mut baseline =
            DependentSession::new(engine.clone(), source.clone(), &inventory, &request, &plan)
                .unwrap();
        let expected: Vec<_> = (0..4)
            .map(|i| {
                let frame = baseline.submit(&plan, i).unwrap().wait().unwrap();
                (frame.metadata, read(&backend, &frame.output))
            })
            .collect();
        drop(baseline);
        for boundary in 0..4 {
            if boundary == 0 && !deferred {
                continue;
            }
            for action in 0..3 {
                let mut session = DependentSession::new(
                    engine.clone(),
                    source.clone(),
                    &inventory,
                    &request,
                    &plan,
                )
                .unwrap();
                for i in 0..2 {
                    drop(session.submit(&plan, i).unwrap().wait().unwrap());
                }
                let mut pending = session.submit(&plan, 2).unwrap();
                decode(&mut pending); // complete LF2
                if deferred {
                    assert_eq!(pending.lf_pending.len(), 1);
                    assert!(!pending.lf_references_ready().unwrap());
                    decode(&mut pending); // complete LF1; LF2's predictor slot has expired
                    assert_eq!(pending.lf_pending.len(), 2);
                    assert!(pending.carry.as_ref().unwrap().lf[1].is_none());
                    if boundary != 0 {
                        decode(&mut pending); // validate the hidden background producer
                        let Some(Stage::Blend(work)) = pending.stage.take() else {
                            panic!("hidden blend")
                        };
                        let surface = pending
                            .output
                            .compositor()
                            .unwrap()
                            .completed_surface(work.wait().unwrap());
                        pending.record(surface).unwrap();
                        assert!(pending.lf_references_ready().unwrap());
                    }
                }
                if boundary >= 2 {
                    let Some(Stage::LfPreview { work, progression }) = pending.stage.take() else {
                        panic!("LF render")
                    };
                    let buffer = work.wait().unwrap();
                    let surface = pending
                        .output
                        .compositor()
                        .unwrap()
                        .import(vec![GpuImageOutput {
                            id: OutputId(0),
                            layout: pending.lf_preview.as_ref().unwrap().layout.clone(),
                            buffer,
                        }])
                        .unwrap();
                    pending
                        .render_refinement(
                            Arc::clone(pending.output.compositor().unwrap()),
                            surface,
                            pending.end - 1,
                            Resume::Advance,
                            progression,
                        )
                        .unwrap();
                }
                if boundary == 3 {
                    let Some(Stage::Refinement {
                        compositor,
                        resume,
                        render: RefinementRender::Blend(work),
                        progression,
                    }) = pending.stage.take()
                    else {
                        panic!("LF blend")
                    };
                    let surface = compositor.completed_surface(work.wait().unwrap());
                    pending.stage = Some(Stage::Refinement {
                        compositor: Arc::clone(&compositor),
                        resume,
                        render: RefinementRender::Pack(compositor.pack(&surface).unwrap()),
                        progression,
                    });
                    pending.submissions.fetch_add(1, Ordering::AcqRel);
                    pending.completed_submissions += 1;
                }
                assert!(matches!(
                    pending.unvalidated(),
                    Err(Error::UnvalidatedOutputNotSubmitted)
                ));
                if action == 0 {
                    drop(pending);
                    assert!(matches!(
                        session.submit(&plan, 2),
                        Err(Error::SessionPoisoned)
                    ));
                } else {
                    let frame = if action == 1 {
                        pending.wait().unwrap()
                    } else {
                        let frame = pollster::block_on(std::future::poll_fn(|context| {
                            pending.poll(context)
                        }))
                        .unwrap();
                        drop(pending);
                        frame
                    };
                    assert_eq!(frame.metadata, expected[2].0);
                    assert_eq!(
                        read(&backend, &frame.output),
                        expected[2].1,
                        "deferred{deferred} boundary{boundary} action{action}"
                    );
                    drop(frame);
                    let next = session.submit(&plan, 3).unwrap().wait().unwrap();
                    assert_eq!(read(&backend, &next.output), expected[3].1);
                }
                drop(session);
                let memory = backend.transient_memory_budget();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while memory.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline
                {
                    backend.device().poll(wgpu::PollType::Poll).unwrap();
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                assert_eq!(
                    memory.snapshot().reserved_bytes,
                    0,
                    "deferred{deferred} boundary{boundary} action{action}"
                );
            }
        }
    }
}

#[test]
fn lf_extra_queued_render_blend_and_pack_stages_release_exact_reservations() {
    let backend = match pollster::block_on(WgpuBackend::request_default(Default::default())) {
        Ok(b) => b,
        Err(jxl_wgpu::Error::NoAdapter) => return,
        Err(e) => panic!("{e:?}"),
    };
    for name in ["nested_modular_gab1", "nested_vardct_gab1"] {
        for composed in [false, true] {
            let text = std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("test-data/lf_extra_channels")
                    .join(format!(
                        "{name}{}.jxl.hex",
                        if composed { ".composed" } else { "" }
                    )),
            )
            .unwrap()
            .split_whitespace()
            .collect::<String>();
            let data = text
                .as_bytes()
                .chunks_exact(2)
                .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
                .collect::<Vec<_>>();
            let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
            let source = Arc::new(
                GpuCodestream::from_shared(Arc::from(data.as_slice()), 0..data.len(), false)
                    .unwrap(),
            );
            let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            let request = GpuOutputRequest::numeric(
                jxl_gpu_formats::PixelFormat::non_color(
                    jxl_gpu_formats::SampleKind::Float,
                    32,
                    &[jxl_gpu_formats::Channel::X],
                ),
                crate::NumericSampleMapping::NormalizedUnsigned,
            )
            .unwrap()
            .with_extra_channel(1)
            .unwrap()
            .with_progressive_output(true);
            let mut baseline =
                DependentSession::new(engine.clone(), source.clone(), &inventory, &request, &plan)
                    .unwrap();
            let frame = baseline.submit(&plan, 0).unwrap().wait().unwrap();
            let expected = read(&backend, &frame.output);
            drop(frame);
            drop(baseline);
            for boundary in 0..4 {
                if !composed && (boundary == 0 || boundary == 3) {
                    continue;
                }
                for action in 0..4 {
                    let mut session = DependentSession::new(
                        engine.clone(),
                        source.clone(),
                        &inventory,
                        &request,
                        &plan,
                    )
                    .unwrap();
                    let mut pending = session.submit(&plan, 0).unwrap();
                    decode(&mut pending);
                    if composed {
                        decode(&mut pending);
                        assert_eq!(pending.lf_pending.len(), 2);
                        assert!(pending.carry.as_ref().unwrap().lf[1].is_none());
                        assert!(
                            pending
                                .lf_pending
                                .iter()
                                .all(|update| update.planes.extras.is_some())
                        );
                        if boundary > 0 {
                            decode(&mut pending);
                        }
                    }
                    let references = |pending: &DependentPending| {
                        pending
                            .carry
                            .as_ref()
                            .unwrap()
                            .references
                            .each_ref()
                            .map(|slot| {
                                slot.as_ref()
                                    .map(|surface| surface.buffer.as_wgpu_buffer().clone())
                            })
                    };
                    let before = references(&pending);
                    if boundary >= 2 {
                        let Some(Stage::LfPreview { work, progression }) = pending.stage.take()
                        else {
                            panic!("LF rendering")
                        };
                        let buffer = work.wait().unwrap();
                        let preview = pending.lf_preview.as_ref().unwrap();
                        let compositor = match &*pending.output {
                            Output::Composed(c) => Arc::clone(c),
                            Output::Native => Arc::clone(preview.compositor.as_ref().unwrap()),
                        };
                        let surface = compositor
                            .import(crate::frame_surface::outputs(
                                &preview.layout,
                                preview.surface.as_ref(),
                                &buffer,
                            ))
                            .unwrap();
                        if action == 3 && boundary == 2 {
                            let memory = backend.transient_memory_budget();
                            let blocker = memory
                                .try_reserve(memory.snapshot().available_bytes)
                                .unwrap();
                            assert!(
                                pending
                                    .render_refinement(
                                        compositor,
                                        surface,
                                        pending.end - 1,
                                        Resume::Advance,
                                        progression
                                    )
                                    .is_err()
                            );
                            drop(blocker);
                        } else {
                            pending
                                .render_refinement(
                                    compositor,
                                    surface,
                                    pending.end - 1,
                                    Resume::Advance,
                                    progression,
                                )
                                .unwrap();
                        }
                    }
                    if boundary == 3 {
                        let Some(Stage::Refinement {
                            compositor,
                            resume,
                            render: RefinementRender::Blend(work),
                            progression,
                        }) = pending.stage.take()
                        else {
                            panic!("LF blending")
                        };
                        let surface = compositor.completed_surface(work.wait().unwrap());
                        if action == 3 {
                            let memory = backend.transient_memory_budget();
                            let blocker = memory
                                .try_reserve(memory.snapshot().available_bytes)
                                .unwrap();
                            assert!(compositor.pack(&surface).is_err());
                            drop(blocker);
                        } else {
                            let work = compositor.pack(&surface).unwrap();
                            pending.stage = Some(Stage::Refinement {
                                compositor,
                                resume,
                                render: RefinementRender::Pack(work),
                                progression,
                            });
                            pending.submissions.fetch_add(1, Ordering::AcqRel);
                            pending.completed_submissions += 1;
                        }
                    }
                    assert_eq!(references(&pending), before);
                    assert!(matches!(
                        pending.unvalidated(),
                        Err(Error::UnvalidatedOutputNotSubmitted)
                    ));
                    if action == 0 || action == 3 {
                        drop(pending);
                    } else {
                        let frame = if action == 1 {
                            pending.wait().unwrap()
                        } else {
                            let frame =
                                pollster::block_on(std::future::poll_fn(|cx| pending.poll(cx)))
                                    .unwrap();
                            drop(pending);
                            frame
                        };
                        assert_eq!(read(&backend, &frame.output), expected);
                        drop(frame);
                    }
                    drop(session);
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                    while backend.transient_memory_budget().snapshot().reserved_bytes != 0
                        && std::time::Instant::now() < deadline
                    {
                        backend.device().poll(wgpu::PollType::Poll).unwrap();
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    assert_eq!(
                        backend.transient_memory_budget().snapshot().reserved_bytes,
                        0,
                        "{name} composed={composed} boundary={boundary} action={action}"
                    );
                }
            }
        }
    }
}
