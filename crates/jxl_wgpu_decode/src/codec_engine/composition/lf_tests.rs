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
        decode.lf = Some(lf_planes(&decode.pending).unwrap());
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
                        .render_refinement(surface, pending.end - 1, Resume::Advance, progression)
                        .unwrap();
                }
                if boundary == 3 {
                    let Some(Stage::Refinement {
                        resume,
                        render: RefinementRender::Blend(work),
                        progression,
                    }) = pending.stage.take()
                    else {
                        panic!("LF blend")
                    };
                    let compositor = pending.output.compositor().unwrap();
                    let surface = compositor.completed_surface(work.wait().unwrap());
                    pending.stage = Some(Stage::Refinement {
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
