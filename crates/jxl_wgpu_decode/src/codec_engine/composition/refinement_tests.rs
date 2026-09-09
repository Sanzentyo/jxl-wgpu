use super::*;
use jxl_wgpu::{ImageReadbackPipeline, WgpuBackend};

fn bytes(frame: &GpuImageFrame, backend: &WgpuBackend) -> Vec<u8> {
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

fn references(pending: &DependentPending) -> [Option<wgpu::Buffer>; 4] {
    std::array::from_fn(|i| {
        pending.carry.as_ref().unwrap().references[i]
            .as_ref()
            .map(|surface| surface.buffer.as_wgpu_buffer().clone())
    })
}

// Drive real hidden producers and stop at the exact submitted refinement boundary. This avoids
// relying on a poll happening to catch a short GPU blend or pack before its callback completes.
fn refine(pending: &mut DependentPending) {
    while pending.physical + 1 != pending.end {
        match pending.stage.take().unwrap() {
            Stage::Decode(PhysicalPending {
                pending: producer,
                count,
                lf,
            }) => {
                assert!(lf.is_none());
                let frame = producer.wait().unwrap();
                assert!(pending.decoded(frame, &count, lf, false).unwrap().is_none());
            }
            Stage::Blend(work) => {
                pending
                    .record(
                        pending
                            .output
                            .compositor()
                            .unwrap()
                            .completed_surface(work.wait().unwrap()),
                    )
                    .unwrap();
            }
            stage => panic!("unexpected hidden producer stage: {stage:?}"),
        }
    }
    let Some(Stage::Decode(decode)) = pending.stage.as_mut() else {
        unreachable!()
    };
    let update = pollster::block_on(std::future::poll_fn(|context| {
        Pin::new(decode.pending.as_mut()).poll_next_update(context)
    }))
    .unwrap();
    update_count(
        pending.completed_submissions,
        &decode.count,
        &pending.submissions,
    )
    .unwrap();
    let SubmittedGpuUpdate::Intermediate { frame, progression } = update else {
        panic!("expected DC update")
    };
    assert!(matches!(
        progression,
        FrameProgression::Coefficients {
            completed_passes: 0,
            ..
        }
    ));
    let before = references(pending);
    pending.refine(frame.output, progression).unwrap();
    assert_eq!(references(pending), before);
}

#[test]
fn submitted_composition_refinements_preserve_references_cancel_and_drain_final_only() {
    let backend = match pollster::block_on(WgpuBackend::request_default(Default::default())) {
        Ok(backend) => backend,
        Err(jxl_wgpu::Error::NoAdapter) => return,
        Err(error) => panic!("{error:?}"),
    };
    let hex = include_str!("../../../test-data/composition_vardct.jxl.hex");
    let compact = hex.split_whitespace().collect::<String>();
    let data: Arc<[u8]> = compact
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect::<Vec<_>>()
        .into();
    let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
    let request = GpuOutputRequest::color(crate::vardct_rgb8_format())
        .unwrap()
        .with_progressive_output(true);
    let source = Arc::new(GpuCodestream::from_shared(data.clone(), 0..data.len(), false).unwrap());
    let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    let mut baseline =
        DependentSession::new(engine.clone(), source.clone(), &inventory, &request, &plan).unwrap();
    let expected: Vec<_> = (0..2)
        .map(|i| {
            let frame = baseline.submit(&plan, i).unwrap().wait().unwrap();
            (frame.metadata, bytes(&frame.output, &backend))
        })
        .collect();
    drop(baseline);
    for boundary in 0..3 {
        let presentation = usize::from(boundary != 0);
        for action in 0..3 {
            let mut session =
                DependentSession::new(engine.clone(), source.clone(), &inventory, &request, &plan)
                    .unwrap();
            for i in 0..presentation {
                drop(session.submit(&plan, i).unwrap().wait().unwrap());
            }
            let mut pending = session.submit(&plan, presentation).unwrap();
            refine(&mut pending);
            assert!(
                matches!(&pending.stage,
                Some(Stage::Refinement { render: RefinementRender::Pack(_), .. }) if boundary == 0)
                    || matches!(&pending.stage, Some(Stage::Refinement { render: RefinementRender::Blend(_), .. }) if boundary != 0)
            );
            if boundary == 2 {
                let Some(Stage::Refinement {
                    resume,
                    render: RefinementRender::Blend(work),
                    progression,
                }) = pending.stage.take()
                else {
                    unreachable!()
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
            let before = references(&pending);
            // An intermediate pack is intentionally absent from final-output escape hatches.
            assert!(matches!(
                pending.unvalidated(),
                Err(Error::UnvalidatedOutputNotSubmitted)
            ));
            if action == 0 {
                drop(pending);
                assert!(matches!(
                    session.submit(&plan, presentation),
                    Err(Error::SessionPoisoned)
                ));
            } else {
                let frame = if action == 1 {
                    pending.wait().unwrap()
                } else {
                    let frame =
                        pollster::block_on(std::future::poll_fn(|context| pending.poll(context)))
                            .unwrap();
                    drop(pending);
                    frame
                };
                assert_eq!(frame.metadata, expected[presentation].0);
                assert_eq!(bytes(&frame.output, &backend), expected[presentation].1);
            }
            drop(before);
            drop(session);
            let memory = backend.transient_memory_budget();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while memory.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline {
                backend.device().poll(wgpu::PollType::Poll).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert_eq!(
                memory.snapshot().reserved_bytes,
                0,
                "boundary{boundary} action{action}"
            );
        }
    }
}
