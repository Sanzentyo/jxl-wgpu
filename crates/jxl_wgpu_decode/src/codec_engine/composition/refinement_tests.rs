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
fn next_intermediate(pending: &mut DependentPending) -> (GpuImageFrame, FrameProgression) {
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
        panic!("expected initial physical-frame update")
    };
    assert_eq!(progression.completed_passes(), Some(0));
    (frame.output, progression)
}

fn require_allocation_failure<T>(backend: &WgpuBackend, operation: impl FnOnce() -> Result<T>) {
    let memory = backend.transient_memory_budget();
    let blocker = memory
        .try_reserve(memory.snapshot().available_bytes)
        .unwrap();
    let error = match operation() {
        Ok(_) => panic!("composition allocated with no available budget"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::MemoryBackpressure(_)), "{error:?}");
    drop(blocker);
}

fn exercise_pending(
    mut pending: DependentPending,
    backend: &WgpuBackend,
    boundary: usize,
    action: usize,
    expected: &(FrameMetadata, Vec<u8>),
) {
    let (frame, progression) = next_intermediate(&mut pending);
    let before = references(&pending);
    if action == 3 && boundary != 2 {
        // Stop after producer validation but before any composition allocation. A public update
        // may legitimately free producer scratch and use those bytes for its following snapshot.
        require_allocation_failure(backend, || pending.refine(frame, progression));
        assert_eq!(references(&pending), before);
        return;
    }
    pending.refine(frame, progression).unwrap();
    assert_eq!(references(&pending), before);
    assert!(
        matches!(&pending.stage,
        Some(Stage::Refinement { render: RefinementRender::Pack(_), .. }) if boundary == 0)
            || matches!(&pending.stage, Some(Stage::Refinement { render: RefinementRender::Blend(_), .. }) if boundary != 0)
    );
    if boundary == 2 {
        let Some(Stage::Refinement {
            compositor,
            resume,
            render: RefinementRender::Blend(work),
            progression,
        }) = pending.stage.take()
        else {
            unreachable!()
        };
        let surface = compositor.completed_surface(work.wait().unwrap());
        if action == 3 {
            require_allocation_failure(backend, || compositor.pack(&surface));
            assert_eq!(references(&pending), before);
            return;
        }
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
        return;
    }
    let frame = if action == 1 {
        pending.wait().unwrap()
    } else {
        let frame =
            pollster::block_on(std::future::poll_fn(|context| pending.poll(context))).unwrap();
        drop(pending);
        frame
    };
    assert_eq!(frame.metadata, expected.0);
    assert_eq!(bytes(&frame.output, backend), expected.1);
}

#[test]
fn submitted_composition_refinements_preserve_references_cancel_and_drain_final_only() {
    let backend = match pollster::block_on(WgpuBackend::request_default(Default::default())) {
        Ok(backend) => backend,
        Err(jxl_wgpu::Error::NoAdapter) => return,
        Err(error) => panic!("{error:?}"),
    };
    for (hex, numeric) in [
        (
            include_str!("../../../test-data/composition_vardct.jxl.hex"),
            false,
        ),
        (
            include_str!("../../../test-data/modular_composition/modular_pass_rgb.jxl.hex"),
            false,
        ),
        (
            include_str!("../../../test-data/modular_composition/modular_pass_gray_alpha.jxl.hex"),
            true,
        ),
        (
            include_str!("../../../test-data/modular_composition/modular_pass_float.jxl.hex"),
            true,
        ),
    ] {
        exercise_submitted_refinements(&backend, hex, numeric);
    }
}

fn exercise_submitted_refinements(backend: &WgpuBackend, hex: &str, numeric: bool) {
    let compact = hex.split_whitespace().collect::<String>();
    let encoded: Vec<u8> = compact
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    let parsed = jxl_gpu_bitstream::parse(&encoded, Default::default()).unwrap();
    let data: Arc<[u8]> = Arc::from(parsed.codestream());
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
    let request = if numeric {
        GpuOutputRequest::numeric(
            jxl_gpu_formats::vpi::VpiPitchLinearFormat::F32.pixel_format(),
            match inventory.image_header.extra_channels[0].bit_depth {
                jxl_gpu_bitstream::SampleBitDepth::Integer { .. } => {
                    crate::NumericSampleMapping::NormalizedUnsigned
                }
                jxl_gpu_bitstream::SampleBitDepth::Float { .. } => {
                    crate::NumericSampleMapping::NativeFloat
                }
            },
        )
        .unwrap()
        .with_extra_channel(0)
        .unwrap()
    } else {
        GpuOutputRequest::color(crate::vardct_rgb8_format()).unwrap()
    }
    .with_progressive_output(true);
    let source = Arc::new(GpuCodestream::from_shared(data.clone(), 0..data.len(), false).unwrap());
    let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    let mut baseline =
        DependentSession::new(engine.clone(), source.clone(), &inventory, &request, &plan).unwrap();
    let expected: Vec<_> = (0..2)
        .map(|i| {
            let frame = baseline.submit(&plan, i).unwrap().wait().unwrap();
            (frame.metadata, bytes(&frame.output, backend))
        })
        .collect();
    drop(baseline);
    for boundary in 0..3 {
        let presentation = usize::from(boundary != 0);
        for action in 0..4 {
            let mut session =
                DependentSession::new(engine.clone(), source.clone(), &inventory, &request, &plan)
                    .unwrap();
            for i in 0..presentation {
                drop(session.submit(&plan, i).unwrap().wait().unwrap());
            }
            exercise_pending(
                session.submit(&plan, presentation).unwrap(),
                backend,
                boundary,
                action,
                &expected[presentation],
            );
            if matches!(action, 0 | 3) {
                assert!(matches!(
                    session.submit(&plan, presentation),
                    Err(Error::SessionPoisoned)
                ));
            }
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
