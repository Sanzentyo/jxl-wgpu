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

pub(super) fn references(pending: &DependentPending) -> [Option<wgpu::Buffer>; 4] {
    std::array::from_fn(|i| {
        pending.carry.as_ref().unwrap().references[i]
            .as_ref()
            .map(|surface| surface.buffer.as_wgpu_buffer().clone())
    })
}

// Drive real hidden producers and stop at the exact submitted refinement boundary. This avoids
// relying on a poll happening to catch a short GPU blend or pack before its callback completes.
fn next_intermediate(pending: &mut DependentPending) -> (GpuImageFrame, FrameProgression) {
    while pending.physical + 1 != pending.end
        || matches!(pending.stage, Some(Stage::PatchDictionary(_)))
    {
        match pending.stage.take().unwrap() {
            Stage::PatchDictionary(parser) => {
                let (dictionary, count) = parser.wait().unwrap();
                pending.dictionary_decoded(dictionary, count).unwrap();
            }
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

pub(super) fn require_allocation_failure<T>(
    backend: &WgpuBackend,
    operation: impl FnOnce() -> Result<T>,
) {
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

#[test]
fn encoded_refinements_transform_for_display_without_committing_references() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let compact = include_str!("../../../test-data/vardct_extras_rgba_progressive.jxl.hex")
        .split_whitespace()
        .collect::<String>();
    let encoded: Vec<_> = compact
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    let parsed = jxl_gpu_bitstream::parse(&encoded, Default::default()).unwrap();
    let data: Arc<[u8]> = Arc::from(parsed.codestream());
    let source = Arc::new(GpuCodestream::from_shared(data.clone(), 0..data.len(), false).unwrap());
    let mut inventory = parsed.codestream_inventory(Default::default()).unwrap();
    assert_eq!(inventory.frames.len(), 1);
    let mut plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
    // Exercise the private producer contract for a displayable pre-transform reference. The
    // original one-frame entropy is unchanged; these metadata-only flags choose its boundary.
    plan.nodes[0].needs_composition = true;
    plan.nodes[0].save_reference = Some(3);
    let request = GpuOutputRequest::color(jxl_gpu_formats::PixelFormat::rgb_f32(
        jxl_gpu_formats::RgbChannelOrder::Rgba,
        false,
        crate::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_progressive_output(true);
    let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    let mut baseline =
        DependentSession::new(engine.clone(), source.clone(), &inventory, &request, &plan).unwrap();
    let mut baseline_pending = baseline.submit(&plan, 0).unwrap();
    let (frame, progression) = next_intermediate(&mut baseline_pending);
    baseline_pending.refine(frame, progression).unwrap();
    let update = pollster::block_on(std::future::poll_fn(|cx| {
        baseline_pending.poll_update(cx, true)
    }))
    .unwrap();
    let SubmittedGpuUpdate::Intermediate { frame, .. } = update else {
        panic!("expected CID");
    };
    let expected_update = bytes(&frame.output, &backend);
    drop(frame);
    let frame = baseline_pending.wait().unwrap();
    let expected_final = bytes(&frame.output, &backend);
    drop((frame, baseline));
    inventory.frames[0].save_before_color_transform = true;
    for action in 0..3 {
        let mut session =
            DependentSession::new(engine.clone(), source.clone(), &inventory, &request, &plan)
                .unwrap();
        let mut pending = session.submit(&plan, 0).unwrap();
        let (frame, progression) = next_intermediate(&mut pending);
        pending.refine(frame, progression).unwrap();
        assert!(matches!(
            pending.stage,
            Some(Stage::Refinement {
                render: RefinementRender::Transform { .. },
                ..
            })
        ));
        assert!(references(&pending).iter().all(Option::is_none));
        if action == 0 {
            drop(pending);
        } else {
            if action == 1 {
                let update =
                    pollster::block_on(std::future::poll_fn(|cx| pending.poll_update(cx, true)))
                        .unwrap();
                let SubmittedGpuUpdate::Intermediate { frame, .. } = update else {
                    panic!("expected transformed CID");
                };
                assert_float_bytes(&bytes(&frame.output, &backend), &expected_update);
                assert!(references(&pending).iter().all(Option::is_none));
            }
            let frame = pending.wait().unwrap();
            assert_float_bytes(&bytes(&frame.output, &backend), &expected_final);
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
            0
        );
    }
}

fn assert_float_bytes(actual: &[u8], expected: &[u8]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
        let a = f32::from_le_bytes(a.try_into().unwrap());
        let b = f32::from_le_bytes(b.try_into().unwrap());
        assert!(
            a.is_finite() && (a - b).abs() < 1e-5 * (1.0 + b.abs()),
            "{a} vs {b}"
        );
    }
}

pub(super) fn step_component_refinement(
    pending: &mut DependentPending,
    backend: &WgpuBackend,
    fail: bool,
) {
    let Some(Stage::Refinement {
        compositor,
        resume,
        render,
        progression,
    }) = pending.stage.take()
    else {
        panic!("expected component refinement");
    };
    let (RefinementRender::Patches {
        work,
        mut source,
        index,
    }
    | RefinementRender::Transform {
        work,
        mut source,
        index,
    }) = render
    else {
        panic!("expected patch or transform work");
    };
    source.buffer = work.wait().unwrap();
    let next = || pending.render_refinement(compositor, source, index, resume, progression);
    if fail {
        require_allocation_failure(backend, next);
    } else {
        next().unwrap();
    }
}

#[test]
fn patch_refinements_preserve_dictionary_and_references_across_cancel_drain_and_admission() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let compact = include_str!("../../../test-data/patches/progressive/vardct.jxl.hex")
        .split_whitespace()
        .collect::<String>();
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
    let source = Arc::new(GpuCodestream::from_shared(data.clone(), 0..data.len(), false).unwrap());
    let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    let request = GpuOutputRequest::color(jxl_gpu_formats::PixelFormat::rgb_f32(
        jxl_gpu_formats::RgbChannelOrder::Rgba,
        false,
        crate::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_progressive_output(true);
    let session = || {
        DependentSession::new(engine.clone(), source.clone(), &inventory, &request, &plan).unwrap()
    };
    let mut baseline = session();
    let mut pending = baseline.submit(&plan, 0).unwrap();
    let update =
        pollster::block_on(std::future::poll_fn(|cx| pending.poll_update(cx, true))).unwrap();
    let SubmittedGpuUpdate::Intermediate { frame, .. } = update else {
        panic!("expected CID")
    };
    let expected_update = bytes(&frame.output, &backend);
    drop(frame);
    let frame = pending.wait().unwrap();
    let expected = (frame.metadata.clone(), bytes(&frame.output, &backend));
    drop((frame, baseline));
    for boundary in 0..3 {
        for action in 0..5 {
            let mut session = session();
            let mut pending = session.submit(&plan, 0).unwrap();
            let (frame, progression) = next_intermediate(&mut pending);
            let references_before = references(&pending);
            let dictionary_before = pending
                .patches
                .as_ref()
                .unwrap()
                .commands
                .as_wgpu_buffer()
                .clone();
            assert_eq!(references_before.iter().filter(|r| r.is_some()).count(), 1);
            if action == 4 && boundary == 0 {
                require_allocation_failure(&backend, || pending.refine(frame, progression));
            } else {
                pending.refine(frame, progression).unwrap();
                assert!(matches!(
                    &pending.stage,
                    Some(Stage::Refinement {
                        render: RefinementRender::Patches { .. },
                        ..
                    })
                ));
                for step in 0..boundary {
                    step_component_refinement(
                        &mut pending,
                        &backend,
                        action == 4 && step + 1 == boundary,
                    );
                }
            }
            assert_eq!(references(&pending), references_before);
            assert_eq!(
                pending.patches.as_ref().unwrap().commands.as_wgpu_buffer(),
                &dictionary_before
            );
            assert!(matches!(
                pending.unvalidated(),
                Err(Error::UnvalidatedOutputNotSubmitted)
            ));
            if action == 0 || action == 4 {
                drop(pending);
                assert!(matches!(
                    session.submit(&plan, 0),
                    Err(Error::SessionPoisoned)
                ));
            } else {
                if action == 3 {
                    let update = pollster::block_on(std::future::poll_fn(|cx| {
                        pending.poll_update(cx, true)
                    }))
                    .unwrap();
                    let SubmittedGpuUpdate::Intermediate { frame, .. } = update else {
                        panic!("expected patched CID")
                    };
                    assert_eq!(bytes(&frame.output, &backend), expected_update);
                    assert_eq!(references(&pending), references_before);
                    assert_eq!(
                        pending.patches.as_ref().unwrap().commands.as_wgpu_buffer(),
                        &dictionary_before
                    );
                }
                let frame = if action == 2 {
                    pollster::block_on(std::future::poll_fn(|cx| pending.poll(cx))).unwrap()
                } else {
                    pending.wait().unwrap()
                };
                assert_eq!(frame.metadata, expected.0);
                assert_eq!(bytes(&frame.output, &backend), expected.1);
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
                "boundary {boundary}, action {action}"
            );
        }
    }
}
