use super::super::refinement_tests::{
    references, require_allocation_failure, step_component_refinement,
};
use super::super::test_support::{drain, fixture, read};
use super::super::*;
use jxl_wgpu::WgpuBackend;

fn open(
    backend: &WgpuBackend,
    name: &str,
    bounded: bool,
    progressive: bool,
) -> (DependentSession, DependentPending) {
    let data = fixture(&format!("patches/features/{name}"));
    let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
    let source = Arc::new(GpuCodestream::from_shared(data.clone(), 0..data.len(), false).unwrap());
    let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    if bounded {
        engine = engine.with_stream_window_limit(std::num::NonZeroU64::new(256).unwrap());
    }
    let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        crate::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_alpha_output_policy(crate::AlphaOutputPolicy::Preserve)
    .with_progressive_output(progressive);
    let mut session = DependentSession::new(engine, source, &inventory, &request, &plan).unwrap();
    let pending = session.submit(&plan, 0).unwrap();
    (session, pending)
}

// Wait at explicit submission boundaries rather than racing the device's completion callback.
fn step(pending: &mut DependentPending, emit: bool) {
    match pending.stage.take().unwrap() {
        Stage::PatchDictionary(parser) => {
            let (dictionary, count) = parser.wait().unwrap();
            pending.dictionary_decoded(dictionary, count).unwrap();
        }
        Stage::Decode(mut decode) => {
            pending.features = decode.features.map(|plan| (pending.physical, plan));
            if pending.needs_lf_output() && decode.lf.is_none() {
                if let WgpuDecodePendingFrame::VarDct(producer) = decode.pending.as_mut() {
                    producer.wait_until_dependency_submitted().unwrap();
                }
                decode.lf = Some(lf_output(&decode.pending).unwrap());
            }
            let frame = decode.pending.wait().unwrap();
            assert!(
                pending
                    .decoded(frame, &decode.count, decode.lf, emit)
                    .unwrap()
                    .is_none()
            );
        }
        Stage::PatchRender { work, mut source } => {
            source.buffer = work.wait().unwrap();
            pending.reconstructed(source).unwrap();
        }
        Stage::Features(work) => pending.feature_complete(work.wait().unwrap()).unwrap(),
        Stage::LfPatchRender(work) => pending
            .complete_lf_features(Some(work.wait().unwrap()), emit)
            .unwrap(),
        Stage::LfFeatures(work) => pending.record_lf(Some(work.wait().unwrap()), emit).unwrap(),
        Stage::ColorTransform { work, mut source } => {
            source.buffer = work.wait().unwrap();
            pending.transformed(source).unwrap();
        }
        Stage::Blend(work) => pending
            .record(
                pending
                    .output
                    .compositor()
                    .unwrap()
                    .completed_surface(work.wait().unwrap()),
            )
            .unwrap(),
        Stage::Advance => pending.advance().unwrap(),
        stage => panic!("unexpected feature test stage: {stage:?}"),
    }
}

fn before_features(pending: &mut DependentPending) {
    for _ in 0..64 {
        if matches!(
            pending.stage,
            Some(Stage::PatchRender { .. } | Stage::LfPatchRender(_))
        ) {
            return;
        }
        step(pending, false);
    }
    panic!("missing patch submission");
}

fn finish_patch(pending: &mut DependentPending, backend: &WgpuBackend, fail: bool) {
    match pending.stage.take().unwrap() {
        Stage::PatchRender { work, mut source } => {
            source.buffer = work.wait().unwrap();
            let next = || pending.reconstructed(source);
            if fail {
                require_allocation_failure(backend, next);
            } else {
                next().unwrap();
            }
        }
        Stage::LfPatchRender(work) => {
            let output = work.wait().unwrap();
            let next = || pending.complete_lf_features(Some(output), false);
            if fail {
                require_allocation_failure(backend, next);
            } else {
                next().unwrap();
            }
        }
        _ => panic!("expected patched components"),
    }
}

#[test]
fn submitted_frame_and_lf_features_cancel_fail_or_drain_without_publishing_references() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for name in [
        "modular_extras_up2_noise",
        "vardct_extras_up8_noise",
        "lf_modular_up2_noise",
        "lf_vardct_up8_noise",
    ] {
        for bounded in [false, true] {
            let (session, pending) = open(&backend, name, bounded, false);
            let frame = pending.wait().unwrap();
            let expected = read(&backend, &frame.output);
            drop((session, frame));
            drain(&backend);
            for action in 0..4 {
                let (session, mut pending) = open(&backend, name, bounded, false);
                before_features(&mut pending);
                let before = references(&pending);
                assert_eq!(before.iter().flatten().count(), 1);
                let lf = pending
                    .carry
                    .as_ref()
                    .unwrap()
                    .lf
                    .iter()
                    .map(|slot| slot.as_ref().map(|frame| frame.frame_index))
                    .collect::<Vec<_>>();
                finish_patch(&mut pending, &backend, action == 3);
                assert_eq!(references(&pending), before);
                assert_eq!(
                    pending
                        .carry
                        .as_ref()
                        .unwrap()
                        .lf
                        .iter()
                        .map(|slot| slot.as_ref().map(|frame| frame.frame_index))
                        .collect::<Vec<_>>(),
                    lf
                );
                if action != 3 {
                    assert!(matches!(
                        pending.stage,
                        Some(Stage::Features(_) | Stage::LfFeatures(_))
                    ));
                }
                assert!(matches!(
                    pending.unvalidated(),
                    Err(Error::UnvalidatedOutputNotSubmitted)
                ));
                match action {
                    0 | 3 => drop(pending),
                    1 => {
                        let frame = pending.wait().unwrap();
                        assert_eq!(read(&backend, &frame.output), expected);
                    }
                    2 => {
                        let frame = pollster::block_on(std::future::poll_fn(|cx| pending.poll(cx)))
                            .unwrap();
                        assert_eq!(read(&backend, &frame.output), expected);
                        drop(pending);
                    }
                    _ => unreachable!(),
                }
                if matches!(action, 0 | 3) {
                    assert!(lock(&session.shared).failed);
                }
                drop(session);
                drain(&backend);
            }
        }
    }
}

#[test]
fn lf_feature_completion_retains_exact_prediction_bytes_and_releases_extra_scratch() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for name in [
        "lf_modular_up2_noise",
        "lf_vardct_up4_noise",
        "lf_modular_noise",
    ] {
        let (session, mut pending) = open(&backend, name, true, false);
        before_features(&mut pending);
        let before = references(&pending);
        finish_patch(&mut pending, &backend, false);
        let Some(Stage::LfFeatures(work)) = pending.stage.take() else {
            panic!("LF features");
        };
        let output = work.wait().unwrap();
        assert!(output.extras.is_none());
        let prediction = u64::from(output.xyb.width()) * u64::from(output.xyb.height()) * 12;
        assert_eq!(
            output
                .xyb
                .planes
                .iter()
                .map(|plane| plane.buffer.size())
                .sum::<u64>(),
            prediction
        );
        let carry = pending.carry.as_ref().unwrap();
        assert!(carry.lf.iter().all(Option::is_none));
        let reference = carry
            .references
            .iter()
            .flatten()
            .map(|surface| surface.buffer.size())
            .sum::<u64>();
        assert_eq!(
            backend.transient_memory_budget().snapshot().reserved_bytes,
            reference + prediction
        );
        assert_eq!(references(&pending), before);
        pending.record_lf(Some(output), false).unwrap();
        drop((pending.wait().unwrap(), session));
        drain(&backend);
    }
}

fn begin_feature_refinement(pending: &mut DependentPending) {
    for _ in 0..64 {
        if pending.physical + 1 == pending.end
            && matches!(pending.stage, Some(Stage::Decode(_) | Stage::LfPreviews(_)))
        {
            break;
        }
        step(pending, true);
    }
    match pending.stage.take().unwrap() {
        Stage::Decode(mut decode) => {
            pending.features = decode.features.clone().map(|plan| (pending.physical, plan));
            let update = pollster::block_on(std::future::poll_fn(|cx| {
                Pin::new(decode.pending.as_mut()).poll_next_update(cx)
            }))
            .unwrap();
            update_count(
                pending.completed_submissions,
                &decode.count,
                &pending.submissions,
            )
            .unwrap();
            pending.stage = Some(Stage::Decode(decode));
            let SubmittedGpuUpdate::Intermediate { frame, progression } = update else {
                panic!("CID");
            };
            pending.refine(frame.output, progression).unwrap();
        }
        Stage::LfPreviews(decode) => {
            pending.features = decode.features.clone().map(|plan| (pending.physical, plan));
            pending
                .start_lf_preview(Resume::LfPreviews(decode))
                .unwrap();
            let Some(Stage::LfPreview {
                work,
                progression,
                resume,
            }) = pending.stage.take()
            else {
                panic!("LF preview");
            };
            let buffer = work.wait().unwrap();
            let preview = pending.lf_preview.as_ref().unwrap();
            let compositor = Arc::clone(pending.output.compositor().unwrap());
            let surface = compositor
                .import_with_encoding(
                    crate::frame_surface::outputs(
                        &preview.layout,
                        preview.surface.as_ref(),
                        &buffer,
                    ),
                    preview.surface_encoding,
                )
                .unwrap();
            pending
                .begin_refinement(compositor, surface, pending.end - 1, resume, progression)
                .unwrap();
        }
        _ => panic!("terminal producer"),
    }
    assert!(matches!(
        pending.stage,
        Some(Stage::Refinement {
            render: RefinementRender::Patches { .. },
            ..
        })
    ));
}

#[test]
fn cid_and_lf_preview_features_preserve_held_images_and_reference_versions() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for name in [
        "vardct_xyb_up2",
        "lf_consumer_modular_noise",
        "lf_consumer_ac_noise",
    ] {
        let (session, mut pending) = open(&backend, name, true, true);
        let SubmittedGpuUpdate::Intermediate { frame, .. } =
            pollster::block_on(std::future::poll_fn(|cx| pending.poll_update(cx, true))).unwrap()
        else {
            panic!("first update");
        };
        let expected_update = read(&backend, &frame.output);
        drop(frame);
        let frame = pending.wait().unwrap();
        let expected_final = read(&backend, &frame.output);
        drop((frame, session));
        drain(&backend);
        for action in 0..5 {
            let (session, mut pending) = open(&backend, name, true, true);
            begin_feature_refinement(&mut pending);
            let before = references(&pending);
            let commands = pending
                .patches
                .as_ref()
                .unwrap()
                .commands
                .as_wgpu_buffer()
                .clone();
            step_component_refinement(&mut pending, &backend, action == 4);
            assert_eq!(references(&pending), before);
            assert_eq!(
                pending.patches.as_ref().unwrap().commands.as_wgpu_buffer(),
                &commands
            );
            if action != 4 {
                assert!(matches!(
                    pending.stage,
                    Some(Stage::Refinement {
                        render: RefinementRender::Features { .. },
                        ..
                    })
                ));
            }
            if matches!(action, 0 | 4) {
                drop(pending);
                assert!(lock(&session.shared).failed);
            } else {
                let held = if action == 3 {
                    let SubmittedGpuUpdate::Intermediate { frame, .. } =
                        pollster::block_on(std::future::poll_fn(|cx| {
                            pending.poll_update(cx, true)
                        }))
                        .unwrap()
                    else {
                        panic!("feature update");
                    };
                    assert_eq!(read(&backend, &frame.output), expected_update);
                    assert_eq!(references(&pending), before);
                    Some(frame)
                } else {
                    None
                };
                let frame = if action == 2 {
                    let frame =
                        pollster::block_on(std::future::poll_fn(|cx| pending.poll(cx))).unwrap();
                    drop(pending);
                    frame
                } else {
                    pending.wait().unwrap()
                };
                assert_eq!(read(&backend, &frame.output), expected_final);
                if let Some(held) = held {
                    assert_eq!(read(&backend, &held.output), expected_update);
                }
            }
            drop(session);
            drain(&backend);
        }
    }
}
