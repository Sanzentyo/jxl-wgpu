use super::test_support::{drain, fixture, read};
use super::*;
use jxl_wgpu::WgpuBackend;

fn backend() -> WgpuBackend {
    pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap()
}

fn open(backend: &WgpuBackend, name: &str) -> (DependentSession, FrameExecutionPlan) {
    let bytes = fixture(&format!("splines/features/{name}"));
    let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
    let source =
        Arc::new(GpuCodestream::from_shared(bytes.clone(), 0..bytes.len(), false).unwrap());
    let engine = WgpuDecodeEngine::new(backend.clone())
        .unwrap()
        .with_stream_window_limit(std::num::NonZeroU64::new(40).unwrap());
    let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        crate::vardct_rgb8_format().color_spec,
    ))
    .unwrap();
    (
        DependentSession::new(engine, source, &inventory, &request, &plan).unwrap(),
        plan,
    )
}

fn step(pending: &mut DependentPending) {
    match pending.stage.take().unwrap() {
        Stage::PatchDictionary(parser) => {
            let (dictionary, count) = parser.wait().unwrap();
            pending.dictionary_decoded(dictionary, count).unwrap();
        }
        Stage::SplineEntropy(parser) => {
            let (program, count) = parser.wait().unwrap();
            pending.splines_decoded(program, count).unwrap();
        }
        Stage::SplineGeometry {
            pending: geometry,
            session,
        } => {
            let (cache, count) = geometry.wait().unwrap();
            pending
                .spline_geometry_completed(cache, count, *session)
                .unwrap();
        }
        Stage::Decode(decode) => {
            pending.features = decode.features.map(|plan| (pending.physical, plan));
            let frame = decode.pending.wait().unwrap();
            assert!(
                pending
                    .decoded(frame, &decode.count, decode.lf, false)
                    .unwrap()
                    .is_none()
            );
        }
        Stage::PatchRender { work, mut source } => {
            source.buffer = work.wait().unwrap();
            pending.reconstructed(source).unwrap();
        }
        Stage::Features(work) => pending.feature_complete(work.wait().unwrap()).unwrap(),
        Stage::ColorTransform { work, mut source } => {
            source.buffer = work.wait().unwrap();
            pending.transformed(source).unwrap();
        }
        Stage::Advance => pending.advance().unwrap(),
        stage => panic!("unexpected spline boundary: {stage:?}"),
    }
}

#[derive(Clone, Copy, Debug)]
enum Boundary {
    Entropy,
    Geometry,
    Raster,
}

fn reached(pending: &DependentPending, boundary: Boundary, physical: usize) -> bool {
    pending.physical == physical
        && matches!(
            (boundary, &pending.stage),
            (Boundary::Entropy, Some(Stage::SplineEntropy(_)))
                | (Boundary::Geometry, Some(Stage::SplineGeometry { .. }))
                | (Boundary::Raster, Some(Stage::Features(_)))
        )
}

#[test]
fn spline_prefix_geometry_and_raster_cancellation_preserve_reference_versions() {
    let backend = backend();
    let (mut session, plan) = open(&backend, "modular_rgb_up2_chain");
    let output = session.submit(&plan, 0).unwrap().wait().unwrap();
    let expected = read(&backend, &output.output);
    drop((output, session));
    drain(&backend);
    for physical in [0, 1, 2] {
        for boundary in [Boundary::Entropy, Boundary::Geometry, Boundary::Raster] {
            for finish in [false, true] {
                let (mut session, plan) = open(&backend, "modular_rgb_up2_chain");
                let mut pending = session.submit(&plan, 0).unwrap();
                for _ in 0..64 {
                    if reached(&pending, boundary, physical) {
                        break;
                    }
                    step(&mut pending);
                }
                assert!(reached(&pending, boundary, physical));
                let references = super::refinement_tests::references(&pending);
                assert_eq!(
                    references.iter().filter(|slot| slot.is_some()).count(),
                    usize::from(physical != 0)
                );
                assert!(matches!(
                    pending.unvalidated(),
                    Err(Error::UnvalidatedOutputNotSubmitted)
                ));
                if finish {
                    let frame = pending.wait().unwrap();
                    assert_eq!(read(&backend, &frame.output), expected);
                    drop(frame);
                } else {
                    drop(pending);
                }
                drop((references, session));
                drain(&backend);
            }
        }
    }
}

#[test]
fn spline_initial_admission_retries_and_geometry_to_body_failure_is_terminal() {
    let backend = backend();
    let (mut session, plan) = open(&backend, "modular_rgb_up2_chain");
    let budget = backend.transient_memory_budget();
    let blocker = budget
        .try_reserve(budget.snapshot().available_bytes)
        .unwrap();
    assert!(matches!(
        session.submit(&plan, 0),
        Err(Error::MemoryBackpressure(_))
    ));
    drop(blocker);
    let mut pending = session.submit(&plan, 0).unwrap();
    step(&mut pending);
    let Some(Stage::SplineGeometry {
        pending: geometry,
        session: producer,
    }) = pending.stage.take()
    else {
        panic!("geometry boundary");
    };
    let (cache, count) = geometry.wait().unwrap();
    let references = super::refinement_tests::references(&pending);
    super::refinement_tests::require_allocation_failure(&backend, || {
        pending.spline_geometry_completed(cache, count, *producer)
    });
    assert_eq!(super::refinement_tests::references(&pending), references);
    drop(pending);
    assert!(matches!(
        session.submit(&plan, 0),
        Err(Error::SessionPoisoned)
    ));
    drop((references, session));
    drain(&backend);
}

#[test]
fn unequal_resampling_keeps_plane_extents_and_ownership_across_feature_admission() {
    let backend = backend();
    for mode in ["modular", "vardct"] {
        for role in ["frame", "lf"] {
            let name = format!("{mode}_floating_resampled_{role}");
            let (mut session, plan) = open(&backend, &name);
            let output = session.submit(&plan, 0).unwrap().wait().unwrap();
            let expected = read(&backend, &output.output);
            drop((output, session));
            drain(&backend);
            for action in 0..3 {
                let (mut session, plan) = open(&backend, &name);
                let mut pending = session.submit(&plan, 0).unwrap();
                for _ in 0..64 {
                    if matches!(pending.stage, Some(Stage::Decode(_))) {
                        break;
                    }
                    step(&mut pending);
                }
                let Some(Stage::Decode(mut decode)) = pending.stage.take() else {
                    panic!("body admission");
                };
                pending.features = decode.features.map(|plan| (pending.physical, plan));
                if pending.needs_lf_output() && decode.lf.is_none() {
                    if let WgpuDecodePendingFrame::VarDct(producer) = decode.pending.as_mut() {
                        producer.wait_until_dependency_submitted().unwrap();
                    }
                    decode.lf = Some(lf_output(&decode.pending).unwrap());
                }
                let frame = decode.pending.wait().unwrap();
                let (color, extras) = if let Some(lf) = &decode.lf {
                    (
                        Extent2d::new(lf.xyb.width(), lf.xyb.height()),
                        lf.extras
                            .as_ref()
                            .unwrap()
                            .planes
                            .iter()
                            .map(|plane| Extent2d::new(plane.width, plane.height))
                            .collect::<Vec<_>>(),
                    )
                } else {
                    (
                        frame.output.outputs[0].layout.extent,
                        frame.output.outputs[1..]
                            .iter()
                            .map(|plane| plane.layout.extent)
                            .collect(),
                    )
                };
                assert_eq!(color, Extent2d::new(13, 9));
                assert_eq!(extras, [Extent2d::new(25, 17); 2]);
                let before = super::refinement_tests::references(&pending);
                assert!(before.iter().all(Option::is_none));
                if action == 2 {
                    super::refinement_tests::require_allocation_failure(&backend, || {
                        pending
                            .decoded(frame, &decode.count, decode.lf, false)
                            .map(|_| ())
                    });
                } else {
                    assert!(
                        pending
                            .decoded(frame, &decode.count, decode.lf, false)
                            .unwrap()
                            .is_none()
                    );
                    assert!(matches!(
                        pending.stage,
                        Some(Stage::Features(_) | Stage::LfFeatures(_))
                    ));
                }
                assert_eq!(super::refinement_tests::references(&pending), before);
                assert!(matches!(
                    pending.unvalidated(),
                    Err(Error::UnvalidatedOutputNotSubmitted)
                ));
                if action == 1 {
                    let output = pending.wait().unwrap();
                    assert_eq!(read(&backend, &output.output), expected);
                    drop(output);
                } else {
                    drop(pending);
                    assert!(matches!(
                        session.submit(&plan, 0),
                        Err(Error::SessionPoisoned)
                    ));
                }
                drop(session);
                drain(&backend);
            }
        }
    }
}
