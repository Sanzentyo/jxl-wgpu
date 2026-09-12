use super::super::refinement_tests::{references, require_allocation_failure};
use super::*;
use crate::progressive_dc::ProgressiveDcOutput;

use jxl_test_support::fixtures::patches as fixtures;
use jxl_test_support::oracles::lf as lf_oracle;
use jxl_test_support::oracles::patches as patch_oracle;

fn open(
    backend: &WgpuBackend,
    family: &str,
    progressive: bool,
    bounded: bool,
) -> (DependentSession, DependentPending) {
    let data = fixture(&format!("patches/lf_producers/{family}"));
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
    let request = GpuOutputRequest::color(jxl_gpu_formats::PixelFormat::rgb_f32(
        jxl_gpu_formats::RgbChannelOrder::Rgba,
        false,
        crate::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_alpha_output_policy(crate::AlphaOutputPolicy::Preserve)
    .with_progressive_output(progressive);
    let mut session = DependentSession::new(engine, source, &inventory, &request, &plan).unwrap();
    let mut pending = session.submit(&plan, 0).unwrap();
    for _ in 0..inventory.frames.len() / 2 {
        decode(&mut pending);
    }
    assert!(matches!(pending.stage, Some(Stage::PatchDictionary(_))));
    assert!(
        pending
            .carry
            .as_ref()
            .unwrap()
            .lf
            .iter()
            .all(Option::is_none)
    );
    assert!(pending.carry.as_ref().unwrap().references[3].is_some());
    (session, pending)
}

fn dictionary(pending: &mut DependentPending) {
    let Some(Stage::PatchDictionary(parser)) = pending.stage.take() else {
        panic!("dictionary");
    };
    let (dictionary, count) = parser.wait().unwrap();
    pending.dictionary_decoded(dictionary, count).unwrap();
}

fn body(
    pending: &mut DependentPending,
) -> (
    SubmittedGpuFrame<GpuImageFrame>,
    Arc<AtomicUsize>,
    ProgressiveDcOutput,
) {
    let Some(Stage::Decode(mut decode)) = pending.stage.take() else {
        panic!("LF body");
    };
    if let WgpuDecodePendingFrame::VarDct(producer) = decode.pending.as_mut() {
        producer.wait_until_dependency_submitted().unwrap();
    }
    let lf = lf_output(&decode.pending).unwrap();
    (decode.pending.wait().unwrap(), decode.count, lf)
}

fn read_buffer(backend: &WgpuBackend, buffer: &jxl_wgpu::GpuBufferLease) -> Vec<f64> {
    let frame = GpuImageFrame {
        token: SubmissionToken(1),
        outputs: vec![GpuImageOutput {
            id: OutputId(0),
            layout: ImageLayout::packed(
                Extent2d::new((buffer.size() / 4) as u32, 1),
                jxl_gpu_formats::vpi::VpiPitchLinearFormat::F32.pixel_format(),
            )
            .unwrap(),
            buffer: buffer.clone(),
        }],
        changed: Default::default(),
    };
    read(backend, &frame)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f64::from(f32::from_le_bytes(*word)))
        .collect()
}

fn components(backend: &WgpuBackend, lf: &ProgressiveDcOutput) -> lf_oracle::Planes {
    let width = lf.xyb.width() as usize;
    let height = lf.xyb.height() as usize;
    let mut channels: Vec<_> = lf
        .xyb
        .planes
        .iter()
        .map(|plane| {
            let data = read_buffer(backend, &plane.buffer);
            (0..height)
                .flat_map(|y| {
                    data[y * plane.stride as usize..y * plane.stride as usize + width]
                        .iter()
                        .copied()
                })
                .collect()
        })
        .collect();
    if let Some(extra) = &lf.extras {
        let data = read_buffer(backend, &extra.buffer);
        channels.extend(extra.planes.iter().map(|plane| {
            (0..height)
                .flat_map(|y| {
                    let row = plane.word_offset as usize + y * plane.row_stride_words as usize;
                    data[row..row + width].iter().copied()
                })
                .collect()
        }));
    }
    lf_oracle::Planes {
        width,
        height,
        channels,
    }
}

fn check_components(actual: &lf_oracle::Planes, expected: &lf_oracle::Planes) {
    assert_eq!(
        (actual.width, actual.height),
        (expected.width, expected.height)
    );
    for (actual, expected) in actual.channels.iter().zip(&expected.channels) {
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(
                a.is_finite() && b.is_finite() && (a - b).abs() < 2e-6 * (1.0 + b.abs()),
                "{a} vs {b}"
            );
        }
    }
}

#[test]
fn lf_patch_planes_match_scalar_algebra_and_release_extras_independently_of_prediction() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for family in [
        "vardct_gab0",
        "modular_gab1",
        "nested_vardct_gab1",
        "nested_modular_gab1",
    ] {
        for progressive in [false, true] {
            let (session, mut pending) = open(&backend, family, progressive, true);
            dictionary(&mut pending);
            let (frame, count, lf) = body(&mut pending);
            // Expected patch algebra uses independently written f64 code and the validated
            // input samples. Native final-image tests independently verify both codec producers.
            let mut expected = components(&backend, &lf);
            let carry = pending.carry.as_ref().unwrap();
            let reference = carry.references[3].as_ref().unwrap();
            let data = read_buffer(&backend, &reference.buffer);
            let pixels = reference.extent.width as usize * reference.extent.height as usize;
            let reference_planes = lf_oracle::Planes {
                width: reference.extent.width as usize,
                height: reference.extent.height as usize,
                channels: (0..expected.channels.len())
                    .map(|channel| {
                        let offset = channel * reference.plane_words as usize;
                        data[offset..offset + pixels].to_vec()
                    })
                    .collect(),
            };
            let image = &carry.source.inventory.image_header;
            let mut small = carry.source.inventory.frames[pending.physical].clone();
            (small.width, small.height) = (expected.width as u32, expected.height as u32);
            let values = fixtures::values(&small, image.extra_channels.len(), 16);
            patch_oracle::apply(&mut expected, &reference_planes, image, &values);
            let before = references(&pending);
            let reference_bytes = carry
                .references
                .iter()
                .flatten()
                .map(|surface| surface.buffer.size())
                .sum::<u64>();
            pending.decoded(frame, &count, Some(lf), true).unwrap();
            let Some(Stage::LfPatchRender(work)) = pending.stage.take() else {
                panic!("LF patch render");
            };
            assert!(
                pending
                    .carry
                    .as_ref()
                    .unwrap()
                    .lf
                    .iter()
                    .all(Option::is_none)
            );
            let output = work.wait().unwrap();
            assert_eq!(output.extras.is_some(), progressive);
            let actual = components(&backend, &output);
            assert_eq!(actual.channels.len(), if progressive { 5 } else { 3 });
            check_components(&actual, &expected);
            assert_eq!(references(&pending), before);
            let prediction_bytes = 3 * actual.width as u64 * actual.height as u64 * 4;
            assert_eq!(
                output
                    .xyb
                    .planes
                    .iter()
                    .map(|plane| plane.buffer.size())
                    .sum::<u64>(),
                prediction_bytes
            );
            let extra_bytes = output
                .extras
                .as_ref()
                .map_or(0, |extras| extras.buffer.size());
            assert_eq!(
                backend.transient_memory_budget().snapshot().reserved_bytes,
                reference_bytes + prediction_bytes + extra_bytes
            );
            // Only the XYB allocation moves to the prediction slot. Drop every extra even when
            // the submitted work originally retained them for progressive delivery.
            pending.record_lf(Some(output), false).unwrap();
            assert!(pending.lf_pending.is_empty());
            assert_eq!(references(&pending), before);
            drop((pending, session));
            drain(&backend);
        }
    }
}

#[test]
fn lf_patch_dictionary_body_and_render_cancellation_or_final_drain_preserve_committed_references() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for family in ["nested_vardct_gab1", "nested_modular_gab1"] {
        for bounded in [false, true] {
            let (session, pending) = open(&backend, family, true, bounded);
            let final_image = pending.wait().unwrap();
            let expected = read(&backend, &final_image.output);
            drop((final_image, session));
            drain(&backend);
            for boundary in 0..3 {
                for action in 0..4 {
                    if action == 3 && boundary != 2 {
                        continue;
                    }
                    let (session, mut pending) = open(&backend, family, true, bounded);
                    let before = references(&pending);
                    if boundary >= 1 {
                        dictionary(&mut pending);
                    }
                    if boundary == 2 {
                        let (frame, count, lf) = body(&mut pending);
                        if action == 3 {
                            require_allocation_failure(&backend, || {
                                pending.decoded(frame, &count, Some(lf), true)
                            });
                        } else {
                            pending.decoded(frame, &count, Some(lf), true).unwrap();
                            assert!(matches!(pending.stage, Some(Stage::LfPatchRender(_))));
                        }
                    }
                    assert_eq!(references(&pending), before);
                    assert!(
                        pending
                            .carry
                            .as_ref()
                            .unwrap()
                            .lf
                            .iter()
                            .all(Option::is_none)
                    );
                    match action {
                        0 | 3 => drop(pending),
                        1 => {
                            let frame = pending.wait().unwrap();
                            assert_eq!(read(&backend, &frame.output), expected);
                        }
                        2 => {
                            let frame =
                                pollster::block_on(std::future::poll_fn(|cx| pending.poll(cx)))
                                    .unwrap();
                            assert_eq!(read(&backend, &frame.output), expected);
                            drop(pending);
                        }
                        _ => unreachable!(),
                    }
                    drop(session);
                    drain(&backend);
                }
            }
        }
    }
}
