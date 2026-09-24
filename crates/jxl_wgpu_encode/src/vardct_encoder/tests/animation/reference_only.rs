use super::*;
use crate::{Determinism, EncodeProfile, FrameEncodeRequest, FrameKind};

fn reference(slot: u8) -> FrameOptions {
    FrameOptions {
        kind: FrameKind::ReferenceOnly,
        save_as_reference: ReferenceSlot::new(slot).unwrap(),
        ..Default::default()
    }
}

fn exercise(
    backend: &WgpuBackend,
    context: &WgpuContext,
    desc: VarDctAnimationDescriptor,
    session: VarDctAnimationSession,
    still: impl FnMut(BufferImageSource) -> Vec<u8>,
    reference_extent: (usize, usize),
) {
    let (width, height) = (desc.canvas_width() as usize, desc.canvas_height() as usize);
    let layers: Vec<_> = (0..4)
        .flat_map(|slot| {
            [
                Layer {
                    width: reference_extent.0,
                    height: reference_extent.1,
                    options: FrameOptions {
                        crop: (reference_extent != (width, height)).then(|| {
                            FrameCrop::new(
                                0,
                                0,
                                reference_extent.0 as u32,
                                reference_extent.1 as u32,
                            )
                            .unwrap()
                        }),
                        ..reference(slot)
                    },
                },
                Layer {
                    width,
                    height,
                    options: options(
                        u32::from(slot) + 2,
                        Some(100 + u32::from(slot)),
                        BlendMode::Add,
                        slot,
                        0,
                    ),
                },
            ]
        })
        .collect();
    let (encoded, samples) = encode_layers(context, session, &layers, still, true);
    check_animation(backend, &encoded, &desc, &layers, &samples, 5);
    let parsed = jxl_gpu_bitstream::parse(&encoded, Default::default()).unwrap();
    let index = jxl_gpu_bitstream::FrameIndex::from_container(&parsed, Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(index.entries().len(), 4);
    assert!(index.entries().iter().all(|entry| entry.frames == 1));
}

#[test]
fn reference_only_frames_share_all_backends_slots_and_implicit_single_pass() {
    let backend = backend();
    let context = WgpuContext::from_backend(&backend);
    let config = VarDctConfig {
        progressive: progressive::combined(),
        ..configuration()
    };
    for (encoder, width, height) in [
        (
            VarDctEncoder::new_with_config(context.clone(), VarDctStrategy::Dct8, config.clone())
                .unwrap(),
            8,
            8,
        ),
        (
            VarDctEncoder::new_with_strategy_map(
                context.clone(),
                mixed::packed_map(25, 17, false),
                config.clone(),
            )
            .unwrap(),
            25,
            17,
        ),
    ] {
        let desc = descriptor(width, height, timebase(60_000, 1001, 2, true));
        let session = encoder.begin_animation(desc.clone()).unwrap();
        exercise(
            &backend,
            &context,
            desc,
            session,
            |s| encoder.encode(s).unwrap(),
            (width, height),
        );
    }
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
    let desc = descriptor(257, 17, timebase(60_000, 1001, 2, true));
    let session = encoder.begin_animation(desc.clone()).unwrap();
    exercise(
        &backend,
        &context,
        desc,
        session,
        |s| encoder.encode(s).unwrap(),
        (259, 19),
    );
}

#[test]
fn reference_only_controls_reject_absent_fields_before_admission() {
    let context = test_context().expect("actual GPU required");
    let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
    let mut session = encoder
        .begin_animation(descriptor(9, 7, timebase(100, 1, 0, true)))
        .unwrap();
    let source = padded_rgb_source_sized(&context, 9, 7, &pixels(9, 7, 0));
    for bad in [
        FrameOptions {
            timing: FrameTiming {
                duration_ticks: 1,
                timecode: None,
            },
            ..reference(0)
        },
        FrameOptions {
            timing: FrameTiming {
                duration_ticks: 0,
                timecode: Some(0),
            },
            ..reference(0)
        },
        FrameOptions {
            color_blend: FrameBlend {
                mode: BlendMode::Add,
                ..Default::default()
            },
            ..reference(0)
        },
        FrameOptions {
            extra_channel_blends: vec![FrameBlend::default()],
            ..reference(0)
        },
        FrameOptions {
            crop: Some(FrameCrop::new(-1, 0, 9, 7).unwrap()),
            ..reference(0)
        },
        FrameOptions {
            crop: Some(FrameCrop::new(0, 0, 8, 7).unwrap()),
            ..reference(0)
        },
        FrameOptions {
            save_before_color_transform: true,
            ..reference(0)
        },
    ] {
        assert!(matches!(
            session.submit_frame(source.clone(), bad),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert_eq!(session.next_frame_index(), FrameIndex::new(0));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
    assert!(matches!(
        session.submit_last_frame(source.clone(), reference(0)),
        Err(EncodeError::InvalidConfiguration(_))
    ));
    let smaller = padded_rgb_source_sized(&context, 8, 7, &pixels(8, 7, 0));
    assert!(matches!(
        session.submit_frame(
            smaller,
            FrameOptions {
                crop: Some(FrameCrop::new(0, 0, 8, 7).unwrap()),
                ..reference(0)
            }
        ),
        Err(EncodeError::InvalidConfiguration(_))
    ));
    let job = session.submit_frame(source, reference(0)).unwrap();
    session.insert(job.wait().unwrap()).unwrap();
    assert!(matches!(
        session.finish_indexed_container(Default::default(), Default::default()),
        Err(EncodeError::MissingFinalFrame)
    ));
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn reference_only_admission_uses_one_pass_and_cancellation_releases_its_exact_budget() {
    let context = test_context().expect("actual GPU required");
    let config = VarDctConfig {
        progressive: progressive::combined(),
        ..configuration()
    };
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
    let source = padded_rgb_source_sized(&context, 17, 9, &pixels(17, 9, 0));
    let animation = timebase(100, 1, 0, false);
    let request = FrameEncodeRequest {
        frame_index: FrameIndex::new(0),
        is_last: false,
        profile: EncodeProfile::VarDct {
            quantization: config.quantization,
        },
        progressive: config.progressive.clone(),
        minimum_determinism: Determinism::SameDevice,
        animation,
        canvas_width: 17,
        canvas_height: 9,
        options: reference(0),
    };
    let bytes = encoder
        .memory_plan_for_request(&source, &request)
        .unwrap()
        .owned_bytes_per_job;
    assert!(bytes < encoder.memory_plan(&source).unwrap().owned_bytes_per_job);
    for limit in [bytes - 1, bytes] {
        let context = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(limit).unwrap(),
        )
        .unwrap();
        let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
        let desc = descriptor(17, 9, animation);
        let mut session = encoder.begin_animation(desc.clone()).unwrap();
        let submitted = session.submit_frame(source.clone(), reference(0));
        if limit < bytes {
            assert!(matches!(submitted, Err(EncodeError::MemoryBackpressure(_))));
            assert_eq!(session.next_frame_index(), FrameIndex::new(0));
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
            continue;
        }
        let submitted = submitted.unwrap();
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, bytes);
        assert!(matches!(
            session.submit_frame(source.clone(), reference(1)),
            Err(EncodeError::MemoryBackpressure(_))
        ));
        assert_eq!(session.next_frame_index(), FrameIndex::new(1));
        let first = submitted.wait().unwrap();
        session.insert(first).unwrap();
        let abandoned = session.submit_frame(source.clone(), reference(1)).unwrap();
        drop(session);
        drop(abandoned);
        context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while encoder.in_flight_memory_stats().reserved_bytes != 0
            && std::time::Instant::now() < deadline
        {
            context.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::yield_now();
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        let mut recovered = encoder.begin_animation(desc).unwrap();
        recovered
            .submit_frame(source.clone(), reference(0))
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}
