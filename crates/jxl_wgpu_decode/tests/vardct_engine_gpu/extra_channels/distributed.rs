use super::*;

fn distributed_fixtures() -> [(&'static str, &'static str); 5] {
    [
        (
            "alpha",
            include_str!("../../../test-data/vardct_extras_distributed_alpha.jxl.hex"),
        ),
        (
            "data",
            include_str!("../../../test-data/vardct_extras_distributed_data.jxl.hex"),
        ),
        (
            "progressive",
            include_str!("../../../test-data/vardct_extras_distributed_progressive.jxl.hex"),
        ),
        (
            "wide",
            include_str!("../../../test-data/vardct_extras_distributed_wide.jxl.hex"),
        ),
        (
            "squeeze",
            include_str!("../../../test-data/vardct_extras_distributed_squeeze.jxl.hex"),
        ),
    ]
}

#[test]
fn distributed_vardct_extras_reconstruct_ac_and_lf_groups_on_gpu() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for limit in [None, NonZeroU64::new(1024)] {
        let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        if let Some(limit) = limit {
            engine = engine.with_stream_window_limit(limit);
        }
        let decoder = GpuDecoder::new(engine);
        for (name, hex) in distributed_fixtures() {
            let data = encoded(hex);
            let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            let (color, planes) = oracle::rust_planes(&data);
            let libjxl = oracle::libjxl_planes(
                &data,
                inventory.image_header.width as usize * inventory.image_header.height as usize,
                planes.len(),
            );
            let image = &inventory.image_header;
            let outputs = (0..planes.len())
                .flat_map(|index| [(Some(index), false), (Some(index), true)])
                .chain(std::iter::once((None, true)));
            for (index, floating) in outputs {
                let bits = index.map(|index| match image.extra_channels[index].bit_depth {
                    jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample } => {
                        bits_per_sample
                    }
                    _ => unreachable!(),
                });
                let request = if let Some(index) = index {
                    scalar::scalar_request(index as u32, bits.unwrap() as u8, floating)
                } else {
                    request(false)
                };
                let mut session = if limit.is_some() {
                    fragmented(&decoder, &data, request)
                } else {
                    decoder
                        .open(&data, request)
                        .unwrap_or_else(|e| panic!("{name}/{index:?}: {e:?}"))
                };
                assert_eq!(session.metadata().extra_channels, image.extra_channels);
                let frame = if limit.is_some() {
                    pollster::block_on(session.next_frame_async())
                } else {
                    session.next_frame()
                }
                .unwrap_or_else(|e| panic!("{name}/{limit:?}/{index:?}/{floating}: {e:?}"))
                .unwrap();
                let memory = session
                    .submission_session()
                    .vardct()
                    .unwrap()
                    .memory_stats()
                    .unwrap();
                assert!(memory.extra_arena_bytes > 0);
                if index.is_some() {
                    assert_eq!(memory.resident_image_bytes, 0);
                }
                let result = ImageReadbackPipeline::new(&backend)
                    .submit(frame.output())
                    .unwrap()
                    .wait()
                    .unwrap();
                let output = &result.frame.outputs[0];
                let original = Extent2d::new(image.width, image.height);
                let oriented = OutputOrientation::from_exif_value(image.orientation)
                    .unwrap()
                    .map_extent(original);
                assert_eq!(
                    output.layout.extent,
                    if floating { oriented } else { original }
                );
                if floating {
                    let actual = oracle::floats(&output.bytes);
                    let expected = index.map_or(&color, |index| &planes[index]);
                    assert_eq!(actual.len(), expected.len());
                    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
                        let tolerance = if index.is_some() || i % 4 == 3 {
                            2e-7
                        } else {
                            3e-4
                        };
                        assert!(
                            (a - b).abs() < tolerance,
                            "{name}/{limit:?}/{index:?}/{i}: {a} vs {b}"
                        );
                        if let Some((lc, lp)) = &libjxl {
                            let expected = index.map_or(lc[i], |index| lp[index][i]);
                            assert!(
                                (a - expected).abs()
                                    < if index.is_some() || i % 4 == 3 {
                                        2e-7
                                    } else {
                                        0.002
                                    },
                                "{name}/{limit:?}/{index:?}/{i}: {a} vs libjxl {expected}"
                            );
                        }
                    }
                } else {
                    let bits = bits.unwrap();
                    let bytes = bits.div_ceil(8) as usize;
                    let mask = (1_u32 << bits) - 1;
                    let channel = if image.grayscale { 1 } else { 3 } + index.unwrap() as u32;
                    for y in 0..image.height {
                        for x in 0..image.width {
                            let code = match x % 11 {
                                0 => 0,
                                1 => mask,
                                _ => {
                                    (193 * x + 317 * y + 97 * channel + (x ^ y) * (23 + channel))
                                        & mask
                                }
                            };
                            let offset = (y * image.width + x) as usize * bytes;
                            let actual = if bytes == 1 {
                                u32::from(output.bytes[offset])
                            } else {
                                u32::from(u16::from_le_bytes(
                                    output.bytes[offset..offset + 2].try_into().unwrap(),
                                ))
                            };
                            assert_eq!(actual, code, "{name}/{limit:?}/{index:?}/{x},{y}");
                        }
                    }
                }
                drop(result);
                drop(frame);
                drop(session);
                assert_eq!(
                    decoder.engine().in_flight_memory_stats().reserved_bytes,
                    0,
                    "{name}/{limit:?}/{index:?}"
                );
            }
        }
    }
}

fn fragmented(
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    data: &[u8],
    request: GpuOutputRequest,
) -> GpuDecodeSession<WgpuDecodeSubmissionSession> {
    let mut stream = decoder.stream(request).unwrap();
    let mut transport = ContainerStreamScanner::new(decoder.container_stream_limits());
    for chunk in data.chunks(347) {
        for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
            stream.push_transport_event(&event).unwrap();
        }
    }
    for event in transport.finish_input().unwrap() {
        stream.push_transport_event(&event).unwrap();
    }
    assert!(stream.stats().retained_spans > 3);
    stream.finish().unwrap()
}

#[test]
fn distributed_extra_stage_keeps_canceled_buffers_budgeted_and_retries_initial_pressure() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let data = encoded(distributed_fixtures()[0].1);
    let probe = GpuDecoder::new(
        VarDctSubmissionEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(128).unwrap()),
    );
    let planned = probe.open(&data, request(false)).unwrap();
    let base = planned
        .submission_session()
        .memory_stats()
        .unwrap()
        .total_frame_bytes;
    assert!(
        planned
            .submission_session()
            .global_modular_memory_stats()
            .is_none()
    );
    drop(planned);
    let budget = MemoryBudget::new(NonZeroU64::new(base * 3).unwrap());
    let decoder = GpuDecoder::new(
        VarDctSubmissionEngine::with_memory_budget(backend.clone(), budget.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(128).unwrap()),
    );
    let mut session = decoder.open(&data, request(false)).unwrap();
    let blocker = budget.try_reserve(base * 3).unwrap();
    let pressure = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert!(matches!(
        pressure.backpressure,
        Some(PrefetchBackpressure::Memory(_))
    ));
    drop(blocker);
    session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert_eq!(budget.snapshot().reserved_bytes, base);
    let initial_submissions = session.submission_session().submissions_per_frame();
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while session.submission_session().submissions_per_frame() == initial_submissions {
        assert!(std::time::Instant::now() < deadline);
        assert!(session.poll_next_frame(&mut context).is_pending());
        std::thread::yield_now();
    }
    assert!(
        budget.snapshot().reserved_bytes > base,
        "the live subimage also has a permit"
    );
    assert!(
        session
            .front_pending_frame()
            .unwrap()
            .unvalidated_gpu_frame()
            .is_err()
    );
    // Abandon a submitted Modular window before polling its cursor or issuing output commands.
    drop(session);
    drain_budget(&backend, &budget);
    let mut retry = decoder.open(&data, request(false)).unwrap();
    let frame = pollster::block_on(retry.next_frame_async())
        .unwrap()
        .unwrap();
    assert_eq!(
        budget.snapshot().reserved_bytes,
        frame.output().outputs[0].buffer.reserved_bytes()
    );
    drop(frame);
    drop(retry);
    assert_eq!(budget.snapshot().reserved_bytes, 0);

    // Capacity for the known frame does not invent capacity for later entropy descriptors.
    let tight = MemoryBudget::new(NonZeroU64::new(base).unwrap());
    let decoder = GpuDecoder::new(
        VarDctSubmissionEngine::with_memory_budget(backend.clone(), tight.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(128).unwrap()),
    );
    let mut session = decoder.open(&data, request(false)).unwrap();
    let error = pollster::block_on(session.next_frame_async()).unwrap_err();
    assert!(
        matches!(error, DecodeError::MemoryBackpressure(MemoryBudgetError::Exhausted { reserved_bytes, .. }) if reserved_bytes == base),
        "{error:?}"
    );
    drop(session);
    drain_budget(&backend, &tight);
}

fn drain_budget(backend: &WgpuBackend, budget: &MemoryBudget) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while budget.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert_eq!(budget.snapshot().reserved_bytes, 0);
}

#[test]
fn damaged_distributed_extra_entropy_never_delivers_color_or_scalar_output() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let mut data = encoded(distributed_fixtures()[0].1);
    let parsed = jxl_gpu_bitstream::parse(&data, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let section = inventory.frames[0]
        .sections
        .iter()
        .rev()
        .find(|section| {
            matches!(
                section.kind,
                jxl_gpu_bitstream::FrameSectionKind::PassGroup { .. }
            )
        })
        .unwrap();
    let end = section.bits.end().unwrap() as usize / 8;
    data[end - 4..end].fill(0);
    for limit in [None, NonZeroU64::new(128)] {
        let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        if let Some(limit) = limit {
            engine = engine.with_stream_window_limit(limit);
        }
        let decoder = GpuDecoder::new(engine);
        for output in [request(false), scalar::scalar_request(0, 5, true)] {
            let mut session = decoder.open(&data, output).unwrap();
            let error = pollster::block_on(session.next_frame_async()).unwrap_err();
            assert!(
                matches!(
                    error,
                    DecodeError::VarDct(VarDctDecodeError::ExtraModularStatus { .. })
                ),
                "{error:?}"
            );
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}
