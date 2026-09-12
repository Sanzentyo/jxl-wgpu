use super::*;

pub(super) fn fragmented(
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    data: &[u8],
    request: GpuOutputRequest,
) -> GpuDecodeSession<WgpuDecodeSubmissionSession> {
    let mut stream = decoder.stream(request).unwrap();
    let mut transport = ContainerStreamScanner::new(decoder.container_stream_limits());
    for chunk in data.chunks(43) {
        for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
            stream.push_transport_event(&event).unwrap();
        }
    }
    for event in transport.finish_input().unwrap() {
        stream.push_transport_event(&event).unwrap();
    }
    assert!(stream.stats().retained_spans >= 3);
    stream.finish().unwrap()
}

#[test]
fn resampled_modular_and_vardct_planes_match_oracles_on_gpu() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let mut whole = std::collections::BTreeMap::new();
    for limit in [None, NonZeroU64::new(1024)] {
        let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        if let Some(limit) = limit {
            engine = engine.with_stream_window_limit(limit);
        }
        let decoder = GpuDecoder::new(engine);
        for (name, hex, _, _) in resampled_fixtures() {
            let data = encoded(hex);
            let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            let image = &inventory.image_header;
            let pixels = image.width as usize * image.height as usize;
            // jxl 0.6 handles dimension_shift inconsistently. libjxl checks those fixtures.
            let rust = (!name.contains("shift")).then(|| oracle::rust_planes(&data));
            let lib = oracle::libjxl_planes(&data, pixels, image.extra_channels.len());
            if rust.is_none() && lib.is_none() {
                eprintln!("{name}: libjxl oracle unavailable");
                continue;
            }
            for selected in 0..=image.extra_channels.len() {
                let color = selected == image.extra_channels.len();
                let bits = if color {
                    0
                } else {
                    match image.extra_channels[selected].bit_depth {
                        jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample } => {
                            bits_per_sample as u8
                        }
                        _ => unreachable!(),
                    }
                };
                for floating in [false, true]
                    .into_iter()
                    .filter(|floating| !color || *floating)
                {
                    let request = if color {
                        request(false)
                    } else {
                        scalar::scalar_request(selected as u32, bits, floating)
                            .with_orientation_policy(OrientationPolicy::Apply)
                    };
                    let mut session = if limit.is_some() {
                        fragmented(&decoder, &data, request)
                    } else {
                        decoder
                            .open(&data, request)
                            .unwrap_or_else(|e| panic!("{name}/{selected} open: {e}"))
                    };
                    assert_eq!(session.metadata().extra_channels, image.extra_channels);
                    let frame = if limit.is_some() {
                        pollster::block_on(session.next_frame_async())
                    } else {
                        session.next_frame()
                    }
                    .unwrap_or_else(|e| panic!("{name}/{selected}/{floating} decode: {e}"))
                    .unwrap();
                    let render_bytes = if let Some(producer) = session.submission_session().vardct()
                    {
                        let stats = producer.memory_stats().unwrap();
                        if !color {
                            assert_eq!(stats.resident_image_bytes, 0);
                        }
                        stats.extra_render_bytes
                    } else {
                        session
                            .submission_session()
                            .modular()
                            .unwrap()
                            .memory_stats()
                            .modular_render_bytes
                    };
                    assert!(render_bytes >= pixels as u64 * 4);
                    let readback = ImageReadbackPipeline::new(&backend)
                        .submit(frame.output())
                        .unwrap()
                        .wait()
                        .unwrap();
                    let output = &readback.frame.outputs[0];
                    let extent = OutputOrientation::from_exif_value(image.orientation)
                        .unwrap()
                        .map_extent(Extent2d::new(image.width, image.height));
                    assert_eq!(output.layout.extent, extent);
                    let key = (name, selected, floating);
                    if limit.is_some() {
                        assert_eq!(
                            &output.bytes,
                            whole.get(&key).unwrap(),
                            "{name}: entropy window changes output"
                        );
                    } else {
                        whole.insert(key, output.bytes.clone());
                    }
                    for (label, reference) in [("rust", &rust), ("libjxl", &lib)] {
                        let Some((colors, extras)) = reference else {
                            continue;
                        };
                        let expected = if color { colors } else { &extras[selected] };
                        if floating {
                            compare_float(
                                name,
                                label,
                                &oracle::floats(&output.bytes),
                                expected,
                                color,
                            );
                        } else {
                            let maximum = ((1u32 << bits) - 1) as f32;
                            let samples: Vec<_> = output
                                .bytes
                                .chunks_exact(usize::from(bits.div_ceil(8)))
                                .map(|bytes| {
                                    if bits <= 8 {
                                        u32::from(bytes[0])
                                    } else {
                                        u32::from(u16::from_le_bytes(bytes.try_into().unwrap()))
                                    }
                                })
                                .collect();
                            assert_eq!(samples.len(), expected.len());
                            for (index, (sample, value)) in samples.iter().zip(expected).enumerate()
                            {
                                let code = (value * maximum).round() as u32;
                                assert!(
                                    sample.abs_diff(code) <= 1,
                                    "{name}/{selected}/{label}/{index}: {sample} vs {code}"
                                );
                            }
                        }
                    }
                    drop(readback);
                    assert_eq!(
                        decoder.engine().in_flight_memory_stats().reserved_bytes,
                        frame.output().outputs[0].buffer.reserved_bytes()
                    );
                    drop(frame);
                    drop(session);
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                }
            }
        }
    }
}

fn compare_float(name: &str, oracle: &str, actual: &[f32], expected: &[f32], color: bool) {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = [0.0_f32; 2];
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite());
        let scalar = !color || index % 4 == 3;
        maximum[usize::from(scalar)] = maximum[usize::from(scalar)].max((a - b).abs());
    }
    assert!(maximum[1] < 4e-7, "{name}/{oracle}: scalar {maximum:?}");
    assert!(
        maximum[0]
            < if name.starts_with("modular") {
                1e-6
            } else if oracle == "rust" {
                3e-4
            } else {
                0.002
            },
        "{name}/{oracle}: color {maximum:?}"
    );
}

#[test]
fn resampled_planes_are_admitted_before_submission_and_released_after_cancellation() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for (name, hex, _, _) in resampled_fixtures()
        .into_iter()
        .filter(|(name, _, _, _)| matches!(*name, "modular2" | "vardct2"))
    {
        let data = encoded(hex);
        let probe = GpuDecoder::wgpu(backend.clone()).unwrap();
        let planned = probe.open(&data, request(false)).unwrap();
        let (base, render_bytes) = if let Some(producer) = planned.submission_session().modular() {
            let stats = producer.memory_stats();
            (stats.per_frame_bytes, stats.modular_render_bytes)
        } else {
            let stats = planned
                .submission_session()
                .vardct()
                .unwrap()
                .memory_stats()
                .unwrap();
            (stats.total_frame_bytes, stats.extra_render_bytes)
        };
        assert!(render_bytes > 517 * 9 * 4);
        drop(planned);
        let limit = base * 2;
        let mut config = jxl_wgpu::WgpuBackendConfig::default();
        config.memory.max_in_flight_transient_bytes = limit;
        let scoped = WgpuBackend::from_device(
            backend.device().clone(),
            backend.queue().clone(),
            backend.adapter_info().clone(),
            config,
        )
        .unwrap();
        let budget = scoped.transient_memory_budget().clone();
        let engine = WgpuDecodeEngine::new(scoped).unwrap();
        let decoder = GpuDecoder::new(engine);
        let mut session = decoder.open(&data, request(false)).unwrap();
        assert_eq!(budget.snapshot().reserved_bytes, 0);
        let held = budget.try_reserve(limit - base + 1).unwrap();
        let pressure = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        assert!(
            matches!(pressure.backpressure, Some(PrefetchBackpressure::Memory(_))),
            "{name}"
        );
        drop(held);
        session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        assert!(budget.snapshot().reserved_bytes >= base, "{name}");
        drop(session);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while budget.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::yield_now();
        }
        assert_eq!(
            budget.snapshot().reserved_bytes,
            0,
            "{name}: abandoned render buffers"
        );
        let mut session = decoder.open(&data, request(false)).unwrap();
        let frame = pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap();
        assert_eq!(
            budget.snapshot().reserved_bytes,
            frame.output().outputs[0].buffer.reserved_bytes()
        );
        drop(frame);
        drop(session);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
    }
}

#[test]
fn custom_resampling_weights_reach_both_gpu_output_producers() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for (name, hex, _, _) in resampled_fixtures()
        .into_iter()
        .filter(|(name, _, _, _)| name.ends_with('2') || name.ends_with('4') || name.ends_with('8'))
        .filter(|(name, _, _, _)| !name.contains("shift"))
    {
        eprintln!("custom {name}");
        let data = corpus::with_custom_upsampling_weights(&encoded(hex));
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert!(
            inventory
                .image_header
                .upsampling_weights
                .up8
                .iter()
                .all(|weight| weight.to_f32() == 0.03125)
        );
        let (rust, _) = oracle::rust_planes(&data);
        let lib = oracle::libjxl_planes(
            &data,
            rust.len() / 4,
            inventory.image_header.extra_channels.len(),
        );
        let mut session = fragmented(&decoder, &data, request(false));
        let frame = pollster::block_on(session.next_frame_async())
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .unwrap();
        let readback = ImageReadbackPipeline::new(&backend)
            .submit(frame.output())
            .unwrap()
            .wait()
            .unwrap();
        let actual = oracle::floats(&readback.frame.outputs[0].bytes);
        compare_float(name, "rust", &actual, &rust, true);
        if let Some((lib, _)) = lib {
            compare_float(name, "libjxl", &actual, &lib, true);
        }
        drop(readback);
        drop(frame);
        drop(session);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}

fn resampled_fixtures() -> [(&'static str, &'static str, u32, u32); 20] {
    [
        (
            "modular_shift8",
            include_str!("../../../test-data/extras_shifted8.jxl.hex"),
            1,
            8,
        ),
        (
            "vardct_shift8",
            include_str!("../../../test-data/vardct_extras_shifted8.jxl.hex"),
            1,
            8,
        ),
        (
            "modular_color4",
            include_str!("../../../test-data/extras_resampled_color4.jxl.hex"),
            4,
            4,
        ),
        (
            "modular_color8",
            include_str!("../../../test-data/extras_resampled_color8.jxl.hex"),
            8,
            8,
        ),
        (
            "modular_shift4",
            include_str!("../../../test-data/extras_shifted4.jxl.hex"),
            2,
            4,
        ),
        (
            "modular_squeeze",
            include_str!("../../../test-data/extras_resampled_squeeze.jxl.hex"),
            1,
            2,
        ),
        (
            "vardct_color4",
            include_str!("../../../test-data/vardct_extras_resampled_color4.jxl.hex"),
            4,
            4,
        ),
        (
            "vardct_color8",
            include_str!("../../../test-data/vardct_extras_resampled_color8.jxl.hex"),
            8,
            8,
        ),
        (
            "vardct_shift4",
            include_str!("../../../test-data/vardct_extras_shifted4.jxl.hex"),
            2,
            4,
        ),
        (
            "vardct_squeeze",
            include_str!("../../../test-data/vardct_extras_resampled_squeeze.jxl.hex"),
            1,
            2,
        ),
        (
            "modular2",
            include_str!("../../../test-data/extras_resampled_2.jxl.hex"),
            1,
            2,
        ),
        (
            "modular4",
            include_str!("../../../test-data/extras_resampled_4.jxl.hex"),
            1,
            4,
        ),
        (
            "modular8",
            include_str!("../../../test-data/extras_resampled_8.jxl.hex"),
            1,
            8,
        ),
        (
            "modular_color",
            include_str!("../../../test-data/extras_resampled_color.jxl.hex"),
            2,
            8,
        ),
        (
            "modular_shift",
            include_str!("../../../test-data/extras_shifted.jxl.hex"),
            1,
            2,
        ),
        (
            "vardct2",
            include_str!("../../../test-data/vardct_extras_resampled_2.jxl.hex"),
            1,
            2,
        ),
        (
            "vardct4",
            include_str!("../../../test-data/vardct_extras_resampled_4.jxl.hex"),
            1,
            4,
        ),
        (
            "vardct8",
            include_str!("../../../test-data/vardct_extras_resampled_8.jxl.hex"),
            1,
            8,
        ),
        (
            "vardct_color",
            include_str!("../../../test-data/vardct_extras_resampled_color.jxl.hex"),
            2,
            8,
        ),
        (
            "vardct_shift",
            include_str!("../../../test-data/vardct_extras_shifted.jxl.hex"),
            1,
            2,
        ),
    ]
}

#[test]
fn resampled_inventory_resolves_coded_grids_and_dimension_shift() {
    for (name, hex, color, extra) in resampled_fixtures() {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let frame = &inventory.frames[0];
        eprintln!(
            "{name}: color={} extras={:?} shift={:?}",
            frame.upsampling,
            frame.extra_channel_upsampling,
            inventory
                .image_header
                .extra_channels
                .iter()
                .map(|e| e.dimension_shift)
                .collect::<Vec<_>>()
        );
        assert_eq!(frame.upsampling, color, "{name}");
        assert!(
            frame
                .extra_channel_upsampling
                .iter()
                .all(|&factor| factor == extra),
            "{name}"
        );
        assert_eq!(
            frame.color_sample_extent(),
            Some((frame.width.div_ceil(color), frame.height.div_ceil(color)))
        );
    }
}
