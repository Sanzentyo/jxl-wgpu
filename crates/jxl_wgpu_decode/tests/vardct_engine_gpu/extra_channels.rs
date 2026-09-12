use super::*;
use jxl_gpu_bitstream::ExtraChannelTypeInventory;
use jxl_gpu_formats::RgbChannelOrder;
use jxl_wgpu_decode::{OrientationPolicy, SpotColorPolicy};

mod associated;
mod composition;
mod distributed;
use jxl_test_support::oracles::extra_channels as oracle;
mod resampled;
mod scalar;
mod spot;

fn fixtures() -> [(&'static str, &'static str); 7] {
    [
        (
            "rgba_progressive",
            include_str!("../../test-data/vardct_extras_rgba_progressive.jxl.hex"),
        ),
        (
            "data_only",
            include_str!("../../test-data/vardct_extras_data_only.jxl.hex"),
        ),
        (
            "rgb12",
            include_str!("../../test-data/vardct_extras_rgb12.jxl.hex"),
        ),
        (
            "gray8",
            include_str!("../../test-data/vardct_extras_gray8.jxl.hex"),
        ),
        (
            "gray_alpha",
            include_str!("../../test-data/vardct_extras_gray_alpha.jxl.hex"),
        ),
        (
            "rgba",
            include_str!("../../test-data/vardct_extras_rgba.jxl.hex"),
        ),
        (
            "transformed",
            include_str!("../../test-data/vardct_extras_transformed.jxl.hex"),
        ),
    ]
}

fn encoded(hex: &str) -> Vec<u8> {
    hex.split_whitespace()
        .collect::<String>()
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn request(keep: bool) -> GpuOutputRequest {
    GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_spot_color_policy(SpotColorPolicy::Preserve)
    .with_orientation_policy(if keep {
        OrientationPolicy::Keep
    } else {
        OrientationPolicy::Apply
    })
}

#[test]
fn global_modular_extras_resume_vardct_color_and_independent_alpha_on_gpu() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for (name, hex) in fixtures() {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        if name == "rgba_progressive" {
            assert!(inventory.frames[0].num_passes > 1);
            assert!(inventory.frames[0].sections.len() > 1);
        }
        let orientation = OutputOrientation::from_exif_value(image.orientation).unwrap();
        let original = Extent2d::new(image.width, image.height);
        let oriented = orientation.map_extent(original);
        let (expected, _) = oracle::rust_planes(&data);
        let libjxl =
            oracle::libjxl_planes(&data, original.area().unwrap(), image.extra_channels.len());
        for keep in [false, true] {
            let mut session = if keep {
                open_incremental(&decoder, &data, request(keep))
            } else {
                decoder
                    .open(&data, request(keep))
                    .unwrap_or_else(|e| panic!("{name}: {e}"))
            };
            assert_eq!(session.metadata().extra_channels, image.extra_channels);
            assert_eq!(
                session.profile(),
                DecodeProfile::VarDct {
                    sample_bit_depth: image.bit_depth
                }
            );
            let producer = session.submission_session().vardct().unwrap();
            assert!(producer.memory_stats().is_none());
            let initial = producer.global_modular_memory_stats().unwrap();
            assert_eq!(producer.submissions_per_frame(), 1);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            let frame = if keep {
                pollster::block_on(session.next_frame_async())
            } else {
                session.next_frame()
            }
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .unwrap();
            let producer = session.submission_session().vardct().unwrap();
            assert!(producer.memory_stats().is_some());
            assert!(producer.submissions_per_frame() > 1);
            assert!(initial.arena_bytes > 0);
            let readback = ImageReadbackPipeline::new(&backend)
                .submit(frame.output())
                .unwrap()
                .wait()
                .unwrap();
            let output = &readback.frame.outputs[0];
            assert_eq!(output.layout.extent, if keep { original } else { oriented });
            let actual = oracle::floats(&output.bytes);
            assert_eq!(actual.len(), expected.len());
            let mut maximum = [0.0_f32; 2];
            for y in 0..image.height {
                for x in 0..image.width {
                    let [ox, oy] = match image.orientation {
                        1 => [x, y],
                        2 => [image.width - 1 - x, y],
                        3 => [image.width - 1 - x, image.height - 1 - y],
                        4 => [x, image.height - 1 - y],
                        5 => [y, x],
                        6 => [image.height - 1 - y, x],
                        7 => [image.height - 1 - y, image.width - 1 - x],
                        8 => [y, image.width - 1 - x],
                        _ => unreachable!(),
                    };
                    let reference_index = (oy * oriented.width + ox) as usize * 4;
                    let actual_index = if keep {
                        (y * image.width + x) as usize * 4
                    } else {
                        reference_index
                    };
                    for c in 0..4 {
                        let value = actual[actual_index + c];
                        assert!(value.is_finite());
                        let difference = (value - expected[reference_index + c]).abs();
                        maximum[usize::from(c == 3)] = maximum[usize::from(c == 3)].max(difference);
                        if let Some((libjxl, _)) = &libjxl {
                            let delta = (value - libjxl[reference_index + c]).abs();
                            assert!(
                                delta < if c == 3 { 2e-7 } else { 0.002 },
                                "{name} libjxl {x},{y}/{c}: {delta}"
                            );
                        }
                    }
                }
            }
            eprintln!("{name} keep={keep}: color/alpha maximum {maximum:?}");
            assert!(maximum[0] < 0.0003, "{name}: color {}", maximum[0]);
            assert!(maximum[1] < 2e-7, "{name}: alpha {}", maximum[1]);
            if !image
                .extra_channels
                .iter()
                .any(|extra| matches!(extra.channel_type, ExtraChannelTypeInventory::Alpha { .. }))
            {
                assert!(
                    actual
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|pixel| pixel[3] == 1.0)
                );
            }
            drop(readback);
            drop(frame);
            drop(session);
            assert_eq!(
                decoder.engine().in_flight_memory_stats().reserved_bytes,
                0,
                "{name}: final lifetime"
            );
        }
    }
}

#[test]
fn global_modular_stage_enforces_caps_and_keeps_abandoned_gpu_buffers_budgeted() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let data = encoded(include_str!("../../test-data/vardct_extras_rgba.jxl.hex"));
    let probe = GpuDecoder::new(
        VarDctSubmissionEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
    );
    let mut session = probe.open(&data, request(false)).unwrap();
    let global = session
        .submission_session()
        .global_modular_memory_stats()
        .unwrap();
    let frame = session.next_frame().unwrap().unwrap();
    let color = session.submission_session().memory_stats().unwrap();
    // HF entropy tables are discovered after LF validation and reserve additional bytes.
    // Leave room for those dynamic tables while filling the budget exactly during the prelude.
    let limit = 2 * (global.total_bytes + global.arena_bytes + color.total_frame_bytes);
    drop(frame);
    drop(session);
    assert_eq!(probe.engine().in_flight_memory_stats().reserved_bytes, 0);

    // Initial-stage planning adapts the upload to total capacity without allocating GPU buffers.
    let budget = MemoryBudget::new(NonZeroU64::new(global.total_bytes).unwrap());
    let adaptive = GpuDecoder::new(
        VarDctSubmissionEngine::with_memory_budget(backend.clone(), budget.clone()).unwrap(),
    );
    let planned = adaptive.open(&data, request(false)).unwrap();
    assert_eq!(
        planned.submission_session().global_modular_memory_stats(),
        Some(global)
    );
    assert_eq!(budget.snapshot().reserved_bytes, 0);
    drop(planned);

    let tight = GpuDecoder::new(
        VarDctSubmissionEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(39).unwrap()),
    );
    assert!(matches!(
        tight.open(&data, request(false)),
        Err(DecodeError::StreamWindowTooSmall {
            limit_bytes: 39,
            minimum_bytes: 40
        })
    ));
    let budget = MemoryBudget::new(NonZeroU64::new(global.total_bytes - 1).unwrap());
    let tight = GpuDecoder::new(
        VarDctSubmissionEngine::with_memory_budget(backend.clone(), budget.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
    );
    assert!(matches!(
        tight.open(&data, request(false)),
        Err(DecodeError::VarDct(
            VarDctDecodeError::MemoryBudgetTooSmall { .. }
        ))
    ));
    assert_eq!(budget.snapshot().reserved_bytes, 0);

    let budget = MemoryBudget::new(NonZeroU64::new(limit).unwrap());
    let decoder = GpuDecoder::new(
        VarDctSubmissionEngine::with_memory_budget(backend.clone(), budget.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(global.stream_bytes).unwrap()),
    );
    let held = budget.try_reserve(limit - global.total_bytes).unwrap();
    let mut abandoned = decoder.open(&data, request(false)).unwrap();
    let mut waiting = decoder.open(&data, request(false)).unwrap();
    abandoned.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert_eq!(budget.snapshot().reserved_bytes, limit);
    assert!(matches!(
        abandoned
            .front_pending_frame()
            .unwrap()
            .unvalidated_gpu_frame(),
        Err(DecodeError::VarDct(
            VarDctDecodeError::UnvalidatedOutputNotSubmitted
        ))
    ));
    let progress = waiting.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert!(matches!(
        progress.backpressure,
        Some(PrefetchBackpressure::Memory(_))
    ));
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while abandoned.submission_session().submissions_per_frame() < 3 {
        assert!(std::time::Instant::now() < deadline);
        assert!(abandoned.poll_next_frame(&mut context).is_pending());
        std::thread::yield_now();
    }
    assert!(abandoned.submission_session().memory_stats().is_none());
    assert_eq!(budget.snapshot().reserved_bytes, limit);
    drop(abandoned);
    drop(held);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while budget.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert_eq!(budget.snapshot().reserved_bytes, 0);
    let frame = pollster::block_on(waiting.next_frame_async())
        .unwrap()
        .unwrap();
    assert_eq!(
        budget.snapshot().reserved_bytes,
        frame.output().outputs[0].buffer.reserved_bytes()
    );
    drop(frame);
    drop(waiting);
    assert_eq!(budget.snapshot().reserved_bytes, 0);
}

#[test]
fn bounded_global_windows_resume_early_and_match_whole_stream_outputs() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for (name, hex) in fixtures() {
        let data = encoded(hex);
        let cap = if name == "transformed" { 1024 } else { 40 };
        let mut whole = None;
        for limit in [u64::MAX, cap] {
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(limit).unwrap()),
            );
            let mut session = if limit == cap {
                open_incremental(&decoder, &data, request(false))
            } else {
                decoder.open(&data, request(false)).unwrap()
            };
            let global = session
                .submission_session()
                .vardct()
                .unwrap()
                .global_modular_memory_stats()
                .unwrap();
            assert!(global.stream_bytes <= limit);
            let frame = if limit == cap {
                pollster::block_on(session.next_frame_async())
            } else {
                session.next_frame()
            }
            .unwrap_or_else(|e| panic!("{name}/{limit}: {e}"))
            .unwrap();
            let readback = ImageReadbackPipeline::new(&backend)
                .submit(frame.output())
                .unwrap()
                .wait()
                .unwrap();
            let bytes = &readback.frame.outputs[0].bytes;
            if let Some(whole) = &whole {
                assert_eq!(bytes, whole, "{name}: windowed output");
            } else {
                whole = Some(bytes.clone());
            }
            let producer = session.submission_session().vardct().unwrap();
            eprintln!(
                "{name} cap={limit}: {} submissions, global stream={}",
                producer.submissions_per_frame(),
                global.stream_bytes
            );
            drop(readback);
            drop(frame);
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn global_modular_entropy_failure_never_exposes_a_color_frame() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let mut data = encoded(include_str!("../../test-data/vardct_extras_rgba.jxl.hex"));
    // Proven global entropy interval from the standalone plane/cursor oracle. Preserve both
    // descriptors and the following LF/AC data so failure is owned by the GPU entropy stage.
    for bit in 885..3436 {
        data[bit / 8] &= !(1 << (bit % 8));
    }
    for cap in [u64::MAX, 40] {
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
        );

        let mut session = decoder.open(&data, request(false)).unwrap();
        session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        assert!(
            session
                .front_pending_frame()
                .unwrap()
                .unvalidated_gpu_frame()
                .is_err()
        );
        let error = pollster::block_on(session.next_frame_async()).unwrap_err();
        assert!(
            matches!(
                error,
                DecodeError::VarDct(VarDctDecodeError::GlobalModularStatus { .. })
            ),
            "{error}"
        );
        assert!(
            session
                .submission_session()
                .vardct()
                .unwrap()
                .memory_stats()
                .is_none()
        );
        drop(session);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}
