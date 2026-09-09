use super::*;

struct NativeUpdate {
    step: usize,
    ratio: u32,
    complete: bool,
    pixels: Vec<u8>,
}

fn native_updates(encoded: &[u8]) -> Option<Vec<NativeUpdate>> {
    use std::process::Command;
    static BINARY: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let binary = BINARY
        .get_or_init(|| {
            let flags = Command::new("pkg-config")
                .args(["--cflags", "--libs", "libjxl"])
                .output()
                .ok()?;
            if !flags.status.success() {
                return None;
            }
            let binary =
                std::env::temp_dir().join(format!("jxl-wgpu-pass-oracle-{}", std::process::id()));
            let compiled = Command::new("cc")
                .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-O2"])
                .arg(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("test-data/decode_progressive.c"),
                )
                .args(
                    std::str::from_utf8(&flags.stdout)
                        .unwrap()
                        .split_whitespace(),
                )
                .arg("-o")
                .arg(&binary)
                .output()
                .unwrap();
            assert!(
                compiled.status.success(),
                "{}",
                String::from_utf8_lossy(&compiled.stderr)
            );
            Some(binary)
        })
        .as_ref()?;
    let directory = std::env::temp_dir().join(format!(
        "jxl-wgpu-pass-input-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&directory).unwrap();
    let input = directory.join("image.jxl");
    let prefix = directory.join("snapshot");
    std::fs::write(&input, encoded).unwrap();
    let decoded = Command::new(binary)
        .arg(&input)
        .arg(encoded.len().to_string())
        .arg(&prefix)
        .output()
        .unwrap();
    assert!(
        decoded.status.success(),
        "{}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    let mut updates = Vec::new();
    for line in std::str::from_utf8(&decoded.stdout).unwrap().lines() {
        let fields = line.split(',').collect::<Vec<_>>();
        if !matches!(fields[0], "progress" | "final") {
            continue;
        }
        assert_eq!(fields[1], "0");
        assert_eq!(fields[5], "0", "nonfinite oracle output");
        let step = fields[2].parse().unwrap();
        updates.push(NativeUpdate {
            step,
            ratio: fields[3].parse().unwrap(),
            complete: fields[0] == "final",
            pixels: std::fs::read(
                directory.join(format!("snapshot-frame0-step{step}-{}.f32", fields[0])),
            )
            .unwrap(),
        });
    }
    std::fs::remove_dir_all(directory).unwrap();
    Some(updates)
}

#[test]
fn gpu_intermediate_pass_pixels_match_native_libjxl_flushes() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let mut errors = Vec::new();
    for (name, encoded, count) in [
        ("spectral", common::vardct_progressive_spectral(), 3),
        ("quantized", common::vardct_progressive_quantized(), 2),
        ("multilf", common::vardct_progressive_multilf(), 3),
    ] {
        let Some(native) = native_updates(encoded) else {
            eprintln!("native libjxl progressive oracle unavailable");
            return;
        };
        assert_eq!(native.len(), count + 1, "{name}: DC, AC passes, final");
        let mut whole = None;
        for cap in [u64::MAX, 256] {
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
            );
            let format = PixelFormat::rgb_f32(
                jxl_gpu_formats::RgbChannelOrder::Rgba,
                false,
                vardct_rgb8_format().color_spec,
            );
            let mut session = decoder
                .open(
                    encoded,
                    GpuOutputRequest::color(format)
                        .unwrap()
                        .with_progressive_output(true),
                )
                .unwrap();
            let mut actual_stages = Vec::new();
            while let Some(frame) = session.next_update().unwrap() {
                let stage = actual_stages.len() + 1;
                let expected = &native[stage];
                assert_eq!(expected.step, stage);
                assert_eq!(frame.is_complete(), expected.complete);
                if let Some(progress) = frame.progression() {
                    assert_eq!(progress.intended_downsampling, expected.ratio);
                }
                let readback = ImageReadbackPipeline::new(&backend)
                    .submit(frame.output())
                    .unwrap()
                    .wait()
                    .unwrap();
                let actual = &readback.frame.outputs[0].bytes;
                assert_eq!(actual.len(), expected.pixels.len());
                let linear = |value: f32| {
                    let magnitude = value.abs();
                    (if magnitude <= 0.04045 {
                        magnitude / 12.92
                    } else {
                        ((magnitude + 0.055) / 1.055).powf(2.4)
                    })
                    .copysign(value)
                };
                let mut max_error = 0.0_f32;
                let mut max_sample = (0.0_f32, 0.0_f32);
                let mut normalized_error = 0.0_f32;
                let mut max_code_error = 0_u32;
                for (actual, expected) in
                    actual.chunks_exact(4).zip(expected.pixels.chunks_exact(4))
                {
                    let actual = f32::from_le_bytes(actual.try_into().unwrap());
                    let expected = f32::from_le_bytes(expected.try_into().unwrap());
                    assert!(actual.is_finite() && expected.is_finite());
                    let difference = (linear(actual) - linear(expected)).abs();
                    if difference > max_error {
                        max_error = difference;
                        max_sample = (linear(actual), linear(expected));
                    }
                    normalized_error =
                        normalized_error.max(difference / linear(expected).abs().max(1.0));
                    let code = |sample: f32| (sample.clamp(0.0, 1.0) * 255.0).round() as u32;
                    max_code_error = max_code_error.max(code(actual).abs_diff(code(expected)));
                }
                eprintln!(
                    "{name} cap {cap} step {stage} linear maxAE {max_error}, sample {max_sample:?}, normalized {normalized_error}"
                );
                assert!(
                    max_code_error <= 1,
                    "{name} step {stage}: native RGB8 code error {max_code_error}"
                );
                errors.push((name, cap, stage, expected.complete, max_error));
                actual_stages.push(actual.clone());
            }
            assert_eq!(actual_stages.len(), count);
            let mut final_only = decoder
                .open(
                    encoded,
                    GpuOutputRequest::color(PixelFormat::rgb_f32(
                        jxl_gpu_formats::RgbChannelOrder::Rgba,
                        false,
                        vardct_rgb8_format().color_spec,
                    ))
                    .unwrap(),
                )
                .unwrap();
            let frame = final_only.next_frame().unwrap().unwrap();
            let readback = ImageReadbackPipeline::new(&backend)
                .submit(frame.output())
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(
                actual_stages.last().unwrap(),
                &readback.frame.outputs[0].bytes,
                "{name}: intermediate rendering changed final output"
            );
            if let Some(expected) = &whole {
                assert_eq!(&actual_stages, expected);
            } else {
                whole = Some(actual_stages);
            }
        }
    }
    // Native and GPU inverse transforms do not sum F32 terms identically. The quantized first
    // pass measures 1.28e-4 in linear light; its fully decoded image remains below 1e-4 and is
    // byte-identical to final-only GPU decoding. Both also obey the existing one-code RGB8 limit.
    assert!(
        errors
            .iter()
            .all(|(_, _, _, complete, error)| *error < if *complete { 1e-4 } else { 2e-4 }),
        "native linear errors: {errors:?}"
    );
}

#[test]
fn completed_gpu_passes_return_immutable_images_before_the_final_frame() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    for (name, encoded, count) in [
        ("spectral", common::vardct_progressive_spectral(), 3),
        ("quantized", common::vardct_progressive_quantized(), 2),
        ("multilf", common::vardct_progressive_multilf(), 3),
    ] {
        let mut whole = None;
        for cap in [u64::MAX, 256] {
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
            );
            let request = GpuOutputRequest::color(vardct_rgb8_format())
                .unwrap()
                .with_progressive_output(true)
                .with_max_frame_slots(NonZeroUsize::new(1).unwrap());
            let mut session = if cap == 256 {
                let mut stream = decoder.stream(request).unwrap();
                let mut transport = ContainerStreamScanner::new(decoder.container_stream_limits());
                for chunk in encoded.chunks(37) {
                    for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
                        stream.push_transport_event(&event).unwrap();
                    }
                }
                for event in transport.finish_input().unwrap() {
                    stream.push_transport_event(&event).unwrap();
                }
                stream.finish().unwrap()
            } else {
                decoder.open(encoded, request).unwrap()
            };
            let mut held = Vec::new();
            let mut pixels = Vec::new();
            while let Some(frame) = if cap == u64::MAX {
                session.next_update().unwrap()
            } else {
                pollster::block_on(session.next_update_async()).unwrap()
            } {
                let readback = ImageReadbackPipeline::new(&backend)
                    .submit(frame.output())
                    .unwrap()
                    .wait()
                    .unwrap();
                pixels.push(readback.frame.outputs[0].bytes.clone());
                held.push(frame);
            }
            assert_eq!(held.len(), count, "{name} cap {cap}");
            for (index, frame) in held.iter().enumerate() {
                assert_eq!(frame.metadata, held[0].metadata);
                if index + 1 < count {
                    assert_eq!(
                        frame.progression().unwrap().completed_passes as usize,
                        index + 1
                    );
                    assert!(!frame.is_complete());
                    assert_ne!(
                        pixels[index],
                        pixels[count - 1],
                        "{name}: intermediate image already includes future passes"
                    );
                } else {
                    assert!(frame.is_complete());
                }
                let reread = ImageReadbackPipeline::new(&backend)
                    .submit(frame.output())
                    .unwrap()
                    .wait()
                    .unwrap();
                assert_eq!(
                    reread.frame.outputs[0].bytes, pixels[index],
                    "{name}: retained update was overwritten"
                );
            }
            assert_eq!(session.frames_submitted(), 1);
            assert_eq!(session.active_frame_slots(), 1);
            if let Some(expected) = &whole {
                assert_eq!(
                    &pixels, expected,
                    "{name}: pass output changed under windowed execution"
                );
            } else {
                whole = Some(pixels);
            }
            drop(held);
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

fn drain_gpu(backend: &WgpuBackend, retained_bytes: u64) {
    let fence = backend.queue().submit(std::iter::empty());
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .unwrap();
    // Poll workers can still be returning from callbacks when the device fence completes.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while backend.transient_memory_budget().snapshot().reserved_bytes != retained_bytes
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
}

#[test]
fn pass_outputs_keep_validation_local_and_admission_and_cancellation_release_their_budgets() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let encoded = common::vardct_progressive_quantized();
    let inventory = jxl_gpu_bitstream::parse(encoded, ParseLimits::default())
        .unwrap()
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    let packet = BoundedVarDctPacketPlan::parse(encoded, &inventory).unwrap();
    let range = packet
        .hf_coefficients
        .as_ref()
        .unwrap()
        .passes
        .last()
        .unwrap()
        .pass_groups
        .iter()
        .max_by_key(|range| range.length)
        .unwrap();
    let end = (range.end().unwrap() / 8) as usize;
    let mut damaged = encoded.to_vec();
    damaged[end - 32..end].fill(0xff);
    for cap in [u64::MAX, 256] {
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
        );
        let request = || {
            GpuOutputRequest::color(vardct_rgb8_format())
                .unwrap()
                .with_progressive_output(true)
                .with_max_frame_slots(NonZeroUsize::new(1).unwrap())
        };
        let mut broken = decoder.open(&damaged, request()).unwrap();
        let memory = broken
            .submission_session()
            .vardct()
            .unwrap()
            .memory_stats()
            .unwrap();
        assert_eq!(memory.intermediate_output_bytes, memory.output_lease_bytes);
        assert!(memory.intermediate_transient_bytes > memory.validation_staging_bytes);
        let prior = broken.next_update().unwrap().unwrap();
        assert!(!prior.is_complete());
        let before = ImageReadbackPipeline::new(&backend)
            .submit(prior.output())
            .unwrap()
            .wait()
            .unwrap()
            .frame
            .outputs[0]
            .bytes
            .clone();
        assert!(matches!(
            broken.next_update(),
            Err(DecodeError::VarDct(VarDctDecodeError::HfCoefficientGpu(_)))
        ));
        assert!(matches!(
            broken.next_frame(),
            Err(DecodeError::SessionPoisoned)
        ));
        drop(broken);
        drain_gpu(&backend, memory.output_lease_bytes);
        assert_eq!(
            decoder.engine().in_flight_memory_stats().reserved_bytes,
            memory.output_lease_bytes
        );
        let after = ImageReadbackPipeline::new(&backend)
            .submit(prior.output())
            .unwrap()
            .wait()
            .unwrap()
            .frame
            .outputs[0]
            .bytes
            .clone();
        assert_eq!(before, after);
        drop(prior);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

        let mut admitted = decoder.open(encoded, request()).unwrap();
        let blocker = backend
            .transient_memory_budget()
            .try_reserve(
                backend.transient_memory_budget().snapshot().limit_bytes - memory.total_frame_bytes
                    + 1,
            )
            .unwrap();
        let progress = admitted.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        assert!(matches!(
            progress.backpressure,
            Some(PrefetchBackpressure::Memory(_))
        ));
        assert_eq!(admitted.frames_submitted(), 0);
        assert_eq!(admitted.active_frame_slots(), 0);
        assert_eq!(
            decoder.engine().in_flight_memory_stats().reserved_bytes,
            backend.transient_memory_budget().snapshot().limit_bytes - memory.total_frame_bytes + 1
        );
        drop(blocker);
        let first = admitted.next_update().unwrap().unwrap();
        drop(admitted);
        drain_gpu(&backend, memory.output_lease_bytes);
        assert_eq!(
            decoder.engine().in_flight_memory_stats().reserved_bytes,
            memory.output_lease_bytes
        );
        assert_eq!(
            ImageReadbackPipeline::new(&backend)
                .submit(first.output())
                .unwrap()
                .wait()
                .unwrap()
                .frame
                .outputs[0]
                .bytes,
            before
        );
        drop(first);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

        let mut abandoned = decoder.open(encoded, request()).unwrap();
        abandoned.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        drop(abandoned);
        drain_gpu(&backend, 0);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}
