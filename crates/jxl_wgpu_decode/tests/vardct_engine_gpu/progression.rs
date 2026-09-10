use super::*;

#[path = "progression/extras.rs"]
mod extras;
#[path = "progression/lf.rs"]
mod lf;
#[path = "progression/sequence.rs"]
mod sequence;
#[path = "progression/sequence_oracle.rs"]
mod sequence_oracle;

fn encoded(hex: &str) -> Vec<u8> {
    let compact = hex.split_whitespace().collect::<String>();
    compact
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn read(backend: &WgpuBackend, frame: &jxl_wgpu::GpuImageFrame) -> Vec<u8> {
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

fn owned(frame: &jxl_wgpu::GpuImageFrame) -> jxl_wgpu::GpuImageFrame {
    jxl_wgpu::GpuImageFrame {
        token: frame.token,
        outputs: frame
            .outputs
            .iter()
            .map(|output| jxl_wgpu::GpuImageOutput {
                id: output.id,
                layout: output.layout.clone(),
                buffer: output.buffer.clone(),
            })
            .collect(),
        changed: frame.changed.clone(),
    }
}

use common::progressive_oracle::*;

fn direct_progressive_cases() -> Vec<(&'static str, Vec<u8>, usize)> {
    vec![
        (
            "raw_progressive",
            common::vardct_progressive_raw_matrix(),
            3,
        ),
        (
            "spectral",
            common::vardct_progressive_spectral().to_vec(),
            3,
        ),
        (
            "quantized",
            common::vardct_progressive_quantized().to_vec(),
            2,
        ),
        ("multilf", common::vardct_progressive_multilf().to_vec(), 3),
        ("jpeg_odd003", common::jpeg_dc_edge_case("003"), 1),
        ("jpeg_odd321", common::jpeg_dc_edge_case("321"), 1),
        ("jpeg_odd111", common::jpeg_dc_edge_case("111"), 1),
        ("jpeg444", common::jpeg_transcode_444().to_vec(), 1),
        ("jpeg422", common::jpeg_transcode_422().to_vec(), 1),
        ("jpeg440", common::jpeg_transcode_440().to_vec(), 1),
        ("jpeg_raw", common::jpeg_transcode_raw_matrix().to_vec(), 1),
        (
            "jpeg_raw_local",
            common::jpeg_transcode_raw_matrix_local().to_vec(),
            1,
        ),
        (
            "jpeg_raw_packets",
            common::jpeg_transcode_raw_matrix_local_packets().to_vec(),
            1,
        ),
        ("gray_progressive", common::vardct_gray("progressive"), 2),
        ("gray_multilf", common::vardct_gray("multilf"), 3),
        ("gray_upsample", common::vardct_gray("upsample"), 3),
        (
            "custom_up4",
            common::with_custom_upsampling_weights(&common::vardct_upsampling("4")),
            3,
        ),
        (
            "custom_up8",
            common::with_custom_upsampling_weights(&common::vardct_upsampling("8")),
            1,
        ),
        ("orientation6", common::vardct_orientation(6), 3),
        ("upsample2", common::vardct_upsampling("2"), 1),
        ("upsample4", common::vardct_upsampling("4"), 3),
        ("upsample8", common::vardct_upsampling("8"), 1),
        (
            "upsample_multilf",
            common::vardct_upsampling("2_multilf"),
            1,
        ),
    ]
}

#[test]
fn gpu_dc_and_ac_images_match_native_libjxl_flushes() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let mut errors = Vec::new();
    for (name, bytes, count) in direct_progressive_cases() {
        let encoded = bytes.as_slice();
        let linear_output = name == "raw_progressive";
        let Some(native) = native_updates(encoded, linear_output) else {
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
            let mut format = PixelFormat::rgb_f32(
                jxl_gpu_formats::RgbChannelOrder::Rgba,
                false,
                vardct_rgb8_format().color_spec,
            );
            if linear_output {
                let jxl_gpu_formats::ColorSpecification::Defined(ref mut spec) = format.color_spec
                else {
                    unreachable!()
                };
                spec.transfer = jxl_gpu_formats::TransferFunction::Linear;
            }
            let mut session = decoder
                .open(
                    encoded,
                    GpuOutputRequest::color(format.clone())
                        .unwrap()
                        .with_progressive_output(true),
                )
                .unwrap();
            let mut actual_stages = Vec::new();
            while let Some(frame) = session.next_update().unwrap() {
                let stage = actual_stages.len();
                let expected = &native[stage];
                assert_eq!(expected.step, stage);
                assert_eq!(frame.is_complete(), expected.complete);
                if let Some(progress) = frame.progression() {
                    assert_eq!(progress.intended_downsampling(), expected.ratio);
                }
                let readback = ImageReadbackPipeline::new(&backend)
                    .submit(frame.output())
                    .unwrap()
                    .wait()
                    .unwrap();
                let actual = &readback.frame.outputs[0].bytes;
                assert_eq!(actual.len(), expected.pixels.len());
                let linear = |value: f32| {
                    if linear_output {
                        return value;
                    }
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
                    let code = |sample: f32| {
                        let sample = sample.clamp(0.0, 1.0);
                        let srgb = if !linear_output {
                            sample
                        } else if sample <= 0.0031308 {
                            sample * 12.92
                        } else {
                            1.055 * sample.powf(1.0 / 2.4) - 0.055
                        };
                        (srgb * 255.0).round() as u32
                    };
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
            assert_eq!(actual_stages.len(), count + 1);
            let mut final_only = decoder
                .open(encoded, GpuOutputRequest::color(format).unwrap())
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
    // byte-identical to final-only GPU decoding. The transplanted raw JPEG matrix intentionally
    // drives extended linear values: final-only decoding has the same 1.915e-3 native difference.
    // Preserve that stress case with an explicit bound; it is not ISO precision evidence.
    // All DC images retain the tighter 1e-5 gate and every stage obeys one-code RGB8 comparison.
    assert!(
        errors.iter().all(|(name, _, stage, complete, error)| *error
            < if *stage == 0 {
                1e-5
            } else if *name == "raw_progressive" {
                2e-3
            } else if *complete {
                1e-4
            } else {
                2e-4
            }),
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
    for (name, bytes, count) in direct_progressive_cases() {
        let encoded = bytes.as_slice();
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
            assert_eq!(held.len(), count + 1, "{name} cap {cap}");
            for (index, frame) in held.iter().enumerate() {
                assert_eq!(frame.metadata, held[0].metadata);
                if index < count {
                    assert_eq!(
                        frame
                            .progression()
                            .unwrap()
                            .completed_passes()
                            .expect("coefficient boundary") as usize,
                        index
                    );
                    assert!(!frame.is_complete());
                    // Constant JPEG-transcode fixtures legitimately have no AC contribution.
                    if !matches!(
                        name,
                        "jpeg444"
                            | "jpeg422"
                            | "jpeg440"
                            | "jpeg_raw"
                            | "jpeg_raw_local"
                            | "jpeg_raw_packets"
                    ) {
                        assert_ne!(
                            pixels[index], pixels[count],
                            "{name}: intermediate image already includes future passes"
                        );
                    }
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
        assert_eq!(
            memory.intermediate_output_bytes,
            2 * memory.output_lease_bytes
        );
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
        let ac = broken.next_update().unwrap().unwrap();
        assert_eq!(
            ac.progression()
                .unwrap()
                .completed_passes()
                .expect("coefficient boundary"),
            1
        );
        drop(ac);
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

#[test]
fn single_entry_dc_survives_deferred_hf_and_final_only_consumers_skip_updates() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    for (name, encoded) in [
        ("custom", common::vardct_upsampling("8_custom")),
        ("thin", common::vardct_upsampling("4_thin")),
        ("single", common::vardct_upsampling("8_single")),
        ("gray", common::vardct_gray("single")),
        ("gray_jpeg", common::vardct_gray("jpeg")),
        ("oriented_jpeg", common::vardct_oriented_jpeg()),
    ] {
        let mut whole = None;
        for cap in [u64::MAX, 40] {
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
            );
            let request = || {
                GpuOutputRequest::color(vardct_rgb8_format())
                    .unwrap()
                    .with_progressive_output(true)
            };
            let read = |frame: &jxl_wgpu::GpuImageFrame| {
                ImageReadbackPipeline::new(&backend)
                    .submit(frame)
                    .unwrap()
                    .wait()
                    .unwrap()
                    .frame
                    .outputs[0]
                    .bytes
                    .clone()
            };
            let mut session = decoder.open(&encoded, request()).unwrap();
            let memory = session
                .submission_session()
                .vardct()
                .unwrap()
                .memory_stats()
                .unwrap();
            assert!(memory.deferred_hf_coefficients, "{name}");
            let dc = pollster::block_on(session.next_update_async())
                .unwrap()
                .unwrap();
            assert_eq!(
                dc.progression()
                    .unwrap()
                    .completed_passes()
                    .expect("coefficient boundary"),
                0
            );
            assert_eq!(dc.progression().unwrap().intended_downsampling(), 8);
            let dc_pixels = read(dc.output());
            let final_frame = session.next_frame().unwrap().unwrap();
            assert!(final_frame.is_complete());
            assert_eq!(dc.metadata, final_frame.metadata);
            let final_pixels = read(final_frame.output());
            assert_eq!(read(dc.output()), dc_pixels);
            assert!(session.next_update().unwrap().is_none());
            let actual = [dc_pixels.clone(), final_pixels.clone()];
            if let Some(expected) = &whole {
                assert_eq!(&actual, expected, "{name}");
            } else {
                whole = Some(actual);
            }
            drop(final_frame);
            drop(dc);
            drop(session);

            // Going straight to final completion must drive the real host continuation even
            // while an unconsumed DC map exists.
            let mut skipped = decoder.open(&encoded, request()).unwrap();
            let frame = pollster::block_on(skipped.next_frame_async())
                .unwrap()
                .unwrap();
            assert_eq!(read(frame.output()), final_pixels);
            assert!(skipped.next_update().unwrap().is_none());
            drop(frame);
            drop(skipped);

            // Damage only the last byte of the HF tail; completed LF/HF metadata and the
            // DC image must remain usable when the following descriptor/coefficient stage fails.
            let mut damaged = encoded.clone();
            let end = damaged.len();
            damaged[end - 1] = 0xff;
            let mut broken = decoder.open(&damaged, request()).unwrap();
            let dc = broken
                .next_update()
                .unwrap_or_else(|error| panic!("{name} cap {cap}: {error:?}"))
                .unwrap();
            assert_eq!(read(dc.output()), dc_pixels, "{name}: later HF changed DC");
            let error = broken.next_update().unwrap_err();
            assert!(matches!(error, DecodeError::VarDct(_)), "{name}: {error:?}");
            assert!(matches!(
                broken.next_update(),
                Err(DecodeError::SessionPoisoned)
            ));
            drop(broken);
            drain_gpu(&backend, memory.output_lease_bytes);
            assert_eq!(
                decoder.engine().in_flight_memory_stats().reserved_bytes,
                memory.output_lease_bytes
            );
            assert_eq!(read(dc.output()), dc_pixels);
            drop(dc);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}
