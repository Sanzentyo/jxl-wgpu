use super::*;
use jxl_wgpu_decode::FrameProgression;

fn lf1_rust_pixels(encoded: &[u8], inventory: &jxl_gpu_bitstream::CodestreamInventory) -> Vec<u8> {
    let last_lf = inventory
        .frames
        .iter()
        .rposition(|frame| frame.lf_level == 1)
        .unwrap();
    let end = inventory.frames[last_lf + 1].header_bits.offset as usize / 8;
    let mut input = &encoded[..end];
    let decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let ProcessingResult::Complete {
        result: mut decoder,
    } = decoder.process(&mut input, None).unwrap()
    else {
        panic!("complete image header");
    };
    decoder.set_pixel_format(JxlPixelFormat::rgb8(0));
    let size = decoder.basic_info().size;
    let ProcessingResult::NeedsMoreInput {
        fallback: mut decoder,
        ..
    } = decoder.process(&mut input, None).unwrap()
    else {
        panic!("LF prefix cannot complete a visible frame header");
    };
    let mut pixels = vec![0; size.0 * size.1 * 3];
    assert!(
        decoder
            .flush_pixels(
                &mut [JxlOutputBuffer::new(&mut pixels, size.1, size.0 * 3)],
                None
            )
            .unwrap()
    );
    pixels
}

#[test]
fn lf_dependencies_publish_immutable_images_before_consumers_and_match_oracles() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    for (name, encoded) in [
        ("dc_ac", corpus::vardct_progressive_dc_ac().to_vec()),
        ("gray_oriented", corpus::vardct_gray("dc_ac")),
        (
            "custom",
            corpus::with_custom_upsampling_weights(corpus::vardct_progressive_dc_ac()),
        ),
        ("noise", {
            let text = include_str!("../../../test-data/noise/lf_progressive_ac.jxl.hex")
                .split_whitespace()
                .collect::<String>();
            text.as_bytes()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
                .collect()
        }),
    ] {
        let inventory = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let rust_lf = lf1_rust_pixels(&encoded, &inventory);
        let native = native_updates(&encoded, true);
        let coefficient_images = inventory.frames.last().unwrap().num_passes as usize + 1;
        let mut expected_whole = None;
        for cap in [u64::MAX, 40] {
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
            let jxl_gpu_formats::ColorSpecification::Defined(ref mut color) = format.color_spec
            else {
                unreachable!()
            };
            color.transfer = jxl_gpu_formats::TransferFunction::Linear;
            let request = || {
                GpuOutputRequest::color(format.clone())
                    .unwrap()
                    .with_progressive_output(true)
                    .with_max_frame_slots(NonZeroUsize::new(1).unwrap())
            };
            let mut session = if cap == u64::MAX {
                decoder.open(&encoded, request()).unwrap()
            } else {
                open_incremental(&decoder, &encoded, request())
            };
            let mut held = Vec::new();
            let mut pixels = Vec::new();
            while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                eprintln!("{name} cap {cap}: {:?}", update.progression());
                let actual = read(&backend, update.output());
                assert!(
                    actual
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|v| f32::from_le_bytes(*v).is_finite())
                );
                if let Some(FrameProgression::LowFrequency { level: 1, .. }) = update.progression()
                {
                    let rgb = actual
                        .as_chunks::<16>()
                        .0
                        .iter()
                        .flat_map(|p| {
                            p[..12].as_chunks::<4>().0.iter().map(|v| {
                                let value = f32::from_le_bytes(*v).clamp(0.0, 1.0);
                                let srgb = if value <= 0.0031308 {
                                    value * 12.92
                                } else {
                                    1.055 * value.powf(1.0 / 2.4) - 0.055
                                };
                                (srgb * 255.0).round() as u8
                            })
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(rgb.len(), rust_lf.len());
                    let error = maximum_error(&rgb, &rust_lf);
                    eprintln!("{name} LF1 Rust code error {error}");
                    assert!(error <= 1, "{name} LF1 Rust code error {error}");
                } else if let Some(native) = native.as_ref().filter(|_| {
                    !matches!(
                        update.progression(),
                        Some(FrameProgression::LowFrequency { .. })
                    )
                }) {
                    let stage = held.len() - 2;
                    let expected = &native[stage];
                    assert_eq!(update.is_complete(), expected.complete);
                    assert_eq!(actual.len(), expected.pixels.len());
                    let error = actual
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(expected.pixels.as_chunks::<4>().0.iter())
                        .map(|(a, b)| {
                            let reference = f32::from_le_bytes(*b);
                            assert!(reference.is_finite());
                            (f32::from_le_bytes(*a) - reference).abs()
                        })
                        .fold(0_f32, f32::max);
                    eprintln!("{name} stage {stage}: native linear error {error}");
                    let limit = if stage == 0 {
                        1e-5
                    } else if expected.complete {
                        1e-4
                    } else {
                        2e-4
                    };
                    assert!(error < limit, "{name} stage {stage}: {error}");
                }
                pixels.push(actual);
                held.push(update);
            }
            assert_eq!(held.len(), 2 + coefficient_images);
            assert_eq!(
                held[0].progression(),
                Some(FrameProgression::LowFrequency {
                    physical_frame_index: 0,
                    level: 2
                })
            );
            assert_eq!(
                held[1].progression(),
                Some(FrameProgression::LowFrequency {
                    physical_frame_index: 1,
                    level: 1
                })
            );
            assert_eq!(session.frames_submitted(), 1);
            assert_eq!(session.active_frame_slots(), 1);
            for (frame, data) in held.iter().zip(&pixels) {
                assert_eq!(frame.metadata, held.last().unwrap().metadata);
                assert_eq!(read(&backend, frame.output()), *data);
            }
            let mut final_only = decoder
                .open(&encoded, GpuOutputRequest::color(format).unwrap())
                .unwrap();
            let final_frame = final_only.next_frame().unwrap().unwrap();
            assert_eq!(
                read(&backend, final_frame.output()),
                *pixels.last().unwrap()
            );
            if let Some(expected) = &expected_whole {
                assert_eq!(&pixels, expected);
            } else {
                expected_whole = Some(pixels);
            }
            drop(final_frame);
            drop(final_only);
            drop(held);
            drop(session);
            drain_gpu(&backend, 0);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}

#[test]
fn lf_publication_defers_admission_and_survives_late_corruption_and_cancellation() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let encoded = corpus::vardct_progressive_dc_ac();
    let inventory = jxl_gpu_bitstream::parse(encoded, ParseLimits::default())
        .unwrap()
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
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
                .with_max_frame_slots(NonZeroUsize::new(1).unwrap())
        };
        let mut reference = decoder.open(encoded, request()).unwrap();
        let a = reference.next_update().unwrap().unwrap();
        let b = reference.next_update().unwrap().unwrap();
        let lf_pixels = [read(&backend, a.output()), read(&backend, b.output())];
        let output_bytes = a.output().outputs[0].buffer.as_wgpu_buffer().size();
        let final_pixels = read(&backend, reference.next_frame().unwrap().unwrap().output());
        drop(a);
        drop(b);
        drop(reference);
        for boundary in 0..=2 {
            let mut session = open_incremental(&decoder, encoded, request());
            session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
            let mut held = None;
            for _ in 0..boundary {
                held = session.next_update().unwrap();
            }
            drop(session);
            drain_gpu(&backend, if boundary == 0 { 0 } else { output_bytes });
            assert_eq!(
                decoder.engine().in_flight_memory_stats().reserved_bytes,
                if boundary == 0 { 0 } else { output_bytes }
            );
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
            if let Some(held) = held {
                assert_eq!(read(&backend, held.output()), lf_pixels[boundary - 1]);
            }
        }
        for completed_lf in [1, 2] {
            let mut session = decoder.open(encoded, request()).unwrap();
            let mut held = None;
            for _ in 0..completed_lf {
                held = session.next_update().unwrap();
            }
            let held = held.unwrap();
            // The next physical producer has not been admitted merely to return this LF image.
            let memory = backend.transient_memory_budget();
            let blocker = memory
                .try_reserve(memory.snapshot().available_bytes - 1)
                .unwrap();
            assert!(session.next_update().is_err());
            assert!(matches!(
                session.next_update(),
                Err(DecodeError::SessionPoisoned)
            ));
            drop(blocker);
            drop(session);
            drain_gpu(&backend, output_bytes);
            assert_eq!(memory.snapshot().reserved_bytes, output_bytes);
            assert_eq!(read(&backend, held.output()), lf_pixels[completed_lf - 1]);
            drop(held);

            let mut corrupt = encoded.to_vec();
            let end = if completed_lf == 1 {
                inventory.frames[2].header_bits.offset as usize / 8
            } else {
                encoded.len()
            };
            corrupt[end - 8..end].fill(0xff);
            let mut broken = decoder.open(&corrupt, request()).unwrap();
            let mut held = Vec::new();
            for expected in &lf_pixels[..completed_lf] {
                let update = broken.next_update().unwrap().unwrap();
                assert_eq!(read(&backend, update.output()), *expected);
                held.push(update);
            }
            loop {
                match broken.next_update() {
                    Ok(Some(update)) => {
                        assert!(!update.is_complete(), "corrupt tail returned a final frame")
                    }
                    Err(_) => break,
                    Ok(None) => panic!("corrupt stream completed"),
                }
            }
            assert!(matches!(
                broken.next_frame(),
                Err(DecodeError::SessionPoisoned)
            ));
            drop(broken);
            drain_gpu(&backend, output_bytes * completed_lf as u64);
            assert_eq!(
                decoder.engine().in_flight_memory_stats().reserved_bytes,
                output_bytes * completed_lf as u64
            );
            for (frame, expected) in held.iter().zip(&lf_pixels) {
                assert_eq!(read(&backend, frame.output()), *expected);
            }
            drop(held);
        }
        let mut skipped = decoder.open(encoded, request()).unwrap();
        assert_eq!(
            read(
                &backend,
                pollster::block_on(skipped.next_frame_async())
                    .unwrap()
                    .unwrap()
                    .output()
            ),
            final_pixels
        );
        assert!(skipped.next_update().unwrap().is_none());
        drop(skipped);
        drain_gpu(&backend, 0);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}
