use super::*;
use std::num::NonZeroUsize;

use common::progressive_oracle::native_updates_with_spots;
use jxl_gpu_bitstream::FrameSectionKind;
use jxl_wgpu_decode::{OrientationPolicy, PrefetchBackpressure, VarDctDecodeError};

fn request(depth: SampleBitDepth, channel: u32) -> GpuOutputRequest {
    scalar(match depth {
        SampleBitDepth::Float { .. } => NumericSampleMapping::NativeFloat,
        SampleBitDepth::Integer { .. } => NumericSampleMapping::NormalizedUnsigned,
    })
    .with_color_channel(channel)
    .unwrap()
    .with_progressive_output(true)
    .with_max_frame_slots(NonZeroUsize::new(1).unwrap())
}

fn native_stages(
    data: &[u8],
    inventory: &CodestreamInventory,
    keep: bool,
) -> Option<Vec<Vec<u32>>> {
    let frame = &inventory.frames[0];
    assert_eq!(inventory.frames.len(), 1);
    let parsed = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let codestream = parsed.codestream();
    let mut stages = Vec::new();
    for completed in 0..frame.num_passes {
        let end = frame
            .sections
            .iter()
            .filter_map(|section| match section.kind {
                FrameSectionKind::PassGroup { pass_index, .. } if pass_index >= completed => {
                    Some(section.bytes.offset as usize)
                }
                _ => None,
            })
            .min()
            .unwrap();
        let native = native_updates_with_spots(&codestream[..end], false, keep, true, false)?;
        stages.push(
            native
                .last()
                .unwrap()
                .pixels
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        );
    }
    let native = native_updates_with_spots(data, false, keep, false, false)?;
    stages.push(
        native
            .last()
            .unwrap()
            .pixels
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    );
    Some(stages)
}

#[test]
fn numeric_color_refinements_match_native_prefixes_and_final_only_delivery() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for name in [
        "vardct_extras_rgba_progressive",
        "floating/vardct_extras_float_squeeze",
    ] {
        let (data, inventory) = fixture(name);
        for keep in [false, true] {
            let Some(expected) = native_stages(&data, &inventory, keep) else {
                eprintln!("skipping numeric prefix oracle: native libjxl is unavailable");
                return;
            };
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(
                        NonZeroU64::new(if keep { 40 } else { u64::MAX }).unwrap(),
                    ),
            );
            for channel in 0..3 {
                let request = request(inventory.image_header.bit_depth, channel)
                    .with_orientation_policy(if keep {
                        OrientationPolicy::Keep
                    } else {
                        OrientationPolicy::Apply
                    })
                    .with_alpha_output_policy(AlphaOutputPolicy::Associated);
                let mut session = open_fragmented(&decoder, &data, request.clone());
                let mut held = Vec::new();
                while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                    let stage = held.len();
                    let actual = read(&backend, &update.output().outputs[0]);
                    let reference = &expected[stage];
                    assert_eq!(actual.len() * 4, reference.len());
                    let mut max = 0_f32;
                    for (&actual, pixel) in actual.iter().zip(reference.chunks_exact(4)) {
                        let actual = f32::from_bits(actual);
                        let reference = f32::from_bits(pixel[channel as usize]);
                        assert!(actual.is_finite() && reference.is_finite());
                        max = max.max((actual - reference).abs() / reference.abs().max(1.0));
                    }
                    assert!(
                        max < if stage == 0 { 2e-5 } else { 6e-4 },
                        "{name}/{channel}/stage{stage}/keep{keep}: {max}"
                    );
                    assert_eq!(
                        update.progression().and_then(|p| p.completed_passes()),
                        (stage + 1 != expected.len()).then_some(stage as u8)
                    );
                    held.push((update, actual));
                }
                assert_eq!(held.len(), expected.len());
                let mut final_only = decoder
                    .open(&data, request.with_progressive_output(false))
                    .unwrap();
                let final_frame = final_only.next_frame().unwrap().unwrap();
                assert_eq!(
                    read(&backend, &final_frame.output().outputs[0]),
                    held.last().unwrap().1
                );
                for (update, bytes) in &held {
                    assert_eq!(update.metadata, final_frame.metadata);
                    assert_eq!(read(&backend, &update.output().outputs[0]), *bytes);
                }
                drop((session, final_only, final_frame, held));
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    0
                );
            }
        }
    }
}

fn drain(backend: &WgpuBackend, retained: u64) {
    backend
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while backend.transient_memory_budget().snapshot().reserved_bytes != retained
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        retained
    );
}

#[test]
fn numeric_refinement_cancellation_corruption_and_admission_preserve_owned_images() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let (data, inventory) = fixture("vardct_extras_rgba_progressive");
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
    );
    let request = request(inventory.image_header.bit_depth, 1);
    let mut reference = decoder.open(&data, request.clone()).unwrap();
    let mut expected = Vec::new();
    while let Some(update) = reference.next_update().unwrap() {
        expected.push(read(&backend, &update.output().outputs[0]));
    }
    drop(reference);
    for completed in 0..expected.len() {
        let mut session = open_fragmented(&decoder, &data, request.clone());
        session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        let mut held = None;
        for _ in 0..completed {
            held = session.next_update().unwrap();
        }
        drop(session);
        let retained = held
            .as_ref()
            .map_or(0, |u| u.output().outputs[0].buffer.size());
        drain(&backend, retained);
        if let Some(update) = &held {
            assert_eq!(
                read(&backend, &update.output().outputs[0]),
                expected[completed - 1]
            );
        }
        drop(held);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
        drain(&backend, 0);
    }
    let frame = &inventory.frames[0];
    let pass = frame.num_passes - 1;
    let packet = frame
        .sections
        .iter()
        .filter(|section| {
            matches!(section.kind,
        FrameSectionKind::PassGroup { pass_index, .. } if pass_index == pass)
        })
        .max_by_key(|section| section.bytes.length)
        .unwrap();
    let mut damaged = data.clone();
    let end = (packet.bytes.offset + packet.bytes.length) as usize;
    damaged[end - (packet.bytes.length as usize).min(16)..end].fill(0xff);
    let mut session = open_fragmented(&decoder, &damaged, request.clone());
    let mut held = Vec::new();
    for expected in &expected[..=pass as usize] {
        let update = session.next_update().unwrap().unwrap();
        assert_eq!(read(&backend, &update.output().outputs[0]), *expected);
        held.push(update);
    }
    assert!(matches!(
        session.next_update(),
        Err(Error::VarDct(VarDctDecodeError::HfCoefficientGpu(_)))
    ));
    assert!(matches!(session.next_frame(), Err(Error::SessionPoisoned)));
    drop(session);
    drain(
        &backend,
        held.iter()
            .map(|u| u.output().outputs[0].buffer.size())
            .sum(),
    );
    for (update, expected) in held.iter().zip(&expected) {
        assert_eq!(read(&backend, &update.output().outputs[0]), *expected);
    }
    drop(held);
    drain(&backend, 0);
    let mut session = decoder.open(&data, request).unwrap();
    let budget = backend.transient_memory_budget();
    let blocker = budget
        .try_reserve(budget.snapshot().available_bytes)
        .unwrap();
    let prefetched = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert!(matches!(
        prefetched.backpressure,
        Some(PrefetchBackpressure::Memory(_))
    ));
    assert_eq!(session.frames_submitted(), 0);
    drop(blocker);
    let first = session.next_update().unwrap().unwrap();
    let final_frame = session.next_frame().unwrap().unwrap();
    assert_eq!(read(&backend, &first.output().outputs[0]), expected[0]);
    assert_eq!(
        read(&backend, &final_frame.output().outputs[0]),
        *expected.last().unwrap()
    );
    drop((first, final_frame, session));
    drain(&backend, 0);
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        0
    );
}

#[test]
fn cropped_lf_numeric_previews_select_original_color_after_reference_composition() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
    );
    for name in [
        "lf_geometry/shifted_float_vardct.crop_left",
        "lf_geometry/shifted_integer_modular.crop_top",
    ] {
        let (data, inventory) = fixture(name);
        assert!(inventory.frames.last().unwrap().have_crop);
        let color_request = GpuOutputRequest::color(PixelFormat::rgb_f32(
            RgbChannelOrder::Rgba,
            false,
            jxl_wgpu_decode::vardct_rgb8_format().color_spec,
        ))
        .unwrap()
        .with_progressive_output(true)
        .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
        .with_spot_color_policy(SpotColorPolicy::Preserve);
        let mut color = decoder.open(&data, color_request).unwrap();
        let mut stages = Vec::new();
        while let Some(update) = color.next_update().unwrap() {
            stages.push((
                update.progression(),
                update.metadata.clone(),
                read(&backend, &update.output().outputs[0]),
            ));
        }
        drop(color);
        assert!(stages.len() >= 3);
        let Some(reference) = oracle::libjxl_output(&data, &["--preserve-alpha"]) else {
            eprintln!("skipping LF numeric final oracle: libjxl is unavailable");
            return;
        };
        let pixels = inventory.image_header.width as usize * inventory.image_header.height as usize;
        let reference = &reference[..pixels * 4];
        for channel in 0..3 {
            let request = request(inventory.image_header.bit_depth, channel);
            let mut session = open_fragmented(&decoder, &data, request.clone());
            let mut held = Vec::new();
            while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                let (progression, metadata, rgb) = &stages[held.len()];
                assert_eq!(update.progression(), *progression);
                assert_eq!(update.metadata, *metadata);
                let actual = read(&backend, &update.output().outputs[0]);
                assert_eq!(actual.len() * 4, rgb.len());
                for (&word, pixel) in actual.iter().zip(rgb.chunks_exact(4)) {
                    let actual = f32::from_bits(word);
                    let expected = f32::from_bits(pixel[channel as usize]);
                    assert!(actual.is_finite() && expected.is_finite());
                    assert!((actual - expected).abs() / expected.abs().max(1.0) < 5e-6);
                }
                if update.is_complete() {
                    for (&word, pixel) in actual.iter().zip(reference.chunks_exact(4)) {
                        let expected = pixel[channel as usize];
                        assert!(expected.is_finite());
                        assert!(
                            (f32::from_bits(word) - expected).abs() / expected.abs().max(1.0)
                                < 1e-3
                        );
                    }
                }
                held.push((update, actual));
            }
            assert_eq!(held.len(), stages.len());
            let mut final_only = decoder
                .open(&data, request.with_progressive_output(false))
                .unwrap();
            let frame = final_only.next_frame().unwrap().unwrap();
            assert_eq!(
                read(&backend, &frame.output().outputs[0]),
                held.last().unwrap().1
            );
            for (update, bytes) in &held {
                assert_eq!(read(&backend, &update.output().outputs[0]), *bytes);
            }
            drop((held, frame, final_only, session));
            drain(&backend, 0);
        }
    }
}
