use super::*;
use jxl_gpu_bitstream::{CodestreamInventory, FrameEncoding, SampleBitDepth};
use jxl_wgpu_decode::{FrameProgression, OrientationPolicy};
use std::num::NonZeroUsize;

pub(super) fn color_request(linear: bool) -> GpuOutputRequest {
    let mut format = request().format().clone();
    if linear && let jxl_gpu_formats::ColorSpecification::Defined(ref mut color) = format.color_spec
    {
        color.transfer = jxl_gpu_formats::TransferFunction::Linear;
    }
    GpuOutputRequest::color(format)
        .unwrap()
        .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
        .with_spot_color_policy(SpotColorPolicy::Preserve)
        .with_progressive_output(true)
        .with_max_frame_slots(NonZeroUsize::new(1).unwrap())
}

fn stages(name: &str, inventory: &CodestreamInventory) -> Vec<Vec<f32>> {
    let image = &inventory.image_header;
    let words = image.width as usize * image.height as usize * (4 + image.extra_channels.len());
    let reference = reference(&format!("progressive/{name}"));
    assert_eq!(
        reference.len(),
        words * (inventory.frames[1].num_passes as usize + 1)
    );
    reference.chunks_exact(words).map(<[f32]>::to_vec).collect()
}

fn assert_progression(
    progression: Option<FrameProgression>,
    index: usize,
    inventory: &CodestreamInventory,
) {
    let frame = &inventory.frames[1];
    if index == frame.num_passes as usize {
        assert!(progression.is_none());
    } else {
        let progression = progression.unwrap();
        assert_eq!(progression.completed_passes(), Some(index as u8));
        assert_eq!(progression.physical_frame_index(), frame.frame_index);
    }
}

pub(super) fn srgb(value: f32) -> f32 {
    let v = f64::from(value);
    (v.signum()
        * if v.abs() <= 0.0031308 {
            v.abs() * 12.92
        } else {
            1.055 * v.abs().powf(1.0 / 2.4) - 0.055
        }) as f32
}

pub(super) fn compare(actual: &[u32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let mut max = 0f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f32::from_bits(actual);
        let error = (actual - expected).abs() / (1.0 + expected.abs());
        assert!(
            actual.is_finite() && expected.is_finite() && error <= tolerance,
            "{label}/{index}: {actual} vs {expected}, error {error}"
        );
        max = max.max(error);
    }
    eprintln!("{label}: max normalized error {max}");
}

pub(super) fn drain(backend: &WgpuBackend, retained: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while backend.transient_memory_budget().snapshot().reserved_bytes != retained
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        retained
    );
}

#[test]
fn patch_passes_match_native_prefixes_and_keep_owned_presentations_immutable() {
    let backend = backend();
    for family in ["vardct", "gray", "modular", "float"] {
        for suffix in ["", "_empty"] {
            let name = format!("{family}{suffix}");
            let data = encoded(&format!("progressive/{name}"));
            let inventory = inventory(&data);
            let expected = stages(&name, &inventory);
            let image = &inventory.image_header;
            let color_words = image.width as usize * image.height as usize * 4;
            for linear in if image.xyb_encoded {
                &[false, true][..]
            } else {
                &[false][..]
            } {
                let mut whole = None;
                for limit in [None, NonZeroU64::new(256)] {
                    let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                    if let Some(limit) = limit {
                        engine = engine.with_stream_window_limit(limit);
                    }
                    let decoder = GpuDecoder::new(engine);
                    let request = color_request(*linear);
                    let mut session = if limit.is_some() {
                        planes::open_fragmented(&decoder, &data, request.clone())
                    } else {
                        decoder.open(&data, request.clone()).unwrap()
                    };
                    let mut held = Vec::new();
                    let mut pixels = Vec::new();
                    while let Some(update) = if limit.is_some() {
                        pollster::block_on(session.next_update_async()).unwrap()
                    } else {
                        session.next_update().unwrap()
                    } {
                        let step = held.len();
                        assert_progression(update.progression(), step, &inventory);
                        let actual = planes::read(&backend, &update.output().outputs[0]);
                        let mut reference = expected[step][..color_words].to_vec();
                        if image.xyb_encoded && !linear {
                            for (index, value) in reference.iter_mut().enumerate() {
                                if index % 4 != 3 {
                                    *value = srgb(*value);
                                }
                            }
                        }
                        let label = format!("{name}/{step}/linear{linear}/{limit:?}");
                        let tolerance = if inventory.frames[1].encoding == FrameEncoding::Modular {
                            1e-5
                        } else {
                            0.003
                        };
                        compare(&actual, &reference, tolerance, &label);
                        compare(
                            &actual
                                .iter()
                                .skip(3)
                                .step_by(4)
                                .copied()
                                .collect::<Vec<_>>(),
                            &reference
                                .iter()
                                .skip(3)
                                .step_by(4)
                                .copied()
                                .collect::<Vec<_>>(),
                            2e-6,
                            &format!("{label}/alpha"),
                        );
                        pixels.push(actual);
                        held.push(update);
                    }
                    assert_eq!(held.len(), expected.len());
                    assert_ne!(
                        pixels.first(),
                        pixels.last(),
                        "{name} already includes future passes"
                    );
                    if let Some(whole) = &whole {
                        assert_eq!(&pixels, whole);
                    }
                    let mut final_only = decoder
                        .open(&data, request.with_progressive_output(false))
                        .unwrap();
                    let final_frame = final_only.next_frame().unwrap().unwrap();
                    assert_eq!(
                        planes::read(&backend, &final_frame.output().outputs[0]),
                        *pixels.last().unwrap()
                    );
                    for (update, expected) in held.iter().zip(&pixels) {
                        assert_eq!(update.metadata, final_frame.metadata);
                        assert_eq!(
                            planes::read(&backend, &update.output().outputs[0]),
                            *expected
                        );
                    }
                    assert!(final_only.next_frame().unwrap().is_none());
                    whole = Some(pixels);
                    drop((held, final_frame, final_only, session));
                    drain(&backend, 0);
                    assert_eq!(
                        decoder.incremental_input_budget().snapshot().reserved_bytes,
                        0
                    );
                }
            }
        }
    }
}

#[test]
fn patch_extra_channel_passes_match_native_scalar_planes() {
    let backend = backend();
    for name in ["vardct", "float"] {
        let data = encoded(&format!("progressive/{name}"));
        let inventory = inventory(&data);
        let image = &inventory.image_header;
        let expected = stages(name, &inventory);
        let pixel_count = image.width as usize * image.height as usize;
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
        );
        for (channel, extra) in image.extra_channels.iter().enumerate() {
            let request = GpuOutputRequest::numeric(
                jxl_gpu_formats::vpi::VpiPitchLinearFormat::F32.pixel_format(),
                match extra.bit_depth {
                    SampleBitDepth::Integer { .. } => NumericSampleMapping::NormalizedUnsigned,
                    SampleBitDepth::Float { .. } => NumericSampleMapping::NativeFloat,
                },
            )
            .unwrap()
            .with_extra_channel(channel as u32)
            .unwrap()
            .with_orientation_policy(OrientationPolicy::Apply)
            .with_progressive_output(true);
            let mut session = planes::open_fragmented(&decoder, &data, request);
            let mut step = 0;
            while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                assert_progression(update.progression(), step, &inventory);
                let actual = planes::read(&backend, &update.output().outputs[0]);
                let offset = pixel_count * (4 + channel);
                compare(
                    &actual,
                    &expected[step][offset..offset + pixel_count],
                    2e-6,
                    &format!("{name}/extra{channel}/step{step}"),
                );
                step += 1;
            }
            assert_eq!(step, expected.len());
            drop(session);
            drain(&backend, 0);
        }
    }
}

#[test]
fn patch_pass_cancellation_and_later_entropy_errors_preserve_prior_images() {
    let backend = backend();
    let data = encoded("progressive/vardct");
    let inventory = inventory(&data);
    let request = color_request(false);
    let mut baseline = GpuDecoder::wgpu(backend.clone())
        .unwrap()
        .open(&data, request.clone())
        .unwrap();
    let mut expected = Vec::new();
    while let Some(update) = baseline.next_update().unwrap() {
        expected.push(planes::read(&backend, &update.output().outputs[0]));
    }
    drop(baseline);
    for limit in [256, u64::MAX] {
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(limit).unwrap()),
        );
        for (completed, finish) in (0..expected.len())
            .flat_map(|completed| [false, true].map(|finish| (completed, finish)))
        {
            let mut session = planes::open_fragmented(&decoder, &data, request.clone());
            session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
            let mut held = None;
            for _ in 0..completed {
                held = pollster::block_on(session.next_update_async()).unwrap();
            }
            // The same submitted producer can switch to final-only delivery after any update.
            if finish {
                let final_frame = pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    planes::read(&backend, &final_frame.output().outputs[0]),
                    *expected.last().unwrap()
                );
            }
            drop(session);
            let retained = held
                .as_ref()
                .map_or(0, |u| u.output().outputs[0].buffer.size());
            drain(&backend, retained);
            if let Some(update) = held {
                assert_eq!(
                    planes::read(&backend, &update.output().outputs[0]),
                    expected[completed - 1]
                );
            }
            drain(&backend, 0);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
        }
        let frame = &inventory.frames[1];
        let pass = frame.num_passes - 1;
        let packet = frame.sections.iter().filter(|section| matches!(section.kind,
            jxl_gpu_bitstream::FrameSectionKind::PassGroup { pass_index, .. } if pass_index == pass
        )).max_by_key(|section| section.bytes.length).unwrap();
        let mut damaged = data.clone();
        let end = packet.bytes.end().unwrap() as usize;
        damaged[end - (packet.bytes.length as usize).min(16)..end].fill(0xff);
        let mut session = planes::open_fragmented(&decoder, &damaged, request.clone());
        let mut held = Vec::new();
        for expected in &expected[..=pass as usize] {
            let update = pollster::block_on(session.next_update_async())
                .unwrap()
                .unwrap();
            assert_eq!(
                planes::read(&backend, &update.output().outputs[0]),
                *expected
            );
            held.push(update);
        }
        assert!(matches!(
            session.next_update(),
            Err(Error::VarDct(
                jxl_wgpu_decode::VarDctDecodeError::HfCoefficientGpu(_)
            ))
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
            assert_eq!(
                planes::read(&backend, &update.output().outputs[0]),
                *expected
            );
        }
        drop(held);
        drain(&backend, 0);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
    }
}
