use super::progression::{color_request, compare, drain, srgb};
use super::*;
use jxl_wgpu_decode::FrameProgression;
use std::num::NonZeroUsize;

const FAMILIES: [&str; 4] = [
    "vardct_gab0",
    "modular_gab1",
    "nested_vardct_gab1",
    "nested_modular_gab1",
];

fn snapshots(name: &str) -> Vec<Vec<f32>> {
    let data = encoded(&format!("lf/{name}"));
    let inventory = inventory(&data);
    let image = &inventory.image_header;
    let words = image.width as usize * image.height as usize * (4 + image.extra_channels.len());
    let values = reference(&format!("lf/{name}"));
    assert_eq!(values.len(), words * (inventory.frames.len() - 1));
    values.chunks_exact(words).map(<[f32]>::to_vec).collect()
}

fn progression(
    actual: Option<FrameProgression>,
    step: usize,
    info: &jxl_gpu_bitstream::CodestreamInventory,
) {
    let levels = info.frames.len() - 2;
    if step == levels + 1 {
        assert_eq!(actual, None);
    } else if step == levels {
        let actual = actual.unwrap();
        assert_eq!(
            actual.physical_frame_index(),
            info.frames.last().unwrap().frame_index
        );
        assert_eq!(actual.completed_passes(), Some(0));
    } else {
        assert_eq!(
            actual,
            Some(FrameProgression::LowFrequency {
                physical_frame_index: info.frames[step].frame_index,
                level: info.frames[step].lf_level as u8,
            })
        );
    }
}

#[test]
fn lf_patch_color_waits_for_dictionary_and_reference_and_matches_independent_oracles() {
    let backend = backend();
    for family in FAMILIES {
        for suffix in ["", "_empty"] {
            let name = format!("{family}{suffix}");
            let data = encoded(&format!("lf/{name}"));
            let info = inventory(&data);
            let reference = &info.frames[info.frames.len() - 2];
            assert_eq!(
                reference.lf_source_frame,
                info.frames.last().unwrap().lf_source_frame
            );
            assert!(reference.save_before_color_transform);
            let expected = snapshots(&name);
            let color_words =
                info.image_header.width as usize * info.image_header.height as usize * 4;
            for linear in [false, true] {
                let request = color_request(linear);
                let mut whole = None;
                for limit in [None, NonZeroU64::new(256)] {
                    let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                    if let Some(limit) = limit {
                        engine = engine.with_stream_window_limit(limit);
                    }
                    let decoder = GpuDecoder::new(engine);
                    let mut session = if limit.is_some() {
                        planes::open_fragmented(&decoder, &data, request.clone())
                    } else {
                        decoder.open(&data, request.clone()).unwrap()
                    };
                    let mut held = Vec::new();
                    let mut words = Vec::new();
                    while let Some(update) = if limit.is_some() {
                        pollster::block_on(session.next_update_async()).unwrap()
                    } else {
                        session.next_update().unwrap()
                    } {
                        let step = held.len();
                        progression(update.progression(), step, &info);
                        let actual = planes::read(&backend, &update.output().outputs[0]);
                        let levels = info.frames.len() - 2;
                        let mut expected = expected[step.min(levels)][..color_words].to_vec();
                        if !linear {
                            for (i, v) in expected.iter_mut().enumerate() {
                                if i % 4 != 3 {
                                    *v = srgb(*v);
                                }
                            }
                        }
                        let label = format!("{name}/{step}/linear{linear}/{limit:?}");
                        if step != levels {
                            compare(
                                &actual,
                                &expected,
                                if step < levels { 5e-4 } else { 0.003 },
                                &label,
                            );
                        } else {
                            // libjxl cannot flush this single-section CID. Its color is checked
                            // for finite, immutable and whole/window-identical delivery below.
                            // Both extras are global here, so their native final values already
                            // apply to this coefficient-zero reconstruction.
                            assert!(actual.iter().all(|v| f32::from_bits(*v).is_finite()));
                        }
                        compare(
                            &actual
                                .iter()
                                .skip(3)
                                .step_by(4)
                                .copied()
                                .collect::<Vec<_>>(),
                            &expected
                                .iter()
                                .skip(3)
                                .step_by(4)
                                .copied()
                                .collect::<Vec<_>>(),
                            2e-6,
                            &format!("{label}/alpha"),
                        );
                        words.push(actual);
                        held.push(update);
                    }
                    assert_eq!(held.len(), expected.len() + 1);
                    assert_ne!(
                        words.first(),
                        words.last(),
                        "{name} early output includes final AC"
                    );
                    if let Some(whole) = &whole {
                        assert_eq!(&words, whole, "{name} bounded output");
                    }
                    let mut final_only = decoder
                        .open(&data, request.clone().with_progressive_output(false))
                        .unwrap();
                    let final_image = final_only.next_frame().unwrap().unwrap();
                    assert_eq!(
                        planes::read(&backend, &final_image.output().outputs[0]),
                        *words.last().unwrap()
                    );
                    for (update, words) in held.iter().zip(&words) {
                        assert_eq!(update.metadata, final_image.metadata);
                        assert_eq!(planes::read(&backend, &update.output().outputs[0]), *words);
                    }
                    whole = Some(words);
                    drop((session, held, final_only, final_image));
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
fn lf_patch_alpha_and_depth_match_scalar_snapshots() {
    let backend = backend();
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    for family in FAMILIES {
        for suffix in ["", "_empty"] {
            let name = format!("{family}{suffix}");
            let data = encoded(&format!("lf/{name}"));
            let info = inventory(&data);
            let expected = snapshots(&name);
            let pixels = info.image_header.width as usize * info.image_header.height as usize;
            for channel in 0..2 {
                let request = GpuOutputRequest::numeric(
                    jxl_gpu_formats::vpi::VpiPitchLinearFormat::F32.pixel_format(),
                    NumericSampleMapping::NormalizedUnsigned,
                )
                .unwrap()
                .with_extra_channel(channel)
                .unwrap()
                .with_progressive_output(true);
                let mut session = planes::open_fragmented(&decoder, &data, request);
                let mut step = 0;
                while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                    progression(update.progression(), step, &info);
                    let start = (4 + channel as usize) * pixels;
                    compare(
                        &planes::read(&backend, &update.output().outputs[0]),
                        &expected[step.min(expected.len() - 1)][start..start + pixels],
                        2e-6,
                        &format!("{name}/extra{channel}/step{step}"),
                    );
                    step += 1;
                }
                assert_eq!(step, expected.len() + 1);
                drop(session);
                drain(&backend, 0);
            }
        }
    }
}

#[test]
fn queued_lf_patches_cancel_or_switch_to_final_delivery_without_mutating_held_updates() {
    let backend = backend();
    let data = encoded("lf/nested_modular_gab1");
    let request = color_request(false);
    let mut baseline = GpuDecoder::wgpu(backend.clone())
        .unwrap()
        .open(&data, request.clone())
        .unwrap();
    let mut expected = Vec::new();
    while let Some(update) = baseline.next_update().unwrap() {
        expected.push(planes::read(&backend, &update.output().outputs[0]));
    }
    assert_eq!(expected.len(), 4);
    drop(baseline);
    for limit in [256, u64::MAX] {
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(limit).unwrap()),
        );
        for completed in 0..expected.len() {
            for action in 0..3 {
                let mut session = planes::open_fragmented(&decoder, &data, request.clone());
                session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
                let mut held = Vec::new();
                for _ in 0..completed {
                    held.push(
                        pollster::block_on(session.next_update_async())
                            .unwrap()
                            .unwrap(),
                    );
                }
                if action != 0 {
                    let frame = if action == 1 {
                        session.next_frame().unwrap().unwrap()
                    } else {
                        pollster::block_on(session.next_frame_async())
                            .unwrap()
                            .unwrap()
                    };
                    assert_eq!(
                        planes::read(&backend, &frame.output().outputs[0]),
                        *expected.last().unwrap()
                    );
                }
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
    }
}

#[test]
fn lf_patch_failures_withhold_unvalidated_dependencies_and_preserve_prior_updates() {
    let backend = backend();
    let original = source("lf_extra_channels/nested_vardct_gab1");
    let info = inventory(&original);
    let mut values = fixtures::values(info.frames.last().unwrap(), 2, 16);
    values[1] = 0; // Only slot three is populated by the hidden reference.
    let invalid_dictionary = fixtures::assemble_shared_lf(&original, &values);
    let data = encoded("lf/nested_vardct_gab1");
    let info = inventory(&data);
    let request = color_request(false);
    for limit in [256, u64::MAX] {
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(limit).unwrap()),
        );
        let mut bad = planes::open_fragmented(&decoder, &invalid_dictionary, request.clone());
        assert!(matches!(
            bad.next_update(),
            Err(Error::PatchDictionary { .. })
        ));
        assert!(matches!(bad.next_frame(), Err(Error::SessionPoisoned)));
        drop(bad);
        drain(&backend, 0);
        let mut baseline = decoder.open(&data, request.clone()).unwrap();
        let mut expected = Vec::new();
        while let Some(update) = baseline.next_update().unwrap() {
            expected.push(planes::read(&backend, &update.output().outputs[0]));
        }
        drop(baseline);
        for final_frame in [false, true] {
            let mut corrupted = data.clone();
            let target = info.frames.len() - if final_frame { 1 } else { 2 };
            let section = info.frames[target].sections.last().unwrap();
            let end = section.bytes.end().unwrap() as usize;
            corrupted[end - 16..end].fill(0xff);
            let mut session = planes::open_fragmented(&decoder, &corrupted, request.clone());
            let mut held = Vec::new();
            if final_frame {
                for expected in &expected[..expected.len() - 1] {
                    let update = pollster::block_on(session.next_update_async())
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        planes::read(&backend, &update.output().outputs[0]),
                        *expected
                    );
                    held.push(update);
                }
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
}
