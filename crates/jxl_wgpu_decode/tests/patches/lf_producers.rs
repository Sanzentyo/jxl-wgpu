use super::progression::{color_request, compare, drain, srgb};
use super::*;
use jxl_wgpu_decode::FrameProgression;

const FAMILIES: [&str; 4] = [
    "vardct_gab0",
    "modular_gab1",
    "nested_vardct_gab1",
    "nested_modular_gab1",
];

#[test]
fn patched_lf_predictions_match_native_finals_and_preserve_progressive_delivery() {
    let backend = backend();
    for family in FAMILIES {
        let unchanged = reference(&format!("lf_producers/{family}_empty"));
        let changed = reference(&format!("lf_producers/{family}"));
        assert_ne!(
            changed, unchanged,
            "LF patches must affect the main prediction"
        );
        let suffixes = ["", "_empty", "_unused"]
            .into_iter()
            .chain(family.contains("vardct").then_some("_padded"));
        for suffix in suffixes {
            let name = format!("lf_producers/{family}{suffix}");
            let data = encoded(&name);
            let info = inventory(&data);
            let pixels = info.image_header.width as usize * info.image_header.height as usize;
            let mut lf = Vec::new();
            let mut dependency = info.frames.last().unwrap().lf_source_frame;
            while let Some(index) = dependency {
                let frame = info
                    .frames
                    .iter()
                    .find(|frame| frame.frame_index == index)
                    .unwrap();
                lf.push(frame);
                dependency = frame.lf_source_frame;
            }
            lf.reverse();
            assert!(lf.iter().all(|frame| frame.lf_level != 0));
            for linear in [false, true] {
                let mut whole = None;
                let mut expected = reference(&name)[..pixels * 4].to_vec();
                if !linear {
                    for (i, value) in expected.iter_mut().enumerate() {
                        if i % 4 != 3 {
                            *value = srgb(*value);
                        }
                    }
                }
                for limit in [None, NonZeroU64::new(256)] {
                    let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                    if let Some(limit) = limit {
                        engine = engine.with_stream_window_limit(limit);
                    }
                    let decoder = GpuDecoder::new(engine);
                    let request = color_request(linear);
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
                        if step < lf.len() {
                            assert_eq!(
                                update.progression(),
                                Some(FrameProgression::LowFrequency {
                                    physical_frame_index: lf[step].frame_index,
                                    level: lf[step].lf_level as u8,
                                })
                            );
                        } else if step == lf.len() {
                            let stage = update.progression().unwrap();
                            assert_eq!(
                                stage.physical_frame_index(),
                                info.frames.last().unwrap().frame_index
                            );
                            assert_eq!(stage.completed_passes(), Some(0));
                        } else {
                            assert!(update.progression().is_none());
                        }
                        let actual = planes::read(&backend, &update.output().outputs[0]);
                        assert!(actual.iter().all(|word| f32::from_bits(*word).is_finite()));
                        words.push(actual);
                        held.push(update);
                    }
                    assert_eq!(held.len(), lf.len() + 2);
                    compare(
                        words.last().unwrap(),
                        &expected,
                        0.003,
                        &format!("{name}/linear{linear}/{limit:?}"),
                    );
                    if let Some(whole) = &whole {
                        assert_eq!(&words, whole);
                    }
                    let mut final_only = decoder
                        .open(&data, request.with_progressive_output(false))
                        .unwrap();
                    let final_image = final_only.next_frame().unwrap().unwrap();
                    assert_eq!(
                        planes::read(&backend, &final_image.output().outputs[0]),
                        *words.last().unwrap()
                    );
                    for (image, old) in held.iter().zip(&words) {
                        assert_eq!(image.metadata, final_image.metadata);
                        assert_eq!(planes::read(&backend, &image.output().outputs[0]), *old);
                    }
                    whole = Some(words);
                    drop((held, session, final_only, final_image));
                    drain(&backend, 0);
                    assert_eq!(
                        decoder.incremental_input_budget().snapshot().reserved_bytes,
                        0
                    );
                }
            }
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
            );
            for channel in 0..2 {
                let request = GpuOutputRequest::numeric(
                    jxl_gpu_formats::vpi::VpiPitchLinearFormat::F32.pixel_format(),
                    NumericSampleMapping::NormalizedUnsigned,
                )
                .unwrap()
                .with_extra_channel(channel)
                .unwrap();
                let mut session = planes::open_fragmented(&decoder, &data, request);
                let image = pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap();
                let expected = reference(&name);
                let offset = (4 + channel as usize) * pixels;
                compare(
                    &planes::read(&backend, &image.output().outputs[0]),
                    &expected[offset..offset + pixels],
                    2e-6,
                    &format!("{name}/extra{channel}"),
                );
                drop((image, session));
                drain(&backend, 0);
            }
        }
    }
}

#[test]
fn lf_dictionary_destinations_use_reduced_coded_geometry_before_publication() {
    let backend = backend();
    for family in FAMILIES {
        let data = encoded(&format!("../lf_extra_channels/{family}"));
        let info = inventory(&data);
        let first = &info.frames[0];
        let (width, height) = first.color_sample_extent().unwrap();
        let mut small = first.clone();
        (small.width, small.height) = (width, height);
        let mut values = fixtures::values(&small, info.image_header.extra_channels.len(), 16);
        values[7] = if first.encoding == jxl_gpu_bitstream::FrameEncoding::VarDct {
            width.div_ceil(8) * 8
        } else {
            width
        };
        let mut dictionaries = vec![None; info.frames.len() - 1];
        dictionaries[0] = Some(values.as_slice());
        let invalid = fixtures::assemble_lf_producers(&data, &dictionaries);
        let bad_info = inventory(&invalid);
        let root = info.frames.len();
        let start = bad_info.frames[root].header_bits.offset as usize / 8;
        let end = bad_info.frames[root + 1].header_bits.offset as usize / 8;
        let mut unused = encoded(&format!("lf_producers/{family}"));
        let offset = inventory(&unused).frames[root].header_bits.offset as usize / 8;
        unused.splice(offset..offset, invalid[start..end].iter().copied());
        assert!(
            jxl_wgpu_decode::FrameExecutionPlan::negotiate(&inventory(&unused))
                .unwrap()
                .nodes[root]
                .lf_last_use
                .is_none()
        );
        for limit in [None, NonZeroU64::new(256)] {
            let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            if let Some(limit) = limit {
                engine = engine.with_stream_window_limit(limit);
            }
            let decoder = GpuDecoder::new(engine);
            for invalid in [&invalid, &unused] {
                let mut session = decoder.open(invalid, color_request(true)).unwrap();
                assert!(matches!(
                    pollster::block_on(session.next_update_async()),
                    Err(Error::PatchDictionary { .. })
                ));
                assert!(matches!(session.next_frame(), Err(Error::SessionPoisoned)));
                drop(session);
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
fn later_lf_entropy_errors_preserve_earlier_patched_images_and_release_reservations() {
    let backend = backend();
    for family in ["nested_vardct_gab1", "nested_modular_gab1"] {
        let data = encoded(&format!("lf_producers/{family}"));
        let info = inventory(&data);
        let mut corrupt = data.clone();
        let frame = &info.frames[info.frames.len() - 2];
        assert_eq!(frame.lf_level, 1);
        let section = frame.sections.last().unwrap();
        assert!(section.bytes.length > 16);
        let end = section.bytes.end().unwrap() as usize;
        corrupt[end - 16..end].fill(0xff);
        for limit in [None, NonZeroU64::new(256)] {
            let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            if let Some(limit) = limit {
                engine = engine.with_stream_window_limit(limit);
            }
            let decoder = GpuDecoder::new(engine);
            let request = color_request(true);
            let mut baseline = decoder.open(&data, request.clone()).unwrap();
            let first = baseline.next_update().unwrap().unwrap();
            let expected = planes::read(&backend, &first.output().outputs[0]);
            drop((baseline, first));
            drain(&backend, 0);
            let mut session = planes::open_fragmented(&decoder, &corrupt, request);
            let first = pollster::block_on(session.next_update_async())
                .unwrap()
                .unwrap();
            assert!(matches!(
                first.progression(),
                Some(FrameProgression::LowFrequency { level: 2, .. })
            ));
            assert_eq!(planes::read(&backend, &first.output().outputs[0]), expected);
            let error = session.next_update().unwrap_err();
            assert!(
                matches!(
                    error,
                    Error::VarDct(jxl_wgpu_decode::VarDctDecodeError::HfCoefficientGpu(_))
                ),
                "{error:?}"
            );
            assert!(matches!(session.next_frame(), Err(Error::SessionPoisoned)));
            drop(session);
            drain(&backend, first.output().outputs[0].buffer.size());
            assert_eq!(planes::read(&backend, &first.output().outputs[0]), expected);
            drop(first);
            drain(&backend, 0);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}
