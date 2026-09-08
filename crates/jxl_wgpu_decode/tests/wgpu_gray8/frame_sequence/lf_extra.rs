use super::independent::{payloads, reassemble, retired};
use super::*;
use jxl_gpu_bitstream::{FrameEncoding, FrameType};
use jxl_gpu_formats::RgbChannelOrder;
use jxl_wgpu_decode::{Error, SpotColorPolicy};

use super::super::extra_channels::extra_channel_oracle as oracle;

fn fixtures() -> [(&'static str, &'static str, &'static str); 8] {
    [
        (
            "distributed_modular_gab1",
            include_str!("../../../test-data/lf_extra_channels/distributed_modular_gab1.jxl.hex"),
            include_str!("../../../test-data/lf_extra_channels/distributed_modular_gab1.f32.hex"),
        ),
        (
            "distributed_vardct_gab1",
            include_str!("../../../test-data/lf_extra_channels/distributed_vardct_gab1.jxl.hex"),
            include_str!("../../../test-data/lf_extra_channels/distributed_vardct_gab1.f32.hex"),
        ),
        (
            "nested_modular_gab1",
            include_str!("../../../test-data/lf_extra_channels/nested_modular_gab1.jxl.hex"),
            include_str!("../../../test-data/lf_extra_channels/nested_modular_gab1.f32.hex"),
        ),
        (
            "nested_vardct_gab1",
            include_str!("../../../test-data/lf_extra_channels/nested_vardct_gab1.jxl.hex"),
            include_str!("../../../test-data/lf_extra_channels/nested_vardct_gab1.f32.hex"),
        ),
        (
            "modular_gab0",
            include_str!("../../../test-data/lf_extra_channels/modular_gab0.jxl.hex"),
            include_str!("../../../test-data/lf_extra_channels/modular_gab0.f32.hex"),
        ),
        (
            "modular_gab1",
            include_str!("../../../test-data/lf_extra_channels/modular_gab1.jxl.hex"),
            include_str!("../../../test-data/lf_extra_channels/modular_gab1.f32.hex"),
        ),
        (
            "vardct_gab0",
            include_str!("../../../test-data/lf_extra_channels/vardct_gab0.jxl.hex"),
            include_str!("../../../test-data/lf_extra_channels/vardct_gab0.f32.hex"),
        ),
        (
            "vardct_gab1",
            include_str!("../../../test-data/lf_extra_channels/vardct_gab1.jxl.hex"),
            include_str!("../../../test-data/lf_extra_channels/vardct_gab1.f32.hex"),
        ),
    ]
}

fn data(hex: &'static str) -> Vec<u8> {
    encoded(&Case {
        name: "lf_extras",
        hex,
        format: LosslessModularFormat::Rgba,
        bits: 8,
        vardct: true,
    })
}

fn output_request(extra: Option<u32>) -> GpuOutputRequest {
    if let Some(extra) = extra {
        GpuOutputRequest::numeric(
            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
            NumericSampleMapping::NormalizedUnsigned,
        )
        .unwrap()
        .with_extra_channel(extra)
        .unwrap()
    } else {
        GpuOutputRequest::color(PixelFormat::rgb_f32(
            RgbChannelOrder::Rgba,
            false,
            jxl_wgpu_decode::vardct_rgb8_format().color_spec,
        ))
        .unwrap()
        .with_spot_color_policy(SpotColorPolicy::Preserve)
    }
}

/// Independently locate real distributed entropy, rather than testing only its frame header.
fn lf_extra_ranges(
    data: &[u8],
    inventory: &jxl_gpu_bitstream::FrameInventory,
) -> Vec<(usize, std::ops::Range<usize>)> {
    use jxl_oxide_common::Bundle;
    let mut bits = jxl_bitstream::Bitstream::new(data);
    let image = std::sync::Arc::new(jxl_image::ImageHeader::parse(&mut bits, ()).unwrap());
    let mut bits = jxl_bitstream::Bitstream::new(data);
    bits.skip_bits(inventory.header_bits.offset as usize)
        .unwrap();
    let pool = jxl_threadpool::JxlThreadPool::none();
    let mut frame = jxl_frame::Frame::parse(
        &mut bits,
        jxl_frame::FrameContext {
            image_header: image,
            tracker: None,
            pool: pool.clone(),
        },
    )
    .unwrap();
    frame.feed_bytes(&data[bits.num_read_bits() / 8..]).unwrap();
    let global = frame.try_parse_lf_global::<i32>().unwrap().unwrap();
    let mut modular = global.gmodular.try_clone().unwrap();
    let groups = modular
        .modular
        .image_mut()
        .unwrap()
        .prepare_groups(frame.pass_shifts())
        .unwrap();
    groups
        .lf_groups
        .into_iter()
        .enumerate()
        .map(|(index, image)| {
            assert!(!image.is_empty());
            let (section_index, section) = inventory
                .sections
                .iter()
                .enumerate()
                .find(|(_, section)| {
                    section.kind
                        == jxl_gpu_bitstream::FrameSectionKind::LowFrequencyGroup {
                            group_index: index as u64,
                        }
                })
                .unwrap();
            let mut bits = jxl_bitstream::Bitstream::new(data);
            bits.skip_bits(section.bits.offset as usize).unwrap();
            let mut recursive = image
                .recursive(&mut bits, global.gmodular.ma_config(), None)
                .unwrap();
            let mut subimage = recursive.prepare_subimage().unwrap();
            let start = bits.num_read_bits();
            subimage
                .decode(
                    &mut bits,
                    1 + frame.header().num_lf_groups() + index as u32,
                    false,
                )
                .unwrap();
            assert!(subimage.finish(&pool));
            let end = bits.num_read_bits();
            assert!(end > start + 16);
            (section_index, start..end)
        })
        .collect()
}

#[test]
fn lf_extra_channels_survive_global_cursor_continuations_and_match_independent_references() {
    let Some(backend) = backend() else { return };
    for (name, hex, reference) in fixtures() {
        let data = data(hex);
        let inventory = parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let nested = name.starts_with("nested");
        let pixels = inventory.image_header.width as usize * inventory.image_header.height as usize;
        if name.starts_with("distributed") {
            assert_eq!(
                lf_extra_ranges(&data, inventory.frames.last().unwrap()).len(),
                2
            );
        }
        assert_eq!(inventory.frames.len(), if nested { 3 } else { 2 });
        assert_eq!(inventory.image_header.extra_channels.len(), 2);
        assert_eq!(inventory.frames[0].frame_type, FrameType::LowFrequency);
        assert_eq!(
            inventory.frames[0].color_sample_extent(),
            Some(if nested {
                (2, 1)
            } else {
                (
                    inventory.image_header.width.div_ceil(8),
                    inventory.image_header.height.div_ceil(8),
                )
            })
        );
        for index in 1..inventory.frames.len() {
            assert_eq!(
                inventory.frames[index].lf_source_frame,
                Some(index as u32 - 1)
            );
        }
        assert_eq!(
            inventory.frames[0].encoding == FrameEncoding::Modular,
            name.contains("modular")
        );
        let reference: Vec<_> = reference
            .split_whitespace()
            .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
            .collect();
        let (rust_color, rust_extras) = oracle::rust_planes(&data);
        assert_eq!(reference.len(), pixels * 6);
        if let Some((color, extras)) = oracle::libjxl_planes(&data, pixels, 2) {
            for (actual, expected) in color.iter().chain(extras.iter().flatten()).zip(&reference) {
                assert!(
                    (actual - expected).abs() <= 2e-6,
                    "{name}: stale native reference"
                );
            }
        }
        let mut whole_submissions = [0; 3];
        let mut whole_planes = [None, None, None];
        for bounded in [false, true] {
            let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            if bounded {
                engine = engine.with_stream_window_limit(NonZeroU64::new(256).unwrap());
            }
            let decoder = GpuDecoder::new(engine);
            for (selection, extra) in [None, Some(0), Some(1)].into_iter().enumerate() {
                let request = output_request(extra);
                let mut session = if bounded {
                    incremental(&decoder, &data, request)
                } else {
                    decoder.open(&data, request).unwrap()
                };
                let frame = if bounded {
                    pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap()
                } else {
                    session.next_frame().unwrap().unwrap()
                };
                let actual = oracle::floats(&read_output(&backend, &frame.output().outputs[0]));
                let (native, rust) = match extra {
                    None => (&reference[..pixels * 4], &rust_color[..]),
                    Some(index) => (
                        &reference[pixels * (4 + index as usize)..pixels * (5 + index as usize)],
                        &rust_extras[index as usize][..],
                    ),
                };
                for expected in [native, rust] {
                    assert_eq!(actual.len(), expected.len());
                    let error = actual
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f32, f32::max);
                    assert!(
                        error <= 5e-4 && actual.iter().all(|v| v.is_finite()),
                        "{name}, bounded={bounded}, extra={extra:?}: error={error}"
                    );
                }
                let submissions = session.submission_session().submissions_per_frame();
                assert!(submissions > 2);
                if bounded {
                    assert!(submissions > whole_submissions[selection]);
                    assert_eq!(
                        Some(&actual),
                        whole_planes[selection].as_ref(),
                        "{name}: bounded output"
                    );
                } else {
                    whole_submissions[selection] = submissions;
                    whole_planes[selection] = Some(actual);
                }
                drop(frame);
                assert!(session.next_frame().unwrap().is_none());
                drop(session);
                retired(&backend);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    0
                );
            }
        }
    }
}

#[test]
fn referenced_lf_distributed_extra_entropy_is_validated_for_every_group_and_output() {
    let Some(backend) = backend() else { return };
    for (name, hex, _) in fixtures()
        .into_iter()
        .filter(|(name, _, _)| name.starts_with("distributed"))
    {
        let data = data(hex);
        let inventory = parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let frame = inventory.frames.last().unwrap();
        for (section, entropy) in lf_extra_ranges(&data, frame) {
            let mut packets = payloads(&data, frame);
            let cut = (entropy.start + (entropy.end - entropy.start) / 2) / 8;
            packets[section].truncate(cut - frame.sections[section].bytes.offset as usize);
            let mut corrupt = data[..frame.header_bits.offset as usize / 8].to_vec();
            corrupt.extend(reassemble(&data, frame, packets));
            // The container/TOC is valid; failure must come from GPU entropy completion.
            parse(&corrupt, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            for bounded in [false, true] {
                let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                if bounded {
                    engine = engine.with_stream_window_limit(NonZeroU64::new(256).unwrap());
                }
                let decoder = GpuDecoder::new(engine);
                for extra in [None, Some(0), Some(1)] {
                    let mut session = if bounded {
                        incremental(&decoder, &corrupt, output_request(extra))
                    } else {
                        decoder.open(&corrupt, output_request(extra)).unwrap()
                    };
                    let result = if bounded {
                        pollster::block_on(session.next_frame_async())
                    } else {
                        session.next_frame()
                    };
                    assert!(
                        matches!(
                            result,
                            Err(Error::VarDct(
                                jxl_wgpu_decode::VarDctDecodeError::ExtraModularStatus { .. }
                            ))
                        ),
                        "{name}, section={section}, bounded={bounded}, extra={extra:?}: {result:?}"
                    );
                    drop(session);
                    retired(&backend);
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                    assert_eq!(
                        decoder.incremental_input_budget().snapshot().reserved_bytes,
                        0
                    );
                }
            }
        }
    }
}

/// Locate the end of the independently decoded global image, including the ignored LF extras.
fn global_end(data: &[u8], frame: &jxl_gpu_bitstream::FrameInventory) -> usize {
    use jxl_oxide_common::Bundle;
    let mut bits = jxl_bitstream::Bitstream::new(data);
    let image = jxl_image::ImageHeader::parse(&mut bits, ()).unwrap();
    let mut bits = jxl_bitstream::Bitstream::new(data);
    bits.skip_bits(frame.header_bits.offset as usize).unwrap();
    let header = jxl_frame::FrameHeader::parse(&mut bits, &image).unwrap();
    let mut bits = jxl_bitstream::Bitstream::new(data);
    bits.skip_bits(frame.sections[0].bits.offset as usize)
        .unwrap();
    jxl_frame::data::LfGlobal::<i32>::parse(
        &mut bits,
        jxl_frame::data::LfGlobalParams::new(&image, &header, None, false),
    )
    .unwrap();
    bits.num_read_bits()
}

#[test]
fn lf_extra_channels_are_validated_even_in_overwritten_roots_and_unselected_consumers() {
    let Some(backend) = backend() else { return };
    for (name, hex, _) in fixtures()
        .into_iter()
        .filter(|(name, _, _)| name.ends_with('0'))
    {
        let data = data(hex);
        let inventory = parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let prefix = &data[..inventory.frames[0].header_bits.offset as usize / 8];
        for physical in [0, 1] {
            let frame = &inventory.frames[physical];
            let mut packets = payloads(&data, frame);
            // Cut the last global extra's entropy, before VarDCT LF/HF metadata. Rebuilding TOC
            // lets inventory succeed, so failure must come from actual GPU entropy validation.
            let end = global_end(&data, frame) / 8 - frame.sections[0].bytes.offset as usize;
            packets[0].truncate(end - 4);
            let mut corrupt = prefix.to_vec();
            if physical == 0 {
                corrupt.extend(reassemble(&data, frame, packets));
                // The valid root overwrites this LF version; the damaged extras still matter.
                corrupt.extend_from_slice(&data[prefix.len()..]);
            } else {
                corrupt.extend(reassemble(
                    &data,
                    &inventory.frames[0],
                    payloads(&data, &inventory.frames[0]),
                ));
                corrupt.extend(reassemble(&data, frame, packets));
            }
            let checked = parse(&corrupt, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            let plan = FrameExecutionPlan::negotiate(&checked).unwrap();
            if physical == 0 {
                assert!(plan.nodes[0].lf_last_use.is_none());
            }
            for bounded in [false, true] {
                let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                if bounded {
                    engine = engine.with_stream_window_limit(NonZeroU64::new(256).unwrap());
                }
                let decoder = GpuDecoder::new(engine);
                let mut session = if bounded {
                    incremental(&decoder, &corrupt, output_request(Some(0)))
                } else {
                    decoder.open(&corrupt, output_request(Some(0))).unwrap()
                };
                let result = if bounded {
                    pollster::block_on(session.next_frame_async())
                } else {
                    session.next_frame()
                };
                assert!(
                    matches!(
                        result,
                        Err(Error::ModularEntropyRejected { .. })
                            | Err(Error::VarDct(
                                jxl_wgpu_decode::VarDctDecodeError::GlobalModularStatus { .. }
                            ))
                    ),
                    "{name} physical={physical} bounded={bounded}: {result:?}"
                );
                drop(session);
                retired(&backend);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    0
                );
            }
        }
    }
}

#[test]
fn cancelling_lf_extra_channels_releases_producer_planes_global_arenas_and_input() {
    use std::task::{Context, Poll, Waker};
    let Some(backend) = backend() else { return };
    for (name, hex, _) in fixtures()
        .into_iter()
        .filter(|(name, _, _)| name.ends_with('1'))
    {
        let data = data(hex);
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(128).unwrap()),
        );
        let mut completed = incremental(&decoder, &data, output_request(None));
        drop(
            pollster::block_on(completed.next_frame_async())
                .unwrap()
                .unwrap(),
        );
        let count = completed.submission_session().submissions_per_frame();
        drop(completed);
        retired(&backend);
        assert!(count >= 8);
        for stop in [1, count / 2, count - 1] {
            let mut session = incremental(&decoder, &data, output_request(None));
            session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
            let mut context = Context::from_waker(Waker::noop());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
            while session.submission_session().submissions_per_frame() < stop {
                assert!(
                    std::time::Instant::now() < deadline,
                    "{name}: cancellation progress timeout"
                );
                assert!(matches!(
                    session.poll_next_frame(&mut context),
                    Poll::Pending
                ));
                std::thread::yield_now();
            }
            assert!(decoder.engine().in_flight_memory_stats().reserved_bytes > 0);
            drop(session);
            retired(&backend);
            assert_eq!(
                decoder.engine().in_flight_memory_stats().reserved_bytes,
                0,
                "{name} stop={stop}/{count}"
            );
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}
