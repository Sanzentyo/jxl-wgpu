use super::*;
use jxl_gpu_bitstream::{BitWriter, FrameInventory, FrameSectionKind, FrameType};
use jxl_wgpu_decode::{Error, VarDctDecodeError};
use jxl_wgpu_encode::{
    BitFragment, FrameGroupLayout, FramePacketSet, GroupPacket, GroupPacketKind, assemble_frame,
};

/// Copy original header bits and rebuild only the section sizes. This leaves frame semantics and
/// the entropy prefix unchanged, so a short payload tests GPU entropy validation after inventory.
pub(super) fn reassemble(data: &[u8], frame: &FrameInventory, payloads: Vec<Vec<u8>>) -> Vec<u8> {
    assert!(!frame.toc_permuted);
    let mut header = BitWriter::new();
    let bits = &frame.header_bits;
    for bit in bits.offset..bits.offset + bits.length {
        header
            .write_bits(u64::from((data[bit as usize / 8] >> (bit % 8)) & 1), 1)
            .unwrap();
    }
    let packets = frame
        .sections
        .iter()
        .zip(payloads)
        .map(|(section, payload)| {
            let kind = match section.kind {
                FrameSectionKind::Single => GroupPacketKind::Single,
                FrameSectionKind::LowFrequencyGlobal => GroupPacketKind::DcGlobal,
                FrameSectionKind::LowFrequencyGroup { group_index } => {
                    GroupPacketKind::DcGroup(group_index.try_into().unwrap())
                }
                FrameSectionKind::HighFrequencyGlobal => GroupPacketKind::AcGlobal,
                FrameSectionKind::PassGroup {
                    pass_index,
                    group_index,
                } => GroupPacketKind::AcGroup {
                    pass: pass_index.try_into().unwrap(),
                    group: group_index.try_into().unwrap(),
                },
            };
            GroupPacket::new(kind, payload)
        });
    assemble_frame(
        FramePacketSet::new(
            BitFragment::new(header.into_bytes(), bits.length as usize).unwrap(),
            FrameGroupLayout::new(
                frame.low_frequency_group_count.try_into().unwrap(),
                frame.group_count.try_into().unwrap(),
                frame.num_passes.try_into().unwrap(),
            )
            .unwrap(),
            packets,
        )
        .unwrap(),
    )
    .unwrap()
    .into_bytes()
}

pub(super) fn payloads(data: &[u8], frame: &FrameInventory) -> Vec<Vec<u8>> {
    frame
        .sections
        .iter()
        .map(|section| {
            let start = section.bytes.offset as usize;
            data[start..start + section.bytes.length as usize].to_vec()
        })
        .collect()
}

/// Reuse the checked-in native-31 still and the libjxl layered still's physical headers.
/// Neither entropy nor precision is derived from the GPU implementation being tested.
fn native_layers(count: usize) -> (Vec<u8>, Vec<u32>) {
    let raw = Case {
        name: "native31",
        hex: include_str!("../../../test-data/integer/31-1-0-33x5-p0-r0.jxl.hex"),
        format: LosslessModularFormat::Gray,
        bits: 31,
        vardct: false,
    };
    let data = encoded(&raw);
    let parsed = parse(&data, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let data = parsed.codestream();
    let template = cases()
        .into_iter()
        .find(|case| case.name == "layered_still")
        .unwrap();
    let template = encoded(&template);
    let parsed_template = parse(&template, Default::default()).unwrap();
    let template_inventory = parsed_template
        .codestream_inventory(Default::default())
        .unwrap();
    let template = parsed_template.codestream();
    let still = &inventory.frames[0];
    let mut result = data[..still.header_bits.offset as usize / 8].to_vec();
    for layer in 0..count {
        let header = if layer + 1 == count {
            template_inventory.frames.last().unwrap()
        } else {
            &template_inventory.frames[0]
        };
        assert!(!header.have_crop);
        assert_eq!(header.encoding, still.encoding);
        assert_eq!(header.flags, still.flags);
        assert_eq!(header.group_size_shift, still.group_size_shift);
        assert_eq!(header.num_passes, still.num_passes);
        assert_eq!(header.sections.len(), still.sections.len());
        result.extend(reassemble(template, header, payloads(data, still)));
    }
    let expected = include_str!("../../../test-data/integer/31-1-0-33x5-p0-r0.u32.hex")
        .split_whitespace()
        .map(|word| u32::from_str_radix(word, 16).unwrap())
        .collect();
    (result, expected)
}

pub(super) fn retired(backend: &WgpuBackend) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while backend.submission_poller().in_flight() != 0 && std::time::Instant::now() < deadline {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(backend.submission_poller().in_flight(), 0);
}

#[test]
fn overwritten_modular_vardct_and_lf_entropy_is_validated_before_presentation() {
    let Some(backend) = backend() else {
        return;
    };
    for case in cases()
        .into_iter()
        .filter(|case| matches!(case.name, "layered_still" | "vardct_rgb" | "vardct_dc"))
    {
        let data = encoded(&case);
        let parsed = parse(&data, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let data = parsed.codestream();
        let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
        let first = &plan.presentations[0];
        let last = first.physical_frames.end - 1;
        let hidden_color = (0..last)
            .rfind(|&i| inventory.frames[i].frame_type != FrameType::LowFrequency)
            .unwrap();
        for index in
            (0..=hidden_color).filter(|&i| case.name == "vardct_dc" || i == 0 || i == hidden_color)
        {
            let mut invalid = data[..inventory.frames[0].header_bits.offset as usize / 8].to_vec();
            for (physical, frame) in inventory.frames.iter().enumerate() {
                let mut packets = payloads(data, frame);
                if physical == index {
                    let packet = packets
                        .iter_mut()
                        .max_by_key(|packet| packet.len())
                        .unwrap();
                    assert!(packet.len() > 16);
                    packet.truncate(packet.len() - 8);
                }
                invalid.extend(reassemble(data, frame, packets));
            }
            assert_eq!(
                FrameExecutionPlan::negotiate(
                    &parse(&invalid, Default::default())
                        .unwrap()
                        .codestream_inventory(Default::default())
                        .unwrap()
                )
                .unwrap(),
                plan,
                "entropy truncation preserves the complete frame plan"
            );
            for bounded in [false, true] {
                eprintln!("reject {} physical {index}, bounded={bounded}", case.name);
                let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                let decoder = GpuDecoder::new(if bounded {
                    engine.with_stream_window_limit(NonZeroU64::new(4096).unwrap())
                } else {
                    engine
                });
                let mut session = if bounded {
                    incremental(
                        &decoder,
                        &invalid,
                        request(&case).with_progressive_output(true),
                    )
                } else {
                    decoder
                        .open(&invalid, request(&case).with_progressive_output(true))
                        .unwrap()
                };
                session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
                assert!(matches!(
                    session
                        .front_pending_frame()
                        .unwrap()
                        .unvalidated_gpu_frame(),
                    Err(Error::UnvalidatedOutputNotSubmitted)
                ));
                let result = if bounded {
                    pollster::block_on(session.next_update_async())
                } else {
                    session.next_update()
                };
                assert!(
                    matches!(
                        result,
                        Err(Error::ModularEntropyRejected { .. }
                            | Error::VarDct(
                                VarDctDecodeError::PacketGpu(_)
                                    | VarDctDecodeError::HfCoefficientGpu(_)
                            ))
                    ),
                    "{} physical {index}: {result:?}",
                    case.name
                );
                assert!(matches!(session.next_update(), Err(Error::SessionPoisoned)));
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
fn native_31_bit_layers_reuse_one_producer_budget_and_count_every_submission() {
    let Some(backend) = backend() else {
        return;
    };
    for count in [2, 17, 129] {
        let (data, expected) = native_layers(count);
        assert!(expected.iter().any(|&word| word != word as f32 as u32));
        let request = GpuOutputRequest::numeric(
            jxl_wgpu_decode::native_modular_pixel_format(
                jxl_wgpu_decode::ModularChannels::Gray,
                31,
            )
            .unwrap(),
            NumericSampleMapping::NativeUnsigned,
        )
        .unwrap();
        for bounded in [false, true] {
            let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            let decoder = GpuDecoder::new(if bounded {
                engine.with_stream_window_limit(NonZeroU64::new(256).unwrap())
            } else {
                engine
            });
            let mut session = if bounded {
                incremental(&decoder, &data, request.clone())
            } else {
                decoder.open(&data, request.clone()).unwrap()
            };
            assert_eq!(
                session.profile(),
                DecodeProfile::FrameSequence {
                    physical_frames: count,
                    presentation_frames: 1,
                }
            );
            session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
            let first_count = session.submission_session().submissions_per_frame();
            assert!(first_count > 0);
            let budget = backend.transient_memory_budget();
            let blocker = budget
                .try_reserve(budget.snapshot().available_bytes)
                .unwrap();
            // All remaining layers must fit in exactly the first producer's admitted footprint.
            let frame = if bounded {
                pollster::block_on(session.next_update_async())
                    .unwrap()
                    .unwrap()
            } else {
                session.next_update().unwrap().unwrap()
            };
            assert_eq!(
                session.submission_session().submissions_per_frame(),
                count * first_count
            );
            let pixels: Vec<_> = read_output(&backend, &frame.output().outputs[0])
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            assert_eq!(pixels, expected, "{count} layers, bounded={bounded}");
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
            drop(frame);
            drop(blocker);
            assert!(session.next_update().unwrap().is_none());
            drop(session);
            retired(&backend);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn cancelling_between_overwritten_layers_releases_source_and_gpu_reservations() {
    use std::task::{Context, Poll, Waker};
    let Some(backend) = backend() else {
        return;
    };
    let (data, _) = native_layers(129);
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    let request = GpuOutputRequest::numeric(
        jxl_wgpu_decode::native_modular_pixel_format(jxl_wgpu_decode::ModularChannels::Gray, 31)
            .unwrap(),
        NumericSampleMapping::NativeUnsigned,
    )
    .unwrap();
    for completed in [0, 1, 7] {
        let mut session = incremental(&decoder, &data, request.clone());
        session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        let first_count = session.submission_session().submissions_per_frame();
        assert!(first_count > 0);
        let mut context = Context::from_waker(Waker::noop());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while session.submission_session().submissions_per_frame() < (completed + 1) * first_count {
            assert!(std::time::Instant::now() < deadline);
            assert!(matches!(
                session.poll_next_frame(&mut context),
                Poll::Pending
            ));
            std::thread::yield_now();
        }
        assert!(matches!(
            session
                .front_pending_frame()
                .unwrap()
                .unvalidated_gpu_frame(),
            Err(Error::UnvalidatedOutputNotSubmitted)
        ));
        assert!(decoder.incremental_input_budget().snapshot().reserved_bytes > 0);
        drop(session);
        retired(&backend);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}
