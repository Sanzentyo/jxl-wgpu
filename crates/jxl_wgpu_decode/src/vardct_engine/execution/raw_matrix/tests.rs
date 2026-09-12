use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};

use jxl_wgpu::{MemoryBudget, WgpuBackend};

use super::*;
use crate::vardct_engine::execution::{FrameDecodeSession, VarDctRuntimeStats};
use crate::vardct_engine::pipeline::VarDctPipelines;
use crate::vardct_engine::source::{VarDctPrepareOptions, prepare_packet_source};
use crate::vardct_packet::BoundedVarDctPacketPlan;
use crate::{
    GpuCodestream, GpuOutputRequest, GpuPendingFrame, GpuSubmissionSession, vardct_rgb8_format,
};

fn session(
    backend: &WgpuBackend,
    pipelines: &Arc<VarDctPipelines>,
    shortage: u64,
    local_packets: bool,
    progressive: bool,
) -> FrameDecodeSession {
    let hex = if local_packets {
        include_str!("../../../../test-data/jpeg_transcode_raw_matrix_local_packets.jxl.hex")
    } else {
        include_str!("../../../../test-data/jpeg_transcode_raw_matrix_local.jxl.hex")
    }
    .split_whitespace()
    .collect::<String>();
    let bytes = hex
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect::<Vec<_>>();
    let parsed = jxl_gpu_bitstream::parse(&bytes, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let source = GpuCodestream::from_shared(
        parsed.codestream().into(),
        0..parsed.codestream().len(),
        false,
    )
    .unwrap();
    let packet = BoundedVarDctPacketPlan::parse(parsed.codestream(), &inventory).unwrap();
    assert_eq!(packet.modular_metadata.is_empty(), local_packets);
    assert_eq!(packet.requires_lf_staging(), local_packets);
    let source = prepare_packet_source(
        backend,
        source,
        &GpuOutputRequest::color(vardct_rgb8_format())
            .unwrap()
            .with_progressive_output(progressive),
        &inventory,
        VarDctPrepareOptions {
            output_variant: pipelines.output_variant,
            stream_window_limit: None,
            memory_limit_bytes: u64::MAX,
        },
        packet,
    )
    .unwrap();
    let plan = source.packet.pending_raw_hf_dequant_side_image().unwrap();
    let end = source.packet.pending_raw_hf_dequant_packet_end().unwrap();
    let minimal = pipelines
        .raw_hf_dequant
        .plan_source(&source.codestream, plan, end, 40)
        .unwrap();
    let whole = pipelines
        .raw_hf_dequant
        .plan_source(&source.codestream, plan, end, u64::MAX)
        .unwrap();
    assert!(whole.memory_bytes > minimal.memory_bytes);
    let capacity = source.memory.total_frame_bytes + minimal.memory_bytes - shortage;
    let runtime_stats = Arc::new(VarDctRuntimeStats {
        submissions_per_frame: Arc::new(AtomicUsize::new(source.submissions_per_frame())),
        hf_packet_stream_batch_count: AtomicUsize::new(0),
    });
    FrameDecodeSession {
        backend: backend.clone(),
        pipelines: Arc::clone(pipelines),
        memory_stats: source.memory,
        runtime_stats,
        memory: MemoryBudget::new(NonZeroU64::new(capacity).unwrap()),
        source: Some(source),
    }
}

#[test]
fn raw_matrix_late_admission_shrinks_uploads_and_cancellation_releases_every_stage() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let pipelines = Arc::new(VarDctPipelines::new(&backend).unwrap());
    for (local_packets, cancel_stage, progressive) in [false, true].into_iter().flat_map(|local| {
        (0..3).flat_map(move |stage| [false, true].map(|progressive| (local, stage, progressive)))
    }) {
        let mut session = session(&backend, &pipelines, 0, local_packets, progressive);
        let memory = session.memory.clone();
        let mut pending = session.submit_next().unwrap().unwrap();
        let (weak_image, weak_frame) = loop {
            if let VarDctPendingStage::RawHfDequant { work, lifetime, .. } = &pending.stage {
                assert_eq!(
                    work.stream.stream_bytes, 40,
                    "late budget must shrink the caller cap"
                );
                assert_eq!(lifetime._permit.bytes(), work.stream.memory_bytes);
                assert!(Arc::ptr_eq(
                    &lifetime._frame,
                    pending.lifetime.as_ref().unwrap()
                ));
                let matches_stage = match cancel_stage {
                    0 => true,
                    1 => work.next_window > 1 && lifetime.job.has_finalization_commands(),
                    2 => !lifetime.job.has_finalization_commands(),
                    _ => unreachable!(),
                };
                if matches_stage {
                    break (Arc::downgrade(lifetime), Arc::downgrade(&lifetime._frame));
                }
            }
            assert!(
                !pending.dependency_submission_ready(),
                "missed raw matrix cancellation stage"
            );
            if let Some(completion) = pending.stage_completion() {
                pending.advance_staged_packet(completion.wait()).unwrap();
            } else {
                pending.resume_after_dc().unwrap();
            }
        };
        assert_eq!(memory.snapshot().available_bytes, 0);
        let submissions = session
            .runtime_stats
            .submissions_per_frame
            .load(Ordering::Acquire);
        drop(pending);
        let fence = backend.queue().submit(std::iter::empty());
        backend
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(fence),
                timeout: None,
            })
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while memory.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::yield_now();
        }
        assert!(weak_image.upgrade().is_none());
        assert!(weak_frame.upgrade().is_none());
        assert_eq!(memory.snapshot().reserved_bytes, 0);
        assert_eq!(
            session
                .runtime_stats
                .submissions_per_frame
                .load(Ordering::Acquire),
            submissions
        );
    }
}

#[test]
fn raw_matrix_rejects_an_unaffordable_minimum_without_retaining_frame_memory() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let pipelines = Arc::new(VarDctPipelines::new(&backend).unwrap());
    for local_packets in [false, true] {
        let mut session = session(&backend, &pipelines, 4, local_packets, false);
        let error = match session.submit_next() {
            Err(error) => error,
            Ok(Some(pending)) => pending.wait().unwrap_err(),
            Ok(None) => panic!("missing frame"),
        };
        assert!(
            matches!(
                error,
                crate::Error::VarDct(VarDctDecodeError::MemoryBackpressure(_))
            ),
            "{error:?}"
        );
        assert_eq!(session.memory.snapshot().reserved_bytes, 0);
    }
}

#[test]
fn deferred_dc_precedes_raw_matrix_admission_and_keeps_its_output_after_failure() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let pipelines = Arc::new(VarDctPipelines::new(&backend).unwrap());
    for local_packets in [false, true] {
        let mut session = session(&backend, &pipelines, 4, local_packets, true);
        let memory = session.memory.clone();
        let mut pending = session.submit_next().unwrap().unwrap();
        let dc = pending.wait_next_update().unwrap();
        let crate::SubmittedGpuUpdate::Intermediate { progression, frame } = dc else {
            panic!("DC missing")
        };
        assert_eq!(
            progression
                .completed_passes()
                .expect("coefficient boundary"),
            0
        );
        assert!(matches!(pending.stage, VarDctPendingStage::AfterDc { .. }));
        assert!(
            pending.expected_hf.is_empty(),
            "descriptors must still be deferred"
        );
        assert!(matches!(
            pending.wait_next_update(),
            Err(crate::Error::VarDct(VarDctDecodeError::MemoryBackpressure(
                _
            )))
        ));
        drop(pending);
        let expected = session.memory_stats.output_lease_bytes;
        assert_eq!(memory.snapshot().reserved_bytes, expected);
        drop(frame);
        assert_eq!(memory.snapshot().reserved_bytes, 0);
    }
}

#[test]
fn invalid_raw_matrix_entropy_cannot_invalidate_an_already_returned_dc_image() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let pipelines = Arc::new(VarDctPipelines::new(&backend).unwrap());
    for local_packets in [false, true] {
        let mut session = session(&backend, &pipelines, 0, local_packets, true);
        session.memory = MemoryBudget::new(NonZeroU64::new(u64::MAX).unwrap());
        let source = session.source.as_mut().unwrap();
        let start = source
            .packet
            .pending_raw_hf_dequant_side_image()
            .unwrap()
            .image
            .token_bit_offset
            .div_ceil(8) as usize;
        let end = (source.packet.pending_raw_hf_dequant_packet_end().unwrap() / 8) as usize;
        let mut damaged = Vec::new();
        source
            .codestream
            .for_each_range_chunk(0..source.codestream.logical_bytes(), |chunk| {
                damaged.extend_from_slice(chunk);
                Ok(())
            })
            .unwrap();
        damaged[start..(start + 8).min(end)].fill(0xff);
        let length = damaged.len();
        source.codestream = GpuCodestream::from_shared(damaged.into(), 0..length, false).unwrap();
        let mut pending = session.submit_next().unwrap().unwrap();
        let dc = pending.wait_next_update().unwrap();
        let crate::SubmittedGpuUpdate::Intermediate { progression, frame } = dc else {
            panic!("DC missing")
        };
        assert_eq!(
            progression
                .completed_passes()
                .expect("coefficient boundary"),
            0
        );
        let read = || {
            jxl_wgpu::ImageReadbackPipeline::new(&backend)
                .submit(&frame.output)
                .unwrap()
                .wait()
                .unwrap()
                .frame
                .outputs[0]
                .bytes
                .clone()
        };
        let before = read();
        let error = pending.wait_next_update().unwrap_err();
        assert!(
            matches!(
                error,
                crate::Error::VarDct(
                    VarDctDecodeError::RawHfDequantStatus { .. }
                        | VarDctDecodeError::RawHfDequantValue { .. }
                        | VarDctDecodeError::RawHfDequantGpu { .. }
                )
            ),
            "{error:?}"
        );
        drop(pending);
        assert_eq!(
            session.memory.snapshot().reserved_bytes,
            session.memory_stats.output_lease_bytes
        );
        assert_eq!(read(), before);
        drop(frame);
        assert_eq!(session.memory.snapshot().reserved_bytes, 0);
    }
}

#[test]
fn cancelling_the_deferred_dc_boundary_releases_queued_and_unsubmitted_work() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let pipelines = Arc::new(VarDctPipelines::new(&backend).unwrap());
    for local_packets in [false, true] {
        let mut session = session(&backend, &pipelines, 0, local_packets, true);
        let mut pending = session.submit_next().unwrap().unwrap();
        while !matches!(pending.stage, VarDctPendingStage::AfterDc { .. }) {
            let completion = pending.stage_completion().unwrap();
            pending.advance_staged_packet(completion.wait()).unwrap();
        }
        let weak = Arc::downgrade(pending.lifetime.as_ref().unwrap());
        assert_eq!(
            session.memory.snapshot().reserved_bytes,
            session.memory_stats.total_frame_bytes
        );
        drop(pending);
        let fence = backend.queue().submit(std::iter::empty());
        backend
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(fence),
                timeout: None,
            })
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while session.memory.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline
        {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::yield_now();
        }
        assert!(weak.upgrade().is_none());
        assert_eq!(session.memory.snapshot().reserved_bytes, 0);
    }
}
