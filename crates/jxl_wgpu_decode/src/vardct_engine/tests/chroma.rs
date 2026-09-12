use crate::frame_surface::FrameSurfaceEncoding;
use crate::vardct_frontend::StandardVarDctProfile;
use crate::{GpuCodestream, GpuOutputRequest, GpuPendingFrame, GpuSubmissionSession};
use jxl_test_support::{fixtures::noise, gpu::planes, offline};
use jxl_wgpu::WgpuBackend;
use std::num::NonZeroU64;
use std::sync::Arc;

use super::super::VarDctSubmissionEngine;

fn source(name: &str) -> (Arc<[u8]>, jxl_gpu_bitstream::CodestreamInventory) {
    let data = offline::unhex(
        &std::fs::read_to_string(
            jxl_test_support::decoder_directory()
                .join(format!("test-data/jpeg_sampling/{name}.jxl.hex")),
        )
        .unwrap(),
    );
    let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    (noise::zero_noise(&data, &inventory, None).into(), inventory)
}

fn drain(backend: &WgpuBackend, bytes: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while backend.transient_memory_budget().snapshot().reserved_bytes != bytes
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        bytes
    );
}

#[test]
fn encoded_jpeg_surfaces_admit_exact_expansion_and_release_after_retry_or_cancellation() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for bounded in [false, true] {
        let mut engine = VarDctSubmissionEngine::new(backend.clone()).unwrap();
        if bounded {
            engine = engine.with_stream_window_limit(NonZeroU64::new(256).unwrap());
        }
        for index in 0..64 {
            let name = format!("odd_{}{}{}", index / 16, index / 4 % 4, index % 4);
            let (data, inventory) = source(&name);
            let profile = StandardVarDctProfile::negotiate(&inventory).unwrap();
            let [blocks_x, blocks_y] = profile.block_extent();
            let full_plane = u64::from(blocks_x) * u64::from(blocks_y) * 64 * 4;
            let shifted = profile
                .channel_shifts
                .iter()
                .filter(|shift| shift.is_subsampled())
                .count() as u64;
            let request = GpuOutputRequest::color(FrameSurfaceEncoding::Srgb.format()).unwrap();
            let open = |request: &GpuOutputRequest| {
                engine
                    .open_with_inventory(
                        GpuCodestream::from_shared(data.clone(), 0..data.len(), false).unwrap(),
                        request,
                        &inventory,
                    )
                    .unwrap()
                    .session
            };
            let plain = open(&request);
            assert_eq!(
                plain.memory_stats().unwrap().pre_restoration_upsample_bytes,
                0
            );
            drop(plain);
            let request = request.for_frame_surface(FrameSurfaceEncoding::Encoded);
            let complete = open(&request);
            let expected_memory = complete.memory_stats().unwrap();
            assert_eq!(
                expected_memory.pre_restoration_upsample_bytes,
                full_plane * shifted,
                "{name}"
            );
            assert_eq!(
                expected_memory.pre_restoration_upsample_uniform_bytes,
                32 * shifted,
                "{name}"
            );
            assert_eq!(expected_memory.restoration_scratch_bytes, 0);
            assert_eq!(expected_memory.noise_bytes, 0);
            drop(complete);
            let request = request.before_frame_features();
            let mut session = open(&request);
            assert_eq!(session.memory_stats().unwrap(), expected_memory);
            assert_eq!(
                backend.transient_memory_budget().snapshot().reserved_bytes,
                0
            );
            if !matches!(index, 0 | 6 | 27 | 54) {
                continue;
            }
            let budget = backend.transient_memory_budget();
            let held = budget
                .try_reserve(
                    budget.snapshot().available_bytes - expected_memory.total_frame_bytes + 1,
                )
                .unwrap();
            let before = budget.snapshot().reserved_bytes;
            for _ in 0..2 {
                assert!(
                    matches!(session.submit_next(), Err(crate::Error::MemoryBackpressure(
                    jxl_wgpu::MemoryBudgetError::Exhausted { requested_bytes, .. }
                )) if requested_bytes == expected_memory.transient_bytes)
                );
                assert_eq!(
                    budget.snapshot().reserved_bytes,
                    before,
                    "failed output admission rolled back"
                );
            }
            drop(held);
            let output = session.submit_next().unwrap().unwrap().wait().unwrap();
            assert!(session.submit_next().unwrap().is_none());
            drop(session);
            drain(&backend, expected_memory.output_lease_bytes);
            let expected = planes::read(&backend, &output.output.outputs[0]);
            assert!(
                expected
                    .iter()
                    .all(|word| f32::from_bits(*word).is_finite())
            );
            drop(output);
            drain(&backend, 0);
            let mut cancelled = open(&request);
            let pending = cancelled.submit_next().unwrap().unwrap();
            drop((pending, cancelled));
            drain(&backend, 0);
            let mut retried = open(&request);
            let output = retried.submit_next().unwrap().unwrap().wait().unwrap();
            assert_eq!(planes::read(&backend, &output.output.outputs[0]), expected);
            drop((output, retried));
            drain(&backend, 0);
        }
    }
}
