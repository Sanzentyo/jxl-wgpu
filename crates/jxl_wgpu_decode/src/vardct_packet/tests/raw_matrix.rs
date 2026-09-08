use jxl_gpu_bitstream::StreamSlice;
use jxl_gpu_protocol::TransformKind;
use jxl_wgpu::{KernelVariant, WgpuBackend};
use wgpu::util::DeviceExt;

use crate::GpuCodestream;
use crate::vardct_resource::{VarDctResourceLayout, hf_matrix_param_index};
use crate::vardct_side_image::RawHfDequantSideImagePlan;
use crate::wgpu_engine::{ModularSideImageStatus, RawHfDequantSideImagePipeline};

struct MatrixResult {
    status: ModularSideImageStatus,
    values: Vec<[f32; 4]>,
    windows: usize,
    finalized: bool,
}

/// Read the destination after every submission, including before the overlay is allowed to run.
fn decode_matrix(
    backend: &WgpuBackend,
    pipeline: &RawHfDequantSideImagePipeline,
    source: &GpuCodestream,
    plan: &RawHfDequantSideImagePlan,
    packet_end: u32,
    cap: u64,
) -> MatrixResult {
    let device = backend.device();
    let layout = VarDctResourceLayout::new(1, 1, 1).unwrap();
    let initial = layout.initial_values().unwrap();
    let transform_index = TransformKind::ALL
        .into_iter()
        .position(|transform| hf_matrix_param_index(transform) == plan.matrix_index)
        .unwrap();
    let offset = layout.matrix_offsets[transform_index] as usize;
    let samples = (plan.image.final_planes[0].width * plan.image.final_planes[0].height) as usize;
    let expected_initial = &initial[offset..offset + samples];
    let resources = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bounded raw matrix test resources"),
        contents: bytemuck::cast_slice(&initial),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bounded raw matrix test readback"),
        size: samples as u64 * 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let stream = pipeline.plan_source(source, plan, packet_end, cap).unwrap();
    assert!(stream.stream_bytes <= cap);
    let permit = backend
        .transient_memory_budget()
        .try_reserve(stream.memory_bytes)
        .unwrap();
    let mut job = pipeline
        .prepare(backend, source, &resources, layout, plan, &stream)
        .unwrap();
    assert_eq!(job.memory_bytes(), stream.memory_bytes);
    let mut commands = job.take_commands().unwrap();
    let mut windows = 1;
    let mut finalized = false;
    let result = loop {
        let mut copy = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        copy.copy_buffer_to_buffer(&resources, offset as u64 * 16, &staging, 0, staging.size());
        let submission = backend.queue().submit([commands, copy.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        job.mark_status_mapped();
        job.status_staging()
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
        let (tx, matrix_rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        matrix_rx.recv().unwrap().unwrap();
        let status = job.finish_status().unwrap();
        let mapped = staging.slice(..).get_mapped_range().unwrap();
        let values = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        assert_eq!(status.expected_cursor, packet_end);
        assert!((plan.image.token_bit_offset..=packet_end).contains(&status.cursor));
        if !status.is_ok() || job.has_finalization_commands() {
            assert_eq!(
                values, expected_initial,
                "cap {cap}, window {windows}: {status:?}"
            );
        }
        if status.is_in_progress() {
            assert!(!finalized);
            commands = job
                .record_next_window(backend, source, stream.segments.get(windows).unwrap())
                .unwrap();
            windows += 1;
        } else if status.is_ok() && job.has_finalization_commands() {
            assert!(!finalized);
            commands = job.take_finalization_commands().unwrap();
            finalized = true;
        } else {
            break MatrixResult {
                status,
                values,
                windows,
                finalized,
            };
        }
    };
    drop(job);
    drop(permit);
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
    result
}

#[test]
fn raw_matrix_windows_preserve_the_destination_until_entropy_is_complete() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let pipeline = RawHfDequantSideImagePipeline::new(&backend, KernelVariant::Lanes64);
    for fixture in [
        include_str!("../../../test-data/jpeg_transcode_raw_matrix.jxl.hex"),
        include_str!("../../../test-data/jpeg_transcode_raw_matrix_local.jxl.hex"),
        include_str!("../../../test-data/jpeg_transcode_raw_matrix_local_packets.jxl.hex"),
    ] {
        let (encoded, plan, packet_end) = super::parse_raw_matrix_fixture(fixture);
        let source = GpuCodestream::from_spans(
            encoded
                .chunks(7)
                .enumerate()
                .map(|(i, bytes)| (i as u64 * 7, StreamSlice::from_shared(bytes.into()))),
        )
        .unwrap();
        assert!(source.contiguous_bytes().is_none());
        let whole = decode_matrix(&backend, &pipeline, &source, &plan, packet_end, u64::MAX);
        assert!(whole.status.is_ok(), "{:?}", whole.status);
        assert_eq!(whole.windows, 1);
        assert!(!whole.finalized);
        assert_eq!(whole.status.decoded_samples, plan.image.decoded_words);
        assert!(
            whole.status.cursor < packet_end,
            "must leave the following HF metadata unread"
        );
        for cap in [40, 44, 64] {
            let bounded = decode_matrix(&backend, &pipeline, &source, &plan, packet_end, cap);
            assert!(bounded.status.is_ok(), "cap {cap}: {:?}", bounded.status);
            assert_eq!(bounded.status.cursor, whole.status.cursor, "cap {cap}");
            assert_eq!(bounded.status.decoded_samples, whole.status.decoded_samples);
            assert_eq!(bounded.values, whole.values, "cap {cap}");
            assert!(bounded.windows > 1, "cap {cap}");
            assert!(bounded.finalized, "cap {cap}");
        }

        // A one-bit truncation at the actual entropy end must fail after successful intermediate
        // windows. The source still contains the original suffix, which must not hide the truncation.
        let truncated = decode_matrix(
            &backend,
            &pipeline,
            &source,
            &plan,
            whole.status.cursor - 1,
            40,
        );
        assert!(
            !truncated.status.is_ok() && !truncated.status.is_in_progress(),
            "{:?}",
            truncated.status
        );
        assert!(truncated.windows > 1);
        assert!(!truncated.finalized);
    }
}
