#![cfg(not(target_arch = "wasm32"))]

use std::num::NonZeroUsize;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use jxl_gpu_formats::{Channel, PixelFormat, SampleKind};
use jxl_wgpu::{DirectReadbackPolicy, GpuImageFrame, WgpuBackend, WgpuBackendConfig};
use jxl_wgpu_decode::{
    GpuDecoder, GpuFrameLease, GpuOutputRequest, NumericSampleMapping, WgpuDecodeBufferPoolLimits,
    WgpuSubmissionEngine,
};

mod common;

use common::gpu_gray8_lossless as indexed_gray8;
const PORTABLE_TRANSIENT_BUFFERS_PER_JOB: u64 = 5;

fn backend() -> Option<WgpuBackend> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("jxl-wgpu decoder buffer-pool test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;
    WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            direct_readback_policy: DirectReadbackPolicy::Disabled,
            ..WgpuBackendConfig::default()
        },
    )
    .ok()
}

fn request() -> GpuOutputRequest {
    GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        NumericSampleMapping::NormalizedGray8,
    )
    .unwrap()
}

fn decode_one(decoder: &GpuDecoder<WgpuSubmissionEngine>) -> GpuFrameLease<GpuImageFrame> {
    let mut session = decoder.open(indexed_gray8(), request()).unwrap();
    session.next_frame().unwrap().unwrap()
}

fn expected_pixels() -> Vec<u8> {
    (0..13u32)
        .flat_map(|y| {
            (0..17u32).map(move |x| {
                if y < 3 {
                    0
                } else {
                    ((x * 17 + y * 31 + (x * y) % 19) & 255) as u8
                }
            })
        })
        .collect()
}

fn read_output(backend: &WgpuBackend, frame: &GpuFrameLease<GpuImageFrame>) -> Vec<u8> {
    let output = &frame.output().outputs[0];
    let source = output.buffer.as_wgpu_buffer();
    let staging = backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("decoder pool output isolation readback"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut commands = backend
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("decoder pool output isolation commands"),
        });
    commands.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    let (sender, receiver) = mpsc::sync_channel(1);
    commands.map_buffer_on_submit(&staging, wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    let submission = backend.queue().submit([commands.finish()]);
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    let bytes = mapped[..output.layout.logical_size as usize].to_vec();
    drop(mapped);
    staging.unmap();
    bytes
}

#[test]
fn repeated_small_decode_reuses_only_transient_exact_matches() {
    let Some(backend) = backend() else {
        eprintln!("skipping decoder buffer-pool reuse test: no adapter");
        return;
    };
    let engine = WgpuSubmissionEngine::new(backend);
    let decoder = GpuDecoder::new(engine.clone());

    let first = decode_one(&decoder);
    let first_stats = engine.buffer_pool_stats();
    assert_eq!(first_stats.misses, PORTABLE_TRANSIENT_BUFFERS_PER_JOB);
    assert_eq!(first_stats.hits, 0);
    assert_eq!(first_stats.idle_buffers, PORTABLE_TRANSIENT_BUFFERS_PER_JOB);
    assert_eq!(first_stats.leased_buffers, 0);
    drop(first);

    let second = decode_one(&decoder);
    let second_stats = engine.buffer_pool_stats();
    assert_eq!(second_stats.misses, first_stats.misses);
    assert_eq!(second_stats.hits, PORTABLE_TRANSIENT_BUFFERS_PER_JOB);
    assert_eq!(
        second_stats.idle_buffers,
        PORTABLE_TRANSIENT_BUFFERS_PER_JOB
    );
    assert_eq!(second_stats.leased_buffers, 0);
    drop(second);
}

#[test]
fn eight_pending_jobs_hold_exclusive_leases_and_outputs_never_alias() {
    let Some(backend) = backend() else {
        eprintln!("skipping concurrent decoder buffer-pool test: no adapter");
        return;
    };
    let engine = WgpuSubmissionEngine::new(backend.clone());
    let decoder = GpuDecoder::new(engine.clone());
    let mut sessions = (0..8)
        .map(|_| decoder.open(indexed_gray8(), request()).unwrap())
        .collect::<Vec<_>>();
    for session in &mut sessions {
        let progress = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        assert_eq!(progress.submitted, 1);
    }

    let pending = engine.buffer_pool_stats();
    assert_eq!(
        pending.leased_buffers,
        8 * PORTABLE_TRANSIENT_BUFFERS_PER_JOB
    );
    assert_eq!(pending.idle_buffers, 0);
    assert_eq!(pending.misses, pending.leased_buffers);

    let frames = sessions
        .iter_mut()
        .map(|session| session.next_frame().unwrap().unwrap())
        .collect::<Vec<_>>();
    let completed = engine.buffer_pool_stats();
    assert_eq!(completed.leased_buffers, 0);
    assert_eq!(
        completed.idle_buffers,
        8 * PORTABLE_TRANSIENT_BUFFERS_PER_JOB
    );

    // Caller-owned outputs are deliberately absent from the pool. Mutating one live output must
    // not affect another concurrently retained frame.
    let first = frames[0].output().outputs[0].buffer.as_wgpu_buffer();
    backend
        .queue()
        .write_buffer(first, 0, &vec![0; first.size() as usize]);
    assert_eq!(read_output(&backend, &frames[1]), expected_pixels());
}

#[test]
fn clear_invalidates_in_flight_generation_without_disrupting_decode() {
    let Some(backend) = backend() else {
        eprintln!("skipping decoder buffer-pool generation test: no adapter");
        return;
    };
    let engine = WgpuSubmissionEngine::new(backend);
    let decoder = GpuDecoder::new(engine.clone());
    let mut session = decoder.open(indexed_gray8(), request()).unwrap();
    session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert_eq!(
        engine.buffer_pool_stats().leased_buffers,
        PORTABLE_TRANSIENT_BUFFERS_PER_JOB
    );

    let generation = engine.clear_buffer_pool();
    assert_eq!(generation, 1);
    let frame = session.next_frame().unwrap().unwrap();
    let stats = engine.buffer_pool_stats();
    assert_eq!(stats.generation, generation);
    assert_eq!(stats.leased_buffers, 0);
    assert_eq!(stats.idle_buffers, 0);
    assert_eq!(stats.evicted, PORTABLE_TRANSIENT_BUFFERS_PER_JOB);
    drop(frame);
}

#[test]
fn abandoned_session_recycles_only_after_map_completion() {
    let Some(backend) = backend() else {
        eprintln!("skipping abandoned decoder buffer-pool test: no adapter");
        return;
    };
    let engine = WgpuSubmissionEngine::new(backend.clone());
    let decoder = GpuDecoder::new(engine.clone());
    let mut session = decoder.open(indexed_gray8(), request()).unwrap();
    session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    drop(session);

    let deadline = Instant::now() + Duration::from_secs(5);
    while engine.buffer_pool_stats().leased_buffers != 0 && Instant::now() < deadline {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    let stats = engine.buffer_pool_stats();
    assert_eq!(stats.leased_buffers, 0);
    assert_eq!(stats.idle_buffers, PORTABLE_TRANSIENT_BUFFERS_PER_JOB);
    assert_eq!(engine.in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn byte_count_and_per_key_limits_are_hard_and_clearable() {
    let Some(backend) = backend() else {
        eprintln!("skipping decoder buffer-pool limit test: no adapter");
        return;
    };
    let engine = WgpuSubmissionEngine::new(backend);
    engine.set_buffer_pool_limits(WgpuDecodeBufferPoolLimits {
        max_idle_bytes: 64,
        max_idle_buffers: 2,
        max_idle_buffers_per_key: 1,
    });
    let decoder = GpuDecoder::new(engine.clone());
    let frame = decode_one(&decoder);
    let stats = engine.buffer_pool_stats();
    assert_eq!(stats.idle_buffers, 2);
    assert_eq!(stats.idle_bytes, 32);
    assert_eq!(stats.evicted, 3);
    assert!(stats.idle_bytes <= stats.limits.max_idle_bytes);
    drop(frame);

    engine.set_buffer_pool_limits(WgpuDecodeBufferPoolLimits {
        max_idle_bytes: 16,
        max_idle_buffers: 1,
        max_idle_buffers_per_key: 1,
    });
    let trimmed = engine.buffer_pool_stats();
    assert_eq!(trimmed.idle_buffers, 1);
    assert_eq!(trimmed.idle_bytes, 16);
    assert_eq!(trimmed.evicted, 4);

    assert_eq!(engine.clear_buffer_pool(), 1);
    let cleared = engine.buffer_pool_stats();
    assert_eq!(cleared.idle_buffers, 0);
    assert_eq!(cleared.idle_bytes, 0);
    assert_eq!(cleared.evicted, 5);
}
