use super::*;
use jxl_wgpu::GpuImageFrame;
use jxl_wgpu_decode::{
    Error, FrameProgression, GpuFrameLease, NumericSampleMapping, OrientationPolicy,
};
use std::task::{Context, Poll, Waker};

fn bytes_equal(actual: &[u8], expected: &[u8], label: &str) {
    assert!(
        actual == expected,
        "{label}: pixel mismatch at {:?}",
        actual.iter().zip(expected).position(|(a, b)| a != b)
    );
}

fn snapshot(
    backend: &WgpuBackend,
    frame: &GpuFrameLease<GpuImageFrame>,
) -> (
    jxl_wgpu_decode::FrameMetadata,
    Option<FrameProgression>,
    Vec<u8>,
) {
    (
        frame.metadata.clone(),
        frame.progression(),
        planes::read_bytes(backend, &frame.output().outputs[0]),
    )
}

#[test]
fn seek_updates_keep_physical_progression_orientation_extras_and_immutable_output() {
    let backend = backend();
    let whole = GpuDecoder::wgpu(backend.clone()).unwrap();
    let bounded = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    for name in [
        "composition_vardct_dc",
        "composition_associated_vardct",
        "modular_composition/modular_pass_gray_alpha",
        "patches/progressive/vardct",
        "patches/progressive/modular",
        "preview/animation_vardct",
    ] {
        let data = source(name);
        let source = inventory(&data);
        let index = BoundFrameIndex::new(source.clone(), None, Default::default()).unwrap();
        let boxed = indexed(&data, index.index(), true);
        let mut requests = vec![request().with_progressive_output(true)];
        if !source.image_header.extra_channels.is_empty() {
            requests.push(
                GpuOutputRequest::numeric(
                    jxl_gpu_formats::vpi::VpiPitchLinearFormat::F32.pixel_format(),
                    NumericSampleMapping::NormalizedUnsigned,
                )
                .unwrap()
                .with_extra_channel(0)
                .unwrap()
                .with_progressive_output(true),
            );
        }
        for request in requests {
            for orientation in [OrientationPolicy::Apply, OrientationPolicy::Keep] {
                let request = request.clone().with_orientation_policy(orientation);
                let mut baseline = whole.open(&data, request.clone()).unwrap();
                let mut expected = Vec::new();
                while let Some(frame) = baseline.next_update().unwrap() {
                    expected.push(snapshot(&backend, &frame));
                }
                let count = baseline.frames_submitted();
                drop(baseline);
                for target in 0..count {
                    let mut seek = bounded
                        .open_seek(
                            &boxed,
                            request.clone(),
                            target,
                            Default::default(),
                            Default::default(),
                        )
                        .unwrap();
                    let wanted: Vec<_> = expected
                        .iter()
                        .filter(|record| record.0.index == target)
                        .collect();
                    let mut held = Vec::new();
                    while let Some(frame) = pollster::block_on(seek.next_update_async()).unwrap() {
                        let actual = snapshot(&backend, &frame);
                        let expected = wanted[held.len()];
                        assert_eq!(actual.0, expected.0, "{name}/{target}");
                        assert_eq!(actual.1, expected.1, "{name}/{target}");
                        bytes_equal(&actual.2, &expected.2, name);
                        held.push((frame, actual.2));
                    }
                    assert_eq!(held.len(), wanted.len(), "{name}/{target}");
                    drop(seek);
                    let retained: u64 = held
                        .iter()
                        .map(|(frame, _)| frame.output().outputs[0].buffer.size())
                        .sum();
                    assert_eq!(
                        backend.transient_memory_budget().snapshot().reserved_bytes,
                        retained
                    );
                    for (frame, expected) in &held {
                        bytes_equal(
                            &planes::read_bytes(&backend, &frame.output().outputs[0]),
                            expected,
                            name,
                        );
                    }
                    drop(held);
                    released(&backend);
                }
            }
        }
    }
}

#[test]
fn seek_admission_retries_and_cancellation_release_source_and_gpu_work() {
    let backend = backend();
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    for name in [
        "composition_vardct_dc",
        "modular_composition/modular_pass_rgb",
    ] {
        for boundary in 0..=2 {
            let data: Arc<[u8]> = source(name).into();
            let weak = Arc::downgrade(&data);
            let mut seek = decoder
                .open_seek_shared(
                    data,
                    request().with_progressive_output(true),
                    2,
                    Default::default(),
                    Default::default(),
                )
                .unwrap();
            assert!(seek.plan().preroll_presentations() > 0);
            let budget = backend.transient_memory_budget();
            let blocker = budget
                .try_reserve(budget.snapshot().available_bytes)
                .unwrap();
            for _ in 0..2 {
                assert!(matches!(
                    seek.next_frame(),
                    Err(Error::MemoryBackpressure(_))
                ));
                assert_eq!(seek.frames_submitted(), 0);
            }
            drop(blocker);
            let mut held = None;
            if boundary == 0 {
                let mut context = Context::from_waker(Waker::noop());
                assert!(matches!(seek.poll_next_update(&mut context), Poll::Pending));
                assert!(seek.frames_submitted() > 0);
            } else {
                let frame = pollster::block_on(seek.next_update_async())
                    .unwrap()
                    .unwrap();
                assert!(!frame.is_complete());
                let expected = planes::read_bytes(&backend, &frame.output().outputs[0]);
                held = Some((frame, expected));
                if boundary == 2 {
                    drop(seek.next_frame().unwrap().unwrap());
                    assert!(seek.next_update().unwrap().is_none());
                }
            }
            drop(seek);
            let retained = held
                .as_ref()
                .map_or(0, |(frame, _)| frame.output().outputs[0].buffer.size());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while (budget.snapshot().reserved_bytes != retained
                || weak.upgrade().is_some()
                || backend.submission_poller().in_flight() != 0)
                && std::time::Instant::now() < deadline
            {
                backend.device().poll(wgpu::PollType::Poll).unwrap();
                std::thread::yield_now();
            }
            assert_eq!(budget.snapshot().reserved_bytes, retained);
            assert!(weak.upgrade().is_none(), "cancelled source retained");
            if let Some((frame, expected)) = &held {
                bytes_equal(
                    &planes::read_bytes(&backend, &frame.output().outputs[0]),
                    expected,
                    name,
                );
            }
            drop(held);
            released(&backend);
        }
    }
}

#[test]
fn malformed_index_and_required_entropy_never_publish_an_unvalidated_target() {
    let backend = backend();
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    for name in [
        "modular_composition/modular_pass_rgb",
        "composition_associated_vardct",
    ] {
        let transport = source(name);
        let data = jxl_gpu_bitstream::parse(&transport, Default::default())
            .unwrap()
            .codestream()
            .to_vec();
        let inventory = inventory(&data);
        let execution = FrameExecutionPlan::negotiate(&inventory).unwrap();
        let index = BoundFrameIndex::new(inventory.clone(), None, Default::default()).unwrap();
        let target = 2;
        let plan = index.seek(target, Default::default()).unwrap();
        assert!(plan.preroll_presentations() > 0);
        for damage_target in [false, true] {
            let physical = if damage_target {
                execution.presentations[target].physical_frames.end - 1
            } else {
                plan.physical_frames().start as usize
            };
            let frame = &inventory.frames[physical];
            let section = frame
                .sections
                .iter()
                .max_by_key(|section| section.bytes.length)
                .unwrap();
            assert!(section.bytes.length > 32);
            let mut damaged = data.clone();
            let end = section.bytes.end().unwrap() as usize;
            damaged[end - 16..end].fill(0xff);
            let boxed = indexed(&damaged, index.index(), true);
            let mut seek = decoder
                .open_seek(
                    &boxed,
                    request().with_progressive_output(true),
                    target,
                    Default::default(),
                    Default::default(),
                )
                .unwrap();
            // Final-only consumption may use validated intermediates internally, but must never
            // turn a corrupt required physical frame into a target lease.
            let error = if damage_target {
                pollster::block_on(seek.next_frame_async()).unwrap_err()
            } else {
                // Even opt-in updates must suppress every required preroll presentation.
                pollster::block_on(seek.next_update_async()).unwrap_err()
            };
            assert!(
                matches!(
                    error,
                    Error::ModularEntropyRejected { .. } | Error::VarDct(_)
                ),
                "{name}: {error:?}"
            );
            assert!(matches!(seek.next_update(), Err(Error::SessionPoisoned)));
            drop(seek);
            released(&backend);
        }
        let mut entries = index.index().entries().to_vec();
        entries[0].codestream_offset += 1;
        let wrong = FrameIndex::new(
            index.index().tick_numerator(),
            index.index().tick_denominator(),
            entries,
            Default::default(),
        )
        .unwrap();
        assert!(matches!(
            decoder.open_seek(
                &indexed(&data, &wrong, false),
                request(),
                target,
                Default::default(),
                Default::default()
            ),
            Err(Error::FrameSeek(FrameSeekError::Offset { .. }))
        ));
        released(&backend);
    }
}

#[test]
fn an_independent_seek_skips_prior_entropy_without_claiming_it_validated() {
    let backend = backend();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let transport = source("sequence_modular_rgb12");
    let data = jxl_gpu_bitstream::parse(&transport, Default::default())
        .unwrap()
        .codestream()
        .to_vec();
    let inventory = inventory(&data);
    let bound = BoundFrameIndex::new(inventory.clone(), None, Default::default()).unwrap();
    let target = 2;
    assert_eq!(
        bound
            .seek(target, Default::default())
            .unwrap()
            .restart_presentation(),
        target
    );
    let mut baseline = decoder.open(&data, request()).unwrap();
    for _ in 0..target {
        drop(baseline.next_frame().unwrap().unwrap());
    }
    let baseline_frame = baseline.next_frame().unwrap().unwrap();
    let expected = snapshot(&backend, &baseline_frame);
    drop(baseline_frame);
    drop(baseline);
    let first = &inventory.frames[0];
    let section = first
        .sections
        .iter()
        .max_by_key(|section| section.bytes.length)
        .unwrap();
    assert!(section.bytes.length > 32);
    let end = section.bytes.end().unwrap() as usize;
    let mut damaged = data;
    damaged[end - 16..end].fill(0xff);
    let mut full = decoder.open(&damaged, request()).unwrap();
    assert!(matches!(
        full.next_frame(),
        Err(Error::ModularEntropyRejected { .. })
    ));
    drop(full);
    let mut seek = decoder
        .open_seek(
            &indexed(&damaged, bound.index(), true),
            request(),
            target,
            Default::default(),
            FrameSeekLimits {
                max_preroll_presentations: 0,
                max_physical_frames: 1,
            },
        )
        .unwrap();
    let frame = seek.next_frame().unwrap().unwrap();
    assert_eq!(frame.metadata, expected.0);
    bytes_equal(
        &planes::read_bytes(&backend, &frame.output().outputs[0]),
        &expected.2,
        "independent target",
    );
    assert_eq!(seek.frames_submitted(), 1);
    drop((frame, seek));
    released(&backend);
}
