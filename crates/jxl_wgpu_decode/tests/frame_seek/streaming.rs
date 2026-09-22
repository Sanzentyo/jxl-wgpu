use super::*;
use jxl_gpu_bitstream::{ContainerStreamScanner, FrameIndexLimits};
use jxl_wgpu_decode::{Error, GpuDecodeSeekStream};
use std::task::{Context, Poll, Waker};

pub(super) fn receive(
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    data: &[u8],
    request: GpuOutputRequest,
) -> GpuDecodeSeekStream<WgpuDecodeEngine> {
    let mut stream = decoder
        .stream_seek(request, FrameIndexLimits::default())
        .unwrap();
    let mut scanner = ContainerStreamScanner::new(decoder.container_stream_limits());
    for chunk in data.chunks(43) {
        for event in scanner.push_chunk(Arc::from(chunk)).unwrap() {
            stream.push_transport_event(&event).unwrap();
        }
    }
    assert!(!stream.is_ready());
    for event in scanner.finish_input().unwrap() {
        stream.push_transport_event(&event).unwrap();
    }
    assert!(stream.is_ready());
    stream
}

fn reordered(data: &[u8], index: &FrameIndex) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let raw = parsed.codestream();
    // Version 1 permits a future final fragment before fragment 0; the index is between them.
    let mut result = jxl_gpu_bitstream::write_container(raw).unwrap()[..32].to_vec();
    result[27] = 1;
    let middle = raw.len() / 2;
    for (id, bytes) in [(0x8000_0001u32, &raw[middle..]), (0, &raw[..middle])] {
        result.extend_from_slice(&(bytes.len() as u32 + 12).to_be_bytes());
        result.extend_from_slice(b"jxlp");
        result.extend_from_slice(&id.to_be_bytes());
        result.extend_from_slice(bytes);
        if id != 0 {
            let payload = index.encode(Default::default()).unwrap();
            result.extend_from_slice(&(payload.len() as u32 + 8).to_be_bytes());
            result.extend_from_slice(b"jxli");
            result.extend_from_slice(&payload);
        }
    }
    result
}

fn input_released(decoder: &GpuDecoder<WgpuDecodeEngine>) {
    let state = decoder.incremental_input_budget().snapshot();
    assert_eq!((state.reserved_bytes, state.reserved_spans), (0, 0));
}

#[test]
fn incremental_seeks_match_contiguous_updates_and_native_final_pixels() {
    let backend = backend();
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    for name in [
        "sequence_modular_rgb12",
        "composition_vardct_dc",
        "composition_gray_alpha",
        "noise/mixed_frames",
        "patches/progressive/vardct",
        "preview/animation_modular",
    ] {
        let data = source(name);
        let bound = BoundFrameIndex::new(inventory(&data), None, Default::default()).unwrap();
        let selected = SelectedImageInventory::new(inventory(&data), ImageSelection::Main).unwrap();
        let count = FrameExecutionPlan::negotiate_selected(
            &selected,
            jxl_wgpu_decode::OrientationPolicy::Apply,
        )
        .unwrap()
        .presentations
        .len();
        let target = (count - 1).min(2);
        let request = request().with_progressive_output(true);
        let mut baseline = decoder
            .open_seek(
                &data,
                request.clone(),
                target,
                Default::default(),
                Default::default(),
            )
            .unwrap();
        let mut expected = Vec::new();
        while let Some(frame) = baseline.next_update().unwrap() {
            expected.push((
                frame.metadata.clone(),
                frame.progression(),
                planes::read_bytes(&backend, &frame.output().outputs[0]),
            ));
        }
        drop(baseline);
        let native = native_updates(&data, false)
            .expect("native oracle required")
            .into_iter()
            .filter(|frame| frame.complete)
            .nth(target)
            .unwrap();
        let pixels = &expected.last().unwrap().2;
        let oracle: Vec<_> = native
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| {
                let value = f32::from_le_bytes(*word);
                assert!(value.is_finite());
                (value.clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect();
        assert_eq!(pixels.len(), oracle.len());
        assert!(
            pixels
                .iter()
                .zip(&oracle)
                .all(|(&a, &b)| a.abs_diff(b) <= 1),
            "{name}: native pixels"
        );
        assert_eq!(expected.last().unwrap().0.duration.ticks, native.duration);
        for boxed in [
            indexed(&data, bound.index(), false),
            reordered(&data, bound.index()),
        ] {
            let stream = receive(&decoder, &boxed, request.clone());
            assert!(stream.stats().retained_spans > 1);
            assert!(stream.index_stats().retained_entries > 0);
            let mut seek = stream.finish(target, Default::default()).unwrap();
            for expected in &expected {
                let frame = pollster::block_on(seek.next_update_async())
                    .unwrap()
                    .unwrap();
                assert_eq!(frame.metadata, expected.0, "{name}");
                assert_eq!(frame.progression(), expected.1, "{name}");
                let pixels = planes::read_bytes(&backend, &frame.output().outputs[0]);
                assert!(pixels == expected.2, "{name}: incremental pixels");
            }
            assert!(seek.next_update().unwrap().is_none());
            drop(seek);
            released(&backend);
            input_released(&decoder);
        }
    }
}

#[test]
fn incremental_seek_cancellation_and_completion_release_input_with_retained_updates() {
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
        let data = source(name);
        let bound = BoundFrameIndex::new(inventory(&data), None, Default::default()).unwrap();
        let data = indexed(&data, bound.index(), true);
        for boundary in 0..4 {
            let stream = receive(&decoder, &data, request().with_progressive_output(true));
            assert!(decoder.incremental_input_budget().snapshot().reserved_bytes > 0);
            if boundary == 0 {
                drop(stream);
                input_released(&decoder);
                continue;
            }
            let mut seek = stream.finish(2, Default::default()).unwrap();
            let budget = backend.transient_memory_budget();
            let blocker = budget
                .try_reserve(budget.snapshot().available_bytes)
                .unwrap();
            assert!(matches!(
                seek.next_frame(),
                Err(Error::MemoryBackpressure(_))
            ));
            drop(blocker);
            let held = if boundary == 1 {
                assert!(matches!(
                    seek.poll_next_update(&mut Context::from_waker(Waker::noop())),
                    Poll::Pending
                ));
                None
            } else {
                let frame = pollster::block_on(seek.next_update_async())
                    .unwrap()
                    .unwrap();
                assert!(!frame.is_complete());
                let pixels = planes::read_bytes(&backend, &frame.output().outputs[0]);
                if boundary == 3 {
                    drop(seek.next_frame().unwrap().unwrap());
                }
                Some((frame, pixels))
            };
            drop(seek);
            let retained = held
                .as_ref()
                .map_or(0, |(frame, _)| frame.output().outputs[0].buffer.size());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while (budget.snapshot().reserved_bytes != retained
                || backend.submission_poller().in_flight() != 0
                || decoder.incremental_input_budget().snapshot().reserved_bytes != 0)
                && std::time::Instant::now() < deadline
            {
                backend.device().poll(wgpu::PollType::Poll).unwrap();
                std::thread::yield_now();
            }
            assert_eq!(budget.snapshot().reserved_bytes, retained);
            assert_eq!(backend.submission_poller().in_flight(), 0);
            input_released(&decoder);
            if let Some((frame, pixels)) = &held {
                assert!(planes::read_bytes(&backend, &frame.output().outputs[0]) == *pixels);
            }
            drop(held);
            released(&backend);
        }
    }
}

#[test]
fn streamed_required_entropy_failure_never_yields_a_target_lease() {
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
        let data = source(name);
        let data = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream()
            .to_vec();
        let inventory = inventory(&data);
        let bound = BoundFrameIndex::new(inventory.clone(), None, Default::default()).unwrap();
        let plan = bound.seek(2, Default::default()).unwrap();
        assert!(plan.preroll_presentations() > 0);
        for physical in [plan.physical_frames().start, plan.physical_frames().end - 1] {
            let section = inventory.frames[physical as usize]
                .sections
                .iter()
                .max_by_key(|section| section.bytes.length)
                .unwrap();
            assert!(section.bytes.length > 32);
            let mut damaged = data.clone();
            let end = section.bytes.end().unwrap() as usize;
            damaged[end - 16..end].fill(255);
            let stream = receive(&decoder, &indexed(&damaged, bound.index(), true), request());
            let mut seek = stream.finish(2, Default::default()).unwrap();
            assert!(matches!(
                pollster::block_on(seek.next_frame_async()),
                Err(Error::ModularEntropyRejected { .. } | Error::VarDct(_))
            ));
            assert!(matches!(seek.next_frame(), Err(Error::SessionPoisoned)));
            drop(seek);
            released(&backend);
            input_released(&decoder);
        }
    }
}
