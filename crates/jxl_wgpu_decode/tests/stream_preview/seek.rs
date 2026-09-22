//! The seek frontend uses the same source-ownership engine probe as early previews.
use super::*;
use jxl_gpu_bitstream::{FrameIndex, FrameIndexError};
use jxl_wgpu_decode::{BoundFrameIndex, FrameSeekError, FrameSeekLimits, GpuDecodeSeekStream};

fn index(raw: &[u8]) -> FrameIndex {
    BoundFrameIndex::new(Arc::new(inventory(raw)), None, Default::default())
        .unwrap()
        .index()
        .clone()
}

fn boxed(raw: &[u8], index: &FrameIndex) -> Vec<u8> {
    jxl_gpu_bitstream::write_container_with_boxes(
        raw,
        &[ContainerBox {
            box_type: *b"jxli",
            payload: &index.encode(Default::default()).unwrap(),
        }],
    )
    .unwrap()
}

fn feed(stream: &mut GpuDecodeSeekStream<Engine>, bytes: &[u8], end: bool) {
    let mut scanner = ContainerStreamScanner::new(Default::default());
    for chunk in bytes.chunks(43) {
        for event in scanner.push_chunk(Arc::from(chunk)).unwrap() {
            stream.push_transport_event(&event).unwrap();
        }
    }
    if end {
        for event in scanner.finish_input().unwrap() {
            stream.push_transport_event(&event).unwrap();
        }
    }
}

fn no_input(decoder: &GpuDecoder<Engine>) {
    let state = decoder.incremental_input_budget().snapshot();
    assert_eq!((state.reserved_bytes, state.reserved_spans), (0, 0));
    assert!(
        decoder
            .engine()
            .opened
            .lock()
            .unwrap()
            .iter()
            .all(|weak| weak.upgrade().is_none())
    );
}

#[test]
fn streamed_seek_opens_only_after_end_and_transfers_original_shared_ranges() {
    for mode in ["modular", "vardct"] {
        let raw = fixture(mode);
        let original = inventory(&raw);
        let index = index(&raw);
        for data in [raw.clone(), boxed(&raw, &index)] {
            for split in 0..=data.len() {
                let decoder = GpuDecoder::new(Engine::default());
                let mut stream = decoder.stream_seek(request(), Default::default()).unwrap();
                let mut scanner = ContainerStreamScanner::new(decoder.container_stream_limits());
                let mut weak = Vec::new();
                for chunk in [&data[..split], &data[split..]] {
                    let storage: Arc<[u8]> = Arc::from(chunk);
                    weak.push(Arc::downgrade(&storage));
                    for event in scanner.push_chunk(storage).unwrap() {
                        stream.push_transport_event(&event).unwrap();
                    }
                }
                assert!(!stream.is_ready());
                assert!(decoder.engine().opened.lock().unwrap().is_empty());
                for event in scanner.finish_input().unwrap() {
                    stream.push_transport_event(&event).unwrap();
                }
                assert!(stream.is_ready());
                assert!(stream.index_stats().authoritative_end);
                let spans = stream.stats().retained_spans;
                let seek = stream.finish(0, Default::default()).unwrap();
                let source = last_source(&decoder);
                assert_eq!(source.bytes.span_count(), spans);
                assert_eq!(source.bytes.logical_bytes(), raw.len() as u64);
                if spans > 1 {
                    assert!(source.bytes.contiguous_bytes().is_none());
                }
                let mut actual = vec![0; raw.len()];
                source
                    .bytes
                    .copy_range(0..raw.len() as u64, &mut actual)
                    .unwrap();
                assert_eq!(actual, raw);
                assert_eq!(
                    source.inventory.source_inventory().complete_inventory(),
                    Some(&original)
                );
                let main = source.inventory.reconstruction_inventory();
                assert!(
                    main.frames
                        .iter()
                        .all(|frame| original.frames.contains(frame))
                );
                assert_eq!(seek.plan().target().index, 0);
                drop((source, seek, scanner));
                assert!(weak.iter().all(|owner| owner.upgrade().is_none()));
                no_input(&decoder);
            }
        }
    }
}

#[test]
fn seek_handoff_failure_drops_input_and_never_opens_engine_for_invalid_authority() {
    let raw = fixture("modular");
    let index = index(&raw);
    let mut wrong_entries = index.entries().to_vec();
    wrong_entries[0].codestream_offset += 1;
    let wrong = FrameIndex::new(
        index.tick_numerator(),
        index.tick_denominator(),
        wrong_entries,
        Default::default(),
    )
    .unwrap();
    for case in 0..5 {
        let decoder = GpuDecoder::new(Engine::default());
        let mut stream = decoder.stream_seek(request(), Default::default()).unwrap();
        feed(
            &mut stream,
            &boxed(&raw, if case == 1 { &wrong } else { &index }),
            case != 0,
        );
        decoder
            .engine()
            .reject_next
            .store(case == 4, Ordering::Release);
        let result = stream.finish(
            usize::from(case == 2),
            FrameSeekLimits {
                max_physical_frames: if case == 3 { 0 } else { 16_384 },
                ..Default::default()
            },
        );
        assert!(
            match (case, result) {
                (0, Err(Error::IncrementalInputIncomplete))
                | (1, Err(Error::FrameSeek(FrameSeekError::Offset { .. })))
                | (2, Err(Error::FrameSeek(FrameSeekError::Target { .. })))
                | (3, Err(Error::FrameSeek(FrameSeekError::PhysicalLimit { .. }))) => true,
                (4, Err(_)) => !decoder.engine().reject_next.load(Ordering::Acquire),
                _ => false,
            },
            "case {case}"
        );
        assert!(decoder.engine().opened.lock().unwrap().is_empty());
        no_input(&decoder);
    }
    let decoder = GpuDecoder::new(Engine::default());
    assert!(matches!(
        decoder.stream_seek(
            request().with_image_selection(ImageSelection::Preview),
            Default::default()
        ),
        Err(Error::FrameSeek(FrameSeekError::PreviewSelection))
    ));
}

#[test]
fn seek_byte_and_span_pressure_retries_identical_event_without_advancing_index() {
    let raw = fixture("modular");
    let boxed = boxed(&raw, &index(&raw));
    for span_limit in [false, true] {
        let mut scanner = ContainerStreamScanner::new(Default::default());
        let events = scanner.push_chunk(Arc::from(boxed.clone())).unwrap();
        let span_count = events
            .iter()
            .filter(|event| {
                matches!(event,
                    ContainerStreamEvent::CodestreamChunk { bytes, .. } if !bytes.is_empty()
                )
            })
            .count();
        let budget = IncrementalInputBudget::with_limits(
            NonZeroU64::new(if span_limit { 4096 } else { raw.len() as u64 }).unwrap(),
            NonZeroUsize::new(if span_limit { span_count } else { 100 }).unwrap(),
        );
        let decoder =
            GpuDecoder::new(Engine::default()).with_incremental_input_budget(budget.clone());
        let mut blocker = decoder.stream(request()).unwrap();
        push(&mut blocker, &raw[..1], 0).unwrap();
        let mut blocker = Some(blocker);
        let mut stream = decoder.stream_seek(request(), Default::default()).unwrap();
        let mut retried = false;
        for event in events {
            let before = (stream.stats(), stream.index_stats());
            if let Err(error) = stream.push_transport_event(&event) {
                assert!(matches!(error, Error::IncrementalInputBudget(_)), "{error}");
                assert_eq!((stream.stats(), stream.index_stats()), before);
                assert!(before.1.retained_entries > 0);
                drop(blocker.take());
                stream.push_transport_event(&event).unwrap();
                retried = true;
            }
        }
        assert!(retried);
        for event in scanner.finish_input().unwrap() {
            stream.push_transport_event(&event).unwrap();
        }
        let seek = stream.finish(0, Default::default()).unwrap();
        assert_eq!(budget.snapshot().reserved_bytes, raw.len() as u64);
        drop(seek);
        no_input(&decoder);
    }
}

#[test]
fn malformed_late_index_releases_main_input_but_keeps_independent_preview() {
    let raw = fixture("modular");
    let original = inventory(&raw);
    let end = preview_end(&original);
    let decoder = GpuDecoder::new(Engine::default());
    let mut stream = decoder.stream_seek(request(), Default::default()).unwrap();
    for (offset, chunk) in [(0, &raw[..end]), (end, &raw[end..])] {
        stream
            .push_transport_event(&ContainerStreamEvent::CodestreamChunk {
                logical_offset: offset as u64,
                bytes: StreamSlice::from_shared(Arc::from(chunk)),
            })
            .unwrap();
    }
    let preview = stream.take_preview(request()).unwrap().unwrap();
    let malformed = jxl_gpu_bitstream::write_container_with_boxes(
        &raw,
        &[ContainerBox {
            box_type: *b"jxli",
            payload: &[0],
        }],
    )
    .unwrap();
    let mut scanner = ContainerStreamScanner::new(Default::default());
    let mut failed = false;
    for event in scanner.push_chunk(Arc::from(malformed)).unwrap() {
        if matches!(event, ContainerStreamEvent::CodestreamChunk { .. }) {
            continue;
        }
        if let Err(error) = stream.push_transport_event(&event) {
            assert!(matches!(
                error,
                Error::FrameSeek(FrameSeekError::Index(FrameIndexError::Empty))
            ));
            failed = true;
            break;
        }
    }
    assert!(failed);
    assert!(stream.index_stats().failed);
    assert_eq!(stream.stats().retained_codestream_bytes, 0);
    assert!(matches!(
        stream.finish(0, Default::default()),
        Err(Error::IncrementalInputPoisoned)
    ));
    check_preview(&last_source(&decoder), &original, &raw);
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        end as u64
    );
    drop(preview);
    no_input(&decoder);
}

#[test]
fn invalid_input_completion_drops_already_parsed_index_and_source() {
    let raw = fixture("modular");
    let data = boxed(&raw, &index(&raw));
    for repeated_end in [false, true] {
        let decoder = GpuDecoder::new(Engine::default());
        let mut stream = decoder.stream_seek(request(), Default::default()).unwrap();
        feed(&mut stream, &data, repeated_end);
        assert!(stream.index_stats().retained_entries > 0);
        assert!(stream.stats().retained_codestream_bytes > 0);
        // An inconsistent first End or a duplicate End cannot authorize a seek.
        assert!(
            stream
                .push_transport_event(&ContainerStreamEvent::End {
                    codestream_bytes: raw.len() as u64 + u64::from(!repeated_end),
                    is_container: true,
                })
                .is_err()
        );
        assert!(!stream.is_ready());
        assert_eq!(
            stream.index_stats(),
            jxl_gpu_bitstream::FrameIndexCollectorStats {
                failed: true,
                ..Default::default()
            }
        );
        assert_eq!(stream.stats().retained_spans, 0);
        assert!(matches!(
            stream.finish(0, Default::default()),
            Err(Error::IncrementalInputPoisoned)
        ));
        assert!(decoder.engine().opened.lock().unwrap().is_empty());
        no_input(&decoder);
    }
}
