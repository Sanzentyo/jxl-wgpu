use super::*;

#[test]
fn raw_matrix_windowed_public_decode_matches_both_oracles_and_releases_memory() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let extent = Extent2d::new(264, 64);
    let expected = rust_jxl_rgb8(common::jpeg_transcode_raw_matrix(), extent);
    for encoded in [
        common::jpeg_transcode_raw_matrix(),
        common::jpeg_transcode_raw_matrix_local(),
        common::jpeg_transcode_raw_matrix_local_packets(),
    ] {
        assert_eq!(
            rust_jxl_rgb8(encoded, extent),
            expected,
            "only descriptors change"
        );
        let djxl = djxl_rgb8(encoded, extent);
        let mut whole_output = None;
        let mut whole_submissions = 0;
        for cap in [u64::MAX, 40, 64, 256] {
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
            );
            let request = || GpuOutputRequest::color(vardct_rgb8_format()).unwrap();
            let mut session = if cap == u64::MAX {
                decoder.open(encoded, request()).unwrap()
            } else {
                let mut stream = decoder.stream(request()).unwrap();
                let mut transport = ContainerStreamScanner::new(decoder.container_stream_limits());
                for chunk in encoded.chunks(7) {
                    for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
                        stream.push_transport_event(&event).unwrap();
                    }
                }
                for event in transport.finish_input().unwrap() {
                    stream.push_transport_event(&event).unwrap();
                }
                assert!(stream.stats().retained_spans > 2);
                stream.finish().unwrap()
            };
            let frame = pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap();
            assert!(session.next_frame().unwrap().is_none());
            let submissions = session.submission_session().submissions_per_frame();
            let readback = ImageReadbackPipeline::new(&backend)
                .submit(frame.output())
                .unwrap()
                .wait()
                .unwrap();
            let actual = &readback.frame.outputs[0].bytes;
            assert!(maximum_error(actual, &expected) <= 1, "cap {cap}: Rust jxl");
            if let Some(reference) = &djxl {
                assert!(maximum_error(actual, reference) <= 1, "cap {cap}: djxl");
            }
            if let Some(whole_output) = &whole_output {
                assert_eq!(actual, whole_output, "cap {cap}");
                assert!(submissions >= whole_submissions, "cap {cap}");
                if cap <= 64 {
                    assert!(
                        submissions > whole_submissions,
                        "cap {cap}: continuation submissions"
                    );
                }
            } else {
                whole_output = Some(actual.clone());
                whole_submissions = submissions;
            }
            drop(readback);
            drop(frame);
            drop(session);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}
