use jxl_test_support::corpus::jpeg_reconstruction::CASES;
use jxl_test_support::gpu::buffer::read_bytes as read;
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{GpuPendingFrame, GpuSubmissionSession, VarDctSubmissionEngine};

#[test]
fn actual_jxl_gpu_reconstruction_matches_every_original_jpeg_byte() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("actual GPU adapter is required");
    eprintln!("Original JPEG adapter: {:?}", backend.adapter_info());
    let engine = VarDctSubmissionEngine::new(backend.clone()).unwrap();
    let bounded = engine
        .clone()
        .with_stream_window_limit(std::num::NonZeroU64::new(40).unwrap());
    for (case, engine, asynchronous) in CASES.iter().map(|case| (case, &engine, false)).chain(
        CASES
            .iter()
            .filter(|case| {
                matches!(
                    case.name,
                    "gray_restart"
                        | "rgb_progressive"
                        | "ycbcr420_progressive_restart"
                        | "ycbcr440_progressive_restart"
                        | "gray_long_eob"
                )
            })
            .map(|case| (case, &bounded, true)),
    ) {
        case.validate();
        eprintln!(
            "Original JPEG: {}, bounded asynchronous: {asynchronous}",
            case.name
        );
        let mut session = engine
            .open_jpeg_reconstruction(case.input, Default::default())
            .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
        let mut pending = session.submit_next().unwrap().unwrap();
        let frame = if asynchronous {
            pollster::block_on(std::future::poll_fn(|context| {
                std::pin::Pin::new(&mut pending).poll_complete(context)
            }))
        } else {
            pending.wait()
        }
        .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
        assert!(session.submit_next().unwrap().is_none());
        assert_eq!(
            frame.output.byte_len(),
            case.jpeg.len() as u64,
            "{} byte length",
            case.name
        );
        let actual = read(&backend, frame.output.buffer());
        if let Some(index) = actual.iter().zip(case.jpeg).position(|(a, b)| a != b) {
            panic!(
                "{} byte {index}: {} != {}",
                case.name, actual[index], case.jpeg[index]
            );
        }
        assert!(actual[case.jpeg.len()..].iter().all(|byte| *byte == 0));
        let retained = frame.output.clone();
        let bytes = retained.buffer().size();
        drop(frame);
        drop(session);
        assert_eq!(engine.in_flight_memory_stats().reserved_bytes, bytes);
        drop(retained);
        assert_eq!(engine.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn host_limits_and_external_metadata_reject_without_gpu_admission() {
    use jxl_gpu_bitstream::{ContainerBox, metadata::MetadataLimits};
    use jxl_wgpu_decode::{Error, JpegReconstructionError as JpegError, JpegReconstructionLimits};
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("actual GPU required");
    let engine = VarDctSubmissionEngine::new(backend).unwrap();
    let case = CASES
        .iter()
        .find(|case| case.name == "metadata_combined")
        .unwrap();
    for limits in [
        JpegReconstructionLimits {
            max_tasks: 0,
            ..Default::default()
        },
        JpegReconstructionLimits {
            max_plan_bytes: 0,
            ..Default::default()
        },
        JpegReconstructionLimits {
            max_output_bytes: 0,
            ..Default::default()
        },
        JpegReconstructionLimits {
            metadata: MetadataLimits {
                max_boxes: 0,
                ..Default::default()
            },
            ..Default::default()
        },
        JpegReconstructionLimits {
            metadata: MetadataLimits {
                max_total_decoded_bytes: 0,
                ..Default::default()
            },
            ..Default::default()
        },
    ] {
        let error = engine
            .open_jpeg_reconstruction(case.input, limits)
            .unwrap_err();
        assert!(
            matches!(
                error,
                Error::JpegReconstruction(JpegError::Limit { .. })
                    | Error::ContainerMetadata(_)
                    | Error::VarDct(jxl_wgpu_decode::VarDctDecodeError::Jpeg(
                        jxl_wgpu_decode::JpegCoefficientError::Metadata(
                            jxl_gpu_bitstream::jpeg_reconstruction::JpegReconstructionError::Limit { .. }
                        )
                    ))
            ),
            "{error:?}"
        );
        assert_eq!(engine.in_flight_memory_stats().reserved_bytes, 0);
    }
    let parsed = jxl_gpu_bitstream::parse(case.input, Default::default()).unwrap();
    let mut boxes: Vec<_> = parsed
        .auxiliary_boxes()
        .iter()
        .map(|item| ContainerBox {
            box_type: item.box_type,
            payload: item.payload,
        })
        .collect();
    let exif = boxes
        .iter()
        .position(|item| {
            item.box_type == *b"Exif"
                || (item.box_type == *b"brob" && item.payload.starts_with(b"Exif"))
        })
        .unwrap();
    let original = boxes[exif];
    boxes.push(original);
    let duplicate =
        jxl_gpu_bitstream::write_container_with_boxes(parsed.codestream(), &boxes).unwrap();
    assert!(matches!(
        engine.open_jpeg_reconstruction(&duplicate, Default::default()),
        Err(Error::JpegReconstruction(JpegError::Invalid(
            "duplicate JPEG metadata box"
        )))
    ));
    boxes.pop();
    boxes.remove(exif);
    let missing =
        jxl_gpu_bitstream::write_container_with_boxes(parsed.codestream(), &boxes).unwrap();
    assert!(matches!(
        engine.open_jpeg_reconstruction(&missing, Default::default()),
        Err(Error::JpegReconstruction(JpegError::Invalid(
            "Exif binding"
        )))
    ));
    assert_eq!(engine.in_flight_memory_stats().reserved_bytes, 0);
}
