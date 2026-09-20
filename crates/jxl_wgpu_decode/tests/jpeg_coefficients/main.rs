use jxl_test_support::corpus::jpeg_reconstruction::CASES;
use jxl_test_support::oracles::jpeg_coefficients::JpegCoefficientOracle;
use jxl_wgpu::{GpuBufferLease, WgpuBackend};
use jxl_wgpu_decode::{GpuPendingFrame, GpuSubmissionSession, VarDctSubmissionEngine};

fn read(backend: &WgpuBackend, source: &GpuBufferLease) -> Vec<i32> {
    let buffer = backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("JPEG coefficient conformance readback"),
        size: source.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut commands = backend.device().create_command_encoder(&Default::default());
    let access = source.try_acquire_gpu_submission().unwrap();
    commands.copy_buffer_to_buffer(source.as_wgpu_buffer(), 0, &buffer, 0, buffer.size());
    backend.queue().submit([commands.finish()]);
    drop(access);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap()
        });
    backend
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    let words = mapped
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| i32::from_le_bytes(*word))
        .collect();
    drop(mapped);
    buffer.unmap();
    words
}

#[test]
fn actual_jxl_gpu_quantizers_and_coefficients_match_original_jpeg_including_mcu_padding() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("actual GPU adapter is required");
    eprintln!("JPEG coefficient adapter: {:?}", backend.adapter_info());
    let engine = VarDctSubmissionEngine::new(backend.clone()).unwrap();
    let oracle = JpegCoefficientOracle::compile();
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
            "JPEG coefficients: {}, bounded asynchronous: {asynchronous}",
            case.name
        );
        let reference = oracle.read(case.jpeg);
        let mut session = engine
            .open_jpeg_coefficients(case.input, Default::default())
            .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
        let bytes = session.layout().storage_bytes();
        assert_eq!(session.layout().extent(), reference.extent);
        assert_eq!(session.layout().planes().len(), reference.components.len());
        for (component, metadata) in reference
            .components
            .iter()
            .zip(session.metadata().components())
        {
            assert_eq!(component.id, u32::from(metadata.id));
        }
        let mut pending = session.submit_next().unwrap().unwrap();
        let frame = if asynchronous {
            assert_eq!(
                session.memory_stats().resolved_stream_window_limit_bytes,
                40
            );
            pollster::block_on(std::future::poll_fn(|context| {
                std::pin::Pin::new(&mut pending).poll_complete(context)
            }))
        } else {
            pending.wait()
        }
        .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
        assert!(session.submit_next().unwrap().is_none());
        assert_eq!(engine.in_flight_memory_stats().reserved_bytes, bytes);
        let actual = read(&backend, frame.output.buffer());
        for (index, (plane, reference)) in frame
            .output
            .layout()
            .planes()
            .iter()
            .zip(&reference.components)
            .enumerate()
        {
            assert_eq!([plane.blocks_per_row, plane.block_rows], reference.blocks);
            assert_eq!(plane.real_blocks, reference.real_blocks);
            assert_eq!(plane.sampling, reference.sampling);
            let q = plane.quantization_word_offset as usize;
            for (k, &expected) in reference.quantization.iter().enumerate() {
                assert_eq!(
                    actual[q + k],
                    i32::from(expected),
                    "{} component {index} quantizer {k}",
                    case.name
                );
            }
            let base = plane.coefficient_word_offset as usize;
            for (k, &expected) in reference.coefficients.iter().enumerate() {
                assert_eq!(
                    actual[base + k],
                    i32::from(expected),
                    "{} component {index} coefficient {k}",
                    case.name
                );
            }
        }
        let retained = frame.output.clone();
        drop(frame);
        drop(session);
        assert_eq!(engine.in_flight_memory_stats().reserved_bytes, bytes);
        drop(retained);
        assert_eq!(engine.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn metadata_profile_and_coefficient_limits_reject_before_gpu_admission() {
    use jxl_wgpu_decode::{Error, JpegCoefficientError, JpegCoefficientLimits, VarDctDecodeError};
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("actual GPU adapter is required");
    let engine = VarDctSubmissionEngine::new(backend).unwrap();
    for case in CASES {
        let exact = engine
            .open_jpeg_coefficients(case.input, Default::default())
            .unwrap();
        let required = u64::from(exact.layout().coefficient_words());
        drop(exact);
        assert!(
            engine
                .open_jpeg_coefficients(
                    case.input,
                    JpegCoefficientLimits {
                        max_coefficient_words: required,
                        ..Default::default()
                    }
                )
                .is_ok()
        );
        let error = engine
            .open_jpeg_coefficients(
                case.input,
                JpegCoefficientLimits {
                    max_coefficient_words: required - 1,
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(
            matches!(
                error,
                Error::VarDct(VarDctDecodeError::Jpeg(
                    JpegCoefficientError::CoefficientLimit { .. }
                ))
            ),
            "{}: {error:?}",
            case.name
        );
        assert_eq!(engine.in_flight_memory_stats().reserved_bytes, 0);
    }
    let gray = CASES
        .iter()
        .find(|case| case.name == "gray_restart")
        .unwrap();
    let rgb = CASES
        .iter()
        .find(|case| case.name == "rgb_sequential")
        .unwrap();
    let parsed_gray = jxl_gpu_bitstream::parse(gray.input, Default::default()).unwrap();
    let parsed_rgb = jxl_gpu_bitstream::parse(rgb.input, Default::default()).unwrap();
    let jbrd = parsed_gray
        .auxiliary_boxes()
        .iter()
        .find(|item| item.box_type == *b"jbrd")
        .unwrap()
        .payload;
    let box_ = jxl_gpu_bitstream::ContainerBox {
        box_type: *b"jbrd",
        payload: jbrd,
    };
    let mismatch =
        jxl_gpu_bitstream::write_container_with_boxes(parsed_rgb.codestream(), &[box_]).unwrap();
    assert!(matches!(
        engine.open_jpeg_coefficients(&mismatch, Default::default()),
        Err(Error::VarDct(VarDctDecodeError::Jpeg(
            JpegCoefficientError::Binding { .. }
        )))
    ));
    assert!(matches!(
        engine.open_jpeg_coefficients(parsed_gray.codestream(), Default::default()),
        Err(Error::VarDct(VarDctDecodeError::Jpeg(
            JpegCoefficientError::MissingMetadata
        )))
    ));
    let unsupported = jxl_gpu_bitstream::write_container_with_boxes(
        jxl_test_support::corpus::green_queen_vardct_nonzero_ac(),
        &[box_],
    )
    .unwrap();
    assert!(
        engine
            .open_jpeg_coefficients(&unsupported, Default::default())
            .is_err()
    );
    let duplicate =
        jxl_gpu_bitstream::write_container_with_boxes(parsed_gray.codestream(), &[box_, box_])
            .unwrap();
    assert!(matches!(
        engine.open_jpeg_coefficients(&duplicate, Default::default()),
        Err(Error::VarDct(VarDctDecodeError::Jpeg(
            JpegCoefficientError::Metadata(
                jxl_gpu_bitstream::jpeg_reconstruction::JpegReconstructionError::DuplicateBox
            )
        )))
    ));
    for end in [0, 2, gray.input.len() / 2, gray.input.len() - 1] {
        assert!(
            engine
                .open_jpeg_coefficients(&gray.input[..end], Default::default())
                .is_err()
        );
    }
    assert_eq!(engine.in_flight_memory_stats().reserved_bytes, 0);
}
