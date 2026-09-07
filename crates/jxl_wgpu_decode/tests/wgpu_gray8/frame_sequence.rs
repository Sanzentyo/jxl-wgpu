use super::*;
use jxl_gpu_bitstream::{ContainerStreamScanner, InventoryLimits, parse};
use jxl_wgpu_decode::{DecodeProfile, FrameExecutionPlan, FramePlanError, WgpuDecodeEngine};
use std::sync::atomic::{AtomicU64, Ordering};

struct Case {
    name: &'static str,
    hex: &'static str,
    format: LosslessModularFormat,
    bits: u8,
    vardct: bool,
}

fn cases() -> [Case; 9] {
    [
        Case {
            name: "modular_gray",
            hex: include_str!("../../test-data/sequence_modular_gray.jxl.hex"),
            format: LosslessModularFormat::Gray,
            bits: 8,
            vardct: false,
        },
        Case {
            name: "modular_rgb12",
            hex: include_str!("../../test-data/sequence_modular_rgb12.jxl.hex"),
            format: LosslessModularFormat::Rgb,
            bits: 12,
            vardct: false,
        },
        Case {
            name: "modular_rgba16",
            hex: include_str!("../../test-data/sequence_modular_rgba16.jxl.hex"),
            format: LosslessModularFormat::Rgba,
            bits: 16,
            vardct: false,
        },
        Case {
            name: "modular_many",
            hex: include_str!("../../test-data/sequence_modular_many.jxl.hex"),
            format: LosslessModularFormat::Gray,
            bits: 8,
            vardct: false,
        },
        Case {
            name: "layered_still",
            hex: include_str!("../../test-data/sequence_layered_still.jxl.hex"),
            format: LosslessModularFormat::Gray,
            bits: 8,
            vardct: false,
        },
        Case {
            name: "vardct_rgb",
            hex: include_str!("../../test-data/sequence_vardct_rgb.jxl.hex"),
            format: LosslessModularFormat::Rgb,
            bits: 8,
            vardct: true,
        },
        Case {
            name: "vardct_gray",
            hex: include_str!("../../test-data/sequence_vardct_gray.jxl.hex"),
            format: LosslessModularFormat::Rgb,
            bits: 8,
            vardct: true,
        },
        Case {
            name: "mixed_jpeg_modular",
            hex: include_str!("../../test-data/sequence_mixed_jpeg_modular.jxl.hex"),
            format: LosslessModularFormat::Rgb,
            bits: 8,
            vardct: true,
        },
        Case {
            name: "vardct_dc",
            hex: include_str!("../../test-data/sequence_vardct_dc.jxl.hex"),
            format: LosslessModularFormat::Rgb,
            bits: 8,
            vardct: true,
        },
    ]
}

fn encoded(case: &Case) -> Vec<u8> {
    let digits = case.hex.split_whitespace().collect::<String>();
    digits
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn request(case: &Case) -> GpuOutputRequest {
    if case.vardct {
        return GpuOutputRequest::color(jxl_wgpu_decode::vardct_rgb8_format())
            .unwrap()
            .with_max_frame_slots(NonZeroUsize::new(3).unwrap());
    }
    let format = case.format.pixel_format(case.bits).unwrap();
    let request = if case.format == LosslessModularFormat::Gray {
        GpuOutputRequest::numeric(format, NumericSampleMapping::NativeUnsigned).unwrap()
    } else {
        GpuOutputRequest::color(format).unwrap()
    };
    request.with_max_frame_slots(NonZeroUsize::new(3).unwrap())
}

fn samples(bytes: &[u8], bits: u8) -> Vec<u16> {
    if bits <= 8 {
        bytes.iter().copied().map(u16::from).collect()
    } else {
        bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect()
    }
}

fn incremental(
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    encoded: &[u8],
    request: GpuOutputRequest,
) -> jxl_wgpu_decode::GpuDecodeSession<jxl_wgpu_decode::WgpuDecodeSubmissionSession> {
    let mut input = decoder.stream(request).unwrap();
    let mut transport = ContainerStreamScanner::new(decoder.container_stream_limits());
    for chunk in encoded.chunks(137) {
        for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
            input.push_transport_event(&event).unwrap();
        }
    }
    for event in transport.finish_input().unwrap() {
        input.push_transport_event(&event).unwrap();
    }
    input.finish().unwrap()
}

fn rust_frames(case: &Case, encoded: &[u8]) -> Vec<(Option<f64>, Vec<u16>)> {
    let mut input = encoded;
    let decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let ProcessingResult::Complete {
        result: mut decoder,
    } = decoder.process(&mut input, None).unwrap()
    else {
        panic!("complete fixture header");
    };
    let size = decoder.basic_info().size;
    let channels = case.format.channel_count() as usize;
    let row_bytes = size.0 * channels * if case.bits <= 8 { 1 } else { 2 };
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: match case.format {
            LosslessModularFormat::Gray => JxlColorType::Grayscale,
            LosslessModularFormat::Rgb => JxlColorType::Rgb,
            LosslessModularFormat::Rgba => JxlColorType::Rgba,
        },
        color_data_format: Some(if case.bits <= 8 {
            JxlDataFormat::U8 {
                bit_depth: case.bits,
            }
        } else {
            JxlDataFormat::U16 {
                endianness: Endianness::LittleEndian,
                bit_depth: case.bits,
            }
        }),
        extra_channel_format: vec![None; usize::from(case.format.has_alpha())],
    });
    let mut frames = Vec::new();
    loop {
        let ProcessingResult::Complete { result: frame } =
            decoder.process(&mut input, None).unwrap()
        else {
            panic!("complete fixture frame");
        };
        let duration = frame.frame_header().duration;
        let mut bytes = vec![0; size.1 * row_bytes];
        let mut outputs = [JxlOutputBuffer::new(&mut bytes, size.1, row_bytes)];
        let ProcessingResult::Complete { result } =
            frame.process(&mut input, &mut outputs, None).unwrap()
        else {
            panic!("complete fixture pixels");
        };
        decoder = result;
        frames.push((duration, samples(&bytes, case.bits)));
        if !decoder.has_more_frames() {
            break;
        }
    }
    frames
}

fn djxl_frames(case: &Case, encoded: &[u8]) -> Option<Vec<Vec<u16>>> {
    if Command::new("djxl").arg("--version").output().is_err() {
        return None;
    }
    static DIRECTORY: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "jxl-sequence-oracle-{}-{}",
        std::process::id(),
        DIRECTORY.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("input.jxl");
    std::fs::write(&path, encoded).unwrap();
    let output = Command::new("djxl")
        .arg(path)
        .arg(dir.join("frame.pam"))
        .args([
            "--quiet",
            "--output_frames",
            &format!("--bits_per_sample={}", case.bits),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut files = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "pam"))
        .collect::<Vec<_>>();
    files.sort();
    let frames = files
        .iter()
        .map(|path| {
            let data = std::fs::read(path).unwrap();
            let end = data
                .windows(7)
                .position(|bytes| bytes == b"ENDHDR\n")
                .unwrap()
                + 7;
            let header = std::str::from_utf8(&data[..end]).unwrap();
            let field = |key: &str| {
                header
                    .lines()
                    .find_map(|line| line.strip_prefix(key))
                    .unwrap()
                    .trim()
                    .parse::<u32>()
                    .unwrap()
            };
            let max = field("MAXVAL ");
            let depth = field("DEPTH ") as usize;
            let raw = if max <= 255 {
                data[end..]
                    .iter()
                    .copied()
                    .map(u16::from)
                    .collect::<Vec<_>>()
            } else {
                data[end..]
                    .chunks_exact(2)
                    .map(|b| u16::from_be_bytes([b[0], b[1]]))
                    .collect()
            };
            raw.chunks_exact(depth)
                .flat_map(|pixel| {
                    let color = if depth == 1 && case.format == LosslessModularFormat::Rgb {
                        vec![pixel[0]; 3]
                    } else {
                        pixel.to_vec()
                    };
                    color.into_iter().map(move |sample| {
                        ((u32::from(sample) * ((1 << case.bits) - 1) + max / 2) / max) as u16
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect();
    std::fs::remove_dir_all(dir).unwrap();
    Some(frames)
}

#[test]
fn modular_and_vardct_sequences_preserve_pixels_timing_and_bounded_input_lifetimes() {
    let Some(backend) = backend() else {
        return;
    };
    for case in cases() {
        eprintln!("sequence {}", case.name);
        let encoded = encoded(&case);
        let inventory = parse(&encoded, Default::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
        assert!(
            plan.nodes.len() > plan.presentations.len(),
            "{} has coalesced layers",
            case.name
        );
        assert_eq!(
            plan.metadata.loop_count,
            if case.name == "layered_still" {
                None
            } else {
                Some(if case.name == "modular_many" { 0 } else { 3 })
            }
        );
        if let Some(clock) = plan.metadata.timebase {
            assert_eq!(
                (
                    clock.ticks_per_second_numerator.get(),
                    clock.ticks_per_second_denominator.get()
                ),
                (30000, 1001)
            );
        } else {
            assert_eq!(plan.presentations.len(), 1);
        }
        if case.name == "vardct_dc" {
            assert!(inventory.frames.iter().any(|frame| frame.lf_level == 2));
        }
        let oracle = rust_frames(&case, &encoded);
        let djxl = djxl_frames(&case, &encoded);
        assert_eq!(oracle.len(), plan.presentations.len());
        let mut whole = Vec::new();
        for bounded in [false, true] {
            let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            let engine = if bounded {
                engine.with_stream_window_limit(NonZeroU64::new(4096).unwrap())
            } else {
                engine
            };
            let decoder = GpuDecoder::new(engine);
            let mut session = if bounded {
                incremental(&decoder, &encoded, request(&case))
            } else {
                decoder.open(&encoded, request(&case)).unwrap()
            };
            assert_eq!(
                session.profile(),
                DecodeProfile::FrameSequence {
                    physical_frames: plan.nodes.len(),
                    presentation_frames: plan.presentations.len()
                }
            );
            assert_eq!(session.metadata(), &plan.metadata);
            for (index, presentation) in plan.presentations.iter().enumerate() {
                let frame = if bounded {
                    pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap()
                } else {
                    session.next_frame().unwrap().unwrap()
                };
                assert_eq!(frame.metadata, presentation.metadata);
                assert!(frame.metadata.name.starts_with("frame-"));
                assert!(frame.metadata.name.ends_with("時間"));
                if let Some(milliseconds) = oracle[index].0 {
                    assert!(
                        (frame.metadata.duration.as_seconds() * 1000.0 - milliseconds).abs() < 1e-9
                    );
                } else {
                    assert!(frame.metadata.duration.timebase.is_none());
                }
                let source_index = frame
                    .metadata
                    .name
                    .split('-')
                    .nth(1)
                    .unwrap()
                    .parse::<u32>()
                    .unwrap();
                assert_eq!(
                    frame.metadata.timecode,
                    plan.metadata
                        .is_animation()
                        .then_some(0x01020000 + source_index)
                );
                let output = &frame.output().outputs[0];
                assert_eq!(output.layout.extent, plan.metadata.extent);
                let pixels = samples(&read_output(&backend, output), case.bits);
                assert_eq!(pixels.len(), oracle[index].1.len());
                let error = pixels
                    .iter()
                    .zip(&oracle[index].1)
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap();
                assert!(
                    error <= u16::from(case.vardct),
                    "{} frame {index}: Rust error {error}",
                    case.name
                );
                if let Some(djxl) = &djxl {
                    assert_eq!(djxl.len(), oracle.len());
                    assert_eq!(pixels.len(), djxl[index].len());
                    let error = pixels
                        .iter()
                        .zip(&djxl[index])
                        .map(|(a, b)| a.abs_diff(*b))
                        .max()
                        .unwrap();
                    assert!(
                        error <= u16::from(case.vardct),
                        "{} frame {index}: djxl error {error}",
                        case.name
                    );
                }
                if bounded {
                    assert_eq!(
                        pixels, whole[index],
                        "{} frame {index} window equality",
                        case.name
                    );
                } else {
                    whole.push(pixels);
                }
            }
            assert!(session.next_frame().unwrap().is_none());
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
            drop(session);
            assert_eq!(
                decoder.engine().in_flight_memory_stats().reserved_bytes,
                0,
                "{}",
                case.name
            );
        }
    }
}

#[test]
fn sequence_admission_retries_and_cancellation_preserve_byte_and_frame_ownership() {
    let Some(backend) = backend() else {
        return;
    };
    for case in cases()
        .into_iter()
        .filter(|case| matches!(case.name, "modular_many" | "vardct_dc"))
    {
        let encoded = encoded(&case);
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(4096).unwrap()),
        );
        let mut session = incremental(&decoder, &encoded, request(&case));
        let retained = decoder.incremental_input_budget().snapshot().reserved_bytes;
        assert!(retained > 0);
        let budget = backend.transient_memory_budget();
        let blocker = budget
            .try_reserve(budget.snapshot().available_bytes)
            .unwrap();
        for _ in 0..2 {
            let pressure = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
            assert_eq!(pressure.submitted, 0);
            assert_eq!(pressure.queued, 0);
            assert!(matches!(
                pressure.backpressure,
                Some(PrefetchBackpressure::Memory(_))
            ));
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                retained
            );
        }
        drop(blocker);
        let target = if case.vardct { 1 } else { 3 };
        let progress = session
            .prefetch(NonZeroUsize::new(target).unwrap())
            .unwrap();
        assert_eq!(progress.queued, target);
        let frame = pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap();
        assert_eq!(frame.metadata.index, 0);
        assert_eq!(frame.metadata.duration.ticks, 4);
        let lease = frame.output().outputs[0].buffer.clone();
        drop(frame);
        drop(session);
        backend
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
        assert_eq!(
            decoder.engine().in_flight_memory_stats().reserved_bytes,
            lease.size()
        );
        drop(lease);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

        // Cancel before consuming the first presentation, including a staged LF cursor map.
        let mut cancelled = incremental(&decoder, &encoded, request(&case));
        cancelled.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        drop(cancelled);
        backend
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while decoder.engine().in_flight_memory_stats().reserved_bytes != 0
            && std::time::Instant::now() < deadline
        {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn frame_sequence_rejects_valid_crop_and_add_before_any_gpu_admission() {
    let Some(backend) = backend() else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend).unwrap();
    for hex in [
        include_str!("../../test-data/sequence_rejected_crop.jxl.hex"),
        include_str!("../../test-data/sequence_rejected_add.jxl.hex"),
    ] {
        let case = Case {
            name: "composition",
            hex,
            format: LosslessModularFormat::Gray,
            bits: 8,
            vardct: false,
        };
        let bytes = encoded(&case);
        assert_eq!(rust_frames(&case, &bytes).len(), 2);
        let result = decoder.open(&bytes, request(&case));
        assert!(matches!(
            result,
            Err(jxl_wgpu_decode::Error::FramePlan(
                FramePlanError::CompositionRequired { frame_index: 1 }
            ))
        ));
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn frame_plan_versions_references_and_rejects_malformed_timing_and_composition() {
    let case = &cases()[0];
    let encoded = encoded(case);
    let mut inventory = parse(&encoded, Default::default())
        .unwrap()
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
    assert_eq!(plan.nodes[0].save_reference, Some(0));
    assert_eq!(plan.nodes[1].references[0].unwrap().frame_index, 0);
    assert_eq!(plan.nodes[3].references[0].unwrap().frame_index, 2);
    assert_eq!(
        plan.presentations
            .iter()
            .map(|frame| frame.metadata.presentation_ticks)
            .collect::<Vec<_>>(),
        vec![0, 4, 14]
    );
    inventory.frames[1].x0 = -1;
    inventory.frames[1].have_crop = true;
    let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
    assert!(plan.nodes[1].needs_composition);
    inventory.frames[0].timecode = None;
    assert!(matches!(
        FrameExecutionPlan::negotiate(&inventory),
        Err(FramePlanError::InvalidFrame { frame_index: 0, .. })
    ));
}
