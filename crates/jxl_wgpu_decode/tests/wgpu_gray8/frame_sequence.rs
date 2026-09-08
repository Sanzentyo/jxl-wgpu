use super::*;
use jxl_gpu_bitstream::{ContainerStreamScanner, InventoryLimits, parse};
use jxl_wgpu_decode::{DecodeProfile, FrameExecutionPlan, FramePlanError, WgpuDecodeEngine};
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "frame_sequence/composition.rs"]
mod composition;
#[path = "frame_sequence/independent.rs"]
mod independent;
#[path = "frame_sequence/reference.rs"]
mod reference;

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

fn composition_cases() -> [Case; 11] {
    use LosslessModularFormat::{Gray, Rgb, Rgba};
    [
        (
            "gray_alpha",
            include_str!("../../test-data/composition_gray_alpha.jxl.hex"),
            Rgba,
            16,
            false,
        ),
        (
            "rgba_mixed_depth",
            include_str!("../../test-data/composition_rgba_mixed_depth.jxl.hex"),
            Rgba,
            12,
            false,
        ),
        (
            "gray",
            include_str!("../../test-data/composition_gray.jxl.hex"),
            Gray,
            8,
            false,
        ),
        (
            "rgb12",
            include_str!("../../test-data/composition_rgb12.jxl.hex"),
            Rgb,
            12,
            false,
        ),
        (
            "rgba8",
            include_str!("../../test-data/composition_rgba8.jxl.hex"),
            Rgba,
            8,
            false,
        ),
        (
            "rgba16",
            include_str!("../../test-data/composition_rgba16.jxl.hex"),
            Rgba,
            16,
            false,
        ),
        (
            "still",
            include_str!("../../test-data/composition_still.jxl.hex"),
            Gray,
            8,
            false,
        ),
        (
            "vardct",
            include_str!("../../test-data/composition_vardct.jxl.hex"),
            Rgb,
            8,
            true,
        ),
        (
            "vardct_gray",
            include_str!("../../test-data/composition_vardct_gray.jxl.hex"),
            Rgb,
            8,
            true,
        ),
        (
            "vardct_dc",
            include_str!("../../test-data/composition_vardct_dc.jxl.hex"),
            Rgb,
            8,
            true,
        ),
        (
            "mixed",
            include_str!("../../test-data/composition_mixed.jxl.hex"),
            Rgb,
            8,
            true,
        ),
    ]
    .map(|(name, hex, format, bits, vardct)| Case {
        name,
        hex,
        format,
        bits,
        vardct,
    })
}

#[test]
fn composed_sequences_validate_every_layer_and_match_two_decoders() {
    let Some(backend) = backend() else {
        return;
    };
    for case in composition_cases() {
        eprintln!("composition {}", case.name);
        let bytes = encoded(&case);
        let inventory = parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
        assert!(plan.nodes.iter().any(|node| node.needs_composition));
        let floats = rust_float_frames(&bytes, case.format);
        let maximum = ((1u32 << case.bits) - 1) as f32;
        let expected = floats
            .iter()
            .map(|frame| {
                (
                    None::<f64>,
                    frame
                        .iter()
                        .map(|value| (value.clamp(0.0, 1.0) * maximum).round() as u16)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let djxl = djxl_frames(&case, &bytes);
        let mut whole = Vec::new();
        for bounded in [false, true] {
            let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            let decoder = GpuDecoder::new(if bounded {
                engine.with_stream_window_limit(NonZeroU64::new(4096).unwrap())
            } else {
                engine
            });
            let mut session = if bounded {
                incremental(&decoder, &bytes, request(&case))
            } else {
                decoder.open(&bytes, request(&case)).unwrap()
            };
            for (index, (_, oracle)) in expected.iter().enumerate() {
                let frame = if bounded {
                    pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap()
                } else {
                    session.next_frame().unwrap().unwrap()
                };
                assert_eq!(frame.metadata, plan.presentations[index].metadata);
                let output = &frame.output().outputs[0];
                assert_eq!(output.layout.extent, plan.metadata.extent);
                let pixels = samples(&read_output(&backend, output), case.bits);
                let error = pixels
                    .iter()
                    .zip(oracle)
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap();
                assert_eq!(pixels.len(), oracle.len());
                assert!(
                    error <= 1,
                    "{} presentation {index}: Rust error {error}",
                    case.name
                );
                if let Some(djxl) = &djxl {
                    assert_eq!(pixels.len(), djxl[index].len());
                    let error = pixels
                        .iter()
                        .zip(&djxl[index])
                        .map(|(a, b)| a.abs_diff(*b))
                        .max()
                        .unwrap();
                    assert!(
                        error <= 1,
                        "{} presentation {index}: djxl error {error}",
                        case.name
                    );
                }
                if bounded {
                    assert_eq!(pixels, whole[index]);
                } else {
                    whole.push(pixels);
                }
            }
            assert!(session.next_frame().unwrap().is_none());
            drop(session);
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

pub(super) fn incremental(
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

fn rust_float_frames(encoded: &[u8], format: LosslessModularFormat) -> Vec<Vec<f32>> {
    let mut input = encoded;
    let decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let ProcessingResult::Complete {
        result: mut decoder,
    } = decoder.process(&mut input, None).unwrap()
    else {
        panic!("complete fixture header");
    };
    let size = decoder.basic_info().size;
    let channels = format.channel_count() as usize;
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: match format {
            LosslessModularFormat::Gray => JxlColorType::Grayscale,
            LosslessModularFormat::Rgb => JxlColorType::Rgb,
            LosslessModularFormat::Rgba => JxlColorType::Rgba,
        },
        color_data_format: Some(JxlDataFormat::F32 {
            endianness: Endianness::LittleEndian,
        }),
        extra_channel_format: vec![None; usize::from(format.has_alpha())],
    });
    let mut frames = Vec::new();
    loop {
        let ProcessingResult::Complete { result: frame } =
            decoder.process(&mut input, None).unwrap()
        else {
            panic!("complete fixture frame");
        };
        let mut bytes = vec![0; size.0 * size.1 * channels * 4];
        let mut outputs = [JxlOutputBuffer::new(
            &mut bytes,
            size.1,
            size.0 * channels * 4,
        )];
        let ProcessingResult::Complete { result } =
            frame.process(&mut input, &mut outputs, None).unwrap()
        else {
            panic!("complete fixture pixels");
        };
        decoder = result;
        frames.push(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        );
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
                    } else if depth == 2 && case.format == LosslessModularFormat::Rgba {
                        vec![pixel[0], pixel[0], pixel[0], pixel[1]]
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

pub(super) fn keep_codestream_order<T: Clone>(
    values: &[T],
    extent: Extent2d,
    channels: usize,
    orientation: u32,
) -> Vec<T> {
    let mut rows: Vec<Vec<Vec<T>>> = values
        .chunks_exact(extent.width as usize * channels)
        .map(|row| row.chunks_exact(channels).map(<[T]>::to_vec).collect())
        .collect();
    if matches!(orientation, 2 | 3 | 6 | 7) {
        for row in &mut rows {
            row.reverse();
        }
    }
    if matches!(orientation, 3 | 4 | 7 | 8) {
        rows.reverse();
    }
    if orientation >= 5 {
        rows = (0..extent.width as usize)
            .map(|x| rows.iter().map(|row| row[x].clone()).collect())
            .collect();
    }
    rows.into_iter().flatten().flatten().collect()
}

#[test]
fn floating_frame_sequences_can_keep_codestream_coordinates() {
    use jxl_gpu_formats::RgbChannelOrder;
    use jxl_wgpu_decode::OrientationPolicy;
    let Some(backend) = backend() else {
        return;
    };
    for case in cases() {
        let encoded = encoded(&case);
        let inventory = parse(&encoded, Default::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let plan = FrameExecutionPlan::negotiate(&inventory).unwrap();
        let kept =
            FrameExecutionPlan::negotiate_with_orientation(&inventory, OrientationPolicy::Keep)
                .unwrap();
        let extent = Extent2d::new(inventory.image_header.width, inventory.image_header.height);
        assert_eq!(kept.metadata.extent, extent);
        assert_eq!(kept.nodes, plan.nodes);
        assert_eq!(kept.presentations, plan.presentations);
        let oracle = rust_frames(&case, &encoded);
        let color = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
        let request =
            GpuOutputRequest::color(PixelFormat::rgb_f32(RgbChannelOrder::Rgba, false, color))
                .unwrap()
                .with_orientation_policy(OrientationPolicy::Keep);
        assert_eq!(request.orientation_policy(), OrientationPolicy::Keep);
        let mut whole = Vec::new();
        for bounded in [false, true] {
            let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            let decoder = GpuDecoder::new(if bounded {
                engine.with_stream_window_limit(NonZeroU64::new(4096).unwrap())
            } else {
                engine
            });
            let mut session = if bounded {
                incremental(&decoder, &encoded, request.clone())
            } else {
                decoder.open(&encoded, request.clone()).unwrap()
            };
            assert_eq!(session.metadata(), &kept.metadata);
            let source_channels = case.format.channel_count() as usize;
            let maximum = ((1u32 << case.bits) - 1) as f32;
            for (index, (_, expected)) in oracle.iter().enumerate() {
                let frame = if bounded {
                    pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap()
                } else {
                    session.next_frame().unwrap().unwrap()
                };
                assert_eq!(frame.metadata, kept.presentations[index].metadata);
                let output = &frame.output().outputs[0];
                assert_eq!(output.layout.extent, extent);
                let bytes = read_output(&backend, output);
                let expected = keep_codestream_order(
                    expected,
                    plan.metadata.extent,
                    source_channels,
                    inventory.image_header.orientation,
                );
                for (actual, expected) in bytes
                    .chunks_exact(16)
                    .zip(expected.chunks_exact(source_channels))
                {
                    for channel in 0..4 {
                        let value = f32::from_le_bytes(
                            actual[channel * 4..channel * 4 + 4].try_into().unwrap(),
                        );
                        assert!(value.is_finite());
                        let expected = if channel == 3 && source_channels != 4 {
                            maximum as u16
                        } else {
                            expected[if source_channels == 1 { 0 } else { channel }]
                        };
                        let code = (value * maximum).round().clamp(0.0, maximum) as u16;
                        assert!(
                            code.abs_diff(expected) <= u16::from(case.vardct),
                            "{} frame {index} channel {channel}: {value} -> {code} != {expected}",
                            case.name
                        );
                    }
                }
                if bounded {
                    assert_eq!(bytes, whole[index]);
                } else {
                    whole.push(bytes);
                }
            }
            assert!(session.next_frame().unwrap().is_none());
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn sequence_admission_retries_and_cancellation_preserve_byte_and_frame_ownership() {
    fn wait_for_callbacks(backend: &WgpuBackend) {
        backend
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        // Device::poll can return while another polling thread runs callbacks it already
        // extracted. Abandoned prefetched frames retire in those callbacks; the native poll
        // permit is released only after they return. Observe that boundary before counting leases.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while backend.submission_poller().in_flight() != 0 && std::time::Instant::now() < deadline {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(backend.submission_poller().in_flight(), 0);
    }
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
        wait_for_callbacks(&backend);
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
        wait_for_callbacks(&backend);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn frame_sequence_composes_crop_and_add_against_both_decoders() {
    let Some(backend) = backend() else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
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
        let oracle = rust_frames(&case, &bytes);
        let djxl = djxl_frames(&case, &bytes);
        assert_eq!(oracle.len(), 2);
        let mut session = decoder.open(&bytes, request(&case)).unwrap();
        let progress = session.prefetch(NonZeroUsize::new(3).unwrap()).unwrap();
        assert_eq!(progress.queued, 1);
        assert_eq!(
            progress.backpressure,
            Some(jxl_wgpu_decode::PrefetchBackpressure::FrameDependency { index: 0 })
        );
        for (index, (_, expected)) in oracle.iter().enumerate() {
            let frame = pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap();
            let pixels = samples(
                &read_output(&backend, &frame.output().outputs[0]),
                case.bits,
            );
            assert_eq!(&pixels, expected);
            if let Some(djxl) = &djxl {
                assert_eq!(pixels, djxl[index]);
            }
        }
        assert!(session.next_frame().unwrap().is_none());
        drop(session);
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
