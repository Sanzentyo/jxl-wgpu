use super::*;
use jxl_gpu_bitstream::{ContainerStreamScanner, InventoryLimits, ParseLimits};
use jxl_gpu_protocol::OutputOrientation;
use jxl_wgpu_decode::{GpuDecodeSession, WgpuDecodeSession};
use std::sync::atomic::{AtomicU64, Ordering};

struct Case {
    name: &'static str,
    hex: &'static str,
    format: LosslessModularFormat,
    bits: u8,
    width: u32,
    height: u32,
    orientation: u32,
}

macro_rules! fixture {
    ($name:literal, $format:ident, $bits:literal, $width:literal, $height:literal, $orientation:literal) => {
        Case {
            name: $name,
            hex: include_str!(concat!(
                "../../test-data/testsrc_modular_orientation_",
                $name,
                ".jxl.hex"
            )),
            format: LosslessModularFormat::$format,
            bits: $bits,
            width: $width,
            height: $height,
            orientation: $orientation,
        }
    };
}

const CASES: &[Case] = &[
    fixture!("gray_1", Gray, 8, 259, 257, 1),
    fixture!("gray_2", Gray, 8, 259, 257, 2),
    fixture!("gray_3", Gray, 8, 259, 257, 3),
    fixture!("gray_4", Gray, 8, 259, 257, 4),
    fixture!("gray_5", Gray, 8, 259, 257, 5),
    fixture!("gray_6", Gray, 8, 259, 257, 6),
    fixture!("gray_7", Gray, 8, 259, 257, 7),
    fixture!("gray_8", Gray, 8, 259, 257, 8),
    fixture!("rgb_1", Rgb, 8, 259, 17, 1),
    fixture!("rgb_2", Rgb, 8, 259, 17, 2),
    fixture!("rgb_3", Rgb, 8, 259, 17, 3),
    fixture!("rgb_4", Rgb, 8, 259, 17, 4),
    fixture!("rgb_5", Rgb, 8, 259, 17, 5),
    fixture!("rgb_6", Rgb, 8, 259, 17, 6),
    fixture!("rgb_7", Rgb, 8, 259, 17, 7),
    fixture!("rgb_8", Rgb, 8, 259, 17, 8),
    fixture!("gray_palette", Gray, 8, 515, 259, 6),
    fixture!("gray_squeeze", Gray, 8, 2051, 259, 8),
    fixture!("gray_single_palette", Gray, 8, 37, 23, 7),
    fixture!("rgba_16", Rgba, 16, 259, 7, 5),
    fixture!("rgb_12_column", Rgb, 12, 1, 257, 6),
    fixture!("gray_row", Gray, 8, 257, 1, 8),
    fixture!("rgba_8", Rgba, 8, 17, 9, 2),
];

impl Case {
    fn encoded(&self) -> Vec<u8> {
        let digits: Vec<_> = self
            .hex
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect();
        digits
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    fn extent(&self) -> Extent2d {
        OutputOrientation::from_exif_value(self.orientation)
            .unwrap()
            .map_extent(Extent2d::new(self.width, self.height))
    }

    fn expected(&self) -> Vec<u16> {
        let maximum = (1u32 << self.bits) - 1;
        let mut rows = Vec::new();
        for y in 0..self.height {
            let mut row = Vec::new();
            for x in 0..self.width {
                let red = if self.name.contains("palette") {
                    [2, 29, 113, 241][((x / 11 + y / 7 + (x * y) % 5) % 4) as usize]
                } else {
                    (613 * x + 107 * y + 43 * (x ^ y)) & maximum
                };
                row.push([
                    red,
                    ((153 * x) ^ (271 * y)) & maximum,
                    (259 * x + 307 * y + 31 * (x ^ y)) & maximum,
                    maximum - ((181 * x + 97 * y) & (maximum - 1)),
                ]);
            }
            rows.push(row);
        }
        // Orient by transposing row/column collections and reversing traversal, independently
        // of the shader's coordinate formulas and group origins.
        if self.orientation >= 5 {
            rows = (0..self.width as usize)
                .map(|x| rows.iter().map(|row| row[x]).collect())
                .collect();
        }
        if matches!(self.orientation, 3 | 4 | 7 | 8) {
            rows.reverse();
        }
        if matches!(self.orientation, 2 | 3 | 6 | 7) {
            for row in &mut rows {
                row.reverse();
            }
        }
        rows.into_iter()
            .flatten()
            .flat_map(|pixel| {
                pixel
                    .into_iter()
                    .take(self.format.channel_count() as usize)
                    .map(|sample| sample as u16)
            })
            .collect()
    }

    fn native_request(&self) -> GpuOutputRequest {
        let format = self.format.pixel_format(self.bits).unwrap();
        if self.format == LosslessModularFormat::Gray {
            GpuOutputRequest::numeric(format, NumericSampleMapping::NativeUnsigned).unwrap()
        } else {
            GpuOutputRequest::color(format).unwrap()
        }
    }
}

fn incremental(
    decoder: &GpuDecoder<WgpuSubmissionEngine>,
    encoded: &[u8],
    request: GpuOutputRequest,
) -> GpuDecodeSession<WgpuDecodeSession> {
    let mut stream = decoder.stream(request).unwrap();
    let mut transport = ContainerStreamScanner::new(decoder.container_stream_limits());
    for chunk in encoded.chunks(137) {
        for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
            stream.push_transport_event(&event).unwrap();
        }
    }
    for event in transport.finish_input().unwrap() {
        stream.push_transport_event(&event).unwrap();
    }
    stream.finish().unwrap()
}

fn decode(
    backend: &WgpuBackend,
    decoder: &GpuDecoder<WgpuSubmissionEngine>,
    encoded: &[u8],
    request: GpuOutputRequest,
    bounded: bool,
) -> (ImageLayout, Vec<u8>) {
    let mut session = if bounded {
        incremental(decoder, encoded, request)
    } else {
        decoder.open(encoded, request).unwrap()
    };
    let stats = session.submission_session().memory_stats();
    if bounded {
        assert!(stats.stream_window_bytes <= 4096);
    }
    let frame = if bounded {
        pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap()
    } else {
        session.next_frame().unwrap().unwrap()
    };
    let output = &frame.output().outputs[0];
    assert_eq!(
        stats.output_lease_bytes,
        output.layout.logical_size.div_ceil(4) * 4
    );
    let result = (output.layout.clone(), read_output(backend, output));
    assert!(session.next_frame().unwrap().is_none());
    drop(frame);
    drop(session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    result
}

fn djxl_samples(case: &Case, encoded: &[u8]) -> Option<Vec<u16>> {
    if Command::new("djxl").arg("--version").output().is_err() {
        return None;
    }
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let prefix = std::env::temp_dir().join(format!(
        "jxl-modular-orientation-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let source = prefix.with_extension("jxl");
    let gray = case.format == LosslessModularFormat::Gray;
    let destination = prefix.with_extension(if gray { "pgm" } else { "ppm" });
    std::fs::write(&source, encoded).unwrap();
    let command = Command::new("djxl")
        .arg(&source)
        .arg(&destination)
        .arg(if gray {
            "--color_space=Gra_D65_Rel_SRG"
        } else {
            "--color_space=RGB_D65_SRG_Rel_SRG"
        })
        .output()
        .unwrap();
    std::fs::remove_file(source).unwrap();
    assert!(
        command.status.success(),
        "djxl {}: {}",
        case.name,
        String::from_utf8_lossy(&command.stderr)
    );
    let bytes = std::fs::read(&destination).unwrap();
    std::fs::remove_file(destination).unwrap();
    let mut cursor = 0;
    let mut token = || {
        loop {
            while bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            if bytes[cursor] != b'#' {
                break;
            }
            while bytes[cursor] != b'\n' {
                cursor += 1;
            }
        }
        let start = cursor;
        while !bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        std::str::from_utf8(&bytes[start..cursor]).unwrap()
    };
    assert_eq!(token(), if gray { "P5" } else { "P6" });
    assert_eq!(token().parse::<u32>().unwrap(), case.extent().width);
    assert_eq!(token().parse::<u32>().unwrap(), case.extent().height);
    assert_eq!(token().parse::<u32>().unwrap(), (1u32 << case.bits) - 1);
    if bytes[cursor] == b'\r' && bytes.get(cursor + 1) == Some(&b'\n') {
        cursor += 1;
    }
    cursor += 1;
    Some(if case.bits <= 8 {
        bytes[cursor..].iter().copied().map(u16::from).collect()
    } else {
        bytes[cursor..]
            .chunks_exact(2)
            .map(|sample| u16::from_be_bytes([sample[0], sample[1]]))
            .collect()
    })
}

fn assert_color_precision(name: &str, bytes: &[u8], expected: &[u8], layout: &ImageLayout) {
    assert_eq!(bytes.len(), expected.len());
    let wide = matches!(
        classify_pixel_format(&layout.format).unwrap(),
        jxl_gpu_formats::PixelFormatClass::Color(jxl_gpu_formats::ColorFormatClass::Luma {
            storage_bits: 16,
            ..
        })
    );
    let maximum = if wide {
        bytes
            .chunks_exact(2)
            .zip(expected.chunks_exact(2))
            .map(|(actual, expected)| {
                u16::from_le_bytes([actual[0], actual[1]])
                    .abs_diff(u16::from_le_bytes([expected[0], expected[1]]))
            })
            .max()
            .unwrap_or(0)
    } else {
        bytes
            .iter()
            .zip(expected)
            .map(|(actual, expected)| u16::from(actual.abs_diff(*expected)))
            .max()
            .unwrap_or(0)
    };
    // CPU color matrices and GPU transfer evaluation can straddle a half-code boundary.
    // Native integer samples and numeric mappings are compared exactly in their own paths.
    assert!(
        maximum <= 1,
        "{name}: color conversion differs by {maximum} codes"
    );
}

#[test]
fn native_modular_orientation_matches_sources_and_both_decoders() {
    let Some(backend) = backend() else {
        return;
    };
    let whole = GpuDecoder::new(WgpuSubmissionEngine::new(backend.clone()));
    let bounded = GpuDecoder::new(
        WgpuSubmissionEngine::new(backend.clone())
            .with_stream_window_limit(NonZeroU64::new(4096).unwrap()),
    );
    let mut saw_fused = false;
    let mut saw_group_inverse = false;
    let mut saw_frame_inverse = false;
    for case in CASES {
        let encoded = case.encoded();
        let inventory = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        assert_eq!(
            inventory.image_header.orientation, case.orientation,
            "{}",
            case.name
        );
        let inspection = whole.open(&encoded, case.native_request()).unwrap();
        let stats = inspection.submission_session().memory_stats();
        saw_fused |= stats.inverse_transform_count == 0;
        saw_group_inverse |=
            stats.inverse_transform_count != 0 && stats.frame_modular_arena_bytes == 0;
        saw_frame_inverse |= stats.frame_modular_arena_bytes != 0;
        if case.name.contains("palette") {
            assert!(stats.palette_dispatch_count != 0);
        }
        if case.name == "gray_squeeze" {
            assert!(stats.progressive_pass_count > 1);
            assert!(stats.low_frequency_group_stream_count > 0);
        }
        drop(inspection);
        let expected = case.expected();
        let (size, rust) = rust_jxl_decode_integer(&encoded, case.format, case.bits).unwrap();
        assert_eq!(
            size,
            (case.extent().width as usize, case.extent().height as usize)
        );
        assert_eq!(rust, expected, "{} Rust oracle", case.name);
        if let Some(djxl) = djxl_samples(case, &encoded) {
            let color = if case.format.has_alpha() {
                expected
                    .chunks_exact(4)
                    .flat_map(|pixel| pixel[..3].iter().copied())
                    .collect()
            } else {
                expected.clone()
            };
            assert_eq!(djxl, color, "{} djxl color oracle", case.name);
        }
        let packed = packed_modular_bytes(&expected, case.bits);
        for (decoder, is_bounded) in [(&whole, false), (&bounded, true)] {
            eprintln!("{} native bounded={is_bounded}", case.name);
            let (layout, bytes) = decode(
                &backend,
                decoder,
                &encoded,
                case.native_request(),
                is_bounded,
            );
            assert_eq!(layout.extent, case.extent());
            assert_eq!(bytes, packed, "{} bounded={is_bounded}", case.name);
        }
    }
    assert!(saw_fused && saw_group_inverse && saw_frame_inverse);
}

#[test]
fn oriented_gray_modular_preserves_all_vpi_color_and_numeric_layouts() {
    let Some(backend) = backend() else {
        return;
    };
    let whole = GpuDecoder::new(WgpuSubmissionEngine::new(backend.clone()));
    let bounded = GpuDecoder::new(
        WgpuSubmissionEngine::new(backend.clone())
            .with_stream_window_limit(NonZeroU64::new(4096).unwrap()),
    );
    for case in CASES
        .iter()
        .filter(|case| case.format == LosslessModularFormat::Gray)
    {
        let encoded = case.encoded();
        let pixels: Vec<_> = case
            .expected()
            .into_iter()
            .map(|sample| sample as u8)
            .collect();
        for vpi in Vpi::ALL {
            let format = vpi.pixel_format();
            let layout = ImageLayout::packed(case.extent(), format.clone()).unwrap();
            let (request, expected) = match classify_pixel_format(&format).unwrap() {
                jxl_gpu_formats::PixelFormatClass::Numeric(_) => {
                    let mapping = if vpi == Vpi::F64 {
                        NumericSampleMapping::NormalizedGray8F64(F64OutputPolicy::ExactF32Widening)
                    } else {
                        NumericSampleMapping::NormalizedGray8
                    };
                    let expected = expected_numeric_bytes(&format, &layout, &pixels, false);
                    (
                        GpuOutputRequest::numeric(format, mapping).unwrap(),
                        expected,
                    )
                }
                jxl_gpu_formats::PixelFormatClass::Color(_) => {
                    let ColorSpecification::Defined(color) = format.color_spec else {
                        unreachable!()
                    };
                    let target: Vec<_> = pixels
                        .iter()
                        .copied()
                        .map(|sample| target_nonlinear(sample, color.transfer))
                        .collect();
                    let expected =
                        convert_rgb_f32([&target, &target, &target], case.extent(), &format)
                            .unwrap();
                    (GpuOutputRequest::color(format).unwrap(), expected.bytes)
                }
            };
            let mut previous = None;
            for (decoder, is_bounded) in [(&whole, false), (&bounded, true)] {
                eprintln!("{} {} bounded={is_bounded}", case.name, vpi.name());
                let (actual_layout, bytes) =
                    decode(&backend, decoder, &encoded, request.clone(), is_bounded);
                assert_eq!(actual_layout, layout);
                let name = format!("{} {} bounded={is_bounded}", case.name, vpi.name());
                if request.mapping() == jxl_wgpu_decode::GpuOutputMapping::Color {
                    assert_color_precision(&name, &bytes, &expected, &layout);
                } else {
                    assert!(bytes == expected, "{name}: numeric bytes differ");
                }
                if let Some(whole) = &previous {
                    assert!(
                        &bytes == whole,
                        "{name}: whole and fragmented results differ"
                    );
                }
                previous = Some(bytes);
            }
        }
    }
}
