//! Native small-image LF producers plus independent f64 expansion before inverse opsin.
//! The native decoder has no LF flush with extras, so these references describe the explicit
//! intermediate-presentation policy; they are not ISO final-image precision measurements.
use super::*;
use jxl_gpu_bitstream::ImageHeaderInventory;
use jxl_wgpu_decode::FrameProgression;

fn up8(
    input: &[f64],
    width: usize,
    height: usize,
    out_w: usize,
    out_h: usize,
    image: &ImageHeaderInventory,
) -> Vec<f64> {
    let mirror = |v: i64, size: usize| {
        let p = size as i64 * 2;
        let v = v.rem_euclid(p);
        v.min(p - 1 - v) as usize
    };
    (0..out_h)
        .flat_map(|y| {
            (0..out_w).map(move |x| {
                let mut sum = 0.0;
                let mut low = f64::INFINITY;
                let mut high = f64::NEG_INFINITY;
                for row in 0..5 {
                    for col in 0..5 {
                        let value = input[mirror((y / 8) as i64 + row - 2, height) * width
                            + mirror((x / 8) as i64 + col - 2, width)];
                        let phase = |p: usize, tap: i64| {
                            p.min(7 - p) as i64 * 5 + if p < 4 { tap } else { 4 - tap }
                        };
                        let (a, b) = (phase(x % 8, col), phase(y % 8, row));
                        let (i, j) = (a.min(b), a.max(b));
                        sum += value
                            * f64::from(
                                image.upsampling_weights.up8[(i * (41 - i) / 2 + j - i) as usize]
                                    .to_f32(),
                            );
                        low = low.min(value);
                        high = high.max(value);
                    }
                }
                sum.clamp(low, high)
            })
        })
        .collect()
}

fn inverse(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut rows: [[f64; 6]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if j < 3 {
                matrix[i][j]
            } else {
                f64::from(j - 3 == i)
            }
        })
    });
    for i in 0..3 {
        let divisor = rows[i][i];
        for value in &mut rows[i] {
            *value /= divisor;
        }
        let pivot = rows[i];
        for (k, row) in rows.iter_mut().enumerate() {
            if k == i {
                continue;
            }
            let scale = row[i];
            for (value, pivot) in row.iter_mut().zip(pivot) {
                *value -= scale * pivot;
            }
        }
    }
    rows.map(|row| row[3..].try_into().unwrap())
}
fn product(matrix: [[f64; 3]; 3], values: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| row.into_iter().zip(values).map(|(a, b)| a * b).sum())
}

fn expected(name: &str, level: u8, image: &ImageHeaderInventory) -> [Vec<f64>; 5] {
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/lf_extra_channels");
    expected_from(&directory, name, level, image)
}

fn expected_from(
    directory: &std::path::Path,
    name: &str,
    level: u8,
    image: &ImageHeaderInventory,
) -> [Vec<f64>; 5] {
    expected_at(directory, name, level, image, [image.width, image.height])
}

fn expected_at(
    directory: &std::path::Path,
    name: &str,
    level: u8,
    image: &ImageHeaderInventory,
    extent: [u32; 2],
) -> [Vec<f64>; 5] {
    let input =
        std::fs::read_to_string(directory.join(format!("{name}.lf{level}.jxl.hex"))).unwrap();
    let compact = input.split_whitespace().collect::<String>();
    let data = compact
        .as_bytes()
        .chunks_exact(2)
        .map(|s| u8::from_str_radix(std::str::from_utf8(s).unwrap(), 16).unwrap())
        .collect::<Vec<_>>();
    let original = read_fixture(&directory.join(format!("{name}.jxl.hex")));
    let source = parse(&original, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let standalone = parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(
        source.image_header.bit_depth,
        standalone.image_header.bit_depth
    );
    assert_eq!(
        source.image_header.extra_channels,
        standalone.image_header.extra_channels
    );
    assert_eq!(
        source.image_header.orientation,
        standalone.image_header.orientation
    );
    assert_eq!(
        source.image_header.grayscale,
        standalone.image_header.grayscale
    );
    for (from, to) in source.frames.iter().zip(&standalone.frames) {
        assert_eq!(from.color_sample_extent(), to.color_sample_extent());
        assert_eq!(from.encoding, to.encoding);
        assert_eq!(from.restoration_filter, to.restoration_filter);
        assert_eq!(from.extra_channel_upsampling, to.extra_channel_upsampling);
        assert_eq!(from.sections.len(), to.sections.len());
        for (a, b) in from.sections.iter().zip(&to.sections) {
            assert_eq!(a.kind, b.kind);
            assert_eq!(
                &original[a.bytes.offset as usize..(a.bytes.offset + a.bytes.length) as usize],
                &data[b.bytes.offset as usize..(b.bytes.offset + b.bytes.length) as usize]
            );
        }
    }
    let hex = std::fs::read_to_string(directory.join(format!("{name}.lf{level}.linear.f32.hex")))
        .unwrap();
    let native = hex
        .split_whitespace()
        .map(|s| f32::from_bits(u32::from_str_radix(s, 16).unwrap()))
        .collect::<Vec<_>>();
    if let Some(actual) = oracle::libjxl_output(
        &data,
        &["--linear", "--preserve-alpha", "--keep-orientation"],
    ) {
        assert_eq!(actual.len(), native.len());
        assert!(
            actual
                .iter()
                .zip(&native)
                .all(|(a, b)| (a - b).abs() <= 2e-6)
        );
    }
    let mut width = image.width.div_ceil(1 << (3 * level)) as usize;
    let mut height = image.height.div_ceil(1 << (3 * level)) as usize;
    let pixels = width * height;
    assert_eq!(native.len(), pixels * 6);
    let opsin = image.opsin_inverse_matrix.unwrap();
    let matrix = opsin
        .inverse_matrix
        .map(|r| r.map(|v| f64::from(v.to_f32())));
    let forward = inverse(matrix);
    let bias = opsin.opsin_bias.map(|v| f64::from(v.to_f32()));
    let intensity = f64::from(image.tone_mapping.intensity_target.to_f32()) / 255.0;
    let mut planes: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::with_capacity(pixels));
    for i in 0..pixels {
        let lms = product(
            forward,
            std::array::from_fn(|c| f64::from(native[i * 4 + c])),
        );
        let mixed: [f64; 3] =
            std::array::from_fn(|c| (lms[c] * intensity - bias[c]).cbrt() + bias[c].cbrt());
        let xyb = [
            (mixed[0] - mixed[1]) / 2.0,
            (mixed[0] + mixed[1]) / 2.0,
            mixed[2],
        ];
        for c in 0..3 {
            planes[c].push(xyb[c]);
        }
        for c in 0..2 {
            planes[3 + c].push(f64::from(native[(4 + c) * pixels + i]));
        }
    }
    // LF prediction addresses a consumer-local rectangle at the producer's top left.
    // Its presentation policy clips that grid before recursive expansion and composition.
    let crop_w = extent[0].div_ceil(1 << (3 * level)) as usize;
    let crop_h = extent[1].div_ceil(1 << (3 * level)) as usize;
    assert!(crop_w <= width && crop_h <= height);
    planes = planes.map(|v| {
        v.chunks_exact(width)
            .take(crop_h)
            .flat_map(|row| row[..crop_w].iter().copied())
            .collect()
    });
    width = crop_w;
    height = crop_h;
    for stage in (0..level).rev() {
        let out_w = extent[0].div_ceil(1 << (3 * stage)) as usize;
        let out_h = extent[1].div_ceil(1 << (3 * stage)) as usize;
        planes = planes.map(|v| up8(&v, width, height, out_w, out_h, image));
        width = out_w;
        height = out_h;
    }
    let [x, y, b, _, _] = &mut planes;
    for ((x, y), b) in x.iter_mut().zip(y).zip(b) {
        let mixed = [*y + *x, *y - *x, *b];
        let lms =
            std::array::from_fn(|c| ((mixed[c] - bias[c].cbrt()).powi(3) + bias[c]) / intensity);
        let rgb = product(matrix, lms).map(|value| {
            let a = value.abs();
            value.signum()
                * if a <= 0.0031308 {
                    a * 12.92
                } else {
                    1.055 * a.powf(1.0 / 2.4) - 0.055
                }
        });
        (*x, *y, *b) = (rgb[0], rgb[1], rgb[2]);
    }
    planes
}

#[path = "lf_conformance.rs"]
mod conformance;

#[test]
fn lf_presentations_preserve_alpha_and_numeric_extras_and_final_bytes() {
    let Some(backend) = backend() else { return };
    for (name, hex, _) in fixtures() {
        let data = data(hex);
        let inventory = parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let expected: Vec<_> = inventory
            .frames
            .iter()
            .filter(|f| f.frame_type == FrameType::LowFrequency)
            .map(|f| expected(name, f.lf_level as u8, &inventory.image_header))
            .collect();
        for bounded in [false, true] {
            let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            if bounded {
                engine = engine.with_stream_window_limit(NonZeroU64::new(40).unwrap());
            }
            let decoder = GpuDecoder::new(engine);
            for extra in [None, Some(0), Some(1)] {
                let request =
                    output_request(extra).with_max_frame_slots(NonZeroUsize::new(1).unwrap());
                let mut baseline = decoder.open(&data, request.clone()).unwrap();
                let final_frame = baseline.next_frame().unwrap().unwrap();
                let final_bytes = read_output(&backend, &final_frame.output().outputs[0]);
                drop(final_frame);
                drop(baseline);
                let request = request.with_progressive_output(true);
                let mut session = if bounded {
                    incremental(&decoder, &data, request)
                } else {
                    decoder.open(&data, request).unwrap()
                };
                let mut count = 0;
                let mut complete = 0;
                let mut held = Vec::new();
                while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                    let output = &update.output().outputs[0];
                    let bytes = read_output(&backend, output);
                    if let Some(FrameProgression::LowFrequency {
                        physical_frame_index,
                        level,
                    }) = update.progression()
                    {
                        assert_eq!(physical_frame_index as usize, count);
                        assert_eq!(level as u32, inventory.frames[count].lf_level);
                        let actual = oracle::floats(&bytes);
                        let reference = &expected[count];
                        let channels = if extra.is_some() { 1 } else { 4 };
                        let mut error = 0.0_f64;
                        for (i, value) in actual.iter().enumerate() {
                            let c = extra.map_or(i % channels, |e| 3 + e as usize);
                            let expected = reference[c][i / channels];
                            error = error.max((f64::from(*value) - expected).abs());
                            assert!(value.is_finite());
                        }
                        let limit = if extra.is_some() { 3e-6 } else { 5e-4 };
                        assert!(
                            error <= limit,
                            "{name} bounded={bounded} extra={extra:?} LF{level}: {error}"
                        );
                        eprintln!(
                            "{name} bounded={bounded} extra={extra:?} LF{level} maxAE={error}"
                        );
                        count += 1;
                        held.push((
                            jxl_wgpu::GpuImageOutput {
                                id: output.id,
                                layout: output.layout.clone(),
                                buffer: output.buffer.clone(),
                            },
                            bytes,
                        ));
                    } else if update.is_complete() {
                        assert_eq!(bytes, final_bytes, "{name} final output changed");
                        complete += 1;
                    }
                }
                assert_eq!(count, expected.len());
                assert_eq!(complete, 1);
                drop(session);
                retired(&backend);
                assert_eq!(
                    decoder.engine().in_flight_memory_stats().reserved_bytes,
                    held.iter().map(|(o, _)| o.buffer.size()).sum::<u64>()
                );
                for (output, bytes) in &held {
                    assert_eq!(read_output(&backend, output), *bytes);
                }
                drop(held);
                retired(&backend);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}

fn fixture_file(name: &str, kind: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test-data/lf_extra_channels")
        .join(format!(
            "{name}{}.jxl.hex",
            if kind.is_empty() {
                String::new()
            } else {
                format!(".{kind}")
            }
        ));
    read_fixture(&path)
}

fn read_fixture(path: &std::path::Path) -> Vec<u8> {
    let text = std::fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .collect::<String>();
    text.as_bytes()
        .chunks_exact(2)
        .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
        .collect()
}

fn compose(planes: &mut [Vec<f64>; 5], background: &[f32]) {
    for i in 0..planes[0].len() {
        let a = planes[3][i].clamp(0.0, 1.0);
        let base_a = f64::from(background[i * 4 + 3]);
        let out_a = a + base_a * (1.0 - a);
        for c in 0..3 {
            planes[c][i] = (planes[c][i] * a
                + f64::from(background[i * 4 + c]) * base_a * (1.0 - a))
                / out_a.max(2_f64.powi(-26));
        }
        planes[3][i] = out_a;
        planes[4][i] += f64::from(background[planes[4].len() * 5 + i]);
    }
}

#[test]
fn composed_lf_extras_wait_for_background_and_preserve_all_output_planes() {
    let Some(backend) = backend() else { return };
    for name in ["nested_modular_gab1", "nested_vardct_gab1"] {
        let data = fixture_file(name, "composed");
        let inventory = parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert_eq!(inventory.frames.len(), 4);
        let Some(background) =
            oracle::libjxl_output(&fixture_file(name, "background"), &["--preserve-alpha"])
        else {
            return;
        };
        let Some(native_final) = oracle::libjxl_output(&data, &["--preserve-alpha"]) else {
            return;
        };
        let mut expected: Vec<_> = (0..2)
            .map(|i| {
                expected(
                    name,
                    inventory.frames[i].lf_level as u8,
                    &inventory.image_header,
                )
            })
            .collect();
        for planes in &mut expected {
            compose(planes, &background);
        }
        let foreground =
            oracle::libjxl_output(&fixture_file(name, "foreground"), &["--preserve-alpha"])
                .unwrap();
        let pixels = inventory.image_header.width as usize * inventory.image_header.height as usize;
        let mut scalar_final: [Vec<f64>; 5] = std::array::from_fn(|channel| {
            (0..pixels)
                .map(|i| {
                    f64::from(if channel < 3 {
                        foreground[i * 4 + channel]
                    } else {
                        foreground[(channel + 1) * pixels + i]
                    })
                })
                .collect()
        });
        compose(&mut scalar_final, &background);
        for channel in 0..5 {
            for i in 0..pixels {
                let expected = f64::from(if channel < 3 {
                    native_final[i * 4 + channel]
                } else {
                    native_final[(channel + 1) * pixels + i]
                });
                assert!(
                    (scalar_final[channel][i] - expected).abs() <= 3e-6,
                    "{name}: scalar final composition differs from native output"
                );
            }
        }

        let pixels = inventory.image_header.width as usize * inventory.image_header.height as usize;
        for bounded in [false, true] {
            let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            if bounded {
                engine = engine.with_stream_window_limit(NonZeroU64::new(40).unwrap());
            }
            let decoder = GpuDecoder::new(engine);
            for extra in [None, Some(0), Some(1)] {
                for native in [false, true] {
                    if native && extra.is_none() {
                        continue;
                    }
                    let request = if native {
                        GpuOutputRequest::numeric(
                            jxl_wgpu_decode::native_modular_pixel_format(
                                jxl_wgpu_decode::ModularChannels::Gray,
                                8,
                            )
                            .unwrap(),
                            NumericSampleMapping::NativeUnsigned,
                        )
                        .unwrap()
                        .with_extra_channel(extra.unwrap())
                        .unwrap()
                    } else {
                        output_request(extra)
                    };
                    let mut baseline = decoder.open(&data, request.clone()).unwrap();
                    let frame = baseline.next_frame().unwrap().unwrap();
                    let final_bytes = read_output(&backend, &frame.output().outputs[0]);
                    drop(frame);
                    drop(baseline);
                    if !native {
                        let floats = oracle::floats(&final_bytes);
                        let reference = extra.map_or(&native_final[..pixels * 4], |e| {
                            &native_final[(4 + e as usize) * pixels..(5 + e as usize) * pixels]
                        });
                        let error = floats
                            .iter()
                            .zip(reference)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0_f32, f32::max);
                        assert!(error <= 5e-4, "{name} final extra={extra:?}: {error}");
                    }
                    let mut session =
                        incremental(&decoder, &data, request.with_progressive_output(true));
                    let mut lf = 0;
                    let mut finals = 0;
                    while let Some(update) =
                        pollster::block_on(session.next_update_async()).unwrap()
                    {
                        let bytes = read_output(&backend, &update.output().outputs[0]);
                        if let Some(FrameProgression::LowFrequency {
                            physical_frame_index,
                            ..
                        }) = update.progression()
                        {
                            assert_eq!(physical_frame_index as usize, lf);
                            let channels = if extra.is_some() { 1 } else { 4 };
                            let actual = if native {
                                bytes.iter().map(|v| f64::from(*v)).collect::<Vec<_>>()
                            } else {
                                oracle::floats(&bytes).into_iter().map(f64::from).collect()
                            };
                            let mut error = 0_f64;
                            for (i, value) in actual.iter().enumerate() {
                                let channel = extra.map_or(i % channels, |e| 3 + e as usize);
                                let value_ref = expected[lf][channel][i / channels];
                                let value_ref = if native {
                                    (value_ref * 255.0).round().clamp(0.0, 255.0)
                                } else {
                                    value_ref
                                };
                                error = error.max((value - value_ref).abs());
                                assert!(value.is_finite());
                            }
                            assert!(
                                error
                                    <= if native {
                                        1.0
                                    } else if extra.is_some() {
                                        3e-6
                                    } else {
                                        5e-4
                                    },
                                "{name} composed bounded={bounded} extra={extra:?} native={native} LF{lf}: {error}"
                            );
                            eprintln!(
                                "{name} composed bounded={bounded} extra={extra:?} native={native} LF{lf} maxAE={error}"
                            );
                            lf += 1;
                        } else if update.is_complete() {
                            assert_eq!(bytes, final_bytes);
                            finals += 1;
                        }
                    }
                    assert_eq!((lf, finals), (2, 1));
                    drop(session);
                    retired(&backend);
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                }
            }
        }
    }
}

#[test]
fn lf_extra_updates_cancel_or_switch_to_blocking_and_async_final_without_leaks() {
    let Some(backend) = backend() else { return };
    for (name, hex, _) in fixtures()
        .into_iter()
        .filter(|(name, _, _)| name.starts_with("nested"))
    {
        for composed in [false, true] {
            let data = if composed {
                fixture_file(name, "composed")
            } else {
                data(hex)
            };
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
            );
            for extra in [None, Some(1)] {
                let request =
                    output_request(extra).with_max_frame_slots(NonZeroUsize::new(1).unwrap());
                let mut baseline = decoder.open(&data, request.clone()).unwrap();
                let frame = baseline.next_frame().unwrap().unwrap();
                let final_bytes = read_output(&backend, &frame.output().outputs[0]);
                drop(frame);
                drop(baseline);
                for boundary in 0..=2 {
                    for action in 0..3 {
                        let mut session = incremental(
                            &decoder,
                            &data,
                            request.clone().with_progressive_output(true),
                        );
                        let memory = backend.transient_memory_budget();
                        let blocker = memory
                            .try_reserve(memory.snapshot().available_bytes)
                            .unwrap();
                        assert_eq!(
                            session
                                .prefetch(NonZeroUsize::new(1).unwrap())
                                .unwrap()
                                .submitted,
                            0
                        );
                        drop(blocker);
                        session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
                        let mut held = Vec::new();
                        for index in 0..boundary {
                            let update = pollster::block_on(session.next_update_async())
                                .unwrap()
                                .unwrap();
                            assert!(
                                matches!(update.progression(),Some(FrameProgression::LowFrequency{physical_frame_index,..}) if physical_frame_index==index)
                            );
                            let output = &update.output().outputs[0];
                            held.push((
                                jxl_wgpu::GpuImageOutput {
                                    id: output.id,
                                    layout: output.layout.clone(),
                                    buffer: output.buffer.clone(),
                                },
                                read_output(&backend, output),
                            ));
                        }
                        if action != 0 {
                            let frame = if action == 1 {
                                session.next_frame().unwrap().unwrap()
                            } else {
                                pollster::block_on(session.next_frame_async())
                                    .unwrap()
                                    .unwrap()
                            };
                            assert_eq!(
                                read_output(&backend, &frame.output().outputs[0]),
                                final_bytes
                            );
                            drop(frame);
                        }
                        drop(session);
                        retired(&backend);
                        assert_eq!(
                            decoder.engine().in_flight_memory_stats().reserved_bytes,
                            held.iter().map(|(o, _)| o.buffer.size()).sum::<u64>()
                        );
                        assert_eq!(
                            decoder.incremental_input_budget().snapshot().reserved_bytes,
                            0
                        );
                        for (output, bytes) in &held {
                            assert_eq!(read_output(&backend, output), *bytes);
                        }
                        drop(held);
                        retired(&backend);
                        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                    }
                }
            }
        }
    }
}

#[test]
fn corrupt_lf_extras_and_late_backgrounds_cannot_publish_unvalidated_previews() {
    let Some(backend) = backend() else { return };
    for name in ["nested_modular_gab1", "nested_vardct_gab1"] {
        let data = fixture_file(name, "composed");
        let inventory = parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        for damaged in 0..4 {
            let mut corrupt = data[..inventory.frames[0].header_bits.offset as usize / 8].to_vec();
            for (index, frame) in inventory.frames.iter().enumerate() {
                let mut packets = payloads(&data, frame);
                if index == damaged {
                    let end =
                        global_end(&data, frame) / 8 - frame.sections[0].bytes.offset as usize;
                    packets[0].truncate(end - 1);
                }
                corrupt.extend(reassemble(&data, frame, packets));
            }
            parse(&corrupt, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
            );
            for extra in [None, Some(1)] {
                let mut session = incremental(
                    &decoder,
                    &corrupt,
                    output_request(extra).with_progressive_output(true),
                );
                let mut held = Vec::new();
                loop {
                    match pollster::block_on(session.next_update_async()) {
                        Ok(Some(update)) => {
                            assert!(matches!(
                                update.progression(),
                                Some(FrameProgression::LowFrequency { .. })
                            ));
                            let output = &update.output().outputs[0];
                            held.push((
                                jxl_wgpu::GpuImageOutput {
                                    id: output.id,
                                    layout: output.layout.clone(),
                                    buffer: output.buffer.clone(),
                                },
                                read_output(&backend, output),
                            ));
                        }
                        Err(error) => {
                            assert!(
                                matches!(
                                    error,
                                    Error::ModularEntropyRejected { .. } | Error::VarDct(_)
                                ),
                                "{name} damaged={damaged}: {error:?}"
                            );
                            break;
                        }
                        Ok(None) => panic!("{name} damaged={damaged}: corrupt entropy accepted"),
                    }
                }
                assert_eq!(held.len(), if damaged == 3 { 2 } else { 0 });
                drop(session);
                retired(&backend);
                assert_eq!(
                    decoder.engine().in_flight_memory_stats().reserved_bytes,
                    held.iter().map(|(o, _)| o.buffer.size()).sum::<u64>()
                );
                for (output, bytes) in &held {
                    assert_eq!(read_output(&backend, output), *bytes);
                }
                drop(held);
                retired(&backend);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    0
                );
            }
        }
    }
}
