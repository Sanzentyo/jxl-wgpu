//! Modular residual-pass images against native libjxl prefix flushes.
use super::*;
use jxl_wgpu_decode::FrameProgression;

mod lifecycle;

mod schedules;

mod composition;

fn fixture(width: u32, height: u32) -> Option<Vec<u8>> {
    static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if Command::new("cjxl").arg("--version").output().is_err() {
        return None;
    }
    let directory = std::env::temp_dir().join(format!(
        "jxl-wgpu-modular-progress-{}-{width}-{height}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let input = directory.join("source.pgm");
    let output = directory.join("image.jxl");
    let mut pixels = format!("P5\n{width} {height}\n255\n").into_bytes();
    pixels.extend((0..width * height).map(|i| {
        let x = i % width;
        let y = i / width;
        (x * 37 + y * 73 + (x * y) % 251) as u8
    }));
    std::fs::write(&input, pixels).unwrap();
    let encoded = Command::new("cjxl")
        .arg(&input)
        .arg(&output)
        .args([
            "-d",
            "0",
            "-m",
            "1",
            "-e",
            "9",
            "-p",
            "-R",
            "1",
            "-x",
            "color_space=Gra_D65_Rel_SRG",
        ])
        .output()
        .unwrap();
    assert!(
        encoded.status.success(),
        "{}",
        String::from_utf8_lossy(&encoded.stderr)
    );
    let data = std::fs::read(output).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    Some(data)
}

fn rgba_request() -> GpuOutputRequest {
    GpuOutputRequest::color(PixelFormat::rgb_f32(
        jxl_gpu_formats::RgbChannelOrder::Rgba,
        false,
        jxl_gpu_formats::vpi::VpiColorSpec::Srgb.specification(),
    ))
    .unwrap()
    .with_progressive_output(true)
    .with_max_frame_slots(NonZeroUsize::new(1).unwrap())
}

fn hex_bytes(hex: &str) -> Vec<u8> {
    let digits = hex.split_whitespace().collect::<String>();
    digits
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn prefix_end(frame: &jxl_gpu_bitstream::FrameInventory, completed: u8) -> usize {
    frame
        .sections
        .iter()
        .filter_map(|section| match section.kind {
            jxl_gpu_bitstream::FrameSectionKind::PassGroup { pass_index, .. }
                if pass_index >= u32::from(completed) =>
            {
                Some(section.bytes.offset as usize)
            }
            _ => None,
        })
        .min()
        .unwrap()
}

#[test]
fn modular_integer_float_and_extra_pass_outputs_match_independent_native_planes() {
    use jxl_gpu_bitstream::SampleBitDepth;
    use jxl_wgpu_decode::{AlphaOutputPolicy, OrientationPolicy, SpotColorPolicy};
    let Some(backend) = backend() else {
        return;
    };
    for (name, hex) in [
        (
            "associated",
            include_str!("../../test-data/extras_associated_squeeze.jxl.hex"),
        ),
        (
            "resampled",
            include_str!("../../test-data/extras_resampled_squeeze.jxl.hex"),
        ),
        (
            "floating",
            include_str!("../../test-data/floating/extras_float_squeeze.jxl.hex"),
        ),
        (
            "global",
            include_str!("../../test-data/floating/extras_float_global.jxl.hex"),
        ),
    ] {
        let encoded = hex_bytes(hex);
        let parsed = jxl_gpu_bitstream::parse(&encoded, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let image = &inventory.image_header;
        let frame = &inventory.frames[0];
        let pixels = (image.width * image.height) as usize;
        let mut requests = vec![(
            None,
            32,
            rgba_request()
                .with_alpha_output_policy(AlphaOutputPolicy::Unassociated)
                .with_spot_color_policy(SpotColorPolicy::Preserve),
        )];
        let mut channels = vec![0, image.extra_channels.len() - 1];
        channels.dedup();
        for channel in channels {
            {
                let bits = 32;
                let mapping = match image.extra_channels[channel].bit_depth {
                    SampleBitDepth::Integer { .. } => NumericSampleMapping::NormalizedUnsigned,
                    SampleBitDepth::Float { .. } => NumericSampleMapping::NativeFloat,
                };
                let request = GpuOutputRequest::numeric(
                    PixelFormat::non_color(SampleKind::Float, bits, &[Channel::X]),
                    mapping,
                )
                .unwrap()
                .with_extra_channel(channel as u32)
                .unwrap()
                .with_progressive_output(true);
                requests.push((Some(channel), bits, request));
            }
        }
        for cap in [40, 1 << 20] {
            let decoder = GpuDecoder::new(
                WgpuSubmissionEngine::new(backend.clone())
                    .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
            );
            for (channel, bits, request) in &requests {
                let request = request
                    .clone()
                    .with_orientation_policy(OrientationPolicy::Keep);
                let mut session = decoder
                    .open(&encoded, request.clone())
                    .unwrap_or_else(|e| panic!("{name} channel={channel:?} bits={bits}: {e}"));
                let stats = session.submission_session().memory_stats();
                let expected_intermediates =
                    stats.intermediate_output_bytes / stats.output_lease_bytes;
                let mut held = Vec::new();
                let mut count = 0;
                while let Some(update) = session
                    .next_update()
                    .unwrap_or_else(|e| panic!("{name} {channel:?}/{bits} cap={cap}: {e}"))
                {
                    let (input, prefix) = if let Some(FrameProgression::Modular {
                        completed_passes,
                        ..
                    }) = update.progression()
                    {
                        count += 1;
                        (
                            &parsed.codestream()[..prefix_end(frame, completed_passes)],
                            true,
                        )
                    } else {
                        (parsed.codestream(), false)
                    };
                    let mut options = vec!["--keep-orientation"];
                    if prefix {
                        options.push("--prefix");
                    }
                    let Some(expected) =
                        extra_channels::extra_channel_oracle::libjxl_output(input, &options)
                    else {
                        return;
                    };
                    assert_eq!(expected.len(), pixels * (4 + image.extra_channels.len()));
                    let expected = channel.map_or(&expected[..pixels * 4], |channel| {
                        &expected[pixels * (4 + channel)..pixels * (5 + channel)]
                    });
                    let bytes = read_output(&backend, &update.output().outputs[0]);
                    let values = bytes
                        .chunks_exact(usize::from(*bits / 8))
                        .map(|value| {
                            if *bits == 64 {
                                f64::from_le_bytes(value.try_into().unwrap())
                            } else {
                                f64::from(f32::from_le_bytes(value.try_into().unwrap()))
                            }
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(values.len(), expected.len());
                    let error = values
                        .iter()
                        .zip(expected)
                        .map(|(&a, &b)| {
                            if !a.is_finite() || !b.is_finite() {
                                assert!(
                                    a == f64::from(b) || (a.is_nan() && b.is_nan()),
                                    "{name} {channel:?}/{bits} cap={cap} {:?}: GPU {a}, native {b}",
                                    update.progression()
                                );
                                return 0.0;
                            }
                            (a - f64::from(b)).abs() / f64::from(b).abs().max(1.0)
                        })
                        .fold(0_f64, f64::max);
                    assert!(
                        error < 3e-5,
                        "{name} channel={channel:?} bits={bits} cap={cap} {:?}: {error}",
                        update.progression()
                    );
                    held.push((update, bytes));
                }
                assert_eq!(count, expected_intermediates, "{name}");
                if name != "global" {
                    assert!(count > 0, "{name} must exercise progression");
                }
                let final_bytes = held.last().unwrap().1.clone();
                drop(session);
                for (update, bytes) in &held {
                    assert_eq!(read_output(&backend, &update.output().outputs[0]), *bytes);
                }
                drop(held);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                let mut final_only = decoder
                    .open(&encoded, request.with_progressive_output(false))
                    .unwrap();
                let final_frame = final_only.next_frame().unwrap().unwrap();
                assert_eq!(
                    read_output(&backend, &final_frame.output().outputs[0]),
                    final_bytes
                );
            }
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn modular_pass_images_match_native_prefixes_and_remain_immutable() {
    let Some(backend) = backend() else {
        return;
    };
    for (width, height) in [(259, 35), (2051, 259)] {
        let Some(encoded) = fixture(width, height) else {
            return;
        };
        let parsed = jxl_gpu_bitstream::parse(&encoded, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let frame = &inventory.frames[0];
        assert_eq!(frame.num_passes, 2);
        let data = parsed.codestream();
        let mut prefixes = Vec::new();
        for completed in 0..frame.num_passes {
            let end = frame
                .sections
                .iter()
                .filter_map(|section| match section.kind {
                    jxl_gpu_bitstream::FrameSectionKind::PassGroup { pass_index, .. }
                        if pass_index >= completed =>
                    {
                        Some(section.bytes.offset as usize)
                    }
                    _ => None,
                })
                .min()
                .unwrap();
            let Some(updates) = jxl_test_support::oracles::progressive::native_updates_options(
                &data[..end],
                false,
                true,
                true,
            ) else {
                return;
            };
            prefixes.push(updates.last().unwrap().pixels.clone());
        }
        // Keep the large fixture on whole packets; tiny windows on the small image exercise
        // continuation within entropy streams without making a conformance test a benchmark.
        for cap in if width < 1000 {
            vec![40, 1 << 20]
        } else {
            vec![1 << 20]
        } {
            let decoder = GpuDecoder::new(
                WgpuSubmissionEngine::new(backend.clone())
                    .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
            );
            let request = rgba_request();
            let mut session = decoder.open(&encoded, request.clone()).unwrap();
            let stats = session.submission_session().memory_stats();
            assert_eq!(
                stats.intermediate_output_bytes,
                stats.output_lease_bytes * 2
            );
            assert!(stats.intermediate_transient_bytes > 0);
            assert_eq!(stats.submissions_per_frame, stats.stream_batch_count + 3);
            let mut held = Vec::new();
            for completed in 0..2 {
                let update = if completed == 0 {
                    session.next_update().unwrap().unwrap()
                } else {
                    pollster::block_on(session.next_update_async())
                        .unwrap()
                        .unwrap()
                };
                assert_eq!(
                    update.progression(),
                    Some(FrameProgression::Modular {
                        physical_frame_index: 0,
                        completed_passes: completed,
                        total_passes: 2,
                        intended_downsampling: if completed == 0 { 8 } else { 2 },
                    })
                );
                let actual = read_output(&backend, &update.output().outputs[0]);
                let expected = &prefixes[usize::from(completed)];
                assert_eq!(actual.len(), expected.len());
                let error = actual
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(expected.as_chunks::<4>().0.iter())
                    .map(|(a, b)| (f32::from_le_bytes(*a) - f32::from_le_bytes(*b)).abs())
                    .fold(0_f32, f32::max);
                assert!(
                    error < 2e-6,
                    "{width}x{height} cap={cap} pass={completed}: {error}"
                );
                held.push((update, actual));
            }
            let final_update = pollster::block_on(session.next_update_async())
                .unwrap()
                .unwrap();
            assert_eq!(final_update.progression(), None);
            let final_bytes = read_output(&backend, &final_update.output().outputs[0]);
            assert!(session.next_update().unwrap().is_none());
            drop(final_update);
            drop(session);
            for (update, bytes) in &held {
                assert_eq!(read_output(&backend, &update.output().outputs[0]), *bytes);
            }
            assert_eq!(
                decoder.engine().in_flight_memory_stats().reserved_bytes,
                stats.intermediate_output_bytes
            );
            drop(held);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            for progressive in [false, true] {
                let mut final_only = decoder
                    .open(
                        &encoded,
                        request.clone().with_progressive_output(progressive),
                    )
                    .unwrap();
                let final_frame = final_only.next_frame().unwrap().unwrap();
                assert_eq!(
                    read_output(&backend, &final_frame.output().outputs[0]),
                    final_bytes
                );
            }
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}
