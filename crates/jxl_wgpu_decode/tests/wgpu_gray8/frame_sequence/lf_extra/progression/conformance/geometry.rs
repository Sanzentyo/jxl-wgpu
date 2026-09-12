//! Intrinsic extra-channel geometry is retained even when LF consumers omit selectors.
use super::*;

fn directory() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/lf_geometry")
}

fn cases() -> Vec<(String, u8)> {
    let cases = include_str!("../../../../../../test-data/lf_geometry/cases.txt")
        .lines()
        .flat_map(|row| {
            let (name, level) = row.split_once(' ').unwrap();
            ["modular", "vardct"].map(|root| (format!("{name}_{root}"), level.parse().unwrap()))
        })
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 8);
    cases
}

#[test]
fn lf_intrinsic_shifts_and_signed_samples_match_native_and_composed_outputs() {
    let Some(backend) = backend() else { return };
    for (name, levels) in cases() {
        for composed in [false, true] {
            let Some(check) = Check::from_directory(&directory(), &name, levels, composed) else {
                return;
            };
            if name.starts_with("signed_float_") {
                assert!(check.native[4].iter().any(|&v| v < 0.0));
                assert!(check.native[4].iter().any(|&v| v > 1.0));
                assert!(check.previews.iter().any(|p| p[4].iter().any(|&v| v < 0.0)));
            }
            for bounded in [false, true] {
                let orientation = if bounded {
                    OrientationPolicy::Apply
                } else {
                    OrientationPolicy::Keep
                };
                let mut preserved = Vec::new();
                for extra in [None, Some(0), Some(1)] {
                    let outputs = check.run(
                        &backend,
                        bounded,
                        orientation,
                        extra,
                        AlphaOutputPolicy::Preserve,
                    );
                    if extra.is_none() {
                        preserved = outputs;
                    }
                }
                if composed && bounded {
                    for policy in [
                        AlphaOutputPolicy::Associated,
                        AlphaOutputPolicy::Unassociated,
                    ] {
                        let converted = check.run(&backend, true, orientation, None, policy);
                        check.check_policy(&preserved, &converted, policy);
                    }
                }
            }
        }
    }
}

fn placed(
    foreground: &[Vec<f64>; 5],
    background: &[Vec<f64>; 5],
    canvas: [u32; 2],
    extent: [u32; 2],
    origin: [i32; 2],
    associated: bool,
) -> [Vec<f64>; 5] {
    let mut output = background.clone();
    for y in 0..extent[1] {
        for x in 0..extent[0] {
            let dx = i64::from(x) + i64::from(origin[0]);
            let dy = i64::from(y) + i64::from(origin[1]);
            if dx < 0 || dy < 0 || dx >= i64::from(canvas[0]) || dy >= i64::from(canvas[1]) {
                continue;
            }
            let source = (y * extent[0] + x) as usize;
            let target = (dy * i64::from(canvas[0]) + dx) as usize;
            let a = foreground[3][source].clamp(0.0, 1.0);
            let base_a = background[3][target];
            let alpha = a + base_a * (1.0 - a);
            for c in 0..3 {
                output[c][target] = if associated {
                    foreground[c][source] + background[c][target] * (1.0 - a)
                } else {
                    (foreground[c][source] * a + background[c][target] * base_a * (1.0 - a))
                        / alpha.max(2_f64.powi(-26))
                };
            }
            output[3][target] = alpha;
            output[4][target] += foreground[4][source];
        }
    }
    output
}

fn cropped(name: &str, levels: u8, suffix: &str, origin: [i32; 2]) -> Option<Check> {
    let mut check = Check::from_directory(&directory(), name, levels, false)?;
    let fixture = |suffix| fixture_from(&directory(), name, suffix);
    check.encoded = fixture(suffix);
    let inventory = parse(&check.encoded, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let terminal = inventory.frames.last().unwrap();
    let extent = [check.image.width - 12, check.image.height - 8];
    let canvas = [check.image.width, check.image.height];
    assert_eq!([terminal.width, terminal.height], extent);
    assert_eq!([terminal.x0, terminal.y0], origin);
    assert!(terminal.have_crop && terminal.uses_lf_frame());
    assert_eq!(inventory.frames.len(), usize::from(levels) + 2);
    let options = &["--preserve-alpha", "--keep-orientation"];
    let foreground = planes(&oracle::libjxl_output(
        &fixture(".crop_foreground"),
        options,
    )?);
    let foreground = foreground.map(|v| {
        v.chunks_exact(canvas[0] as usize)
            .take(extent[1] as usize)
            .flat_map(|row| row[..extent[0] as usize].iter().copied())
            .collect()
    });
    let background = planes(&oracle::libjxl_output(&fixture(".background"), options)?);
    let associated = matches!(
        check.image.extra_channels[0].channel_type,
        ExtraChannelTypeInventory::Alpha { associated: true }
    );
    check.native = planes(&oracle::libjxl_output(&check.encoded, options)?);
    let expected = placed(&foreground, &background, canvas, extent, origin, associated);
    for (c, expected) in expected.iter().enumerate() {
        assert_error(
            &check.native[c]
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<_>>(),
            expected,
            3e-6,
            &format!("{name}{suffix} scalar final channel={c}"),
            false,
        );
    }
    check.previews = (1..=levels)
        .rev()
        .map(|level| {
            let foreground = expected_at(&directory(), name, level, &check.image, extent);
            placed(&foreground, &background, canvas, extent, origin, associated)
        })
        .collect();
    check.name = format!("{name}{suffix}");
    Some(check)
}

#[test]
fn lf_cropped_consumers_keep_local_prediction_and_signed_canvas_placement() {
    let Some(backend) = backend() else { return };
    for (name, levels) in cases() {
        for (suffix, origin) in [(".crop_left", [-3, 5]), (".crop_top", [5, -3])] {
            let Some(check) = cropped(&name, levels, suffix, origin) else {
                return;
            };
            for bounded in [false, true] {
                for extra in [None, Some(0), Some(1)] {
                    check.run(
                        &backend,
                        bounded,
                        if bounded {
                            OrientationPolicy::Apply
                        } else {
                            OrientationPolicy::Keep
                        },
                        extra,
                        AlphaOutputPolicy::Preserve,
                    );
                }
            }
        }
    }
}

#[test]
fn lf_cropped_native_samples_survive_cancellation_and_final_only_drain() {
    let Some(backend) = backend() else { return };
    for (name, levels) in cases().into_iter().filter(|(name, _)| {
        name.starts_with("shifted_integer_") || name.starts_with("signed_float_")
    }) {
        let Some(check) = cropped(&name, levels, ".crop_left", [-3, 5]) else {
            return;
        };
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
        );
        for extra in 0..2 {
            let depth = check.image.extra_channels[extra].bit_depth;
            let request = match depth {
                SampleBitDepth::Integer { bits_per_sample } => GpuOutputRequest::numeric(
                    jxl_wgpu_decode::native_modular_pixel_format(
                        jxl_wgpu_decode::ModularChannels::Gray,
                        bits_per_sample.try_into().unwrap(),
                    )
                    .unwrap(),
                    NumericSampleMapping::NativeUnsigned,
                )
                .unwrap()
                .with_extra_channel(extra as u32)
                .unwrap(),
                SampleBitDepth::Float { .. } => request(&check.image, Some(extra as u32)),
            }
            .with_max_frame_slots(NonZeroUsize::new(1).unwrap());
            let compare = |bytes: &[u8], planes: &[Vec<f64>; 5]| {
                let values = check.values(
                    planes,
                    Some(extra as u32),
                    OrientationPolicy::Apply,
                    AlphaOutputPolicy::Preserve,
                );
                match depth {
                    SampleBitDepth::Integer { bits_per_sample } => {
                        lifecycle::integer_error(bytes, &values, bits_per_sample)
                    }
                    SampleBitDepth::Float { .. } => assert_error(
                        &oracle::floats(bytes),
                        &values,
                        3e-6,
                        &format!("{name} cropped native extra={extra}"),
                        false,
                    ),
                }
            };
            let mut baseline = decoder.open(&check.encoded, request.clone()).unwrap();
            let frame = baseline.next_frame().unwrap().unwrap();
            let final_bytes = read_output(&backend, &frame.output().outputs[0]);
            compare(&final_bytes, &check.native);
            drop(frame);
            drop(baseline);
            for boundary in 0..=usize::from(levels) {
                let mut session = incremental(
                    &decoder,
                    &check.encoded,
                    request.clone().with_progressive_output(true),
                );
                session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
                let mut held = Vec::new();
                for index in 0..boundary {
                    let update = pollster::block_on(session.next_update_async())
                        .unwrap()
                        .unwrap();
                    assert!(matches!(update.progression(),
                        Some(FrameProgression::LowFrequency { physical_frame_index, .. })
                        if physical_frame_index as usize == index));
                    let output = lifecycle::hold(&backend, &update.output().outputs[0]);
                    compare(&output.1, &check.previews[index]);
                    held.push(output);
                }
                if boundary == usize::from(levels) {
                    let frame = pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        read_output(&backend, &frame.output().outputs[0]),
                        final_bytes
                    );
                }
                drop(session);
                lifecycle::released(&backend, &decoder, held);
            }
        }
    }
}

#[test]
fn lf_cropped_and_shifted_corruption_keeps_validated_outputs_alive() {
    let Some(backend) = backend() else { return };
    for (name, levels) in cases() {
        lifecycle::reject_corruption(
            &backend,
            &name,
            levels,
            &fixture_from(&directory(), &name, ".crop_top"),
        );
    }
}
