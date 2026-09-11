//! Cross-feature LF streams; native decoding is an independent test oracle only.
use super::*;
use jxl_gpu_bitstream::{ExtraChannelTypeInventory, SampleBitDepth};
use jxl_wgpu_decode::{AlphaOutputPolicy, OrientationPolicy};

#[path = "lf_conformance_lifecycle.rs"]
mod lifecycle;

fn directory() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/lf_conformance")
}

fn fixture(name: &str, suffix: &str) -> Vec<u8> {
    read_fixture(&directory().join(format!("{name}{suffix}.jxl.hex")))
}

fn cases() -> Vec<(String, u8)> {
    let manifest = include_str!("../../../test-data/lf_conformance/cases.txt");
    let cases = manifest
        .lines()
        .flat_map(|row| {
            let (name, level) = row.split_once(' ').unwrap();
            ["modular", "vardct"].map(|root| (format!("{name}_{root}"), level.parse().unwrap()))
        })
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 10);
    cases
}

fn assert_metadata(name: &str, level: u8, image: &ImageHeaderInventory) {
    let integer = |bits_per_sample| SampleBitDepth::Integer { bits_per_sample };
    let float = |bits_per_sample, exponent_bits_per_sample| SampleBitDepth::Float {
        bits_per_sample,
        exponent_bits_per_sample,
    };
    let (depth, alpha, extra, associated, orientation, levels) =
        if name.starts_with("integer_associated_") {
            (integer(12), integer(16), integer(20), true, 6, 4)
        } else if name.starts_with("resampled_associated_") {
            (integer(16), integer(16), integer(20), true, 2, 1)
        } else if name.starts_with("floating_resampled_") {
            (float(32, 8), float(32, 8), float(32, 8), true, 7, 1)
        } else if name.starts_with("gray_float_") {
            (float(32, 8), float(16, 5), float(24, 7), true, 5, 2)
        } else {
            assert!(name.starts_with("floating_"));
            (float(24, 7), float(16, 5), float(24, 7), false, 8, 3)
        };
    assert_eq!(level, levels);
    assert_eq!(image.bit_depth, depth);
    assert_eq!(image.extra_channels.len(), 2);
    assert_eq!(image.extra_channels[0].bit_depth, alpha);
    assert_eq!(image.extra_channels[1].bit_depth, extra);
    assert_eq!(
        image.extra_channels[0].channel_type,
        ExtraChannelTypeInventory::Alpha { associated }
    );
    assert_eq!(image.orientation, orientation);
    assert_eq!(image.grayscale, name.starts_with("gray_"));
}

fn assert_error(actual: &[f32], expected: &[f64], tolerance: f64, label: &str, relative: bool) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let mut error = 0.0_f64;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite(), "{label}");
        error = error.max((f64::from(a) - b).abs() / if relative { b.abs().max(1.0) } else { 1.0 });
    }
    assert!(error <= tolerance, "{label}: maxAE={error}");
    eprintln!("{label}: relative={relative} error={error}");
}

fn request(image: &ImageHeaderInventory, extra: Option<u32>) -> GpuOutputRequest {
    if let Some(extra) = extra {
        GpuOutputRequest::numeric(
            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
            match image.extra_channels[extra as usize].bit_depth {
                SampleBitDepth::Float { .. } => NumericSampleMapping::NativeFloat,
                SampleBitDepth::Integer { .. } => NumericSampleMapping::NormalizedUnsigned,
            },
        )
        .unwrap()
        .with_extra_channel(extra)
        .unwrap()
    } else {
        output_request(None)
    }
    .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
}

fn planes(native: &[f32]) -> [Vec<f64>; 5] {
    assert_eq!(native.len() % 6, 0);
    let pixels = native.len() / 6;
    std::array::from_fn(|channel| {
        (0..pixels)
            .map(|i| {
                f64::from(if channel < 3 {
                    native[i * 4 + channel]
                } else {
                    native[(channel + 1) * pixels + i]
                })
            })
            .collect()
    })
}

fn blend(foreground: &mut [Vec<f64>; 5], background: &[Vec<f64>; 5], associated: bool) {
    for i in 0..foreground[0].len() {
        let a = foreground[3][i].clamp(0.0, 1.0);
        let base_a = background[3][i];
        let alpha = a + base_a * (1.0 - a);
        for c in 0..3 {
            foreground[c][i] = if associated {
                foreground[c][i] + background[c][i] * (1.0 - a)
            } else {
                (foreground[c][i] * a + background[c][i] * base_a * (1.0 - a))
                    / alpha.max(2_f64.powi(-26))
            };
        }
        foreground[3][i] = alpha;
        foreground[4][i] += background[4][i];
    }
}

struct Check {
    name: String,
    encoded: Vec<u8>,
    image: ImageHeaderInventory,
    previews: Vec<[Vec<f64>; 5]>,
    native: [Vec<f64>; 5],
}

impl Check {
    fn new(name: &str, levels: u8, composed: bool) -> Option<Self> {
        let encoded = fixture(name, if composed { ".composed" } else { "" });
        let inventory = parse(&encoded, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = inventory.image_header;
        assert_metadata(name, levels, &image);
        assert_eq!(
            inventory.frames.len(),
            usize::from(levels) + if composed { 2 } else { 1 }
        );
        for (index, frame) in inventory.frames.iter().take(levels as usize).enumerate() {
            assert_eq!(frame.lf_level, u32::from(levels) - index as u32);
            assert_eq!(
                frame.lf_source_frame,
                index.checked_sub(1).map(|i| i as u32)
            );
        }
        assert_eq!(
            inventory.frames.last().unwrap().lf_source_frame,
            Some(u32::from(levels) - 1)
        );
        assert_eq!(
            inventory.frames[0].upsampling,
            if name.contains("resampled") { 2 } else { 1 }
        );
        assert_eq!(
            inventory.frames[0].extra_channel_upsampling,
            vec![if name.contains("resampled") { 8 } else { 1 }; 2]
        );
        let options = &["--preserve-alpha", "--keep-orientation"];
        let native = planes(&oracle::libjxl_output(&encoded, options)?);
        let mut previews: Vec<_> = (1..=levels)
            .rev()
            .map(|level| expected_from(&directory(), name, level, &image))
            .collect();
        if composed {
            let background = planes(&oracle::libjxl_output(
                &fixture(name, ".background"),
                options,
            )?);
            let mut foreground = planes(&oracle::libjxl_output(&fixture(name, ""), options)?);
            let associated = matches!(
                image.extra_channels[0].channel_type,
                ExtraChannelTypeInventory::Alpha { associated: true }
            );
            blend(&mut foreground, &background, associated);
            for c in 0..5 {
                assert_error(
                    &native[c].iter().map(|&v| v as f32).collect::<Vec<_>>(),
                    &foreground[c],
                    3e-6,
                    &format!("{name} scalar final channel={c}"),
                    false,
                );
            }
            for preview in &mut previews {
                blend(preview, &background, associated);
            }
        }
        Some(Self {
            name: format!("{name} composed={composed}"),
            encoded,
            image,
            previews,
            native,
        })
    }

    fn values(
        &self,
        planes: &[Vec<f64>; 5],
        extra: Option<u32>,
        orientation: OrientationPolicy,
        alpha: AlphaOutputPolicy,
    ) -> Vec<f64> {
        let associated = matches!(
            self.image.extra_channels[0].channel_type,
            ExtraChannelTypeInventory::Alpha { associated: true }
        );
        let pixels = self.image.width as usize * self.image.height as usize;
        let rgba = (0..pixels)
            .flat_map(|i| {
                (0..4).map(move |c| {
                    if let Some(extra) = extra {
                        return planes[3 + extra as usize][i];
                    }
                    let value = planes[c][i];
                    if c == 3 {
                        return value;
                    }
                    match (alpha, associated) {
                        (AlphaOutputPolicy::Unassociated, true) => {
                            value / planes[3][i].max(2_f64.powi(-26))
                        }
                        (AlphaOutputPolicy::Associated, false) => {
                            value * planes[3][i].max(2_f64.powi(-26))
                        }
                        _ => value,
                    }
                })
            })
            .collect::<Vec<_>>();
        let oriented = crate::common::composed_oracle::orient(
            &rgba,
            self.image.width as usize,
            self.image.height as usize,
            if orientation == OrientationPolicy::Apply {
                self.image.orientation
            } else {
                1
            },
        );
        if extra.is_some() {
            oriented.into_iter().step_by(4).collect()
        } else {
            oriented
        }
    }

    fn run(
        &self,
        backend: &WgpuBackend,
        bounded: bool,
        orientation: OrientationPolicy,
        extra: Option<u32>,
        alpha: AlphaOutputPolicy,
    ) -> Vec<Vec<f32>> {
        let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        if bounded {
            engine = engine.with_stream_window_limit(NonZeroU64::new(256).unwrap());
        }
        let decoder = GpuDecoder::new(engine);
        let request = request(&self.image, extra)
            .with_alpha_output_policy(alpha)
            .with_orientation_policy(orientation)
            .with_max_frame_slots(NonZeroUsize::new(1).unwrap());
        let label = format!(
            "{} bounded={bounded} {orientation:?} {alpha:?} extra={extra:?}",
            self.name
        );
        let mut final_only = decoder.open(&self.encoded, request.clone()).unwrap();
        let final_frame = final_only.next_frame().unwrap().unwrap();
        let final_bytes = read_output(backend, &final_frame.output().outputs[0]);
        self.check_output(
            &oracle::floats(&final_bytes),
            &self.values(&self.native, extra, orientation, alpha),
            extra,
            alpha,
            &format!("{label} final"),
        );
        drop(final_frame);
        drop(final_only);
        let request = request.with_progressive_output(true);
        let mut session = if bounded {
            incremental(&decoder, &self.encoded, request)
        } else {
            decoder.open(&self.encoded, request).unwrap()
        };
        let mut returned = 0;
        let mut complete = 0;
        let mut held = Vec::new();
        while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
            let output = &update.output().outputs[0];
            let bytes = read_output(backend, output);
            if let Some(FrameProgression::LowFrequency {
                physical_frame_index,
                level,
            }) = update.progression()
            {
                assert_eq!(physical_frame_index as usize, returned);
                assert_eq!(level as usize, self.previews.len() - returned);
                self.check_output(
                    &oracle::floats(&bytes),
                    &self.values(&self.previews[returned], extra, orientation, alpha),
                    extra,
                    alpha,
                    &format!("{label} LF{level}"),
                );
                held.push((
                    jxl_wgpu::GpuImageOutput {
                        id: output.id,
                        layout: output.layout.clone(),
                        buffer: output.buffer.clone(),
                    },
                    bytes,
                ));
                returned += 1;
            } else if update.is_complete() {
                assert_eq!(bytes, final_bytes, "{label}: final-only output changed");
                complete += 1;
            }
        }
        assert_eq!(returned, self.previews.len());
        assert_eq!(complete, 1);
        drop(session);
        retired(backend);
        assert_eq!(
            decoder.engine().in_flight_memory_stats().reserved_bytes,
            held.iter()
                .map(|(output, _)| output.buffer.size())
                .sum::<u64>()
        );
        for (output, bytes) in &held {
            assert_eq!(read_output(backend, output), *bytes);
        }
        let result = std::iter::once(oracle::floats(&final_bytes))
            .chain(held.iter().map(|(_, bytes)| oracle::floats(bytes)))
            .collect();
        drop(held);
        retired(backend);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        result
    }

    fn check_output(
        &self,
        actual: &[f32],
        expected: &[f64],
        extra: Option<u32>,
        alpha: AlphaOutputPolicy,
        label: &str,
    ) {
        let tolerance = if extra.is_some() { 3e-6 } else { 5e-4 };
        if extra.is_none()
            && alpha == AlphaOutputPolicy::Unassociated
            && matches!(
                self.image.extra_channels[0].channel_type,
                ExtraChannelTypeInventory::Alpha { associated: true }
            )
        {
            // Unpremultiplication magnifies the native/GPU IDCT difference near zero alpha.
            // Measure reconstruction in its associated domain; the separate policy check
            // checks delivered straight values against the independently validated GPU source.
            let weight = |i: usize| {
                if i % 4 == 3 {
                    1.0
                } else {
                    expected[i / 4 * 4 + 3].max(2_f64.powi(-26))
                }
            };
            let weighted = actual
                .iter()
                .enumerate()
                .map(|(i, &v)| (f64::from(v) * weight(i)) as f32)
                .collect::<Vec<_>>();
            let expected = expected
                .iter()
                .enumerate()
                .map(|(i, &v)| v * weight(i))
                .collect::<Vec<_>>();
            assert_error(
                &weighted,
                &expected,
                tolerance,
                &format!("{label} associated-domain"),
                false,
            );
        } else {
            assert_error(actual, expected, tolerance, label, false);
        }
    }

    fn check_policy(
        &self,
        preserved: &[Vec<f32>],
        converted: &[Vec<f32>],
        policy: AlphaOutputPolicy,
    ) {
        assert_eq!(preserved.len(), converted.len());
        let associated = matches!(
            self.image.extra_channels[0].channel_type,
            ExtraChannelTypeInventory::Alpha { associated: true }
        );
        for (index, (source, output)) in preserved.iter().zip(converted).enumerate() {
            let expected = source
                .iter()
                .enumerate()
                .map(|(i, &value)| {
                    let value = f64::from(value);
                    if i % 4 == 3 {
                        return value;
                    }
                    let alpha = f64::from(source[i / 4 * 4 + 3]).max(2_f64.powi(-26));
                    match (policy, associated) {
                        (AlphaOutputPolicy::Unassociated, true) => value / alpha,
                        (AlphaOutputPolicy::Associated, false) => value * alpha,
                        _ => value,
                    }
                })
                .collect::<Vec<_>>();
            assert_error(
                output,
                &expected,
                3e-6,
                &format!("{} policy={policy:?} output={index}", self.name),
                true,
            );
        }
    }
}

#[test]
fn lf_precision_association_resampling_and_all_dependency_levels_match_native_producers() {
    let Some(backend) = backend() else { return };
    for (name, levels) in cases() {
        let Some(check) = Check::new(&name, levels, false) else {
            return;
        };
        for bounded in [false, true] {
            for extra in [None, Some(0), Some(1)] {
                check.run(
                    &backend,
                    bounded,
                    OrientationPolicy::Keep,
                    extra,
                    AlphaOutputPolicy::Preserve,
                );
            }
        }
    }
}

#[test]
fn lf_composition_orientation_and_alpha_policies_match_independent_layers() {
    let Some(backend) = backend() else { return };
    for (name, levels) in cases() {
        let Some(check) = Check::new(&name, levels, true) else {
            return;
        };
        for bounded in [false, true] {
            let orientation = if bounded {
                OrientationPolicy::Apply
            } else {
                OrientationPolicy::Keep
            };
            let mut preserved = Vec::new();
            for extra in [None, Some(0), Some(1)] {
                let result = check.run(
                    &backend,
                    bounded,
                    orientation,
                    extra,
                    AlphaOutputPolicy::Preserve,
                );
                if extra.is_none() {
                    preserved = result;
                }
            }
            for alpha in [
                AlphaOutputPolicy::Unassociated,
                AlphaOutputPolicy::Associated,
            ] {
                let result = check.run(&backend, bounded, orientation, None, alpha);
                check.check_policy(&preserved, &result, alpha);
            }
        }
        let Some(check) = Check::new(&name, levels, false) else {
            return;
        };
        for extra in [None, Some(0), Some(1)] {
            check.run(
                &backend,
                true,
                OrientationPolicy::Apply,
                extra,
                AlphaOutputPolicy::Preserve,
            );
        }
    }
}
