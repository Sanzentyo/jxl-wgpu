#![cfg(not(target_arch = "wasm32"))]

use std::{
    num::{NonZeroU64, NonZeroUsize},
    ops::Range,
};

use jxl_gpu_formats::{Channel, ColorSpecification, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_gpu_protocol::Extent2d;
use jxl_test_support::gpu::planes;
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, NumericSampleMapping, SpotColorPolicy,
    WgpuDecodeEngine,
};

mod cases;
mod existing_families;
mod extended;
mod profiles;
mod reference;
use reference::{Reference, ReferenceColor};

fn decode(
    backend: &WgpuBackend,
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    reference: &Reference,
    request: GpuOutputRequest,
    fragmented: bool,
) -> Vec<u32> {
    assert_eq!(reference.descriptor.frames.len(), 1);
    decode_frames(backend, decoder, reference, request, fragmented).remove(0)
}

fn decode_frames(
    backend: &WgpuBackend,
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    reference: &Reference,
    request: GpuOutputRequest,
    fragmented: bool,
) -> Vec<Vec<u32>> {
    let expected_frames = reference.descriptor.frames.len();
    let request = request.with_max_frame_slots(NonZeroUsize::new(expected_frames).unwrap());
    let mut session = if fragmented {
        planes::open_fragmented(decoder, &reference.input, request)
    } else {
        decoder.open(&reference.input, request).unwrap()
    };
    let metadata = session.metadata();
    if let Some(animation) = reference.animation {
        let timebase = metadata.timebase.unwrap();
        assert_eq!(
            timebase.ticks_per_second_numerator.get(),
            animation.ticks_per_second_numerator
        );
        assert_eq!(
            timebase.ticks_per_second_denominator.get(),
            animation.ticks_per_second_denominator
        );
        assert_eq!(metadata.loop_count, Some(animation.num_loops));
        assert_eq!(metadata.has_timecodes, Some(animation.have_timecodes));
    } else {
        assert!(metadata.timebase.is_none() && metadata.loop_count.is_none());
    }
    let mut frames = Vec::new();
    let mut ticks = 0;
    while let Some(frame) = pollster::block_on(session.next_frame_async()).unwrap() {
        let index = frames.len();
        assert!(index < expected_frames);
        let descriptor = &reference.descriptor.frames[index];
        assert_eq!(frame.metadata.index, index);
        assert_eq!(frame.metadata.name, descriptor.name);
        assert_eq!(frame.metadata.presentation_ticks, ticks);
        ticks += u64::from(frame.metadata.duration.ticks);
        if let Some(seconds) = descriptor.duration {
            assert_eq!(frame.metadata.duration.as_seconds(), seconds);
        } else {
            assert_eq!(
                frame.metadata.duration,
                jxl_wgpu_decode::FrameDuration::still()
            );
        }
        assert_eq!(frame.metadata.is_last, index + 1 == expected_frames);
        assert_eq!(frame.output().outputs.len(), 1);
        assert_eq!(
            frame.output().outputs[0].layout.extent,
            Extent2d::new(reference.width as u32, reference.height as u32)
        );
        frames.push(frame);
    }
    assert_eq!(frames.len(), expected_frames);
    drop(session);
    assert!(backend.transient_memory_stats().reserved_bytes > 0);
    let words = frames
        .iter()
        .map(|frame| planes::read(backend, &frame.output().outputs[0]))
        .collect();
    drop(frames);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
    words
}

fn compare(
    reference: &Reference,
    words: &[u32],
    channels: Range<usize>,
    window: Option<NonZeroU64>,
    frame: usize,
) {
    let count = reference.width * reference.height;
    assert_eq!(words.len(), count * channels.len());
    let mut squared = vec![0.0; channels.len()];
    let mut peaks = vec![0.0_f64; channels.len()];
    let bounds = &reference.descriptor.frames[frame];
    for (pixel, actual) in words.chunks_exact(channels.len()).enumerate() {
        for (lane, channel) in channels.clone().enumerate() {
            let expected = reference.pixels[(frame * count + pixel) * reference.channels + channel];
            let value = f32::from_bits(actual[lane]);
            assert!(value.is_finite() && expected.is_finite());
            if bounds.rms_error == 0.0 && bounds.peak_error == 0.0 {
                assert_eq!(
                    actual[lane],
                    expected.to_bits(),
                    "{} pixel {pixel} channel {channel}",
                    reference.name
                );
            }
            let delta = f64::from(value) - f64::from(expected);
            squared[lane] += delta * delta;
            peaks[lane] = peaks[lane].max(delta.abs());
        }
    }
    let rms: Vec<_> = squared
        .iter()
        .map(|sum| (sum / count as f64).sqrt())
        .collect();
    eprintln!(
        "{} frame {frame} window {window:?} channels {channels:?}: RMSE {rms:?}, peak {peaks:?}",
        reference.name
    );
    assert!(
        rms.iter().all(|&error| error <= bounds.rms_error),
        "{} RMSE {rms:?}",
        reference.name
    );
    assert!(
        peaks.iter().all(|&error| error <= bounds.peak_error),
        "{} peak {peaks:?}",
        reference.name
    );
}

fn run_case(name: &str) {
    let case = cases::CASES.iter().find(|case| case.name == name).unwrap();
    let reference = case.load();
    let alternate = cases::ALTERNATES
        .iter()
        .find(|a| a.primary == name)
        .map(|a| case.load_alternate(a));
    let mut config = jxl_wgpu::WgpuBackendConfig::default();
    config.memory.max_transient_bytes = 2 * 1024 * 1024 * 1024;
    config.memory.max_in_flight_transient_bytes = 2 * 1024 * 1024 * 1024;
    let backend = pollster::block_on(WgpuBackend::request_default(config)).unwrap();
    eprintln!("official conformance {}", case.name);
    let original_components = matches!(reference.color, ReferenceColor::OriginalNumeric);
    let has_alpha = reference.channels > reference.colors;
    let color_components = reference.colors + usize::from(has_alpha);
    let mut specification = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
    match reference.color {
        ReferenceColor::Transfer(transfer) => {
            let ColorSpecification::Defined(ref mut color) = specification else {
                unreachable!()
            };
            color.transfer = transfer;
        }
        ReferenceColor::Profile => {
            specification = ColorSpecification::Icc(
                jxl_gpu_protocol::icc::IccProfile::parse(
                    reference.profile.clone().into(),
                    Default::default(),
                )
                .unwrap(),
            );
        }
        ReferenceColor::OriginalNumeric => {}
    }
    let format = if reference.colors == 1 {
        PixelFormat::gray_f32(has_alpha, false, specification)
    } else {
        PixelFormat::rgb_f32(
            if has_alpha {
                RgbChannelOrder::Rgba
            } else {
                RgbChannelOrder::Rgb
            },
            false,
            specification,
        )
    };
    let mut whole = Vec::new();
    for window in [None, NonZeroU64::new(16 * 1024)] {
        let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        if let Some(limit) = window {
            engine = engine.with_stream_window_limit(limit);
        }
        let decoder = GpuDecoder::new(engine);
        // The official runner preserves associated alpha and disables spot rendering.
        let request = GpuOutputRequest::color(format.clone())
            .unwrap()
            .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
            .with_spot_color_policy(SpotColorPolicy::Preserve);
        let mut outputs = Vec::new();
        if !original_components {
            outputs.push(decode_frames(
                &backend,
                &decoder,
                &reference,
                request,
                window.is_some(),
            ));
            for (frame, words) in outputs[0].iter().enumerate() {
                compare(&reference, words, 0..color_components, window, frame);
                if let Some(alternate) = &alternate {
                    compare(alternate, words, 0..color_components, window, frame);
                }
            }
        }
        let first_component = if original_components {
            0
        } else {
            color_components
        };
        for channel in first_component..reference.channels {
            let depth = if channel < reference.colors {
                reference.depths[0]
            } else {
                reference.depths[1 + channel - reference.colors]
            };
            let request = GpuOutputRequest::numeric(
                PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                if matches!(depth, jxl_gpu_bitstream::SampleBitDepth::Float { .. }) {
                    NumericSampleMapping::NativeFloat
                } else {
                    NumericSampleMapping::NormalizedUnsigned
                },
            )
            .unwrap();
            let request = if channel < reference.colors {
                request.with_color_channel(channel as u32)
            } else {
                request.with_extra_channel((channel - reference.colors) as u32)
            }
            .unwrap();
            let words = decode_frames(&backend, &decoder, &reference, request, window.is_some());
            for (frame, words) in words.iter().enumerate() {
                compare(&reference, words, channel..channel + 1, window, frame);
                if let Some(alternate) = &alternate {
                    compare(alternate, words, channel..channel + 1, window, frame);
                }
            }
            outputs.push(words);
        }
        if window.is_none() {
            whole = outputs;
        } else {
            assert_eq!(outputs, whole, "{} bounded output", reference.name);
        }
    }
}

#[test]
fn lossless_pfm() {
    run_case("lossless_pfm");
}

#[test]
fn alpha_nonpremultiplied() {
    run_case("alpha_nonpremultiplied");
}

#[test]
fn alpha_premultiplied() {
    run_case("alpha_premultiplied");
}

#[test]
fn alpha_triangles() {
    run_case("alpha_triangles");
}

#[test]
fn spot() {
    run_case("spot");
}

fn invalid_profile_is_rejected_before_gpu_admission(
    name: &str,
    expected_field: &'static str,
    expected_offset: u64,
) {
    use jxl_gpu_protocol::icc::{IccError, IccProfile};
    let reference = cases::CASES
        .iter()
        .find(|case| case.name == name)
        .unwrap()
        .load();
    assert!(matches!(
        IccProfile::parse(reference.profile.into(), Default::default()),
        Err(IccError::Invalid { field, offset })
            if field == expected_field && offset == expected_offset
    ));
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let memory = backend.transient_memory_budget();
    let held = memory
        .try_reserve(memory.snapshot().available_bytes)
        .unwrap();
    let before = memory.snapshot().reserved_bytes;
    for _ in 0..2 {
        let request = GpuOutputRequest::color(jxl_wgpu_decode::vardct_rgb8_format()).unwrap();
        assert!(matches!(
            decoder.open(&reference.input, request),
            Err(jxl_wgpu_decode::Error::Icc(IccError::Invalid { field, offset }))
                if field == expected_field && offset == expected_offset
        ));
        assert_eq!(memory.snapshot().reserved_bytes, before);
        assert_eq!(
            decoder.engine().in_flight_memory_stats().reserved_bytes,
            before
        );
    }
    drop(held);
    assert_eq!(memory.snapshot().reserved_bytes, 0);
}

#[test]
fn spot_color_requires_a_valid_profile_before_gpu_admission() {
    invalid_profile_is_rejected_before_gpu_admission("spot", "PCS D50 illuminant", 68);
}

#[test]
fn bench_color_requires_a_valid_profile_before_gpu_admission() {
    invalid_profile_is_rejected_before_gpu_admission("bench_oriented_brg", "reserved header", 84);
}

#[test]
fn sunset_logo() {
    run_case("sunset_logo");
}

#[test]
fn blendmodes() {
    run_case("blendmodes");
}

#[test]
fn delta_palette() {
    run_case("delta_palette");
}

#[test]
fn grayscale() {
    run_case("grayscale");
}

#[test]
fn grayscale_jpeg_pixels() {
    run_case("grayscale_jpeg");
}

#[test]
fn lz77_flower() {
    run_case("lz77_flower");
}

#[test]
fn patches_lossless() {
    run_case("patches_lossless");
}

#[test]
fn animation_icos4d() {
    run_case("animation_icos4d");
}

#[test]
fn animation_newtons_cradle() {
    run_case("animation_newtons_cradle");
}

#[test]
fn bench_oriented_brg() {
    run_case("bench_oriented_brg");
}

#[test]
fn bicycles() {
    run_case("bicycles");
}

#[test]
fn bike() {
    run_case("bike");
}

#[test]
fn cafe() {
    run_case("cafe");
}

#[test]
fn grayscale_public_university() {
    run_case("grayscale_public_university");
}

#[test]
fn noise() {
    run_case("noise");
}

#[test]
fn opsin_inverse() {
    run_case("opsin_inverse");
}

#[test]
fn patches() {
    run_case("patches");
}

#[test]
fn progressive() {
    run_case("progressive");
}

#[test]
fn upsampling() {
    run_case("upsampling");
}

#[test]
fn mul_no_extra_channels() {
    run_case("mul_no_extra_channels");
}
