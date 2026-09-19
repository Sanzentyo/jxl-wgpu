#![cfg(not(target_arch = "wasm32"))]

use std::{num::NonZeroU64, ops::Range};

use jxl_gpu_formats::{Channel, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_gpu_protocol::Extent2d;
use jxl_test_support::gpu::planes;
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, NumericSampleMapping, SpotColorPolicy,
    WgpuDecodeEngine,
};

mod cases;
mod extended;
mod reference;
use reference::Reference;

fn decode(
    backend: &WgpuBackend,
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    reference: &Reference,
    request: GpuOutputRequest,
    fragmented: bool,
) -> Vec<u32> {
    let mut session = if fragmented {
        planes::open_fragmented(decoder, &reference.input, request)
    } else {
        decoder.open(&reference.input, request).unwrap()
    };
    let frame = pollster::block_on(session.next_frame_async())
        .unwrap()
        .unwrap();
    assert_eq!(frame.metadata.index, 0);
    assert_eq!(frame.metadata.name, reference.descriptor.frames[0].name);
    assert_eq!(frame.metadata.presentation_ticks, 0);
    assert!(frame.metadata.is_last);
    assert_eq!(frame.output().outputs.len(), 1);
    assert_eq!(
        frame.output().outputs[0].layout.extent,
        Extent2d::new(reference.width as u32, reference.height as u32)
    );
    assert!(
        pollster::block_on(session.next_frame_async())
            .unwrap()
            .is_none()
    );
    drop(session);
    assert!(backend.transient_memory_stats().reserved_bytes > 0);
    let words = planes::read(backend, &frame.output().outputs[0]);
    drop(frame);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
    words
}

fn compare(
    reference: &Reference,
    words: &[u32],
    channels: Range<usize>,
    window: Option<NonZeroU64>,
) {
    let count = reference.width * reference.height;
    assert_eq!(words.len(), count * channels.len());
    let mut squared = vec![0.0; channels.len()];
    let mut peaks = vec![0.0_f64; channels.len()];
    let bounds = &reference.descriptor.frames[0];
    for (pixel, actual) in words.chunks_exact(channels.len()).enumerate() {
        for (lane, channel) in channels.clone().enumerate() {
            let expected = reference.pixels[pixel * reference.channels + channel];
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
        "{} window {window:?} channels {channels:?}: RMSE {rms:?}, peak {peaks:?}",
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
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    eprintln!("official still {}", case.name);
    // An original-profile reference describes component values, not a requested CMS transform.
    // In particular, spot's untouched v2 profile has a noncanonical PCS illuminant.
    let original_components = reference.descriptor.original_icc.is_some();
    let has_alpha = reference.channels > 3;
    let color_components = if has_alpha { 4 } else { 3 };
    let format = PixelFormat::rgb_f32(
        if has_alpha {
            RgbChannelOrder::Rgba
        } else {
            RgbChannelOrder::Rgb
        },
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    );
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
            outputs.push(decode(
                &backend,
                &decoder,
                &reference,
                request,
                window.is_some(),
            ));
            compare(&reference, &outputs[0], 0..color_components, window);
        }
        let first_component = if original_components {
            0
        } else {
            color_components
        };
        for channel in first_component..reference.channels {
            let request = GpuOutputRequest::numeric(
                PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                NumericSampleMapping::NormalizedUnsigned,
            )
            .unwrap();
            let request = if channel < 3 {
                request.with_color_channel(channel as u32)
            } else {
                request.with_extra_channel((channel - 3) as u32)
            }
            .unwrap();
            let words = decode(&backend, &decoder, &reference, request, window.is_some());
            compare(&reference, &words, channel..channel + 1, window);
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

#[test]
fn spot_color_requires_a_valid_profile_before_gpu_admission() {
    use jxl_gpu_protocol::icc::{IccError, IccProfile};
    let reference = cases::CASES
        .iter()
        .find(|case| case.name == "spot")
        .unwrap()
        .load();
    assert!(matches!(
        IccProfile::parse(reference.profile.into(), Default::default()),
        Err(IccError::Invalid {
            field: "PCS D50 illuminant",
            offset: 68
        })
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
            Err(jxl_wgpu_decode::Error::Icc(IccError::Invalid {
                field: "PCS D50 illuminant",
                offset: 68
            }))
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
fn sunset_logo() {
    run_case("sunset_logo");
}
