#![cfg(not(target_arch = "wasm32"))]

use std::num::NonZeroU64;

use jxl_gpu_formats::{Channel, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_test_support::{fixtures::modular_ycbcr, gpu::planes, offline::hex};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, NumericSampleMapping, OrientationPolicy,
    WgpuDecodeEngine,
};

mod admission;
mod progression;

fn color_request() -> GpuOutputRequest {
    GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
    .with_orientation_policy(OrientationPolicy::Keep)
}

fn reference(name: &str) -> (Vec<u8>, Vec<f32>) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/modular_ycbcr");
    let bytes = hex::unhex(&std::fs::read_to_string(root.join(format!("{name}.jxl.hex"))).unwrap());
    (bytes, samples(name))
}

fn samples(name: &str) -> Vec<f32> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/modular_ycbcr");
    std::fs::read_to_string(root.join(format!("{name}.f32.hex")))
        .unwrap()
        .split_whitespace()
        .map(|word| f32::from_bits(u32::from_str_radix(word, 16).unwrap()))
        .collect()
}

fn require_samples(name: &str, actual: &[u32], expected: impl Iterator<Item = f32>) {
    let expected: Vec<_> = expected.collect();
    assert_eq!(actual.len(), expected.len(), "{name}");
    for (index, (&word, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f32::from_bits(word);
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 2e-6,
            "{name} at {index}: {actual} != {expected}"
        );
    }
}

fn local_wide_samples(case: &modular_ycbcr::Case) -> bool {
    case.has_local_transforms()
        && match case.bit_depth {
            jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample } => bits_per_sample > 16,
            jxl_gpu_bitstream::SampleBitDepth::Float { .. } => true,
        }
}

fn decoder_with_limit(
    backend: &WgpuBackend,
    limit: Option<NonZeroU64>,
) -> GpuDecoder<WgpuDecodeEngine> {
    let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    if let Some(limit) = limit {
        engine = engine.with_stream_window_limit(limit);
    }
    GpuDecoder::new(engine)
}

#[test]
fn all_component_selectors_match_native_color_and_numeric_output() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoders = [
        decoder_with_limit(&backend, None),
        decoder_with_limit(&backend, NonZeroU64::new(40)),
        decoder_with_limit(&backend, None),
    ];
    for case in modular_ycbcr::cases()
        .into_iter()
        .filter(|case| !local_wide_samples(case))
    {
        require_case(&backend, &decoders, &case);
    }
}

#[test]
fn local_wide_sample_transforms_match_independent_color_and_numeric_references() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoders = [
        decoder_with_limit(&backend, None),
        decoder_with_limit(&backend, NonZeroU64::new(40)),
        decoder_with_limit(&backend, None),
    ];
    for case in modular_ycbcr::cases()
        .into_iter()
        .filter(local_wide_samples)
    {
        require_case(&backend, &decoders, &case);
    }
}

fn require_case(
    backend: &WgpuBackend,
    decoders: &[GpuDecoder<WgpuDecodeEngine>; 3],
    case: &modular_ycbcr::Case,
) {
    let (bytes, expected) = reference(&case.name);
    let info = jxl_gpu_bitstream::parse(&bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    case.validate(&info);
    let pixels = (case.size[0] * case.size[1]) as usize;
    let rgba = &expected[..pixels * 4];
    let request = color_request();
    let mut prior = None;
    for (decoder, fragmented) in decoders[..2].iter().zip([false, true]) {
        let mut session = if fragmented {
            planes::open_fragmented(decoder, &bytes, request.clone())
        } else {
            decoder.open(&bytes, request.clone()).unwrap()
        };
        let frame = pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap();
        let actual = planes::read(backend, &frame.output().outputs[0]);
        require_samples(&case.name, &actual, rgba.iter().copied());
        if let Some(prior) = &prior {
            assert_eq!(&actual, prior, "{} fragmented", case.name);
        }
        prior = Some(actual);
        drop((frame, session));
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
    let decoder = &decoders[2];
    for channel in 0..if case.grayscale { 1 } else { 3 } {
        let request = GpuOutputRequest::numeric(
            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
            mapping(case.bit_depth),
        )
        .unwrap()
        .with_color_channel(channel)
        .unwrap()
        .with_orientation_policy(OrientationPolicy::Keep);
        let mut session = decoder.open(&bytes, request).unwrap();
        let frame = session.next_frame().unwrap().unwrap();
        require_samples(
            &case.name,
            &planes::read(backend, &frame.output().outputs[0]),
            rgba.as_chunks::<4>()
                .0
                .iter()
                .map(|pixel| pixel[channel as usize]),
        );
        drop((frame, session));
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
    for (channel, extra) in info.image_header.extra_channels.iter().enumerate() {
        let request = GpuOutputRequest::numeric(
            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
            mapping(extra.bit_depth),
        )
        .unwrap()
        .with_extra_channel(channel as u32)
        .unwrap()
        .with_orientation_policy(OrientationPolicy::Keep);
        let mut session = planes::open_fragmented(decoder, &bytes, request);
        let frame = session.next_frame().unwrap().unwrap();
        require_samples(
            &case.name,
            &planes::read(backend, &frame.output().outputs[0]),
            expected[(4 + channel) * pixels..][..pixels].iter().copied(),
        );
        drop((frame, session));
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
    if case.orientation != 1 {
        let oriented = samples(&format!("{}.oriented", case.name));
        let mut session = decoder
            .open(
                &bytes,
                request.with_orientation_policy(OrientationPolicy::Apply),
            )
            .unwrap();
        let frame = session.next_frame().unwrap().unwrap();
        require_samples(
            &case.name,
            &planes::read(backend, &frame.output().outputs[0]),
            oriented[..pixels * 4].iter().copied(),
        );
        drop((frame, session));
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
    eprintln!(
        "{}: color, numeric components and 40-byte windows",
        case.name
    );
}

fn mapping(depth: jxl_gpu_bitstream::SampleBitDepth) -> NumericSampleMapping {
    match depth {
        jxl_gpu_bitstream::SampleBitDepth::Integer { .. } => {
            NumericSampleMapping::NormalizedUnsigned
        }
        jxl_gpu_bitstream::SampleBitDepth::Float { .. } => NumericSampleMapping::NativeFloat,
    }
}
