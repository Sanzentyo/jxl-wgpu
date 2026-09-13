use super::{corpus, inventory, output};
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction};
use jxl_gpu_protocol::WhitePointAdaptation;
use jxl_gpu_protocol::icc::{IccLimits, IccProfile, IccRenderingIntent};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuOutputRequest, OrientationPolicy};
use std::num::NonZeroU64;

mod animation;
mod progression;

fn reference(case: corpus::Case, name: &str) -> Vec<u32> {
    let bytes = std::fs::read(
        corpus::directory()
            .parent()
            .unwrap()
            .join("embedded_icc_xyb")
            .join(format!("{}.{name}.f32le", case.name())),
    )
    .unwrap();
    let (words, tail) = bytes.as_chunks::<4>();
    assert!(tail.is_empty());
    words.iter().copied().map(u32::from_le_bytes).collect()
}

fn compare(
    actual: &[u32],
    expected: &[u32],
    channels: usize,
    planar: bool,
    bound: f32,
    context: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    assert_eq!(expected.len(), 153 * channels);
    let mut maximum = 0.0_f32;
    for (index, &expected) in expected.iter().enumerate() {
        let channel = index % channels;
        let pixel = index / channels;
        let actual = actual[if planar { channel * 153 + pixel } else { index }];
        if channel == channels - 1 {
            assert_eq!(actual, expected, "{context}: alpha {pixel}");
        } else {
            let actual = f32::from_bits(actual);
            let expected = f32::from_bits(expected);
            assert!(actual.is_finite() && expected.is_finite());
            let error = (actual - expected).abs();
            assert!(
                error <= bound,
                "{context}: {index}: {actual} vs {expected}, error {error}"
            );
            maximum = maximum.max(error);
        }
    }
    eprintln!("{context}: maxAE={maximum}");
}

fn linear_reference(data: &[u8], gray: bool) -> Vec<u32> {
    let mut image = jxl_oxide::JxlImage::read_with_defaults(data).unwrap();
    image.set_render_spot_color(false);
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).unwrap();
    let pixels = render.image_all_channels();
    assert_eq!(pixels.channels(), 4);
    pixels
        .buf()
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|pixel| {
            let rgb = if gray {
                [(f64::from(pixel[0]) * 0.2126
                    + f64::from(pixel[1]) * 0.7152
                    + f64::from(pixel[2]) * 0.0722) as f32; 3]
            } else {
                [pixel[0], pixel[1], pixel[2]]
            };
            [rgb[0], rgb[1], rgb[2], pixel[3]].map(f32::to_bits)
        })
        .collect()
}

#[test]
fn icc_xyb_direct_linear_output_uses_the_native_basis_without_selecting_original_cms() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for case in corpus::cases().filter(|case| case.xyb) {
        let data = case.bytes();
        case.validate(&inventory(&data));
        let reference = reference(case, "linear.native");
        let native: Vec<_> = if case.gray {
            reference
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|pixel| [pixel[0], pixel[0], pixel[0], pixel[1]])
                .collect()
        } else {
            reference
        };
        let expected = linear_reference(&data, case.gray);
        // Match the original-color corpus's native XYB conformance bound. The primary
        // GPU comparison below keeps a tighter 2e-5 bound against independent jxl-oxide.
        for (&native, &expected) in native.iter().zip(&expected) {
            let native = f32::from_bits(native);
            let expected = f32::from_bits(expected);
            assert!((native - expected).abs() <= (1.0 + native.abs()) / 1024.0);
        }
        for planar in [false, true] {
            let mut color = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
            let ColorSpecification::Defined(ref mut specification) = color else {
                unreachable!()
            };
            specification.transfer = TransferFunction::Linear;
            let format = PixelFormat::rgb_f32(RgbChannelOrder::Rgba, planar, color);
            let mut baseline = None;
            for intent in [
                IccRenderingIntent::Relative,
                IccRenderingIntent::Perceptual,
                IccRenderingIntent::Absolute,
                IccRenderingIntent::Saturation,
            ] {
                for limit in [None, NonZeroU64::new(256)] {
                    let request = GpuOutputRequest::color(format.clone())
                        .unwrap()
                        .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                        .with_orientation_policy(OrientationPolicy::Keep)
                        .with_white_point_adaptation(WhitePointAdaptation::None)
                        .with_icc_rendering_intent(intent);
                    let frames = output::frames(&backend, &data, request, limit);
                    assert_eq!(frames.len(), 1);
                    compare(
                        &frames[0],
                        &expected,
                        4,
                        planar,
                        2e-5,
                        &format!(
                            "{} linear planar={planar} {intent:?} {limit:?}",
                            case.name()
                        ),
                    );
                    if let Some(baseline) = &baseline {
                        assert_eq!(&frames, baseline);
                    }
                    baseline = Some(frames);
                }
            }
        }
    }
}

#[test]
fn icc_xyb_requested_profiles_match_independent_device_values_and_plane_counts() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for case in corpus::cases().filter(|case| case.xyb) {
        let data = case.bytes();
        case.validate(&inventory(&data));
        for gray in [false, true] {
            let profile = corpus::cases()
                .find(|case| case.gray == gray)
                .unwrap()
                .profile();
            let profile = IccProfile::parse(profile.into(), IccLimits::default()).unwrap();
            let expected = reference(case, if gray { "gray.scalar" } else { "rgb.scalar" });
            for planar in [false, true] {
                let color = ColorSpecification::Icc(profile.clone());
                let format = if gray {
                    PixelFormat::gray_f32(true, planar, color)
                } else {
                    PixelFormat::rgb_f32(RgbChannelOrder::Rgba, planar, color)
                };
                let request = GpuOutputRequest::color(format)
                    .unwrap()
                    .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                    .with_orientation_policy(OrientationPolicy::Keep);
                let mut baseline = None;
                for limit in [None, NonZeroU64::new(256)] {
                    let frames = output::frames(&backend, &data, request.clone(), limit);
                    assert_eq!(frames.len(), 1);
                    // Retain the established embedded-ICC end-to-end F32 bound.
                    compare(
                        &frames[0],
                        &expected,
                        if gray { 2 } else { 4 },
                        planar,
                        2e-4,
                        &format!("{} to gray={gray} planar={planar} {limit:?}", case.name()),
                    );
                    if let Some(baseline) = &baseline {
                        assert_eq!(&frames, baseline);
                    }
                    baseline = Some(frames);
                }
            }
        }
    }
}
