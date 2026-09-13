use super::{
    compare, corpus, device_format, device_request, donor, frames, inventory, numeric, profile,
};
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction};
use jxl_test_support::fixtures::{modular_ycbcr, original_color};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuOutputRequest, OrientationPolicy};
use std::num::NonZeroU64;

fn reference(name: &str, kind: &str, field: &str) -> Vec<f32> {
    std::fs::read_to_string(
        corpus::directory()
            .parent()
            .unwrap()
            .join("embedded_icc_ycbcr")
            .join(format!("{name}.{kind}.{field}.f32.hex")),
    )
    .unwrap()
    .split_whitespace()
    .map(|word| f32::from_bits(u32::from_str_radix(word, 16).unwrap()))
    .collect()
}

#[test]
fn converted_icc_ycbcr_matches_independent_intervals_after_reconstruction() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for name in [
        "sampling_123",
        "gray",
        "associated",
        "resampling_8",
        "vardct_ycbcr_bt709_srgb_still",
    ] {
        let (source, expected, gray, vardct) = if let Some(case) = modular_ycbcr::cases()
            .into_iter()
            .find(|case| case.name == name)
        {
            let (data, words) = numeric::sample_fixture("modular_ycbcr", name);
            case.validate(&inventory(&data));
            let pixels = (case.size[0] * case.size[1]) as usize;
            (
                data,
                words[..pixels * 4]
                    .iter()
                    .map(|word| f32::from_bits(*word))
                    .collect::<Vec<_>>(),
                case.grayscale,
                false,
            )
        } else {
            let case = original_color::cases()
                .into_iter()
                .find(|case| case.name == name)
                .unwrap();
            let data = case.bytes();
            case.validate(&inventory(&data));
            (data, case.reference(), false, true)
        };
        let data = profile::replace(&source, &donor(gray).bytes());
        let channels = if gray { 2 } else { 4 };
        let expected: Vec<_> = if gray {
            expected
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|pixel| [pixel[0], pixel[3]])
                .collect()
        } else {
            expected
        };
        let device = frames(
            &backend,
            &data,
            device_request(device_format(donor(gray))),
            None,
        );
        assert_eq!(device.len(), 1, "{name}");
        // Verify the exact source error contract used to propagate the independent ICC bounds.
        compare(
            &device[0],
            &expected,
            channels,
            |value| {
                if vardct {
                    (1.0 + value.abs()) / 1024.0
                } else {
                    2e-6
                }
            },
            |value| 2e-6 * if vardct { 1.0 + value.abs() } else { 1.0 },
            &format!("{name} device interval"),
        );
        for kind in ["linear", "srgb", "other"] {
            let (specification, output_gray) = if kind == "other" {
                (device_format(donor(!gray)).color_spec, !gray)
            } else {
                let mut color = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
                if kind == "linear"
                    && let ColorSpecification::Defined(ref mut color) = color
                {
                    color.transfer = TransferFunction::Linear;
                }
                (color, false)
            };
            let outputs = if output_gray { 2 } else { 4 };
            let center = reference(name, kind, "scalar");
            let lower = reference(name, kind, "lower");
            let upper = reference(name, kind, "upper");
            let pixels = expected.len() / channels;
            assert_eq!(center.len(), pixels * outputs, "{name} {kind}");
            assert_eq!(lower.len(), center.len());
            assert_eq!(upper.len(), center.len());
            for planar in [false, true] {
                let format = if output_gray {
                    PixelFormat::gray_f32(true, planar, specification.clone())
                } else {
                    PixelFormat::rgb_f32(RgbChannelOrder::Rgba, planar, specification.clone())
                };
                let request = GpuOutputRequest::color(format)
                    .unwrap()
                    .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                    .with_orientation_policy(OrientationPolicy::Keep);
                let mut whole = None;
                for limit in [None, NonZeroU64::new(256)] {
                    let decoded = frames(&backend, &data, request.clone(), limit);
                    assert_eq!(decoded.len(), 1, "{name} {kind}");
                    assert_eq!(decoded[0].len(), center.len());
                    let mut maximum = 0.0_f32;
                    for index in 0..center.len() {
                        let pixel = index / outputs;
                        let channel = index % outputs;
                        let stored = if planar {
                            channel * pixels + pixel
                        } else {
                            index
                        };
                        let word = decoded[0][stored];
                        let actual = f32::from_bits(word);
                        assert!(
                            center[index].is_finite()
                                && lower[index] <= center[index]
                                && center[index] <= upper[index]
                        );
                        assert!(
                            actual.is_finite() && actual >= lower[index] && actual <= upper[index],
                            "{name} {kind} planar={planar} {limit:?} {index}: {actual} outside [{}, {}], center {}",
                            lower[index],
                            upper[index],
                            center[index]
                        );
                        if channel == outputs - 1 {
                            assert_eq!(
                                word,
                                device[0][pixel * channels + channels - 1],
                                "{name} alpha"
                            );
                        }
                        maximum = maximum.max((actual - center[index]).abs());
                    }
                    eprintln!("{name} {kind} planar={planar} {limit:?}: maxAE={maximum}");
                    if let Some(whole) = &whole {
                        assert_eq!(&decoded, whole, "{name} {kind} fragmented");
                    }
                    whole = Some(decoded);
                }
            }
        }
    }
}
