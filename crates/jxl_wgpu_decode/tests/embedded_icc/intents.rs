use super::{color, corpus, output};
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder};
use jxl_gpu_protocol::icc::{IccProfile, IccRenderingIntent};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuOutputRequest};
use std::num::NonZeroU64;
use std::path::Path;

fn references(directory: &Path, name: &str) -> Vec<[f32; 6]> {
    let bytes = std::fs::read(directory.join(format!("decoder/{name}.reference"))).unwrap();
    let (records, tail) = bytes.as_chunks::<28>();
    assert!(tail.is_empty());
    records
        .iter()
        .map(|record| {
            let values = std::array::from_fn(|c| {
                f32::from_le_bytes(record[c * 4..c * 4 + 4].try_into().unwrap())
            });
            let [native, exact, lower, upper, native_lower, native_upper] = values;
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(lower <= exact && exact <= upper);
            assert!(native_lower <= native && native <= native_upper);
            assert_eq!(u32::from_le_bytes(record[24..].try_into().unwrap()), 0);
            values
        })
        .collect()
}

#[test]
fn requested_icc_intents_convert_original_and_xyb_pixels_with_white_and_black_adjustments() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../jxl_wgpu/test-data/icc/intents");
    let mut distinct_from_relative = [0; 4];
    let mut conversions = 0;
    for case in corpus::cases() {
        let data = case.bytes();
        let source = if case.xyb {
            let bytes = std::fs::read(
                corpus::directory()
                    .parent()
                    .unwrap()
                    .join("embedded_icc_xyb")
                    .join(format!("{}.linear.native.f32le", case.name())),
            )
            .unwrap();
            let (words, tail) = bytes.as_chunks::<4>();
            assert!(tail.is_empty());
            words.iter().copied().map(u32::from_le_bytes).collect()
        } else {
            color::reference(case, "native")
        };
        let source_colors = if case.gray { 1 } else { 3 };
        for target in ["rgb_2", "gray_2", "rgb_5", "gray_5", "rgb_8", "gray_8"] {
            let profile = IccProfile::parse(
                std::fs::read(directory.join(format!("{target}.icc")))
                    .unwrap()
                    .into(),
                Default::default(),
            )
            .unwrap();
            let gray = target.starts_with("gray_");
            let colors = if gray { 1 } else { 3 };
            let prefix = format!("{}_to_{target}", case.name());
            let relative = references(&directory, &format!("{prefix}_1"));
            for intent in [
                IccRenderingIntent::Perceptual,
                IccRenderingIntent::Relative,
                IccRenderingIntent::Saturation,
                IccRenderingIntent::Absolute,
            ] {
                let name = format!("{prefix}_{}", intent as u32);
                let reference = references(&directory, &name);
                assert_eq!(reference.len(), 153 * colors);
                distinct_from_relative[intent as usize] += reference
                    .iter()
                    .zip(&relative)
                    .filter(|(selected, relative)| {
                        selected[3] < relative[2] || relative[3] < selected[2]
                    })
                    .count();
                for planar in [false, true] {
                    let color = ColorSpecification::Icc(profile.clone());
                    let format = if gray {
                        PixelFormat::gray_f32(true, planar, color)
                    } else {
                        PixelFormat::rgb_f32(RgbChannelOrder::Rgba, planar, color)
                    };
                    for limit in [None, NonZeroU64::new(256)] {
                        let request = GpuOutputRequest::color(format.clone())
                            .unwrap()
                            .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                            .with_icc_rendering_intent(intent);
                        let frames = output::frames(&backend, &data, request, limit);
                        assert_eq!(frames.len(), 1);
                        let actual = &frames[0];
                        assert_eq!(actual.len(), 153 * (colors + 1));
                        for pixel in 0..153 {
                            let word = |channel| {
                                actual[if planar {
                                    channel * 153 + pixel
                                } else {
                                    pixel * (colors + 1) + channel
                                }]
                            };
                            assert_eq!(
                                word(colors),
                                source[pixel * (source_colors + 1) + source_colors],
                                "alpha {name}"
                            );
                            for c in 0..colors {
                                let value = f32::from_bits(word(c));
                                let expected = reference[pixel * colors + c];
                                assert!(
                                    value.is_finite()
                                        && expected[2] <= value
                                        && value <= expected[3],
                                    "{name} pixel {pixel} channel {c}, planar={planar} limit={limit:?}: {value} outside {expected:?}"
                                );
                            }
                        }
                        conversions += 1;
                    }
                }
            }
        }
    }
    assert_eq!(conversions, 768);
    assert_eq!(distinct_from_relative[1], 0);
    assert!(
        [0, 2, 3]
            .into_iter()
            .all(|i| distinct_from_relative[i] > 100)
    );
    eprintln!(
        "ICC intent decoder: {conversions} presentations; distinct components {distinct_from_relative:?}"
    );
}
