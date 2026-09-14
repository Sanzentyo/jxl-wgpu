use super::{color, corpus, output, references};
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder};
use jxl_gpu_protocol::icc::{IccProfile, IccRenderingIntent};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuOutputRequest};
use std::{num::NonZeroU64, path::Path};

#[test]
fn decoded_original_icc_pixels_execute_requested_mpe_stages_and_preserve_alpha() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../jxl_wgpu/test-data/icc/mpe");
    let mut presentations = 0;
    let mut distinct = [0; 4];
    for case in corpus::cases().filter(|case| !case.xyb) {
        let data = case.bytes();
        let native = color::reference(case, "native");
        let colors = if case.gray { 1 } else { 3 };
        for target in ["segments", "clut4", "lab"] {
            let profile = IccProfile::parse(
                std::fs::read(directory.join(format!("{target}.icc")))
                    .unwrap()
                    .into(),
                Default::default(),
            )
            .unwrap();
            let prefix = format!("{}_to_{target}", case.name());
            let relative = references::read(&directory, &format!("{prefix}_1"));
            for intent in [
                IccRenderingIntent::Perceptual,
                IccRenderingIntent::Relative,
                IccRenderingIntent::Saturation,
                IccRenderingIntent::Absolute,
            ] {
                let expected = references::read(&directory, &format!("{prefix}_{}", intent as u32));
                assert_eq!(expected.len(), 153 * 3);
                distinct[intent as usize] += expected
                    .iter()
                    .zip(&relative)
                    .filter(|(a, b)| a[3] < b[2] || b[3] < a[2])
                    .count();
                for planar in [false, true] {
                    let format = PixelFormat::rgb_f32(
                        RgbChannelOrder::Rgba,
                        planar,
                        ColorSpecification::Icc(profile.clone()),
                    );
                    for limit in [None, NonZeroU64::new(256)] {
                        let request = GpuOutputRequest::color(format.clone())
                            .unwrap()
                            .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                            .with_icc_rendering_intent(intent);
                        let frames = output::frames(&backend, &data, request, limit);
                        assert_eq!(frames.len(), 1);
                        let pixels = &frames[0];
                        assert_eq!(pixels.len(), 153 * 4);
                        for pixel in 0..153 {
                            let word = |channel| {
                                pixels[if planar {
                                    channel * 153 + pixel
                                } else {
                                    pixel * 4 + channel
                                }]
                            };
                            assert_eq!(
                                word(3),
                                native[pixel * (colors + 1) + colors],
                                "alpha {prefix}"
                            );
                            for c in 0..3 {
                                let actual = f32::from_bits(word(c));
                                let reference = expected[pixel * 3 + c];
                                assert!(
                                    actual.is_finite()
                                        && actual >= reference[2]
                                        && actual <= reference[3],
                                    "{prefix}, {intent:?} pixel {pixel} channel {c}, planar {planar} window {limit:?}: {actual} outside {reference:?}"
                                );
                            }
                        }
                        presentations += 1;
                    }
                }
            }
        }
    }
    assert_eq!(presentations, 192);
    assert_eq!(distinct[1], 0);
    assert!([0, 2, 3].into_iter().all(|i| distinct[i] > 100));
    eprintln!("MPE decoder {presentations} presentations, distinct from relative {distinct:?}");
}
