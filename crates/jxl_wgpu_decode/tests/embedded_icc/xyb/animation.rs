use super::{corpus, inventory, output};
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder};
use jxl_gpu_protocol::icc::{IccLimits, IccProfile, IccRenderingIntent};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuOutputRequest, OrientationPolicy};
use std::num::{NonZeroU64, NonZeroUsize};

#[test]
fn icc_xyb_saved_references_blend_in_original_device_values_and_preserve_above_one() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let directory = corpus::directory()
        .parent()
        .unwrap()
        .join("embedded_icc_xyb/animation");
    for case in corpus::cases().filter(|case| case.xyb) {
        let name = case.name().strip_suffix("_xyb").unwrap().to_owned();
        let data = std::fs::read(directory.join(format!("{name}.jxl"))).unwrap();
        let parsed = inventory(&data);
        assert!(parsed.image_header.xyb_encoded);
        assert_eq!(parsed.frames.len(), 2);
        assert_eq!(parsed.frames[0].save_as_reference, 1);
        assert!(!parsed.frames[0].save_before_color_transform);
        assert!(parsed.frames.iter().all(|f| f.encoding == case.encoding));
        assert_eq!(
            parsed
                .image_header
                .embedded_icc
                .as_ref()
                .unwrap()
                .profile
                .as_ref(),
            case.profile()
        );
        let colors = if case.gray { 1 } else { 3 };
        let reference = |frame: &str, kind: &str| {
            let bytes =
                std::fs::read(directory.join(format!("{name}.{frame}.device.{kind}.f32le")))
                    .unwrap();
            let (words, tail) = bytes.as_chunks::<4>();
            assert!(tail.is_empty());
            let words: Vec<_> = words.iter().copied().map(u32::from_le_bytes).collect();
            assert_eq!(words.len(), 153 * colors);
            words
                .chunks_exact(colors)
                .flat_map(|pixel| pixel.iter().copied().chain([1.0_f32.to_bits()]))
                .collect::<Vec<_>>()
        };
        let expected = ["frame0", "composed"].map(|frame| reference(frame, "scalar"));
        let lower = ["frame0", "composed"].map(|frame| reference(frame, "lower"));
        let upper = ["frame0", "composed"].map(|frame| reference(frame, "upper"));
        assert!(expected[1].iter().any(|&word| f32::from_bits(word) > 1.1));
        let profile = IccProfile::parse(case.profile().into(), IccLimits::default()).unwrap();
        for planar in [false, true] {
            let format = if case.gray {
                PixelFormat::gray_f32(true, planar, ColorSpecification::Icc(profile.clone()))
            } else {
                PixelFormat::rgb_f32(
                    RgbChannelOrder::Rgba,
                    planar,
                    ColorSpecification::Icc(profile.clone()),
                )
            };
            let mut baseline = None;
            for intent in [IccRenderingIntent::Relative, IccRenderingIntent::Perceptual] {
                for limit in [None, NonZeroU64::new(256)] {
                    let request = GpuOutputRequest::color(format.clone())
                        .unwrap()
                        .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                        .with_orientation_policy(OrientationPolicy::Keep)
                        .with_icc_rendering_intent(intent)
                        .with_max_frame_slots(NonZeroUsize::new(3).unwrap());
                    let frames = output::frames(&backend, &data, request, limit);
                    assert_eq!(frames.len(), 2);
                    for (frame, (actual, expected)) in frames.iter().zip(&expected).enumerate() {
                        assert_eq!(actual.len(), expected.len());
                        for (index, (&expected, (&lower, &upper))) in expected
                            .iter()
                            .zip(lower[frame].iter().zip(&upper[frame]))
                            .enumerate()
                        {
                            let channel = index % (colors + 1);
                            let pixel = index / (colors + 1);
                            let actual = actual[if planar { channel * 153 + pixel } else { index }];
                            if channel == colors {
                                assert_eq!(actual, expected, "opaque synthetic alpha");
                            } else {
                                let actual = f32::from_bits(actual);
                                let lower = f32::from_bits(lower);
                                let upper = f32::from_bits(upper);
                                assert!(
                                    actual.is_finite() && actual >= lower && actual <= upper,
                                    "{name} frame {frame} sample {index} planar={planar} {intent:?} {limit:?}: {actual} outside {lower}..{upper}"
                                );
                            }
                        }
                    }
                    if let Some(baseline) = &baseline {
                        assert_eq!(&frames, baseline);
                    }
                    baseline = Some(frames);
                }
            }
        }
    }
}
