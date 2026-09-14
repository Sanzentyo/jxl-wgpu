use super::{corpus, inventory, output};
use jxl_gpu_bitstream::{ExtraChannelTypeInventory, FrameBlendMode, SampleBitDepth};
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder};
use jxl_gpu_protocol::icc::{IccLimits, IccProfile};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuOutputRequest, OrientationPolicy};
use std::num::{NonZeroU64, NonZeroUsize};

#[test]
fn icc_xyb_alpha_references_preserve_association_and_execute_every_blend_in_device_color() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let directory = corpus::directory()
        .parent()
        .unwrap()
        .join("embedded_icc_xyb/alpha");
    for case in corpus::cases().filter(|case| case.xyb) {
        let base = case.name().strip_suffix("_xyb").unwrap().to_owned();
        let profile = IccProfile::parse(case.profile().into(), IccLimits::default()).unwrap();
        for associated in [false, true] {
            for mode in [
                FrameBlendMode::Replace,
                FrameBlendMode::Add,
                FrameBlendMode::Blend,
                FrameBlendMode::MultiplyAdd,
                FrameBlendMode::Multiply,
            ] {
                for alpha_reference in [false, true] {
                    if alpha_reference && mode != FrameBlendMode::Blend {
                        continue;
                    }
                    let association = if associated { "associated" } else { "straight" };
                    let suffix = if alpha_reference { "_alpha_ref1" } else { "" };
                    let name = format!("{base}_{association}_m{}{suffix}", mode as u8);
                    let data = std::fs::read(directory.join(format!("{name}.jxl"))).unwrap();
                    let parsed = inventory(&data);
                    let image = &parsed.image_header;
                    assert_eq!((image.width, image.height), (17, 9));
                    assert_eq!(image.grayscale, case.gray);
                    assert!(image.xyb_encoded);
                    assert_eq!(
                        image.embedded_icc.as_ref().unwrap().profile.as_ref(),
                        case.profile()
                    );
                    assert_eq!(image.extra_channels.len(), 1);
                    assert_eq!(
                        image.extra_channels[0].channel_type,
                        ExtraChannelTypeInventory::Alpha { associated }
                    );
                    assert_eq!(
                        image.extra_channels[0].bit_depth,
                        SampleBitDepth::Float {
                            bits_per_sample: 32,
                            exponent_bits_per_sample: 8
                        }
                    );
                    assert_eq!(parsed.frames.len(), 2);
                    assert!(
                        parsed
                            .frames
                            .iter()
                            .all(|frame| frame.encoding == case.encoding && !frame.do_ycbcr)
                    );
                    assert_eq!(parsed.frames[0].save_as_reference, 1);
                    assert!(!parsed.frames[0].save_before_color_transform);
                    assert_eq!(parsed.frames[0].color_blend.mode, FrameBlendMode::Replace);
                    assert_eq!(parsed.frames[1].color_blend.mode, mode);
                    assert_eq!(
                        parsed.frames[1].color_blend.source,
                        u32::from(mode != FrameBlendMode::Replace)
                    );
                    for (index, frame) in parsed.frames.iter().enumerate() {
                        assert_eq!(frame.extra_channel_blends.len(), 1);
                        let blend = &frame.extra_channel_blends[0];
                        let retained = index == 1 && alpha_reference;
                        assert_eq!(
                            blend.mode,
                            if retained {
                                FrameBlendMode::Blend
                            } else {
                                FrameBlendMode::Replace
                            }
                        );
                        assert_eq!(blend.source, u32::from(retained));
                    }
                    if matches!(mode, FrameBlendMode::Blend | FrameBlendMode::MultiplyAdd) {
                        assert_eq!(parsed.frames[1].color_blend.alpha_channel, Some(0));
                        assert!(parsed.frames[1].color_blend.clamp);
                    }
                    let channels = if case.gray { 2 } else { 4 };
                    let reference = |frame: &str, kind: &str| {
                        let bytes = std::fs::read(
                            directory.join(format!("{name}.{frame}.device.{kind}.f32le")),
                        )
                        .unwrap();
                        let (words, tail) = bytes.as_chunks::<4>();
                        assert!(tail.is_empty());
                        assert_eq!(words.len(), 153 * channels);
                        words
                            .iter()
                            .copied()
                            .map(u32::from_le_bytes)
                            .collect::<Vec<_>>()
                    };
                    let expected = ["frame0", "composed"].map(|frame| reference(frame, "scalar"));
                    let lower = ["frame0", "composed"].map(|frame| reference(frame, "lower"));
                    let upper = ["frame0", "composed"].map(|frame| reference(frame, "upper"));
                    assert!(
                        expected[0]
                            .chunks_exact(channels)
                            .any(|p| p[channels - 1] == 0.0_f32.to_bits())
                    );
                    assert!(
                        expected[0]
                            .chunks_exact(channels)
                            .any(|p| p[channels - 1] == 1.0_f32.to_bits())
                    );
                    if !associated
                        && matches!(mode, FrameBlendMode::Add | FrameBlendMode::MultiplyAdd)
                    {
                        assert!(expected[1].iter().any(|&word| f32::from_bits(word) > 1.0));
                    }
                    for planar in [false, true] {
                        let color = ColorSpecification::Icc(profile.clone());
                        let format = if case.gray {
                            PixelFormat::gray_f32(true, planar, color)
                        } else {
                            PixelFormat::rgb_f32(RgbChannelOrder::Rgba, planar, color)
                        };
                        let request = GpuOutputRequest::color(format)
                            .unwrap()
                            .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                            .with_orientation_policy(OrientationPolicy::Keep)
                            .with_max_frame_slots(NonZeroUsize::new(3).unwrap());
                        let mut baseline = None;
                        for limit in [None, NonZeroU64::new(256)] {
                            let frames = output::frames(&backend, &data, request.clone(), limit);
                            assert_eq!(frames.len(), 2);
                            for (frame, actual) in frames.iter().enumerate() {
                                assert_eq!(actual.len(), expected[frame].len());
                                for (index, &expected) in expected[frame].iter().enumerate() {
                                    let channel = index % channels;
                                    let pixel = index / channels;
                                    let actual =
                                        actual[if planar { channel * 153 + pixel } else { index }];
                                    if channel == channels - 1 {
                                        assert_eq!(
                                            actual, expected,
                                            "{name} frame {frame} alpha {pixel}"
                                        );
                                    } else {
                                        let actual = f32::from_bits(actual);
                                        let lo = f32::from_bits(lower[frame][index]);
                                        let hi = f32::from_bits(upper[frame][index]);
                                        assert!(
                                            actual.is_finite() && actual >= lo && actual <= hi,
                                            "{name} frame {frame} sample {index} planar={planar} {limit:?}: {actual} outside {lo}..{hi}"
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
                    eprintln!("{name}: both frames, layouts, and input modes passed");
                }
            }
        }
    }
}
