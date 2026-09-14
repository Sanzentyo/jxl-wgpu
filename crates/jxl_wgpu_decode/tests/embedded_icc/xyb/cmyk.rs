use crate::{inventory, lut::reference, numeric_float, output};
use jxl_gpu_bitstream::{ExtraChannelTypeInventory, FrameEncoding};
use jxl_gpu_formats::{ColorSample, ColorStorage, PixelFormat};
use jxl_gpu_protocol::{
    WhitePointAdaptation,
    icc::{IccProfile, IccRenderingIntent},
};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuOutputRequest, SpotColorPolicy};
use serde::Deserialize;
use std::{
    num::{NonZeroU64, NonZeroUsize},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
struct Manifest {
    width: usize,
    height: usize,
    frames: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    profile: String,
    modular: bool,
    black: usize,
}

fn directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/cmyk_xyb")
}

fn cases() -> Vec<Case> {
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(directory().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        (manifest.width, manifest.height, manifest.frames),
        (17, 9, 3)
    );
    assert_eq!(manifest.cases.len(), 12);
    manifest.cases
}

impl Case {
    fn inputs(&self) -> (Vec<u8>, IccProfile, Vec<u32>) {
        let data = std::fs::read(directory().join(format!("{}.jxl", self.name))).unwrap();
        let parsed = inventory(&data);
        assert!(parsed.image_header.xyb_encoded);
        assert!(!parsed.image_header.grayscale);
        assert_eq!(
            (parsed.image_header.width, parsed.image_header.height),
            (17, 9)
        );
        assert!(matches!(self.black, 0 | 2));
        assert_eq!(parsed.image_header.extra_channels.len(), 3);
        assert_eq!(
            parsed.image_header.extra_channels[self.black].channel_type,
            ExtraChannelTypeInventory::Black
        );
        assert!(matches!(
            parsed.image_header.extra_channels[1].channel_type,
            ExtraChannelTypeInventory::Alpha { associated: false }
        ));
        let profile = IccProfile::parse(
            std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../jxl_wgpu/test-data/icc/lut")
                    .join(format!("{}.icc", self.profile)),
            )
            .unwrap()
            .into(),
            Default::default(),
        )
        .unwrap();
        assert_eq!(
            &parsed.image_header.embedded_icc.as_ref().unwrap().profile,
            profile.bytes()
        );
        assert_eq!(
            profile.header().rendering_intent,
            IccRenderingIntent::Relative
        );
        assert_eq!(parsed.frames.len(), 3);
        for (index, frame) in parsed.frames.iter().enumerate() {
            assert_eq!(
                frame.encoding,
                if self.modular {
                    FrameEncoding::Modular
                } else {
                    FrameEncoding::VarDct
                }
            );
            assert!(!frame.can_be_referenced());
            assert!(!frame.do_ycbcr);
            assert_eq!(frame.duration_ticks, index as u32 + 1);
        }
        let source = std::fs::read(directory().join(format!("{}.f32", self.name))).unwrap();
        let (words, tail) = source.as_chunks::<4>();
        assert!(tail.is_empty());
        assert_eq!(words.len(), 3 * 153 * 6);
        (
            data,
            profile,
            words.iter().copied().map(u32::from_le_bytes).collect(),
        )
    }
}

fn request(format: PixelFormat) -> GpuOutputRequest {
    GpuOutputRequest::color(format)
        .unwrap()
        .with_spot_color_policy(SpotColorPolicy::Preserve)
        .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
        .with_max_frame_slots(NonZeroUsize::new(3).unwrap())
}

#[test]
fn cmyk_xyb_linear_basis_matches_native_with_unchanged_extra_samples() {
    use jxl_gpu_formats::{ColorSpecification, RgbChannelOrder, TransferFunction};
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let mut color = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
    let ColorSpecification::Defined(ref mut spec) = color else {
        unreachable!()
    };
    spec.transfer = TransferFunction::Linear;
    let format = PixelFormat::rgb_f32(RgbChannelOrder::Rgba, false, color);
    for case in cases() {
        let (data, _, native) = case.inputs();
        let mut baseline = None;
        for limit in [None, NonZeroU64::new(256)] {
            let frames = output::frames(&backend, &data, request(format.clone()), limit);
            assert_eq!(frames.len(), 3);
            for (frame, pixels) in frames.iter().enumerate() {
                assert_eq!(pixels.len(), 153 * 4);
                for (pixel, rgba) in pixels.as_chunks::<4>().0.iter().enumerate() {
                    let source = &native[(frame * 153 + pixel) * 6..];
                    assert_eq!(rgba[3], source[4]);
                    for c in 0..3 {
                        let actual = f32::from_bits(rgba[c]);
                        let expected = f32::from_bits(source[c]);
                        assert!(
                            actual.is_finite()
                                && (actual - expected).abs() <= (1.0 + expected.abs()) / 1024.0,
                            "{} frame {frame} pixel {pixel} channel {c}: {actual} vs {expected}",
                            case.name
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

#[test]
fn legal_cmyk_xyb_output_keeps_generated_k_and_alpha_independent() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let mut components = 0;
    let mut presentations = 0;
    for case in cases() {
        eprintln!("CMYK XYB device {}", case.name);
        let (data, profile, native) = case.inputs();
        for intent in [
            IccRenderingIntent::Perceptual,
            IccRenderingIntent::Relative,
            IccRenderingIntent::Saturation,
            IccRenderingIntent::Absolute,
        ] {
            let expected = reference(&directory(), &case.name, intent);
            assert_eq!(expected.len(), 3 * 153 * 4);
            assert!(
                expected
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(native.as_chunks::<6>().0)
                    .filter(|(device, source)| {
                        let black = 1.0 - f32::from_bits(source[3 + case.black]);
                        black < device[3][2] - 0.01 || black > device[3][3] + 0.01
                    })
                    .count()
                    > 153
            );
            for storage in [ColorStorage::Planar, ColorStorage::Interleaved] {
                let format =
                    PixelFormat::icc_device(profile.clone(), ColorSample::F32, storage, true)
                        .unwrap();
                let mut baseline = None;
                for limit in [None, NonZeroU64::new(256)] {
                    let frames = output::frames(
                        &backend,
                        &data,
                        request(format.clone()).with_icc_rendering_intent(intent),
                        limit,
                    );
                    assert_eq!(frames.len(), 3);
                    for (frame, pixels) in frames.iter().enumerate() {
                        assert_eq!(pixels.len(), 153 * 5);
                        for pixel in 0..153 {
                            let word = |c| {
                                pixels[if storage == ColorStorage::Planar {
                                    c * 153 + pixel
                                } else {
                                    pixel * 5 + c
                                }]
                            };
                            assert_eq!(word(4), native[(frame * 153 + pixel) * 6 + 4]);
                            for c in 0..4 {
                                let actual = f32::from_bits(word(c));
                                let interval = expected[(frame * 153 + pixel) * 4 + c];
                                assert!(
                                    actual.is_finite()
                                        && interval[2] <= actual
                                        && actual <= interval[3],
                                    "{} {intent:?} {storage:?} {limit:?} frame {frame} pixel {pixel} component {c}: {actual} vs {interval:?}",
                                    case.name
                                );
                                components += 1;
                            }
                        }
                        presentations += 1;
                    }
                    if let Some(baseline) = &baseline {
                        assert_eq!(&frames, baseline);
                    }
                    baseline = Some(frames);
                }
            }
        }
    }
    assert_eq!((presentations, components), (576, 352512));
    eprintln!("CMYK XYB device: {presentations} presentations, {components} components");
}

#[test]
fn cmyk_xyb_numeric_cmy_reconstruction_preserves_encoded_black_alpha_and_spot_samples() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let mut components = 0;
    for case in cases() {
        eprintln!("CMYK XYB numeric {}", case.name);
        let (data, _, native) = case.inputs();
        let expected = reference(&directory(), &case.name, IccRenderingIntent::Relative);
        for channel in 0..6 {
            let scalar = numeric_float().with_max_frame_slots(NonZeroUsize::new(3).unwrap());
            let scalar = if channel < 3 {
                scalar.with_color_channel(channel)
            } else {
                scalar.with_extra_channel(channel - 3)
            }
            .unwrap();
            let mut baseline = None;
            for limit in [None, NonZeroU64::new(256)] {
                eprintln!("{} numeric {channel} window={limit:?}", case.name);
                let scalar = scalar
                    .clone()
                    .with_white_point_adaptation(WhitePointAdaptation::None)
                    .with_spot_color_policy(SpotColorPolicy::Render)
                    .with_alpha_output_policy(AlphaOutputPolicy::Associated)
                    .with_icc_rendering_intent(if limit.is_some() {
                        IccRenderingIntent::Saturation
                    } else {
                        IccRenderingIntent::Absolute
                    });
                let frames = output::frames(&backend, &data, scalar, limit);
                assert_eq!(frames.len(), 3);
                for (frame, pixels) in frames.iter().enumerate() {
                    assert_eq!(pixels.len(), 153);
                    for (pixel, &word) in pixels.iter().enumerate() {
                        if channel < 3 {
                            let actual = f64::from(f32::from_bits(word));
                            let interval = expected[(frame * 153 + pixel) * 4 + channel as usize];
                            // One final F32 subtraction complements the generated CMY component.
                            let lower = 1.0 - f64::from(interval[3]) - f64::from(f32::EPSILON);
                            let upper = 1.0 - f64::from(interval[2]) + f64::from(f32::EPSILON);
                            assert!(
                                actual.is_finite() && lower <= actual && actual <= upper,
                                "{} {limit:?} frame {frame} pixel {pixel} component {channel}: {actual} vs [{lower}, {upper}]",
                                case.name
                            );
                        } else {
                            assert_eq!(
                                word,
                                native[(frame * 153 + pixel) * 6 + channel as usize],
                                "{} frame {frame} pixel {pixel} extra {}",
                                case.name,
                                channel - 3
                            );
                        }
                        components += 1;
                    }
                }
                if let Some(baseline) = &baseline {
                    assert_eq!(&frames, baseline);
                }
                baseline = Some(frames);
            }
        }
    }
    assert_eq!(components, 66096);
    eprintln!("CMYK XYB numeric: {components} component comparisons");
}
