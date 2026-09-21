use super::{backend, corpus, frames, oracle};
use jxl_gpu_formats::{
    ColorSpace, ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction,
};
use jxl_gpu_protocol::{GamutMapping, LuminanceRange};
use jxl_test_support::{
    fixtures::tone_mapping::{self, Metadata},
    oracles::{color, gamut_mapping as gamut, tone_mapping::Mapping},
};
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuOutputRequest};

mod reference;

#[test]
fn hdr_gamut_and_tone_mapping_follow_composition_and_preserve_protected_light() {
    let backend = backend();
    let reader = frames::FrameReader::new(&backend);
    let mut presentations = 0;
    let mut components = 0;
    for (index, case) in corpus::cases().into_iter().enumerate() {
        let relative = index % 7 == 1;
        let threshold = [0.0, 0.125, 20.0, 80.0, 100.0, 4000.0, 0.0][index % 7];
        let black = if index % 4 == 0 { 0.0625 } else { 0.0 };
        let data = tone_mapping::replace(
            &case.bytes(),
            Metadata {
                intensity_target: case.nits as f32,
                min_nits: black,
                relative_to_max_display: relative,
                linear_below: threshold,
            },
        );
        let originals = frames::read(
            &reader,
            &data,
            GpuOutputRequest::color(case.format(case.transfer, case.space))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve),
            false,
            4,
            false,
        );
        let linear = case.xyb && !case.sequence;
        let rgba = case.reference(linear);
        let source_transfer = if linear {
            TransferFunction::Linear
        } else {
            case.transfer
        };
        let target_space =
            [ColorSpace::Bt709, ColorSpace::DisplayP3, ColorSpace::Bt2020][index % 3];
        let preference = [0.0, 0.1, 0.5, 0.9, 1.0][index % 5];
        for tone in [false, true] {
            let mapping = tone.then_some(Mapping {
                source: [f64::from(black), case.nits],
                target: [0.0, 80.0],
                protected: if relative {
                    f64::from(threshold) * 80.0
                } else {
                    f64::from(threshold)
                },
            });
            let target_nits = if tone { 80.0 } else { case.nits };
            for transfer in [
                TransferFunction::Linear,
                TransferFunction::Pq,
                TransferFunction::Hlg,
            ] {
                let base_format = case.format(transfer, target_space);
                let mut baseline = None;
                for planar in [false, true] {
                    let format = PixelFormat::rgb_f32(
                        RgbChannelOrder::Rgba,
                        planar,
                        base_format.color_spec.clone(),
                    );
                    for bounded in [false, true] {
                        let mut request = GpuOutputRequest::color(format.clone())
                            .unwrap()
                            .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                            .with_gamut_mapping(GamutMapping::new(preference).unwrap())
                            .unwrap();
                        if tone {
                            request =
                                request.with_tone_mapping(LuminanceRange::new(0.0, 80.0).unwrap());
                        }
                        let actual = frames::read(&reader, &data, request, planar, 4, bounded);
                        assert_eq!(actual.len(), case.frame_count());
                        for (frame, pixels) in actual.iter().enumerate() {
                            for (p, pixel) in pixels.as_chunks::<4>().0.iter().enumerate() {
                                let offset = frame * case.frame_words() + p * 4;
                                let rgb = [rgba[offset], rgba[offset + 1], rgba[offset + 2]]
                                    .map(f64::from);
                                let light =
                                    oracle::to_linear(rgb, source_transfer, case.space, case.nits);
                                let interval = oracle::linear_interval(
                                    rgb,
                                    source_transfer,
                                    case.space,
                                    case.nits,
                                    f64::from(case.tolerance()),
                                );
                                let (mapped, bounds) = reference::map(
                                    light,
                                    interval,
                                    color::matrix(case.space, target_space),
                                    target_space,
                                    mapping,
                                    f64::from(preference),
                                );
                                let expected = oracle::from_linear(
                                    mapped,
                                    transfer,
                                    target_space,
                                    target_nits,
                                );
                                let bounds = oracle::from_linear_interval(
                                    bounds,
                                    transfer,
                                    target_space,
                                    target_nits,
                                );
                                for c in 0..3 {
                                    let value = f64::from(f32::from_bits(pixel[c]));
                                    let round = 5e-5 * (1.0 + expected[c].abs());
                                    assert!(
                                        value.is_finite()
                                            && value >= bounds[c][0] - round
                                            && value <= bounds[c][1] + round,
                                        "{} gamut {tone}/{transfer:?}/{frame}/{p}/{c}: {value}, {} in {:?}",
                                        case.name,
                                        expected[c],
                                        bounds[c]
                                    );
                                    components += 1;
                                }
                                assert_eq!(
                                    pixel[3],
                                    originals[frame][p * 4 + 3],
                                    "gamut preserves alpha words"
                                );
                            }
                        }
                        if let Some(baseline) = &baseline {
                            assert_eq!(&actual, baseline, "{} gamut input/layout", case.name);
                        }
                        baseline = Some(actual);
                        presentations += case.frame_count();
                    }
                }
            }
        }
    }
    assert_eq!(presentations, 1920);
    assert_eq!(components, 3_362_688);
    eprintln!("Gamut HDR: {presentations} presentations, {components} components");
}

#[test]
fn embedded_profiles_map_to_requested_rgb_gamut_after_icc_conversion() {
    let backend = backend();
    let reader = frames::FrameReader::new(&backend);
    for (index, case) in jxl_test_support::fixtures::embedded_icc::cases().enumerate() {
        let data = case.bytes();
        let rgba = case.linear_reference();
        let intensity = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap()
            .image_header
            .tone_mapping
            .intensity_target
            .to_f32();
        let target_space =
            [ColorSpace::Bt2020, ColorSpace::DisplayP3, ColorSpace::Bt709][index % 3];
        let ColorSpecification::Defined(mut spec) =
            jxl_wgpu_decode::vardct_rgb8_format().color_spec
        else {
            unreachable!()
        };
        spec.space = target_space;
        spec.transfer = TransferFunction::Linear;
        for tone in [false, true] {
            let mapping = tone.then_some(Mapping {
                source: [0.0, f64::from(intensity)],
                target: [0.0, 80.0],
                protected: 0.0,
            });
            let mut baseline = None;
            for planar in [false, true] {
                for bounded in [false, true] {
                    let mut request = GpuOutputRequest::color(PixelFormat::rgb_f32(
                        RgbChannelOrder::Rgba,
                        planar,
                        ColorSpecification::Defined(spec),
                    ))
                    .unwrap()
                    .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                    .with_gamut_mapping(GamutMapping::default())
                    .unwrap();
                    if tone {
                        request =
                            request.with_tone_mapping(LuminanceRange::new(0.0, 80.0).unwrap());
                    }
                    let actual = frames::read(&reader, &data, request, planar, 4, bounded);
                    assert_eq!(actual.len(), 1);
                    for (p, pixel) in actual[0].as_chunks::<4>().0.iter().enumerate() {
                        let light = [rgba[p * 4], rgba[p * 4 + 1], rgba[p * 4 + 2]].map(f64::from);
                        let interval = light.map(|v| {
                            let radius = if case.xyb {
                                (1.0 + v.abs()) / 1024.0
                            } else {
                                2e-4
                            };
                            [v - radius, v + radius]
                        });
                        let (expected, bounds) = reference::map(
                            light,
                            interval,
                            color::matrix(ColorSpace::Bt709, target_space),
                            target_space,
                            mapping,
                            f64::from(0.1_f32),
                        );
                        for c in 0..3 {
                            let actual = f64::from(f32::from_bits(pixel[c]));
                            assert!(
                                actual.is_finite()
                                    && actual >= bounds[c][0]
                                    && actual <= bounds[c][1],
                                "{} ICC gamut {tone}/{p}/{c}: {actual}, {} in {:?}",
                                case.name(),
                                expected[c],
                                bounds[c]
                            );
                        }
                        assert_eq!(pixel[3], rgba[p * 4 + 3].to_bits());
                    }
                    if let Some(baseline) = &baseline {
                        assert_eq!(&actual, baseline);
                    }
                    baseline = Some(actual);
                }
            }
        }
    }
}
