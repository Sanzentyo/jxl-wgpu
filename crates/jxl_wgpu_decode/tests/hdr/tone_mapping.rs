use super::{backend, corpus, frames, oracle};
use jxl_gpu_formats::{
    ColorSpace, ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction,
};
use jxl_gpu_protocol::{LuminanceRange, icc::IccProfile};
use jxl_test_support::{
    fixtures::tone_mapping::{self, Metadata},
    oracles::{color, tone_mapping::Mapping},
};
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuOutputRequest};
use std::path::Path;

#[test]
fn embedded_icc_and_xyb_tone_mapping_reach_enumerated_display_light() {
    let backend = backend();
    for case in jxl_test_support::fixtures::embedded_icc::cases() {
        let data = case.bytes();
        let image = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap()
            .image_header;
        let scale = if case.xyb {
            f64::from(image.tone_mapping.intensity_target.to_f32()) / 1000.0
        } else {
            1.0
        };
        let data = tone_mapping::replace(
            &data,
            Metadata {
                intensity_target: 1000.0,
                min_nits: 0.0625,
                relative_to_max_display: true,
                linear_below: 0.125,
            },
        );
        let reference = case.linear_reference();
        let mapping = Mapping {
            source: [0.0625, 1000.0],
            target: [0.0, 80.0],
            protected: 10.0,
        };
        let ColorSpecification::Defined(mut color) =
            jxl_wgpu_decode::vardct_rgb8_format().color_spec
        else {
            unreachable!()
        };
        color.transfer = TransferFunction::Linear;
        let mut baseline = None;
        for planar in [false, true] {
            for bounded in [false, true] {
                let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
                    RgbChannelOrder::Rgba,
                    planar,
                    ColorSpecification::Defined(color),
                ))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                .with_tone_mapping(LuminanceRange::new(0.0, 80.0).unwrap());
                let actual = frames::read(&backend, &data, request, planar, 4, bounded);
                assert_eq!(actual.len(), 1);
                for (p, pixel) in actual[0].as_chunks::<4>().0.iter().enumerate() {
                    let original = [reference[p * 4], reference[p * 4 + 1], reference[p * 4 + 2]]
                        .map(f64::from);
                    let linear = original.map(|v| v * scale);
                    let range = original.map(|v| {
                        let radius = if case.xyb {
                            (1.0 + v.abs()) / 1024.0
                        } else {
                            2e-4
                        };
                        [(v - radius) * scale, (v + radius) * scale]
                    });
                    let expected =
                        mapping.apply(linear, oracle::luminance(ColorSpace::Bt709), [1.0; 3]);
                    let bounds =
                        mapping.interval(range, oracle::luminance(ColorSpace::Bt709), [1.0; 3]);
                    for c in 0..3 {
                        let value = f64::from(f32::from_bits(pixel[c]));
                        let round = 8e-5 * (1.0 + expected[c].abs());
                        assert!(
                            value.is_finite()
                                && value >= bounds[c][0] - round
                                && value <= bounds[c][1] + round,
                            "{} tone {p}/{c}: {value}, {} in {:?}",
                            case.name(),
                            expected[c],
                            bounds[c]
                        );
                    }
                    assert_eq!(pixel[3], reference[p * 4 + 3].to_bits());
                }
                if let Some(baseline) = &baseline {
                    assert_eq!(&actual, baseline);
                }
                baseline = Some(actual);
            }
        }
    }
}

#[test]
fn image_tone_metadata_maps_hdr_and_icc_presentation_after_composition() {
    let backend = backend();
    let identity = IccProfile::parse(
        std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../jxl_wgpu/test-data/icc/mpe/identity.icc"),
        )
        .unwrap()
        .into(),
        Default::default(),
    )
    .unwrap();
    let mut presentations = 0;
    let mut components = 0;
    for (index, case) in corpus::cases().into_iter().enumerate() {
        let threshold = [0.0, 0.125, 20.0][index % 3];
        let relative = index % 3 == 1;
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
        let display_black = if index % 4 == 1 { 0.03125 } else { 0.0 };
        let target = LuminanceRange::new(display_black, 80.0).unwrap();
        let original = frames::read(
            &backend,
            &data,
            GpuOutputRequest::color(case.format(case.transfer, case.space))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve),
            false,
            4,
            false,
        );
        let mapping = Mapping {
            source: [f64::from(black), case.nits],
            target: [f64::from(display_black), 80.0],
            protected: if relative {
                f64::from(threshold) * 80.0
            } else {
                f64::from(threshold)
            },
        };
        let linear = case.xyb && !case.sequence;
        let rgba = case.reference(linear);
        let source_transfer = if linear {
            TransferFunction::Linear
        } else {
            case.transfer
        };
        // The three targets exercise source/target HLG OOTFs, absolute PQ units, and the
        // shared tone stage in D50 PCS. Target RGB has its own primary luminances.
        for target_transfer in [
            TransferFunction::Hlg,
            TransferFunction::Pq,
            TransferFunction::Linear,
        ] {
            let icc = target_transfer == TransferFunction::Linear;
            let mut format = if icc {
                PixelFormat::rgb_f32(
                    RgbChannelOrder::Rgba,
                    false,
                    ColorSpecification::Icc(identity.clone()),
                )
            } else {
                case.format(target_transfer, ColorSpace::DisplayP3)
            };
            let matrix = if icc {
                color::pcs_matrix(case.space)
            } else {
                color::matrix(case.space, ColorSpace::DisplayP3)
            };
            let luminance = if icc {
                [0.0, 1.0, 0.0]
            } else {
                oracle::luminance(ColorSpace::DisplayP3)
            };
            let neutral = if icc { [0.9642, 1.0, 0.8249] } else { [1.0; 3] };
            let mut baseline = None;
            for planar in [false, true] {
                format =
                    PixelFormat::rgb_f32(RgbChannelOrder::Rgba, planar, format.color_spec.clone());
                for bounded in [false, true] {
                    let request = GpuOutputRequest::color(format.clone())
                        .unwrap()
                        .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                        .with_tone_mapping(target);
                    let actual = frames::read(&backend, &data, request, planar, 4, bounded);
                    assert_eq!(actual.len(), case.frame_count());
                    for (frame, pixels) in actual.iter().enumerate() {
                        for (p, pixel) in pixels.as_chunks::<4>().0.iter().enumerate() {
                            let input = &rgba[frame * case.frame_words() + p * 4..][..4];
                            let rgb = [input[0], input[1], input[2]].map(f64::from);
                            let light =
                                oracle::to_linear(rgb, source_transfer, case.space, case.nits);
                            let range = oracle::linear_interval(
                                rgb,
                                source_transfer,
                                case.space,
                                case.nits,
                                f64::from(case.tolerance()),
                            );
                            let converted =
                                matrix.map(|row| (0..3).map(|c| row[c] * light[c]).sum());
                            let bounds = matrix.map(|row| {
                                std::array::from_fn(|edge| {
                                    (0..3)
                                        .map(|c| {
                                            row[c]
                                                * range[c]
                                                    [if row[c] >= 0.0 { edge } else { 1 - edge }]
                                        })
                                        .sum()
                                })
                            });
                            let mapped = mapping.apply(converted, luminance, neutral);
                            let interval = mapping.interval(bounds, luminance, neutral);
                            let expected = if icc {
                                mapped
                            } else {
                                oracle::from_linear(
                                    mapped,
                                    target_transfer,
                                    ColorSpace::DisplayP3,
                                    80.0,
                                )
                            };
                            // Propagate the declared tone arithmetic allowance before nonlinear
                            // output encoding, independently of codec and primary uncertainty.
                            let interval = std::array::from_fn(|c| {
                                let round = 8e-5 * (1.0 + mapped[c].abs());
                                [interval[c][0] - round, interval[c][1] + round]
                            });
                            let bounds = if icc {
                                interval
                            } else {
                                oracle::from_linear_interval(
                                    interval,
                                    target_transfer,
                                    ColorSpace::DisplayP3,
                                    80.0,
                                )
                            };
                            for c in 0..3 {
                                let value = f64::from(f32::from_bits(pixel[c]));
                                let round = 5e-5 * (1.0 + expected[c].abs());
                                assert!(
                                    value.is_finite()
                                        && value >= bounds[c][0] - round
                                        && value <= bounds[c][1] + round,
                                    "{} {target_transfer:?} {mapping:?}/{frame}/{p}/{c}: {value}, {} in {:?}",
                                    case.name,
                                    expected[c],
                                    bounds[c]
                                );
                                components += 1;
                            }
                            assert!((f32::from_bits(pixel[3]) - input[3]).abs() <= 2e-6);
                            assert_eq!(
                                pixel[3],
                                original[frame][p * 4 + 3],
                                "tone mapping preserves alpha words"
                            );
                        }
                    }
                    if let Some(baseline) = &baseline {
                        assert_eq!(&actual, baseline, "{} tone layout/input", case.name);
                    }
                    baseline = Some(actual);
                    presentations += case.frame_count();
                }
            }
        }
    }
    assert_eq!(presentations, 960);
    assert_eq!(components, 1_681_344);
    eprintln!("Tone presentation: {presentations} frames, {components} color components");
}
