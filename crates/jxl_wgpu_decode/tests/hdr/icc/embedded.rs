use super::*;

#[test]
fn embedded_rgb_gray_icc_and_xyb_output_hdr_against_independent_linear_references() {
    use jxl_test_support::fixtures::embedded_icc as embedded;
    let backend = backend();
    for case in embedded::cases() {
        let data = case.bytes();
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        case.validate(&inventory);
        let reference_nits = f64::from(
            inventory
                .image_header
                .tone_mapping
                .intensity_target
                .to_f32(),
        );
        let bytes = if case.xyb {
            std::fs::read(
                embedded::directory()
                    .parent()
                    .unwrap()
                    .join("embedded_icc_xyb")
                    .join(format!("{}.linear.native.f32le", case.name())),
            )
            .unwrap()
        } else {
            jxl_test_support::offline::unhex(
                &std::fs::read_to_string(
                    embedded::directory().join(format!("{}.linear.scalar.f32.hex", case.name())),
                )
                .unwrap(),
            )
        };
        let values: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        let rgba: Vec<_> = if case.xyb && case.gray {
            values
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[0], p[0], p[1]])
                .collect()
        } else {
            values
        };
        assert_eq!(rgba.len(), 17 * 9 * 4);
        for nits in [100, 255, 1000, 4000] {
            let data = corpus::intensity::replace(&data, nits);
            let independent = jxl_oxide::JxlImage::read_with_defaults(data.as_slice()).unwrap();
            assert_eq!(
                independent
                    .image_header()
                    .metadata
                    .tone_mapping
                    .intensity_target,
                f32::from(nits)
            );
            let nits = f64::from(nits);
            let scale = if case.xyb { reference_nits / nits } else { 1.0 };
            for transfer in [TransferFunction::Pq, TransferFunction::Hlg] {
                for space in [ColorSpace::Bt2020, ColorSpace::DisplayP3] {
                    let ColorSpecification::Defined(mut color) =
                        jxl_wgpu_decode::vardct_rgb8_format().color_spec
                    else {
                        unreachable!()
                    };
                    color.space = space;
                    color.transfer = transfer;
                    let mut baseline = None;
                    for planar in [false, true] {
                        let format = PixelFormat::rgb_f32(
                            RgbChannelOrder::Rgba,
                            planar,
                            ColorSpecification::Defined(color),
                        );
                        for bounded in [false, true] {
                            let request = GpuOutputRequest::color(format.clone())
                                .unwrap()
                                .with_alpha_output_policy(AlphaOutputPolicy::Preserve);
                            let actual = frames(&backend, &data, request, planar, 4, bounded);
                            assert_eq!(actual.len(), 1);
                            for (p, pixel) in actual[0].as_chunks::<4>().0.iter().enumerate() {
                                let original =
                                    [rgba[p * 4], rgba[p * 4 + 1], rgba[p * 4 + 2]].map(f64::from);
                                let rgb = original.map(|v| v * scale);
                                let input_range: [[f64; 2]; 3] = std::array::from_fn(|c| {
                                    let radius = if case.xyb {
                                        (1.0 + original[c].abs()) / 1024.0
                                    } else {
                                        2e-4
                                    };
                                    [
                                        (original[c] - radius) * scale,
                                        (original[c] + radius) * scale,
                                    ]
                                });
                                let expected = oracle::convert(
                                    rgb,
                                    TransferFunction::Linear,
                                    transfer,
                                    ColorSpace::Bt709,
                                    space,
                                    nits,
                                );
                                let matrix = color::matrix(ColorSpace::Bt709, space);
                                let target_range = matrix.map(|row| {
                                    std::array::from_fn(|edge| {
                                        (0..3)
                                            .map(|c| {
                                                row[c]
                                                    * input_range[c][if row[c] >= 0.0 {
                                                        edge
                                                    } else {
                                                        1 - edge
                                                    }]
                                            })
                                            .sum()
                                    })
                                });
                                let bounds = oracle::from_linear_interval(
                                    target_range,
                                    transfer,
                                    space,
                                    nits,
                                );
                                for c in 0..3 {
                                    let value = f64::from(f32::from_bits(pixel[c]));
                                    let round = 5e-5 * (1.0 + expected[c].abs());
                                    assert!(
                                        value.is_finite()
                                            && value >= bounds[c][0] - round
                                            && value <= bounds[c][1] + round,
                                        "{} {transfer:?} {space:?} {p}/{c}: {value}, {}, {:?}",
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
    }
}
