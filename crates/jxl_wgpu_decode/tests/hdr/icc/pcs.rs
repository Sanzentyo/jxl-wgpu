use super::*;

#[test]
fn hdr_original_and_composed_sources_reach_relative_icc_pcs_with_image_luminance() {
    let backend = backend();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let identity = IccProfile::parse(
        std::fs::read(root.join("../jxl_wgpu/test-data/icc/mpe/identity.icc"))
            .unwrap()
            .into(),
        Default::default(),
    )
    .unwrap();
    for case in corpus::cases() {
        let linear = case.xyb && !case.sequence;
        let rgba = case.reference(linear);
        let transfer = if linear {
            TransferFunction::Linear
        } else {
            case.transfer
        };
        let matrix = color::pcs_matrix(case.space);
        let mut baseline = None;
        for planar in [false, true] {
            let format = PixelFormat::rgb_f32(
                RgbChannelOrder::Rgba,
                planar,
                ColorSpecification::Icc(identity.clone()),
            );
            for bounded in [false, true] {
                let request = GpuOutputRequest::color(format.clone())
                    .unwrap()
                    .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                    .with_icc_rendering_intent(IccRenderingIntent::Relative);
                let actual = frames::read(&backend, &case.bytes(), request, planar, 4, bounded);
                assert_eq!(actual.len(), case.frame_count());
                for (frame, actual) in actual.iter().enumerate() {
                    assert_eq!(actual.len(), case.frame_words());
                    for (p, pixel) in actual.as_chunks::<4>().0.iter().enumerate() {
                        let input = &rgba[frame * case.frame_words() + p * 4..][..4];
                        let rgb = [input[0], input[1], input[2]].map(f64::from);
                        let light = oracle::to_linear(rgb, transfer, case.space, case.nits);
                        let range = oracle::linear_interval(
                            rgb,
                            transfer,
                            case.space,
                            case.nits,
                            f64::from(case.tolerance()),
                        );
                        for c in 0..3 {
                            let expected: f64 = (0..3).map(|i| matrix[c][i] * light[i]).sum();
                            let bounds: [f64; 2] = std::array::from_fn(|edge| {
                                (0..3)
                                    .map(|i| {
                                        matrix[c][i]
                                            * range[i]
                                                [if matrix[c][i] >= 0.0 { edge } else { 1 - edge }]
                                    })
                                    .sum()
                            });
                            let round = 5e-5 * (1.0 + expected.abs());
                            let value = f64::from(f32::from_bits(pixel[c]));
                            assert!(
                                value.is_finite()
                                    && value >= bounds[0] - round
                                    && value <= bounds[1] + round,
                                "{} {frame}/{p}/{c}: {value}, PCS {expected}, {bounds:?}",
                                case.name
                            );
                        }
                        assert!((f32::from_bits(pixel[3]) - input[3]).abs() <= 2e-6);
                    }
                }
                if let Some(baseline) = &baseline {
                    assert_eq!(&actual, baseline, "{} HDR ICC packing", case.name);
                }
                baseline = Some(actual);
            }
        }
    }
}
