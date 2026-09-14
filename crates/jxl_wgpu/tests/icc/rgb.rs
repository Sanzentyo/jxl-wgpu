use super::*;
use jxl_gpu_formats::{ColorSpace, TransferFunction as Tf};
use jxl_gpu_protocol::{GammaExponent, RgbColorEncoding};
use jxl_test_support::oracles::color;

#[test]
fn enumerated_transfers_and_color_geometry_match_independent_f64_in_both_directions() {
    let Some(backend) = backend() else { return };
    let manifest: linear::Manifest =
        serde_json::from_slice(&std::fs::read(directory().join("linear/manifest.json")).unwrap())
            .unwrap();
    let profile = profile("mpe/identity");

    let extent = Extent2d::new(17, 13);
    let edges = [
        -0.25_f32,
        -0.081,
        -0.04045,
        -0.0031308,
        -0.0,
        0.0,
        f32::MIN_POSITIVE,
        0.0031308,
        0.018,
        0.04045,
        0.081,
        0.5,
        1.0,
        1.25,
    ];
    let mut components = 0;
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Tile16x16,
    ] {
        let pipeline = ResidentIccPipeline::with_variant(backend.device(), variant).unwrap();
        for space in &manifest.spaces {
            let geometry = space.encoding();
            let c = geometry.chromaticities().unwrap();
            let color_space = ColorSpace::CustomRgb(c);
            for transfer in [
                Tf::Linear,
                Tf::Srgb,
                Tf::Bt709,
                Tf::Bt2020,
                Tf::Pq,
                Tf::Hlg,
                Tf::Gamma(GammaExponent::new(0.4545455).unwrap()),
                Tf::Dci,
            ] {
                let encoding = RgbColorEncoding {
                    space: geometry,
                    transfer: transfer.rgb_transfer().unwrap(),
                };
                for forward in [true, false] {
                    let transform = if forward {
                        IccTransform::from_rgb(encoding, &profile, IccRenderingIntent::Relative)
                    } else {
                        IccTransform::to_rgb(&profile, encoding, IccRenderingIntent::Relative)
                    }
                    .unwrap();
                    let input: Vec<f32> = (0..extent.area().unwrap() * 3)
                        .map(|i| {
                            let value = if i < edges.len() * 3 {
                                edges[(i / 3 + i % 3 * 5) % edges.len()]
                            } else {
                                ((i * 37 + 11) % 1009) as f32 / 1008.0
                            };
                            // PQ's singular extension above the signal domain is not a finite-color test.
                            if forward && transfer == Tf::Pq {
                                value.clamp(-1.0, 1.0)
                            } else {
                                value
                            }
                        })
                        .collect();
                    let actual = run(&backend, &pipeline, &transform, extent, &input, 5);
                    let matrix = if forward {
                        color::pcs_matrix(color_space)
                    } else {
                        color::inverse_pcs_matrix(color_space)
                    };
                    let (source, target) = if forward {
                        (transfer, Tf::Linear)
                    } else {
                        (Tf::Linear, transfer)
                    };
                    for (pixel, (input, actual)) in input
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .zip(actual.as_chunks::<3>().0)
                        .enumerate()
                    {
                        let input = input.map(f64::from);
                        let exact = color::convert(input, source, target, matrix);
                        // F32 matrix/power arithmetic allowance, propagated through signed matrices
                        // and piecewise transfers. PQ's near-cancelling constants require more ulps.
                        let arithmetic = if transfer == Tf::Pq { 5e-5 } else { 2e-6 };
                        let bounds = color::interval(input, source, target, matrix, arithmetic);
                        for c in 0..3 {
                            let actual = f64::from(actual[c]);
                            let round = arithmetic * (1.0 + exact[c].abs());
                            assert!(
                                actual.is_finite()
                                    && actual >= bounds[c][0] - round
                                    && actual <= bounds[c][1] + round,
                                "{}/{transfer:?}/{forward}/{pixel}/{c}: {actual}, exact {}, bounds {:?}, arithmetic {round}",
                                space.name,
                                exact[c],
                                bounds[c]
                            );
                            components += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(components, 159_120);
    eprintln!("enumerated ICC transfer/geometry: {components} independent components");
}
