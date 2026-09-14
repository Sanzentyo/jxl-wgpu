use super::*;
use jxl_gpu_formats::{ColorSpace, TransferFunction};
use jxl_gpu_protocol::{DisplayIntensity, RgbColorEncoding};
use jxl_test_support::oracles::{color, hdr};

#[test]
fn display_relative_hdr_endpoints_match_independent_pcs_equations() {
    let backend = backend().expect("HDR ICC needs an actual adapter");
    eprintln!("HDR ICC adapter: {:?}", backend.adapter_info());
    let identity = profile("mpe/identity");
    let samples: [[f32; 3]; 12] = [
        [0.0; 3],
        [1e-10, 2e-10, 3e-10],
        [0.001, 0.002, 0.003],
        [0.08, 0.18, 0.01],
        [0.5; 3],
        [1.0; 3],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-0.01, 0.5, 0.2],
        [-0.1, -0.2, -0.3],
        [0.0, -0.001, 0.00001],
        [-1.0; 3],
    ];
    let extent = Extent2d::new(4, 3);
    let mut components = 0;
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Tile16x16,
    ] {
        let pipeline = ResidentIccPipeline::with_variant(backend.device(), variant).unwrap();
        for nits in [
            100.0, 255.0, 280.0, 300.0, 320.0, 500.0, 1000.0, 4000.0, 10000.0,
        ] {
            let intensity = DisplayIntensity::new(nits as f32).unwrap();
            for space in [ColorSpace::Bt709, ColorSpace::Bt2020, ColorSpace::DisplayP3] {
                for transfer in [TransferFunction::Pq, TransferFunction::Hlg] {
                    let encoding = RgbColorEncoding {
                        space: space.rgb_space().unwrap(),
                        transfer: transfer.rgb_transfer().unwrap(),
                    };
                    for forward in [true, false] {
                        let transform = if forward {
                            IccTransform::from_rgb_with_intensity(
                                encoding,
                                intensity,
                                &identity,
                                IccRenderingIntent::Relative,
                            )
                        } else {
                            IccTransform::to_rgb_with_intensity(
                                &identity,
                                encoding,
                                intensity,
                                IccRenderingIntent::Relative,
                            )
                        }
                        .unwrap();
                        let input: Vec<_> = samples.into_iter().flatten().collect();
                        let output = run(&backend, &pipeline, &transform, extent, &input, 5);
                        let matrix = if forward {
                            color::pcs_matrix(space)
                        } else {
                            color::inverse_pcs_matrix(space)
                        };
                        for (p, actual) in output.as_chunks::<3>().0.iter().enumerate() {
                            let input = samples[p].map(f64::from);
                            let (expected, bounds) = if forward {
                                let linear = hdr::to_linear(input, transfer, space, nits);
                                let range =
                                    hdr::linear_interval(input, transfer, space, nits, 5e-5);
                                let expected =
                                    matrix.map(|row| (0..3).map(|c| row[c] * linear[c]).sum());
                                let bounds = matrix.map(|row| {
                                    std::array::from_fn(|edge| {
                                        (0..3)
                                            .map(|c| {
                                                row[c]
                                                    * range[c][if row[c] >= 0.0 {
                                                        edge
                                                    } else {
                                                        1 - edge
                                                    }]
                                            })
                                            .sum()
                                    })
                                });
                                (expected, bounds)
                            } else {
                                let linear =
                                    matrix.map(|row| (0..3).map(|c| row[c] * input[c]).sum());
                                let range = std::array::from_fn(|r| {
                                    let radius = 5e-7
                                        * (1.0
                                            + (0..3)
                                                .map(|c| (matrix[r][c] * input[c]).abs())
                                                .sum::<f64>());
                                    [linear[r] - radius, linear[r] + radius]
                                });
                                (
                                    hdr::from_linear(linear, transfer, space, nits),
                                    hdr::from_linear_interval(range, transfer, space, nits),
                                )
                            };
                            for c in 0..3 {
                                let value = f64::from(actual[c]);
                                let round = 5e-5 * (1.0 + expected[c].abs());
                                assert!(
                                    value.is_finite()
                                        && value >= bounds[c][0] - round
                                        && value <= bounds[c][1] + round,
                                    "{variant:?} {space:?} {transfer:?} {nits} {forward}/{p}/{c}: {value}, expected {}, bounds {:?}",
                                    expected[c],
                                    bounds[c]
                                );
                                components += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(components, 11_664);
}
