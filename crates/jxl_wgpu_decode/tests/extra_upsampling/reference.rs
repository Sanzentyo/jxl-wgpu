pub(super) use jxl_test_support::oracles::resampling::{Arithmetic, Plane, Sample};

#[test]
fn cropping_the_intermediate_grid_changes_edge_samples_beyond_rounding_error() {
    let case = super::corpus::cases()
        .into_iter()
        .find(|case| case.width == 129 && case.color_factor == 1 && case.extra_factor == 64)
        .unwrap();
    let (_, inventory, expected) = case.load(Arithmetic::Wgsl, false);
    let weights = &inventory.image_header.upsampling_weights;
    let samples = (0..2)
        .flat_map(|y| {
            (0..3).map(move |x| {
                let code = (x * 311 + y * 997 + x * y * 53 + 3 * 4013) % 65521;
                Sample {
                    value: f64::from((code % 97) - 43) / 16.0,
                    error: 0.0,
                }
            })
        })
        .collect();
    let wrong = Plane {
        width: 3,
        height: 2,
        samples,
    }
    .filter(8, weights, Arithmetic::Wgsl)
    .crop(17, 13)
    .filter(8, weights, Arithmetic::Wgsl)
    .crop(129, 97);
    let differing = expected[3]
        .samples
        .iter()
        .zip(&wrong.samples)
        .filter(|(correct, wrong)| {
            (correct.value - wrong.value).abs() > correct.error + wrong.error
        })
        .count();
    assert!(
        differing > 100,
        "fixture must distinguish premature cropping"
    );
}
