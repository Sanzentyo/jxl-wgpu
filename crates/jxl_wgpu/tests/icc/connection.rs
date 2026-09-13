use super::*;
use jxl_gpu_protocol::{Chromaticity, RgbChromaticities, RgbColorSpace};

#[derive(Deserialize)]
struct LinearManifest {
    spaces: Vec<LinearSpace>,
}

#[derive(Deserialize)]
struct LinearSpace {
    name: String,
    white: [f64; 2],
    primaries: [[f64; 2]; 3],
}

impl LinearSpace {
    fn encoding(&self) -> RgbColorSpace {
        let [red, green, blue] = self
            .primaries
            .map(|[x, y]| Chromaticity::new(x, y).unwrap());
        let white = Chromaticity::new(self.white[0], self.white[1]).unwrap();
        let color = RgbChromaticities {
            red,
            green,
            blue,
            white,
        };
        match self.name.as_str() {
            "bt709" => {
                assert_eq!(color, RgbChromaticities::BT709);
                RgbColorSpace::Bt709
            }
            "bt2020" => {
                assert_eq!(color, RgbChromaticities::BT2020);
                RgbColorSpace::Bt2020
            }
            "display_p3" => {
                assert_eq!(color, RgbChromaticities::DISPLAY_P3);
                RgbColorSpace::DisplayP3
            }
            "equal_white" => {
                assert_eq!(white, Chromaticity::E);
                RgbColorSpace::Custom(color)
            }
            "native_rgb" => RgbColorSpace::Custom(color),
            name => panic!("unknown linear reference endpoint {name}"),
        }
    }
}

#[test]
fn native_and_independent_linear_connections_keep_signed_rgb_and_exact_icc_curves() {
    let Some(backend) = backend() else {
        return;
    };
    let profiles: Manifest =
        serde_json::from_slice(&std::fs::read(directory().join("manifest.json")).unwrap()).unwrap();
    let linear: LinearManifest =
        serde_json::from_slice(&std::fs::read(directory().join("linear/manifest.json")).unwrap())
            .unwrap();
    assert_eq!(linear.spaces.len(), 5);
    let extent = Extent2d::new(profiles.width, profiles.height);
    let pipeline = ResidentIccPipeline::new(backend.device()).unwrap();
    let linear_input = floats("linear/input.f32le");
    assert!(linear_input.iter().any(|v| *v < 0.0) && linear_input.iter().any(|v| *v > 1.0));
    let mut total = 0;
    let mut outside_unit = [0; 2];
    let mut native_semantics = [0; 4];
    let mut maximum_error = 0.0_f32;
    for space in &linear.spaces {
        for record in &profiles.profiles {
            let profile = profile(&record.name);
            for to_linear in [true, false] {
                let transform = if to_linear {
                    IccTransform::to_linear_rgb(
                        &profile,
                        space.encoding(),
                        IccRenderingIntent::Relative,
                    )
                } else {
                    IccTransform::from_linear_rgb(
                        space.encoding(),
                        &profile,
                        IccRenderingIntent::Relative,
                    )
                }
                .unwrap();
                assert_eq!(
                    transform.source().channels(),
                    if to_linear { record.channels } else { 3 }
                );
                assert_eq!(
                    transform.target().channels(),
                    if to_linear { 3 } else { record.channels }
                );
                let input = if to_linear {
                    floats(&format!("{}_input.f32le", record.name))
                } else {
                    linear_input.clone()
                };
                let actual = run(&backend, &pipeline, &transform, extent, &input, 9);
                let name = if to_linear {
                    format!("{}_to_{}", record.name, space.name)
                } else {
                    format!("{}_to_{}", space.name, record.name)
                };
                let expected = references(&format!("linear/{name}.reference"));
                assert_eq!(actual.len(), expected.len());
                for (i, (value, reference)) in actual.into_iter().zip(expected).enumerate() {
                    assert!(
                        reference.lower <= reference.exact && reference.exact <= reference.upper
                    );
                    assert!(
                        value.is_finite() && value >= reference.lower && value <= reference.upper,
                        "{name} sample {i}: GPU {value}, independent reference {reference:?}"
                    );
                    assert!(reference.native_semantics < 4);
                    native_semantics[reference.native_semantics as usize] += 1;
                    if reference.native_semantics == 0 {
                        assert!(
                            reference.native >= reference.native_lower
                                && reference.native <= reference.native_upper,
                            "{name} sample {i}: native outside independent interval: {reference:?}"
                        );
                    }
                    if to_linear {
                        outside_unit[0] += usize::from(value < -1e-4);
                        outside_unit[1] += usize::from(value > 1.0001);
                    } else {
                        assert!((0.0..=1.0).contains(&value));
                    }
                    maximum_error = maximum_error.max((value - reference.exact).abs());
                    total += 1;
                }
            }
        }
    }
    assert_eq!(total, 182410);
    assert!(outside_unit.iter().all(|count| *count > 100));
    assert!(native_semantics[0] > 180000);
    eprintln!(
        "ICC linear connections {total} components; native semantics {native_semantics:?}; signed/above-one outputs {outside_unit:?}; maximum absolute error {maximum_error}"
    );
}
