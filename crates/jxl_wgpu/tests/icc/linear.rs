use jxl_gpu_protocol::{Chromaticity, RgbChromaticities, RgbColorSpace};
use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct Manifest {
    pub(super) spaces: Vec<Space>,
}

#[derive(Deserialize)]
pub(super) struct Space {
    pub(super) name: String,
    white: [f64; 2],
    primaries: [[f64; 2]; 3],
}

impl Space {
    pub(super) fn encoding(&self) -> RgbColorSpace {
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
