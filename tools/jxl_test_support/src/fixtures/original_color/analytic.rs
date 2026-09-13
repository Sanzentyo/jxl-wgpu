//! Explicit extra profiles; the original D65 corpus remains unchanged.
use super::{Case, MODES, Profile, Transfer};
use jxl_gpu_bitstream::{
    ChromaticityInventory as Xy, PrimariesInventory as P, TransferFunctionInventory as Tf,
    WhitePointInventory as W,
};

pub fn cases() -> Vec<Case> {
    let profiles = [
        Profile {
            name: "e_bt709",
            primaries: P::Srgb,
            grayscale: false,
            white: W::E,
        },
        Profile {
            name: "dci_p3",
            primaries: P::P3,
            grayscale: false,
            white: W::Dci,
        },
        Profile {
            name: "d50_adobe",
            primaries: P::Custom {
                red: Xy {
                    x: 640000,
                    y: 330000,
                },
                green: Xy {
                    x: 210000,
                    y: 710000,
                },
                blue: Xy {
                    x: 150000,
                    y: 60000,
                },
            },
            grayscale: false,
            white: W::Custom(Xy {
                x: 345670,
                y: 358500,
            }),
        },
        Profile {
            name: "d65_custom",
            primaries: P::Custom {
                red: Xy {
                    x: 734700,
                    y: 265300,
                },
                green: Xy {
                    x: 115200,
                    y: 826400,
                },
                blue: Xy {
                    x: 156600,
                    y: 17700,
                },
            },
            grayscale: false,
            white: W::D65,
        },
        Profile {
            name: "e_gray",
            primaries: P::Srgb,
            grayscale: true,
            white: W::E,
        },
        Profile {
            name: "dci_gray",
            primaries: P::Srgb,
            grayscale: true,
            white: W::Dci,
        },
    ];
    let transfers = [
        Transfer {
            name: "srgb",
            transfer: Tf::Srgb,
        },
        Transfer {
            name: "dci",
            transfer: Tf::Dci,
        },
        Transfer {
            name: "gamma22",
            transfer: Tf::Gamma {
                scaled_gamma: 4545455,
                inverted: true,
            },
        },
        Transfer {
            name: "linear",
            transfer: Tf::Linear,
        },
        Transfer {
            name: "gamma2",
            transfer: Tf::Gamma {
                scaled_gamma: 5000000,
                inverted: true,
            },
        },
        Transfer {
            name: "dci",
            transfer: Tf::Dci,
        },
    ];
    let mut result = Vec::new();
    for mode in MODES {
        for (index, (&profile, &transfer)) in profiles.iter().zip(&transfers).enumerate() {
            if mode.ycbcr() && index > 1 {
                continue;
            }
            for floating in [false, true] {
                if floating && (mode.ycbcr() || ![1, 2, 4].contains(&index)) {
                    continue;
                }
                for sequence in [false, true] {
                    let mut case = Case::new(mode, profile, transfer, sequence, floating);
                    case.name.insert_str(0, "analytic_");
                    result.push(case);
                }
            }
        }
    }
    result
}
