use super::*;
use crate::{GammaExponent, RgbColorEncoding, RgbColorSpace, TransferFunction};

#[test]
fn enumerated_transfers_surround_the_pcs_connection_without_device_curve_clipping() {
    let profile = parse(profile_bytes(&rgb_tags())).unwrap();
    for transfer in [
        TransferFunction::Linear,
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Bt2020,
        TransferFunction::Pq,
        TransferFunction::Hlg,
        TransferFunction::Gamma(GammaExponent::new(0.4545455).unwrap()),
        TransferFunction::Dci,
    ] {
        let encoding = RgbColorEncoding {
            space: RgbColorSpace::DisplayP3,
            transfer,
        };
        for intent in [
            IccRenderingIntent::Perceptual,
            IccRenderingIntent::Relative,
            IccRenderingIntent::Saturation,
            IccRenderingIntent::Absolute,
        ] {
            for to_linear in [true, false] {
                let transform = if to_linear {
                    IccTransform::from_rgb(encoding, &profile, intent)
                } else {
                    IccTransform::to_rgb(&profile, encoding, intent)
                }
                .unwrap();
                let endpoint = if to_linear {
                    transform.source()
                } else {
                    transform.target()
                };
                assert_eq!(endpoint, &IccTransformEndpoint::Rgb(encoding));
                let stages = transform.program().stages();
                if transfer == TransferFunction::Linear {
                    assert!(
                        !stages
                            .iter()
                            .any(|s| matches!(s, IccStage::RgbTransfer { .. }))
                    );
                    let linear = if to_linear {
                        IccTransform::from_linear_rgb(encoding.space, &profile, intent)
                    } else {
                        IccTransform::to_linear_rgb(&profile, encoding.space, intent)
                    }
                    .unwrap();
                    assert_eq!(transform, linear);
                } else {
                    let stage = if to_linear {
                        stages.first()
                    } else {
                        stages.last()
                    };
                    assert_eq!(
                        stage,
                        Some(&IccStage::RgbTransfer {
                            transfer,
                            to_linear
                        })
                    );
                    assert_eq!(
                        stages
                            .iter()
                            .filter(|s| matches!(s, IccStage::RgbTransfer { .. }))
                            .count(),
                        1
                    );
                }
            }
        }
    }
}
