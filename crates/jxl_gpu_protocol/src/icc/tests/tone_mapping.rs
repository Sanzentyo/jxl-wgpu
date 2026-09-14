use super::*;
use crate::{LuminanceRange, RgbColorEncoding, ToneMapping};

#[test]
fn tone_mapping_separates_source_and_target_matrices_even_for_the_same_profile() {
    let profile = parse(profile_bytes(&rgb_tags())).unwrap();
    let mapping = ToneMapping::new(
        LuminanceRange::new(0.0, 1000.0).unwrap(),
        LuminanceRange::new(0.0, 100.0).unwrap(),
        20.0,
    )
    .unwrap();
    for intent in [
        IccRenderingIntent::Perceptual,
        IccRenderingIntent::Relative,
        IccRenderingIntent::Saturation,
        IccRenderingIntent::Absolute,
    ] {
        let unmapped = IccTransform::new(&profile, &profile, intent).unwrap();
        let mapped = unmapped.clone().with_tone_mapping(mapping).unwrap();
        assert_ne!(mapped, unmapped);
        let stages = mapped.program().stages();
        let index = stages
            .iter()
            .position(|stage| matches!(stage, IccStage::ToneMapping(_)))
            .unwrap();
        assert!(matches!(stages[index - 1], IccStage::Matrix(_)));
        assert!(matches!(stages[index + 1], IccStage::Matrix(_)));
        assert_eq!(
            stages
                .iter()
                .filter(|stage| matches!(stage, IccStage::ToneMapping(_)))
                .count(),
            1
        );
        assert_eq!(
            mapped.clone().with_tone_mapping(mapping).unwrap(),
            mapped,
            "builder replaces, never stacks"
        );
        for forward in [true, false] {
            let transform = if forward {
                IccTransform::from_rgb(RgbColorEncoding::LINEAR_BT709, &profile, intent)
            } else {
                IccTransform::to_rgb(&profile, RgbColorEncoding::LINEAR_BT709, intent)
            }
            .unwrap()
            .with_tone_mapping(mapping)
            .unwrap();
            let endpoint = if forward {
                transform.source()
            } else {
                transform.target()
            };
            assert!(
                matches!(endpoint, IccTransformEndpoint::Rgb { intensity: Some(value), .. }
                if *value == if forward { mapping.source().white() } else { mapping.target().white() })
            );
        }
    }
}
