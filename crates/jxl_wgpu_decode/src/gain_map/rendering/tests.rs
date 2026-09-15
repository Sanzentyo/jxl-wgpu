use super::*;
use jxl_gpu_bitstream::gain_map::UnsignedFraction;

#[test]
fn headroom_selection_preserves_direction_endpoints_and_tiny_nonzero_weights() {
    let mut metadata = GainMapMetadata::default();
    metadata.base_hdr_headroom.numerator = 1;
    metadata.alternate_hdr_headroom.numerator = 3;
    for reverse in [false, true] {
        if reverse {
            std::mem::swap(
                &mut metadata.base_hdr_headroom,
                &mut metadata.alternate_hdr_headroom,
            );
        }
        let render = |h| {
            GainMapRendering {
                rendition: GainMapRendition::DisplayHeadroom(h),
                ..Default::default()
            }
            .weight(&metadata)
            .unwrap()
        };
        let expected = if reverse {
            [
                Weight::Apply(-1.0),
                Weight::Apply(-1.0),
                Weight::Apply(-0.5),
                Weight::Baseline,
                Weight::Baseline,
            ]
        } else {
            [
                Weight::Baseline,
                Weight::Baseline,
                Weight::Apply(0.5),
                Weight::Apply(1.0),
                Weight::Apply(1.0),
            ]
        };
        for (h, expected) in [0.0, 1.0, 2.0, 3.0, f64::MAX].into_iter().zip(expected) {
            assert_eq!(render(h), expected);
        }
    }
    metadata.base_hdr_headroom = UnsignedFraction {
        numerator: u32::MAX - 1,
        denominator: u32::MAX,
    };
    metadata.alternate_hdr_headroom = UnsignedFraction {
        numerator: u32::MAX - 2,
        denominator: u32::MAX - 1,
    };
    assert_eq!(
        metadata.base_hdr_headroom.value(),
        metadata.alternate_hdr_headroom.value()
    );
    assert_eq!(
        GainMapRendering::default().weight(&metadata).unwrap(),
        Weight::Apply(-1.0)
    );
    let rounded_endpoint = GainMapRendering {
        rendition: GainMapRendition::DisplayHeadroom(metadata.base_hdr_headroom.value()),
        ..Default::default()
    };
    // The shared rounded value lies above both exact fractions, not at either endpoint.
    assert_eq!(
        rounded_endpoint.weight(&metadata).unwrap(),
        Weight::Baseline
    );
    std::mem::swap(
        &mut metadata.base_hdr_headroom,
        &mut metadata.alternate_hdr_headroom,
    );
    assert_eq!(
        rounded_endpoint.weight(&metadata).unwrap(),
        Weight::Apply(1.0)
    );
    metadata.base_hdr_headroom.numerator = 0;
    let tiny = GainMapRendering {
        rendition: GainMapRendition::DisplayHeadroom(f64::from_bits(1)),
        ..Default::default()
    };
    assert_eq!(tiny.weight(&metadata).unwrap(), Weight::Apply(0.0));
    metadata.alternate_hdr_headroom.numerator = 0;
    assert_eq!(tiny.weight(&metadata).unwrap(), Weight::Baseline);
    for h in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            GainMapRendering {
                rendition: GainMapRendition::DisplayHeadroom(h),
                ..Default::default()
            }
            .weight(&metadata)
            .is_err()
        );
    }
}
