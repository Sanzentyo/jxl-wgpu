use super::*;

#[test]
fn luminance_metadata_has_canonical_identity_and_checked_boundaries() {
    assert_eq!(
        LuminanceRange::new(-0.0, 100.0),
        LuminanceRange::new(0.0, 100.0)
    );
    for black in [f32::NAN, f32::INFINITY, -1.0, 100.01] {
        assert!(LuminanceRange::new(black, 100.0).is_none());
    }
    for white in [f32::NAN, f32::INFINITY, -1.0, 0.0] {
        assert!(LuminanceRange::new(0.0, white).is_none());
    }
    let range = LuminanceRange::new(100.0, 100.0).unwrap();
    assert_eq!(range.black_nits(), range.white().nits());
    assert_eq!(
        ToneMapping::new(range, range, -0.0),
        ToneMapping::new(range, range, 0.0)
    );
    for threshold in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(ToneMapping::new(range, range, threshold).is_none());
    }
    assert_eq!(
        ToneMapping::new(range, range, 200.0)
            .unwrap()
            .linear_below_nits(),
        200.0
    );
}
