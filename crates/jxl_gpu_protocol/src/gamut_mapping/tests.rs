use super::GamutMapping;

#[test]
fn preferences_validate_endpoints_and_canonicalize_zero() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
        assert!(GamutMapping::new(value).is_none());
    }
    for value in [0.0, 0.1, 0.5, 1.0] {
        assert_eq!(
            GamutMapping::new(value).unwrap().preserve_saturation(),
            value
        );
    }
    assert_eq!(GamutMapping::new(-0.0), GamutMapping::new(0.0));
    assert_eq!(GamutMapping::default(), GamutMapping::new(0.1).unwrap());
}
