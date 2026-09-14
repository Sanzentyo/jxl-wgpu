use super::*;

mod mpe;

fn affine(transform: &IccTransform) -> &IccAffine {
    let matrices = transform
        .program()
        .stages()
        .iter()
        .filter_map(|stage| match stage {
            IccStage::Matrix(matrix) => Some(matrix),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(matrices.len(), 1);
    matrices[0]
}
fn matrix(transform: &IccTransform) -> [[f64; 3]; 3] {
    if !transform
        .program()
        .stages()
        .iter()
        .any(|s| matches!(s, IccStage::Matrix(_)))
    {
        return crate::color::matrix::IDENTITY;
    }
    let matrix = affine(transform);
    std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            if r < matrix.offset().len() && c < matrix.input_channels() {
                matrix.matrix()[r * matrix.input_channels() + c]
            } else {
                0.0
            }
        })
    })
}
fn offset(transform: &IccTransform) -> [f64; 3] {
    if !transform
        .program()
        .stages()
        .iter()
        .any(|s| matches!(s, IccStage::Matrix(_)))
    {
        return [0.0; 3];
    }
    std::array::from_fn(|c| affine(transform).offset().get(c).copied().unwrap_or(0.0))
}

fn put32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
}

fn element(kind: &[u8; 4], values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::from(*kind);
    bytes.extend_from_slice(&[0; 4]);
    for value in values {
        bytes.extend_from_slice(&value.to_be_bytes());
    }
    bytes
}

fn rgb_tags() -> Vec<([u8; 4], Vec<u8>)> {
    vec![
        (*b"wtpt", element(b"XYZ ", &[0xf6d6, 65536, 0xd32d])),
        (*b"rXYZ", element(b"XYZ ", &[40000, 10000, 1000])),
        (*b"gXYZ", element(b"XYZ ", &[20000, 50000, 5000])),
        (*b"bXYZ", element(b"XYZ ", &[3174, 5536, 48061])),
        (*b"rTRC", element(b"para", &[0, 65536 * 2])),
        (*b"gTRC", element(b"para", &[0, 65536 * 3])),
        (*b"bTRC", element(b"curv", &[0])),
    ]
}

fn profile_bytes(tags: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
    let mut bytes = vec![0; 132 + tags.len() * 12];
    put32(&mut bytes, 8, 0x0440_0000);
    bytes[12..16].copy_from_slice(b"mntr");
    bytes[16..20].copy_from_slice(b"RGB ");
    bytes[20..24].copy_from_slice(b"XYZ ");
    bytes[36..40].copy_from_slice(b"acsp");
    put32(&mut bytes, 64, 1);
    for (i, value) in [0xf6d6, 65536, 0xd32d].into_iter().enumerate() {
        put32(&mut bytes, 68 + i * 4, value);
    }
    put32(&mut bytes, 128, tags.len() as u32);
    for (i, (signature, payload)) in tags.iter().enumerate() {
        let offset = bytes.len() as u32;
        let entry = 132 + i * 12;
        bytes[entry..entry + 4].copy_from_slice(signature);
        put32(&mut bytes, entry + 4, offset);
        put32(&mut bytes, entry + 8, payload.len() as u32);
        bytes.extend(payload);
        bytes.resize(bytes.len().next_multiple_of(4), 0);
    }
    let size = bytes.len() as u32;
    put32(&mut bytes, 0, size);
    bytes
}

fn parse(bytes: Vec<u8>) -> Result<IccProfile, IccError> {
    IccProfile::parse(bytes.into(), IccLimits::default())
}

#[test]
fn exact_colorants_independent_curves_and_original_bytes_survive_selection() {
    let bytes = profile_bytes(&rgb_tags());
    let profile = parse(bytes.clone()).unwrap();
    let restricted = IccProfile::parse(
        bytes.clone().into(),
        IccLimits {
            max_curve_samples: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(profile, restricted, "limits do not change profile identity");
    assert_eq!(&**profile.bytes(), bytes);
    assert_eq!(profile.header().version, 0x0440_0000);
    let selected = profile
        .matrix_trc(IccDirection::DeviceToPcs, IccRenderingIntent::Relative)
        .unwrap();
    assert_eq!(
        selected.matrix()[0],
        [40000.0 / 65536.0, 20000.0 / 65536.0, 3174.0 / 65536.0]
    );
    assert_ne!(selected.curves()[0], selected.curves()[1]);
    assert_eq!(selected.curves()[2].kind(), &IccCurveKind::Identity);
    assert_eq!(selected.media_white(), [0xf6d6, 65536, 0xd32d]);
    assert_eq!(selected.chromatic_adaptation(), None);
    let transform = IccTransform::new(&profile, &profile, IccRenderingIntent::Relative).unwrap();
    for (r, row) in matrix(&transform).iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            assert_eq!(*value, f64::from(r == c));
        }
    }
}

#[test]
fn tag_priority_is_directional_and_never_silently_discards_a_lut() {
    for (signature, direction) in [
        (*b"A2B0", IccDirection::DeviceToPcs),
        (*b"A2B1", IccDirection::DeviceToPcs),
        (*b"D2B1", IccDirection::DeviceToPcs),
        (*b"B2A0", IccDirection::PcsToDevice),
        (*b"B2A1", IccDirection::PcsToDevice),
        (*b"B2D1", IccDirection::PcsToDevice),
    ] {
        let mut tags = rgb_tags();
        tags.push((
            signature,
            if signature[0] == b'D' || signature[2] == b'D' {
                mpe::identity_mpe(3)
            } else {
                element(b"mAB ", &[])
            },
        ));
        let profile = parse(profile_bytes(&tags)).unwrap();
        assert_eq!(
            profile.matrix_trc(direction, IccRenderingIntent::Relative),
            Err(IccError::TransformTag {
                tag: IccSignature(signature)
            })
        );
        let other = match direction {
            IccDirection::DeviceToPcs => IccDirection::PcsToDevice,
            IccDirection::PcsToDevice => IccDirection::DeviceToPcs,
        };
        assert!(
            profile
                .matrix_trc(other, IccRenderingIntent::Relative)
                .is_ok()
        );
    }
    let profile = parse(profile_bytes(&rgb_tags())).unwrap();
    for intent in [
        IccRenderingIntent::Perceptual,
        IccRenderingIntent::Absolute,
        IccRenderingIntent::Saturation,
    ] {
        let transform = IccTransform::new(&profile, &profile, intent).unwrap();
        assert_eq!(matrix(&transform), crate::color::matrix::IDENTITY);
        assert_eq!(offset(&transform), [0.0; 3]);
    }
}

#[test]
fn profile_ranges_sharing_padding_and_limits_are_checked_before_use() {
    let original = profile_bytes(&rgb_tags());
    for length in 0..original.len() {
        let mut bytes = original[..length].to_vec();
        if length >= 4 {
            put32(&mut bytes, 0, length as u32);
        }
        assert!(parse(bytes).is_err(), "truncation {length}");
    }
    assert!(matches!(
        IccProfile::parse(
            original.clone().into(),
            IccLimits {
                max_tags: 6,
                ..Default::default()
            }
        ),
        Err(IccError::Limit {
            resource: "tag count",
            required: 7,
            limit: 6
        })
    ));
    assert!(matches!(
        IccProfile::parse(
            original.clone().into(),
            IccLimits {
                max_profile_bytes: 131,
                ..Default::default()
            }
        ),
        Err(IccError::Limit {
            resource: "profile bytes",
            ..
        })
    ));
    for (offset, value) in [
        (136, 0),
        (136, u32::MAX),
        (140, u32::MAX),
        (128, u32::MAX),
        (64, 4),
        (68, 0),
        (100, 1),
    ] {
        let mut bytes = original.clone();
        put32(&mut bytes, offset, value);
        assert!(parse(bytes).is_err(), "corrupt field {offset}");
    }
    let mut duplicate = original.clone();
    duplicate[144..148].copy_from_slice(b"wtpt");
    assert!(matches!(
        parse(duplicate),
        Err(IccError::DuplicateTag { .. })
    ));
    let first = u32::from_be_bytes(original[136..140].try_into().unwrap());
    let mut overlap = original.clone();
    put32(&mut overlap, 148, first);
    put32(&mut overlap, 152, 24);
    assert!(matches!(parse(overlap), Err(IccError::TagOverlap { .. })));
    // v2 permits gaps. Shared complete payloads remain one physical element.
    let mut shared = original;
    put32(&mut shared, 8, 0x0240_0000);
    put32(&mut shared, 148, first);
    let parsed = parse(shared).unwrap();
    assert_eq!(
        parsed.tag_data(IccSignature(*b"rXYZ")),
        parsed.tag_data(IccSignature(*b"wtpt"))
    );
}

#[test]
fn curve_inversion_accepts_flat_intervals_and_rejects_undefined_inverses() {
    for (values, expected) in [
        (
            vec![0, 0, 10000, 10000, 65535, 65535],
            Some(IccInverseDirection::Increasing),
        ),
        (
            vec![65535, 65535, 10000, 10000, 0, 0],
            Some(IccInverseDirection::Decreasing),
        ),
        (vec![0, 40000, 20000, 65535], None),
        (vec![123; 6], None),
    ] {
        let mut curve = element(b"curv", &[values.len() as u32]);
        for value in values {
            curve.extend_from_slice(&u16::to_be_bytes(value));
        }
        let mut tags = rgb_tags();
        tags[4].1 = curve;
        let profile = parse(profile_bytes(&tags)).unwrap();
        let forward = profile
            .matrix_trc(IccDirection::DeviceToPcs, IccRenderingIntent::Relative)
            .unwrap();
        assert_eq!(forward.curves()[0].inverse_direction().ok(), expected);
        assert_eq!(
            profile
                .matrix_trc(IccDirection::PcsToDevice, IccRenderingIntent::Relative)
                .is_ok(),
            expected.is_some()
        );
    }
    for (function, parameters) in [
        (0, vec![0]),
        (1, vec![65536, 0, 0]),
        (3, vec![65536, 65536, -65536i32 as u32, 65536, 0]),
    ] {
        let mut tags = rgb_tags();
        let mut payload = vec![function << 16];
        payload.extend(parameters);
        tags[4].1 = element(b"para", &payload);
        let profile = parse(profile_bytes(&tags)).unwrap();
        assert!(matches!(
            profile.matrix_trc(IccDirection::DeviceToPcs, IccRenderingIntent::Relative),
            Err(IccError::CurveParameters { .. })
        ));
    }
}

#[test]
fn gray_uses_pcs_y_and_chad_is_preserved_without_double_adaptation() {
    let mut tags = rgb_tags();
    tags.push((
        *b"chad",
        element(b"sf32", &[65536, 4096, 0, 0, 65536, 0, 0, 0, 65536]),
    ));
    let rgb = parse(profile_bytes(&tags)).unwrap();
    let mut bytes = profile_bytes(&[tags[0].clone(), (*b"kTRC", element(b"curv", &[0]))]);
    bytes[16..20].copy_from_slice(b"GRAY");
    let gray = parse(bytes).unwrap();
    let transform = IccTransform::new(&rgb, &gray, IccRenderingIntent::Relative).unwrap();
    assert_eq!(
        matrix(&transform)[0],
        [10000.0 / 65536.0, 50000.0 / 65536.0, 5536.0 / 65536.0]
    );
    assert_eq!(
        transform
            .source()
            .profile()
            .unwrap()
            .matrix_trc()
            .unwrap()
            .chromatic_adaptation()
            .unwrap()[1],
        4096
    );
    assert_eq!(transform.target().channels(), 1);
    let reverse = IccTransform::new(&gray, &rgb, IccRenderingIntent::Relative).unwrap();
    assert_eq!(
        reverse
            .source()
            .profile()
            .unwrap()
            .matrix_trc()
            .unwrap()
            .matrix()[0][0],
        f64::from(0xf6d6) / 65536.0
    );
}

#[test]
fn a_singular_input_matrix_is_usable_forward_but_has_no_inverse() {
    let target = parse(profile_bytes(&rgb_tags())).unwrap();
    let mut tags = rgb_tags();
    tags[1].1 = element(b"XYZ ", &[0, 0, 0]);
    let source = parse(profile_bytes(&tags)).unwrap();
    let transform = IccTransform::new(&source, &target, IccRenderingIntent::Relative).unwrap();
    assert!(matrix(&transform).iter().all(|row| row[0] == 0.0));
    assert_eq!(
        IccTransform::new(&target, &source, IccRenderingIntent::Relative),
        Err(IccError::Matrix)
    );
    assert!(
        IccTransform::to_linear_rgb(
            &source,
            crate::RgbColorSpace::Bt709,
            IccRenderingIntent::Relative
        )
        .is_ok()
    );
    assert_eq!(
        IccTransform::from_linear_rgb(
            crate::RgbColorSpace::Bt709,
            &source,
            IccRenderingIntent::Relative
        ),
        Err(IccError::Matrix)
    );
}

#[test]
fn linear_connections_honor_directional_tag_priority_intent_and_geometry() {
    use crate::RgbColorSpace;
    for (tag, forward) in [
        (*b"A2B0", true),
        (*b"A2B1", true),
        (*b"B2A0", false),
        (*b"B2A1", false),
    ] {
        let mut tags = rgb_tags();
        tags.push((tag, element(b"mAB ", &[])));
        let profile = parse(profile_bytes(&tags)).unwrap();
        let to = IccTransform::to_linear_rgb(
            &profile,
            RgbColorSpace::Bt709,
            IccRenderingIntent::Relative,
        );
        let from = IccTransform::from_linear_rgb(
            RgbColorSpace::Bt709,
            &profile,
            IccRenderingIntent::Relative,
        );
        let (rejected, accepted) = if forward { (to, from) } else { (from, to) };
        assert_eq!(
            rejected,
            Err(IccError::TransformTag {
                tag: IccSignature(tag)
            })
        );
        assert!(accepted.is_ok());
    }
    let profile = parse(profile_bytes(&rgb_tags())).unwrap();
    for intent in [
        IccRenderingIntent::Perceptual,
        IccRenderingIntent::Absolute,
        IccRenderingIntent::Saturation,
    ] {
        let to = IccTransform::to_linear_rgb(&profile, RgbColorSpace::Bt709, intent).unwrap();
        let from = IccTransform::from_linear_rgb(RgbColorSpace::Bt709, &profile, intent).unwrap();
        let product = crate::color::matrix::multiply(matrix(&from), matrix(&to));
        for (r, row) in product.into_iter().enumerate() {
            for (c, value) in row.into_iter().enumerate() {
                assert!((value - f64::from(r == c)).abs() < 1e-12);
            }
        }
        assert_eq!(offset(&to), [0.0; 3]);
        assert_eq!(offset(&from), [0.0; 3]);
    }
    assert_eq!(
        IccTransform::to_linear_rgb(
            &profile,
            RgbColorSpace::Undefined,
            IccRenderingIntent::Relative
        ),
        Err(IccError::LinearRgb(
            crate::ColorMatrixError::UndefinedTarget
        ))
    );
    assert_eq!(
        IccTransform::from_linear_rgb(
            RgbColorSpace::Undefined,
            &profile,
            IccRenderingIntent::Relative
        ),
        Err(IccError::LinearRgb(
            crate::ColorMatrixError::UndefinedSource
        ))
    );
}

#[test]
fn every_intent_selects_its_mpe_then_its_lut_then_the_default_lut() {
    for direction in [IccDirection::DeviceToPcs, IccDirection::PcsToDevice] {
        for intent in [
            IccRenderingIntent::Perceptual,
            IccRenderingIntent::Relative,
            IccRenderingIntent::Saturation,
            IccRenderingIntent::Absolute,
        ] {
            let (mut mpe, mut lut, base) = match direction {
                IccDirection::DeviceToPcs => (*b"D2B0", *b"A2B0", *b"A2B0"),
                IccDirection::PcsToDevice => (*b"B2D0", *b"B2A0", *b"B2A0"),
            };
            mpe[3] += intent as u8;
            lut[3] += if intent == IccRenderingIntent::Absolute {
                1
            } else {
                intent as u8
            };
            let mut tags = rgb_tags();
            tags.push((base, element(b"mAB ", &[])));
            if lut != base {
                tags.push((lut, element(b"mAB ", &[])));
            }
            tags.push((mpe, self::mpe::identity_mpe(3)));
            for expected in [mpe, lut, base] {
                if !tags.iter().any(|(signature, _)| *signature == expected) {
                    continue;
                }
                let profile = parse(profile_bytes(&tags)).unwrap();
                assert_eq!(
                    profile.matrix_trc(direction, intent),
                    Err(IccError::TransformTag {
                        tag: IccSignature(expected)
                    })
                );
                tags.retain(|(signature, _)| *signature != expected);
            }
            assert!(
                parse(profile_bytes(&tags))
                    .unwrap()
                    .matrix_trc(direction, intent)
                    .is_ok()
            );
        }
    }
}
