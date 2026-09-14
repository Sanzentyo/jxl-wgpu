use super::*;

fn header(kind: &[u8; 4], p: u16, q: u16) -> Vec<u8> {
    element(kind, &[(u32::from(p) << 16) | u32::from(q)])
}
fn matrix_element(p: u16, q: u16) -> Vec<u8> {
    let mut bytes = header(b"matf", p, q);
    for row in 0..q {
        for c in 0..p {
            bytes.extend_from_slice(&f32::from(u8::from(row == c)).to_be_bytes());
        }
    }
    bytes.resize(bytes.len() + usize::from(q) * 4, 0);
    bytes
}
fn container(kind: &[u8; 4], p: u16, q: u16, elements: &[Vec<u8>], order: &[usize]) -> Vec<u8> {
    let mut bytes = header(kind, p, q);
    if kind == b"mpet" {
        bytes.extend_from_slice(&(order.len() as u32).to_be_bytes());
    }
    let start = bytes.len();
    bytes.resize(start + order.len() * 8, 0);
    let mut positions = Vec::new();
    for value in elements {
        positions.push((bytes.len() as u32, value.len() as u32));
        bytes.extend_from_slice(value);
        bytes.resize(bytes.len().next_multiple_of(4), 0);
    }
    for (i, &index) in order.iter().enumerate() {
        let (offset, size) = positions[index];
        put32(&mut bytes, start + i * 8, offset);
        put32(&mut bytes, start + i * 8 + 4, size);
    }
    bytes
}
pub(super) fn identity_mpe(channels: u16) -> Vec<u8> {
    container(
        b"mpet",
        channels,
        channels,
        &[matrix_element(channels, channels)],
        &[0],
    )
}
fn select(payload: Vec<u8>) -> Result<IccProfileProgram, IccError> {
    let mut tags = rgb_tags();
    tags.push((*b"D2B1", payload));
    parse(profile_bytes(&tags))?.select(IccDirection::DeviceToPcs, IccRenderingIntent::Relative)
}

#[test]
fn processing_order_is_independent_of_storage_and_shared_elements_remain_shared() {
    let mut a = matrix_element(3, 3);
    put32(&mut a, 48, 0.125_f32.to_bits());
    let b = matrix_element(3, 3);
    let selected = select(container(b"mpet", 3, 3, &[a, b], &[1, 0, 1])).unwrap();
    let stages = selected.program().stages();
    assert_eq!(stages.len(), 3);
    let IccStage::Matrix(first) = &stages[0] else {
        panic!()
    };
    let IccStage::Matrix(middle) = &stages[1] else {
        panic!()
    };
    let IccStage::Matrix(last) = &stages[2] else {
        panic!()
    };
    assert_eq!(middle.offset(), &[0.125, 0.0, 0.0]);
    assert_eq!(first.matrix().as_ptr(), last.matrix().as_ptr());
    assert_eq!(selected.tag(), Some(IccSignature(*b"D2B1")));
    assert!(selected.matrix_trc().is_none());
}

#[test]
fn unknown_mpe_elements_fall_back_but_known_broken_elements_do_not() {
    let mut unknown = matrix_element(3, 3);
    unknown[..4].copy_from_slice(b"new!");
    let selected = select(container(b"mpet", 3, 3, &[unknown.clone()], &[0])).unwrap();
    assert!(selected.matrix_trc().is_some());
    let mut tags = rgb_tags();
    tags.push((*b"D2B1", container(b"mpet", 3, 3, &[unknown], &[0])));
    tags.push((*b"A2B1", element(b"mAB ", &[])));
    assert!(
        matches!(parse(profile_bytes(&tags)).unwrap().select(IccDirection::DeviceToPcs, IccRenderingIntent::Relative), Err(IccError::TransformTag { tag: IccSignature(signature) }) if signature == *b"A2B1")
    );
    for bits in [
        f32::INFINITY.to_bits(),
        f32::NAN.to_bits(),
        f32::NEG_INFINITY.to_bits(),
    ] {
        let mut broken = matrix_element(3, 3);
        put32(&mut broken, 12, bits);
        assert!(matches!(
            select(container(b"mpet", 3, 3, &[broken], &[0])),
            Err(IccError::Invalid {
                field: "non-finite float",
                ..
            })
        ));
    }
}

#[test]
fn mpe_truncations_positions_channels_and_resource_bounds_fail_before_execution() {
    let payload = identity_mpe(3);
    for length in 0..payload.len() {
        let result = select(payload[..length].to_vec());
        assert!(result.is_err(), "truncation {length}");
    }
    for (offset, value) in [
        (12, 0),
        (12, u32::MAX),
        (16, 16),
        (16, 25),
        (20, u32::MAX),
        (32, 0x00020003),
        (28, 1),
    ] {
        let mut bytes = payload.clone();
        put32(&mut bytes, offset, value);
        assert!(select(bytes).is_err(), "field {offset}={value}");
    }
    let mut tags = rgb_tags();
    tags.push((*b"D2B1", payload));
    for limits in [
        IccLimits {
            max_processing_elements: 0,
            ..Default::default()
        },
        IccLimits {
            max_processing_channels: 2,
            ..Default::default()
        },
    ] {
        let profile = IccProfile::parse(profile_bytes(&tags).into(), limits).unwrap();
        assert!(matches!(
            profile.select(IccDirection::DeviceToPcs, IccRenderingIntent::Relative),
            Err(IccError::Limit { .. })
        ));
    }
    let mismatched = container(
        b"mpet",
        3,
        3,
        &[matrix_element(3, 4), matrix_element(3, 3)],
        &[0, 1],
    );
    assert!(matches!(
        select(mismatched),
        Err(IccError::Invalid {
            field: "MPE channel continuity",
            ..
        })
    ));
}

fn formula(function: u16, values: &[f32]) -> Vec<u8> {
    let mut bytes = element(b"parf", &[u32::from(function) << 16]);
    for value in values {
        bytes.extend_from_slice(&value.to_be_bytes());
    }
    bytes
}
fn segmented() -> Vec<u8> {
    let mut curve = element(b"curf", &[3 << 16, 0_f32.to_bits(), 1_f32.to_bits()]);
    curve.extend(formula(0, &[1.0, 2.0, 0.0, -0.25]));
    curve.extend(element(
        b"samf",
        &[2, 0.5_f32.to_bits(), 1.25_f32.to_bits()],
    ));
    curve.extend(formula(0, &[1.0, 2.0, -0.75, 0.0]));
    curve
}
#[test]
fn shared_segmented_curves_reconstruct_the_implicit_sample_without_clipping() {
    let cvst = container(b"cvst", 3, 3, &[segmented()], &[0, 0, 0]);
    let selected = select(container(b"mpet", 3, 3, &[cvst], &[0])).unwrap();
    let IccStage::SegmentedCurves(curves) = &selected.program().stages()[0] else {
        panic!()
    };
    assert_eq!(curves[0].segments().as_ptr(), curves[2].segments().as_ptr());
    let IccCurveSegmentKind::Samples(samples) = &curves[0].segments()[1].kind else {
        panic!()
    };
    assert_eq!(&**samples, &[-0.25, 0.5, 1.25]);
    assert_eq!(curves[0].segments()[0].lower, f32::NEG_INFINITY);
    assert_eq!(curves[0].segments()[2].upper, f32::INFINITY);
}

#[test]
fn clut_dimensions_payloads_and_sampled_curves_are_bounded_and_validated() {
    let mut clut = header(b"clut", 3, 3);
    clut.extend([255; 16]);
    assert!(select(container(b"mpet", 3, 3, &[clut.clone()], &[0])).is_err());
    clut[15..28].fill(0);
    assert!(matches!(
        select(container(b"mpet", 3, 3, &[clut], &[0])),
        Err(IccError::Limit {
            resource: "CLUT values",
            ..
        })
    ));
    for (offset, value) in [
        (8, 0),
        (12, f32::NAN.to_bits()),
        (16, (-1_f32).to_bits()),
        (56, u32::MAX),
    ] {
        let mut curve = segmented();
        put32(&mut curve, offset, value);
        assert!(
            select(container(
                b"mpet",
                3,
                3,
                &[container(b"cvst", 3, 3, &[curve], &[0, 0, 0])],
                &[0]
            ))
            .is_err()
        );
    }
    let mut pass = header(b"bACS", 3, 3);
    pass.extend_from_slice(b"test");
    let selected = select(container(b"mpet", 3, 3, &[pass], &[0])).unwrap();
    assert!(selected.program().stages().is_empty());
}

#[test]
fn formula_domains_are_checked_even_when_all_stored_parameters_are_finite() {
    for (function, parameters, accepted) in [
        (0, vec![0.5, 1.0, 0.0, 0.0], false),
        (0, vec![2.0, 1.0, 0.0, -0.5], true),
        (0, vec![-1.0, 1.0, 0.0, 0.0], false),
        (1, vec![1.0, 1.0, 1.0, 0.0, 0.0], false),
        (1, vec![2.0, 1.0, 1.0, 1.0, 0.0], true),
        (2, vec![1.0, -2.0, 1.0, 0.0, 0.0], false),
    ] {
        let mut curve = element(b"curf", &[1 << 16]);
        curve.extend(formula(function, &parameters));
        let set = container(b"cvst", 3, 3, &[curve], &[0, 0, 0]);
        assert_eq!(
            select(container(b"mpet", 3, 3, &[set], &[0])).is_ok(),
            accepted,
            "formula {function}: {parameters:?}"
        );
    }
}

#[test]
fn absolute_mpe_connections_do_not_apply_media_white_twice() {
    let build = |tag, white: [u32; 3]| {
        let mut tags = rgb_tags();
        tags[0].1 = element(b"XYZ ", &white);
        tags.push((tag, identity_mpe(3)));
        parse(profile_bytes(&tags)).unwrap()
    };
    let source = build(*b"D2B3", [52000, 59000, 43000]);
    let target = build(*b"B2D3", [47000, 54000, 39000]);
    let absolute = IccTransform::new(&source, &target, IccRenderingIntent::Absolute).unwrap();
    assert!(
        absolute.program().stages().is_empty(),
        "both methods already use absolute PCS"
    );
    let mut tags = rgb_tags();
    tags[0].1 = element(b"XYZ ", &[52000, 59000, 43000]);
    let relative_source = parse(profile_bytes(&tags)).unwrap();
    let mixed = IccTransform::new(&relative_source, &target, IccRenderingIntent::Absolute).unwrap();
    let base = relative_source
        .matrix_trc(IccDirection::DeviceToPcs, IccRenderingIntent::Relative)
        .unwrap();
    for (r, d50) in [0.9642, 1.0, 0.8249].into_iter().enumerate() {
        for c in 0..3 {
            let expected = base.matrix()[r][c] * f64::from(base.media_white()[r]) / 65536.0 / d50;
            assert!((matrix(&mixed)[r][c] - expected).abs() < 1e-15);
        }
    }
}
