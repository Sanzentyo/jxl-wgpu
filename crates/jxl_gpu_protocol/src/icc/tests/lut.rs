use super::*;

fn header(kind: &[u8; 4], inputs: u8, outputs: u8) -> Vec<u8> {
    let mut bytes = element(kind, &[]);
    bytes.resize(32, 0);
    bytes[8] = inputs;
    bytes[9] = outputs;
    bytes
}

fn curves(count: u8, gamma: u16) -> Vec<u8> {
    (0..count)
        .flat_map(|_| {
            let mut bytes = element(b"curv", &[1]);
            bytes.extend(gamma.to_be_bytes());
            bytes.resize(16, 0);
            bytes
        })
        .collect()
}

pub(super) fn identity_lut(direction: IccDirection) -> Vec<u8> {
    let mut bytes = header(
        if direction == IccDirection::DeviceToPcs {
            b"mAB "
        } else {
            b"mBA "
        },
        3,
        3,
    );
    put32(&mut bytes, 12, 32);
    for _ in 0..3 {
        bytes.extend(element(b"curv", &[0]));
    }
    bytes
}

fn multi(direction: IccDirection, channels: u8, matrix: bool, clut: bool) -> Vec<u8> {
    let reverse = direction == IccDirection::PcsToDevice;
    let (inputs, outputs) = if reverse {
        (3, channels)
    } else {
        (channels, 3)
    };
    let mut bytes = header(if reverse { b"mBA " } else { b"mAB " }, inputs, outputs);
    let mut elements = vec![(12, curves(3, 384))];
    if matrix {
        let mut coefficients = vec![0; 48];
        for i in [0, 4, 8] {
            put32(&mut coefficients, i * 4, 65536);
        }
        put32(&mut coefficients, 36, (-16384_i32) as u32);
        elements.push((16, coefficients));
        elements.push((20, curves(3, 512)));
    }
    if clut {
        let mut table = vec![0; 20];
        table[..usize::from(inputs)].fill(2);
        table[16] = 1;
        table.resize(20 + (1_usize << inputs) * usize::from(outputs), 127);
        elements.push((24, table));
        elements.push((28, curves(channels, 128)));
    }
    // Reverse physical storage; readers must use each named offset and execution order.
    for (field, data) in elements.into_iter().rev() {
        let offset = bytes.len() as u32;
        put32(&mut bytes, field, offset);
        bytes.extend(data);
        bytes.resize(bytes.len().next_multiple_of(4), 0);
    }
    bytes
}

fn tables(wide: bool) -> Vec<u8> {
    let mut bytes = header(if wide { b"mft2" } else { b"mft1" }, 3, 3);
    bytes.resize(if wide { 52 } else { 48 }, 0);
    bytes[10] = 2;
    for i in [0, 4, 8] {
        put32(&mut bytes, 12 + i * 4, 65536);
    }
    let (input, output) = if wide { (17_u16, 33_u16) } else { (256, 256) };
    if wide {
        bytes[48..50].copy_from_slice(&input.to_be_bytes());
        bytes[50..52].copy_from_slice(&output.to_be_bytes());
    }
    for (entries, count) in [(input, 3), (8, 3), (output, 3)] {
        for _ in 0..count {
            for n in 0..entries {
                if wide {
                    bytes.extend(
                        ((u32::from(n) * 65535 / u32::from(entries - 1)) as u16).to_be_bytes(),
                    );
                } else {
                    bytes.push((u32::from(n) * 255 / u32::from(entries - 1)) as u8);
                }
            }
        }
    }
    bytes
}

fn select_lut(
    payload: Vec<u8>,
    direction: IccDirection,
    space: [u8; 4],
    lab: bool,
    limits: IccLimits,
) -> Result<IccProfileProgram, IccError> {
    let mut tags = rgb_tags();
    tags.push((
        if direction == IccDirection::DeviceToPcs {
            *b"A2B1"
        } else {
            *b"B2A1"
        },
        payload,
    ));
    let mut bytes = profile_bytes(&tags);
    bytes[16..20].copy_from_slice(&space);
    if lab {
        bytes[20..24].copy_from_slice(b"Lab ");
    }
    IccProfile::parse(bytes.into(), limits)?.select(direction, IccRenderingIntent::Relative)
}

#[test]
fn table_luts_retain_sample_precision_pcs_scaling_and_forward_curve_order() {
    for wide in [false, true] {
        for direction in [IccDirection::DeviceToPcs, IccDirection::PcsToDevice] {
            let selected = select_lut(
                tables(wide),
                direction,
                *b"RGB ",
                false,
                IccLimits::default(),
            )
            .unwrap();
            let stages = selected.program().stages();
            let curves = stages
                .iter()
                .filter_map(|stage| match stage {
                    IccStage::Curves {
                        curves,
                        inverse: false,
                    } => Some(curves),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(curves.len(), 2);
            for (set, count) in curves
                .into_iter()
                .zip(if wide { [17, 33] } else { [256, 256] })
            {
                for curve in &**set {
                    let IccCurveKind::Sampled(samples) = curve.kind() else {
                        panic!()
                    };
                    assert_eq!(samples.len(), count);
                    assert_eq!((samples[0], samples[count - 1]), (0, 65535));
                }
            }
            let clut = stages
                .iter()
                .find_map(|stage| match stage {
                    IccStage::Clut(clut) => Some(clut),
                    _ => None,
                })
                .unwrap();
            assert_eq!(clut.grid(), &[2, 2, 2]);
            assert_eq!(clut.values().len(), 24);
            assert_eq!(clut.interpolation(), IccClutInterpolation::Tetrahedral);
            let boundary = if direction == IccDirection::DeviceToPcs {
                stages.last().unwrap()
            } else {
                &stages[0]
            };
            let IccStage::Matrix(boundary) = boundary else {
                panic!()
            };
            assert_eq!(
                boundary.matrix()[0],
                if direction == IccDirection::DeviceToPcs {
                    65535.0 / 32768.0
                } else {
                    32768.0 / 65535.0
                }
            );
            assert_eq!(
                boundary.clamp_output(),
                direction == IccDirection::PcsToDevice
            );
        }
    }
}

#[test]
fn all_ab_stage_combinations_use_their_declared_direction_and_clamped_matrix() {
    for direction in [IccDirection::DeviceToPcs, IccDirection::PcsToDevice] {
        for matrix in [false, true] {
            for clut in [false, true] {
                for channels in if clut { &[1, 3, 4][..] } else { &[3][..] } {
                    let space = match channels {
                        1 => *b"GRAY",
                        3 => *b"RGB ",
                        _ => *b"CMYK",
                    };
                    let selected = select_lut(
                        multi(direction, *channels, matrix, clut),
                        direction,
                        space,
                        true,
                        IccLimits::default(),
                    )
                    .unwrap();
                    let mut actual = Vec::new();
                    for stage in selected.program().stages() {
                        match stage {
                            IccStage::Curves { curves, inverse } => {
                                assert!(!inverse);
                                let IccCurveKind::Gamma(gamma) = curves[0].kind() else {
                                    panic!()
                                };
                                actual.push(*gamma);
                            }
                            IccStage::Clut(table) => {
                                assert_eq!(
                                    table.interpolation(),
                                    if direction == IccDirection::PcsToDevice {
                                        IccClutInterpolation::Multilinear
                                    } else {
                                        IccClutInterpolation::Tetrahedral
                                    }
                                );
                                actual.push(0);
                            }
                            IccStage::Matrix(value) if value.offset()[0] == -0.25 => {
                                assert!(value.clamp_output());
                                actual.push(1);
                            }
                            _ => {}
                        }
                    }
                    let mut expected = Vec::new();
                    if clut {
                        expected.extend([128, 0]);
                    }
                    if matrix {
                        expected.extend([512, 1]);
                    }
                    expected.push(384);
                    if direction == IccDirection::PcsToDevice {
                        expected.reverse();
                    }
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

#[test]
fn ab_curve_sets_can_share_complete_curves_and_suffixes_but_not_matrix_storage() {
    let direction = IccDirection::DeviceToPcs;
    let mut bytes = multi(direction, 1, true, true);
    let b = u32::from_be_bytes(bytes[12..16].try_into().unwrap());
    for c in 0..3 {
        let start = b as usize + c * 16;
        put32(&mut bytes, start + 8, 2);
        bytes[start + 12..start + 16].copy_from_slice(&[0, 0, 255, 255]);
    }
    put32(&mut bytes, 20, b);
    put32(&mut bytes, 28, b + 32);
    let selected = select_lut(
        bytes.clone(),
        direction,
        *b"GRAY",
        false,
        IccLimits::default(),
    )
    .unwrap();
    let sets = selected
        .program()
        .stages()
        .iter()
        .filter_map(|stage| match stage {
            IccStage::Curves { curves, .. } => Some(curves),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(sets[0][0], sets[2][2]);
    assert_eq!(sets[1], sets[2]);
    let samples = |curve: &IccCurve| {
        let IccCurveKind::Sampled(samples) = curve.kind() else {
            panic!("shared sampled payload")
        };
        samples.as_ptr()
    };
    assert_eq!(samples(&sets[0][0]), samples(&sets[2][2]));
    for (m, b) in sets[1].iter().zip(sets[2].iter()) {
        assert_eq!(samples(m), samples(b));
    }
    put32(&mut bytes, 16, b);
    assert!(matches!(
        select_lut(bytes, direction, *b"GRAY", false, IccLimits::default()),
        Err(IccError::Invalid {
            field: "overlapping LUT elements",
            ..
        })
    ));
}

#[test]
fn lut_truncation_counts_offsets_padding_and_limits_fail_before_execution() {
    let direction = IccDirection::DeviceToPcs;
    for bytes in [tables(false), tables(true), multi(direction, 3, true, true)] {
        // Only trailing tag-alignment padding may be absent.
        for length in 0..bytes.len() - 3 {
            assert!(
                select_lut(
                    bytes[..length].to_vec(),
                    direction,
                    *b"RGB ",
                    false,
                    IccLimits::default()
                )
                .is_err(),
                "truncation {length}"
            );
        }
        for limits in [
            IccLimits {
                max_processing_elements: 0,
                ..Default::default()
            },
            IccLimits {
                max_processing_channels: 2,
                ..Default::default()
            },
            IccLimits {
                max_clut_values: 2,
                ..Default::default()
            },
        ] {
            assert!(matches!(
                select_lut(bytes.clone(), direction, *b"RGB ", false, limits),
                Err(IccError::Limit { .. })
            ));
        }
    }
    for (field, value) in [
        (12, 0),
        (12, 31),
        (16, 0),
        (20, 0),
        (24, 0),
        (28, 0),
        (28, u32::MAX),
    ] {
        let mut bytes = multi(direction, 3, true, true);
        put32(&mut bytes, field, value);
        assert!(select_lut(bytes, direction, *b"RGB ", false, IccLimits::default()).is_err());
    }
    let mut bytes = multi(direction, 3, false, false);
    bytes[46] = 1;
    assert!(matches!(
        select_lut(bytes, direction, *b"RGB ", false, IccLimits::default()),
        Err(IccError::Invalid {
            field: "LUT element padding",
            ..
        })
    ));
    for count in [0_u16, 1, 4097] {
        let mut bytes = tables(true);
        bytes[48..50].copy_from_slice(&count.to_be_bytes());
        assert!(matches!(
            select_lut(bytes, direction, *b"RGB ", false, IccLimits::default()),
            Err(IccError::Invalid {
                field: "LUT table entries",
                ..
            })
        ));
    }
    let mut bytes = tables(true);
    put32(&mut bytes, 12, 32768);
    assert!(matches!(
        select_lut(bytes, direction, *b"RGB ", false, IccLimits::default()),
        Err(IccError::Invalid {
            field: "LUT matrix requires PCS XYZ input",
            ..
        })
    ));
}

#[test]
fn legacy_xyz_device_signature_does_not_turn_input_data_into_pcs() {
    assert!(
        select_lut(
            tables(true),
            IccDirection::DeviceToPcs,
            *b"XYZ ",
            true,
            IccLimits::default()
        )
        .is_ok()
    );
    let mut bytes = tables(true);
    put32(&mut bytes, 12, 32768);
    assert!(matches!(
        select_lut(
            bytes,
            IccDirection::DeviceToPcs,
            *b"XYZ ",
            true,
            IccLimits::default(),
        ),
        Err(IccError::Invalid {
            field: "LUT matrix requires PCS XYZ input",
            ..
        })
    ));
}

#[test]
fn v2_luts_plan_gpu_black_detection_only_for_connections_that_need_it() {
    for wide in [false, true] {
        let mut tags = rgb_tags();
        tags.extend([(*b"A2B0", tables(wide)), (*b"B2A0", tables(wide))]);
        let mut bytes = profile_bytes(&tags);
        put32(&mut bytes, 8, 0x0240_0000);
        let v2 = IccProfile::parse(bytes.into(), Default::default()).unwrap();
        let v4 = IccProfile::parse(profile_bytes(&rgb_tags()).into(), Default::default()).unwrap();
        for intent in [
            IccRenderingIntent::Perceptual,
            IccRenderingIntent::Relative,
            IccRenderingIntent::Saturation,
            IccRenderingIntent::Absolute,
        ] {
            assert!(IccTransform::new(&v2, &v2, intent).is_ok());
            assert!(IccTransform::new(&v4, &v2, intent).is_ok());
            assert!(
                IccTransform::from_linear_rgb(crate::RgbColorSpace::Bt709, &v2, intent).is_ok()
            );
            for selected in [
                IccTransform::new(&v2, &v4, intent),
                IccTransform::to_linear_rgb(&v2, crate::RgbColorSpace::Bt709, intent),
            ] {
                let transform = selected.unwrap();
                let probes = transform
                    .program()
                    .stages()
                    .iter()
                    .filter_map(|stage| match stage {
                        IccStage::BlackPointConnection(connection) => Some(connection),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if matches!(
                    intent,
                    IccRenderingIntent::Perceptual | IccRenderingIntent::Saturation
                ) {
                    assert_eq!(probes.len(), 1);
                    assert_eq!(probes[0].input(), &[0.0; 3]);
                    assert_eq!(
                        probes[0].source(),
                        v2.select(IccDirection::DeviceToPcs, intent)
                            .unwrap()
                            .program()
                    );
                } else {
                    assert!(probes.is_empty());
                }
            }
        }
    }
}

#[test]
fn v2_black_metadata_uses_device_endpoints_and_zero_for_unavailable_estimates() {
    let v4 = IccProfile::parse(profile_bytes(&rgb_tags()).into(), Default::default()).unwrap();
    for (space, endpoint) in [
        (*b"RGB ", Some([0.0; 3])),
        (*b"CMY ", Some([1.0; 3])),
        (*b"Lab ", Some([0.0, 128.0 / 255.0, 128.0 / 255.0])),
        (*b"XYZ ", None),
        (*b"3CLR", None),
    ] {
        let mut tags = rgb_tags();
        tags.push((*b"A2B0", tables(true)));
        let mut bytes = profile_bytes(&tags);
        put32(&mut bytes, 8, 0x0240_0000);
        bytes[16..20].copy_from_slice(&space);
        let source = IccProfile::parse(bytes.into(), Default::default()).unwrap();
        let transform = IccTransform::new(&source, &v4, IccRenderingIntent::Perceptual).unwrap();
        let probe = transform
            .program()
            .stages()
            .iter()
            .find_map(|stage| match stage {
                IccStage::BlackPointConnection(connection) => Some(connection),
                _ => None,
            });
        assert_eq!(
            probe.map(|connection| connection.input()),
            endpoint.as_ref().map(|values| values.as_slice())
        );
        if endpoint.is_none() {
            // This target's zero black makes the unavailable-estimate connection identical
            // to relative intent. Falling back to v4 reference black would change the program.
            assert_eq!(
                transform.program(),
                IccTransform::new(&source, &v4, IccRenderingIntent::Relative)
                    .unwrap()
                    .program()
            );
        }
    }
}

#[test]
fn malformed_selected_lut_never_falls_back_to_another_valid_method() {
    assert_eq!(
        select_lut(
            element(b"????", &[]),
            IccDirection::DeviceToPcs,
            *b"RGB ",
            false,
            IccLimits::default(),
        )
        .unwrap_err(),
        IccError::TagType {
            tag: IccSignature(*b"A2B1"),
            kind: IccSignature(*b"????"),
        }
    );
    let mut tags = rgb_tags();
    tags.push((*b"A2B0", identity_lut(IccDirection::DeviceToPcs)));
    tags.push((*b"A2B1", header(b"mAB ", 3, 3)));
    let profile = IccProfile::parse(profile_bytes(&tags).into(), Default::default()).unwrap();
    assert!(
        profile
            .select(IccDirection::DeviceToPcs, IccRenderingIntent::Perceptual)
            .is_ok()
    );
    assert!(matches!(
        profile.select(IccDirection::DeviceToPcs, IccRenderingIntent::Relative),
        Err(IccError::Invalid {
            field: "LUT processing combination",
            ..
        })
    ));
    let reversed = identity_lut(IccDirection::PcsToDevice);
    assert!(matches!(
        select_lut(
            reversed,
            IccDirection::DeviceToPcs,
            *b"RGB ",
            false,
            IccLimits::default()
        ),
        Err(IccError::TagType { .. })
    ));
}
