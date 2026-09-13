use super::*;
use jxl_gpu_protocol::icc::{IccDirection, IccError, IccSignature};
use jxl_wgpu::ResidentIccError;

fn sampled_profile(samples: &[u16]) -> IccProfile {
    let mut curve = b"curv\0\0\0\0".to_vec();
    curve.extend_from_slice(&(samples.len() as u32).to_be_bytes());
    for sample in samples {
        curve.extend_from_slice(&sample.to_be_bytes());
    }
    gray_curve_profile(curve)
}

fn parametric_profile(function: u16, parameters: &[f64]) -> IccProfile {
    let mut curve = b"para\0\0\0\0".to_vec();
    curve.extend_from_slice(&function.to_be_bytes());
    curve.extend_from_slice(&[0; 2]);
    for parameter in parameters {
        curve.extend_from_slice(&((*parameter * 65536.0).round() as i32).to_be_bytes());
    }
    gray_curve_profile(curve)
}

fn gray_curve_profile(curve: Vec<u8>) -> IccProfile {
    let original = profile("gray");
    let mut tags = original
        .tags()
        .iter()
        .map(|tag| {
            (
                tag.signature,
                original.tag_data(tag.signature).unwrap().to_vec(),
            )
        })
        .collect::<Vec<_>>();
    tags.iter_mut()
        .find(|(tag, _)| *tag == IccSignature(*b"kTRC"))
        .unwrap()
        .1 = curve;
    let mut bytes = original.bytes()[..128].to_vec();
    bytes.extend_from_slice(&(tags.len() as u32).to_be_bytes());
    bytes.resize(132 + tags.len() * 12, 0);
    for (i, (signature, data)) in tags.iter().enumerate() {
        let entry = 132 + 12 * i;
        bytes[entry..entry + 4].copy_from_slice(&signature.0);
        let offset = bytes.len() as u32;
        bytes[entry + 4..entry + 8].copy_from_slice(&offset.to_be_bytes());
        bytes[entry + 8..entry + 12].copy_from_slice(&(data.len() as u32).to_be_bytes());
        bytes.extend_from_slice(data);
        bytes.resize(bytes.len().next_multiple_of(4), 0);
    }
    let length = bytes.len() as u32;
    bytes[..4].copy_from_slice(&length.to_be_bytes());
    IccProfile::parse(bytes.into(), Default::default()).unwrap()
}

#[test]
fn parametric_inverse_solves_offsets_gaps_and_clipped_plateaus_without_rounding_search() {
    let Some(backend) = backend() else {
        return;
    };
    let linear = sampled_profile(&[0, 65535]);
    let pipeline = ResidentIccPipeline::new(backend.device()).unwrap();
    let cases = [
        (
            0,
            vec![2.0],
            vec![0.0, 0.0625, 0.25, 1.0],
            vec![0.0, 0.25, 0.5, 1.0],
        ),
        (
            1,
            vec![2.0, 2.0, -1.0],
            vec![0.0, 0.0625, 0.25, 1.0],
            vec![0.5, 0.625, 0.75, 1.0],
        ),
        (
            2,
            vec![2.0, 1.0, 0.0, 0.25],
            vec![0.0, 0.25, 0.5, 1.0],
            vec![0.0, 0.0, 0.5, 0.75_f32.sqrt()],
        ),
        (
            3,
            vec![2.0, 1.0, 0.0, 0.0, 0.5],
            vec![0.0, 0.0625, 0.125, 0.1875, 0.25, 1.0],
            vec![0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
        ),
        (
            4,
            vec![2.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            vec![0.0, 0.0625, 0.25, 1.0],
            vec![0.0, 0.125, 0.5, 1.0],
        ),
        (
            4,
            vec![1.0, 1.0, 0.0, 2.0, 1.5, 0.0, -0.5],
            vec![-1.0, 0.0, 0.5, 1.0, 2.0],
            vec![0.25, 0.25, 0.5, 0.75, 0.75],
        ),
        (
            4,
            vec![1.0, 1.0, 0.0, -2.0, 1.5, 0.0, 1.5],
            vec![-1.0, 0.0, 0.5, 1.0, 2.0],
            vec![0.75, 0.75, 0.5, 0.25, 0.25],
        ),
    ];
    for (function, parameters, input, expected) in cases {
        let target = parametric_profile(function, &parameters);
        let transform = IccTransform::new(&linear, &target, IccRenderingIntent::Relative).unwrap();
        let actual = run(
            &backend,
            &pipeline,
            &transform,
            Extent2d::new(input.len() as u32, 1),
            &input,
            9,
        );
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 2e-7,
                "function {function}, {parameters:?}, sample {index}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn uploaded_profile_program_reuses_metadata_across_extents_pitches_and_abandoned_encoding() {
    let Some(backend) = backend() else {
        return;
    };
    let source_profile = parametric_profile(0, &[2.0]);
    let target_profile = sampled_profile(&[0, 65535]);
    let transform = IccTransform::new(
        &source_profile,
        &target_profile,
        IccRenderingIntent::Relative,
    )
    .unwrap();
    let program = ResidentIccProgram::new(backend.device(), &transform).unwrap();
    let pipeline = ResidentIccPipeline::new(backend.device()).unwrap();
    assert_eq!(
        (program.input_channels(), program.output_channels()),
        (1, 1)
    );
    assert_eq!(program.memory_plan().program_bytes, 184); // 80 header + 48 parametric + 48 sampled + 8 samples.
    let extent = Extent2d::new(2, 1);
    let source = Storage::new(&backend, extent, 1, Some(&[0.25, 0.5]), 4);
    let target = Storage::new(&backend, extent, 1, None, 8);
    let mut abandoned = backend.device().create_command_encoder(&Default::default());
    let uniform = pipeline
        .encode(
            backend.device(),
            &mut abandoned,
            &program,
            ResidentIccInputs {
                input: source.binding(),
                output: target.binding(),
                extent,
                input_planes: &source.planes,
                output_planes: &target.planes,
            },
        )
        .unwrap();
    drop(abandoned);
    drop(uniform);
    for (extent, padding) in [
        (Extent2d::new(1, 1), 0),
        (Extent2d::new(257, 3), 7),
        (Extent2d::new(5, 129), 17),
    ] {
        let samples = (0..extent.area().unwrap())
            .map(|i| (i % 17) as f32 / 16.0)
            .collect::<Vec<_>>();
        let actual = run_program(&backend, &pipeline, &program, extent, &samples, padding);
        for (actual, sample) in actual.iter().zip(samples) {
            assert!((actual - sample * sample).abs() <= 2e-7);
        }
    }
}

#[test]
fn large_irregular_sample_tables_keep_the_exact_f32_input_coordinate() {
    let Some(backend) = backend() else {
        return;
    };
    let linear = sampled_profile(&[0, 65535]);
    let pipeline = ResidentIccPipeline::new(backend.device()).unwrap();
    let mut input = vec![
        0.0,
        1.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        2_f32.powi(-50),
        2_f32.powi(-40),
        2_f32.powi(-20),
        0.000001,
        0.01,
        0.1,
        0.2,
        0.3,
        0.5,
        0.99,
        f32::from_bits(1_f32.to_bits() - 1),
    ];
    for i in 0..1024_u32 {
        input.push(f32::from_bits(
            0x3f00_0000 | (i.wrapping_mul(7919) & 0x7f_ffff),
        ));
    }
    for count in [1001, 1_000_003] {
        let table = (0..count)
            .map(|i| (i as u64 * 7919 % 65536) as u16)
            .collect::<Vec<_>>();
        let source = sampled_profile(&table);
        let transform = IccTransform::new(&source, &linear, IccRenderingIntent::Relative).unwrap();
        let actual = run(
            &backend,
            &pipeline,
            &transform,
            Extent2d::new(input.len() as u32, 1),
            &input,
            11,
        );
        for (i, (&actual, &x)) in actual.iter().zip(&input).enumerate() {
            // f64 exactly represents the product of this F32 significand and the <=20-bit
            // integer interval count. It supplies an independent interpolation coordinate.
            let position = f64::from(x) * (count - 1) as f64;
            let left = (position as usize).min(count - 2);
            let a = f64::from(table[left]) / 65535.0;
            let b = f64::from(table[left + 1]) / 65535.0;
            let expected = a + (b - a) * (position - left as f64);
            assert!(
                (f64::from(actual) - expected).abs() <= 2e-7,
                "{count} samples, input {i} ({x}): {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn inverse_plateaus_follow_icc_endpoint_rules_in_both_monotone_directions() {
    let Some(backend) = backend() else {
        return;
    };
    let linear = sampled_profile(&[0, 65535]);
    let samples = [0, 0, 16384, 16384, 49151, 65535, 65535];
    let q1 = f32::from(samples[2]) / 65535.0;
    let q2 = f32::from(samples[4]) / 65535.0;
    let inputs = [-1.0, 0.0, q1 * 0.5, q1, (q1 + q2) * 0.5, q2, 1.0, 2.0];
    let expected = [
        1.0 / 6.0,
        1.0 / 6.0,
        1.5 / 6.0,
        3.0 / 6.0,
        3.5 / 6.0,
        4.0 / 6.0,
        5.0 / 6.0,
        5.0 / 6.0,
    ];
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Tile16x16,
    ] {
        let pipeline = ResidentIccPipeline::with_variant(backend.device(), variant).unwrap();
        for decreasing in [false, true] {
            let target = sampled_profile(&samples.map(|v| if decreasing { 65535 - v } else { v }));
            let transform =
                IccTransform::new(&linear, &target, IccRenderingIntent::Relative).unwrap();
            let input = inputs.map(|v| if decreasing { 1.0 - v } else { v });
            let actual = run(
                &backend,
                &pipeline,
                &transform,
                Extent2d::new(8, 1),
                &input,
                7,
            );
            for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 2e-7,
                    "{variant:?} decreasing={decreasing} sample {i}: {actual} vs {expected}"
                );
            }
        }
    }
    let constant = sampled_profile(&[0; 3]);
    assert!(
        constant
            .matrix_trc(IccDirection::DeviceToPcs, IccRenderingIntent::Relative)
            .is_ok()
    );
    assert!(matches!(
        IccTransform::new(&linear, &constant, IccRenderingIntent::Relative),
        Err(IccError::CurveInverse { .. })
    ));
}

#[test]
fn resident_icc_rejects_invalid_bindings_aliasing_and_unadmitted_program_sizes() {
    let Some(backend) = backend() else {
        return;
    };
    let transform = IccTransform::new(
        &profile("sampled"),
        &profile("srgb"),
        IccRenderingIntent::Relative,
    )
    .unwrap();
    let plan = ResidentIccMemoryPlan::new(&transform, &backend.device().limits()).unwrap();
    let exact = wgpu::Limits {
        max_buffer_size: plan.program_bytes,
        max_storage_buffer_binding_size: plan.program_bytes,
        ..backend.device().limits()
    };
    assert_eq!(
        ResidentIccMemoryPlan::new(&transform, &exact).unwrap(),
        plan
    );
    assert!(matches!(
        ResidentIccMemoryPlan::new(
            &transform,
            &wgpu::Limits {
                max_buffer_size: plan.program_bytes - 1,
                ..exact
            }
        ),
        Err(ResidentIccError::Limit {
            resource: "program bytes",
            ..
        })
    ));
    let program = ResidentIccProgram::new(backend.device(), &transform).unwrap();
    let pipeline = ResidentIccPipeline::new(backend.device()).unwrap();
    let extent = Extent2d::new(3, 5);
    let source = Storage::new(&backend, extent, 3, Some(&[0.5; 45]), 3);
    let target = Storage::new(&backend, extent, 3, None, 3);
    let mut encoder = backend.device().create_command_encoder(&Default::default());
    let input = ResidentIccInputs {
        input: source.binding(),
        output: target.binding(),
        extent,
        input_planes: &source.planes,
        output_planes: &target.planes,
    };
    let overlapping = [target.planes[0]; 3];
    let overflowing = [ResidentIccPlane {
        offset: u32::MAX,
        stride: 3,
    }; 3];
    let short_stride = [ResidentIccPlane {
        offset: 0,
        stride: 2,
    }; 3];
    let mut check = |inputs| {
        pipeline
            .encode(backend.device(), &mut encoder, &program, inputs)
            .err()
            .unwrap()
    };
    assert!(matches!(
        check(ResidentIccInputs {
            input_planes: &source.planes[..2],
            ..input
        }),
        ResidentIccError::Channels {
            role: "input",
            expected: 3,
            actual: 2
        }
    ));
    assert_eq!(
        check(ResidentIccInputs {
            output: source.binding(),
            ..input
        }),
        ResidentIccError::Aliasing
    );
    assert_eq!(
        check(ResidentIccInputs {
            output_planes: &overlapping,
            ..input
        }),
        ResidentIccError::OutputOverlap
    );
    assert!(matches!(
        check(ResidentIccInputs {
            input: ResidentStorageBinding {
                offset: 1,
                ..source.binding()
            },
            ..input
        }),
        ResidentIccError::Binding { role: "input" }
    ));
    assert!(matches!(
        check(ResidentIccInputs {
            extent: Extent2d::new(0, 5),
            ..input
        }),
        ResidentIccError::Plane { .. }
    ));
    assert!(matches!(
        check(ResidentIccInputs {
            input_planes: &overflowing,
            ..input
        }),
        ResidentIccError::Addressing
    ));
    assert!(matches!(
        check(ResidentIccInputs {
            input_planes: &short_stride,
            ..input
        }),
        ResidentIccError::Plane { role: "input", .. }
    ));
    // Validation failure never records a dispatch; finishing/submitting this encoder is valid.
    let submission = backend.queue().submit([encoder.finish()]);
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
}
