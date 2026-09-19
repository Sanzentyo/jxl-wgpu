use super::*;
use jxl_gpu_protocol::{GamutMapping, OutputOrientation};
use jxl_test_support::oracles::{color as oracle, gamut_mapping, hdr};

#[test]
fn gray_projects_mapped_linear_light_before_transfer_and_packs_alpha() {
    let backend = backend().expect("gray output conformance requires an adapter");
    let extent = Extent2d::new(3, 2);
    let pixels = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-0.25, 0.125, 1.5],
        [-0.25; 3],
        [1.5; 3],
    ];
    let channels = std::array::from_fn(|channel| pixels.iter().map(|p| p[channel]).collect());
    let source = RgbColorEncoding {
        space: RgbColorSpace::Bt2020,
        transfer: SourceTransferFunction::Linear,
    };
    let gamma = jxl_gpu_protocol::GammaExponent::new(0.4545455).unwrap();
    for target in [ColorSpace::Bt709, ColorSpace::Bt2020, ColorSpace::DisplayP3] {
        let matrix = oracle::matrix(ColorSpace::Bt2020, target);
        let luminance = hdr::luminance(target);
        for mapped in [false, true] {
            for transfer in [
                TransferFunction::Linear,
                TransferFunction::Srgb,
                TransferFunction::Bt709,
                TransferFunction::Bt2020,
                TransferFunction::Gamma(gamma),
                TransferFunction::Dci,
                TransferFunction::Pq,
                TransferFunction::Hlg,
            ] {
                let expected = pixels.map(|pixel| {
                    let mut linear = matrix.map(|row| {
                        row.into_iter()
                            .zip(pixel)
                            .map(|(a, b)| a * f64::from(b))
                            .sum()
                    });
                    if mapped {
                        linear = gamut_mapping::apply(linear, luminance, f64::from(0.1_f32));
                    }
                    let y = linear.into_iter().zip(luminance).map(|(v, l)| v * l).sum();
                    oracle::from_linear(y, transfer)
                });
                for (alpha, planar) in [(false, false), (true, false), (true, true)] {
                    for floating in [false, true] {
                        let color = rgb_color(target, transfer);
                        let format = if floating {
                            PixelFormat::gray_f32(alpha, planar, color)
                        } else {
                            PixelFormat::gray8(alpha, planar, color)
                        };
                        let mut session = backend
                            .create_session(&frame_desc(extent), plan(extent, source))
                            .unwrap();
                        enqueue(&mut session, extent, &channels);
                        let mut request = ImageOutputRequest::new(source, format.clone());
                        if mapped {
                            request = request.with_gamut_mapping(GamutMapping::default());
                        }
                        let token = session.submit_image(RenderIntent::Final, request).unwrap();
                        let actual = session.wait_image(token).unwrap().outputs.remove(0);
                        assert_eq!(actual.layout, ImageLayout::packed(extent, format).unwrap());
                        let components = 1 + usize::from(alpha);
                        let sample_bytes = if floating { 4 } else { 1 };
                        for (pixel, expected) in expected.into_iter().enumerate() {
                            for component in 0..components {
                                let expected = if component == 0 { expected } else { 1.0 };
                                let plane =
                                    &actual.layout.planes[if planar { component } else { 0 }];
                                let offset = plane.offset as usize
                                    + pixel / 3 * plane.row_stride as usize
                                    + pixel % 3
                                        * sample_bytes
                                        * if planar { 1 } else { components }
                                    + if planar { 0 } else { component * sample_bytes };
                                if floating {
                                    let value = f32::from_le_bytes(
                                        actual.bytes[offset..offset + 4].try_into().unwrap(),
                                    );
                                    if component == 1 {
                                        assert_eq!(value.to_bits(), 1.0_f32.to_bits());
                                    } else {
                                        let error = (f64::from(value) - expected).abs();
                                        assert!(
                                            value.is_finite()
                                                && error <= 5e-5 * expected.abs().max(1.0),
                                            "{target:?}/{transfer:?} mapped={mapped}, pixel={pixel}: {value} != {expected}"
                                        );
                                    }
                                } else {
                                    let code = (expected.clamp(0.0, 1.0) * 255.0).round() as u8;
                                    assert!(
                                        actual.bytes[offset].abs_diff(code)
                                            <= u8::from(component == 0)
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
}

#[test]
fn gray_color_contract_rejects_missing_metadata_and_limited_range() {
    let extent = Extent2d::new(3, 2);
    let source = jxl_wgpu::ImageOutputSource {
        extent,
        orientation: OutputOrientation::Identity,
        strides: [3; 3],
        encoding: RgbColorEncoding::LINEAR_BT709,
    };
    let mut limited = rgb_color(ColorSpace::Bt709, TransferFunction::Linear);
    let ColorSpecification::Defined(ref mut color) = limited else {
        unreachable!()
    };
    color.range = ColorRange::Limited;
    for color in [
        limited,
        ColorSpecification::Undefined,
        rgb_color(ColorSpace::Sensor, TransferFunction::Linear),
        rgb_color(ColorSpace::Bt709, TransferFunction::Undefined),
    ] {
        let layout =
            ImageLayout::packed(extent, PixelFormat::gray_f32(false, false, color)).unwrap();
        assert!(matches!(
            jxl_wgpu::ImageOutputParams::new(
                &layout,
                source,
                64,
                jxl_gpu_protocol::WhitePointAdaptation::Bradford
            ),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn gray_projection_follows_all_eight_output_orientations() {
    let backend = backend().expect("gray orientation conformance requires an adapter");
    let extent = Extent2d::new(3, 2);
    let channels = [
        vec![1.0, 0.0, 0.0, -0.25, 1.25, 0.5],
        vec![0.0, 1.0, 0.0, 0.5, -0.25, 0.25],
        vec![0.0, 0.0, 1.0, 0.75, 0.5, -0.5],
    ];
    let luminance = hdr::luminance(ColorSpace::Bt709);
    for (exif, pixels) in [
        (1, [0, 1, 2, 3, 4, 5]),
        (2, [2, 1, 0, 5, 4, 3]),
        (3, [5, 4, 3, 2, 1, 0]),
        (4, [3, 4, 5, 0, 1, 2]),
        (5, [0, 3, 1, 4, 2, 5]),
        (6, [3, 0, 4, 1, 5, 2]),
        (7, [5, 2, 4, 1, 3, 0]),
        (8, [2, 5, 1, 4, 0, 3]),
    ] {
        let orientation = OutputOrientation::from_exif_value(exif).unwrap();
        let output_extent = if exif < 5 {
            extent
        } else {
            Extent2d::new(2, 3)
        };
        for planar in [false, true] {
            let format = PixelFormat::gray_f32(
                true,
                planar,
                rgb_color(ColorSpace::Bt709, TransferFunction::Linear),
            );
            let mut render = plan(extent, RgbColorEncoding::LINEAR_BT709);
            let render_mut = Arc::get_mut(&mut render).unwrap();
            let RenderOp::Save(save) = &mut render_mut.nodes[0].op else {
                unreachable!()
            };
            save.orientation = orientation;
            render_mut.outputs[0].extent = output_extent;
            let mut session = backend.create_session(&frame_desc(extent), render).unwrap();
            enqueue(&mut session, extent, &channels);
            let token = session
                .submit_image(
                    RenderIntent::Final,
                    ImageOutputRequest::new(RgbColorEncoding::LINEAR_BT709, format.clone()),
                )
                .unwrap();
            let output = session.wait_image(token).unwrap().outputs.remove(0);
            assert_eq!(
                output.layout,
                ImageLayout::packed(output_extent, format).unwrap()
            );
            for (pixel, source) in pixels.into_iter().enumerate() {
                for component in 0..2 {
                    let plane = &output.layout.planes[if planar { component } else { 0 }];
                    let offset = plane.offset as usize
                        + pixel / output_extent.width as usize * plane.row_stride as usize
                        + pixel % output_extent.width as usize * if planar { 4 } else { 8 }
                        + if planar { 0 } else { component * 4 };
                    let value =
                        f32::from_le_bytes(output.bytes[offset..offset + 4].try_into().unwrap());
                    let expected = if component == 1 {
                        1.0
                    } else {
                        (0..3)
                            .map(|c| f64::from(channels[c][source]) * luminance[c])
                            .sum()
                    };
                    assert!(
                        (f64::from(value) - expected).abs() <= 2e-6,
                        "{orientation:?}/{planar}/{pixel}/{component}: {value} != {expected}"
                    );
                }
            }
        }
    }
    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
}
