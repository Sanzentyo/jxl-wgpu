use super::*;
use jxl_gpu_protocol::GamutMapping;
use jxl_test_support::oracles::{color, gamut_mapping as oracle, hdr};

#[test]
fn mapped_output_precedes_transfer_yuv_packing_and_quantization() {
    let backend = backend().expect("gamut mapping requires an adapter");
    let extent = Extent2d::new(7, 1);
    let inputs = [
        [-0.2, 0.1, 1.25],
        [1.3, 0.2, 0.3],
        [0.1, 0.4, 0.7],
        [4.0; 3],
        [-1.0; 3],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let channels = std::array::from_fn(|c| inputs.iter().map(|p| p[c]).collect());
    let source = RgbColorEncoding {
        space: RgbColorSpace::Bt2020,
        transfer: SourceTransferFunction::Linear,
    };
    for target in [ColorSpace::Bt709, ColorSpace::Bt2020, ColorSpace::DisplayP3] {
        for transfer in [
            TransferFunction::Linear,
            TransferFunction::Srgb,
            TransferFunction::Pq,
            TransferFunction::Hlg,
        ] {
            let color_spec = rgb_color(target, transfer);
            let mut formats = vec![
                (
                    PixelFormat::rgb8(RgbChannelOrder::Rgb, false, color_spec.clone()),
                    false,
                ),
                (
                    PixelFormat::rgb_f32(RgbChannelOrder::Bgra, true, color_spec.clone()),
                    false,
                ),
                (
                    PixelFormat::rgb_f32(RgbChannelOrder::Rgba, false, color_spec.clone()),
                    false,
                ),
            ];
            if transfer == TransferFunction::Srgb {
                let ColorSpecification::Defined(mut yuv) = color_spec else {
                    unreachable!()
                };
                yuv.encoding = YcbcrEncoding::Bt709;
                yuv.chroma_location = ChromaLocation2d::CENTER;
                let yuv = ColorSpecification::Defined(yuv);
                formats.extend([
                    (PixelFormat::nv12(yuv.clone()), false),
                    (PixelFormat::i444(12, 16, yuv).unwrap(), true),
                ]);
            }
            let matrix = color::matrix(ColorSpace::Bt2020, target);
            let expected: [Vec<_>; 3] = std::array::from_fn(|c| {
                inputs
                    .iter()
                    .map(|p| {
                        let converted = matrix.map(|row| {
                            row.into_iter().zip(p).map(|(a, b)| a * f64::from(*b)).sum()
                        });
                        let mapped =
                            oracle::apply(converted, hdr::luminance(target), f64::from(0.1_f32));
                        // Generic transfers use absolute PQ units and scene HLG, without image white.
                        color::from_linear(mapped[c], transfer) as f32
                    })
                    .collect()
            });
            for (format, sixteen_bit) in formats {
                let mut session = backend
                    .create_session(&frame_desc(extent), plan(extent, source))
                    .unwrap();
                enqueue(&mut session, extent, &channels);
                let token = session
                    .submit_image(
                        RenderIntent::Final,
                        ImageOutputRequest::new(source, format.clone())
                            .with_gamut_mapping(GamutMapping::default()),
                    )
                    .unwrap();
                let actual = session.wait_image(token).unwrap().outputs.remove(0);
                let expected =
                    convert_rgb_f32([&expected[0], &expected[1], &expected[2]], extent, &format)
                        .unwrap();
                assert_eq!(actual.layout, expected.layout);
                if format.sample_kind == jxl_gpu_formats::SampleKind::Float {
                    for (component, (actual, expected)) in actual
                        .bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(expected.bytes.as_chunks::<4>().0)
                        .enumerate()
                    {
                        assert!(
                            (f32::from_le_bytes(*actual) - f32::from_le_bytes(*expected)).abs()
                                <= 5e-5,
                            "{format:?}/{component}: GPU {}, expected {}",
                            f32::from_le_bytes(*actual),
                            f32::from_le_bytes(*expected)
                        );
                    }
                } else if sixteen_bit {
                    for (actual, expected) in actual
                        .bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(expected.bytes.as_chunks::<2>().0)
                    {
                        assert!(
                            u16::from_le_bytes(*actual).abs_diff(u16::from_le_bytes(*expected))
                                <= 16
                        );
                    }
                } else {
                    assert!(
                        actual
                            .bytes
                            .iter()
                            .zip(expected.bytes)
                            .all(|(a, b)| a.abs_diff(b) <= 1),
                        "{format:?}"
                    );
                }
            }
        }
    }
}
