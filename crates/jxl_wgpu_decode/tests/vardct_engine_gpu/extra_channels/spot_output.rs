use super::*;
use jxl_gpu_formats::{
    ChromaLocation2d, ColorRange, ColorSpec, ColorSpecification, Packed422Order, TransferFunction,
    YcbcrEncoding, convert_rgb_f32,
};
use jxl_wgpu_encode::LosslessModularFormat;

#[test]
fn spot_presentation_precedes_native_quantization_and_target_chroma_subsampling() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let srgb = ColorSpecification::Defined(ColorSpec {
        transfer: TransferFunction::Srgb,
        ..ColorSpec::bt709(ColorRange::Limited, ChromaLocation2d::CENTER)
    });
    let cl = ColorSpecification::Defined(ColorSpec {
        encoding: YcbcrEncoding::Bt2020ConstantLuminance,
        ..ColorSpec::bt2020_ncl(ColorRange::Limited, ChromaLocation2d::CENTER)
    });
    for (name, hex) in multiple_spots()
        .into_iter()
        .filter(|(name, _)| name.ends_with("resampled"))
    {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        for policy in [
            AlphaOutputPolicy::Preserve,
            AlphaOutputPolicy::Unassociated,
            AlphaOutputPolicy::Associated,
        ] {
            for bits in [8, 12, 16] {
                for kind in [LosslessModularFormat::Rgb, LosslessModularFormat::Rgba] {
                    let mut format = kind.pixel_format(bits).unwrap();
                    format.color_spec = vardct_rgb8_format().color_spec;
                    let request = GpuOutputRequest::color(format)
                        .unwrap()
                        .with_alpha_output_policy(policy);
                    let frames = associated::decode(&backend, &data, request.clone(), false);
                    assert_eq!(
                        frames,
                        associated::decode(&backend, &data, request, true),
                        "{name}/{bits}: native async"
                    );
                    if let Some(mut expected) =
                        formula(&data, &inventory.image_header, false, false)
                    {
                        associated::associate(&mut expected, policy, true);
                        let channels = if kind == LosslessModularFormat::Rgb {
                            3
                        } else {
                            4
                        };
                        let mask = ((1u32 << bits) - 1) as f32;
                        for (index, sample) in frames[0]
                            .1
                            .chunks_exact(if bits <= 8 { 1 } else { 2 })
                            .enumerate()
                        {
                            let actual = if bits <= 8 {
                                u16::from(sample[0])
                            } else {
                                u16::from_le_bytes(sample.try_into().unwrap())
                            };
                            let expected = (expected[index / channels * 4 + index % channels]
                                .clamp(0.0, 1.0)
                                * mask)
                                .round() as u16;
                            let tolerance = if inventory.image_header.xyb_encoded {
                                (mask * 0.00005).ceil() as u16
                            } else {
                                1
                            };
                            assert!(
                                actual.abs_diff(expected) <= tolerance,
                                "{name}/{policy:?}/{bits}/{index}: {actual} vs {expected}"
                            );
                        }
                    }
                }
            }
            for format in [
                PixelFormat::nv12(srgb),
                PixelFormat::p010(cl),
                PixelFormat::packed_yuv4228(Packed422Order::Yuyv, srgb),
                PixelFormat::packed_yuv4228(Packed422Order::Uyvy, srgb),
            ] {
                let ColorSpecification::Defined(mut color) = format.color_spec else {
                    unreachable!()
                };
                color.range = ColorRange::Full;
                let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
                    RgbChannelOrder::Rgba,
                    false,
                    ColorSpecification::Defined(color),
                ))
                .unwrap()
                .with_alpha_output_policy(policy);
                let rgba = associated::decode(&backend, &data, request, false);
                let values = oracle::floats(&rgba[0].1);
                let planes: [Vec<f32>; 3] =
                    std::array::from_fn(|c| values.chunks_exact(4).map(|p| p[c]).collect());
                let expected = if format.color_spec == cl {
                    associated::p010_cl(&values, rgba[0].0.extent, &format)
                } else {
                    convert_rgb_f32(
                        [&planes[0], &planes[1], &planes[2]],
                        rgba[0].0.extent,
                        &format,
                    )
                    .unwrap()
                };
                let request = GpuOutputRequest::color(format)
                    .unwrap()
                    .with_alpha_output_policy(policy);
                let actual = associated::decode(&backend, &data, request.clone(), false);
                assert_eq!(actual, associated::decode(&backend, &data, request, true));
                assert_eq!(actual[0].0, expected.layout);
                let high = expected.layout.format.color_spec == cl;
                for (a, b) in actual[0]
                    .1
                    .chunks_exact(if high { 2 } else { 1 })
                    .zip(expected.bytes.chunks_exact(if high { 2 } else { 1 }))
                {
                    let (a, b) = if high {
                        (
                            u16::from_le_bytes(a.try_into().unwrap()),
                            u16::from_le_bytes(b.try_into().unwrap()),
                        )
                    } else {
                        (u16::from(a[0]), u16::from(b[0]))
                    };
                    if high {
                        assert_eq!(a & 63, 0);
                    }
                    assert!(
                        a.abs_diff(b) <= if high { 64 } else { 1 },
                        "{name}/{policy:?}: {a} vs {b}"
                    );
                }
            }
        }
    }
}

#[test]
fn spot_policy_never_tints_selected_numeric_planes_and_hdr_remains_explicit() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for (name, hex) in multiple_spots()
        .into_iter()
        .filter(|(name, _)| name.ends_with("rgb"))
    {
        let data = encoded(hex);
        for index in [1, 3, 4, 6, 7] {
            let bits = [16, 1, 7, 12, 4, 8, 6, 10, 15][index];
            for (format, mapping) in [
                (
                    PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                    NumericSampleMapping::NormalizedUnsigned,
                ),
                (
                    LosslessModularFormat::Gray.pixel_format(bits).unwrap(),
                    NumericSampleMapping::NativeUnsigned,
                ),
            ] {
                let request = GpuOutputRequest::numeric(format, mapping)
                    .unwrap()
                    .with_extra_channel(index as u32)
                    .unwrap();
                let rendered = associated::decode(&backend, &data, request.clone(), false);
                let preserved = associated::decode(
                    &backend,
                    &data,
                    request.with_spot_color_policy(SpotColorPolicy::Preserve),
                    true,
                );
                assert_eq!(rendered, preserved, "{name}/{index}: raw spot data");
            }
        }
        if name.starts_with("vardct") {
            let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
            for transfer in [TransferFunction::Pq, TransferFunction::Hlg] {
                let ColorSpecification::Defined(mut color) = vardct_rgb8_format().color_spec else {
                    unreachable!()
                };
                color.transfer = transfer;
                let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
                    RgbChannelOrder::Rgb,
                    false,
                    ColorSpecification::Defined(color),
                ))
                .unwrap();
                assert!(matches!(decoder.open(&data, request), Err(DecodeError::VarDct(VarDctDecodeError::Output(
                    jxl_wgpu_decode::color_output::ColorOutputError::HdrLuminanceMappingRequired)))));
            }
            assert_eq!(
                backend.transient_memory_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}
