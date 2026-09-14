use super::{directory, floats, oracle};
use jxl_gpu_bitstream::gain_map::{GainMapBundle, JHGM};
use jxl_gpu_formats::{
    ColorSpace, ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction,
};
use jxl_test_support::{gpu::planes, oracles::hdr};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, OrientationPolicy};

#[test]
fn alternate_output_preserves_hdr_luminance_alpha_and_planar_or_packed_storage() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    // Lossless baseline, RGB Modular-XYB gain samples, asymmetric resampling and alternate
    // BT.2020 application primaries. Output Display-P3 adds a second primary conversion.
    let name = "case_9";
    let bytes = std::fs::read(directory().join(format!("{name}.jxl"))).unwrap();
    let parsed = jxl_gpu_bitstream::parse(&bytes, Default::default()).unwrap();
    let payload = parsed
        .auxiliary_boxes()
        .iter()
        .find(|b| b.box_type == JHGM)
        .unwrap()
        .payload;
    let bundle = GainMapBundle::parse(payload, Default::default()).unwrap();
    let reference = oracle(
        &floats(&format!("{name}.base.f32")),
        &floats(&format!("{name}.gain.f32")),
        29,
        13,
        bundle.metadata(),
        true,
    );
    let mut images = 0;
    for transfer in [
        TransferFunction::Linear,
        TransferFunction::Srgb,
        TransferFunction::Pq,
        TransferFunction::Hlg,
    ] {
        for planar in [false, true] {
            for integer in [false, true] {
                for associated in [false, true] {
                    let ColorSpecification::Defined(mut color) = super::format(true).color_spec
                    else {
                        unreachable!()
                    };
                    color.space = ColorSpace::DisplayP3;
                    color.transfer = transfer;
                    let format = if integer {
                        PixelFormat::rgb8(
                            RgbChannelOrder::Bgra,
                            planar,
                            ColorSpecification::Defined(color),
                        )
                    } else {
                        PixelFormat::rgb_f32(
                            RgbChannelOrder::Bgra,
                            planar,
                            ColorSpecification::Defined(color),
                        )
                    };
                    let request = GpuOutputRequest::color(format.clone())
                        .unwrap()
                        .with_orientation_policy(OrientationPolicy::Keep)
                        .with_alpha_output_policy(if associated {
                            AlphaOutputPolicy::Associated
                        } else {
                            AlphaOutputPolicy::Unassociated
                        });
                    let frame = pollster::block_on(decoder.decode_alternate(
                        &bytes,
                        request,
                        Default::default(),
                    ))
                    .unwrap();
                    let output = &frame.output().outputs[0];
                    assert_eq!(output.layout.format, format);
                    assert_eq!(output.layout.extent, jxl_gpu_protocol::Extent2d::new(17, 9));
                    let bytes = planes::read_bytes(&backend, output);
                    let mut occupied = vec![false; bytes.len()];
                    for pixel in 0..17 * 9 {
                        let rgb = std::array::from_fn(|c| reference[pixel * 4 + c]);
                        let converted = hdr::convert(
                            rgb,
                            TransferFunction::Linear,
                            transfer,
                            ColorSpace::Bt2020,
                            ColorSpace::DisplayP3,
                            203.0,
                        );
                        // Propagate the already-fixed gain reconstruction bound through the
                        // signed primary matrix, PQ scaling and coupled HLG OOTF.
                        let range = hdr::interval(
                            rgb,
                            TransferFunction::Linear,
                            transfer,
                            ColorSpace::Bt2020,
                            ColorSpace::DisplayP3,
                            203.0,
                            2e-4,
                        );
                        let alpha = reference[pixel * 4 + 3];
                        for (storage, channel) in [2, 1, 0, 3].into_iter().enumerate() {
                            let plane = &output.layout.planes[if planar { storage } else { 0 }];
                            let component_bytes = if integer { 1 } else { 4 };
                            let index = plane.offset as usize
                                + pixel / 17 * plane.row_stride as usize
                                + (pixel % 17 * if planar { 1 } else { 4 }
                                    + if planar { 0 } else { storage })
                                    * component_bytes;
                            occupied[index..index + component_bytes].fill(true);
                            let actual = if integer {
                                f64::from(bytes[index]) / 255.0
                            } else {
                                f64::from(f32::from_le_bytes(
                                    bytes[index..][..4].try_into().unwrap(),
                                ))
                            };
                            let (mut low, mut high, packing) = if channel == 3 {
                                (alpha - 2e-7, alpha + 2e-7, 0.0)
                            } else {
                                let scale = if associated {
                                    alpha.max(2.0_f64.powi(-26))
                                } else {
                                    1.0
                                };
                                (
                                    range[channel][0] * scale,
                                    range[channel][1] * scale,
                                    5e-5 * (1.0 + converted[channel].abs()) * scale,
                                )
                            };
                            let quantization = if integer {
                                low = low.clamp(0.0, 1.0);
                                high = high.clamp(0.0, 1.0);
                                1.0 / 255.0
                            } else {
                                0.0
                            };
                            assert!(
                                actual.is_finite()
                                    && actual >= low - packing - quantization
                                    && actual <= high + packing + quantization,
                                "{transfer:?} planar={planar} integer={integer} associated={associated} {pixel}/{channel}: {actual}, F64 interval [{low},{high}]"
                            );
                        }
                    }
                    assert!(
                        bytes
                            .iter()
                            .zip(occupied)
                            .all(|(&byte, used)| used || byte == 0),
                        "nonzero output padding"
                    );
                    drop(frame);
                    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                    images += 1;
                }
            }
        }
    }
    assert_eq!(images, 32);
    eprintln!(
        "gain-map {images} output layouts/transfers/association cases, {} values",
        images * 17 * 9 * 4
    );
}
