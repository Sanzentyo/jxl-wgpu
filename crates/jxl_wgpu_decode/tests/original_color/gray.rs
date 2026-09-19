use super::{corpus, oracle, planes, tolerance};
use jxl_gpu_formats::{ColorSpace, ColorSpecification, ImageLayout, PixelFormat, TransferFunction};
use jxl_test_support::oracles::hdr;
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, WgpuDecodeEngine};
use std::num::NonZeroU64;

#[test]
fn gray_and_alpha_outputs_match_native_color_and_f64_projection_in_both_codecs() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let cases: Vec<_> = corpus::cases()
        .into_iter()
        .filter(|case| {
            !case.sequence
                && !case.floating
                && case.transfer.name == "srgb"
                && (case.profile.name == "bt2020"
                    || (case.profile.grayscale
                        && matches!(
                            case.mode,
                            corpus::Mode::ModularRgb | corpus::Mode::VarDctXyb
                        )))
        })
        .collect();
    assert_eq!(cases.len(), 8);
    let luminance = hdr::luminance(ColorSpace::Bt709);
    for case in cases {
        let data = case.bytes();
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        case.validate(&inventory);
        assert_eq!(
            inventory.image_header.extra_channels[0].channel_type,
            jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha { associated: false },
        );
        let reference = case.reference();
        assert!(
            reference
                .as_chunks::<4>()
                .0
                .iter()
                .any(|p| p[3] > 0.0 && p[3] < 1.0)
        );
        let ColorSpecification::Defined(source) = case.format().color_spec else {
            unreachable!()
        };
        let matrix = oracle::matrix(source.space, ColorSpace::Bt709);
        for (floating, alpha, planar, association) in [
            (true, false, false, AlphaOutputPolicy::Preserve),
            (true, true, false, AlphaOutputPolicy::Unassociated),
            (true, true, true, AlphaOutputPolicy::Associated),
            (false, false, false, AlphaOutputPolicy::Associated),
            (false, true, true, AlphaOutputPolicy::Preserve),
            (false, true, false, AlphaOutputPolicy::Associated),
        ] {
            let ColorSpecification::Defined(mut target) =
                jxl_wgpu_decode::vardct_rgb8_format().color_spec
            else {
                unreachable!()
            };
            target.transfer = if floating {
                TransferFunction::Linear
            } else {
                TransferFunction::Srgb
            };
            let color = ColorSpecification::Defined(target);
            let format = if floating {
                PixelFormat::gray_f32(alpha, planar, color)
            } else {
                PixelFormat::gray8(alpha, planar, color)
            };
            let mut baseline = None;
            for bounded in [false, true] {
                let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                if bounded {
                    engine = engine.with_stream_window_limit(NonZeroU64::new(256).unwrap());
                }
                let decoder = GpuDecoder::new(engine);
                let request = GpuOutputRequest::color(format.clone())
                    .unwrap()
                    .with_alpha_output_policy(association);
                let mut session = if bounded {
                    planes::open_fragmented(&decoder, &data, request)
                } else {
                    decoder.open(&data, request).unwrap()
                };
                let frame = pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap();
                assert!(
                    pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .is_none()
                );
                drop(session);
                assert!(decoder.engine().in_flight_memory_stats().reserved_bytes > 0);
                let output = &frame.output().outputs[0];
                assert_eq!(
                    output.layout,
                    ImageLayout::packed(jxl_gpu_protocol::Extent2d::new(37, 19), format.clone())
                        .unwrap()
                );
                let bytes = planes::read_bytes(&backend, output);
                for (pixel, reference) in reference.as_chunks::<4>().0.iter().enumerate() {
                    let rgb = [reference[0], reference[1], reference[2]].map(f64::from);
                    let linear =
                        oracle::convert(rgb, source.transfer, TransferFunction::Linear, matrix);
                    let range = oracle::interval(
                        rgb,
                        source.transfer,
                        TransferFunction::Linear,
                        matrix,
                        f64::from(tolerance(&case)),
                    );
                    let expected = oracle::from_linear(
                        (0..3).map(|c| linear[c] * luminance[c]).sum(),
                        target.transfer,
                    );
                    let edges: [f64; 2] = std::array::from_fn(|edge| {
                        oracle::from_linear(
                            (0..3).map(|c| range[c][edge] * luminance[c]).sum(),
                            target.transfer,
                        )
                    });
                    for component in 0..1 + usize::from(alpha) {
                        let (mut expected, mut low, mut high) = if component == 0 {
                            (expected, edges[0], edges[1])
                        } else {
                            let a = f64::from(reference[3]);
                            (a, a - 2e-6, a + 2e-6)
                        };
                        if component == 0 && association == AlphaOutputPolicy::Associated {
                            let a = f64::from(reference[3]);
                            if a <= 2.0_f64.powi(-26) {
                                expected = 0.0;
                                low = 0.0;
                                high = 0.0;
                            } else {
                                let bounds = [
                                    low * (a - 2e-6),
                                    low * (a + 2e-6),
                                    high * (a - 2e-6),
                                    high * (a + 2e-6),
                                ];
                                low = bounds.into_iter().fold(f64::INFINITY, f64::min);
                                high = bounds.into_iter().fold(f64::NEG_INFINITY, f64::max);
                                expected *= a;
                            }
                        }
                        let plane = &output.layout.planes[if planar { component } else { 0 }];
                        let sample_bytes = if floating { 4 } else { 1 };
                        let offset = plane.offset as usize
                            + pixel / 37 * plane.row_stride as usize
                            + pixel % 37
                                * sample_bytes
                                * if planar { 1 } else { 1 + usize::from(alpha) }
                            + if planar { 0 } else { component * sample_bytes };
                        let actual = if floating {
                            f64::from(f32::from_le_bytes(
                                bytes[offset..offset + 4].try_into().unwrap(),
                            ))
                        } else {
                            low = low.clamp(0.0, 1.0);
                            high = high.clamp(0.0, 1.0);
                            f64::from(bytes[offset]) / 255.0
                        };
                        let rounding = if component == 0 {
                            5e-6 * (1.0 + expected.abs())
                        } else {
                            0.0
                        };
                        let quantization = if floating { 0.0 } else { 1.0 / 255.0 };
                        assert!(
                            actual.is_finite()
                                && actual >= low - rounding - quantization
                                && actual <= high + rounding + quantization,
                            "{} float={floating} alpha={alpha} planar={planar} {association:?} {pixel}/{component}: {actual}, f64 {expected}, [{low}, {high}]",
                            case.name
                        );
                    }
                }
                if let Some(baseline) = &baseline {
                    assert_eq!(&bytes, baseline, "fragmented {}", case.name);
                }
                baseline = Some(bytes);
                drop(frame);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}
