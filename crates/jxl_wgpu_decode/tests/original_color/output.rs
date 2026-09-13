use super::{corpus, oracle, planes, tolerance};
use jxl_gpu_formats::{
    Channel, ColorSpace, ColorSpecification, PixelFormat, RgbChannelOrder, SampleKind,
    TransferFunction,
};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, ModularChannels, NumericSampleMapping,
    native_modular_pixel_format,
};

#[derive(Clone, Copy, Debug)]
enum Output {
    LinearBt709,
    AbsoluteLinearBt709,
    Srgb8,
    Native12,
    ColorPlane,
    AlphaPlane,
}

impl Output {
    fn request(self, case: &corpus::Case) -> GpuOutputRequest {
        let scalar = || PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]);
        match self {
            Self::LinearBt709 | Self::AbsoluteLinearBt709 => {
                let mut color = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
                let ColorSpecification::Defined(ref mut spec) = color else {
                    unreachable!()
                };
                spec.transfer = TransferFunction::Linear;
                GpuOutputRequest::color(PixelFormat::rgb_f32(RgbChannelOrder::Rgba, false, color))
                    .unwrap()
            }
            Self::Srgb8 => GpuOutputRequest::color(PixelFormat::rgb8(
                RgbChannelOrder::Rgba,
                false,
                jxl_wgpu_decode::vardct_rgb8_format().color_spec,
            ))
            .unwrap(),
            Self::Native12 => GpuOutputRequest::color(
                native_modular_pixel_format(ModularChannels::Rgba, 12).unwrap(),
            )
            .unwrap(),
            Self::ColorPlane => GpuOutputRequest::numeric(
                scalar(),
                if case.floating {
                    NumericSampleMapping::NativeFloat
                } else {
                    NumericSampleMapping::NormalizedUnsigned
                },
            )
            .unwrap()
            .with_color_channel(u32::from(!case.profile.grayscale))
            .unwrap(),
            Self::AlphaPlane => {
                GpuOutputRequest::numeric(scalar(), NumericSampleMapping::NormalizedUnsigned)
                    .unwrap()
                    .with_extra_channel(0)
                    .unwrap()
            }
        }
        .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
        .with_white_point_adaptation(if matches!(self, Self::AbsoluteLinearBt709) {
            jxl_gpu_protocol::WhitePointAdaptation::None
        } else {
            jxl_gpu_protocol::WhitePointAdaptation::Bradford
        })
    }
}

#[test]
fn requested_color_and_numeric_outputs_keep_original_encoding_and_alpha_independent() {
    check_outputs(corpus::cases());
}

#[test]
fn analytic_color_and_numeric_outputs_keep_original_encoding_and_alpha_independent() {
    check_outputs(corpus::analytic_cases());
}

fn check_outputs(cases: Vec<corpus::Case>) {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for case in cases {
        let data = case.bytes();
        let expected = case.reference();
        let linear_original = (case.mode.xyb()
            && !case.sequence
            && matches!(
                case.transfer.transfer,
                jxl_gpu_bitstream::TransferFunctionInventory::Gamma { .. }
                    | jxl_gpu_bitstream::TransferFunctionInventory::Dci
            ))
        .then(|| oracle::linear_original_still(&case));
        let ColorSpecification::Defined(source) = case.format().color_spec else {
            unreachable!()
        };
        let mut outputs = vec![
            Output::LinearBt709,
            Output::Srgb8,
            Output::Native12,
            Output::ColorPlane,
            Output::AlphaPlane,
        ];
        if case.name.starts_with("analytic_") {
            outputs.push(Output::AbsoluteLinearBt709);
        }
        for output in outputs {
            let matrix = oracle::matrix_with_adaptation(
                source.space,
                ColorSpace::Bt709,
                !matches!(output, Output::AbsoluteLinearBt709),
            );
            eprintln!("{} {output:?}", case.name);
            let mut session = decoder.open(&data, output.request(&case)).unwrap();
            let mut frames = 0;
            while let Some(frame) = session.next_frame().unwrap() {
                let result = &frame.output().outputs[0];
                let bytes = planes::read_bytes(&backend, result);
                let (samples, channels): (Vec<f64>, usize) = match output {
                    Output::Native12 => (
                        bytes
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|v| f64::from(u16::from_le_bytes(*v)) / 4095.0)
                            .collect(),
                        4,
                    ),
                    Output::Srgb8 => (bytes.iter().map(|&v| f64::from(v) / 255.0).collect(), 4),
                    _ => (
                        bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|v| f64::from(f32::from_le_bytes(*v)))
                            .collect(),
                        if matches!(output, Output::LinearBt709 | Output::AbsoluteLinearBt709) {
                            4
                        } else {
                            1
                        },
                    ),
                };
                assert_eq!(samples.len(), 37 * 19 * channels);
                for (pixel, actual) in samples.chunks_exact(channels).enumerate() {
                    let reference = &expected[(frames * 37 * 19 + pixel) * 4..][..4];
                    let rgb = [reference[0], reference[1], reference[2]].map(f64::from);
                    let (rgb, source_transfer) = if let Some(linear) = &linear_original
                        && matches!(
                            output,
                            Output::LinearBt709 | Output::AbsoluteLinearBt709 | Output::Srgb8
                        ) {
                        (
                            [linear[pixel][0], linear[pixel][1], linear[pixel][2]],
                            TransferFunction::Linear,
                        )
                    } else {
                        (rgb, source.transfer)
                    };
                    let target = if matches!(output, Output::Srgb8) {
                        TransferFunction::Srgb
                    } else {
                        TransferFunction::Linear
                    };
                    let converted = oracle::convert(rgb, source_transfer, target, matrix);
                    let interval = oracle::interval(
                        rgb,
                        source_transfer,
                        target,
                        matrix,
                        f64::from(tolerance(&case)),
                    );
                    for (channel, &actual) in actual.iter().enumerate() {
                        let (expected, mut low, mut high, quantization) = match output {
                            Output::ColorPlane | Output::AlphaPlane => {
                                let channel = if matches!(output, Output::AlphaPlane) {
                                    3
                                } else {
                                    usize::from(!case.profile.grayscale)
                                };
                                let value = f64::from(reference[channel]);
                                let bound = if channel == 3 {
                                    2e-6
                                } else {
                                    f64::from(tolerance(&case))
                                } * (1.0 + value.abs());
                                (value, value - bound, value + bound, 0.0)
                            }
                            _ if channel == 3 => {
                                let value = f64::from(reference[3]);
                                (
                                    value,
                                    value - 2e-6,
                                    value + 2e-6,
                                    match output {
                                        Output::Srgb8 => 1.0 / 255.0,
                                        Output::Native12 => 1.0 / 4095.0,
                                        _ => 0.0,
                                    },
                                )
                            }
                            Output::Native12 => {
                                let value = f64::from(reference[channel]);
                                let bound = f64::from(tolerance(&case)) * (1.0 + value.abs());
                                (value, value - bound, value + bound, 1.0 / 4095.0)
                            }
                            _ => (
                                converted[channel],
                                interval[channel][0],
                                interval[channel][1],
                                if matches!(output, Output::Srgb8) {
                                    1.0 / 255.0
                                } else {
                                    0.0
                                },
                            ),
                        };
                        if quantization != 0.0 {
                            low = low.clamp(0.0, 1.0);
                            high = high.clamp(0.0, 1.0);
                        }
                        // The color-output matrix coefficients are F32; its independent scalar
                        // tests also validate conversion without codec reconstruction error.
                        let packing = if matches!(
                            output,
                            Output::LinearBt709 | Output::AbsoluteLinearBt709 | Output::Srgb8
                        ) && channel < 3
                        {
                            5e-6 * (1.0 + expected.abs())
                        } else {
                            0.0
                        };
                        assert!(
                            actual.is_finite()
                                && actual >= low - quantization - packing
                                && actual <= high + quantization + packing,
                            "{}/{output:?}/{frames}/{pixel}/{channel}: {actual}, f64 {expected}, interval [{low}, {high}], packing {packing}, quantization {quantization}",
                            case.name
                        );
                    }
                }
                frames += 1;
            }
            assert_eq!(frames, if case.sequence { 4 } else { 1 });
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}
