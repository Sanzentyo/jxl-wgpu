use super::*;
use jxl_gpu_formats::{ColorSpecification, TransferFunction};

// Independent f64 equations, evaluated from the unmodified official sRGB samples.
fn converted(encoded: f64, target: TransferFunction) -> f64 {
    let magnitude = encoded.abs();
    let linear = if magnitude <= 0.04045 {
        magnitude / 12.92
    } else {
        ((magnitude + 0.055) / 1.055).powf(2.4)
    };
    let signed = linear.copysign(encoded);
    match target {
        TransferFunction::Linear => signed,
        TransferFunction::Bt709 if signed < 0.018 => 4.5 * signed,
        TransferFunction::Bt709 => 1.099 * signed.powf(0.45) - 0.099,
        TransferFunction::Bt2020 => {
            let nonlinear = if linear < 0.018_053_968_510_807 {
                4.5 * linear
            } else {
                1.099_296_826_809_44 * linear.powf(0.45) - 0.099_296_826_809_44
            };
            nonlinear.copysign(encoded)
        }
        _ => unreachable!("test transfer"),
    }
}

#[test]
fn signed_color_uses_the_requested_sdr_transfer_without_clipping() {
    let reference = cases::CASES
        .iter()
        .find(|case| case.name == "alpha_triangles")
        .unwrap()
        .load();
    assert!(reference.pixels.iter().any(|&value| value < -0.5));
    assert!(reference.pixels.iter().any(|&value| value > 1.5));
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for transfer in [
        TransferFunction::Linear,
        TransferFunction::Bt709,
        TransferFunction::Bt2020,
    ] {
        let ColorSpecification::Defined(mut color) =
            jxl_wgpu_decode::vardct_rgb8_format().color_spec
        else {
            unreachable!()
        };
        color.transfer = transfer;
        let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
            RgbChannelOrder::Rgba,
            false,
            ColorSpecification::Defined(color),
        ))
        .unwrap()
        .with_alpha_output_policy(AlphaOutputPolicy::Preserve);
        let words = decode(&backend, &decoder, &reference, request, false);
        let mut peak = 0.0_f64;
        for (index, (&actual, &encoded)) in words.iter().zip(&reference.pixels).enumerate() {
            if index % 4 == 3 {
                assert_eq!(actual, encoded.to_bits(), "alpha {index}");
                continue;
            }
            let expected = converted(f64::from(encoded), transfer);
            let error = (f64::from(f32::from_bits(actual)) - expected).abs();
            assert!(error.is_finite());
            // Sixteen F32 epsilons allow the independent pow implementation and normalization;
            // all original sRGB conformance bounds remain independently enforced by run_case.
            let bound = 16.0 * f64::from(f32::EPSILON) * expected.abs().max(1.0);
            assert!(error <= bound, "{transfer:?} {index}: {error} > {bound}");
            peak = peak.max(error);
        }
        eprintln!("extended {transfer:?}: peak {peak}");
    }
}
