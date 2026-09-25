//! Large-transform precision uses the pinned scalar oracle's exact inverse division.
//! Keep native SIMD's independent sRGB code comparison as well.

use super::*;
use jxl_gpu_formats::{RgbChannelOrder, TransferFunction};
use jxl_test_support::gpu::planes::open_fragmented;
use jxl_test_support::oracles::extra_channels::{floats, libjxl_output, rust_planes};
use jxl_test_support::oracles::progressive::scalar_linear_updates;
use jxl_wgpu_decode::WgpuDecodeEngine;

pub(super) struct LinearPixelOracles {
    whole: GpuDecoder<WgpuDecodeEngine>,
    windowed: GpuDecoder<WgpuDecodeEngine>,
    readback: ImageReadbackPipeline,
}

impl LinearPixelOracles {
    pub(super) fn new(backend: &WgpuBackend) -> Self {
        Self {
            whole: GpuDecoder::wgpu(backend.clone()).unwrap(),
            windowed: GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
            ),
            readback: ImageReadbackPipeline::new(backend),
        }
    }

    pub(super) fn check(&self, encoded: &[u8], input: &[[u32; 3]], bits: u8) {
        let native = libjxl_output(encoded, &["--original"]).expect("native SIMD oracle required");
        let (rust, extras) = rust_planes(encoded);
        assert!(extras.is_empty());
        assert_eq!(native.len(), rust.len());
        for (&a, &b) in native.iter().zip(&rust) {
            assert!(a.is_finite() && b.is_finite());
            let code = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            assert!(code(a).abs_diff(code(b)) <= 1);
        }
        check_quality(&native, input, bits);
        let scalar = scalar_linear_updates(encoded);
        let scalar = floats(&scalar.last().filter(|u| u.complete).unwrap().pixels);
        let rust_linear: Vec<_> = rust
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i % 4 == 3 {
                    return v;
                }
                let sign = v.signum();
                let v = f64::from(v.abs());
                if v <= 0.04045 {
                    sign * (v / 12.92) as f32
                } else {
                    sign * ((v + 0.055) / 1.055).powf(2.4) as f32
                }
            })
            .collect();
        compare("Rust/scalar", &rust_linear, &scalar);
        let color = match vardct_rgb8_format().color_spec {
            ColorSpecification::Defined(mut color) => {
                color.transfer = TransferFunction::Linear;
                ColorSpecification::Defined(color)
            }
            _ => panic!("explicit RGB color specification required"),
        };
        let format = PixelFormat::rgb_f32(RgbChannelOrder::Rgba, false, color);
        let mut whole = Vec::new();
        for (decoder, fragmented) in [(&self.whole, false), (&self.windowed, true)] {
            let request = GpuOutputRequest::color(format.clone()).unwrap();
            let mut session = if fragmented {
                open_fragmented(decoder, encoded, request)
            } else {
                decoder.open(encoded, request).unwrap()
            };
            let frame = session.next_frame().unwrap().unwrap();
            let read = self
                .readback
                .submit(frame.output())
                .unwrap()
                .wait()
                .unwrap();
            let gpu = floats(&read.frame.outputs[0].bytes);
            compare("GPU/scalar", &gpu, &scalar);
            compare("GPU/Rust", &gpu, &rust_linear);
            if fragmented {
                assert_eq!(gpu, whole);
            } else {
                whole = gpu;
            }
            drop(read);
            drop(frame);
            assert!(session.next_frame().unwrap().is_none());
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}

fn compare(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert!(
            (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
            "{label} linear sample {i}: {a} vs {b}"
        );
    }
}
