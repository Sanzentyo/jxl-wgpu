//! Original-sRGB coding: independent coefficients, pixels, headers and ownership.

use super::*;
use crate::{
    AnimationHeader, Determinism, EncodeProfile, FrameEncodeRequest, FrameIndex, FrameOptions,
    GpuEncodeBackend, GpuFrameSource, VarDctBackend,
};
use jxl_gpu_formats::{PixelFormat, RgbChannelOrder};
use jxl_test_support::gpu::planes::open_fragmented;
use jxl_test_support::oracles::extra_channels::{floats, libjxl_output, rust_planes};
use jxl_wgpu_decode::WgpuDecodeEngine;

fn configuration() -> VarDctConfig {
    VarDctConfig {
        color_transform: VarDctColorTransform::Original,
        quantization: VarDctQuantization::new(
            35_252,
            16,
            super::super::VarDctHfMultiplier::new(12).unwrap(),
        )
        .unwrap(),
        ..Default::default()
    }
}

#[test]
fn original_rgb_tiled_still_interoperates() {
    let context = test_context().expect("actual GPU required for original RGB encoding");
    let (width, height) = (13, 21);
    let pixels = (0..width * height)
        .map(|i| {
            let (x, y) = (i % width, i / width);
            [
                (x * 19 % 256) as u8,
                (y * 13 % 256) as u8,
                ((x * 7 + y * 9) % 256) as u8,
            ]
        })
        .collect::<Vec<_>>();
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), configuration()).unwrap();
    assert_eq!(encoder.color_transform(), VarDctColorTransform::Original);
    let source = padded_rgb_source_sized(&context, width, height, &pixels);
    let encoded = encoder.encode(source).unwrap();
    let inventory = jxl_gpu_bitstream::parse(&encoded, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert!(!inventory.image_header.xyb_encoded);
    assert!(!inventory.frames[0].do_ycbcr);
    assert_eq!(
        (
            inventory.frames[0].x_qm_scale,
            inventory.frames[0].b_qm_scale
        ),
        (2, 2)
    );
    let decoded = quantization::assert_decoders_agree(&encoded, width, height);
    assert!(psnr(&pixels, &decoded) > 30.0);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}

fn texture(width: usize, height: usize) -> Vec<[u8; 3]> {
    (0..width * height)
        .map(|i| {
            let (x, y) = (i % width, i / width);
            [
                ((x * 37 + y * 19) % 256) as u8,
                ((x * 13 + y * 53) % 256) as u8,
                ((x * 71 + y * 11) % 256) as u8,
            ]
        })
        .collect()
}

fn compare(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert!(
            (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
            "sample {i}: {a} vs {b}"
        );
        let code = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
        assert!(code(a).abs_diff(code(b)) <= 1, "RGB8 sample {i}");
    }
}

pub(super) struct PixelOracles {
    whole: GpuDecoder<WgpuDecodeEngine>,
    windowed: GpuDecoder<WgpuDecodeEngine>,
    readback: ImageReadbackPipeline,
}

impl PixelOracles {
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

    fn check(&self, encoded: &[u8], input: &[[u8; 3]]) {
        let native = self.check_decoders(encoded);
        let codes: Vec<_> = native
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| {
                p[..3]
                    .iter()
                    .map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
            })
            .collect();
        assert!(
            psnr(input, &codes) > 30.0,
            "source PSNR {}",
            psnr(input, &codes)
        );
    }

    pub(super) fn check_decoders(&self, encoded: &[u8]) -> Vec<f32> {
        self.check_decoders_with_rust(encoded).0
    }

    pub(super) fn check_decoders_with_rust(&self, encoded: &[u8]) -> (Vec<f32>, Vec<f32>) {
        self.check_declared_color(encoded, vardct_rgb8_format().color_spec)
    }

    pub(super) fn check_declared_color(
        &self,
        encoded: &[u8],
        color: jxl_gpu_formats::ColorSpecification,
    ) -> (Vec<f32>, Vec<f32>) {
        let (rust, extras) = rust_planes(encoded);
        assert!(extras.is_empty());
        let native = self.check_with_reference(encoded, color, &rust);
        (native, rust)
    }

    pub(super) fn check_with_reference(
        &self,
        encoded: &[u8],
        color: jxl_gpu_formats::ColorSpecification,
        reference: &[f32],
    ) -> Vec<f32> {
        let native =
            libjxl_output(encoded, &["--original"]).expect("required pinned native RGB oracle");
        compare(reference, &native);
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
            compare(&gpu, &native);
            compare(&gpu, reference);
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
        native
    }
}

#[test]
fn original_rgb_all_strategies_match_native_coefficients_and_pixels() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("actual GPU required for original RGB encoding");
    let context = WgpuContext::from_backend(&backend);
    let pixels = PixelOracles::new(&backend);
    for (strategy, oracle) in VarDctStrategy::ALL
        .into_iter()
        .zip(native::native_oracles())
    {
        let extent = strategy.pixel_extent();
        let (w, h) = (extent.width as usize, extent.height as usize);
        let input = texture(w, h);
        let normalized: Vec<_> = input
            .iter()
            .map(|p| p.map(|c| f64::from(c) / 255.0))
            .collect();
        let coefficients = native::forward_samples(&normalized, w, h, &oracle);
        let config = VarDctConfig {
            coefficient_orders: orders::selected([strategy]),
            lf_metadata: custom_lf_metadata(),
            ..configuration()
        };
        let encoder = VarDctBackend::new_with_config(&context, strategy, config.clone()).unwrap();
        let source = padded_rgb_source_sized(&context, w, h, &input);
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::VarDct {
                quantization: config.quantization,
            },
            progressive: config.progressive.clone(),
            minimum_determinism: Determinism::SameDevice,
            animation: AnimationHeader::Still,
            canvas_width: extent.width,
            canvas_height: extent.height,
            options: FrameOptions::default(),
        };
        let job = encoder
            .submit(&context, GpuFrameSource::Buffer(source.clone()), &request)
            .unwrap();
        let (words, bits, artifacts) = job.wait_with_ac_for_test().unwrap();
        assert!(native::check_ac(&words, bits, &coefficients, &oracle, config.clone()) > 0);
        let frame = assemble_frame(artifacts.packets).unwrap();
        let mut encoded = super::image_header_with_color(
            extent.width,
            extent.height,
            AnimationHeader::Still,
            &encoder.color_plan,
        )
        .unwrap()
        .bytes()
        .to_vec();
        encoded.extend_from_slice(frame.bytes());
        let convenience =
            VarDctEncoder::new_with_config(context.clone(), strategy, config).unwrap();
        assert_eq!(
            convenience.color_transform(),
            VarDctColorTransform::Original
        );
        assert_eq!(convenience.encode(source).unwrap(), encoded);
        let inventory = jxl_gpu_bitstream::parse(&encoded, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert!(!inventory.image_header.xyb_encoded);
        assert!(!inventory.frames[0].do_ycbcr);
        assert_eq!(
            (
                inventory.frames[0].x_qm_scale,
                inventory.frames[0].b_qm_scale
            ),
            (2, 2)
        );
        pixels.check(&encoded, &input);
    }
}

#[test]
fn original_rgb_progressive_mapped_and_tiled_variants_keep_edges_and_matrices() {
    let (device, queue, info) = test_device().expect("actual GPU required");
    let backend = WgpuBackend::from_device(
        device.as_ref().clone(),
        queue.as_ref().clone(),
        info.clone(),
        WgpuBackendConfig {
            enable_timestamps: false,
            ..Default::default()
        },
    )
    .unwrap();
    let pixels = PixelOracles::new(&backend);
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Lanes64,
        KernelVariant::Lanes128,
        KernelVariant::Lanes256,
    ] {
        let context = test_context_with_variants(
            &device,
            &queue,
            &info,
            &[(FORWARD_KERNEL_KEY, variant), (TILED_KERNEL_KEY, variant)],
        )
        .unwrap();
        for (w, h) in [(1, 1), (25, 17), (259, 19)] {
            let input = texture(w, h);
            let source = padded_rgb_source_sized(&context, w, h, &input);
            let map = mixed::packed_map(w as u32, h as u32, false);
            let strategies = map
                .transforms()
                .iter()
                .map(|t| t.strategy)
                .chain([VarDctStrategy::Dct8])
                .collect::<Vec<_>>();
            let config = VarDctConfig {
                progressive: progressive::combined(),
                group_order: super::super::VarDctGroupOrder::saliency_first(),
                coefficient_orders: orders::selected(strategies.iter().copied()),
                dequant_matrices: if w == 25 {
                    matrices::selected()
                } else {
                    raw_matrices::selected(strategies)
                },
                ..configuration()
            };
            let tiled =
                TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
            assert_eq!(tiled.workgroup_variant(), variant);
            let encoded = tiled.encode_container(source.clone()).unwrap();
            pixels.check(&encoded, &input);
            let mapped =
                VarDctEncoder::new_with_strategy_map(context.clone(), map, config).unwrap();
            assert_eq!(mapped.workgroup_variant(), variant);
            let encoded = mapped.encode(source).unwrap();
            pixels.check(&encoded, &input);
            assert_eq!(context.memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn original_rgb_exact_admission_cancellation_and_wrong_color_rejection() {
    let context = test_context().expect("actual GPU required");
    let input = texture(25, 17);
    let source = padded_rgb_source_sized(&context, 25, 17, &input);
    let config = VarDctConfig {
        progressive: progressive::combined(),
        ..configuration()
    };
    for mapped in [false, true] {
        let make = |context: WgpuContext, color_transform| {
            let config = VarDctConfig {
                color_transform,
                ..config.clone()
            };
            if mapped {
                VarDctBackend::new_with_strategy_map(
                    &context,
                    mixed::packed_map(25, 17, false),
                    config,
                )
            } else {
                VarDctBackend::new_tiled_dct8_with_config(&context, config)
            }
            .unwrap()
        };
        let original = make(context.clone(), VarDctColorTransform::Original);
        let xyb = make(context.clone(), VarDctColorTransform::Xyb);
        assert_eq!(
            original.memory_plan(&source).unwrap(),
            xyb.memory_plan(&source).unwrap()
        );
        let bytes = original.memory_plan(&source).unwrap().owned_bytes_per_job;
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::VarDct {
                quantization: config.quantization,
            },
            progressive: config.progressive.clone(),
            minimum_determinism: Determinism::SameDevice,
            animation: AnimationHeader::Still,
            canvas_width: 25,
            canvas_height: 17,
            options: FrameOptions::default(),
        };
        for limit in [bytes - 1, bytes] {
            let context = WgpuContext::with_memory_budget(
                Arc::new(context.device().clone()),
                Arc::new(context.queue().clone()),
                NonZeroU64::new(limit).unwrap(),
            )
            .unwrap();
            let encoder = make(context.clone(), VarDctColorTransform::Original);
            for format in [
                PixelFormat::gray8(false, false, jxl_gpu_formats::ColorSpecification::Default),
                PixelFormat::rgb8(
                    RgbChannelOrder::Rgb,
                    false,
                    jxl_gpu_formats::ColorSpecification::Undefined,
                ),
            ] {
                let mut wrong = source.clone();
                wrong.layout.format = format;
                assert!(matches!(
                    encoder.submit(&context, GpuFrameSource::Buffer(wrong), &request),
                    Err(EncodeError::Unsupported(UnsupportedFeature::InputFormat))
                ));
            }
            assert_eq!(context.memory_stats().reserved_bytes, 0);
            let job = encoder.submit(&context, GpuFrameSource::Buffer(source.clone()), &request);
            if limit < bytes {
                assert!(matches!(job, Err(EncodeError::MemoryBackpressure(_))));
                assert_eq!(context.memory_stats().reserved_bytes, 0);
                continue;
            }
            let job = job.unwrap();
            assert_eq!(context.memory_stats().reserved_bytes, bytes);
            assert!(matches!(
                encoder.submit(&context, GpuFrameSource::Buffer(source.clone()), &request),
                Err(EncodeError::MemoryBackpressure(_))
            ));
            drop(job);
            context
                .device()
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while context.memory_stats().reserved_bytes != 0 && std::time::Instant::now() < deadline
            {
                context.device().poll(wgpu::PollType::Poll).unwrap();
                std::thread::yield_now();
            }
            assert_eq!(context.memory_stats().reserved_bytes, 0);
            use crate::GpuEncodeJob;
            encoder
                .submit(&context, GpuFrameSource::Buffer(source.clone()), &request)
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(context.memory_stats().reserved_bytes, 0);
        }
    }
}
