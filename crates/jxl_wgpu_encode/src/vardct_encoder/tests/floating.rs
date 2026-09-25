//! Floating source semantics are checked independently of the GPU bit-field conversion.

mod boundaries;

use super::*;
use crate::{
    AnimationHeader, BackendError, ColorSampleFormat, Determinism, EncodeProfile,
    FrameEncodeRequest, FrameIndex, FrameOptions, GpuEncodeBackend, GpuEncodeJob, GpuFrameSource,
    VarDctBackend,
};

use jxl_gpu_formats::{FloatPrecision, SampleKind};

pub(super) fn all_precisions() -> Vec<FloatPrecision> {
    let result: Vec<_> = (2..=8)
        .flat_map(|exponent| {
            (2..=23).map(move |fraction| {
                FloatPrecision::new(1 + exponent + fraction, exponent).unwrap()
            })
        })
        .collect();
    assert_eq!(result.len(), 154);
    result
}

pub(super) fn format(precision: FloatPrecision) -> ColorSampleFormat {
    ColorSampleFormat::float(
        crate::ColorChannels::Rgb,
        precision.bits(),
        precision.exponent_bits(),
    )
    .unwrap()
}

/// Arithmetic decoding in F64, independent of the shader's binary32 field rebasing.
pub(super) fn value(word: u32, precision: FloatPrecision) -> f64 {
    let fraction_bits = precision.bits() - precision.exponent_bits() - 1;
    let fraction = word & ((1 << fraction_bits) - 1);
    let exponent = (word >> fraction_bits) & ((1 << precision.exponent_bits()) - 1);
    let bias = (1i32 << (precision.exponent_bits() - 1)) - 1;
    let magnitude = if exponent == 0 {
        f64::from(fraction) * 2f64.powi(1 - bias - i32::from(fraction_bits))
    } else {
        (1.0 + f64::from(fraction) / f64::from(1u32 << fraction_bits))
            * 2f64.powi(exponent as i32 - bias)
    };
    if word & (1 << (precision.bits() - 1)) != 0 {
        -magnitude
    } else {
        magnitude
    }
}

pub(super) fn pixels(width: usize, height: usize, p: FloatPrecision) -> Vec<[u32; 3]> {
    let fraction_bits = p.bits() - p.exponent_bits() - 1;
    let bias = (1 << (p.exponent_bits() - 1)) - 1;
    let one = bias << fraction_bits;
    let half = if bias == 1 {
        1 << (fraction_bits - 1)
    } else {
        (bias - 1) << fraction_bits
    };
    let three_quarters = half + (1 << (fraction_bits - 2 + u8::from(bias != 1)));
    let values = [
        0,
        1,
        (1 << fraction_bits) - 1,
        1 << (p.bits() - 1),
        half,
        three_quarters,
        one,
    ];
    (0..width * height)
        .map(|i| std::array::from_fn(|c| values[(i * 5 + c * 3) % values.len()]))
        .collect()
}

pub(super) fn source(
    context: &WgpuContext,
    w: usize,
    h: usize,
    p: FloatPrecision,
    words: &[[u32; 3]],
) -> BufferImageSource {
    // Standard IEEE encodings deliberately use the equivalent CustomFloat spelling.
    precision::source_with_kind(
        context,
        w,
        h,
        p.bits(),
        SampleKind::CustomFloat(p),
        words,
        true,
    )
}

fn request(w: usize, h: usize, config: &VarDctConfig) -> FrameEncodeRequest {
    FrameEncodeRequest {
        frame_index: FrameIndex::new(0),
        is_last: true,
        profile: EncodeProfile::VarDct {
            quantization: config.quantization,
        },
        progressive: config.progressive.clone(),
        minimum_determinism: Determinism::SameDevice,
        animation: AnimationHeader::Still,
        canvas_width: w as u32,
        canvas_height: h as u32,
        options: FrameOptions::default(),
    }
}

fn normalized(words: &[[u32; 3]], p: FloatPrecision) -> Vec<[f64; 3]> {
    words.iter().map(|v| v.map(|v| value(v, p))).collect()
}

pub(super) fn check_header(encoded: &[u8], p: FloatPrecision) {
    let inventory = jxl_gpu_bitstream::parse(encoded, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(inventory.image_header.bit_depth, format(p).bit_depth());
    assert!(!inventory.image_header.modular_16bit_buffers);
}

pub(super) fn check_pixels(
    oracles: &color::PixelOracles,
    encoded: &[u8],
    words: &[[u32; 3]],
    p: FloatPrecision,
) -> Vec<f32> {
    let (actual, rust) = oracles.check_decoders_with_rust(encoded);
    precision::check_normalized_quality(&actual, &normalized(words, p));
    rust
}

fn single(
    context: &WgpuContext,
    config: &VarDctConfig,
    strategy: VarDctStrategy,
    oracle: &native::Oracle,
    words: &[[u32; 3]],
) -> Vec<u8> {
    let extent = strategy.pixel_extent();
    let (w, h) = (extent.width as usize, extent.height as usize);
    let p = config.sample_format.float_precision().unwrap();
    let components: Vec<_> = normalized(words, p)
        .into_iter()
        .map(|v| match config.color_transform {
            VarDctColorTransform::Xyb => reference::xyb_normalized(v),
            VarDctColorTransform::Original => v,
        })
        .collect();
    let coefficients = native::forward_samples(&components, w, h, oracle);
    let backend = VarDctBackend::new_with_config(context, strategy, config.clone()).unwrap();
    let input = source(context, w, h, p, words);
    assert!(backend.memory_plan(&input).unwrap().source_validation_bytes >= 256);
    let (ac, bits, artifacts) = backend
        .submit(
            context,
            GpuFrameSource::Buffer(input),
            &request(w, h, config),
        )
        .unwrap()
        .wait_with_ac_for_test()
        .unwrap();
    native::check_ac(&ac, bits, &coefficients, oracle, config.clone());
    let mut encoded = super::image_header_with_color(
        w as u32,
        h as u32,
        AnimationHeader::Still,
        &backend.color_plan,
    )
    .unwrap()
    .bytes()
    .to_vec();
    encoded.extend_from_slice(assemble_frame(artifacts.packets).unwrap().bytes());
    encoded
}

#[test]
fn floating_precision_all_formats_match_native_coefficients_and_pixels() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let oracles = color::PixelOracles::new(&gpu);
    let native = native::native_oracles();
    for p in all_precisions() {
        for color in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
            eprintln!("{p:?}/{color:?}");
            let config = VarDctConfig {
                sample_format: format(p),
                ..precision::configuration(8, color)
            };
            let words = pixels(8, 8, p);
            let encoded = single(&context, &config, VarDctStrategy::Dct8, &native[0], &words);
            check_header(&encoded, p);
            check_pixels(&oracles, &encoded, &words, p);
            let mut clean = precision::source_with_kind(
                &context,
                8,
                8,
                p.bits(),
                format(p).pixel_format().sample_kind,
                &words,
                false,
            );
            clean.layout.format = format(p).pixel_format();
            assert_eq!(
                encoded,
                VarDctEncoder::new_with_config(
                    context.clone(),
                    VarDctStrategy::Dct8,
                    config.clone()
                )
                .unwrap()
                .encode(clean)
                .unwrap()
            );
            for (w, h, mapped) in [(25, 17, true), (259, 3, false)] {
                let config = VarDctConfig {
                    progressive: progressive::combined(),
                    group_order: crate::VarDctGroupOrder::saliency_first(),
                    ..config.clone()
                };
                let words = pixels(w, h, p);
                let input = source(&context, w, h, p, &words);
                let encoded = if mapped {
                    VarDctEncoder::new_with_strategy_map(
                        context.clone(),
                        mixed::packed_map(w as u32, h as u32, false),
                        config,
                    )
                    .unwrap()
                    .encode(input)
                } else {
                    TiledVarDctEncoder::new_with_config(context.clone(), config)
                        .unwrap()
                        .encode(input)
                }
                .unwrap();
                check_header(&encoded, p);
                check_pixels(&oracles, &encoded, &words, p);
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn floating_precision_all_strategies_preserve_custom_orders_and_lf_metadata() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let oracles = precision::linear::LinearPixelOracles::new(&gpu);
    let native = native::native_oracles();
    for (p, color) in [
        (FloatPrecision::BINARY16, VarDctColorTransform::Xyb),
        (FloatPrecision::BINARY32, VarDctColorTransform::Original),
    ] {
        for (strategy, oracle) in VarDctStrategy::ALL.into_iter().zip(&native) {
            eprintln!("{p:?}/{color:?}/{strategy:?}");
            let extent = strategy.pixel_extent();
            let words = pixels(extent.width as usize, extent.height as usize, p);
            let config = VarDctConfig {
                sample_format: format(p),
                coefficient_orders: orders::selected([strategy]),
                lf_metadata: custom_lf_metadata(),
                quantization: VarDctQuantization::new(
                    65536,
                    256,
                    crate::VarDctHfMultiplier::new(256).unwrap(),
                )
                .unwrap(),
                ..precision::configuration(8, color)
            };
            let encoded = single(&context, &config, strategy, oracle, &words);
            check_header(&encoded, p);
            oracles.check_normalized(&encoded, &normalized(&words, p));
        }
    }
}

#[test]
fn floating_precision_rejects_non_finite_sources_before_artifact_publication() {
    let context = test_context().expect("actual GPU required");
    for p in all_precisions() {
        let fraction = p.bits() - p.exponent_bits() - 1;
        let infinity = ((1u32 << p.exponent_bits()) - 1) << fraction;
        let sign = 1u32 << (p.bits() - 1);
        let config = VarDctConfig {
            sample_format: format(p),
            ..precision::configuration(8, VarDctColorTransform::Original)
        };
        for mapped in [false, true] {
            let config = VarDctConfig {
                color_transform: if mapped {
                    VarDctColorTransform::Xyb
                } else {
                    VarDctColorTransform::Original
                },
                ..config.clone()
            };
            let (w, h) = (13, 9);
            let encoder = if mapped {
                VarDctBackend::new_with_strategy_map(
                    &context,
                    mixed::packed_map(w as u32, h as u32, false),
                    config.clone(),
                )
            } else {
                VarDctBackend::new_tiled_dct8_with_config(&context, config.clone())
            }
            .unwrap();
            for (i, invalid) in [
                infinity,
                infinity | sign,
                infinity | 1,
                infinity | sign | (1 << (fraction - 1)),
            ]
            .into_iter()
            .enumerate()
            {
                let mut words = pixels(w, h, p);
                words[if i % 2 == 0 { 0 } else { w * h - 1 }][i % 3] = invalid;
                let result = encoder
                    .submit(
                        &context,
                        GpuFrameSource::Buffer(source(&context, w, h, p, &words)),
                        &request(w, h, &config),
                    )
                    .unwrap()
                    .wait();
                assert!(
                    matches!(
                        result,
                        Err(EncodeError::Backend(BackendError::VarDctNonFiniteSource))
                    ),
                    "{p:?}/{mapped}/{invalid:x}"
                );
                assert_eq!(context.memory_stats().reserved_bytes, 0);
            }
        }
    }
}
