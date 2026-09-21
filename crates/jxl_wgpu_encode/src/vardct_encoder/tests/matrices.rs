//! Independent matrix parser/expander and GPU custom-matrix regressions.

use super::*;
use crate::{VarDctDequantMatrices, VarDctMatrixEncoding};
use jxl_oxide_common::Bundle;
use jxl_vardct::{DequantMatrixSet, DequantMatrixSetParams, TransformType};

fn bands(count: usize) -> [Vec<FiniteF16>; 3] {
    std::array::from_fn(|channel| {
        (0..count)
            .map(|band| {
                f16(if band == 0 {
                    [0x5200, 0x4880, 0x4800][channel]
                } else {
                    [0xb400, 0x3000, 0x3800][(band + channel) % 3]
                })
            })
            .collect()
    })
}

fn encoding(mode: u8, count: usize) -> VarDctMatrixEncoding {
    fn fixed<const N: usize>(large: bool) -> [[FiniteF16; N]; 3] {
        std::array::from_fn(|channel| {
            std::array::from_fn(|index| {
                f16(if large {
                    [0x4000, 0x4400, 0x4800]
                } else {
                    [0x3c00, 0x4000, 0x4200]
                }[(channel + index) % 3])
            })
        })
    }
    match mode {
        0 => VarDctMatrixEncoding::Default,
        1 => VarDctMatrixEncoding::Hornuss(fixed(true)),
        2 => VarDctMatrixEncoding::Dct2(fixed(true)),
        3 => VarDctMatrixEncoding::Dct4 {
            params: fixed(false),
            dct_params: bands(count),
        },
        4 => VarDctMatrixEncoding::Dct4x8 {
            params: fixed(false),
            dct_params: bands(count),
        },
        5 => VarDctMatrixEncoding::Afv {
            params: fixed(false),
            dct_params: bands(count),
            dct4x4_params: bands(17 - count),
        },
        6 => VarDctMatrixEncoding::Dct(bands(count)),
        _ => unreachable!(),
    }
}

pub(super) fn selected() -> VarDctDequantMatrices {
    VarDctStrategy::ALL
        .into_iter()
        .fold(VarDctDequantMatrices::default(), |set, strategy| {
            let mode = match strategy {
                VarDctStrategy::Hornuss => 1,
                VarDctStrategy::Dct2x2 => 2,
                VarDctStrategy::Dct4x4 => 3,
                VarDctStrategy::Dct4x8 | VarDctStrategy::Dct8x4 => 4,
                VarDctStrategy::Afv0
                | VarDctStrategy::Afv1
                | VarDctStrategy::Afv2
                | VarDctStrategy::Afv3 => 5,
                _ => 6,
            };
            set.with_matrix(strategy, encoding(mode, 3)).unwrap()
        })
}

pub(super) fn oracle(set: &VarDctDequantMatrices) -> DequantMatrixSet {
    let mut output = BitWriter::new();
    set.write(&mut output).unwrap();
    let bits = output.bit_len();
    let bytes = output.into_bytes();
    let mut reader = jxl_bitstream::Bitstream::new(&bytes);
    let pool = jxl_threadpool::JxlThreadPool::none();
    let parsed = DequantMatrixSet::parse(
        &mut reader,
        DequantMatrixSetParams::new(8, 1, None, None, &pool),
    )
    .unwrap();
    assert_eq!(reader.num_read_bits(), bits);
    parsed
}

pub(super) fn oracle_scales(
    oracle: &DequantMatrixSet,
    strategy: VarDctStrategy,
    set: &VarDctDequantMatrices,
) -> [Vec<f64>; 3] {
    if let Some(encoding) = set.encoding(strategy) {
        let parameters: Vec<_> = match encoding {
            VarDctMatrixEncoding::Hornuss(p) => p.iter().flatten().map(|v| v.to_bits()).collect(),
            VarDctMatrixEncoding::Dct2(p) => p.iter().flatten().map(|v| v.to_bits()).collect(),
            _ => Vec::new(),
        };
        if !parameters.is_empty() {
            // These modes are checked against native decode/expansion, because the older
            // jxl-vardct parser omits their wire-unit ×64 conversion.
            let native = jxl_test_support::oracles::vardct_matrices::records()
                .iter()
                .find(|r| r.mode == encoding.encoding_id() && r.variant == 1)
                .unwrap();
            assert_eq!(parameters, native.parameters);
            return std::array::from_fn(|channel| {
                native
                    .scales
                    .iter()
                    .map(|v| f64::from(v[channel]))
                    .collect()
            });
        }
    }
    use TransformType::*;
    // Independent decoder enum mapping; do not use the production expansion.
    let transform = [
        Dct8, Hornuss, Dct2, Dct4, Dct16, Dct32, Dct16x8, Dct8x16, Dct32x8, Dct8x32, Dct32x16,
        Dct16x32, Dct4x8, Dct8x4, Afv0, Afv1, Afv2, Afv3, Dct64, Dct64x32, Dct32x64, Dct128,
        Dct128x64, Dct64x128, Dct256, Dct256x128, Dct128x256,
    ][strategy.codestream_id() as usize];
    let extent = strategy.pixel_extent();
    std::array::from_fn(|channel| {
        let raster = if strategy.needs_transpose() {
            oracle.get_transposed(channel, transform)
        } else {
            oracle.get(channel, transform)
        };
        let mut wire = vec![0.0; raster.len()];
        for y in 0..extent.height {
            for x in 0..extent.width {
                let index = if strategy.is_special() || extent.height < extent.width {
                    y * extent.width + x
                } else {
                    x * extent.height + y
                } as usize;
                wire[index] = f64::from(raster[(y * extent.width + x) as usize]);
            }
        }
        wire
    })
}

#[test]
fn all_parametric_matrix_modes_bands_and_orientations_match_independent_expansion() {
    for count in [1, 3, 16] {
        // Default matrices have a pinned native libjxl oracle in jxl_gpu_protocol.
        // jxl-vardct 0.11.1 has different default Y/B constants for the 256×128 family;
        // use this independent parser only for explicitly serialized parameters here.
        for mode in 1..=6 {
            let mut set = VarDctDequantMatrices::default();
            for strategy in VarDctStrategy::ALL {
                if (1..=5).contains(&mode) && strategy.pixel_extent().area().unwrap() != 64 {
                    continue;
                }
                set = set.with_matrix(strategy, encoding(mode, count)).unwrap();
            }
            let oracle = oracle(&set);
            for strategy in VarDctStrategy::ALL {
                if set.encoding(strategy).is_none() {
                    continue;
                }
                let expected = oracle_scales(&oracle, strategy, &set);
                let actual = set.metadata(strategy, &Default::default()).unwrap();
                for (index, entry) in actual.iter().enumerate().skip(usize::from(mode <= 2)) {
                    for channel in 0..3 {
                        assert_eq!(
                            entry[channel],
                            (expected[channel][index] as f32).to_bits(),
                            "mode {mode}, bands {count}, {strategy:?}, channel {channel}, index {index}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_matrices_are_rejected_before_gpu_construction() {
    use jxl_gpu_protocol::VarDctMatrixError;
    let invalid = |encoding| {
        assert!(matches!(
            VarDctDequantMatrices::default().with_matrix(VarDctStrategy::Dct8, encoding),
            Err(EncodeError::VarDctMatrix(VarDctMatrixError::Value {
                matrix: 0,
                ..
            }))
        ))
    };
    for count in [0, 17] {
        invalid(VarDctMatrixEncoding::Dct(bands(count)));
    }
    let mut unequal = bands(3);
    unequal[1].pop();
    invalid(VarDctMatrixEncoding::Dct(unequal));
    invalid(VarDctMatrixEncoding::Dct4 {
        params: [[f16(0x3c00); 2]; 3],
        dct_params: bands(0),
    });
    invalid(VarDctMatrixEncoding::Dct4x8 {
        params: [[f16(0x3c00); 1]; 3],
        dct_params: bands(17),
    });
    invalid(VarDctMatrixEncoding::Afv {
        params: [[f16(0x3c00); 9]; 3],
        dct_params: bands(1),
        dct4x4_params: bands(0),
    });
    for value in [0x0000, 0x8000, 0xbc00] {
        invalid(VarDctMatrixEncoding::Hornuss([[f16(value); 3]; 3]));
        invalid(VarDctMatrixEncoding::Dct(std::array::from_fn(|_| {
            vec![f16(value)]
        })));
    }
    invalid(VarDctMatrixEncoding::Dct4 {
        params: [[f16(0); 2]; 3],
        dct_params: bands(1),
    });
    invalid(VarDctMatrixEncoding::Dct(std::array::from_fn(|_| {
        vec![f16(0x7bff); 16]
    })));
    for mode in 1..=5 {
        assert!(matches!(
            VarDctDequantMatrices::default()
                .with_matrix(VarDctStrategy::Dct16x16, encoding(mode, 3)),
            Err(EncodeError::VarDctMatrix(VarDctMatrixError::Encoding {
                matrix: 4,
                ..
            }))
        ));
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            jxl_gpu_protocol::VarDctMatrixEncoding::Hornuss([[value; 3]; 3])
                .expand(VarDctStrategy::Dct8),
            Err(VarDctMatrixError::Value { matrix: 0, .. })
        ));
    }
}

#[test]
fn matrix_families_share_parameters_and_default_reset_restores_wire_shortcut() {
    let mut set = VarDctDequantMatrices::default()
        .with_matrix(VarDctStrategy::Dct16x8, encoding(6, 3))
        .unwrap();
    assert_eq!(
        set.encoding(VarDctStrategy::Dct16x8),
        set.encoding(VarDctStrategy::Dct8x16)
    );
    assert!(set.encoding(VarDctStrategy::Dct8).is_none());
    set = set
        .with_matrix(VarDctStrategy::Dct8x16, VarDctMatrixEncoding::Default)
        .unwrap();
    assert_eq!(set, VarDctDequantMatrices::default());
    let mut output = BitWriter::new();
    set.write(&mut output).unwrap();
    assert_eq!(output.bit_len(), 1);
    assert_eq!(output.into_bytes(), [1]);
}

#[test]
fn custom_tiled_matrices_interoperate_across_groups_windows_and_variants() {
    let (device, queue, info) = test_device().expect("actual GPU required for custom matrices");
    let context = WgpuContext::new(device.clone(), queue.clone()).unwrap();
    let config = VarDctConfig {
        dequant_matrices: selected(),
        coefficient_orders: orders::selected([VarDctStrategy::Dct8]),
        lf_metadata: custom_lf_metadata(),
        ..Default::default()
    };
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
    let mut cases = Vec::new();
    for (width, height) in [(1, 1), (13, 21), (257, 17), (2057, 17)] {
        let pixels = reference::pattern(width, height);
        let source = padded_rgb_source_sized(&context, width, height, &pixels);
        let actual = encoder.encode(source).unwrap();
        quantization::assert_decoders_agree(&actual, width, height);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        cases.push((width, height, pixels, actual));
    }
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Lanes64,
        KernelVariant::Lanes128,
        KernelVariant::Lanes256,
    ] {
        let context =
            test_context_with_variants(&device, &queue, &info, &[(TILED_KERNEL_KEY, variant)])
                .unwrap();
        let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
        for (width, height, pixels, expected) in &cases {
            let source = padded_rgb_source_sized(&context, *width, *height, pixels);
            assert_eq!(
                &encoder.encode(source).unwrap(),
                expected,
                "{variant:?}/{width}x{height}"
            );
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn extreme_valid_matrices_return_typed_gpu_overflow_and_release_memory() {
    let context = test_context().expect("actual GPU required for custom matrix overflow");
    let config = VarDctConfig {
        dequant_matrices: VarDctDequantMatrices::default()
            .with_matrix(
                VarDctStrategy::Dct8,
                VarDctMatrixEncoding::Dct(std::array::from_fn(|_| vec![f16(0x7bff); 3])),
            )
            .unwrap(),
        ..Default::default()
    };
    let source = padded_rgb_source_sized(&context, 8, 8, &reference::pattern(8, 8));
    let single =
        VarDctEncoder::new_with_config(context.clone(), VarDctStrategy::Dct8, config.clone())
            .unwrap();
    let tiled = TiledVarDctEncoder::new_with_config(context, config).unwrap();
    for result in [single.encode(source.clone()), tiled.encode(source)] {
        assert!(
            matches!(
                result,
                Err(EncodeError::Backend(
                    crate::BackendError::VarDctQuantizationOverflow {
                        low_frequency: false,
                        high_frequency: true,
                    }
                ))
            ),
            "{result:?}"
        );
    }
    assert_eq!(single.in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(tiled.in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn equivalent_fixed_matrix_modes_preserve_pixels_across_native_and_gpu_decoders() {
    let context = test_context().expect("actual GPU required for special matrix modes");
    let pixels = reference::pattern(8, 8);
    let source = padded_rgb_source_sized(&context, 8, 8, &pixels);
    for strategy in [
        VarDctStrategy::Dct8,
        VarDctStrategy::Hornuss,
        VarDctStrategy::Dct2x2,
    ] {
        let mut original = None;
        for mode in [6, 1, 2] {
            let config = VarDctConfig {
                dequant_matrices: VarDctDequantMatrices::default()
                    .with_matrix(
                        strategy,
                        match mode {
                            1 => VarDctMatrixEncoding::Hornuss([[f16(0x4000); 3]; 3]),
                            2 => VarDctMatrixEncoding::Dct2([[f16(0x4000); 6]; 3]),
                            _ => VarDctMatrixEncoding::Dct(std::array::from_fn(|_| {
                                vec![f16(0x4000)]
                            })),
                        },
                    )
                    .unwrap(),
                ..Default::default()
            };
            let encoder =
                VarDctEncoder::new_with_config(context.clone(), strategy, config).unwrap();
            let bytes = encoder.encode(source.clone()).unwrap();
            let rust = decode_rgb8_sized(&bytes, 8, 8);
            if let Some(original) = &original {
                assert_eq!(&rust, original, "equivalent matrices");
            } else {
                original = Some(rust);
            }
            eprintln!("special matrix case {strategy:?}/{mode}");
            quantization::assert_decoders_agree(&bytes, 8, 8);
        }
    }
}
