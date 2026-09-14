#![cfg(not(target_arch = "wasm32"))]

mod dispatch;

use jxl_gpu_formats::ColorSpace;
use jxl_gpu_protocol::GamutMapping;
use jxl_test_support::oracles::{gamut_mapping as oracle, hdr::luminance};
use jxl_wgpu::{GamutMappingParams, WgpuBackend};

const SPACES: [ColorSpace; 3] = [ColorSpace::Bt709, ColorSpace::Bt2020, ColorSpace::DisplayP3];

#[test]
fn gamut_mapping_matches_native_libjxl_and_independent_cube_intersections() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    eprintln!("Gamut adapter: {:?}", backend.adapter_info());
    let records: Vec<[f32; 8]> = include_str!("../../test-data/gamut_mapping/native.txt")
        .lines()
        .map(|line| {
            line.split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect();
    assert_eq!(records.len(), 5000);
    let mut components = 0;
    for workgroup in [1, 32, 256] {
        let pipeline = dispatch::pipeline(backend.device(), workgroup);
        for rows in records.chunk_by(|a, b| a[..2] == b[..2]) {
            let space = SPACES[rows[0][0] as usize];
            let preference = rows[0][1];
            let input: Vec<_> = rows.iter().map(|r| [r[2], r[3], r[4], 0.375]).collect();
            let params = GamutMappingParams::new(
                space.rgb_space().unwrap(),
                GamutMapping::new(preference).unwrap(),
            )
            .unwrap();
            let actual = dispatch::run(&backend, &pipeline, workgroup, &input, params);
            for (index, (row, actual)) in rows.iter().zip(actual).enumerate() {
                let rgb = [row[2], row[3], row[4]];
                let expected =
                    oracle::apply(rgb.map(f64::from), luminance(space), f64::from(preference));
                for c in 0..3 {
                    // Predeclared allowance for F32 luminance, intersections and normalization.
                    let gpu = f64::from(actual[c]);
                    let native = f64::from(row[c + 5]);
                    assert!(
                        (native - expected[c]).abs() <= 4e-6,
                        "native {space:?}/{preference}/{index}/{c}: {native}, {}",
                        expected[c]
                    );
                    assert!(
                        gpu.is_finite()
                            && (0.0..=1.0).contains(&gpu)
                            && (gpu - expected[c]).abs() <= 4e-6,
                        "GPU {workgroup}/{space:?}/{preference}/{index}/{c}: {gpu}, {}",
                        expected[c]
                    );
                    if rgb.iter().all(|v| (0.0..=1.0).contains(v)) {
                        assert_eq!(actual[c].to_bits(), rgb[c].to_bits());
                    }
                    components += 1;
                }
                assert_eq!(actual[3].to_bits(), 0.375_f32.to_bits());
            }
        }
    }
    assert_eq!(components, 45_000);
    eprintln!("Gamut native comparison: {components} GPU components");
}

#[test]
fn negative_light_and_extended_finite_colors_have_bounded_outputs() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let mut inputs: Vec<_> = [-1e30_f32, -2.0, -0.1, 0.0, 0.5, 1.0, 2.0, 1e30]
        .into_iter()
        .map(|v| [v, v, v, 0.625])
        .collect();
    inputs.extend([
        [-1e30, 1e30, 1e30, 0.625],
        [1e30, 1e30, -1e30, 0.625],
        [1e30, -1e30, 1e30, 0.625],
        [-0.5, 0.01, 0.01, 0.625],
        [1.0_f32.next_down(), 0.5, 1.0_f32.next_up(), 0.625],
    ]);
    let pipeline = dispatch::pipeline(backend.device(), 32);
    for space in SPACES {
        for preference in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let params = GamutMappingParams::new(
                space.rgb_space().unwrap(),
                GamutMapping::new(preference).unwrap(),
            )
            .unwrap();
            let actual = dispatch::run(&backend, &pipeline, 32, &inputs, params);
            for (input, actual) in inputs.iter().zip(actual) {
                let expected = oracle::apply(
                    [input[0], input[1], input[2]].map(f64::from),
                    luminance(space),
                    f64::from(preference),
                );
                for c in 0..3 {
                    assert!(
                        actual[c].is_finite()
                            && (0.0..=1.0).contains(&actual[c])
                            && (f64::from(actual[c]) - expected[c]).abs() <= 4e-6,
                        "{space:?}/{preference}/{input:?}: {actual:?}, {expected:?}"
                    );
                }
                assert_eq!(actual[3].to_bits(), input[3].to_bits());
            }
        }
    }
}
