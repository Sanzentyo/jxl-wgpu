use super::*;
use crate::{GpuEncodeBackend, GpuFrameSource, VarDctBackend};

struct Profile {
    name: &'static str,
    channels: usize,
}

fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../jxl_wgpu/test-data/icc")
}

fn centers(path: PathBuf) -> Vec<f64> {
    let bytes = fs::read(path).unwrap();
    let (records, tail) = bytes.as_chunks::<28>();
    assert!(tail.is_empty());
    records
        .iter()
        .map(|r| {
            let value = f32::from_le_bytes(r[4..8].try_into().unwrap());
            let lower = f32::from_le_bytes(r[8..12].try_into().unwrap());
            let upper = f32::from_le_bytes(r[12..16].try_into().unwrap());
            assert!(lower <= value && value <= upper);
            f64::from(value)
        })
        .collect()
}

fn xyb(linear: [f64; 3], intensity: f64) -> [f64; 3] {
    let bias = 0.0037930732552754493_f64;
    let lms = [
        [0.3, 0.622, 0.078],
        [0.23, 0.692, 0.078],
        [0.2434226892, 0.2047674442, 0.5518098665],
    ]
    .map(|row| {
        (bias
            + (0..3)
                .map(|c| row[c] * linear[c] * intensity / 255.0)
                .sum::<f64>())
        .max(0.0)
        .cbrt()
            - bias.cbrt()
    });
    [(lms[0] - lms[1]) * 0.5, (lms[0] + lms[1]) * 0.5, lms[2]]
}

#[test]
fn icc_working_coefficients_match_independent_scalar_color_and_native_bases() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let pixels = Pixels::new(&gpu);
    let source_pixels = 37 * 17;
    let profiles = [
        "srgb",
        "gamma_v2",
        "gamma_v4",
        "threshold",
        "offset",
        "piecewise",
        "affine",
        "sampled",
        "gray",
        "wide",
    ]
    .map(|name| Profile {
        name,
        channels: if name == "gray" { 1 } else { 3 },
    });
    let bases = native::native_oracles();
    // All ten matrix/TRC families have DCT8 evidence; the remaining passes cover every
    // standard transform with both original and ICC-normalized XYB working components.
    let cases = profiles.iter().map(|p| (p, VarDctStrategy::Dct8)).chain(
        VarDctStrategy::ALL
            .into_iter()
            .enumerate()
            .map(|(i, strategy)| (&profiles[i % profiles.len()], strategy)),
    );
    for (profile_case, strategy) in cases {
        let profile = IccProfile::parse(
            fs::read(directory().join(format!("{}.icc", profile_case.name)))
                .unwrap()
                .into(),
            Default::default(),
        )
        .unwrap();
        let samples = extra_channels::floats(
            &fs::read(directory().join(format!("{}_input.f32le", profile_case.name))).unwrap(),
        );
        assert_eq!(samples.len(), source_pixels * profile_case.channels);
        // Each committed 28-byte record contains independent exact/lower/upper values,
        // plus separately classified native-CMM values; use its independent center here.
        let linear =
            centers(directory().join(format!("linear/{}_to_bt709.reference", profile_case.name)));
        assert_eq!(linear.len(), source_pixels * 3);
        let extent = strategy.pixel_extent();
        let source_indices: Vec<_> = (0..extent.area().unwrap())
            .map(|i| i % source_pixels)
            .collect();
        let words: Vec<[u32; 3]> = source_indices
            .iter()
            .map(|&i| {
                std::array::from_fn(|c| {
                    samples
                        [i * profile_case.channels + if profile_case.channels == 1 { 0 } else { c }]
                    .to_bits()
                })
            })
            .collect();
        let base = &bases[strategy as usize];
        for transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
            let config = config(&profile, transform);
            let components: Vec<_> = source_indices
                .iter()
                .zip(&words)
                .map(|(&index, words)| {
                    if transform == VarDctColorTransform::Original {
                        return words.map(|v| f64::from(f32::from_bits(v)));
                    }
                    let rgb: [f64; 3] = linear[index * 3..index * 3 + 3].try_into().unwrap();
                    let working = if profile_case.channels == 1 {
                        let row = jxl_test_support::oracles::color::pcs_matrix(
                            jxl_gpu_formats::ColorSpace::Bt709,
                        )[1];
                        [(0..3).map(|c| row[c] * rgb[c]).sum(); 3]
                    } else {
                        rgb
                    };
                    xyb(working, 255.0)
                })
                .collect();
            let expected = native::forward_samples(
                &components,
                extent.width as usize,
                extent.height as usize,
                base,
            );
            let backend =
                VarDctBackend::new_with_config(&context, strategy, config.clone()).unwrap();
            let (ac, bits, artifacts) = backend
                .submit(
                    &context,
                    GpuFrameSource::Buffer(upload(&context, extent, &config, &words, true)),
                    &layouts::request(extent, &config),
                )
                .unwrap()
                .wait_with_ac_for_test()
                .unwrap();
            native::check_ac(&ac, bits, &expected, base, config.clone());
            let mut bytes = image_header_with_color(
                extent.width,
                extent.height,
                crate::AnimationHeader::Still,
                &backend.color_plan,
            )
            .unwrap()
            .into_bytes();
            bytes.extend_from_slice(assemble_frame(artifacts.packets).unwrap().bytes());
            pixels.check(&bytes, &config);
        }
    }
}

#[test]
fn icc_legacy_lut_and_mpe_sources_match_independent_pcs_and_native_coefficients() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let pixels = Pixels::new(&gpu);
    let profiles = IccProfileOracle::compile();
    let bases = native::native_oracles();
    let matrix =
        jxl_test_support::oracles::color::inverse_pcs_matrix(jxl_gpu_formats::ColorSpace::Bt709);
    let mut counts = [0; 2];
    for (family, corpus) in ["lut", "mpe"].into_iter().enumerate() {
        let mut entries: Vec<_> = fs::read_dir(directory().join(corpus))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| {
                p.extension().is_some_and(|e| e == "icc") && p.file_stem().unwrap() != "identity"
            })
            .collect();
        entries.sort();
        for path in entries {
            let profile =
                IccProfile::parse(fs::read(&path).unwrap().into(), Default::default()).unwrap();
            let channels = match &profile.header().device_space.0 {
                b"GRAY" => 1,
                b"RGB " => 3,
                // Wider device spaces are outside the checked Gray/RGB encoder contract.
                _ => continue,
            };
            let name = path.file_stem().unwrap().to_str().unwrap();
            let input = extra_channels::floats(
                &fs::read(directory().join(corpus).join(if family == 0 {
                    format!("{name}_forward.f32le")
                } else {
                    "input.f32le".to_owned()
                }))
                .unwrap(),
            );
            let pcs = centers(
                directory()
                    .join(corpus)
                    .join(format!("{name}_forward_1.reference")),
            );
            assert_eq!(pcs.len() / 3, input.len() / channels);
            let strategy = VarDctStrategy::Dct8;
            let extent = strategy.pixel_extent();
            // A full corpus image is already tested in the shared resident CMM. This checks
            // the encoder boundary and every selected program family with the same frozen oracle.
            let words: Vec<_> = (0..64)
                .map(|i| {
                    std::array::from_fn(|c| {
                        input[i * channels + if channels == 1 { 0 } else { c }].to_bits()
                    })
                })
                .collect();
            for transform in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
                let config = config(&profile, transform);
                let components: Vec<_> = words
                    .iter()
                    .enumerate()
                    .map(|(i, words)| {
                        if transform == VarDctColorTransform::Original {
                            return words.map(|w| f64::from(f32::from_bits(w)));
                        }
                        let linear = if channels == 1 {
                            [pcs[i * 3 + 1]; 3]
                        } else {
                            matrix.map(|row| (0..3).map(|c| row[c] * pcs[i * 3 + c]).sum::<f64>())
                        };
                        xyb(linear, 255.0)
                    })
                    .collect();
                let expected = native::forward_samples(&components, 8, 8, &bases[0]);
                let backend =
                    VarDctBackend::new_with_config(&context, strategy, config.clone()).unwrap();
                let (ac, bits, artifacts) = backend
                    .submit(
                        &context,
                        GpuFrameSource::Buffer(upload(&context, extent, &config, &words, true)),
                        &layouts::request(extent, &config),
                    )
                    .unwrap()
                    .wait_with_ac_for_test()
                    .unwrap();
                eprintln!("ICC {corpus}/{name} {transform:?}");
                native::check_ac(&ac, bits, &expected, &bases[0], config.clone());
                let mut bytes = image_header_with_color(
                    8,
                    8,
                    crate::AnimationHeader::Still,
                    &backend.color_plan,
                )
                .unwrap()
                .into_bytes();
                bytes.extend_from_slice(assemble_frame(artifacts.packets).unwrap().bytes());
                assert_eq!(profiles.read(&bytes).profile, profile.bytes().as_ref());
                pixels.check(&bytes, &config);
            }
            counts[family] += 1;
        }
    }
    assert_eq!(counts, [30, 9]);
}

#[test]
fn icc_normalization_preserves_all_integer_and_float_precisions_before_color_conversion() {
    use jxl_gpu_protocol::icc::{IccDirection, IccRenderingIntent};
    let context = test_context().unwrap();
    let bases = native::native_oracles();
    let matrix =
        jxl_test_support::oracles::color::inverse_pcs_matrix(jxl_gpu_formats::ColorSpace::Bt709);
    let extent = Extent2d::new(8, 8);
    for gray in [false, true] {
        let channels = if gray {
            ColorChannels::Gray
        } else {
            ColorChannels::Rgb
        };
        let profile = jxl_test_support::fixtures::icc::with_matrix_mpe(
            &profile(gray),
            IccDirection::DeviceToPcs,
            IccRenderingIntent::Relative,
        );
        let formats =
            (1..=31)
                .map(|bits| ColorSampleFormat::integer(channels, bits).unwrap())
                .chain(floating::all_precisions().into_iter().map(|p| {
                    ColorSampleFormat::float(channels, p.bits(), p.exponent_bits()).unwrap()
                }));
        let mut count = 0;
        for format in formats {
            let mut config = config(&profile, VarDctColorTransform::Xyb);
            config.sample_format = format;
            let mut input = if let Some(p) = format.float_precision() {
                floating::pixels(8, 8, p)
            } else {
                precision::pixels(8, 8, format.bits_per_sample(), 39)
            };
            if gray {
                input.iter_mut().for_each(|v| *v = [v[0]; 3]);
            }
            let components: Vec<_> = input
                .iter()
                .map(|word| {
                    let values = word.map(|word| {
                        if let Some(p) = format.float_precision() {
                            floating::value(word, p)
                        } else {
                            f64::from(word) / ((1_u64 << format.bits_per_sample()) - 1) as f64
                        }
                    });
                    // The synthetic MPE has an independently specified identity RGB -> XYZ
                    // matrix, or one Gray column equal to these exact binary32 D50 constants.
                    let pcs = if gray {
                        [0.9642_f32, 1.0, 0.8249].map(|v| f64::from(v) * values[0])
                    } else {
                        values
                    };
                    xyb(
                        if gray {
                            [pcs[1]; 3]
                        } else {
                            matrix.map(|row| (0..3).map(|c| row[c] * pcs[c]).sum())
                        },
                        255.0,
                    )
                })
                .collect();
            let expected = native::forward_samples(&components, 8, 8, &bases[0]);
            let backend =
                VarDctBackend::new_with_config(&context, VarDctStrategy::Dct8, config.clone())
                    .unwrap();
            let (ac, bits, _) = backend
                .submit(
                    &context,
                    GpuFrameSource::Buffer(upload(&context, extent, &config, &input, true)),
                    &layouts::request(extent, &config),
                )
                .unwrap()
                .wait_with_ac_for_test()
                .unwrap();
            native::check_ac(&ac, bits, &expected, &bases[0], config);
            assert_eq!(context.memory_stats().reserved_bytes, 0);
            count += 1;
        }
        assert_eq!(count, 185);
    }
}
