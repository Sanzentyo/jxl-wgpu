use super::*;
use crate::CmykSampleEncoding;

#[test]
fn cmyk_icc_working_coefficients_match_frozen_pcs_and_every_native_transform_basis() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoder = GpuDecoder::wgpu(gpu.clone()).unwrap();
    let profiles = IccProfileOracle::compile();
    let bases = native::native_oracles();
    let matrix =
        jxl_test_support::oracles::color::inverse_pcs_matrix(jxl_gpu_formats::ColorSpace::Bt709);
    let names = [
        "lut8_xyz_4",
        "lut8_lab_4",
        "lut16_xyz_4",
        "lut16_lab_4",
        "ab_xyz_4",
        "ab_lab_4",
        "ab_clut_xyz_4",
        "ab_clut_lab_4",
    ];
    let cases = names
        .into_iter()
        .map(|name| (name, VarDctStrategy::Dct8))
        .chain(
            VarDctStrategy::ALL
                .into_iter()
                .enumerate()
                .map(|(i, strategy)| (names[i % names.len()], strategy)),
        );
    for (name, strategy) in cases {
        let path = directory().join("lut");
        let profile = IccProfile::parse(
            fs::read(path.join(format!("{name}.icc"))).unwrap().into(),
            Default::default(),
        )
        .unwrap();
        let samples =
            extra_channels::floats(&fs::read(path.join(format!("{name}_forward.f32le"))).unwrap());
        let pcs = centers(path.join(format!("{name}_forward_1.reference")));
        assert_eq!(samples.len() / 4, pcs.len() / 3);
        // Select frozen corpus inputs whose floating complement is exactly reversible.
        // No rounded replacement input or enlarged color bound is attributed to the reference.
        let indices: Vec<_> = samples
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
            .filter_map(|(i, pixel)| {
                pixel
                    .iter()
                    .all(|&v| (1.0 - (1.0 - v)).to_bits() == v.to_bits())
                    .then_some(i)
            })
            .collect();
        assert_eq!(indices.len(), 36);
        let extent = strategy.pixel_extent();
        let selected: Vec<_> = (0..extent.area().unwrap())
            .map(|i| indices[i % indices.len()])
            .collect();
        let words: Vec<_> = selected
            .iter()
            .flat_map(|&i| {
                samples[i * 4..i * 4 + 4]
                    .iter()
                    .map(|&v| (1.0 - v).to_bits())
            })
            .collect();
        for transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
            let config = config(&profile, transform);
            let components: Vec<_> = selected
                .iter()
                .enumerate()
                .map(|(p, &i)| {
                    if transform == VarDctColorTransform::Original {
                        return std::array::from_fn(|c| {
                            f64::from(f32::from_bits(words[p * 4 + c]))
                        });
                    }
                    xyb(
                        matrix.map(|row| (0..3).map(|c| row[c] * pcs[i * 3 + c]).sum()),
                        255.0,
                    )
                })
                .collect();
            let base = &bases[strategy as usize];
            let expected = native::forward_samples(
                &components,
                extent.width as usize,
                extent.height as usize,
                base,
            );
            let mut format = config.pixel_format();
            format.byte_order = ByteOrder::Big;
            let (layout, bytes) = Packing {
                storage: Storage::Planar,
                reversed: true,
                shifted: true,
            }
            .pack(format, extent, &words, 1031);
            let source =
                BufferImageSource::new(
                    Arc::new(context.device().create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("CMYK frozen ICC coefficient input"),
                            contents: &bytes,
                            usage: wgpu::BufferUsages::STORAGE,
                        },
                    )),
                    layout,
                )
                .unwrap()
                .with_cmyk_encoding(CmykSampleEncoding::Complemented)
                .unwrap();
            let backend =
                VarDctBackend::new_with_config(&context, strategy, config.clone()).unwrap();
            let (ac, bits, artifacts) = backend
                .submit(
                    &context,
                    GpuFrameSource::Buffer(source),
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
            assert_eq!(profiles.read(&bytes).profile, profile.bytes().as_ref());
            let black = jxl_test_support::oracles::modular_integer::vardct_extra_words(&bytes, 0);
            assert_eq!(black.len(), 1);
            assert_eq!(
                black[0].words,
                words.iter().skip(3).step_by(4).copied().collect::<Vec<_>>()
            );
            let native = extra_channels::libjxl_output(
                &bytes,
                &[
                    if transform == VarDctColorTransform::Original {
                        "--original-icc"
                    } else {
                        "--linear"
                    },
                    "--no-cms",
                ],
            )
            .unwrap();
            let pixels = extent.area().unwrap();
            assert_eq!(native.len(), pixels * 5);
            for channel in 0..3 {
                let request = if transform == VarDctColorTransform::Original {
                    GpuOutputRequest::numeric(
                        crate::SamplePrecision::float(32, 8).unwrap().pixel_format(),
                        jxl_wgpu_decode::NumericSampleMapping::NativeFloat,
                    )
                    .unwrap()
                    .with_color_channel(channel)
                    .unwrap()
                } else {
                    let mut color = vardct_rgb8_format().color_spec;
                    let ColorSpecification::Defined(ref mut spec) = color else {
                        unreachable!()
                    };
                    spec.transfer = jxl_gpu_formats::TransferFunction::Linear;
                    GpuOutputRequest::color(PixelFormat::rgb_f32(
                        RgbChannelOrder::Rgb,
                        false,
                        color,
                    ))
                    .unwrap()
                };
                let mut session = decoder.open(&bytes, request).unwrap();
                let frame = session.next_frame().unwrap().unwrap();
                let actual = extra_channels::floats(&jxl_test_support::gpu::planes::read_bytes(
                    &gpu,
                    &frame.output().outputs[0],
                ));
                let reference: Vec<_> = native[..pixels * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| {
                        if transform == VarDctColorTransform::Original {
                            vec![p[channel as usize]]
                        } else {
                            p[..3].to_vec()
                        }
                    })
                    .collect();
                compare(&actual, &reference);
                if transform == VarDctColorTransform::Xyb {
                    break;
                }
            }
        }
    }
}
