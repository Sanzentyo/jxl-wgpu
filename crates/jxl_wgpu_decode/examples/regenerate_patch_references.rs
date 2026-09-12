//! Freeze native component-reference results while reusing checked-in entropy sources.
use jxl_test_support::{fixtures::patch_references, offline, oracles::extra_channels as native};

fn main() {
    let output = jxl_test_support::decoder_directory().join("test-data/patches/references");
    std::fs::create_dir_all(&output).unwrap();
    for case in patch_references::cases() {
        eprintln!("{}: {:?}", case.name, case.sources);
        let encoded = case.encode();
        let native = native::libjxl_output(
            &encoded,
            &["--linear", "--preserve-alpha", "--keep-orientation"],
        )
        .expect("native libjxl must accept every component reference fixture");
        let samples = match case.reference_source() {
            patch_references::ReferenceSource::NativeLinear => native,
            patch_references::ReferenceSource::NativeSrgb => {
                let inventory = jxl_gpu_bitstream::parse(&encoded, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap();
                assert!(inventory.image_header.extra_channels.is_empty());
                let srgb =
                    native::libjxl_output(&encoded, &["--preserve-alpha", "--keep-orientation"])
                        .unwrap();
                assert_eq!(srgb.len(), native.len());
                srgb.into_iter()
                    .enumerate()
                    .map(|(i, value)| {
                        if i % 4 == 3 {
                            return value;
                        }
                        let value = f64::from(value);
                        (value.signum()
                            * if value.abs() <= 0.04045 {
                                value.abs() / 12.92
                            } else {
                                ((value.abs() + 0.055) / 1.055).powf(2.4)
                            }) as f32
                    })
                    .collect()
            }
            patch_references::ReferenceSource::JxlOxideLinear => {
                let arithmetic = case.encode_arithmetic_reference();
                let equivalent = native::libjxl_output(
                    &arithmetic,
                    &["--linear", "--preserve-alpha", "--keep-orientation"],
                )
                .unwrap();
                assert!(
                    native
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(equivalent.iter().map(|v| v.to_bits())),
                    "implicit-alpha arithmetic changed pixels"
                );
                let mut image =
                    jxl_oxide::JxlImage::read_with_defaults(arithmetic.as_slice()).unwrap();
                image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                    jxl_oxide::RenderingIntent::Relative,
                ));
                let render = image.render_frame(0).unwrap();
                let pixels = render.image_all_channels();
                let samples: Vec<_> = pixels
                    .buf()
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2], 1.0])
                    .collect();
                assert_eq!(samples.len(), native.len());
                let difference = samples
                    .iter()
                    .zip(native)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "{}: pinned jxl-oxide reference; native fast-renderer maxAE {difference}",
                    case.name
                );
                samples
            }
        };
        assert!(samples.iter().all(|sample| sample.is_finite()));
        std::fs::write(
            output.join(format!("{}.jxl.hex", case.name)),
            offline::hex(&encoded),
        )
        .unwrap();
        std::fs::write(
            output.join(format!("{}.f32.hex", case.name)),
            offline::float_hex(
                &samples
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
        )
        .unwrap();
    }
}
