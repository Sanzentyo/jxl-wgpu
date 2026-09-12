//! Native references for explicit spline, patch, upsampling, noise and LF combinations.
use jxl_test_support::{fixtures::splines, offline, oracles::extra_channels};

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = root.join("splines/features");
    std::fs::create_dir_all(&output).unwrap();
    for case in splines::cases() {
        let source = offline::unhex(
            &std::fs::read_to_string(root.join(format!("{}.jxl.hex", case.source))).unwrap(),
        );
        let encoded = case.assemble(&source);
        let info = jxl_gpu_bitstream::parse(&encoded, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let mut options = vec!["--preserve-alpha", "--keep-orientation"];
        if info.image_header.xyb_encoded {
            options.push("--linear");
        }
        let mut reference = extra_channels::libjxl_output(&encoded, &options)
            .expect("libjxl must independently decode every spline fixture");
        assert_eq!(
            reference.len(),
            info.image_header.width as usize
                * info.image_header.height as usize
                * (4 + info.image_header.extra_channels.len())
        );
        assert!(reference.iter().all(|value| value.is_finite()));
        if !info.image_header.xyb_encoded {
            // Native CMS approximates extended sRGB. Preserve its unconverted original RGB
            // and apply the declared transfer in f64, including negative and >1 components.
            let pixels = info.image_header.width as usize * info.image_header.height as usize;
            for (index, value) in reference[..pixels * 4].iter_mut().enumerate() {
                if index % 4 != 3 {
                    let code = f64::from(*value);
                    *value = (code.signum()
                        * if code.abs() <= 0.04045 {
                            code.abs() / 12.92
                        } else {
                            ((code.abs() + 0.055) / 1.055).powf(2.4)
                        }) as f32;
                }
            }
        }
        if case.reference == splines::ReferenceSource::JxlOxide {
            assert!(info.image_header.extra_channels.is_empty());
            let mut image = jxl_oxide::JxlImage::read_with_defaults(encoded.as_slice()).unwrap();
            image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));
            let render = image.render_frame(0).unwrap();
            let samples: Vec<_> = render
                .image_all_channels()
                .buf()
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 1.0])
                .collect();
            assert_eq!(samples.len(), reference.len());
            let difference = samples
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "{}: independent jxl-oxide reference; native fast-renderer maxAE {difference}",
                case.name
            );
            reference = samples;
        }
        std::fs::write(
            output.join(format!("{}.jxl.hex", case.name)),
            offline::hex(&encoded),
        )
        .unwrap();
        std::fs::write(
            output.join(format!("{}.f32.hex", case.name)),
            offline::float_hex(
                &reference
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
        )
        .unwrap();
        eprintln!(
            "{}: {} frames, {} bytes",
            case.name,
            info.frames.len(),
            encoded.len()
        );
    }
    let output = root.join("splines/progressive");
    std::fs::create_dir_all(&output).unwrap();
    for &(name, source) in splines::PROGRESSIVE_SOURCES {
        let source = offline::unhex(
            &std::fs::read_to_string(root.join(format!("{source}.jxl.hex"))).unwrap(),
        );
        let encoded = splines::progressive(&source);
        let info = jxl_gpu_bitstream::parse(&encoded, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let frame = info.frames.last().unwrap();
        let mut snapshots = Vec::new();
        for completed in 0..=frame.num_passes {
            let end = frame
                .sections
                .iter()
                .filter_map(|section| match section.kind {
                    jxl_gpu_bitstream::FrameSectionKind::PassGroup { pass_index, .. }
                        if pass_index >= completed =>
                    {
                        Some(section.bytes.offset as usize)
                    }
                    _ => None,
                })
                .min()
                .unwrap_or(encoded.len());
            let updates = jxl_test_support::oracles::progressive::native_updates_all_channels(
                &encoded[..end],
                info.image_header.xyb_encoded,
                true,
                completed < frame.num_passes,
            )
            .expect("libjxl must independently decode each spline-bearing pass prefix");
            let last = updates.last().unwrap();
            assert_eq!(last.complete, completed == frame.num_passes);
            snapshots.extend_from_slice(&last.pixels);
        }
        std::fs::write(
            output.join(format!("{name}.jxl.hex")),
            offline::hex(&encoded),
        )
        .unwrap();
        std::fs::write(
            output.join(format!("{name}.f32.hex")),
            offline::float_hex(&snapshots),
        )
        .unwrap();
        eprintln!("progressive {name}: {} passes", frame.num_passes);
    }
}
