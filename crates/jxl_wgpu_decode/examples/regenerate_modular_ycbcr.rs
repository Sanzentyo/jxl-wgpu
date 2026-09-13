//! Generate independently encoded Modular YCbCr streams and native unclipped F32 references.
use std::path::PathBuf;
use std::process::Command;

use jxl_test_support::{fixtures::modular_ycbcr, offline, oracles::extra_channels};

fn main() {
    let generator = std::env::args_os()
        .nth(1)
        .expect("native generator executable path");
    let output = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data/modular_ycbcr");
    let temporary =
        std::env::temp_dir().join(format!("jxl-wgpu-modular-ycbcr-{}", std::process::id()));
    std::fs::create_dir(&temporary).unwrap();
    std::fs::create_dir_all(&output).unwrap();
    offline::run(Command::new(generator).arg(&temporary));
    let cases = modular_ycbcr::cases();
    assert_eq!(
        std::fs::read_dir(&temporary).unwrap().count(),
        cases.len()
            + cases
                .iter()
                .filter(|case| case.gaborish || case.epf_iterations != 0)
                .count()
            + cases
                .iter()
                .filter(|case| !case.transforms.is_empty())
                .count()
    );
    for case in cases {
        let bytes = std::fs::read(temporary.join(format!("{}.jxl", case.name))).unwrap();
        let info = jxl_gpu_bitstream::parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        case.validate(&info);
        if !case.transforms.is_empty() {
            let name = format!("{}.topology", case.name);
            let native = std::fs::read_to_string(temporary.join(&name)).unwrap();
            let topology = modular_ycbcr::NativeTopology::parse(&native);
            assert_eq!(topology.transform_count, case.transforms.len());
            std::fs::write(output.join(name), native).unwrap();
        }
        if case.passes > 1 {
            let mut snapshots = Vec::new();
            for completed in 0..=case.passes {
                let end =
                    info.frames[0]
                        .sections
                        .iter()
                        .filter_map(|section| match section.kind {
                            jxl_gpu_bitstream::FrameSectionKind::PassGroup {
                                pass_index, ..
                            } if pass_index >= completed => Some(section.bytes.offset as usize),
                            _ => None,
                        })
                        .min()
                        .unwrap_or(bytes.len());
                let updates = jxl_test_support::oracles::progressive::native_updates_all_channels(
                    &bytes[..end],
                    false,
                    true,
                    completed < case.passes,
                )
                .expect("native progressive reference");
                let last = updates.last().unwrap();
                assert_eq!(last.complete, completed == case.passes);
                snapshots.extend_from_slice(&last.pixels);
            }
            std::fs::write(
                output.join(format!("{}.progressive.f32.hex", case.name)),
                offline::float_hex(&snapshots),
            )
            .unwrap();
        }
        let fast =
            extra_channels::libjxl_output(&bytes, &["--preserve-alpha", "--keep-orientation"])
                .expect("native libjxl decoder is required");
        let reference = if case.gaborish || case.epf_iterations != 0 {
            // A scalar expansion produces equivalent 4:4:4 input, isolating restoration from
            // libjxl's vertical-sampling filter defect. Both independent decoders must agree.
            assert_eq!(case.orientation, 1);
            assert!(!case.grayscale);
            let expanded =
                std::fs::read(temporary.join(format!("{}.expanded.jxl", case.name))).unwrap();
            let expanded_info = jxl_gpu_bitstream::parse(&expanded, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            assert_eq!(expanded_info.frames[0].jpeg_upsampling, [0; 3]);
            let native = extra_channels::libjxl_output(
                &expanded,
                &["--preserve-alpha", "--keep-orientation"],
            )
            .unwrap();
            let image = jxl_oxide::JxlImage::read_with_defaults(expanded.as_slice()).unwrap();
            let render = image.render_frame(0).unwrap();
            let planar = render.image_planar();
            assert_eq!(planar.len(), 3 + case.extra_factors.len());
            let pixels = (case.size[0] * case.size[1]) as usize;
            let mut samples = Vec::with_capacity(fast.len());
            for i in 0..pixels {
                samples.extend([
                    planar[0].buf()[i],
                    planar[1].buf()[i],
                    planar[2].buf()[i],
                    planar.get(3).map_or(1.0, |alpha| alpha.buf()[i]),
                ]);
            }
            samples.extend(
                planar[3..]
                    .iter()
                    .flat_map(|plane| plane.buf().iter().copied()),
            );
            assert_eq!(samples.len(), fast.len());
            assert_eq!(samples.len(), native.len());
            assert!(
                samples
                    .iter()
                    .chain(&native)
                    .chain(&fast)
                    .all(|value| value.is_finite()),
                "{} independent references contain non-finite samples",
                case.name
            );
            let independent = samples
                .iter()
                .zip(&native)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                independent <= 2e-6,
                "{} independent expanded references differ: {independent}",
                case.name
            );
            let difference = fast
                .iter()
                .zip(&samples)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "{}: equivalent libjxl/jxl-oxide maxAE {independent}; original native/equivalent maxAE {difference}",
                case.name
            );
            std::fs::write(
                output.join(format!("{}.expanded.jxl.hex", case.name)),
                offline::hex(&expanded),
            )
            .unwrap();
            native
        } else {
            fast
        };
        assert_eq!(
            reference.len(),
            info.image_header.width as usize
                * info.image_header.height as usize
                * (4 + case.extra_factors.len())
        );
        assert!(reference.iter().all(|value| value.is_finite()));
        std::fs::write(
            output.join(format!("{}.jxl.hex", case.name)),
            offline::hex(&bytes),
        )
        .unwrap();
        if case.orientation != 1 {
            let oriented = extra_channels::libjxl_output(&bytes, &["--preserve-alpha"])
                .expect("native libjxl decoder is required");
            std::fs::write(
                output.join(format!("{}.oriented.f32.hex", case.name)),
                offline::float_hex(
                    &oriented
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<_>>(),
                ),
            )
            .unwrap();
        }
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
        eprintln!("{}: {} bytes, {:?}", case.name, bytes.len(), case.selectors);
    }
    std::fs::remove_dir_all(temporary).unwrap();
}
