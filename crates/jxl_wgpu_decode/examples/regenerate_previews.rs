//! Reproduce independent preview entropy and combine it with checked main-image codestreams.
use std::path::{Path, PathBuf};
use std::process::Command;

use jxl_test_support::fixtures::preview;
use jxl_test_support::offline;

fn source(root: &Path, name: &str) -> Vec<u8> {
    offline::unhex(&std::fs::read_to_string(root.join(format!("{name}.jxl.hex"))).unwrap())
}

fn encoded(temporary: &Path, mode: &str, width: u32, height: u32, main: &[u8]) -> Vec<u8> {
    let color = match preview::inventory(main).image_header.colour_encoding {
        jxl_gpu_bitstream::ColourEncodingInventory::Enumerated {
            rendering_intent: jxl_gpu_bitstream::RenderingIntentInventory::Relative,
            ..
        } => "color_space=RGB_D65_SRG_Rel_SRG",
        jxl_gpu_bitstream::ColourEncodingInventory::Enumerated {
            rendering_intent: jxl_gpu_bitstream::RenderingIntentInventory::Perceptual,
            ..
        } => "color_space=RGB_D65_SRG_Per_SRG",
        _ => panic!("unexpected main color encoding"),
    };
    let mut pixels = format!("P6\n{width} {height}\n255\n").into_bytes();
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                pixels.push((32 + (x * (channel + 2) * 3 + y * (7 - channel)) % 192) as u8);
            }
        }
    }
    let input = temporary.join("source.ppm");
    let output = temporary.join("source.jxl");
    std::fs::write(&input, pixels).unwrap();
    let mut command = Command::new("cjxl");
    command.arg(&input).arg(&output).args([
        "--distance=1",
        "--photon_noise_iso=800",
        "--gaborish=0",
        "--epf=0",
        "-x",
        color,
        "--quiet",
    ]);
    if mode == "modular" {
        command.arg("--modular=1");
    }
    offline::run(&mut command);
    std::fs::read(output).unwrap()
}

fn write(root: &Path, name: &str, data: Vec<u8>) {
    let info = preview::inventory(&data);
    std::fs::write(root.join(format!("{name}.jxl.hex")), offline::hex(&data)).unwrap();
    eprintln!(
        "{name}: {} bytes, {:?}, {} physical frames",
        data.len(),
        info.image_header.preview_size,
        info.frames.len()
    );
}

fn main() {
    let source_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| source_root.join("preview"));
    std::fs::create_dir_all(&output).unwrap();
    let temporary = std::env::temp_dir().join(format!("jxl-wgpu-preview-{}", std::process::id()));
    std::fs::create_dir(&temporary).unwrap();
    let main = source(&source_root, "noise/vardct_257x17");
    for mode in ["modular", "vardct"] {
        let base = encoded(&temporary, mode, 15, 27, &main);
        write(
            &output,
            mode,
            preview::combine(&main, &base, Default::default()),
        );
        write(
            &output,
            &format!("nonfinal_{mode}"),
            preview::combine(
                &main,
                &base,
                preview::Options {
                    is_last: false,
                    ..Default::default()
                },
            ),
        );
        write(
            &output,
            &format!("main_modular_preview_{mode}"),
            preview::combine(
                &source(&source_root, "noise/modular_257x17"),
                &base,
                Default::default(),
            ),
        );
        let lf = source(&source_root, "noise/lf_progressive_ac");
        let lf_preview = encoded(&temporary, mode, 15, 27, &lf);
        write(
            &output,
            &format!("lf_{mode}"),
            preview::combine(&lf, &lf_preview, Default::default()),
        );
        let animation = source(&source_root, "noise/mixed_frames");
        let animation_preview = encoded(&temporary, mode, 15, 27, &animation);
        write(
            &output,
            &format!("animation_{mode}"),
            preview::combine(
                &animation,
                &animation_preview,
                preview::Options {
                    is_last: false,
                    duration: 7,
                    timecode: 0x12345678,
                    ..Default::default()
                },
            ),
        );
        for div8 in [false, true] {
            let height = if div8 { 16 } else { 27 };
            for ratio in 0..8 {
                if !div8 && ratio == 0 {
                    continue;
                }
                let width = match ratio {
                    0 => 8,
                    1 => height,
                    2 => height * 12 / 10,
                    3 => height * 4 / 3,
                    4 => height * 3 / 2,
                    5 => height * 16 / 9,
                    6 => height * 5 / 4,
                    7 => height * 2,
                    _ => unreachable!(),
                };
                let preview = encoded(&temporary, mode, width, height, &main);
                let name = format!("ratio_{mode}_{}_{ratio}", u32::from(div8));
                write(
                    &output,
                    &name,
                    preview::combine(
                        &main,
                        &preview,
                        preview::Options {
                            div8,
                            ratio,
                            ..Default::default()
                        },
                    ),
                );
            }
        }
    }
    for (name, original) in [
        ("alpha_modular", "extras_rgba"),
        ("alpha_vardct", "vardct_extras_rgba"),
        ("rgb_modular", "noise/modular_rgb_group256"),
        ("rgb_vardct", "noise/vardct_rgb_257x17"),
        ("jpeg_420", "noise/jpeg_420"),
        ("float_vardct", "noise/vardct_rgb_float32_up4"),
        ("resampled_modular", "noise/modular_up4"),
        ("resampled_vardct", "noise/vardct_up4"),
    ] {
        let data = source(&source_root, original);
        write(
            &output,
            name,
            preview::combine(&data, &data, Default::default()),
        );
    }
    std::fs::remove_dir_all(temporary).unwrap();
}
