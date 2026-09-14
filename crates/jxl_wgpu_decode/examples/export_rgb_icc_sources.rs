//! Offline source export: independent decoder/colorimetry, never production GPU pixels.
use jxl_gpu_bitstream::FrameEncoding;
use jxl_gpu_formats::{ColorSpecification, TransferFunction};
use jxl_test_support::{fixtures::original_color as corpus, oracles::color};
use std::{io::Write, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = PathBuf::from(std::env::args().nth(1).expect("new output directory"));
    assert!(!output.exists());
    std::fs::create_dir_all(&output)?;
    let mut manifest = std::fs::File::create(output.join("manifest.tsv"))?;
    let targets = [
        "gamma_v4",
        "gray",
        "intents/rgb_2",
        "intents/rgb_8",
        "mpe/identity",
        "lut/lut8_xyz_3",
        "lut/lut16_lab_3",
        "lut/ab_lab_1",
        "lut/lut16_v2_xyz_3",
    ];
    let cases: Vec<_> = corpus::cases()
        .into_iter()
        .chain(corpus::analytic_cases())
        .collect();
    assert_eq!(cases.len(), 228);
    for (index, case) in cases.iter().enumerate() {
        let ColorSpecification::Defined(encoding) = case.format().color_spec else {
            unreachable!()
        };
        let linear = case.mode.xyb() && !case.sequence;
        let rgba: Vec<[f64; 4]> = if linear {
            color::linear_original_still(case)
        } else {
            case.reference()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| p.map(f64::from))
                .collect()
        };
        let matrix = color::pcs_matrix(encoding.space);
        let transfer = if linear {
            TransferFunction::Linear
        } else {
            encoding.transfer
        };
        let tolerance = if case.mode.encoding() == FrameEncoding::Modular && !case.mode.xyb() {
            1e-5
        } else {
            1.0 / 1024.0
        };
        let mut file = std::fs::File::create(output.join(format!("{}.pcs", case.name)))?;
        for pixel in &rgba {
            let rgb = [pixel[0], pixel[1], pixel[2]];
            let center = color::convert(rgb, transfer, TransferFunction::Linear, matrix);
            let range = color::interval(rgb, transfer, TransferFunction::Linear, matrix, tolerance);
            for c in 0..3 {
                // Established codec reconstruction uncertainty plus F32 output colorimetry.
                // This interval is fixed before any ICC target or GPU result is evaluated.
                let radius = (center[c] - range[c][0])
                    .abs()
                    .max((range[c][1] - center[c]).abs())
                    + 5e-6 * (1.0 + center[c].abs());
                file.write_all(&center[c].to_le_bytes())?;
                file.write_all(&radius.to_le_bytes())?;
            }
        }
        assert_eq!(rgba.len(), 37 * 19 * if case.sequence { 4 } else { 1 });
        let target = targets[(index + index / targets.len()) % targets.len()];
        writeln!(
            manifest,
            "{}\t{}\t{}",
            case.name,
            target,
            rgba.len() / (37 * 19)
        )?;
    }
    Ok(())
}
