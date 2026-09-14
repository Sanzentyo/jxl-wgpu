//! Offline HDR-to-PCS sources and predeclared error intervals; no GPU input.
use jxl_gpu_formats::TransferFunction;
use jxl_test_support::{
    fixtures::hdr as corpus,
    oracles::{color, hdr},
};
use std::{io::Write, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = PathBuf::from(std::env::args().nth(1).expect("new output directory"));
    assert!(!output.exists());
    std::fs::create_dir_all(&output)?;
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
    let mut manifest = std::fs::File::create(output.join("manifest.tsv"))?;
    for (i, case) in corpus::cases().iter().enumerate() {
        let linear = case.xyb && !case.sequence;
        let rgba = case.reference(linear);
        let transfer = if linear {
            TransferFunction::Linear
        } else {
            case.transfer
        };
        let matrix = color::pcs_matrix(case.space);
        let mut file = std::fs::File::create(output.join(format!("{}.pcs", case.name)))?;
        for pixel in rgba.as_chunks::<4>().0 {
            let input = [pixel[0], pixel[1], pixel[2]].map(f64::from);
            let light = hdr::to_linear(input, transfer, case.space, case.nits);
            let range = hdr::linear_interval(
                input,
                transfer,
                case.space,
                case.nits,
                f64::from(case.tolerance()),
            );
            for row in matrix {
                let center: f64 = (0..3).map(|c| row[c] * light[c]).sum();
                let bounds: [f64; 2] = std::array::from_fn(|edge| {
                    (0..3)
                        .map(|c| row[c] * range[c][if row[c] >= 0.0 { edge } else { 1 - edge }])
                        .sum()
                });
                let radius = (center - bounds[0]).abs().max((bounds[1] - center).abs())
                    + 5e-5 * (1.0 + center.abs());
                assert!(center.is_finite() && radius.is_finite());
                file.write_all(&center.to_le_bytes())?;
                file.write_all(&radius.to_le_bytes())?;
            }
        }
        let target = targets[(i + i / targets.len()) % targets.len()];
        writeln!(
            manifest,
            "{}\t{}\t{}\t{}\t{}",
            case.name,
            target,
            case.width,
            case.height,
            case.frame_count()
        )?;
    }
    Ok(())
}
