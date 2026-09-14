//! Assemble explicitly declared CMYK/YCCK corpus sources before native reference generation.
use std::path::Path;

#[derive(serde::Deserialize)]
struct Manifest {
    cases: Vec<Case>,
}

#[derive(serde::Deserialize)]
struct Case {
    name: String,
    mode: u32,
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "assemble_cmyk SEED_DIRECTORY NEW_OUTPUT_DIRECTORY"
    );
    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    assert!(!output.exists());
    std::fs::create_dir_all(output).unwrap();
    let manifest = std::fs::read(input.join("manifest.json")).unwrap();
    let cases: Manifest = serde_json::from_slice(&manifest).unwrap();
    assert_eq!(cases.cases.len(), 18);
    for case in cases.cases {
        let name = format!("{}.jxl", case.name);
        let bytes = std::fs::read(input.join(&name)).unwrap();
        let assembled = if case.mode == 2 {
            jxl_test_support::fixtures::frame_features::as_ycbcr_444(&bytes)
        } else {
            bytes.clone()
        };
        let inventory = |bytes: &[u8]| {
            let parsed = jxl_gpu_bitstream::parse(bytes, Default::default()).unwrap();
            (
                parsed.codestream_inventory(Default::default()).unwrap(),
                parsed.codestream().to_vec(),
            )
        };
        let (before, before_stream) = inventory(&bytes);
        let (after, after_stream) = inventory(&assembled);
        assert_eq!(before.image_header, after.image_header);
        assert_eq!(after.frames.len(), 3);
        for (before, after) in before.frames.iter().zip(&after.frames) {
            assert_eq!(after.do_ycbcr, case.mode == 2);
            assert_eq!(before.color_blend, after.color_blend);
            assert_eq!(before.extra_channel_blends, after.extra_channel_blends);
            assert_eq!(before.sections.len(), after.sections.len());
            for section in &before.sections {
                let rebuilt = after
                    .sections
                    .iter()
                    .find(|other| other.kind == section.kind)
                    .unwrap();
                assert!(
                    before_stream
                        [section.bytes.offset as usize..section.bytes.end().unwrap() as usize]
                        == after_stream
                            [rebuilt.bytes.offset as usize..rebuilt.bytes.end().unwrap() as usize],
                    "{} frame {} section {:?}: changed entropy",
                    case.name,
                    before.frame_index,
                    section.kind
                );
            }
        }
        std::fs::write(output.join(name), assembled).unwrap();
    }
    std::fs::write(output.join("manifest.json"), manifest).unwrap();
}
