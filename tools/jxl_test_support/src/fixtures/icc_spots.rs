//! Native, explicitly inventoried spot/color/profile combinations.

use jxl_gpu_bitstream::{CodestreamInventory, FrameEncoding, SampleBitDepth};

#[derive(Debug)]
pub struct Case {
    pub name: String,
    pub icc: bool,
    pub gray: bool,
    pub modular: bool,
    pub original: bool,
    pub sequence: bool,
}

pub fn directory() -> std::path::PathBuf {
    crate::decoder_directory().join("test-data/icc_spots")
}

pub fn cases() -> Vec<Case> {
    let manifest = std::fs::read_to_string(directory().join("manifest.tsv")).unwrap();
    let mut lines = manifest.lines();
    assert_eq!(
        lines.next(),
        Some("name\ticc\tgray\tmodular\toriginal\tsequence")
    );
    let boolean = |value| match value {
        "0" => false,
        "1" => true,
        _ => panic!("invalid manifest boolean"),
    };
    lines
        .map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 6);
            Case {
                name: fields[0].into(),
                icc: boolean(fields[1]),
                gray: boolean(fields[2]),
                modular: boolean(fields[3]),
                original: boolean(fields[4]),
                sequence: boolean(fields[5]),
            }
        })
        .collect()
}

impl Case {
    pub fn bytes(&self) -> Vec<u8> {
        std::fs::read(directory().join(format!("{}.jxl", self.name))).unwrap()
    }

    pub fn validate(&self, inventory: &CodestreamInventory) {
        let image = &inventory.image_header;
        assert_eq!((image.width, image.height), (17, 9));
        assert_eq!(image.grayscale, self.gray);
        assert_eq!(image.xyb_encoded, !self.original);
        assert_eq!(image.embedded_icc.is_some(), self.icc);
        assert_eq!(image.extra_channels.len(), 9);
        assert_eq!(
            image.bit_depth,
            SampleBitDepth::Float {
                bits_per_sample: 32,
                exponent_bits_per_sample: 8
            }
        );
        assert_eq!(inventory.frames.len(), if self.sequence { 3 } else { 1 });
        for frame in &inventory.frames {
            assert_eq!(
                frame.encoding,
                if self.modular {
                    FrameEncoding::Modular
                } else {
                    FrameEncoding::VarDct
                }
            );
        }
    }
}
