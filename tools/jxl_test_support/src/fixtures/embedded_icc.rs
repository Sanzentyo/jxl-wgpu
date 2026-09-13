//! Native JPEG XL streams whose per-channel / sampled ICC curves remain embedded.
use jxl_gpu_bitstream::{
    CodestreamInventory, ColourEncodingInventory, ColourSpaceInventory, FrameEncoding,
    SampleBitDepth,
};

#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub gray: bool,
    pub encoding: FrameEncoding,
    pub xyb: bool,
}

pub fn cases() -> impl Iterator<Item = Case> {
    [false, true].into_iter().flat_map(|gray| {
        [FrameEncoding::Modular, FrameEncoding::VarDct]
            .into_iter()
            .flat_map(move |encoding| {
                [false, true].into_iter().map(move |xyb| Case {
                    gray,
                    encoding,
                    xyb,
                })
            })
    })
}

pub fn directory() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/jxl_wgpu_decode/test-data/embedded_icc")
}

impl Case {
    fn color_name(self) -> &'static str {
        if self.gray { "gray" } else { "rgb" }
    }

    pub fn name(self) -> String {
        format!(
            "{}_{}_{}",
            self.color_name(),
            match self.encoding {
                FrameEncoding::Modular => "modular",
                FrameEncoding::VarDct => "vardct",
            },
            if self.xyb { "xyb" } else { "original" },
        )
    }

    pub fn bytes(self) -> Vec<u8> {
        crate::offline::unhex(
            &std::fs::read_to_string(directory().join(format!("{}.jxl.hex", self.name()))).unwrap(),
        )
    }

    pub fn profile(self) -> Vec<u8> {
        std::fs::read(directory().join(format!("{}.icc", self.color_name()))).unwrap()
    }

    pub fn input(self) -> Vec<u32> {
        crate::offline::unhex(
            &std::fs::read_to_string(
                directory().join(format!("{}.input.f32.hex", self.color_name())),
            )
            .unwrap(),
        )
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(u32::from_le_bytes)
        .collect()
    }

    pub fn validate(self, inventory: &CodestreamInventory) {
        let image = &inventory.image_header;
        assert_eq!((image.width, image.height), (17, 9));
        assert_eq!(image.grayscale, self.gray);
        assert_eq!(image.xyb_encoded, self.xyb);
        assert_eq!(
            image.colour_encoding,
            ColourEncodingInventory::IccProfile {
                colour_space: if self.gray {
                    ColourSpaceInventory::Grey
                } else {
                    ColourSpaceInventory::Rgb
                },
            }
        );
        assert_eq!(
            image.embedded_icc.as_ref().unwrap().profile.as_ref(),
            self.profile()
        );
        assert_eq!(image.extra_channels.len(), 1);
        assert_eq!(
            image.bit_depth,
            SampleBitDepth::Float {
                bits_per_sample: 32,
                exponent_bits_per_sample: 8,
            }
        );
        assert_eq!(image.extra_channels[0].bit_depth, image.bit_depth);
        assert_eq!(inventory.frames.len(), 1);
        assert_eq!(inventory.frames[0].encoding, self.encoding);
        assert!(!inventory.frames[0].do_ycbcr);
    }
}
