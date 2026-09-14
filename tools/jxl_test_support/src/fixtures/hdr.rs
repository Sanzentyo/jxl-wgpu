use jxl_gpu_bitstream::{
    CodestreamInventory, ColourEncodingInventory, ColourSpaceInventory, FrameEncoding,
    PrimariesInventory, SampleBitDepth, TransferFunctionInventory, WhitePointInventory,
};
use jxl_gpu_formats::{
    ColorSpace, ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction,
};
use std::path::PathBuf;

pub struct Case {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub modular: bool,
    pub xyb: bool,
    pub space: ColorSpace,
    pub gray: bool,
    pub transfer: TransferFunction,
    pub nits: f64,
    pub sequence: bool,
    pub floating: bool,
}

fn directory() -> PathBuf {
    crate::decoder_directory().join("test-data/hdr")
}

pub fn cases() -> Vec<Case> {
    let manifest = std::fs::read_to_string(directory().join("manifest.txt")).unwrap();
    let cases: Vec<_> = manifest
        .lines()
        .map(|line| {
            let columns: [&str; 10] = line
                .split_whitespace()
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            Case {
                name: columns[0].to_owned(),
                width: columns[1].parse().unwrap(),
                height: columns[2].parse().unwrap(),
                modular: columns[3] == "1",
                xyb: columns[4] == "1",
                space: match columns[5] {
                    "bt709" | "gray" => ColorSpace::Bt709,
                    "bt2020" => ColorSpace::Bt2020,
                    "p3" => ColorSpace::DisplayP3,
                    value => panic!("unknown HDR primaries {value}"),
                },
                gray: columns[5] == "gray",
                transfer: match columns[6] {
                    "pq" => TransferFunction::Pq,
                    "hlg" => TransferFunction::Hlg,
                    value => panic!("unknown HDR transfer {value}"),
                },
                nits: columns[7].parse().unwrap(),
                sequence: columns[8] == "1",
                floating: columns[9] == "1",
            }
        })
        .collect();
    assert_eq!(cases.len(), 56);
    cases
}

impl Case {
    pub fn bytes(&self) -> Vec<u8> {
        crate::offline::hex::unhex(
            &std::fs::read_to_string(directory().join(format!("{}.jxl.hex", self.name))).unwrap(),
        )
    }

    pub fn reference(&self, linear: bool) -> Vec<f32> {
        let domain = if linear { "linear" } else { "original" };
        std::fs::read_to_string(directory().join(format!("{}.{domain}.f32.hex", self.name)))
            .unwrap()
            .split_whitespace()
            .map(|word| f32::from_bits(u32::from_str_radix(word, 16).unwrap()))
            .collect()
    }

    pub fn format(&self, transfer: TransferFunction, space: ColorSpace) -> PixelFormat {
        let ColorSpecification::Defined(mut color) =
            jxl_wgpu_decode::vardct_rgb8_format().color_spec
        else {
            unreachable!()
        };
        color.transfer = transfer;
        color.space = space;
        PixelFormat::rgb_f32(
            RgbChannelOrder::Rgba,
            false,
            ColorSpecification::Defined(color),
        )
    }

    pub fn frame_words(&self) -> usize {
        self.width * self.height * 4
    }
    pub fn frame_count(&self) -> usize {
        if self.sequence { 4 } else { 1 }
    }
    pub fn tolerance(&self) -> f32 {
        if self.modular && !self.xyb {
            1e-5
        } else {
            1.0 / 1024.0
        }
    }

    pub fn validate(&self, inventory: &CodestreamInventory) {
        let image = &inventory.image_header;
        assert_eq!(
            (image.width as usize, image.height as usize),
            (self.width, self.height)
        );
        assert_eq!(image.grayscale, self.gray);
        assert_eq!(image.xyb_encoded, self.xyb);
        assert_eq!(
            f64::from(image.tone_mapping.intensity_target.to_f32()),
            self.nits
        );
        let ColourEncodingInventory::Enumerated {
            colour_space,
            white_point,
            primaries,
            transfer_function,
            ..
        } = image.colour_encoding
        else {
            panic!("HDR corpus must retain enumerated metadata")
        };
        assert_eq!(
            colour_space,
            if self.gray {
                ColourSpaceInventory::Grey
            } else {
                ColourSpaceInventory::Rgb
            }
        );
        assert_eq!(white_point, WhitePointInventory::D65);
        assert_eq!(
            primaries,
            match self.space {
                ColorSpace::Bt709 => PrimariesInventory::Srgb,
                ColorSpace::Bt2020 => PrimariesInventory::Bt2100,
                ColorSpace::DisplayP3 => PrimariesInventory::P3,
                _ => unreachable!(),
            }
        );
        assert_eq!(
            transfer_function,
            match self.transfer {
                TransferFunction::Pq => TransferFunctionInventory::Pq,
                TransferFunction::Hlg => TransferFunctionInventory::Hlg,
                _ => unreachable!(),
            }
        );
        assert!(image.embedded_icc.is_none());
        assert_eq!(
            image.bit_depth,
            if self.floating {
                SampleBitDepth::Float {
                    bits_per_sample: 32,
                    exponent_bits_per_sample: 8,
                }
            } else {
                SampleBitDepth::Integer {
                    bits_per_sample: 16,
                }
            }
        );
        assert_eq!(image.extra_channels.len(), 1);
        assert_eq!(
            image.extra_channels[0].bit_depth,
            SampleBitDepth::Integer {
                bits_per_sample: 10
            }
        );
        assert_eq!(inventory.frames.len(), if self.sequence { 6 } else { 1 });
        for frame in &inventory.frames {
            assert_eq!(
                frame.encoding,
                if self.modular {
                    FrameEncoding::Modular
                } else {
                    FrameEncoding::VarDct
                }
            );
            assert!(!frame.do_ycbcr);
        }
    }
}
