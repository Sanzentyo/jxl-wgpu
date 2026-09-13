//! Original SDR profiles with independently encoded RGB/XYB sources and explicit YCbCr recipes.

use jxl_gpu_bitstream::{
    BitReader, BitWriter, CodestreamInventory, ColourEncodingInventory, ColourSpaceInventory,
    FrameEncoding, PrimariesInventory, RenderingIntentInventory, SampleBitDepth,
    TransferFunctionInventory, WhitePointInventory,
};
use jxl_gpu_formats::{
    ColorSpace, ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction,
};

use super::frame_features;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    ModularRgb,
    VarDctRgb,
    ModularXyb,
    VarDctXyb,
    ModularYcbcr,
    VarDctYcbcr,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::ModularRgb => "modular_rgb",
            Self::VarDctRgb => "vardct_rgb",
            Self::ModularXyb => "modular_xyb",
            Self::VarDctXyb => "vardct_xyb",
            Self::ModularYcbcr => "modular_ycbcr",
            Self::VarDctYcbcr => "vardct_ycbcr",
        }
    }
    pub fn encoding(self) -> FrameEncoding {
        match self {
            Self::ModularRgb | Self::ModularXyb | Self::ModularYcbcr => FrameEncoding::Modular,
            _ => FrameEncoding::VarDct,
        }
    }
    pub fn ycbcr(self) -> bool {
        matches!(self, Self::ModularYcbcr | Self::VarDctYcbcr)
    }
    pub fn xyb(self) -> bool {
        matches!(self, Self::ModularXyb | Self::VarDctXyb)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Profile {
    pub name: &'static str,
    pub primaries: PrimariesInventory,
    pub grayscale: bool,
}
#[derive(Clone, Copy, Debug)]
pub struct Transfer {
    pub name: &'static str,
    pub transfer: TransferFunctionInventory,
}
#[derive(Clone, Debug)]
pub struct Case {
    pub name: String,
    pub mode: Mode,
    pub profile: Profile,
    pub transfer: Transfer,
    pub sequence: bool,
    pub floating: bool,
}

const PROFILES: [Profile; 4] = [
    Profile {
        name: "bt709",
        primaries: PrimariesInventory::Srgb,
        grayscale: false,
    },
    Profile {
        name: "bt2020",
        primaries: PrimariesInventory::Bt2100,
        grayscale: false,
    },
    Profile {
        name: "p3",
        primaries: PrimariesInventory::P3,
        grayscale: false,
    },
    Profile {
        name: "gray",
        primaries: PrimariesInventory::Srgb,
        grayscale: true,
    },
];
const TRANSFERS: [Transfer; 3] = [
    Transfer {
        name: "linear",
        transfer: TransferFunctionInventory::Linear,
    },
    Transfer {
        name: "srgb",
        transfer: TransferFunctionInventory::Srgb,
    },
    Transfer {
        name: "bt709",
        transfer: TransferFunctionInventory::Bt709,
    },
];
const MODES: [Mode; 6] = [
    Mode::ModularRgb,
    Mode::VarDctRgb,
    Mode::ModularXyb,
    Mode::VarDctXyb,
    Mode::ModularYcbcr,
    Mode::VarDctYcbcr,
];

pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for mode in MODES {
        for profile in PROFILES {
            if mode.ycbcr() && profile.grayscale {
                continue;
            }
            for transfer in TRANSFERS {
                for sequence in [false, true] {
                    cases.push(Case::new(mode, profile, transfer, sequence, false));
                }
            }
        }
    }
    for mode in MODES[..4].iter().copied() {
        for sequence in [false, true] {
            for (profile, transfer) in [(PROFILES[1], TRANSFERS[0]), (PROFILES[2], TRANSFERS[2])] {
                cases.push(Case::new(mode, profile, transfer, sequence, true));
            }
        }
    }
    cases
}

impl Case {
    fn new(
        mode: Mode,
        profile: Profile,
        transfer: Transfer,
        sequence: bool,
        floating: bool,
    ) -> Self {
        Self {
            name: format!(
                "{}_{}_{}_{}{}",
                mode.name(),
                profile.name,
                transfer.name,
                if floating { "float_" } else { "" },
                if sequence { "sequence" } else { "still" }
            ),
            mode,
            profile,
            transfer,
            sequence,
            floating,
        }
    }

    pub fn format(&self) -> PixelFormat {
        let mut color = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
        let ColorSpecification::Defined(ref mut spec) = color else {
            unreachable!()
        };
        spec.space = match self.profile.primaries {
            PrimariesInventory::Srgb => ColorSpace::Bt709,
            PrimariesInventory::Bt2100 => ColorSpace::Bt2020,
            PrimariesInventory::P3 => ColorSpace::DisplayP3,
            _ => unreachable!(),
        };
        spec.transfer = match self.transfer.transfer {
            TransferFunctionInventory::Linear => TransferFunction::Linear,
            TransferFunctionInventory::Srgb => TransferFunction::Srgb,
            TransferFunctionInventory::Bt709 => TransferFunction::Bt709,
            _ => unreachable!(),
        };
        PixelFormat::rgb_f32(RgbChannelOrder::Rgba, false, color)
    }

    pub fn bytes(&self) -> Vec<u8> {
        read_bytes(&self.name)
    }

    pub fn reference(&self) -> Vec<f32> {
        std::fs::read_to_string(directory().join(format!("{}.original.f32.hex", self.name)))
            .unwrap()
            .split_whitespace()
            .map(|word| f32::from_bits(u32::from_str_radix(word, 16).unwrap()))
            .collect()
    }

    pub fn validate(&self, inventory: &CodestreamInventory) {
        let image = &inventory.image_header;
        assert_eq!((image.width, image.height), (37, 19), "{}", self.name);
        assert_eq!(image.grayscale, self.profile.grayscale);
        assert_eq!(image.xyb_encoded, self.mode.xyb());
        assert_eq!(
            image.colour_encoding,
            ColourEncodingInventory::Enumerated {
                colour_space: if self.profile.grayscale {
                    ColourSpaceInventory::Grey
                } else {
                    ColourSpaceInventory::Rgb
                },
                white_point: WhitePointInventory::D65,
                primaries: self.profile.primaries,
                transfer_function: self.transfer.transfer,
                rendering_intent: RenderingIntentInventory::Relative,
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
                    bits_per_sample: if self.mode.ycbcr() { 8 } else { 12 },
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
            assert_eq!(frame.encoding, self.mode.encoding(), "{}", self.name);
            assert_eq!(frame.do_ycbcr, self.mode.ycbcr(), "{}", self.name);
            assert_eq!(frame.jpeg_upsampling, [0; 3]);
        }
    }

    /// The public native encoder overrides YCbCr. Interpret its unchanged 8-bit component
    /// entropy as 4:4:4 YCbCr using the normative frame flag and selector fields. Image precision,
    /// channel counts, transform headers, section bytes and composition metadata do not change.
    /// Native libjxl must independently decode the resulting stream before it becomes a fixture.
    pub fn encode_ycbcr(&self) -> Vec<u8> {
        assert!(self.mode.ycbcr());
        let source = read_bytes(&format!("{}.rgb-source", self.name));
        let parsed = jxl_gpu_bitstream::parse(&source, Default::default()).unwrap();
        let info = parsed.codestream_inventory(Default::default()).unwrap();
        assert!(!info.image_header.xyb_encoded && !info.image_header.grayscale);
        assert_eq!(
            info.image_header.bit_depth,
            SampleBitDepth::Integer { bits_per_sample: 8 }
        );
        let source = parsed.codestream();
        let mut output = source[..info.frames[0].header_bits.offset as usize / 8].to_vec();
        for frame in &info.frames {
            assert!(!frame.do_ycbcr && !frame.uses_lf_frame());
            let start = frame.header_bits.offset;
            let mut reader = BitReader::new(source);
            reader.skip_bits(start).unwrap();
            assert_eq!(reader.read_bits(1).unwrap(), 0, "explicit frame header");
            reader.skip_bits(3).unwrap();
            let mut flags = BitWriter::new();
            frame_features::flags(&mut flags, frame.flags);
            let mut expected = BitReader::new(flags.as_bytes());
            assert_eq!(
                reader.read_bits(flags.bit_len() as u8).unwrap(),
                expected.read_bits(flags.bit_len() as u8).unwrap()
            );
            assert_eq!(reader.read_bits(1).unwrap(), 0, "RGB source");
            let transform_offset = start + 4 + flags.bit_len() as u64;
            let mut header = BitWriter::new();
            frame_features::copy_bits(&mut header, source, start, transform_offset);
            header.write_bits(1, 1).unwrap();
            header.write_bits(0, 6).unwrap(); // Three independent full-resolution selectors.
            frame_features::copy_bits(
                &mut header,
                source,
                transform_offset + 1,
                frame.header_bits.end().unwrap(),
            );
            output.extend(frame_features::packet_frame_prefix(
                source,
                frame,
                jxl_wgpu_encode::BitFragment::new(header.as_bytes().to_vec(), header.bit_len())
                    .unwrap(),
                None,
            ));
        }
        output
    }
}

pub fn directory() -> std::path::PathBuf {
    crate::decoder_directory().join("test-data/original_color")
}

fn read_bytes(name: &str) -> Vec<u8> {
    crate::offline::unhex(
        &std::fs::read_to_string(directory().join(format!("{name}.jxl.hex"))).unwrap(),
    )
}
