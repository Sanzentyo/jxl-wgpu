use std::{collections::BTreeMap, io::Read, path::Path};

use jxl_gpu_bitstream::{
    ColourEncodingInventory, ColourSpaceInventory, ExtraChannelTypeInventory, PrimariesInventory,
    RenderingIntentInventory, SampleBitDepth, TransferFunctionInventory, WhitePointInventory,
};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug)]
pub(super) enum ReferenceColor {
    Transfer(jxl_gpu_formats::TransferFunction),
    OriginalNumeric,
    Profile,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum OriginalProfile {
    Unspecified,
    Embedded,
    Enumerated {
        grayscale: bool,
        transfer: TransferFunctionInventory,
        intent: RenderingIntentInventory,
    },
}

pub(super) struct Case {
    pub name: &'static str,
    pub input_sha256: &'static str,
    pub descriptor_sha256: &'static str,
    // A few upstream NPYs are Git blobs rather than descriptor-addressed objects.
    pub inline_pixels_sha256: Option<&'static str>,
    pub reference_parts: usize,
    pub color: ReferenceColor,
    pub original: OriginalProfile,
}

pub(super) struct Alternate {
    pub name: &'static str,
    pub primary: &'static str,
    pub input_sha256: &'static str,
    pub descriptor_sha256: &'static str,
    pub distinct_pixels: bool,
}

#[derive(serde::Deserialize)]
pub(super) struct Frame {
    pub name: String,
    pub rms_error: f64,
    pub peak_error: f64,
    pub duration: Option<f64>,
}

#[derive(serde::Deserialize)]
pub(super) struct Descriptor {
    pub frames: Vec<Frame>,
    pub original_icc: Option<String>,
    bits_per_sample: Vec<u32>,
    exp_bits_per_sample: Vec<u32>,
    extra_channel_type: Vec<String>,
    intensity_target: f32,
    min_nits: f32,
    relative_to_max_display: u32,
    linear_below: f32,
    sha256sums: BTreeMap<String, String>,
}

pub(super) struct Reference {
    pub name: &'static str,
    pub input: Vec<u8>,
    pub profile: Vec<u8>,
    pub descriptor: Descriptor,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub colors: usize,
    pub color: ReferenceColor,
    pub depths: Vec<SampleBitDepth>,
    pub pixels: Vec<f32>,
    pub animation: Option<jxl_gpu_bitstream::AnimationInventory>,
}

fn check(bytes: &[u8], expected: &str) {
    assert_eq!(
        Sha256::digest(bytes).as_slice(),
        jxl_test_support::offline::hex::unhex(expected)
    );
}

impl Case {
    pub fn load(&self) -> Reference {
        self.load_inner(None)
    }

    pub fn load_alternate(&self, alternate: &Alternate) -> Reference {
        assert_eq!(self.name, alternate.primary);
        self.load_inner(Some(alternate))
    }

    fn load_inner(&self, alternate: Option<&Alternate>) -> Reference {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/official_conformance");
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test-data/official_conformance")
            .join(self.name);
        let input = std::fs::read(directory.join("input.jxl")).unwrap();
        check(
            &input,
            alternate.map_or(self.input_sha256, |a| a.input_sha256),
        );
        let descriptor_path = alternate.map_or_else(
            || directory.join("test.json"),
            |a| root.join("variants").join(format!("{}.json", a.name)),
        );
        let descriptor = std::fs::read(descriptor_path).unwrap();
        // The descriptor hash pins the published bounds as well as all referenced object hashes.
        check(
            &descriptor,
            alternate.map_or(self.descriptor_sha256, |a| a.descriptor_sha256),
        );
        let descriptor: Descriptor = serde_json::from_slice(&descriptor).unwrap();
        assert!(!descriptor.frames.is_empty());
        let profile = std::fs::read(directory.join("reference.icc")).unwrap();
        check(&profile, &descriptor.sha256sums["reference.icc"]);
        let mut bytes = Vec::new();
        let pixels_path = match alternate {
            Some(a) if a.distinct_pixels => {
                root.join("variants").join(format!("{}.npy.gz", a.name))
            }
            _ => directory.join("reference.npy.gz"),
        };
        // Splitting a large gzip stream keeps each stored part bounded; the joined stream's
        // decompressed bytes still have exactly the original upstream NPY digest below.
        assert!((1..=3).contains(&self.reference_parts));
        let mut compressed: Box<dyn Read> = Box::new(std::io::empty());
        for part in 0..self.reference_parts {
            let path = if self.reference_parts == 1 {
                pixels_path.clone()
            } else {
                std::path::PathBuf::from(format!("{}.part{part:02}", pixels_path.display()))
            };
            compressed = Box::new(compressed.chain(std::fs::File::open(path).unwrap()));
        }
        flate2::read::GzDecoder::new(compressed)
            .read_to_end(&mut bytes)
            .unwrap();
        let pixels_hash = match descriptor.sha256sums.get("reference_image.npy") {
            Some(hash) => {
                assert!(self.inline_pixels_sha256.is_none());
                hash.as_str()
            }
            None => {
                assert!(alternate.is_none());
                self.inline_pixels_sha256.expect("pinned upstream NPY blob")
            }
        };
        check(&bytes, pixels_hash);
        assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
        let offset = 10 + usize::from(u16::from_le_bytes(bytes[8..10].try_into().unwrap()));
        let header = std::str::from_utf8(&bytes[10..offset]).unwrap();
        assert!(header.contains("'descr': '<f4'") && header.contains("'fortran_order': False"));
        let shape: Vec<usize> = header
            .split("'shape': (")
            .nth(1)
            .unwrap()
            .split(')')
            .next()
            .unwrap()
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(|part| part.parse().unwrap())
            .collect();
        assert_eq!(shape.len(), 4);
        assert_eq!(shape[0], descriptor.frames.len());
        let (words, tail) = bytes[offset..].as_chunks::<4>();
        assert!(tail.is_empty());
        assert_eq!(words.len(), shape.iter().product::<usize>());
        let pixels = words.iter().copied().map(f32::from_le_bytes).collect();
        let info = jxl_gpu_bitstream::parse(&input, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &info.image_header;
        if image.animation.is_none() {
            assert_eq!(shape[0], 1);
        }
        assert_eq!(
            image.animation.is_some(),
            descriptor
                .frames
                .iter()
                .any(|frame| frame.duration.is_some())
        );
        let (width, height) = if image.orientation >= 5 {
            (image.height, image.width)
        } else {
            (image.width, image.height)
        };
        assert_eq!([shape[1], shape[2]], [height as usize, width as usize]);
        let colors = if image.grayscale { 1 } else { 3 };
        assert_eq!(shape[3], colors + image.extra_channels.len());
        let depths: Vec<_> = std::iter::once(image.bit_depth)
            .chain(image.extra_channels.iter().map(|extra| extra.bit_depth))
            .collect();
        let declared_depths: Vec<_> = depths
            .iter()
            .copied()
            .map(|depth| match depth {
                SampleBitDepth::Integer { bits_per_sample } => (bits_per_sample, 0),
                SampleBitDepth::Float {
                    bits_per_sample,
                    exponent_bits_per_sample,
                } => (bits_per_sample, exponent_bits_per_sample),
            })
            .collect();
        assert_eq!(
            declared_depths
                .iter()
                .map(|depth| depth.0)
                .collect::<Vec<_>>(),
            descriptor.bits_per_sample
        );
        assert_eq!(
            declared_depths
                .iter()
                .map(|depth| depth.1)
                .collect::<Vec<_>>(),
            descriptor.exp_bits_per_sample
        );
        let kinds: Vec<_> = image
            .extra_channels
            .iter()
            .map(|extra| match extra.channel_type {
                ExtraChannelTypeInventory::Alpha { .. } => "Alpha",
                ExtraChannelTypeInventory::SpotColour { .. } => "SpotColor",
                _ => panic!("unexpected official extra channel"),
            })
            .collect();
        assert_eq!(kinds, descriptor.extra_channel_type);
        // These cases have at most one alpha, immediately after the color samples.
        // Packed color comparison must never silently reorder an independent extra plane.
        assert!(kinds.is_empty() || kinds[0] == "Alpha");
        assert!(kinds.iter().skip(1).all(|&kind| kind != "Alpha"));
        assert_eq!(
            image.tone_mapping.intensity_target.to_f32(),
            descriptor.intensity_target
        );
        assert_eq!(image.tone_mapping.min_nits.to_f32(), descriptor.min_nits);
        assert_eq!(
            u32::from(image.tone_mapping.relative_to_max_display),
            descriptor.relative_to_max_display
        );
        assert_eq!(
            image.tone_mapping.linear_below.to_f32(),
            descriptor.linear_below
        );
        if let Some(original) = &descriptor.original_icc {
            let original_hash = &descriptor.sha256sums[original];
            let original_bytes = if original_hash == &descriptor.sha256sums["reference.icc"] {
                profile.clone()
            } else {
                std::fs::read(directory.join(original)).unwrap()
            };
            check(&original_bytes, original_hash);
            match self.original {
                OriginalProfile::Embedded => {
                    assert_eq!(
                        &*image.embedded_icc.as_ref().unwrap().profile,
                        original_bytes
                    );
                }
                OriginalProfile::Enumerated {
                    grayscale,
                    transfer,
                    intent,
                } => {
                    assert!(image.embedded_icc.is_none());
                    assert_eq!(
                        image.colour_encoding,
                        ColourEncodingInventory::Enumerated {
                            colour_space: if grayscale {
                                ColourSpaceInventory::Grey
                            } else {
                                ColourSpaceInventory::Rgb
                            },
                            white_point: WhitePointInventory::D65,
                            primaries: PrimariesInventory::Srgb,
                            transfer_function: transfer,
                            rendering_intent: intent,
                        }
                    );
                }
                OriginalProfile::Unspecified => panic!("unclassified original profile"),
            }
        } else {
            assert!(matches!(self.original, OriginalProfile::Unspecified));
        }
        Reference {
            name: alternate.map_or(self.name, |a| a.name),
            input,
            profile,
            descriptor,
            width: shape[2],
            height: shape[1],
            channels: shape[3],
            colors,
            color: self.color,
            depths,
            pixels,
            animation: image.animation,
        }
    }
}
