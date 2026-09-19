use std::{collections::BTreeMap, io::Read, path::Path};

use jxl_gpu_bitstream::{ExtraChannelTypeInventory, SampleBitDepth};
use sha2::{Digest, Sha256};

pub(super) struct Case {
    pub name: &'static str,
    pub input_sha256: &'static str,
    pub descriptor_sha256: &'static str,
}

#[derive(serde::Deserialize)]
pub(super) struct Frame {
    pub name: String,
    pub rms_error: f64,
    pub peak_error: f64,
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
    pub pixels: Vec<f32>,
}

fn check(bytes: &[u8], expected: &str) {
    assert_eq!(
        Sha256::digest(bytes).as_slice(),
        jxl_test_support::offline::hex::unhex(expected)
    );
}

impl Case {
    pub fn load(&self) -> Reference {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test-data/official_stills")
            .join(self.name);
        let input = std::fs::read(directory.join("input.jxl")).unwrap();
        check(&input, self.input_sha256);
        let descriptor = std::fs::read(directory.join("test.json")).unwrap();
        // The descriptor hash pins the published bounds as well as all referenced object hashes.
        check(&descriptor, self.descriptor_sha256);
        let descriptor: Descriptor = serde_json::from_slice(&descriptor).unwrap();
        assert_eq!(descriptor.frames.len(), 1);
        let profile = std::fs::read(directory.join("reference.icc")).unwrap();
        check(&profile, &descriptor.sha256sums["reference.icc"]);
        let mut bytes = Vec::new();
        flate2::read::GzDecoder::new(
            std::fs::File::open(directory.join("reference.npy.gz")).unwrap(),
        )
        .read_to_end(&mut bytes)
        .unwrap();
        check(&bytes, &descriptor.sha256sums["reference_image.npy"]);
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
        assert_eq!(shape[0], 1);
        let (words, tail) = bytes[offset..].as_chunks::<4>();
        assert!(tail.is_empty());
        assert_eq!(words.len(), shape.iter().product::<usize>());
        let pixels = words.iter().copied().map(f32::from_le_bytes).collect();
        let info = jxl_gpu_bitstream::parse(&input, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &info.image_header;
        assert!(!image.grayscale);
        assert!(image.animation.is_none());
        assert_eq!(shape[3], 3 + image.extra_channels.len());
        let depths: Vec<_> = std::iter::once(image.bit_depth)
            .chain(image.extra_channels.iter().map(|extra| extra.bit_depth))
            .map(|depth| match depth {
                SampleBitDepth::Integer { bits_per_sample } => (bits_per_sample, 0),
                SampleBitDepth::Float {
                    bits_per_sample,
                    exponent_bits_per_sample,
                } => (bits_per_sample, exponent_bits_per_sample),
            })
            .collect();
        assert_eq!(
            depths.iter().map(|depth| depth.0).collect::<Vec<_>>(),
            descriptor.bits_per_sample
        );
        assert_eq!(
            depths.iter().map(|depth| depth.1).collect::<Vec<_>>(),
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
            let embedded = &image.embedded_icc.as_ref().unwrap().profile;
            check(embedded, &descriptor.sha256sums[original]);
            assert_eq!(&**embedded, profile);
        }
        Reference {
            name: self.name,
            input,
            profile,
            descriptor,
            width: shape[2],
            height: shape[1],
            channels: shape[3],
            pixels,
        }
    }
}
