use jxl_gpu_bitstream::{CodestreamInventory, FrameEncoding};
use std::path::{Path, PathBuf};

use super::reference::{Arithmetic, Plane, Sample};

pub(super) struct Case {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub color_factor: u32,
    pub extra_factor: u32,
}

fn directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/extra_upsampling")
}

pub(super) fn cases() -> Vec<Case> {
    let manifest = std::fs::read_to_string(directory().join("manifest.tsv")).unwrap();
    let cases: Vec<_> = manifest
        .lines()
        .map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 5);
            Case {
                name: fields[0].into(),
                width: fields[1].parse().unwrap(),
                height: fields[2].parse().unwrap(),
                color_factor: fields[3].parse().unwrap(),
                extra_factor: fields[4].parse().unwrap(),
            }
        })
        .collect();
    assert_eq!(cases.len(), 68);
    cases
}

impl Case {
    pub fn load(
        &self,
        arithmetic: Arithmetic,
        custom_weights: bool,
    ) -> (Vec<u8>, CodestreamInventory, Vec<Plane>) {
        let mut data = std::fs::read(directory().join(format!("{}.jxl", self.name))).unwrap();
        if custom_weights {
            data = jxl_test_support::corpus::with_custom_upsampling_weights(&data);
        }
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert_eq!(inventory.frames.len(), 1);
        let frame = &inventory.frames[0];
        let image = &inventory.image_header;
        assert_eq!(frame.encoding, FrameEncoding::Modular);
        assert_eq!(
            (image.width as usize, image.height as usize),
            (self.width, self.height)
        );
        assert!(!image.xyb_encoded);
        assert_eq!(frame.upsampling, self.color_factor);
        assert_eq!(frame.extra_channel_upsampling, [self.extra_factor; 4]);
        assert_eq!(image.extra_channels.len(), 4);
        assert!(
            image
                .extra_channels
                .iter()
                .all(|extra| extra.dimension_shift == 3)
        );
        let coded = std::fs::read(directory().join(format!("{}.coded", self.name))).unwrap();
        let (words, tail) = coded.as_chunks::<4>();
        assert!(tail.is_empty());
        let mut words = words.iter().map(|word| u32::from_le_bytes(*word));
        let mut expected = Vec::new();
        for channel in 0..7 {
            let factor = if channel < 3 {
                self.color_factor
            } else {
                self.extra_factor
            };
            let depth = if channel < 3 {
                image.bit_depth
            } else {
                image.extra_channels[channel - 3].bit_depth
            };
            let width = self.width.div_ceil(factor as usize);
            let height = self.height.div_ceil(factor as usize);
            let samples = (0..width * height)
                .map(|index| {
                    let word = words.next().unwrap();
                    let x = index % width;
                    let y = index / width;
                    let code = (x * 311 + y * 997 + x * y * 53 + channel * 4013) % 65521;
                    let source_word = match depth {
                        jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample } => {
                            code as u32 & ((1_u32 << bits_per_sample) - 1)
                        }
                        jxl_gpu_bitstream::SampleBitDepth::Float { .. } => {
                            (((code % 97) as i32 - 43) as f32 / 16.0).to_bits()
                        }
                    };
                    assert_eq!(
                        word, source_word,
                        "{} coded channel {channel} sample {index}",
                        self.name
                    );
                    Sample::decoded(word, depth, arithmetic)
                })
                .collect();
            expected.push(
                Plane {
                    width,
                    height,
                    samples,
                }
                .reconstruct(
                    factor,
                    self.width,
                    self.height,
                    &image.upsampling_weights,
                    arithmetic,
                ),
            );
        }
        assert!(words.next().is_none());
        (data, inventory, expected)
    }
}
