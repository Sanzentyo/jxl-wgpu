//! Component-domain patch references with explicit source, oracle and control metadata.
use super::{frame_features, noise, patches};
use jxl_gpu_bitstream::CodestreamInventory;

mod jpeg;
mod mixed;
mod modular_ycbcr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Jpeg,
    Mixed,
    ModularYcbcr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    Empty,
    AllModes,
    Overwrite,
    PaddedEdge,
    JpegPadding,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReferenceSource {
    NativeLinear,
    NativeSrgb,
    JxlOxideLinear,
    /// Independently expanded 4:4:4 sources traverse the complete feature sequence and
    /// isolate libjxl's vertically subsampled restoration defect.
    ExpandedSrgb {
        sources: [Source; 2],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Noise {
    Preserve,
    Zero,
    Inject,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorErrorScale {
    Component,
    /// Unclamped alpha can produce large, opposing YCbCr components. Bound the RGB vector
    /// instead of dividing by a single component which may cancel to almost zero.
    PixelRgb,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    pub name: String,
    pub noise: Noise,
}

impl Source {
    fn encode(&self) -> Vec<u8> {
        let data = crate::offline::unhex(
            &std::fs::read_to_string(
                crate::decoder_directory().join(format!("test-data/{}.jxl.hex", self.name)),
            )
            .unwrap(),
        );
        match self.noise {
            Noise::Preserve => data,
            Noise::Inject => frame_features::with_noise(&data, false),
            Noise::Zero => {
                let info = inventory(&data);
                let data = if info.frames.iter().any(|frame| frame.flags & 1 != 0) {
                    data
                } else {
                    frame_features::with_noise(&data, false)
                };
                noise::zero_noise(&data, &inventory(&data), None)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Case {
    pub name: String,
    pub sources: [Source; 2],
    pub family: Family,
    pub pattern: Pattern,
    pub reference_source: ReferenceSource,
    pub linear_tolerance: f32,
    /// Also freeze and directly compare native sRGB, without deriving it from GPU output.
    pub encoded_tolerance: Option<f32>,
    pub color_error_scale: ColorErrorScale,
    /// Cases whose final reference must differ, such as an empty dictionary or zero noise.
    pub controls: Vec<String>,
}

pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    jpeg::extend(&mut cases);
    mixed::extend(&mut cases);
    modular_ycbcr::extend(&mut cases);
    cases
}

impl Case {
    fn new(name: String, sources: [String; 2], family: Family, pattern: Pattern) -> Self {
        Self {
            name,
            sources: sources.map(|name| Source {
                name,
                noise: Noise::Preserve,
            }),
            family,
            pattern,
            reference_source: ReferenceSource::NativeLinear,
            linear_tolerance: 1.0 / 1024.0,
            encoded_tolerance: None,
            color_error_scale: ColorErrorScale::Component,
            controls: Vec::new(),
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        self.encode_sources(&self.sources, false)
    }

    /// jxl-oxide 0.12.6 rejects implicit alpha in patch modes 4–7. The reference generator
    /// proves this arithmetic equivalent matches the original stream bit-for-bit in libjxl.
    pub fn encode_arithmetic_reference(&self) -> Vec<u8> {
        assert_eq!(self.reference_source, ReferenceSource::JxlOxideLinear);
        assert_eq!(self.pattern, Pattern::AllModes);
        self.encode_sources(&self.sources, true)
    }

    pub fn encode_expanded_reference(&self) -> Vec<u8> {
        self.encode_sources(self.expanded_sources(), false)
    }

    pub fn encode_expanded_arithmetic_reference(&self) -> Vec<u8> {
        self.encode_sources(self.expanded_sources(), true)
    }

    fn expanded_sources(&self) -> &[Source; 2] {
        let ReferenceSource::ExpandedSrgb { sources } = &self.reference_source else {
            panic!("case does not declare expanded reference sources");
        };
        sources
    }

    fn encode_sources(&self, sources: &[Source; 2], arithmetic: bool) -> Vec<u8> {
        let sources = sources.each_ref().map(Source::encode);
        let dictionaries = sources.each_ref().map(|data| {
            let inventory = inventory(data);
            let frame = inventory.frames.last().unwrap();
            let mut coded = frame.clone();
            (coded.width, coded.height) = frame.color_sample_extent().unwrap();
            if self.pattern == Pattern::PaddedEdge {
                // A two-pixel patch crosses both visible edges, but stays inside the VarDCT
                // padded reconstruction surface used before filters and presentation cropping.
                let mut values = vec![1, 3, 0, 0, 1, 1, 0, coded.width - 1, coded.height - 1];
                values.extend(std::iter::repeat_n(
                    1,
                    1 + inventory.image_header.extra_channels.len(),
                ));
                values
            } else if self.pattern == Pattern::JpegPadding {
                assert!(frame.do_ycbcr && inventory.image_header.extra_channels.is_empty());
                // Independent raw-factor table, including equal nonzero triples. Put the
                // final 2x2 patch exactly against the padded corner, beyond the visible image.
                let h = frame
                    .jpeg_upsampling
                    .map(|v| [1, 2, 2, 1][v as usize])
                    .into_iter()
                    .max()
                    .unwrap()
                    * 8;
                let v = frame
                    .jpeg_upsampling
                    .map(|v| [1, 2, 1, 2][v as usize])
                    .into_iter()
                    .max()
                    .unwrap()
                    * 8;
                vec![
                    1,
                    3,
                    0,
                    0,
                    1,
                    1,
                    0,
                    coded.width.div_ceil(h) * h - 2,
                    coded.height.div_ceil(v) * v - 2,
                    1,
                ]
            } else if arithmetic && self.pattern != Pattern::Empty {
                assert!(inventory.image_header.extra_channels.is_empty());
                patches::arithmetic_values(&coded, 16)
            } else {
                patches::values(
                    &coded,
                    inventory.image_header.extra_channels.len(),
                    if self.pattern == Pattern::Empty {
                        0
                    } else {
                        16
                    },
                )
            }
        });
        if self.pattern == Pattern::Overwrite {
            // Alternate producers and overwrite all four slots; the final read of slot zero
            // must observe its second completed version, including patches and noise.
            let dictionaries: Vec<_> = (0..5)
                .map(|i| {
                    let mut values = dictionaries[(i + 1) % 2].clone();
                    values[1] = (i % 4) as u32;
                    values
                })
                .collect();
            let mut frames = vec![frame_features::Frame {
                codestream: &sources[0],
                reference: Some((0, true)),
                patches: None,
                splines: None,
            }];
            for (i, values) in dictionaries.iter().enumerate() {
                frames.push(frame_features::Frame {
                    codestream: &sources[(i + 1) % 2],
                    reference: (i < 4).then_some((((i + 1) % 4) as u32, true)),
                    patches: Some(values),
                    splines: None,
                });
            }
            frame_features::assemble_frames(&frames)
        } else {
            frame_features::assemble_frames(&[
                frame_features::Frame {
                    codestream: &sources[0],
                    reference: Some((3, true)),
                    patches: None,
                    splines: None,
                },
                frame_features::Frame {
                    codestream: &sources[1],
                    reference: None,
                    patches: Some(&dictionaries[1]),
                    splines: None,
                },
            ])
        }
    }
}

fn inventory(bytes: &[u8]) -> CodestreamInventory {
    jxl_gpu_bitstream::parse(bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
}
