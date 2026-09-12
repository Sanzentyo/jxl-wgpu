//! Component-domain patch references across coding modes and JPEG sampling layouts.
use super::{frame_features, noise, patches};
use jxl_gpu_bitstream::CodestreamInventory;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    Empty,
    AllModes,
    Overwrite,
    PaddedEdge,
    JpegPadding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceSource {
    NativeLinear,
    NativeSrgb,
    JxlOxideLinear,
}

pub struct Case {
    pub name: String,
    pub sources: [String; 2],
    pub zero_noise: bool,
    pub pattern: Pattern,
}

pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for cb in 0..4 {
        for y in 0..4 {
            for cr in 0..4 {
                for zero_noise in [false, true] {
                    cases.push(Case {
                        name: format!("jpeg_{cb}{y}{cr}{}", if zero_noise { "_zero" } else { "" }),
                        // Different producer and consumer layouts prevent a matching stride
                        // mistake in both frames from hiding behind a same-source patch.
                        sources: [
                            format!("jpeg_sampling/odd_{cr}{cb}{y}"),
                            format!("jpeg_sampling/odd_{cb}{y}{cr}"),
                        ],
                        zero_noise,
                        pattern: Pattern::AllModes,
                    });
                }
                cases.push(Case {
                    name: format!("jpeg_{cb}{y}{cr}_padding"),
                    sources: [
                        format!("jpeg_sampling/odd_{cr}{cb}{y}"),
                        format!("jpeg_sampling/odd_{cb}{y}{cr}"),
                    ],
                    zero_noise: true,
                    pattern: Pattern::JpegPadding,
                });
            }
        }
    }
    for sampling in ["422", "440", "420"] {
        for filter in ["gab", "epf1", "gab_epf2", "gab_epf3"] {
            cases.push(Case {
                name: format!("jpeg_{sampling}_{filter}"),
                sources: [
                    format!("noise/jpeg_{sampling}"),
                    format!("noise/jpeg_{sampling}_{filter}"),
                ],
                zero_noise: false,
                pattern: Pattern::AllModes,
            });
        }
        for (suffix, pattern) in [("empty", Pattern::Empty), ("edge", Pattern::PaddedEdge)] {
            cases.push(Case {
                name: format!("jpeg_{sampling}_{suffix}"),
                sources: std::array::from_fn(|_| format!("noise/jpeg_{sampling}")),
                zero_noise: true,
                pattern,
            });
        }
    }
    for (name, sources) in [
        ("xyb", ["noise/modular_257x17", "noise/vardct_257x17"]),
        ("xyb_up2", ["noise/modular_up2", "noise/vardct_up2"]),
        ("xyb_up4", ["noise/modular_up4", "noise/vardct_up4"]),
        ("xyb_up8", ["noise/modular_up8", "noise/vardct_up8"]),
        (
            "rgb",
            ["noise/modular_rgb_group256", "noise/vardct_rgb_257x17"],
        ),
        ("rgb_up2", ["noise/modular_rgb_up2", "noise/vardct_rgb_up2"]),
        ("rgb_up4", ["noise/modular_rgb_up4", "noise/vardct_rgb_up4"]),
        ("rgb_up8", ["noise/modular_rgb_up8", "noise/vardct_rgb_up8"]),
        (
            "extras_up2",
            [
                "lf_patch_features/equal_up2_modular.lf1",
                "lf_patch_features/equal_up2_vardct.lf1",
            ],
        ),
        (
            "extras_up4",
            [
                "lf_patch_features/equal_up4_modular.lf1",
                "lf_patch_features/equal_up4_vardct.lf1",
            ],
        ),
        (
            "extras_up8",
            [
                "lf_patch_features/equal_up8_modular.lf1",
                "lf_patch_features/equal_up8_vardct.lf1",
            ],
        ),
        (
            "lf_up8",
            [
                "lf_patch_features/equal_up8_modular",
                "lf_patch_features/equal_up8_vardct",
            ],
        ),
        (
            "jpeg_modular",
            ["noise/modular_rgb_group256", "noise/jpeg_420"],
        ),
        ("jpeg_rgb", ["noise/vardct_rgb_257x17", "noise/jpeg_422"]),
    ] {
        for reverse in [false, true] {
            let sources = if reverse {
                [sources[1], sources[0]]
            } else {
                sources
            };
            for (suffix, pattern) in [("empty", Pattern::Empty), ("patches", Pattern::AllModes)] {
                cases.push(Case {
                    name: format!("mixed_{name}_{}_{suffix}", u32::from(reverse)),
                    sources: sources.map(str::to_owned),
                    zero_noise: false,
                    pattern,
                });
            }
            if matches!(
                name,
                "xyb" | "rgb" | "extras_up8" | "jpeg_modular" | "jpeg_rgb"
            ) {
                cases.push(Case {
                    name: format!("mixed_{name}_{}_overwrite", u32::from(reverse)),
                    sources: sources.map(str::to_owned),
                    zero_noise: false,
                    pattern: Pattern::Overwrite,
                });
            }
        }
    }
    cases
}

impl Case {
    pub fn reference_source(&self) -> ReferenceSource {
        match self.name.as_str() {
            // These component-domain crossings can produce sRGB values above two. libjxl's
            // CMS approximates the extended-range curve; freeze its unconverted RGB and
            // evaluate the specified transfer in f64 instead of relaxing precision bounds.
            name if name.starts_with("mixed_jpeg_") => ReferenceSource::NativeSrgb,
            // libjxl 0.12's fast renderer has a vertical-subsampling restoration defect.
            // The existing noise_combinations test audits this exception with scalar f64
            // Gaborish and jxl-oxide. Keep the same explicit selection for patched frames.
            "jpeg_440_gab" | "jpeg_420_gab" | "jpeg_440_gab_epf3" | "jpeg_420_gab_epf3" => {
                ReferenceSource::JxlOxideLinear
            }
            _ => ReferenceSource::NativeLinear,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        self.encode_with_arithmetic(false)
    }

    /// jxl-oxide 0.12.6 rejects implicit alpha in patch modes 4–7. The reference generator
    /// proves this arithmetic equivalent matches the original stream bit-for-bit in libjxl.
    pub fn encode_arithmetic_reference(&self) -> Vec<u8> {
        assert_eq!(self.reference_source(), ReferenceSource::JxlOxideLinear);
        assert_eq!(self.pattern, Pattern::AllModes);
        self.encode_with_arithmetic(true)
    }

    fn encode_with_arithmetic(&self, arithmetic: bool) -> Vec<u8> {
        let sources = self.sources.each_ref().map(|name| {
            let data = crate::offline::unhex(
                &std::fs::read_to_string(
                    crate::decoder_directory().join(format!("test-data/{name}.jxl.hex")),
                )
                .unwrap(),
            );
            if self.zero_noise {
                noise::zero_noise(&data, &inventory(&data), None)
            } else {
                data
            }
        });
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
            } else if arithmetic {
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
