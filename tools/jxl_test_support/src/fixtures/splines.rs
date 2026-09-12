//! Quantized spline scenarios and per-physical-frame insertion independent of GPU decoding.

use super::frame_features::{copy_bits, dictionary, feature_header, packet_frame_prefix};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scenario {
    Main,
    PatchChain,
    LowFrequency,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceSource {
    Native,
    JxlOxide,
}

pub const PROGRESSIVE_SOURCES: &[(&str, &str)] = &[
    ("vardct", "vardct_extras_rgba_progressive"),
    ("gray", "testsrc_vardct_gray_progressive"),
    ("modular", "modular_passes/squeeze"),
    ("float", "floating/vardct_extras_float_squeeze"),
];

pub fn progressive(source: &[u8]) -> Vec<u8> {
    let info = jxl_gpu_bitstream::parse(source, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(info.frames.len(), 1);
    let frame = &info.frames[0];
    assert!(frame.num_passes > 1);
    let spline = values(frame, 8);
    let patches = super::patches::values(frame, info.image_header.extra_channels.len(), 16);
    super::frame_features::assemble_frames(&[
        super::frame_features::Frame {
            codestream: source,
            reference: Some((3, true)),
            patches: None,
            splines: Some(&spline),
        },
        super::frame_features::Frame {
            codestream: source,
            reference: None,
            patches: Some(&patches),
            splines: Some(&spline),
        },
    ])
}

pub struct Case {
    pub name: String,
    pub source: String,
    pub scenario: Scenario,
    pub adjustment: i32,
    pub reference: ReferenceSource,
}

pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for mode in ["modular", "vardct"] {
        for original_rgb in [false, true] {
            for factor in [1, 2, 4, 8] {
                let color = if original_rgb { "rgb_" } else { "" };
                let suffix = if factor == 1 {
                    if mode == "modular" && original_rgb {
                        "group256".to_owned()
                    } else {
                        "257x17".to_owned()
                    }
                } else {
                    format!("up{factor}")
                };
                for (scenario, variant, adjustment) in [
                    (Scenario::Main, "", 8),
                    (Scenario::PatchChain, "_chain", -8),
                ] {
                    cases.push(Case {
                        name: format!("{mode}_{color}{suffix}{variant}"),
                        source: format!("noise/{mode}_{color}{suffix}"),
                        scenario,
                        adjustment,
                        reference: ReferenceSource::Native,
                    });
                }
            }
        }
        for factor in [2, 4, 8] {
            for (scenario, extension, prefix) in [
                (Scenario::Main, ".lf1", "extras"),
                (Scenario::LowFrequency, "", "lf"),
            ] {
                cases.push(Case {
                    name: format!("{mode}_{prefix}_up{factor}"),
                    source: format!("lf_patch_features/equal_up{factor}_{mode}{extension}"),
                    scenario,
                    adjustment: 0,
                    reference: ReferenceSource::Native,
                });
            }
        }
        for nested in [false, true] {
            let prefix = if nested { "lf_nested" } else { "lf" };
            cases.push(Case {
                name: format!("{mode}_{prefix}_noise"),
                source: format!("noise/{prefix}_{mode}_gab1"),
                scenario: Scenario::LowFrequency,
                adjustment: 8,
                reference: ReferenceSource::Native,
            });
        }
    }
    for (name, source, scenario) in [
        (
            "base_correlation",
            "noise/vardct_base_correlation",
            Scenario::Main,
        ),
        (
            "lf_correlation",
            "noise/vardct_lf_correlation",
            Scenario::Main,
        ),
        ("modular_early_extras", "extras_shifted8", Scenario::Main),
        (
            "vardct_early_extras",
            "vardct_extras_shifted8",
            Scenario::Main,
        ),
        (
            "lf_consumer",
            "testsrc_vardct_progressive_dc_ac",
            Scenario::Main,
        ),
        (
            "lf_consumer_chain",
            "testsrc_vardct_progressive_dc_ac",
            Scenario::PatchChain,
        ),
        (
            "custom_up8",
            "testsrc_vardct_upsample_8_custom",
            Scenario::Main,
        ),
    ] {
        cases.push(Case {
            name: name.into(),
            source: source.into(),
            scenario,
            adjustment: -8,
            reference: ReferenceSource::Native,
        });
    }
    // libjxl 0.12 and Rust jxl share a vertical-subsampling restoration defect already
    // audited by noise_combinations. Preserve that explicit independent-oracle selection.
    for (sampling, reference) in [
        ("422", ReferenceSource::Native),
        ("440", ReferenceSource::JxlOxide),
        ("420", ReferenceSource::JxlOxide),
    ] {
        cases.push(Case {
            name: format!("jpeg_{sampling}"),
            source: format!("noise/jpeg_{sampling}_gab_epf3"),
            scenario: Scenario::Main,
            adjustment: 0,
            reference,
        });
    }
    cases
}

impl Case {
    pub fn assemble(&self, source: &[u8]) -> Vec<u8> {
        let info = jxl_gpu_bitstream::parse(source, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let main = info.frames.last().unwrap();
        match self.scenario {
            Scenario::Main | Scenario::LowFrequency => {
                let programs: Vec<_> = info
                    .frames
                    .iter()
                    .enumerate()
                    .map(|(index, frame)| {
                        (if self.scenario == Scenario::Main {
                            index + 1 == info.frames.len()
                        } else {
                            frame.lf_level != 0
                        })
                        .then(|| values(frame, self.adjustment))
                    })
                    .collect();
                insert(
                    source,
                    &programs.iter().map(Option::as_deref).collect::<Vec<_>>(),
                )
            }
            Scenario::PatchChain => {
                let spline = values(main, self.adjustment);
                let mut coded = main.clone();
                (coded.width, coded.height) = main.color_sample_extent().unwrap();
                let patches =
                    super::patches::values(&coded, info.image_header.extra_channels.len(), 16);
                super::frame_features::assemble_frames(&[
                    super::frame_features::Frame {
                        codestream: source,
                        reference: Some((3, true)),
                        patches: None,
                        splines: Some(&spline),
                    },
                    super::frame_features::Frame {
                        codestream: source,
                        reference: Some((3, true)),
                        patches: Some(&patches),
                        splines: Some(&spline),
                    },
                    super::frame_features::Frame {
                        codestream: source,
                        reference: None,
                        patches: Some(&patches),
                        splines: Some(&spline),
                    },
                ])
            }
        }
    }
}
use jxl_gpu_bitstream::{BitWriter, FrameInventory};

fn pack(value: i32) -> u32 {
    ((value as u32) << 1) ^ (value >> 31) as u32
}

/// Two curved splines, including negative double deltas, correlated color, changing width and
/// both signs of DCT coefficients. The second curve leaves and re-enters the coded frame.
pub fn values(frame: &FrameInventory, adjustment: i32) -> Vec<u32> {
    let (width, height) = frame.color_sample_extent().unwrap();
    let point_limit = u64::from(width) * u64::from(height) / 2;
    assert!(
        point_limit > 0,
        "fixture has no legal spline control-point budget"
    );
    let w = width.max(4) as i32;
    let h = height.max(4) as i32;
    let mut curves = vec![
        vec![[1, 1], [w / 2, h - 1], [w - 1, h / 3]],
        vec![[w - 1, h / 2], [w + 2, -2], [-3, h + 1], [1, h / 2]],
    ];
    // Very small reduced LF images legally admit only one short or point-like spline.
    if point_limit < 7 {
        curves.truncate(1);
        curves[0].truncate(point_limit as usize);
    }
    let mut result = vec![curves.len() as u32 - 1];
    let mut previous = [0; 2];
    for (index, curve) in curves.iter().enumerate() {
        for axis in 0..2 {
            result.push(if index == 0 {
                curve[0][axis] as u32
            } else {
                pack(curve[0][axis] - previous[axis])
            });
        }
        previous = curve[0];
    }
    result.push(pack(adjustment));
    for (curve, points) in curves.into_iter().enumerate() {
        result.push(points.len() as u32 - 1);
        let mut velocity = [0; 2];
        for pair in points.windows(2) {
            for (axis, previous) in velocity.iter_mut().enumerate() {
                let delta = pair[1][axis] - pair[0][axis];
                result.push(pack(delta - *previous));
                *previous = delta;
            }
        }
        let mut coefficients = [0i32; 128];
        for (i, value) in [
            (0, 9),
            (1, -3),
            (32, 2),
            (34, 1),
            (64, -2),
            (67, 1),
            (96, 4),
            (97, -1),
        ] {
            coefficients[i] = if curve == 0 { value } else { -value };
        }
        result.extend(coefficients.into_iter().map(pack));
    }
    result
}

/// Preserve each physical header and body, adding splines only at explicitly selected frames.
pub fn insert(bytes: &[u8], programs: &[Option<&[u32]>]) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(bytes, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    assert_eq!(programs.len(), inventory.frames.len());
    let bytes = parsed.codestream();
    let mut output = bytes[..inventory.frames[0].header_bits.offset as usize / 8].to_vec();
    for (frame, &program) in inventory.frames.iter().zip(programs) {
        if let Some(program) = program {
            assert_eq!(
                frame.flags & 0x12,
                0,
                "insertion requires a feature-free prefix"
            );
            output.extend(packet_frame_prefix(
                bytes,
                frame,
                feature_header(bytes, frame, frame.flags | 16),
                Some(&dictionary(program)),
            ));
        } else {
            let mut header = BitWriter::new();
            copy_bits(
                &mut header,
                bytes,
                frame.header_bits.offset,
                frame.header_bits.end().unwrap(),
            );
            output.extend(packet_frame_prefix(
                bytes,
                frame,
                jxl_wgpu_encode::BitFragment::new(header.as_bytes().to_vec(), header.bit_len())
                    .unwrap(),
                None,
            ));
        }
    }
    output
}
