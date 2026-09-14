use super::{inventory, lut::reference, output::frames_bytes};

mod compare;
use jxl_gpu_formats::{ColorSample, ColorStorage, ImageLayout, PixelFormat};
use jxl_gpu_protocol::{
    Extent2d,
    icc::{IccProfile, IccRenderingIntent},
};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuOutputRequest, SpotColorPolicy};
use serde::Deserialize;
use std::{
    num::{NonZeroU64, NonZeroUsize},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
struct Manifest {
    width: u32,
    height: u32,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    source: String,
    target: String,
    channels: usize,
    target_channels: usize,
    frames: usize,
    mode: u32,
    black: usize,
}

fn directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/device_output")
}
fn profiles() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../jxl_wgpu/test-data/icc/lut")
}
fn profile(name: &str) -> IccProfile {
    IccProfile::parse(
        std::fs::read(profiles().join(format!("{name}.icc")))
            .unwrap()
            .into(),
        Default::default(),
    )
    .unwrap()
}

impl Case {
    fn inputs(&self, extent: Extent2d) -> (Vec<u8>, Vec<f32>) {
        let folder = if self.channels == 4 {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/cmyk/generated")
        } else {
            profiles().join("decoder")
        };
        let data = std::fs::read(folder.join(format!("{}.jxl", self.name))).unwrap();
        let image = inventory(&data);
        assert_eq!(
            (image.image_header.width, image.image_header.height),
            (extent.width, extent.height)
        );
        assert_eq!(image.frames.len(), self.frames);
        assert!(!image.image_header.xyb_encoded);
        assert_eq!(
            image
                .image_header
                .embedded_icc
                .as_ref()
                .unwrap()
                .profile
                .as_ref(),
            profile(&self.source).bytes().as_ref()
        );
        for frame in &image.frames {
            assert_eq!(
                frame.encoding,
                if self.mode == 0 {
                    jxl_gpu_bitstream::FrameEncoding::Modular
                } else {
                    jxl_gpu_bitstream::FrameEncoding::VarDct
                }
            );
            assert_eq!(frame.do_ycbcr, self.mode == 2);
        }
        let extension = if self.channels == 4 { "f32" } else { "f32le" };
        let bytes = std::fs::read(folder.join(format!("{}.{extension}", self.name))).unwrap();
        let (words, tail) = bytes.as_chunks::<4>();
        assert!(tail.is_empty());
        let original = words
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect::<Vec<_>>();
        assert_eq!(
            original.len(),
            extent.area().unwrap() * self.frames * self.source_stride()
        );
        (data, original)
    }
    fn source_stride(&self) -> usize {
        if self.channels == 4 {
            6
        } else {
            self.channels + 1
        }
    }
    fn alpha(&self, original: &[f32], pixel: usize) -> f32 {
        original[pixel * self.source_stride() + if self.channels == 4 { 4 } else { self.channels }]
    }
}

#[derive(Clone, Copy)]
struct Packing {
    sample: ColorSample,
    storage: ColorStorage,
    alpha: bool,
    associated: bool,
}
const PACKINGS: [Packing; 4] = [
    Packing {
        sample: ColorSample::F32,
        storage: ColorStorage::Interleaved,
        alpha: true,
        associated: false,
    },
    Packing {
        sample: ColorSample::F32,
        storage: ColorStorage::Planar,
        alpha: true,
        associated: true,
    },
    Packing {
        sample: ColorSample::U8,
        storage: ColorStorage::Interleaved,
        alpha: false,
        associated: false,
    },
    Packing {
        sample: ColorSample::U8,
        storage: ColorStorage::Planar,
        alpha: true,
        associated: true,
    },
];

impl Packing {
    fn format(self, profile: IccProfile) -> PixelFormat {
        let mut format =
            PixelFormat::icc_device(profile, self.sample, self.storage, self.alpha).unwrap();
        // Physical order is deliberately the reverse of profile/alpha order.
        if self.storage == ColorStorage::Planar {
            format.planes.reverse();
        } else {
            format.planes[0].words.reverse();
        }
        format
    }
    fn policy(self) -> AlphaOutputPolicy {
        if self.associated {
            AlphaOutputPolicy::Associated
        } else {
            AlphaOutputPolicy::Preserve
        }
    }
}

#[test]
fn requested_device_profiles_match_independent_native_intervals_after_composition() {
    verify(false);
}

#[test]
fn original_device_output_preserves_profile_components_with_independent_alpha() {
    verify(true);
}

fn verify(passthrough: bool) {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(directory().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest.cases.len(), 42);
    let extent = Extent2d::new(manifest.width, manifest.height);
    let count = extent.area().unwrap();
    assert_eq!(count, 153);
    let mut components = 0;
    let mut presentations = 0;
    for case in manifest.cases {
        eprintln!("device output {} passthrough={passthrough}", case.name);
        let (data, original) = case.inputs(extent);
        let target = profile(if passthrough {
            &case.source
        } else {
            &case.target
        });
        let channels = if passthrough {
            case.channels
        } else {
            case.target_channels
        };
        assert_eq!(
            target.header().device_space.device_channels(),
            Some(channels as u8)
        );
        let intents = if passthrough {
            &[IccRenderingIntent::Relative][..]
        } else {
            &[
                IccRenderingIntent::Perceptual,
                IccRenderingIntent::Relative,
                IccRenderingIntent::Saturation,
                IccRenderingIntent::Absolute,
            ]
        };
        for &spots in if !passthrough && case.channels == 4 {
            &[false, true][..]
        } else {
            &[false]
        } {
            for &intent in intents {
                let expected = if passthrough {
                    Vec::new()
                } else {
                    let expected = reference(
                        &directory(),
                        &format!("{}_{}", case.name, u32::from(spots)),
                        intent,
                    );
                    assert_eq!(expected.len(), count * case.frames * channels);
                    expected
                };
                let oracle = compare::Oracle {
                    case: &case,
                    original: &original,
                    expected: &expected,
                    passthrough,
                    intent,
                    spots,
                };
                for packing in PACKINGS {
                    let format = packing.format(target.clone());
                    let layout = ImageLayout::packed(extent, format.clone()).unwrap();
                    let mut whole = Vec::new();
                    for limit in [None, NonZeroU64::new(256)] {
                        let request = GpuOutputRequest::color(format.clone())
                            .unwrap()
                            .with_max_frame_slots(NonZeroUsize::new(case.frames).unwrap())
                            .with_alpha_output_policy(packing.policy())
                            .with_spot_color_policy(if spots {
                                SpotColorPolicy::Render
                            } else {
                                SpotColorPolicy::Preserve
                            })
                            .with_icc_rendering_intent(intent);
                        let frames = frames_bytes(&backend, &data, request, limit);
                        assert_eq!(frames.len(), case.frames);
                        for (frame, bytes) in frames.iter().enumerate() {
                            components += oracle.assert_frame(&layout, packing, frame, bytes);
                            presentations += 1;
                        }
                        if limit.is_none() {
                            whole = frames;
                        } else {
                            assert_eq!(frames, whole, "{} bounded output", case.name);
                        }
                    }
                }
            }
        }
    }
    assert_eq!(presentations, if passthrough { 624 } else { 4224 });
    assert_eq!(components, if passthrough { 323136 } else { 3231360 });
    eprintln!(
        "device output passthrough={passthrough}: {presentations} presentations, {components} checked components"
    );
}
