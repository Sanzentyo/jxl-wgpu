//! ICC declarations, independent working coefficients, reconstruction and byte ownership.
use super::*;
use crate::{ColorChannels, ColorSampleFormat};
use jxl_gpu_formats::{
    ByteOrder, ColorSample, ColorSpecification, ColorStorage, PixelFormat, RgbChannelOrder,
};
use jxl_gpu_protocol::icc::IccProfile;
use jxl_test_support::{
    fixtures::source_layout::{Packing, Storage},
    gpu::planes::open_fragmented,
    oracles::{extra_channels, icc_profile::IccProfileOracle},
};
use jxl_wgpu_decode::WgpuDecodeEngine;

mod boundaries;
mod coefficients;
mod sequence;

pub(super) fn profile(gray: bool) -> IccProfile {
    IccProfile::parse(
        fs::read(
            jxl_test_support::fixtures::embedded_icc::directory().join(if gray {
                "gray.icc"
            } else {
                "rgb.icc"
            }),
        )
        .unwrap()
        .into(),
        Default::default(),
    )
    .unwrap()
}

pub(super) fn config(profile: &IccProfile, transform: VarDctColorTransform) -> VarDctConfig {
    VarDctConfig {
        sample_format: ColorSampleFormat::float(
            if profile.header().device_space.0 == *b"GRAY" {
                ColorChannels::Gray
            } else {
                ColorChannels::Rgb
            },
            32,
            8,
        )
        .unwrap(),
        source_color: ColorSpecification::Icc(profile.clone()),
        color_options: crate::ImageColorOptions {
            rendering_intent: profile.header().rendering_intent,
            ..Default::default()
        },
        ..precision::configuration(8, transform)
    }
}

fn words(extent: Extent2d, channels: ColorChannels) -> Vec<[u32; 3]> {
    (0..extent.area().unwrap())
        .map(|index| {
            let values: [u32; 3] =
                std::array::from_fn(|c| (((index * 23 + c * 31) % 199) as f32 / 198.0).to_bits());
            if channels == ColorChannels::Gray {
                [values[0]; 3]
            } else {
                values
            }
        })
        .collect()
}

fn upload(
    context: &WgpuContext,
    extent: Extent2d,
    config: &VarDctConfig,
    words: &[[u32; 3]],
    alternate: bool,
) -> BufferImageSource {
    let mut source = if config.sample_format.channels() == ColorChannels::Gray {
        gray::upload(
            context,
            extent,
            config.sample_format,
            &words.iter().map(|v| v[0]).collect::<Vec<_>>(),
            alternate,
        )
    } else {
        layouts::upload(
            context,
            extent,
            config.sample_format,
            words,
            Packing {
                storage: if alternate {
                    Storage::Split
                } else {
                    Storage::Packed
                },
                reversed: alternate,
                shifted: alternate,
            },
            if alternate {
                ByteOrder::Big
            } else {
                ByteOrder::Native
            },
        )
    };
    source.layout.format.color_spec = config.source_color.clone();
    source
}

fn compare(actual: &[f32], reference: &[f32]) {
    assert_eq!(actual.len(), reference.len());
    for (index, (&a, &r)) in actual.iter().zip(reference).enumerate() {
        assert!(
            a.is_finite() && r.is_finite() && (a - r).abs() <= 2e-4 * (1.0 + r.abs()),
            "ICC component {index}: {a} vs {r}"
        );
        let code = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        assert!(
            code(a).abs_diff(code(r)) <= 1,
            "ICC rounded component {index}"
        );
    }
}

struct Pixels {
    decoders: [GpuDecoder<WgpuDecodeEngine>; 2],
    readback: ImageReadbackPipeline,
}

impl Pixels {
    fn new(gpu: &WgpuBackend) -> Self {
        Self {
            decoders: [
                GpuDecoder::wgpu(gpu.clone()).unwrap(),
                GpuDecoder::new(
                    WgpuDecodeEngine::new(gpu.clone())
                        .unwrap()
                        .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
                ),
            ],
            readback: ImageReadbackPipeline::new(gpu),
        }
    }

    fn check(&self, bytes: &[u8], config: &VarDctConfig) {
        self.check_frames(bytes, config, 1);
    }

    fn check_frames(&self, bytes: &[u8], config: &VarDctConfig, frames: usize) {
        let xyb = config.color_transform == VarDctColorTransform::Xyb;
        let ColorSpecification::Icc(profile) = &config.source_color else {
            panic!("ICC test source")
        };
        let native = extra_channels::libjxl_output(
            bytes,
            &[if xyb { "--linear" } else { "--original-icc" }, "--no-cms"],
        )
        .expect("required native ICC output oracle");
        let (rust, actual_profile) = extra_channels::rust_frame_planes_with_profile(bytes);
        assert_eq!(rust.len(), frames);
        assert!(rust.iter().all(|frame| frame.1.is_empty()));
        if xyb {
            assert!(
                actual_profile
                    == jxl::api::JxlColorProfile::Simple(jxl::api::JxlColorEncoding::linear_srgb(
                        config.sample_format.channels() == ColorChannels::Gray,
                    ))
            );
        }
        if !xyb && frames > 1 {
            // Rust jxl reverses the clamped-Multiply operand (the existing animation
            // corpus records this defect). Retain its initial Replace comparison and
            // use independent jxl-oxide for every composed presentation, at the same bound.
            compare(&rust[0].0, &native[..rust[0].0.len()]);
            let mut oxide = jxl_oxide::JxlImage::read_with_defaults(bytes).unwrap();
            oxide.request_icc(profile.bytes()).unwrap();
            assert_eq!(oxide.num_loaded_keyframes(), frames);
            let mut composed = Vec::new();
            for i in 0..frames {
                let render = oxide.render_frame(i).unwrap().image_all_channels();
                let channels = render.channels();
                for p in render.buf().chunks_exact(channels) {
                    if channels == 1 {
                        composed.extend([p[0], p[0], p[0], 1.0]);
                    } else {
                        assert_eq!(channels, 3);
                        composed.extend([p[0], p[1], p[2], 1.0]);
                    }
                }
            }
            compare(&composed, &native);
        } else {
            let rust: Vec<_> = rust.into_iter().flat_map(|frame| frame.0).collect();
            compare(&rust, &native);
        }
        let format = if xyb {
            let mut color = vardct_rgb8_format().color_spec;
            let ColorSpecification::Defined(ref mut spec) = color else {
                unreachable!()
            };
            spec.transfer = jxl_gpu_formats::TransferFunction::Linear;
            PixelFormat::rgb_f32(RgbChannelOrder::Rgb, false, color)
        } else {
            PixelFormat::icc_device(
                profile.clone(),
                ColorSample::F32,
                ColorStorage::Interleaved,
                false,
            )
            .unwrap()
        };
        let count = if xyb {
            3
        } else {
            config.sample_format.channels().count() as usize
        };
        let expected: Vec<_> = native
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| p[..count].iter().copied())
            .collect();
        let mut whole = None;
        for (decoder, fragmented) in self.decoders.iter().zip([false, true]) {
            let request = GpuOutputRequest::color(format.clone()).unwrap();
            let mut session = if fragmented {
                open_fragmented(decoder, bytes, request)
            } else {
                decoder.open(bytes, request).unwrap()
            };
            let mut actual = Vec::new();
            let mut count = 0;
            while let Some(image) = session.next_frame().unwrap() {
                let read = self
                    .readback
                    .submit(image.output())
                    .unwrap()
                    .wait()
                    .unwrap();
                actual.extend(extra_channels::floats(&read.frame.outputs[0].bytes));
                count += 1;
            }
            assert_eq!(count, frames);
            compare(&actual, &expected);
            if let Some(whole) = &whole {
                assert_eq!(&actual, whole);
            } else {
                whole = Some(actual);
            }
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}

#[test]
fn icc_sources_preserve_profiles_and_match_independent_pixels_on_every_topology() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let pixels = Pixels::new(&gpu);
    let profiles = IccProfileOracle::compile();
    for gray in [false, true] {
        let profile = profile(gray);
        for transform in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
            let config = config(&profile, transform);
            for (topology, extent) in [
                (layouts::Topology::Single, Extent2d::new(8, 8)),
                (layouts::Topology::Map, Extent2d::new(25, 17)),
                (layouts::Topology::Tiled, Extent2d::new(259, 3)),
            ] {
                let mut config = config.clone();
                if matches!(topology, layouts::Topology::Tiled) {
                    config.group_order = crate::VarDctGroupOrder::saliency_first();
                }
                let backend = topology.backend(&context, extent, &config);
                let input = words(extent, config.sample_format.channels());
                let source = upload(&context, extent, &config, &input, false);
                let bytes = layouts::encode(&context, &backend, &config, source);
                assert_eq!(
                    bytes,
                    layouts::encode(
                        &context,
                        &backend,
                        &config,
                        upload(&context, extent, &config, &input, true)
                    )
                );
                assert_eq!(profiles.read(&bytes).profile, profile.bytes().as_ref());
                let device = PixelFormat::icc_device(
                    profile.clone(),
                    ColorSample::F32,
                    ColorStorage::Interleaved,
                    false,
                )
                .unwrap();
                let components: Vec<_> = input
                    .iter()
                    .flat_map(|p| {
                        p[..config.sample_format.channels().count() as usize]
                            .iter()
                            .copied()
                    })
                    .collect();
                let (layout, data) = Packing {
                    storage: Storage::Planar,
                    reversed: false,
                    shifted: true,
                }
                .pack(device, extent, &components, 4099);
                let buffer =
                    context
                        .device()
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("explicit ICC device components"),
                            contents: &data,
                            usage: wgpu::BufferUsages::STORAGE,
                        });
                assert_eq!(
                    bytes,
                    layouts::encode(
                        &context,
                        &backend,
                        &config,
                        BufferImageSource::new(Arc::new(buffer), layout).unwrap()
                    )
                );
                pixels.check(&bytes, &config);
                assert_eq!(context.memory_budget().snapshot().reserved_bytes, 0);
            }
        }
    }
}
