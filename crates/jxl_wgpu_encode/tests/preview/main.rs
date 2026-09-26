#![cfg(not(target_arch = "wasm32"))]

mod lifetime;
mod profiles;
mod sequences;

use std::num::{NonZeroU8, NonZeroU64};
use std::sync::Arc;

use jxl_gpu_formats::{Channel, PackingFieldKind, PixelFormat, Swizzle};
use jxl_gpu_protocol::{Extent2d, OutputOrientation};
use jxl_test_support::fixtures::source_layout::{Packing, Storage};
use jxl_test_support::gpu::planes::{open_fragmented, read_bytes};
use jxl_test_support::oracles::extra_channels as oracle;
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, ImageSelection, OrientationPolicy,
    WgpuDecodeEngine,
};
use jxl_wgpu_encode::*;
use wgpu::util::DeviceExt;

struct Rig {
    gpu: WgpuBackend,
    context: WgpuContext,
    decoder: GpuDecoder<WgpuDecodeEngine>,
}
impl Rig {
    fn new() -> Self {
        let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
        Self {
            context: WgpuContext::from_backend(&gpu),
            decoder: GpuDecoder::new(
                WgpuDecodeEngine::new(gpu.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
            ),
            gpu,
        }
    }

    fn check(
        &self,
        bytes: &[u8],
        main: Extent2d,
        preview: PreviewSize,
        orientation: OutputOrientation,
        extras: usize,
    ) {
        let inventory = jxl_gpu_bitstream::parse(bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert_eq!(
            inventory.image_header.preview_size,
            Some((preview.extent().width, preview.extent().height))
        );
        assert!(inventory.frames[0].is_preview);
        for selection in [ImageSelection::Main, ImageSelection::Preview] {
            let extent = if selection == ImageSelection::Main {
                main
            } else {
                preview.extent()
            };
            for keep in [false, true] {
                let mut flags = vec!["--original", "--preserve-alpha"];
                if selection == ImageSelection::Preview {
                    flags.push("--preview");
                }
                if keep {
                    flags.push("--keep-orientation");
                }
                let native = oracle::libjxl_output(bytes, &flags)
                    .expect("required native preview/main output");
                let pixels = (extent.width * extent.height) as usize;
                let stride = pixels
                    * (4 + if selection == ImageSelection::Main {
                        extras
                    } else {
                        0
                    });
                assert_eq!(native.len() % stride, 0);
                let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
                    jxl_gpu_formats::RgbChannelOrder::Rgba,
                    false,
                    jxl_wgpu_decode::vardct_rgb8_format().color_spec,
                ))
                .unwrap()
                .with_image_selection(selection)
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                .with_orientation_policy(if keep {
                    OrientationPolicy::Keep
                } else {
                    OrientationPolicy::Apply
                });
                let mut whole = Vec::new();
                for fragmented in [false, true] {
                    let mut session = if fragmented {
                        open_fragmented(&self.decoder, bytes, request.clone())
                    } else {
                        self.decoder.open(bytes, request.clone()).unwrap()
                    };
                    let mut held = Vec::new();
                    for expected in native.chunks_exact(stride) {
                        let frame = pollster::block_on(session.next_frame_async())
                            .unwrap()
                            .unwrap();
                        let plane = &frame.output().outputs[0];
                        assert_eq!(
                            plane.layout.extent,
                            if keep {
                                extent
                            } else {
                                orientation.map_extent(extent)
                            }
                        );
                        let actual = read_bytes(&self.gpu, plane);
                        let values = oracle::floats(&actual);
                        assert_eq!(values.len(), pixels * 4);
                        for (i, (&a, &b)) in values.iter().zip(&expected[..pixels * 4]).enumerate()
                        {
                            assert!(
                                a.is_finite()
                                    && b.is_finite()
                                    && (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
                                "{selection:?}/{orientation:?}/{keep}/{i}: {a} vs {b}"
                            );
                        }
                        if fragmented {
                            assert_eq!(actual, whole[held.len()]);
                        } else {
                            whole.push(actual);
                        }
                        held.push(frame);
                    }
                    assert!(session.next_frame().unwrap().is_none());
                    drop(session);
                    for (frame, expected) in held.iter().zip(&whole) {
                        assert_eq!(&read_bytes(&self.gpu, &frame.output().outputs[0]), expected);
                    }
                    drop(held);
                    assert_eq!(
                        self.decoder
                            .engine()
                            .in_flight_memory_stats()
                            .reserved_bytes,
                        0
                    );
                }
            }
        }
    }
}

fn format(samples: ColorSampleFormat, alpha: bool) -> PixelFormat {
    let mut format = samples.pixel_format();
    if alpha {
        let mut word = format.planes[0].words[0].clone();
        word.fields.last_mut().unwrap().kind = PackingFieldKind::Channel(Channel::W);
        format.planes[0].words.push(word);
        format.swizzle = if samples.channels() == ColorChannels::Gray {
            Swizzle::X00W
        } else {
            Swizzle::XYZW
        };
    }
    format
}

fn source(
    rig: &Rig,
    extent: Extent2d,
    samples: ColorSampleFormat,
    alpha: bool,
    seed: u32,
) -> BufferImageSource {
    packed_source(
        rig,
        extent,
        samples,
        alpha,
        &sample_words(extent, samples, alpha, seed),
    )
}

fn sample_words(extent: Extent2d, samples: ColorSampleFormat, alpha: bool, seed: u32) -> Vec<u32> {
    (0..extent.width * extent.height * (samples.channels().count() + u32::from(alpha)))
        .map(|i| {
            let code = (i * 37 + i / 3 * 17 + seed * 11) % 127 + 48;
            if samples.exponent_bits() == 0 {
                (u64::from(code) * ((1u64 << samples.bits_per_sample()) - 1) / 255) as u32
            } else {
                (code as f32 / 256.0).to_bits()
            }
        })
        .collect()
}

fn packed_source(
    rig: &Rig,
    extent: Extent2d,
    samples: ColorSampleFormat,
    alpha: bool,
    words: &[u32],
) -> BufferImageSource {
    let (layout, bytes) = Packing {
        storage: Storage::Planar,
        reversed: true,
        shifted: true,
    }
    .pack(format(samples, alpha), extent, words, 4099);
    let buffer = rig
        .context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("independent preview/main source"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    BufferImageSource::new(Arc::new(buffer), layout).unwrap()
}

fn progression() -> ProgressivePlan {
    ProgressivePlan::new(
        [(2, 2), (4, 0), (8, 0)]
            .into_iter()
            .map(|(square, shift)| ProgressivePass {
                coefficient_square: NonZeroU8::new(square).unwrap(),
                shift,
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn preview_dimensions_cover_every_bucket_with_both_codecs_and_independent_main() {
    let rig = Rig::new();
    let main = Extent2d::new(17, 9);
    for (index, width) in [1, 64, 65, 320, 321, 1344, 1345, 4096]
        .into_iter()
        .enumerate()
    {
        let orientation = OutputOrientation::from_exif_value(index as u32 + 1).unwrap();
        let preview = if index % 2 == 0 {
            PreviewSize::new(width, 3).unwrap()
        } else {
            PreviewSize::new(3, width).unwrap()
        };
        let samples = if index % 2 == 0 {
            ColorSampleFormat::integer(ColorChannels::Gray, 13).unwrap()
        } else {
            ColorSampleFormat::float(ColorChannels::Rgb, 32, 8).unwrap()
        };
        let image_options = ImageOptions {
            orientation,
            ..Default::default()
        };
        let factor = [
            UpsamplingFactor::One,
            UpsamplingFactor::Two,
            UpsamplingFactor::Four,
            UpsamplingFactor::Eight,
        ][index % 4];
        let preview_source = source(
            &rig,
            factor.source_extent(preview.extent()),
            samples,
            true,
            1,
        );
        let main_source = source(&rig, main, samples, true, 7);
        for modular in [true, false] {
            let name = CodestreamName::new("preview方向\0").unwrap();
            let options = FrameOptions {
                name,
                upsampling: factor,
                ..Default::default()
            };
            let bytes = if modular {
                let encoder = LosslessModularEncoder::with_config(
                    rig.context.clone(),
                    LosslessModularConfig {
                        group_size: LosslessModularGroupSize::Pixels128,
                        entropy: if index % 2 == 0 {
                            LosslessModularEntropyCoding::Prefix
                        } else {
                            LosslessModularEntropyCoding::Ans
                        },
                        local_transforms: LosslessModularSqueeze::HorizontalThenVertical.into(),
                        ..Default::default()
                    },
                )
                .with_image_options(image_options)
                .unwrap();
                let descriptor = LosslessModularSequenceDescriptor::from_pixel_format(
                    main.width,
                    main.height,
                    &main_source.layout.format,
                    AnimationHeader::Still,
                )
                .unwrap()
                .with_preview(preview);
                let mut session = encoder.begin_sequence(descriptor).unwrap();
                let preview_job = session
                    .submit_preview(preview_source.clone(), options)
                    .unwrap();
                assert_eq!(session.next_frame_index().get(), 0);
                let frame = session
                    .submit_last_frame(main_source.clone(), FrameOptions::default())
                    .unwrap()
                    .wait()
                    .unwrap();
                session.insert(frame).unwrap();
                session
                    .insert_preview(pollster::block_on(preview_job).unwrap())
                    .unwrap();
                session.finish_raw().unwrap()
            } else {
                let encoder = TiledVarDctEncoder::new_with_config(
                    rig.context.clone(),
                    VarDctConfig {
                        sample_format: samples,
                        alpha: Some(AlphaAssociation::Unassociated),
                        color_transform: if index % 2 == 0 {
                            VarDctColorTransform::Original
                        } else {
                            VarDctColorTransform::Xyb
                        },
                        image_options,
                        progressive: progression(),
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut session = encoder
                    .begin_sequence(
                        ImageSequenceDescriptor::new(
                            main.width,
                            main.height,
                            AnimationHeader::Still,
                        )
                        .unwrap()
                        .with_preview(preview),
                    )
                    .unwrap();
                let preview_job = session
                    .submit_preview(preview_source.clone(), options)
                    .unwrap();
                assert_eq!(session.next_frame_index().get(), 0);
                let frame = session
                    .submit_last_frame(main_source.clone(), FrameOptions::default())
                    .unwrap()
                    .wait()
                    .unwrap();
                session.insert(frame).unwrap();
                session.insert_preview(preview_job.wait().unwrap()).unwrap();
                session.finish_raw().unwrap()
            };
            if modular {
                let words = jxl_test_support::oracles::modular_words::original_preview(&bytes);
                let coded = factor.source_extent(preview.extent());
                assert_eq!((words.width, words.height), (coded.width, coded.height));
                assert_eq!(words.bits, u32::from(samples.bits_per_sample()));
                assert_eq!(words.exponent_bits, u32::from(samples.exponent_bits()));
                let expected = sample_words(coded, samples, true, 1);
                let channels = samples.channels().count() as usize + 1;
                assert_eq!(words.planes.len(), channels);
                for (channel, plane) in words.planes.iter().enumerate() {
                    assert_eq!(
                        plane.iter().map(|&word| word as u32).collect::<Vec<_>>(),
                        expected
                            .chunks_exact(channels)
                            .map(|pixel| pixel[channel])
                            .collect::<Vec<_>>()
                    );
                }
            }
            rig.check(&bytes, main, preview, orientation, 1);
            assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
        }
    }
}
