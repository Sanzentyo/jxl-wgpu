#![cfg(not(target_arch = "wasm32"))]

mod admission;
mod display;
mod sequences;
mod stills;
mod tone;

use std::num::{NonZeroU8, NonZeroU64};
use std::sync::Arc;

use jxl_gpu_formats::{Channel, PackingFieldKind, PixelFormat, Swizzle};
use jxl_gpu_protocol::{Extent2d, OutputOrientation};
use jxl_test_support::fixtures::source_layout::{Packing, Storage};
use jxl_test_support::gpu::planes::{open_fragmented, read_bytes};
use jxl_test_support::oracles::{extra_channels, modular_words};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, OrientationPolicy, WgpuDecodeEngine,
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

    fn check_output(
        &self,
        bytes: &[u8],
        extent: Extent2d,
        orientation: OutputOrientation,
        alpha: bool,
    ) {
        for keep in [false, true] {
            let mut flags = vec!["--original", "--preserve-alpha"];
            if keep {
                flags.push("--keep-orientation");
            }
            let native = extra_channels::libjxl_output(bytes, &flags).expect("libjxl 0.12 output");
            let pixels = (extent.width * extent.height) as usize;
            let stride = pixels * (4 + usize::from(alpha));
            assert_eq!(native.len() % stride, 0);
            let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
                jxl_gpu_formats::RgbChannelOrder::Rgba,
                false,
                jxl_wgpu_decode::vardct_rgb8_format().color_spec,
            ))
            .unwrap()
            .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
            .with_orientation_policy(if keep {
                OrientationPolicy::Keep
            } else {
                OrientationPolicy::Apply
            });
            let output_extent = if keep {
                extent
            } else {
                orientation.map_extent(extent)
            };
            let mut whole = Vec::new();
            for fragmented in [false, true] {
                let mut session = if fragmented {
                    open_fragmented(&self.decoder, bytes, request.clone())
                } else {
                    self.decoder.open(bytes, request.clone()).unwrap()
                };
                let mut held = Vec::new();
                for expected in native.chunks_exact(stride) {
                    let frame = session.next_frame().unwrap().unwrap();
                    let plane = &frame.output().outputs[0];
                    assert_eq!(plane.layout.extent, output_extent);
                    let actual = read_bytes(&self.gpu, plane);
                    let values = extra_channels::floats(&actual);
                    assert_eq!(values.len(), pixels * 4);
                    for (i, (&a, &b)) in values.iter().zip(&expected[..pixels * 4]).enumerate() {
                        assert!(
                            a.is_finite()
                                && b.is_finite()
                                && (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
                            "{orientation:?} keep={keep}, pixel component {i}: {a} vs {b}"
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

fn name(index: usize) -> CodestreamName {
    let length = [0, 1, 15, 16, 47, 48, 1070, 1071][index % 8];
    let mut bytes = vec![b'n'; length];
    if length >= 15 {
        bytes[..7].copy_from_slice("方向\0".as_bytes());
    }
    CodestreamName::new(bytes).unwrap()
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
) -> (BufferImageSource, Vec<u32>) {
    let words = (0..extent.width * extent.height * (samples.channels().count() + u32::from(alpha)))
        .map(|i| {
            let code = (i * 17 + i / 3 * 13 + seed * 7) % 127 + 48;
            if samples.exponent_bits() == 0 {
                code * ((1u32 << samples.bits_per_sample()) - 1) / 255
            } else {
                (code as f32 / 256.0).to_bits()
            }
        })
        .collect::<Vec<_>>();
    let (layout, bytes) = Packing {
        storage: Storage::Planar,
        reversed: true,
        shifted: true,
    }
    .pack(format(samples, alpha), extent, &words, 4099);
    let buffer = rig
        .context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("orientation input with poisoned padding"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    (
        BufferImageSource::new(Arc::new(buffer), layout).unwrap(),
        words,
    )
}

fn check_headers(bytes: &[u8], orientation: OutputOrientation, names: &[CodestreamName]) {
    assert_eq!(
        modular_words::presentation_headers(bytes),
        names
            .iter()
            .map(|name| (
                orientation.to_exif_value(),
                name.as_str().as_bytes().to_vec()
            ))
            .collect::<Vec<_>>()
    );
    let inventory = jxl_gpu_bitstream::parse(bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(
        inventory.image_header.orientation,
        orientation.to_exif_value()
    );
    assert_eq!(inventory.frames.len(), names.len());
    for (frame, name) in inventory.frames.iter().zip(names) {
        assert_eq!(frame.name_bytes, name.as_str().as_bytes());
    }
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
fn names_validate_byte_limits_utf8_and_shared_ownership_before_use() {
    assert_eq!(CodestreamName::MAX_BYTES, 1071);
    for bytes in [
        vec![b'a'; 1072],
        vec![0xff],
        vec![0xe6, 0x96],
        "方".repeat(358).into_bytes(),
    ] {
        assert!(matches!(
            CodestreamName::new(&bytes),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            ExtraChannel::new(
                ExtraChannelKind::Depth,
                SamplePrecision::integer(8).unwrap(),
                0,
                bytes
            ),
            Err(EncodeError::InvalidConfiguration(_))
        ));
    }
    for index in 0..8 {
        let original = name(index);
        let cloned = original.clone();
        assert_eq!(cloned.as_str().as_ptr(), original.as_str().as_ptr());
        let extra = ExtraChannel::new(
            ExtraChannelKind::Depth,
            SamplePrecision::integer(8).unwrap(),
            0,
            original.as_str().as_bytes().to_vec(),
        )
        .unwrap();
        assert_eq!(extra.name(), original.as_str().as_bytes());
        drop(original);
        assert_eq!(cloned, name(index));
    }
    assert_eq!(
        CodestreamName::new("方".repeat(357))
            .unwrap()
            .as_str()
            .len(),
        1071
    );
}
