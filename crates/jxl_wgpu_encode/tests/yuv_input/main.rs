#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;

use jxl_gpu_formats::*;
use jxl_gpu_protocol::Extent2d;
use jxl_test_support::oracles::{modular_integer, modular_words, yuv::CodePlanes};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_encode::*;
use wgpu::util::DeviceExt;

mod boundaries;
mod sequence;

#[derive(Clone, Copy, Debug)]
enum Packing {
    Planar,
    Semi(ChromaOrder),
    Packed(Packed422Order),
}

struct Case {
    layout: ImageLayout,
    bytes: Vec<u8>,
    codes: CodePlanes,
    color: ColorSpec,
}

fn case(
    extent: Extent2d,
    packing: Packing,
    sampling: ChromaSubsampling,
    bits: u8,
    big: bool,
    color: ColorSpec,
) -> Case {
    let storage = if bits == 8 { 8 } else { 16 };
    let spec = ColorSpecification::Defined(color);
    let mut format = match packing {
        Packing::Planar => PixelFormat::yuv_planar(sampling, bits, storage, spec).unwrap(),
        Packing::Semi(order) => {
            PixelFormat::yuv_semiplanar(sampling, bits, storage, order, spec).unwrap()
        }
        Packing::Packed(order) => PixelFormat::packed_yuv4228(order, spec),
    };
    format.byte_order = if big {
        ByteOrder::Big
    } else {
        ByteOrder::Little
    };
    let mut planes = ImageLayout::packed(extent, format.clone()).unwrap().planes;
    let mut end = 37;
    for plane in &mut planes {
        plane.offset = end;
        plane.row_stride += 7;
        end = plane.end_offset().unwrap() + 11;
    }
    // Reverse physical plane order to catch code assuming contiguous or sorted planes.
    if planes.len() == 3 {
        let mut offset = 37;
        for plane in planes.iter_mut().rev() {
            plane.offset = offset;
            offset = plane.end_offset().unwrap() + 11;
        }
    }
    let layout = ImageLayout::from_planes(extent, format, planes).unwrap();
    let (dx, dy) = sampling.chroma_divisors().unwrap();
    let cw = extent.width.div_ceil(u32::from(dx));
    let ch = extent.height.div_ceil(u32::from(dy));
    let max = (1u32 << bits) - 1;
    let values = |len: usize, seed: usize| {
        (0..len)
            .map(|i| {
                let landmarks = [
                    0,
                    max,
                    16 << (bits - 8),
                    235 << (bits - 8),
                    128 << (bits - 8),
                ];
                if (i + seed) % 7 < landmarks.len() {
                    landmarks[(i + seed) % 7] as u16
                } else {
                    ((i * 4093 + seed * 1777) as u32 & max) as u16
                }
            })
            .collect()
    };
    let codes = CodePlanes {
        extent,
        subsampling: sampling,
        bits,
        y: values(extent.area().unwrap(), 0),
        cb: values((cw * ch) as usize, 2),
        cr: values((cw * ch) as usize, 4),
    };
    let mut bytes = vec![0xa7; layout.logical_size.div_ceil(4) as usize * 4];
    let mut write = |plane: usize, x: u32, y: u32, v: u16| {
        let p = &layout.planes[plane];
        let offset = (p.offset
            + u64::from(y) * p.row_stride
            + u64::from(x) * u64::from(storage / 8)) as usize;
        let pad = storage - bits;
        let v = (v << pad) | ((1u16 << pad) - 1);
        let word = if big {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        };
        if storage == 8 {
            bytes[offset] = v as u8;
        } else {
            bytes[offset..offset + 2].copy_from_slice(&word);
        }
    };
    for y in 0..extent.height {
        for x in 0..extent.width {
            let v = codes.y[(y * extent.width + x) as usize];
            let pos = match packing {
                Packing::Packed(order) => {
                    x / 2 * 4 + x % 2 * 2 + u32::from(order == Packed422Order::Uyvy)
                }
                _ => x,
            };
            write(0, pos, y, v);
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            let i = (y * cw + x) as usize;
            match packing {
                Packing::Planar => {
                    write(1, x, y, codes.cb[i]);
                    write(2, x, y, codes.cr[i]);
                }
                Packing::Semi(order) => {
                    let swap = u32::from(order == ChromaOrder::CrCb);
                    write(1, 2 * x + swap, y, codes.cb[i]);
                    write(1, 2 * x + 1 - swap, y, codes.cr[i]);
                }
                Packing::Packed(order) => {
                    let offset = u32::from(order == Packed422Order::Yuyv);
                    write(0, 4 * x + offset, y, codes.cb[i]);
                    write(0, 4 * x + offset + 2, y, codes.cr[i]);
                }
            }
        }
    }
    Case {
        layout,
        bytes,
        codes,
        color,
    }
}

fn upload(context: &WgpuContext, layout: ImageLayout, bytes: &[u8]) -> BufferImageSource {
    BufferImageSource::new(
        Arc::new(
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("YUV independent code planes"),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                }),
        ),
        layout,
    )
    .unwrap()
}

impl Case {
    fn input(&self, context: &WgpuContext, transfer: YuvRgbTransfer) -> YuvImageSource {
        YuvImageSource::new(upload(context, self.layout.clone(), &self.bytes), transfer).unwrap()
    }
    fn texture_input(&self, context: &WgpuContext, transfer: YuvRgbTransfer) -> YuvImageSource {
        let planes = jxl_test_support::gpu::textures::upload_planes(
            context.device(),
            context.queue(),
            &self.layout,
            &self.bytes,
        )
        .into_iter()
        .map(|texture| {
            let format = texture.format();
            TexturePlaneSource::new(texture, format, 1, 1).unwrap()
        })
        .collect();
        YuvImageSource::new(
            TexturePlanesSource::new(self.layout.extent, self.layout.format.clone(), planes)
                .unwrap(),
            transfer,
        )
        .unwrap()
    }
    fn stored_input(
        &self,
        context: &WgpuContext,
        transfer: YuvRgbTransfer,
        textures: bool,
    ) -> YuvImageSource {
        if textures {
            self.texture_input(context, transfer)
        } else {
            self.input(context, transfer)
        }
    }
    fn assert_rgb(&self, channels: &[modular_integer::ExtraWords], linear: bool) {
        let expected = self.codes.rgb(self.color, linear);
        assert!(channels.len() >= 3);
        for (c, plane) in channels.iter().take(3).enumerate() {
            assert_eq!(
                (plane.width, plane.height),
                (self.layout.extent.width, self.layout.extent.height)
            );
            for (i, &word) in plane.words.iter().enumerate() {
                let value = f64::from(f32::from_bits(word));
                // NCL's short affine arithmetic uses an absolute bound; transfer evaluation
                // adds a relative bound for pow/exp and their propagated input rounding.
                let bound = if linear {
                    8e-6 * expected[i][c].abs().max(1.0)
                } else {
                    2e-6
                };
                assert!(
                    value.is_finite() && (value - expected[i][c]).abs() <= bound,
                    "{:?} {:?} pixel={i} component={c}: {value} vs {} bound={bound}",
                    self.layout.format,
                    self.color,
                    expected[i][c]
                );
            }
        }
    }
}

fn decoded_input(
    context: &WgpuContext,
    format: PixelFormat,
    channels: &[modular_integer::ExtraWords],
) -> BufferImageSource {
    let extent = Extent2d::new(channels[0].width, channels[0].height);
    let bytes: Vec<_> = (0..extent.area().unwrap())
        .flat_map(|i| (0..3).flat_map(move |c| channels[c].words[i].to_le_bytes()))
        .collect();
    upload(
        context,
        ImageLayout::packed(extent, format).unwrap(),
        &bytes,
    )
}

fn config(input: &YuvImageSource) -> VarDctConfig {
    VarDctConfig {
        sample_format: ColorSampleFormat::float(ColorChannels::Rgb, 32, 8).unwrap(),
        source_color: input.pixel_format().color_spec.clone(),
        color_transform: VarDctColorTransform::Original,
        ..Default::default()
    }
}

#[test]
fn yuv_packing_depth_range_and_chroma_phases_match_independent_f64_and_native_words() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let encoders = [
        LosslessModularEntropyCoding::Prefix,
        LosslessModularEntropyCoding::Ans,
    ]
    .map(|entropy| {
        LosslessModularEncoder::with_config(
            context.clone(),
            LosslessModularConfig {
                entropy,
                ..Default::default()
            },
        )
    });
    let mut cases = Vec::new();
    for sampling in [
        ChromaSubsampling::Cs444,
        ChromaSubsampling::Cs422,
        ChromaSubsampling::Cs422R,
        ChromaSubsampling::Cs411,
        ChromaSubsampling::Cs411R,
        ChromaSubsampling::Cs420,
    ] {
        for (packing, bits, big) in [
            (Packing::Planar, 8, false),
            (Packing::Planar, 10, true),
            (Packing::Semi(ChromaOrder::CbCr), 12, false),
            (Packing::Semi(ChromaOrder::CrCb), 16, true),
        ] {
            cases.push((packing, sampling, bits, big));
        }
    }
    for order in [Packed422Order::Yuyv, Packed422Order::Uyvy] {
        cases.push((Packing::Packed(order), ChromaSubsampling::Cs422, 8, false));
    }
    for (index, &(packing, sampling, bits, big)) in cases.iter().enumerate() {
        for (phase, location) in [
            ChromaLocation::Even,
            ChromaLocation::Center,
            ChromaLocation::Odd,
        ]
        .into_iter()
        .enumerate()
        {
            for range in [ColorRange::Full, ColorRange::Limited] {
                let color = ColorSpec::bt709(
                    range,
                    ChromaLocation2d {
                        horizontal: location,
                        vertical: location,
                    },
                );
                let fixture = case(Extent2d::new(9, 5), packing, sampling, bits, big, color);
                let input = fixture.input(&context, YuvRgbTransfer::Preserve);
                let encoder = &encoders[(index + phase) % 2];
                let encoded = encoder.encode(input).unwrap();
                assert_eq!(
                    encoded,
                    encoder
                        .encode(fixture.texture_input(&context, YuvRgbTransfer::Preserve))
                        .unwrap()
                );
                let native = modular_words::channel_frames(&encoded).remove(0);
                fixture.assert_rgb(&native, false);
                assert_eq!(modular_integer::modular_channel_words(&encoded, 0), native);
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn yuv_color_matrices_transfers_excursions_and_single_pixel_edges_are_explicit() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let encoder = LosslessModularEncoder::new(context.clone());
    for matrix in [
        YcbcrEncoding::Bt601,
        YcbcrEncoding::Bt709,
        YcbcrEncoding::Bt2020,
        YcbcrEncoding::Bt2020ConstantLuminance,
    ] {
        for transfer in [
            TransferFunction::Linear,
            TransferFunction::Srgb,
            TransferFunction::Sycc,
            TransferFunction::Bt709,
            TransferFunction::Bt2020,
            TransferFunction::Dci,
            TransferFunction::Hlg,
            TransferFunction::Pq,
            TransferFunction::Gamma(jxl_gpu_protocol::GammaExponent::new(0.4).unwrap()),
        ] {
            if matrix == YcbcrEncoding::Bt2020ConstantLuminance
                && transfer != TransferFunction::Bt2020
            {
                continue;
            }
            for output in [YuvRgbTransfer::Preserve, YuvRgbTransfer::Linear] {
                if output == YuvRgbTransfer::Preserve
                    && (matrix == YcbcrEncoding::Bt2020ConstantLuminance
                        || transfer == TransferFunction::Bt2020)
                {
                    continue;
                }
                if output == YuvRgbTransfer::Linear
                    && matches!(transfer, TransferFunction::Pq | TransferFunction::Gamma(_))
                {
                    continue;
                }
                let mut color = ColorSpec::bt2020_ncl(
                    ColorRange::Limited,
                    ChromaLocation2d {
                        horizontal: ChromaLocation::Odd,
                        vertical: ChromaLocation::Center,
                    },
                );
                color.encoding = matrix;
                color.transfer = transfer;
                let fixture = case(
                    Extent2d::new(1, 9),
                    Packing::Semi(ChromaOrder::CrCb),
                    ChromaSubsampling::Cs420,
                    10,
                    true,
                    color,
                );
                let input = fixture.input(&context, output);
                let encoded = encoder.encode(input).unwrap();
                assert_eq!(
                    encoded,
                    encoder
                        .encode(fixture.texture_input(&context, output))
                        .unwrap()
                );
                let native = modular_words::channel_frames(&encoded).remove(0);
                fixture.assert_rgb(&native, output == YuvRgbTransfer::Linear);
                assert_eq!(modular_integer::modular_channel_words(&encoded, 0), native);
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

/// Retention checks must cover every caller-owned plane, not just the first handle.
enum SourceOwners {
    Buffer(std::sync::Weak<wgpu::Buffer>),
    Textures(Vec<std::sync::Weak<wgpu::Texture>>),
}
impl SourceOwners {
    fn of(source: &YuvImageSource) -> Self {
        match source.source() {
            ImageSourceStorage::Buffer(source) => Self::Buffer(Arc::downgrade(&source.buffer)),
            ImageSourceStorage::TexturePlanes(source) => Self::Textures(
                source
                    .planes
                    .iter()
                    .map(|plane| Arc::downgrade(&plane.texture))
                    .collect(),
            ),
            ImageSourceStorage::Texture(source) => {
                Self::Textures(vec![Arc::downgrade(&source.texture)])
            }
        }
    }
    fn released(&self) -> bool {
        match self {
            Self::Buffer(owner) => owner.upgrade().is_none(),
            Self::Textures(owners) => owners.iter().all(|owner| owner.upgrade().is_none()),
        }
    }
}
