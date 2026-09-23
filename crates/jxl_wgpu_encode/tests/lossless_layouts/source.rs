use super::*;
use jxl_gpu_formats::{
    Channel, ImageLayout, PackingField, PackingWord, PlaneFormat, PlaneSampling, Swizzle,
    SwizzleComponent,
};
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug)]
pub(super) enum Storage {
    Packed,
    Planar,
    Split,
    SharedWord,
    ThreeBytes,
    MixedWords,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Case {
    pub(super) format: LosslessModularFormat,
    pub(super) bits: u8,
    pub(super) kind: SampleKind,
    pub(super) storage: Storage,
    pub(super) reversed: bool,
    pub(super) byte_order: ByteOrder,
    pub(super) shifted: bool,
}

// Explicit source/reference pairs include both signs, subnormals, infinities, and NaN payloads.
const HALF: [(u32, u32); 16] = [
    (0, 0),
    (0x8000, 0x8000_0000),
    (1, 0x3380_0000),
    (0x8001, 0xb380_0000),
    (0x3ff, 0x387f_c000),
    (0x83ff, 0xb87f_c000),
    (0x400, 0x3880_0000),
    (0x7bff, 0x477f_e000),
    (0xfbff, 0xc77f_e000),
    (0x7c00, 0x7f80_0000),
    (0xfc00, 0xff80_0000),
    (0x7c01, 0x7f80_2000),
    (0xfc01, 0xff80_2000),
    (0x7fff, 0x7fff_e000),
    (0x3c00, 0x3f80_0000),
    (0xbc00, 0xbf80_0000),
];

impl Case {
    pub(super) fn canonical(self) -> Self {
        Self {
            storage: Storage::Packed,
            reversed: false,
            byte_order: ByteOrder::Native,
            shifted: false,
            ..self
        }
    }

    pub(super) fn samples(self, extent: Extent2d) -> Vec<u32> {
        let mask = u32::MAX >> (32 - self.bits);
        (0..extent.width * extent.height * self.format.channel_count())
            .map(|index| {
                if self.kind == SampleKind::Float {
                    if self.bits == 16 {
                        HALF[index as usize % HALF.len()].0
                    } else {
                        let values = [
                            0,
                            0x8000_0000,
                            1,
                            0x8000_0001,
                            0x007f_ffff,
                            0x0080_0000,
                            0x7f7f_ffff,
                            0xff7f_ffff,
                            0x7f80_0000,
                            0xff80_0000,
                            0x7f80_0001,
                            0xff80_0001,
                            0x7fff_ffff,
                            0xffff_ffff,
                            0x3f80_0000,
                            0xbf80_0000,
                        ];
                        values[index as usize % values.len()]
                    }
                } else if index % 64 < 16 {
                    [0, mask, 1, mask - 1][index as usize % 4]
                } else {
                    index.wrapping_mul(0x9e37_79b9).rotate_left(11) & mask
                }
            })
            .collect()
    }

    pub(super) fn normalized(self, word: u32) -> f32 {
        if self.kind == SampleKind::Unsigned {
            (f64::from(word) / f64::from(u32::MAX >> (32 - self.bits))) as f32
        } else if self.bits == 32 {
            f32::from_bits(word)
        } else {
            f32::from_bits(HALF.iter().find(|&&(half, _)| half == word).map_or_else(
                || jxl_test_support::oracles::sample_bits::binary32(word, 16),
                |&(_, expected)| expected,
            ))
        }
    }
}

struct Sample {
    logical: usize,
    word: usize,
    shift: u8,
}

pub(super) fn upload(
    context: &WgpuContext,
    case: &Case,
    extent: Extent2d,
    expected: &[u32],
    plane_gap: u64,
) -> BufferImageSource {
    let channels = case.format.channel_count() as usize;
    let mut pixel_format = if case.kind == SampleKind::Float {
        case.format.float_pixel_format(case.bits).unwrap()
    } else {
        case.format.pixel_format(case.bits).unwrap()
    };
    pixel_format.byte_order = case.byte_order;
    let mut order: Vec<_> = (0..channels).collect();
    if case.reversed && case.format == LosslessModularFormat::GrayAlpha {
        order.reverse();
        pixel_format.swizzle = Swizzle::Xyzw([
            SwizzleComponent::W,
            SwizzleComponent::Zero,
            SwizzleComponent::Zero,
            SwizzleComponent::X,
        ]);
    }
    if case.reversed && channels >= 3 {
        order[..3].reverse();
        pixel_format.swizzle = if channels == 4 {
            Swizzle::ZYXW
        } else {
            Swizzle::ZYX1
        };
        if matches!(case.storage, Storage::Split) && channels == 4 {
            order = vec![3, 0, 1, 2];
            pixel_format.swizzle = Swizzle::Xyzw([
                SwizzleComponent::Y,
                SwizzleComponent::Z,
                SwizzleComponent::W,
                SwizzleComponent::X,
            ]);
        }
    }
    let ids = if case.format == LosslessModularFormat::GrayAlpha {
        [Channel::X, Channel::W, Channel::Y, Channel::Z]
    } else {
        [Channel::X, Channel::Y, Channel::Z, Channel::W]
    };
    let physical_planes: Vec<Vec<usize>> = match case.storage {
        Storage::Planar => (0..channels).map(|index| vec![index]).collect(),
        Storage::Split if channels > 1 => vec![(0..channels - 1).collect(), vec![channels - 1]],
        _ => vec![(0..channels).collect()],
    };
    let mut assignments = Vec::new();
    pixel_format.planes = physical_planes
        .iter()
        .map(|physical| {
            let mut samples = Vec::new();
            let words = if matches!(case.storage, Storage::SharedWord) {
                let storage_bits = (channels as u8 * case.bits + 1).div_ceil(8) * 8;
                assert!(storage_bits <= 32);
                let mut fields = Vec::new();
                for (index, &position) in physical.iter().enumerate() {
                    let shift = storage_bits - case.bits * (index as u8 + 1);
                    fields.push(PackingField::channel(ids[position], case.bits));
                    samples.push(Sample {
                        logical: order[position],
                        word: 0,
                        shift,
                    });
                }
                fields.push(PackingField::padding(
                    storage_bits - channels as u8 * case.bits,
                ));
                vec![PackingWord { fields }]
            } else {
                physical
                    .iter()
                    .enumerate()
                    .map(|(word, &position)| {
                        let storage_bits = match case.storage {
                            Storage::ThreeBytes => 24,
                            Storage::MixedWords => [8, 16, 24, 32][position],
                            _ if case.shifted && case.bits == 16 => 32,
                            _ => case.bits.next_power_of_two().max(8),
                        };
                        let shift = if case.shifted {
                            (storage_bits - case.bits).div_ceil(2)
                        } else {
                            0
                        };
                        let mut fields = Vec::new();
                        if storage_bits > case.bits + shift {
                            fields.push(PackingField::padding(storage_bits - case.bits - shift));
                        }
                        fields.push(PackingField::channel(ids[position], case.bits));
                        if shift != 0 {
                            fields.push(PackingField::padding(shift));
                        }
                        samples.push(Sample {
                            logical: order[position],
                            word,
                            shift,
                        });
                        PackingWord { fields }
                    })
                    .collect()
            };
            assignments.push(samples);
            PlaneFormat {
                sampling: PlaneSampling::FULL,
                pixels_per_element: 1,
                words,
            }
        })
        .collect();
    let mut layout = ImageLayout::packed(extent, pixel_format.clone()).unwrap();
    // Store planes in reverse physical order, with unrelated unaligned pitches and poisoned gaps.
    let mut end = 5;
    for index in (0..layout.planes.len()).rev() {
        let plane = &mut layout.planes[index];
        plane.offset = end;
        plane.row_stride = plane.row_bytes + 5 + 2 * index as u64;
        end = plane.end_offset().unwrap() + plane_gap + 3;
    }
    let layout = ImageLayout::from_planes(extent, pixel_format, layout.planes).unwrap();
    let mut bytes = vec![0xa5; layout.logical_size.div_ceil(4) as usize * 4];
    let mask = u32::MAX >> (32 - case.bits);
    for (plane_index, plane) in layout.planes.iter().enumerate() {
        let word_widths: Vec<_> = layout.format.planes[plane_index]
            .words
            .iter()
            .map(|word| word.bits() as usize / 8)
            .collect();
        let pixel_bytes: usize = word_widths.iter().sum();
        for y in 0..extent.height {
            for x in 0..extent.width {
                let mut packed = vec![u32::MAX; word_widths.len()];
                for sample in &assignments[plane_index] {
                    let value =
                        expected[((y * extent.width + x) as usize) * channels + sample.logical];
                    packed[sample.word] =
                        (packed[sample.word] & !(mask << sample.shift)) | (value << sample.shift);
                }
                let mut address = (plane.offset + u64::from(y) * plane.row_stride) as usize
                    + x as usize * pixel_bytes;
                for (word, &width) in packed.into_iter().zip(&word_widths) {
                    if case.byte_order == ByteOrder::Big {
                        bytes[address..address + width]
                            .copy_from_slice(&word.to_be_bytes()[4 - width..]);
                    } else {
                        bytes[address..address + width]
                            .copy_from_slice(&word.to_le_bytes()[..width]);
                    }
                    address += width;
                }
            }
        }
    }
    let buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lossless layout source with poisoned padding"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    BufferImageSource::new(Arc::new(buffer), layout).unwrap()
}
