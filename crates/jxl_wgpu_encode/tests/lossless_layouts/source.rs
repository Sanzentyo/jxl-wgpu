use super::*;
use wgpu::util::DeviceExt;

pub(super) use jxl_test_support::fixtures::source_layout::Storage;

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
    pub(super) fn is_float(self) -> bool {
        matches!(self.kind, SampleKind::Float | SampleKind::CustomFloat(_))
    }

    pub(super) fn exponent_bits(self) -> u8 {
        match self.kind {
            SampleKind::CustomFloat(precision) => precision.exponent_bits(),
            SampleKind::Float if self.bits == 16 => 5,
            SampleKind::Float if self.bits == 32 => 8,
            SampleKind::Unsigned => 0,
            _ => panic!("unsupported encoder sample precision"),
        }
    }

    pub(super) fn pixel_format(self) -> PixelFormat {
        match self.kind {
            SampleKind::CustomFloat(precision) => {
                assert_eq!(self.bits, precision.bits());
                self.format.custom_float_pixel_format(precision)
            }
            SampleKind::Float => self.format.float_pixel_format(self.bits).unwrap(),
            SampleKind::Unsigned => self.format.pixel_format(self.bits).unwrap(),
            SampleKind::Signed => panic!("unsupported signed encoder input"),
        }
    }

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
                if let SampleKind::CustomFloat(precision) = self.kind {
                    let fraction_bits = self.bits - precision.exponent_bits() - 1;
                    let fraction = (1 << fraction_bits) - 1;
                    let special = ((1 << precision.exponent_bits()) - 1) << fraction_bits;
                    let bias = (1 << (precision.exponent_bits() - 1)) - 1;
                    let values = [
                        0,
                        1,
                        fraction,
                        1 << fraction_bits,
                        special - 1,
                        special,
                        special | 1,
                        special | (1 << (fraction_bits - 1)),
                        special | fraction,
                        bias << fraction_bits,
                    ];
                    // Both signs of every boundary and NaN payload, then deterministic words.
                    // Components get different phases without omitting any value in a plane.
                    let channels = self.format.channel_count();
                    let phase = index / channels + (index % channels) * 7;
                    if phase % 32 < 20 {
                        values[(phase % 20 / 2) as usize] | ((phase & 1) << (self.bits - 1))
                    } else {
                        index.wrapping_mul(0x9e37_79b9).rotate_left(11) & mask
                    }
                } else if self.kind == SampleKind::Float {
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
        } else if let SampleKind::CustomFloat(precision) = self.kind {
            f32::from_bits(jxl_test_support::oracles::sample_bits::custom_binary32(
                word,
                precision.bits(),
                precision.exponent_bits(),
            ))
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

pub(super) fn upload(
    context: &WgpuContext,
    case: &Case,
    extent: Extent2d,
    expected: &[u32],
    plane_gap: u64,
) -> BufferImageSource {
    let mut format = case.pixel_format();
    format.byte_order = case.byte_order;
    let (layout, bytes) = jxl_test_support::fixtures::source_layout::Packing {
        storage: case.storage,
        reversed: case.reversed,
        shifted: case.shifted,
    }
    .pack(format, extent, expected, plane_gap);
    let buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lossless layout source with poisoned padding"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    BufferImageSource::new(Arc::new(buffer), layout).unwrap()
}
