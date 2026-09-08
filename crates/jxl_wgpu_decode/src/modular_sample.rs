//! Original sample encoding, independent of Modular working-word geometry and render state.

use jxl_gpu_bitstream::SampleBitDepth;

use crate::modular_transform::GpuModularChannelLayout;

/// Validated JPEG XL sample precision, packed as total bits and exponent bits for the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularSampleEncoding(u32);

impl ModularSampleEncoding {
    pub(crate) const fn new(depth: SampleBitDepth) -> Option<Self> {
        match depth {
            SampleBitDepth::Integer { bits_per_sample } => {
                if bits_per_sample >= 1 && bits_per_sample <= 31 {
                    Some(Self(bits_per_sample))
                } else {
                    None
                }
            }
            SampleBitDepth::Float {
                bits_per_sample,
                exponent_bits_per_sample,
            } => {
                if exponent_bits_per_sample >= 2
                    && exponent_bits_per_sample <= 8
                    && bits_per_sample >= exponent_bits_per_sample + 3
                    && bits_per_sample <= exponent_bits_per_sample + 24
                {
                    Some(Self(bits_per_sample | (exponent_bits_per_sample << 8)))
                } else {
                    None
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) const fn integer(bits: u32) -> Option<Self> {
        Self::new(SampleBitDepth::Integer {
            bits_per_sample: bits,
        })
    }

    pub(crate) const fn bits(self) -> u8 {
        (self.0 & 255) as u8
    }
    pub(crate) const fn is_float(self) -> bool {
        self.0 >> 8 != 0
    }
    pub(crate) const fn packed(self) -> u32 {
        self.0
    }

    pub(crate) const fn depth(self) -> SampleBitDepth {
        if self.is_float() {
            SampleBitDepth::Float {
                bits_per_sample: self.bits() as u32,
                exponent_bits_per_sample: self.0 >> 8,
            }
        } else {
            SampleBitDepth::Integer {
                bits_per_sample: self.bits() as u32,
            }
        }
    }
}

/// An inverse-transformed view and its independently declared source sample interpretation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularOutputPlane {
    pub layout: GpuModularChannelLayout,
    pub encoding: ModularSampleEncoding,
}

impl ModularOutputPlane {
    pub(crate) const fn new(
        layout: GpuModularChannelLayout,
        encoding: ModularSampleEncoding,
    ) -> Self {
        Self { layout, encoding }
    }
}

pub(crate) fn shader(source: &str) -> String {
    source.replace(
        "/*__JXL_MODULAR_SAMPLE__*/",
        include_str!("modular_sample.wgsl"),
    )
}

#[cfg(test)]
pub(crate) fn integer_planes(planes: &[GpuModularChannelLayout]) -> Vec<ModularOutputPlane> {
    planes
        .iter()
        .map(|&layout| {
            ModularOutputPlane::new(
                layout,
                ModularSampleEncoding::integer(layout.bit_depth).unwrap(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_abi_rejects_unrepresentable_declarations() {
        for (depth, packed) in [
            (
                SampleBitDepth::Integer {
                    bits_per_sample: 16,
                },
                0x10,
            ),
            (
                SampleBitDepth::Float {
                    bits_per_sample: 16,
                    exponent_bits_per_sample: 5,
                },
                0x510,
            ),
            (
                SampleBitDepth::Float {
                    bits_per_sample: 32,
                    exponent_bits_per_sample: 8,
                },
                0x820,
            ),
        ] {
            let encoding = ModularSampleEncoding::new(depth).unwrap();
            assert_eq!(encoding.depth(), depth);
            assert_eq!(encoding.packed(), packed);
        }
        for bits_per_sample in [0, 32, u32::MAX] {
            assert!(
                ModularSampleEncoding::new(SampleBitDepth::Integer { bits_per_sample }).is_none()
            );
        }
        for (bits_per_sample, exponent_bits_per_sample) in [
            (4, 2),
            (27, 2),
            (32, 7),
            (16, 1),
            (16, 9),
            (0, 0),
            (u32::MAX, 8),
            (32, u32::MAX),
        ] {
            assert!(
                ModularSampleEncoding::new(SampleBitDepth::Float {
                    bits_per_sample,
                    exponent_bits_per_sample
                })
                .is_none()
            );
        }
    }
}
