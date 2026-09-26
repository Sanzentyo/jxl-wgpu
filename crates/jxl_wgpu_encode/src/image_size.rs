//! One checked JPEG XL SizeHeader representation for the canvas and intrinsic display hint.

use jxl_gpu_bitstream::BitWriter;

use crate::EncodeError;

const MAX_EXPLICIT_AXIS: u32 = 1 << 30;
const RATIOS: [(u32, u32); 7] = [(1, 1), (12, 10), (4, 3), (3, 2), (16, 9), (5, 4), (2, 1)];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ImageSize {
    width: u32,
    height: u32,
    ratio: u8,
}

impl ImageSize {
    /// The executable canvas profile retains its narrower coordinate bound. Intrinsic
    /// display hints use the full SizeHeader grammar without authorizing pixel allocation.
    pub(crate) fn canvas(width: u32, height: u32) -> Result<Self, EncodeError> {
        if width >= MAX_EXPLICIT_AXIS || height >= MAX_EXPLICIT_AXIS {
            return Err(EncodeError::InvalidConfiguration(
                "image canvas dimensions must be in 1..2^30",
            ));
        }
        Self::new(width, height)
    }

    pub(crate) fn new(width: u32, height: u32) -> Result<Self, EncodeError> {
        if width == 0 || !(1..=MAX_EXPLICIT_AXIS).contains(&height) {
            return Err(EncodeError::InvalidConfiguration(
                "image size is not representable",
            ));
        }
        // Keep existing explicit-size bytes. A wider axis is legal only through a normative
        // aspect ratio; derive it with integer truncation, not a rounded floating ratio.
        let ratio = if width <= MAX_EXPLICIT_AXIS {
            0
        } else {
            RATIOS
                .iter()
                .position(|&(numerator, denominator)| {
                    u64::from(height) * u64::from(numerator) / u64::from(denominator)
                        == u64::from(width)
                })
                .map(|index| index as u8 + 1)
                .ok_or(EncodeError::InvalidConfiguration(
                    "image width is not representable",
                ))?
        };
        Ok(Self {
            width,
            height,
            ratio,
        })
    }

    pub(crate) const fn width(self) -> u32 {
        self.width
    }
    pub(crate) const fn height(self) -> u32 {
        self.height
    }

    pub(crate) fn write(self, output: &mut BitWriter) -> Result<(), EncodeError> {
        output.write_bits(0, 1)?; // explicit (non-div8) height
        write_axis(output, self.height)?;
        output.write_bits(u64::from(self.ratio), 3)?;
        if self.ratio == 0 {
            write_axis(output, self.width)?;
        }
        Ok(())
    }
}

fn write_axis(output: &mut BitWriter, size: u32) -> Result<(), EncodeError> {
    let value = size - 1;
    let (selector, bits) = if value < 1 << 9 {
        (0, 9)
    } else if value < 1 << 13 {
        (1, 13)
    } else if value < 1 << 18 {
        (2, 18)
    } else {
        (3, 30)
    };
    output.write_bits(selector, 2)?;
    output.write_bits(u64::from(value), bits)?;
    Ok(())
}

/// Intended display dimensions, before orientation. This is a metadata hint, not a resize.
/// Height and explicit width reach 2^30; wider widths require an exact JPEG XL aspect ratio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntrinsicSize(ImageSize);

impl IntrinsicSize {
    pub fn new(width: u32, height: u32) -> Result<Self, EncodeError> {
        Ok(Self(ImageSize::new(width, height)?))
    }
    #[must_use]
    pub const fn width(self) -> u32 {
        self.0.width()
    }
    #[must_use]
    pub const fn height(self) -> u32 {
        self.0.height()
    }
    pub(crate) fn write(self, output: &mut BitWriter) -> Result<(), EncodeError> {
        self.0.write(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_oxide_common::Bundle;

    #[test]
    fn size_buckets_and_wide_ratio_boundaries_roundtrip_independently() {
        let mut sizes = Vec::new();
        for height in [1, 512, 513, 8192, 8193, 262144, 262145, 1 << 30] {
            for width in [1, 512, 513, 8192, 8193, 262144, 262145, 1 << 30] {
                sizes.push((width, height));
            }
        }
        // Every non-square derived ratio also exercises its integer truncation boundary.
        for height in [(1 << 30) - 1, 1 << 30] {
            for (numerator, denominator) in [(12, 10), (4, 3), (3, 2), (16, 9), (5, 4), (2, 1)] {
                sizes.push(((u64::from(height) * numerator / denominator) as u32, height));
            }
        }
        for (width, height) in sizes {
            let size = IntrinsicSize::new(width, height).unwrap();
            assert_eq!((size.width(), size.height()), (width, height));
            let mut writer = BitWriter::new();
            size.write(&mut writer).unwrap();
            let bytes = writer.into_bytes();
            let parsed =
                jxl_image::SizeHeader::parse(&mut jxl_bitstream::Bitstream::new(&bytes), ())
                    .unwrap();
            assert_eq!((parsed.width, parsed.height), (width, height));
        }
    }

    #[test]
    fn unrepresentable_sizes_are_rejected_before_header_construction() {
        for (width, height) in [
            (0, 1),
            (1, 0),
            (1, (1 << 30) + 1),
            ((1 << 30) + 1, 1),
            ((1 << 31) + 1, 1 << 30),
            (u32::MAX, u32::MAX),
        ] {
            assert!(matches!(
                IntrinsicSize::new(width, height),
                Err(EncodeError::InvalidConfiguration(_))
            ));
            assert!(matches!(
                crate::ImageSequenceDescriptor::new(width, height, crate::AnimationHeader::Still),
                Err(EncodeError::InvalidConfiguration(_))
            ));
        }
    }

    #[test]
    fn intrinsic_hints_do_not_extend_the_executable_canvas_profile() {
        for (width, height) in [(1 << 30, 1), (1, 1 << 30), (1 << 31, 1 << 30)] {
            assert!(IntrinsicSize::new(width, height).is_ok());
            assert!(matches!(
                crate::ImageSequenceDescriptor::new(width, height, crate::AnimationHeader::Still),
                Err(EncodeError::InvalidConfiguration(_))
            ));
            assert!(matches!(
                ImageSize::canvas(width, height),
                Err(EncodeError::InvalidConfiguration(_))
            ));
        }
    }
}
