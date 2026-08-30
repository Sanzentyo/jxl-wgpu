use std::ops::Range;

use jxl_gpu_protocol::Extent2d;
use thiserror::Error;

use crate::{PixelFormat, PixelFormatError, PlaneFormat};

/// Addressing for one plane in a single pitch-linear byte allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PitchLinearPlaneLayout {
    pub plane_index: usize,
    pub offset: u64,
    pub row_stride: u64,
    pub sample_extent: Extent2d,
    /// Minimum number of bytes containing the elements in one row. Padding up
    /// to `row_stride` is not part of this value.
    pub row_bytes: u64,
}

impl PitchLinearPlaneLayout {
    pub fn end_offset(&self) -> Result<u64, LayoutError> {
        let preceding_rows = u64::from(self.sample_extent.height.saturating_sub(1));
        self.offset
            .checked_add(
                self.row_stride
                    .checked_mul(preceding_rows)
                    .ok_or(LayoutError::SizeOverflow)?,
            )
            .and_then(|offset| offset.checked_add(self.row_bytes))
            .ok_or(LayoutError::SizeOverflow)
    }

    pub fn byte_range(&self) -> Result<Range<u64>, LayoutError> {
        Ok(self.offset..self.end_offset()?)
    }
}

/// Concrete offsets and row pitches for a [`PixelFormat`] in one directly
/// addressable allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageLayout {
    pub extent: Extent2d,
    pub format: PixelFormat,
    pub planes: Vec<PitchLinearPlaneLayout>,
    /// Last addressable byte plus one. This intentionally excludes any unused
    /// tail capacity in a larger GPU buffer.
    pub logical_size: u64,
}

impl ImageLayout {
    /// Builds a compact layout. Plane starts are aligned to four bytes so the
    /// layout is convenient for portable `wgpu` storage-buffer word access;
    /// row pitches remain tight.
    pub fn packed(extent: Extent2d, format: PixelFormat) -> Result<Self, LayoutError> {
        validate_extent(extent)?;
        format.validate()?;

        let mut offset = 0_u64;
        let mut planes = Vec::with_capacity(format.planes.len());
        for (plane_index, plane_format) in format.planes.iter().enumerate() {
            offset = align_up(offset, 4)?;
            let sample_extent = plane_extent(extent, plane_format)?;
            let row_bytes = minimum_row_bytes(sample_extent.width, plane_format)?;
            let plane = PitchLinearPlaneLayout {
                plane_index,
                offset,
                row_stride: row_bytes,
                sample_extent,
                row_bytes,
            };
            offset = plane.end_offset()?;
            planes.push(plane);
        }

        Self::from_planes(extent, format, planes)
    }

    /// Validates caller-selected plane offsets and pitches. Plane descriptions
    /// must be in format order and must not overlap in the shared allocation.
    pub fn from_planes(
        extent: Extent2d,
        format: PixelFormat,
        planes: Vec<PitchLinearPlaneLayout>,
    ) -> Result<Self, LayoutError> {
        validate_extent(extent)?;
        format.validate()?;
        if planes.len() != format.planes.len() {
            return Err(LayoutError::PlaneCount {
                expected: format.planes.len(),
                actual: planes.len(),
            });
        }

        let mut logical_size = 0_u64;
        let mut ranges = Vec::with_capacity(planes.len());
        for (plane_index, (plane, plane_format)) in planes.iter().zip(&format.planes).enumerate() {
            if plane.plane_index != plane_index {
                return Err(LayoutError::PlaneIndex {
                    expected: plane_index,
                    actual: plane.plane_index,
                });
            }
            let expected_extent = plane_extent(extent, plane_format)?;
            if plane.sample_extent != expected_extent {
                return Err(LayoutError::PlaneExtent {
                    plane: plane_index,
                    expected: expected_extent,
                    actual: plane.sample_extent,
                });
            }
            let expected_row_bytes = minimum_row_bytes(expected_extent.width, plane_format)?;
            if plane.row_bytes != expected_row_bytes {
                return Err(LayoutError::RowBytes {
                    plane: plane_index,
                    expected: expected_row_bytes,
                    actual: plane.row_bytes,
                });
            }
            if plane.row_stride < expected_row_bytes {
                return Err(LayoutError::ShortRowStride {
                    plane: plane_index,
                    minimum: expected_row_bytes,
                    actual: plane.row_stride,
                });
            }
            let range = plane.byte_range()?;
            logical_size = logical_size.max(range.end);
            ranges.push((plane_index, range));
        }

        ranges.sort_unstable_by_key(|(_, range)| range.start);
        for pair in ranges.windows(2) {
            let (left_index, left) = &pair[0];
            let (right_index, right) = &pair[1];
            if left.end > right.start {
                return Err(LayoutError::PlaneOverlap {
                    first: *left_index,
                    second: *right_index,
                });
            }
        }

        Ok(Self {
            extent,
            format,
            planes,
            logical_size,
        })
    }

    #[must_use]
    pub fn plane(&self, plane_index: usize) -> Option<&PitchLinearPlaneLayout> {
        self.planes.get(plane_index)
    }
}

fn validate_extent(extent: Extent2d) -> Result<(), LayoutError> {
    if extent.is_empty() {
        return Err(LayoutError::EmptyExtent);
    }
    Ok(())
}

fn plane_extent(extent: Extent2d, format: &PlaneFormat) -> Result<Extent2d, LayoutError> {
    let horizontal = u32::from(format.sampling.horizontal_divisor);
    let vertical = u32::from(format.sampling.vertical_divisor);
    if horizontal == 0 || vertical == 0 {
        return Err(LayoutError::SizeOverflow);
    }
    Ok(Extent2d::new(
        extent.width.div_ceil(horizontal),
        extent.height.div_ceil(vertical),
    ))
}

fn minimum_row_bytes(width: u32, format: &PlaneFormat) -> Result<u64, LayoutError> {
    let elements = width.div_ceil(u32::from(format.pixels_per_element));
    let bits = u64::from(elements)
        .checked_mul(format.bits_per_element())
        .ok_or(LayoutError::SizeOverflow)?;
    Ok(bits.div_ceil(8))
}

fn align_up(value: u64, alignment: u64) -> Result<u64, LayoutError> {
    let mask = alignment.checked_sub(1).ok_or(LayoutError::SizeOverflow)?;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or(LayoutError::SizeOverflow)
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum LayoutError {
    #[error("image extent must be non-empty")]
    EmptyExtent,
    #[error(transparent)]
    PixelFormat(#[from] PixelFormatError),
    #[error("image layout size overflow")]
    SizeOverflow,
    #[error("layout has {actual} planes; format requires {expected}")]
    PlaneCount { expected: usize, actual: usize },
    #[error("layout plane index {actual} is out of order; expected {expected}")]
    PlaneIndex { expected: usize, actual: usize },
    #[error("plane {plane} has extent {actual:?}; expected {expected:?}")]
    PlaneExtent {
        plane: usize,
        expected: Extent2d,
        actual: Extent2d,
    },
    #[error("plane {plane} has row byte count {actual}; expected {expected}")]
    RowBytes {
        plane: usize,
        expected: u64,
        actual: u64,
    },
    #[error("plane {plane} has row stride {actual}; expected at least {minimum}")]
    ShortRowStride {
        plane: usize,
        minimum: u64,
        actual: u64,
    },
    #[error("planes {first} and {second} overlap")]
    PlaneOverlap { first: usize, second: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChromaLocation2d, ColorRange, ColorSpec, ColorSpecification, PixelFormat};

    fn spec() -> ColorSpecification {
        ColorSpecification::Defined(ColorSpec::bt709(
            ColorRange::Limited,
            ChromaLocation2d::CENTER,
        ))
    }

    #[test]
    fn packs_odd_i420_without_losing_tail_samples() {
        let layout = ImageLayout::packed(
            Extent2d::new(5, 3),
            PixelFormat::i420(8, 8, spec()).unwrap(),
        )
        .unwrap();
        assert_eq!(layout.planes[0].sample_extent, Extent2d::new(5, 3));
        assert_eq!(layout.planes[0].row_bytes, 5);
        assert_eq!(layout.planes[1].sample_extent, Extent2d::new(3, 2));
        assert_eq!(layout.planes[1].row_bytes, 3);
        assert_eq!(layout.planes[2].sample_extent, Extent2d::new(3, 2));
        assert_eq!(layout.logical_size, 30);
    }

    #[test]
    fn p010_rows_use_sixteen_bit_words() {
        let layout = ImageLayout::packed(Extent2d::new(5, 3), PixelFormat::p010(spec())).unwrap();
        assert_eq!(layout.planes[0].row_bytes, 10);
        assert_eq!(layout.planes[1].sample_extent, Extent2d::new(3, 2));
        assert_eq!(layout.planes[1].row_bytes, 12);
    }

    #[test]
    fn accepts_padded_non_overlapping_rows() {
        let format = PixelFormat::nv12(spec());
        let planes = vec![
            PitchLinearPlaneLayout {
                plane_index: 0,
                offset: 0,
                row_stride: 16,
                sample_extent: Extent2d::new(5, 3),
                row_bytes: 5,
            },
            PitchLinearPlaneLayout {
                plane_index: 1,
                offset: 48,
                row_stride: 16,
                sample_extent: Extent2d::new(3, 2),
                row_bytes: 6,
            },
        ];
        let layout = ImageLayout::from_planes(Extent2d::new(5, 3), format, planes).unwrap();
        assert_eq!(layout.logical_size, 70);
    }

    #[test]
    fn rejects_overlapping_planes() {
        let packed = ImageLayout::packed(Extent2d::new(4, 4), PixelFormat::nv12(spec())).unwrap();
        let mut planes = packed.planes;
        planes[1].offset = 4;
        assert!(matches!(
            ImageLayout::from_planes(Extent2d::new(4, 4), PixelFormat::nv12(spec()), planes),
            Err(LayoutError::PlaneOverlap { .. })
        ));
    }
}
