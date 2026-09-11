//! Private frame storage with an explicit codec-component or RGB domain. All planes share one
//! accounted allocation; output
//! views carry their actual offsets instead of hiding extra samples beyond an RGB layout.

use jxl_gpu_formats::{Channel, ImageLayout, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_gpu_protocol::{ChangedRegions, Extent2d, OutputId, Region};
use jxl_wgpu::{GpuBufferLease, GpuImageOutput, UnvalidatedGpuImageOutput};

/// The sample domain at the post-reconstruction boundary. Patch references retain codec
/// components, frame blending uses the original encoding, and an unreferenced XYB presentation
/// can retain linear RGB until output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FrameSurfaceEncoding {
    Srgb,
    Linear,
    /// Codec components before the inverse color transform. The private producer contract
    /// carries this tag explicitly; a pixel format alone can never identify this domain.
    Encoded,
}

impl FrameSurfaceEncoding {
    pub(crate) fn format(self) -> PixelFormat {
        let mut color = crate::vardct_rgb8_format().color_spec;
        if self != Self::Srgb
            && let jxl_gpu_formats::ColorSpecification::Defined(ref mut color) = color
        {
            color.transfer = jxl_gpu_formats::TransferFunction::Linear;
        }
        PixelFormat::rgb_f32(RgbChannelOrder::Rgb, true, color)
    }

    pub(crate) fn from_format(format: &PixelFormat) -> Option<Self> {
        [Self::Srgb, Self::Linear]
            .into_iter()
            .find(|encoding| *format == encoding.format())
    }

    pub(crate) const fn rgb_encoding(self) -> jxl_gpu_protocol::RgbColorEncoding {
        match self {
            Self::Srgb => jxl_gpu_protocol::RgbColorEncoding::SRGB_BT709,
            Self::Linear | Self::Encoded => jxl_gpu_protocol::RgbColorEncoding::LINEAR_BT709,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FrameSurfaceError {
    #[error(transparent)]
    Layout(#[from] jxl_gpu_formats::LayoutError),
    #[error("frame surface {resource} requires {required} bytes, available {available}")]
    Limit {
        resource: &'static str,
        required: u64,
        available: u64,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct FrameSurfaceLayout {
    pub color: ImageLayout,
    pub extras: Vec<ImageLayout>,
    pub plane_bytes: u64,
    pub storage_bytes: u64,
}

impl FrameSurfaceLayout {
    pub(crate) fn new(
        extent: Extent2d,
        extra_count: usize,
        limits: &wgpu::Limits,
    ) -> Result<Self, FrameSurfaceError> {
        Self::with_encoding(extent, extra_count, FrameSurfaceEncoding::Srgb, limits)
    }

    pub(crate) fn with_encoding(
        extent: Extent2d,
        extra_count: usize,
        encoding: FrameSurfaceEncoding,
        limits: &wgpu::Limits,
    ) -> Result<Self, FrameSurfaceError> {
        let format = encoding.format();
        let color = ImageLayout::packed(extent, format)?;
        let scalar_bytes = color.planes[0].end_offset()?;
        let alignment = u64::from(limits.min_storage_buffer_offset_alignment).max(4);
        let available = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size)
            .min(u64::from(u32::MAX - 3));
        let overflow = || FrameSurfaceError::Limit {
            resource: "storage",
            required: u64::MAX,
            available,
        };
        let plane_bytes = scalar_bytes
            .checked_add(alignment - 1)
            .map(|bytes| bytes / alignment * alignment)
            .ok_or_else(overflow)?;
        let storage_bytes = (extra_count as u64)
            .checked_add(3)
            .and_then(|count| count.checked_mul(plane_bytes))
            .ok_or_else(overflow)?;
        if storage_bytes > available {
            return Err(FrameSurfaceError::Limit {
                resource: "storage",
                required: storage_bytes,
                available,
            });
        }
        let planes = color
            .planes
            .into_iter()
            .enumerate()
            .map(|(index, mut plane)| {
                plane.offset = index as u64 * plane_bytes;
                plane
            })
            .collect();
        let color = ImageLayout::from_planes(extent, color.format, planes)?;
        let scalar = ImageLayout::packed(
            extent,
            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
        )?;
        let extras = (0..extra_count)
            .map(|index| {
                let mut planes = scalar.planes.clone();
                planes[0].offset = (3 + index as u64) * plane_bytes;
                ImageLayout::from_planes(extent, scalar.format.clone(), planes)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            color,
            extras,
            plane_bytes,
            storage_bytes,
        })
    }

    pub(crate) fn layouts(&self) -> impl Iterator<Item = &ImageLayout> {
        std::iter::once(&self.color).chain(&self.extras)
    }
}

pub(crate) fn outputs(
    color: &ImageLayout,
    surface: Option<&FrameSurfaceLayout>,
    buffer: &GpuBufferLease,
) -> Vec<GpuImageOutput> {
    std::iter::once(color)
        .chain(surface.into_iter().flat_map(|surface| &surface.extras))
        .enumerate()
        .map(|(index, layout)| GpuImageOutput {
            id: OutputId(index as u32),
            layout: layout.clone(),
            buffer: buffer.clone(),
        })
        .collect()
}

pub(crate) fn unvalidated_outputs(
    color: &ImageLayout,
    surface: Option<&FrameSurfaceLayout>,
    buffer: &GpuBufferLease,
) -> Vec<UnvalidatedGpuImageOutput> {
    outputs(color, surface, buffer)
        .into_iter()
        .map(|output| UnvalidatedGpuImageOutput {
            id: output.id,
            layout: output.layout,
            buffer: output.buffer,
        })
        .collect()
}

pub(crate) fn changed_regions(
    color: &ImageLayout,
    surface: Option<&FrameSurfaceLayout>,
) -> ChangedRegions {
    ChangedRegions {
        outputs: std::iter::once(color)
            .chain(surface.into_iter().flat_map(|surface| &surface.extras))
            .enumerate()
            .map(|(index, layout)| {
                (
                    OutputId(index as u32),
                    vec![Region::new(0, 0, layout.extent.width, layout.extent.height)],
                )
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retention_checks_the_complete_allocation_before_building_channel_views() {
        let limits = wgpu::Limits {
            max_buffer_size: 65536,
            max_storage_buffer_binding_size: 65536,
            ..Default::default()
        };
        let extent = Extent2d::new(513, 5);
        let rgb = FrameSurfaceLayout::new(extent, 0, &limits).unwrap();
        assert!(rgb.storage_bytes < 65536);
        assert!(matches!(
            FrameSurfaceLayout::new(extent, 9, &limits),
            Err(FrameSurfaceError::Limit {
                resource: "storage",
                ..
            })
        ));
        assert!(FrameSurfaceLayout::new(Extent2d::new(1, 1), usize::MAX, &limits).is_err());
        assert!(FrameSurfaceLayout::new(Extent2d::new(u32::MAX, u32::MAX), 1, &limits).is_err());
    }

    #[test]
    fn independent_views_address_only_their_plane_and_include_the_real_prefix() {
        let limits = wgpu::Limits::default();
        let surface = FrameSurfaceLayout::new(Extent2d::new(3, 5), 9, &limits).unwrap();
        let mut end = 0;
        for plane in surface.layouts().flat_map(|layout| &layout.planes) {
            assert!(plane.offset >= end);
            assert!(
                plane
                    .offset
                    .is_multiple_of(u64::from(limits.min_storage_buffer_offset_alignment))
            );
            assert_eq!(plane.end_offset().unwrap() - plane.offset, 60);
            end = plane.end_offset().unwrap();
        }
        assert!(end <= surface.storage_bytes);
        assert_eq!(surface.extras[8].logical_size, end);
        assert_eq!(
            surface.color.logical_size,
            surface.color.planes[2].end_offset().unwrap()
        );
        assert_eq!(
            changed_regions(&surface.color, Some(&surface))
                .outputs
                .len(),
            10
        );
    }
}
