//! Private frame storage with an explicit codec-component or RGB domain. All planes share one
//! accounted allocation; each view carries its own extent and offset, including extra channels
//! that have already been upsampled while color components still await frame features.

use jxl_gpu_formats::{Channel, ImageLayout, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_gpu_protocol::{ChangedRegions, Extent2d, OutputId, Region};
use jxl_wgpu::{GpuBufferLease, GpuImageOutput, UnvalidatedGpuImageOutput};

/// A component surface can leave the producer before or after frame features. This is
/// independent of its sample domain: saved encoded references have completed all features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FrameRenderStage {
    Complete,
    BeforeFeatures,
}

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
    pub color_plane_bytes: u64,
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
        Self::with_extra_extents(
            extent,
            std::iter::repeat_n(extent, extra_count),
            encoding,
            limits,
        )
    }

    pub(crate) fn with_extra_extents(
        extent: Extent2d,
        extra_extents: impl ExactSizeIterator<Item = Extent2d> + Clone,
        encoding: FrameSurfaceEncoding,
        limits: &wgpu::Limits,
    ) -> Result<Self, FrameSurfaceError> {
        let format = encoding.format();
        let color = ImageLayout::packed(extent, format)?;
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
        let aligned_bytes = |bytes: u64| {
            bytes
                .checked_add(alignment - 1)
                .map(|bytes| bytes / alignment * alignment)
                .ok_or_else(overflow)
        };
        let color_plane_bytes = aligned_bytes(color.planes[0].end_offset()?)?;
        // Reject an impossible channel count before walking the iterator or allocating views.
        let minimum_bytes = (extra_extents.len() as u64)
            .checked_add(3)
            .and_then(|count| count.checked_mul(alignment))
            .ok_or_else(overflow)?;
        if minimum_bytes > available {
            return Err(FrameSurfaceError::Limit {
                resource: "storage",
                required: minimum_bytes,
                available,
            });
        }
        let scalar_format = PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]);
        let mut storage_bytes = color_plane_bytes.checked_mul(3).ok_or_else(overflow)?;
        for extra_extent in extra_extents.clone() {
            let scalar = ImageLayout::packed(extra_extent, scalar_format.clone())?;
            storage_bytes = storage_bytes
                .checked_add(aligned_bytes(scalar.logical_size)?)
                .ok_or_else(overflow)?;
            if storage_bytes > available {
                return Err(FrameSurfaceError::Limit {
                    resource: "storage",
                    required: storage_bytes,
                    available,
                });
            }
        }
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
                plane.offset = index as u64 * color_plane_bytes;
                plane
            })
            .collect();
        let color = ImageLayout::from_planes(extent, color.format, planes)?;
        let mut offset = 3 * color_plane_bytes;
        let extras = extra_extents
            .map(|extra_extent| {
                let mut scalar = ImageLayout::packed(extra_extent, scalar_format.clone())?;
                scalar.planes[0].offset = offset;
                offset += aligned_bytes(scalar.logical_size)?;
                Ok::<_, FrameSurfaceError>(ImageLayout::from_planes(
                    extra_extent,
                    scalar.format,
                    scalar.planes,
                )?)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            color,
            extras,
            color_plane_bytes,
            storage_bytes,
        })
    }

    pub(crate) fn layouts(&self) -> impl Iterator<Item = &ImageLayout> {
        std::iter::once(&self.color).chain(&self.extras)
    }

    pub(crate) fn has_uniform_extent(&self) -> bool {
        self.extras
            .iter()
            .all(|extra| extra.extent == self.color.extent)
    }

    pub(crate) fn set_encoding(&mut self, encoding: FrameSurfaceEncoding) {
        self.color.format = encoding.format();
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

    #[test]
    fn early_extra_resampling_has_independent_extents_and_checked_offsets() {
        let limits = wgpu::Limits::default();
        let color = Extent2d::new(13, 9);
        let extra = Extent2d::new(25, 17);
        let layout = FrameSurfaceLayout::with_extra_extents(
            color,
            [extra, color].into_iter(),
            FrameSurfaceEncoding::Encoded,
            &limits,
        )
        .unwrap();
        assert!(!layout.has_uniform_extent());
        assert_eq!(layout.extras[0].extent, extra);
        assert_eq!(layout.extras[1].extent, color);
        assert_eq!(
            layout.extras[0].planes[0].offset,
            layout.color_plane_bytes * 3
        );
        assert_eq!(
            layout.extras[1].planes[0].offset,
            layout.color_plane_bytes * 3 + 1792
        );
        assert_eq!(layout.storage_bytes, layout.color_plane_bytes * 4 + 1792);
        let changed = changed_regions(&layout.color, Some(&layout));
        assert_eq!(changed.outputs[&OutputId(1)], [Region::new(0, 0, 25, 17)]);
        let exact = wgpu::Limits {
            max_buffer_size: layout.storage_bytes,
            ..limits.clone()
        };
        assert!(
            FrameSurfaceLayout::with_extra_extents(
                color,
                [extra, color].into_iter(),
                FrameSurfaceEncoding::Encoded,
                &exact
            )
            .is_ok()
        );
        let too_small = wgpu::Limits {
            max_buffer_size: layout.storage_bytes - 1,
            ..limits
        };
        assert!(matches!(
            FrameSurfaceLayout::with_extra_extents(
                color,
                [extra, color].into_iter(),
                FrameSurfaceEncoding::Encoded,
                &too_small
            ),
            Err(FrameSurfaceError::Limit { .. })
        ));
    }
}
