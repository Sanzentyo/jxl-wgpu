//! Private frame storage with an explicit codec-component or color domain. All planes share one
//! accounted allocation; each view carries its own extent and offset, including extra channels
//! that have already been upsampled while color components still await frame features.

use jxl_gpu_formats::{Channel, ImageLayout, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_gpu_protocol::icc::{IccProfile, IccSignature};
use jxl_gpu_protocol::{
    ChangedRegions, Extent2d, OutputId, Region, RgbColorEncoding, RgbColorSpace,
};
use jxl_wgpu::{GpuBufferLease, GpuImageOutput, UnvalidatedGpuImageOutput};

pub(crate) mod copy;

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
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FrameSurfaceEncoding {
    Rgb(RgbColorEncoding),
    /// Original device values with their exact profile, with one gray or three RGB planes.
    Icc(IccProfile),
    /// Original JPEG XL CMY components and their independently stored Black extra channel.
    /// Stored samples are complements of ICC ink amounts. Reference blending stays in this
    /// codestream domain; a four-channel ICC view borrows Black without duplicating it.
    Cmyk {
        profile: IccProfile,
        black_extra: usize,
    },
    /// Codec components before the inverse color transform. The private producer contract
    /// carries this tag explicitly; a pixel format alone can never identify this domain.
    Encoded,
}

impl FrameSurfaceEncoding {
    pub(crate) fn format(&self) -> PixelFormat {
        if let Self::Icc(profile) = self {
            let color = jxl_gpu_formats::ColorSpecification::Icc(profile.clone());
            return if profile.header().device_space == IccSignature(*b"GRAY") {
                PixelFormat::gray_f32(false, true, color)
            } else {
                PixelFormat::rgb_f32(RgbChannelOrder::Rgb, true, color)
            };
        }
        if matches!(self, Self::Encoded | Self::Cmyk { .. }) {
            let mut format = PixelFormat::non_color(
                SampleKind::Float,
                32,
                &[Channel::X, Channel::Y, Channel::Z],
            );
            format.planes = [Channel::X, Channel::Y, Channel::Z]
                .into_iter()
                .map(|channel| {
                    jxl_gpu_formats::PlaneFormat::separate_words(
                        jxl_gpu_formats::PlaneSampling::FULL,
                        1,
                        &[channel],
                        32,
                    )
                })
                .collect();
            return format;
        }
        let mut color = crate::vardct_rgb8_format().color_spec;
        let encoding = self.rgb_encoding().expect("RGB surface encoding");
        if let jxl_gpu_formats::ColorSpecification::Defined(ref mut color) = color {
            color.space = match encoding.space {
                RgbColorSpace::Bt709 => jxl_gpu_formats::ColorSpace::Bt709,
                RgbColorSpace::Bt2020 => jxl_gpu_formats::ColorSpace::Bt2020,
                RgbColorSpace::DisplayP3 => jxl_gpu_formats::ColorSpace::DisplayP3,
                RgbColorSpace::Custom(value) => jxl_gpu_formats::ColorSpace::CustomRgb(value),
                RgbColorSpace::Undefined => unreachable!("validated frame RGB primaries"),
            };
            color.transfer = match encoding.transfer {
                jxl_gpu_protocol::TransferFunction::Linear => {
                    jxl_gpu_formats::TransferFunction::Linear
                }
                jxl_gpu_protocol::TransferFunction::Srgb => jxl_gpu_formats::TransferFunction::Srgb,
                jxl_gpu_protocol::TransferFunction::Bt709 => {
                    jxl_gpu_formats::TransferFunction::Bt709
                }
                jxl_gpu_protocol::TransferFunction::Gamma(exponent) => {
                    jxl_gpu_formats::TransferFunction::Gamma(exponent)
                }
                jxl_gpu_protocol::TransferFunction::Bt2020 => {
                    jxl_gpu_formats::TransferFunction::Bt2020
                }
                jxl_gpu_protocol::TransferFunction::Pq => jxl_gpu_formats::TransferFunction::Pq,
                jxl_gpu_protocol::TransferFunction::Hlg => jxl_gpu_formats::TransferFunction::Hlg,
                jxl_gpu_protocol::TransferFunction::Dci => jxl_gpu_formats::TransferFunction::Dci,
            };
        }
        PixelFormat::rgb_f32(RgbChannelOrder::Rgb, true, color)
    }

    /// Recognize only a complete canonical color layout. Codec components and CMYK's
    /// separately stored Black channel always require the producer's explicit domain.
    pub(crate) fn from_format(format: &PixelFormat) -> Option<Self> {
        format.validate().ok()?;
        if let jxl_gpu_formats::ColorSpecification::Icc(profile) = &format.color_spec {
            let encoding = Self::Icc(profile.clone());
            return (*format == encoding.format()).then_some(encoding);
        }
        let jxl_gpu_formats::ColorSpecification::Defined(color) = format.color_spec else {
            return None;
        };
        let encoding = Self::Rgb(RgbColorEncoding {
            space: color.space.rgb_space()?,
            transfer: color.transfer.rgb_transfer()?,
        });
        (*format == encoding.format()).then_some(encoding)
    }

    pub(crate) const fn rgb_encoding(&self) -> Option<RgbColorEncoding> {
        match self {
            Self::Rgb(encoding) => Some(*encoding),
            Self::Icc(_) | Self::Cmyk { .. } | Self::Encoded => None,
        }
    }

    pub(crate) fn icc_profile(&self) -> Option<&IccProfile> {
        match self {
            Self::Icc(profile) | Self::Cmyk { profile, .. } => Some(profile),
            Self::Rgb(_) | Self::Encoded => None,
        }
    }

    pub(crate) fn icc_sample_encoding(&self) -> jxl_wgpu::ResidentIccSampleEncoding {
        if matches!(self, Self::Cmyk { .. }) {
            jxl_wgpu::ResidentIccSampleEncoding::Complement
        } else {
            jxl_wgpu::ResidentIccSampleEncoding::Direct
        }
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum FrameSurfaceError {
    #[error(transparent)]
    Layout(#[from] jxl_gpu_formats::LayoutError),
    #[error("frame surface {resource} requires {required} bytes, available {available}")]
    Limit {
        resource: &'static str,
        required: u64,
        available: u64,
    },
    #[error("frame plane copy {role} {plane}: {reason}")]
    Copy {
        role: &'static str,
        plane: usize,
        reason: &'static str,
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
    pub(crate) fn icc_planes(
        &self,
        encoding: &FrameSurfaceEncoding,
    ) -> crate::Result<Vec<jxl_wgpu::ResidentIccPlane>> {
        let black = if let FrameSurfaceEncoding::Cmyk { black_extra, .. } = encoding {
            let black = self
                .extras
                .get(*black_extra)
                .ok_or(crate::Error::EngineContract(
                    "CMYK Black plane is outside the frame surface",
                ))?;
            if self.color.planes.len() != 3
                || black.planes.len() != 1
                || black.extent != self.color.extent
            {
                return Err(crate::Error::EngineContract(
                    "CMYK planes have incompatible extents",
                ));
            }
            Some(&black.planes[0])
        } else {
            None
        };
        Ok(self
            .color
            .planes
            .iter()
            .chain(black)
            .map(|plane| jxl_wgpu::ResidentIccPlane {
                offset: (plane.offset / 4) as u32,
                stride: (plane.row_stride / 4) as u32,
            })
            .collect())
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
        let color_count = color.planes.len() as u64;
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
            .checked_add(color_count)
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
        let mut storage_bytes = color_plane_bytes
            .checked_mul(color_count)
            .ok_or_else(overflow)?;
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
        let mut offset = color_count * color_plane_bytes;
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
    fn canonical_rgb_layouts_round_trip_without_claiming_a_component_domain() {
        for primaries in [
            RgbColorSpace::Bt709,
            RgbColorSpace::Bt2020,
            RgbColorSpace::DisplayP3,
            RgbColorSpace::Custom(jxl_gpu_protocol::RgbChromaticities {
                white: jxl_gpu_protocol::Chromaticity::DCI,
                ..jxl_gpu_protocol::RgbChromaticities::DISPLAY_P3
            }),
        ] {
            for transfer in [
                jxl_gpu_protocol::TransferFunction::Linear,
                jxl_gpu_protocol::TransferFunction::Srgb,
                jxl_gpu_protocol::TransferFunction::Bt709,
                jxl_gpu_protocol::TransferFunction::Dci,
                jxl_gpu_protocol::TransferFunction::Gamma(
                    jxl_gpu_protocol::GammaExponent::new(0.5).unwrap(),
                ),
            ] {
                let encoding = FrameSurfaceEncoding::Rgb(RgbColorEncoding {
                    space: primaries,
                    transfer,
                });
                assert_eq!(
                    FrameSurfaceEncoding::from_format(&encoding.format()),
                    Some(encoding.clone())
                );
                let interleaved =
                    PixelFormat::rgb_f32(RgbChannelOrder::Rgb, false, encoding.format().color_spec);
                assert_eq!(FrameSurfaceEncoding::from_format(&interleaved), None);
            }
        }
        assert_eq!(FrameSurfaceEncoding::Encoded.rgb_encoding(), None);
        assert_eq!(
            FrameSurfaceEncoding::from_format(&FrameSurfaceEncoding::Encoded.format()),
            None
        );
        assert_eq!(
            FrameSurfaceEncoding::Encoded.format().color_spec,
            jxl_gpu_formats::ColorSpecification::Undefined
        );
    }

    #[test]
    fn retention_checks_the_complete_allocation_before_building_channel_views() {
        let limits = wgpu::Limits {
            max_buffer_size: 65536,
            max_storage_buffer_binding_size: 65536,
            ..Default::default()
        };
        let extent = Extent2d::new(513, 5);
        let rgb = FrameSurfaceLayout::with_encoding(
            extent,
            0,
            FrameSurfaceEncoding::Rgb(jxl_gpu_protocol::RgbColorEncoding::SRGB_BT709),
            &limits,
        )
        .unwrap();
        assert!(rgb.storage_bytes < 65536);
        assert!(matches!(
            FrameSurfaceLayout::with_encoding(
                extent,
                9,
                FrameSurfaceEncoding::Rgb(jxl_gpu_protocol::RgbColorEncoding::SRGB_BT709),
                &limits
            ),
            Err(FrameSurfaceError::Limit {
                resource: "storage",
                ..
            })
        ));
        assert!(
            FrameSurfaceLayout::with_encoding(
                Extent2d::new(1, 1),
                usize::MAX,
                FrameSurfaceEncoding::Rgb(jxl_gpu_protocol::RgbColorEncoding::SRGB_BT709),
                &limits
            )
            .is_err()
        );
        assert!(
            FrameSurfaceLayout::with_encoding(
                Extent2d::new(u32::MAX, u32::MAX),
                1,
                FrameSurfaceEncoding::Rgb(jxl_gpu_protocol::RgbColorEncoding::SRGB_BT709),
                &limits
            )
            .is_err()
        );
    }

    #[test]
    fn independent_views_address_only_their_plane_and_include_the_real_prefix() {
        let limits = wgpu::Limits::default();
        let surface = FrameSurfaceLayout::with_encoding(
            Extent2d::new(3, 5),
            9,
            FrameSurfaceEncoding::Rgb(jxl_gpu_protocol::RgbColorEncoding::SRGB_BT709),
            &limits,
        )
        .unwrap();
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
