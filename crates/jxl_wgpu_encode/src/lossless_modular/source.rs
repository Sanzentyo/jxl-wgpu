use std::num::NonZeroU64;

use bytemuck::Zeroable;
use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, ImageLayout,
    PackingFieldKind, PixelFormat, PlaneSampling, SampleKind, Swizzle, SwizzleComponent,
};

use super::color::ModularColorEncoding;
use super::grid::LosslessModularGroup;
use super::memory::align_up;
use super::types::{LosslessModularFormat, ModularParams, ModularSourceParams};
use crate::{EncodeError, UnsupportedFeature};

#[derive(Clone, Copy, Debug)]
pub(super) struct LosslessModularSourceSpec {
    pub(super) format: LosslessModularFormat,
    pub(super) color: ModularColorEncoding,
    pub(super) bits_per_sample: u8,
    pub(super) bytes_per_sample: u8,
    pub(super) exponent_bits_per_sample: u8,
    pub(super) big_endian: bool,
    components: [ModularSourceParams; 4],
}

fn component_index(component: SwizzleComponent) -> Option<usize> {
    match component {
        SwizzleComponent::X => Some(0),
        SwizzleComponent::Y => Some(1),
        SwizzleComponent::Z => Some(2),
        SwizzleComponent::W => Some(3),
        SwizzleComponent::Zero | SwizzleComponent::One => None,
    }
}

pub(super) fn lossless_modular_source_spec(
    format: &PixelFormat,
) -> Result<LosslessModularSourceSpec, EncodeError> {
    if format.validate().is_err()
        || !matches!(format.sample_kind, SampleKind::Unsigned | SampleKind::Float)
        || format.chroma_subsampling != ChromaSubsampling::None
        || format.planes.len() > 4
    {
        return Err(UnsupportedFeature::InputFormat.into());
    }
    let Swizzle::Xyzw(swizzle) = format.swizzle else {
        return Err(UnsupportedFeature::InputFormat.into());
    };
    let logical_format = match (format.model, format.swizzle, &format.color_spec) {
        (ColorModel::NonColor, Swizzle::X000, ColorSpecification::Undefined)
        | (
            ColorModel::Gray,
            Swizzle::X001,
            ColorSpecification::Default
            | ColorSpecification::Undefined
            | ColorSpecification::Defined(_),
        ) => LosslessModularFormat::Gray,
        (ColorModel::Rgb, _, _) => {
            if swizzle[3] == SwizzleComponent::One {
                LosslessModularFormat::Rgb
            } else if component_index(swizzle[3]).is_some() {
                LosslessModularFormat::Rgba
            } else {
                return Err(UnsupportedFeature::InputFormat.into());
            }
        }
        _ => return Err(UnsupportedFeature::InputFormat.into()),
    };
    let mut stored = [None; 4];
    let mut bits_per_sample = None;
    let mut bytes_per_sample = 0;
    for (plane_index, plane) in format.planes.iter().enumerate() {
        if plane.sampling != PlaneSampling::FULL || plane.pixels_per_element != 1 {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        let mut word_offset = 0u32;
        let mut have_component = false;
        for word in &plane.words {
            let word_bits = word
                .fields
                .iter()
                .try_fold(0u32, |sum, field| sum.checked_add(u32::from(field.bits)))
                .ok_or(UnsupportedFeature::InputFormat)?;
            if !matches!(word_bits, 8 | 16 | 24 | 32) {
                return Err(UnsupportedFeature::InputFormat.into());
            }
            let mut bit_shift = word_bits;
            for field in &word.fields {
                bit_shift -= u32::from(field.bits);
                let index = match field.kind {
                    PackingFieldKind::Channel(Channel::X) => 0,
                    PackingFieldKind::Channel(Channel::Y) => 1,
                    PackingFieldKind::Channel(Channel::Z) => 2,
                    PackingFieldKind::Channel(Channel::W) => 3,
                    PackingFieldKind::Padding => continue,
                    _ => return Err(UnsupportedFeature::InputFormat.into()),
                };
                let supported_depth = if format.sample_kind == SampleKind::Float {
                    matches!(field.bits, 16 | 32)
                } else {
                    (1..=31).contains(&field.bits)
                };
                if !supported_depth
                    || stored[index].is_some()
                    || bits_per_sample.is_some_and(|bits| bits != field.bits)
                {
                    return Err(UnsupportedFeature::InputFormat.into());
                }
                stored[index] = Some(ModularSourceParams {
                    row_stride: 0,
                    byte_offset: word_offset,
                    pixel_stride: 0,
                    word_bytes: word_bits / 8,
                    bit_shift,
                    plane: plane_index as u32,
                });
                bits_per_sample = Some(field.bits);
                bytes_per_sample = bytes_per_sample.max(word_bits as u8 / 8);
                have_component = true;
            }
            word_offset = word_offset
                .checked_add(word_bits / 8)
                .ok_or(UnsupportedFeature::InputFormat)?;
        }
        if !have_component {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        for component in stored.iter_mut().flatten() {
            if component.plane == plane_index as u32 {
                component.pixel_stride = word_offset;
            }
        }
    }
    let mut components = [ModularSourceParams::zeroed(); 4];
    for (logical, component) in components
        .iter_mut()
        .take(logical_format.channel_count() as usize)
        .enumerate()
    {
        let index = component_index(swizzle[logical]).ok_or(UnsupportedFeature::InputFormat)?;
        *component = stored[index]
            .take()
            .ok_or(UnsupportedFeature::InputFormat)?;
    }
    // Duplicated, absent or discarded channels are not a lossless source contract.
    if stored.iter().any(Option::is_some) {
        return Err(UnsupportedFeature::InputFormat.into());
    }
    let bits_per_sample = bits_per_sample.ok_or(UnsupportedFeature::InputFormat)?;
    Ok(LosslessModularSourceSpec {
        format: logical_format,
        color: ModularColorEncoding::from_format(format)?,
        bits_per_sample,
        bytes_per_sample,
        exponent_bits_per_sample: if format.sample_kind == SampleKind::Float {
            if bits_per_sample == 16 { 5 } else { 8 }
        } else {
            0
        },
        big_endian: format.byte_order == ByteOrder::Big,
        components,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct SourceWindow {
    start: u64,
    end: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ModularSourceWindows([SourceWindow; 4]);

impl ModularSourceWindows {
    pub(super) fn merge(self, other: Self) -> Self {
        Self(std::array::from_fn(|index| {
            let (left, right) = (self.0[index], other.0[index]);
            if left.end == 0 {
                return right;
            }
            if right.end == 0 {
                return left;
            }
            SourceWindow {
                start: left.start.min(right.start),
                end: left.end.max(right.end),
            }
        }))
    }

    pub(super) fn maximum_bytes(self) -> u64 {
        self.0
            .iter()
            .map(|window| window.end - window.start)
            .max()
            .unwrap_or(0)
    }

    // Alignment can make neighboring plane windows overlap. Count each addressed byte once;
    // unused bindings alias the first window and do not increase source exposure.
    pub(super) fn addressed_bytes(self) -> Result<u64, EncodeError> {
        let mut windows = self.0;
        windows.sort_unstable_by_key(|window| window.start);
        let mut end = 0;
        let mut bytes = 0u64;
        for window in windows {
            if window.end > end {
                bytes = bytes
                    .checked_add(window.end - window.start.max(end))
                    .ok_or(EncodeError::InvalidSource("source window size overflow"))?;
                end = window.end;
            }
        }
        Ok(bytes)
    }

    pub(super) fn validate(self, limit: u64) -> Result<(), EncodeError> {
        let required = self.maximum_bytes();
        if required > limit {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffer_binding_size",
                required,
                available: limit,
            }
            .into());
        }
        if required == 0 || required - 1 > u64::from(u32::MAX) {
            return Err(EncodeError::InvalidSource(
                "source window exceeds WGSL u32 addressing",
            ));
        }
        Ok(())
    }

    pub(super) fn rebase(
        self,
        params: &mut ModularParams,
        absolute_offsets: [u64; 4],
    ) -> Result<(), EncodeError> {
        for (component, absolute) in params
            .sources
            .iter_mut()
            .zip(absolute_offsets)
            .take(params.channels as usize)
        {
            let window = self.0[component.plane as usize];
            component.byte_offset = u32::try_from(absolute.checked_sub(window.start).ok_or(
                EncodeError::InvalidSource("source window address underflow"),
            )?)
            .map_err(|_| EncodeError::InvalidSource("source window address exceeds WGSL u32"))?;
        }
        Ok(())
    }

    pub(super) fn entries(self, buffer: &wgpu::Buffer) -> [wgpu::BindGroupEntry<'_>; 4] {
        std::array::from_fn(|index| {
            let window = if self.0[index].end == 0 {
                self.0[0]
            } else {
                self.0[index]
            };
            wgpu::BindGroupEntry {
                binding: [0, 3, 4, 5][index],
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: window.start,
                    size: NonZeroU64::new(window.end - window.start),
                }),
            }
        })
    }
}

pub(super) struct ModularSourceLayout {
    pub(super) spec: LosslessModularSourceSpec,
    pub(super) full_windows: ModularSourceWindows,
    offsets: [u64; 4],
    alignment: u64,
}

pub(super) struct ModularGroupSource {
    pub(super) components: [ModularSourceParams; 4],
    pub(super) offsets: [u64; 4],
    pub(super) windows: ModularSourceWindows,
}

impl ModularSourceLayout {
    pub(super) fn new(
        layout: &ImageLayout,
        buffer_bytes: u64,
        alignment: u64,
    ) -> Result<Self, EncodeError> {
        let mut spec = lossless_modular_source_spec(&layout.format)?;
        // Public layout fields can be modified after construction. Revalidate every plane,
        // including extents, pitches, non-overlap, and the claimed logical allocation size.
        let checked =
            ImageLayout::from_planes(layout.extent, layout.format.clone(), layout.planes.clone())?;
        if checked != *layout {
            return Err(EncodeError::InvalidSource(
                "inconsistent logical source size",
            ));
        }
        let alignment = alignment.max(4);
        let mut full_windows = ModularSourceWindows::default();
        for (index, plane) in layout.planes.iter().enumerate() {
            let end = align_up(plane.end_offset()?, 4)
                .ok_or(EncodeError::InvalidSource("source binding size overflow"))?;
            if end > buffer_bytes {
                return Err(EncodeError::InvalidSource(
                    "source binding does not contain the final addressable sample word",
                ));
            }
            full_windows.0[index] = SourceWindow {
                start: plane.offset - plane.offset % alignment,
                end,
            };
        }
        let mut offsets = [0; 4];
        for (component, offset) in spec
            .components
            .iter_mut()
            .zip(&mut offsets)
            .take(spec.format.channel_count() as usize)
        {
            let plane = &layout.planes[component.plane as usize];
            component.row_stride = u32::try_from(plane.row_stride).map_err(|_| {
                EncodeError::InvalidSource("row stride exceeds WGSL u32 addressing")
            })?;
            *offset = plane
                .offset
                .checked_add(u64::from(component.byte_offset))
                .ok_or(EncodeError::InvalidSource(
                    "source component offset overflow",
                ))?;
            component.byte_offset = 0;
        }
        Ok(Self {
            spec,
            full_windows,
            offsets,
            alignment,
        })
    }

    pub(super) fn group(
        &self,
        group: LosslessModularGroup,
    ) -> Result<ModularGroupSource, EncodeError> {
        let mut offsets = [0; 4];
        let mut windows = ModularSourceWindows::default();
        for (index, component) in self
            .spec
            .components
            .iter()
            .take(self.spec.format.channel_count() as usize)
            .enumerate()
        {
            let start = self.offsets[index]
                .checked_add(u64::from(group.y) * u64::from(component.row_stride))
                .and_then(|value| {
                    value.checked_add(u64::from(group.x) * u64::from(component.pixel_stride))
                })
                .ok_or(EncodeError::InvalidSource("source group offset overflow"))?;
            let end = start
                .checked_add(u64::from(group.height - 1) * u64::from(component.row_stride))
                .and_then(|value| {
                    value
                        .checked_add(u64::from(group.width - 1) * u64::from(component.pixel_stride))
                })
                .and_then(|value| value.checked_add(u64::from(component.word_bytes)))
                .and_then(|value| align_up(value, 4))
                .ok_or(EncodeError::InvalidSource("source group end overflow"))?;
            let mut component_window = ModularSourceWindows::default();
            component_window.0[component.plane as usize] = SourceWindow {
                start: start - start % self.alignment,
                end,
            };
            windows = windows.merge(component_window);
            offsets[index] = start;
        }
        Ok(ModularGroupSource {
            components: self.spec.components,
            offsets,
            windows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_gpu_formats::{PackingField, PackingWord};
    use jxl_gpu_protocol::Extent2d;

    #[test]
    fn every_integer_bit_position_in_supported_words_has_exact_address_metadata() {
        for bits in 1..=31 {
            for word_bits in [8, 16, 24, 32].into_iter().filter(|&width| width >= bits) {
                for shift in 0..=word_bits - bits {
                    let mut format = LosslessModularFormat::Gray.pixel_format(bits).unwrap();
                    let mut fields = Vec::new();
                    if word_bits > bits + shift {
                        fields.push(PackingField::padding(word_bits - bits - shift));
                    }
                    fields.push(PackingField::channel(Channel::X, bits));
                    if shift != 0 {
                        fields.push(PackingField::padding(shift));
                    }
                    format.planes[0].words = vec![PackingWord { fields }];
                    let spec = lossless_modular_source_spec(&format).unwrap();
                    assert_eq!(spec.bits_per_sample, bits);
                    assert_eq!(spec.components[0].bit_shift, u32::from(shift));
                    assert_eq!(spec.components[0].word_bytes, u32::from(word_bits / 8));
                    assert_eq!(spec.components[0].pixel_stride, u32::from(word_bits / 8));
                }
            }
        }
    }

    #[test]
    fn large_absolute_plane_offsets_are_rebased_without_binding_the_gaps() {
        let extent = Extent2d::new(257, 3);
        let format = PixelFormat::rgb8(
            jxl_gpu_formats::RgbChannelOrder::Bgr,
            true,
            ColorSpecification::Undefined,
        );
        let mut layout = ImageLayout::packed(extent, format.clone()).unwrap();
        for (index, plane) in layout.planes.iter_mut().enumerate() {
            plane.offset = (1u64 << 34) * (index as u64 + 1) + 3;
            plane.row_stride = 263;
        }
        let layout = ImageLayout::from_planes(extent, format, layout.planes).unwrap();
        let source = ModularSourceLayout::new(&layout, layout.logical_size + 3, 256).unwrap();
        let group = super::super::grid::LosslessModularGroupGrid::for_extent(257, 3)
            .unwrap()
            .group(0)
            .unwrap();
        let tile = source.group(group).unwrap();
        assert!(
            tile.offsets
                .iter()
                .take(3)
                .all(|&offset| offset > u64::from(u32::MAX))
        );
        tile.windows.validate(1024).unwrap();
        assert_eq!(tile.windows.maximum_bytes(), 788);
        assert_eq!(tile.windows.addressed_bytes().unwrap(), 3 * 788);
        let mut params = ModularParams::zeroed();
        params.sources = tile.components;
        params.channels = 3;
        tile.windows.rebase(&mut params, tile.offsets).unwrap();
        assert!(
            params.sources[..3]
                .iter()
                .all(|source| source.byte_offset == 3)
        );
        assert!(tile.windows.validate(787).is_err());
    }

    #[test]
    fn shared_alignment_prefixes_are_counted_once() {
        let layout = ImageLayout::packed(
            Extent2d::new(1, 1),
            PixelFormat::rgb8(
                jxl_gpu_formats::RgbChannelOrder::Rgb,
                true,
                ColorSpecification::Undefined,
            ),
        )
        .unwrap();
        let source = ModularSourceLayout::new(&layout, 12, 256).unwrap();
        assert_eq!(source.full_windows.addressed_bytes().unwrap(), 12);
        assert_eq!(source.full_windows.maximum_bytes(), 12);
    }
}
