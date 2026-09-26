//! Checked source packing and bounded plane windows shared by encoder backends.

use std::num::NonZeroU64;

use bytemuck::Zeroable;
use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, ImageLayout,
    PackingField, PackingFieldKind, PackingWord, PixelFormat, PlaneFormat, PlaneSampling,
    SampleKind, Swizzle, SwizzleComponent,
};

use crate::{EncodeError, UnsupportedFeature};

mod cmyk;
pub use cmyk::CmykSampleEncoding;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SourceParams {
    pub(crate) row_stride: u32,
    pub(crate) byte_offset: u32,
    pub(crate) pixel_stride: u32,
    pub(crate) word_bytes: u32,
    /// Low five bits are the sample shift; bit five selects exact integer complement.
    pub(crate) bit_shift: u32,
    pub(crate) plane: u32,
}

/// Color and alpha topology shared by checked encoder input plans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceChannels {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
}

impl SourceChannels {
    #[must_use]
    pub const fn channel_count(self) -> u32 {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::GrayAlpha | Self::Rgba)
    }

    #[must_use]
    pub const fn color_channels(self) -> crate::ColorChannels {
        match self {
            Self::Gray | Self::GrayAlpha => crate::ColorChannels::Gray,
            Self::Rgb | Self::Rgba => crate::ColorChannels::Rgb,
        }
    }

    #[must_use]
    pub const fn color_channel_count(self) -> u32 {
        self.color_channels().count()
    }

    /// Constructs the canonical pitch-linear source format for an unsigned integer depth.
    ///
    /// Depths `1..=8` use one native-endian `u8` word per component. Depths `9..=16` use one
    /// native-endian `u16` word per component, and `17..=31` use `u32`. Samples occupy the low bits;
    /// the high padding bits are outside the valid sample and are ignored by the encoder.
    pub fn pixel_format(self, bits_per_sample: u8) -> Result<PixelFormat, EncodeError> {
        if !(1..=31).contains(&bits_per_sample) {
            return Err(EncodeError::InvalidConfiguration(
                "lossless Modular integer depth must be in 1..=31",
            ));
        }
        Ok(self.packed_pixel_format(bits_per_sample, SampleKind::Unsigned))
    }

    /// Constructs native IEEE binary16 or binary32 storage, preserving every source bit.
    ///
    /// Components remain in the declared sRGB/gray domain. No floating-point arithmetic,
    /// normalization or alpha association is performed by the lossless encoder.
    pub fn float_pixel_format(self, bits_per_sample: u8) -> Result<PixelFormat, EncodeError> {
        if !matches!(bits_per_sample, 16 | 32) {
            return Err(EncodeError::InvalidConfiguration(
                "lossless Modular floating storage must be binary16 or binary32",
            ));
        }
        Ok(self.packed_pixel_format(bits_per_sample, SampleKind::Float))
    }

    /// Constructs raw binary floating storage with explicitly checked sample/exponent widths.
    /// Like integer input, each component occupies the low bits of an 8/16/32-bit word.
    /// All components, including alpha, share this precision. No F32 conversion occurs.
    ///
    /// ```
    /// use jxl_gpu_formats::{FloatPrecision, SampleKind};
    /// use jxl_wgpu_encode::LosslessModularFormat;
    /// let precision = FloatPrecision::new(24, 7).unwrap();
    /// let format = LosslessModularFormat::Rgba.custom_float_pixel_format(precision);
    /// assert_eq!(format.sample_kind, SampleKind::CustomFloat(precision));
    /// format.validate().unwrap();
    /// ```
    #[must_use]
    pub fn custom_float_pixel_format(
        self,
        precision: jxl_gpu_formats::FloatPrecision,
    ) -> PixelFormat {
        self.packed_pixel_format(precision.bits(), SampleKind::CustomFloat(precision))
    }

    fn packed_pixel_format(self, bits_per_sample: u8, sample_kind: SampleKind) -> PixelFormat {
        let storage_bits = bits_per_sample.next_power_of_two().max(8);
        let (model, color_spec, swizzle, channels): (_, _, _, &[Channel]) = match self {
            Self::Gray => (
                ColorModel::NonColor,
                ColorSpecification::Undefined,
                Swizzle::X000,
                &[Channel::X],
            ),
            Self::GrayAlpha => (
                ColorModel::Gray,
                ColorSpecification::Default,
                Swizzle::X00W,
                &[Channel::X, Channel::W],
            ),
            Self::Rgb => (
                ColorModel::Rgb,
                ColorSpecification::Default,
                Swizzle::XYZ1,
                &[Channel::X, Channel::Y, Channel::Z],
            ),
            Self::Rgba => (
                ColorModel::Rgb,
                ColorSpecification::Default,
                Swizzle::XYZW,
                &[Channel::X, Channel::Y, Channel::Z, Channel::W],
            ),
        };
        let words = channels
            .iter()
            .copied()
            .map(|channel| {
                let mut fields = Vec::with_capacity(2);
                if bits_per_sample < storage_bits {
                    fields.push(PackingField::padding(storage_bits - bits_per_sample));
                }
                fields.push(PackingField::channel(channel, bits_per_sample));
                PackingWord { fields }
            })
            .collect();
        PixelFormat {
            model,
            color_spec,
            chroma_subsampling: ChromaSubsampling::None,
            sample_kind,
            byte_order: ByteOrder::Native,
            swizzle,
            planes: vec![PlaneFormat {
                sampling: PlaneSampling::FULL,
                pixels_per_element: 1,
                words,
            }],
        }
    }
}

pub(crate) fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment.checked_sub(1)?)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}

#[derive(Clone, Debug)]
pub(crate) struct SourceSpec {
    pub(crate) format: SourceChannels,
    pub(crate) bits_per_sample: u8,
    pub(crate) bytes_per_sample: u8,
    pub(crate) exponent_bits_per_sample: u8,
    pub(crate) big_endian: bool,
    components: [SourceParams; 4],
    black: Option<SourceParams>,
}

impl SourceSpec {
    pub(crate) fn is_cmyk(&self) -> bool {
        self.black.is_some()
    }
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

pub(crate) fn source_spec(format: &PixelFormat) -> Result<SourceSpec, EncodeError> {
    if format.validate().is_err()
        || !matches!(
            format.sample_kind,
            SampleKind::Unsigned | SampleKind::Float | SampleKind::CustomFloat(_)
        )
        || format.chroma_subsampling != ChromaSubsampling::None
        || format.planes.len() > 5
    {
        return Err(UnsupportedFeature::InputFormat.into());
    }
    let device_channels = if format.model == ColorModel::IccDevice {
        let ColorSpecification::Icc(profile) = &format.color_spec else {
            return Err(UnsupportedFeature::InputFormat.into());
        };
        if format.swizzle != Swizzle::Device {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        match &profile.header().device_space.0 {
            b"GRAY" => Some(1),
            b"RGB " => Some(3),
            b"CMYK" => Some(4),
            _ => return Err(UnsupportedFeature::InputFormat.into()),
        }
    } else {
        None
    };
    let (model, logical_swizzle) = if let Some(channels) = device_channels {
        let alpha = format
            .planes
            .iter()
            .flat_map(|plane| &plane.words)
            .any(|word| {
                word.fields
                    .iter()
                    .any(|field| field.kind == PackingFieldKind::Channel(Channel::Alpha))
            });
        if channels == 1 {
            (
                ColorModel::Gray,
                Swizzle::Xyzw([
                    SwizzleComponent::X,
                    SwizzleComponent::Zero,
                    SwizzleComponent::Zero,
                    if alpha {
                        SwizzleComponent::Y
                    } else {
                        SwizzleComponent::One
                    },
                ]),
            )
        } else {
            (
                ColorModel::Rgb,
                if alpha { Swizzle::XYZW } else { Swizzle::XYZ1 },
            )
        }
    } else {
        (format.model, format.swizzle)
    };
    let Swizzle::Xyzw(swizzle) = logical_swizzle else {
        return Err(UnsupportedFeature::InputFormat.into());
    };
    let logical_format = match (model, logical_swizzle, &format.color_spec) {
        (ColorModel::NonColor, Swizzle::X000, ColorSpecification::Undefined) => {
            SourceChannels::Gray
        }
        (
            ColorModel::Gray,
            Swizzle::Xyzw(
                [
                    _,
                    SwizzleComponent::Zero,
                    SwizzleComponent::Zero,
                    SwizzleComponent::One,
                ],
            ),
            _,
        ) => SourceChannels::Gray,
        (
            ColorModel::Gray,
            Swizzle::Xyzw([_, SwizzleComponent::Zero, SwizzleComponent::Zero, alpha]),
            _,
        ) if component_index(alpha).is_some() => SourceChannels::GrayAlpha,
        (ColorModel::Rgb, _, _) => {
            if swizzle[3] == SwizzleComponent::One {
                SourceChannels::Rgb
            } else if component_index(swizzle[3]).is_some() {
                SourceChannels::Rgba
            } else {
                return Err(UnsupportedFeature::InputFormat.into());
            }
        }
        _ => return Err(UnsupportedFeature::InputFormat.into()),
    };
    let mut stored = [None; 5];
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
                    PackingFieldKind::Channel(Channel::X) if device_channels.is_none() => 0,
                    PackingFieldKind::Channel(Channel::Y) if device_channels.is_none() => 1,
                    PackingFieldKind::Channel(Channel::Z) if device_channels.is_none() => 2,
                    PackingFieldKind::Channel(Channel::W) if device_channels.is_none() => 3,
                    PackingFieldKind::Channel(Channel::Device(index))
                        if device_channels.is_some_and(|count| index < count) =>
                    {
                        usize::from(index)
                    }
                    PackingFieldKind::Channel(Channel::Alpha) if device_channels.is_some() => {
                        usize::from(device_channels.expect("checked ICC device source"))
                    }
                    PackingFieldKind::Padding => continue,
                    _ => return Err(UnsupportedFeature::InputFormat.into()),
                };
                let supported_depth = match format.sample_kind {
                    SampleKind::Float => matches!(field.bits, 16 | 32),
                    SampleKind::CustomFloat(precision) => field.bits == precision.bits(),
                    SampleKind::Unsigned => (1..=31).contains(&field.bits),
                    SampleKind::Signed => false,
                };
                if !supported_depth
                    || stored[index].is_some()
                    || bits_per_sample.is_some_and(|bits| bits != field.bits)
                {
                    return Err(UnsupportedFeature::InputFormat.into());
                }
                stored[index] = Some(SourceParams {
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
    let mut components = [SourceParams::zeroed(); 4];
    for (logical, component) in components
        .iter_mut()
        .take(logical_format.channel_count() as usize)
        .enumerate()
    {
        let swizzle_index = if logical == logical_format.color_channel_count() as usize {
            3 // Gray+alpha's second logical channel selects the canonical alpha component.
        } else {
            logical
        };
        let index = if device_channels == Some(4) && swizzle_index == 3 {
            4 // CMYK's physical alpha follows K; the coded main view remains CMY/alpha.
        } else {
            component_index(swizzle[swizzle_index]).ok_or(UnsupportedFeature::InputFormat)?
        };
        *component = stored[index]
            .take()
            .ok_or(UnsupportedFeature::InputFormat)?;
    }
    let black = if device_channels == Some(4) {
        Some(stored[3].take().ok_or(UnsupportedFeature::InputFormat)?)
    } else {
        None
    };
    // Duplicated, absent or discarded channels are not a lossless source contract.
    if stored.iter().any(Option::is_some) {
        return Err(UnsupportedFeature::InputFormat.into());
    }
    let bits_per_sample = bits_per_sample.ok_or(UnsupportedFeature::InputFormat)?;
    Ok(SourceSpec {
        format: logical_format,
        bits_per_sample,
        bytes_per_sample,
        exponent_bits_per_sample: match format.sample_kind {
            SampleKind::Float => {
                let precision = if bits_per_sample == 16 {
                    jxl_gpu_formats::FloatPrecision::BINARY16
                } else {
                    jxl_gpu_formats::FloatPrecision::BINARY32
                };
                precision.exponent_bits()
            }
            SampleKind::CustomFloat(precision) => precision.exponent_bits(),
            SampleKind::Unsigned | SampleKind::Signed => 0,
        },
        big_endian: format.byte_order == ByteOrder::Big,
        components,
        black,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct SourceWindow {
    start: u64,
    end: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SourceWindows([SourceWindow; 4]);

impl SourceWindows {
    /// Caller-buffer union, including aliased scalars. Encoder-owned copies pass None;
    /// their complete allocation is already charged by the input preparation plan.
    pub(crate) fn addressed_bytes_many<'a>(
        bindings: impl IntoIterator<Item = (Option<&'a wgpu::Buffer>, Self)>,
    ) -> Result<u64, EncodeError> {
        let mut allocations: Vec<(&wgpu::Buffer, Vec<SourceWindow>)> = Vec::new();
        for (buffer, windows) in bindings {
            let Some(buffer) = buffer else {
                continue;
            };
            if let Some((_, spans)) = allocations.iter_mut().find(|(known, _)| *known == buffer) {
                spans.extend(windows.0);
            } else {
                allocations.push((buffer, windows.0.to_vec()));
            }
        }
        allocations
            .into_iter()
            .try_fold(0u64, |mut total, (_, mut spans)| {
                spans.sort_unstable_by_key(|span| span.start);
                let mut end = 0;
                for span in spans {
                    if span.end > end {
                        total = total
                            .checked_add(span.end - span.start.max(end))
                            .ok_or(EncodeError::InvalidSource("source window size overflow"))?;
                        end = span.end;
                    }
                }
                Ok(total)
            })
    }

    pub(crate) fn merge(self, other: Self) -> Self {
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

    pub(crate) fn maximum_bytes(self) -> u64 {
        self.0
            .iter()
            .map(|window| window.end - window.start)
            .max()
            .unwrap_or(0)
    }

    // Alignment can make neighboring plane windows overlap. Count each addressed byte once;
    // unused bindings alias the first window and do not increase source exposure.
    pub(crate) fn addressed_bytes(self) -> Result<u64, EncodeError> {
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

    pub(crate) fn validate(self, limit: u64) -> Result<(), EncodeError> {
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

    pub(crate) fn rebase(
        self,
        components: &mut [SourceParams],
        absolute_offsets: [u64; 4],
    ) -> Result<(), EncodeError> {
        for (component, absolute) in components.iter_mut().zip(absolute_offsets) {
            let window = self.0[component.plane as usize];
            component.byte_offset = u32::try_from(absolute.checked_sub(window.start).ok_or(
                EncodeError::InvalidSource("source window address underflow"),
            )?)
            .map_err(|_| EncodeError::InvalidSource("source window address exceeds WGSL u32"))?;
        }
        Ok(())
    }

    pub(crate) fn entries(
        self,
        buffer: &wgpu::Buffer,
        bindings: [u32; 4],
    ) -> [wgpu::BindGroupEntry<'_>; 4] {
        std::array::from_fn(|index| {
            let window = if self.0[index].end == 0 {
                self.0[0]
            } else {
                self.0[index]
            };
            wgpu::BindGroupEntry {
                binding: bindings[index],
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: window.start,
                    size: NonZeroU64::new(window.end - window.start),
                }),
            }
        })
    }
}

#[derive(Clone)]
pub(crate) struct SourceLayout {
    pub(crate) spec: SourceSpec,
    pub(crate) full_windows: SourceWindows,
    offsets: [u64; 4],
    alignment: u64,
    extent: jxl_gpu_protocol::Extent2d,
    pub(crate) black: Option<Box<SourceLayout>>,
    pub(crate) cmyk_color: Option<Box<SourceLayout>>,
}

pub(crate) struct SourceRegion {
    pub(crate) components: [SourceParams; 4],
    pub(crate) offsets: [u64; 4],
    pub(crate) windows: SourceWindows,
}

impl SourceLayout {
    pub(crate) fn for_input(
        source: &crate::source_input::FrameInputPlan,
        alignment: u64,
    ) -> Result<Self, EncodeError> {
        Self::with_encoding(
            &source.layout,
            source.buffer_bytes(),
            alignment,
            source.cmyk_encoding(),
        )
    }
    #[cfg(test)]
    pub(crate) fn new(
        layout: &ImageLayout,
        buffer_bytes: u64,
        alignment: u64,
    ) -> Result<Self, EncodeError> {
        Self::with_encoding(
            layout,
            buffer_bytes,
            alignment,
            CmykSampleEncoding::default(),
        )
    }

    pub(crate) fn for_source(
        source: &crate::BufferImageSource,
        alignment: u64,
    ) -> Result<Self, EncodeError> {
        Self::with_encoding(
            &source.layout,
            source.buffer.size(),
            alignment,
            source.cmyk_encoding(),
        )
    }

    fn with_encoding(
        layout: &ImageLayout,
        buffer_bytes: u64,
        alignment: u64,
        encoding: CmykSampleEncoding,
    ) -> Result<Self, EncodeError> {
        let spec = source_spec(&layout.format)?.with_cmyk_encoding(encoding)?;
        // Public layout fields can be modified after construction. Revalidate every plane,
        // including extents, pitches, non-overlap, and the claimed logical allocation size.
        let checked =
            ImageLayout::from_planes(layout.extent, layout.format.clone(), layout.planes.clone())?;
        if checked != *layout {
            return Err(EncodeError::InvalidSource(
                "inconsistent logical source size",
            ));
        }
        for plane in &layout.planes {
            let end = align_up(plane.end_offset()?, 4)
                .ok_or(EncodeError::InvalidSource("source binding size overflow"))?;
            if end > buffer_bytes {
                return Err(EncodeError::InvalidSource(
                    "source binding does not contain the final addressable sample word",
                ));
            }
        }
        Self::view(layout, spec, alignment.max(4))
    }

    /// Bind only the physical planes addressed by this view. CMY/alpha and Black can
    /// therefore share a five-plane input without adding a fifth shader source binding.
    fn view(
        layout: &ImageLayout,
        mut spec: SourceSpec,
        alignment: u64,
    ) -> Result<Self, EncodeError> {
        let cmyk_color = spec
            .black
            .map(|black| {
                let mut components = spec.components;
                components[3] = black;
                Self::view(
                    layout,
                    SourceSpec {
                        format: SourceChannels::Rgba,
                        components,
                        black: None,
                        ..spec.clone()
                    },
                    alignment,
                )
                .map(Box::new)
            })
            .transpose()?;
        let black = spec
            .black
            .map(|component| {
                let mut components = [SourceParams::zeroed(); 4];
                components[0] = component;
                Self::view(
                    layout,
                    SourceSpec {
                        format: SourceChannels::Gray,
                        components,
                        black: None,
                        ..spec.clone()
                    },
                    alignment,
                )
                .map(Box::new)
            })
            .transpose()?;
        let mut planes: Vec<_> = spec.components[..spec.format.channel_count() as usize]
            .iter()
            .map(|component| component.plane as usize)
            .collect();
        planes.sort_unstable();
        planes.dedup();
        let mut full_windows = SourceWindows::default();
        for (binding, &index) in planes.iter().enumerate() {
            let plane = &layout.planes[index];
            let end = align_up(plane.end_offset()?, 4)
                .ok_or(EncodeError::InvalidSource("source binding size overflow"))?;
            full_windows.0[binding] = SourceWindow {
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
            component.plane = planes
                .binary_search(&(component.plane as usize))
                .expect("selected source plane") as u32;
        }
        Ok(Self {
            spec,
            full_windows,
            offsets,
            alignment,
            extent: layout.extent,
            black,
            cmyk_color,
        })
    }

    pub(crate) fn region(
        &self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<SourceRegion, EncodeError> {
        if width == 0
            || height == 0
            || x.checked_add(width)
                .is_none_or(|end| end > self.extent.width)
            || y.checked_add(height)
                .is_none_or(|end| end > self.extent.height)
        {
            return Err(EncodeError::InvalidSource(
                "source region is outside the image",
            ));
        }
        let mut offsets = [0; 4];
        let mut windows = SourceWindows::default();
        for (index, component) in self
            .spec
            .components
            .iter()
            .take(self.spec.format.channel_count() as usize)
            .enumerate()
        {
            let start = self.offsets[index]
                .checked_add(u64::from(y) * u64::from(component.row_stride))
                .and_then(|value| {
                    value.checked_add(u64::from(x) * u64::from(component.pixel_stride))
                })
                .ok_or(EncodeError::InvalidSource("source group offset overflow"))?;
            let end = start
                .checked_add(u64::from(height - 1) * u64::from(component.row_stride))
                .and_then(|value| {
                    value.checked_add(u64::from(width - 1) * u64::from(component.pixel_stride))
                })
                .and_then(|value| value.checked_add(u64::from(component.word_bytes)))
                .and_then(|value| align_up(value, 4))
                .ok_or(EncodeError::InvalidSource("source group end overflow"))?;
            let mut component_window = SourceWindows::default();
            component_window.0[component.plane as usize] = SourceWindow {
                start: start - start % self.alignment,
                end,
            };
            windows = windows.merge(component_window);
            offsets[index] = start;
        }
        Ok(SourceRegion {
            components: self.spec.components,
            offsets,
            windows,
        })
    }
}

pub(crate) const SHADER: &str = include_str!("source.wgsl");

pub(crate) fn shader(source: &str) -> String {
    source.replace("/*__JXL_SOURCE__*/", SHADER)
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
                    let mut format = SourceChannels::Gray.pixel_format(bits).unwrap();
                    let mut fields = Vec::new();
                    if word_bits > bits + shift {
                        fields.push(PackingField::padding(word_bits - bits - shift));
                    }
                    fields.push(PackingField::channel(Channel::X, bits));
                    if shift != 0 {
                        fields.push(PackingField::padding(shift));
                    }
                    format.planes[0].words = vec![PackingWord { fields }];
                    let spec = source_spec(&format).unwrap();
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
        let source = SourceLayout::new(&layout, layout.logical_size + 3, 256).unwrap();
        let tile = source.region(0, 0, 256, 3).unwrap();
        assert!(
            tile.offsets
                .iter()
                .take(3)
                .all(|&offset| offset > u64::from(u32::MAX))
        );
        tile.windows.validate(1024).unwrap();
        assert_eq!(tile.windows.maximum_bytes(), 788);
        assert_eq!(tile.windows.addressed_bytes().unwrap(), 3 * 788);
        let mut components = tile.components;
        tile.windows
            .rebase(&mut components[..3], tile.offsets)
            .unwrap();
        assert!(components[..3].iter().all(|source| source.byte_offset == 3));
        assert!(tile.windows.validate(787).is_err());
    }

    #[test]
    fn regions_and_final_word_are_checked_before_gpu_addressing() {
        let layout = ImageLayout::packed(
            Extent2d::new(17, 9),
            SourceChannels::Rgb.pixel_format(8).unwrap(),
        )
        .unwrap();
        assert!(SourceLayout::new(&layout, layout.logical_size, 256).is_err());
        let source =
            SourceLayout::new(&layout, align_up(layout.logical_size, 4).unwrap(), 256).unwrap();
        for (x, y, w, h) in [
            (0, 0, 0, 1),
            (0, 0, 1, 0),
            (17, 0, 1, 1),
            (0, 9, 1, 1),
            (u32::MAX, 0, 2, 1),
            (0, u32::MAX, 1, 2),
        ] {
            assert!(source.region(x, y, w, h).is_err());
        }
        let mut tile = source.region(16, 8, 1, 1).unwrap();
        tile.windows.validate(256).unwrap();
        tile.windows
            .rebase(&mut tile.components[..3], tile.offsets)
            .unwrap();
        assert_eq!(tile.components[0].byte_offset, 200);
        assert_eq!(tile.components[2].byte_offset, 202);
        assert_eq!(tile.windows.addressed_bytes().unwrap(), 204);
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
        let source = SourceLayout::new(&layout, 12, 256).unwrap();
        assert_eq!(source.full_windows.addressed_bytes().unwrap(), 12);
        assert_eq!(source.full_windows.maximum_bytes(), 12);
    }
}
