//! Packing for profile-owned device components and independent alpha.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_formats::{
    ColorFormatClass, ColorSample, ColorSpecification, ColorStorage, ImageLayout, PixelFormatClass,
    classify_pixel_format,
};
use jxl_gpu_protocol::{Extent2d, OutputOrientation, icc::IccProfile};

use crate::{AlphaConversion, Error, ResidentIccPlane, ResidentIccSampleEncoding, Result};

pub const DEVICE_OUTPUT_SHADER: &str = concat!(
    include_str!("../shaders/image_orientation.wgsl"),
    "\n",
    include_str!("../shaders/alpha_output.wgsl"),
    "\n",
    include_str!("../shaders/device_output.wgsl"),
);

#[derive(Clone, Copy, Debug)]
pub struct DeviceOutputSource<'a> {
    pub profile: &'a IccProfile,
    pub extent: Extent2d,
    pub orientation: OutputOrientation,
    pub planes: &'a [ResidentIccPlane],
    pub alpha: Option<ResidentIccPlane>,
    pub input_words: u64,
    pub sample_encoding: ResidentIccSampleEncoding,
    pub alpha_conversion: AlphaConversion,
}

/// Validated 320-byte uniform. Source addressing uses F32 words; output addressing uses bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DeviceOutputParams {
    geometry: [u32; 4],
    channels: [u32; 4],
    mapping: [u32; 4],
    sizes: [u32; 4],
    input_offsets: [[u32; 4]; 4],
    input_strides: [[u32; 4]; 4],
    output_offsets: [[u32; 4]; 4],
    output_strides: [[u32; 4]; 4],
}

impl DeviceOutputParams {
    pub fn new(
        layout: &ImageLayout,
        source: DeviceOutputSource<'_>,
        dispatch_width: u32,
    ) -> Result<Self> {
        let invalid = || {
            Error::InvalidPayload(
                "ICC device output layout or source planes are inconsistent".into(),
            )
        };
        let PixelFormatClass::Color(ColorFormatClass::IccDevice {
            sample,
            storage,
            channels,
            alpha,
        }) = classify_pixel_format(&layout.format).map_err(|_| invalid())?
        else {
            return Err(invalid());
        };
        if !matches!(&layout.format.color_spec, ColorSpecification::Icc(profile) if profile == source.profile)
            || source.extent.is_empty()
            || source.orientation.map_extent(source.extent) != layout.extent
            || source.planes.len() != usize::from(channels)
            || source.input_words > u64::from(u32::MAX)
            || dispatch_width == 0
        {
            return Err(invalid());
        }
        let validated =
            ImageLayout::from_planes(layout.extent, layout.format.clone(), layout.planes.clone())?;
        if validated.logical_size != layout.logical_size
            || layout.logical_size > u64::from(u32::MAX - 3)
        {
            return Err(invalid());
        }
        let components = u32::from(channels) + u32::from(alpha);
        let bytes = if sample == ColorSample::F32 { 4 } else { 1 };
        let pixel_step = if storage == ColorStorage::Planar {
            bytes
        } else {
            components * bytes
        };
        let mut result = Self {
            geometry: [
                layout.extent.width,
                layout.extent.height,
                source.extent.width,
                source.extent.height,
            ],
            channels: [
                u32::from(channels),
                u32::from(alpha),
                bytes,
                source.orientation.to_exif_value() - 1,
            ],
            mapping: [
                u32::from(source.alpha.is_some()),
                source.sample_encoding as u32,
                source.alpha_conversion as u32,
                pixel_step,
            ],
            sizes: [layout.logical_size as u32, dispatch_width, 0, 0],
            input_offsets: [[0; 4]; 4],
            input_strides: [[0; 4]; 4],
            output_offsets: [[0; 4]; 4],
            output_strides: [[0; 4]; 4],
        };
        for (i, plane) in source
            .planes
            .iter()
            .chain(source.alpha.as_ref())
            .enumerate()
        {
            let end = u64::from(source.extent.height - 1)
                .checked_mul(u64::from(plane.stride))
                .and_then(|n| n.checked_add(u64::from(plane.offset)))
                .and_then(|n| n.checked_add(u64::from(source.extent.width)))
                .ok_or_else(invalid)?;
            if plane.stride < source.extent.width || end > source.input_words {
                return Err(invalid());
            }
            result.input_offsets[i / 4][i % 4] = plane.offset;
            result.input_strides[i / 4][i % 4] = plane.stride;
        }
        for (plane, packing) in layout.planes.iter().zip(&layout.format.planes) {
            for (position, word) in packing.words.iter().enumerate() {
                let i = match word.fields[0].kind {
                    jxl_gpu_formats::PackingFieldKind::Channel(
                        jxl_gpu_formats::Channel::Device(index),
                    ) => usize::from(index),
                    jxl_gpu_formats::PackingFieldKind::Channel(jxl_gpu_formats::Channel::Alpha) => {
                        usize::from(channels)
                    }
                    _ => return Err(invalid()),
                };
                let offset = plane.offset + position as u64 * u64::from(bytes);
                result.output_offsets[i / 4][i % 4] =
                    u32::try_from(offset).map_err(|_| invalid())?;
                result.output_strides[i / 4][i % 4] =
                    u32::try_from(plane.row_stride).map_err(|_| invalid())?;
            }
        }
        Ok(result)
    }
}

const _: () = {
    assert!(std::mem::size_of::<DeviceOutputParams>() == 320);
    assert!(std::mem::offset_of!(DeviceOutputParams, input_offsets) == 64);
    assert!(std::mem::offset_of!(DeviceOutputParams, output_strides) == 256);
};
