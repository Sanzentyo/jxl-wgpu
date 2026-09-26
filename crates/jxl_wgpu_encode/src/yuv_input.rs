//! Explicit YCbCr-to-RGB preparation, shared by every encoder frontend.
use std::num::NonZeroU64;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use jxl_gpu_formats::{
    ByteOrder, ChromaLocation, ChromaOrder, ColorFormatClass, ColorRange, ColorSpecification,
    ImageLayout, Packed422Order, PixelFormat, PixelFormatClass, RgbChannelOrder, TransferFunction,
    YcbcrEncoding, classify_pixel_format,
};
use wgpu::util::DeviceExt;

use crate::{BufferImageSource, EncodeError, UnsupportedFeature, WgpuContext};

/// RGB transfer after the explicitly requested GPU YCbCr conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YuvRgbTransfer {
    /// Preserve the source transfer. Requires a non-constant-luminance matrix and a
    /// transfer representable in JPEG XL. No transfer curve is evaluated or clipped.
    Preserve,
    /// Produce linear RGB in the source primaries. Supports Linear, sRGB/sYCC, BT.709,
    /// BT.2020, DCI and scene-linear HLG. PQ and arbitrary gamma require `Preserve`.
    /// No display OOTF, tone mapping, gamut mapping or alpha conversion is applied.
    Linear,
}

/// Explicit GPU conversion of a pitch-linear integer YCbCr buffer to RGB binary32.
///
/// Chroma is bilinearly reconstructed at the declared locations with edge replication.
/// Nominal-range excursions are retained. Modular preserves the resulting RGB words,
/// not the original subsampled YCbCr samples. Independent scalar attachments are unchanged.
#[derive(Clone, Debug)]
pub struct YuvImageSource {
    source: BufferImageSource,
    transfer: YuvRgbTransfer,
    output_format: PixelFormat,
}

impl YuvImageSource {
    pub fn new(source: BufferImageSource, transfer: YuvRgbTransfer) -> Result<Self, EncodeError> {
        let plan = YuvPlan::new(&source, transfer)?;
        Ok(Self {
            source,
            transfer,
            output_format: plan.layout.format,
        })
    }

    #[must_use]
    pub fn source(&self) -> &BufferImageSource {
        &self.source
    }

    #[must_use]
    pub fn rgb_transfer(&self) -> YuvRgbTransfer {
        self.transfer
    }

    #[must_use]
    pub fn pixel_format(&self) -> &PixelFormat {
        &self.output_format
    }

    /// Attach independent scalars in the image declaration's order.
    pub fn with_extra_channels(
        mut self,
        channels: Vec<BufferImageSource>,
    ) -> Result<Self, EncodeError> {
        self.source = self.source.with_extra_channels(channels)?;
        Ok(self)
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct YuvParams {
    width: u32,
    height: u32,
    kind: u32,
    order: u32,
    bits: u32,
    storage_bytes: u32,
    big_endian: u32,
    matrix: u32,
    limited: u32,
    subsample_x: u32,
    subsample_y: u32,
    linear: u32,
    siting_x: f32,
    siting_y: f32,
    transfer: u32,
    reserved: u32,
    planes: [[u32; 4]; 3],
}

pub(crate) struct YuvPlan {
    pub(crate) layout: ImageLayout,
    pub(crate) source_bytes: u64,
    params: YuvParams,
}

const _: () = {
    assert!(size_of::<YuvParams>() == 112);
    assert!(align_of::<YuvParams>() == 4);
};

impl YuvPlan {
    pub(crate) fn new(
        source: &BufferImageSource,
        transfer: YuvRgbTransfer,
    ) -> Result<Self, EncodeError> {
        let input = &source.layout;
        let checked =
            ImageLayout::from_planes(input.extent, input.format.clone(), input.planes.clone())?;
        let source_bytes =
            checked
                .logical_size
                .checked_add(3)
                .ok_or(EncodeError::InvalidSource(
                    "YUV source binding size overflow",
                ))?
                & !3;
        if checked != *input
            || input.extent.width == 0
            || input.extent.height == 0
            || source_bytes > source.buffer.size()
            || !source.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(EncodeError::InvalidSource(
                "YUV conversion requires a canonical nonempty storage layout with word padding",
            ));
        }
        // Classification owns packing semantics. This kernel also implements big-endian words.
        let mut packing = input.format.clone();
        if !matches!(
            packing.swizzle,
            jxl_gpu_formats::Swizzle::XYZ0 | jxl_gpu_formats::Swizzle::XYZ1
        ) {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        packing.byte_order = ByteOrder::Little;
        let class = classify_pixel_format(&packing).map_err(|_| UnsupportedFeature::InputFormat)?;
        let (kind, order, bits, storage_bits) = match class {
            PixelFormatClass::Color(ColorFormatClass::YuvPlanar {
                bits, storage_bits, ..
            }) => (0, 0, bits, storage_bits),
            PixelFormatClass::Color(ColorFormatClass::YuvSemiplanar {
                bits,
                storage_bits,
                chroma_order,
                ..
            }) => (
                1,
                u32::from(chroma_order == ChromaOrder::CrCb),
                bits,
                storage_bits,
            ),
            PixelFormatClass::Color(ColorFormatClass::Yuv422Packed { order }) => {
                (2, u32::from(order == Packed422Order::Uyvy), 8, 8)
            }
            _ => return Err(UnsupportedFeature::InputFormat.into()),
        };
        let ColorSpecification::Defined(color) = input.format.color_spec else {
            return Err(UnsupportedFeature::InputFormat.into());
        };
        let matrix = match color.encoding {
            YcbcrEncoding::Bt601 => 0,
            YcbcrEncoding::Bt709 => 1,
            YcbcrEncoding::Bt2020 => 2,
            YcbcrEncoding::Bt2020ConstantLuminance
                if color.space == jxl_gpu_formats::ColorSpace::Bt2020
                    && color.transfer == TransferFunction::Bt2020
                    && transfer == YuvRgbTransfer::Linear =>
            {
                3
            }
            _ => return Err(UnsupportedFeature::InputFormat.into()),
        };
        // These curves are finite for every possible unsigned YUV code, including excursions.
        // PQ's pole and arbitrary gamma powers cannot promise that without pixel validation;
        // those retain their encoded domain through Preserve instead.
        let transfer_id = if transfer == YuvRgbTransfer::Preserve {
            0
        } else {
            match color.transfer {
                TransferFunction::Linear => 0,
                TransferFunction::Srgb | TransferFunction::Sycc => 1,
                TransferFunction::Bt709 => 2,
                TransferFunction::Hlg => 4,
                TransferFunction::Bt2020 => 5,
                TransferFunction::Dci => 7,
                _ => return Err(UnsupportedFeature::InputFormat.into()),
            }
        };
        let mut rgb_color = color;
        rgb_color.encoding = YcbcrEncoding::Undefined;
        rgb_color.range = ColorRange::Full;
        rgb_color.chroma_location = jxl_gpu_formats::ChromaLocation2d::BOTH;
        if transfer == YuvRgbTransfer::Linear {
            rgb_color.transfer = TransferFunction::Linear;
        }
        let output = PixelFormat::rgb_f32(
            RgbChannelOrder::Rgb,
            false,
            ColorSpecification::Defined(rgb_color),
        );
        crate::source_color::SourceColorEncoding::from_format(&output)?;
        let layout = ImageLayout::packed(input.extent, output)?;
        let (sx, sy) = input
            .format
            .chroma_subsampling
            .chroma_divisors()
            .ok_or(UnsupportedFeature::InputFormat)?;
        fn siting(location: ChromaLocation, divisor: u8) -> Result<f32, EncodeError> {
            if divisor == 1 {
                return Ok(0.0);
            }
            match location {
                ChromaLocation::Even => Ok(0.0),
                ChromaLocation::Center => Ok(f32::from(divisor - 1) / 2.0),
                ChromaLocation::Odd => Ok(f32::from(divisor - 1)),
                ChromaLocation::Both => Err(UnsupportedFeature::InputFormat.into()),
            }
        }
        let word_address = |value| {
            u32::try_from(value)
                .map_err(|_| EncodeError::InvalidSource("YUV preparation address exceeds u32"))
        };
        word_address(source_bytes)?;
        word_address(layout.logical_size)?;
        let mut planes = [[0; 4]; 3];
        for (target, plane) in planes.iter_mut().zip(&input.planes) {
            *target = [
                word_address(plane.offset)?,
                word_address(plane.row_stride)?,
                0,
                0,
            ];
        }
        let params = YuvParams {
            width: input.extent.width,
            height: input.extent.height,
            kind,
            order,
            bits: u32::from(bits),
            storage_bytes: u32::from(storage_bits / 8),
            big_endian: u32::from(input.format.byte_order == ByteOrder::Big),
            matrix,
            limited: u32::from(color.range == ColorRange::Limited),
            subsample_x: u32::from(sx),
            subsample_y: u32::from(sy),
            linear: u32::from(transfer == YuvRgbTransfer::Linear),
            siting_x: siting(color.chroma_location.horizontal, sx)?,
            siting_y: siting(color.chroma_location.vertical, sy)?,
            transfer: transfer_id,
            reserved: 0,
            planes,
        };
        Ok(Self {
            layout,
            source_bytes,
            params,
        })
    }

    pub(crate) fn owned_bytes(&self) -> u64 {
        self.layout.logical_size + size_of::<YuvParams>() as u64
    }

    pub(crate) fn largest_allocation(&self) -> u64 {
        self.layout.logical_size.max(size_of::<YuvParams>() as u64)
    }

    pub(crate) fn requires_unassociated_alpha(&self) -> bool {
        self.params.linear != 0 && self.params.transfer != 0
    }

    pub(crate) fn validate_limits(&self, limits: &wgpu::Limits) -> Result<(), EncodeError> {
        for (name, required, available) in [
            (
                "max_storage_buffer_binding_size",
                self.source_bytes.max(self.layout.logical_size),
                limits.max_storage_buffer_binding_size,
            ),
            (
                "max_compute_workgroups_per_dimension",
                u64::from(
                    self.params
                        .width
                        .div_ceil(16)
                        .max(self.params.height.div_ceil(16)),
                ),
                u64::from(limits.max_compute_workgroups_per_dimension),
            ),
            (
                "max_uniform_buffer_binding_size",
                size_of::<YuvParams>() as u64,
                limits.max_uniform_buffer_binding_size,
            ),
            (
                "max_compute_invocations_per_workgroup",
                256,
                u64::from(limits.max_compute_invocations_per_workgroup),
            ),
            (
                "max_compute_workgroup_size_x",
                16,
                u64::from(limits.max_compute_workgroup_size_x),
            ),
            (
                "max_compute_workgroup_size_y",
                16,
                u64::from(limits.max_compute_workgroup_size_y),
            ),
        ] {
            if required > available {
                return Err(UnsupportedFeature::DeviceLimit {
                    name,
                    required,
                    available,
                }
                .into());
            }
        }
        Ok(())
    }

    pub(crate) fn materialize(
        &self,
        context: &WgpuContext,
        input: YuvImageSource,
        output: &wgpu::Buffer,
    ) -> PreparedYuv {
        let pipeline = context.yuv_pipeline();
        let uniform = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("YUV input parameters"),
                contents: bytemuck::bytes_of(&self.params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("YUV input preparation"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &input.source.buffer,
                            offset: 0,
                            size: NonZeroU64::new(self.source_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
        PreparedYuv {
            _input: input,
            _uniform: uniform,
            pipeline: pipeline.clone(),
            bind_group,
            grid: [
                self.params.width.div_ceil(16),
                self.params.height.div_ceil(16),
            ],
        }
    }
}

pub(crate) fn pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("YUV input conversion"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                "{}\n{}",
                jxl_wgpu::IMAGE_TRANSFER_SHADER,
                include_str!("yuv_input.wgsl")
            )
            .into(),
        ),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("YUV input conversion"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

pub(crate) struct PreparedYuv {
    _input: YuvImageSource,
    _uniform: wgpu::Buffer,
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group: wgpu::BindGroup,
    grid: [u32; 2],
}

impl PreparedYuv {
    pub(crate) fn record(&self, commands: &mut wgpu::CommandEncoder) {
        let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("prepare RGB from YUV input"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid[0], self.grid[1], 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuv_shader_has_a_portable_checked_uniform_abi() {
        let text = format!(
            "{}\n{}",
            jxl_wgpu::IMAGE_TRANSFER_SHADER,
            include_str!("yuv_input.wgsl")
        );
        let module = naga::front::wgsl::parse_str(&text).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        let (_, ty) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("Params"))
            .unwrap();
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            panic!("uniform struct");
        };
        assert_eq!(*span, size_of::<YuvParams>() as u32);
        assert_eq!(
            members.iter().map(|m| m.offset).collect::<Vec<_>>(),
            (0..=16).map(|i| i * 4).collect::<Vec<_>>()
        );
    }

    #[test]
    fn yuv_preparation_checks_allocation_binding_and_dispatch_limits() {
        let layout = ImageLayout::packed(
            jxl_gpu_protocol::Extent2d::new(17, 33),
            PixelFormat::rgb_f32(RgbChannelOrder::Rgb, false, ColorSpecification::Default),
        )
        .unwrap();
        let plan = YuvPlan {
            layout,
            source_bytes: 8192,
            params: YuvParams {
                width: 17,
                height: 33,
                ..YuvParams::zeroed()
            },
        };
        let defaults = wgpu::Limits::default();
        plan.validate_limits(&defaults).unwrap();
        for (name, limits) in [
            (
                "max_storage_buffer_binding_size",
                wgpu::Limits {
                    max_storage_buffer_binding_size: 8191,
                    ..defaults.clone()
                },
            ),
            (
                "max_compute_workgroups_per_dimension",
                wgpu::Limits {
                    max_compute_workgroups_per_dimension: 2,
                    ..defaults.clone()
                },
            ),
            (
                "max_uniform_buffer_binding_size",
                wgpu::Limits {
                    max_uniform_buffer_binding_size: 111,
                    ..defaults.clone()
                },
            ),
            (
                "max_compute_invocations_per_workgroup",
                wgpu::Limits {
                    max_compute_invocations_per_workgroup: 255,
                    ..defaults.clone()
                },
            ),
            (
                "max_compute_workgroup_size_x",
                wgpu::Limits {
                    max_compute_workgroup_size_x: 15,
                    ..defaults.clone()
                },
            ),
            (
                "max_compute_workgroup_size_y",
                wgpu::Limits {
                    max_compute_workgroup_size_y: 15,
                    ..defaults
                },
            ),
        ] {
            assert!(
                matches!(plan.validate_limits(&limits),Err(EncodeError::Unsupported(UnsupportedFeature::DeviceLimit { name:actual,.. })) if actual==name)
            );
        }
    }
}
