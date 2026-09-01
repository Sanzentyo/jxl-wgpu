//! Default resource table and LF dequantization for the bounded VarDCT profile.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::TransformKind;
use jxl_oxide_common::Bundle;
use jxl_vardct::{DequantMatrixSet, DequantMatrixSetParams, TransformType};
use jxl_wgpu::{KernelVariant, VAR_DCT_AFV_BASIS};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::vardct_frontend::VarDctChannelShift;

const RESOURCE_SHADER: &str = include_str!("vardct_resource.wgsl");

/// Vec4-indexed resource table consumed by the resident VarDCT renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctResourceLayout {
    pub quant_offset: u32,
    pub quant_count: u32,
    pub correlation_offset: u32,
    pub lf_offsets: [u32; 3],
    pub lf_strides: [u32; 3],
    pub lf_extents: [[u32; 2]; 3],
    pub matrix_offsets: [u32; TransformKind::ALL.len()],
    pub afv_basis_offset: u32,
    pub vector_count: u32,
    pub block_count: u32,
    pub correlation_count: u32,
}

impl VarDctResourceLayout {
    pub fn new(
        blocks_x: u32,
        blocks_y: u32,
        quant_count: u32,
    ) -> Result<Self, VarDctResourceError> {
        Self::with_channel_shifts(
            blocks_x,
            blocks_y,
            quant_count,
            [VarDctChannelShift::default(); 3],
        )
    }

    pub fn with_channel_shifts(
        blocks_x: u32,
        blocks_y: u32,
        quant_count: u32,
        channel_shifts: [VarDctChannelShift; 3],
    ) -> Result<Self, VarDctResourceError> {
        validate_channel_shifts(channel_shifts)?;
        let block_count =
            blocks_x
                .checked_mul(blocks_y)
                .ok_or(VarDctResourceError::ArithmeticOverflow {
                    field: "LF block count",
                })?;
        let correlation_count = blocks_x
            .div_ceil(8)
            .checked_mul(blocks_y.div_ceil(8))
            .ok_or(VarDctResourceError::ArithmeticOverflow {
                field: "correlation cell count",
            })?;
        let quant_offset = 0_u32;
        if quant_count == 0 {
            return Err(VarDctResourceError::ZeroQuantizationEntries);
        }
        let correlation_offset = quant_count;
        let lf_start = correlation_offset.checked_add(correlation_count).ok_or(
            VarDctResourceError::ArithmeticOverflow {
                field: "LF resource offset",
            },
        )?;
        let lf_extents = channel_shifts.map(|shift| {
            shift.shifted_extent(blocks_x, blocks_y).ok_or(
                VarDctResourceError::ArithmeticOverflow {
                    field: "shifted LF resource extent",
                },
            )
        });
        let lf_extents = [lf_extents[0]?, lf_extents[1]?, lf_extents[2]?];
        let lf_strides = lf_extents.map(|extent| extent[0]);
        let same_geometry = lf_extents
            .into_iter()
            .all(|extent| extent == [blocks_x, blocks_y]);
        let (lf_offsets, mut cursor) = if same_geometry {
            (
                [lf_start; 3],
                lf_start.checked_add(block_count).ok_or(
                    VarDctResourceError::ArithmeticOverflow {
                        field: "matrix offset",
                    },
                )?,
            )
        } else {
            let mut offsets = [0; 3];
            let mut cursor = lf_start;
            for (offset, [width, height]) in offsets.iter_mut().zip(lf_extents) {
                *offset = cursor;
                cursor = cursor
                    .checked_add(width.checked_mul(height).ok_or(
                        VarDctResourceError::ArithmeticOverflow {
                            field: "shifted LF resource channel size",
                        },
                    )?)
                    .ok_or(VarDctResourceError::ArithmeticOverflow {
                        field: "shifted LF resource channel end",
                    })?;
            }
            (offsets, cursor)
        };
        let mut matrix_offsets = [0; TransformKind::ALL.len()];
        for (index, transform) in TransformKind::ALL.into_iter().enumerate() {
            matrix_offsets[index] = cursor;
            let area =
                transform
                    .pixel_extent()
                    .area()
                    .ok_or(VarDctResourceError::ArithmeticOverflow {
                        field: "dequant matrix area",
                    })?;
            cursor = cursor
                .checked_add(u32::try_from(area).map_err(|_| {
                    VarDctResourceError::ArithmeticOverflow {
                        field: "dequant matrix area",
                    }
                })?)
                .ok_or(VarDctResourceError::ArithmeticOverflow {
                    field: "dequant matrix end",
                })?;
        }
        let afv_basis_offset = cursor;
        let basis_vectors = u32::try_from(VAR_DCT_AFV_BASIS.len() / 4).map_err(|_| {
            VarDctResourceError::ArithmeticOverflow {
                field: "AFV basis vectors",
            }
        })?;
        let vector_count = afv_basis_offset.checked_add(basis_vectors).ok_or(
            VarDctResourceError::ArithmeticOverflow {
                field: "resource vector count",
            },
        )?;
        Ok(Self {
            quant_offset,
            quant_count,
            correlation_offset,
            lf_offsets,
            lf_strides,
            lf_extents,
            matrix_offsets,
            afv_basis_offset,
            vector_count,
            block_count,
            correlation_count,
        })
    }

    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.vector_count as u64 * 16
    }

    /// Builds immutable correlation defaults, all normative default dequantization matrices, and
    /// the AFV basis. Per-task and per-channel quantization scales are populated by GPU artifact
    /// lowering from decoded `hf_mul` and the frame header.
    pub fn initial_values(self) -> Result<Vec<[f32; 4]>, VarDctResourceError> {
        let mut values = vec![[0.0; 4]; self.vector_count as usize];
        let correlation_end = self.correlation_offset + self.correlation_count;
        values[self.correlation_offset as usize..correlation_end as usize]
            .fill([0.0, 1.0, 0.0, 0.0]);
        let matrices = default_dequant_matrices()?;
        for (strategy, transform) in TransformKind::ALL.into_iter().enumerate() {
            let matrix_offset = self.matrix_offsets[strategy] as usize;
            let matrix = matrices.matrix(transform);
            values[matrix_offset..matrix_offset + matrix.len()].copy_from_slice(&matrix);
        }
        for (destination, basis) in values[self.afv_basis_offset as usize..]
            .iter_mut()
            .zip(VAR_DCT_AFV_BASIS.chunks_exact(4))
        {
            destination.copy_from_slice(basis);
        }
        Ok(values)
    }

    pub(crate) fn install_dequant_matrix_words(
        self,
        values: &mut [[f32; 4]],
        words: &[[u32; 4]],
    ) -> Result<(), VarDctResourceError> {
        self.validate_dequant_matrix_words(words)?;
        let start = self.matrix_offsets[0] as usize;
        let end = self.afv_basis_offset as usize;
        let destination =
            values
                .get_mut(start..end)
                .ok_or(VarDctResourceError::ArithmeticOverflow {
                    field: "dequant matrix resource range",
                })?;
        for (destination, source) in destination.iter_mut().zip(words) {
            *destination = source.map(f32::from_bits);
        }
        Ok(())
    }

    pub(crate) fn validate_dequant_matrix_words(
        self,
        words: &[[u32; 4]],
    ) -> Result<(), VarDctResourceError> {
        let expected = usize::try_from(
            self.afv_basis_offset
                .checked_sub(self.matrix_offsets[0])
                .ok_or(VarDctResourceError::ArithmeticOverflow {
                    field: "dequant matrix resource length",
                })?,
        )
        .map_err(|_| VarDctResourceError::ArithmeticOverflow {
            field: "dequant matrix resource length",
        })?;
        if words.len() != expected {
            return Err(VarDctResourceError::DequantMatrixVectorCount {
                expected,
                actual: words.len(),
            });
        }
        Ok(())
    }

    #[must_use]
    pub(crate) const fn dequant_matrix_byte_offset(self) -> u64 {
        self.matrix_offsets[0] as u64 * 16
    }
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum VarDctResourceError {
    #[error("VarDCT resource arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
    #[error("LF output stride {actual} is smaller than required extent {required}")]
    InvalidOutputStride { required: u32, actual: u32 },
    #[error(
        "VarDCT JPEG channel {channel} has invalid shift {horizontal}x{vertical}; each axis is at most one bit"
    )]
    InvalidChannelShift {
        channel: usize,
        horizontal: u32,
        vertical: u32,
    },
    #[error("VarDCT resource preparation requires at least one quantization entry")]
    ZeroQuantizationEntries,
    #[error("failed to construct the normative default VarDCT dequantization matrices")]
    DefaultDequantMatrices,
    #[error(
        "VarDCT dequantization matrix payload has {actual} vectors; expected exactly {expected}"
    )]
    DequantMatrixVectorCount { expected: usize, actual: usize },
    #[error("VarDCT resource preparation requires a linear workgroup, got {variant:?}")]
    WorkgroupShape { variant: KernelVariant },
    #[error("VarDCT resource workgroup variant {variant:?} exceeds device limits")]
    WorkgroupVariant { variant: KernelVariant },
}

/// Exact 144-byte LF preparation uniform.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VarDctResourceParams {
    pub geometry: [u32; 4],
    pub source_geometry: [[u32; 4]; 3],
    pub destination_geometry: [[u32; 4]; 3],
    pub scales: [f32; 4],
    pub correlation: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct VarDctResourceConfig {
    pub block_extent: [u32; 2],
    pub output_origin: [u32; 2],
    pub channel_shifts: [VarDctChannelShift; 3],
    pub lf_offsets: [u32; 3],
    pub lf_strides: [u32; 3],
    pub apply_chroma_from_luma: bool,
    pub global_scale: u32,
    pub quant_lf: u32,
    pub lf_dequantization: [f32; 3],
    pub lf_correlation: [f32; 2],
    pub extra_precision: u8,
}

impl VarDctResourceParams {
    pub fn new(config: VarDctResourceConfig) -> Result<Self, VarDctResourceError> {
        let VarDctResourceConfig {
            block_extent: [blocks_x, blocks_y],
            output_origin,
            channel_shifts,
            lf_offsets,
            lf_strides,
            apply_chroma_from_luma,
            global_scale,
            quant_lf,
            lf_dequantization,
            lf_correlation,
            extra_precision,
        } = config;
        validate_channel_shifts(channel_shifts)?;
        let blocks =
            blocks_x
                .checked_mul(blocks_y)
                .ok_or(VarDctResourceError::ArithmeticOverflow {
                    field: "LF preparation block count",
                })?;
        let output_extents = channel_shifts.map(|shift| {
            shift.shifted_extent(blocks_x, blocks_y).ok_or(
                VarDctResourceError::ArithmeticOverflow {
                    field: "shifted LF group extent",
                },
            )
        });
        let output_extents = [output_extents[0]?, output_extents[1]?, output_extents[2]?];
        let mut destination_geometry = [[0; 4]; 3];
        for (channel, destination) in destination_geometry.iter_mut().enumerate() {
            let shift = channel_shifts[channel];
            let origin = [
                output_origin[0] >> shift.horizontal,
                output_origin[1] >> shift.vertical,
            ];
            let required = origin[0].checked_add(output_extents[channel][0]).ok_or(
                VarDctResourceError::ArithmeticOverflow {
                    field: "LF output horizontal extent",
                },
            )?;
            let stride = lf_strides[channel];
            if stride < required {
                return Err(VarDctResourceError::InvalidOutputStride {
                    required,
                    actual: stride,
                });
            }
            *destination = [stride, origin[0], origin[1], lf_offsets[channel]];
        }
        let source_order = [1usize, 0, 2];
        let mut source_geometry = [[0; 4]; 3];
        for (channel, source) in source_geometry.iter_mut().enumerate() {
            let source_channel = source_order[channel];
            let [width, height] = output_extents[channel];
            *source = [
                width,
                height,
                u32::try_from(source_channel)
                    .ok()
                    .and_then(|channel| channel.checked_mul(blocks))
                    .ok_or(VarDctResourceError::ArithmeticOverflow {
                        field: "LF source channel offset",
                    })?,
                source_channel as u32,
            ];
        }
        let denominator = global_scale as f32 * quant_lf as f32;
        let precision_divisor = (1u32 << extra_precision) as f32;
        let coalesced_lf = lf_offsets.into_iter().all(|offset| offset == lf_offsets[0]);
        let reconstruction_mode = if apply_chroma_from_luma {
            1
        } else if coalesced_lf {
            2
        } else {
            0
        };
        Ok(Self {
            geometry: [blocks_x, blocks_y, blocks, reconstruction_mode],
            source_geometry,
            destination_geometry,
            scales: [
                512.0 * lf_dequantization[0] / denominator / precision_divisor,
                512.0 * lf_dequantization[1] / denominator / precision_divisor,
                512.0 * lf_dequantization[2] / denominator / precision_divisor,
                precision_divisor,
            ],
            correlation: [lf_correlation[0], lf_correlation[1], 0.0, 0.0],
        })
    }

    #[must_use]
    pub const fn dispatch_with_variant(self, variant: KernelVariant) -> u32 {
        self.geometry[2].div_ceil(variant.workgroup_size().0)
    }

    #[must_use]
    pub fn smoothing_thresholds(self) -> [f32; 3] {
        [
            self.scales[0] * self.scales[3],
            self.scales[1] * self.scales[3],
            self.scales[2] * self.scales[3],
        ]
    }
}

fn validate_channel_shifts(
    channel_shifts: [VarDctChannelShift; 3],
) -> Result<(), VarDctResourceError> {
    for (channel, shift) in channel_shifts.into_iter().enumerate() {
        if shift.horizontal > 1 || shift.vertical > 1 {
            return Err(VarDctResourceError::InvalidChannelShift {
                channel,
                horizontal: shift.horizontal,
                vertical: shift.vertical,
            });
        }
    }
    Ok(())
}

struct DefaultDequantMatrices(DequantMatrixSet);

impl DefaultDequantMatrices {
    fn matrix(&self, transform: TransformKind) -> Vec<[f32; 4]> {
        let transform_type = vardct_transform_type(transform);
        let channel = |index| {
            if transform.needs_transpose() {
                self.0.get_transposed(index, transform_type)
            } else {
                self.0.get(index, transform_type)
            }
        };
        let [x, y, b] = [channel(0), channel(1), channel(2)];
        let extent = transform.pixel_extent();
        let mut packed = vec![[0.0; 4]; x.len()];
        for frequency_y in 0..extent.height {
            for frequency_x in 0..extent.width {
                let raster = (frequency_y * extent.width + frequency_x) as usize;
                let packed_index = backend_matrix_index(transform, frequency_x, frequency_y);
                packed[packed_index] = [x[raster], y[raster], b[raster], 0.0];
            }
        }
        packed
    }
}

#[must_use]
pub(crate) const fn hf_matrix_param_index(transform: TransformKind) -> usize {
    match transform {
        TransformKind::Dct8 => 0,
        TransformKind::Hornuss => 1,
        TransformKind::Dct2x2 => 2,
        TransformKind::Dct4x4 => 3,
        TransformKind::Dct16x16 => 4,
        TransformKind::Dct32x32 => 5,
        TransformKind::Dct16x8 | TransformKind::Dct8x16 => 6,
        TransformKind::Dct32x8 | TransformKind::Dct8x32 => 7,
        TransformKind::Dct32x16 | TransformKind::Dct16x32 => 8,
        TransformKind::Dct4x8 | TransformKind::Dct8x4 => 9,
        TransformKind::Afv0 | TransformKind::Afv1 | TransformKind::Afv2 | TransformKind::Afv3 => 10,
        TransformKind::Dct64x64 => 11,
        TransformKind::Dct64x32 | TransformKind::Dct32x64 => 12,
        TransformKind::Dct128x128 => 13,
        TransformKind::Dct128x64 | TransformKind::Dct64x128 => 14,
        TransformKind::Dct256x256 => 15,
        TransformKind::Dct256x128 | TransformKind::Dct128x256 => 16,
    }
}

fn backend_matrix_index(transform: TransformKind, frequency_x: u32, frequency_y: u32) -> usize {
    let extent = transform.pixel_extent();
    let index = if transform.is_special() || extent.height < extent.width {
        frequency_y * extent.width + frequency_x
    } else {
        frequency_x * extent.height + frequency_y
    };
    index as usize
}

fn default_dequant_matrices() -> Result<DefaultDequantMatrices, VarDctResourceError> {
    let encoded_default = [1u8];
    let mut bitstream = jxl_bitstream::Bitstream::new(&encoded_default);
    let pool = jxl_threadpool::JxlThreadPool::none();
    let params = DequantMatrixSetParams::new(8, 1, None, None, &pool);
    DequantMatrixSet::parse(&mut bitstream, params)
        .map(DefaultDequantMatrices)
        .map_err(|_| VarDctResourceError::DefaultDequantMatrices)
}

pub(crate) const fn vardct_transform_type(transform: TransformKind) -> TransformType {
    match transform {
        TransformKind::Dct8 => TransformType::Dct8,
        TransformKind::Hornuss => TransformType::Hornuss,
        TransformKind::Dct2x2 => TransformType::Dct2,
        TransformKind::Dct4x4 => TransformType::Dct4,
        TransformKind::Dct16x16 => TransformType::Dct16,
        TransformKind::Dct32x32 => TransformType::Dct32,
        TransformKind::Dct16x8 => TransformType::Dct16x8,
        TransformKind::Dct8x16 => TransformType::Dct8x16,
        TransformKind::Dct32x8 => TransformType::Dct32x8,
        TransformKind::Dct8x32 => TransformType::Dct8x32,
        TransformKind::Dct32x16 => TransformType::Dct32x16,
        TransformKind::Dct16x32 => TransformType::Dct16x32,
        TransformKind::Dct4x8 => TransformType::Dct4x8,
        TransformKind::Dct8x4 => TransformType::Dct8x4,
        TransformKind::Afv0 => TransformType::Afv0,
        TransformKind::Afv1 => TransformType::Afv1,
        TransformKind::Afv2 => TransformType::Afv2,
        TransformKind::Afv3 => TransformType::Afv3,
        TransformKind::Dct64x64 => TransformType::Dct64,
        TransformKind::Dct64x32 => TransformType::Dct64x32,
        TransformKind::Dct32x64 => TransformType::Dct32x64,
        TransformKind::Dct128x128 => TransformType::Dct128,
        TransformKind::Dct128x64 => TransformType::Dct128x64,
        TransformKind::Dct64x128 => TransformType::Dct64x128,
        TransformKind::Dct256x256 => TransformType::Dct256,
        TransformKind::Dct256x128 => TransformType::Dct256x128,
        TransformKind::Dct128x256 => TransformType::Dct128x256,
    }
}

pub struct VarDctResourceBuffers<'a> {
    pub quantized_lf: &'a wgpu::Buffer,
    pub dequantized_lf: &'a wgpu::Buffer,
}

pub struct VarDctResourcePipeline {
    pipeline: wgpu::ComputePipeline,
    variant: KernelVariant,
}

fn validate_workgroup_variant(
    variant: KernelVariant,
    limits: &wgpu::Limits,
) -> Result<(), VarDctResourceError> {
    if !variant.is_linear() {
        return Err(VarDctResourceError::WorkgroupShape { variant });
    }
    variant
        .validate_for("vardct_resource", limits, 0)
        .map_err(|_| VarDctResourceError::WorkgroupVariant { variant })
}

impl VarDctResourcePipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, VarDctResourceError> {
        Self::with_variant(device, KernelVariant::Lanes64)
    }

    pub fn with_variant(
        device: &wgpu::Device,
        variant: KernelVariant,
    ) -> Result<Self, VarDctResourceError> {
        validate_workgroup_variant(variant, &device.limits())?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu VarDCT LF resource preparation"),
            source: wgpu::ShaderSource::Wgsl(RESOURCE_SHADER.into()),
        });
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        let constants = [
            ("wg_x", f64::from(workgroup_x)),
            ("wg_y", f64::from(workgroup_y)),
        ];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu VarDCT LF resource preparation"),
            layout: None,
            module: &module,
            entry_point: Some("prepare_lf"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
        Ok(Self { pipeline, variant })
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffers: VarDctResourceBuffers<'_>,
        params: VarDctResourceParams,
    ) -> wgpu::Buffer {
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu VarDCT LF resource params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu VarDCT LF resource bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, buffers.quantized_lf),
                entry(1, buffers.dequantized_lf),
                entry(2, &uniform),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu VarDCT LF resource preparation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(params.dispatch_with_variant(self.variant), 1, 1);
        drop(pass);
        uniform
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

const _: () = {
    assert!(std::mem::size_of::<VarDctResourceParams>() == 144);
    assert!(std::mem::align_of::<VarDctResourceParams>() == 16);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_and_shader_are_bounded() {
        let layout = VarDctResourceLayout::new(4, 2, 1).unwrap();
        assert_eq!(layout.correlation_count, 1);
        assert_eq!(layout.lf_offsets, [2; 3]);
        assert_eq!(layout.matrix_offsets[0], 10);
        for index in 1..TransformKind::ALL.len() {
            let previous_area = TransformKind::ALL[index - 1].pixel_extent().area().unwrap() as u32;
            assert_eq!(
                layout.matrix_offsets[index],
                layout.matrix_offsets[index - 1] + previous_area
            );
        }
        assert_eq!(
            layout.initial_values().unwrap().len(),
            layout.vector_count as usize
        );
        let module = naga::front::wgsl::parse_str(RESOURCE_SHADER).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn jpeg_component_shifts_are_bounded_before_layout_arithmetic() {
        let shifts = [
            VarDctChannelShift {
                horizontal: 2,
                vertical: 0,
            },
            VarDctChannelShift::default(),
            VarDctChannelShift::default(),
        ];
        assert_eq!(
            VarDctResourceLayout::with_channel_shifts(4, 2, 1, shifts).unwrap_err(),
            VarDctResourceError::InvalidChannelShift {
                channel: 0,
                horizontal: 2,
                vertical: 0,
            }
        );
    }

    #[test]
    fn correlation_grid_scales_past_one_frequency_cell() {
        let layout = VarDctResourceLayout::new(17, 9, 1).unwrap();
        assert_eq!(layout.correlation_count, 6);
        assert_eq!(layout.correlation_offset, 1);
        assert_eq!(layout.lf_offsets, [7; 3]);
        let values = layout.initial_values().unwrap();
        assert_eq!(&values[1..7], &[[0.0, 1.0, 0.0, 0.0]; 6]);
    }

    #[test]
    fn multiple_quantization_entries_shift_every_following_region() {
        let layout = VarDctResourceLayout::new(17, 9, 5).unwrap();
        assert_eq!(layout.quant_offset, 0);
        assert_eq!(layout.quant_count, 5);
        assert_eq!(layout.correlation_offset, 5);
        assert_eq!(layout.correlation_count, 6);
        assert_eq!(layout.lf_offsets, [11; 3]);
        assert_eq!(layout.matrix_offsets[0], 164);

        let values = layout.initial_values().unwrap();
        assert_eq!(&values[..5], &[[0.0; 4]; 5]);
        assert_eq!(&values[5..11], &[[0.0, 1.0, 0.0, 0.0]; 6]);
    }

    #[test]
    fn default_dct8_matrix_matches_normative_band_interpolation_samples() {
        let matrix = default_dequant_matrices()
            .unwrap()
            .matrix(TransformKind::Dct8);
        let expected = [
            [0.000_317_460_3, 0.001_785_714_3, 0.001_953_125],
            [0.000_745_078_5, 0.003_473_115_4, 0.016_986_076],
            [0.002_613_333_3, 0.005_100_178_5, 0.070_312_24],
        ];
        let samples = [
            matrix[backend_matrix_index(TransformKind::Dct8, 0, 0)],
            matrix[backend_matrix_index(TransformKind::Dct8, 7, 0)],
            matrix[backend_matrix_index(TransformKind::Dct8, 7, 7)],
        ];
        for (actual, expected) in samples.into_iter().zip(expected) {
            for (actual, expected) in actual[..3].iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= 5.0e-8,
                    "actual={actual:?} expected={expected:?}"
                );
            }
            assert_eq!(actual[3], 0.0);
        }
        assert_eq!(
            matrix[backend_matrix_index(TransformKind::Dct8, 1, 0)],
            matrix[backend_matrix_index(TransformKind::Dct8, 0, 1)],
        );
    }

    #[test]
    fn rectangular_default_matrices_follow_wire_transposition() {
        let matrices = default_dequant_matrices().unwrap();
        let tall = matrices.matrix(TransformKind::Dct16x8);
        let wide = matrices.matrix(TransformKind::Dct8x16);
        assert_eq!(tall.len(), wide.len());
        for y in 0..16 {
            for x in 0..8 {
                assert_eq!(
                    tall[backend_matrix_index(TransformKind::Dct16x8, x, y)],
                    wide[backend_matrix_index(TransformKind::Dct8x16, y, x)],
                );
            }
        }
    }

    #[test]
    fn tiled_workgroup_is_rejected_before_pipeline_creation() {
        assert_eq!(
            validate_workgroup_variant(KernelVariant::Tile8x8, &wgpu::Limits::default())
                .unwrap_err(),
            VarDctResourceError::WorkgroupShape {
                variant: KernelVariant::Tile8x8,
            }
        );
    }
}
