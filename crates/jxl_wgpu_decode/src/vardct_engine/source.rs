use std::num::NonZeroU64;

use jxl_gpu_bitstream::{
    ColourEncodingInventory, ColourSpaceInventory, PrimariesInventory, TransferFunctionInventory,
    WhitePointInventory,
};
use jxl_gpu_formats::ImageLayout;
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{
    KernelVariant, ResidentChromaUpsampleMemoryPlan, ResidentEpfMemoryPlan,
    ResidentGaborishWeights, ResidentImageUpsampleWeights, ResidentVarDctMemoryPlan,
    UpsamplingFactor, WgpuBackend,
};

use crate::entropy_window::MIN_STREAM_WINDOW_BYTES;
use crate::progressive_dc::ProgressiveDcXybPlanes;
use crate::vardct_artifact::{
    HfMetadataArtifactConfig, HfMetadataLoweringParams, VarDctArtifactDeviceLimits,
    VarDctArtifactLayout,
};
use crate::vardct_frontend::{
    AdaptiveLfPlan, StandardVarDctProfile, VarDctChannelShift, VarDctColorTransform,
};
use crate::vardct_output::{
    PackedU8Format, VarDctInverseOpsin, VarDctOutputPlan, VarDctOutputTransform,
};
use crate::vardct_packet::{BoundedVarDctPacketPlan, VarDctModularParams, VarDctPacketControl};
use crate::vardct_pass_group::{HfCoefficientExecutionPlan, HfCoefficientGroupExecutionPlan};
use crate::vardct_resource::{VarDctResourceConfig, VarDctResourceLayout, VarDctResourceParams};
use crate::{GpuCodestream, GpuOutputMapping, GpuOutputRequest};

use super::restoration::{VarDctEpfPlan, dequant_matrix_multiplier, restoration_config};
use super::types::{
    ADAPTIVE_LF_WORKGROUP_BYTES, DeferredHfCoefficientLayout, PACKET_STATUS_BYTES,
    VarDctDecodeError, VarDctDecodeMemoryInputs, VarDctDecodeMemoryStats,
};
use super::window_plan::{
    AdaptiveStreamLimitDecision, CombinedPacketWindowExecutionPlan, LfPacketWindowExecutionPlan,
    VarDctEntropyPlanSelection, select_budget_adaptive_stream_limit,
};

pub(super) struct VarDctSource {
    pub(super) codestream: GpuCodestream,
    pub(super) packet: BoundedVarDctPacketPlan,
    pub(super) groups: Vec<VarDctGroupSource>,
    pub(super) lf_packet_windows: Option<LfPacketWindowExecutionPlan>,
    pub(super) combined_packet_windows: Option<CombinedPacketWindowExecutionPlan>,
    pub(super) stream_limit: u64,
    pub(super) resource_layout: VarDctResourceLayout,
    pub(super) hf_coefficients: Option<HfCoefficientExecutionPlan>,
    pub(super) deferred_hf: Option<DeferredHfCoefficientLayout>,
    pub(super) gaborish: Option<ResidentGaborishWeights>,
    pub(super) epf: Option<VarDctEpfPlan>,
    pub(super) frame_upsampling: Option<ResidentImageUpsampleWeights>,
    pub(super) output_plan: VarDctOutputPlan,
    pub(super) output_format: PackedU8Format,
    pub(super) layout: ImageLayout,
    pub(super) output_transform: VarDctOutputTransform,
    pub(super) quant_biases: [f32; 4],
    pub(super) frame_name: String,
    pub(super) memory: VarDctDecodeMemoryStats,
    pub(super) adaptive_lf: AdaptiveLfPlan,
    pub(super) external_lf: Option<ProgressiveDcXybPlanes>,
}

impl VarDctSource {
    #[must_use]
    pub(super) const fn adaptive_lf_smoothing(&self) -> bool {
        self.adaptive_lf.executes()
    }

    pub(super) fn staged_lf_submission_count(&self) -> usize {
        if self.packet.profile.uses_lf_frame() {
            0
        } else {
            self.lf_packet_windows
                .as_ref()
                .map_or(1, LfPacketWindowExecutionPlan::batch_count)
        }
    }

    pub(super) fn submissions_per_frame(&self) -> usize {
        if self.deferred_hf.is_some() {
            if self.packet.pending_raw_hf_dequant_side_image().is_some()
                && !self.packet.requires_local_tree_staging()
            {
                // The final AC/render submission is known up front. Each raw side image adds its
                // own submission as the resumable HF-global parser discovers it.
                return 1;
            }
            if self.packet.profile.uses_lf_frame() {
                return 2;
            }
            let coefficient_batches = self.hf_coefficients.as_ref().map_or(0, |coefficients| {
                if coefficients.uses_bounded_stream_windows() {
                    coefficients.stream_batch_count()
                } else {
                    0
                }
            });
            return self
                .staged_lf_submission_count()
                .saturating_add(coefficient_batches)
                .saturating_add(2);
        }
        let local_lf = if self.packet.requires_local_tree_staging() {
            self.lf_packet_windows
                .as_ref()
                .map_or(1, LfPacketWindowExecutionPlan::batch_count)
        } else {
            0
        };
        let combined_packet_extra = self
            .combined_packet_windows
            .as_ref()
            .map_or(0, |plan| plan.batch_count().saturating_sub(1));
        if let Some(coefficients) = &self.hf_coefficients
            && coefficients.uses_bounded_stream_windows()
        {
            // Packet/pre-coefficient work, one ordered submission per reusable upload, then the
            // resident inverse-transform/render/status tail. Local-tree frames additionally map
            // their LF cursors before this sequence.
            local_lf + combined_packet_extra + 2 + coefficients.stream_batch_count()
        } else {
            local_lf + combined_packet_extra + 1
        }
    }
}

pub(super) struct VarDctGroupSource {
    pub(super) control: VarDctPacketControl,
    pub(super) resource_params: VarDctResourceParams,
    pub(super) artifact_layout: VarDctArtifactLayout,
    pub(super) artifact_params: HfMetadataLoweringParams,
    pub(super) quant_offset: u32,
}

#[derive(Clone, Copy)]
pub(super) struct VarDctPrepareOptions {
    pub(super) output_variant: KernelVariant,
    pub(super) stream_window_limit: Option<NonZeroU64>,
    pub(super) memory_limit_bytes: u64,
    pub(super) progressive_dc_final: Option<bool>,
}

pub(super) fn prepare_source(
    backend: &WgpuBackend,
    codestream: GpuCodestream,
    request: &GpuOutputRequest,
    inventory: &jxl_gpu_bitstream::CodestreamInventory,
    options: VarDctPrepareOptions,
) -> Result<VarDctSource, VarDctDecodeError> {
    let output_format = match request.mapping() {
        GpuOutputMapping::Color => PackedU8Format::try_from(request.format())?,
        _ => return Err(VarDctDecodeError::UnsupportedOutput),
    };
    if inventory.image_header.orientation != 1 {
        return Err(VarDctDecodeError::UnsupportedOrientation {
            orientation: inventory.image_header.orientation,
        });
    }
    if !matches!(
        inventory.image_header.colour_encoding,
        ColourEncodingInventory::Enumerated {
            colour_space: ColourSpaceInventory::Rgb,
            white_point: WhitePointInventory::D65,
            primaries: PrimariesInventory::Srgb,
            transfer_function: TransferFunctionInventory::Srgb,
            rendering_intent: _,
        }
    ) {
        return Err(VarDctDecodeError::UnsupportedColorEncoding);
    }
    let frame = inventory
        .frames
        .first()
        .ok_or(VarDctDecodeError::MissingFrame)?;
    let (gaborish, epf_header) = restoration_config(frame.restoration_filter)?;
    let dequant_channel_scales = [
        dequant_matrix_multiplier("X", frame.x_qm_scale)?,
        1.0,
        dequant_matrix_multiplier("B", frame.b_qm_scale)?,
    ];
    let profile = options.progressive_dc_final.map_or_else(
        || StandardVarDctProfile::negotiate(inventory),
        |is_final| StandardVarDctProfile::negotiate_progressive_dc(inventory, is_final),
    )?;
    let packet = BoundedVarDctPacketPlan::parse_source(&codestream, profile)?;
    let adaptive_lf = packet.profile.adaptive_lf();
    let deferred_hf = DeferredHfCoefficientLayout::plan(&packet)?;
    let codestream_bytes = codestream.logical_bytes();
    let codestream_len =
        usize::try_from(codestream_bytes).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
            field: "codestream source length",
        })?;
    let staged_local_trees = packet.requires_local_tree_staging();
    let limits = backend.device().limits();
    let configured_stream_limit = options
        .stream_window_limit
        .map_or(u64::MAX, NonZeroU64::get)
        .min(limits.max_buffer_size)
        .min(limits.max_storage_buffer_binding_size);
    if configured_stream_limit < MIN_STREAM_WINDOW_BYTES {
        return Err(VarDctDecodeError::EntropyStreamWindowTooSmall {
            limit_bytes: configured_stream_limit,
            minimum_bytes: MIN_STREAM_WINDOW_BYTES,
        });
    }
    let [blocks_x, blocks_y] = packet.block_extent();
    let resource_layout = VarDctResourceLayout::with_channel_shifts(
        blocks_x,
        blocks_y,
        packet.total_task_capacity()?,
        packet.profile.channel_shifts(),
    )?;
    let correlation_width = packet.profile.width().div_ceil(64);
    let pass_group_dim_blocks = packet.profile.group_dimension().checked_div(8).ok_or(
        VarDctDecodeError::ArithmeticOverflow {
            field: "pass-group block dimension",
        },
    )?;
    let mut quant_offset = resource_layout.quant_offset;
    let mut groups = Vec::with_capacity(packet.groups.len());
    for packet_group in &packet.groups {
        let control = if let Some(continuation) = &packet_group.external_lf_hf {
            packet_group.hf_stage_control(&packet, continuation)?
        } else if staged_local_trees {
            packet_group.lf_stage_control(&packet)?
        } else {
            packet_group.packet_control(&packet)?
        };
        let [group_blocks_x, group_blocks_y] = packet_group.block_extent();
        let block_origin = [packet_group.rect.x / 8, packet_group.rect.y / 8];
        let lf_offsets = if adaptive_lf.executes() {
            [0; 3]
        } else {
            resource_layout.lf_offsets
        };
        let resource_params = VarDctResourceParams::new(VarDctResourceConfig {
            block_extent: [group_blocks_x, group_blocks_y],
            output_origin: block_origin,
            channel_shifts: packet.profile.channel_shifts(),
            lf_offsets,
            lf_strides: resource_layout.lf_strides,
            apply_chroma_from_luma: packet.profile.uses_chroma_from_luma(),
            global_scale: packet.global_scale,
            quant_lf: packet.quant_lf,
            lf_dequantization: packet.lf_dequantization.multipliers,
            lf_correlation: packet.lf_correlation.lf_slopes(),
            extra_precision: packet_group.extra_precision,
        })?;
        let group_correlation_width = packet_group.rect.width.div_ceil(64);
        let group_correlation_height = packet_group.rect.height.div_ceil(64);
        let correlation_origin = [packet_group.rect.x / 64, packet_group.rect.y / 64];
        let correlation_offset = correlation_origin[1]
            .checked_mul(correlation_width)
            .and_then(|offset| offset.checked_add(correlation_origin[0]))
            .and_then(|offset| resource_layout.correlation_offset.checked_add(offset))
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "LF-group correlation resource offset",
            })?;
        let artifact_config = HfMetadataArtifactConfig {
            blocks_width: group_blocks_x,
            blocks_height: group_blocks_y,
            block_info_entries: packet_group.task_capacity,
            strategy_offset_words: control.offsets[2],
            hf_mul_offset_words: control.offsets[3],
            raw_metadata_words: u64::from(control.capacities[1]),
            pass_group_dim_blocks,
            lf_stride: blocks_x,
            correlation_stride: correlation_width,
            correlation_width: group_correlation_width,
            correlation_height: group_correlation_height,
            destination_origin: [packet_group.rect.x, packet_group.rect.y],
            afv_basis_offset: resource_layout.afv_basis_offset,
            quant_offset,
            correlation_offset,
            global_scale: packet.global_scale,
            channel_shifts: packet.profile.channel_shifts(),
            lf_offsets: resource_layout.lf_offsets,
            lf_strides: resource_layout.lf_strides,
            matrix_offsets: resource_layout.matrix_offsets,
        };
        let artifact_layout = VarDctArtifactLayout::plan(
            &artifact_config,
            VarDctArtifactDeviceLimits::from_wgpu(&backend.device().limits()),
        )?;
        let artifact_params = HfMetadataLoweringParams::new(
            &artifact_config,
            artifact_layout,
            dequant_channel_scales,
            packet.lf_correlation.hf_params(),
        )?;
        groups.push(VarDctGroupSource {
            control,
            resource_params,
            artifact_layout,
            artifact_params,
            quant_offset,
        });
        quant_offset = quant_offset.checked_add(packet_group.task_capacity).ok_or(
            VarDctDecodeError::ArithmeticOverflow {
                field: "LF-group quantization resource range",
            },
        )?;
    }
    debug_assert_eq!(quant_offset, resource_layout.correlation_offset);
    let epf = epf_header
        .map(|header| header.plan(&packet, &groups, packet.global_scale))
        .transpose()?;
    let epf_sigma_memory = epf
        .as_ref()
        .and_then(|plan| plan.sigma_groups.first())
        .map(|config| config.plan())
        .transpose()?;
    let artifacts = groups
        .iter()
        .map(|group| group.artifact_layout)
        .collect::<Vec<_>>();
    let frame_upsampling = match packet.profile.upsampling() {
        1 => None,
        factor => {
            let factor = UpsamplingFactor::try_from(factor).map_err(|_| {
                VarDctDecodeError::EngineContract {
                    detail: "parsed frame upsampling factor is not 1, 2, 4, or 8",
                }
            })?;
            let compact = match factor {
                UpsamplingFactor::X2 => inventory
                    .image_header
                    .upsampling_weights
                    .up2
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
                UpsamplingFactor::X4 => inventory
                    .image_header
                    .upsampling_weights
                    .up4
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
                UpsamplingFactor::X8 => inventory
                    .image_header
                    .upsampling_weights
                    .up8
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
            };
            Some(ResidentImageUpsampleWeights::new(factor, &compact)?)
        }
    };
    let output_plan = VarDctOutputPlan::for_limits_with_variant(
        packet.profile.presentation_width(),
        packet.profile.presentation_height(),
        output_format,
        &backend.device().limits(),
        options.output_variant,
    )?;
    let layout = ImageLayout::packed(
        Extent2d::new(
            packet.profile.presentation_width(),
            packet.profile.presentation_height(),
        ),
        output_format.pixel_format(),
    )?;
    let (output_transform, quant_biases) = match packet.profile.color_transform() {
        VarDctColorTransform::Xyb => {
            let opsin = inventory
                .image_header
                .opsin_inverse_matrix
                .ok_or(VarDctDecodeError::MissingInverseOpsin)?;
            (
                VarDctOutputTransform::Xyb(VarDctInverseOpsin {
                    opsin_bias: opsin.opsin_bias.map(|value| value.to_f32()),
                    inverse_opsin_matrix: opsin
                        .inverse_matrix
                        .map(|row| row.map(|value| value.to_f32())),
                    intensity_target: inventory
                        .image_header
                        .tone_mapping
                        .intensity_target
                        .to_f32(),
                }),
                [
                    opsin.quant_bias[0].to_f32(),
                    opsin.quant_bias[1].to_f32(),
                    opsin.quant_bias[2].to_f32(),
                    opsin.quant_bias_numerator.to_f32(),
                ],
            )
        }
        VarDctColorTransform::Ycbcr => {
            let channel_shifts =
                if gaborish.is_some() || epf.is_some() || frame_upsampling.is_some() {
                    [VarDctChannelShift::default(); 3]
                } else {
                    packet.profile.channel_shifts()
                };
            (
                VarDctOutputTransform::Ycbcr { channel_shifts },
                // Non-XYB image metadata omits the optional opsin object that otherwise carries these
                // TransformData defaults, but VarDCT coefficient biasing still uses their exact F32
                // roundings.
                [
                    1.0 - 0.054_650_072,
                    1.0 - 0.070_054_5,
                    1.0 - 0.049_935_102,
                    0.145,
                ],
            )
        }
    };
    let resident_memory = packet
        .groups
        .iter()
        .map(|group| ResidentVarDctMemoryPlan::new(group.coefficient_words()))
        .collect::<Result<Vec<_>, _>>()?;
    let plan_at_limit =
        |stream_limit: u64| -> Result<VarDctEntropyPlanSelection, VarDctDecodeError> {
            let lf_packet_windows = staged_local_trees
                .then(|| LfPacketWindowExecutionPlan::new(codestream_bytes, &packet, stream_limit))
                .transpose()?
                .flatten();
            let combined_packet_windows = (!staged_local_trees
                && !packet.profile.uses_lf_frame()
                && packet.pending_raw_hf_dequant_side_image().is_none())
            .then(|| {
                CombinedPacketWindowExecutionPlan::new(codestream_bytes, &packet, stream_limit)
            })
            .transpose()?
            .flatten();
            let hf_coefficients = packet
                .hf_coefficients
                .as_ref()
                .map(|entropy| {
                    HfCoefficientExecutionPlan::new(
                        &packet,
                        entropy,
                        &artifacts,
                        codestream_bytes,
                        stream_limit,
                    )
                })
                .transpose()?;
            let deferred_hf_plan = if combined_packet_windows.is_some() {
                None
            } else {
                deferred_hf.as_ref()
            };
            let memory = VarDctDecodeMemoryStats::plan(VarDctDecodeMemoryInputs {
                stream_limit,
                codestream_len,
                packet: &packet,
                groups: &groups,
                lf_packet_windows: lf_packet_windows.as_ref(),
                combined_packet_windows: combined_packet_windows.as_ref(),
                resource: resource_layout,
                hf_coefficients: hf_coefficients.as_ref(),
                deferred_hf: deferred_hf_plan,
                adaptive_lf_smoothing: adaptive_lf.executes(),
                restoration_scratch: gaborish.is_some() || epf.is_some(),
                gaborish: gaborish.is_some(),
                epf_sigma: epf_sigma_memory,
                epf_iterations: epf.as_ref().map_or(0, |plan| plan.passes.len() as u32),
                frame_upsampling: frame_upsampling.as_ref(),
                resident: &resident_memory,
                output: output_plan,
            })?;
            Ok(VarDctEntropyPlanSelection {
                stream_limit,
                lf_packet_windows,
                combined_packet_windows,
                hf_coefficients,
                memory,
            })
        };
    let selected_stream_limit = match select_budget_adaptive_stream_limit(
        configured_stream_limit,
        options.memory_limit_bytes,
        |stream_limit| Ok(plan_at_limit(stream_limit)?.memory.into()),
    )? {
        AdaptiveStreamLimitDecision::Selected(stream_limit) => stream_limit,
        AdaptiveStreamLimitDecision::BudgetTooSmall { required_bytes } => {
            return Err(VarDctDecodeError::MemoryBudgetTooSmall {
                required_bytes,
                limit_bytes: options.memory_limit_bytes,
            });
        }
    };
    let entropy_plan = plan_at_limit(selected_stream_limit)?;
    let VarDctEntropyPlanSelection {
        stream_limit,
        lf_packet_windows,
        combined_packet_windows,
        hf_coefficients,
        memory,
    } = entropy_plan;
    let deferred_hf = if combined_packet_windows.is_some() {
        None
    } else {
        deferred_hf
    };
    if packet.requires_hf_global_staging()
        && combined_packet_windows.is_none()
        && !packet.profile.uses_lf_frame()
    {
        for (packet_group, group) in packet.groups.iter().zip(&mut groups) {
            group.control = packet_group.lf_stage_control(&packet)?;
        }
    }
    validate_device_limits(
        backend.device(),
        memory,
        &packet,
        &groups,
        hf_coefficients.as_ref(),
    )?;
    let frame_name = packet.profile.frame_name().to_string();
    Ok(VarDctSource {
        codestream,
        packet,
        groups,
        lf_packet_windows,
        combined_packet_windows,
        stream_limit,
        resource_layout,
        hf_coefficients,
        deferred_hf,
        gaborish,
        epf,
        frame_upsampling,
        output_plan,
        output_format,
        layout,
        output_transform,
        quant_biases,
        frame_name,
        memory,
        adaptive_lf,
        external_lf: None,
    })
}

fn validate_device_limits(
    device: &wgpu::Device,
    memory: VarDctDecodeMemoryStats,
    packet: &BoundedVarDctPacketPlan,
    groups: &[VarDctGroupSource],
    hf_coefficients: Option<&HfCoefficientExecutionPlan>,
) -> Result<(), VarDctDecodeError> {
    let limits = device.limits();
    let group_count =
        u64::try_from(groups.len()).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
            field: "LF-group device-limit count",
        })?;
    if group_count == 0 {
        return Err(VarDctDecodeError::GroupPlanCount {
            component: "device-limit source",
            expected: 1,
            actual: 0,
        });
    }
    let predictor_capacity = packet.needs_self_correcting || packet.requires_local_tree_staging();
    let mut reconstruction_storage_bytes = 0_u64;
    for (index, group) in packet.groups.iter().enumerate() {
        let reconstruction = u64::from(group.reconstructed_words(predictor_capacity)?)
            .checked_mul(4)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "LF-group reconstruction binding bytes",
            })?;
        let lz77 = hf_coefficients
            .and_then(|plan| plan.groups.get(index))
            .map_or(0, HfCoefficientGroupExecutionPlan::lz77_scratch_bytes);
        let execution_state = hf_coefficients
            .and_then(|plan| plan.groups.get(index))
            .map_or(0, HfCoefficientGroupExecutionPlan::execution_state_bytes);
        reconstruction_storage_bytes = reconstruction_storage_bytes.max(
            reconstruction
                .checked_add(lz77)
                .and_then(|bytes| bytes.checked_add(execution_state))
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "LF-group reconstruction binding bytes",
                })?,
        );
    }
    let raw_metadata_bytes = groups
        .iter()
        .map(|group| u64::from(group.control.capacities[1]) * 4)
        .max()
        .unwrap_or(0);
    let coefficient_bytes = packet
        .groups
        .iter()
        .map(|group| u64::from(group.coefficient_words()) * 4)
        .max()
        .unwrap_or(0);
    let artifact_bytes = groups
        .iter()
        .map(|group| group.artifact_layout.artifact_bytes)
        .max()
        .unwrap_or(0);
    let occupancy_bytes = groups
        .iter()
        .map(|group| group.artifact_layout.occupancy_bytes)
        .max()
        .unwrap_or(0);
    let hf_params_bytes = memory.hf_params_bytes;
    let hf_status_bytes = hf_coefficients
        .map(|plan| {
            plan.groups
                .iter()
                .map(HfCoefficientGroupExecutionPlan::status_bytes)
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0);
    let modular_metadata_binding_bytes =
        if packet.requires_local_tree_staging() || packet.profile.uses_lf_frame() {
            packet
                .groups
                .iter()
                .map(|group| group.lf_modular.metadata.len() as u64 * 4)
                .max()
                .unwrap_or(0)
        } else {
            memory.modular_metadata_bytes
        };
    for (resource, required, storage) in [
        ("codestream upload", memory.codestream_bytes, true),
        ("Modular metadata", modular_metadata_binding_bytes, true),
        (
            "LF reconstruction and HF LZ77 storage",
            reconstruction_storage_bytes,
            true,
        ),
        ("raw HF metadata", raw_metadata_bytes, true),
        ("coefficients", coefficient_bytes, true),
        ("packet status", PACKET_STATUS_BYTES, true),
        (
            "Modular parameters",
            std::mem::size_of::<VarDctModularParams>() as u64,
            true,
        ),
        (
            "local-tree packet entropy stream window",
            memory.packet_stream_window_bytes,
            true,
        ),
        ("validation staging", memory.validation_staging_bytes, false),
        ("LF temporary", memory.lf_temporary_bytes, true),
        ("VarDCT resources", memory.resource_bytes, true),
        ("VarDCT artifact", artifact_bytes, true),
        ("artifact occupancy", occupancy_bytes, true),
        ("HF entropy bundle", memory.hf_entropy_bundle_bytes, true),
        (
            "HF entropy stream window",
            memory.hf_stream_window_bytes,
            true,
        ),
        ("HF pass-group parameters", hf_params_bytes, true),
        ("HF pass-group status", hf_status_bytes, true),
        (
            "HF coefficient order table",
            memory.hf_order_table_bytes,
            true,
        ),
        (
            "one resident component plane",
            memory.resident_plane_bytes.into_iter().max().unwrap_or(0),
            true,
        ),
        (
            "one component upsample plane",
            if memory.component_upsample_bytes == 0 {
                0
            } else {
                let shifted = packet
                    .profile
                    .channel_shifts()
                    .into_iter()
                    .filter(|shift| shift.is_subsampled())
                    .count() as u64;
                memory
                    .component_upsample_bytes
                    .checked_div(shifted)
                    .unwrap_or(0)
            },
            true,
        ),
        (
            "one restoration scratch plane",
            memory.restoration_scratch_bytes / 3,
            true,
        ),
        (
            "one frame upsample plane",
            memory.frame_upsample_image_bytes / 3,
            true,
        ),
        (
            "frame upsample weights",
            memory.frame_upsample_weight_bytes,
            true,
        ),
        ("EPF sigma plane", memory.epf_sigma_bytes, true),
        ("packed RGB8 output", memory.output_lease_bytes, true),
    ] {
        check_limit(resource, required, limits.max_buffer_size)?;
        if storage {
            check_limit(resource, required, limits.max_storage_buffer_binding_size)?;
        }
    }
    let epf_sigma_uniform_bytes = if memory.epf_sigma_uniform_bytes == 0 {
        0
    } else {
        memory.epf_sigma_uniform_bytes / group_count
    };
    for (resource, required) in [
        (
            "packet control uniform",
            std::mem::size_of::<VarDctPacketControl>() as u64,
        ),
        (
            "LF resource uniform",
            std::mem::size_of::<VarDctResourceParams>() as u64,
        ),
        ("adaptive LF uniform", memory.adaptive_lf_uniform_bytes),
        (
            "artifact uniform",
            std::mem::size_of::<HfMetadataLoweringParams>() as u64,
        ),
        (
            "HF coefficient sink uniform",
            if hf_coefficients.is_some() {
                std::mem::size_of::<crate::vardct_artifact::HfCoefficientSinkParams>() as u64
            } else {
                0
            },
        ),
        ("Gaborish uniform", memory.gaborish_uniform_bytes),
        (
            "component upsample uniform",
            if memory.component_upsample_uniform_bytes == 0 {
                0
            } else {
                ResidentChromaUpsampleMemoryPlan::UNIFORM_BYTES
            },
        ),
        ("EPF sigma uniform", epf_sigma_uniform_bytes),
        (
            "one EPF filter uniform",
            if memory.epf_filter_uniform_bytes == 0 {
                0
            } else {
                ResidentEpfMemoryPlan::UNIFORM_BYTES
            },
        ),
        (
            "frame upsample uniform",
            memory.frame_upsample_uniform_bytes,
        ),
        ("output uniform", memory.output_uniform_bytes),
    ] {
        check_limit(resource, required, limits.max_uniform_buffer_binding_size)?;
    }
    check_limit(
        "adaptive LF workgroup storage",
        ADAPTIVE_LF_WORKGROUP_BYTES,
        u64::from(limits.max_compute_workgroup_storage_size),
    )?;
    check_limit(
        "adaptive LF workgroup invocations",
        16 * 16,
        u64::from(limits.max_compute_invocations_per_workgroup),
    )?;
    check_limit(
        "adaptive LF workgroup Y size",
        16,
        u64::from(limits.max_compute_workgroup_size_y),
    )?;
    Ok(())
}

pub(super) fn check_limit(
    resource: &'static str,
    required: u64,
    available: u64,
) -> Result<(), VarDctDecodeError> {
    if required > available {
        return Err(VarDctDecodeError::DeviceLimit {
            resource,
            required,
            available,
        });
    }
    Ok(())
}
