use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use jxl_gpu_formats::{
    ChromaOrder, ColorFormatClass, ColorRange, ColorSpecification, ImageLayout, Packed422Order,
    PixelFormat, PixelFormatClass, RgbChannelOrder, RgbStorage, SampleKind, TransferFunction,
    classify_pixel_format,
};
use jxl_gpu_protocol::{Extent2d, SubmissionToken};
use jxl_wgpu::{
    GpuBufferLease, KernelVariant, ResidentStorageBinding, SubmissionPollPermit, WgpuBackend,
};

use crate::buffer_pool::{DecodeBufferLease, DecodeBufferPool};
use crate::entropy::EntropyStreamParams;
use crate::entropy_window::{
    GroupEntropyRange, GroupStreamSegment, MIN_STREAM_WINDOW_BYTES, StreamBatch,
    build_stream_batches_for_len as build_entropy_stream_batches,
};
use crate::model::native_modular_format;
use crate::modular_finalize::{
    ModularFinalizeBindings, ModularFinalizeOutput, ModularFinalizeParams, ModularFinalizeRegion,
};
use crate::modular_inverse::{ModularInverseJob, ModularInversePlan};
use crate::modular_palette::ModularPaletteWeightedParams;
use crate::modular_rct::{ModularRctArena, ModularRctParams};
use crate::modular_squeeze::{ModularSqueezeArena, ModularSqueezeParams};
use crate::modular_tree::EntropyCoderIr;
use crate::profile::{
    ModularGroup, ResidentModularFramePlan, ResidentModularGroupPlan, StandardModularProfile,
};
use crate::progressive_dc::{
    ProgressiveDcConvertInputs, ProgressiveDcPipeline, ProgressiveDcXybPlanes,
};
use crate::{
    Error, F64OutputPolicy, GpuOutputMapping, GpuOutputRequest, ModularPredictor,
    NumericSampleMapping, Result, WgpuPendingFrame,
};

use super::lifetime::{DecodeJobLifetime, DecodeMemoryPermits, DecodeSource, MapCompletion};
use super::pipeline::{reconstruction_specialization, uses_generalized_channel_layout};
use super::types::{
    DecodeStatus, DispatchControl, ENTROPY_EXECUTION_STATE_BYTES, F64OutputPath,
    GENERIC_PREDICTOR_EXECUTION_STATE_BYTES, GENERIC_WEIGHTED_EXECUTION_STATE_BYTES,
    ModularEntropyCoding, ModularInversePipelines, ModularOutputSpecialization,
    ModularReconstructionSpecialization, NATIVE_F64_DUMMY_WORD_BYTES, OutputWritePath,
    STATUS_BYTES, ShaderParams, WATCHDOG_PARALLEL_GROUP_LANE_CAP, WgpuDecodeCapabilities,
    WgpuDecodeMemoryStats,
};
#[derive(Clone, Debug)]
pub(super) struct GroupDispatchLayout {
    pub(super) group_workgroup_size: u32,
    pub(super) reconstruction_lane_stride: u64,
    pub(super) execution_state_bytes_per_lane: u64,
    pub(super) entropy_coding: ModularEntropyCoding,
    pub(super) max_logical_reconstruction_sample_words: u32,
    pub(super) max_physical_reconstruction_sample_words: u32,
    pub(super) resident_modular_arena_bytes: u64,
    pub(super) frame_modular_arena_bytes: u64,
    pub(super) global_reconstruction_sample_words: u32,
    pub(super) low_frequency_group_stream_count: usize,
    pub(super) progressive_pass_count: u32,
    pub(super) inverse_transform_count: usize,
    pub(super) palette_dispatch_count: usize,
    pub(super) inverse_transform_uniform_bytes: u64,
    pub(super) final_output_uniform_bytes: u64,
    pub(super) max_lz77_window_words: u32,
    pub(super) max_lz77_scratch_words: u32,
    pub(super) parallel_group_lanes: usize,
    pub(super) reconstructed_bytes: u64,
    pub(super) global_stream_segments: Arc<[GroupStreamSegment]>,
    pub(super) global_stream_batches: Arc<[StreamBatch]>,
    pub(super) stream_segments: Arc<[GroupStreamSegment]>,
    pub(super) stream_batches: Arc<[StreamBatch]>,
    pub(super) stream_bytes: u64,
    pub(super) status_stride: u64,
    pub(super) status_bytes: u64,
    pub(super) params_stride: u64,
    pub(super) params_bytes: u64,
    pub(super) output_write_path: OutputWritePath,
    pub(super) output_specialization: ModularOutputSpecialization,
    pub(super) reconstruction_specialization: ModularReconstructionSpecialization,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct GroupDispatchOptions {
    pub(super) requested_frame_slots: usize,
    pub(super) memory_limit_bytes: u64,
    pub(super) kernel_variant: KernelVariant,
    pub(super) stream_window_limit: Option<NonZeroU64>,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FixedGradientOutputMode {
    FinalizePass = 0,
    DirectNormalizedGray8 = 1,
    CompactNormalizedGray8 = 2,
    ResidentOnly = 3,
    CursorContinuation = 4,
}

impl GroupDispatchLayout {
    pub(super) fn new(
        device: &wgpu::Device,
        codestream_bytes: u64,
        profile: &StandardModularProfile,
        modular_metadata: &[u32],
        output: &OutputPlan,
        options: GroupDispatchOptions,
    ) -> Result<Self> {
        let generalized_channels = uses_generalized_channel_layout(profile);
        let output_write_path = if generalized_channels {
            OutputWritePath::AtomicBytes
        } else {
            output.write_path_for_groups(&profile.groups)?
        };
        let reconstruction_specialization = reconstruction_specialization(profile);
        let needs_self_correcting = profile.resident_entropy_plans.iter().any(|plan| {
            plan.ma_config
                .resolve(&profile.ma_config)
                .needs_self_correcting()
        }) || (profile.global_stream.is_some()
            && profile.resident_frame_plan.as_ref().is_some_and(|plan| {
                plan.ma_config
                    .resolve(&profile.ma_config)
                    .needs_self_correcting()
            }));
        let execution_state_bytes_per_lane =
            modular_execution_state_bytes(reconstruction_specialization, needs_self_correcting);
        let mut saw_prefix = false;
        let mut saw_ans = false;
        for plan in &profile.resident_entropy_plans {
            match plan.ma_config.resolve(&profile.ma_config).entropy.coder {
                EntropyCoderIr::Prefix(_) => saw_prefix = true,
                EntropyCoderIr::Ans { .. } => saw_ans = true,
            }
        }
        if let (Some(frame_plan), Some(_)) =
            (profile.resident_frame_plan.as_ref(), profile.global_stream)
        {
            match frame_plan
                .ma_config
                .resolve(&profile.ma_config)
                .entropy
                .coder
            {
                EntropyCoderIr::Prefix(_) => saw_prefix = true,
                EntropyCoderIr::Ans { .. } => saw_ans = true,
            }
        }
        let entropy_coding = match (saw_prefix, saw_ans) {
            (true, false) => ModularEntropyCoding::Prefix,
            (false, true) => ModularEntropyCoding::Ans,
            (true, true) => ModularEntropyCoding::Mixed,
            (false, false) => {
                return Err(Error::EngineContract(
                    "Modular frame has no group entropy descriptors",
                ));
            }
        };
        let limits = device.limits();
        let group_workgroup_size = options.kernel_variant.workgroup_size().0;
        let mut reconstruction_lane_stride = 0u64;
        let mut max_logical_reconstruction_sample_words = 0u32;
        let mut max_physical_reconstruction_sample_words = 0u32;
        let mut max_lz77_window_words = 0u32;
        let mut max_lz77_scratch_words = 0u32;
        let output_mode = if profile.progressive_dc.is_some() {
            FixedGradientOutputMode::ResidentOnly
        } else {
            fixed_gradient_output_mode(
                profile.channels.count(),
                profile.bits_per_sample,
                output,
                reconstruction_specialization,
            )
        };
        let output_specialization = match output_mode {
            FixedGradientOutputMode::FinalizePass => ModularOutputSpecialization::FinalizePass,
            FixedGradientOutputMode::DirectNormalizedGray8
            | FixedGradientOutputMode::CompactNormalizedGray8 => {
                ModularOutputSpecialization::DirectNormalizedGray8
            }
            FixedGradientOutputMode::ResidentOnly => ModularOutputSpecialization::FinalizePass,
            FixedGradientOutputMode::CursorContinuation => {
                ModularOutputSpecialization::FinalizePass
            }
        };
        let resident_modular_arena_bytes = if generalized_channels {
            profile
                .resident_entropy_plans
                .iter()
                .map(|plan| plan.inverse_plan.arena_bytes())
                .max()
                .unwrap_or(0)
        } else {
            0
        };
        let frame_workspace = profile
            .resident_frame_plan
            .as_ref()
            .map(|plan| frame_arena_workspace(profile, plan, execution_state_bytes_per_lane))
            .transpose()?;
        let frame_modular_arena_bytes = frame_workspace.map_or(0, |workspace| workspace.bytes);
        let global_reconstruction_sample_words =
            frame_workspace.map_or(0, |workspace| workspace.decoded_words);
        let inverse_transform_count = if generalized_channels {
            profile
                .resident_entropy_plans
                .iter()
                .flat_map(|plan| plan.inverse_plan.jobs())
                .chain(
                    profile
                        .resident_frame_plan
                        .iter()
                        .flat_map(|plan| plan.inverse_plan.jobs()),
                )
                .try_fold(0usize, |total, job| {
                    let dispatches = match job {
                        ModularInverseJob::Palette { job } => job.dispatch_count() as usize,
                        ModularInverseJob::Squeeze { .. } | ModularInverseJob::Rct { .. } => 1,
                    };
                    total.checked_add(dispatches)
                })
                .ok_or_else(|| Error::backend("Modular inverse dispatch count overflow"))?
        } else {
            0
        };
        let palette_dispatch_count = if generalized_channels {
            profile
                .resident_entropy_plans
                .iter()
                .flat_map(|plan| plan.inverse_plan.jobs())
                .chain(
                    profile
                        .resident_frame_plan
                        .iter()
                        .flat_map(|plan| plan.inverse_plan.jobs()),
                )
                .try_fold(0usize, |total, job| {
                    let dispatches = match job {
                        ModularInverseJob::Palette { job } => job.dispatch_count() as usize,
                        ModularInverseJob::Squeeze { .. } | ModularInverseJob::Rct { .. } => 0,
                    };
                    total.checked_add(dispatches)
                })
                .ok_or_else(|| Error::backend("Modular Palette dispatch count overflow"))?
        } else {
            0
        };
        let inverse_transform_uniform_bytes = if generalized_channels {
            profile
                .resident_entropy_plans
                .iter()
                .flat_map(|plan| plan.inverse_plan.jobs())
                .chain(
                    profile
                        .resident_frame_plan
                        .iter()
                        .flat_map(|plan| plan.inverse_plan.jobs()),
                )
                .try_fold(0u64, |total, job| {
                    let bytes = match job {
                        ModularInverseJob::Squeeze { .. } => {
                            std::mem::size_of::<ModularSqueezeParams>() as u64
                        }
                        ModularInverseJob::Rct { .. } => {
                            std::mem::size_of::<ModularRctParams>() as u64
                        }
                        ModularInverseJob::Palette { job } => job.uniform_bytes(),
                    };
                    total.checked_add(bytes)
                })
                .ok_or_else(|| Error::backend("Modular inverse uniform size overflow"))?
        } else {
            0
        };
        let final_output_uniform_bytes = if generalized_channels && profile.progressive_dc.is_none()
        {
            (std::mem::size_of::<ModularFinalizeParams>() as u64)
                .checked_mul(
                    u64::try_from(if profile.resident_frame_plan.is_some() {
                        1
                    } else {
                        profile.groups.len()
                    })
                    .map_err(|_| {
                        Error::backend("Modular group count exceeds uniform accounting")
                    })?,
                )
                .ok_or_else(|| Error::backend("Modular finalizer uniform size overflow"))?
        } else {
            0
        };
        for (group_index, group) in profile.entropy_groups.iter().copied().enumerate() {
            let decoded_symbol_count = group_decoded_symbol_count(profile, group_index, group)?;
            let lz77_window_words =
                group_lz77_window_words(profile, group_index, group, decoded_symbol_count)?;
            let physical_lz77_words = lz77_scratch_words(lz77_window_words);
            let group_output_mode =
                refine_fixed_gradient_output_mode(output_mode, lz77_window_words);
            let physical_sample_words =
                if group_output_mode == FixedGradientOutputMode::CompactNormalizedGray8 {
                    compact_gray8_sample_words(group)?
                } else if uses_generalized_channel_layout(profile) {
                    resident_entropy_plan(profile, group_index)?
                        .inverse_plan
                        .arena_words()
                } else {
                    decoded_symbol_count
                };
            max_logical_reconstruction_sample_words =
                max_logical_reconstruction_sample_words.max(decoded_symbol_count);
            max_physical_reconstruction_sample_words =
                max_physical_reconstruction_sample_words.max(physical_sample_words);
            max_lz77_window_words = max_lz77_window_words.max(lz77_window_words);
            max_lz77_scratch_words = max_lz77_scratch_words.max(physical_lz77_words);
            let group_bytes = group_reconstructed_bytes(
                profile,
                group_index,
                group,
                decoded_symbol_count,
                physical_sample_words,
                execution_state_bytes_per_lane,
            )?;
            let lane_alignment = if generalized_channels && profile.entropy_groups.len() > 1 {
                u64::from(limits.min_storage_buffer_offset_alignment).max(4)
            } else {
                4
            };
            reconstruction_lane_stride = reconstruction_lane_stride.max(align_to(
                group_bytes,
                lane_alignment,
                "reconstruction lane",
            )?);
        }
        if reconstruction_lane_stride == 0 {
            return Err(Error::backend("Modular reconstruction lane is empty"));
        }
        let device_stream_limit = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let stream_limit = options
            .stream_window_limit
            .map_or(device_stream_limit, |limit| {
                device_stream_limit.min(limit.get())
            });
        if stream_limit < MIN_STREAM_WINDOW_BYTES {
            return Err(Error::StreamWindowTooSmall {
                limit_bytes: stream_limit,
                minimum_bytes: MIN_STREAM_WINDOW_BYTES,
            });
        }
        let group_count = u64::try_from(profile.entropy_groups.len())
            .map_err(|_| Error::backend("Modular group count exceeds u64"))?;
        let (global_stream_segments, global_stream_batches, global_stream_bytes) =
            profile.global_stream.map_or_else(
                || Ok((Vec::new(), Vec::new(), 0)),
                |stream| {
                    build_entropy_stream_batches(
                        codestream_bytes,
                        &[GroupEntropyRange {
                            token_bit_offset: stream.token_bit_offset,
                            token_bit_end: stream.token_bit_end,
                        }],
                        stream_limit,
                        1,
                    )
                },
            )?;
        let status_record_count = group_count
            .checked_add(u64::from(profile.global_stream.is_some()))
            .ok_or_else(|| Error::backend("Modular status record count overflow"))?;
        let status_stride = STATUS_BYTES;
        let status_bytes = status_stride
            .checked_mul(status_record_count)
            .ok_or_else(|| Error::backend("Modular status buffer size overflow"))?;
        let params_stride = std::mem::size_of::<ShaderParams>() as u64;
        let params_bytes = params_stride
            .checked_mul(status_record_count)
            .ok_or_else(|| Error::backend("Modular parameter buffer size overflow"))?;
        let fixed_bytes = [
            modular_metadata_bytes(modular_metadata)?,
            align4(output.layout.logical_size)?,
            if output.f64_output_path == Some(F64OutputPath::NativeArithmetic) {
                NATIVE_F64_DUMMY_WORD_BYTES
            } else {
                0
            },
            status_bytes,
            status_bytes,
            params_bytes,
            std::mem::size_of::<DispatchControl>() as u64,
            inverse_transform_uniform_bytes,
            final_output_uniform_bytes,
            frame_modular_arena_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| Error::backend("parallel Modular fixed memory size overflow"))?;
        let device_lane_cap = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / reconstruction_lane_stride;
        let device_lane_cap = usize::try_from(device_lane_cap).unwrap_or(usize::MAX);
        let workgroup_cap = u64::from(limits.max_compute_workgroups_per_dimension)
            .checked_mul(u64::from(group_workgroup_size))
            .and_then(|lanes| usize::try_from(lanes).ok())
            .unwrap_or(usize::MAX);
        let lane_cap = WATCHDOG_PARALLEL_GROUP_LANE_CAP
            .min(profile.entropy_groups.len())
            .min(device_lane_cap)
            .min(workgroup_cap);
        if lane_cap == 0 {
            return Err(Error::backend(
                "device limits cannot bind one Modular reconstruction lane",
            ));
        }
        let requested_slots = u64::try_from(options.requested_frame_slots.max(1))
            .map_err(|_| Error::backend("requested frame-slot count exceeds u64"))?;
        let requested_target = options.memory_limit_bytes / requested_slots;
        let selected = match select_parallel_group_layout(
            codestream_bytes,
            &profile.entropy_groups,
            ParallelGroupLimits {
                stream_limit,
                lane_cap,
                lane_stride: reconstruction_lane_stride,
                fixed_bytes,
                per_frame_target: requested_target,
            },
        ) {
            Ok(Some(selected)) => Some(selected),
            Ok(None) => select_parallel_group_layout(
                codestream_bytes,
                &profile.entropy_groups,
                ParallelGroupLimits {
                    stream_limit,
                    lane_cap,
                    lane_stride: reconstruction_lane_stride,
                    fixed_bytes,
                    per_frame_target: options.memory_limit_bytes,
                },
            )?,
            Err(error) => return Err(error),
        }
        .ok_or_else(|| {
            Error::backend(format!(
                "one bounded Modular lane plus fixed allocations exceeds the shared {}-byte budget",
                options.memory_limit_bytes
            ))
        })?;
        let (parallel_group_lanes, stream_segments, stream_batches, group_stream_bytes) = selected;
        let stream_bytes = group_stream_bytes.max(global_stream_bytes);
        let reconstructed_bytes = reconstruction_lane_stride
            .checked_mul(u64::try_from(parallel_group_lanes).unwrap_or(u64::MAX))
            .ok_or_else(|| Error::backend("parallel Modular scratch size overflow"))?;
        Ok(Self {
            group_workgroup_size,
            reconstruction_lane_stride,
            execution_state_bytes_per_lane,
            entropy_coding,
            max_logical_reconstruction_sample_words,
            max_physical_reconstruction_sample_words,
            resident_modular_arena_bytes,
            frame_modular_arena_bytes,
            global_reconstruction_sample_words,
            low_frequency_group_stream_count: profile.low_frequency_entropy_group_count,
            progressive_pass_count: profile.pass_count,
            inverse_transform_count,
            palette_dispatch_count,
            inverse_transform_uniform_bytes,
            final_output_uniform_bytes,
            max_lz77_window_words,
            max_lz77_scratch_words,
            parallel_group_lanes,
            reconstructed_bytes,
            global_stream_segments: global_stream_segments.into(),
            global_stream_batches: global_stream_batches.into(),
            stream_segments: stream_segments.into(),
            stream_batches: stream_batches.into(),
            stream_bytes,
            status_stride,
            status_bytes,
            params_stride,
            params_bytes,
            output_write_path,
            output_specialization,
            reconstruction_specialization,
        })
    }
}

pub(super) fn build_stream_batches(
    codestream_bytes: u64,
    groups: &[ModularGroup],
    stream_limit: u64,
    max_groups_per_batch: usize,
) -> Result<(Vec<GroupStreamSegment>, Vec<StreamBatch>, u64)> {
    let ranges = groups
        .iter()
        .map(|group| GroupEntropyRange {
            token_bit_offset: group.token_bit_offset,
            token_bit_end: group.token_bit_end,
        })
        .collect::<Vec<_>>();
    build_entropy_stream_batches(
        codestream_bytes,
        &ranges,
        stream_limit,
        max_groups_per_batch,
    )
}

pub(super) type ParallelGroupLayout = (usize, Vec<GroupStreamSegment>, Vec<StreamBatch>, u64);

#[derive(Clone, Copy, Debug)]
pub(super) struct ParallelGroupLimits {
    pub(super) stream_limit: u64,
    pub(super) lane_cap: usize,
    pub(super) lane_stride: u64,
    pub(super) fixed_bytes: u64,
    pub(super) per_frame_target: u64,
}

pub(super) fn select_parallel_group_layout(
    codestream_bytes: u64,
    groups: &[ModularGroup],
    limits: ParallelGroupLimits,
) -> Result<Option<ParallelGroupLayout>> {
    let available = match limits.per_frame_target.checked_sub(limits.fixed_bytes) {
        Some(available) => available,
        None => return Ok(None),
    };
    let budget_lane_cap = usize::try_from(available / limits.lane_stride).unwrap_or(usize::MAX);
    let mut lanes = limits.lane_cap.min(budget_lane_cap);
    while lanes != 0 {
        let scratch_bytes = limits
            .lane_stride
            .checked_mul(u64::try_from(lanes).unwrap_or(u64::MAX))
            .ok_or_else(|| Error::backend("parallel Modular scratch size overflow"))?;
        let Some(stream_budget) = available.checked_sub(scratch_bytes) else {
            lanes -= 1;
            continue;
        };
        let effective_stream_limit = limits.stream_limit.min(stream_budget);
        if effective_stream_limit < MIN_STREAM_WINDOW_BYTES {
            lanes -= 1;
            continue;
        }
        let (segments, batches, stream_bytes) =
            build_stream_batches(codestream_bytes, groups, effective_stream_limit, lanes)?;
        let required = limits
            .fixed_bytes
            .checked_add(stream_bytes)
            .and_then(|bytes| bytes.checked_add(scratch_bytes))
            .ok_or_else(|| Error::backend("parallel Modular memory target overflow"))?;
        if required <= limits.per_frame_target {
            return Ok(Some((lanes, segments, batches, stream_bytes)));
        }
        lanes -= 1;
    }
    Ok(None)
}

pub(super) fn compact_gray8_sample_words(group: ModularGroup) -> Result<u32> {
    group
        .width
        .checked_mul(group.height.min(2))
        .ok_or_else(|| Error::backend("two-row normalized Gray8 workspace size overflow"))
}

pub(super) fn group_reconstructed_bytes(
    profile: &StandardModularProfile,
    group_index: usize,
    group: ModularGroup,
    decoded_symbol_count: u32,
    physical_sample_words: u32,
    execution_state_bytes: u64,
) -> Result<u64> {
    let predictor_words = if group_ma_config(profile, group_index)?.needs_self_correcting() {
        u64::from(group_maximum_channel_width(profile, group_index, group)?)
            .checked_mul(5)
            .ok_or_else(|| Error::backend("weighted predictor workspace overflow"))?
    } else {
        0
    };
    let entropy_words = u64::from(lz77_scratch_words(group_lz77_window_words(
        profile,
        group_index,
        group,
        decoded_symbol_count,
    )?));
    let working_bytes = u64::from(physical_sample_words)
        .checked_add(predictor_words)
        .and_then(|words| words.checked_add(entropy_words))
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| Error::backend("group reconstruction workspace size overflow"))?;
    align16(working_bytes)?
        .checked_add(execution_state_bytes)
        .ok_or_else(|| Error::backend("group execution state size overflow"))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FrameArenaWorkspace {
    pub(super) bytes: u64,
    pub(super) entropy_state_offset_words: u32,
    pub(super) decoded_words: u32,
    pub(super) maximum_width: u32,
    pub(super) lz77_window_words: u32,
}

pub(super) fn frame_arena_workspace(
    profile: &StandardModularProfile,
    plan: &ResidentModularFramePlan,
    execution_state_bytes: u64,
) -> Result<FrameArenaWorkspace> {
    let arena_words = plan.inverse_plan.arena_words();
    let decoded_words = plan
        .channel_metadata
        .channels
        .last()
        .map_or(0, |channel| channel.decoded_end);
    let maximum_width = plan
        .channel_metadata
        .channels
        .iter()
        .map(|channel| channel.width)
        .max()
        .unwrap_or(0);
    if profile.global_stream.is_none() {
        return Ok(FrameArenaWorkspace {
            bytes: u64::from(arena_words) * 4,
            entropy_state_offset_words: arena_words,
            decoded_words,
            maximum_width,
            lz77_window_words: 0,
        });
    }
    let ma_config = plan.ma_config.resolve(&profile.ma_config);
    let predictor_words = if ma_config.needs_self_correcting() {
        u64::from(maximum_width)
            .checked_mul(5)
            .ok_or_else(|| Error::backend("DC-global weighted predictor workspace overflow"))?
    } else {
        0
    };
    let lz77_window_words = ma_config
        .entropy
        .lz77_window_words(maximum_width, decoded_words)?;
    let entropy_words = u64::from(lz77_scratch_words(lz77_window_words));
    let working_bytes = u64::from(arena_words)
        .checked_add(predictor_words)
        .and_then(|words| words.checked_add(entropy_words))
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| Error::backend("DC-global reconstruction workspace size overflow"))?;
    let aligned_bytes = align16(working_bytes)?;
    let bytes = aligned_bytes
        .checked_add(execution_state_bytes)
        .ok_or_else(|| Error::backend("DC-global execution state size overflow"))?;
    let entropy_state_offset_words = u32::try_from(aligned_bytes / 4)
        .map_err(|_| Error::backend("DC-global entropy state offset exceeds WGSL u32"))?;
    Ok(FrameArenaWorkspace {
        bytes,
        entropy_state_offset_words,
        decoded_words,
        maximum_width,
        lz77_window_words,
    })
}

pub(super) const fn modular_execution_state_bytes(
    reconstruction: ModularReconstructionSpecialization,
    needs_self_correcting: bool,
) -> u64 {
    match (reconstruction, needs_self_correcting) {
        (
            ModularReconstructionSpecialization::ChannelFixed {
                predictor: ModularPredictor::Gradient,
                ..
            },
            false,
        ) => ENTROPY_EXECUTION_STATE_BYTES,
        (_, true) => GENERIC_WEIGHTED_EXECUTION_STATE_BYTES,
        _ => GENERIC_PREDICTOR_EXECUTION_STATE_BYTES,
    }
}

pub(super) fn group_entropy_state_offset_words(
    profile: &StandardModularProfile,
    group_index: usize,
    group: ModularGroup,
    decoded_symbol_count: u32,
    physical_sample_words: u32,
) -> Result<u32> {
    let predictor_words = if group_ma_config(profile, group_index)?.needs_self_correcting() {
        u64::from(group_maximum_channel_width(profile, group_index, group)?)
            .checked_mul(5)
            .ok_or_else(|| Error::backend("weighted predictor workspace overflow"))?
    } else {
        0
    };
    let entropy_words = u64::from(lz77_scratch_words(group_lz77_window_words(
        profile,
        group_index,
        group,
        decoded_symbol_count,
    )?));
    u64::from(physical_sample_words)
        .checked_add(predictor_words)
        .and_then(|words| words.checked_add(entropy_words))
        .and_then(|words| words.checked_mul(4))
        .and_then(|bytes| align16(bytes).ok())
        .and_then(|bytes| u32::try_from(bytes / 4).ok())
        .ok_or_else(|| Error::backend("group entropy execution state offset exceeds WGSL u32"))
}

pub(super) fn fixed_gradient_output_mode(
    source_channels: u32,
    source_bits: u8,
    output: &OutputPlan,
    specialization: ModularReconstructionSpecialization,
) -> FixedGradientOutputMode {
    if source_channels == 1
        && source_bits == 8
        && output.kind == OutputKind::NumericUnsigned
        && output.numeric_mapping == 1
        && output.channels == 1
        && output.bits == 8
        && output.storage_bits == 8
        && matches!(
            specialization,
            ModularReconstructionSpecialization::ChannelFixed {
                predictor: ModularPredictor::Gradient,
                offset: 0,
                multiplier: 1,
                channel_count: 1,
                ..
            }
        )
    {
        FixedGradientOutputMode::DirectNormalizedGray8
    } else {
        FixedGradientOutputMode::FinalizePass
    }
}

pub(super) fn refine_fixed_gradient_output_mode(
    output_mode: FixedGradientOutputMode,
    lz77_window_words: u32,
) -> FixedGradientOutputMode {
    if output_mode == FixedGradientOutputMode::DirectNormalizedGray8
        && lz77_scratch_words(lz77_window_words) == 0
    {
        FixedGradientOutputMode::CompactNormalizedGray8
    } else {
        output_mode
    }
}

pub(super) fn group_decoded_symbol_count(
    profile: &StandardModularProfile,
    group_index: usize,
    group: ModularGroup,
) -> Result<u32> {
    if uses_generalized_channel_layout(profile) {
        return Ok(resident_entropy_plan(profile, group_index)?
            .inverse_plan
            .entropy_words());
    }
    group
        .sample_count()?
        .checked_mul(profile.channels.count())
        .ok_or_else(|| Error::backend("group reconstruction sample count overflow"))
}

pub(super) fn group_maximum_channel_width(
    profile: &StandardModularProfile,
    group_index: usize,
    group: ModularGroup,
) -> Result<u32> {
    if !uses_generalized_channel_layout(profile) {
        return Ok(group.width);
    }
    resident_entropy_plan(profile, group_index)?
        .channel_metadata
        .channels
        .iter()
        .map(|channel| channel.width)
        .max()
        .filter(|width| *width != 0)
        .ok_or_else(|| Error::backend("generalized Modular channel layout is empty"))
}

pub(super) fn group_lz77_window_words(
    profile: &StandardModularProfile,
    group_index: usize,
    group: ModularGroup,
    decoded_symbol_count: u32,
) -> Result<u32> {
    group_ma_config(profile, group_index)?
        .entropy
        .lz77_window_words(
            group_maximum_channel_width(profile, group_index, group)?,
            decoded_symbol_count,
        )
}

pub(super) fn group_ma_config(
    profile: &StandardModularProfile,
    group_index: usize,
) -> Result<&crate::modular_tree::MaConfigIr> {
    Ok(resident_entropy_plan(profile, group_index)?
        .ma_config
        .resolve(&profile.ma_config))
}

pub(super) fn resident_entropy_plan(
    profile: &StandardModularProfile,
    group_index: usize,
) -> Result<&ResidentModularGroupPlan> {
    profile
        .resident_entropy_plans
        .get(group_index)
        .ok_or(Error::EngineContract(
            "resident Modular entropy plan is missing",
        ))
}

pub(super) const fn lz77_scratch_words(window_words: u32) -> u32 {
    if window_words <= 1 { 0 } else { window_words }
}

pub(super) fn modular_metadata_bytes(metadata: &[u32]) -> Result<u64> {
    u64::try_from(metadata.len())
        .ok()
        .and_then(|words| words.checked_mul(std::mem::size_of::<u32>() as u64))
        .ok_or_else(|| Error::backend("Modular metadata size overflow"))
}

pub(super) fn modular_finalize_params(
    profile: &StandardModularProfile,
    output: &OutputPlan,
    group_index: usize,
    group: ModularGroup,
) -> Result<ModularFinalizeParams> {
    let finalize_output = modular_finalize_output(output)?;
    let resident = resident_entropy_plan(profile, group_index)?;
    ModularFinalizeParams::new(
        ModularFinalizeRegion {
            source_extent: Extent2d::new(group.width, group.height),
            canvas_extent: output.layout.extent,
            origin_x: group.x,
            origin_y: group.y,
            status_index: u32::try_from(group_index)
                .map_err(|_| Error::backend("Modular finalizer status index exceeds u32"))?,
        },
        profile.bits_per_sample,
        &resident.inverse_plan.final_gpu_layouts(),
        resident.inverse_plan.arena_words(),
        finalize_output,
    )
    .map_err(Error::from)
}

pub(super) fn modular_frame_finalize_params(
    profile: &StandardModularProfile,
    output: &OutputPlan,
    frame_plan: &ResidentModularFramePlan,
) -> Result<ModularFinalizeParams> {
    let status_index = if profile.global_stream.is_some() {
        u32::try_from(profile.entropy_groups.len())
            .map_err(|_| Error::backend("DC-global finalizer status index exceeds u32"))?
    } else {
        0
    };
    ModularFinalizeParams::new(
        ModularFinalizeRegion {
            source_extent: Extent2d::new(profile.width, profile.height),
            canvas_extent: output.layout.extent,
            origin_x: 0,
            origin_y: 0,
            status_index,
        },
        profile.bits_per_sample,
        &frame_plan.inverse_plan.final_gpu_layouts(),
        frame_plan.inverse_plan.arena_words(),
        modular_finalize_output(output)?,
    )
    .map_err(Error::from)
}

pub(super) fn modular_finalize_output(output: &OutputPlan) -> Result<ModularFinalizeOutput> {
    let to_u32 = |value: u64, name: &'static str| {
        u32::try_from(value).map_err(|_| Error::backend(format!("{name} exceeds WGSL u32")))
    };
    let mut plane_offsets = [0u32; 4];
    let mut plane_strides = [0u32; 4];
    for plane in &output.layout.planes {
        let index = plane.plane_index;
        let offset = plane_offsets
            .get_mut(index)
            .ok_or_else(|| Error::backend("Modular final output has more than four planes"))?;
        *offset = to_u32(plane.offset, "final output plane offset")?;
        plane_strides[index] = to_u32(plane.row_stride, "final output plane stride")?;
    }
    let chroma_extent = output
        .layout
        .plane(1)
        .map_or(Extent2d::new(0, 0), |plane| plane.sample_extent);
    Ok(ModularFinalizeOutput {
        kind: output.kind as u32,
        transfer: output.transfer,
        limited_range: output.limited_range,
        channels: output.channels,
        order: output.order,
        bits: output.bits,
        storage_bits: output.storage_bits,
        numeric_mapping: output.numeric_mapping,
        plane_offsets,
        plane_strides,
        logical_size: to_u32(output.layout.logical_size, "final output size")?,
        chroma_extent,
    })
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OutputKind {
    NumericUnsigned = 0,
    Luma = 1,
    YuvSemiplanar = 2,
    YuvPlanar = 3,
    Yuv422Packed = 4,
    RgbInterleaved = 5,
    RgbPlanar = 6,
    NumericSigned = 7,
    NumericFloat = 8,
    NativeModular = 9,
}

pub(super) struct OutputPlan {
    pub(super) layout: ImageLayout,
    pub(super) kind: OutputKind,
    pub(super) transfer: u32,
    pub(super) limited_range: bool,
    pub(super) channels: u32,
    pub(super) order: u32,
    pub(super) bits: u32,
    pub(super) storage_bits: u32,
    pub(super) numeric_mapping: u32,
    pub(super) f64_output_path: Option<F64OutputPath>,
}

impl OutputPlan {
    pub(super) fn new(
        extent: Extent2d,
        request: &GpuOutputRequest,
        source_channels: crate::ModularChannels,
        source_bits: u8,
        capabilities: WgpuDecodeCapabilities,
    ) -> Result<Self> {
        let format = request.format().clone();
        if let Some(native) = native_modular_format(&format) {
            let native_mapping = matches!(
                (native.channels, request.mapping()),
                (
                    crate::ModularChannels::Gray,
                    GpuOutputMapping::Numeric(NumericSampleMapping::NativeUnsigned)
                ) | (
                    crate::ModularChannels::Rgb | crate::ModularChannels::Rgba,
                    GpuOutputMapping::Color
                )
            );
            if native_mapping {
                if native.channels != source_channels || native.bits_per_sample != source_bits {
                    return Err(Error::UnsupportedOutputFormat(format!(
                        "native Modular output {native:?} does not match {:?} {}-bit source",
                        source_channels, source_bits
                    )));
                }
                let output = Self {
                    layout: ImageLayout::packed(extent, format)?,
                    kind: OutputKind::NativeModular,
                    transfer: 0,
                    limited_range: false,
                    channels: native.channels.count(),
                    order: 0,
                    bits: u32::from(native.bits_per_sample),
                    storage_bits: u32::from(native.storage_bits),
                    numeric_mapping: 3,
                    f64_output_path: None,
                };
                output.validate_shader_layout()?;
                return Ok(output);
            }
        }
        if source_channels != crate::ModularChannels::Gray || source_bits != 8 {
            return Err(Error::UnsupportedOutputFormat(
                "RGB/RGBA and non-8-bit Modular sources currently require their exact canonical native output descriptor"
                    .into(),
            ));
        }
        let class = classify_pixel_format(&format)
            .map_err(|error| Error::UnsupportedOutputFormat(format!("{format:?}: {error}")))?;
        let (
            kind,
            transfer,
            limited_range,
            channels,
            order,
            bits,
            storage_bits,
            numeric_mapping,
            f64_output_path,
        ) = match (class, request.mapping()) {
            (
                PixelFormatClass::Numeric(numeric),
                GpuOutputMapping::Numeric(NumericSampleMapping::NormalizedGray8),
            ) => {
                if numeric.sample_kind == SampleKind::Float && numeric.bits_per_component == 64 {
                    return Err(Error::F64OutputPolicyRequired);
                }
                let kind = match numeric.sample_kind {
                    SampleKind::Unsigned => OutputKind::NumericUnsigned,
                    SampleKind::Signed => OutputKind::NumericSigned,
                    SampleKind::Float => OutputKind::NumericFloat,
                };
                (
                    kind,
                    0,
                    false,
                    u32::from(numeric.components),
                    0,
                    u32::from(numeric.bits_per_component),
                    u32::from(numeric.bits_per_component),
                    1,
                    None,
                )
            }
            (
                PixelFormatClass::Numeric(numeric),
                GpuOutputMapping::Numeric(NumericSampleMapping::NormalizedGray8F64(policy)),
            ) => {
                if numeric.sample_kind != SampleKind::Float
                    || numeric.bits_per_component != 64
                    || numeric.components != 1
                {
                    return Err(Error::F64OutputPolicyForNonF64);
                }
                let path = resolve_f64_output_path(policy, capabilities)?;
                (
                    OutputKind::NumericFloat,
                    0,
                    false,
                    1,
                    0,
                    64,
                    64,
                    match path {
                        F64OutputPath::ExactF32Widening => 1,
                        F64OutputPath::NativeArithmetic => 2,
                    },
                    Some(path),
                )
            }
            (PixelFormatClass::Numeric(_), GpuOutputMapping::Color) => {
                return Err(Error::NumericMappingRequired);
            }
            (
                PixelFormatClass::Numeric(_),
                GpuOutputMapping::Numeric(NumericSampleMapping::NativeUnsigned),
            ) => {
                return Err(Error::UnsupportedOutputFormat(
                    "native unsigned output descriptor does not match the Modular source".into(),
                ));
            }
            (PixelFormatClass::Color(_), GpuOutputMapping::Numeric(_)) => {
                return Err(Error::NumericMappingForColorOutput);
            }
            (PixelFormatClass::Color(color), GpuOutputMapping::Color) => {
                let (transfer, limited_range) = color_conversion(&format)?;
                let (kind, channels, order, bits, storage_bits) = match color {
                    ColorFormatClass::Rgb8 { storage, order } => {
                        if limited_range {
                            return Err(Error::UnsupportedOutputFormat(
                                "RGB output requires an explicit full-range color specification"
                                    .into(),
                            ));
                        }
                        let (channels, order) = rgb_output_shape(order);
                        let kind = match storage {
                            RgbStorage::Interleaved => OutputKind::RgbInterleaved,
                            RgbStorage::Planar => OutputKind::RgbPlanar,
                        };
                        (kind, channels, order, 8, 8)
                    }
                    ColorFormatClass::Luma { bits, storage_bits }
                        if matches!((bits, storage_bits), (8, 8) | (16, 16)) =>
                    {
                        (
                            OutputKind::Luma,
                            1,
                            0,
                            u32::from(bits),
                            u32::from(storage_bits),
                        )
                    }
                    ColorFormatClass::YuvSemiplanar {
                        bits: 8,
                        storage_bits: 8,
                        chroma_order,
                        ..
                    } => (
                        OutputKind::YuvSemiplanar,
                        3,
                        match chroma_order {
                            ChromaOrder::CbCr => 0,
                            ChromaOrder::CrCb => 1,
                        },
                        8,
                        8,
                    ),
                    ColorFormatClass::YuvPlanar {
                        bits: 8,
                        storage_bits: 8,
                        ..
                    } => (OutputKind::YuvPlanar, 3, 0, 8, 8),
                    ColorFormatClass::Yuv422Packed { order } => (
                        OutputKind::Yuv422Packed,
                        3,
                        match order {
                            Packed422Order::Yuyv => 0,
                            Packed422Order::Uyvy => 1,
                        },
                        8,
                        8,
                    ),
                    unsupported => {
                        return Err(Error::UnsupportedOutputFormat(format!(
                            "the 8-bit Gray GPU conversion path does not implement color storage {unsupported:?}"
                        )));
                    }
                };
                (
                    kind,
                    transfer,
                    limited_range,
                    channels,
                    order,
                    bits,
                    storage_bits,
                    0,
                    None,
                )
            }
            (_, GpuOutputMapping::ExtraChannel { index }) => {
                return Err(Error::UnsupportedOutputFormat(format!(
                    "extra channel {index} output mapping is not supported by this engine"
                )));
            }
        };
        let output = Self {
            layout: ImageLayout::packed(extent, format)?,
            kind,
            transfer,
            limited_range,
            channels,
            order,
            bits,
            storage_bits,
            numeric_mapping,
            f64_output_path,
        };
        output.validate_shader_layout()?;
        Ok(output)
    }

    fn validate_shader_layout(&self) -> Result<()> {
        let expected_planes = match self.kind {
            OutputKind::NumericUnsigned
            | OutputKind::NumericSigned
            | OutputKind::NumericFloat
            | OutputKind::Luma
            | OutputKind::Yuv422Packed
            | OutputKind::RgbInterleaved
            | OutputKind::NativeModular => 1,
            OutputKind::YuvSemiplanar => 2,
            OutputKind::YuvPlanar => 3,
            OutputKind::RgbPlanar => usize::try_from(self.channels)
                .map_err(|_| Error::backend("RGB plane count overflow"))?,
        };
        if self.layout.planes.len() != expected_planes || expected_planes > 4 {
            return Err(Error::backend(format!(
                "requested output has {} planes; {:?} requires {expected_planes}",
                self.layout.planes.len(),
                self.kind
            )));
        }
        u32::try_from(self.layout.logical_size)
            .map_err(|_| Error::backend("requested output exceeds the WGSL u32 address space"))?;
        for plane in &self.layout.planes {
            for (name, value) in [
                ("offset", plane.offset),
                ("row stride", plane.row_stride),
                ("row bytes", plane.row_bytes),
                ("end offset", plane.end_offset()?),
            ] {
                u32::try_from(value).map_err(|_| {
                    Error::backend(format!(
                        "output plane {} {name} exceeds the WGSL u32 address space",
                        plane.plane_index
                    ))
                })?;
            }
        }
        if matches!(self.kind, OutputKind::Yuv422Packed)
            || (self.kind == OutputKind::RgbInterleaved && self.channels == 4)
        {
            let plane = &self.layout.planes[0];
            if !plane.offset.is_multiple_of(4) || !plane.row_stride.is_multiple_of(4) {
                return Err(Error::backend(
                    "four-byte packed output requires four-byte-aligned rows",
                ));
            }
        }
        if matches!(
            self.kind,
            OutputKind::NumericUnsigned | OutputKind::NumericSigned | OutputKind::NumericFloat
        ) && self.bits >= 32
        {
            let plane = &self.layout.planes[0];
            if !plane.offset.is_multiple_of(4) || !plane.row_stride.is_multiple_of(4) {
                return Err(Error::backend(
                    "32/64-bit numeric output requires four-byte-aligned rows",
                ));
            }
        }
        if self.kind == OutputKind::NumericFloat && self.bits == 64 {
            let plane = &self.layout.planes[0];
            if !plane.offset.is_multiple_of(8) || !plane.row_stride.is_multiple_of(8) {
                return Err(Error::backend(
                    "F64 numeric output requires eight-byte-aligned rows",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn write_path_for_groups(&self, groups: &[ModularGroup]) -> Result<OutputWritePath> {
        if groups.is_empty() {
            return Err(Error::backend("Modular output has no pass groups"));
        }
        if self
            .layout
            .planes
            .iter()
            .any(|plane| !plane.offset.is_multiple_of(4) || !plane.row_stride.is_multiple_of(4))
        {
            return Ok(OutputWritePath::AtomicBytes);
        }
        for &group in groups {
            if !self.group_row_span_is_word_isolated(group)? {
                return Ok(OutputWritePath::AtomicBytes);
            }
        }
        Ok(OutputWritePath::WordAligned)
    }

    /// Proves that one group's row writes cannot share a storage word with a horizontal neighbor.
    /// Plane offsets and strides are checked separately by [`Self::write_path_for_groups`].
    fn group_row_span_is_word_isolated(&self, group: ModularGroup) -> Result<bool> {
        let end_x = group
            .x
            .checked_add(group.width)
            .ok_or_else(|| Error::backend("Modular group horizontal extent overflow"))?;
        if end_x > self.layout.extent.width {
            return Err(Error::backend("Modular group exceeds the output width"));
        }
        let internal_right_boundary = end_x != self.layout.extent.width;
        if self.kind == OutputKind::Yuv422Packed {
            // Each output word owns a pair. An odd internal edge would make adjacent groups write
            // the same pair even though both plane rows themselves begin on word boundaries.
            return Ok(
                group.x.is_multiple_of(2) && (!internal_right_boundary || end_x.is_multiple_of(2))
            );
        }
        let bytes_per_pixel = match self.kind {
            OutputKind::NumericUnsigned | OutputKind::NumericSigned | OutputKind::NumericFloat => {
                u64::from(self.channels)
                    .checked_mul(u64::from(self.bits / 8))
                    .ok_or_else(|| Error::backend("numeric output pixel size overflow"))?
            }
            OutputKind::Luma | OutputKind::YuvSemiplanar | OutputKind::YuvPlanar => {
                u64::from(self.storage_bits / 8)
            }
            OutputKind::RgbInterleaved => u64::from(self.channels),
            OutputKind::RgbPlanar => 1,
            OutputKind::NativeModular => u64::from(self.channels)
                .checked_mul(u64::from(self.storage_bits / 8))
                .ok_or_else(|| Error::backend("native Modular output pixel size overflow"))?,
            OutputKind::Yuv422Packed => unreachable!("packed 4:2:2 was handled above"),
        };
        let start = u64::from(group.x)
            .checked_mul(bytes_per_pixel)
            .ok_or_else(|| Error::backend("Modular group output start overflow"))?;
        let end = u64::from(end_x)
            .checked_mul(bytes_per_pixel)
            .ok_or_else(|| Error::backend("Modular group output end overflow"))?;
        Ok(start.is_multiple_of(4) && (!internal_right_boundary || end.is_multiple_of(4)))
    }
}

pub(super) fn resolve_f64_output_path(
    policy: F64OutputPolicy,
    capabilities: WgpuDecodeCapabilities,
) -> Result<F64OutputPath> {
    match policy {
        F64OutputPolicy::NativeRequired => capabilities
            .native_f64_arithmetic
            .then_some(F64OutputPath::NativeArithmetic)
            .ok_or(Error::NativeF64Unavailable),
        F64OutputPolicy::NativeOrExactF32Widening => Ok(if capabilities.native_f64_arithmetic {
            F64OutputPath::NativeArithmetic
        } else {
            F64OutputPath::ExactF32Widening
        }),
        F64OutputPolicy::ExactF32Widening => Ok(F64OutputPath::ExactF32Widening),
    }
}

pub(super) fn rgb_output_shape(order: RgbChannelOrder) -> (u32, u32) {
    match order {
        RgbChannelOrder::Rgb => (3, 0),
        RgbChannelOrder::Bgr => (3, 1),
        RgbChannelOrder::Rgba => (4, 2),
        RgbChannelOrder::Bgra => (4, 3),
    }
}

pub(super) fn color_conversion(format: &PixelFormat) -> Result<(u32, bool)> {
    let ColorSpecification::Defined(spec) = format.color_spec else {
        return Err(Error::UnsupportedOutputFormat(
            "YCbCr output requires an explicit color specification".into(),
        ));
    };
    let transfer = match spec.transfer {
        TransferFunction::Srgb | TransferFunction::Sycc => 0,
        TransferFunction::Bt709 | TransferFunction::Bt2020 => 1,
        TransferFunction::Linear => 2,
        transfer => {
            return Err(Error::UnsupportedOutputFormat(format!(
                "the 8-bit Gray GPU conversion path does not implement {transfer:?} output transfer"
            )));
        }
    };
    Ok((transfer, spec.range == ColorRange::Limited))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ModularMetadataInventory {
    pub(super) local_ma_stream_count: usize,
    pub(super) unique_ma_config_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DeviceAdmissionOptions {
    pub(super) requested_frame_slots: usize,
    pub(super) memory_limit_bytes: u64,
    pub(super) progressive_dc: bool,
}

pub(super) fn validate_device_limits(
    device: &wgpu::Device,
    modular_metadata: &[u32],
    metadata_inventory: ModularMetadataInventory,
    dispatch: &GroupDispatchLayout,
    output: &OutputPlan,
    options: DeviceAdmissionOptions,
) -> Result<WgpuDecodeMemoryStats> {
    let storage_limit = device.limits().max_storage_buffer_binding_size;
    let buffer_limit = device.limits().max_buffer_size;
    let stream_bytes = dispatch.stream_bytes;
    let metadata_bytes = modular_metadata_bytes(modular_metadata)?;
    let output_bytes = align4(output.layout.logical_size)?;
    let dispatch_control_bytes = std::mem::size_of::<DispatchControl>() as u64;
    let native_f64_dummy_bytes = if output.f64_output_path == Some(F64OutputPath::NativeArithmetic)
    {
        NATIVE_F64_DUMMY_WORD_BYTES
    } else {
        0
    };
    let progressive_dc_plane_bytes = if options.progressive_dc {
        u64::from(output.layout.extent.width)
            .checked_mul(u64::from(output.layout.extent.height))
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>() as u64))
            .and_then(|bytes| bytes.checked_mul(3))
            .ok_or_else(|| Error::backend("progressive-DC plane byte count overflow"))?
    } else {
        0
    };
    let progressive_dc_uniform_bytes = if options.progressive_dc {
        std::mem::size_of::<crate::progressive_dc::ProgressiveDcConvertParams>() as u64
    } else {
        0
    };
    for (name, required) in [
        ("bounded group stream window", stream_bytes),
        ("Modular metadata", metadata_bytes),
        (
            "parallel reconstructed samples",
            dispatch.reconstructed_bytes,
        ),
        (
            "frame-resident Modular arena",
            dispatch.frame_modular_arena_bytes,
        ),
        ("requested output", output_bytes),
        ("group statuses", dispatch.status_bytes),
        ("group parameters", dispatch.params_bytes),
    ] {
        if required > storage_limit || required > buffer_limit {
            return Err(Error::backend(format!(
                "{name} buffer requires {required} bytes, exceeding the device limit"
            )));
        }
    }
    for (name, required) in [
        ("group status readback", dispatch.status_bytes),
        ("parallel dispatch control", dispatch_control_bytes),
    ] {
        if required > buffer_limit {
            return Err(Error::backend(format!(
                "{name} buffer requires {required} bytes, exceeding the device buffer limit"
            )));
        }
    }
    if dispatch_control_bytes > device.limits().max_uniform_buffer_binding_size {
        return Err(Error::backend(
            "parallel dispatch control exceeds the device uniform-binding limit",
        ));
    }
    let per_frame = [
        stream_bytes,
        metadata_bytes,
        dispatch.reconstructed_bytes,
        dispatch.frame_modular_arena_bytes,
        output_bytes,
        native_f64_dummy_bytes,
        dispatch.status_bytes,
        dispatch.status_bytes,
        dispatch.params_bytes,
        dispatch_control_bytes,
        dispatch.inverse_transform_uniform_bytes,
        dispatch.final_output_uniform_bytes,
        progressive_dc_plane_bytes,
        progressive_dc_uniform_bytes,
    ]
    .into_iter()
    .try_fold(0u64, |total, bytes| total.checked_add(bytes))
    .ok_or_else(|| Error::backend("Modular GPU memory budget overflow"))?;
    let affordable_slots = options.memory_limit_bytes / per_frame;
    if affordable_slots == 0 {
        return Err(Error::backend(format!(
            "one Modular GPU frame requires {per_frame} bytes, exceeding the shared {}-byte budget",
            options.memory_limit_bytes
        )));
    }
    let max_frame_slots = options
        .requested_frame_slots
        .min(usize::try_from(affordable_slots).unwrap_or(usize::MAX));
    let max_frame_window_bytes = per_frame
        .checked_mul(
            u64::try_from(max_frame_slots)
                .map_err(|_| Error::backend("resolved frame-slot count exceeds u64"))?,
        )
        .ok_or_else(|| Error::backend("bounded in-flight GPU memory budget overflow"))?;
    let transient_bytes = per_frame
        .checked_sub(output_bytes)
        .ok_or_else(|| Error::backend("Modular transient memory accounting underflow"))?;
    let max_dispatch_workgroups = dispatch
        .global_stream_batches
        .iter()
        .chain(dispatch.stream_batches.iter())
        .try_fold(0u32, |maximum, batch| {
            u32::try_from(batch.group_count)
                .map(|groups| maximum.max(groups.div_ceil(dispatch.group_workgroup_size)))
                .map_err(|_| Error::backend("batch group count exceeds WGSL u32"))
        })?;
    if max_dispatch_workgroups == 0 {
        return Err(Error::backend("Modular stream batch layout is empty"));
    }
    Ok(WgpuDecodeMemoryStats {
        per_frame_bytes: per_frame,
        modular_metadata_bytes: metadata_bytes,
        local_ma_stream_count: metadata_inventory.local_ma_stream_count,
        unique_ma_config_count: metadata_inventory.unique_ma_config_count,
        output_lease_bytes: output_bytes,
        transient_bytes,
        max_frame_slots,
        max_frame_window_bytes,
        stream_window_bytes: dispatch.stream_bytes,
        reconstruction_scratch_bytes: dispatch.reconstructed_bytes,
        reconstruction_lane_stride_bytes: dispatch.reconstruction_lane_stride,
        execution_state_bytes_per_lane: dispatch.execution_state_bytes_per_lane,
        entropy_coding: dispatch.entropy_coding,
        max_logical_reconstruction_sample_words: dispatch.max_logical_reconstruction_sample_words,
        max_physical_reconstruction_sample_words: dispatch.max_physical_reconstruction_sample_words,
        resident_modular_arena_bytes: dispatch.resident_modular_arena_bytes,
        frame_modular_arena_bytes: dispatch.frame_modular_arena_bytes,
        global_reconstruction_sample_words: dispatch.global_reconstruction_sample_words,
        low_frequency_group_stream_count: dispatch.low_frequency_group_stream_count,
        progressive_pass_count: dispatch.progressive_pass_count,
        inverse_transform_count: dispatch.inverse_transform_count,
        palette_dispatch_count: dispatch.palette_dispatch_count,
        inverse_transform_uniform_bytes: dispatch.inverse_transform_uniform_bytes,
        final_output_uniform_bytes: dispatch.final_output_uniform_bytes,
        progressive_dc_plane_bytes,
        progressive_dc_uniform_bytes,
        max_lz77_window_words: dispatch.max_lz77_window_words,
        max_lz77_scratch_words: dispatch.max_lz77_scratch_words,
        stream_batch_count: dispatch
            .global_stream_batches
            .len()
            .checked_add(dispatch.stream_batches.len())
            .ok_or_else(|| Error::backend("Modular stream batch count overflow"))?,
        submissions_per_frame: dispatch
            .global_stream_batches
            .len()
            .checked_add(dispatch.stream_batches.len())
            .ok_or_else(|| Error::backend("Modular submission count overflow"))?,
        parallel_group_lanes: dispatch.parallel_group_lanes,
        group_workgroup_size: dispatch.group_workgroup_size,
        max_dispatch_workgroups,
        output_write_path: dispatch.output_write_path,
        output_specialization: dispatch.output_specialization,
        reconstruction_specialization: dispatch.reconstruction_specialization,
    })
}

#[derive(Clone, Copy)]
pub(super) struct SubmitPipelines<'a> {
    pub(super) decode: &'a wgpu::ComputePipeline,
    pub(super) inverse: Option<&'a ModularInversePipelines>,
    pub(super) progressive_dc: Option<&'a ProgressiveDcPipeline>,
}

pub(super) fn submit_decode(
    backend: &WgpuBackend,
    pipelines: SubmitPipelines<'_>,
    source: &DecodeSource,
    buffers: &Arc<DecodeBufferPool>,
    memory_permits: DecodeMemoryPermits,
    poll_permit: SubmissionPollPermit,
) -> Result<WgpuPendingFrame> {
    let device = backend.device();
    // Only a bounded batch of LF/pass-subimage packets is storage-bound at once. The host keeps the
    // validated shared span table, while queue ordering lets every batch reuse this one GPU
    // window. Cross-chunk ranges are copied directly into that bounded upload, never into a
    // whole-codestream allocation.
    let stream = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu decode bounded group stream window"),
        size: source.dispatch_layout.stream_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let metadata_bytes = u64::try_from(source.modular_metadata.len())
        .ok()
        .and_then(|entries| entries.checked_mul(u64::try_from(std::mem::size_of::<u32>()).ok()?))
        .ok_or_else(|| Error::backend("Modular metadata size overflow"))?;
    let metadata_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let metadata_buffer = buffers.checkout(
        "jxl-wgpu decode Modular metadata",
        metadata_bytes,
        metadata_usage,
        std::mem::align_of::<u32>() as u64,
    );
    backend.queue().write_buffer(
        metadata_buffer.buffer(),
        0,
        bytemuck::cast_slice(source.modular_metadata.as_ref()),
    );

    let reconstructed_usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let reconstructed = buffers.checkout(
        "jxl-wgpu decoded Modular samples",
        source.dispatch_layout.reconstructed_bytes,
        reconstructed_usage,
        std::mem::align_of::<u32>() as u64,
    );
    let frame_arena = (source.dispatch_layout.frame_modular_arena_bytes != 0).then(|| {
        buffers.checkout(
            "jxl-wgpu frame-resident Modular arena",
            source.dispatch_layout.frame_modular_arena_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            std::mem::align_of::<u32>() as u64,
        )
    });
    let progressive_dc_planes = source
        .profile
        .progressive_dc
        .map(|_| {
            ProgressiveDcXybPlanes::new(device, source.profile.width, source.profile.height, 0)
        })
        .transpose()?;
    let output_size = align4(source.output.layout.logical_size)?;
    let mut output_usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    if backend.direct_readback_enabled() {
        output_usage |= wgpu::BufferUsages::MAP_READ;
    }
    let output = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu decoded image output"),
        size: output_size,
        usage: output_usage,
        mapped_at_creation: false,
    }));
    // The native shader declares the caller-visible allocation exactly once as `array<f64>`.
    // Its otherwise-unused raw-word binding receives a distinct dummy allocation, avoiding two
    // writable storage aliases for the same buffer.
    let native_f64_dummy_words =
        (source.output.f64_output_path == Some(F64OutputPath::NativeArithmetic)).then(|| {
            buffers.checkout(
                "jxl-wgpu native F64 dummy word output",
                NATIVE_F64_DUMMY_WORD_BYTES,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                std::mem::align_of::<u32>() as u64,
            )
        });
    let status_usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let status = buffers.checkout(
        "jxl-wgpu decode status",
        source.dispatch_layout.status_bytes,
        status_usage,
        std::mem::align_of::<DecodeStatus>() as u64,
    );
    let status_staging_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
    let status_staging = buffers.checkout(
        "jxl-wgpu decode status readback",
        source.dispatch_layout.status_bytes,
        status_staging_usage,
        wgpu::COPY_BUFFER_ALIGNMENT,
    );

    let params_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let params_buffer = buffers.checkout(
        "jxl-wgpu decode Modular parameters",
        source.dispatch_layout.params_bytes,
        params_usage,
        std::mem::align_of::<u32>() as u64,
    );
    let dispatch_control = buffers.checkout(
        "jxl-wgpu decode Modular dispatch control",
        std::mem::size_of::<DispatchControl>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        std::mem::align_of::<DispatchControl>() as u64,
    );

    let word_output_binding = native_f64_dummy_words.as_ref().map_or_else(
        || output.as_entire_binding(),
        |buffer| buffer.buffer().as_entire_binding(),
    );
    let bind_group_layout = pipelines.decode.get_bind_group_layout(0);
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: stream.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: metadata_buffer.buffer().as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: reconstructed.buffer().as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: word_output_binding.clone(),
        },
        wgpu::BindGroupEntry {
            binding: 4,
            resource: status.buffer().as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 5,
            resource: params_buffer.buffer().as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 7,
            resource: dispatch_control.buffer().as_entire_binding(),
        },
    ];
    if source.output.f64_output_path == Some(F64OutputPath::NativeArithmetic) {
        entries.push(wgpu::BindGroupEntry {
            binding: 6,
            resource: output.as_entire_binding(),
        });
    }
    let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jxl-wgpu decode parallel Modular group bindings"),
        layout: &bind_group_layout,
        entries: &entries,
    });
    let global_binding = frame_arena.as_ref().map(|frame_arena| {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: stream.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: metadata_buffer.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: frame_arena.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: word_output_binding.clone(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: status.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params_buffer.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: dispatch_control.buffer().as_entire_binding(),
            },
        ];
        if source.output.f64_output_path == Some(F64OutputPath::NativeArithmetic) {
            entries.push(wgpu::BindGroupEntry {
                binding: 6,
                resource: output.as_entire_binding(),
            });
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu decode DC-global Modular bindings"),
            layout: &bind_group_layout,
            entries: &entries,
        })
    });
    let completion = Arc::new(MapCompletion::default());
    let lifetime = Arc::new(DecodeJobLifetime {
        output: GpuBufferLease::from_tracked(output.as_ref().clone(), memory_permits.output),
        _modular_metadata: metadata_buffer,
        _reconstructed: reconstructed,
        _frame_arena: frame_arena,
        _native_f64_dummy_words: native_f64_dummy_words,
        _status: status,
        status_staging,
        status_mapped: AtomicBool::new(false),
        _params: params_buffer,
        _dispatch_control: dispatch_control,
        _transient_permit: memory_permits.transient,
        progressive_dc_planes,
        _progressive_dc_uniform: Mutex::new(None),
    });
    let upload_len = usize::try_from(source.dispatch_layout.stream_bytes)
        .map_err(|_| Error::backend("bounded stream upload exceeds host address space"))?;
    let mut stream_upload = vec![0u8; upload_len];
    let mut final_submission = None;
    let has_global_stream = !source.dispatch_layout.global_stream_batches.is_empty();
    if has_global_stream {
        let global_record_index = source.profile.entropy_groups.len();
        let global_record_index_u32 = u32::try_from(global_record_index)
            .map_err(|_| Error::backend("DC-global status index exceeds WGSL u32"))?;
        let global_params_offset = u64::try_from(global_record_index)
            .ok()
            .and_then(|index| index.checked_mul(source.dispatch_layout.params_stride))
            .ok_or_else(|| Error::backend("DC-global parameter offset overflow"))?;
        for (batch_index, batch) in source
            .dispatch_layout
            .global_stream_batches
            .iter()
            .enumerate()
        {
            stream_upload.fill(0);
            let segment_index = batch.segments.start;
            if batch.segments.end != segment_index + 1 {
                return Err(Error::EngineContract(
                    "one DC-global entropy batch must contain exactly one segment",
                ));
            }
            let segment = source
                .dispatch_layout
                .global_stream_segments
                .get(segment_index)
                .copied()
                .ok_or(Error::EngineContract(
                    "DC-global entropy stream segment is missing",
                ))?;
            copy_stream_segment(source, segment, &mut stream_upload, "DC-global")?;
            let params = build_global_params(segment, global_record_index_u32, source)?;
            backend.queue().write_buffer(
                lifetime._params.buffer(),
                global_params_offset,
                bytemuck::bytes_of(&params),
            );
            backend.queue().write_buffer(&stream, 0, &stream_upload);
            let control = DispatchControl {
                first_group: global_record_index_u32,
                group_count: 1,
                lane_stride_words: u32::try_from(
                    source.dispatch_layout.reconstruction_lane_stride / 4,
                )
                .map_err(|_| Error::backend("reconstruction lane stride exceeds WGSL u32"))?,
                _padding: 0,
            };
            backend.queue().write_buffer(
                lifetime._dispatch_control.buffer(),
                0,
                bytemuck::bytes_of(&control),
            );

            let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu validate bounded DC-global Modular entropy"),
            });
            if batch_index == 0 {
                commands.clear_buffer(lifetime._reconstructed.buffer(), 0, None);
                if let Some(frame_arena) = &lifetime._frame_arena {
                    commands.clear_buffer(frame_arena.buffer(), 0, None);
                }
                commands.clear_buffer(lifetime.output.as_wgpu_buffer(), 0, None);
                if let Some(dummy) = &lifetime._native_f64_dummy_words {
                    commands.clear_buffer(dummy.buffer(), 0, None);
                }
                commands.clear_buffer(lifetime._status.buffer(), 0, None);
            }
            {
                let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("jxl-wgpu DC-global Modular entropy reconstruction"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipelines.decode);
                pass.set_bind_group(0, global_binding.as_ref().unwrap_or(&binding), &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            backend.queue().submit([commands.finish()]);
        }
    }
    for (batch_index, batch) in source.dispatch_layout.stream_batches.iter().enumerate() {
        stream_upload.fill(0);
        for segment_index in batch.segments.clone() {
            let segment = source
                .dispatch_layout
                .stream_segments
                .get(segment_index)
                .copied()
                .ok_or_else(|| Error::backend("group stream segment is missing"))?;
            copy_stream_segment(source, segment, &mut stream_upload, "group")?;

            let group = source
                .profile
                .entropy_groups
                .get(segment.group_index)
                .copied()
                .ok_or_else(|| Error::backend("stream segment group index is invalid"))?;
            let status_index = u32::try_from(segment.group_index)
                .map_err(|_| Error::backend("group status index exceeds WGSL u32"))?;
            let params = build_params(
                group,
                segment.group_index,
                segment,
                status_index,
                source,
                source.dispatch_layout.reconstruction_specialization,
                segment.group_index == 0,
            )?;
            let params_offset = u64::try_from(segment.group_index)
                .ok()
                .and_then(|index| index.checked_mul(source.dispatch_layout.params_stride))
                .ok_or_else(|| Error::backend("group parameter offset overflow"))?;
            let params_end = params_offset
                .checked_add(std::mem::size_of::<ShaderParams>() as u64)
                .ok_or_else(|| Error::backend("group parameter range overflow"))?;
            if params_end > source.dispatch_layout.params_bytes {
                return Err(Error::backend("group parameter buffer is truncated"));
            }
            backend.queue().write_buffer(
                lifetime._params.buffer(),
                params_offset,
                bytemuck::bytes_of(&params),
            );
        }
        backend.queue().write_buffer(&stream, 0, &stream_upload);
        let control = DispatchControl {
            first_group: u32::try_from(batch.first_group)
                .map_err(|_| Error::backend("batch group index exceeds WGSL u32"))?,
            group_count: u32::try_from(batch.group_count)
                .map_err(|_| Error::backend("batch group count exceeds WGSL u32"))?,
            lane_stride_words: u32::try_from(source.dispatch_layout.reconstruction_lane_stride / 4)
                .map_err(|_| Error::backend("reconstruction lane stride exceeds WGSL u32"))?,
            _padding: 0,
        };
        backend.queue().write_buffer(
            lifetime._dispatch_control.buffer(),
            0,
            bytemuck::bytes_of(&control),
        );

        let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu decode bounded Modular batch"),
        });
        if batch_index == 0 {
            commands.clear_buffer(lifetime._reconstructed.buffer(), 0, None);
            if !has_global_stream {
                if let Some(frame_arena) = &lifetime._frame_arena {
                    commands.clear_buffer(frame_arena.buffer(), 0, None);
                }
                commands.clear_buffer(lifetime.output.as_wgpu_buffer(), 0, None);
                if let Some(dummy) = &lifetime._native_f64_dummy_words {
                    commands.clear_buffer(dummy.buffer(), 0, None);
                }
                commands.clear_buffer(lifetime._status.buffer(), 0, None);
            }
        }
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu generic Modular entropy and MA reconstruction"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipelines.decode);
            pass.set_bind_group(0, &binding, &[]);
            pass.dispatch_workgroups(
                control
                    .group_count
                    .div_ceil(source.dispatch_layout.group_workgroup_size),
                1,
                1,
            );
        }
        let final_batch = batch_index + 1 == source.dispatch_layout.stream_batches.len();
        let mut inverse_uniforms = Vec::new();
        if !source.channel_layout_offsets.is_empty() {
            let inverse = pipelines.inverse.ok_or(Error::EngineContract(
                "descriptor reconstruction is missing resident inverse pipelines",
            ))?;
            for segment_index in batch.segments.clone() {
                let segment = source
                    .dispatch_layout
                    .stream_segments
                    .get(segment_index)
                    .copied()
                    .ok_or_else(|| Error::backend("group stream segment is missing"))?;
                if segment.flags & GroupStreamSegment::FINAL == 0 {
                    continue;
                }
                let lane_index = segment
                    .group_index
                    .checked_sub(batch.first_group)
                    .filter(|lane| *lane < batch.group_count)
                    .ok_or_else(|| {
                        Error::backend("final Modular group lane is outside its batch")
                    })?;
                inverse_uniforms.extend(encode_modular_inverse(
                    device,
                    &mut commands,
                    source,
                    lifetime._reconstructed.buffer(),
                    inverse,
                    segment.group_index,
                    lane_index,
                )?);
                if source.profile.resident_frame_plan.is_some() {
                    encode_subimage_plane_copies(
                        &mut commands,
                        source,
                        lifetime._reconstructed.buffer(),
                        lifetime
                            ._frame_arena
                            .as_ref()
                            .ok_or(Error::EngineContract(
                                "Modular subimage assembly is missing its frame arena",
                            ))?
                            .buffer(),
                        segment.group_index,
                        lane_index,
                    )?;
                } else if source.profile.progressive_dc.is_none() {
                    inverse_uniforms.push(encode_modular_finalize(
                        device,
                        &mut commands,
                        source,
                        &lifetime,
                        inverse,
                        segment.group_index,
                        lane_index,
                    )?);
                }
            }
        }
        if final_batch {
            if let Some(frame_plan) = &source.profile.resident_frame_plan {
                let inverse = pipelines.inverse.ok_or(Error::EngineContract(
                    "frame Modular reconstruction is missing resident inverse pipelines",
                ))?;
                let frame_arena = lifetime._frame_arena.as_ref().ok_or(Error::EngineContract(
                    "frame-wide Modular inverse is missing its resident arena",
                ))?;
                let arena_size = NonZeroU64::new(frame_plan.inverse_plan.arena_bytes()).ok_or(
                    Error::EngineContract(
                        "frame Modular reconstruction produced an empty inverse arena",
                    ),
                )?;
                inverse_uniforms.extend(encode_modular_inverse_jobs(
                    device,
                    &mut commands,
                    ResidentStorageBinding {
                        buffer: frame_arena.buffer(),
                        offset: 0,
                        size: arena_size,
                    },
                    &frame_plan.inverse_plan,
                    frame_plan.wp_header,
                    inverse,
                )?);
                if source.profile.progressive_dc.is_none() {
                    inverse_uniforms.push(encode_frame_modular_finalize(
                        device,
                        &mut commands,
                        source,
                        &lifetime,
                        inverse,
                    )?);
                }
            }
            if let Some(progressive) = source.profile.progressive_dc {
                let pipeline = pipelines.progressive_dc.ok_or(Error::EngineContract(
                    "progressive-DC Modular conversion pipeline is missing",
                ))?;
                let planes =
                    lifetime
                        .progressive_dc_planes
                        .as_ref()
                        .ok_or(Error::EngineContract(
                            "progressive-DC Modular conversion planes are missing",
                        ))?;
                let (arena, final_planes) =
                    if let Some(frame_plan) = &source.profile.resident_frame_plan {
                        let arena = lifetime._frame_arena.as_ref().ok_or(Error::EngineContract(
                            "progressive-DC frame inverse arena is missing",
                        ))?;
                        (arena.buffer(), frame_plan.inverse_plan.final_gpu_layouts())
                    } else {
                        let [group_plan] = source.profile.resident_entropy_plans.as_slice() else {
                            return Err(Error::EngineContract(
                                "progressive-DC Modular root requires one resident frame topology",
                            ));
                        };
                        (
                            lifetime._reconstructed.buffer(),
                            group_plan.inverse_plan.final_gpu_layouts(),
                        )
                    };
                let source_planes = final_planes.try_into().map_err(|_| {
                    Error::EngineContract(
                        "progressive-DC Modular root must reconstruct exactly three XYB planes",
                    )
                })?;
                let arena_size = NonZeroU64::new(arena.size()).ok_or(Error::EngineContract(
                    "progressive-DC Modular arena is empty",
                ))?;
                let uniform = pipeline.encode_convert(
                    device,
                    &mut commands,
                    ProgressiveDcConvertInputs {
                        arena: ResidentStorageBinding {
                            buffer: arena,
                            offset: 0,
                            size: arena_size,
                        },
                        source_planes,
                        outputs: planes,
                        multipliers: progressive.lf_dequantization(),
                    },
                )?;
                *lifetime
                    ._progressive_dc_uniform
                    .lock()
                    .map_err(|_| Error::backend("progressive-DC uniform lock was poisoned"))? =
                    Some(uniform);
            }
            commands.copy_buffer_to_buffer(
                lifetime._status.buffer(),
                0,
                lifetime.status_staging.buffer(),
                0,
                source.dispatch_layout.status_bytes,
            );
            let callback_lifetime = Arc::clone(&lifetime);
            let callback_completion = Arc::clone(&completion);
            commands.map_buffer_on_submit(
                lifetime.status_staging.buffer(),
                wgpu::MapMode::Read,
                ..,
                move |result| {
                    // Release the callback's ownership before waking a waiter. The pending frame
                    // keeps the job alive through validation; an abandoned pending frame instead
                    // makes this the final Arc and safely unmaps/recycles staging.
                    if result.is_ok() {
                        callback_lifetime
                            .status_mapped
                            .store(true, Ordering::Release);
                    }
                    drop(callback_lifetime);
                    callback_completion.complete(
                        result.map_err(|error| format!("GPU status mapping failed: {error}")),
                    );
                },
            );
        }
        let submission = backend.queue().submit([commands.finish()]);
        drop(inverse_uniforms);
        if final_batch {
            final_submission = Some(submission);
        }
    }
    let submission = final_submission
        .ok_or_else(|| Error::backend("bounded Modular stream produced no GPU submission"))?;
    let poll_completion = Arc::clone(&completion);
    if let Err(error) = poll_permit.register(submission, move |error| {
        poll_completion.complete(Err(error));
    }) {
        completion.complete(Err(format!("GPU poll registration failed: {error}")));
    }

    Ok(WgpuPendingFrame {
        device: backend.device().clone(),
        lifetime: Some(lifetime),
        token: SubmissionToken(1),
        layout: source.output.layout.clone(),
        completion,
        stream_sample_counts: {
            let mut expected = source
                .profile
                .entropy_groups
                .iter()
                .copied()
                .enumerate()
                .map(|(group_index, group)| {
                    group_decoded_symbol_count(&source.profile, group_index, group)
                })
                .collect::<Result<Vec<_>>>()?;
            if source.profile.global_stream.is_some() {
                expected.push(
                    source
                        .profile
                        .resident_frame_plan
                        .as_ref()
                        .and_then(|plan| plan.channel_metadata.channels.last())
                        .map_or(0, |channel| channel.decoded_end),
                );
            }
            expected.into()
        },
        status_stride: source.dispatch_layout.status_stride,
    })
}

pub(super) fn copy_stream_segment(
    source: &DecodeSource,
    segment: GroupStreamSegment,
    upload: &mut [u8],
    stream_name: &'static str,
) -> Result<()> {
    let input_len = segment
        .input_end
        .checked_sub(segment.input_start)
        .ok_or_else(|| Error::backend(format!("{stream_name} stream input range underflow")))?;
    let end = segment
        .upload_offset
        .checked_add(input_len)
        .ok_or_else(|| Error::backend(format!("{stream_name} stream upload range overflow")))?;
    let destination = upload
        .get_mut(segment.upload_offset..end)
        .ok_or_else(|| Error::backend(format!("{stream_name} stream upload range is truncated")))?;
    source.codestream.copy_range(
        u64::try_from(segment.input_start)
            .map_err(|_| Error::backend(format!("{stream_name} stream start exceeds u64")))?
            ..u64::try_from(segment.input_end)
                .map_err(|_| Error::backend(format!("{stream_name} stream end exceeds u64")))?,
        destination,
    )
}

pub(super) fn build_global_params(
    stream_segment: GroupStreamSegment,
    status_index: u32,
    source: &DecodeSource,
) -> Result<ShaderParams> {
    let Some(plan) = source.profile.resident_frame_plan.as_ref() else {
        let mut params = <ShaderParams as bytemuck::Zeroable>::zeroed();
        params.entropy = EntropyStreamParams {
            token_start: 0,
            token_end: stream_segment.available_token_end,
            lz77_window_mask: 0,
        };
        params.window_logical_start = stream_segment.window_logical_start;
        params.window_upload_start = stream_segment.window_upload_start;
        params.stream_token_end = stream_segment.stream_token_end;
        params.window_yield_end = stream_segment.window_yield_end;
        params.window_flags = stream_segment.flags;
        params.status_index = status_index;
        params.fixed_output_mode = FixedGradientOutputMode::DirectNormalizedGray8 as u32;
        return Ok(params);
    };
    let workspace = frame_arena_workspace(
        &source.profile,
        plan,
        source.dispatch_layout.execution_state_bytes_per_lane,
    )?;
    let mut params = <ShaderParams as bytemuck::Zeroable>::zeroed();
    params.entropy = EntropyStreamParams {
        token_start: 0,
        token_end: stream_segment.available_token_end,
        lz77_window_mask: workspace.lz77_window_words.saturating_sub(1),
    };
    params.window_logical_start = stream_segment.window_logical_start;
    params.window_upload_start = stream_segment.window_upload_start;
    params.stream_token_end = stream_segment.stream_token_end;
    params.window_yield_end = stream_segment.window_yield_end;
    params.window_flags = stream_segment.flags;
    params.entropy_state_offset = workspace.entropy_state_offset_words;
    params.width = workspace.maximum_width;
    params.height = 1;
    params.sample_count = workspace.decoded_words;
    params.source_channels = source.profile.channels.count();
    params.channel_layout_offset =
        source
            .global_channel_layout_offset
            .ok_or(Error::EngineContract(
                "DC-global channel metadata offset is missing",
            ))?;
    params.metadata_base = source
        .global_ma_metadata_offset
        .ok_or(Error::EngineContract(
            "DC-global MA metadata offset is missing",
        ))?;
    params.source_bits = u32::from(source.profile.bits_per_sample);
    params.source_mask = (1u32 << source.profile.bits_per_sample) - 1;
    params.needs_self_correcting = u32::from(
        plan.ma_config
            .resolve(&source.profile.ma_config)
            .needs_self_correcting(),
    );
    params.status_index = status_index;
    params.stream_index = 0;
    params.fixed_output_mode = FixedGradientOutputMode::DirectNormalizedGray8 as u32;
    let wp_header = plan.wp_header;
    params.wp_p1 = wp_header.p1;
    params.wp_p2 = wp_header.p2;
    params.wp_p3a = wp_header.p3a;
    params.wp_p3b = wp_header.p3b;
    params.wp_p3c = wp_header.p3c;
    params.wp_p3d = wp_header.p3d;
    params.wp_p3e = wp_header.p3e;
    params.wp_w0 = wp_header.w0;
    params.wp_w1 = wp_header.w1;
    params.wp_w2 = wp_header.w2;
    params.wp_w3 = wp_header.w3;
    Ok(params)
}

pub(super) fn encode_modular_inverse(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    source: &DecodeSource,
    reconstructed: &wgpu::Buffer,
    pipelines: &ModularInversePipelines,
    group_index: usize,
    lane_index: usize,
) -> Result<Vec<wgpu::Buffer>> {
    let plan = resident_entropy_plan(&source.profile, group_index)?;
    let arena_size = NonZeroU64::new(plan.inverse_plan.arena_bytes()).ok_or(
        Error::EngineContract("descriptor reconstruction produced an empty resident arena"),
    )?;
    let arena_offset = u64::try_from(lane_index)
        .ok()
        .and_then(|lane| lane.checked_mul(source.dispatch_layout.reconstruction_lane_stride))
        .ok_or_else(|| Error::backend("resident Modular lane offset overflow"))?;
    if arena_offset
        .checked_add(arena_size.get())
        .is_none_or(|end| end > reconstructed.size())
    {
        return Err(Error::EngineContract(
            "resident Modular inverse arena exceeds its reconstruction lane",
        ));
    }
    let storage = ResidentStorageBinding {
        buffer: reconstructed,
        offset: arena_offset,
        size: arena_size,
    };
    encode_modular_inverse_jobs(
        device,
        encoder,
        storage,
        &plan.inverse_plan,
        plan.wp_header,
        pipelines,
    )
}

pub(super) fn encode_modular_inverse_jobs(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    storage: ResidentStorageBinding<'_>,
    plan: &ModularInversePlan,
    wp_header: crate::modular_tree::WpHeaderIr,
    pipelines: &ModularInversePipelines,
) -> Result<Vec<wgpu::Buffer>> {
    let mut uniforms = Vec::new();
    for job in plan.jobs() {
        match *job {
            ModularInverseJob::Squeeze { params } => {
                let pipeline = pipelines.squeeze.as_ref().ok_or(Error::EngineContract(
                    "resident Modular Squeeze job is missing its pipeline",
                ))?;
                uniforms.push(
                    pipeline
                        .encode(
                            device,
                            encoder,
                            ModularSqueezeArena::from_storage(storage),
                            params,
                        )
                        .map_err(crate::ModularInversePlanError::from)
                        .map_err(Error::from)?,
                );
            }
            ModularInverseJob::Rct { params } => {
                let pipeline = pipelines.rct.as_ref().ok_or(Error::EngineContract(
                    "resident Modular RCT job is missing its pipeline",
                ))?;
                uniforms.push(
                    pipeline
                        .encode(
                            device,
                            encoder,
                            ModularRctArena::from_storage(storage),
                            params,
                        )
                        .map_err(crate::ModularInversePlanError::from)
                        .map_err(Error::from)?,
                );
            }
            ModularInverseJob::Palette { job } => {
                let pipeline = pipelines.palette.as_ref().ok_or(Error::EngineContract(
                    "resident Modular Palette job is missing its pipeline",
                ))?;
                uniforms.extend(pipeline.encode(
                    device,
                    encoder,
                    storage,
                    job,
                    ModularPaletteWeightedParams::from(wp_header),
                )?);
            }
        }
    }
    Ok(uniforms)
}

pub(super) fn encode_subimage_plane_copies(
    encoder: &mut wgpu::CommandEncoder,
    source: &DecodeSource,
    group_arena: &wgpu::Buffer,
    frame_arena: &wgpu::Buffer,
    group_index: usize,
    lane_index: usize,
) -> Result<()> {
    let frame_plan = source
        .profile
        .resident_frame_plan
        .as_ref()
        .ok_or(Error::EngineContract(
            "Modular subimage assembly is missing its frame plan",
        ))?;
    let copies = frame_plan
        .subimage_plane_copies
        .get(group_index)
        .ok_or(Error::EngineContract(
            "Modular subimage assembly is missing its copy plan",
        ))?;
    let lane_offset = u64::try_from(lane_index)
        .ok()
        .and_then(|lane| lane.checked_mul(source.dispatch_layout.reconstruction_lane_stride))
        .ok_or_else(|| Error::backend("Modular subimage copy lane offset overflow"))?;
    for copy in copies {
        if copy.source.width != copy.destination.width
            || copy.source.height != copy.destination.height
            || copy.source.bit_depth != copy.destination.bit_depth
        {
            return Err(Error::EngineContract(
                "Modular subimage source and frame-arena destination geometries disagree",
            ));
        }
        let row_bytes = u64::from(copy.source.width)
            .checked_mul(4)
            .ok_or_else(|| Error::backend("Modular subimage copy row size overflow"))?;
        for row in 0..copy.source.height {
            let source_offset = u64::from(copy.source.word_offset)
                .checked_add(u64::from(row) * u64::from(copy.source.row_stride_words))
                .and_then(|words| words.checked_mul(4))
                .and_then(|bytes| lane_offset.checked_add(bytes))
                .ok_or_else(|| Error::backend("Modular subimage copy source offset overflow"))?;
            let destination_offset = u64::from(copy.destination.word_offset)
                .checked_add(u64::from(row) * u64::from(copy.destination.row_stride_words))
                .and_then(|words| words.checked_mul(4))
                .ok_or_else(|| {
                    Error::backend("Modular subimage copy destination offset overflow")
                })?;
            if source_offset
                .checked_add(row_bytes)
                .is_none_or(|end| end > group_arena.size())
                || destination_offset
                    .checked_add(row_bytes)
                    .is_none_or(|end| end > frame_arena.size())
            {
                return Err(Error::EngineContract(
                    "Modular subimage plane copy exceeds a resident GPU arena",
                ));
            }
            encoder.copy_buffer_to_buffer(
                group_arena,
                source_offset,
                frame_arena,
                destination_offset,
                row_bytes,
            );
        }
    }
    Ok(())
}

pub(super) fn encode_modular_finalize(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    source: &DecodeSource,
    lifetime: &DecodeJobLifetime,
    pipelines: &ModularInversePipelines,
    group_index: usize,
    lane_index: usize,
) -> Result<wgpu::Buffer> {
    let params = source
        .finalize_params
        .get(group_index)
        .copied()
        .ok_or(Error::EngineContract(
            "descriptor reconstruction is missing final-output parameters",
        ))?;
    let plan = resident_entropy_plan(&source.profile, group_index)?;
    let arena_size = NonZeroU64::new(plan.inverse_plan.arena_bytes()).ok_or(
        Error::EngineContract("descriptor reconstruction produced an empty resident arena"),
    )?;
    let arena_offset = u64::try_from(lane_index)
        .ok()
        .and_then(|lane| lane.checked_mul(source.dispatch_layout.reconstruction_lane_stride))
        .ok_or_else(|| Error::backend("resident Modular lane offset overflow"))?;
    let output = lifetime.output.as_wgpu_buffer();
    let output_size = NonZeroU64::new(output.size()).ok_or(Error::EngineContract(
        "descriptor reconstruction produced an empty output allocation",
    ))?;
    let native_f64_dummy_words = lifetime
        ._native_f64_dummy_words
        .as_ref()
        .map(DecodeBufferLease::buffer);
    let output_words_buffer = native_f64_dummy_words.unwrap_or(output);
    let output_words_size = NonZeroU64::new(output_words_buffer.size()).ok_or(
        Error::EngineContract("descriptor reconstruction produced an empty word output"),
    )?;
    pipelines
        .finalize
        .encode(
            device,
            encoder,
            ModularFinalizeBindings {
                arena: ResidentStorageBinding {
                    buffer: lifetime._reconstructed.buffer(),
                    offset: arena_offset,
                    size: arena_size,
                },
                output_words: ResidentStorageBinding {
                    buffer: output_words_buffer,
                    offset: 0,
                    size: output_words_size,
                },
                status: ResidentStorageBinding {
                    buffer: lifetime._status.buffer(),
                    offset: 0,
                    size: NonZeroU64::new(lifetime._status.buffer().size()).ok_or(
                        Error::EngineContract(
                            "descriptor reconstruction produced an empty status allocation",
                        ),
                    )?,
                },
                output_f64: native_f64_dummy_words.map(|_| ResidentStorageBinding {
                    buffer: output,
                    offset: 0,
                    size: output_size,
                }),
            },
            params,
        )
        .map_err(Error::from)
}

pub(super) fn encode_frame_modular_finalize(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    source: &DecodeSource,
    lifetime: &DecodeJobLifetime,
    pipelines: &ModularInversePipelines,
) -> Result<wgpu::Buffer> {
    let params = source
        .finalize_params
        .first()
        .copied()
        .ok_or(Error::EngineContract(
            "frame Modular reconstruction is missing final-output parameters",
        ))?;
    let plan = source
        .profile
        .resident_frame_plan
        .as_ref()
        .ok_or(Error::EngineContract(
            "frame Modular reconstruction is missing its inverse plan",
        ))?;
    let frame_arena = lifetime
        ._frame_arena
        .as_ref()
        .ok_or(Error::EngineContract(
            "frame Modular reconstruction is missing its resident arena",
        ))?
        .buffer();
    let arena_size = NonZeroU64::new(plan.inverse_plan.arena_bytes()).ok_or(
        Error::EngineContract("frame Modular reconstruction produced an empty arena"),
    )?;
    let output = lifetime.output.as_wgpu_buffer();
    let output_size = NonZeroU64::new(output.size()).ok_or(Error::EngineContract(
        "frame Modular reconstruction produced an empty output allocation",
    ))?;
    let native_f64_dummy_words = lifetime
        ._native_f64_dummy_words
        .as_ref()
        .map(DecodeBufferLease::buffer);
    let output_words_buffer = native_f64_dummy_words.unwrap_or(output);
    let output_words_size = NonZeroU64::new(output_words_buffer.size()).ok_or(
        Error::EngineContract("frame Modular reconstruction produced an empty word output"),
    )?;
    pipelines
        .finalize
        .encode(
            device,
            encoder,
            ModularFinalizeBindings {
                arena: ResidentStorageBinding {
                    buffer: frame_arena,
                    offset: 0,
                    size: arena_size,
                },
                output_words: ResidentStorageBinding {
                    buffer: output_words_buffer,
                    offset: 0,
                    size: output_words_size,
                },
                status: ResidentStorageBinding {
                    buffer: lifetime._status.buffer(),
                    offset: 0,
                    size: NonZeroU64::new(lifetime._status.buffer().size()).ok_or(
                        Error::EngineContract(
                            "frame Modular reconstruction produced an empty status allocation",
                        ),
                    )?,
                },
                output_f64: native_f64_dummy_words.map(|_| ResidentStorageBinding {
                    buffer: output,
                    offset: 0,
                    size: output_size,
                }),
            },
            params,
        )
        .map_err(Error::from)
}

pub(super) fn build_params(
    group: ModularGroup,
    group_index: usize,
    stream_segment: GroupStreamSegment,
    status_index: u32,
    source: &DecodeSource,
    reconstruction_specialization: ModularReconstructionSpecialization,
    initialize_chroma: bool,
) -> Result<ShaderParams> {
    let to_u32 = |value: u64, name: &'static str| {
        u32::try_from(value).map_err(|_| Error::backend(format!("{name} exceeds WGSL u32")))
    };
    let plane = |index: usize| -> Result<(u32, u32)> {
        source
            .output
            .layout
            .plane(index)
            .map_or(Ok((0, 0)), |plane| {
                Ok((
                    to_u32(plane.offset, "plane offset")?,
                    to_u32(plane.row_stride, "plane row stride")?,
                ))
            })
    };
    let (plane0_offset, plane0_stride) = plane(0)?;
    let (plane1_offset, plane1_stride) = plane(1)?;
    let (plane2_offset, plane2_stride) = plane(2)?;
    let (plane3_offset, plane3_stride) = plane(3)?;
    let chroma = source.output.layout.plane(1);
    let (fixed_leaf_predictor, fixed_leaf_offset, fixed_leaf_multiplier, fixed_leaf_clusters) =
        match reconstruction_specialization {
            ModularReconstructionSpecialization::ChannelFixed {
                predictor,
                offset,
                multiplier,
                clusters,
                ..
            } => (
                u32::from(predictor.index()),
                u32::from_ne_bytes(offset.to_ne_bytes()),
                multiplier,
                clusters.map(u32::from),
            ),
            ModularReconstructionSpecialization::GenericMetaAdaptive => (0, 0, 0, [0; 4]),
            ModularReconstructionSpecialization::DescriptorMetaAdaptive => (0, 0, 0, [0; 4]),
        };
    let decoded_symbol_count = group_decoded_symbol_count(&source.profile, group_index, group)?;
    let lz77_window_words =
        group_lz77_window_words(&source.profile, group_index, group, decoded_symbol_count)?;
    let fixed_output_mode = if source.profile.progressive_dc.is_some() {
        FixedGradientOutputMode::ResidentOnly
    } else {
        refine_fixed_gradient_output_mode(
            fixed_gradient_output_mode(
                source.profile.channels.count(),
                source.profile.bits_per_sample,
                &source.output,
                reconstruction_specialization,
            ),
            lz77_window_words,
        )
    };
    let physical_sample_words =
        if fixed_output_mode == FixedGradientOutputMode::CompactNormalizedGray8 {
            compact_gray8_sample_words(group)?
        } else if uses_generalized_channel_layout(&source.profile) {
            resident_entropy_plan(&source.profile, group_index)?
                .inverse_plan
                .arena_words()
        } else {
            decoded_symbol_count
        };
    let entropy_state_offset = group_entropy_state_offset_words(
        &source.profile,
        group_index,
        group,
        decoded_symbol_count,
        physical_sample_words,
    )?;
    let group_plan = resident_entropy_plan(&source.profile, group_index)?;
    let wp_header = group_plan.wp_header;
    let ma_config = group_plan.ma_config.resolve(&source.profile.ma_config);
    Ok(ShaderParams {
        entropy: EntropyStreamParams {
            token_start: 0,
            token_end: stream_segment.available_token_end,
            lz77_window_mask: lz77_window_words.saturating_sub(1),
        },
        window_logical_start: stream_segment.window_logical_start,
        window_upload_start: stream_segment.window_upload_start,
        stream_token_end: stream_segment.stream_token_end,
        window_yield_end: stream_segment.window_yield_end,
        window_flags: stream_segment.flags,
        entropy_state_offset,
        width: group.width,
        height: group.height,
        origin_x: group.x,
        origin_y: group.y,
        sample_count: group.sample_count()?,
        initialize_chroma: u32::from(initialize_chroma),
        source_channels: source.profile.channels.count(),
        channel_layout_offset: source
            .channel_layout_offsets
            .get(group_index)
            .copied()
            .unwrap_or(0),
        metadata_base: source.ma_metadata_offsets.get(group_index).copied().ok_or(
            Error::EngineContract("Modular group MA metadata offset is missing"),
        )?,
        source_bits: u32::from(source.profile.bits_per_sample),
        source_mask: (1u32 << source.profile.bits_per_sample) - 1,
        needs_self_correcting: u32::from(ma_config.needs_self_correcting()),
        output_kind: source.output.kind as u32,
        transfer: source.output.transfer,
        limited_range: u32::from(source.output.limited_range),
        channels: source.output.channels,
        order: source.output.order,
        bits: source.output.bits,
        storage_bits: source.output.storage_bits,
        plane0_offset,
        plane0_stride,
        plane1_offset,
        plane1_stride,
        plane2_offset,
        plane2_stride,
        plane3_offset,
        plane3_stride,
        chroma_width: chroma.map_or(0, |plane| plane.sample_extent.width),
        chroma_height: chroma.map_or(0, |plane| plane.sample_extent.height),
        logical_size: to_u32(source.output.layout.logical_size, "output logical size")?,
        numeric_mapping: source.output.numeric_mapping,
        status_index,
        stream_index: group.stream_index,
        fixed_leaf_predictor,
        fixed_leaf_offset,
        fixed_leaf_multiplier,
        fixed_leaf_cluster0: fixed_leaf_clusters[0],
        fixed_leaf_cluster1: fixed_leaf_clusters[1],
        fixed_leaf_cluster2: fixed_leaf_clusters[2],
        fixed_leaf_cluster3: fixed_leaf_clusters[3],
        fixed_output_mode: fixed_output_mode as u32,
        wp_p1: wp_header.p1,
        wp_p2: wp_header.p2,
        wp_p3a: wp_header.p3a,
        wp_p3b: wp_header.p3b,
        wp_p3c: wp_header.p3c,
        wp_p3d: wp_header.p3d,
        wp_p3e: wp_header.p3e,
        wp_w0: wp_header.w0,
        wp_w1: wp_header.w1,
        wp_w2: wp_header.w2,
        wp_w3: wp_header.w3,
    })
}

pub(super) fn align4(value: u64) -> Result<u64> {
    value
        .checked_add(3)
        .map(|value| value & !3)
        .ok_or_else(|| Error::backend("GPU buffer size overflow"))
}

pub(super) fn align16(value: u64) -> Result<u64> {
    value
        .checked_add(15)
        .map(|value| value & !15)
        .ok_or_else(|| Error::backend("GPU buffer size overflow"))
}

pub(super) fn align_to(value: u64, alignment: u64, name: &'static str) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Error::backend(format!(
            "{name} alignment is not a non-zero power of two"
        )));
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| Error::backend(format!("{name} size overflow")))
}
