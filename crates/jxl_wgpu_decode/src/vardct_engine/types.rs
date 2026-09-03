use jxl_gpu_formats::LayoutError;
use jxl_wgpu::{
    ImageUpsampleDispatchMemoryPlan, MemoryBudgetError, PreparedUpsamplingMemoryPlan,
    ResidentChromaUpsampleError, ResidentChromaUpsampleMemoryPlan, ResidentEpfError,
    ResidentEpfMemoryPlan, ResidentGaborishError, ResidentGaborishMemoryPlan,
    ResidentImageUpsampleError, ResidentImageUpsampleWeights, ResidentVarDctError,
    ResidentVarDctMemoryPlan, SubmissionPollerError,
};
use thiserror::Error;

use crate::Error as DecodeError;
use crate::progressive_dc::{ProgressiveDcGpuError, ProgressiveDcPackParams};
use crate::vardct_artifact::{
    GpuVarDctArtifactStatus, GpuVarDctLoweringError, HfMetadataLoweringParams, VarDctArtifactError,
};
use crate::vardct_epf::{EpfSigmaError, EpfSigmaMemoryPlan};
use crate::vardct_lf::AdaptiveLfParams;
use crate::vardct_output::{VarDctOutputError, VarDctOutputPlan};
use crate::vardct_packet::{
    BoundedVarDctPacketError, BoundedVarDctPacketPlan, GpuVarDctPacketError, GpuVarDctPacketStatus,
    VarDctModularParams, VarDctPacketControl, packet_execution_state_bytes,
};
use crate::vardct_pass_group::{
    GpuHfCoefficientError, HF_COEFFICIENT_EXECUTION_STATE_BYTES, HF_COEFFICIENT_STATUS_BYTES,
    HfCoefficientExecutionPlan, HfCoefficientPlanError,
};
use crate::vardct_resource::{VarDctResourceError, VarDctResourceLayout, VarDctResourceParams};

use super::source::VarDctGroupSource;
use super::window_plan::{CombinedPacketWindowExecutionPlan, LfPacketWindowExecutionPlan};

pub(super) const PACKET_STATUS_BYTES: u64 = std::mem::size_of::<GpuVarDctPacketStatus>() as u64;
pub(super) const ARTIFACT_STATUS_BYTES: u64 = std::mem::size_of::<GpuVarDctArtifactStatus>() as u64;
pub(super) const ADAPTIVE_LF_WORKGROUP_BYTES: u64 = 18 * 18 * 16;
pub(super) const VAR_DCT_PARSE_LIMIT_BYTES: u64 = 16 * 1024 * 1024;

fn align4(value: u64) -> Result<u64, VarDctDecodeError> {
    value
        .checked_add(3)
        .map(|value| value & !3)
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "four-byte buffer alignment",
        })
}

/// Typed production-path failure for GPU-resident VarDCT decode.
#[derive(Debug, Error)]
pub enum VarDctDecodeError {
    #[error("the bounded VarDCT engine does not support the requested output format")]
    UnsupportedOutput,
    #[error("the bounded VarDCT engine does not implement image orientation {orientation}")]
    UnsupportedOrientation { orientation: u32 },
    #[error("the bounded VarDCT engine requires the standard sRGB D65 presentation encoding")]
    UnsupportedColorEncoding,
    #[error("the VarDCT frame declares invalid EPF iteration count {iterations}")]
    InvalidEpfIterations { iterations: u32 },
    #[error("the VarDCT frame declares invalid {channel} quant-matrix scale {scale}")]
    InvalidQuantMatrixScale { channel: &'static str, scale: u32 },
    #[error("the XYB image header does not contain an inverse opsin matrix")]
    MissingInverseOpsin,
    #[error("the bounded VarDCT engine requires exactly one image frame")]
    MissingFrame,
    #[error(transparent)]
    Frontend(#[from] crate::vardct_frontend::VarDctFrontendError),
    #[error(transparent)]
    Packet(#[from] BoundedVarDctPacketError),
    #[error(transparent)]
    PacketGpu(#[from] GpuVarDctPacketError),
    #[error(transparent)]
    Artifact(#[from] VarDctArtifactError),
    #[error(transparent)]
    ArtifactGpu(#[from] GpuVarDctLoweringError),
    #[error(transparent)]
    HfCoefficientPlan(#[from] HfCoefficientPlanError),
    #[error(transparent)]
    HfCoefficientGpu(#[from] GpuHfCoefficientError),
    #[error(transparent)]
    Resource(#[from] VarDctResourceError),
    #[error(transparent)]
    Resident(#[from] ResidentVarDctError),
    #[error(transparent)]
    ChromaUpsample(#[from] ResidentChromaUpsampleError),
    #[error(transparent)]
    ImageUpsample(#[from] ResidentImageUpsampleError),
    #[error(transparent)]
    Gaborish(#[from] ResidentGaborishError),
    #[error(transparent)]
    Epf(#[from] ResidentEpfError),
    #[error(transparent)]
    EpfSigma(#[from] EpfSigmaError),
    #[error(transparent)]
    Output(#[from] VarDctOutputError),
    #[error(transparent)]
    ProgressiveDc(#[from] ProgressiveDcGpuError),
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error("VarDCT GPU memory backpressure: {0}")]
    MemoryBackpressure(#[from] MemoryBudgetError),
    #[error("VarDCT GPU submission-poll backpressure: {0}")]
    PollBackpressure(#[from] SubmissionPollerError),
    #[error("VarDCT GPU memory arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
    #[error("a progressive-DC VarDCT frame was submitted without its resident LF source")]
    MissingProgressiveDcSource,
    #[error("a resident progressive-DC LF source was supplied to a frame that does not use one")]
    UnexpectedProgressiveDcSource,
    #[error("VarDCT engine contract failed: {detail}")]
    EngineContract { detail: &'static str },
    #[error(
        "the bounded VarDCT entropy window is {limit_bytes} bytes, but at least {minimum_bytes} bytes are required"
    )]
    EntropyStreamWindowTooSmall {
        limit_bytes: u64,
        minimum_bytes: u64,
    },
    #[error(
        "the minimum-window VarDCT frame needs {required_bytes} bytes, but the shared memory budget permits {limit_bytes} bytes"
    )]
    MemoryBudgetTooSmall {
        required_bytes: u64,
        limit_bytes: u64,
    },
    #[error("VarDCT {resource} needs {required} bytes, device permits {available}")]
    DeviceLimit {
        resource: &'static str,
        required: u64,
        available: u64,
    },
    #[error("VarDCT artifact status {field} is {actual}, expected {expected}")]
    ArtifactStatus {
        field: &'static str,
        expected: u32,
        actual: u32,
    },
    #[error("mapped VarDCT validation status has an invalid {status} ABI")]
    StatusAbi { status: &'static str },
    #[error("raw HF dequantization matrix {matrix} GPU setup or execution failed")]
    RawHfDequantGpu {
        matrix: usize,
        #[source]
        source: Box<DecodeError>,
    },
    #[error(
        "raw HF dequantization matrix {matrix} failed with status {code} after {decoded_samples}/{expected_samples} samples at bit {cursor}/{expected_cursor}"
    )]
    RawHfDequantStatus {
        matrix: usize,
        code: u32,
        decoded_samples: u32,
        expected_samples: u32,
        cursor: u32,
        expected_cursor: u32,
    },
    #[error("raw HF dequantization matrix {matrix} produced a non-positive or oversized weight")]
    RawHfDequantValue { matrix: usize },
    #[error("VarDCT {component} has {actual} LF-group plans; expected {expected}")]
    GroupPlanCount {
        component: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("VarDCT GPU completion was consumed more than once")]
    CompletionConsumed,
    #[error(
        "unvalidated VarDCT output is unavailable until the local-tree HF submission is queued"
    )]
    UnvalidatedOutputNotSubmitted,
    #[error("VarDCT kernel '{kernel}' configuration failed: {message}")]
    KernelPolicy {
        kernel: &'static str,
        message: String,
    },
    #[error("VarDCT entropy-window planning failed")]
    EntropyWindowPlan {
        #[source]
        source: Box<DecodeError>,
    },
    #[error("VarDCT entropy-window execution contract failed: {detail}")]
    EntropyWindowContract { detail: &'static str },
    #[error("VarDCT codestream source contract failed")]
    CodestreamSource {
        #[source]
        source: Box<DecodeError>,
    },
    #[error("failed to access the mapped VarDCT codestream buffer: {source}")]
    CodestreamMap {
        #[source]
        source: wgpu::MapRangeError,
    },
}

/// Exact GPU buffer accounting for one bounded VarDCT frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctDecodeMemoryStats {
    /// Effective packet/AC entropy upload cap after caller, device, and frame-budget policy.
    pub resolved_stream_window_limit_bytes: u64,
    pub codestream_bytes: u64,
    pub modular_metadata_bytes: u64,
    /// True when HF-local metadata size is discovered after the LF cursor map. The initial
    /// reservation covers all LF-local metadata; a larger HF peak is admitted through the same
    /// shared byte budget before the second submission.
    pub deferred_hf_modular_metadata: bool,
    /// True when a single-entry progressive-DC packet discovers HF-global after a GPU metadata
    /// stage. Fixed scratch/status capacity is admitted up front; descriptor/order uploads are
    /// admitted from the same budget after the cursor map.
    pub deferred_hf_coefficients: bool,
    pub reconstructed_bytes: u64,
    pub raw_metadata_bytes: u64,
    pub coefficient_bytes: u64,
    pub packet_status_bytes: u64,
    pub validation_staging_bytes: u64,
    pub packet_control_bytes: u64,
    pub modular_params_bytes: u64,
    /// Reusable upload shared by staged local-tree LF and HF packet entropy. Zero when every
    /// packet binds the retained whole codestream.
    pub packet_stream_window_bytes: u64,
    /// Ordered initial packet batches. For local trees this is the staged-LF count; for a
    /// combined/global-tree packet it covers the complete LF/HF packet.
    pub packet_stream_batch_count: usize,
    /// Resume records included in `reconstructed_bytes`, reported separately for ABI auditing.
    pub packet_execution_state_bytes: u64,
    pub lf_temporary_bytes: u64,
    pub resource_bytes: u64,
    pub resource_uniform_bytes: u64,
    pub progressive_dc_pack_uniform_bytes: u64,
    pub adaptive_lf_uniform_bytes: u64,
    pub artifact_bytes: u64,
    pub occupancy_bytes: u64,
    pub artifact_uniform_bytes: u64,
    pub hf_entropy_bundle_bytes: u64,
    /// Reusable AC entropy upload. Zero when every pass group binds the original whole stream.
    pub hf_stream_window_bytes: u64,
    /// Ordered AC dispatch batches when the reusable stream path is active.
    pub hf_stream_batch_count: usize,
    pub hf_params_bytes: u64,
    pub hf_lz77_scratch_bytes: u64,
    pub hf_execution_state_bytes: u64,
    pub hf_status_bytes: u64,
    pub hf_order_table_bytes: u64,
    pub hf_sink_uniform_bytes: u64,
    /// Exact Cb/X, Y, and Cr/B resident plane allocations. Subsampled JPEG components use
    /// physically smaller buffers rather than full-resolution padding.
    pub resident_plane_bytes: [u64; 3],
    pub resident_image_bytes: u64,
    /// Encoded-resolution destinations allocated for shifted components before restoration or
    /// frame upsampling.
    pub component_upsample_bytes: u64,
    /// One 32-byte interpolation uniform for each shifted component.
    pub component_upsample_uniform_bytes: u64,
    /// Three full-resolution ping-pong destinations shared by Gaborish and EPF.
    pub restoration_scratch_bytes: u64,
    pub gaborish_uniform_bytes: u64,
    pub epf_sigma_bytes: u64,
    pub epf_sigma_uniform_bytes: u64,
    pub epf_filter_uniform_bytes: u64,
    /// Three presentation-resolution F32 planes produced by frame upsampling.
    pub frame_upsample_image_bytes: u64,
    /// Expanded phase-major 5x5 kernels retained by the fused frame upsampling dispatch.
    pub frame_upsample_weight_bytes: u64,
    /// Fused three-plane frame upsampling uniform.
    pub frame_upsample_uniform_bytes: u64,
    pub resident_transient_bytes: u64,
    pub output_uniform_bytes: u64,
    /// Packed RGB8 storage retained until the final [`GpuBufferLease`] clone is dropped.
    pub output_lease_bytes: u64,
    /// All non-output GPU buffers retained through status validation.
    pub transient_bytes: u64,
    pub total_frame_bytes: u64,
}

impl VarDctDecodeMemoryStats {
    pub(super) fn plan(inputs: VarDctDecodeMemoryInputs<'_>) -> Result<Self, VarDctDecodeError> {
        let VarDctDecodeMemoryInputs {
            stream_limit,
            codestream_len,
            packet,
            groups,
            lf_packet_windows,
            combined_packet_windows,
            resource,
            hf_coefficients,
            deferred_hf,
            adaptive_lf_smoothing,
            restoration_scratch,
            gaborish,
            epf_sigma,
            epf_iterations,
            frame_upsampling,
            resident,
            output,
        } = inputs;
        fn checked_sum(
            values: impl IntoIterator<Item = u64>,
            field: &'static str,
        ) -> Result<u64, VarDctDecodeError> {
            values.into_iter().try_fold(0_u64, |total, value| {
                total
                    .checked_add(value)
                    .ok_or(VarDctDecodeError::ArithmeticOverflow { field })
            })
        }
        let checked_words = |words: u64, field: &'static str| {
            words
                .checked_mul(4)
                .ok_or(VarDctDecodeError::ArithmeticOverflow { field })
        };
        let codestream_bytes = align4(u64::try_from(codestream_len).map_err(|_| {
            VarDctDecodeError::ArithmeticOverflow {
                field: "codestream upload length",
            }
        })?)?;
        let modular_metadata_words =
            if packet.requires_local_tree_staging() || packet.profile.uses_lf_frame {
                packet.groups.iter().try_fold(0_u64, |total, group| {
                    let words = u64::try_from(group.lf_modular.metadata.len()).map_err(|_| {
                        VarDctDecodeError::ArithmeticOverflow {
                            field: "LF-local Modular metadata length",
                        }
                    })?;
                    total
                        .checked_add(words)
                        .ok_or(VarDctDecodeError::ArithmeticOverflow {
                            field: "LF-local Modular metadata words",
                        })
                })?
            } else {
                u64::try_from(packet.modular_metadata.len()).map_err(|_| {
                    VarDctDecodeError::ArithmeticOverflow {
                        field: "Modular metadata length",
                    }
                })?
            };
        let modular_metadata_bytes =
            checked_words(modular_metadata_words, "Modular metadata bytes")?;
        let predictor_capacity =
            packet.needs_self_correcting || packet.requires_local_tree_staging();
        let mut reconstructed_words = Vec::with_capacity(packet.groups.len());
        for group in &packet.groups {
            reconstructed_words.push(u64::from(group.reconstructed_words(predictor_capacity)?));
        }
        let reconstructed_bytes = checked_words(
            checked_sum(reconstructed_words, "LF reconstruction word total")?,
            "LF reconstruction bytes",
        )?;
        let raw_metadata_bytes = checked_words(
            checked_sum(
                groups
                    .iter()
                    .map(|group| u64::from(group.control.capacities[1])),
                "raw HF metadata word total",
            )?,
            "raw HF metadata bytes",
        )?;
        let coefficient_bytes = checked_words(
            u64::from(packet.total_coefficient_words()?),
            "coefficient bytes",
        )?;
        let group_count =
            u64::try_from(groups.len()).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
                field: "LF-group count",
            })?;
        let packet_status_bytes = group_count.checked_mul(PACKET_STATUS_BYTES).ok_or(
            VarDctDecodeError::ArithmeticOverflow {
                field: "packet status bytes",
            },
        )?;
        let hf_entropy_bundle_bytes = hf_coefficients
            .map(|plan| checked_words(plan.entropy_words.len() as u64, "HF entropy bundle bytes"))
            .transpose()?
            .unwrap_or(0);
        let hf_uses_stream_windows =
            hf_coefficients.is_some_and(HfCoefficientExecutionPlan::uses_bounded_stream_windows);
        let hf_stream_window_bytes = if hf_uses_stream_windows {
            hf_coefficients.map_or(0, HfCoefficientExecutionPlan::stream_window_bytes)
        } else {
            0
        };
        let hf_stream_batch_count = if hf_uses_stream_windows {
            hf_coefficients.map_or(0, HfCoefficientExecutionPlan::stream_batch_count)
        } else {
            0
        };
        let hf_params_bytes = if hf_uses_stream_windows {
            hf_coefficients.map_or(0, HfCoefficientExecutionPlan::reusable_params_bytes)
        } else {
            hf_coefficients
                .map(|plan| {
                    checked_sum(
                        plan.groups.iter().map(|group| group.params.len() as u64),
                        "HF parameter count",
                    )?
                    .checked_mul(std::mem::size_of::<
                        crate::vardct_pass_group::HfCoefficientPassParams,
                    >() as u64)
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "HF parameter bytes",
                    })
                })
                .transpose()?
                .unwrap_or(0)
        };
        let hf_lz77_scratch_bytes = hf_coefficients.map_or_else(
            || deferred_hf.map_or(0, |plan| plan.lz77_scratch_bytes),
            HfCoefficientExecutionPlan::lz77_scratch_bytes,
        );
        let hf_execution_state_bytes = hf_coefficients.map_or_else(
            || deferred_hf.map_or(0, |plan| plan.execution_state_bytes),
            HfCoefficientExecutionPlan::execution_state_bytes,
        );
        let hf_status_bytes = hf_coefficients.map_or_else(
            || deferred_hf.map_or(0, |plan| plan.status_bytes),
            HfCoefficientExecutionPlan::status_bytes,
        );
        let hf_order_table_bytes = hf_coefficients
            .map(|plan| checked_words(plan.order_words.len() as u64, "HF order-table bytes"))
            .transpose()?
            .unwrap_or(0);
        let hf_sink_uniform_bytes = hf_coefficients
            .map(|plan| {
                (plan.groups.len() as u64)
                    .checked_mul(
                        std::mem::size_of::<crate::vardct_artifact::HfCoefficientSinkParams>()
                            as u64,
                    )
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "HF sink uniform bytes",
                    })
            })
            .transpose()?
            .unwrap_or_else(|| deferred_hf.map_or(0, |plan| plan.sink_uniform_bytes));
        let hf_params_bytes = if hf_coefficients.is_none() {
            deferred_hf.map_or(hf_params_bytes, |plan| plan.params_bytes)
        } else {
            hf_params_bytes
        };
        let artifact_status_bytes = group_count.checked_mul(ARTIFACT_STATUS_BYTES).ok_or(
            VarDctDecodeError::ArithmeticOverflow {
                field: "artifact status bytes",
            },
        )?;
        let validation_staging_bytes = packet_status_bytes
            .checked_add(artifact_status_bytes)
            .and_then(|bytes| bytes.checked_add(hf_status_bytes))
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "VarDCT validation staging bytes",
            })?;
        let packet_control_bytes = group_count
            .checked_mul(std::mem::size_of::<VarDctPacketControl>() as u64)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "packet control bytes",
            })?;
        let modular_params_bytes = group_count
            .checked_mul(std::mem::size_of::<VarDctModularParams>() as u64)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "Modular parameter bytes",
            })?;
        debug_assert!(lf_packet_windows.is_none() || combined_packet_windows.is_none());
        let packet_stream_window_bytes = lf_packet_windows
            .map(|plan| plan.stream_bytes)
            .or_else(|| combined_packet_windows.map(|plan| plan.stream_bytes))
            .unwrap_or(0);
        let packet_stream_batch_count = lf_packet_windows
            .map(LfPacketWindowExecutionPlan::batch_count)
            .or_else(|| combined_packet_windows.map(CombinedPacketWindowExecutionPlan::batch_count))
            .unwrap_or(0);
        let packet_execution_state_bytes = packet
            .groups
            .iter()
            .try_fold(0_u64, |total, _| {
                total.checked_add(packet_execution_state_bytes(
                    packet.needs_self_correcting || packet.requires_local_tree_staging(),
                ))
            })
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "VarDCT packet execution-state bytes",
            })?;
        let lf_temporary_bytes = u64::from(resource.block_count).checked_mul(16).ok_or(
            VarDctDecodeError::ArithmeticOverflow {
                field: "LF temporary bytes",
            },
        )?;
        let lf_temporary_bytes = if packet.profile.uses_lf_frame || !adaptive_lf_smoothing {
            0
        } else {
            lf_temporary_bytes
        };
        let resource_bytes = resource.bytes();
        let resource_uniform_bytes = if packet.profile.uses_lf_frame {
            0
        } else {
            group_count
                .checked_mul(std::mem::size_of::<VarDctResourceParams>() as u64)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "resource uniform bytes",
                })?
        };
        let progressive_dc_pack_uniform_bytes = if packet.profile.uses_lf_frame {
            std::mem::size_of::<ProgressiveDcPackParams>() as u64
        } else {
            0
        };
        let adaptive_lf_uniform_bytes = if adaptive_lf_smoothing {
            std::mem::size_of::<AdaptiveLfParams>() as u64
        } else {
            0
        };
        let artifact_bytes = checked_sum(
            groups
                .iter()
                .map(|group| group.artifact_layout.artifact_bytes),
            "artifact byte total",
        )?;
        let occupancy_bytes = checked_sum(
            groups
                .iter()
                .map(|group| group.artifact_layout.occupancy_bytes),
            "occupancy byte total",
        )?;
        let artifact_uniform_bytes = group_count
            .checked_mul(std::mem::size_of::<HfMetadataLoweringParams>() as u64)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "artifact uniform bytes",
            })?;
        let [blocks_x, blocks_y] = packet.block_extent();
        let resident_plane_bytes = packet.profile.channel_shifts.map(|shift| {
            u64::from(blocks_x >> shift.horizontal)
                .checked_mul(8)
                .and_then(|width| {
                    u64::from(blocks_y >> shift.vertical)
                        .checked_mul(8)
                        .and_then(|height| width.checked_mul(height))
                })
                .and_then(|pixels| pixels.checked_mul(std::mem::size_of::<f32>() as u64))
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "resident component plane bytes",
                })
        });
        let [resident_x, resident_y, resident_b] = resident_plane_bytes;
        let resident_plane_bytes = [resident_x?, resident_y?, resident_b?];
        let resident_image_bytes =
            checked_sum(resident_plane_bytes, "resident component image bytes")?;
        let full_plane_bytes = u64::from(blocks_x)
            .checked_mul(8)
            .and_then(|width| {
                u64::from(blocks_y)
                    .checked_mul(8)
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| pixels.checked_mul(std::mem::size_of::<f32>() as u64))
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "full-resolution restoration plane bytes",
            })?;
        let shifted_channel_count = packet
            .profile
            .channel_shifts
            .into_iter()
            .filter(|shift| shift.is_subsampled())
            .count() as u64;
        let component_upsample_required = restoration_scratch || frame_upsampling.is_some();
        let component_upsample_bytes = if component_upsample_required {
            full_plane_bytes.checked_mul(shifted_channel_count).ok_or(
                VarDctDecodeError::ArithmeticOverflow {
                    field: "component upsample bytes",
                },
            )?
        } else {
            0
        };
        let component_upsample_uniform_bytes = if component_upsample_required {
            ResidentChromaUpsampleMemoryPlan::UNIFORM_BYTES
                .checked_mul(shifted_channel_count)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "component upsample uniform bytes",
                })?
        } else {
            0
        };
        let restoration_scratch_bytes = if restoration_scratch {
            full_plane_bytes
                .checked_mul(3)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "restoration scratch bytes",
                })?
        } else {
            0
        };
        let gaborish_uniform_bytes = if gaborish {
            ResidentGaborishMemoryPlan::new().uniform_bytes
        } else {
            0
        };
        let epf_sigma_bytes = epf_sigma.map_or(0, |plan| plan.sigma_bytes);
        let epf_sigma_uniform_bytes = epf_sigma
            .map(|plan| {
                plan.uniform_bytes.checked_mul(group_count).ok_or(
                    VarDctDecodeError::ArithmeticOverflow {
                        field: "EPF sigma uniform bytes",
                    },
                )
            })
            .transpose()?
            .unwrap_or(0);
        let epf_filter_uniform_bytes = u64::from(epf_iterations)
            .checked_mul(ResidentEpfMemoryPlan::new().uniform_bytes)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "EPF filter uniform bytes",
            })?;
        let frame_upsample_image_bytes = if frame_upsampling.is_some() {
            u64::from(packet.profile.presentation_width)
                .checked_mul(u64::from(packet.profile.presentation_height))
                .and_then(|pixels| pixels.checked_mul(std::mem::size_of::<f32>() as u64))
                .and_then(|plane| plane.checked_mul(3))
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "frame upsample image bytes",
                })?
        } else {
            0
        };
        let frame_upsample_weight_bytes = frame_upsampling
            .map(|weights| PreparedUpsamplingMemoryPlan::from(weights).storage_bytes)
            .unwrap_or(0);
        let frame_upsample_uniform_bytes = if frame_upsampling.is_some() {
            ImageUpsampleDispatchMemoryPlan::UNIFORM_BYTES
        } else {
            0
        };
        let resident_transient_bytes = checked_sum(
            resident.iter().map(|plan| plan.total_bytes),
            "resident VarDCT transient bytes",
        )?;
        let output_uniform_bytes = output.memory.uniform_bytes;
        let output_lease_bytes = output.memory.output_storage_bytes;
        let transient_bytes = [
            codestream_bytes,
            modular_metadata_bytes,
            reconstructed_bytes,
            raw_metadata_bytes,
            coefficient_bytes,
            packet_status_bytes,
            validation_staging_bytes,
            packet_control_bytes,
            modular_params_bytes,
            packet_stream_window_bytes,
            lf_temporary_bytes,
            resource_bytes,
            resource_uniform_bytes,
            progressive_dc_pack_uniform_bytes,
            adaptive_lf_uniform_bytes,
            artifact_bytes,
            occupancy_bytes,
            artifact_uniform_bytes,
            hf_entropy_bundle_bytes,
            hf_stream_window_bytes,
            hf_params_bytes,
            hf_lz77_scratch_bytes,
            hf_execution_state_bytes,
            hf_status_bytes,
            hf_order_table_bytes,
            hf_sink_uniform_bytes,
            resident_image_bytes,
            component_upsample_bytes,
            component_upsample_uniform_bytes,
            restoration_scratch_bytes,
            gaborish_uniform_bytes,
            epf_sigma_bytes,
            epf_sigma_uniform_bytes,
            epf_filter_uniform_bytes,
            frame_upsample_image_bytes,
            frame_upsample_weight_bytes,
            frame_upsample_uniform_bytes,
            resident_transient_bytes,
            output_uniform_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |total, value| {
            total
                .checked_add(value)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "VarDCT transient byte total",
                })
        })?;
        let total_frame_bytes = transient_bytes.checked_add(output_lease_bytes).ok_or(
            VarDctDecodeError::ArithmeticOverflow {
                field: "VarDCT frame byte total",
            },
        )?;
        Ok(Self {
            resolved_stream_window_limit_bytes: stream_limit,
            codestream_bytes,
            modular_metadata_bytes,
            deferred_hf_modular_metadata: packet.requires_local_tree_staging(),
            deferred_hf_coefficients: deferred_hf.is_some(),
            reconstructed_bytes,
            raw_metadata_bytes,
            coefficient_bytes,
            packet_status_bytes,
            validation_staging_bytes,
            packet_control_bytes,
            modular_params_bytes,
            packet_stream_window_bytes,
            packet_stream_batch_count,
            packet_execution_state_bytes,
            lf_temporary_bytes,
            resource_bytes,
            resource_uniform_bytes,
            progressive_dc_pack_uniform_bytes,
            adaptive_lf_uniform_bytes,
            artifact_bytes,
            occupancy_bytes,
            artifact_uniform_bytes,
            hf_entropy_bundle_bytes,
            hf_stream_window_bytes,
            hf_stream_batch_count,
            hf_params_bytes,
            hf_lz77_scratch_bytes,
            hf_execution_state_bytes,
            hf_status_bytes,
            hf_order_table_bytes,
            hf_sink_uniform_bytes,
            resident_plane_bytes,
            resident_image_bytes,
            component_upsample_bytes,
            component_upsample_uniform_bytes,
            restoration_scratch_bytes,
            gaborish_uniform_bytes,
            epf_sigma_bytes,
            epf_sigma_uniform_bytes,
            epf_filter_uniform_bytes,
            frame_upsample_image_bytes,
            frame_upsample_weight_bytes,
            frame_upsample_uniform_bytes,
            resident_transient_bytes,
            output_uniform_bytes,
            output_lease_bytes,
            transient_bytes,
            total_frame_bytes,
        })
    }
}

pub(super) struct VarDctDecodeMemoryInputs<'a> {
    pub(super) stream_limit: u64,
    pub(super) codestream_len: usize,
    pub(super) packet: &'a BoundedVarDctPacketPlan,
    pub(super) groups: &'a [VarDctGroupSource],
    pub(super) lf_packet_windows: Option<&'a LfPacketWindowExecutionPlan>,
    pub(super) combined_packet_windows: Option<&'a CombinedPacketWindowExecutionPlan>,
    pub(super) resource: VarDctResourceLayout,
    pub(super) hf_coefficients: Option<&'a HfCoefficientExecutionPlan>,
    pub(super) deferred_hf: Option<&'a DeferredHfCoefficientLayout>,
    pub(super) adaptive_lf_smoothing: bool,
    pub(super) restoration_scratch: bool,
    pub(super) gaborish: bool,
    pub(super) epf_sigma: Option<EpfSigmaMemoryPlan>,
    pub(super) epf_iterations: u32,
    pub(super) frame_upsampling: Option<&'a ResidentImageUpsampleWeights>,
    pub(super) resident: &'a [ResidentVarDctMemoryPlan],
    pub(super) output: VarDctOutputPlan,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DeferredHfCoefficientGroupLayout {
    pub(super) lz77_scratch_bytes: u64,
    pub(super) execution_state_bytes: u64,
}

#[derive(Clone, Debug)]
pub(super) struct DeferredHfCoefficientLayout {
    pub(super) groups: Vec<DeferredHfCoefficientGroupLayout>,
    pub(super) lz77_scratch_bytes: u64,
    pub(super) execution_state_bytes: u64,
    pub(super) status_bytes: u64,
    pub(super) params_bytes: u64,
    pub(super) sink_uniform_bytes: u64,
}

impl DeferredHfCoefficientLayout {
    pub(super) fn plan(
        packet: &BoundedVarDctPacketPlan,
    ) -> Result<Option<Self>, VarDctDecodeError> {
        if !packet.requires_deferred_hf_coefficients() {
            return Ok(None);
        }
        let pass_group_count = usize::try_from(packet.profile.group_count).map_err(|_| {
            VarDctDecodeError::ArithmeticOverflow {
                field: "deferred HF pass-group count",
            }
        })?;
        let mut local_counts = vec![0_u64; packet.groups.len()];
        for global_group_index in 0..packet.profile.group_count {
            let lf_group = packet
                .profile
                .low_frequency_group_index_for_pass_group(global_group_index)
                .map_err(HfCoefficientPlanError::from)?;
            let lf_group =
                usize::try_from(lf_group).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
                    field: "deferred HF LF-group index",
                })?;
            let count =
                local_counts
                    .get_mut(lf_group)
                    .ok_or(VarDctDecodeError::GroupPlanCount {
                        component: "deferred HF coefficient",
                        expected: packet.groups.len(),
                        actual: lf_group.saturating_add(1),
                    })?;
            *count = count
                .checked_add(1)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "deferred HF local pass-group count",
                })?;
        }
        let max_group_blocks = packet
            .profile
            .group_dimension
            .div_ceil(8)
            .checked_mul(packet.profile.group_dimension.div_ceil(8))
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "deferred HF pass-group block count",
            })?;
        let max_lz77_words = max_group_blocks
            .checked_mul(3 * 64)
            .and_then(u32::checked_next_power_of_two)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "deferred HF LZ77 capacity",
            })?;
        let mut groups = Vec::with_capacity(local_counts.len());
        let mut lz77_scratch_bytes = 0_u64;
        let mut execution_state_bytes = 0_u64;
        for count in local_counts {
            let lz77 = count
                .checked_mul(u64::from(max_lz77_words))
                .and_then(|words| words.checked_mul(4))
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "deferred HF LZ77 bytes",
                })?;
            let execution = count
                .checked_mul(HF_COEFFICIENT_EXECUTION_STATE_BYTES)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "deferred HF execution-state bytes",
                })?;
            lz77_scratch_bytes = lz77_scratch_bytes.checked_add(lz77).ok_or(
                VarDctDecodeError::ArithmeticOverflow {
                    field: "deferred HF LZ77 total",
                },
            )?;
            execution_state_bytes = execution_state_bytes.checked_add(execution).ok_or(
                VarDctDecodeError::ArithmeticOverflow {
                    field: "deferred HF execution-state total",
                },
            )?;
            groups.push(DeferredHfCoefficientGroupLayout {
                lz77_scratch_bytes: lz77,
                execution_state_bytes: execution,
            });
        }
        let pass_group_count =
            u64::try_from(pass_group_count).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
                field: "deferred HF pass-group byte count",
            })?;
        Ok(Some(Self {
            groups,
            lz77_scratch_bytes,
            execution_state_bytes,
            status_bytes: pass_group_count
                .checked_mul(HF_COEFFICIENT_STATUS_BYTES)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "deferred HF status bytes",
                })?,
            params_bytes: pass_group_count
                .checked_mul(
                    std::mem::size_of::<crate::vardct_pass_group::HfCoefficientPassParams>() as u64,
                )
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "deferred HF parameter bytes",
                })?,
            sink_uniform_bytes: u64::try_from(packet.groups.len())
                .ok()
                .and_then(|groups| {
                    groups.checked_mul(std::mem::size_of::<
                        crate::vardct_artifact::HfCoefficientSinkParams,
                    >() as u64)
                })
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "deferred HF sink uniform bytes",
                })?,
        }))
    }
}
