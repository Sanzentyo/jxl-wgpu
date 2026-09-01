//! Runtime-neutral GPU submission engine for the bounded standard VarDCT profile.
//!
//! The accepted codestream profile is intentionally bounded and authoritative: one still XYB
//! frame, independently bounded LF groups, GPU-decoded mixed strategy/quantization/correlation metadata, and
//! GPU-decoded single-pass AC coefficients for every JPEG XL VarDCT strategy. No pixel,
//! coefficient, transform, quantization, residual, or entropy fallback runs on the CPU.

use std::collections::BTreeMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use jxl_gpu_bitstream::{
    CodestreamInventory, ColourEncodingInventory, ColourSpaceInventory,
    EdgePreservingFilterInventory, GaborishInventory, InventoryLimits, ParseLimits,
    PrimariesInventory, RestorationFilterInventory, TransferFunctionInventory, WhitePointInventory,
};
use jxl_gpu_formats::{
    ChromaLocation2d, ColorRange, ColorSpace, ColorSpec, ColorSpecification, ImageLayout,
    LayoutError, PixelFormat, RgbChannelOrder, TransferFunction, YcbcrEncoding,
};
use jxl_gpu_protocol::{
    ChangedRegions, EpfPass, Extent2d, OutputId, Region, SubmissionToken, TransformKind,
};
use jxl_wgpu::{
    GpuBufferLease, GpuImageFrame, GpuImageOutput, KernelVariant, MemoryBudget,
    MemoryBudgetSnapshot, MemoryPermit, ResidentEpfError, ResidentEpfInputs, ResidentEpfMemoryPlan,
    ResidentEpfParameters, ResidentEpfPipeline, ResidentF32Plane, ResidentGaborishError,
    ResidentGaborishInputs, ResidentGaborishMemoryPlan, ResidentGaborishPipeline,
    ResidentGaborishWeights, ResidentStorageBinding, ResidentVarDctError, ResidentVarDctInputs,
    ResidentVarDctMemoryPlan, ResidentVarDctRenderConfig, ResidentVarDctRenderer,
    ResidentVarDctScratch, SubmissionPollPermit, UnvalidatedGpuImageFrame,
    UnvalidatedGpuImageOutput, WgpuBackend,
};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::entropy_window::{
    GroupEntropyRange, GroupStreamSegment, MIN_STREAM_WINDOW_BYTES, StreamBatch,
    build_stream_batches_for_len,
};
use crate::vardct_artifact::{
    GpuVarDctArtifactStatus, GpuVarDctLoweringError, HfMetadataArtifactConfig,
    HfMetadataLoweringBuffers, HfMetadataLoweringParams, HfMetadataLoweringPipeline,
    VarDctArtifactDeviceLimits, VarDctArtifactError, VarDctArtifactLayout,
};
use crate::vardct_epf::{EpfSigmaConfig, EpfSigmaError, EpfSigmaMemoryPlan, EpfSigmaPipeline};
use crate::vardct_lf::{AdaptiveLfBuffers, AdaptiveLfParams, AdaptiveLfPipeline};
use crate::vardct_output::{
    VarDctInverseOpsin, VarDctOutputConfig, VarDctOutputError, VarDctOutputInputs,
    VarDctOutputPacker, VarDctOutputPlan, VarDctOutputPlane, VarDctOutputScratch,
};
use crate::vardct_packet::{
    BoundedHfMetadataContinuation, BoundedVarDctPacketError, BoundedVarDctPacketPlan,
    GpuVarDctPacketError, GpuVarDctPacketStatus, VarDctModularParams, VarDctPacketBuffers,
    VarDctPacketControl, VarDctPacketPipeline, VarDctPacketValidation,
    packet_execution_state_bytes,
};
use crate::vardct_pass_group::{
    GpuHfCoefficientError, GpuHfCoefficientStatus, HfCoefficientBuffers,
    HfCoefficientExecutionPlan, HfCoefficientGroupExecutionPlan, HfCoefficientPipeline,
    HfCoefficientPlanError,
};
use crate::vardct_resource::{
    VarDctResourceBuffers, VarDctResourceConfig, VarDctResourceError, VarDctResourceLayout,
    VarDctResourceParams, VarDctResourcePipeline,
};
use crate::{
    AnimationMetadata, DecodeProfile, Error as DecodeError, FrameDuration, FrameMetadata,
    GpuCodestream, GpuOutputMapping, GpuOutputRequest, GpuPendingFrame, GpuSubmissionEngine,
    GpuSubmissionSession, PreparedGpuSession, Result as DecodeResult, SubmittedGpuFrame,
};

const PACKET_STATUS_BYTES: u64 = std::mem::size_of::<GpuVarDctPacketStatus>() as u64;
const ARTIFACT_STATUS_BYTES: u64 = std::mem::size_of::<GpuVarDctArtifactStatus>() as u64;
const ADAPTIVE_LF_WORKGROUP_BYTES: u64 = 18 * 18 * 16;
const VAR_DCT_PARSE_LIMIT_BYTES: u64 = 16 * 1024 * 1024;

/// Typed production-path failure for GPU-resident VarDCT decode.
#[derive(Debug, Error)]
pub enum VarDctDecodeError {
    #[error("the bounded VarDCT engine only produces tightly packed sRGB D65 RGB8 output")]
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
    Gaborish(#[from] ResidentGaborishError),
    #[error(transparent)]
    Epf(#[from] ResidentEpfError),
    #[error(transparent)]
    EpfSigma(#[from] EpfSigmaError),
    #[error(transparent)]
    Output(#[from] VarDctOutputError),
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error("VarDCT GPU memory arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
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

/// Exact canonical output supported by [`VarDctSubmissionEngine`].
#[must_use]
pub fn vardct_rgb8_format() -> PixelFormat {
    PixelFormat::rgb8(
        RgbChannelOrder::Rgb,
        false,
        ColorSpecification::Defined(ColorSpec {
            space: ColorSpace::Bt709,
            encoding: YcbcrEncoding::Undefined,
            transfer: TransferFunction::Srgb,
            range: ColorRange::Full,
            chroma_location: ChromaLocation2d::BOTH,
        }),
    )
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
    pub xyb_plane_bytes: u64,
    pub restoration_scratch_bytes: u64,
    pub gaborish_uniform_bytes: u64,
    pub epf_sigma_bytes: u64,
    pub epf_sigma_uniform_bytes: u64,
    pub epf_filter_uniform_bytes: u64,
    pub resident_transient_bytes: u64,
    pub output_uniform_bytes: u64,
    /// Packed RGB8 storage retained until the final [`GpuBufferLease`] clone is dropped.
    pub output_lease_bytes: u64,
    /// All non-output GPU buffers retained through status validation.
    pub transient_bytes: u64,
    pub total_frame_bytes: u64,
}

impl VarDctDecodeMemoryStats {
    fn plan(inputs: VarDctDecodeMemoryInputs<'_>) -> Result<Self, VarDctDecodeError> {
        let VarDctDecodeMemoryInputs {
            stream_limit,
            codestream_len,
            packet,
            groups,
            lf_packet_windows,
            combined_packet_windows,
            resource,
            hf_coefficients,
            adaptive_lf_smoothing,
            restoration_scratch,
            gaborish,
            epf_sigma,
            epf_iterations,
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
        let modular_metadata_words = if packet.requires_local_tree_staging() {
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
        let hf_lz77_scratch_bytes =
            hf_coefficients.map_or(0, HfCoefficientExecutionPlan::lz77_scratch_bytes);
        let hf_execution_state_bytes =
            hf_coefficients.map_or(0, HfCoefficientExecutionPlan::execution_state_bytes);
        let hf_status_bytes = hf_coefficients.map_or(0, HfCoefficientExecutionPlan::status_bytes);
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
            .unwrap_or(0);
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
        let resource_bytes = resource.bytes();
        let resource_uniform_bytes = group_count
            .checked_mul(std::mem::size_of::<VarDctResourceParams>() as u64)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "resource uniform bytes",
            })?;
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
        let pixels = u64::from(blocks_x)
            .checked_mul(8)
            .and_then(|width| {
                u64::from(blocks_y)
                    .checked_mul(8)
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "XYB pixel count",
            })?;
        let xyb_plane_bytes =
            pixels
                .checked_mul(12)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "XYB plane bytes",
                })?;
        let restoration_scratch_bytes = if restoration_scratch {
            xyb_plane_bytes
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
            xyb_plane_bytes,
            restoration_scratch_bytes,
            gaborish_uniform_bytes,
            epf_sigma_bytes,
            epf_sigma_uniform_bytes,
            epf_filter_uniform_bytes,
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
            xyb_plane_bytes,
            restoration_scratch_bytes,
            gaborish_uniform_bytes,
            epf_sigma_bytes,
            epf_sigma_uniform_bytes,
            epf_filter_uniform_bytes,
            resident_transient_bytes,
            output_uniform_bytes,
            output_lease_bytes,
            transient_bytes,
            total_frame_bytes,
        })
    }
}

struct VarDctDecodeMemoryInputs<'a> {
    stream_limit: u64,
    codestream_len: usize,
    packet: &'a BoundedVarDctPacketPlan,
    groups: &'a [VarDctGroupSource],
    lf_packet_windows: Option<&'a LfPacketWindowExecutionPlan>,
    combined_packet_windows: Option<&'a CombinedPacketWindowExecutionPlan>,
    resource: VarDctResourceLayout,
    hf_coefficients: Option<&'a HfCoefficientExecutionPlan>,
    adaptive_lf_smoothing: bool,
    restoration_scratch: bool,
    gaborish: bool,
    epf_sigma: Option<EpfSigmaMemoryPlan>,
    epf_iterations: u32,
    resident: &'a [ResidentVarDctMemoryPlan],
    output: VarDctOutputPlan,
}

struct VarDctPipelines {
    packet: VarDctPacketPipeline,
    resource: VarDctResourcePipeline,
    adaptive_lf: AdaptiveLfPipeline,
    artifact: HfMetadataLoweringPipeline,
    hf_coefficients: HfCoefficientPipeline,
    renderer: ResidentVarDctRenderer,
    gaborish: ResidentGaborishPipeline,
    epf_sigma: EpfSigmaPipeline,
    epf: ResidentEpfPipeline,
    output: VarDctOutputPacker,
    output_variant: KernelVariant,
}

impl VarDctPipelines {
    fn new(backend: &WgpuBackend) -> Result<Self, VarDctDecodeError> {
        let resource_variant =
            resolve_kernel_variant(backend, "vardct_resource", KernelVariant::Lanes64)?;
        let output_variant =
            resolve_kernel_variant(backend, "vardct_output", KernelVariant::Lanes256)?;
        let gaborish_variant =
            resolve_kernel_variant(backend, "vardct_gaborish", KernelVariant::Tile16x16)?;
        let epf_sigma_variant =
            resolve_kernel_variant(backend, "vardct_epf_sigma", KernelVariant::Lanes64)?;
        let epf_variant = resolve_kernel_variant(backend, "vardct_epf", KernelVariant::Tile16x16)?;
        let device = backend.device();
        Ok(Self {
            packet: VarDctPacketPipeline::new(device),
            resource: VarDctResourcePipeline::with_variant(device, resource_variant)?,
            adaptive_lf: AdaptiveLfPipeline::new(device),
            artifact: HfMetadataLoweringPipeline::new(device),
            hf_coefficients: HfCoefficientPipeline::new(device),
            renderer: ResidentVarDctRenderer::new(device),
            gaborish: ResidentGaborishPipeline::with_variant(device, gaborish_variant)?,
            epf_sigma: EpfSigmaPipeline::with_variant(device, epf_sigma_variant)?,
            epf: ResidentEpfPipeline::with_variant(device, epf_variant)?,
            output: VarDctOutputPacker::with_variant(device, output_variant)?,
            output_variant,
        })
    }
}

fn resolve_kernel_variant(
    backend: &WgpuBackend,
    kernel: &'static str,
    default: KernelVariant,
) -> Result<KernelVariant, VarDctDecodeError> {
    let variant = backend
        .kernel_policy()
        .variant_for(kernel, default)
        .map_err(|error| VarDctDecodeError::KernelPolicy {
            kernel,
            message: error.to_string(),
        })?;
    variant
        .validate_for(kernel, &backend.device().limits(), 0)
        .map_err(|error| VarDctDecodeError::KernelPolicy {
            kernel,
            message: error.to_string(),
        })?;
    Ok(variant)
}

/// GPU-only submission engine for the bounded standard regular-VarDCT profile.
#[derive(Clone)]
pub struct VarDctSubmissionEngine {
    backend: WgpuBackend,
    pipelines: Arc<VarDctPipelines>,
    memory: MemoryBudget,
    stream_window_limit: Option<NonZeroU64>,
}

impl VarDctSubmissionEngine {
    pub fn new(backend: WgpuBackend) -> Result<Self, VarDctDecodeError> {
        let memory = backend.transient_memory_budget().clone();
        Self::with_memory_budget(backend, memory)
    }

    /// Uses an explicitly shared byte budget for output, entropy, render, and validation buffers.
    pub fn with_memory_budget(
        backend: WgpuBackend,
        memory: MemoryBudget,
    ) -> Result<Self, VarDctDecodeError> {
        let pipelines = Arc::new(VarDctPipelines::new(&backend)?);
        Ok(Self {
            backend,
            pipelines,
            memory,
            stream_window_limit: None,
        })
    }

    /// Caps reusable VarDCT entropy uploads.
    ///
    /// Combined/global-tree packets, staged local-tree LF/HF packets, and AC pass groups enforce
    /// this caller upper bound. Device limits and the shared per-frame byte budget may resolve a
    /// smaller four-byte-aligned cap. Recursive entropy streams will adopt the same policy with
    /// their resume state.
    #[must_use]
    pub fn with_stream_window_limit(mut self, limit: NonZeroU64) -> Self {
        self.stream_window_limit = Some(limit);
        self
    }

    /// Returns the caller-supplied upper bound, not a session's budget-resolved cap. The latter is
    /// reported by [`VarDctDecodeMemoryStats::resolved_stream_window_limit_bytes`].
    #[must_use]
    pub const fn stream_window_limit(&self) -> Option<NonZeroU64> {
        self.stream_window_limit
    }

    #[must_use]
    pub const fn backend(&self) -> &WgpuBackend {
        &self.backend
    }

    #[must_use]
    pub fn in_flight_memory_stats(&self) -> MemoryBudgetSnapshot {
        self.memory.snapshot()
    }

    pub(crate) fn open_with_inventory(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
        inventory: &CodestreamInventory,
    ) -> DecodeResult<PreparedGpuSession<VarDctDecodeSession>> {
        self.open_with_inventory_data(codestream, request, inventory)
    }

    pub(crate) fn open_with_inventory_data(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
        inventory: &CodestreamInventory,
    ) -> DecodeResult<PreparedGpuSession<VarDctDecodeSession>> {
        let source = prepare_source(
            &self.backend,
            codestream,
            request,
            inventory,
            self.pipelines.output_variant,
            self.stream_window_limit,
            self.memory.snapshot().limit_bytes,
        )?;
        let extent = source.layout.extent;
        let profile = DecodeProfile::VarDct { bits_per_sample: 8 };
        let submissions_per_frame = source.submissions_per_frame();
        let runtime_stats = Arc::new(VarDctRuntimeStats {
            submissions_per_frame: AtomicUsize::new(submissions_per_frame),
            hf_packet_stream_batch_count: AtomicUsize::new(0),
        });
        Ok(PreparedGpuSession::new(
            profile,
            AnimationMetadata::still(extent),
            VarDctDecodeSession {
                backend: self.backend.clone(),
                pipelines: Arc::clone(&self.pipelines),
                memory_stats: source.memory,
                runtime_stats,
                source: Some(source),
                memory: self.memory.clone(),
            },
        )
        .with_resolved_frame_slots(NonZeroUsize::new(1).expect("one is nonzero")))
    }
}

impl std::fmt::Debug for VarDctSubmissionEngine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VarDctSubmissionEngine")
            .field("backend", &self.backend)
            .field("memory", &self.memory.snapshot())
            .finish_non_exhaustive()
    }
}

struct VarDctSource {
    codestream: GpuCodestream,
    packet: BoundedVarDctPacketPlan,
    groups: Vec<VarDctGroupSource>,
    lf_packet_windows: Option<LfPacketWindowExecutionPlan>,
    combined_packet_windows: Option<CombinedPacketWindowExecutionPlan>,
    stream_limit: u64,
    resource_layout: VarDctResourceLayout,
    hf_coefficients: Option<HfCoefficientExecutionPlan>,
    gaborish: Option<ResidentGaborishWeights>,
    epf: Option<VarDctEpfPlan>,
    output_plan: VarDctOutputPlan,
    layout: ImageLayout,
    inverse_opsin: VarDctInverseOpsin,
    quant_biases: [f32; 4],
    frame_name: String,
    memory: VarDctDecodeMemoryStats,
}

struct VarDctEntropyPlanSelection {
    stream_limit: u64,
    lf_packet_windows: Option<LfPacketWindowExecutionPlan>,
    combined_packet_windows: Option<CombinedPacketWindowExecutionPlan>,
    hf_coefficients: Option<HfCoefficientExecutionPlan>,
    memory: VarDctDecodeMemoryStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdaptiveStreamMemory {
    total_frame_bytes: u64,
    packet_stream_window_bytes: u64,
    hf_stream_window_bytes: u64,
}

impl From<VarDctDecodeMemoryStats> for AdaptiveStreamMemory {
    fn from(memory: VarDctDecodeMemoryStats) -> Self {
        Self {
            total_frame_bytes: memory.total_frame_bytes,
            packet_stream_window_bytes: memory.packet_stream_window_bytes,
            hf_stream_window_bytes: memory.hf_stream_window_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdaptiveStreamLimitDecision {
    Selected(u64),
    BudgetTooSmall { required_bytes: u64 },
}

fn select_budget_adaptive_stream_limit(
    configured_limit: u64,
    memory_limit_bytes: u64,
    mut memory_at_limit: impl FnMut(u64) -> Result<AdaptiveStreamMemory, VarDctDecodeError>,
) -> Result<AdaptiveStreamLimitDecision, VarDctDecodeError> {
    let configured_limit = configured_limit & !3;
    let configured = memory_at_limit(configured_limit)?;
    if configured.total_frame_bytes <= memory_limit_bytes {
        return Ok(AdaptiveStreamLimitDecision::Selected(configured_limit));
    }

    let minimum = memory_at_limit(MIN_STREAM_WINDOW_BYTES)?;
    if minimum.total_frame_bytes > memory_limit_bytes {
        return Ok(AdaptiveStreamLimitDecision::BudgetTooSmall {
            required_bytes: minimum.total_frame_bytes,
        });
    }

    let active_stream_windows = u64::from(minimum.packet_stream_window_bytes != 0)
        + u64::from(minimum.hf_stream_window_bytes != 0);
    debug_assert!(active_stream_windows != 0);
    let non_stream_bytes = minimum
        .total_frame_bytes
        .checked_sub(minimum.packet_stream_window_bytes)
        .and_then(|bytes| bytes.checked_sub(minimum.hf_stream_window_bytes))
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "minimum-window VarDCT non-stream bytes",
        })?;
    let available_stream_bytes = memory_limit_bytes.saturating_sub(non_stream_bytes);
    let suggested_limit = available_stream_bytes
        .checked_div(active_stream_windows)
        .unwrap_or(MIN_STREAM_WINDOW_BYTES)
        .min(configured_limit)
        .max(MIN_STREAM_WINDOW_BYTES)
        & !3;

    let mut best_limit = MIN_STREAM_WINDOW_BYTES;
    let mut failing_limit = configured_limit;
    if suggested_limit > best_limit && suggested_limit < failing_limit {
        let suggested = memory_at_limit(suggested_limit)?;
        if suggested.total_frame_bytes <= memory_limit_bytes {
            best_limit = suggested_limit;
        } else {
            failing_limit = suggested_limit;
        }
    }
    for _ in 0..32 {
        let remaining_steps = failing_limit.saturating_sub(best_limit) / 4;
        if remaining_steps <= 1 {
            break;
        }
        let midpoint = best_limit.checked_add((remaining_steps / 2) * 4).ok_or(
            VarDctDecodeError::ArithmeticOverflow {
                field: "adaptive VarDCT stream-window midpoint",
            },
        )?;
        let candidate = memory_at_limit(midpoint)?;
        if candidate.total_frame_bytes <= memory_limit_bytes {
            best_limit = midpoint;
        } else {
            failing_limit = midpoint;
        }
    }
    Ok(AdaptiveStreamLimitDecision::Selected(best_limit))
}

#[derive(Clone, Debug)]
struct LfPacketWindowExecutionPlan {
    stream_segments: Arc<[GroupStreamSegment]>,
    stream_batches: Arc<[StreamBatch]>,
    segment_params: Arc<[VarDctModularParams]>,
    stream_bytes: u64,
}

#[derive(Clone, Debug)]
struct CombinedPacketWindowExecutionPlan {
    stream_segments: Arc<[GroupStreamSegment]>,
    stream_batches: Arc<[StreamBatch]>,
    segment_params: Arc<[VarDctModularParams]>,
    stream_bytes: u64,
}

#[derive(Clone, Debug)]
struct HfPacketWindowExecutionPlan {
    stream_segments: Arc<[GroupStreamSegment]>,
    stream_batches: Arc<[StreamBatch]>,
    segment_params: Arc<[VarDctModularParams]>,
    stream_bytes: u64,
}

fn map_packet_window_plan_error(error: DecodeError) -> VarDctDecodeError {
    match error {
        DecodeError::StreamWindowTooSmall {
            limit_bytes,
            minimum_bytes,
        } => VarDctDecodeError::EntropyStreamWindowTooSmall {
            limit_bytes,
            minimum_bytes,
        },
        source => VarDctDecodeError::EntropyWindowPlan {
            source: Box::new(source),
        },
    }
}

fn map_codestream_source_error(source: DecodeError) -> VarDctDecodeError {
    VarDctDecodeError::CodestreamSource {
        source: Box::new(source),
    }
}

fn copy_stream_segment(
    codestream: &GpuCodestream,
    segment: GroupStreamSegment,
    upload: &mut [u8],
    detail: &'static str,
) -> Result<(), VarDctDecodeError> {
    let input_len = segment
        .input_end
        .checked_sub(segment.input_start)
        .ok_or(VarDctDecodeError::EntropyWindowContract { detail })?;
    let output_end = segment.upload_offset.checked_add(input_len).ok_or(
        VarDctDecodeError::ArithmeticOverflow {
            field: "bounded stream upload end",
        },
    )?;
    let output = upload
        .get_mut(segment.upload_offset..output_end)
        .ok_or(VarDctDecodeError::EntropyWindowContract { detail })?;
    let input_start =
        u64::try_from(segment.input_start).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
            field: "bounded stream input start",
        })?;
    let input_end =
        u64::try_from(segment.input_end).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
            field: "bounded stream input end",
        })?;
    if input_end > codestream.logical_bytes() {
        return Err(VarDctDecodeError::EntropyWindowContract { detail });
    }
    codestream
        .copy_range(input_start..input_end, output)
        .map_err(map_codestream_source_error)
}

impl LfPacketWindowExecutionPlan {
    fn new(
        codestream_bytes: u64,
        packet: &BoundedVarDctPacketPlan,
        stream_limit: u64,
    ) -> Result<Option<Self>, VarDctDecodeError> {
        let ranges = packet
            .groups
            .iter()
            .map(|group| {
                let token_bit_end =
                    group
                        .lf_group
                        .end()
                        .ok_or(VarDctDecodeError::ArithmeticOverflow {
                            field: "LF packet stream end",
                        })?;
                Ok(GroupEntropyRange {
                    token_bit_offset: u64::from(group.lf_entropy_bit_offset),
                    token_bit_end,
                })
            })
            .collect::<Result<Vec<_>, VarDctDecodeError>>()?;
        let (segments, batches, _) =
            build_stream_batches_for_len(codestream_bytes, &ranges, stream_limit, 1)
                .map_err(map_packet_window_plan_error)?;
        let uses_windows = segments.iter().any(|segment| {
            segment.flags != (GroupStreamSegment::FIRST | GroupStreamSegment::FINAL)
        });
        if !uses_windows {
            return Ok(None);
        }
        let mut segment_params = Vec::with_capacity(segments.len());
        for &segment in &segments {
            let group = packet.groups.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "LF packet segment references an absent group",
                },
            )?;
            // Local-tree staging reserves the conservative predictor/LZ layout because the HF
            // descriptor is discovered only after this stage. The LF state itself records only
            // the predictor fields selected by its parsed tree.
            let state_offset = group.packet_execution_state_offset_words(true)?;
            segment_params.push(
                VarDctModularParams::default()
                    .with_lz77_window(group.lf_modular.lz77_window_words)
                    .with_self_correcting(group.lf_modular.needs_self_correcting)
                    .with_stream_segment(segment, group.lf_entropy_bit_offset, state_offset),
            );
        }
        Ok(Some(Self {
            stream_segments: segments.into(),
            stream_batches: batches.into(),
            segment_params: segment_params.into(),
            stream_bytes: stream_limit & !3,
        }))
    }

    fn batch_count(&self) -> usize {
        self.stream_batches.len()
    }
}

impl CombinedPacketWindowExecutionPlan {
    fn new(
        codestream_bytes: u64,
        packet: &BoundedVarDctPacketPlan,
        stream_limit: u64,
    ) -> Result<Option<Self>, VarDctDecodeError> {
        let mut stream_bases = Vec::with_capacity(packet.groups.len());
        let ranges = packet
            .groups
            .iter()
            .map(|group| {
                let control = group.packet_control(packet)?;
                let token_bit_offset = u64::from(control.section_bits[0]);
                let token_bit_end = u64::from(control.section_bits[1]);
                stream_bases.push(control.section_bits[0]);
                Ok(GroupEntropyRange {
                    token_bit_offset,
                    token_bit_end,
                })
            })
            .collect::<Result<Vec<_>, VarDctDecodeError>>()?;
        let (segments, batches, stream_bytes) = build_stream_batches_for_len(
            codestream_bytes,
            &ranges,
            stream_limit,
            packet.groups.len().max(1),
        )
        .map_err(map_packet_window_plan_error)?;
        let uses_windows = segments.iter().any(|segment| {
            segment.flags != (GroupStreamSegment::FIRST | GroupStreamSegment::FINAL)
        });
        if !uses_windows {
            return Ok(None);
        }
        let mut segment_params = Vec::with_capacity(segments.len());
        for &segment in &segments {
            let group = packet.groups.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "combined packet segment references an absent group",
                },
            )?;
            let stream_base_bit = *stream_bases.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "combined packet segment has no stream base",
                },
            )?;
            let state_offset =
                group.packet_execution_state_offset_words(packet.needs_self_correcting)?;
            segment_params.push(
                VarDctModularParams::default()
                    .with_lz77_window(group.lz77_window_words)
                    .with_self_correcting(packet.needs_self_correcting)
                    .with_stream_segment(segment, stream_base_bit, state_offset),
            );
        }
        Ok(Some(Self {
            stream_segments: segments.into(),
            stream_batches: batches.into(),
            segment_params: segment_params.into(),
            stream_bytes,
        }))
    }

    fn batch_count(&self) -> usize {
        self.stream_batches.len()
    }
}

impl HfPacketWindowExecutionPlan {
    fn new(
        codestream_bytes: u64,
        packet: &BoundedVarDctPacketPlan,
        continuations: &[BoundedHfMetadataContinuation],
        stream_limit: u64,
    ) -> Result<Option<Self>, VarDctDecodeError> {
        if packet.groups.len() != continuations.len() {
            return Err(VarDctDecodeError::GroupPlanCount {
                component: "HF packet continuation",
                expected: packet.groups.len(),
                actual: continuations.len(),
            });
        }
        let ranges = packet
            .groups
            .iter()
            .zip(continuations)
            .map(|(group, continuation)| {
                let token_bit_end =
                    group
                        .lf_group
                        .end()
                        .ok_or(VarDctDecodeError::ArithmeticOverflow {
                            field: "HF packet stream end",
                        })?;
                Ok(GroupEntropyRange {
                    token_bit_offset: u64::from(continuation.token_bit_offset),
                    token_bit_end,
                })
            })
            .collect::<Result<Vec<_>, VarDctDecodeError>>()?;
        let (segments, batches, stream_bytes) = build_stream_batches_for_len(
            codestream_bytes,
            &ranges,
            stream_limit,
            packet.groups.len().max(1),
        )
        .map_err(map_packet_window_plan_error)?;
        let uses_windows = segments.iter().any(|segment| {
            segment.flags != (GroupStreamSegment::FIRST | GroupStreamSegment::FINAL)
        });
        if !uses_windows {
            return Ok(None);
        }
        let mut segment_params = Vec::with_capacity(segments.len());
        for &segment in &segments {
            let group = packet.groups.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "HF packet segment references an absent group",
                },
            )?;
            let continuation = continuations.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "HF packet segment references an absent continuation",
                },
            )?;
            let state_offset = group.packet_execution_state_offset_words(true)?;
            segment_params.push(
                VarDctModularParams::default()
                    .with_lz77_window(continuation.modular.lz77_window_words)
                    .with_self_correcting(continuation.modular.needs_self_correcting)
                    .with_stream_segment(segment, continuation.token_bit_offset, state_offset),
            );
        }
        Ok(Some(Self {
            stream_segments: segments.into(),
            stream_batches: batches.into(),
            segment_params: segment_params.into(),
            stream_bytes,
        }))
    }
}

impl VarDctSource {
    fn submissions_per_frame(&self) -> usize {
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

struct VarDctGroupSource {
    control: VarDctPacketControl,
    resource_params: VarDctResourceParams,
    artifact_layout: VarDctArtifactLayout,
    artifact_params: HfMetadataLoweringParams,
    quant_offset: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct VarDctEpfHeader {
    iterations: u32,
    sharp_lut: [f32; 8],
    channel_scale: [f32; 3],
    quant_mul: f32,
    pass0_sigma_scale: f32,
    pass2_sigma_scale: f32,
    border_sad_mul: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct VarDctEpfPlan {
    sigma_groups: Vec<EpfSigmaConfig>,
    passes: Vec<ResidentEpfParameters>,
}

impl VarDctEpfHeader {
    fn plan(
        self,
        packet: &BoundedVarDctPacketPlan,
        groups: &[VarDctGroupSource],
        global_scale: u32,
    ) -> Result<VarDctEpfPlan, VarDctDecodeError> {
        let mut passes = Vec::with_capacity(self.iterations as usize);
        if self.iterations >= 3 {
            passes.push(ResidentEpfParameters {
                pass: EpfPass::Pass0,
                sigma_scale: self.pass0_sigma_scale,
                border_sad_mul: self.border_sad_mul,
                channel_scale: self.channel_scale,
            });
        }
        if self.iterations >= 1 {
            passes.push(ResidentEpfParameters {
                pass: EpfPass::Pass1,
                sigma_scale: 1.0,
                border_sad_mul: self.border_sad_mul,
                channel_scale: self.channel_scale,
            });
        }
        if self.iterations >= 2 {
            passes.push(ResidentEpfParameters {
                pass: EpfPass::Pass2,
                sigma_scale: self.pass2_sigma_scale,
                border_sad_mul: self.border_sad_mul,
                channel_scale: self.channel_scale,
            });
        }
        debug_assert_eq!(passes.len(), self.iterations as usize);
        let [output_blocks_x, output_blocks_y] = packet.block_extent();
        let sigma_groups = packet
            .groups
            .iter()
            .zip(groups)
            .map(|(packet_group, group)| {
                let [blocks_x, blocks_y] = packet_group.block_extent();
                Ok(EpfSigmaConfig {
                    blocks_x,
                    blocks_y,
                    output_blocks_x,
                    output_blocks_y,
                    output_origin: [packet_group.rect.x / 8, packet_group.rect.y / 8],
                    task_count: packet_group.task_capacity,
                    sharpness_offset_words: group.control.expected[3],
                    artifact_status_offset_words: group.artifact_layout.status_offset_words,
                    task_metadata_offset_words: group.artifact_layout.task_metadata_offset_words,
                    global_scale,
                    quant_mul: self.quant_mul,
                    sharp_lut: self.sharp_lut,
                })
            })
            .collect::<Result<Vec<_>, VarDctDecodeError>>()?;
        Ok(VarDctEpfPlan {
            sigma_groups,
            passes,
        })
    }
}

impl GpuSubmissionEngine for VarDctSubmissionEngine {
    type Session = VarDctDecodeSession;

    fn parse_limits(&self) -> ParseLimits {
        ParseLimits {
            max_input_bytes: VAR_DCT_PARSE_LIMIT_BYTES,
            max_boxes: 32,
            max_box_bytes: VAR_DCT_PARSE_LIMIT_BYTES,
            max_codestream_bytes: VAR_DCT_PARSE_LIMIT_BYTES,
        }
    }

    fn inventory_limits(&self) -> InventoryLimits {
        InventoryLimits {
            max_frames: 1,
            max_total_section_bytes: self.parse_limits().max_codestream_bytes,
            ..InventoryLimits::default()
        }
    }

    fn open(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
        inventory: Arc<CodestreamInventory>,
    ) -> DecodeResult<PreparedGpuSession<Self::Session>> {
        self.open_with_inventory(codestream, request, &inventory)
    }
}

fn prepare_source(
    backend: &WgpuBackend,
    codestream: GpuCodestream,
    request: &GpuOutputRequest,
    inventory: &jxl_gpu_bitstream::CodestreamInventory,
    output_variant: KernelVariant,
    stream_window_limit: Option<NonZeroU64>,
    memory_limit_bytes: u64,
) -> Result<VarDctSource, VarDctDecodeError> {
    if request.mapping() != GpuOutputMapping::Color || request.format() != &vardct_rgb8_format() {
        return Err(VarDctDecodeError::UnsupportedOutput);
    }
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
    let packet = BoundedVarDctPacketPlan::parse_source(&codestream, inventory)?;
    let codestream_bytes = codestream.logical_bytes();
    let codestream_len =
        usize::try_from(codestream_bytes).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
            field: "codestream source length",
        })?;
    let staged_local_trees = packet.requires_local_tree_staging();
    let limits = backend.device().limits();
    let configured_stream_limit = stream_window_limit
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
    let resource_layout =
        VarDctResourceLayout::new(blocks_x, blocks_y, packet.total_task_capacity()?)?;
    let correlation_width = packet.profile.width.div_ceil(64);
    let pass_group_dim_blocks = packet.profile.group_dimension.checked_div(8).ok_or(
        VarDctDecodeError::ArithmeticOverflow {
            field: "pass-group block dimension",
        },
    )?;
    let mut quant_offset = resource_layout.quant_offset;
    let mut groups = Vec::with_capacity(packet.groups.len());
    for packet_group in &packet.groups {
        let control = if staged_local_trees {
            packet_group.lf_stage_control(&packet)?
        } else {
            packet_group.packet_control(&packet)?
        };
        let [group_blocks_x, group_blocks_y] = packet_group.block_extent();
        let block_origin = [packet_group.rect.x / 8, packet_group.rect.y / 8];
        let resource_params = VarDctResourceParams::new(VarDctResourceConfig {
            block_extent: [group_blocks_x, group_blocks_y],
            output_stride: blocks_x,
            output_origin: block_origin,
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
    let output_plan = VarDctOutputPlan::for_limits_with_variant(
        packet.profile.width,
        packet.profile.height,
        &backend.device().limits(),
        output_variant,
    )?;
    let layout = ImageLayout::packed(
        Extent2d::new(packet.profile.width, packet.profile.height),
        vardct_rgb8_format(),
    )?;
    let opsin = inventory
        .image_header
        .opsin_inverse_matrix
        .ok_or(VarDctDecodeError::MissingInverseOpsin)?;
    let inverse_opsin = VarDctInverseOpsin {
        opsin_bias: opsin.opsin_bias.map(|value| value.to_f32()),
        inverse_opsin_matrix: opsin
            .inverse_matrix
            .map(|row| row.map(|value| value.to_f32())),
        intensity_target: inventory
            .image_header
            .tone_mapping
            .intensity_target
            .to_f32(),
    };
    let quant_biases = [
        opsin.quant_bias[0].to_f32(),
        opsin.quant_bias[1].to_f32(),
        opsin.quant_bias[2].to_f32(),
        opsin.quant_bias_numerator.to_f32(),
    ];
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
            let combined_packet_windows = (!staged_local_trees)
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
            let memory = VarDctDecodeMemoryStats::plan(VarDctDecodeMemoryInputs {
                stream_limit,
                codestream_len,
                packet: &packet,
                groups: &groups,
                lf_packet_windows: lf_packet_windows.as_ref(),
                combined_packet_windows: combined_packet_windows.as_ref(),
                resource: resource_layout,
                hf_coefficients: hf_coefficients.as_ref(),
                adaptive_lf_smoothing: packet.profile.adaptive_lf_smoothing,
                restoration_scratch: gaborish.is_some() || epf.is_some(),
                gaborish: gaborish.is_some(),
                epf_sigma: epf_sigma_memory,
                epf_iterations: epf.as_ref().map_or(0, |plan| plan.passes.len() as u32),
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
        memory_limit_bytes,
        |stream_limit| Ok(plan_at_limit(stream_limit)?.memory.into()),
    )? {
        AdaptiveStreamLimitDecision::Selected(stream_limit) => stream_limit,
        AdaptiveStreamLimitDecision::BudgetTooSmall { required_bytes } => {
            return Err(VarDctDecodeError::MemoryBudgetTooSmall {
                required_bytes,
                limit_bytes: memory_limit_bytes,
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
    validate_device_limits(
        backend.device(),
        memory,
        &packet,
        &groups,
        hf_coefficients.as_ref(),
    )?;
    let frame_name = packet.profile.frame_name.clone();
    Ok(VarDctSource {
        codestream,
        packet,
        groups,
        lf_packet_windows,
        combined_packet_windows,
        stream_limit,
        resource_layout,
        hf_coefficients,
        gaborish,
        epf,
        output_plan,
        layout,
        inverse_opsin,
        quant_biases,
        frame_name,
        memory,
    })
}

fn dequant_matrix_multiplier(channel: &'static str, scale: u32) -> Result<f32, VarDctDecodeError> {
    // JPEG XL 3-bit X/B quant-matrix scale: (1 / 1.25)^(scale - 2).
    const MULTIPLIERS: [f32; 8] = [1.5625, 1.25, 1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768];
    MULTIPLIERS
        .get(scale as usize)
        .copied()
        .ok_or(VarDctDecodeError::InvalidQuantMatrixScale { channel, scale })
}

fn restoration_config(
    restoration: RestorationFilterInventory,
) -> Result<(Option<ResidentGaborishWeights>, Option<VarDctEpfHeader>), VarDctDecodeError> {
    let (gaborish, epf) = match restoration {
        RestorationFilterInventory::Default => (
            GaborishInventory::Default,
            EdgePreservingFilterInventory::default(),
        ),
        RestorationFilterInventory::Custom { gaborish, epf } => (gaborish, epf),
    };
    let gaborish = match gaborish {
        GaborishInventory::Disabled => None,
        GaborishInventory::Default => Some(ResidentGaborishWeights::DEFAULT),
        GaborishInventory::Custom { weights } => Some(ResidentGaborishWeights {
            x: weights[0].map(|value| value.to_f32()),
            y: weights[1].map(|value| value.to_f32()),
            b: weights[2].map(|value| value.to_f32()),
        }),
    };
    let epf = match epf {
        EdgePreservingFilterInventory::Disabled => None,
        EdgePreservingFilterInventory::Enabled {
            iterations,
            sharp_lut,
            weights,
            sigma,
            sigma_for_modular: _,
        } => {
            if !(1..=3).contains(&iterations) {
                return Err(VarDctDecodeError::InvalidEpfIterations { iterations });
            }
            let sharp_lut = sharp_lut.map_or(
                [
                    0.0,
                    1.0 / 7.0,
                    2.0 / 7.0,
                    3.0 / 7.0,
                    4.0 / 7.0,
                    5.0 / 7.0,
                    6.0 / 7.0,
                    1.0,
                ],
                |values| values.map(|value| value.to_f32()),
            );
            let channel_scale = weights.map_or([40.0, 5.0, 3.5], |weights| {
                weights.channel_scale.map(|value| value.to_f32())
            });
            let (quant_mul, pass0_sigma_scale, pass2_sigma_scale, border_sad_mul) =
                sigma.map_or((0.46, 0.9, 6.5, 2.0 / 3.0), |sigma| {
                    (
                        sigma.quant_mul.map_or(0.46, |value| value.to_f32()),
                        sigma.pass0_sigma_scale.to_f32(),
                        sigma.pass2_sigma_scale.to_f32(),
                        sigma.border_sad_mul.to_f32(),
                    )
                });
            Some(VarDctEpfHeader {
                iterations,
                sharp_lut,
                channel_scale,
                quant_mul,
                pass0_sigma_scale,
                pass2_sigma_scale,
                border_sad_mul,
            })
        }
    };
    Ok((gaborish, epf))
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
    let modular_metadata_binding_bytes = if packet.requires_local_tree_staging() {
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
        ("one XYB plane", memory.xyb_plane_bytes / 3, true),
        (
            "one restoration scratch plane",
            memory.restoration_scratch_bytes / 3,
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
        ("EPF sigma uniform", epf_sigma_uniform_bytes),
        (
            "one EPF filter uniform",
            if memory.epf_filter_uniform_bytes == 0 {
                0
            } else {
                ResidentEpfMemoryPlan::UNIFORM_BYTES
            },
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

fn check_limit(
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

/// One-frame submission state for [`VarDctSubmissionEngine`].
pub struct VarDctDecodeSession {
    backend: WgpuBackend,
    pipelines: Arc<VarDctPipelines>,
    memory_stats: VarDctDecodeMemoryStats,
    runtime_stats: Arc<VarDctRuntimeStats>,
    source: Option<VarDctSource>,
    memory: MemoryBudget,
}

#[derive(Debug)]
struct VarDctRuntimeStats {
    submissions_per_frame: AtomicUsize,
    hf_packet_stream_batch_count: AtomicUsize,
}

impl VarDctDecodeSession {
    #[must_use]
    pub const fn memory_stats(&self) -> VarDctDecodeMemoryStats {
        self.memory_stats
    }

    #[must_use]
    pub fn in_flight_memory_stats(&self) -> MemoryBudgetSnapshot {
        self.memory.snapshot()
    }

    #[must_use]
    pub fn submissions_per_frame(&self) -> usize {
        self.runtime_stats
            .submissions_per_frame
            .load(Ordering::Acquire)
    }

    /// Exact staged HF packet batch count once the LF cursor map has completed. It is zero before
    /// that dynamic plan exists and when every HF packet binds the retained whole codestream.
    #[must_use]
    pub fn hf_packet_stream_batch_count(&self) -> usize {
        self.runtime_stats
            .hf_packet_stream_batch_count
            .load(Ordering::Acquire)
    }
}

impl std::fmt::Debug for VarDctDecodeSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VarDctDecodeSession")
            .field("submitted", &self.source.is_none())
            .field("memory_stats", &self.memory_stats())
            .finish_non_exhaustive()
    }
}

impl GpuSubmissionSession for VarDctDecodeSession {
    type Frame = GpuImageFrame;
    type Pending = VarDctPendingFrame;

    fn submit_next(&mut self) -> DecodeResult<Option<Self::Pending>> {
        let Some(source) = self.source.as_ref() else {
            return Ok(None);
        };
        let poll_permit = self
            .backend
            .submission_poller()
            .try_reserve()
            .map_err(DecodeError::PollBackpressure)?;
        let output_permit = self.memory.try_reserve(source.memory.output_lease_bytes)?;
        let transient_permit = self.memory.try_reserve(source.memory.transient_bytes)?;
        let source = self
            .source
            .take()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let pending = submit_vardct(
            &self.backend,
            Arc::clone(&self.pipelines),
            self.memory.clone(),
            Arc::clone(&self.runtime_stats),
            source,
            VarDctMemoryPermits {
                output: output_permit,
                transient: transient_permit,
            },
            poll_permit,
        )?;
        Ok(Some(pending))
    }
}

struct VarDctMemoryPermits {
    output: MemoryPermit,
    transient: MemoryPermit,
}

struct HfCoefficientJobBuffers {
    entropy_bundle: wgpu::Buffer,
    order_table: wgpu::Buffer,
    stream_window: Option<wgpu::Buffer>,
    params_window: Option<wgpu::Buffer>,
    groups: Vec<HfCoefficientGroupJobBuffers>,
}

struct HfCoefficientGroupJobBuffers {
    params: Option<wgpu::Buffer>,
    status: wgpu::Buffer,
    sink_params: wgpu::Buffer,
}

struct HfCoefficientBatchSubmission {
    stream_upload: Box<[u8]>,
    params_upload: Box<[u8]>,
    commands: wgpu::CommandBuffer,
}

struct LfPacketBatchSubmission {
    group_index: usize,
    stream_upload: Box<[u8]>,
    params: VarDctModularParams,
    commands: wgpu::CommandBuffer,
}

struct CombinedPacketGroupUpload {
    group_index: usize,
    params: VarDctModularParams,
}

struct CombinedPacketBatchSubmission {
    stream_upload: Box<[u8]>,
    groups: Box<[CombinedPacketGroupUpload]>,
    commands: wgpu::CommandBuffer,
}

struct HfPacketGroupUpload {
    group_index: usize,
    control: VarDctPacketControl,
    params: VarDctModularParams,
}

struct HfPacketBatchSubmission {
    stream_upload: Box<[u8]>,
    groups: Box<[HfPacketGroupUpload]>,
    commands: wgpu::CommandBuffer,
}

enum LfPacketCommands {
    Whole(wgpu::CommandBuffer),
    Windowed(Vec<LfPacketBatchSubmission>),
}

enum VarDctDownstreamCommands {
    Whole(wgpu::CommandBuffer),
    Windowed {
        before_coefficients: wgpu::CommandBuffer,
        coefficient_batches: Vec<HfCoefficientBatchSubmission>,
        after_coefficients: wgpu::CommandBuffer,
    },
}

fn submit_lf_packet_commands(
    queue: &wgpu::Queue,
    commands: LfPacketCommands,
    lifetime: &VarDctJobLifetime,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    match commands {
        LfPacketCommands::Whole(commands) => Ok(queue.submit([commands])),
        LfPacketCommands::Windowed(batches) => {
            let stream = lifetime._packet_stream_window.as_ref().ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "windowed LF packet commands have no retained stream upload",
                },
            )?;
            let mut last_submission = None;
            for batch in batches {
                let group = lifetime._groups.get(batch.group_index).ok_or(
                    VarDctDecodeError::EntropyWindowContract {
                        detail: "windowed LF packet batch references an absent group",
                    },
                )?;
                queue.write_buffer(stream, 0, &batch.stream_upload);
                queue.write_buffer(&group.modular_params, 0, bytemuck::bytes_of(&batch.params));
                last_submission = Some(queue.submit([batch.commands]));
            }
            last_submission.ok_or(VarDctDecodeError::EntropyWindowContract {
                detail: "windowed LF packet execution has no batches",
            })
        }
    }
}

fn write_combined_packet_batch(
    queue: &wgpu::Queue,
    stream: &wgpu::Buffer,
    batch: &CombinedPacketBatchSubmission,
    lifetime: &VarDctJobLifetime,
) -> Result<(), VarDctDecodeError> {
    queue.write_buffer(stream, 0, &batch.stream_upload);
    for upload in &batch.groups {
        let group = lifetime._groups.get(upload.group_index).ok_or(
            VarDctDecodeError::EntropyWindowContract {
                detail: "windowed combined packet batch references an absent group",
            },
        )?;
        queue.write_buffer(&group.modular_params, 0, bytemuck::bytes_of(&upload.params));
    }
    Ok(())
}

fn submit_combined_packet_commands(
    queue: &wgpu::Queue,
    mut batches: Vec<CombinedPacketBatchSubmission>,
    downstream: VarDctDownstreamCommands,
    lifetime: &VarDctJobLifetime,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    let stream = lifetime._packet_stream_window.as_ref().ok_or(
        VarDctDecodeError::EntropyWindowContract {
            detail: "windowed combined packet commands have no retained stream upload",
        },
    )?;
    let final_batch = batches
        .pop()
        .ok_or(VarDctDecodeError::EntropyWindowContract {
            detail: "windowed combined packet execution has no batches",
        })?;
    for batch in batches {
        write_combined_packet_batch(queue, stream, &batch, lifetime)?;
        queue.submit([batch.commands]);
    }
    write_combined_packet_batch(queue, stream, &final_batch, lifetime)?;
    submit_vardct_downstream(queue, vec![final_batch.commands], downstream, lifetime)
}

fn write_hf_packet_batch(
    queue: &wgpu::Queue,
    stream: &wgpu::Buffer,
    batch: &HfPacketBatchSubmission,
    lifetime: &VarDctJobLifetime,
) -> Result<(), VarDctDecodeError> {
    queue.write_buffer(stream, 0, &batch.stream_upload);
    for upload in &batch.groups {
        let group = lifetime._groups.get(upload.group_index).ok_or(
            VarDctDecodeError::EntropyWindowContract {
                detail: "windowed HF packet batch references an absent group",
            },
        )?;
        queue.write_buffer(
            &group.packet_control,
            0,
            bytemuck::bytes_of(&upload.control),
        );
        queue.write_buffer(&group.modular_params, 0, bytemuck::bytes_of(&upload.params));
    }
    Ok(())
}

fn submit_hf_packet_commands(
    queue: &wgpu::Queue,
    mut batches: Vec<HfPacketBatchSubmission>,
    downstream: VarDctDownstreamCommands,
    lifetime: &VarDctJobLifetime,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    let stream = lifetime._packet_stream_window.as_ref().ok_or(
        VarDctDecodeError::EntropyWindowContract {
            detail: "windowed HF packet commands have no retained stream upload",
        },
    )?;
    let final_batch = batches
        .pop()
        .ok_or(VarDctDecodeError::EntropyWindowContract {
            detail: "windowed HF packet execution has no batches",
        })?;
    for batch in batches {
        write_hf_packet_batch(queue, stream, &batch, lifetime)?;
        queue.submit([batch.commands]);
    }
    write_hf_packet_batch(queue, stream, &final_batch, lifetime)?;
    submit_vardct_downstream(queue, vec![final_batch.commands], downstream, lifetime)
}

fn submit_vardct_downstream(
    queue: &wgpu::Queue,
    mut prefix: Vec<wgpu::CommandBuffer>,
    downstream: VarDctDownstreamCommands,
    lifetime: &VarDctJobLifetime,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    match downstream {
        VarDctDownstreamCommands::Whole(commands) => {
            prefix.push(commands);
            Ok(queue.submit(prefix))
        }
        VarDctDownstreamCommands::Windowed {
            before_coefficients,
            coefficient_batches,
            after_coefficients,
        } => {
            prefix.push(before_coefficients);
            queue.submit(prefix);
            let buffers = lifetime._hf_coefficients.as_ref().ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "windowed AC commands have no retained coefficient buffers",
                },
            )?;
            let stream =
                buffers
                    .stream_window
                    .as_ref()
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "windowed AC commands have no stream upload",
                    })?;
            let params =
                buffers
                    .params_window
                    .as_ref()
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "windowed AC commands have no parameter upload",
                    })?;
            for batch in coefficient_batches {
                queue.write_buffer(stream, 0, &batch.stream_upload);
                queue.write_buffer(params, 0, &batch.params_upload);
                queue.submit([batch.commands]);
            }
            Ok(queue.submit([after_coefficients]))
        }
    }
}

struct VarDctGroupJobBuffers {
    reconstructed: wgpu::Buffer,
    raw_metadata: wgpu::Buffer,
    coefficients: wgpu::Buffer,
    packet_status: wgpu::Buffer,
    packet_control: wgpu::Buffer,
    modular_params: wgpu::Buffer,
    artifact: wgpu::Buffer,
    occupancy: wgpu::Buffer,
    artifact_uniform: wgpu::Buffer,
}

struct RestorationCursor<'a> {
    image: &'a [wgpu::Buffer; 3],
    scratch: &'a [wgpu::Buffer; 3],
    current_is_scratch: bool,
}

impl<'a> RestorationCursor<'a> {
    fn new(image: &'a [wgpu::Buffer; 3], scratch: &'a [wgpu::Buffer; 3]) -> Self {
        Self {
            image,
            scratch,
            current_is_scratch: false,
        }
    }

    fn advance(&mut self) -> (&'a [wgpu::Buffer; 3], &'a [wgpu::Buffer; 3]) {
        let pair = if self.current_is_scratch {
            (self.scratch, self.image)
        } else {
            (self.image, self.scratch)
        };
        self.current_is_scratch = !self.current_is_scratch;
        pair
    }

    fn current(&self) -> &'a [wgpu::Buffer; 3] {
        if self.current_is_scratch {
            self.scratch
        } else {
            self.image
        }
    }
}

struct RestorationJobBuffers {
    _planes: [wgpu::Buffer; 3],
    _gaborish_uniform: Option<wgpu::Buffer>,
    _epf_sigma: Option<wgpu::Buffer>,
    _epf_sigma_uniforms: Vec<wgpu::Buffer>,
    _epf_uniforms: Vec<wgpu::Buffer>,
}

struct VarDctJobLifetime {
    output: GpuBufferLease,
    status_staging: wgpu::Buffer,
    status_mapped: AtomicBool,
    _transient_permits: Mutex<Vec<MemoryPermit>>,
    _codestream: wgpu::Buffer,
    _packet_stream_window: Option<wgpu::Buffer>,
    _modular_metadata: Mutex<Vec<wgpu::Buffer>>,
    _groups: Vec<VarDctGroupJobBuffers>,
    _lf_temporary: wgpu::Buffer,
    _resources: wgpu::Buffer,
    _resource_uniforms: Vec<wgpu::Buffer>,
    _adaptive_lf_uniform: Option<wgpu::Buffer>,
    _hf_coefficients: Option<HfCoefficientJobBuffers>,
    _xyb_planes: [wgpu::Buffer; 3],
    _restoration: Option<RestorationJobBuffers>,
    _resident_scratch: Vec<ResidentVarDctScratch>,
    _output_scratch: VarDctOutputScratch,
}

impl Drop for VarDctJobLifetime {
    fn drop(&mut self) {
        if self.status_mapped.swap(false, Ordering::AcqRel) {
            self.status_staging.unmap();
        }
    }
}

#[derive(Clone, Debug)]
struct VarDctGroupValidation {
    uniform_transform: Option<TransformKind>,
    expected_lf_samples: u32,
    expected_coefficients: u32,
    expected_blocks: u32,
    correlation_samples: u32,
    task_capacity: u32,
    expected_global_scale: u32,
    expected_quant_lf: u32,
    expected_extra_precision: u8,
}

/// Submitted VarDCT frame awaiting one aggregate map of every LF/pass-group status record.
pub struct VarDctPendingFrame {
    backend: WgpuBackend,
    pipelines: Arc<VarDctPipelines>,
    memory: MemoryBudget,
    runtime_stats: Arc<VarDctRuntimeStats>,
    lifetime: Option<Arc<VarDctJobLifetime>>,
    stage: VarDctPendingStage,
    token: SubmissionToken,
    layout: ImageLayout,
    frame_name: String,
    expected_groups: Vec<VarDctGroupValidation>,
    expected_hf_group_indices: Vec<u32>,
}

enum VarDctPendingStage {
    LocalLf {
        completion: Arc<MapCompletion>,
        source: Box<VarDctSource>,
        downstream: Option<VarDctDownstreamCommands>,
    },
    Final {
        completion: Arc<MapCompletion>,
    },
}

impl std::fmt::Debug for VarDctPendingFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VarDctPendingFrame")
            .field("token", &self.token)
            .field("layout", &self.layout)
            .field("lf_group_count", &self.expected_groups.len())
            .field(
                "stage",
                &match &self.stage {
                    VarDctPendingStage::LocalLf { .. } => "local-lf",
                    VarDctPendingStage::Final { .. } => "final",
                },
            )
            .finish_non_exhaustive()
    }
}

impl VarDctPendingFrame {
    /// Same-queue, budget-tracked access before packet/artifact status becomes authoritative.
    pub fn unvalidated_gpu_frame(&self) -> DecodeResult<UnvalidatedGpuImageFrame> {
        if matches!(self.stage, VarDctPendingStage::LocalLf { .. }) {
            return Err(VarDctDecodeError::UnvalidatedOutputNotSubmitted.into());
        }
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        Ok(UnvalidatedGpuImageFrame {
            token: self.token,
            outputs: vec![UnvalidatedGpuImageOutput {
                id: OutputId(0),
                layout: self.layout.clone(),
                buffer: lifetime.output.clone(),
            }],
        })
    }

    fn stage_completion(&self) -> Arc<MapCompletion> {
        match &self.stage {
            VarDctPendingStage::LocalLf { completion, .. }
            | VarDctPendingStage::Final { completion } => Arc::clone(completion),
        }
    }

    fn take_local_stage(&mut self) -> Option<(Box<VarDctSource>, VarDctDownstreamCommands)> {
        let placeholder = VarDctPendingStage::Final {
            completion: Arc::new(MapCompletion::default()),
        };
        let stage = std::mem::replace(&mut self.stage, placeholder);
        match stage {
            VarDctPendingStage::LocalLf {
                source,
                mut downstream,
                ..
            } => downstream.take().map(|commands| (source, commands)),
            final_stage @ VarDctPendingStage::Final { .. } => {
                self.stage = final_stage;
                None
            }
        }
    }

    fn submit_hf_stage(
        &mut self,
        mapping: Result<(), String>,
        source: Box<VarDctSource>,
        downstream: VarDctDownstreamCommands,
    ) -> DecodeResult<()> {
        mapping.map_err(DecodeError::backend)?;
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let mapped = lifetime
            .status_staging
            .slice(..)
            .get_mapped_range()
            .map_err(DecodeError::backend)?;
        let mut cursors = Vec::with_capacity(self.expected_groups.len());
        for (index, expected) in self.expected_groups.iter().enumerate() {
            let offset = index.checked_mul(PACKET_STATUS_BYTES as usize).ok_or(
                VarDctDecodeError::StatusAbi {
                    status: "LF cursor offset",
                },
            )?;
            let status: GpuVarDctPacketStatus = mapped
                .get(offset..offset + PACKET_STATUS_BYTES as usize)
                .and_then(|bytes| bytemuck::try_pod_read_unaligned(bytes).ok())
                .ok_or(VarDctDecodeError::StatusAbi {
                    status: "LF cursor",
                })?;
            cursors.push(
                status
                    .validate_lf_stage(
                        expected.expected_lf_samples,
                        expected.expected_global_scale,
                        expected.expected_quant_lf,
                        expected.expected_extra_precision,
                    )
                    .map_err(VarDctDecodeError::from)?,
            );
        }
        drop(mapped);
        lifetime.status_staging.unmap();
        lifetime.status_mapped.store(false, Ordering::Release);

        let continuations = source
            .packet
            .groups
            .iter()
            .zip(cursors)
            .map(|(group, cursor)| {
                source
                    .packet
                    .parse_hf_continuation_source(&source.codestream, group, cursor)
                    .map_err(VarDctDecodeError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let hf_packet_windows = HfPacketWindowExecutionPlan::new(
            source.codestream.logical_bytes(),
            &source.packet,
            &continuations,
            source.stream_limit,
        )?;
        if let Some(plan) = &hf_packet_windows {
            let available = lifetime
                ._packet_stream_window
                .as_ref()
                .map_or(0, wgpu::Buffer::size);
            if plan.stream_bytes > available {
                return Err(VarDctDecodeError::DeviceLimit {
                    resource: "shared local-tree packet stream window",
                    required: plan.stream_bytes,
                    available,
                }
                .into());
            }
        }
        let hf_metadata_bytes = continuations
            .iter()
            .try_fold(0_u64, |total, continuation| {
                let words = u64::try_from(continuation.modular.metadata.len()).map_err(|_| {
                    VarDctDecodeError::ArithmeticOverflow {
                        field: "HF-local Modular metadata length",
                    }
                })?;
                total
                    .checked_add(words.checked_mul(4).ok_or(
                        VarDctDecodeError::ArithmeticOverflow {
                            field: "HF-local Modular metadata bytes",
                        },
                    )?)
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "HF-local Modular metadata total",
                    })
            })?;
        let additional_permit = hf_metadata_bytes
            .checked_sub(source.memory.modular_metadata_bytes)
            .filter(|&bytes| bytes != 0)
            .map(|bytes| self.memory.try_reserve(bytes))
            .transpose()?;
        let limits = self.backend.device().limits();
        for continuation in &continuations {
            let bytes = u64::try_from(continuation.modular.metadata.len())
                .ok()
                .and_then(|words| words.checked_mul(4))
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "HF-local Modular metadata binding",
                })?;
            check_limit("HF-local Modular metadata", bytes, limits.max_buffer_size)?;
            check_limit(
                "HF-local Modular metadata",
                bytes,
                limits.max_storage_buffer_binding_size,
            )?;
        }
        let poll_permit = self
            .backend
            .submission_poller()
            .try_reserve()
            .map_err(DecodeError::PollBackpressure)?;
        let device = self.backend.device();
        lock_unpoisoned(&lifetime._modular_metadata).clear();
        let mut metadata_buffers = Vec::with_capacity(continuations.len());
        for continuation in &continuations {
            metadata_buffers.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jxl-wgpu VarDCT HF-local Modular metadata"),
                    contents: bytemuck::cast_slice(&continuation.modular.metadata),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            );
        }
        {
            let mut retained = lock_unpoisoned(&lifetime._modular_metadata);
            retained.extend(metadata_buffers.iter().cloned());
        }
        if let Some(permit) = additional_permit {
            lock_unpoisoned(&lifetime._transient_permits).push(permit);
        }
        let controls = source
            .packet
            .groups
            .iter()
            .zip(&source.groups)
            .zip(&continuations)
            .map(|((packet_group, group), continuation)| {
                let control = packet_group
                    .hf_stage_control(&source.packet, continuation)
                    .map_err(VarDctDecodeError::from)?;
                debug_assert_eq!(group.control.geometry, control.geometry);
                Ok(control)
            })
            .collect::<Result<Vec<_>, VarDctDecodeError>>()?;

        let windowed_batches = if let Some(plan) = &hf_packet_windows {
            let stream = lifetime._packet_stream_window.as_ref().ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "windowed HF packet plan has no shared stream buffer",
                },
            )?;
            let upload_len = usize::try_from(plan.stream_bytes).map_err(|_| {
                VarDctDecodeError::ArithmeticOverflow {
                    field: "HF packet stream window host length",
                }
            })?;
            let mut submissions = Vec::with_capacity(plan.stream_batches.len());
            for batch in plan.stream_batches.iter() {
                if batch.group_count == 0 || batch.segments.is_empty() {
                    return Err(VarDctDecodeError::EntropyWindowContract {
                        detail: "HF packet batch contains no segment",
                    }
                    .into());
                }
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu bounded HF packet stream batch"),
                });
                let mut stream_upload = vec![0_u8; upload_len];
                let mut group_uploads = Vec::with_capacity(batch.group_count);
                for segment_index in batch.segments.clone() {
                    let segment = *plan.stream_segments.get(segment_index).ok_or(
                        VarDctDecodeError::EntropyWindowContract {
                            detail: "HF packet batch references an absent segment",
                        },
                    )?;
                    let buffers = lifetime._groups.get(segment.group_index).ok_or(
                        VarDctDecodeError::EntropyWindowContract {
                            detail: "HF packet segment references an absent GPU group",
                        },
                    )?;
                    let metadata = metadata_buffers.get(segment.group_index).ok_or(
                        VarDctDecodeError::EntropyWindowContract {
                            detail: "HF packet segment references absent Modular metadata",
                        },
                    )?;
                    let control = *controls.get(segment.group_index).ok_or(
                        VarDctDecodeError::EntropyWindowContract {
                            detail: "HF packet segment references an absent control record",
                        },
                    )?;
                    let params = *plan.segment_params.get(segment_index).ok_or(
                        VarDctDecodeError::EntropyWindowContract {
                            detail: "HF packet segment has no parameter record",
                        },
                    )?;
                    copy_stream_segment(
                        &source.codestream,
                        segment,
                        &mut stream_upload,
                        "HF packet segment exceeds the source or reusable upload",
                    )?;
                    self.pipelines.packet.encode_hf(
                        device,
                        &mut encoder,
                        VarDctPacketBuffers {
                            codestream: stream,
                            modular_metadata: metadata,
                            reconstructed_lf: &buffers.reconstructed,
                            raw_hf_metadata: &buffers.raw_metadata,
                            coefficients: &buffers.coefficients,
                            status: &buffers.packet_status,
                            control: &buffers.packet_control,
                            modular_params: &buffers.modular_params,
                        },
                    );
                    group_uploads.push(HfPacketGroupUpload {
                        group_index: segment.group_index,
                        control,
                        params,
                    });
                }
                if group_uploads.len() != batch.group_count {
                    return Err(VarDctDecodeError::EntropyWindowContract {
                        detail: "HF packet batch group count disagrees with its segments",
                    }
                    .into());
                }
                submissions.push(HfPacketBatchSubmission {
                    stream_upload: stream_upload.into_boxed_slice(),
                    groups: group_uploads.into_boxed_slice(),
                    commands: encoder.finish(),
                });
            }
            Some(submissions)
        } else {
            None
        };
        let completion = Arc::new(MapCompletion::default());
        let submission = if let Some(batches) = windowed_batches {
            let batch_count = batches.len();
            let additional_submissions =
                batch_count
                    .checked_sub(1)
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "windowed HF packet execution has no batches",
                    })?;
            let current_submissions = self
                .runtime_stats
                .submissions_per_frame
                .load(Ordering::Acquire);
            let total_submissions = current_submissions
                .checked_add(additional_submissions)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "VarDCT dynamic submission count",
                })?;
            self.runtime_stats
                .hf_packet_stream_batch_count
                .store(batch_count, Ordering::Release);
            self.runtime_stats
                .submissions_per_frame
                .store(total_submissions, Ordering::Release);
            submit_hf_packet_commands(self.backend.queue(), batches, downstream, lifetime)?
        } else {
            let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu bounded VarDCT HF-local stage"),
            });
            for (((buffers, continuation), metadata), control) in lifetime
                ._groups
                .iter()
                .zip(&continuations)
                .zip(&metadata_buffers)
                .zip(&controls)
            {
                let params = VarDctModularParams::default()
                    .with_lz77_window(continuation.modular.lz77_window_words)
                    .with_self_correcting(continuation.modular.needs_self_correcting);
                self.backend.queue().write_buffer(
                    &buffers.packet_control,
                    0,
                    bytemuck::bytes_of(control),
                );
                self.backend.queue().write_buffer(
                    &buffers.modular_params,
                    0,
                    bytemuck::bytes_of(&params),
                );
                self.pipelines.packet.encode_hf(
                    device,
                    &mut commands,
                    VarDctPacketBuffers {
                        codestream: &lifetime._codestream,
                        modular_metadata: metadata,
                        reconstructed_lf: &buffers.reconstructed,
                        raw_hf_metadata: &buffers.raw_metadata,
                        coefficients: &buffers.coefficients,
                        status: &buffers.packet_status,
                        control: &buffers.packet_control,
                        modular_params: &buffers.modular_params,
                    },
                );
            }
            submit_vardct_downstream(
                self.backend.queue(),
                vec![commands.finish()],
                downstream,
                lifetime,
            )?
        };
        arm_status_map(lifetime, &completion, "VarDCT final validation mapping");
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll_permit.register(submission, move |error| {
            poll_completion.complete(Err(error));
        }) {
            completion.complete(Err(format!("VarDCT GPU poll registration failed: {error}")));
        }
        self.stage = VarDctPendingStage::Final { completion };
        Ok(())
    }

    fn finish(
        &mut self,
        mapping: Result<(), String>,
    ) -> DecodeResult<SubmittedGpuFrame<GpuImageFrame>> {
        mapping.map_err(DecodeError::backend)?;
        let lifetime = self
            .lifetime
            .take()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let mapped = lifetime
            .status_staging
            .slice(..)
            .get_mapped_range()
            .map_err(DecodeError::backend)?;
        let group_count = self.expected_groups.len();
        let packet_bytes = group_count
            .checked_mul(PACKET_STATUS_BYTES as usize)
            .ok_or(VarDctDecodeError::StatusAbi {
                status: "packet count",
            })?;
        let artifact_bytes = group_count
            .checked_mul(ARTIFACT_STATUS_BYTES as usize)
            .ok_or(VarDctDecodeError::StatusAbi {
                status: "artifact count",
            })?;
        let hf_offset =
            packet_bytes
                .checked_add(artifact_bytes)
                .ok_or(VarDctDecodeError::StatusAbi {
                    status: "aggregate offset",
                })?;
        for (index, expected) in self.expected_groups.iter().enumerate() {
            let packet_offset = index * PACKET_STATUS_BYTES as usize;
            let packet_status: GpuVarDctPacketStatus = mapped
                .get(packet_offset..packet_offset + PACKET_STATUS_BYTES as usize)
                .and_then(|bytes| bytemuck::try_pod_read_unaligned(bytes).ok())
                .ok_or(VarDctDecodeError::StatusAbi { status: "packet" })?;
            let artifact_offset = packet_bytes + index * ARTIFACT_STATUS_BYTES as usize;
            let artifact: GpuVarDctArtifactStatus = mapped
                .get(artifact_offset..artifact_offset + ARTIFACT_STATUS_BYTES as usize)
                .and_then(|bytes| bytemuck::try_pod_read_unaligned(bytes).ok())
                .ok_or(VarDctDecodeError::StatusAbi { status: "artifact" })?;
            let packet = packet_status
                .validate(VarDctPacketValidation {
                    expected_strategy: expected.uniform_transform,
                    expected_lf_samples: expected.expected_lf_samples,
                    block_count: expected.expected_blocks,
                    correlation_samples: expected.correlation_samples,
                    task_capacity: expected.task_capacity,
                    expected_global_scale: expected.expected_global_scale,
                    expected_quant_lf: expected.expected_quant_lf,
                    expected_extra_precision: expected.expected_extra_precision,
                })
                .map_err(VarDctDecodeError::from)?;
            if packet_status.coefficient_words != expected.expected_coefficients {
                return Err(VarDctDecodeError::ArtifactStatus {
                    field: "packet coefficient_words",
                    expected: expected.expected_coefficients,
                    actual: packet_status.coefficient_words,
                }
                .into());
            }
            artifact.validate().map_err(VarDctDecodeError::from)?;
            for (field, expected, actual) in [
                ("task_count", packet.first_blocks, artifact.task_count),
                (
                    "coefficient_words",
                    expected.expected_coefficients,
                    artifact.coefficient_words,
                ),
                (
                    "covered_blocks",
                    expected.expected_blocks,
                    artifact.covered_blocks,
                ),
                (
                    "consumed_block_info_entries",
                    packet.first_blocks,
                    artifact.consumed_block_info_entries,
                ),
                ("backend_requirements", 0, artifact.backend_requirements),
            ] {
                if actual != expected {
                    return Err(VarDctDecodeError::ArtifactStatus {
                        field,
                        expected,
                        actual,
                    }
                    .into());
                }
            }
        }
        let hf_status_bytes = mapped
            .get(hf_offset..)
            .ok_or(VarDctDecodeError::StatusAbi {
                status: "HF coefficient",
            })?;
        let hf_statuses = bytemuck::try_cast_slice::<u8, GpuHfCoefficientStatus>(hf_status_bytes)
            .map_err(|_| VarDctDecodeError::StatusAbi {
            status: "HF coefficient",
        })?;
        if hf_statuses.len() != self.expected_hf_group_indices.len() {
            return Err(VarDctDecodeError::StatusAbi {
                status: "HF coefficient count",
            }
            .into());
        }
        for (&group, status) in self
            .expected_hf_group_indices
            .iter()
            .zip(hf_statuses.iter().copied())
        {
            status.validate(group).map_err(VarDctDecodeError::from)?;
        }
        drop(mapped);
        let output_id = OutputId(0);
        let mut regions = BTreeMap::new();
        regions.insert(
            output_id,
            vec![Region::new(
                0,
                0,
                self.layout.extent.width,
                self.layout.extent.height,
            )],
        );
        Ok(SubmittedGpuFrame::new(
            FrameMetadata {
                index: 0,
                duration: FrameDuration::still(),
                presentation_ticks: 0,
                timecode: None,
                is_last: true,
                is_keyframe: true,
                name: std::mem::take(&mut self.frame_name),
            },
            GpuImageFrame {
                token: self.token,
                outputs: vec![GpuImageOutput {
                    id: output_id,
                    layout: self.layout.clone(),
                    buffer: lifetime.output.clone(),
                }],
                changed: ChangedRegions { outputs: regions },
            },
        ))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl GpuPendingFrame for VarDctPendingFrame {
    type Frame = GpuImageFrame;

    fn wait(mut self) -> DecodeResult<SubmittedGpuFrame<Self::Frame>> {
        loop {
            let mapping = self.stage_completion().wait();
            if let Some((source, downstream)) = self.take_local_stage() {
                self.submit_hf_stage(mapping, source, downstream)?;
            } else {
                return self.finish(mapping);
            }
        }
    }

    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<DecodeResult<SubmittedGpuFrame<Self::Frame>>> {
        loop {
            if let Err(error) = self.backend.device().poll(wgpu::PollType::Poll) {
                return Poll::Ready(Err(DecodeError::backend(error)));
            }
            let Some(mapping) = self.stage_completion().poll(context) else {
                return Poll::Pending;
            };
            if let Some((source, downstream)) = self.take_local_stage() {
                if let Err(error) = self.submit_hf_stage(mapping, source, downstream) {
                    return Poll::Ready(Err(error));
                }
            } else {
                return Poll::Ready(self.finish(mapping));
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl GpuPendingFrame for VarDctPendingFrame {
    type Frame = GpuImageFrame;

    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<DecodeResult<SubmittedGpuFrame<Self::Frame>>> {
        loop {
            if let Err(error) = self.backend.device().poll(wgpu::PollType::Poll) {
                return Poll::Ready(Err(DecodeError::backend(error)));
            }
            let Some(mapping) = self.stage_completion().poll(context) else {
                return Poll::Pending;
            };
            if let Some((source, downstream)) = self.take_local_stage() {
                if let Err(error) = self.submit_hf_stage(mapping, source, downstream) {
                    return Poll::Ready(Err(error));
                }
            } else {
                return Poll::Ready(self.finish(mapping));
            }
        }
    }
}

fn resident_binding(
    buffer: &wgpu::Buffer,
) -> Result<ResidentStorageBinding<'_>, VarDctDecodeError> {
    Ok(ResidentStorageBinding::entire(buffer)?)
}

fn resident_image_planes<'a>(
    buffers: &'a [wgpu::Buffer; 3],
    width: u32,
    height: u32,
    stride: u32,
) -> Result<[ResidentF32Plane<'a>; 3], VarDctDecodeError> {
    Ok([
        ResidentF32Plane {
            storage: resident_binding(&buffers[0])?,
            width,
            height,
            stride,
        },
        ResidentF32Plane {
            storage: resident_binding(&buffers[1])?,
            width,
            height,
            stride,
        },
        ResidentF32Plane {
            storage: resident_binding(&buffers[2])?,
            width,
            height,
            stride,
        },
    ])
}

fn upload_codestream(
    codestream: &GpuCodestream,
    buffer: &wgpu::Buffer,
    padded_bytes: u64,
) -> Result<(), VarDctDecodeError> {
    if padded_bytes < codestream.logical_bytes() || !padded_bytes.is_multiple_of(4) {
        return Err(VarDctDecodeError::EntropyWindowContract {
            detail: "GPU codestream buffer does not cover an aligned logical source",
        });
    }
    let logical_size = usize::try_from(codestream.logical_bytes()).map_err(|_| {
        VarDctDecodeError::ArithmeticOverflow {
            field: "codestream upload length",
        }
    })?;
    let mut mapped = buffer
        .get_mapped_range_mut(..)
        .map_err(|source| VarDctDecodeError::CodestreamMap { source })?;
    let upload_result = (|| {
        let mut mapped_cursor = 0usize;
        codestream
            .for_each_range_chunk(0..codestream.logical_bytes(), |chunk| -> DecodeResult<()> {
                let mapped_end = mapped_cursor
                    .checked_add(chunk.len())
                    .ok_or_else(|| DecodeError::backend("codestream mapped offset overflow"))?;
                if mapped_end > mapped.len() {
                    return Err(DecodeError::EngineContract(
                        "codestream span exceeds the mapped GPU buffer",
                    ));
                }
                mapped
                    .slice(mapped_cursor..mapped_end)
                    .copy_from_slice(chunk);
                mapped_cursor = mapped_end;
                Ok(())
            })
            .map_err(map_codestream_source_error)?;
        if mapped_cursor != logical_size {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "codestream spans did not fill the logical mapped range",
            });
        }
        if logical_size < mapped.len() {
            mapped.slice(logical_size..).fill(0);
        }
        Ok(())
    })();
    drop(mapped);
    buffer.unmap();
    upload_result
}

fn submit_vardct(
    backend: &WgpuBackend,
    pipelines: Arc<VarDctPipelines>,
    memory: MemoryBudget,
    runtime_stats: Arc<VarDctRuntimeStats>,
    source: VarDctSource,
    permits: VarDctMemoryPermits,
    poll_permit: SubmissionPollPermit,
) -> Result<VarDctPendingFrame, VarDctDecodeError> {
    let device = backend.device();
    let codestream_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu VarDCT codestream"),
        size: source.memory.codestream_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: true,
    });
    upload_codestream(
        &source.codestream,
        &codestream_buffer,
        source.memory.codestream_bytes,
    )?;
    let staged_local_trees = source.packet.requires_local_tree_staging();
    let modular_metadata = if staged_local_trees {
        source
            .packet
            .groups
            .iter()
            .map(|group| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jxl-wgpu VarDCT LF-local Modular metadata"),
                    contents: bytemuck::cast_slice(&group.lf_modular.metadata),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect::<Vec<_>>()
    } else {
        vec![
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu VarDCT global Modular metadata"),
                contents: bytemuck::cast_slice(&source.packet.modular_metadata),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        ]
    };
    let storage = |label: &'static str, size: u64, extra: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | extra,
            mapped_at_creation: false,
        })
    };
    let packet_stream_window_bytes = source
        .lf_packet_windows
        .as_ref()
        .map(|plan| plan.stream_bytes)
        .or_else(|| {
            source
                .combined_packet_windows
                .as_ref()
                .map(|plan| plan.stream_bytes)
        });
    let packet_stream_window = packet_stream_window_bytes.map(|bytes| {
        storage(
            "jxl-wgpu reusable packet entropy stream window",
            bytes,
            wgpu::BufferUsages::COPY_DST,
        )
    });
    if source.groups.len() != source.packet.groups.len() {
        return Err(VarDctDecodeError::GroupPlanCount {
            component: "packet source",
            expected: source.packet.groups.len(),
            actual: source.groups.len(),
        });
    }
    if let Some(plan) = &source.hf_coefficients
        && plan.groups.len() != source.packet.groups.len()
    {
        return Err(VarDctDecodeError::GroupPlanCount {
            component: "HF coefficient",
            expected: source.packet.groups.len(),
            actual: plan.groups.len(),
        });
    }
    let mut group_buffers = Vec::with_capacity(source.groups.len());
    for (index, (packet_group, group)) in
        source.packet.groups.iter().zip(&source.groups).enumerate()
    {
        let predictor_capacity = source.packet.needs_self_correcting || staged_local_trees;
        let reconstructed_bytes = u64::from(packet_group.reconstructed_words(predictor_capacity)?)
            .checked_mul(4)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "LF-group reconstruction bytes",
            })?;
        let hf_lz77_bytes = source
            .hf_coefficients
            .as_ref()
            .and_then(|plan| plan.groups.get(index))
            .map_or(0, HfCoefficientGroupExecutionPlan::lz77_scratch_bytes);
        let hf_execution_state_bytes = source
            .hf_coefficients
            .as_ref()
            .and_then(|plan| plan.groups.get(index))
            .map_or(0, HfCoefficientGroupExecutionPlan::execution_state_bytes);
        let reconstructed = storage(
            "jxl-wgpu VarDCT LF-group reconstruction",
            reconstructed_bytes
                .checked_add(hf_lz77_bytes)
                .and_then(|bytes| bytes.checked_add(hf_execution_state_bytes))
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "LF-group reconstruction, HF LZ77, and execution-state bytes",
                })?,
            wgpu::BufferUsages::COPY_DST,
        );
        let raw_metadata = storage(
            "jxl-wgpu VarDCT LF-group raw HF metadata",
            u64::from(group.control.capacities[1]) * 4,
            wgpu::BufferUsages::COPY_DST,
        );
        let coefficients = storage(
            "jxl-wgpu VarDCT LF-group coefficients",
            u64::from(packet_group.coefficient_words()) * 4,
            wgpu::BufferUsages::COPY_DST,
        );
        let packet_status = storage(
            "jxl-wgpu VarDCT LF-group packet status",
            PACKET_STATUS_BYTES,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let packet_control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu VarDCT LF-group packet control"),
            contents: bytemuck::bytes_of(&group.control),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let modular = &packet_group.lf_modular;
        let params = VarDctModularParams::default()
            .with_lz77_window(if staged_local_trees {
                modular.lz77_window_words
            } else {
                packet_group.lz77_window_words
            })
            .with_self_correcting(if staged_local_trees {
                modular.needs_self_correcting
            } else {
                source.packet.needs_self_correcting
            });
        let modular_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu VarDCT LF-group Modular params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let artifact = storage(
            "jxl-wgpu VarDCT LF-group resident artifact",
            group.artifact_layout.artifact_bytes,
            wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let occupancy = storage(
            "jxl-wgpu VarDCT LF-group artifact occupancy",
            group.artifact_layout.occupancy_bytes,
            wgpu::BufferUsages::COPY_DST,
        );
        let artifact_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu VarDCT LF-group artifact params"),
            contents: bytemuck::bytes_of(&group.artifact_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        group_buffers.push(VarDctGroupJobBuffers {
            reconstructed,
            raw_metadata,
            coefficients,
            packet_status,
            packet_control,
            modular_params,
            artifact,
            occupancy,
            artifact_uniform,
        });
    }
    let lf_temporary = storage(
        "jxl-wgpu VarDCT dequantized LF temporary",
        source.memory.lf_temporary_bytes,
        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let resource_values = source.resource_layout.initial_values()?;
    let resources = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu VarDCT resource vectors"),
        contents: bytemuck::cast_slice(&resource_values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let hf_coefficient_buffers = source.hf_coefficients.as_ref().map(|plan| {
        let windowed = plan.uses_bounded_stream_windows();
        HfCoefficientJobBuffers {
            entropy_bundle: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu HF entropy bundle"),
                contents: bytemuck::cast_slice(&plan.entropy_words),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            order_table: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu HF natural-order table"),
                contents: bytemuck::cast_slice(&plan.order_words),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            stream_window: windowed.then(|| {
                storage(
                    "jxl-wgpu reusable HF coefficient stream window",
                    plan.stream_window_bytes(),
                    wgpu::BufferUsages::COPY_DST,
                )
            }),
            params_window: windowed.then(|| {
                storage(
                    "jxl-wgpu reusable HF coefficient parameter window",
                    plan.reusable_params_bytes(),
                    wgpu::BufferUsages::COPY_DST,
                )
            }),
            groups: plan
                .groups
                .iter()
                .map(|group| HfCoefficientGroupJobBuffers {
                    params: (!windowed).then(|| {
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("jxl-wgpu LF-group HF pass-group params"),
                            contents: bytemuck::cast_slice(&group.params),
                            usage: wgpu::BufferUsages::STORAGE,
                        })
                    }),
                    status: storage(
                        "jxl-wgpu LF-group HF pass-group status",
                        group.status_bytes(),
                        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    ),
                    sink_params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("jxl-wgpu LF-group HF coefficient sink params"),
                        contents: bytemuck::bytes_of(&group.sink_params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    }),
                })
                .collect(),
        }
    });
    let plane_bytes = source.memory.xyb_plane_bytes / 3;
    let xyb_planes = [
        "jxl-wgpu VarDCT X plane",
        "jxl-wgpu VarDCT Y plane",
        "jxl-wgpu VarDCT B plane",
    ]
    .map(|label| storage(label, plane_bytes, wgpu::BufferUsages::COPY_DST));
    let restoration_planes = (source.gaborish.is_some() || source.epf.is_some()).then(|| {
        [
            "jxl-wgpu VarDCT restoration scratch X plane",
            "jxl-wgpu VarDCT restoration scratch Y plane",
            "jxl-wgpu VarDCT restoration scratch B plane",
        ]
        .map(|label| storage(label, plane_bytes, wgpu::BufferUsages::empty()))
    });
    let epf_sigma = source.epf.as_ref().map(|_| {
        storage(
            "jxl-wgpu VarDCT EPF inverse-sigma plane",
            source.memory.epf_sigma_bytes,
            wgpu::BufferUsages::COPY_DST,
        )
    });
    let mut output_usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    if backend.direct_readback_enabled() {
        output_usage |= wgpu::BufferUsages::MAP_READ;
    }
    let output = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu VarDCT packed RGB8 output"),
        size: source.memory.output_lease_bytes,
        usage: output_usage,
        mapped_at_creation: false,
    }));
    let status_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu VarDCT aggregate validation staging"),
        size: source.memory.validation_staging_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut packet_commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("jxl-wgpu bounded VarDCT packet stage"),
    });
    for buffer in [
        &lf_temporary,
        &xyb_planes[0],
        &xyb_planes[1],
        &xyb_planes[2],
        output.as_ref(),
    ] {
        packet_commands.clear_buffer(buffer, 0, None);
    }
    for group in &group_buffers {
        for buffer in [
            &group.reconstructed,
            &group.raw_metadata,
            &group.coefficients,
            &group.packet_status,
            &group.artifact,
            &group.occupancy,
        ] {
            packet_commands.clear_buffer(buffer, 0, None);
        }
    }
    if let Some(buffers) = &hf_coefficient_buffers {
        for group in &buffers.groups {
            packet_commands.clear_buffer(&group.status, 0, None);
        }
    }
    if let Some(sigma) = &epf_sigma {
        packet_commands.clear_buffer(sigma, 0, None);
    }
    let (lf_stage_commands, combined_packet_batches, mut commands) = if let Some(plan) =
        &source.lf_packet_windows
    {
        if !staged_local_trees {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "LF packet windows require staged local trees",
            });
        }
        let stream =
            packet_stream_window
                .as_ref()
                .ok_or(VarDctDecodeError::EntropyWindowContract {
                    detail: "windowed LF packet plan has no stream buffer",
                })?;
        let upload_len = usize::try_from(plan.stream_bytes).map_err(|_| {
            VarDctDecodeError::ArithmeticOverflow {
                field: "LF packet stream window host length",
            }
        })?;
        let mut first_commands = Some(packet_commands);
        let mut submissions = Vec::with_capacity(plan.stream_batches.len());
        for (batch_index, batch) in plan.stream_batches.iter().enumerate() {
            if batch.group_count != 1 || batch.segments.end != batch.segments.start + 1 {
                return Err(VarDctDecodeError::EntropyWindowContract {
                    detail: "serial LF packet batch does not contain exactly one segment",
                });
            }
            let segment_index = batch.segments.start;
            let segment = *plan.stream_segments.get(segment_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "LF packet batch references an absent segment",
                },
            )?;
            if segment.group_index != batch.first_group {
                return Err(VarDctDecodeError::EntropyWindowContract {
                    detail: "LF packet segment and batch group indices disagree",
                });
            }
            let buffers = group_buffers.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "LF packet segment references an absent GPU group",
                },
            )?;
            let metadata = modular_metadata.get(segment.group_index).ok_or(
                VarDctDecodeError::GroupPlanCount {
                    component: "LF-local Modular metadata",
                    expected: group_buffers.len(),
                    actual: modular_metadata.len(),
                },
            )?;
            let mut encoder = first_commands.take().unwrap_or_else(|| {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu bounded LF packet stream batch"),
                })
            });
            pipelines.packet.encode_lf(
                device,
                &mut encoder,
                VarDctPacketBuffers {
                    codestream: stream,
                    modular_metadata: metadata,
                    reconstructed_lf: &buffers.reconstructed,
                    raw_hf_metadata: &buffers.raw_metadata,
                    coefficients: &buffers.coefficients,
                    status: &buffers.packet_status,
                    control: &buffers.packet_control,
                    modular_params: &buffers.modular_params,
                },
            );
            if batch_index + 1 == plan.stream_batches.len() {
                for (index, buffers) in group_buffers.iter().enumerate() {
                    encoder.copy_buffer_to_buffer(
                        &buffers.packet_status,
                        0,
                        &status_staging,
                        u64::try_from(index).map_err(|_| {
                            VarDctDecodeError::ArithmeticOverflow {
                                field: "LF staging status index",
                            }
                        })? * PACKET_STATUS_BYTES,
                        PACKET_STATUS_BYTES,
                    );
                }
            }
            let mut stream_upload = vec![0_u8; upload_len];
            copy_stream_segment(
                &source.codestream,
                segment,
                &mut stream_upload,
                "LF packet segment exceeds the source or reusable upload",
            )?;
            let params = *plan.segment_params.get(segment_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "LF packet segment has no parameter record",
                },
            )?;
            submissions.push(LfPacketBatchSubmission {
                group_index: segment.group_index,
                stream_upload: stream_upload.into_boxed_slice(),
                params,
                commands: encoder.finish(),
            });
        }
        if first_commands.is_some() || submissions.is_empty() {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "windowed LF packet execution has no dispatch",
            });
        }
        (
            Some(LfPacketCommands::Windowed(submissions)),
            None,
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu bounded VarDCT downstream stage"),
            }),
        )
    } else if let Some(plan) = &source.combined_packet_windows {
        if staged_local_trees {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "combined packet windows cannot stage local trees",
            });
        }
        let stream =
            packet_stream_window
                .as_ref()
                .ok_or(VarDctDecodeError::EntropyWindowContract {
                    detail: "windowed combined packet plan has no stream buffer",
                })?;
        let metadata = modular_metadata
            .first()
            .ok_or(VarDctDecodeError::GroupPlanCount {
                component: "global Modular metadata",
                expected: 1,
                actual: 0,
            })?;
        let upload_len = usize::try_from(plan.stream_bytes).map_err(|_| {
            VarDctDecodeError::ArithmeticOverflow {
                field: "combined packet stream window host length",
            }
        })?;
        let mut first_commands = Some(packet_commands);
        let mut submissions = Vec::with_capacity(plan.stream_batches.len());
        for batch in plan.stream_batches.iter() {
            if batch.group_count == 0 || batch.segments.is_empty() {
                return Err(VarDctDecodeError::EntropyWindowContract {
                    detail: "combined packet batch contains no segment",
                });
            }
            let mut encoder = first_commands.take().unwrap_or_else(|| {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu bounded combined packet stream batch"),
                })
            });
            let mut stream_upload = vec![0_u8; upload_len];
            let mut group_uploads = Vec::with_capacity(batch.group_count);
            for segment_index in batch.segments.clone() {
                let segment = *plan.stream_segments.get(segment_index).ok_or(
                    VarDctDecodeError::EntropyWindowContract {
                        detail: "combined packet batch references an absent segment",
                    },
                )?;
                let buffers = group_buffers.get(segment.group_index).ok_or(
                    VarDctDecodeError::EntropyWindowContract {
                        detail: "combined packet segment references an absent GPU group",
                    },
                )?;
                let params = *plan.segment_params.get(segment_index).ok_or(
                    VarDctDecodeError::EntropyWindowContract {
                        detail: "combined packet segment has no parameter record",
                    },
                )?;
                copy_stream_segment(
                    &source.codestream,
                    segment,
                    &mut stream_upload,
                    "combined packet segment exceeds the source or reusable upload",
                )?;
                pipelines.packet.encode(
                    device,
                    &mut encoder,
                    VarDctPacketBuffers {
                        codestream: stream,
                        modular_metadata: metadata,
                        reconstructed_lf: &buffers.reconstructed,
                        raw_hf_metadata: &buffers.raw_metadata,
                        coefficients: &buffers.coefficients,
                        status: &buffers.packet_status,
                        control: &buffers.packet_control,
                        modular_params: &buffers.modular_params,
                    },
                );
                group_uploads.push(CombinedPacketGroupUpload {
                    group_index: segment.group_index,
                    params,
                });
            }
            if group_uploads.len() != batch.group_count {
                return Err(VarDctDecodeError::EntropyWindowContract {
                    detail: "combined packet batch group count disagrees with its segments",
                });
            }
            submissions.push(CombinedPacketBatchSubmission {
                stream_upload: stream_upload.into_boxed_slice(),
                groups: group_uploads.into_boxed_slice(),
                commands: encoder.finish(),
            });
        }
        if first_commands.is_some() || submissions.is_empty() {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "windowed combined packet execution has no dispatch",
            });
        }
        (
            None,
            Some(submissions),
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu bounded VarDCT downstream stage"),
            }),
        )
    } else {
        for (index, buffers) in group_buffers.iter().enumerate() {
            let metadata = if staged_local_trees {
                modular_metadata
                    .get(index)
                    .ok_or(VarDctDecodeError::GroupPlanCount {
                        component: "LF-local Modular metadata",
                        expected: group_buffers.len(),
                        actual: modular_metadata.len(),
                    })?
            } else {
                modular_metadata
                    .first()
                    .ok_or(VarDctDecodeError::GroupPlanCount {
                        component: "global Modular metadata",
                        expected: 1,
                        actual: 0,
                    })?
            };
            let buffers = VarDctPacketBuffers {
                codestream: &codestream_buffer,
                modular_metadata: metadata,
                reconstructed_lf: &buffers.reconstructed,
                raw_hf_metadata: &buffers.raw_metadata,
                coefficients: &buffers.coefficients,
                status: &buffers.packet_status,
                control: &buffers.packet_control,
                modular_params: &buffers.modular_params,
            };
            if staged_local_trees {
                pipelines
                    .packet
                    .encode_lf(device, &mut packet_commands, buffers);
            } else {
                pipelines
                    .packet
                    .encode(device, &mut packet_commands, buffers);
            }
        }
        if staged_local_trees {
            for (index, buffers) in group_buffers.iter().enumerate() {
                packet_commands.copy_buffer_to_buffer(
                    &buffers.packet_status,
                    0,
                    &status_staging,
                    u64::try_from(index).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
                        field: "LF staging status index",
                    })? * PACKET_STATUS_BYTES,
                    PACKET_STATUS_BYTES,
                );
            }
        }
        if staged_local_trees {
            (
                Some(LfPacketCommands::Whole(packet_commands.finish())),
                None,
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu bounded VarDCT downstream stage"),
                }),
            )
        } else {
            (None, None, packet_commands)
        }
    };
    let mut resource_uniforms = Vec::with_capacity(source.groups.len());
    for (group, buffers) in source.groups.iter().zip(&group_buffers) {
        resource_uniforms.push(pipelines.resource.encode(
            device,
            &mut commands,
            VarDctResourceBuffers {
                quantized_lf: &buffers.reconstructed,
                dequantized_lf: &lf_temporary,
            },
            group.resource_params,
        ));
    }
    let [blocks_x, blocks_y] = source.packet.block_extent();
    let smoothing_thresholds = source
        .groups
        .first()
        .ok_or(VarDctDecodeError::GroupPlanCount {
            component: "packet source",
            expected: 1,
            actual: 0,
        })?
        .resource_params
        .smoothing_thresholds();
    let adaptive_lf_uniform = if source.packet.profile.adaptive_lf_smoothing {
        Some(pipelines.adaptive_lf.encode(
            device,
            &mut commands,
            AdaptiveLfBuffers {
                input: &lf_temporary,
                output: &resources,
            },
            AdaptiveLfParams::new(
                blocks_x,
                blocks_y,
                0,
                source.resource_layout.lf_offset,
                smoothing_thresholds,
            ),
        ))
    } else {
        commands.copy_buffer_to_buffer(
            &lf_temporary,
            0,
            &resources,
            u64::from(source.resource_layout.lf_offset) * 16,
            source.memory.lf_temporary_bytes,
        );
        None
    };
    for buffers in &group_buffers {
        pipelines.artifact.encode(
            device,
            &mut commands,
            HfMetadataLoweringBuffers {
                raw_metadata: &buffers.raw_metadata,
                artifact: &buffers.artifact,
                occupancy: &buffers.occupancy,
                resources: &resources,
                params: &buffers.artifact_uniform,
            },
        );
    }
    let mut epf_sigma_uniforms = Vec::new();
    match (source.epf.as_ref(), epf_sigma.as_ref()) {
        (Some(plan), Some(sigma)) => {
            if plan.sigma_groups.len() != group_buffers.len() {
                return Err(VarDctDecodeError::GroupPlanCount {
                    component: "EPF sigma",
                    expected: group_buffers.len(),
                    actual: plan.sigma_groups.len(),
                });
            }
            epf_sigma_uniforms.reserve(plan.sigma_groups.len());
            for (&config, buffers) in plan.sigma_groups.iter().zip(&group_buffers) {
                epf_sigma_uniforms.push(pipelines.epf_sigma.encode(
                    device,
                    &mut commands,
                    &buffers.raw_metadata,
                    &buffers.artifact,
                    sigma,
                    config,
                )?);
            }
        }
        (None, None) => {}
        _ => unreachable!("EPF plan and sigma buffer are constructed together"),
    }
    let mut windowed_before_coefficients = None;
    let mut windowed_coefficient_batches = Vec::new();
    if let (Some(plan), Some(buffers)) = (
        source.hf_coefficients.as_ref(),
        hf_coefficient_buffers.as_ref(),
    ) {
        if plan.uses_bounded_stream_windows() {
            windowed_before_coefficients = Some(commands.finish());
            commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu bounded VarDCT post-coefficient stage"),
            });
            let stream_window =
                buffers
                    .stream_window
                    .as_ref()
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "windowed AC plan has no stream buffer",
                    })?;
            let params_window =
                buffers
                    .params_window
                    .as_ref()
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "windowed AC plan has no parameter buffer",
                    })?;
            let upload_len = usize::try_from(plan.stream_window_bytes()).map_err(|_| {
                VarDctDecodeError::ArithmeticOverflow {
                    field: "HF stream window host length",
                }
            })?;
            for ((group_plan, hf_buffers), group_buffers) in
                plan.groups.iter().zip(&buffers.groups).zip(&group_buffers)
            {
                for batch in &group_plan.stream_batches {
                    let mut stream_upload = vec![0_u8; upload_len];
                    for segment in &group_plan.stream_segments[batch.segments.clone()] {
                        copy_stream_segment(
                            &source.codestream,
                            *segment,
                            &mut stream_upload,
                            "HF stream segment exceeds the source or reusable upload",
                        )?;
                    }
                    let params_upload =
                        bytemuck::cast_slice(&group_plan.segment_params[batch.segments.clone()])
                            .to_vec()
                            .into_boxed_slice();
                    let mut batch_commands =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("jxl-wgpu bounded HF coefficient stream batch"),
                        });
                    pipelines.hf_coefficients.encode(
                        device,
                        &mut batch_commands,
                        HfCoefficientBuffers {
                            codestream: stream_window,
                            entropy_bundle: &buffers.entropy_bundle,
                            reconstruction: &group_buffers.reconstructed,
                            params: params_window,
                            status: &hf_buffers.status,
                            artifact: &group_buffers.artifact,
                            order_table: &buffers.order_table,
                            coefficients: &group_buffers.coefficients,
                            sink_params: &hf_buffers.sink_params,
                        },
                        u32::try_from(batch.group_count).map_err(|_| {
                            VarDctDecodeError::ArithmeticOverflow {
                                field: "HF stream batch dispatch count",
                            }
                        })?,
                    );
                    windowed_coefficient_batches.push(HfCoefficientBatchSubmission {
                        stream_upload: stream_upload.into_boxed_slice(),
                        params_upload,
                        commands: batch_commands.finish(),
                    });
                }
            }
        } else {
            for ((group_plan, hf_buffers), group_buffers) in
                plan.groups.iter().zip(&buffers.groups).zip(&group_buffers)
            {
                let params =
                    hf_buffers
                        .params
                        .as_ref()
                        .ok_or(VarDctDecodeError::EntropyWindowContract {
                            detail: "whole-range AC plan has no parameter buffer",
                        })?;
                pipelines.hf_coefficients.encode(
                    device,
                    &mut commands,
                    HfCoefficientBuffers {
                        codestream: &codestream_buffer,
                        entropy_bundle: &buffers.entropy_bundle,
                        reconstruction: &group_buffers.reconstructed,
                        params,
                        status: &hf_buffers.status,
                        artifact: &group_buffers.artifact,
                        order_table: &buffers.order_table,
                        coefficients: &group_buffers.coefficients,
                        sink_params: &hf_buffers.sink_params,
                    },
                    u32::try_from(group_plan.params.len()).map_err(|_| {
                        VarDctDecodeError::ArithmeticOverflow {
                            field: "LF-group HF pass-group dispatch count",
                        }
                    })?,
                );
            }
        }
    }
    let padded_width = blocks_x
        .checked_mul(8)
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "padded output width",
        })?;
    let padded_height = blocks_y
        .checked_mul(8)
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "padded output height",
        })?;
    let correlation_width = source.packet.profile.width.div_ceil(64);
    let correlation_height = source.packet.profile.height.div_ceil(64);
    let mut resident_scratch = Vec::with_capacity(source.groups.len());
    for ((packet_group, group), buffers) in source
        .packet
        .groups
        .iter()
        .zip(&source.groups)
        .zip(&group_buffers)
    {
        resident_scratch.push(pipelines.renderer.encode(
            device,
            &mut commands,
            ResidentVarDctInputs {
                coefficients: resident_binding(&buffers.coefficients)?,
                artifact: resident_binding(&buffers.artifact)?,
                resources: resident_binding(&resources)?,
                outputs: resident_image_planes(
                    &xyb_planes,
                    padded_width,
                    padded_height,
                    padded_width,
                )?,
                indirect: &buffers.artifact,
                indirect_base_offset: u64::from(group.artifact_layout.indirect_offset_words) * 4,
                config: ResidentVarDctRenderConfig {
                    task_capacity: packet_group.task_capacity,
                    scratch_scalars: packet_group.coefficient_words(),
                    task_word_offset: group.artifact_layout.tasks_offset_words,
                    bucket_word_offset: group.artifact_layout.buckets_offset_words,
                    quant_offset: group.quant_offset,
                    correlation_offset: source.resource_layout.correlation_offset,
                    lf_offset: source.resource_layout.lf_offset,
                    lf_stride: blocks_x,
                    correlation_width,
                    correlation_height,
                    quant_biases: source.quant_biases,
                },
            },
        )?);
    }
    let image_width = source.packet.profile.width;
    let image_height = source.packet.profile.height;
    let mut restoration = restoration_planes
        .as_ref()
        .map(|scratch| RestorationCursor::new(&xyb_planes, scratch));
    let gaborish_uniform = match (source.gaborish, restoration.as_mut()) {
        (Some(weights), Some(restoration)) => {
            let (input_buffers, output_buffers) = restoration.advance();
            let uniform = pipelines.gaborish.encode(
                device,
                &mut commands,
                ResidentGaborishInputs {
                    inputs: resident_image_planes(
                        input_buffers,
                        image_width,
                        image_height,
                        padded_width,
                    )?,
                    outputs: resident_image_planes(
                        output_buffers,
                        image_width,
                        image_height,
                        padded_width,
                    )?,
                    weights,
                },
            )?;
            Some(uniform)
        }
        (None, _) => None,
        (Some(_), None) => unreachable!("Gaborish requires restoration scratch planes"),
    };
    let mut epf_uniforms =
        Vec::with_capacity(source.epf.as_ref().map_or(0, |plan| plan.passes.len()));
    if let Some(epf) = &source.epf {
        let restoration = restoration
            .as_mut()
            .unwrap_or_else(|| unreachable!("EPF requires restoration scratch planes"));
        let sigma_buffer = epf_sigma
            .as_ref()
            .unwrap_or_else(|| unreachable!("EPF requires a sigma plane"));
        let sigma = ResidentF32Plane {
            storage: resident_binding(sigma_buffer)?,
            width: blocks_x,
            height: blocks_y,
            stride: blocks_x,
        };
        for &parameters in &epf.passes {
            let (input_buffers, output_buffers) = restoration.advance();
            epf_uniforms.push(pipelines.epf.encode(
                device,
                &mut commands,
                ResidentEpfInputs {
                    inputs: resident_image_planes(
                        input_buffers,
                        image_width,
                        image_height,
                        padded_width,
                    )?,
                    outputs: resident_image_planes(
                        output_buffers,
                        image_width,
                        image_height,
                        padded_width,
                    )?,
                    sigma,
                    parameters,
                },
            )?);
        }
    }
    let presentation_planes = restoration
        .as_ref()
        .map_or(&xyb_planes, RestorationCursor::current);
    let output_scratch = pipelines.output.encode(
        device,
        &mut commands,
        VarDctOutputInputs {
            planes: [
                VarDctOutputPlane {
                    storage: resident_binding(&presentation_planes[0])?,
                    stride: padded_width,
                },
                VarDctOutputPlane {
                    storage: resident_binding(&presentation_planes[1])?,
                    stride: padded_width,
                },
                VarDctOutputPlane {
                    storage: resident_binding(&presentation_planes[2])?,
                    stride: padded_width,
                },
            ],
            output: resident_binding(&output)?,
            config: VarDctOutputConfig {
                width: source.packet.profile.width,
                height: source.packet.profile.height,
                inverse_opsin: source.inverse_opsin,
            },
        },
    )?;
    debug_assert_eq!(output_scratch.plan, source.output_plan);
    let restoration_buffers = restoration_planes.map(|planes| RestorationJobBuffers {
        _planes: planes,
        _gaborish_uniform: gaborish_uniform,
        _epf_sigma: epf_sigma,
        _epf_sigma_uniforms: epf_sigma_uniforms,
        _epf_uniforms: epf_uniforms,
    });
    let packet_status_end = source.memory.packet_status_bytes;
    let artifact_status_end = packet_status_end
        .checked_add(
            u64::try_from(group_buffers.len())
                .map_err(|_| VarDctDecodeError::ArithmeticOverflow {
                    field: "LF-group status count",
                })?
                .checked_mul(ARTIFACT_STATUS_BYTES)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "artifact status staging bytes",
                })?,
        )
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "artifact status staging end",
        })?;
    for (index, (group, buffers)) in source.groups.iter().zip(&group_buffers).enumerate() {
        let index = u64::try_from(index).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
            field: "LF-group status index",
        })?;
        commands.copy_buffer_to_buffer(
            &buffers.packet_status,
            0,
            &status_staging,
            index * PACKET_STATUS_BYTES,
            PACKET_STATUS_BYTES,
        );
        commands.copy_buffer_to_buffer(
            &buffers.artifact,
            u64::from(group.artifact_layout.status_offset_words) * 4,
            &status_staging,
            packet_status_end + index * ARTIFACT_STATUS_BYTES,
            ARTIFACT_STATUS_BYTES,
        );
    }
    if let Some(buffers) = &hf_coefficient_buffers {
        let mut offset = artifact_status_end;
        for group in &buffers.groups {
            let status_bytes = group.status.size();
            commands.copy_buffer_to_buffer(&group.status, 0, &status_staging, offset, status_bytes);
            offset =
                offset
                    .checked_add(status_bytes)
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "HF status staging offset",
                    })?;
        }
        debug_assert_eq!(offset, source.memory.validation_staging_bytes);
    }

    let after_coefficients = commands.finish();
    let downstream_commands = if let Some(before_coefficients) = windowed_before_coefficients {
        VarDctDownstreamCommands::Windowed {
            before_coefficients,
            coefficient_batches: windowed_coefficient_batches,
            after_coefficients,
        }
    } else {
        VarDctDownstreamCommands::Whole(after_coefficients)
    };
    let lifetime = Arc::new(VarDctJobLifetime {
        output: GpuBufferLease::from_tracked(output.as_ref().clone(), permits.output),
        status_staging,
        status_mapped: AtomicBool::new(false),
        _transient_permits: Mutex::new(vec![permits.transient]),
        _codestream: codestream_buffer,
        _packet_stream_window: packet_stream_window,
        _modular_metadata: Mutex::new(modular_metadata),
        _groups: group_buffers,
        _lf_temporary: lf_temporary,
        _resources: resources,
        _resource_uniforms: resource_uniforms,
        _adaptive_lf_uniform: adaptive_lf_uniform,
        _hf_coefficients: hf_coefficient_buffers,
        _xyb_planes: xyb_planes,
        _restoration: restoration_buffers,
        _resident_scratch: resident_scratch,
        _output_scratch: output_scratch,
    });
    let completion = Arc::new(MapCompletion::default());
    let (submission, downstream) = if let Some(lf_stage_commands) = lf_stage_commands {
        (
            submit_lf_packet_commands(backend.queue(), lf_stage_commands, &lifetime)?,
            Some(downstream_commands),
        )
    } else if let Some(batches) = combined_packet_batches {
        (
            submit_combined_packet_commands(
                backend.queue(),
                batches,
                downstream_commands,
                &lifetime,
            )?,
            None,
        )
    } else {
        (
            submit_vardct_downstream(backend.queue(), Vec::new(), downstream_commands, &lifetime)?,
            None,
        )
    };
    arm_status_map(
        &lifetime,
        &completion,
        if staged_local_trees {
            "VarDCT LF cursor mapping"
        } else {
            "VarDCT validation mapping"
        },
    );
    let poll_completion = Arc::clone(&completion);
    if let Err(error) = poll_permit.register(submission, move |error| {
        poll_completion.complete(Err(error));
    }) {
        completion.complete(Err(format!("VarDCT GPU poll registration failed: {error}")));
    }
    let mut expected_groups = Vec::with_capacity(source.packet.groups.len());
    for group in &source.packet.groups {
        let [group_blocks_x, group_blocks_y] = group.block_extent();
        let expected_blocks = group_blocks_x.checked_mul(group_blocks_y).ok_or(
            VarDctDecodeError::ArithmeticOverflow {
                field: "LF-group validation block count",
            },
        )?;
        let correlation_samples = group
            .rect
            .width
            .div_ceil(64)
            .checked_mul(group.rect.height.div_ceil(64))
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "LF-group validation correlation samples",
            })?;
        expected_groups.push(VarDctGroupValidation {
            uniform_transform: source.packet.uniform_transform,
            expected_lf_samples: expected_blocks.checked_mul(3).ok_or(
                VarDctDecodeError::ArithmeticOverflow {
                    field: "LF-group validation sample count",
                },
            )?,
            expected_coefficients: group.coefficient_words(),
            expected_blocks,
            correlation_samples,
            task_capacity: group.task_capacity,
            expected_global_scale: source.packet.global_scale,
            expected_quant_lf: source.packet.quant_lf,
            expected_extra_precision: group.extra_precision,
        });
    }
    let expected_hf_group_indices = source
        .hf_coefficients
        .iter()
        .flat_map(|plan| &plan.groups)
        .flat_map(HfCoefficientGroupExecutionPlan::global_group_indices)
        .collect();
    let layout = source.layout.clone();
    let frame_name = source.frame_name.clone();
    let stage = if let Some(downstream) = downstream {
        VarDctPendingStage::LocalLf {
            completion,
            source: Box::new(source),
            downstream: Some(downstream),
        }
    } else {
        VarDctPendingStage::Final { completion }
    };
    Ok(VarDctPendingFrame {
        backend: backend.clone(),
        pipelines,
        memory,
        runtime_stats,
        lifetime: Some(lifetime),
        stage,
        token: SubmissionToken(1),
        layout,
        frame_name,
        expected_groups,
        expected_hf_group_indices,
    })
}

fn arm_status_map(
    lifetime: &Arc<VarDctJobLifetime>,
    completion: &Arc<MapCompletion>,
    stage: &'static str,
) {
    let callback_lifetime = Arc::clone(lifetime);
    let callback_completion = Arc::clone(completion);
    lifetime
        .status_staging
        .map_async(wgpu::MapMode::Read, .., move |result| {
            if result.is_ok() {
                callback_lifetime
                    .status_mapped
                    .store(true, Ordering::Release);
            }
            drop(callback_lifetime);
            callback_completion
                .complete(result.map_err(|error| format!("{stage} failed: {error}")));
        });
}

#[derive(Default)]
struct MapCompletion {
    state: Mutex<MapState>,
    condition: Condvar,
}

#[derive(Default)]
struct MapState {
    result: Option<Result<(), String>>,
    waker: Option<Waker>,
}

impl MapCompletion {
    fn complete(&self, result: Result<(), String>) {
        let waker = {
            let mut state = lock_unpoisoned(&self.state);
            if state.result.is_some() {
                return;
            }
            state.result = Some(result);
            state.waker.take()
        };
        self.condition.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn poll(&self, context: &Context<'_>) -> Option<Result<(), String>> {
        let mut state = lock_unpoisoned(&self.state);
        if state.result.is_none() {
            state.waker = Some(context.waker().clone());
        }
        state.result.take()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self) -> Result<(), String> {
        let mut state = lock_unpoisoned(&self.state);
        while state.result.is_none() {
            state = self
                .condition
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state
            .result
            .take()
            .expect("mapping result was checked as present")
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn align4(value: u64) -> Result<u64, VarDctDecodeError> {
    value
        .checked_add(3)
        .map(|value| value & !3)
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "four-byte buffer alignment",
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_gpu_bitstream::StreamSlice;

    fn synthetic_stream_memory(
        fixed_bytes: u64,
        packet_window: bool,
        hf_window: bool,
        stream_limit: u64,
    ) -> AdaptiveStreamMemory {
        let packet_stream_window_bytes = if packet_window { stream_limit } else { 0 };
        let hf_stream_window_bytes = if hf_window { stream_limit } else { 0 };
        AdaptiveStreamMemory {
            total_frame_bytes: fixed_bytes + packet_stream_window_bytes + hf_stream_window_bytes,
            packet_stream_window_bytes,
            hf_stream_window_bytes,
        }
    }

    #[test]
    fn adaptive_stream_limit_uses_the_largest_aligned_cap_that_fits() {
        let decision = select_budget_adaptive_stream_limit(256, 1_120, |stream_limit| {
            Ok(synthetic_stream_memory(1_000, true, true, stream_limit))
        })
        .unwrap();
        assert_eq!(decision, AdaptiveStreamLimitDecision::Selected(60));

        let unchanged = select_budget_adaptive_stream_limit(256, 1_512, |stream_limit| {
            Ok(synthetic_stream_memory(1_000, true, true, stream_limit))
        })
        .unwrap();
        assert_eq!(unchanged, AdaptiveStreamLimitDecision::Selected(256));
    }

    #[test]
    fn adaptive_stream_limit_reports_the_exact_minimum_window_layout() {
        let decision = select_budget_adaptive_stream_limit(256, 1_079, |stream_limit| {
            Ok(synthetic_stream_memory(1_000, true, true, stream_limit))
        })
        .unwrap();
        assert_eq!(
            decision,
            AdaptiveStreamLimitDecision::BudgetTooSmall {
                required_bytes: 1_080,
            }
        );
    }

    #[test]
    fn adaptive_stream_limit_normalizes_the_caller_cap_to_four_bytes() {
        let decision = select_budget_adaptive_stream_limit(255, 2_000, |stream_limit| {
            Ok(synthetic_stream_memory(1_000, true, false, stream_limit))
        })
        .unwrap();
        assert_eq!(decision, AdaptiveStreamLimitDecision::Selected(252));
    }

    #[test]
    fn bounded_stream_upload_crosses_physical_codestream_spans() {
        let bytes: Arc<[u8]> = Arc::from([10, 11, 12, 13, 14, 15, 16]);
        let source = GpuCodestream::from_spans([
            (
                0,
                StreamSlice::from_shared_range(Arc::clone(&bytes), 0..3).unwrap(),
            ),
            (
                3,
                StreamSlice::from_shared_range(Arc::clone(&bytes), 3..5).unwrap(),
            ),
            (5, StreamSlice::from_shared_range(bytes, 5..7).unwrap()),
        ])
        .unwrap();
        let segment = GroupStreamSegment {
            group_index: 0,
            input_start: 2,
            input_end: 7,
            upload_offset: 3,
            window_logical_start: 0,
            window_upload_start: 0,
            available_token_end: 0,
            stream_token_end: 0,
            window_yield_end: 0,
            flags: 0,
        };
        let mut upload = [0u8; 9];
        copy_stream_segment(&source, segment, &mut upload, "test source range").unwrap();
        assert_eq!(upload, [0, 0, 0, 12, 13, 14, 15, 16, 0]);

        assert!(matches!(
            copy_stream_segment(
                &source,
                GroupStreamSegment {
                    input_end: 8,
                    ..segment
                },
                &mut upload,
                "test source range",
            ),
            Err(VarDctDecodeError::EntropyWindowContract {
                detail: "test source range"
            })
        ));
    }

    #[test]
    fn quant_matrix_scales_cover_all_wire_values_and_reject_out_of_range() {
        let expected = [1.5625, 1.25, 1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768];
        for (scale, expected) in expected.into_iter().enumerate() {
            assert_eq!(
                dequant_matrix_multiplier("X", scale as u32).unwrap(),
                expected
            );
        }
        assert!(matches!(
            dequant_matrix_multiplier("B", 8),
            Err(VarDctDecodeError::InvalidQuantMatrixScale {
                channel: "B",
                scale: 8
            })
        ));
    }

    #[test]
    fn restoration_contract_rejects_invalid_epf_iterations() {
        let error = restoration_config(RestorationFilterInventory::Custom {
            gaborish: GaborishInventory::Disabled,
            epf: EdgePreservingFilterInventory::Enabled {
                iterations: 0,
                sharp_lut: None,
                weights: None,
                sigma: None,
                sigma_for_modular: None,
            },
        })
        .unwrap_err();
        assert!(matches!(
            error,
            VarDctDecodeError::InvalidEpfIterations { iterations: 0 }
        ));
    }

    #[test]
    fn restoration_contract_preserves_disabled_and_standard_defaults() {
        let disabled = restoration_config(RestorationFilterInventory::Custom {
            gaborish: GaborishInventory::Disabled,
            epf: EdgePreservingFilterInventory::Disabled,
        })
        .unwrap();
        assert_eq!(disabled, (None, None));

        let (gaborish, epf) = restoration_config(RestorationFilterInventory::Default).unwrap();
        assert_eq!(gaborish, Some(ResidentGaborishWeights::DEFAULT));
        assert_eq!(
            epf,
            Some(VarDctEpfHeader {
                iterations: 2,
                sharp_lut: [
                    0.0,
                    1.0 / 7.0,
                    2.0 / 7.0,
                    3.0 / 7.0,
                    4.0 / 7.0,
                    5.0 / 7.0,
                    6.0 / 7.0,
                    1.0,
                ],
                channel_scale: [40.0, 5.0, 3.5],
                quant_mul: 0.46,
                pass0_sigma_scale: 0.9,
                pass2_sigma_scale: 6.5,
                border_sad_mul: 2.0 / 3.0,
            })
        );
    }

    #[test]
    fn restoration_contract_preserves_custom_gaborish_and_epf_values() {
        let half = jxl_gpu_bitstream::FiniteF16::from_bits(0x3800).unwrap();
        let quarter = jxl_gpu_bitstream::FiniteF16::from_bits(0x3400).unwrap();
        let one = jxl_gpu_bitstream::FiniteF16::from_bits(0x3c00).unwrap();
        let two = jxl_gpu_bitstream::FiniteF16::from_bits(0x4000).unwrap();
        let zero = jxl_gpu_bitstream::FiniteF16::from_bits(0).unwrap();
        let weights = [[half, quarter], [quarter, zero], [zero, half]];
        let (gaborish, epf) = restoration_config(RestorationFilterInventory::Custom {
            gaborish: GaborishInventory::Custom { weights },
            epf: EdgePreservingFilterInventory::Enabled {
                iterations: 3,
                sharp_lut: Some([zero, quarter, half, one, zero, quarter, half, one]),
                weights: Some(jxl_gpu_bitstream::EpfWeightsInventory {
                    channel_scale: [one, half, quarter],
                    pass1_zeroflush: half,
                    pass2_zeroflush: quarter,
                }),
                sigma: Some(jxl_gpu_bitstream::EpfSigmaInventory {
                    quant_mul: Some(one),
                    pass0_sigma_scale: half,
                    pass2_sigma_scale: quarter,
                    border_sad_mul: two,
                }),
                sigma_for_modular: None,
            },
        })
        .unwrap();
        assert_eq!(
            gaborish,
            Some(ResidentGaborishWeights {
                x: [0.5, 0.25],
                y: [0.25, 0.0],
                b: [0.0, 0.5],
            })
        );
        assert_eq!(
            epf,
            Some(VarDctEpfHeader {
                iterations: 3,
                sharp_lut: [0.0, 0.25, 0.5, 1.0, 0.0, 0.25, 0.5, 1.0],
                channel_scale: [1.0, 0.5, 0.25],
                quant_mul: 1.0,
                pass0_sigma_scale: 0.5,
                pass2_sigma_scale: 0.25,
                border_sad_mul: 2.0,
            })
        );
    }
}
