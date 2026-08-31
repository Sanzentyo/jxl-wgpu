//! Runtime-neutral GPU submission engine for the bounded standard VarDCT profile.
//!
//! The accepted codestream profile is intentionally bounded and authoritative: one still XYB
//! frame, either one image-sized regular zero-AC transform or one LF group of tiled DCT8 tasks,
//! stream-provided quantization, GPU-decoded LF/HF metadata, and GPU-decoded single-pass DCT8 AC
//! coefficients. No pixel, coefficient, transform, quantization, residual, or entropy fallback
//! runs on the CPU.

use std::collections::BTreeMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use jxl_gpu_bitstream::{
    CodestreamInventory, ColourEncodingInventory, ColourSpaceInventory,
    EdgePreservingFilterInventory, GaborishInventory, InventoryLimits, ParseLimits,
    PrimariesInventory, RenderingIntentInventory, RestorationFilterInventory,
    TransferFunctionInventory, WhitePointInventory,
};
use jxl_gpu_formats::{
    ChromaLocation2d, ColorRange, ColorSpace, ColorSpec, ColorSpecification, ImageLayout,
    LayoutError, PixelFormat, RgbChannelOrder, TransferFunction, YcbcrEncoding,
};
use jxl_gpu_protocol::{
    ChangedRegions, Extent2d, OutputId, Region, SubmissionToken, TransformKind,
};
use jxl_wgpu::{
    GpuBufferLease, GpuImageFrame, GpuImageOutput, KernelVariant, MemoryBudget,
    MemoryBudgetSnapshot, MemoryPermit, ResidentStorageBinding, ResidentVarDctError,
    ResidentVarDctInputs, ResidentVarDctMemoryPlan, ResidentVarDctOutputPlane,
    ResidentVarDctRenderConfig, ResidentVarDctRenderer, ResidentVarDctScratch,
    SubmissionPollPermit, UnvalidatedGpuImageFrame, UnvalidatedGpuImageOutput, WgpuBackend,
};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::vardct_artifact::{
    GpuVarDctArtifactStatus, GpuVarDctLoweringError, HfDispatchStage, HfMetadataArtifactConfig,
    HfMetadataLoweringBuffers, HfMetadataLoweringParams, HfMetadataLoweringPipeline,
    VAR_DCT_STRATEGY_COUNT, VarDctArtifactDeviceLimits, VarDctArtifactError, VarDctArtifactLayout,
};
use crate::vardct_lf::{AdaptiveLfBuffers, AdaptiveLfParams, AdaptiveLfPipeline};
use crate::vardct_output::{
    VarDctInverseOpsin, VarDctOutputConfig, VarDctOutputError, VarDctOutputInputs,
    VarDctOutputPacker, VarDctOutputPlan, VarDctOutputPlane, VarDctOutputScratch,
};
use crate::vardct_packet::{
    BoundedVarDctPacketError, BoundedVarDctPacketPlan, GpuVarDctPacketError, GpuVarDctPacketStatus,
    VarDctModularParams, VarDctPacketBuffers, VarDctPacketControl, VarDctPacketPipeline,
};
use crate::vardct_pass_group::{
    GpuHfCoefficientError, GpuHfCoefficientStatus, HfCoefficientBuffers,
    HfCoefficientExecutionPlan, HfCoefficientPipeline, HfCoefficientPlanError,
};
use crate::vardct_resource::{
    VarDctResourceBuffers, VarDctResourceError, VarDctResourceLayout, VarDctResourceParams,
    VarDctResourcePipeline,
};
use crate::{
    AnimationMetadata, DecodeProfile, Error as DecodeError, FrameDuration, FrameMetadata,
    GpuCodestream, GpuOutputMapping, GpuOutputRequest, GpuPendingFrame, GpuSubmissionEngine,
    GpuSubmissionSession, PreparedGpuSession, Result as DecodeResult, SubmittedGpuFrame,
};

const PACKET_STATUS_BYTES: u64 = std::mem::size_of::<GpuVarDctPacketStatus>() as u64;
const ARTIFACT_STATUS_BYTES: u64 = std::mem::size_of::<GpuVarDctArtifactStatus>() as u64;
const BASE_VALIDATION_STAGING_BYTES: u64 = PACKET_STATUS_BYTES + ARTIFACT_STATUS_BYTES;
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
    #[error("the bounded VarDCT engine requires disabled Gaborish and EPF restoration")]
    UnsupportedRestoration,
    #[error("the bounded VarDCT engine requires frame quant-matrix scales X=3 and B=2")]
    UnsupportedQuantMatrixScale,
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
    Output(#[from] VarDctOutputError),
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error("VarDCT GPU memory arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
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
    #[error("VarDCT GPU completion was consumed more than once")]
    CompletionConsumed,
    #[error("VarDCT kernel '{kernel}' configuration failed: {message}")]
    KernelPolicy {
        kernel: &'static str,
        message: String,
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
    pub codestream_bytes: u64,
    pub modular_metadata_bytes: u64,
    pub reconstructed_bytes: u64,
    pub raw_metadata_bytes: u64,
    pub coefficient_bytes: u64,
    pub packet_status_bytes: u64,
    pub validation_staging_bytes: u64,
    pub packet_control_bytes: u64,
    pub modular_params_bytes: u64,
    pub lf_temporary_bytes: u64,
    pub resource_bytes: u64,
    pub resource_uniform_bytes: u64,
    pub adaptive_lf_uniform_bytes: u64,
    pub artifact_bytes: u64,
    pub occupancy_bytes: u64,
    pub artifact_uniform_bytes: u64,
    pub hf_entropy_bundle_bytes: u64,
    pub hf_params_bytes: u64,
    pub hf_lz77_scratch_bytes: u64,
    pub hf_status_bytes: u64,
    pub hf_order_table_bytes: u64,
    pub hf_sink_uniform_bytes: u64,
    pub xyb_plane_bytes: u64,
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
            codestream_len,
            packet,
            control,
            resource,
            artifact,
            hf_coefficients,
            resident,
            output,
        } = inputs;
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
        let modular_metadata_bytes = checked_words(
            u64::try_from(packet.modular_metadata.len()).map_err(|_| {
                VarDctDecodeError::ArithmeticOverflow {
                    field: "Modular metadata length",
                }
            })?,
            "Modular metadata bytes",
        )?;
        let reconstructed_bytes = checked_words(
            u64::from(packet.reconstructed_words()?),
            "LF reconstruction bytes",
        )?;
        let raw_metadata_bytes =
            checked_words(u64::from(control.capacities[1]), "raw HF metadata bytes")?;
        let coefficient_bytes =
            checked_words(u64::from(packet.coefficient_words()), "coefficient bytes")?;
        let packet_status_bytes = PACKET_STATUS_BYTES;
        let hf_entropy_bundle_bytes = hf_coefficients
            .map(|plan| checked_words(plan.entropy_words.len() as u64, "HF entropy bundle bytes"))
            .transpose()?
            .unwrap_or(0);
        let hf_params_bytes = hf_coefficients
            .map(|plan| {
                (plan.params.len() as u64)
                    .checked_mul(std::mem::size_of::<
                        crate::vardct_pass_group::HfCoefficientPassParams,
                    >() as u64)
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "HF parameter bytes",
                    })
            })
            .transpose()?
            .unwrap_or(0);
        let hf_lz77_scratch_bytes =
            hf_coefficients.map_or(0, HfCoefficientExecutionPlan::lz77_scratch_bytes);
        let hf_status_bytes = hf_coefficients.map_or(0, HfCoefficientExecutionPlan::status_bytes);
        let hf_order_table_bytes = hf_coefficients
            .map(|plan| checked_words(plan.order_words.len() as u64, "HF order-table bytes"))
            .transpose()?
            .unwrap_or(0);
        let hf_sink_uniform_bytes = hf_coefficients
            .map(|_| std::mem::size_of::<crate::vardct_artifact::HfCoefficientSinkParams>() as u64)
            .unwrap_or(0);
        let validation_staging_bytes = BASE_VALIDATION_STAGING_BYTES
            .checked_add(hf_status_bytes)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "VarDCT validation staging bytes",
        })?;
        let packet_control_bytes = std::mem::size_of::<VarDctPacketControl>() as u64;
        let modular_params_bytes = std::mem::size_of::<VarDctModularParams>() as u64;
        let lf_temporary_bytes = u64::from(resource.block_count).checked_mul(16).ok_or(
            VarDctDecodeError::ArithmeticOverflow {
                field: "LF temporary bytes",
            },
        )?;
        let resource_bytes = resource.bytes();
        let resource_uniform_bytes = std::mem::size_of::<VarDctResourceParams>() as u64;
        let adaptive_lf_uniform_bytes = std::mem::size_of::<AdaptiveLfParams>() as u64;
        let artifact_bytes = artifact.artifact_bytes;
        let occupancy_bytes = artifact.occupancy_bytes;
        let artifact_uniform_bytes = std::mem::size_of::<HfMetadataLoweringParams>() as u64;
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
        let resident_transient_bytes = resident.total_bytes;
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
            lf_temporary_bytes,
            resource_bytes,
            resource_uniform_bytes,
            adaptive_lf_uniform_bytes,
            artifact_bytes,
            occupancy_bytes,
            artifact_uniform_bytes,
            hf_entropy_bundle_bytes,
            hf_params_bytes,
            hf_lz77_scratch_bytes,
            hf_status_bytes,
            hf_order_table_bytes,
            hf_sink_uniform_bytes,
            xyb_plane_bytes,
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
            codestream_bytes,
            modular_metadata_bytes,
            reconstructed_bytes,
            raw_metadata_bytes,
            coefficient_bytes,
            packet_status_bytes,
            validation_staging_bytes,
            packet_control_bytes,
            modular_params_bytes,
            lf_temporary_bytes,
            resource_bytes,
            resource_uniform_bytes,
            adaptive_lf_uniform_bytes,
            artifact_bytes,
            occupancy_bytes,
            artifact_uniform_bytes,
            hf_entropy_bundle_bytes,
            hf_params_bytes,
            hf_lz77_scratch_bytes,
            hf_status_bytes,
            hf_order_table_bytes,
            hf_sink_uniform_bytes,
            xyb_plane_bytes,
            resident_transient_bytes,
            output_uniform_bytes,
            output_lease_bytes,
            transient_bytes,
            total_frame_bytes,
        })
    }
}

struct VarDctDecodeMemoryInputs<'a> {
    codestream_len: usize,
    packet: &'a BoundedVarDctPacketPlan,
    control: VarDctPacketControl,
    resource: VarDctResourceLayout,
    artifact: VarDctArtifactLayout,
    hf_coefficients: Option<&'a HfCoefficientExecutionPlan>,
    resident: ResidentVarDctMemoryPlan,
    output: VarDctOutputPlan,
}

struct VarDctPipelines {
    packet: VarDctPacketPipeline,
    resource: VarDctResourcePipeline,
    adaptive_lf: AdaptiveLfPipeline,
    artifact: HfMetadataLoweringPipeline,
    hf_coefficients: HfCoefficientPipeline,
    renderer: ResidentVarDctRenderer,
    output: VarDctOutputPacker,
    output_variant: KernelVariant,
}

impl VarDctPipelines {
    fn new(backend: &WgpuBackend) -> Result<Self, VarDctDecodeError> {
        let resource_variant =
            resolve_kernel_variant(backend, "vardct_resource", KernelVariant::Lanes64)?;
        let output_variant =
            resolve_kernel_variant(backend, "vardct_output", KernelVariant::Lanes256)?;
        let device = backend.device();
        Ok(Self {
            packet: VarDctPacketPipeline::new(device),
            resource: VarDctResourcePipeline::with_variant(device, resource_variant)?,
            adaptive_lf: AdaptiveLfPipeline::new(device),
            artifact: HfMetadataLoweringPipeline::new(device),
            hf_coefficients: HfCoefficientPipeline::new(device),
            renderer: ResidentVarDctRenderer::new(device),
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
        })
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
        let source = prepare_source(
            &self.backend,
            codestream,
            request,
            inventory,
            self.pipelines.output_variant,
        )?;
        let extent = source.layout.extent;
        let profile = DecodeProfile::VarDctRegular {
            bits_per_sample: 8,
            transform: source.packet.transform,
        };
        Ok(PreparedGpuSession::new(
            profile,
            AnimationMetadata::still(extent),
            VarDctDecodeSession {
                backend: self.backend.clone(),
                pipelines: Arc::clone(&self.pipelines),
                memory_stats: source.memory,
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
    codestream_storage: Arc<[u8]>,
    codestream_range: Range<usize>,
    packet: BoundedVarDctPacketPlan,
    control: VarDctPacketControl,
    resource_layout: VarDctResourceLayout,
    resource_params: VarDctResourceParams,
    artifact_layout: VarDctArtifactLayout,
    artifact_params: HfMetadataLoweringParams,
    hf_coefficients: Option<HfCoefficientExecutionPlan>,
    output_plan: VarDctOutputPlan,
    layout: ImageLayout,
    inverse_opsin: VarDctInverseOpsin,
    quant_biases: [f32; 4],
    frame_name: String,
    transform_index: usize,
    memory: VarDctDecodeMemoryStats,
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

    fn open(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
    ) -> DecodeResult<PreparedGpuSession<Self::Session>> {
        let parsed = jxl_gpu_bitstream::parse(codestream.bytes(), self.parse_limits())?;
        let inventory = parsed.codestream_inventory(InventoryLimits {
            max_frames: 1,
            max_total_section_bytes: u64::try_from(codestream.bytes().len())
                .map_err(|_| DecodeError::backend("VarDCT codestream size exceeds u64"))?,
            ..InventoryLimits::default()
        })?;
        self.open_with_inventory(codestream, request, &inventory)
    }
}

fn prepare_source(
    backend: &WgpuBackend,
    codestream: GpuCodestream,
    request: &GpuOutputRequest,
    inventory: &jxl_gpu_bitstream::CodestreamInventory,
    output_variant: KernelVariant,
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
            rendering_intent: RenderingIntentInventory::Relative,
        }
    ) {
        return Err(VarDctDecodeError::UnsupportedColorEncoding);
    }
    let frame = inventory
        .frames
        .first()
        .ok_or(VarDctDecodeError::MissingFrame)?;
    if !matches!(
        frame.restoration_filter,
        RestorationFilterInventory::Custom {
            gaborish: GaborishInventory::Disabled,
            epf: EdgePreservingFilterInventory::Disabled,
        }
    ) {
        return Err(VarDctDecodeError::UnsupportedRestoration);
    }
    if frame.x_qm_scale != 3 || frame.b_qm_scale != 2 {
        return Err(VarDctDecodeError::UnsupportedQuantMatrixScale);
    }
    let packet = BoundedVarDctPacketPlan::parse(codestream.bytes(), inventory)?;
    let control = packet.packet_control()?;
    let [blocks_x, blocks_y] = packet.block_extent();
    let transform_extent = packet.transform.pixel_extent();
    let transform_area = transform_extent
        .width
        .checked_mul(transform_extent.height)
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "transform area",
        })?;
    let resource_layout =
        VarDctResourceLayout::new(blocks_x, blocks_y, transform_area, packet.task_count)?;
    let resource_params =
        VarDctResourceParams::new(blocks_x, blocks_y, packet.global_scale, packet.quant_lf)?;
    let transform_index = TransformKind::ALL
        .iter()
        .position(|&candidate| candidate == packet.transform)
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "regular transform index",
        })?;
    let matrix_offsets = [resource_layout.matrix_offset; VAR_DCT_STRATEGY_COUNT];
    let correlation_width = packet.profile.width.div_ceil(64);
    let correlation_height = packet.profile.height.div_ceil(64);
    let pass_group_dim_blocks = packet.profile.group_dimension.checked_div(8).ok_or(
        VarDctDecodeError::ArithmeticOverflow {
            field: "pass-group block dimension",
        },
    )?;
    let artifact_config = HfMetadataArtifactConfig {
        blocks_width: blocks_x,
        blocks_height: blocks_y,
        block_info_entries: packet.task_count,
        strategy_offset_words: control.offsets[2],
        hf_mul_offset_words: control.offsets[3],
        raw_metadata_words: u64::from(control.capacities[1]),
        pass_group_dim_blocks,
        lf_stride: blocks_x,
        correlation_width,
        correlation_height,
        destination_origin: [0, 0],
        afv_basis_offset: resource_layout.matrix_offset,
        quant_offset: resource_layout.quant_offset,
        global_scale: packet.global_scale,
        matrix_offsets,
    };
    let artifact_layout = VarDctArtifactLayout::plan(
        &artifact_config,
        VarDctArtifactDeviceLimits::from_wgpu(&backend.device().limits()),
    )?;
    let artifact_params = HfMetadataLoweringParams::new(&artifact_config, artifact_layout)?;
    let hf_coefficients = packet
        .hf_coefficients
        .as_ref()
        .map(|entropy| HfCoefficientExecutionPlan::new(&packet, entropy, artifact_layout))
        .transpose()?;
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
    let scratch_scalars = transform_area
        .checked_mul(packet.task_count)
        .and_then(|area| area.checked_mul(3))
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "resident transform scratch scalars",
        })?;
    let resident_memory = ResidentVarDctMemoryPlan::new(scratch_scalars)?;
    let memory = VarDctDecodeMemoryStats::plan(VarDctDecodeMemoryInputs {
        codestream_len: codestream.bytes().len(),
        packet: &packet,
        control,
        resource: resource_layout,
        artifact: artifact_layout,
        hf_coefficients: hf_coefficients.as_ref(),
        resident: resident_memory,
        output: output_plan,
    })?;
    validate_device_limits(backend.device(), memory)?;
    let frame_name = packet.profile.frame_name.clone();
    Ok(VarDctSource {
        codestream_storage: codestream.shared_storage(),
        codestream_range: codestream.storage_range(),
        packet,
        control,
        resource_layout,
        resource_params,
        artifact_layout,
        artifact_params,
        hf_coefficients,
        output_plan,
        layout,
        inverse_opsin,
        quant_biases,
        frame_name,
        transform_index,
        memory,
    })
}

fn validate_device_limits(
    device: &wgpu::Device,
    memory: VarDctDecodeMemoryStats,
) -> Result<(), VarDctDecodeError> {
    let limits = device.limits();
    for (resource, required, storage) in [
        ("codestream upload", memory.codestream_bytes, true),
        ("Modular metadata", memory.modular_metadata_bytes, true),
        ("LF reconstruction", memory.reconstructed_bytes, true),
        ("raw HF metadata", memory.raw_metadata_bytes, true),
        ("coefficients", memory.coefficient_bytes, true),
        ("packet status", memory.packet_status_bytes, true),
        ("validation staging", memory.validation_staging_bytes, false),
        ("LF temporary", memory.lf_temporary_bytes, true),
        ("VarDCT resources", memory.resource_bytes, true),
        ("VarDCT artifact", memory.artifact_bytes, true),
        ("artifact occupancy", memory.occupancy_bytes, true),
        ("HF entropy bundle", memory.hf_entropy_bundle_bytes, true),
        ("HF pass-group parameters", memory.hf_params_bytes, true),
        ("HF LZ77 scratch", memory.hf_lz77_scratch_bytes, true),
        ("HF pass-group status", memory.hf_status_bytes, true),
        (
            "HF coefficient order table",
            memory.hf_order_table_bytes,
            true,
        ),
        ("one XYB plane", memory.xyb_plane_bytes / 3, true),
        ("packed RGB8 output", memory.output_lease_bytes, true),
    ] {
        check_limit(resource, required, limits.max_buffer_size)?;
        if storage {
            check_limit(resource, required, limits.max_storage_buffer_binding_size)?;
        }
    }
    for (resource, required) in [
        ("packet control uniform", memory.packet_control_bytes),
        ("LF resource uniform", memory.resource_uniform_bytes),
        ("adaptive LF uniform", memory.adaptive_lf_uniform_bytes),
        ("artifact uniform", memory.artifact_uniform_bytes),
        ("HF coefficient sink uniform", memory.hf_sink_uniform_bytes),
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
    source: Option<VarDctSource>,
    memory: MemoryBudget,
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

    /// All VarDCT compute phases and validation copies are recorded into one queue submission.
    #[must_use]
    pub const fn submissions_per_frame(&self) -> usize {
        1
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
        let pending = submit_vardct(
            &self.backend,
            &self.pipelines,
            source,
            VarDctMemoryPermits {
                output: output_permit,
                transient: transient_permit,
            },
            poll_permit,
        )?;
        self.source = None;
        Ok(Some(pending))
    }
}

struct VarDctMemoryPermits {
    output: MemoryPermit,
    transient: MemoryPermit,
}

struct HfCoefficientJobBuffers {
    entropy_bundle: wgpu::Buffer,
    lz77_scratch: wgpu::Buffer,
    params: wgpu::Buffer,
    status: wgpu::Buffer,
    order_table: wgpu::Buffer,
    sink_params: wgpu::Buffer,
}

struct VarDctJobLifetime {
    output: GpuBufferLease,
    status_staging: wgpu::Buffer,
    status_mapped: AtomicBool,
    _transient_permit: MemoryPermit,
    _codestream: wgpu::Buffer,
    _modular_metadata: wgpu::Buffer,
    _reconstructed: wgpu::Buffer,
    _raw_metadata: wgpu::Buffer,
    _coefficients: wgpu::Buffer,
    _packet_status: wgpu::Buffer,
    _packet_control: wgpu::Buffer,
    _modular_params: wgpu::Buffer,
    _lf_temporary: wgpu::Buffer,
    _resources: wgpu::Buffer,
    _resource_uniform: wgpu::Buffer,
    _adaptive_lf_uniform: wgpu::Buffer,
    _artifact: wgpu::Buffer,
    _occupancy: wgpu::Buffer,
    _artifact_uniform: wgpu::Buffer,
    _hf_coefficients: Option<HfCoefficientJobBuffers>,
    _xyb_planes: [wgpu::Buffer; 3],
    _resident_scratch: ResidentVarDctScratch,
    _output_scratch: VarDctOutputScratch,
}

impl Drop for VarDctJobLifetime {
    fn drop(&mut self) {
        if self.status_mapped.swap(false, Ordering::AcqRel) {
            self.status_staging.unmap();
        }
    }
}

/// Submitted VarDCT frame awaiting one mapped packet/artifact validation record.
pub struct VarDctPendingFrame {
    device: wgpu::Device,
    lifetime: Option<Arc<VarDctJobLifetime>>,
    completion: Arc<MapCompletion>,
    token: SubmissionToken,
    layout: ImageLayout,
    transform: TransformKind,
    frame_name: String,
    expected_lf_samples: u32,
    expected_hf_samples: u32,
    expected_coefficients: u32,
    expected_blocks: u32,
    expected_tasks: u32,
    expected_hf_groups: u32,
    expected_global_scale: u32,
    expected_quant_lf: u32,
}

impl std::fmt::Debug for VarDctPendingFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VarDctPendingFrame")
            .field("token", &self.token)
            .field("layout", &self.layout)
            .field("transform", &self.transform)
            .finish_non_exhaustive()
    }
}

impl VarDctPendingFrame {
    /// Same-queue, budget-tracked access before packet/artifact status becomes authoritative.
    pub fn unvalidated_gpu_frame(&self) -> DecodeResult<UnvalidatedGpuImageFrame> {
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
        let packet: GpuVarDctPacketStatus = mapped
            .get(..PACKET_STATUS_BYTES as usize)
            .and_then(|bytes| bytemuck::try_pod_read_unaligned(bytes).ok())
            .ok_or(VarDctDecodeError::StatusAbi { status: "packet" })?;
        let artifact: GpuVarDctArtifactStatus = mapped
            .get(PACKET_STATUS_BYTES as usize..BASE_VALIDATION_STAGING_BYTES as usize)
            .and_then(|bytes| bytemuck::try_pod_read_unaligned(bytes).ok())
            .ok_or(VarDctDecodeError::StatusAbi { status: "artifact" })?;
        packet
            .validate(
                self.transform,
                self.expected_lf_samples,
                self.expected_hf_samples,
                self.expected_global_scale,
                self.expected_quant_lf,
            )
            .map_err(VarDctDecodeError::from)?;
        if packet.coefficient_words != self.expected_coefficients {
            return Err(VarDctDecodeError::ArtifactStatus {
                field: "packet coefficient_words",
                expected: self.expected_coefficients,
                actual: packet.coefficient_words,
            }
            .into());
        }
        artifact.validate().map_err(VarDctDecodeError::from)?;
        let hf_status_bytes = mapped.get(BASE_VALIDATION_STAGING_BYTES as usize..).ok_or(
            VarDctDecodeError::StatusAbi {
                status: "HF coefficient",
            },
        )?;
        let hf_statuses = bytemuck::try_cast_slice::<u8, GpuHfCoefficientStatus>(hf_status_bytes)
            .map_err(|_| VarDctDecodeError::StatusAbi {
            status: "HF coefficient",
        })?;
        if hf_statuses.len() != self.expected_hf_groups as usize {
            return Err(VarDctDecodeError::StatusAbi {
                status: "HF coefficient count",
            }
            .into());
        }
        for (group, status) in hf_statuses.iter().copied().enumerate() {
            status
                .validate(group as u32)
                .map_err(VarDctDecodeError::from)?;
        }
        drop(mapped);
        for (field, expected, actual) in [
            ("task_count", self.expected_tasks, artifact.task_count),
            (
                "coefficient_words",
                self.expected_coefficients,
                artifact.coefficient_words,
            ),
            (
                "covered_blocks",
                self.expected_blocks,
                artifact.covered_blocks,
            ),
            (
                "consumed_block_info_entries",
                self.expected_tasks,
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
        let mapping = self.completion.wait();
        self.finish(mapping)
    }

    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<DecodeResult<SubmittedGpuFrame<Self::Frame>>> {
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Poll::Ready(Err(DecodeError::backend(error)));
        }
        match self.completion.poll(context) {
            Some(mapping) => Poll::Ready(self.finish(mapping)),
            None => Poll::Pending,
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
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Poll::Ready(Err(DecodeError::backend(error)));
        }
        match self.completion.poll(context) {
            Some(mapping) => Poll::Ready(self.finish(mapping)),
            None => Poll::Pending,
        }
    }
}

fn resident_binding(
    buffer: &wgpu::Buffer,
) -> Result<ResidentStorageBinding<'_>, VarDctDecodeError> {
    Ok(ResidentStorageBinding::entire(buffer)?)
}

fn submit_vardct(
    backend: &WgpuBackend,
    pipelines: &VarDctPipelines,
    source: &VarDctSource,
    permits: VarDctMemoryPermits,
    poll_permit: SubmissionPollPermit,
) -> Result<VarDctPendingFrame, VarDctDecodeError> {
    let device = backend.device();
    let codestream = source
        .codestream_storage
        .get(source.codestream_range.clone())
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "codestream storage range",
        })?;
    let upload_bytes = usize::try_from(source.memory.codestream_bytes).map_err(|_| {
        VarDctDecodeError::ArithmeticOverflow {
            field: "codestream upload host length",
        }
    })?;
    let mut upload = Vec::with_capacity(upload_bytes);
    upload.extend_from_slice(codestream);
    upload.resize(upload_bytes, 0);
    let codestream_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu VarDCT codestream"),
        contents: &upload,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let modular_metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu VarDCT Modular metadata"),
        contents: bytemuck::cast_slice(&source.packet.modular_metadata),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let storage = |label: &'static str, size: u64, extra: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | extra,
            mapped_at_creation: false,
        })
    };
    let reconstructed = storage(
        "jxl-wgpu VarDCT LF reconstruction",
        source.memory.reconstructed_bytes,
        wgpu::BufferUsages::COPY_DST,
    );
    let raw_metadata = storage(
        "jxl-wgpu VarDCT raw HF metadata",
        source.memory.raw_metadata_bytes,
        wgpu::BufferUsages::COPY_DST,
    );
    let coefficients = storage(
        "jxl-wgpu VarDCT coefficients",
        source.memory.coefficient_bytes,
        wgpu::BufferUsages::COPY_DST,
    );
    let packet_status = storage(
        "jxl-wgpu VarDCT packet status",
        PACKET_STATUS_BYTES,
        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let packet_control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu VarDCT packet control"),
        contents: bytemuck::bytes_of(&source.control),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let params = VarDctModularParams::default()
        .with_lz77_window(source.packet.lz77_window_words)
        .with_self_correcting(source.packet.needs_self_correcting);
    let modular_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu VarDCT Modular params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let lf_temporary = storage(
        "jxl-wgpu VarDCT dequantized LF temporary",
        source.memory.lf_temporary_bytes,
        wgpu::BufferUsages::COPY_DST,
    );
    let resource_values = source.resource_layout.initial_values();
    let resources = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu VarDCT resource vectors"),
        contents: bytemuck::cast_slice(&resource_values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let artifact = storage(
        "jxl-wgpu VarDCT resident artifact",
        source.artifact_layout.artifact_bytes,
        wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let occupancy = storage(
        "jxl-wgpu VarDCT artifact occupancy",
        source.artifact_layout.occupancy_bytes,
        wgpu::BufferUsages::COPY_DST,
    );
    let artifact_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu VarDCT artifact params"),
        contents: bytemuck::bytes_of(&source.artifact_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let hf_coefficient_buffers =
        source
            .hf_coefficients
            .as_ref()
            .map(|plan| HfCoefficientJobBuffers {
                entropy_bundle: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jxl-wgpu HF entropy bundle"),
                    contents: bytemuck::cast_slice(&plan.entropy_words),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                lz77_scratch: storage(
                    "jxl-wgpu HF LZ77 scratch",
                    plan.lz77_scratch_bytes(),
                    wgpu::BufferUsages::COPY_DST,
                ),
                params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jxl-wgpu HF pass-group params"),
                    contents: bytemuck::cast_slice(&plan.params),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                status: storage(
                    "jxl-wgpu HF pass-group status",
                    plan.status_bytes(),
                    wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                ),
                order_table: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jxl-wgpu HF natural-order table"),
                    contents: bytemuck::cast_slice(&plan.order_words),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                sink_params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jxl-wgpu HF coefficient sink params"),
                    contents: bytemuck::bytes_of(&plan.sink_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                }),
            });
    let plane_bytes = source.memory.xyb_plane_bytes / 3;
    let xyb_planes = [
        "jxl-wgpu VarDCT X plane",
        "jxl-wgpu VarDCT Y plane",
        "jxl-wgpu VarDCT B plane",
    ]
    .map(|label| storage(label, plane_bytes, wgpu::BufferUsages::COPY_DST));
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

    let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("jxl-wgpu bounded VarDCT decode"),
    });
    for buffer in [
        &reconstructed,
        &raw_metadata,
        &coefficients,
        &packet_status,
        &lf_temporary,
        &artifact,
        &occupancy,
        &xyb_planes[0],
        &xyb_planes[1],
        &xyb_planes[2],
        output.as_ref(),
    ] {
        commands.clear_buffer(buffer, 0, None);
    }
    if let Some(buffers) = &hf_coefficient_buffers {
        commands.clear_buffer(&buffers.lz77_scratch, 0, None);
        commands.clear_buffer(&buffers.status, 0, None);
    }
    pipelines.packet.encode(
        device,
        &mut commands,
        VarDctPacketBuffers {
            codestream: &codestream_buffer,
            modular_metadata: &modular_metadata,
            reconstructed_lf: &reconstructed,
            raw_hf_metadata: &raw_metadata,
            coefficients: &coefficients,
            status: &packet_status,
            control: &packet_control,
            modular_params: &modular_params,
        },
    );
    let resource_uniform = pipelines.resource.encode(
        device,
        &mut commands,
        VarDctResourceBuffers {
            quantized_lf: &reconstructed,
            dequantized_lf: &lf_temporary,
        },
        source.resource_params,
    );
    let adaptive_lf_uniform = pipelines.adaptive_lf.encode(
        device,
        &mut commands,
        AdaptiveLfBuffers {
            input: &lf_temporary,
            output: &resources,
        },
        AdaptiveLfParams::new(
            source.control.geometry[2],
            source.control.geometry[3],
            0,
            source.resource_layout.lf_offset,
            source.resource_params.smoothing_thresholds(),
        ),
    );
    pipelines.artifact.encode(
        device,
        &mut commands,
        HfMetadataLoweringBuffers {
            raw_metadata: &raw_metadata,
            artifact: &artifact,
            occupancy: &occupancy,
            resources: &resources,
            params: &artifact_uniform,
        },
    );
    if let (Some(plan), Some(buffers)) = (
        source.hf_coefficients.as_ref(),
        hf_coefficient_buffers.as_ref(),
    ) {
        pipelines.hf_coefficients.encode(
            device,
            &mut commands,
            HfCoefficientBuffers {
                codestream: &codestream_buffer,
                entropy_bundle: &buffers.entropy_bundle,
                lz77_scratch: &buffers.lz77_scratch,
                params: &buffers.params,
                status: &buffers.status,
                artifact: &artifact,
                order_table: &buffers.order_table,
                coefficients: &coefficients,
                sink_params: &buffers.sink_params,
            },
            u32::try_from(plan.params.len()).map_err(|_| {
                VarDctDecodeError::ArithmeticOverflow {
                    field: "HF pass-group dispatch count",
                }
            })?,
        );
    }

    let (tasks_offset, tasks_size) = source.artifact_layout.task_binding();
    let tasks = ResidentStorageBinding {
        buffer: &artifact,
        offset: tasks_offset,
        size: NonZeroU64::new(tasks_size)
            .ok_or(ResidentVarDctError::EmptyBinding { role: "task" })?,
    };
    let extent = source.packet.transform.pixel_extent();
    let blocks = source.resource_layout.block_count;
    let scratch_scalars = extent
        .width
        .checked_mul(extent.height)
        .and_then(|area| area.checked_mul(source.packet.task_count))
        .and_then(|area| area.checked_mul(3))
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "resident scratch scalars",
        })?;
    let indirect_offsets = [
        source
            .artifact_layout
            .indirect_offset(source.transform_index, HfDispatchStage::Dequantize)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "dequantize indirect offset",
            })?,
        source
            .artifact_layout
            .indirect_offset(source.transform_index, HfDispatchStage::Horizontal)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "horizontal indirect offset",
            })?,
        source
            .artifact_layout
            .indirect_offset(source.transform_index, HfDispatchStage::Vertical)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "vertical indirect offset",
            })?,
    ];
    let padded_width =
        source.control.geometry[2]
            .checked_mul(8)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "padded output width",
            })?;
    let padded_height =
        source.control.geometry[3]
            .checked_mul(8)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "padded output height",
            })?;
    let resident_outputs = xyb_planes.each_ref().map(|plane| {
        resident_binding(plane).map(|storage| ResidentVarDctOutputPlane {
            storage,
            width: padded_width,
            height: padded_height,
            stride: padded_width,
        })
    });
    let [output_x, output_y, output_b] = resident_outputs;
    let resident_scratch = pipelines.renderer.encode(
        device,
        &mut commands,
        ResidentVarDctInputs {
            coefficients: resident_binding(&coefficients)?,
            tasks,
            resources: resident_binding(&resources)?,
            outputs: [output_x?, output_y?, output_b?],
            indirect: &artifact,
            indirect_offsets,
            config: ResidentVarDctRenderConfig {
                transform: source.packet.transform,
                task_base: 0,
                task_capacity: source.packet.task_count,
                scratch_scalars,
                quant_offset: source.resource_layout.quant_offset,
                correlation_offset: source.resource_layout.correlation_offset,
                lf_offset: source.resource_layout.lf_offset,
                correlation_width: source.packet.profile.width.div_ceil(64),
                correlation_height: source.packet.profile.height.div_ceil(64),
                quant_biases: source.quant_biases,
            },
        },
    )?;
    let output_scratch = pipelines.output.encode(
        device,
        &mut commands,
        VarDctOutputInputs {
            planes: [
                VarDctOutputPlane {
                    storage: resident_binding(&xyb_planes[0])?,
                    stride: padded_width,
                },
                VarDctOutputPlane {
                    storage: resident_binding(&xyb_planes[1])?,
                    stride: padded_width,
                },
                VarDctOutputPlane {
                    storage: resident_binding(&xyb_planes[2])?,
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
    commands.copy_buffer_to_buffer(&packet_status, 0, &status_staging, 0, PACKET_STATUS_BYTES);
    commands.copy_buffer_to_buffer(
        &artifact,
        u64::from(source.artifact_layout.status_offset_words) * 4,
        &status_staging,
        PACKET_STATUS_BYTES,
        ARTIFACT_STATUS_BYTES,
    );
    if let Some(buffers) = &hf_coefficient_buffers {
        commands.copy_buffer_to_buffer(
            &buffers.status,
            0,
            &status_staging,
            BASE_VALIDATION_STAGING_BYTES,
            source.memory.hf_status_bytes,
        );
    }

    let completion = Arc::new(MapCompletion::default());
    let lifetime = Arc::new(VarDctJobLifetime {
        output: GpuBufferLease::from_tracked(output.as_ref().clone(), permits.output),
        status_staging,
        status_mapped: AtomicBool::new(false),
        _transient_permit: permits.transient,
        _codestream: codestream_buffer,
        _modular_metadata: modular_metadata,
        _reconstructed: reconstructed,
        _raw_metadata: raw_metadata,
        _coefficients: coefficients,
        _packet_status: packet_status,
        _packet_control: packet_control,
        _modular_params: modular_params,
        _lf_temporary: lf_temporary,
        _resources: resources,
        _resource_uniform: resource_uniform,
        _adaptive_lf_uniform: adaptive_lf_uniform,
        _artifact: artifact,
        _occupancy: occupancy,
        _artifact_uniform: artifact_uniform,
        _hf_coefficients: hf_coefficient_buffers,
        _xyb_planes: xyb_planes,
        _resident_scratch: resident_scratch,
        _output_scratch: output_scratch,
    });
    let callback_lifetime = Arc::clone(&lifetime);
    let callback_completion = Arc::clone(&completion);
    commands.map_buffer_on_submit(
        &lifetime.status_staging,
        wgpu::MapMode::Read,
        ..,
        move |result| {
            if result.is_ok() {
                callback_lifetime
                    .status_mapped
                    .store(true, Ordering::Release);
            }
            drop(callback_lifetime);
            callback_completion.complete(
                result.map_err(|error| format!("VarDCT validation mapping failed: {error}")),
            );
        },
    );
    let submission = backend.queue().submit([commands.finish()]);
    let poll_completion = Arc::clone(&completion);
    if let Err(error) = poll_permit.register(submission, move |error| {
        poll_completion.complete(Err(error));
    }) {
        completion.complete(Err(format!("VarDCT GPU poll registration failed: {error}")));
    }
    let correlations = source
        .packet
        .profile
        .width
        .div_ceil(64)
        .checked_mul(source.packet.profile.height.div_ceil(64))
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "correlation samples",
        })?;
    let expected_hf_samples = source
        .packet
        .task_count
        .checked_mul(2)
        .and_then(|tasks| blocks.checked_add(tasks))
        .and_then(|samples| {
            correlations
                .checked_mul(2)
                .and_then(|cfl| samples.checked_add(cfl))
        })
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "HF metadata sample count",
        })?;
    Ok(VarDctPendingFrame {
        device: device.clone(),
        lifetime: Some(lifetime),
        completion,
        token: SubmissionToken(1),
        layout: source.layout.clone(),
        transform: source.packet.transform,
        frame_name: source.frame_name.clone(),
        expected_lf_samples: blocks * 3,
        expected_hf_samples,
        expected_coefficients: source.packet.coefficient_words(),
        expected_blocks: blocks,
        expected_tasks: source.packet.task_count,
        expected_hf_groups: source
            .hf_coefficients
            .as_ref()
            .map(|plan| plan.params.len() as u32)
            .unwrap_or(0),
        expected_global_scale: source.packet.global_scale,
        expected_quant_lf: source.packet.quant_lf,
    })
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
