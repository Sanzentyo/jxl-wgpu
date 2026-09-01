//! GPU dispatch, artifact validation, and VarDCT encoder handles.

use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use jxl_gpu_bitstream::PrefixCodeEntry;
use jxl_wgpu::{KernelVariant, MemoryPermit};

use super::bitstream::{build_frame_packet, image_header, pack_signed_control};
use super::entropy::{HfEntropyPlan, fixed_prefix_code, prefix_entries};
use super::types::{
    DCT8_COEFFICIENTS, DCT8_NATURAL_ORDER, GLOBAL_SCALE, HF_QUANTIZATION, MAX_AC_FRAGMENT_WORDS,
    MAX_BLOCKS, MAX_COEFFICIENTS, MAX_DC_FRAGMENT_WORDS, MAX_HF_QUANTIZED_MAGNITUDE, QUANT_LF,
    SCALABLE_ARTIFACT_READY, SCALABLE_HEADER_WORDS, ScalableArtifactLayout,
    ScalableDcFragmentDescriptor, ScalableVarDctArtifactHeader, ScalableVarDctKernelParams,
    TiledVarDctGrid, VarDctArtifactData, VarDctColorEncoding, VarDctFrameLayout,
    VarDctKernelArtifact, VarDctKernelParams, VarDctLfMetadata, VarDctMemoryPlan, VarDctStrategy,
    VarDctTopology,
};
use crate::prefix::{PrefixCode, RAW_SYMBOLS};
use crate::{
    AnimationHeader, BackendError, BitFragment, BufferImageSource, Determinism, EncodeError,
    EncodeProfile, EncoderCapabilities, FrameEncodeRequest, FrameIndex, FrameOptions,
    FrameSubmission, GpuEncodeBackend, GpuEncodeJob, GpuEncoder, GpuFrameArtifacts, GpuFrameSource,
    KernelStage, PerceptualDistance, ProfileCapability, ProgressivePlan, UnsupportedFeature,
    WgpuContext, assemble_frame,
};

pub(super) const SHADER: &str = include_str!("../vardct_encoder.wgsl");
pub(super) const LARGE_SHADER: &str = include_str!("../vardct_large_encoder.wgsl");
pub(super) const PROFILE_DISTANCE: f32 = 25.0;
pub(super) const BOUNDED_KERNEL_KEY: &str = "vardct_encode_bounded";
pub(super) const SCALABLE_QUANTIZE_KERNEL_KEY: &str = "vardct_encode_quantize";
pub(super) const BOUNDED_WORKGROUP_STORAGE_BYTES: u32 = 1_024 * 16;
pub(super) const LARGE_WORKGROUP_STORAGE_BYTES: u32 = 64 * 16;

#[derive(Clone, Copy, Debug)]
pub(super) struct VarDctDispatchPlan {
    source_binding_offset: u64,
    source_binding_size: NonZeroU64,
    kernel: VarDctKernelPlan,
    memory: VarDctMemoryPlan,
    frame: VarDctFrameLayout,
}

#[derive(Clone, Copy, Debug)]
enum VarDctKernelPlan {
    Bounded(VarDctKernelParams),
    Scalable {
        params: ScalableVarDctKernelParams,
        layout: ScalableArtifactLayout,
    },
}

enum VarDctPipelines {
    Bounded(Arc<wgpu::ComputePipeline>),
    Scalable {
        quantize: Arc<wgpu::ComputePipeline>,
        serialize: Arc<wgpu::ComputePipeline>,
    },
}

/// GPU backend for one standard VarDCT still-image strategy.
///
/// The source extent must equal the selected transform extent. The backend
/// emits a standards-compliant VarDCT frame and does not route pixels or
/// coefficients through a CPU codec.
pub struct VarDctBackend {
    pipelines: VarDctPipelines,
    workgroup_variant: KernelVariant,
    code: PrefixCode,
    hf_entropy: HfEntropyPlan,
    topology: VarDctTopology,
    lf_metadata: VarDctLfMetadata,
    capabilities: EncoderCapabilities,
    max_storage_binding_size: u64,
    max_buffer_size: u64,
    max_compute_workgroups_per_dimension: u32,
    storage_offset_alignment: u64,
}

impl VarDctBackend {
    /// Creates a standard VarDCT strategy backend and its compute pipeline.
    ///
    /// # Errors
    ///
    /// Returns an encoder error if the fixed standard entropy tree cannot be
    /// represented by the JPEG XL prefix-code writer.
    pub fn new(context: &WgpuContext, strategy: VarDctStrategy) -> Result<Self, EncodeError> {
        Self::new_with_lf_metadata(context, strategy, VarDctLfMetadata::default())
    }

    /// Creates a standard VarDCT strategy backend with explicit LF metadata.
    pub fn new_with_lf_metadata(
        context: &WgpuContext,
        strategy: VarDctStrategy,
        lf_metadata: VarDctLfMetadata,
    ) -> Result<Self, EncodeError> {
        Self::new_with_topology(
            context,
            VarDctTopology::SingleTransform(strategy),
            lf_metadata,
        )
    }

    /// Creates the bounded tiled-DCT8 profile used by [`TiledVarDctEncoder`].
    /// Every padded 8x8 block is an independent regular transform. The source
    /// extent selects the checked block, LF-group, and AC-group grids at
    /// submission time.
    pub fn new_tiled_dct8(context: &WgpuContext) -> Result<Self, EncodeError> {
        Self::new_tiled_dct8_with_lf_metadata(context, VarDctLfMetadata::default())
    }

    /// Creates the tiled DCT8 backend with explicit LF metadata.
    pub fn new_tiled_dct8_with_lf_metadata(
        context: &WgpuContext,
        lf_metadata: VarDctLfMetadata,
    ) -> Result<Self, EncodeError> {
        Self::new_with_topology(context, VarDctTopology::TiledDct8, lf_metadata)
    }

    fn new_with_topology(
        context: &WgpuContext,
        topology: VarDctTopology,
        lf_metadata: VarDctLfMetadata,
    ) -> Result<Self, EncodeError> {
        let code = fixed_prefix_code()?;
        let hf_entropy = HfEntropyPlan::single_cluster_prefix()?;
        let limits = context.device().limits();
        let (kernel_key, default_variant, workgroup_storage_bytes) =
            if topology.uses_scalable_kernel() {
                (
                    SCALABLE_QUANTIZE_KERNEL_KEY,
                    KernelVariant::Lanes64,
                    LARGE_WORKGROUP_STORAGE_BYTES,
                )
            } else {
                (
                    BOUNDED_KERNEL_KEY,
                    KernelVariant::Lanes256,
                    BOUNDED_WORKGROUP_STORAGE_BYTES,
                )
            };
        let workgroup_variant = context
            .kernel_policy()
            .variant_for(kernel_key, default_variant)?;
        workgroup_variant.validate_for(kernel_key, &limits, workgroup_storage_bytes)?;
        let (workgroup_x, _) = workgroup_variant.workgroup_size();
        let workgroup_constants = [("wg_x", f64::from(workgroup_x))];
        let pipelines = if topology.uses_scalable_kernel() {
            validate_scalable_device_limits(&limits)?;
            let module = context
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("jxl-wgpu scalable VarDCT kernel"),
                    source: wgpu::ShaderSource::Wgsl(LARGE_SHADER.into()),
                });
            VarDctPipelines::Scalable {
                quantize: Arc::new(context.device().create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("jxl-wgpu scalable VarDCT block quantization"),
                        layout: None,
                        module: &module,
                        entry_point: Some("quantize_blocks"),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &workgroup_constants,
                            ..Default::default()
                        },
                        cache: None,
                    },
                )),
                serialize: Arc::new(context.device().create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("jxl-wgpu scalable VarDCT control serialization"),
                        layout: None,
                        module: &module,
                        entry_point: Some("serialize_control"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    },
                )),
            }
        } else {
            let module = context
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("jxl-wgpu VarDCT forward-transform kernel"),
                    source: wgpu::ShaderSource::Wgsl(SHADER.into()),
                });
            VarDctPipelines::Bounded(Arc::new(context.device().create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("jxl-wgpu VarDCT strategy pipeline"),
                    layout: None,
                    module: &module,
                    entry_point: Some("encode"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &workgroup_constants,
                        ..Default::default()
                    },
                    cache: None,
                },
            )))
        };
        let distance = profile_distance();
        Ok(Self {
            pipelines,
            workgroup_variant,
            code,
            hf_entropy,
            topology,
            lf_metadata,
            capabilities: EncoderCapabilities {
                profiles: vec![ProfileCapability::VarDct {
                    min_distance: distance,
                    max_distance: distance,
                }],
                max_progressive_passes: 1,
                animation: false,
                determinism: Determinism::SameDevice,
                implemented_stages: vec![
                    KernelStage::InputNormalization,
                    KernelStage::ColorTransform,
                    KernelStage::ForwardTransform,
                    KernelStage::Quantization,
                    KernelStage::CoefficientTokenization,
                    KernelStage::HistogramReduction,
                ],
            },
            max_storage_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            storage_offset_alignment: u64::from(limits.min_storage_buffer_offset_alignment),
        })
    }

    /// Selected linear workgroup for the parallel forward/quantization pass.
    ///
    /// The scalable control serializer remains a separate fixed scalar pass because its DC
    /// prediction and bit-offset state are sequential.
    #[must_use]
    pub const fn workgroup_variant(&self) -> KernelVariant {
        self.workgroup_variant
    }

    #[must_use]
    pub const fn lf_metadata(&self) -> VarDctLfMetadata {
        self.lf_metadata
    }

    /// Computes the exact memory admission and source binding before a job is
    /// submitted.
    pub fn memory_plan(&self, source: &BufferImageSource) -> Result<VarDctMemoryPlan, EncodeError> {
        Ok(self.dispatch_plan(source)?.memory)
    }

    fn dispatch_plan(&self, source: &BufferImageSource) -> Result<VarDctDispatchPlan, EncodeError> {
        let extent = source.layout.extent;
        let frame = match self.topology {
            VarDctTopology::SingleTransform(strategy) => {
                let frame = VarDctFrameLayout::single(strategy);
                if extent.width != frame.width || extent.height != frame.height {
                    return Err(EncodeError::InvalidSource(
                        "the VarDCT source extent must equal the selected transform extent",
                    ));
                }
                frame
            }
            VarDctTopology::TiledDct8 => {
                VarDctFrameLayout::tiled_dct8(extent.width, extent.height)?
            }
        };
        if source.layout.format != VarDctColorEncoding::SrgbD65.pixel_format()
            || source.layout.planes.len() != 1
            || !source.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        let plane = source
            .layout
            .plane(0)
            .ok_or(EncodeError::InvalidSource("missing VarDCT RGB plane"))?;
        let row_bytes = u64::from(extent.width) * 3;
        if plane.row_bytes != row_bytes || plane.row_stride < row_bytes {
            return Err(EncodeError::InvalidSource(
                "the VarDCT RGB plane has an invalid row layout",
            ));
        }
        let row_stride = u32::try_from(plane.row_stride)
            .map_err(|_| EncodeError::InvalidSource("VarDCT row stride exceeds WGSL u32"))?;
        let sample_end = plane
            .row_stride
            .checked_mul(u64::from(extent.height - 1))
            .and_then(|rows| plane.offset.checked_add(rows))
            .and_then(|offset| offset.checked_add(row_bytes))
            .ok_or(EncodeError::InvalidSource(
                "VarDCT source address arithmetic overflow",
            ))?;
        let binding_end = align_up(sample_end, 4).ok_or(EncodeError::InvalidSource(
            "VarDCT source binding size overflow",
        ))?;
        if binding_end > source.buffer.size() {
            return Err(EncodeError::InvalidSource(
                "VarDCT source binding does not contain the final sample word",
            ));
        }
        let alignment = self.storage_offset_alignment.max(4);
        let source_binding_offset = plane.offset - plane.offset % alignment;
        let source_binding_bytes =
            binding_end
                .checked_sub(source_binding_offset)
                .ok_or(EncodeError::InvalidSource(
                    "VarDCT source binding range underflow",
                ))?;
        if source_binding_bytes > self.max_storage_binding_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffer_binding_size",
                required: source_binding_bytes,
                available: self.max_storage_binding_size,
            }
            .into());
        }
        let source_binding_size = NonZeroU64::new(source_binding_bytes).ok_or(
            EncodeError::InvalidSource("VarDCT source binding must not be empty"),
        )?;
        let relative_offset =
            plane
                .offset
                .checked_sub(source_binding_offset)
                .ok_or(EncodeError::InvalidSource(
                    "VarDCT source address arithmetic underflow",
                ))?;
        let shader_last_byte = sample_end
            .checked_sub(source_binding_offset)
            .and_then(|end| end.checked_sub(1))
            .ok_or(EncodeError::InvalidSource(
                "VarDCT source address arithmetic underflow",
            ))?;
        u32::try_from(shader_last_byte).map_err(|_| {
            EncodeError::InvalidSource("VarDCT source address exceeds the WGSL u32 space")
        })?;
        let byte_offset = u32::try_from(relative_offset).map_err(|_| {
            EncodeError::InvalidSource("VarDCT source offset exceeds the WGSL u32 space")
        })?;
        let blocks_x = frame.blocks_x;
        let blocks_y = frame.blocks_y;
        let (lf_quantization, lf_correlation) = self.lf_metadata.forward_quantization();
        let hf_correlation = self.lf_metadata.hf_correlation();
        let common_strategy = u32::from(frame.topology.strategy().codestream_id());
        let (kernel, memory) = if frame.topology.uses_scalable_kernel() {
            let layout = match frame.topology {
                VarDctTopology::SingleTransform(strategy) => {
                    ScalableArtifactLayout::new(strategy, &self.code)?
                }
                VarDctTopology::TiledDct8 => ScalableArtifactLayout::for_block_grid(
                    blocks_x,
                    blocks_y,
                    frame.lf_group_count()?,
                    &self.code,
                )?,
            };
            let required_workgroup_axis = blocks_x.max(blocks_y);
            if required_workgroup_axis > self.max_compute_workgroups_per_dimension {
                return Err(UnsupportedFeature::DeviceLimit {
                    name: "max_compute_workgroups_per_dimension",
                    required: u64::from(required_workgroup_axis),
                    available: u64::from(self.max_compute_workgroups_per_dimension),
                }
                .into());
            }
            let artifact_bytes = layout.artifact_bytes();
            if artifact_bytes > self.max_storage_binding_size {
                return Err(UnsupportedFeature::DeviceLimit {
                    name: "max_storage_buffer_binding_size",
                    required: artifact_bytes,
                    available: self.max_storage_binding_size,
                }
                .into());
            }
            if artifact_bytes > self.max_buffer_size {
                return Err(UnsupportedFeature::DeviceLimit {
                    name: "max_buffer_size",
                    required: artifact_bytes,
                    available: self.max_buffer_size,
                }
                .into());
            }
            (
                VarDctKernelPlan::Scalable {
                    params: ScalableVarDctKernelParams {
                        row_stride,
                        byte_offset,
                        width: extent.width,
                        height: extent.height,
                        blocks_x,
                        blocks_y,
                        strategy: common_strategy,
                        global_scale: GLOBAL_SCALE,
                        quant_lf: QUANT_LF,
                        raw_prefix: prefix_entries(&self.code),
                        strategy_offset: layout.strategy_offset,
                        dc_offset: layout.dc_offset,
                        token_offset: layout.token_offset,
                        extra_offset: layout.extra_offset,
                        fragment_offset: layout.fragment_offset,
                        fragment_word_capacity: layout.fragment_word_capacity,
                        artifact_words: layout.artifact_words,
                        topology: frame.topology.artifact_id(),
                        fragment_descriptor_offset: layout.fragment_descriptor_offset,
                        fragment_descriptor_len: layout.fragment_descriptor_len,
                        lf_groups_x: frame.lf_groups_x,
                        lf_groups_y: frame.lf_groups_y,
                        lf_quantization,
                        lf_correlation,
                    },
                    layout,
                },
                VarDctMemoryPlan::scalable(
                    source_binding_bytes,
                    artifact_bytes,
                    frame.topology.kernel_layout(),
                ),
            )
        } else {
            (
                VarDctKernelPlan::Bounded(VarDctKernelParams {
                    row_stride,
                    byte_offset,
                    width: extent.width,
                    height: extent.height,
                    blocks_x,
                    blocks_y,
                    strategy: common_strategy,
                    global_scale: GLOBAL_SCALE,
                    quant_lf: QUANT_LF,
                    dc_prefix: prefix_entries(&self.code),
                    hf_prefix: self.hf_entropy.gpu_entries(),
                    lf_quantization,
                    lf_correlation,
                    hf_correlation,
                    hf_quantization: HF_QUANTIZATION,
                    padding: [0; 33],
                }),
                VarDctMemoryPlan::fixed(source_binding_bytes),
            )
        };
        Ok(VarDctDispatchPlan {
            source_binding_offset,
            source_binding_size,
            kernel,
            memory,
            frame,
        })
    }
}

pub(super) fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let adjustment = alignment.checked_sub(1)?;
    value
        .checked_add(adjustment)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}

fn validate_scalable_device_limits(limits: &wgpu::Limits) -> Result<(), EncodeError> {
    let checks = [(
        "max_storage_buffers_per_shader_stage",
        3,
        u64::from(limits.max_storage_buffers_per_shader_stage),
    )];
    if let Some((name, required, available)) = checks
        .into_iter()
        .find(|(_, required, available)| required > available)
    {
        return Err(UnsupportedFeature::DeviceLimit {
            name,
            required,
            available,
        }
        .into());
    }
    Ok(())
}

pub(super) fn profile_distance() -> PerceptualDistance {
    PerceptualDistance::new(PROFILE_DISTANCE)
        .expect("the fixed VarDCT distance is within the public validated range")
}

fn validate_vardct_request(
    request: &FrameEncodeRequest,
    frame: VarDctFrameLayout,
) -> Result<(), EncodeError> {
    if request.frame_index != FrameIndex::new(0)
        || !request.is_last
        || request.animation != AnimationHeader::Still
        || request.canvas_width != frame.width
        || request.canvas_height != frame.height
        || request.options != FrameOptions::default()
        || request.progressive != ProgressivePlan::single()
    {
        return Err(EncodeError::InvalidConfiguration(
            "the VarDCT profile requires one full-canvas final transform-sized still frame",
        ));
    }
    if request.profile
        != (EncodeProfile::VarDct {
            distance: profile_distance(),
        })
    {
        return Err(EncodeError::InvalidConfiguration(
            "the requested VarDCT distance does not match the fixed LF-first profile",
        ));
    }
    Ok(())
}

impl GpuEncodeBackend for VarDctBackend {
    type Job = VarDctJob;

    fn capabilities(&self) -> &EncoderCapabilities {
        &self.capabilities
    }

    fn supports_input(&self, source: &GpuFrameSource) -> bool {
        let GpuFrameSource::Buffer(source) = source else {
            return false;
        };
        self.dispatch_plan(source).is_ok()
    }

    fn submit(
        &self,
        context: &WgpuContext,
        source: GpuFrameSource,
        request: &FrameEncodeRequest,
    ) -> Result<Self::Job, EncodeError> {
        let GpuFrameSource::Buffer(source) = source else {
            return Err(UnsupportedFeature::InputFormat.into());
        };
        let plan = self.dispatch_plan(&source)?;
        validate_vardct_request(request, plan.frame)?;
        let memory_permit = context
            .memory_budget()
            .try_reserve(plan.memory.owned_bytes_per_job)?;

        let parameters = Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu VarDCT parameters"),
            size: plan.memory.parameter_storage_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let artifact = Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu VarDCT artifact"),
            size: plan.memory.artifact_storage_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let readback = Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu VarDCT readback"),
            size: plan.memory.readback_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        context.queue().write_buffer(
            &parameters,
            0,
            match &plan.kernel {
                VarDctKernelPlan::Bounded(params) => bytemuck::bytes_of(params),
                VarDctKernelPlan::Scalable { params, .. } => bytemuck::bytes_of(params),
            },
        );

        let source_binding = wgpu::BufferBinding {
            buffer: &source.buffer,
            offset: plan.source_binding_offset,
            size: Some(plan.source_binding_size),
        };
        let params_binding_size = NonZeroU64::new(plan.memory.parameter_storage_bytes)
            .expect("the VarDCT parameter ABI is non-empty");
        let artifact_binding_size = NonZeroU64::new(plan.memory.artifact_storage_bytes)
            .expect("the VarDCT artifact ABI is non-empty");
        let create_bind_group = |pipeline: &wgpu::ComputePipeline, label| {
            context
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(source_binding.clone()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &parameters,
                                offset: 0,
                                size: Some(params_binding_size),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &artifact,
                                offset: 0,
                                size: Some(artifact_binding_size),
                            }),
                        },
                    ],
                })
        };
        let mut commands =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu VarDCT encode"),
                });
        commands.clear_buffer(&artifact, 0, None);
        let job_layout = match (&self.pipelines, plan.kernel) {
            (VarDctPipelines::Bounded(pipeline), VarDctKernelPlan::Bounded(_)) => {
                let bind_group = create_bind_group(pipeline, "jxl-wgpu VarDCT bindings");
                let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("jxl-wgpu VarDCT forward transform and tokenization"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
                VarDctJobLayout::Bounded
            }
            (
                VarDctPipelines::Scalable {
                    quantize,
                    serialize,
                },
                VarDctKernelPlan::Scalable { params, layout },
            ) => {
                let quantize_bind_group =
                    create_bind_group(quantize, "jxl-wgpu scalable VarDCT quantization bindings");
                {
                    let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("jxl-wgpu scalable VarDCT 8x8 DC quantization"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(quantize);
                    pass.set_bind_group(0, &quantize_bind_group, &[]);
                    pass.dispatch_workgroups(params.blocks_x, params.blocks_y, 1);
                }
                // A separate WebGPU pass is the explicit global storage
                // visibility boundary for all block workgroups before the
                // single deterministic prediction/serialization invocation.
                // The control entry point intentionally has no source binding;
                // automatic pipeline layouts therefore retain only bindings
                // 1 and 2 for this pass.
                let serialize_bind_group =
                    context
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("jxl-wgpu scalable VarDCT serialization bindings"),
                            layout: &serialize.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &parameters,
                                        offset: 0,
                                        size: Some(params_binding_size),
                                    }),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &artifact,
                                        offset: 0,
                                        size: Some(artifact_binding_size),
                                    }),
                                },
                            ],
                        });
                {
                    let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("jxl-wgpu scalable VarDCT control and entropy serialization"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(serialize);
                    pass.set_bind_group(0, &serialize_bind_group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                VarDctJobLayout::Scalable(layout)
            }
            _ => {
                return Err(BackendError::Invariant(
                    "VarDCT strategy selected incompatible GPU pipelines",
                )
                .into());
            }
        };
        commands.copy_buffer_to_buffer(
            &artifact,
            0,
            &readback,
            0,
            plan.memory.artifact_storage_bytes,
        );

        let completion = Arc::new(VarDctMapCompletion::default());
        let callback_completion = Arc::clone(&completion);
        let readback_for_map = Arc::clone(&readback);
        let lifetime = Arc::new(VarDctJobLifetime {
            _parameters: parameters,
            _artifact: artifact,
            readback,
            _memory_permit: memory_permit,
            mapped: AtomicBool::new(false),
        });
        let callback_lifetime = Arc::clone(&lifetime);
        commands.map_buffer_on_submit(
            &readback_for_map,
            wgpu::MapMode::Read,
            0..plan.memory.readback_bytes,
            move |result| {
                if result.is_ok() {
                    callback_lifetime.mapped.store(true, Ordering::Release);
                }
                callback_completion.complete(result.map_err(BackendError::ArtifactMapping));
                drop(callback_lifetime);
            },
        );
        let poll_permit = context.submission_poller().try_reserve()?;
        let submission_index = context.queue().submit([commands.finish()]);
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll_permit.register(submission_index, move |error| {
            poll_completion.complete(Err(BackendError::PollWorker(error)));
        }) {
            completion.complete(Err(BackendError::PollRegistration(error)));
        }

        Ok(VarDctJob {
            lifetime: Some(lifetime),
            completion,
            code: self.code.clone(),
            hf_entropy: self.hf_entropy.clone(),
            lf_metadata: self.lf_metadata,
            frame_layout: plan.frame,
            artifact_layout: job_layout,
            frame_index: request.frame_index,
            is_last: request.is_last,
        })
    }
}

#[derive(Default)]
struct VarDctMapCompletion {
    state: Mutex<VarDctMapState>,
    condition: Condvar,
}

#[derive(Default)]
struct VarDctMapState {
    result: Option<Result<(), BackendError>>,
    waker: Option<Waker>,
}

impl VarDctMapCompletion {
    fn complete(&self, result: Result<(), BackendError>) {
        let waker = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
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

    fn poll(&self, cx: &Context<'_>) -> Option<Result<(), BackendError>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.result.is_none() {
            state.waker = Some(cx.waker().clone());
        }
        state.result.take()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self) -> Result<(), BackendError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.result.is_none() {
            state = self
                .condition
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state
            .result
            .take()
            .expect("VarDCT map completion was checked as present")
    }
}

struct VarDctJobLifetime {
    _parameters: Arc<wgpu::Buffer>,
    _artifact: Arc<wgpu::Buffer>,
    readback: Arc<wgpu::Buffer>,
    _memory_permit: MemoryPermit,
    mapped: AtomicBool,
}

impl Drop for VarDctJobLifetime {
    fn drop(&mut self) {
        if self.mapped.swap(false, Ordering::AcqRel) {
            self.readback.unmap();
        }
    }
}

/// Runtime-neutral completion for one standard VarDCT GPU submission.
#[derive(Clone, Copy, Debug)]
enum VarDctJobLayout {
    Bounded,
    Scalable(ScalableArtifactLayout),
}

pub struct VarDctJob {
    lifetime: Option<Arc<VarDctJobLifetime>>,
    completion: Arc<VarDctMapCompletion>,
    code: PrefixCode,
    hf_entropy: HfEntropyPlan,
    lf_metadata: VarDctLfMetadata,
    frame_layout: VarDctFrameLayout,
    artifact_layout: VarDctJobLayout,
    frame_index: FrameIndex,
    is_last: bool,
}

impl VarDctJob {
    fn finish(
        &mut self,
        mapping: Result<(), BackendError>,
    ) -> Result<GpuFrameArtifacts, EncodeError> {
        let lifetime = self.lifetime.take().ok_or(BackendError::Invariant(
            "VarDCT GPU job was already consumed",
        ))?;
        mapping?;
        let mapped = match lifetime.readback.slice(..).get_mapped_range() {
            Ok(mapped) => mapped,
            Err(error) => {
                lifetime.readback.unmap();
                lifetime.mapped.store(false, Ordering::Release);
                return Err(BackendError::ArtifactRange(error).into());
            }
        };
        let result = (|| {
            let artifact = match self.artifact_layout {
                VarDctJobLayout::Bounded => {
                    let artifact = bytemuck::try_from_bytes::<VarDctKernelArtifact>(&mapped)
                        .map_err(|_| {
                            BackendError::InvalidArtifact("VarDCT ABI size or alignment")
                        })?;
                    validate_artifact(artifact, &self.code, &self.hf_entropy, self.frame_layout)?
                }
                VarDctJobLayout::Scalable(layout) => {
                    validate_scalable_artifact(&mapped, layout, &self.code, self.frame_layout)?
                }
            };
            Ok(GpuFrameArtifacts {
                frame_index: self.frame_index,
                is_last: self.is_last,
                packets: build_frame_packet(
                    artifact,
                    &self.code,
                    &self.hf_entropy,
                    self.frame_layout,
                    self.lf_metadata,
                )?,
                acceleration: None,
            })
        })();
        drop(mapped);
        lifetime.readback.unmap();
        lifetime.mapped.store(false, Ordering::Release);
        drop(lifetime);
        result
    }
}

impl GpuEncodeJob for VarDctJob {
    fn poll_complete(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<GpuFrameArtifacts, EncodeError>> {
        match self.completion.poll(cx) {
            Some(result) => Poll::Ready(self.finish(result)),
            None => Poll::Pending,
        }
    }

    fn wait(self) -> Result<GpuFrameArtifacts, EncodeError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut job = self;
            let result = job.completion.wait();
            job.finish(result)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(BackendError::Invariant(
                "blocking GPU waits are unavailable on browser WebGPU; await the submission",
            )
            .into())
        }
    }
}

fn validate_artifact<'a>(
    artifact: &'a VarDctKernelArtifact,
    code: &PrefixCode,
    hf_entropy: &HfEntropyPlan,
    frame: VarDctFrameLayout,
) -> Result<VarDctArtifactData<'a>, BackendError> {
    let VarDctTopology::SingleTransform(strategy) = frame.topology else {
        return Err(BackendError::InvalidArtifact(
            "the fixed VarDCT artifact cannot represent a tiled frame",
        ));
    };
    let (blocks_x, blocks_y) = strategy.block_grid();
    let block_count = usize::try_from(blocks_x * blocks_y)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT block count does not fit usize"))?;
    let expected_strategy = u32::from(strategy.codestream_id());
    if artifact.strategy != expected_strategy
        || artifact.block_count != block_count as u32
        || artifact.dc_sample_count != (3 * block_count) as u32
    {
        return Err(BackendError::InvalidArtifact(
            "VarDCT strategy or live-count header mismatch",
        ));
    }
    for block in 0..MAX_BLOCKS {
        let expected = if block < block_count {
            expected_strategy | u32::from(block == 0) << 8
        } else {
            0
        };
        if artifact.strategy_map[block] != expected {
            return Err(BackendError::InvalidArtifact(
                "VarDCT GPU strategy map is malformed",
            ));
        }
    }

    let coefficient_count =
        usize::from(strategy.block_extent().0) * usize::from(strategy.block_extent().1);
    let quantized_live_end = if strategy == VarDctStrategy::Dct8 {
        DCT8_COEFFICIENTS
    } else {
        block_count
    };
    let xyb_channels = [1usize, 0, 2];
    for (dc_channel, &xyb_channel) in xyb_channels.iter().enumerate() {
        let dc_base = dc_channel * MAX_BLOCKS;
        let coefficient_base = xyb_channel * MAX_COEFFICIENTS;
        for block in 0..block_count {
            if artifact.quantized_dc_yxb[dc_base + block]
                != artifact.quantized_xyb[coefficient_base + block]
            {
                return Err(BackendError::InvalidArtifact(
                    "VarDCT DC channel ordering mismatch",
                ));
            }
        }
        if artifact.quantized_dc_yxb[dc_base + block_count..dc_base + MAX_BLOCKS]
            .iter()
            .any(|&value| value != 0)
            || artifact.quantized_xyb
                [coefficient_base + quantized_live_end..coefficient_base + MAX_COEFFICIENTS]
                .iter()
                .any(|&value| value != 0)
        {
            return Err(BackendError::InvalidArtifact(
                "the VarDCT profile produced a nonzero coefficient padding token",
            ));
        }
    }
    if artifact
        .forward_xyb_bits
        .chunks_exact(MAX_COEFFICIENTS)
        .flat_map(|channel| &channel[..coefficient_count])
        .any(|&bits| !f32::from_bits(bits).is_finite())
    {
        return Err(BackendError::InvalidArtifact(
            "VarDCT forward transform produced a non-finite coefficient",
        ));
    }

    let entries = code.raw_entries();
    let mut expected_histogram = [0u32; RAW_SYMBOLS];
    let mut bit_offset = 0u32;
    for channel in 0..3 {
        let base = channel * MAX_BLOCKS;
        for block in 0..block_count {
            let block_x = block % blocks_x as usize;
            let block_y = block / blocks_x as usize;
            let left = if block_x > 0 {
                artifact.quantized_dc_yxb[base + block - 1]
            } else if block_y > 0 {
                artifact.quantized_dc_yxb[base + block - blocks_x as usize]
            } else {
                0
            };
            let top = if block_y > 0 {
                artifact.quantized_dc_yxb[base + block - blocks_x as usize]
            } else {
                left
            };
            let top_left = if block_x > 0 && block_y > 0 {
                artifact.quantized_dc_yxb[base + block - blocks_x as usize - 1]
            } else {
                left
            };
            let residual =
                gradient_residual_i32(artifact.quantized_dc_yxb[base + block], top, left, top_left);
            let (token, extra_bit_count, extra) = signed_token(residual)?;
            let slot = base + block;
            if artifact.dc_raw_tokens[slot] != token || artifact.dc_extra_bits[slot] != extra {
                return Err(BackendError::InvalidArtifact(
                    "VarDCT DC token does not match its predicted residual",
                ));
            }
            let token_index = usize::try_from(token).map_err(|_| {
                BackendError::InvalidArtifact("VarDCT DC token index does not fit usize")
            })?;
            let entry = entries
                .get(token_index)
                .ok_or(BackendError::InvalidArtifact(
                    "VarDCT DC token exceeds the fixed entropy alphabet",
                ))?;
            if read_fragment_bits(artifact, bit_offset, u32::from(entry.bit_len))?
                != u32::from(entry.bits)
            {
                return Err(BackendError::InvalidArtifact(
                    "VarDCT GPU prefix fragment does not match its token",
                ));
            }
            bit_offset += u32::from(entry.bit_len);
            if read_fragment_bits(artifact, bit_offset, extra_bit_count)? != extra {
                return Err(BackendError::InvalidArtifact(
                    "VarDCT GPU extra-bit fragment does not match its token",
                ));
            }
            bit_offset += extra_bit_count;
            expected_histogram[token_index] += 1;
        }
        if artifact.dc_raw_tokens[base + block_count..base + MAX_BLOCKS]
            .iter()
            .chain(&artifact.dc_extra_bits[base + block_count..base + MAX_BLOCKS])
            .any(|&value| value != 0)
        {
            return Err(BackendError::InvalidArtifact(
                "VarDCT DC token padding is nonzero",
            ));
        }
    }
    if bit_offset != artifact.dc_fragment_bit_len || artifact.raw_histogram != expected_histogram {
        return Err(BackendError::InvalidArtifact(
            "VarDCT GPU entropy fragment length or histogram mismatch",
        ));
    }
    validate_fixed_ac_artifact(artifact, &hf_entropy.code, strategy)?;
    Ok(fixed_artifact_data(artifact))
}

fn validate_fixed_ac_artifact(
    artifact: &VarDctKernelArtifact,
    code: &PrefixCode,
    strategy: VarDctStrategy,
) -> Result<(), BackendError> {
    if artifact.dc_padding.iter().any(|&word| word != 0)
        || artifact.ac_padding.iter().any(|&word| word != 0)
    {
        return Err(BackendError::InvalidArtifact(
            "bounded VarDCT artifact padding is nonzero",
        ));
    }
    if strategy != VarDctStrategy::Dct8 {
        if artifact.ac_fragment_bit_len != 0
            || artifact.ac_token_count != 0
            || artifact.ac_histogram.iter().any(|&count| count != 0)
            || artifact.ac_fragment_words.iter().any(|&word| word != 0)
        {
            return Err(BackendError::InvalidArtifact(
                "bounded non-DCT8 artifact contains an AC entropy fragment",
            ));
        }
        return Ok(());
    }
    let coefficient_nonzero =
        artifact
            .quantized_xyb
            .chunks_exact(MAX_COEFFICIENTS)
            .any(|channel| {
                channel[1..DCT8_COEFFICIENTS]
                    .iter()
                    .any(|&value| value != 0)
            });
    if !coefficient_nonzero {
        if artifact.ac_fragment_bit_len != 0
            || artifact.ac_token_count != 0
            || artifact.ac_histogram.iter().any(|&count| count != 0)
            || artifact.ac_fragment_words.iter().any(|&word| word != 0)
        {
            return Err(BackendError::InvalidArtifact(
                "zero-HF VarDCT artifact contains an AC entropy fragment",
            ));
        }
        return Ok(());
    }
    if artifact.ac_fragment_bit_len == 0
        || artifact.ac_fragment_bit_len
            > u32::try_from(MAX_AC_FRAGMENT_WORDS * 32).expect("bounded AC artifact fits u32")
    {
        return Err(BackendError::InvalidArtifact(
            "bounded VarDCT AC fragment length is invalid",
        ));
    }

    let entries = code.raw_entries();
    let mut expected_histogram = [0u32; RAW_SYMBOLS];
    let mut bit_offset = 0u32;
    let mut token_count = 0u32;
    for &xyb_channel in &[1usize, 0, 2] {
        let coefficient_base = xyb_channel * MAX_COEFFICIENTS;
        let nonzero_count = DCT8_NATURAL_ORDER[1..]
            .iter()
            .filter(|&&offset| artifact.quantized_xyb[coefficient_base + offset] != 0)
            .count();
        validate_ac_token(
            artifact,
            &entries,
            &mut expected_histogram,
            &mut bit_offset,
            u32::try_from(nonzero_count)
                .map_err(|_| BackendError::InvalidArtifact("DCT8 nonzero count exceeds u32"))?,
        )?;
        token_count += 1;
        if nonzero_count == 0 {
            continue;
        }

        let mut remaining = nonzero_count;
        for &offset in &DCT8_NATURAL_ORDER[1..] {
            let coefficient = artifact.quantized_xyb[coefficient_base + offset];
            if coefficient.unsigned_abs() > MAX_HF_QUANTIZED_MAGNITUDE as u32 {
                return Err(BackendError::InvalidArtifact(
                    "DCT8 coefficient exceeds the fixed HF token alphabet",
                ));
            }
            let packed = pack_signed_control(coefficient);
            validate_ac_token(
                artifact,
                &entries,
                &mut expected_histogram,
                &mut bit_offset,
                packed,
            )?;
            token_count += 1;
            if coefficient != 0 {
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
        }
    }
    if bit_offset != artifact.ac_fragment_bit_len
        || token_count != artifact.ac_token_count
        || expected_histogram != artifact.ac_histogram
    {
        return Err(BackendError::InvalidArtifact(
            "bounded VarDCT AC fragment length, token count, or histogram mismatch",
        ));
    }
    validate_fragment_padding(&artifact.ac_fragment_words, artifact.ac_fragment_bit_len)
}

fn validate_ac_token(
    artifact: &VarDctKernelArtifact,
    entries: &[PrefixCodeEntry; RAW_SYMBOLS],
    histogram: &mut [u32; RAW_SYMBOLS],
    bit_offset: &mut u32,
    value: u32,
) -> Result<(), BackendError> {
    let (token, extra_bit_count, extra) = unsigned_token(value)?;
    let token_index = usize::try_from(token)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT AC token index does not fit usize"))?;
    let entry = entries
        .get(token_index)
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT AC token exceeds the fixed entropy alphabet",
        ))?;
    if read_fragment_slice(
        &artifact.ac_fragment_words,
        artifact.ac_fragment_bit_len,
        *bit_offset,
        u32::from(entry.bit_len),
    )? != u32::from(entry.bits)
    {
        return Err(BackendError::InvalidArtifact(
            "VarDCT AC prefix fragment does not match its token",
        ));
    }
    *bit_offset += u32::from(entry.bit_len);
    if read_fragment_slice(
        &artifact.ac_fragment_words,
        artifact.ac_fragment_bit_len,
        *bit_offset,
        extra_bit_count,
    )? != extra
    {
        return Err(BackendError::InvalidArtifact(
            "VarDCT AC extra-bit fragment does not match its token",
        ));
    }
    *bit_offset += extra_bit_count;
    histogram[token_index] += 1;
    Ok(())
}

pub(super) fn fixed_artifact_data(artifact: &VarDctKernelArtifact) -> VarDctArtifactData<'_> {
    VarDctArtifactData {
        strategy: artifact.strategy,
        dc_fragment_words: &artifact.dc_fragment_words,
        dc_fragment_bit_len: artifact.dc_fragment_bit_len,
        dc_fragment_descriptors: &[],
        ac_fragment_words: &artifact.ac_fragment_words,
        ac_fragment_bit_len: artifact.ac_fragment_bit_len,
    }
}

fn validate_scalable_artifact<'a>(
    mapped: &'a [u8],
    layout: ScalableArtifactLayout,
    code: &PrefixCode,
    frame: VarDctFrameLayout,
) -> Result<VarDctArtifactData<'a>, BackendError> {
    let expected_bytes = usize::try_from(layout.artifact_bytes()).map_err(|_| {
        BackendError::InvalidArtifact("scalable VarDCT artifact size does not fit usize")
    })?;
    if mapped.len() != expected_bytes {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT mapped artifact has the wrong byte length",
        ));
    }
    let words = bytemuck::try_cast_slice::<u8, u32>(mapped).map_err(|_| {
        BackendError::InvalidArtifact("scalable VarDCT artifact word ABI alignment")
    })?;
    let header_bytes = mapped
        .get(..std::mem::size_of::<ScalableVarDctArtifactHeader>())
        .ok_or(BackendError::InvalidArtifact(
            "scalable VarDCT artifact header is truncated",
        ))?;
    let header = bytemuck::try_from_bytes::<ScalableVarDctArtifactHeader>(header_bytes)
        .map_err(|_| BackendError::InvalidArtifact("scalable VarDCT header ABI alignment"))?;
    let blocks_x = frame.blocks_x;
    let blocks_y = frame.blocks_y;
    let block_count = blocks_x
        .checked_mul(blocks_y)
        .ok_or(BackendError::InvalidArtifact(
            "scalable VarDCT block count overflow",
        ))?;
    let dc_sample_count = block_count
        .checked_mul(3)
        .ok_or(BackendError::InvalidArtifact(
            "scalable VarDCT sample count overflow",
        ))?;
    let strategy = frame.topology.strategy();
    let lf_group_count = frame
        .lf_group_count()
        .map_err(|_| BackendError::InvalidArtifact("scalable VarDCT LF group count overflow"))?;
    if header.status != SCALABLE_ARTIFACT_READY
        || header.block_count != block_count
        || header.dc_sample_count != dc_sample_count
        || header.strategy != u32::from(strategy.codestream_id())
        || header.ac_all_zero != 1
        || header.strategy_offset != layout.strategy_offset
        || header.strategy_len != layout.strategy_len
        || header.dc_offset != layout.dc_offset
        || header.dc_len != layout.dc_len
        || header.token_offset != layout.token_offset
        || header.token_len != layout.token_len
        || header.extra_offset != layout.extra_offset
        || header.extra_len != layout.extra_len
        || header.fragment_offset != layout.fragment_offset
        || header.fragment_word_capacity != layout.fragment_word_capacity
        || header.artifact_words != layout.artifact_words
        || header.width != frame.width
        || header.height != frame.height
        || header.blocks_x != blocks_x
        || header.blocks_y != blocks_y
        || header.topology != frame.topology.artifact_id()
        || header.fragment_descriptor_offset != layout.fragment_descriptor_offset
        || header.fragment_descriptor_len != layout.fragment_descriptor_len
        || header.lf_groups_x != frame.lf_groups_x
        || header.lf_groups_y != frame.lf_groups_y
        || header.lf_group_count != lf_group_count
    {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT status, live counts, orientation, or layout metadata mismatch",
        ));
    }
    if header.padding.iter().any(|&word| word != 0) {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT header padding is nonzero",
        ));
    }
    if header.dc_fragment_bit_len > layout.fragment_max_bits
        || header.dc_fragment_bit_len
            > layout
                .fragment_word_capacity
                .checked_mul(32)
                .ok_or(BackendError::InvalidArtifact(
                    "scalable VarDCT fragment capacity overflow",
                ))?
    {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT entropy fragment exceeds its checked capacity",
        ));
    }

    let descriptor_words = artifact_words(
        words,
        layout.fragment_descriptor_offset,
        layout.fragment_descriptor_len,
    )?;
    let fragment_descriptors =
        bytemuck::try_cast_slice::<u32, ScalableDcFragmentDescriptor>(descriptor_words).map_err(
            |_| BackendError::InvalidArtifact("scalable VarDCT fragment descriptor ABI alignment"),
        )?;
    if fragment_descriptors.len()
        != usize::try_from(lf_group_count).map_err(|_| {
            BackendError::InvalidArtifact("scalable VarDCT LF group count does not fit usize")
        })?
    {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT fragment descriptor count mismatch",
        ));
    }
    let strategy_map = artifact_words(words, layout.strategy_offset, layout.strategy_len)?;
    let quantized_dc = artifact_words(words, layout.dc_offset, layout.dc_len)?;
    let raw_tokens = artifact_words(words, layout.token_offset, layout.token_len)?;
    let extra_bits = artifact_words(words, layout.extra_offset, layout.extra_len)?;
    let fragment_words =
        artifact_words(words, layout.fragment_offset, layout.fragment_word_capacity)?;
    validate_zero_gap(
        words,
        SCALABLE_HEADER_WORDS,
        layout.fragment_descriptor_offset,
    )?;
    validate_zero_gap(
        words,
        layout.fragment_descriptor_offset + layout.fragment_descriptor_len,
        layout.strategy_offset,
    )?;
    validate_zero_gap(
        words,
        layout.strategy_offset + layout.strategy_len,
        layout.dc_offset,
    )?;
    validate_zero_gap(words, layout.dc_offset + layout.dc_len, layout.token_offset)?;
    validate_zero_gap(
        words,
        layout.token_offset + layout.token_len,
        layout.extra_offset,
    )?;
    validate_zero_gap(
        words,
        layout.extra_offset + layout.extra_len,
        layout.fragment_offset,
    )?;
    validate_zero_gap(
        words,
        layout.fragment_offset + layout.fragment_word_capacity,
        layout.artifact_words,
    )?;

    let expected_strategy = u32::from(strategy.codestream_id());
    for (block, &value) in strategy_map.iter().enumerate() {
        let is_first = match frame.topology {
            VarDctTopology::SingleTransform(_) => block == 0,
            VarDctTopology::TiledDct8 => true,
        };
        let expected = expected_strategy | u32::from(is_first) << 8;
        if value != expected {
            return Err(BackendError::InvalidArtifact(
                "scalable VarDCT GPU strategy map is malformed",
            ));
        }
    }

    let block_count_usize = usize::try_from(block_count)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT block count does not fit usize"))?;
    let entries = code.raw_entries();
    let mut expected_histogram = [0u32; RAW_SYMBOLS];
    let mut bit_offset = 0u32;
    for (group_index, descriptor) in fragment_descriptors.iter().enumerate() {
        let group_index = u32::try_from(group_index).map_err(|_| {
            BackendError::InvalidArtifact("scalable VarDCT LF group index exceeds u32")
        })?;
        let group = frame.lf_group_blocks(group_index).map_err(|_| {
            BackendError::InvalidArtifact("scalable VarDCT LF group geometry mismatch")
        })?;
        if descriptor.bit_offset != bit_offset {
            return Err(BackendError::InvalidArtifact(
                "scalable VarDCT fragment descriptors are not contiguous",
            ));
        }
        for channel in 0..3usize {
            let base = channel * block_count_usize;
            for local_y in 0..group.height as usize {
                for local_x in 0..group.width as usize {
                    let block_x = group.origin_x as usize + local_x;
                    let block_y = group.origin_y as usize + local_y;
                    let block = block_y * blocks_x as usize + block_x;
                    let left = if local_x > 0 {
                        quantized_dc[base + block - 1] as i32
                    } else if local_y > 0 {
                        quantized_dc[base + block - blocks_x as usize] as i32
                    } else {
                        0
                    };
                    let top = if local_y > 0 {
                        quantized_dc[base + block - blocks_x as usize] as i32
                    } else {
                        left
                    };
                    let top_left = if local_x > 0 && local_y > 0 {
                        quantized_dc[base + block - blocks_x as usize - 1] as i32
                    } else {
                        left
                    };
                    let actual = quantized_dc[base + block] as i32;
                    let residual = gradient_residual_i32(actual, top, left, top_left);
                    let (token, extra_bit_count, extra) = signed_token(residual)?;
                    let slot = base + block;
                    if raw_tokens[slot] != token || extra_bits[slot] != extra {
                        return Err(BackendError::InvalidArtifact(
                            "scalable VarDCT DC token does not match its predicted residual",
                        ));
                    }
                    let token_index = usize::try_from(token).map_err(|_| {
                        BackendError::InvalidArtifact("VarDCT DC token index does not fit usize")
                    })?;
                    let entry = entries
                        .get(token_index)
                        .ok_or(BackendError::InvalidArtifact(
                            "VarDCT DC token exceeds the fixed entropy alphabet",
                        ))?;
                    if read_fragment_slice(
                        fragment_words,
                        header.dc_fragment_bit_len,
                        bit_offset,
                        u32::from(entry.bit_len),
                    )? != u32::from(entry.bits)
                    {
                        return Err(BackendError::InvalidArtifact(
                            "scalable VarDCT GPU prefix fragment does not match its token",
                        ));
                    }
                    bit_offset += u32::from(entry.bit_len);
                    if read_fragment_slice(
                        fragment_words,
                        header.dc_fragment_bit_len,
                        bit_offset,
                        extra_bit_count,
                    )? != extra
                    {
                        return Err(BackendError::InvalidArtifact(
                            "scalable VarDCT GPU extra-bit fragment does not match its token",
                        ));
                    }
                    bit_offset += extra_bit_count;
                    expected_histogram[token_index] += 1;
                }
            }
        }
        if descriptor.bit_len != bit_offset - descriptor.bit_offset {
            return Err(BackendError::InvalidArtifact(
                "scalable VarDCT fragment descriptor length mismatch",
            ));
        }
    }
    if bit_offset != header.dc_fragment_bit_len || header.raw_histogram != expected_histogram {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT entropy fragment length or histogram mismatch",
        ));
    }
    validate_fragment_padding(fragment_words, header.dc_fragment_bit_len)?;
    Ok(VarDctArtifactData {
        strategy: expected_strategy,
        dc_fragment_words: fragment_words,
        dc_fragment_bit_len: header.dc_fragment_bit_len,
        dc_fragment_descriptors: fragment_descriptors,
        ac_fragment_words: &[],
        ac_fragment_bit_len: 0,
    })
}

fn artifact_words(words: &[u32], offset: u32, len: u32) -> Result<&[u32], BackendError> {
    let start = usize::try_from(offset)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT artifact offset does not fit usize"))?;
    let len = usize::try_from(len)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT artifact length does not fit usize"))?;
    let end = start.checked_add(len).ok_or(BackendError::InvalidArtifact(
        "VarDCT artifact range overflow",
    ))?;
    words.get(start..end).ok_or(BackendError::InvalidArtifact(
        "VarDCT artifact range is out of bounds",
    ))
}

fn validate_zero_gap(words: &[u32], start: u32, end: u32) -> Result<(), BackendError> {
    if artifact_words(
        words,
        start,
        end.checked_sub(start).ok_or(BackendError::InvalidArtifact(
            "VarDCT artifact section order is invalid",
        ))?,
    )?
    .iter()
    .any(|&word| word != 0)
    {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT artifact alignment padding is nonzero",
        ));
    }
    Ok(())
}

fn validate_fragment_padding(words: &[u32], bit_len: u32) -> Result<(), BackendError> {
    let used_words = bit_len
        .checked_add(31)
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT fragment word count overflow",
        ))?
        / 32;
    let used_words = usize::try_from(used_words)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT fragment size does not fit usize"))?;
    if let Some(&last_word) = used_words.checked_sub(1).and_then(|index| words.get(index)) {
        let live_bits = bit_len % 32;
        if live_bits != 0 && last_word & !((1u32 << live_bits) - 1) != 0 {
            return Err(BackendError::InvalidArtifact(
                "scalable VarDCT fragment has nonzero high padding bits",
            ));
        }
    }
    if words
        .get(used_words..)
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT fragment used-word count is out of bounds",
        ))?
        .iter()
        .any(|&word| word != 0)
    {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT fragment word padding is nonzero",
        ));
    }
    Ok(())
}

fn read_fragment_slice(
    words: &[u32],
    bit_len: u32,
    start: u32,
    count: u32,
) -> Result<u32, BackendError> {
    let end = start
        .checked_add(count)
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT GPU fragment address overflow",
        ))?;
    let capacity = u32::try_from(words.len())
        .ok()
        .and_then(|len| len.checked_mul(32))
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT GPU fragment capacity overflow",
        ))?;
    if end > bit_len || end > capacity {
        return Err(BackendError::InvalidArtifact(
            "VarDCT GPU fragment is truncated",
        ));
    }
    let mut value = 0u32;
    for index in 0..count {
        let bit = start + index;
        value |= ((words[(bit / 32) as usize] >> (bit % 32)) & 1) << index;
    }
    Ok(value)
}

pub(super) fn clamped_gradient_i32(top: i32, left: i32, top_left: i32) -> i32 {
    top.wrapping_add(left)
        .wrapping_sub(top_left)
        .clamp(top.min(left), top.max(left))
}

pub(super) fn gradient_residual_i32(actual: i32, top: i32, left: i32, top_left: i32) -> i32 {
    actual.wrapping_sub(clamped_gradient_i32(top, left, top_left))
}

pub(super) fn signed_token(value: i32) -> Result<(u32, u32, u32), BackendError> {
    let packed = if value >= 0 {
        u64::from(value as u32) * 2
    } else {
        u64::try_from(-i64::from(value)).expect("the negated i32 value fits u64") * 2 - 1
    };
    let packed = u32::try_from(packed).map_err(|_| {
        BackendError::InvalidArtifact("VarDCT signed coefficient exceeds the token alphabet")
    })?;
    unsigned_token(packed)
}

fn unsigned_token(value: u32) -> Result<(u32, u32, u32), BackendError> {
    if value == 0 {
        return Ok((0, 0, 0));
    }
    let extra_bit_count = 31 - value.leading_zeros();
    let token = extra_bit_count + 1;
    if token as usize >= RAW_SYMBOLS {
        return Err(BackendError::InvalidArtifact(
            "VarDCT token exceeds the fixed entropy alphabet",
        ));
    }
    Ok((token, extra_bit_count, value - (1 << extra_bit_count)))
}

fn read_fragment_bits(
    artifact: &VarDctKernelArtifact,
    start: u32,
    count: u32,
) -> Result<u32, BackendError> {
    let end = start
        .checked_add(count)
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT GPU fragment address overflow",
        ))?;
    if end > artifact.dc_fragment_bit_len
        || end > u32::try_from(MAX_DC_FRAGMENT_WORDS * 32).expect("fixed artifact fits u32")
    {
        return Err(BackendError::InvalidArtifact(
            "VarDCT GPU fragment is truncated",
        ));
    }
    let mut value = 0u32;
    for index in 0..count {
        let bit = start + index;
        let word = artifact.dc_fragment_words[(bit / 32) as usize];
        value |= ((word >> (bit % 32)) & 1) << index;
    }
    Ok(value)
}

/// GPU-only convenience encoder for one standard VarDCT transform.
pub struct VarDctEncoder {
    encoder: GpuEncoder<VarDctBackend>,
    strategy: VarDctStrategy,
}

impl VarDctEncoder {
    /// Creates the profile backend.
    ///
    /// # Errors
    ///
    /// Returns an encoder error if the fixed standard entropy tree cannot be
    /// constructed or the selected device cannot execute the strategy's
    /// checked storage/workgroup/dispatch requirements.
    pub fn new(context: WgpuContext, strategy: VarDctStrategy) -> Result<Self, EncodeError> {
        Self::new_with_lf_metadata(context, strategy, VarDctLfMetadata::default())
    }

    /// Creates the profile backend with explicit LF dequantization and channel correlation.
    pub fn new_with_lf_metadata(
        context: WgpuContext,
        strategy: VarDctStrategy,
        lf_metadata: VarDctLfMetadata,
    ) -> Result<Self, EncodeError> {
        let backend = VarDctBackend::new_with_lf_metadata(&context, strategy, lf_metadata)?;
        Ok(Self {
            encoder: GpuEncoder::new(context, backend),
            strategy,
        })
    }

    #[must_use]
    pub fn capabilities(&self) -> &EncoderCapabilities {
        self.encoder.capabilities()
    }

    #[must_use]
    pub const fn strategy(&self) -> VarDctStrategy {
        self.strategy
    }

    /// Workgroup selected for this encoder's parallel VarDCT pass.
    #[must_use]
    pub fn workgroup_variant(&self) -> KernelVariant {
        self.encoder.backend().workgroup_variant()
    }

    #[must_use]
    pub fn lf_metadata(&self) -> VarDctLfMetadata {
        self.encoder.backend().lf_metadata()
    }

    #[must_use]
    pub const fn color_encoding(&self) -> VarDctColorEncoding {
        VarDctColorEncoding::SrgbD65
    }

    #[must_use]
    pub fn distance(&self) -> PerceptualDistance {
        profile_distance()
    }

    #[must_use]
    pub fn in_flight_memory_stats(&self) -> jxl_wgpu::MemoryBudgetSnapshot {
        self.encoder.memory_stats()
    }

    pub fn memory_plan(&self, source: &BufferImageSource) -> Result<VarDctMemoryPlan, EncodeError> {
        self.encoder.backend().memory_plan(source)
    }

    pub fn submit(&self, source: BufferImageSource) -> Result<VarDctSubmission, EncodeError> {
        self.submit_inner(source, false)
    }

    pub fn submit_container(
        &self,
        source: BufferImageSource,
    ) -> Result<VarDctSubmission, EncodeError> {
        self.submit_inner(source, true)
    }

    pub fn encode(&self, source: BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit(source)?.wait()
    }

    pub fn encode_container(&self, source: BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit_container(source)?.wait()
    }

    fn submit_inner(
        &self,
        source: BufferImageSource,
        container: bool,
    ) -> Result<VarDctSubmission, EncodeError> {
        self.memory_plan(&source)?;
        let (width, height) = self.strategy.block_extent();
        let width = u32::from(width);
        let height = u32::from(height);
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::VarDct {
                distance: profile_distance(),
            },
            progressive: ProgressivePlan::single(),
            minimum_determinism: Determinism::SameDevice,
            animation: AnimationHeader::Still,
            canvas_width: width,
            canvas_height: height,
            options: FrameOptions::default(),
        };
        let frame = self
            .encoder
            .submit_frame(GpuFrameSource::Buffer(source), request)?;
        Ok(VarDctSubmission {
            frame: Some(frame),
            codestream_header: image_header(width, height)?,
            container,
        })
    }
}

/// GPU-only JPEG XL VarDCT encoder for a rectangular grid of independent
/// regular DCT8 transforms.
///
/// The current executable subset accepts RGB8 dimensions through 16,384 pixels
/// on each axis when at least one axis exceeds 256 pixels, including partial
/// 8x8 edge blocks. This guarantees an explicit multi-section TOC with every
/// 2,048x2,048 LF/DC group and at least two 256x256 AC/pass groups. AC
/// coefficients are deliberately zero, so decoded quality is the profile's
/// LF-only contract rather than a general distance-25 guarantee.
pub struct TiledVarDctEncoder {
    encoder: GpuEncoder<VarDctBackend>,
}

impl TiledVarDctEncoder {
    /// Creates the tiled DCT8 backend.
    ///
    /// # Errors
    ///
    /// Returns an encoder error if the fixed entropy tree cannot be built or
    /// the device cannot execute the checked scalable kernel ABI.
    pub fn new(context: WgpuContext) -> Result<Self, EncodeError> {
        Self::new_with_lf_metadata(context, VarDctLfMetadata::default())
    }

    /// Creates the tiled DCT8 backend with explicit LF dequantization and channel correlation.
    pub fn new_with_lf_metadata(
        context: WgpuContext,
        lf_metadata: VarDctLfMetadata,
    ) -> Result<Self, EncodeError> {
        let backend = VarDctBackend::new_tiled_dct8_with_lf_metadata(&context, lf_metadata)?;
        Ok(Self {
            encoder: GpuEncoder::new(context, backend),
        })
    }

    #[must_use]
    pub fn capabilities(&self) -> &EncoderCapabilities {
        self.encoder.capabilities()
    }

    /// Workgroup selected for block quantization. Control serialization remains scalar.
    #[must_use]
    pub fn workgroup_variant(&self) -> KernelVariant {
        self.encoder.backend().workgroup_variant()
    }

    #[must_use]
    pub fn lf_metadata(&self) -> VarDctLfMetadata {
        self.encoder.backend().lf_metadata()
    }

    #[must_use]
    pub const fn color_encoding(&self) -> VarDctColorEncoding {
        VarDctColorEncoding::SrgbD65
    }

    #[must_use]
    pub fn distance(&self) -> PerceptualDistance {
        profile_distance()
    }

    #[must_use]
    pub fn in_flight_memory_stats(&self) -> jxl_wgpu::MemoryBudgetSnapshot {
        self.encoder.memory_stats()
    }

    pub fn memory_plan(&self, source: &BufferImageSource) -> Result<VarDctMemoryPlan, EncodeError> {
        self.encoder.backend().memory_plan(source)
    }

    pub fn grid(&self, source: &BufferImageSource) -> Result<TiledVarDctGrid, EncodeError> {
        self.memory_plan(source)?;
        TiledVarDctGrid::new(source.layout.extent.width, source.layout.extent.height)
    }

    pub fn submit(&self, source: BufferImageSource) -> Result<VarDctSubmission, EncodeError> {
        self.submit_inner(source, false)
    }

    pub fn submit_container(
        &self,
        source: BufferImageSource,
    ) -> Result<VarDctSubmission, EncodeError> {
        self.submit_inner(source, true)
    }

    pub fn encode(&self, source: BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit(source)?.wait()
    }

    pub fn encode_container(&self, source: BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit_container(source)?.wait()
    }

    fn submit_inner(
        &self,
        source: BufferImageSource,
        container: bool,
    ) -> Result<VarDctSubmission, EncodeError> {
        let frame =
            VarDctFrameLayout::tiled_dct8(source.layout.extent.width, source.layout.extent.height)?;
        self.memory_plan(&source)?;
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::VarDct {
                distance: profile_distance(),
            },
            progressive: ProgressivePlan::single(),
            minimum_determinism: Determinism::SameDevice,
            animation: AnimationHeader::Still,
            canvas_width: frame.width,
            canvas_height: frame.height,
            options: FrameOptions::default(),
        };
        let frame_submission = self
            .encoder
            .submit_frame(GpuFrameSource::Buffer(source), request)?;
        Ok(VarDctSubmission {
            frame: Some(frame_submission),
            codestream_header: image_header(frame.width, frame.height)?,
            container,
        })
    }
}

/// Executor-independent future for a complete standard VarDCT codestream.
pub struct VarDctSubmission {
    frame: Option<FrameSubmission<VarDctJob>>,
    codestream_header: BitFragment,
    container: bool,
}

impl VarDctSubmission {
    pub fn wait(mut self) -> Result<Vec<u8>, EncodeError> {
        let frame = self
            .frame
            .take()
            .expect("a VarDCT submission can only complete once")
            .wait()?;
        self.assemble(frame)
    }

    fn assemble(&self, frame: GpuFrameArtifacts) -> Result<Vec<u8>, EncodeError> {
        let encoded_frame = assemble_frame(frame.packets)?;
        let mut codestream = self.codestream_header.bytes().to_vec();
        codestream.extend_from_slice(encoded_frame.bytes());
        if self.container {
            Ok(jxl_gpu_bitstream::write_container(&codestream)?)
        } else {
            Ok(codestream)
        }
    }
}

impl Future for VarDctSubmission {
    type Output = Result<Vec<u8>, EncodeError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let submission = self.get_mut();
        let frame = submission
            .frame
            .as_mut()
            .expect("a VarDCT submission must not be polled after completion");
        match Pin::new(frame).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                submission.frame.take();
                Poll::Ready(result.and_then(|frame| submission.assemble(frame)))
            }
        }
    }
}
