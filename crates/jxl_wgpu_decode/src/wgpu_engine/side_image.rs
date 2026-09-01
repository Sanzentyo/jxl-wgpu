use std::num::NonZeroU64;
use std::sync::atomic::AtomicBool;

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::TransformKind;
use jxl_wgpu::{KernelVariant, ResidentStorageBinding, WgpuBackend};
use wgpu::util::DeviceExt;

use crate::entropy::EntropyStreamParams;
use crate::modular_inverse::ModularInverseJob;
use crate::vardct_resource::{VarDctResourceLayout, hf_matrix_param_index};
use crate::vardct_side_image::RawHfDequantSideImagePlan;
use crate::{Error, Result};

use super::execution::{
    FixedGradientOutputMode, align16, encode_modular_inverse_jobs, lz77_scratch_words,
    modular_execution_state_bytes,
};
use super::pipeline::{create_decode_pipeline, shader_source};
use super::types::{
    DecodeStatus, DispatchControl, F64OutputPath, ModularInversePipelineCache,
    ModularReconstructionSpecialization, OutputWritePath, STATUS_OK, ShaderParams,
};

const OVERLAY_SHADER: &str = include_str!("../vardct_raw_matrix.wgsl");
const WINDOW_FIRST: u32 = 1;
const WINDOW_FINAL: u32 = 2;
const ERROR_RAW_MATRIX_VALUE: u32 = 15;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RawMatrixParams {
    denominator: f32,
    width: u32,
    height: u32,
    target_count: u32,
    plane_offsets: [u32; 4],
    plane_strides: [u32; 4],
    target_offsets: [u32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<RawMatrixParams>() == 64);
    assert!(std::mem::align_of::<RawMatrixParams>() == 16);
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawHfDequantSideImageStatus {
    pub(crate) code: u32,
    pub(crate) decoded_samples: u32,
    pub(crate) cursor: u32,
    pub(crate) expected_cursor: u32,
}

pub(crate) struct RawHfDequantSideImagePipeline {
    decode: wgpu::ComputePipeline,
    overlay: wgpu::ComputePipeline,
    inverse: ModularInversePipelineCache,
    variant: KernelVariant,
}

impl RawHfDequantSideImagePipeline {
    pub(crate) fn new(backend: &WgpuBackend, variant: KernelVariant) -> Self {
        let decode = create_decode_pipeline(
            backend,
            "jxl-wgpu raw HF dequant Modular side image",
            &shader_source(
                F64OutputPath::ExactF32Widening,
                OutputWritePath::WordAligned,
                ModularReconstructionSpecialization::DescriptorMetaAdaptive,
            ),
            variant,
            true,
        );
        let module = backend
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("jxl-wgpu raw HF dequant matrix overlay"),
                source: wgpu::ShaderSource::Wgsl(OVERLAY_SHADER.into()),
            });
        let (workgroup_x, _) = variant.workgroup_size();
        let overlay = backend
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu raw HF dequant matrix overlay"),
                layout: None,
                module: &module,
                entry_point: Some("overlay"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("wg_x", f64::from(workgroup_x))],
                    ..Default::default()
                },
                cache: None,
            });
        Self {
            decode,
            overlay,
            inverse: ModularInversePipelineCache::default(),
            variant,
        }
    }

    pub(crate) fn prepare(
        &self,
        backend: &WgpuBackend,
        codestream: &wgpu::Buffer,
        resources: &wgpu::Buffer,
        resource_layout: VarDctResourceLayout,
        plan: &RawHfDequantSideImagePlan,
        packet_end: u32,
    ) -> Result<RawHfDequantSideImageJob> {
        if plan.token_bit_offset > packet_end {
            return Err(Error::EngineContract(
                "raw HF dequant entropy starts after its packet",
            ));
        }
        let device = backend.device();
        let (metadata, channel_layout_offset) = packed_metadata(plan)?;
        let workspace = workspace(plan)?;
        let memory_bytes = frame_bytes(plan, metadata.len(), workspace.bytes)?;
        validate_limits(device, metadata.len(), workspace.bytes)?;

        let source_mask = if plan.bit_depth == 32 {
            u32::MAX
        } else {
            1_u32
                .checked_shl(plan.bit_depth)
                .and_then(|value| value.checked_sub(1))
                .ok_or_else(|| Error::backend("raw HF dequant bit depth exceeds WGSL u32"))?
        };
        let mut params = <ShaderParams as Zeroable>::zeroed();
        params.entropy = EntropyStreamParams {
            token_start: plan.token_bit_offset,
            token_end: packet_end,
            lz77_window_mask: plan.lz77_window_words.saturating_sub(1),
        };
        params.stream_token_end = packet_end;
        params.window_yield_end = packet_end;
        params.window_flags = WINDOW_FIRST | WINDOW_FINAL;
        params.entropy_state_offset = workspace.entropy_state_offset_words;
        params.width = plan.maximum_width;
        params.height = 1;
        params.sample_count = plan.decoded_words;
        params.source_channels = 3;
        params.channel_layout_offset = channel_layout_offset;
        params.source_bits = plan.bit_depth;
        params.source_mask = source_mask;
        params.needs_self_correcting = u32::from(plan.needs_self_correcting);
        params.stream_index = plan.stream_index;
        params.fixed_output_mode = FixedGradientOutputMode::CursorContinuation as u32;
        params.wp_p1 = plan.wp_header.p1;
        params.wp_p2 = plan.wp_header.p2;
        params.wp_p3a = plan.wp_header.p3a;
        params.wp_p3b = plan.wp_header.p3b;
        params.wp_p3c = plan.wp_header.p3c;
        params.wp_p3d = plan.wp_header.p3d;
        params.wp_p3e = plan.wp_header.p3e;
        params.wp_w0 = plan.wp_header.w0;
        params.wp_w1 = plan.wp_header.w1;
        params.wp_w2 = plan.wp_header.w2;
        params.wp_w3 = plan.wp_header.w3;
        let control = DispatchControl {
            first_group: 0,
            group_count: 1,
            lane_stride_words: u32::try_from(workspace.bytes / 4)
                .map_err(|_| Error::backend("raw HF dequant workspace exceeds WGSL u32"))?,
            _padding: 0,
        };
        let overlay_params = overlay_params(resource_layout, plan)?;

        let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu raw HF dequant Modular metadata"),
            contents: bytemuck::cast_slice(&metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu raw HF dequant resident arena"),
            size: workspace.bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu raw HF dequant unused output"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu raw HF dequant status"),
            size: std::mem::size_of::<DecodeStatus>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let status_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu raw HF dequant status readback"),
            size: std::mem::size_of::<DecodeStatus>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu raw HF dequant decode parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu raw HF dequant dispatch control"),
            contents: bytemuck::bytes_of(&control),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let overlay_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu raw HF dequant overlay parameters"),
            contents: bytemuck::bytes_of(&overlay_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let decode_binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu raw HF dequant Modular bindings"),
            layout: &self.decode.get_bind_group_layout(0),
            entries: &[
                entry(0, codestream),
                entry(1, &metadata),
                entry(2, &arena),
                entry(3, &dummy_output),
                entry(4, &status),
                entry(5, &params),
                entry(7, &control),
            ],
        });
        let overlay_binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu raw HF dequant overlay bindings"),
            layout: &self.overlay.get_bind_group_layout(0),
            entries: &[
                entry(0, &arena),
                entry(1, resources),
                entry(2, &status),
                entry(3, &overlay_params),
            ],
        });
        let needs_palette = plan
            .inverse_plan
            .jobs()
            .iter()
            .any(|job| matches!(job, ModularInverseJob::Palette { .. }));
        let needs_squeeze = plan
            .inverse_plan
            .jobs()
            .iter()
            .any(|job| matches!(job, ModularInverseJob::Squeeze { .. }));
        let needs_rct = plan
            .inverse_plan
            .jobs()
            .iter()
            .any(|job| matches!(job, ModularInverseJob::Rct { .. }));
        let inverse = self.inverse.get(
            backend,
            F64OutputPath::ExactF32Widening,
            needs_palette,
            needs_squeeze,
            needs_rct,
        )?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu raw HF dequant side-image stage"),
        });
        encoder.clear_buffer(&arena, 0, None);
        encoder.clear_buffer(&dummy_output, 0, None);
        encoder.clear_buffer(&status, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu raw HF dequant Modular decode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.decode);
            pass.set_bind_group(0, &decode_binding, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let inverse_uniforms = encode_modular_inverse_jobs(
            device,
            &mut encoder,
            ResidentStorageBinding {
                buffer: &arena,
                offset: 0,
                size: NonZeroU64::new(plan.inverse_plan.arena_bytes()).ok_or(
                    Error::EngineContract("raw HF dequant inverse arena is empty"),
                )?,
            },
            &plan.inverse_plan,
            plan.wp_header,
            &inverse,
        )?;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu raw HF dequant matrix overlay"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.overlay);
            pass.set_bind_group(0, &overlay_binding, &[]);
            let sample_count = plan.final_planes[0]
                .width
                .checked_mul(plan.final_planes[0].height)
                .ok_or_else(|| Error::backend("raw HF dequant overlay extent overflow"))?;
            pass.dispatch_workgroups(sample_count.div_ceil(self.variant.workgroup_size().0), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &status,
            0,
            &status_staging,
            0,
            std::mem::size_of::<DecodeStatus>() as u64,
        );

        Ok(RawHfDequantSideImageJob {
            commands: Some(encoder.finish()),
            status_staging,
            status_mapped: AtomicBool::new(false),
            memory_bytes,
            _metadata: metadata,
            _arena: arena,
            _dummy_output: dummy_output,
            _status: status,
            _params: params,
            _control: control,
            _overlay_params: overlay_params,
            _inverse_uniforms: inverse_uniforms,
        })
    }

    pub(crate) fn memory_bytes(&self, plan: &RawHfDequantSideImagePlan) -> Result<u64> {
        let (metadata, _) = packed_metadata(plan)?;
        let workspace = workspace(plan)?;
        frame_bytes(plan, metadata.len(), workspace.bytes)
    }
}

pub(crate) struct RawHfDequantSideImageJob {
    commands: Option<wgpu::CommandBuffer>,
    status_staging: wgpu::Buffer,
    status_mapped: AtomicBool,
    memory_bytes: u64,
    _metadata: wgpu::Buffer,
    _arena: wgpu::Buffer,
    _dummy_output: wgpu::Buffer,
    _status: wgpu::Buffer,
    _params: wgpu::Buffer,
    _control: wgpu::Buffer,
    _overlay_params: wgpu::Buffer,
    _inverse_uniforms: Vec<wgpu::Buffer>,
}

impl RawHfDequantSideImageJob {
    pub(crate) const fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    pub(crate) fn take_commands(&mut self) -> Result<wgpu::CommandBuffer> {
        self.commands.take().ok_or(Error::EngineContract(
            "raw HF dequant side-image commands were consumed twice",
        ))
    }

    pub(crate) const fn status_staging(&self) -> &wgpu::Buffer {
        &self.status_staging
    }

    pub(crate) fn mark_status_mapped(&self) {
        self.status_mapped
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub(crate) fn finish_status(&self) -> Result<RawHfDequantSideImageStatus> {
        let mapped = self
            .status_staging
            .slice(..)
            .get_mapped_range()
            .map_err(Error::backend)?;
        let status = mapped
            .get(..std::mem::size_of::<DecodeStatus>())
            .and_then(|bytes| bytemuck::try_pod_read_unaligned::<DecodeStatus>(bytes).ok())
            .ok_or(Error::EngineContract(
                "raw HF dequant status has an invalid ABI",
            ))?;
        drop(mapped);
        self.status_staging.unmap();
        self.status_mapped
            .store(false, std::sync::atomic::Ordering::Release);
        Ok(RawHfDequantSideImageStatus {
            code: status.code,
            decoded_samples: status.decoded_samples,
            cursor: status.cursor,
            expected_cursor: status.expected_cursor,
        })
    }
}

impl Drop for RawHfDequantSideImageJob {
    fn drop(&mut self) {
        if self
            .status_mapped
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            self.status_staging.unmap();
        }
    }
}

#[derive(Clone, Copy)]
struct Workspace {
    bytes: u64,
    entropy_state_offset_words: u32,
}

fn workspace(plan: &RawHfDequantSideImagePlan) -> Result<Workspace> {
    let predictor_words = if plan.needs_self_correcting {
        u64::from(plan.maximum_width)
            .checked_mul(5)
            .ok_or_else(|| Error::backend("raw HF dequant predictor workspace overflow"))?
    } else {
        0
    };
    let working_words = u64::from(plan.inverse_plan.arena_words())
        .checked_add(predictor_words)
        .and_then(|words| words.checked_add(u64::from(lz77_scratch_words(plan.lz77_window_words))))
        .ok_or_else(|| Error::backend("raw HF dequant workspace overflow"))?;
    let aligned_bytes = align16(
        working_words
            .checked_mul(4)
            .ok_or_else(|| Error::backend("raw HF dequant workspace byte overflow"))?,
    )?;
    let execution_state_bytes = modular_execution_state_bytes(
        ModularReconstructionSpecialization::DescriptorMetaAdaptive,
        plan.needs_self_correcting,
    );
    Ok(Workspace {
        bytes: aligned_bytes
            .checked_add(execution_state_bytes)
            .ok_or_else(|| Error::backend("raw HF dequant execution state overflow"))?,
        entropy_state_offset_words: u32::try_from(aligned_bytes / 4)
            .map_err(|_| Error::backend("raw HF dequant state offset exceeds WGSL u32"))?,
    })
}

fn packed_metadata(plan: &RawHfDequantSideImagePlan) -> Result<(Vec<u32>, u32)> {
    let mut metadata = plan.metadata.clone();
    let channel_layout_offset = plan.channel_metadata.append_to(
        &mut metadata,
        plan.inverse_plan.arena_words(),
        &plan.final_planes,
    )?;
    Ok((metadata, channel_layout_offset))
}

fn frame_bytes(
    plan: &RawHfDequantSideImagePlan,
    metadata_words: usize,
    workspace_bytes: u64,
) -> Result<u64> {
    let metadata_bytes = u64::try_from(metadata_words)
        .ok()
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| Error::backend("raw HF dequant metadata byte size overflow"))?;
    let inverse_uniform_bytes = plan
        .inverse_plan
        .jobs()
        .iter()
        .try_fold(0_u64, |total, job| {
            let bytes = match *job {
                ModularInverseJob::Squeeze { .. } => {
                    std::mem::size_of::<crate::modular_squeeze::ModularSqueezeParams>() as u64
                }
                ModularInverseJob::Rct { .. } => {
                    std::mem::size_of::<crate::modular_rct::ModularRctParams>() as u64
                }
                ModularInverseJob::Palette { job } => job.uniform_bytes(),
            };
            total
                .checked_add(bytes)
                .ok_or_else(|| Error::backend("raw HF dequant inverse uniform overflow"))
        })?;
    [
        metadata_bytes,
        workspace_bytes,
        4,
        std::mem::size_of::<DecodeStatus>() as u64,
        std::mem::size_of::<DecodeStatus>() as u64,
        std::mem::size_of::<ShaderParams>() as u64,
        std::mem::size_of::<DispatchControl>() as u64,
        std::mem::size_of::<RawMatrixParams>() as u64,
        inverse_uniform_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or_else(|| Error::backend("raw HF dequant frame byte total overflow"))
    })
}

fn validate_limits(
    device: &wgpu::Device,
    metadata_words: usize,
    workspace_bytes: u64,
) -> Result<()> {
    let limits = device.limits();
    let metadata_bytes = u64::try_from(metadata_words)
        .ok()
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| Error::backend("raw HF dequant metadata binding size overflow"))?;
    for (name, bytes) in [
        ("raw HF dequant Modular metadata", metadata_bytes),
        ("raw HF dequant resident arena", workspace_bytes),
    ] {
        if bytes > limits.max_buffer_size || bytes > limits.max_storage_buffer_binding_size {
            return Err(Error::backend(format!(
                "{name} needs {bytes} bytes beyond the device storage limit"
            )));
        }
    }
    Ok(())
}

fn overlay_params(
    resource_layout: VarDctResourceLayout,
    plan: &RawHfDequantSideImagePlan,
) -> Result<RawMatrixParams> {
    let mut target_offsets = [0_u32; 4];
    let mut target_count = 0_usize;
    for (index, transform) in TransformKind::ALL.into_iter().enumerate() {
        if hf_matrix_param_index(transform) == plan.matrix_index {
            let target = target_offsets
                .get_mut(target_count)
                .ok_or(Error::EngineContract(
                    "raw HF dequant matrix has too many resource targets",
                ))?;
            *target = resource_layout.matrix_offsets[index];
            target_count += 1;
        }
    }
    if target_count == 0 {
        return Err(Error::EngineContract(
            "raw HF dequant matrix has no resource target",
        ));
    }
    Ok(RawMatrixParams {
        denominator: plan.denominator,
        width: plan.final_planes[0].width,
        height: plan.final_planes[0].height,
        target_count: u32::try_from(target_count)
            .map_err(|_| Error::backend("raw HF dequant target count exceeds WGSL u32"))?,
        plane_offsets: [
            plan.final_planes[0].word_offset,
            plan.final_planes[1].word_offset,
            plan.final_planes[2].word_offset,
            0,
        ],
        plane_strides: [
            plan.final_planes[0].row_stride_words,
            plan.final_planes[1].row_stride_words,
            plan.final_planes[2].row_stride_words,
            0,
        ],
        target_offsets,
    })
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(crate) const fn raw_matrix_value_error(code: u32) -> bool {
    code == ERROR_RAW_MATRIX_VALUE
}

pub(crate) const fn raw_matrix_status_ok(code: u32) -> bool {
    code == STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_shader_and_uniform_validate_semantically() {
        let module = naga::front::wgsl::parse_str(OVERLAY_SHADER).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(std::mem::size_of::<RawMatrixParams>(), 64);
        assert_eq!(std::mem::align_of::<RawMatrixParams>(), 16);
    }

    #[test]
    fn representative_targets_share_one_canonical_raster() {
        let layout = VarDctResourceLayout::new(1, 1, 1).unwrap();
        let expected_counts = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 1, 2, 1, 2, 1, 2];
        for (matrix, expected) in expected_counts.into_iter().enumerate() {
            let actual = TransformKind::ALL
                .into_iter()
                .filter(|&transform| hf_matrix_param_index(transform) == matrix)
                .count();
            assert_eq!(actual, expected);
            let offsets = TransformKind::ALL
                .into_iter()
                .enumerate()
                .filter(|(_, transform)| hf_matrix_param_index(*transform) == matrix)
                .map(|(index, _)| layout.matrix_offsets[index])
                .collect::<Vec<_>>();
            assert_eq!(offsets.len(), expected);
        }
    }
}
