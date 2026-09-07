use std::num::NonZeroU64;
use std::sync::atomic::AtomicBool;

use bytemuck::Zeroable;

use jxl_wgpu::{KernelVariant, ResidentStorageBinding, WgpuBackend};
use wgpu::util::DeviceExt;

use crate::entropy::EntropyStreamParams;
use crate::modular_inverse::ModularInverseJob;

use crate::modular_side_image::ModularSideImagePlan;
use crate::{Error, Result};

use super::super::execution::{
    FixedGradientOutputMode, align16, encode_modular_inverse_jobs, lz77_scratch_words,
    modular_execution_state_bytes,
};
use super::super::pipeline::{create_decode_pipeline, shader_source};
use super::super::types::{
    DecodeStatus, DispatchControl, F64OutputPath, ModularInversePipelineCache,
    ModularReconstructionSpecialization, OutputWritePath, ShaderParams,
};

const WINDOW_FIRST: u32 = 1;
const WINDOW_FINAL: u32 = 2;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ModularSideImageStatus {
    pub(crate) code: u32,
    pub(crate) decoded_samples: u32,
    pub(crate) cursor: u32,
    pub(crate) expected_cursor: u32,
}

pub(crate) struct ModularSideImagePipeline {
    decode: wgpu::ComputePipeline,
    inverse: ModularInversePipelineCache,
}

impl ModularSideImagePipeline {
    pub(crate) fn new(backend: &WgpuBackend, variant: KernelVariant) -> Self {
        let decode = create_decode_pipeline(
            backend,
            "jxl-wgpu Modular side-image decode",
            &shader_source(
                F64OutputPath::ExactF32Widening,
                OutputWritePath::WordAligned,
                ModularReconstructionSpecialization::DescriptorMetaAdaptive,
            ),
            variant,
            true,
        );
        Self {
            decode,
            inverse: ModularInversePipelineCache::default(),
        }
    }

    pub(crate) fn record(
        &self,
        backend: &WgpuBackend,
        codestream: &wgpu::Buffer,
        plan: &ModularSideImagePlan,
        packet_end: u32,
    ) -> Result<ModularSideImageRecording> {
        if plan.token_bit_offset > packet_end {
            return Err(Error::EngineContract(
                "Modular side image entropy starts after its packet",
            ));
        }
        let device = backend.device();
        let stream = stream_window(codestream, plan.token_bit_offset, packet_end)?;
        let (metadata, channel_layout_offset) = packed_metadata(plan)?;
        let workspace = workspace(plan)?;
        let memory_bytes = frame_bytes(plan, metadata.len(), workspace.bytes, stream.bytes)?;
        validate_limits(device, metadata.len(), workspace.bytes, stream.bytes)?;

        let source_mask = if plan.bit_depth == 32 {
            u32::MAX
        } else {
            1_u32
                .checked_shl(plan.bit_depth)
                .and_then(|value| value.checked_sub(1))
                .ok_or_else(|| Error::backend("Modular side image bit depth exceeds WGSL u32"))?
        };
        let mut params = <ShaderParams as Zeroable>::zeroed();
        params.entropy = EntropyStreamParams {
            token_start: stream.token_start,
            token_end: stream.token_end,
            lz77_window_mask: plan.lz77_window_words.saturating_sub(1),
        };
        params.stream_token_end = stream.token_end;
        params.window_yield_end = stream.token_end;
        params.window_flags = WINDOW_FIRST | WINDOW_FINAL;
        params.entropy_state_offset = workspace.entropy_state_offset_words;
        params.width = plan.maximum_width;
        params.height = 1;
        params.sample_count = plan.decoded_words;
        params.source_channels = u32::try_from(plan.final_planes.len())
            .map_err(|_| Error::backend("Modular side-image channel count exceeds u32"))?;
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
                .map_err(|_| Error::backend("Modular side image workspace exceeds WGSL u32"))?,
            _padding: 0,
        };

        let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu Modular side-image metadata"),
            contents: bytemuck::cast_slice(&metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let stream_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu Modular side image bounded stream"),
            size: stream.bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu Modular side image resident arena"),
            size: workspace.bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu Modular side image unused output"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu Modular side image status"),
            size: std::mem::size_of::<DecodeStatus>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let status_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu Modular side image status readback"),
            size: std::mem::size_of::<DecodeStatus>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu Modular side image decode parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu Modular side image dispatch control"),
            contents: bytemuck::bytes_of(&control),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let decode_binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu Modular side-image bindings"),
            layout: &self.decode.get_bind_group_layout(0),
            entries: &[
                entry(0, &stream_buffer),
                entry(1, &metadata),
                entry(2, &arena),
                entry(3, &dummy_output),
                entry(4, &status),
                entry(5, &params),
                entry(7, &control),
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
            label: Some("jxl-wgpu Modular side-image stage"),
        });
        encoder.copy_buffer_to_buffer(
            codestream,
            stream.source_offset,
            &stream_buffer,
            0,
            stream.bytes,
        );
        encoder.clear_buffer(&arena, 0, None);
        encoder.clear_buffer(&dummy_output, 0, None);
        encoder.clear_buffer(&status, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu Modular side-image decode"),
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
                    Error::EngineContract("Modular side image inverse arena is empty"),
                )?,
            },
            &plan.inverse_plan,
            plan.wp_header,
            &inverse,
        )?;
        Ok(ModularSideImageRecording {
            encoder,
            job: ModularSideImageJob {
                commands: None,
                status_staging,
                status_mapped: AtomicBool::new(false),
                memory_bytes,
                cursor_base_bits: stream.cursor_base_bits,
                _stream: stream_buffer,
                _metadata: metadata,
                arena,
                _dummy_output: dummy_output,
                status,
                _params: params,
                _control: control,
                _uniforms: inverse_uniforms,
            },
        })
    }

    pub(crate) fn memory_bytes(&self, plan: &ModularSideImagePlan, packet_end: u32) -> Result<u64> {
        let (metadata, _) = packed_metadata(plan)?;
        let workspace = workspace(plan)?;
        let stream = stream_window_geometry(plan.token_bit_offset, packet_end)?;
        frame_bytes(plan, metadata.len(), workspace.bytes, stream.bytes)
    }
}

pub(crate) struct ModularSideImageJob {
    commands: Option<wgpu::CommandBuffer>,
    status_staging: wgpu::Buffer,
    status_mapped: AtomicBool,
    memory_bytes: u64,
    cursor_base_bits: u32,
    _stream: wgpu::Buffer,
    _metadata: wgpu::Buffer,
    arena: wgpu::Buffer,
    _dummy_output: wgpu::Buffer,
    status: wgpu::Buffer,
    _params: wgpu::Buffer,
    _control: wgpu::Buffer,
    _uniforms: Vec<wgpu::Buffer>,
}

impl ModularSideImageJob {
    pub(crate) const fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    pub(crate) fn take_commands(&mut self) -> Result<wgpu::CommandBuffer> {
        self.commands.take().ok_or(Error::EngineContract(
            "Modular side-image commands were consumed twice",
        ))
    }

    pub(crate) const fn status_staging(&self) -> &wgpu::Buffer {
        &self.status_staging
    }

    pub(crate) fn mark_status_mapped(&self) {
        self.status_mapped
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub(crate) fn finish_status(&self) -> Result<ModularSideImageStatus> {
        let mapped = self
            .status_staging
            .slice(..)
            .get_mapped_range()
            .map_err(Error::backend)?;
        let status = mapped
            .get(..std::mem::size_of::<DecodeStatus>())
            .and_then(|bytes| bytemuck::try_pod_read_unaligned::<DecodeStatus>(bytes).ok())
            .ok_or(Error::EngineContract(
                "Modular side image status has an invalid ABI",
            ))?;
        drop(mapped);
        self.status_staging.unmap();
        self.status_mapped
            .store(false, std::sync::atomic::Ordering::Release);
        Ok(ModularSideImageStatus {
            code: status.code,
            decoded_samples: status.decoded_samples,
            cursor: status
                .cursor
                .checked_add(self.cursor_base_bits)
                .ok_or_else(|| Error::backend("Modular side image cursor rebasing overflow"))?,
            expected_cursor: status
                .expected_cursor
                .checked_add(self.cursor_base_bits)
                .ok_or_else(|| {
                    Error::backend("Modular side image expected-cursor rebasing overflow")
                })?,
        })
    }
}

impl Drop for ModularSideImageJob {
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
struct StreamWindow {
    source_offset: u64,
    bytes: u64,
    cursor_base_bits: u32,
    token_start: u32,
    token_end: u32,
}

fn stream_window(
    codestream: &wgpu::Buffer,
    token_start: u32,
    token_end: u32,
) -> Result<StreamWindow> {
    let window = stream_window_geometry(token_start, token_end)?;
    let source_end = window
        .source_offset
        .checked_add(window.bytes)
        .ok_or_else(|| Error::backend("Modular side image stream range overflow"))?;
    if source_end > codestream.size() {
        return Err(Error::backend(
            "Modular side image stream range exceeds the retained codestream",
        ));
    }
    Ok(window)
}

fn stream_window_geometry(token_start: u32, token_end: u32) -> Result<StreamWindow> {
    if token_start >= token_end {
        return Err(Error::EngineContract(
            "Modular side image entropy range is empty or reversed",
        ));
    }
    let first_word = token_start / 32;
    let end_word = token_end.div_ceil(32);
    let word_count = end_word
        .checked_sub(first_word)
        .ok_or_else(|| Error::backend("Modular side image stream word range underflow"))?;
    let bytes = u64::from(word_count)
        .checked_mul(4)
        .ok_or_else(|| Error::backend("Modular side image stream byte size overflow"))?;
    let source_offset = u64::from(first_word)
        .checked_mul(4)
        .ok_or_else(|| Error::backend("Modular side image stream byte offset overflow"))?;
    let cursor_base_bits = first_word
        .checked_mul(32)
        .ok_or_else(|| Error::backend("Modular side image cursor base overflow"))?;
    Ok(StreamWindow {
        source_offset,
        bytes,
        cursor_base_bits,
        token_start: token_start - cursor_base_bits,
        token_end: token_end - cursor_base_bits,
    })
}

#[derive(Clone, Copy)]
struct Workspace {
    bytes: u64,
    entropy_state_offset_words: u32,
}

fn workspace(plan: &ModularSideImagePlan) -> Result<Workspace> {
    let predictor_words = if plan.needs_self_correcting {
        u64::from(plan.maximum_width)
            .checked_mul(5)
            .ok_or_else(|| Error::backend("Modular side image predictor workspace overflow"))?
    } else {
        0
    };
    let working_words = u64::from(plan.inverse_plan.arena_words())
        .checked_add(predictor_words)
        .and_then(|words| words.checked_add(u64::from(lz77_scratch_words(plan.lz77_window_words))))
        .ok_or_else(|| Error::backend("Modular side image workspace overflow"))?;
    let aligned_bytes = align16(
        working_words
            .checked_mul(4)
            .ok_or_else(|| Error::backend("Modular side image workspace byte overflow"))?,
    )?;
    let execution_state_bytes = modular_execution_state_bytes(
        ModularReconstructionSpecialization::DescriptorMetaAdaptive,
        plan.needs_self_correcting,
    );
    Ok(Workspace {
        bytes: aligned_bytes
            .checked_add(execution_state_bytes)
            .ok_or_else(|| Error::backend("Modular side image execution state overflow"))?,
        entropy_state_offset_words: u32::try_from(aligned_bytes / 4)
            .map_err(|_| Error::backend("Modular side image state offset exceeds WGSL u32"))?,
    })
}

fn packed_metadata(plan: &ModularSideImagePlan) -> Result<(Vec<u32>, u32)> {
    let mut metadata = plan.metadata.clone();
    let channel_layout_offset = plan.channel_metadata.append_to(
        &mut metadata,
        plan.inverse_plan.arena_words(),
        &plan.final_planes,
    )?;
    Ok((metadata, channel_layout_offset))
}

fn frame_bytes(
    plan: &ModularSideImagePlan,
    metadata_words: usize,
    workspace_bytes: u64,
    stream_bytes: u64,
) -> Result<u64> {
    let metadata_bytes = u64::try_from(metadata_words)
        .ok()
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| Error::backend("Modular side image metadata byte size overflow"))?;
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
                .ok_or_else(|| Error::backend("Modular side image inverse uniform overflow"))
        })?;
    [
        stream_bytes,
        metadata_bytes,
        workspace_bytes,
        4,
        std::mem::size_of::<DecodeStatus>() as u64,
        std::mem::size_of::<DecodeStatus>() as u64,
        std::mem::size_of::<ShaderParams>() as u64,
        std::mem::size_of::<DispatchControl>() as u64,
        inverse_uniform_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or_else(|| Error::backend("Modular side image frame byte total overflow"))
    })
}

fn validate_limits(
    device: &wgpu::Device,
    metadata_words: usize,
    workspace_bytes: u64,
    stream_bytes: u64,
) -> Result<()> {
    let limits = device.limits();
    let metadata_bytes = u64::try_from(metadata_words)
        .ok()
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| Error::backend("Modular side image metadata binding size overflow"))?;
    for (name, bytes) in [
        ("Modular side image bounded stream", stream_bytes),
        ("Modular side-image metadata", metadata_bytes),
        ("Modular side image resident arena", workspace_bytes),
    ] {
        if bytes > limits.max_buffer_size || bytes > limits.max_storage_buffer_binding_size {
            return Err(Error::backend(format!(
                "{name} needs {bytes} bytes beyond the device storage limit"
            )));
        }
    }
    Ok(())
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(crate) struct ModularSideImageRecording {
    pub(super) encoder: wgpu::CommandEncoder,
    job: ModularSideImageJob,
}

impl ModularSideImageRecording {
    pub(super) fn arena(&self) -> &wgpu::Buffer {
        &self.job.arena
    }
    pub(super) fn status(&self) -> &wgpu::Buffer {
        &self.job.status
    }
    pub(super) fn retain_uniform(&mut self, uniform: wgpu::Buffer) -> Result<()> {
        self.job.memory_bytes = self
            .job
            .memory_bytes
            .checked_add(uniform.size())
            .ok_or_else(|| Error::backend("Modular side-image retained uniform bytes overflow"))?;
        self.job._uniforms.push(uniform);
        Ok(())
    }
    pub(crate) fn finish(mut self) -> ModularSideImageJob {
        self.encoder.copy_buffer_to_buffer(
            &self.job.status,
            0,
            &self.job.status_staging,
            0,
            std::mem::size_of::<DecodeStatus>() as u64,
        );
        self.job.commands = Some(self.encoder.finish());
        self.job
    }
}

#[cfg(test)]
mod tests;
