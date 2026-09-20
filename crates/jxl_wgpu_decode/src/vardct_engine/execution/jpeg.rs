//! Restore JPEG integers directly from the validated VarDCT decoder's resident artifacts.

use bytemuck::{Pod, Zeroable};

use super::super::jpeg::{GpuJpegCoefficients, JpegCoefficientLayout, STATUS_BYTES};
use super::*;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RestoreParams {
    lf: [[u32; 4]; 3],
    outputs: [[u32; 4]; 3],
    shifts: [[u32; 4]; 3],
    group: [u32; 4],
    artifact: [u32; 4],
    config: [u32; 4],
}

pub(crate) const RESTORE_UNIFORM_BYTES: u64 = size_of::<RestoreParams>() as u64;
const _: () = {
    assert!(size_of::<RestoreParams>() == 192);
    assert!(align_of::<RestoreParams>() == 16);
};

pub(crate) struct JpegRestorePipeline {
    pipeline: wgpu::ComputePipeline,
}

impl JpegRestorePipeline {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu JPEG integer restoration"),
            source: wgpu::ShaderSource::Wgsl(include_str!("jpeg.wgsl").into()),
        });
        Self {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu JPEG integer restoration"),
                layout: None,
                module: &shader,
                entry_point: Some("restore"),
                compilation_options: Default::default(),
                cache: None,
            }),
        }
    }
}

pub(super) struct JpegRestoreScratch {
    pub(super) status: wgpu::Buffer,
    pub(super) layout: Arc<JpegCoefficientLayout>,
    _uniforms: Vec<wgpu::Buffer>,
}

pub(super) fn encode_restore(
    device: &wgpu::Device,
    commands: &mut wgpu::CommandEncoder,
    pipelines: &VarDctPipelines,
    source: &VarDctSource,
    groups: &[VarDctGroupJobBuffers],
    output: &wgpu::Buffer,
    layout: Arc<JpegCoefficientLayout>,
) -> Result<JpegRestoreScratch, VarDctDecodeError> {
    let pipeline = pipelines
        .jpeg
        .get_or_init(|| JpegRestorePipeline::new(device));
    let status = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu JPEG coefficient status"),
        size: STATUS_BYTES,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut outputs = [[u32::MAX, 0, 0, 0]; 3];
    for plane in layout.planes() {
        outputs[plane.channel as usize] = [
            plane.coefficient_word_offset,
            plane.blocks_per_row,
            plane.block_rows,
            plane.quantization_word_offset,
        ];
    }
    let mut uniforms = Vec::with_capacity(groups.len());
    for ((packet_group, group), buffers) in
        source.packet.groups.iter().zip(&source.groups).zip(groups)
    {
        let [width, height] = packet_group.block_extent();
        let dispatch_width = group
            .artifact_layout
            .task_capacity
            .min(device.limits().max_compute_workgroups_per_dimension);
        if dispatch_width == 0 {
            return Err(VarDctDecodeError::EngineContract {
                detail: "empty JPEG restoration dispatch",
            });
        }
        check_limit(
            "JPEG restoration dispatch rows",
            group.artifact_layout.task_capacity.div_ceil(dispatch_width) as u64,
            device.limits().max_compute_workgroups_per_dimension as u64,
        )?;
        let params = RestoreParams {
            lf: group.resource_params.source_geometry,
            outputs,
            shifts: source
                .packet
                .profile
                .channel_shifts
                .map(|shift| [shift.horizontal, shift.vertical, 0, 0]),
            group: [
                packet_group.rect.x / 8,
                packet_group.rect.y / 8,
                width,
                height,
            ],
            artifact: [
                group.artifact_layout.task_metadata_offset_words,
                group.artifact_layout.status_offset_words,
                group.artifact_layout.task_capacity,
                packet_group.coefficient_words(),
            ],
            config: [
                u32::from(layout.rgb),
                u32::from(layout.cfl),
                u32::from(packet_group.extra_precision()),
                dispatch_width,
            ],
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu JPEG restoration parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu JPEG restoration bindings"),
            layout: &pipeline.pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &buffers.reconstructed),
                entry(1, &buffers.coefficients),
                entry(2, &buffers.artifact),
                entry(3, &buffers.raw_metadata),
                entry(4, output),
                entry(5, &status),
                entry(6, &uniform),
            ],
        });
        let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu restore JPEG coefficients"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(
            dispatch_width,
            group.artifact_layout.task_capacity.div_ceil(dispatch_width),
            1,
        );
        drop(pass);
        uniforms.push(uniform);
    }
    Ok(JpegRestoreScratch {
        status,
        layout,
        _uniforms: uniforms,
    })
}

/// A JPEG coefficient submission with the same staged validation and cancellation lifetime as VarDCT.
#[derive(Debug)]
pub struct JpegCoefficientPending {
    pub(crate) inner: FramePendingFrame,
    pub(crate) layout: Arc<JpegCoefficientLayout>,
}

impl JpegCoefficientPending {
    fn finish(
        &mut self,
        mapping: Result<(), String>,
    ) -> DecodeResult<SubmittedGpuFrame<GpuJpegCoefficients>> {
        let life = self.inner.finish_validated(mapping)?;
        Ok(SubmittedGpuFrame::new(
            FrameMetadata {
                index: 0,
                duration: FrameDuration::still(),
                presentation_ticks: 0,
                timecode: None,
                is_last: true,
                is_keyframe: true,
                name: std::mem::take(&mut self.inner.frame_name),
            },
            GpuJpegCoefficients {
                layout: Arc::clone(&self.layout),
                buffer: life.output.clone(),
            },
        ))
    }
}

impl GpuPendingFrame for JpegCoefficientPending {
    type Frame = GpuJpegCoefficients;
    #[cfg(not(target_arch = "wasm32"))]
    fn wait(mut self) -> DecodeResult<SubmittedGpuFrame<Self::Frame>> {
        loop {
            let Some(completion) = self.inner.stage_completion() else {
                self.inner.resume_after_dc()?;
                continue;
            };
            let mapping = completion.wait();
            if self.inner.dependency_submission_ready() {
                return self.finish(mapping);
            }
            self.inner.advance_staged_packet(mapping)?;
        }
    }
    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<DecodeResult<SubmittedGpuFrame<Self::Frame>>> {
        self.inner
            .backend
            .device()
            .poll(wgpu::PollType::Poll)
            .map_err(DecodeError::backend)?;
        let Some(completion) = self.inner.stage_completion() else {
            self.inner.resume_after_dc()?;
            context.waker().wake_by_ref();
            return Poll::Pending;
        };
        let Some(mapping) = completion.poll(context) else {
            return Poll::Pending;
        };
        if self.inner.dependency_submission_ready() {
            return Poll::Ready(self.finish(mapping));
        }
        self.inner.advance_staged_packet(mapping)?;
        context.waker().wake_by_ref();
        Poll::Pending
    }
}

#[cfg(test)]
mod tests;
