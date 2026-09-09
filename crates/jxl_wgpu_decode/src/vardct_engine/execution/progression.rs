//! Immutable pass images, copied validation evidence, and queue submission at image-wide barriers.

use super::*;

pub(super) struct IntermediatePermits {
    pub(super) output: MemoryPermit,
    pub(super) transient: MemoryPermit,
    pub(super) poll: SubmissionPollPermit,
}

pub(super) struct IntermediateFrame {
    output: GpuBufferLease,
    status: wgpu::Buffer,
    mapped: AtomicBool,
    completion: Arc<MapCompletion>,
    progression: crate::FrameProgression,
    spatial_groups: u64,
    _render: render::FrameRenderResult,
    _transient: MemoryPermit,
}

impl Drop for IntermediateFrame {
    fn drop(&mut self) {
        if self.mapped.swap(false, Ordering::AcqRel) {
            self.status.unmap();
        }
    }
}

pub(super) struct IntermediateCommands {
    commands: wgpu::CommandBuffer,
    poll: SubmissionPollPermit,
}

pub(super) enum PassCommands {
    Whole(Vec<wgpu::CommandBuffer>),
    Windowed(HfCoefficientWindowCommands),
}

pub(super) struct IntermediateRenderInputs<'a> {
    pub(super) source: &'a VarDctSource,
    pub(super) groups: &'a [VarDctGroupJobBuffers],
    pub(super) hf: Option<&'a HfCoefficientJobBuffers>,
    pub(super) resources: &'a wgpu::Buffer,
    pub(super) planes: Option<&'a [wgpu::Buffer; 3]>,
    pub(super) post: &'a PostTransformJobBuffers,
}

pub(super) fn record_intermediates(
    backend: &WgpuBackend,
    pipelines: &VarDctPipelines,
    inputs: IntermediateRenderInputs<'_>,
    permits: Vec<IntermediatePermits>,
) -> Result<(Vec<Arc<IntermediateFrame>>, Vec<IntermediateCommands>), VarDctDecodeError> {
    let source = inputs.source;
    if permits.len() != source.intermediate_passes.len() {
        return Err(VarDctDecodeError::EngineContract {
            detail: "intermediate output admission differs from the pass schedule",
        });
    }
    let device = backend.device();
    let mut frames = Vec::with_capacity(permits.len());
    let mut recordings = Vec::with_capacity(permits.len());
    for (progression, mut permits) in source.intermediate_passes.iter().copied().zip(permits) {
        let hf = inputs.hf.ok_or(VarDctDecodeError::EngineContract {
            detail: "intermediate pass image has no coefficient buffers",
        })?;
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        if backend.direct_readback_enabled() {
            usage |= wgpu::BufferUsages::MAP_READ;
        }
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu immutable intermediate output"),
            size: source.memory.output_lease_bytes,
            usage,
            mapped_at_creation: false,
        });
        let status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu intermediate validation snapshot"),
            size: source.memory.validation_staging_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu render a completed coefficient pass"),
        });
        let rendered = render::encode_frame_render(
            device,
            &mut commands,
            pipelines,
            render::FrameRenderInputs {
                source,
                group_buffers: inputs.groups,
                resources: inputs.resources,
                output: &output,
                resident_planes: inputs.planes,
                rendered_extra: None,
                post_transform: PostTransformJobBuffers {
                    _pre_restoration_planes: inputs.post._pre_restoration_planes.clone(),
                    _restoration_planes: inputs.post._restoration_planes.clone(),
                    _frame_upsample_planes: inputs.post._frame_upsample_planes.clone(),
                    _frame_upsample_weights: source
                        .frame_upsample
                        .as_ref()
                        .map(|k| k.upload(device))
                        .transpose()?,
                    _epf_sigma: inputs.post._epf_sigma.clone(),
                    ..Default::default()
                },
                transient_permit: &mut permits.transient,
            },
        )?;
        let packet_bytes = source.memory.packet_status_bytes;
        for (index, (group, buffers)) in source.groups.iter().zip(inputs.groups).enumerate() {
            commands.copy_buffer_to_buffer(
                &buffers.packet_status,
                0,
                &status,
                index as u64 * PACKET_STATUS_BYTES,
                PACKET_STATUS_BYTES,
            );
            commands.copy_buffer_to_buffer(
                &buffers.artifact,
                u64::from(group.artifact_layout.status_offset_words) * 4,
                &status,
                packet_bytes + index as u64 * ARTIFACT_STATUS_BYTES,
                ARTIFACT_STATUS_BYTES,
            );
        }
        let mut offset = packet_bytes + inputs.groups.len() as u64 * ARTIFACT_STATUS_BYTES;
        for group in &hf.groups {
            commands.copy_buffer_to_buffer(&group.status, 0, &status, offset, group.status.size());
            offset += group.status.size();
        }
        if offset != status.size() || rendered.lf_planes.is_some() {
            return Err(VarDctDecodeError::EngineContract {
                detail: "intermediate image requires color-only main-frame validation",
            });
        }
        frames.push(Arc::new(IntermediateFrame {
            output: GpuBufferLease::from_tracked(output, permits.output),
            status,
            mapped: AtomicBool::new(false),
            completion: Arc::new(MapCompletion::default()),
            progression,
            spatial_groups: source.packet.profile.group_count,
            _render: rendered,
            _transient: permits.transient,
        }));
        recordings.push(IntermediateCommands {
            commands: commands.finish(),
            poll: permits.poll,
        });
    }
    Ok((frames, recordings))
}

fn arm_snapshot(
    frame: &Arc<IntermediateFrame>,
    poll: SubmissionPollPermit,
    submission: wgpu::SubmissionIndex,
) {
    let callback_frame = Arc::clone(frame);
    frame
        .status
        .map_async(wgpu::MapMode::Read, .., move |result| {
            if result.is_ok() {
                callback_frame.mapped.store(true, Ordering::Release);
            }
            callback_frame
                .completion
                .complete(result.map_err(|error| error.to_string()));
        });
    let completion = Arc::clone(&frame.completion);
    if let Err(error) = poll.register(submission, move |error| completion.complete(Err(error))) {
        frame.completion.complete(Err(format!(
            "intermediate GPU poll registration failed: {error}"
        )));
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn submit_passes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &VarDctPipelines,
    lifetime: &VarDctJobLifetime,
    coefficients: PassCommands,
    intermediate_commands: Vec<IntermediateCommands>,
    mut prefix: Vec<wgpu::CommandBuffer>,
    after: wgpu::CommandBuffer,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    if intermediate_commands.len() != lifetime.intermediates.len() {
        return Err(VarDctDecodeError::EngineContract {
            detail: "intermediate submissions lost a render stage",
        });
    }
    match coefficients {
        PassCommands::Whole(passes) => {
            if passes.len() != intermediate_commands.len() + 1 {
                return Err(VarDctDecodeError::EngineContract {
                    detail: "whole coefficient pass count changed",
                });
            }
            let mut passes = passes.into_iter();
            for (frame, recording) in lifetime.intermediates.iter().zip(intermediate_commands) {
                prefix.push(passes.next().expect("pass count was checked"));
                prefix.push(recording.commands);
                let submission = queue.submit(std::mem::take(&mut prefix));
                arm_snapshot(frame, recording.poll, submission);
            }
            prefix.push(passes.next().expect("one final pass remains"));
            prefix.push(after);
            Ok(queue.submit(prefix))
        }
        PassCommands::Windowed(windows) => {
            queue.submit(prefix);
            let retained = lock_unpoisoned(&lifetime._hf_coefficients);
            let hf = retained.as_ref().ok_or(VarDctDecodeError::EngineContract {
                detail: "intermediate stream lost coefficient buffers",
            })?;
            for (pass, (frame, recording)) in lifetime
                .intermediates
                .iter()
                .zip(intermediate_commands)
                .enumerate()
            {
                windows.submit_pass(device, queue, pipelines, hf, &lifetime._groups, pass, None)?;
                let submission = queue.submit([recording.commands]);
                arm_snapshot(frame, recording.poll, submission);
            }
            windows.submit_pass(
                device,
                queue,
                pipelines,
                hf,
                &lifetime._groups,
                lifetime.intermediates.len(),
                None,
            )?;
            Ok(queue.submit([after]))
        }
    }
}

impl FramePendingFrame {
    pub(super) fn poll_intermediate(
        &mut self,
        context: &Context<'_>,
    ) -> Poll<DecodeResult<crate::SubmittedGpuUpdate<GpuImageFrame>>> {
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let frame = lifetime
            .intermediates
            .get(self.next_intermediate)
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let Some(mapping) = frame.completion.poll(context) else {
            return Poll::Pending;
        };
        mapping.map_err(DecodeError::backend)?;
        let mapped = frame
            .status
            .slice(..)
            .get_mapped_range()
            .map_err(DecodeError::backend)?;
        self.validate_mapped_status(
            lifetime,
            &mapped,
            Some((frame.progression.completed_passes, frame.spatial_groups)),
        )?;
        drop(mapped);
        frame.status.unmap();
        frame.mapped.store(false, Ordering::Release);
        self.next_intermediate += 1;
        Poll::Ready(Ok(crate::SubmittedGpuUpdate::Intermediate {
            progression: frame.progression,
            frame: SubmittedGpuFrame::new(
                FrameMetadata {
                    index: 0,
                    duration: FrameDuration::still(),
                    presentation_ticks: 0,
                    timecode: None,
                    is_last: true,
                    is_keyframe: true,
                    name: self.frame_name.clone(),
                },
                GpuImageFrame {
                    token: self.token,
                    outputs: crate::frame_surface::outputs(&self.layout, None, &frame.output),
                    changed: crate::frame_surface::changed_regions(&self.layout, None),
                },
            ),
        }))
    }
}
