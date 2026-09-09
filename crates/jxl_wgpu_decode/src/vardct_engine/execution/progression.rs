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
    submitted: AtomicBool,
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
    commands: Vec<wgpu::CommandBuffer>,
    needs_hf_status: bool,
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
    if permits.len() != source.intermediate_outputs.len() {
        return Err(VarDctDecodeError::EngineContract {
            detail: "intermediate output admission differs from the pass schedule",
        });
    }
    let device = backend.device();
    let mut frames = Vec::with_capacity(permits.len());
    let mut recordings = Vec::with_capacity(permits.len());
    for (plan, mut permits) in source.intermediate_outputs.iter().zip(permits) {
        let progression = plan.progression;
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
            size: source
                .memory
                .intermediate_validation_bytes(&plan.reconstruction)?,
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
                reconstruction: &plan.reconstruction,
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
        let offset = packet_bytes + inputs.groups.len() as u64 * ARTIFACT_STATUS_BYTES;
        let needs_hf_status = progression
            .completed_passes()
            .expect("coefficient boundary")
            != 0
            && inputs.hf.is_none();
        if progression
            .completed_passes()
            .expect("coefficient boundary")
            != 0
        {
            if let Some(hf) = inputs.hf {
                copy_hf_status(&mut commands, hf, &status, offset)?;
            }
        } else if offset != status.size() {
            return Err(VarDctDecodeError::EngineContract {
                detail: "DC validation size changed",
            });
        }
        if rendered.lf_planes.is_some() {
            return Err(VarDctDecodeError::EngineContract {
                detail: "intermediate image requires color-only main-frame validation",
            });
        }
        frames.push(Arc::new(IntermediateFrame {
            output: GpuBufferLease::from_tracked(output, permits.output),
            status,
            mapped: AtomicBool::new(false),
            submitted: AtomicBool::new(false),
            completion: Arc::new(MapCompletion::default()),
            progression,
            spatial_groups: source.packet.profile.group_count,
            _render: rendered,
            _transient: permits.transient,
        }));
        recordings.push(IntermediateCommands {
            commands: vec![commands.finish()],
            needs_hf_status,
            poll: permits.poll,
        });
    }
    Ok((frames, recordings))
}

fn arm_snapshot(
    frame: &Arc<IntermediateFrame>,
    poll: SubmissionPollPermit,
    submission: wgpu::SubmissionIndex,
    retained_job: Option<Arc<VarDctJobLifetime>>,
) {
    frame.submitted.store(true, Ordering::Release);
    let callback_frame = Arc::clone(frame);
    frame
        .status
        .map_async(wgpu::MapMode::Read, .., move |result| {
            if result.is_ok() {
                callback_frame.mapped.store(true, Ordering::Release);
            }
            // Deferred DC can be the last queued work when its consumer cancels. Preserve the
            // common reconstruction buffers and their permits until that submission completes.
            drop(retained_job);
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

fn copy_hf_status(
    commands: &mut wgpu::CommandEncoder,
    hf: &HfCoefficientJobBuffers,
    status: &wgpu::Buffer,
    mut offset: u64,
) -> Result<(), VarDctDecodeError> {
    for group in &hf.groups {
        commands.copy_buffer_to_buffer(&group.status, 0, status, offset, group.status.size());
        offset += group.status.size();
    }
    if offset != status.size() {
        return Err(VarDctDecodeError::EngineContract {
            detail: "intermediate HF validation size changed",
        });
    }
    Ok(())
}

impl IntermediateFrame {
    pub(super) fn is_submitted(&self) -> bool {
        self.submitted.load(Ordering::Acquire)
    }
}

/// DC has no coefficient dependency; publish it before parsing any deferred HF-global tail.
pub(super) fn submit_dc(
    queue: &wgpu::Queue,
    lifetime: &Arc<VarDctJobLifetime>,
    recording: IntermediateCommands,
    prefix: Option<wgpu::CommandBuffer>,
) -> Result<(), VarDctDecodeError> {
    let frame = lifetime
        .intermediates
        .first()
        .ok_or(VarDctDecodeError::EngineContract {
            detail: "deferred DC has no admitted image",
        })?;
    if frame
        .progression
        .completed_passes()
        .expect("coefficient boundary")
        != 0
        || recording.needs_hf_status
        || frame.is_submitted()
    {
        return Err(VarDctDecodeError::EngineContract {
            detail: "invalid deferred DC boundary",
        });
    }
    let submission = queue.submit(prefix.into_iter().chain(recording.commands));
    arm_snapshot(
        frame,
        recording.poll,
        submission,
        Some(Arc::clone(lifetime)),
    );
    Ok(())
}

/// Descriptor-dependent status buffers become available only after the DC image was submitted.
pub(super) fn attach_hf_status(
    device: &wgpu::Device,
    lifetime: &VarDctJobLifetime,
    first: usize,
    hf: &HfCoefficientJobBuffers,
    recordings: &mut [IntermediateCommands],
) -> Result<(), VarDctDecodeError> {
    let frames = lifetime
        .intermediates
        .get(first..)
        .ok_or(VarDctDecodeError::EngineContract {
            detail: "deferred intermediate start is out of range",
        })?;
    if frames.len() != recordings.len() {
        return Err(VarDctDecodeError::EngineContract {
            detail: "deferred intermediate count changed",
        });
    }
    let offset = lifetime._groups.len() as u64 * (PACKET_STATUS_BYTES + ARTIFACT_STATUS_BYTES);
    for (frame, recording) in frames.iter().zip(recordings) {
        if !recording.needs_hf_status {
            continue;
        }
        let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu deferred intermediate coefficient status"),
        });
        copy_hf_status(&mut commands, hf, &frame.status, offset)?;
        recording.commands.push(commands.finish());
        recording.needs_hf_status = false;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn submit_passes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &VarDctPipelines,
    lifetime: &VarDctJobLifetime,
    first_intermediate: usize,
    coefficients: PassCommands,
    intermediate_commands: Vec<IntermediateCommands>,
    mut prefix: Vec<wgpu::CommandBuffer>,
    after: Vec<wgpu::CommandBuffer>,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    let frames = lifetime.intermediates.get(first_intermediate..).ok_or(
        VarDctDecodeError::EngineContract {
            detail: "intermediate submission start is out of range",
        },
    )?;
    let total_passes = match &coefficients {
        PassCommands::Whole(passes) => passes.len(),
        PassCommands::Windowed(windows) => windows.pass_count(),
    };
    let mut previous = None;
    for frame in frames {
        let completed = usize::from(
            frame
                .progression
                .completed_passes()
                .expect("coefficient boundary"),
        );
        if usize::from(
            frame
                .progression
                .total_passes()
                .expect("coefficient boundary"),
        ) != total_passes
            || completed >= total_passes
            || previous.is_some_and(|previous| completed <= previous)
        {
            return Err(VarDctDecodeError::EngineContract {
                detail: "invalid intermediate pass schedule",
            });
        }
        previous = Some(completed);
    }
    if intermediate_commands.len() != frames.len()
        || intermediate_commands
            .iter()
            .any(|commands| commands.needs_hf_status)
    {
        return Err(VarDctDecodeError::EngineContract {
            detail: "intermediate submissions lost a validation stage",
        });
    }
    match coefficients {
        PassCommands::Whole(passes) => {
            let mut passes = passes.into_iter();
            let mut completed = 0;
            for (frame, recording) in frames.iter().zip(intermediate_commands) {
                let target = usize::from(
                    frame
                        .progression
                        .completed_passes()
                        .expect("coefficient boundary"),
                );
                for _ in completed..target {
                    prefix.push(passes.next().ok_or(VarDctDecodeError::EngineContract {
                        detail: "intermediate image exceeds available coefficient passes",
                    })?);
                }
                completed = target;
                prefix.extend(recording.commands);
                let submission = queue.submit(std::mem::take(&mut prefix));
                arm_snapshot(frame, recording.poll, submission, None);
            }
            prefix.extend(passes);
            prefix.extend(after);
            Ok(queue.submit(prefix))
        }
        PassCommands::Windowed(windows) => {
            if !prefix.is_empty() {
                queue.submit(prefix);
            }
            let retained = lock_unpoisoned(&lifetime._hf_coefficients);
            let hf = retained.as_ref().ok_or(VarDctDecodeError::EngineContract {
                detail: "intermediate stream lost coefficient buffers",
            })?;
            let mut completed = 0;
            for (frame, recording) in frames.iter().zip(intermediate_commands) {
                let target = usize::from(
                    frame
                        .progression
                        .completed_passes()
                        .expect("coefficient boundary"),
                );
                for pass in completed..target {
                    windows.submit_pass(
                        device,
                        queue,
                        pipelines,
                        hf,
                        &lifetime._groups,
                        pass,
                        None,
                    )?;
                }
                completed = target;
                let submission = queue.submit(recording.commands);
                arm_snapshot(frame, recording.poll, submission, None);
            }
            for pass in completed..windows.pass_count() {
                windows.submit_pass(device, queue, pipelines, hf, &lifetime._groups, pass, None)?;
            }
            Ok(queue.submit(after))
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
            Some((
                frame
                    .progression
                    .completed_passes()
                    .expect("coefficient boundary"),
                frame.spatial_groups,
            )),
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
                    outputs: crate::frame_surface::outputs(
                        &self.layout,
                        self.surface.as_deref(),
                        &frame.output,
                    ),
                    changed: crate::frame_surface::changed_regions(
                        &self.layout,
                        self.surface.as_deref(),
                    ),
                },
            ),
        }))
    }
}
