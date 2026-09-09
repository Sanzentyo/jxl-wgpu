//! Extra-channel continuations at the same barriers as color coefficients.

use std::collections::VecDeque;

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum ExtraPhase {
    Coefficients,
    Image,
}

pub(in super::super) struct ExtraProgression {
    coefficients: PassCommands,
    recordings: VecDeque<IntermediateCommands>,
    after: Vec<wgpu::CommandBuffer>,
    next_image: usize,
    pub(in super::super) phase: ExtraPhase,
    pub(in super::super) completed: usize,
    pub(in super::super) total: usize,
}

impl ExtraProgression {
    fn submit_coefficients(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipelines: &VarDctPipelines,
        lifetime: &VarDctJobLifetime,
        mut prefix: Vec<wgpu::CommandBuffer>,
    ) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
        let target = lifetime
            .intermediates
            .get(self.next_image)
            .map_or(self.total, |image| {
                usize::from(
                    image
                        .progression
                        .completed_passes()
                        .expect("coefficient image"),
                )
            });
        if target <= self.completed || target > self.total {
            return Err(VarDctDecodeError::EngineContract {
                detail: "invalid progressive extra pass boundary",
            });
        }
        let tail = if target == self.total {
            std::mem::take(&mut self.after)
        } else {
            let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu extra cursors after a coefficient pass"),
            });
            let retained = lock_unpoisoned(&lifetime._hf_coefficients);
            let hf = retained.as_ref().ok_or(VarDctDecodeError::EngineContract {
                detail: "progressive extra pass has no coefficient status",
            })?;
            let mut offset =
                lifetime._groups.len() as u64 * (PACKET_STATUS_BYTES + ARTIFACT_STATUS_BYTES);
            for group in &hf.groups {
                commands.copy_buffer_to_buffer(
                    &group.status,
                    0,
                    &lifetime.status_staging,
                    offset,
                    group.status.size(),
                );
                offset += group.status.size();
            }
            vec![commands.finish()]
        };
        let submission = match &mut self.coefficients {
            PassCommands::Whole(passes) => {
                prefix.extend(passes.drain(..target - self.completed));
                prefix.extend(tail);
                queue.submit(prefix)
            }
            PassCommands::Windowed(windows) => {
                if !prefix.is_empty() {
                    queue.submit(prefix);
                }
                let retained = lock_unpoisoned(&lifetime._hf_coefficients);
                let hf = retained.as_ref().ok_or(VarDctDecodeError::EngineContract {
                    detail: "progressive extra pass lost its coefficient buffers",
                })?;
                for pass in self.completed..target {
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
                queue.submit(tail)
            }
        };
        self.completed = target;
        self.phase = ExtraPhase::Coefficients;
        Ok(submission)
    }

    fn submit_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lifetime: &Arc<VarDctJobLifetime>,
        mut prefix: Vec<wgpu::CommandBuffer>,
    ) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
        let image = lifetime.intermediates.get(self.next_image).ok_or(
            VarDctDecodeError::EngineContract {
                detail: "progressive extra image has no admitted snapshot",
            },
        )?;
        if usize::from(
            image
                .progression
                .completed_passes()
                .expect("coefficient image"),
        ) != self.completed
        {
            return Err(VarDctDecodeError::EngineContract {
                detail: "progressive extra image disagrees with completed passes",
            });
        }
        let recording = self
            .recordings
            .pop_front()
            .ok_or(VarDctDecodeError::EngineContract {
                detail: "progressive extra image lost its recording",
            })?;
        prefix.extend(recording.commands);
        // Force the shared status map to fence this image without mapping image pixels. Snapshot
        // validation remains in its independent map; subsequent cursor stages reuse this map.
        let mut fence = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu progressive extra image completion"),
        });
        fence.copy_buffer_to_buffer(
            &lifetime._groups[0].packet_status,
            0,
            &lifetime.status_staging,
            0,
            4,
        );
        prefix.push(fence.finish());
        let submission = queue.submit(prefix);
        arm_snapshot(
            image,
            recording.poll,
            submission.clone(),
            Some(Arc::clone(lifetime)),
        );
        self.next_image += 1;
        self.phase = ExtraPhase::Image;
        Ok(submission)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn start(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &VarDctPipelines,
    lifetime: &Arc<VarDctJobLifetime>,
    first: usize,
    coefficients: PassCommands,
    recordings: Vec<IntermediateCommands>,
    prefix: Vec<wgpu::CommandBuffer>,
    after: Vec<wgpu::CommandBuffer>,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    let mut slot = lock_unpoisoned(&lifetime.progressive_extra);
    if slot.is_some() {
        return Err(VarDctDecodeError::EngineContract {
            detail: "progressive extra schedule was already installed",
        });
    }
    let total = match &coefficients {
        PassCommands::Whole(passes) => passes.len(),
        PassCommands::Windowed(windows) => windows.pass_count(),
    };
    let mut progress = ExtraProgression {
        coefficients,
        recordings: recordings.into(),
        after,
        next_image: first,
        phase: ExtraPhase::Coefficients,
        completed: 0,
        total,
    };
    let submission = if first == 0 {
        progress.submit_image(device, queue, lifetime, prefix)?
    } else {
        progress.submit_coefficients(device, queue, pipelines, lifetime, prefix)?
    };
    *slot = Some(progress);
    Ok(submission)
}

impl FramePendingFrame {
    pub(in super::super) fn finish_extra_image(
        &mut self,
        mapping: Result<(), String>,
        source: Box<VarDctSource>,
    ) -> DecodeResult<()> {
        mapping.map_err(DecodeError::backend)?;
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        lifetime.status_staging.unmap();
        lifetime.status_mapped.store(false, Ordering::Release);
        let poll = self
            .backend
            .submission_poller()
            .try_reserve()
            .map_err(DecodeError::PollBackpressure)?;
        let submission = lock_unpoisoned(&lifetime.progressive_extra)
            .as_mut()
            .ok_or(DecodeError::EngineContract(
                "completed extra image lost its pass schedule",
            ))?
            .submit_coefficients(
                self.backend.device(),
                self.backend.queue(),
                &self.pipelines,
                lifetime,
                Vec::new(),
            )?;
        self.arm_extra_progression(submission, poll, source)
    }

    pub(in super::super) fn finish_extra_pass(
        &mut self,
        source: Box<VarDctSource>,
    ) -> DecodeResult<()> {
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let mut progress = lock_unpoisoned(&lifetime.progressive_extra);
        let Some(schedule) = progress.as_mut() else {
            drop(progress);
            return self.submit_extra_output();
        };
        if schedule.completed == schedule.total {
            *progress = None;
            drop(progress);
            return self.submit_extra_output();
        }
        let poll = self
            .backend
            .submission_poller()
            .try_reserve()
            .map_err(DecodeError::PollBackpressure)?;
        let submission = schedule.submit_image(
            self.backend.device(),
            self.backend.queue(),
            lifetime,
            Vec::new(),
        )?;
        drop(progress);
        self.arm_extra_progression(submission, poll, source)
    }

    fn arm_extra_progression(
        &mut self,
        submission: wgpu::SubmissionIndex,
        poll: SubmissionPollPermit,
        source: Box<VarDctSource>,
    ) -> DecodeResult<()> {
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let completion = Arc::new(MapCompletion::default());
        arm_status_map(
            lifetime,
            &completion,
            "VarDCT progressive extra continuation",
        );
        let callback = Arc::clone(&completion);
        if let Err(error) = poll.register(submission, move |error| callback.complete(Err(error))) {
            completion.complete(Err(error.to_string()));
        }
        self.stage = self.after_coefficients_stage(completion, source);
        Ok(())
    }
}
