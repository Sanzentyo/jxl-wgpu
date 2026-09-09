//! GPU cursor continuations and bounded, sequential assembly of Modular extra channels.

use std::collections::VecDeque;

use crate::modular_assembly::encode_plane_copies;
use crate::vardct_extra::VarDctExtraSubimage;
use crate::vardct_pass_group::{HF_COEFFICIENT_STATUS_BYTES, HfCoefficientStreamEnd};
use crate::wgpu_engine::{ModularSideImageJob, ModularSideImageStreamPlan};

use super::*;

pub(super) fn map_extra_error(source: DecodeError) -> VarDctDecodeError {
    VarDctDecodeError::ModularExtra {
        source: Box::new(source),
    }
}

pub(super) fn prepare_frame_arena(
    backend: &WgpuBackend,
    source: &mut VarDctSource,
    permit: Option<MemoryPermit>,
) -> DecodeResult<Option<GpuBufferLease>> {
    let Some(plan) = &source.packet.extra_channels else {
        if permit.is_some() {
            return Err(DecodeError::EngineContract(
                "extra arena admitted without a plan",
            ));
        }
        return Ok(None);
    };
    let permit = permit.ok_or(DecodeError::EngineContract(
        "extra frame arena was not admitted",
    ))?;
    let buffer = backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu assembled VarDCT Modular extra channels"),
        size: plan.inverse.arena_bytes(),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let arena = GpuBufferLease::from_tracked(buffer, permit);
    for &index in &source.extra_indices {
        let declaration =
            source
                .extra_declarations
                .get(index)
                .ok_or(DecodeError::EngineContract(
                    "selected extra channel has no declaration",
                ))?;
        let encoding = crate::modular_sample::ModularSampleEncoding::new(declaration.bit_depth)
            .ok_or(DecodeError::EngineContract(
                "extra-channel profile changed precision",
            ))?;
        let plane = plan.inverse.final_gpu_layouts().get(index).copied().ok_or(
            DecodeError::EngineContract("selected extra channel has no reconstructed plane"),
        )?;
        source
            .extra_planes
            .push(super::super::staging::ResidentModularPlane {
                index: index as u32,
                arena: arena.clone(),
                plane,
                encoding,
            });
    }
    Ok(Some(arena))
}

pub(super) struct HfValidation {
    index: u32,
    token_start: u32,
    token_end: u32,
    cursor_base: u32,
    continuation: bool,
}

impl HfValidation {
    pub(super) fn is_completed_by(&self, passes: u8, spatial_groups: u64) -> bool {
        u64::from(self.index) < u64::from(passes) * spatial_groups
    }

    pub(super) fn plan(
        source: &VarDctSource,
        execution: &HfCoefficientExecutionPlan,
    ) -> DecodeResult<Vec<Self>> {
        let entropy = source
            .packet
            .hf_coefficients
            .as_ref()
            .ok_or(DecodeError::EngineContract(
                "AC validation has no entropy plan",
            ))?;
        let groups = source.packet.profile.group_count;
        execution
            .groups
            .iter()
            .flat_map(HfCoefficientGroupExecutionPlan::global_group_indices)
            .map(|index| {
                let pass = (u64::from(index) / groups) as usize;
                let group = (u64::from(index) % groups) as usize;
                let range = entropy
                    .passes
                    .get(pass)
                    .and_then(|pass| pass.pass_groups.get(group))
                    .ok_or(DecodeError::EngineContract(
                        "AC validation references an absent packet",
                    ))?;
                let start = u32::try_from(range.offset)
                    .map_err(|_| DecodeError::EngineContract("AC packet start exceeds u32"))?;
                let end = range
                    .end()
                    .and_then(|end| u32::try_from(end).ok())
                    .ok_or(DecodeError::EngineContract("AC packet end exceeds u32"))?;
                let cursor_base = if execution.uses_bounded_stream_windows() {
                    start
                } else {
                    0
                };
                let continuation = source
                    .packet
                    .extra_channels
                    .as_ref()
                    .map(|plan| plan.ac_stream_end(&source.packet.profile, pass, group as u32))
                    .transpose()
                    .map_err(VarDctDecodeError::from)?
                    == Some(HfCoefficientStreamEnd::Continuation);
                Ok(Self {
                    index,
                    token_start: start - cursor_base,
                    token_end: end - cursor_base,
                    cursor_base,
                    continuation,
                })
            })
            .collect()
    }

    pub(super) fn validate(&self, status: GpuHfCoefficientStatus) -> DecodeResult<u32> {
        let cursor = status
            .validate_cursor(self.index, self.token_start, self.token_end)
            .map_err(VarDctDecodeError::from)?;
        if !self.continuation {
            status
                .validate(self.index)
                .map_err(VarDctDecodeError::from)?;
        }
        cursor
            .checked_add(self.cursor_base)
            .ok_or(DecodeError::EngineContract("AC cursor overflow"))
    }
}

struct ExtraRequest {
    pass: Option<usize>,
    group: u32,
    cursor: u32,
}

enum ExtraResume {
    Lf {
        commands: PostLfCommands,
        cursors: Vec<u32>,
    },
    Ac,
}

pub(super) struct ExtraWork {
    source: Box<VarDctSource>,
    remaining: VecDeque<ExtraRequest>,
    resume: ExtraResume,
    current: ExtraRequest,
    subimage: VarDctExtraSubimage,
    stream: ModularSideImageStreamPlan,
    next_window: usize,
    copied: bool,
}

pub(super) struct ExtraLifetime {
    job: ModularSideImageJob,
    _permit: MemoryPermit,
    frame: Arc<VarDctJobLifetime>,
}

impl FramePendingFrame {
    pub(super) fn after_coefficients_stage(
        &self,
        completion: Arc<MapCompletion>,
        source: Box<VarDctSource>,
    ) -> VarDctPendingStage {
        if self.extra_output_commands.is_some() {
            VarDctPendingStage::AcExtra { completion, source }
        } else {
            VarDctPendingStage::Final { completion }
        }
    }

    pub(super) fn start_lf_extras(
        &mut self,
        source: Box<VarDctSource>,
        commands: PostLfCommands,
        cursors: Vec<u32>,
    ) -> DecodeResult<()> {
        let requests = cursors
            .iter()
            .enumerate()
            .map(|(group, &cursor)| ExtraRequest {
                pass: None,
                group: group as u32,
                cursor,
            })
            .collect();
        self.start_extra_subimage(source, requests, ExtraResume::Lf { commands, cursors })
    }

    pub(super) fn finish_ac_extra(
        &mut self,
        mapping: Result<(), String>,
        source: Box<VarDctSource>,
    ) -> DecodeResult<()> {
        mapping.map_err(DecodeError::backend)?;
        let frame = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let mapped = frame
            .status_staging
            .slice(..)
            .get_mapped_range()
            .map_err(DecodeError::backend)?;
        let offset =
            self.expected_groups.len() * (PACKET_STATUS_BYTES + ARTIFACT_STATUS_BYTES) as usize;
        let bytes = self.expected_hf.len() * HF_COEFFICIENT_STATUS_BYTES as usize;
        let statuses = mapped
            .get(offset..offset + bytes)
            .ok_or(VarDctDecodeError::StatusAbi {
                status: "AC extra cursor range",
            })?;
        let statuses =
            bytemuck::try_cast_slice::<u8, GpuHfCoefficientStatus>(statuses).map_err(|_| {
                VarDctDecodeError::StatusAbi {
                    status: "AC extra cursor",
                }
            })?;
        let mut requests = VecDeque::new();
        for (expected, &status) in self.expected_hf.iter().zip(statuses) {
            let cursor = expected.validate(status)?;
            if expected.continuation {
                let groups = source.packet.profile.group_count;
                requests.push_back(ExtraRequest {
                    pass: Some((u64::from(expected.index) / groups) as usize),
                    group: (u64::from(expected.index) % groups) as u32,
                    cursor,
                });
            }
        }
        drop(mapped);
        frame.status_staging.unmap();
        frame.status_mapped.store(false, Ordering::Release);
        self.start_extra_subimage(source, requests, ExtraResume::Ac)
    }

    fn start_extra_subimage(
        &mut self,
        source: Box<VarDctSource>,
        mut remaining: VecDeque<ExtraRequest>,
        resume: ExtraResume,
    ) -> DecodeResult<()> {
        while let Some(current) = remaining.pop_front() {
            let Some(subimage) = source.packet.parse_extra_subimage_source(
                &source.codestream,
                current.pass,
                current.group,
                current.cursor,
            )?
            else {
                continue;
            };
            let pipeline = self.pipelines.raw_hf_dequant.modular();
            let mut stream = pipeline.plan_source(
                &source.codestream,
                &subimage.image,
                subimage.packet_end,
                source.stream_limit,
            )?;
            let available = self.memory.snapshot().available_bytes;
            if stream.memory_bytes > available {
                let fixed = stream.memory_bytes - stream.stream_bytes;
                let limit = available.saturating_sub(fixed) & !3;
                if limit
                    >= stream
                        .stream_bytes
                        .min(crate::entropy_window::MIN_STREAM_WINDOW_BYTES)
                {
                    stream = pipeline.plan_source(
                        &source.codestream,
                        &subimage.image,
                        subimage.packet_end,
                        limit,
                    )?;
                }
            }
            let permit = self.memory.try_reserve(stream.memory_bytes)?;
            let poll = self
                .backend
                .submission_poller()
                .try_reserve()
                .map_err(DecodeError::PollBackpressure)?;
            let mut job = pipeline
                .record_source(&self.backend, &source.codestream, &subimage.image, &stream)?
                .finish();
            if job.memory_bytes() != stream.memory_bytes {
                return Err(DecodeError::EngineContract(
                    "extra subimage allocation disagrees with admission",
                ));
            }
            let commands = job.take_commands()?;
            let frame = Arc::clone(
                self.lifetime
                    .as_ref()
                    .ok_or(VarDctDecodeError::CompletionConsumed)?,
            );
            let lifetime = Arc::new(ExtraLifetime {
                job,
                _permit: permit,
                frame,
            });
            let work = Box::new(ExtraWork {
                source,
                remaining,
                resume,
                current,
                subimage,
                stream,
                next_window: 1,
                copied: false,
            });
            self.submit_extra_commands(work, lifetime, poll, vec![commands]);
            return Ok(());
        }
        match resume {
            ExtraResume::Lf { commands, cursors } => {
                self.submit_hf_from_cursors(source, commands, cursors)
            }
            ExtraResume::Ac => self.submit_extra_output(),
        }
    }

    pub(super) fn finish_extra_subimage(
        &mut self,
        mapping: Result<(), String>,
        mut work: Box<ExtraWork>,
        lifetime: Arc<ExtraLifetime>,
    ) -> DecodeResult<()> {
        mapping.map_err(DecodeError::backend)?;
        let status = lifetime.job.finish_status()?;
        let plan = &work.subimage.image;
        let segment =
            work.stream
                .segments
                .get(work.next_window - 1)
                .ok_or(DecodeError::EngineContract(
                    "extra subimage has no submitted window",
                ))?;
        if !(status.is_ok() || status.is_in_progress())
            || status.decoded_samples > plan.decoded_words
            || (status.is_ok() && status.decoded_samples != plan.decoded_words)
            || status.cursor < plan.token_bit_offset
            || status.cursor > plan.token_bit_offset + segment.available_token_end
            || status.expected_cursor != work.subimage.packet_end
        {
            return Err(VarDctDecodeError::ExtraModularStatus {
                stream: plan.stream_index,
                code: status.code,
                decoded_samples: status.decoded_samples,
                expected_samples: plan.decoded_words,
                cursor: status.cursor,
                packet_end: work.subimage.packet_end,
            }
            .into());
        }
        let mut lifetime = Arc::try_unwrap(lifetime).map_err(|_| {
            DecodeError::EngineContract("extra subimage callback retained completed resources")
        })?;
        if work.copied {
            if let ExtraResume::Lf { cursors, .. } = &mut work.resume {
                cursors[work.current.group as usize] = status.cursor;
            }
            drop(lifetime);
            return self.start_extra_subimage(work.source, work.remaining, work.resume);
        }
        let poll = self
            .backend
            .submission_poller()
            .try_reserve()
            .map_err(DecodeError::PollBackpressure)?;
        let commands = if status.is_in_progress() {
            let next =
                work.stream
                    .segments
                    .get(work.next_window)
                    .ok_or(DecodeError::EngineContract(
                        "extra subimage yielded after its final window",
                    ))?;
            let commands =
                lifetime
                    .job
                    .record_next_window(&self.backend, &work.source.codestream, next)?;
            work.next_window += 1;
            vec![commands]
        } else {
            if work.current.pass.is_some()
                && (u64::from(status.cursor).div_ceil(8) * 8 != u64::from(work.subimage.packet_end)
                    || !work.source.codestream.bits_are_zero(
                        u64::from(status.cursor),
                        u64::from(work.subimage.packet_end),
                    )?)
            {
                return Err(VarDctDecodeError::ExtraModularTrailingBits {
                    stream: plan.stream_index,
                    cursor: status.cursor,
                    packet_end: work.subimage.packet_end,
                }
                .into());
            }
            if plan.final_planes.len() != work.subimage.targets.len() {
                return Err(DecodeError::EngineContract(
                    "extra subimage reconstruction changed its output plane count",
                ));
            }
            let mut copies =
                self.backend
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("jxl-wgpu assemble reconstructed Modular extra subimage"),
                    });
            let arena = lifetime
                .frame
                .extra_frame
                .as_ref()
                .ok_or(DecodeError::EngineContract(
                    "extra subimage has no retained frame arena",
                ))?;
            encode_plane_copies(
                &mut copies,
                lifetime.job.arena(),
                arena.as_wgpu_buffer(),
                0,
                plan.final_planes
                    .iter()
                    .copied()
                    .zip(work.subimage.targets.iter().copied()),
            )?;
            lifetime.job.encode_status_copy(&mut copies);
            let mut commands: Vec<_> = lifetime
                .job
                .take_finalization_commands()
                .into_iter()
                .collect();
            commands.push(copies.finish());
            work.copied = true;
            commands
        };
        self.submit_extra_commands(work, Arc::new(lifetime), poll, commands);
        Ok(())
    }

    fn submit_extra_commands(
        &mut self,
        work: Box<ExtraWork>,
        lifetime: Arc<ExtraLifetime>,
        poll: SubmissionPollPermit,
        commands: Vec<wgpu::CommandBuffer>,
    ) {
        let submission = self.backend.queue().submit(commands);
        let completion = Arc::new(MapCompletion::default());
        lifetime.job.mark_status_mapped();
        let callback_lifetime = Arc::clone(&lifetime);
        let callback_completion = Arc::clone(&completion);
        lifetime
            .job
            .status_staging()
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                drop(callback_lifetime);
                callback_completion.complete(result.map_err(|error| error.to_string()));
            });
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll.register(submission, move |error| {
            poll_completion.complete(Err(error))
        }) {
            completion.complete(Err(error.to_string()));
        }
        self.runtime_stats
            .submissions_per_frame
            .fetch_add(1, Ordering::AcqRel);
        self.stage = VarDctPendingStage::ModularExtra {
            completion,
            work,
            lifetime,
        };
    }

    fn submit_extra_output(&mut self) -> DecodeResult<()> {
        let poll = self
            .backend
            .submission_poller()
            .try_reserve()
            .map_err(DecodeError::PollBackpressure)?;
        let commands = self
            .extra_output_commands
            .take()
            .ok_or(DecodeError::EngineContract(
                "assembled extra channels have no frame output commands",
            ))?;
        let frame = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let submission = self.backend.queue().submit([commands]);
        let completion = Arc::new(MapCompletion::default());
        arm_status_map(frame, &completion, "VarDCT assembled extra-channel output");
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll.register(submission, move |error| {
            poll_completion.complete(Err(error))
        }) {
            completion.complete(Err(error.to_string()));
        }
        self.runtime_stats
            .submissions_per_frame
            .fetch_add(1, Ordering::AcqRel);
        self.stage = VarDctPendingStage::Final { completion };
        Ok(())
    }
}
