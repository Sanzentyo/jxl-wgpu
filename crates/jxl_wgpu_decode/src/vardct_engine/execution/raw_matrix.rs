//! Bounded raw-quantization image execution and the following HF-global cursor.

use std::sync::{Arc, atomic::Ordering};

use jxl_wgpu::{MemoryPermit, SubmissionPollPermit};

use crate::entropy_window::GroupStreamSegment;
use crate::vardct_side_image::RawHfDequantSideImagePlan;
use crate::wgpu_engine::{
    ModularSideImageJob, ModularSideImageStatus, ModularSideImageStreamPlan, raw_matrix_status_ok,
    raw_matrix_value_error,
};
use crate::{Error as DecodeError, Result as DecodeResult};

use super::{
    DeferredHfGlobalCommands, FramePendingFrame, MapCompletion, VarDctDecodeError,
    VarDctJobLifetime, VarDctPendingStage, VarDctSource,
};

pub(super) struct RawMatrixWork {
    source: Box<VarDctSource>,
    commands: DeferredHfGlobalCommands,
    stream: ModularSideImageStreamPlan,
    next_window: usize,
}

/// The callback owns the image and frame allocations until the submitted map completes.
pub(super) struct RawMatrixLifetime {
    job: ModularSideImageJob,
    _permit: MemoryPermit,
    _frame: Arc<VarDctJobLifetime>,
}

impl FramePendingFrame {
    pub(super) fn start_raw_hf_dequant_stage(
        &mut self,
        source: Box<VarDctSource>,
        mut commands: DeferredHfGlobalCommands,
        poll_permit: Option<SubmissionPollPermit>,
    ) -> Result<(), VarDctDecodeError> {
        let frame = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let plan = source.packet.pending_raw_hf_dequant_side_image().ok_or(
            VarDctDecodeError::EngineContract {
                detail: "raw HF dequant stage has no pending side image",
            },
        )?;
        let packet_end = source.packet.pending_raw_hf_dequant_packet_end().ok_or(
            VarDctDecodeError::EngineContract {
                detail: "raw HF dequant stage has no bounded packet end",
            },
        )?;
        let error = |source| VarDctDecodeError::RawHfDequantGpu {
            matrix: plan.matrix_index,
            source: Box::new(source),
        };
        let pipeline = &self.pipelines.raw_hf_dequant;
        let mut stream = pipeline
            .plan_source(&source.codestream, plan, packet_end, source.stream_limit)
            .map_err(error)?;
        let available = self.memory.snapshot().available_bytes;
        if stream.memory_bytes > available {
            let fixed = stream.memory_bytes - stream.stream_bytes;
            let limit = available.saturating_sub(fixed) & !3;
            if limit
                >= stream
                    .stream_bytes
                    .min(crate::entropy_window::MIN_STREAM_WINDOW_BYTES)
            {
                stream = pipeline
                    .plan_source(
                        &source.codestream,
                        plan,
                        packet_end,
                        source.stream_limit.min(limit),
                    )
                    .map_err(error)?;
            }
        }
        let permit = self.memory.try_reserve(stream.memory_bytes)?;
        let poll = match poll_permit {
            Some(poll) => poll,
            None => self
                .backend
                .submission_poller()
                .try_reserve()
                .map_err(VarDctDecodeError::PollBackpressure)?,
        };
        let mut job = pipeline
            .prepare(
                &self.backend,
                &source.codestream,
                &frame._resources,
                source.resource_layout,
                plan,
                &stream,
            )
            .map_err(error)?;
        if job.memory_bytes() != stream.memory_bytes {
            return Err(VarDctDecodeError::EngineContract {
                detail: "raw HF dequant allocation disagrees with its byte admission",
            });
        }
        let mut submission = Vec::with_capacity(2);
        submission.extend(commands.before_coefficients.take());
        submission.push(job.take_commands().map_err(error)?);
        let lifetime = Arc::new(RawMatrixLifetime {
            job,
            _permit: permit,
            _frame: Arc::clone(frame),
        });
        let work = Box::new(RawMatrixWork {
            source,
            commands,
            stream,
            next_window: 1,
        });
        self.submit_raw_hf_commands(work, lifetime, poll, submission);
        Ok(())
    }

    pub(super) fn finish_raw_hf_dequant_stage(
        &mut self,
        mapping: Result<(), String>,
        mut work: Box<RawMatrixWork>,
        lifetime: Arc<RawMatrixLifetime>,
    ) -> DecodeResult<()> {
        mapping.map_err(DecodeError::backend)?;
        let plan = work
            .source
            .packet
            .pending_raw_hf_dequant_side_image()
            .ok_or(VarDctDecodeError::EngineContract {
                detail: "completed raw HF dequant stage has no parser continuation",
            })?;
        let packet_end = work
            .source
            .packet
            .pending_raw_hf_dequant_packet_end()
            .ok_or(VarDctDecodeError::EngineContract {
                detail: "completed raw HF dequant stage has no bounded packet end",
            })?;
        let error = |source| VarDctDecodeError::RawHfDequantGpu {
            matrix: plan.matrix_index,
            source: Box::new(source),
        };
        let status = lifetime.job.finish_status().map_err(error)?;
        let segment =
            work.stream
                .segments
                .get(work.next_window - 1)
                .ok_or(DecodeError::EngineContract(
                    "raw matrix has no submitted window",
                ))?;
        validate_status(plan, packet_end, segment, status)?;
        let mut lifetime = Arc::try_unwrap(lifetime).map_err(|_| {
            DecodeError::EngineContract("raw matrix callback retained resources after completion")
        })?;
        if status.is_in_progress() || lifetime.job.has_finalization_commands() {
            let poll = self
                .backend
                .submission_poller()
                .try_reserve()
                .map_err(DecodeError::PollBackpressure)?;
            let commands = if status.is_in_progress() {
                let next = work.stream.segments.get(work.next_window).ok_or(
                    DecodeError::EngineContract("raw matrix yielded after its final window"),
                )?;
                let commands = lifetime
                    .job
                    .record_next_window(&self.backend, &work.source.codestream, next)
                    .map_err(error)?;
                work.next_window += 1;
                commands
            } else {
                lifetime
                    .job
                    .take_finalization_commands()
                    .ok_or(DecodeError::EngineContract(
                        "raw matrix lost its finalization commands",
                    ))?
            };
            self.submit_raw_hf_commands(work, Arc::new(lifetime), poll, vec![commands]);
            return Ok(());
        }
        drop(lifetime);
        work.source
            .packet
            .resume_hf_global_after_raw_side_image_source(&work.source.codestream, status.cursor)
            .map_err(VarDctDecodeError::from)?;
        if work
            .source
            .packet
            .pending_raw_hf_dequant_side_image()
            .is_some()
        {
            self.start_raw_hf_dequant_stage(work.source, work.commands, None)?;
            Ok(())
        } else {
            self.submit_deferred_hf_coefficients(work.source, work.commands)
        }
    }

    fn submit_raw_hf_commands(
        &mut self,
        work: Box<RawMatrixWork>,
        lifetime: Arc<RawMatrixLifetime>,
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
            poll_completion.complete(Err(error));
        }) {
            completion.complete(Err(format!(
                "raw HF dequant GPU poll registration failed: {error}"
            )));
        }
        self.runtime_stats
            .submissions_per_frame
            .fetch_add(1, Ordering::AcqRel);
        self.stage = VarDctPendingStage::RawHfDequant {
            completion,
            work,
            lifetime,
        };
    }
}

fn validate_status(
    plan: &RawHfDequantSideImagePlan,
    packet_end: u32,
    segment: GroupStreamSegment,
    status: ModularSideImageStatus,
) -> Result<(), VarDctDecodeError> {
    if raw_matrix_value_error(status.code) {
        return Err(VarDctDecodeError::RawHfDequantValue {
            matrix: plan.matrix_index,
        });
    }
    let available_end = plan
        .image
        .token_bit_offset
        .checked_add(segment.available_token_end)
        .ok_or(VarDctDecodeError::EngineContract {
            detail: "raw matrix window end overflows its cursor",
        })?;
    if !(raw_matrix_status_ok(status.code) || status.is_in_progress())
        || status.decoded_samples > plan.image.decoded_words
        || (status.is_ok() && status.decoded_samples != plan.image.decoded_words)
        || status.cursor < plan.image.token_bit_offset
        || status.cursor > available_end
        || status.expected_cursor != packet_end
    {
        return Err(VarDctDecodeError::RawHfDequantStatus {
            matrix: plan.matrix_index,
            code: status.code,
            decoded_samples: status.decoded_samples,
            expected_samples: plan.image.decoded_words,
            cursor: status.cursor,
            expected_cursor: status.expected_cursor,
        });
    }
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
