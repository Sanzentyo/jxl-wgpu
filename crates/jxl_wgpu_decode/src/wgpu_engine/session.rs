use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use jxl_gpu_formats::ImageLayout;
use jxl_gpu_protocol::SubmissionToken;
use jxl_wgpu::{
    GpuImageFrame, MemoryBudget, MemoryBudgetSnapshot, UnvalidatedGpuImageFrame, WgpuBackend,
};

use crate::buffer_pool::DecodeBufferPool;
use crate::modular_render::ModularReconstructionPipeline;
use crate::progressive_dc::ProgressiveDcXybPlanes;
use crate::{
    Error, FrameDuration, FrameMetadata, GpuPendingFrame, GpuSubmissionSession, Result,
    SubmittedGpuFrame, SubmittedGpuUpdate,
};

use super::execution::{DecodeExecution, SubmitPipelines, submit_decode};
use super::lifetime::{DecodeJobLifetime, DecodeMemoryPermits, DecodeSource, MapCompletion};
use super::progression::IntermediateFrame;
use super::types::{
    DecodeStatus, F64OutputPath, ModularInversePipelines, STATUS_BYTES, STATUS_OK,
    WgpuDecodeMemoryStats,
};
/// One-frame runtime-neutral GPU decode session for the standard lossless Modular profile.
pub struct WgpuDecodeSession {
    pub(super) backend: WgpuBackend,
    pub(super) pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) source: Option<Arc<DecodeSource>>,
    pub(super) memory_stats: WgpuDecodeMemoryStats,
    pub(super) memory_budget: MemoryBudget,
    pub(super) buffers: Arc<DecodeBufferPool>,
    pub(super) f64_output_path: Option<F64OutputPath>,
    pub(super) inverse_pipelines: Option<Arc<ModularInversePipelines>>,
    pub(super) lf_reconstruction_pipeline: Option<Arc<ModularReconstructionPipeline>>,
}

impl std::fmt::Debug for WgpuDecodeSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuDecodeSession")
            .field("submitted", &self.source.is_none())
            .field("memory_stats", &self.memory_stats)
            .finish_non_exhaustive()
    }
}

impl GpuSubmissionSession for WgpuDecodeSession {
    type Frame = GpuImageFrame;
    type Pending = WgpuPendingFrame;

    fn submit_next(&mut self) -> Result<Option<Self::Pending>> {
        let Some(source) = self.source.as_ref() else {
            return Ok(None);
        };
        // Admission must precede Queue::submit and source consumption. Saturation leaves the
        // exact decode source available for a later prefetch attempt.
        let poll_permit = self
            .backend
            .submission_poller()
            .try_reserve()
            .map_err(Error::PollBackpressure)?;
        let output_permit = self
            .memory_budget
            .try_reserve(self.memory_stats.output_lease_bytes)?;
        let transient_permit = self
            .memory_budget
            .try_reserve(self.memory_stats.transient_bytes)?;
        let pending = submit_decode(
            &self.backend,
            SubmitPipelines {
                decode: Arc::clone(&self.pipeline),
                inverse: self.inverse_pipelines.clone(),
                lf_reconstruction: self.lf_reconstruction_pipeline.clone(),
            },
            source,
            &self.buffers,
            DecodeMemoryPermits {
                output: output_permit,
                transient: transient_permit,
            },
            poll_permit,
        )?;
        self.source = None;
        Ok(Some(pending))
    }
}

impl WgpuDecodeSession {
    #[must_use]
    pub const fn memory_stats(&self) -> WgpuDecodeMemoryStats {
        self.memory_stats
    }

    /// Maximum byte exposure allowed by this session's requested frame window.
    #[must_use]
    pub const fn max_frame_window_gpu_bytes(&self) -> u64 {
        self.memory_stats.max_frame_window_bytes
    }

    /// Reports allocations currently retained by jobs and output leases across engine clones.
    #[must_use]
    pub fn in_flight_memory_stats(&self) -> MemoryBudgetSnapshot {
        self.memory_budget.snapshot()
    }

    /// Resolved F64 path for this session, or `None` when the requested output is not F64.
    #[must_use]
    pub const fn f64_output_path(&self) -> Option<F64OutputPath> {
        self.f64_output_path
    }
}

/// A Modular frame whose next image boundary is submitted. Progressive requests resume later
/// entropy passes as updates are consumed; mapped validation may still be pending.
pub struct WgpuPendingFrame {
    pub(super) frame_name: String,
    pub(super) device: wgpu::Device,
    pub(super) lifetime: Option<Arc<DecodeJobLifetime>>,
    pub(super) token: SubmissionToken,
    pub(super) layout: ImageLayout,
    pub(super) surface: Option<Arc<crate::frame_surface::FrameSurfaceLayout>>,
    pub(super) completion: Arc<MapCompletion>,
    pub(super) stream_sample_counts: Arc<[u32]>,
    pub(super) status_stride: u64,
    pub(super) execution: Option<Box<DecodeExecution>>,
    pub(super) intermediate: Option<Arc<IntermediateFrame>>,
}

impl std::fmt::Debug for WgpuPendingFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuPendingFrame")
            .field("token", &self.token)
            .field("layout", &self.layout)
            .field("stream_sample_counts", &self.stream_sample_counts)
            .finish_non_exhaustive()
    }
}

impl WgpuPendingFrame {
    pub(crate) fn progressive_dc_planes(&self) -> Result<ProgressiveDcXybPlanes> {
        self.lifetime
            .as_ref()
            .and_then(|lifetime| lifetime.progressive_dc_planes.clone())
            .ok_or(Error::EngineContract(
                "Modular pending frame does not retain progressive-DC XYB planes",
            ))
    }

    /// Clones a budget-tracked lease to the queue-submitted output before validation completes.
    ///
    /// Submit consumers only to the same [`WgpuBackend`] device and queue that created this decode
    /// session. Queue ordering then permits display, readback, or custom GPU work without a host
    /// wait. This value deliberately has no authoritative frame metadata or changed regions. If
    /// [`GpuDecodeSession::next_frame`](crate::GpuDecodeSession::next_frame) later returns an error,
    /// already-submitted consumer work cannot be rolled back and all derived data must be
    /// discarded.
    ///
    /// The returned [`jxl_wgpu::GpuBufferLease`] clone retains the output allocation's shared byte-budget
    /// permit. Keep that lease alive instead of cloning its raw wgpu buffer handle.
    pub fn unvalidated_gpu_frame(&self) -> Result<UnvalidatedGpuImageFrame> {
        if self.execution.is_some() {
            return Err(Error::EngineContract(
                "progressive Modular final output has not been submitted",
            ));
        }
        let lifetime = self.lifetime.as_ref().ok_or(Error::EngineContract(
            "Modular GPU pending frame was already consumed",
        ))?;
        Ok(UnvalidatedGpuImageFrame {
            token: self.token,
            outputs: crate::frame_surface::unvalidated_outputs(
                &self.layout,
                self.surface.as_deref(),
                &lifetime.output,
            ),
        })
    }

    fn finish(
        &mut self,
        mapping: std::result::Result<(), String>,
    ) -> Result<SubmittedGpuFrame<GpuImageFrame>> {
        mapping.map_err(Error::backend)?;
        let lifetime = self.lifetime.take().ok_or(Error::EngineContract(
            "Modular GPU completion was consumed more than once",
        ))?;
        let mapped = lifetime
            .status_staging
            .buffer()
            .slice(..)
            .get_mapped_range()
            .map_err(Error::backend)?;
        validate_statuses(
            &mapped,
            self.status_stride,
            self.stream_sample_counts.iter().copied().enumerate(),
        )?;
        drop(mapped);

        Ok(self.image_frame(&lifetime.output))
    }

    fn image_frame(&self, output: &jxl_wgpu::GpuBufferLease) -> SubmittedGpuFrame<GpuImageFrame> {
        SubmittedGpuFrame::new(
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
                    output,
                ),
                changed: crate::frame_surface::changed_regions(
                    &self.layout,
                    self.surface.as_deref(),
                ),
            },
        )
    }

    fn resume(&mut self) -> Result<()> {
        if self.intermediate.is_none()
            && let Some(execution) = &mut self.execution
        {
            let lifetime = self
                .lifetime
                .as_ref()
                .ok_or(Error::EngineContract("Modular completion was consumed"))?;
            self.intermediate = execution.resume(lifetime, &self.completion)?;
            if self.intermediate.is_none() {
                self.execution = None;
            }
        }
        Ok(())
    }

    fn finish_intermediate(
        &mut self,
        mapping: std::result::Result<(), String>,
        validate: bool,
    ) -> Result<SubmittedGpuUpdate<GpuImageFrame>> {
        mapping.map_err(Error::backend)?;
        let image = self
            .intermediate
            .take()
            .ok_or(Error::EngineContract("Modular intermediate was consumed"))?;
        if validate {
            let mapped = image
                .status_staging
                .buffer()
                .slice(..)
                .get_mapped_range()
                .map_err(Error::backend)?;
            let group_end = image.boundary.group_end;
            let has_global = self
                .execution
                .as_ref()
                .is_some_and(|execution| execution.has_global_stream());
            let global_index = self.stream_sample_counts.len().saturating_sub(1);
            validate_statuses(
                &mapped,
                self.status_stride,
                self.stream_sample_counts
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|&(index, _)| {
                        index < group_end || (has_global && index == global_index)
                    }),
            )?;
        }
        Ok(SubmittedGpuUpdate::Intermediate {
            frame: self.image_frame(&image.output),
            progression: image.boundary.progression,
        })
    }

    fn poll_update(
        &mut self,
        context: &mut Context<'_>,
        intermediates: bool,
    ) -> Poll<Result<SubmittedGpuUpdate<GpuImageFrame>>> {
        loop {
            self.resume()?;
            self.device
                .poll(wgpu::PollType::Poll)
                .map_err(Error::backend)?;
            if let Some(image) = &self.intermediate {
                let Some(mapping) = image.completion.poll(context) else {
                    return Poll::Pending;
                };
                let update = self.finish_intermediate(mapping, intermediates)?;
                if intermediates {
                    return Poll::Ready(Ok(update));
                }
            } else {
                return match self.completion.poll(context) {
                    Some(mapping) => {
                        Poll::Ready(self.finish(mapping).map(SubmittedGpuUpdate::Complete))
                    }
                    None => Poll::Pending,
                };
            }
        }
    }
}

impl GpuPendingFrame for WgpuPendingFrame {
    type Frame = GpuImageFrame;

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(mut self) -> Result<SubmittedGpuFrame<Self::Frame>> {
        loop {
            self.resume()?;
            if let Some(image) = &self.intermediate {
                let mapping = image.completion.wait();
                self.finish_intermediate(mapping, false)?;
            } else {
                let mapping = self.completion.wait();
                return self.finish(mapping);
            }
        }
    }

    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        self.poll_update(context, false).map(|result| {
            result.and_then(|update| match update {
                SubmittedGpuUpdate::Complete(frame) => Ok(frame),
                SubmittedGpuUpdate::Intermediate { .. } => Err(Error::EngineContract(
                    "final Modular poll returned an intermediate",
                )),
            })
        })
    }

    fn poll_next_update(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuUpdate<Self::Frame>>> {
        self.poll_update(context, true)
    }
}

fn validate_statuses(
    mapped: &[u8],
    stride: u64,
    counts: impl Iterator<Item = (usize, u32)>,
) -> Result<()> {
    let statuses = counts
        .map(|(group_index, expected_samples)| {
            let start = u64::try_from(group_index)
                .ok()
                .and_then(|index| index.checked_mul(stride))
                .and_then(|offset| usize::try_from(offset).ok())
                .ok_or_else(|| Error::backend("GPU status offset overflow"))?;
            let end = start
                .checked_add(STATUS_BYTES as usize)
                .ok_or_else(|| Error::backend("GPU status range overflow"))?;
            let bytes = mapped
                .get(start..end)
                .ok_or_else(|| Error::backend("GPU status buffer was truncated"))?;
            let status = bytemuck::try_cast_slice::<u8, DecodeStatus>(bytes)
                .map_err(|_| Error::backend("GPU status buffer has an invalid ABI layout"))?
                .first()
                .copied()
                .ok_or_else(|| Error::backend("GPU status buffer was truncated"))?;
            Ok((group_index, expected_samples, status))
        })
        .collect::<Result<Vec<_>>>();
    for (group_index, expected_samples, status) in statuses? {
        if status.code != STATUS_OK
            || status.decoded_samples != expected_samples
            || status.cursor != status.expected_cursor
        {
            return Err(Error::ModularEntropyRejected {
                group_index,
                status: status.code,
                decoded_samples: status.decoded_samples,
                expected_samples,
                cursor: status.cursor,
                expected_cursor: status.expected_cursor,
            });
        }
    }

    Ok(())
}
