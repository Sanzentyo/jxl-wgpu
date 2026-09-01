use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use jxl_gpu_formats::ImageLayout;
use jxl_gpu_protocol::{ChangedRegions, OutputId, Region, SubmissionToken};
use jxl_wgpu::{
    GpuImageFrame, GpuImageOutput, MemoryBudget, MemoryBudgetSnapshot, UnvalidatedGpuImageFrame,
    UnvalidatedGpuImageOutput, WgpuBackend,
};

use crate::buffer_pool::DecodeBufferPool;
use crate::progressive_dc::{ProgressiveDcPipeline, ProgressiveDcXybPlanes};
use crate::{
    Error, FrameDuration, FrameMetadata, GpuPendingFrame, GpuSubmissionSession, Result,
    SubmittedGpuFrame,
};

use super::execution::{SubmitPipelines, submit_decode};
use super::lifetime::{DecodeJobLifetime, DecodeMemoryPermits, DecodeSource, MapCompletion};
use super::types::{
    DecodeStatus, F64OutputPath, ModularInversePipelines, STATUS_BYTES, STATUS_OK,
    WgpuDecodeMemoryStats,
};
/// One-frame runtime-neutral GPU decode session for the standard lossless Modular profile.
pub struct WgpuDecodeSession {
    pub(super) backend: WgpuBackend,
    pub(super) pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) source: Option<DecodeSource>,
    pub(super) memory_stats: WgpuDecodeMemoryStats,
    pub(super) memory_budget: MemoryBudget,
    pub(super) buffers: Arc<DecodeBufferPool>,
    pub(super) f64_output_path: Option<F64OutputPath>,
    pub(super) inverse_pipelines: Option<Arc<ModularInversePipelines>>,
    pub(super) progressive_dc_pipeline: Option<Arc<ProgressiveDcPipeline>>,
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
                decode: &self.pipeline,
                inverse: self.inverse_pipelines.as_deref(),
                progressive_dc: self.progressive_dc_pipeline.as_deref(),
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

/// One submitted stock Modular frame. Queue submission has completed, while mapped validation may
/// still be pending.
pub struct WgpuPendingFrame {
    pub(super) device: wgpu::Device,
    pub(super) lifetime: Option<Arc<DecodeJobLifetime>>,
    pub(super) token: SubmissionToken,
    pub(super) layout: ImageLayout,
    pub(super) completion: Arc<MapCompletion>,
    pub(super) stream_sample_counts: Arc<[u32]>,
    pub(super) status_stride: u64,
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
        let lifetime = self.lifetime.as_ref().ok_or(Error::EngineContract(
            "Modular GPU pending frame was already consumed",
        ))?;
        Ok(UnvalidatedGpuImageFrame {
            token: self.token,
            outputs: vec![UnvalidatedGpuImageOutput {
                id: OutputId(0),
                layout: self.layout.clone(),
                buffer: lifetime.output.clone(),
            }],
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
        let statuses = self
            .stream_sample_counts
            .iter()
            .copied()
            .enumerate()
            .map(|(group_index, expected_samples)| {
                let start = u64::try_from(group_index)
                    .ok()
                    .and_then(|index| index.checked_mul(self.status_stride))
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
        drop(mapped);
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

        let output_id = OutputId(0);
        let mut regions = BTreeMap::new();
        regions.insert(
            output_id,
            vec![Region::new(
                0,
                0,
                self.layout.extent.width,
                self.layout.extent.height,
            )],
        );
        Ok(SubmittedGpuFrame::new(
            FrameMetadata {
                index: 0,
                duration: FrameDuration::still(),
                presentation_ticks: 0,
                timecode: None,
                is_last: true,
                is_keyframe: true,
                name: String::new(),
            },
            GpuImageFrame {
                token: self.token,
                outputs: vec![GpuImageOutput {
                    id: output_id,
                    layout: self.layout.clone(),
                    buffer: lifetime.output.clone(),
                }],
                changed: ChangedRegions { outputs: regions },
            },
        ))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl GpuPendingFrame for WgpuPendingFrame {
    type Frame = GpuImageFrame;

    fn wait(mut self) -> Result<SubmittedGpuFrame<Self::Frame>> {
        let mapping = self.completion.wait();
        self.finish(mapping)
    }

    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Poll::Ready(Err(Error::backend(error)));
        }
        match self.completion.poll(context) {
            Some(mapping) => Poll::Ready(self.finish(mapping)),
            None => Poll::Pending,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl GpuPendingFrame for WgpuPendingFrame {
    type Frame = GpuImageFrame;

    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Poll::Ready(Err(Error::backend(error)));
        }
        match self.completion.poll(context) {
            Some(mapping) => Poll::Ready(self.finish(mapping)),
            None => Poll::Pending,
        }
    }
}
