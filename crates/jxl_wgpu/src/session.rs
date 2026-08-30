// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::BTreeMap;
use std::sync::Arc;

use jxl_gpu_protocol::{
    AcceleratedFrame, ChangedRegions, Extent2d, FrameSessionDesc, GroupId, GroupPayload, OutputId,
    OutputLayout, Region, RenderIntent, RenderPlan, ResourceData, ResourceId, ResourceUpdate,
    SampleType, SubmissionToken,
};
#[cfg(not(target_arch = "wasm32"))]
use jxl_gpu_protocol::{AcceleratedFrameSession, AcceleratorError};

use crate::buffer_pool::PooledBuffer;
use crate::context::WgpuAccelerator;
use crate::readback::{ReadbackRequest, resolve_outputs};
use crate::scheduler::Scheduler;
use crate::video::{
    CpuImageFrame, GpuImageFrame, GpuImageOutput, ImageOutputRequest, ImageReadbackRequest,
    resolve_image_outputs,
};
use crate::{Error, ExecutionPlan, Result};

/// One packed output that remains owned by the caller after its frame session is dropped.
///
/// Commands submitted later to the same [`wgpu::Queue`] may consume `buffer` immediately;
/// WebGPU queue ordering makes an explicit host wait unnecessary.
#[derive(Clone, Debug)]
pub struct GpuOutputBuffer {
    pub id: OutputId,
    pub extent: Extent2d,
    pub sample_type: SampleType,
    pub channels: u8,
    pub layout: OutputLayout,
    /// Number of meaningful bytes in `buffer`. The allocation may be padded for WebGPU copy and
    /// binding alignment requirements.
    pub logical_size: u64,
    pub buffer: Arc<wgpu::Buffer>,
}

/// Non-blocking result of [`WgpuFrameSession::submit_gpu`].
#[derive(Clone, Debug)]
pub struct GpuFrame {
    pub token: SubmissionToken,
    pub outputs: Vec<GpuOutputBuffer>,
    pub changed: ChangedRegions,
}

/// Execution mode associated with a pending submission token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmissionMode {
    CpuReadback,
    CpuImageReadback,
    GpuOnly,
}

/// Dispatch counts recorded for the most recently submitted frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WgpuSubmissionStats {
    pub planned_dispatches: u32,
    pub compute_dispatches: u32,
    pub fused_dispatches: u32,
    /// True when the final storage allocation was mapped directly instead of copied to staging.
    pub direct_readback: bool,
    /// Physical resident-plane bytes addressed by this submission.
    pub resident_bytes: u64,
    /// Explicit uniforms, packet uploads, packed outputs, and staging bytes
    /// allocated for this submission. Driver-private allocations are excluded.
    pub transient_bytes: u64,
}

#[derive(Debug)]
enum PendingSubmission {
    CpuReadback {
        submission: wgpu::SubmissionIndex,
        readbacks: Vec<ReadbackRequest>,
        changed: ChangedRegions,
        recycle_after_wait: Vec<PooledBuffer>,
        transient_bytes: u64,
    },
    CpuImageReadback {
        submission: wgpu::SubmissionIndex,
        readbacks: Vec<ImageReadbackRequest>,
        changed: ChangedRegions,
        recycle_after_wait: Vec<PooledBuffer>,
        transient_bytes: u64,
    },
    GpuOnly {
        #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
        submission: wgpu::SubmissionIndex,
        transient_bytes: u64,
    },
}

impl PendingSubmission {
    const fn mode(&self) -> SubmissionMode {
        match self {
            Self::CpuReadback { .. } => SubmissionMode::CpuReadback,
            Self::CpuImageReadback { .. } => SubmissionMode::CpuImageReadback,
            Self::GpuOnly { .. } => SubmissionMode::GpuOnly,
        }
    }

    const fn transient_bytes(&self) -> u64 {
        match self {
            Self::CpuReadback {
                transient_bytes, ..
            }
            | Self::CpuImageReadback {
                transient_bytes, ..
            }
            | Self::GpuOnly {
                transient_bytes, ..
            } => *transient_bytes,
        }
    }
}

pub struct WgpuFrameSession {
    accelerator: WgpuAccelerator,
    frame: FrameSessionDesc,
    plan: Arc<RenderPlan>,
    execution: ExecutionPlan,
    groups: BTreeMap<GroupId, GroupPayload>,
    resources: BTreeMap<ResourceId, ResourceUpdate>,
    pending: BTreeMap<SubmissionToken, PendingSubmission>,
    next_token: u64,
    last_submission_stats: Option<WgpuSubmissionStats>,
    pending_transient_bytes: u64,
}

impl std::fmt::Debug for WgpuFrameSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuFrameSession")
            .field("frame", &self.frame)
            .field("group_count", &self.groups.len())
            .field("resource_count", &self.resources.len())
            .field("pending_count", &self.pending.len())
            .field("pending_transient_bytes", &self.pending_transient_bytes)
            .field("last_submission_stats", &self.last_submission_stats)
            .field("execution", &self.execution)
            .finish_non_exhaustive()
    }
}

impl WgpuFrameSession {
    pub(crate) fn new(
        accelerator: WgpuAccelerator,
        frame: FrameSessionDesc,
        plan: Arc<RenderPlan>,
        execution: ExecutionPlan,
    ) -> Result<Self> {
        Scheduler::validate(&plan)?;
        Ok(Self {
            accelerator,
            frame,
            plan,
            execution,
            groups: BTreeMap::new(),
            resources: BTreeMap::new(),
            pending: BTreeMap::new(),
            next_token: 1,
            last_submission_stats: None,
            pending_transient_bytes: 0,
        })
    }

    pub const fn last_submission_stats(&self) -> Option<WgpuSubmissionStats> {
        self.last_submission_stats
    }

    /// Conservative sum of explicit transient bytes for tokens that have not
    /// been waited. Caller-owned GPU outputs may outlive their token and are
    /// therefore no longer included after `wait_gpu`.
    pub const fn pending_transient_bytes(&self) -> u64 {
        self.pending_transient_bytes
    }

    pub fn update_resource(&mut self, update: ResourceUpdate) -> Result<()> {
        if let Some(previous) = self.resources.get(&update.id)
            && update.revision <= previous.revision
        {
            return Err(Error::InvalidPayload(format!(
                "resource {:?} revision {} does not follow revision {}",
                update.id, update.revision, previous.revision
            )));
        }
        if let ResourceData::Plane(plane) = &update.data {
            plane
                .validate()
                .map_err(|error| Error::InvalidPayload(error.to_string()))?;
        }
        self.resources.insert(update.id, update);
        Ok(())
    }

    pub fn enqueue(&mut self, payload: GroupPayload) -> Result<()> {
        if payload.group.0 >= self.frame.group_count {
            return Err(Error::InvalidPayload(format!(
                "group {:?} is outside declared group count {}",
                payload.group, self.frame.group_count
            )));
        }
        if let Some(previous) = self.groups.get(&payload.group)
            && payload.revision <= previous.revision
        {
            return Err(Error::InvalidPayload(format!(
                "group {:?} revision {} does not follow revision {}",
                payload.group, payload.revision, previous.revision
            )));
        }
        if let Some(packet) = &payload.vardct {
            if !crate::vardct::has_node(&self.plan) {
                return Err(Error::InvalidPayload(
                    "a VarDCT packet was supplied to a plan without a VarDCT node".into(),
                ));
            }
            if packet.revision != payload.revision {
                return Err(Error::InvalidPayload(format!(
                    "VarDCT packet revision {} does not match group revision {}",
                    packet.revision, payload.revision
                )));
            }
            crate::vardct::validate_packet(packet)?;
        }
        for plane in &payload.planes {
            plane
                .validate()
                .map_err(|error| Error::InvalidPayload(error.to_string()))?;
            let Some(desc) = self.plan.planes.iter().find(|desc| desc.id == plane.id) else {
                return Err(Error::InvalidPayload(format!(
                    "group {:?} contains unknown plane {:?}",
                    payload.group, plane.id
                )));
            };
            if !matches!(
                desc.role,
                jxl_gpu_protocol::PlaneRole::Source | jxl_gpu_protocol::PlaneRole::Parameter
            ) {
                return Err(Error::InvalidPayload(format!(
                    "group {:?} attempts to upload non-source plane {:?}",
                    payload.group, plane.id
                )));
            }
            if desc.sample_type != plane.data.sample_type() {
                return Err(Error::InvalidPayload(format!(
                    "plane {:?} has {:?} data, expected {:?}",
                    plane.id,
                    plane.data.sample_type(),
                    desc.sample_type
                )));
            }
        }
        self.groups.insert(payload.group, payload);
        Ok(())
    }

    pub fn submit(&mut self, intent: RenderIntent) -> Result<SubmissionToken> {
        self.validate_submission(intent)?;

        let encoded = Scheduler::encode(
            &self.accelerator,
            &self.plan,
            &self.execution,
            &self.groups,
            &self.resources,
        )?;
        let stats = WgpuSubmissionStats {
            planned_dispatches: encoded.planned_dispatches,
            compute_dispatches: encoded.compute_dispatches,
            fused_dispatches: encoded.fused_dispatches,
            direct_readback: encoded.direct_readback,
            resident_bytes: encoded.resident_bytes,
            transient_bytes: encoded.transient_bytes,
        };
        let token = self.allocate_token()?;
        self.reserve_pending_transient(encoded.transient_bytes)?;
        let submission = self.accelerator.queue.submit([encoded.command_buffer]);
        recycle_submitted(encoded.recycle_after_submit);
        self.last_submission_stats = Some(stats);
        let changed = self.changed_regions();
        self.pending.insert(
            token,
            PendingSubmission::CpuReadback {
                submission,
                readbacks: encoded.readbacks,
                changed,
                recycle_after_wait: encoded.recycle_after_wait,
                transient_bytes: encoded.transient_bytes,
            },
        );
        Ok(token)
    }

    /// Submits the frame and returns packed GPU buffers without allocating, copying, or mapping a
    /// CPU readback buffer.
    ///
    /// This method does not wait for GPU completion. The returned buffers may be referenced by a
    /// later command submitted to the accelerator's queue immediately; queue ordering guarantees
    /// that the save kernels complete before the dependent command executes.
    pub fn submit_gpu(&mut self, intent: RenderIntent) -> Result<GpuFrame> {
        self.validate_submission(intent)?;

        let encoded = Scheduler::encode_gpu(
            &self.accelerator,
            &self.plan,
            &self.execution,
            &self.groups,
            &self.resources,
        )?;
        let stats = WgpuSubmissionStats {
            planned_dispatches: encoded.planned_dispatches,
            compute_dispatches: encoded.compute_dispatches,
            fused_dispatches: encoded.fused_dispatches,
            direct_readback: encoded.direct_readback,
            resident_bytes: encoded.resident_bytes,
            transient_bytes: encoded.transient_bytes,
        };
        let outputs = encoded
            .packed_outputs
            .into_iter()
            .map(|output| GpuOutputBuffer {
                id: output.id,
                extent: output.extent,
                sample_type: output.sample_type,
                channels: output.channels,
                layout: output.layout,
                logical_size: output.logical_size,
                buffer: output.buffer,
            })
            .collect();
        let token = self.allocate_token()?;
        self.reserve_pending_transient(encoded.transient_bytes)?;
        let submission = self.accelerator.queue.submit([encoded.command_buffer]);
        recycle_submitted(encoded.recycle_after_submit);
        debug_assert!(encoded.recycle_after_wait.is_empty());
        self.last_submission_stats = Some(stats);
        self.pending.insert(
            token,
            PendingSubmission::GpuOnly {
                submission,
                transient_bytes: encoded.transient_bytes,
            },
        );
        Ok(GpuFrame {
            token,
            outputs,
            changed: self.changed_regions(),
        })
    }

    /// Submits a generic pitch-linear image conversion and schedules CPU readback.
    ///
    /// The GPU writes the requested planar or NV12 layout directly from the render plan's first
    /// three saved F32 channels. No RGB(A) output buffer or RGB(A) readback is created.
    pub fn submit_image(
        &mut self,
        intent: RenderIntent,
        request: ImageOutputRequest,
    ) -> Result<SubmissionToken> {
        self.validate_submission(intent)?;
        let encoded = Scheduler::encode_image(
            &self.accelerator,
            &self.plan,
            &self.execution,
            &self.groups,
            &self.resources,
            &request,
        )?;
        let stats = WgpuSubmissionStats {
            planned_dispatches: encoded.planned_dispatches,
            compute_dispatches: encoded.compute_dispatches,
            fused_dispatches: encoded.fused_dispatches,
            direct_readback: encoded.direct_readback,
            resident_bytes: encoded.resident_bytes,
            transient_bytes: encoded.transient_bytes,
        };
        let token = self.allocate_token()?;
        self.reserve_pending_transient(encoded.transient_bytes)?;
        let submission = self.accelerator.queue.submit([encoded.command_buffer]);
        recycle_submitted(encoded.recycle_after_submit);
        self.last_submission_stats = Some(stats);
        let changed = self.changed_regions();
        self.pending.insert(
            token,
            PendingSubmission::CpuImageReadback {
                submission,
                readbacks: encoded.image_readbacks,
                changed,
                recycle_after_wait: encoded.recycle_after_wait,
                transient_bytes: encoded.transient_bytes,
            },
        );
        Ok(token)
    }

    /// Submits a generic pitch-linear image conversion and returns its GPU allocation.
    ///
    /// The returned buffer is tightly strided according to each output's descriptor. Commands on
    /// the accelerator's queue may consume it immediately because WebGPU preserves queue order.
    pub fn submit_gpu_image(
        &mut self,
        intent: RenderIntent,
        request: ImageOutputRequest,
    ) -> Result<GpuImageFrame> {
        self.validate_submission(intent)?;
        let encoded = Scheduler::encode_gpu_image(
            &self.accelerator,
            &self.plan,
            &self.execution,
            &self.groups,
            &self.resources,
            &request,
        )?;
        let stats = WgpuSubmissionStats {
            planned_dispatches: encoded.planned_dispatches,
            compute_dispatches: encoded.compute_dispatches,
            fused_dispatches: encoded.fused_dispatches,
            direct_readback: encoded.direct_readback,
            resident_bytes: encoded.resident_bytes,
            transient_bytes: encoded.transient_bytes,
        };
        let outputs = encoded
            .packed_image_outputs
            .into_iter()
            .map(|output| GpuImageOutput {
                id: output.id,
                layout: output.layout,
                buffer: output.buffer,
            })
            .collect();
        let token = self.allocate_token()?;
        self.reserve_pending_transient(encoded.transient_bytes)?;
        let submission = self.accelerator.queue.submit([encoded.command_buffer]);
        recycle_submitted(encoded.recycle_after_submit);
        debug_assert!(encoded.recycle_after_wait.is_empty());
        self.last_submission_stats = Some(stats);
        self.pending.insert(
            token,
            PendingSubmission::GpuOnly {
                submission,
                transient_bytes: encoded.transient_bytes,
            },
        );
        Ok(GpuImageFrame {
            token,
            outputs,
            changed: self.changed_regions(),
        })
    }

    fn validate_submission(&self, intent: RenderIntent) -> Result<()> {
        if matches!(intent, RenderIntent::Final) {
            for group_index in 0..self.frame.group_count {
                let id = GroupId(group_index);
                let Some(group) = self.groups.get(&id) else {
                    return Err(Error::InvalidPayload(format!(
                        "final submission is missing group {id:?}"
                    )));
                };
                if !group.complete {
                    return Err(Error::InvalidPayload(format!(
                        "final submission contains incomplete group {id:?}"
                    )));
                }
                if crate::vardct::has_node(&self.plan) && group.vardct.is_none() {
                    return Err(Error::InvalidPayload(format!(
                        "final submission is missing a VarDCT packet for group {id:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn allocate_token(&mut self) -> Result<SubmissionToken> {
        let token = SubmissionToken(self.next_token);
        self.next_token = self
            .next_token
            .checked_add(1)
            .ok_or_else(|| Error::Execution("submission token space was exhausted".into()))?;
        Ok(token)
    }

    fn reserve_pending_transient(&mut self, bytes: u64) -> Result<()> {
        self.pending_transient_bytes = self
            .pending_transient_bytes
            .checked_add(bytes)
            .ok_or(Error::BufferSizeOverflow)?;
        Ok(())
    }

    fn release_pending_transient(&mut self, bytes: u64) -> Result<()> {
        self.pending_transient_bytes = self
            .pending_transient_bytes
            .checked_sub(bytes)
            .ok_or_else(|| Error::Execution("pending transient accounting underflowed".into()))?;
        Ok(())
    }

    fn changed_regions(&self) -> ChangedRegions {
        ChangedRegions {
            outputs: self
                .plan
                .outputs
                .iter()
                .map(|output| {
                    (
                        output.id,
                        vec![Region::new(0, 0, output.extent.width, output.extent.height)],
                    )
                })
                .collect(),
        }
    }

    pub fn wait(&mut self, token: SubmissionToken) -> Result<AcceleratedFrame> {
        let pending = self
            .pending
            .get(&token)
            .ok_or(Error::UnknownSubmission(token.0))?;
        if pending.mode() != SubmissionMode::CpuReadback {
            return Err(Error::SubmissionModeMismatch {
                token: token.0,
                expected: SubmissionMode::CpuReadback,
                actual: pending.mode(),
            });
        }
        let pending = self
            .pending
            .remove(&token)
            .ok_or(Error::UnknownSubmission(token.0))?;
        self.release_pending_transient(pending.transient_bytes())?;
        let PendingSubmission::CpuReadback {
            submission,
            readbacks,
            changed,
            recycle_after_wait,
            transient_bytes: _,
        } = pending
        else {
            unreachable!("submission mode was checked before removal")
        };
        let outputs = resolve_outputs(&self.accelerator.device, submission, readbacks)?;
        recycle_unmapped(recycle_after_wait);
        Ok(AcceleratedFrame {
            token,
            outputs,
            changed,
        })
    }

    /// Waits for and maps a token returned by [`Self::submit_image`].
    pub fn wait_image(&mut self, token: SubmissionToken) -> Result<CpuImageFrame> {
        let pending = self
            .pending
            .get(&token)
            .ok_or(Error::UnknownSubmission(token.0))?;
        if pending.mode() != SubmissionMode::CpuImageReadback {
            return Err(Error::SubmissionModeMismatch {
                token: token.0,
                expected: SubmissionMode::CpuImageReadback,
                actual: pending.mode(),
            });
        }
        let pending = self
            .pending
            .remove(&token)
            .ok_or(Error::UnknownSubmission(token.0))?;
        self.release_pending_transient(pending.transient_bytes())?;
        let PendingSubmission::CpuImageReadback {
            submission,
            readbacks,
            changed,
            recycle_after_wait,
            transient_bytes: _,
        } = pending
        else {
            unreachable!("submission mode was checked before removal")
        };
        let outputs = resolve_image_outputs(&self.accelerator.device, submission, readbacks)?;
        recycle_unmapped(recycle_after_wait);
        Ok(CpuImageFrame {
            token,
            outputs,
            changed,
        })
    }

    /// Waits for native GPU completion without mapping or copying an output buffer.
    ///
    /// Browser WebGPU cannot implement a synchronous host wait; on `wasm32` this method returns a
    /// typed [`Error::Unsupported`] and leaves the token pending. Consumers submitted to the same
    /// queue never need this wait because WebGPU preserves submission order.
    pub fn wait_gpu(&mut self, token: SubmissionToken) -> Result<()> {
        let pending = self
            .pending
            .get(&token)
            .ok_or(Error::UnknownSubmission(token.0))?;
        if pending.mode() != SubmissionMode::GpuOnly {
            return Err(Error::SubmissionModeMismatch {
                token: token.0,
                expected: SubmissionMode::GpuOnly,
                actual: pending.mode(),
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            Err(browser_wait_error())
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let pending = self
                .pending
                .remove(&token)
                .ok_or(Error::UnknownSubmission(token.0))?;
            self.release_pending_transient(pending.transient_bytes())?;
            let PendingSubmission::GpuOnly {
                submission,
                transient_bytes: _,
            } = pending
            else {
                unreachable!("submission mode was checked before removal")
            };
            self.accelerator.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })?;
            Ok(())
        }
    }
}

/// Returning these buffers immediately after queue submission is safe: every acquisition belongs
/// to this accelerator and all subsequent commands enter the same totally ordered WebGPU queue.
fn recycle_submitted(buffers: Vec<PooledBuffer>) {
    for buffer in buffers {
        let _ = buffer.recycle();
    }
}

/// Direct readback buffers are mapped by `map_buffer_on_submit`, so their stronger reuse boundary
/// is successful decode followed by `Buffer::unmap` inside the resolver.
fn recycle_unmapped(buffers: Vec<PooledBuffer>) {
    for buffer in buffers {
        let _ = buffer.recycle();
    }
}

#[cfg(target_arch = "wasm32")]
fn browser_wait_error() -> Error {
    Error::Unsupported(
        "WgpuFrameSession::wait_gpu cannot synchronously block browser WebGPU; submit a dependent \
         command to the same queue or use the queue completion callback"
            .into(),
    )
}

#[cfg(not(target_arch = "wasm32"))]
impl AcceleratedFrameSession for WgpuFrameSession {
    fn update_resource(
        &mut self,
        update: ResourceUpdate,
    ) -> std::result::Result<(), AcceleratorError> {
        WgpuFrameSession::update_resource(self, update).map_err(Into::into)
    }

    fn enqueue(&mut self, payload: GroupPayload) -> std::result::Result<(), AcceleratorError> {
        WgpuFrameSession::enqueue(self, payload).map_err(Into::into)
    }

    fn submit(
        &mut self,
        intent: RenderIntent,
    ) -> std::result::Result<SubmissionToken, AcceleratorError> {
        WgpuFrameSession::submit(self, intent).map_err(Into::into)
    }

    fn wait(
        &mut self,
        token: SubmissionToken,
    ) -> std::result::Result<AcceleratedFrame, AcceleratorError> {
        WgpuFrameSession::wait(self, token).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::{
        ChromaLocation2d, ColorRange, ColorSpec, ColorSpecification, ImageOutputRequest,
        PixelFormat,
    };
    use crate::{WgpuAcceleratorConfig, WgpuMemoryPolicy};
    #[cfg(not(target_arch = "wasm32"))]
    use jxl_gpu_protocol::OutputOrientation;
    use jxl_gpu_protocol::{
        Border2d, ChromaAxis, FallbackGranularity, GaborishParams, HostPlane, MemoryMode,
        OutputDesc, OutputId, OutputLayout, PlaneData, PlaneDesc, PlaneId, PlaneRole,
        PrecisionContract, PrecisionPolicy, RenderNode, RenderOp, SaveParams, Scale2d,
        UpsampleParams,
    };

    fn test_accelerator() -> Option<WgpuAccelerator> {
        match pollster::block_on(WgpuAccelerator::request_default(WgpuAcceleratorConfig {
            enable_timestamps: false,
            ..WgpuAcceleratorConfig::default()
        })) {
            Ok(accelerator) => Some(accelerator),
            Err(Error::NoAdapter) => {
                eprintln!("skipping GPU test: no wgpu adapter is available");
                None
            }
            Err(error) => panic!("failed to initialize GPU test device: {error}"),
        }
    }

    fn frame_desc(extent: Extent2d) -> FrameSessionDesc {
        FrameSessionDesc {
            frame_extent: extent,
            group_extent: extent,
            group_count: 1,
            precision: PrecisionPolicy::F32Only,
            memory_mode: MemoryMode::Resident,
            max_resident_bytes: 16 * 1024 * 1024,
            max_scratch_bytes: 16 * 1024 * 1024,
            fallback: FallbackGranularity::WholeFrame,
        }
    }

    fn plane_desc(
        id: u32,
        extent: Extent2d,
        sample_type: SampleType,
        role: PlaneRole,
    ) -> PlaneDesc {
        PlaneDesc {
            id: PlaneId(id),
            extent,
            stride: extent.width,
            sample_type,
            role,
        }
    }

    fn render_node(
        name: &'static str,
        op: RenderOp,
        inputs: &[u32],
        outputs: &[u32],
        scale: Scale2d,
        precision: PrecisionContract,
    ) -> RenderNode {
        RenderNode {
            name: name.into(),
            op,
            inputs: inputs.iter().copied().map(PlaneId).collect(),
            outputs: outputs.iter().copied().map(PlaneId).collect(),
            resources: Vec::new(),
            scale,
            border: Border2d::default(),
            precision,
        }
    }

    fn copy_chain_plan(extent: Extent2d) -> Arc<RenderPlan> {
        let output = OutputId(0);
        Arc::new(RenderPlan {
            planes: vec![
                plane_desc(0, extent, SampleType::F32, PlaneRole::Source),
                plane_desc(1, extent, SampleType::F32, PlaneRole::Intermediate),
                plane_desc(2, extent, SampleType::F32, PlaneRole::Intermediate),
                plane_desc(3, extent, SampleType::F32, PlaneRole::Intermediate),
            ],
            nodes: vec![
                render_node(
                    "copy a",
                    RenderOp::Copy,
                    &[0],
                    &[1],
                    Scale2d::IDENTITY,
                    PrecisionContract::Exact,
                ),
                render_node(
                    "copy b",
                    RenderOp::Copy,
                    &[1],
                    &[2],
                    Scale2d::IDENTITY,
                    PrecisionContract::Exact,
                ),
                render_node(
                    "copy c",
                    RenderOp::Copy,
                    &[2],
                    &[3],
                    Scale2d::IDENTITY,
                    PrecisionContract::Exact,
                ),
                render_node(
                    "save copy chain",
                    RenderOp::Save(SaveParams {
                        output,
                        sample_type: SampleType::F32,
                        channels: vec![PlaneId(3)],
                        layout: OutputLayout::Planar,
                        orientation: jxl_gpu_protocol::OutputOrientation::Identity,
                    }),
                    &[3],
                    &[],
                    Scale2d::IDENTITY,
                    PrecisionContract::Exact,
                ),
            ],
            outputs: vec![OutputDesc {
                id: output,
                extent,
                sample_type: SampleType::F32,
                channels: 1,
                layout: OutputLayout::Planar,
            }],
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn enqueue_copy_source(session: &mut WgpuFrameSession, extent: Extent2d, values: Vec<f32>) {
        session
            .enqueue(GroupPayload {
                group: GroupId(0),
                revision: 0,
                complete: true,
                planes: vec![HostPlane {
                    id: PlaneId(0),
                    extent,
                    stride: extent.width,
                    origin: (0, 0),
                    data: PlaneData::F32(values),
                }],
                vardct: None,
            })
            .expect("enqueue copy source");
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn copy_gpu_output_to_host(accelerator: &WgpuAccelerator, output: &GpuOutputBuffer) -> Vec<u8> {
        use std::sync::mpsc;

        let copy_size = output.buffer.size();
        let staging = accelerator.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu zero-copy consumer test staging"),
            size: copy_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            accelerator
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu zero-copy consumer test"),
                });
        encoder.copy_buffer_to_buffer(&output.buffer, 0, &staging, 0, copy_size);
        let submission = accelerator.queue().submit([encoder.finish()]);
        let (sender, receiver) = mpsc::sync_channel(1);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        accelerator
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("wait for dependent zero-copy consumer");
        receiver
            .recv()
            .expect("zero-copy consumer mapping callback")
            .expect("map zero-copy consumer staging");
        let mapped = staging
            .slice(..)
            .get_mapped_range()
            .expect("read zero-copy consumer staging");
        let logical_size = usize::try_from(output.logical_size).expect("logical size fits usize");
        let bytes = mapped[..logical_size].to_vec();
        drop(mapped);
        staging.unmap();
        bytes
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn f32_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte f32")))
            .collect()
    }

    #[test]
    fn modular_kernel_executes_and_reads_back() {
        let Some(accelerator) = test_accelerator() else {
            return;
        };

        let extent = Extent2d::new(2, 2);
        let source_id = PlaneId(0);
        let converted_id = PlaneId(1);
        let output_id = OutputId(0);
        let plan = Arc::new(RenderPlan {
            planes: vec![
                PlaneDesc {
                    id: source_id,
                    extent,
                    stride: 2,
                    sample_type: SampleType::I32,
                    role: PlaneRole::Source,
                },
                PlaneDesc {
                    id: converted_id,
                    extent,
                    stride: 2,
                    sample_type: SampleType::F32,
                    role: PlaneRole::Intermediate,
                },
            ],
            nodes: vec![
                RenderNode {
                    name: "modular".into(),
                    op: RenderOp::ModularToF32 {
                        multiplier: 0.5,
                        bias: 1.0,
                    },
                    inputs: vec![source_id],
                    outputs: vec![converted_id],
                    resources: Vec::new(),
                    scale: Scale2d::IDENTITY,
                    border: Border2d::default(),
                    precision: PrecisionContract::default(),
                },
                RenderNode {
                    name: "save".into(),
                    op: RenderOp::Save(SaveParams {
                        output: output_id,
                        sample_type: SampleType::F32,
                        channels: vec![converted_id],
                        layout: OutputLayout::Planar,
                        orientation: jxl_gpu_protocol::OutputOrientation::Identity,
                    }),
                    inputs: vec![converted_id],
                    outputs: Vec::new(),
                    resources: Vec::new(),
                    scale: Scale2d::IDENTITY,
                    border: Border2d::default(),
                    precision: PrecisionContract::Exact,
                },
            ],
            outputs: vec![OutputDesc {
                id: output_id,
                extent,
                sample_type: SampleType::F32,
                channels: 1,
                layout: OutputLayout::Planar,
            }],
        });
        let frame = frame_desc(extent);
        let mut session = accelerator
            .create_session(&frame, plan)
            .expect("create GPU frame session");
        session
            .enqueue(GroupPayload {
                group: GroupId(0),
                revision: 0,
                complete: true,
                planes: vec![HostPlane {
                    id: source_id,
                    extent,
                    stride: 2,
                    origin: (0, 0),
                    data: PlaneData::I32(vec![1, 2, 3, 4]),
                }],
                vardct: None,
            })
            .expect("enqueue source plane");
        let token = session
            .submit(RenderIntent::Final)
            .expect("submit GPU work");
        let frame = session.wait(token).expect("read GPU output");
        let [output] = frame.outputs.as_slice() else {
            panic!("expected one output, got {}", frame.outputs.len());
        };
        let PlaneData::F32(values) = &output.data else {
            panic!("expected F32 output");
        };
        assert_eq!(values, &[1.5, 2.0, 2.5, 3.0]);
        assert_eq!(
            frame.changed.outputs[&output_id][0],
            Region::new(0, 0, 2, 2)
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn save_applies_all_eight_output_orientations_on_gpu() {
        let Some(accelerator) = test_accelerator() else {
            return;
        };
        let source_extent = Extent2d::new(3, 2);
        let source = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cases = [
            (
                OutputOrientation::Identity,
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
            (
                OutputOrientation::FlipHorizontal,
                vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
            ),
            (
                OutputOrientation::Rotate180,
                vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ),
            (
                OutputOrientation::FlipVertical,
                vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0],
            ),
            (
                OutputOrientation::Transpose,
                vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            ),
            (
                OutputOrientation::Rotate90Cw,
                vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0],
            ),
            (
                OutputOrientation::AntiTranspose,
                vec![6.0, 3.0, 5.0, 2.0, 4.0, 1.0],
            ),
            (
                OutputOrientation::Rotate90Ccw,
                vec![3.0, 6.0, 2.0, 5.0, 1.0, 4.0],
            ),
        ];

        for (orientation, expected) in cases {
            let output_extent = orientation.map_extent(source_extent);
            let plan = Arc::new(RenderPlan {
                planes: vec![plane_desc(
                    0,
                    source_extent,
                    SampleType::F32,
                    PlaneRole::Source,
                )],
                nodes: vec![render_node(
                    "oriented save",
                    RenderOp::Save(SaveParams {
                        output: OutputId(0),
                        sample_type: SampleType::F32,
                        channels: vec![PlaneId(0)],
                        layout: OutputLayout::Planar,
                        orientation,
                    }),
                    &[0],
                    &[],
                    Scale2d::IDENTITY,
                    PrecisionContract::Exact,
                )],
                outputs: vec![OutputDesc {
                    id: OutputId(0),
                    extent: output_extent,
                    sample_type: SampleType::F32,
                    channels: 1,
                    layout: OutputLayout::Planar,
                }],
            });
            let mut session = accelerator
                .create_session(&frame_desc(source_extent), plan)
                .expect("create oriented Save session");
            enqueue_copy_source(&mut session, source_extent, source.clone());
            let token = session
                .submit(RenderIntent::Final)
                .expect("submit oriented Save");
            let rendered = session.wait(token).expect("read oriented Save");
            let PlaneData::F32(actual) = &rendered.outputs[0].data else {
                panic!("expected oriented F32 output");
            };
            assert_eq!(actual, &expected, "orientation {orientation:?}");
            assert_eq!(rendered.outputs[0].extent, output_extent);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_yuv_applies_orientation_before_subsampling_and_packing() {
        let Some(accelerator) = test_accelerator() else {
            return;
        };
        let source_extent = Extent2d::new(3, 2);
        let source = vec![0.0, 0.1, 0.2, 0.3, 0.4, 1.0];
        let cases = [
            (
                OutputOrientation::Identity,
                vec![0.0, 0.1, 0.2, 0.3, 0.4, 1.0],
            ),
            (
                OutputOrientation::FlipHorizontal,
                vec![0.2, 0.1, 0.0, 1.0, 0.4, 0.3],
            ),
            (
                OutputOrientation::Rotate180,
                vec![1.0, 0.4, 0.3, 0.2, 0.1, 0.0],
            ),
            (
                OutputOrientation::FlipVertical,
                vec![0.3, 0.4, 1.0, 0.0, 0.1, 0.2],
            ),
            (
                OutputOrientation::Transpose,
                vec![0.0, 0.3, 0.1, 0.4, 0.2, 1.0],
            ),
            (
                OutputOrientation::Rotate90Cw,
                vec![0.3, 0.0, 0.4, 0.1, 1.0, 0.2],
            ),
            (
                OutputOrientation::AntiTranspose,
                vec![1.0, 0.2, 0.4, 0.1, 0.3, 0.0],
            ),
            (
                OutputOrientation::Rotate90Ccw,
                vec![0.2, 1.0, 0.1, 0.4, 0.0, 0.3],
            ),
        ];
        let format = PixelFormat::i444(
            8,
            8,
            ColorSpecification::Defined(ColorSpec::bt709(
                ColorRange::Full,
                ChromaLocation2d::CENTER,
            )),
        )
        .expect("I444 format");

        for (orientation, expected) in cases {
            let output_extent = orientation.map_extent(source_extent);
            let channels = vec![PlaneId(0), PlaneId(1), PlaneId(2)];
            let plan = Arc::new(RenderPlan {
                planes: channels
                    .iter()
                    .map(|&id| PlaneDesc {
                        id,
                        extent: source_extent,
                        stride: source_extent.width,
                        sample_type: SampleType::F32,
                        role: PlaneRole::Source,
                    })
                    .collect(),
                nodes: vec![RenderNode {
                    name: "oriented native YUV save".into(),
                    op: RenderOp::Save(SaveParams {
                        output: OutputId(0),
                        sample_type: SampleType::F32,
                        channels: channels.clone(),
                        layout: OutputLayout::Interleaved,
                        orientation,
                    }),
                    inputs: channels.clone(),
                    outputs: Vec::new(),
                    resources: Vec::new(),
                    scale: Scale2d::IDENTITY,
                    border: Border2d::default(),
                    precision: PrecisionContract::Exact,
                }],
                outputs: vec![OutputDesc {
                    id: OutputId(0),
                    extent: output_extent,
                    sample_type: SampleType::F32,
                    channels: 3,
                    layout: OutputLayout::Interleaved,
                }],
            });
            let mut session = accelerator
                .create_session(&frame_desc(source_extent), plan)
                .expect("create oriented native YUV session");
            session
                .enqueue(GroupPayload {
                    group: GroupId(0),
                    revision: 0,
                    complete: true,
                    planes: channels
                        .iter()
                        .map(|&id| HostPlane {
                            id,
                            extent: source_extent,
                            stride: source_extent.width,
                            origin: (0, 0),
                            data: PlaneData::F32(source.clone()),
                        })
                        .collect(),
                    vardct: None,
                })
                .expect("enqueue oriented native YUV source");
            let token = session
                .submit_image(RenderIntent::Final, ImageOutputRequest::new(format.clone()))
                .expect("submit oriented native YUV");
            let rendered = session.wait_image(token).expect("read oriented native YUV");
            let output = &rendered.outputs[0];
            let y_plane = output.layout.plane(0).expect("oriented Y plane");
            let mut actual = Vec::new();
            for y in 0..output_extent.height {
                let start = usize::try_from(y_plane.offset + u64::from(y) * y_plane.row_stride)
                    .expect("Y row offset");
                let end = start + output_extent.width as usize;
                actual.extend_from_slice(&output.bytes[start..end]);
            }
            let expected = expected
                .into_iter()
                .map(|value| (value * 255.0_f32 + 0.5).floor() as u8)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "YUV orientation {orientation:?}");
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_output_is_packed_queue_ordered_and_outlives_session() {
        let Some(accelerator) = test_accelerator() else {
            return;
        };
        let extent = Extent2d::new(3, 2);
        let expected = vec![-3.0, 0.25, 1.5, 7.0, -0.0, 22.25];
        let mut session = accelerator
            .create_session(&frame_desc(extent), copy_chain_plan(extent))
            .expect("create zero-copy session");
        enqueue_copy_source(&mut session, extent, expected.clone());

        let frame = session
            .submit_gpu(RenderIntent::Final)
            .expect("submit zero-copy GPU output");
        let stats = session
            .last_submission_stats()
            .expect("submission publishes memory and dispatch stats");
        assert_eq!(stats.planned_dispatches, 4);
        assert_eq!(stats.compute_dispatches, 4);
        assert_eq!(stats.fused_dispatches, 0);
        assert!(!stats.direct_readback);
        assert!(stats.resident_bytes > 0);
        assert!(stats.transient_bytes > 0);
        assert_eq!(session.pending_transient_bytes(), stats.transient_bytes);
        let [output] = frame.outputs.as_slice() else {
            panic!("expected one zero-copy output");
        };
        assert_eq!(output.id, OutputId(0));
        assert_eq!(output.extent, extent);
        assert_eq!(output.sample_type, SampleType::F32);
        assert_eq!(output.channels, 1);
        assert_eq!(output.layout, OutputLayout::Planar);
        assert_eq!(output.logical_size, 24);
        assert!(output.buffer.size() >= output.logical_size);
        assert_eq!(
            frame.changed.outputs[&OutputId(0)],
            [Region::new(0, 0, 3, 2)]
        );

        let output = output.clone();
        drop(frame);
        drop(session);

        // This dependent copy is submitted without waiting for the frame submission. Waiting only
        // for this later command proves same-queue ordering and that the Arc-owned output survives
        // destruction of its frame session.
        let actual = f32_values(&copy_gpu_output_to_host(&accelerator, &output));
        assert_eq!(actual, expected);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn pending_transient_accounting_accumulates_and_releases() {
        let Some(accelerator) = test_accelerator() else {
            return;
        };
        let extent = Extent2d::new(3, 2);
        let mut session = accelerator
            .create_session(&frame_desc(extent), copy_chain_plan(extent))
            .expect("create pending-memory session");
        enqueue_copy_source(&mut session, extent, vec![0.5; 6]);

        let first = session
            .submit_gpu(RenderIntent::Final)
            .expect("submit first pending frame");
        let per_submission = session
            .last_submission_stats()
            .expect("first submission stats")
            .transient_bytes;
        let second = session
            .submit_gpu(RenderIntent::Final)
            .expect("submit second pending frame");
        assert_eq!(
            session.pending_transient_bytes(),
            per_submission.checked_mul(2).expect("test total fits")
        );

        session.wait_gpu(first.token).expect("wait first frame");
        assert_eq!(session.pending_transient_bytes(), per_submission);
        session.wait_gpu(second.token).expect("wait second frame");
        assert_eq!(session.pending_transient_bytes(), 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn internal_buffers_are_reused_across_sessions_but_public_gpu_output_is_not_pooled() {
        let Some(accelerator) = test_accelerator() else {
            return;
        };
        accelerator.clear_buffer_pool();
        let extent = Extent2d::new(7, 5);
        let plan = copy_chain_plan(extent);

        let mut first = accelerator
            .create_session(&frame_desc(extent), Arc::clone(&plan))
            .expect("create first pooled CPU session");
        let resident_slots = first
            .execution
            .arena
            .allocations
            .iter()
            .map(|allocation| allocation.offset)
            .collect::<std::collections::BTreeSet<_>>()
            .len() as u64;
        enqueue_copy_source(&mut first, extent, vec![3.0; 35]);
        let token = first
            .submit(RenderIntent::Final)
            .expect("submit first pooled CPU frame");
        first.wait(token).expect("wait first pooled CPU frame");
        let after_first = accelerator.buffer_pool_stats();
        assert_eq!(after_first.misses, resident_slots + 1);
        assert_eq!(after_first.cached_buffers, resident_slots + 1);

        let mut second = accelerator
            .create_session(&frame_desc(extent), Arc::clone(&plan))
            .expect("create second pooled CPU session");
        enqueue_copy_source(&mut second, extent, vec![-9.0; 35]);
        let token = second
            .submit(RenderIntent::Final)
            .expect("submit second pooled CPU frame");
        let rendered = second.wait(token).expect("wait second pooled CPU frame");
        let PlaneData::F32(values) = &rendered.outputs[0].data else {
            panic!("expected pooled F32 output");
        };
        assert!(values.iter().all(|&value| value == -9.0));
        let after_second = accelerator.buffer_pool_stats();
        assert_eq!(after_second.hits - after_first.hits, resident_slots + 1);

        accelerator.clear_buffer_pool();
        let before_gpu = accelerator.buffer_pool_stats();
        let mut gpu = accelerator
            .create_session(&frame_desc(extent), plan)
            .expect("create GPU-only pool-boundary session");
        enqueue_copy_source(&mut gpu, extent, vec![11.0; 35]);
        let gpu_frame = gpu
            .submit_gpu(RenderIntent::Final)
            .expect("submit GPU-only pool-boundary frame");
        gpu.wait_gpu(gpu_frame.token)
            .expect("wait GPU-only pool-boundary frame");
        assert_eq!(gpu.pending_transient_bytes(), 0);
        let after_gpu = accelerator.buffer_pool_stats();
        assert_eq!(after_gpu.misses - before_gpu.misses, resident_slots);
        assert_eq!(after_gpu.cached_buffers, resident_slots);

        // Clearing all internal buffers cannot invalidate the caller-owned packed output.
        accelerator.clear_buffer_pool();
        let actual = f32_values(&copy_gpu_output_to_host(
            &accelerator,
            &gpu_frame.outputs[0],
        ));
        assert!(actual.iter().all(|&value| value == 11.0));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "manual release-mode allocation benchmark"]
    fn repeated_cpu_decode_pool_release_benchmark() {
        fn run(max_cached_buffer_bytes: u64) -> Option<(std::time::Duration, u64)> {
            let config = WgpuAcceleratorConfig {
                enable_timestamps: false,
                memory: WgpuMemoryPolicy {
                    max_cached_buffer_bytes,
                    ..WgpuMemoryPolicy::default()
                },
                ..WgpuAcceleratorConfig::default()
            };
            let accelerator = match pollster::block_on(WgpuAccelerator::request_default(config)) {
                Ok(accelerator) => accelerator,
                Err(Error::NoAdapter) => return None,
                Err(error) => panic!("request benchmark adapter: {error}"),
            };
            let extent = Extent2d::new(512, 512);
            let plan = copy_chain_plan(extent);
            let source = vec![0.25; extent.area().expect("benchmark extent")];

            // Compile pipelines before measuring allocation reuse.
            let mut warm = accelerator
                .create_session(&frame_desc(extent), Arc::clone(&plan))
                .expect("create benchmark warmup session");
            enqueue_copy_source(&mut warm, extent, source.clone());
            let token = warm
                .submit(RenderIntent::Final)
                .expect("submit benchmark warmup");
            warm.wait(token).expect("wait benchmark warmup");
            accelerator.clear_buffer_pool();

            let started = std::time::Instant::now();
            for _ in 0..20 {
                let mut session = accelerator
                    .create_session(&frame_desc(extent), Arc::clone(&plan))
                    .expect("create repeated benchmark session");
                enqueue_copy_source(&mut session, extent, source.clone());
                let token = session
                    .submit(RenderIntent::Final)
                    .expect("submit repeated benchmark frame");
                session.wait(token).expect("wait repeated benchmark frame");
            }
            Some((started.elapsed(), accelerator.buffer_pool_stats().hits))
        }

        let Some((uncached, uncached_hits)) = run(0) else {
            eprintln!("skipping GPU buffer pool benchmark: no adapter");
            return;
        };
        let Some((cached, cached_hits)) = run(32 * 1024 * 1024) else {
            eprintln!("skipping GPU buffer pool benchmark: no adapter");
            return;
        };
        eprintln!(
            "20x 512x512 CPU readback: pool off {uncached:?}, pool on {cached:?}, ratio {:.3}",
            cached.as_secs_f64() / uncached.as_secs_f64()
        );
        assert_eq!(uncached_hits, 0);
        assert!(cached_hits > 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn cpu_and_gpu_submission_tokens_reject_the_wrong_wait_mode() {
        let Some(accelerator) = test_accelerator() else {
            return;
        };
        let extent = Extent2d::new(1, 1);

        let mut cpu = accelerator
            .create_session(&frame_desc(extent), copy_chain_plan(extent))
            .expect("create CPU-readback token session");
        enqueue_copy_source(&mut cpu, extent, vec![4.0]);
        let cpu_token = cpu
            .submit(RenderIntent::Final)
            .expect("submit CPU readback");
        assert!(matches!(
            cpu.wait_gpu(cpu_token),
            Err(Error::SubmissionModeMismatch {
                token,
                expected: SubmissionMode::GpuOnly,
                actual: SubmissionMode::CpuReadback,
            }) if token == cpu_token.0
        ));
        cpu.wait(cpu_token)
            .expect("mode mismatch must leave CPU token pending");

        let mut gpu = accelerator
            .create_session(&frame_desc(extent), copy_chain_plan(extent))
            .expect("create GPU-only token session");
        enqueue_copy_source(&mut gpu, extent, vec![8.0]);
        let frame = gpu
            .submit_gpu(RenderIntent::Final)
            .expect("submit GPU-only output");
        assert!(matches!(
            gpu.wait(frame.token),
            Err(Error::SubmissionModeMismatch {
                token,
                expected: SubmissionMode::CpuReadback,
                actual: SubmissionMode::GpuOnly,
            }) if token == frame.token.0
        ));
        gpu.wait_gpu(frame.token)
            .expect("mode mismatch must leave GPU token pending");
    }

    #[test]
    fn disjoint_lifetimes_reuse_physical_slots_on_gpu() {
        let Some(accelerator) = test_accelerator() else {
            return;
        };
        let extent = Extent2d::new(19, 11);
        let mut session = accelerator
            .create_session(&frame_desc(extent), copy_chain_plan(extent))
            .expect("create aliased copy chain session");
        let offset = |plane| {
            session
                .execution
                .arena
                .allocation(PlaneId(plane))
                .expect("copy plane allocation")
                .offset
        };
        assert_eq!(offset(0), offset(2));
        assert_eq!(offset(1), offset(3));
        assert_ne!(offset(0), offset(1));

        let values = (0..extent.width * extent.height)
            .map(|index| index as f32 * 0.125 - 3.0)
            .collect::<Vec<_>>();
        session
            .enqueue(GroupPayload {
                group: GroupId(0),
                revision: 0,
                complete: true,
                planes: vec![HostPlane {
                    id: PlaneId(0),
                    extent,
                    stride: extent.width,
                    origin: (0, 0),
                    data: PlaneData::F32(values.clone()),
                }],
                vardct: None,
            })
            .expect("enqueue aliased copy source");
        let token = session
            .submit(RenderIntent::Final)
            .expect("submit aliased copy chain");
        let rendered = session.wait(token).expect("read aliased copy output");
        let PlaneData::F32(actual) = &rendered.outputs[0].data else {
            panic!("expected F32 aliased copy output");
        };
        assert_eq!(actual, &values);
    }

    #[test]
    fn simultaneously_live_slot_alias_returns_typed_error_before_gpu_validation() {
        let Some(accelerator) = test_accelerator() else {
            return;
        };
        let extent = Extent2d::new(3, 3);
        let mut session = accelerator
            .create_session(&frame_desc(extent), copy_chain_plan(extent))
            .expect("create copy chain session");
        let source_offset = session
            .execution
            .arena
            .allocation(PlaneId(0))
            .expect("source allocation")
            .offset;
        session
            .execution
            .arena
            .allocations
            .iter_mut()
            .find(|allocation| allocation.plane == PlaneId(1))
            .expect("first output allocation")
            .offset = source_offset;
        session
            .enqueue(GroupPayload {
                group: GroupId(0),
                revision: 0,
                complete: true,
                planes: vec![HostPlane {
                    id: PlaneId(0),
                    extent,
                    stride: extent.width,
                    origin: (0, 0),
                    data: PlaneData::F32(vec![1.0; 9]),
                }],
                vardct: None,
            })
            .expect("enqueue malformed-plan source");

        assert!(matches!(
            session.submit(RenderIntent::Final),
            Err(Error::Execution(message)) if message.contains("simultaneously live")
        ));
    }

    #[test]
    fn gpu_only_budget_counts_one_packed_buffer_and_no_readback() {
        // Three copy uniforms (48), one save uniform (32), and one 36-byte packed output fit 116
        // bytes exactly. CPU mode additionally needs a 36-byte readback and must fail at 152.
        let config = WgpuAcceleratorConfig {
            enable_timestamps: false,
            memory: WgpuMemoryPolicy {
                max_transient_bytes: 116,
                ..WgpuMemoryPolicy::default()
            },
            ..WgpuAcceleratorConfig::default()
        };
        let accelerator = match pollster::block_on(WgpuAccelerator::request_default(config)) {
            Ok(accelerator) => accelerator,
            Err(Error::NoAdapter) => {
                eprintln!("skipping GPU test: no wgpu adapter is available");
                return;
            }
            Err(error) => panic!("failed to initialize GPU test device: {error}"),
        };
        let extent = Extent2d::new(3, 3);
        let mut session = accelerator
            .create_session(&frame_desc(extent), copy_chain_plan(extent))
            .expect("create transient-budget session");
        session
            .enqueue(GroupPayload {
                group: GroupId(0),
                revision: 0,
                complete: true,
                planes: vec![HostPlane {
                    id: PlaneId(0),
                    extent,
                    stride: extent.width,
                    origin: (0, 0),
                    data: PlaneData::F32(vec![1.0; 9]),
                }],
                vardct: None,
            })
            .expect("enqueue transient-budget source");

        if accelerator.direct_readback_enabled() {
            let token = session
                .submit(RenderIntent::Final)
                .expect("direct readback fits without staging");
            assert_eq!(
                session
                    .last_submission_stats()
                    .map(|stats| stats.direct_readback),
                Some(true)
            );
            session.wait(token).expect("map direct primary output");
        } else {
            assert!(matches!(
                session.submit(RenderIntent::Final),
                Err(Error::ResourceLimit(message))
                    if message.contains("submission needs 152 bytes")
                        && message.contains("limit of 116 bytes")
            ));
        }
        let frame = session
            .submit_gpu(RenderIntent::Final)
            .expect("GPU-only submission fits without a readback buffer");
        assert_eq!(frame.outputs.len(), 1);
        session
            .wait_gpu(frame.token)
            .expect("wait for budget-boundary GPU submission");
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn browser_wait_contract_reports_synchronous_wait_as_unsupported() {
        assert!(matches!(
            browser_wait_error(),
            Error::Unsupported(message)
                if message.contains("cannot synchronously block browser WebGPU")
                    && message.contains("same queue")
        ));
    }

    #[test]
    fn every_portable_shader_executes_in_one_submission() {
        let Some(accelerator) = test_accelerator() else {
            return;
        };
        let one = Extent2d::new(1, 1);
        let two = Extent2d::new(2, 2);
        let output_id = OutputId(0);
        let mut weights = vec![0.0; 4 * 25];
        for phase in 0..4 {
            weights[phase * 25 + 12] = 1.0;
        }
        let plan = Arc::new(RenderPlan {
            planes: vec![
                plane_desc(0, one, SampleType::I32, PlaneRole::Source),
                plane_desc(1, one, SampleType::F32, PlaneRole::Intermediate),
                plane_desc(2, one, SampleType::F32, PlaneRole::Intermediate),
                plane_desc(3, one, SampleType::F32, PlaneRole::Intermediate),
                plane_desc(4, two, SampleType::F32, PlaneRole::Intermediate),
                plane_desc(5, two, SampleType::F32, PlaneRole::Source),
                plane_desc(6, two, SampleType::F32, PlaneRole::Source),
                plane_desc(7, two, SampleType::F32, PlaneRole::Intermediate),
                plane_desc(8, two, SampleType::F32, PlaneRole::Intermediate),
                plane_desc(9, two, SampleType::F32, PlaneRole::Intermediate),
                plane_desc(10, two, SampleType::F32, PlaneRole::Source),
                plane_desc(11, two, SampleType::F32, PlaneRole::Intermediate),
                plane_desc(12, two, SampleType::F32, PlaneRole::Intermediate),
                plane_desc(13, two, SampleType::F32, PlaneRole::Intermediate),
                plane_desc(14, two, SampleType::F32, PlaneRole::Intermediate),
                plane_desc(15, two, SampleType::F32, PlaneRole::Intermediate),
            ],
            nodes: vec![
                render_node(
                    "modular",
                    RenderOp::ModularToF32 {
                        multiplier: 1.0,
                        bias: 0.0,
                    },
                    &[0],
                    &[1],
                    Scale2d::IDENTITY,
                    PrecisionContract::default(),
                ),
                render_node(
                    "copy",
                    RenderOp::Copy,
                    &[1],
                    &[2],
                    Scale2d::IDENTITY,
                    PrecisionContract::Exact,
                ),
                {
                    let mut node = render_node(
                        "gaborish",
                        RenderOp::Gaborish(GaborishParams {
                            channel: 0,
                            weight0: 1.0,
                            weight1: 0.0,
                            weight2: 0.0,
                        }),
                        &[2],
                        &[3],
                        Scale2d::IDENTITY,
                        PrecisionContract::default(),
                    );
                    node.border = Border2d::symmetric(1, 1);
                    node
                },
                {
                    let mut node = render_node(
                        "upsample",
                        RenderOp::Upsample(UpsampleParams {
                            factor: 2,
                            weights: weights.into(),
                        }),
                        &[3],
                        &[4],
                        Scale2d::new(2, 2),
                        PrecisionContract::default(),
                    );
                    node.border = Border2d::symmetric(2, 2);
                    node
                },
                render_node(
                    "ycbcr",
                    RenderOp::YcbcrToRgb,
                    &[5, 4, 6],
                    &[7, 8, 9],
                    Scale2d::IDENTITY,
                    PrecisionContract::default(),
                ),
                render_node(
                    "premultiply",
                    RenderOp::PremultiplyAlpha {
                        alpha_plane: PlaneId(10),
                    },
                    &[7, 8, 9, 10],
                    &[11, 12, 13, 14],
                    Scale2d::IDENTITY,
                    PrecisionContract::default(),
                ),
                render_node(
                    "convert",
                    RenderOp::Convert {
                        output_type: SampleType::F32,
                    },
                    &[11],
                    &[15],
                    Scale2d::IDENTITY,
                    PrecisionContract::default(),
                ),
                render_node(
                    "save",
                    RenderOp::Save(SaveParams {
                        output: output_id,
                        sample_type: SampleType::F32,
                        channels: vec![PlaneId(15), PlaneId(12), PlaneId(13), PlaneId(14)],
                        layout: OutputLayout::Interleaved,
                        orientation: jxl_gpu_protocol::OutputOrientation::Identity,
                    }),
                    &[15, 12, 13, 14],
                    &[],
                    Scale2d::IDENTITY,
                    PrecisionContract::Exact,
                ),
            ],
            outputs: vec![OutputDesc {
                id: output_id,
                extent: two,
                sample_type: SampleType::F32,
                channels: 4,
                layout: OutputLayout::Interleaved,
            }],
        });
        let mut session = accelerator
            .create_session(&frame_desc(two), plan)
            .expect("create portable shader session");
        session
            .enqueue(GroupPayload {
                group: GroupId(0),
                revision: 0,
                complete: true,
                planes: vec![
                    HostPlane {
                        id: PlaneId(0),
                        extent: one,
                        stride: 1,
                        origin: (0, 0),
                        data: PlaneData::I32(vec![0]),
                    },
                    HostPlane {
                        id: PlaneId(5),
                        extent: two,
                        stride: 2,
                        origin: (0, 0),
                        data: PlaneData::F32(vec![0.0; 4]),
                    },
                    HostPlane {
                        id: PlaneId(6),
                        extent: two,
                        stride: 2,
                        origin: (0, 0),
                        data: PlaneData::F32(vec![0.0; 4]),
                    },
                    HostPlane {
                        id: PlaneId(10),
                        extent: two,
                        stride: 2,
                        origin: (0, 0),
                        data: PlaneData::F32(vec![0.5; 4]),
                    },
                ],
                vardct: None,
            })
            .expect("enqueue shader chain inputs");
        let token = session
            .submit(RenderIntent::Final)
            .expect("submit portable shader chain");
        let rendered = session.wait(token).expect("read portable shader output");
        let PlaneData::F32(values) = &rendered.outputs[0].data else {
            panic!("expected F32 shader output");
        };
        for pixel in values.chunks_exact(4) {
            let expected_color = 0.5 * (128.0 / 255.0);
            assert!((pixel[0] - expected_color).abs() < 1.0e-6);
            assert!((pixel[1] - expected_color).abs() < 1.0e-6);
            assert!((pixel[2] - expected_color).abs() < 1.0e-6);
            assert_eq!(pixel[3], 0.5);
        }
    }

    #[test]
    fn upsample_2x_4x_and_8x_execute_on_gpu() {
        let Some(accelerator) = test_accelerator() else {
            return;
        };
        for factor in [2_u8, 4, 8] {
            let input_extent = Extent2d::new(1, 1);
            let output_extent = Extent2d::new(u32::from(factor), u32::from(factor));
            let phase_count = usize::from(factor) * usize::from(factor);
            let mut weights = vec![0.0; phase_count * 25];
            for phase in 0..phase_count {
                weights[phase * 25 + 12] = 1.0;
            }
            let output_id = OutputId(0);
            let plan = Arc::new(RenderPlan {
                planes: vec![
                    plane_desc(0, input_extent, SampleType::F32, PlaneRole::Source),
                    plane_desc(1, output_extent, SampleType::F32, PlaneRole::Intermediate),
                ],
                nodes: vec![
                    {
                        let mut node = render_node(
                            "upsample",
                            RenderOp::Upsample(UpsampleParams {
                                factor,
                                weights: weights.into(),
                            }),
                            &[0],
                            &[1],
                            Scale2d::new(factor, factor),
                            PrecisionContract::default(),
                        );
                        node.border = Border2d::symmetric(2, 2);
                        node
                    },
                    render_node(
                        "save",
                        RenderOp::Save(SaveParams {
                            output: output_id,
                            sample_type: SampleType::F32,
                            channels: vec![PlaneId(1)],
                            layout: OutputLayout::Planar,
                            orientation: jxl_gpu_protocol::OutputOrientation::Identity,
                        }),
                        &[1],
                        &[],
                        Scale2d::IDENTITY,
                        PrecisionContract::Exact,
                    ),
                ],
                outputs: vec![OutputDesc {
                    id: output_id,
                    extent: output_extent,
                    sample_type: SampleType::F32,
                    channels: 1,
                    layout: OutputLayout::Planar,
                }],
            });
            let mut session = accelerator
                .create_session(&frame_desc(output_extent), plan)
                .expect("create upsample session");
            session
                .enqueue(GroupPayload {
                    group: GroupId(0),
                    revision: 0,
                    complete: true,
                    planes: vec![HostPlane {
                        id: PlaneId(0),
                        extent: input_extent,
                        stride: 1,
                        origin: (0, 0),
                        data: PlaneData::F32(vec![0.75]),
                    }],
                    vardct: None,
                })
                .expect("enqueue upsample source");
            let token = session
                .submit(RenderIntent::Final)
                .expect("submit upsample");
            let rendered = session.wait(token).expect("read upsample output");
            let PlaneData::F32(values) = &rendered.outputs[0].data else {
                panic!("expected F32 upsample output");
            };
            assert_eq!(values.len(), phase_count);
            assert!(values.iter().all(|value| *value == 0.75));
        }
    }

    fn execute_chroma_case(
        accelerator: &WgpuAccelerator,
        axis: ChromaAxis,
        input_extent: Extent2d,
        output_extent: Extent2d,
        input_values: Vec<f32>,
    ) -> Vec<f32> {
        let output_id = OutputId(0);
        let mut chroma = render_node(
            "chroma",
            RenderOp::ChromaUpsample { axis },
            &[0],
            &[1],
            match axis {
                ChromaAxis::Horizontal => Scale2d::new(2, 1),
                ChromaAxis::Vertical => Scale2d::new(1, 2),
            },
            PrecisionContract::default(),
        );
        chroma.border = match axis {
            ChromaAxis::Horizontal => Border2d::symmetric(1, 0),
            ChromaAxis::Vertical => Border2d::symmetric(0, 1),
        };
        let plan = Arc::new(RenderPlan {
            planes: vec![
                plane_desc(0, input_extent, SampleType::F32, PlaneRole::Source),
                plane_desc(1, output_extent, SampleType::F32, PlaneRole::Intermediate),
            ],
            nodes: vec![
                chroma,
                render_node(
                    "save chroma",
                    RenderOp::Save(SaveParams {
                        output: output_id,
                        sample_type: SampleType::F32,
                        channels: vec![PlaneId(1)],
                        layout: OutputLayout::Planar,
                        orientation: jxl_gpu_protocol::OutputOrientation::Identity,
                    }),
                    &[1],
                    &[],
                    Scale2d::IDENTITY,
                    PrecisionContract::Exact,
                ),
            ],
            outputs: vec![OutputDesc {
                id: output_id,
                extent: output_extent,
                sample_type: SampleType::F32,
                channels: 1,
                layout: OutputLayout::Planar,
            }],
        });
        let mut session = accelerator
            .create_session(&frame_desc(output_extent), plan)
            .expect("create chroma upsample session");
        session
            .enqueue(GroupPayload {
                group: GroupId(0),
                revision: 0,
                complete: true,
                planes: vec![HostPlane {
                    id: PlaneId(0),
                    extent: input_extent,
                    stride: input_extent.width,
                    origin: (0, 0),
                    data: PlaneData::F32(input_values),
                }],
                vardct: None,
            })
            .expect("enqueue chroma source");
        let token = session
            .submit(RenderIntent::Final)
            .expect("submit chroma upsample");
        let rendered = session.wait(token).expect("read chroma output");
        let PlaneData::F32(values) = &rendered.outputs[0].data else {
            panic!("expected F32 chroma output");
        };
        values.clone()
    }

    #[test]
    fn chroma_upsample_matches_codec_edges_on_gpu() {
        let Some(accelerator) = test_accelerator() else {
            return;
        };
        assert_eq!(
            execute_chroma_case(
                &accelerator,
                ChromaAxis::Horizontal,
                Extent2d::new(3, 1),
                Extent2d::new(5, 1),
                vec![1.0, 2.0, 4.0],
            ),
            [1.0, 1.25, 1.75, 2.5, 3.5]
        );
        assert_eq!(
            execute_chroma_case(
                &accelerator,
                ChromaAxis::Horizontal,
                Extent2d::new(2, 1),
                Extent2d::new(4, 1),
                vec![1.0, 3.0],
            ),
            [1.0, 1.5, 2.5, 3.0]
        );
        assert_eq!(
            execute_chroma_case(
                &accelerator,
                ChromaAxis::Horizontal,
                Extent2d::new(1, 1),
                Extent2d::new(2, 1),
                vec![7.0],
            ),
            [7.0, 7.0]
        );
        assert_eq!(
            execute_chroma_case(
                &accelerator,
                ChromaAxis::Vertical,
                Extent2d::new(1, 3),
                Extent2d::new(1, 5),
                vec![1.0, 2.0, 4.0],
            ),
            [1.0, 1.25, 1.75, 2.5, 3.5]
        );
    }
}
