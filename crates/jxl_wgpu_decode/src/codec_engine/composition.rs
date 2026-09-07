//! Serial physical-frame execution with resident reference versions and lazy producer admission.
//! Caller-visible buffers are packed only after all zero-duration layers have been validated.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};

use jxl_gpu_bitstream::{
    CodestreamInventory, ColourEncodingInventory, ColourSpaceInventory, FrameType,
    PrimariesInventory, TransferFunctionInventory, WhitePointInventory,
};
use jxl_gpu_formats::{PixelFormat, RgbChannelOrder};
use jxl_gpu_protocol::{
    ChangedRegions, Extent2d, OutputId, OutputOrientation, Region, SubmissionToken,
};
use jxl_wgpu::{
    GpuImageFrame, GpuImageOutput, UnvalidatedGpuImageFrame, UnvalidatedGpuImageOutput,
};

use super::sequence::{SequenceSource, submission_counter};
use super::{WgpuDecodeEngine, WgpuDecodePendingFrame, WgpuDecodeSubmissionSession};
use crate::{
    Error, FrameExecutionPlan, FrameMetadata, FramePlanError, GpuCodestream, GpuOutputRequest,
    GpuPendingFrame, GpuSubmissionSession, OrientationPolicy, Result, SubmittedGpuFrame,
    UnsupportedCodestreamFeature, UnsupportedProfile,
};

mod gpu;
use gpu::{Compositor, GpuWork, Surface};

#[derive(Debug)]
struct Carry {
    source: SequenceSource,
    references: [Option<Surface>; 4],
    prepared: Option<WgpuDecodeSubmissionSession>,
}

#[derive(Debug)]
struct Shared {
    carry: Option<Carry>,
    in_flight: Option<usize>,
    failed: bool,
}

/// The session and its one active presentation share only a handoff cell. GPU completion never
/// locks the cell: the pending state returns the next reference version after validation.
#[derive(Debug)]
pub(super) struct CompositionSession {
    shared: Arc<Mutex<Shared>>,
    compositor: Arc<Compositor>,
    submissions: Arc<AtomicUsize>,
}

impl CompositionSession {
    pub(super) fn new(
        engine: WgpuDecodeEngine,
        codestream: Arc<GpuCodestream>,
        inventory: &CodestreamInventory,
        request: &GpuOutputRequest,
        plan: &FrameExecutionPlan,
    ) -> Result<Self> {
        validate(inventory, plan)?;
        let image = &inventory.image_header;
        let working = GpuOutputRequest::color(PixelFormat::rgb_f32(
            RgbChannelOrder::Rgba,
            false,
            crate::vardct_rgb8_format().color_spec,
        ))?
        .with_orientation_policy(OrientationPolicy::Keep)
        .with_max_frame_slots(request.max_frame_slots());
        let compositor = Arc::new(Compositor::new(
            engine.backend().clone(),
            Extent2d::new(image.width, image.height),
            !image.extra_channels.is_empty(),
            image.grayscale,
            OutputOrientation::from_exif_value(image.orientation).ok_or(
                Error::InvalidImageOrientation {
                    value: image.orientation,
                },
            )?,
            request,
        )?);
        let source = SequenceSource {
            engine,
            codestream,
            inventory: inventory.clone(),
            request: working,
        };
        // Initial metadata and output negotiation is performed at open, like the still engines.
        // Subsequent physical producers are prepared one at a time while their pending frame runs.
        let first = next_producer(
            &source,
            plan.presentations[0].physical_frames.start,
            plan.presentations[0].physical_frames.end,
        )?;
        let prepared = Some(source.prepare_physical(first)?.session);
        Ok(Self {
            shared: Arc::new(Mutex::new(Shared {
                carry: Some(Carry {
                    source,
                    references: std::array::from_fn(|_| None),
                    prepared,
                }),
                in_flight: None,
                failed: false,
            })),
            compositor,
            submissions: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub(super) fn submissions(&self) -> usize {
        self.submissions.load(Ordering::Acquire)
    }

    pub(super) fn submit(
        &mut self,
        plan: &FrameExecutionPlan,
        index: usize,
    ) -> Result<CompositionPending> {
        let mut shared = lock(&self.shared);
        if shared.failed {
            return Err(Error::SessionPoisoned);
        }
        if let Some(index) = shared.in_flight {
            return Err(Error::FrameDependencyBackpressure { index });
        }
        let presentation = &plan.presentations[index];
        let carry = shared
            .carry
            .as_mut()
            .ok_or(Error::EngineContract("composition source was lost"))?;
        let physical = next_producer(
            &carry.source,
            presentation.physical_frames.start,
            presentation.physical_frames.end,
        )?;
        if carry.prepared.is_none() {
            carry.prepared = Some(carry.source.prepare_physical(physical)?.session);
        }
        let producer = carry.prepared.as_mut().expect("physical producer prepared");
        let pending = producer
            .submit_next()?
            .ok_or(Error::EngineContract("physical producer returned no frame"))?;
        let count = submission_counter(&pending, producer.submissions_per_frame());
        carry.prepared = None;
        // Nothing leaves the queue until the first physical GPU submission is admitted.
        let carry = shared.carry.take().expect("physical producer admitted");
        shared.in_flight = Some(index);
        self.submissions.store(0, Ordering::Release);
        Ok(CompositionPending {
            shared: Arc::clone(&self.shared),
            carry: Some(carry),
            compositor: Arc::clone(&self.compositor),
            metadata: presentation.metadata.clone(),
            nodes: plan.nodes[presentation.physical_frames.clone()].to_vec(),
            first: presentation.physical_frames.start,
            end: presentation.physical_frames.end,
            physical,
            stage: Some(Stage::Decode(Box::new(pending), count)),
            submissions: Arc::clone(&self.submissions),
            finished: false,
        })
    }
}

fn validate(inventory: &CodestreamInventory, plan: &FrameExecutionPlan) -> Result<()> {
    let image = &inventory.image_header;
    if !matches!(
        image.colour_encoding,
        ColourEncodingInventory::Enumerated {
            colour_space: ColourSpaceInventory::Grey | ColourSpaceInventory::Rgb,
            white_point: WhitePointInventory::D65,
            primaries: PrimariesInventory::Srgb,
            transfer_function: TransferFunctionInventory::Srgb,
            ..
        }
    ) || image.embedded_icc.is_some()
    {
        return Err(UnsupportedProfile::new(
            UnsupportedCodestreamFeature::ColorEncoding,
            "frame composition requires the original enumerated D65 sRGB encoding",
        )
        .into());
    }
    for (node, frame) in plan.nodes.iter().zip(&inventory.frames) {
        if !node.needs_composition {
            continue;
        }
        for blend in std::iter::once(&frame.color_blend).chain(&frame.extra_channel_blends) {
            if blend
                .alpha_channel
                .is_some_and(|channel| channel as usize >= image.extra_channels.len())
            {
                return Err(FramePlanError::InvalidFrame {
                    frame_index: frame.frame_index,
                    reason: "blend alpha channel is outside the image extra channels",
                }
                .into());
            }
            let Some(reference) = node.references[blend.source as usize] else {
                continue;
            };
            if reference.before_color_transform {
                return Err(FramePlanError::InvalidFrame {
                    frame_index: frame.frame_index,
                    reason: "post-transform blending cannot consume a pre-transform reference",
                }
                .into());
            }
            let producer = &inventory.frames[reference.frame_index as usize];
            if !plan.nodes[reference.frame_index as usize].needs_composition
                && (producer.x0 != 0
                    || producer.y0 != 0
                    || producer.width < image.width
                    || producer.height < image.height)
            {
                return Err(FramePlanError::InvalidFrame {
                    frame_index: frame.frame_index,
                    reason: "a cropped reference cannot be used as a canvas background",
                }
                .into());
            }
        }
    }
    Ok(())
}

fn next_producer(source: &SequenceSource, start: usize, end: usize) -> Result<usize> {
    (start..end)
        .find(|&i| source.inventory.frames[i].frame_type != FrameType::LowFrequency)
        .ok_or(Error::EngineContract("presentation has no color producer"))
}

#[derive(Debug)]
enum Stage {
    Decode(Box<WgpuDecodePendingFrame>, Arc<AtomicUsize>),
    Blend(GpuWork),
    Pack(GpuWork),
}

#[derive(Debug)]
pub(super) struct CompositionPending {
    shared: Arc<Mutex<Shared>>,
    carry: Option<Carry>,
    compositor: Arc<Compositor>,
    metadata: FrameMetadata,
    nodes: Vec<crate::FrameExecutionNode>,
    first: usize,
    end: usize,
    physical: usize,
    stage: Option<Stage>,
    submissions: Arc<AtomicUsize>,
    finished: bool,
}

impl CompositionPending {
    pub(super) fn unvalidated(&self) -> Result<UnvalidatedGpuImageFrame> {
        let Some(Stage::Pack(work)) = &self.stage else {
            return Err(Error::UnvalidatedOutputNotSubmitted);
        };
        Ok(UnvalidatedGpuImageFrame {
            token: SubmissionToken(1),
            outputs: vec![UnvalidatedGpuImageOutput {
                id: OutputId(0),
                layout: self.compositor.layout.clone(),
                buffer: work.unvalidated()?,
            }],
        })
    }

    fn decoded(
        &mut self,
        frame: SubmittedGpuFrame<GpuImageFrame>,
        count: &AtomicUsize,
    ) -> Result<()> {
        self.submissions
            .fetch_add(count.load(Ordering::Acquire), Ordering::AcqRel);
        let carry = self
            .carry
            .as_ref()
            .ok_or(Error::EngineContract("composition carry was lost"))?;
        let mut outputs = frame.output.outputs;
        if outputs.len() != 1 {
            return Err(Error::EngineContract(
                "physical producer requires one color output",
            ));
        }
        let surface = Surface::from_output(outputs.remove(0), carry.source.request.format())?;
        let node = &self.nodes[self.physical - self.first];
        if node.needs_composition {
            let header = &carry.source.inventory.frames[self.physical];
            let color = carry.references[header.color_blend.source as usize].as_ref();
            let alpha = header
                .extra_channel_blends
                .first()
                .and_then(|blend| carry.references[blend.source as usize].as_ref());
            self.stage = Some(Stage::Blend(
                self.compositor.blend(&surface, color, alpha, header)?,
            ));
            self.submissions.fetch_add(1, Ordering::AcqRel);
            Ok(())
        } else {
            self.record(surface)
        }
    }

    fn record(&mut self, surface: Surface) -> Result<()> {
        let carry = self
            .carry
            .as_mut()
            .ok_or(Error::EngineContract("composition carry was lost"))?;
        let node = &self.nodes[self.physical - self.first];
        let header = &carry.source.inventory.frames[self.physical];
        if let Some(slot) = node.save_reference {
            // Pre-transform references belong to patches, which this producer does not yet
            // execute. Never reinterpret an RGB surface as XYB/YCbCr patch storage.
            carry.references[slot as usize] =
                (!header.save_before_color_transform).then(|| surface.clone());
        }
        if self.physical + 1 == self.end {
            self.stage = Some(Stage::Pack(self.compositor.pack(&surface)?));
            self.submissions.fetch_add(1, Ordering::AcqRel);
        } else {
            self.physical = next_producer(&carry.source, self.physical + 1, self.end)?;
            let mut prepared = carry.source.prepare_physical(self.physical)?.session;
            let pending = prepared
                .submit_next()?
                .ok_or(Error::EngineContract("physical producer returned no frame"))?;
            let count = submission_counter(&pending, prepared.submissions_per_frame());
            self.stage = Some(Stage::Decode(Box::new(pending), count));
        }
        Ok(())
    }

    fn finish(
        &mut self,
        buffer: jxl_wgpu::GpuBufferLease,
    ) -> Result<SubmittedGpuFrame<GpuImageFrame>> {
        let carry = self
            .carry
            .take()
            .ok_or(Error::EngineContract("composition finished twice"))?;
        let mut shared = lock(&self.shared);
        shared.in_flight = None;
        shared.carry = (!self.metadata.is_last).then_some(carry);
        self.finished = true;
        let extent = self.compositor.layout.extent;
        Ok(SubmittedGpuFrame::new(
            self.metadata.clone(),
            GpuImageFrame {
                token: SubmissionToken(1),
                outputs: vec![GpuImageOutput {
                    id: OutputId(0),
                    layout: self.compositor.layout.clone(),
                    buffer,
                }],
                changed: ChangedRegions {
                    outputs: BTreeMap::from([(
                        OutputId(0),
                        vec![Region::new(0, 0, extent.width, extent.height)],
                    )]),
                },
            },
        ))
    }

    pub(super) fn poll(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<GpuImageFrame>>> {
        loop {
            let stage = self
                .stage
                .as_mut()
                .ok_or(Error::EngineContract("composition stage was lost"))?;
            match stage {
                Stage::Decode(pending, _) => {
                    let frame = match Pin::new(pending.as_mut()).poll_complete(context) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    let Some(Stage::Decode(_, count)) = self.stage.take() else {
                        unreachable!()
                    };
                    self.decoded(frame, &count)?;
                }
                Stage::Blend(work) => {
                    let buffer = match work.poll(context) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    self.stage = None;
                    let extent = &self
                        .carry
                        .as_ref()
                        .expect("live composition source")
                        .source
                        .inventory
                        .image_header;
                    self.record(Surface {
                        buffer,
                        extent: Extent2d::new(extent.width, extent.height),
                    })?;
                }
                Stage::Pack(work) => {
                    let buffer = match work.poll(context) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    self.stage = None;
                    return Poll::Ready(self.finish(buffer));
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn wait(mut self) -> Result<SubmittedGpuFrame<GpuImageFrame>> {
        loop {
            match self
                .stage
                .take()
                .ok_or(Error::EngineContract("composition stage was lost"))?
            {
                Stage::Decode(pending, count) => self.decoded(pending.wait()?, &count)?,
                Stage::Blend(work) => {
                    let image = &self
                        .carry
                        .as_ref()
                        .expect("live composition source")
                        .source
                        .inventory
                        .image_header;
                    let extent = Extent2d::new(image.width, image.height);
                    self.record(Surface {
                        buffer: work.wait()?,
                        extent,
                    })?;
                }
                Stage::Pack(work) => return self.finish(work.wait()?),
            }
        }
    }
}

impl Drop for CompositionPending {
    fn drop(&mut self) {
        if !self.finished {
            let mut shared = lock(&self.shared);
            shared.in_flight = None;
            shared.failed = true;
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
