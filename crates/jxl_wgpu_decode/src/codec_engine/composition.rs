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
use crate::frame_surface::FrameSurfaceEncoding;
use crate::progressive_dc::ProgressiveDcXybPlanes;
use crate::{
    Error, FrameExecutionPlan, FrameMetadata, FramePlanError, GpuCodestream, GpuOutputRequest,
    GpuPendingFrame, GpuSubmissionSession, Result, SubmittedGpuFrame, UnsupportedCodestreamFeature,
    UnsupportedProfile,
};

mod blend;
mod gpu;
mod spot;
mod submission;
use gpu::{Compositor, Surface};
use submission::GpuWork;

/// Keep producer selection and sequence dispatch consistent about presentation-only conversion.
pub(super) fn needs_surface(
    inventory: &CodestreamInventory,
    request: &GpuOutputRequest,
    plan: &FrameExecutionPlan,
) -> bool {
    use jxl_gpu_formats::{
        ColorFormatClass, ColorRange, ColorSpace, ColorSpecification, PixelFormatClass, RgbSample,
        TransferFunction, classify_pixel_format,
    };
    let direct_float = matches!(
        classify_pixel_format(request.format()),
        Ok(PixelFormatClass::Color(ColorFormatClass::Rgb {
            sample: RgbSample::F32,
            ..
        }))
    ) && matches!(request.format().color_spec, ColorSpecification::Defined(spec)
            if spec.space == ColorSpace::Bt709 && spec.range == ColorRange::Full
                && matches!(spec.transfer, TransferFunction::Srgb | TransferFunction::Sycc
                    | TransferFunction::Linear | TransferFunction::Bt709 | TransferFunction::Bt2020));
    let image = &inventory.image_header;
    let native = crate::model::native_modular_format(request.format());
    let direct_integer = native.is_some_and(|format| {
        image.bit_depth
            == (jxl_gpu_bitstream::SampleBitDepth::Integer {
                bits_per_sample: u32::from(format.bits_per_sample),
            })
    });
    let source_conversion = match image.bit_depth {
        jxl_gpu_bitstream::SampleBitDepth::Float { .. } => !direct_float,
        jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample } => {
            bits_per_sample > 16 && !direct_float && !direct_integer
        }
    };
    let wide_vardct_output = native.is_some_and(|format| format.bits_per_sample > 16)
        && inventory
            .frames
            .iter()
            .any(|frame| frame.encoding == jxl_gpu_bitstream::FrameEncoding::VarDct);
    let modular_rendering = request.extra_channel().is_none()
        && inventory.frames.iter().any(|frame| {
            frame.encoding == jxl_gpu_bitstream::FrameEncoding::Modular
                && frame.lf_level == 0
                && (image.xyb_encoded
                    || frame.restoration_filter
                        != jxl_gpu_bitstream::RestorationFilterInventory::Custom {
                            gaborish: jxl_gpu_bitstream::GaborishInventory::Disabled,
                            epf: jxl_gpu_bitstream::EdgePreservingFilterInventory::Disabled,
                        })
        });
    modular_rendering
        || ((source_conversion || wide_vardct_output)
            && request.mapping() == crate::GpuOutputMapping::Color)
        || request.renders_spot_colors(&inventory.image_header.extra_channels)
        || plan.nodes.iter().any(|node| node.needs_composition)
        || inventory
            .frames
            .iter()
            .any(|frame| frame.frame_type == FrameType::ReferenceOnly)
}

#[derive(Debug)]
enum Output {
    Native,
    Composed(Arc<Compositor>),
}

impl Output {
    fn compositor(&self) -> Result<&Compositor> {
        match self {
            Self::Composed(compositor) => Ok(compositor),
            Self::Native => Err(Error::EngineContract(
                "native sequence requested composition",
            )),
        }
    }
}

#[derive(Debug)]
struct LfFrame {
    frame_index: u32,
    last_use: u32,
    planes: ProgressiveDcXybPlanes,
}

#[derive(Debug)]
struct Carry {
    source: SequenceSource,
    references: [Option<Surface>; 4],
    lf: [Option<LfFrame>; 4],
    prepared: Option<WgpuDecodeSubmissionSession>,
}

impl Carry {
    fn prepare(
        &self,
        index: usize,
        node: &crate::FrameExecutionNode,
    ) -> Result<WgpuDecodeSubmissionSession> {
        let mut session = self.source.prepare_physical(index)?.session;
        if let Some(source) = node.lf_source_frame {
            let level = self.source.inventory.frames[index].lf_level;
            let planes = self
                .lf
                .get(level as usize)
                .and_then(Option::as_ref)
                .filter(|lf| lf.frame_index == source)
                .ok_or(Error::EngineContract(
                    "LF slot version does not match the execution plan",
                ))?;
            let WgpuDecodeSubmissionSession::VarDct(producer) = &mut session else {
                return Err(Error::EngineContract("LF consumer is not VarDCT"));
            };
            producer.set_progressive_dc_source(planes.planes.clone())?;
        }
        Ok(session)
    }
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
pub(super) struct DependentSession {
    shared: Arc<Mutex<Shared>>,
    output: Arc<Output>,
    submissions: Arc<AtomicUsize>,
}

impl DependentSession {
    pub(super) fn new(
        engine: WgpuDecodeEngine,
        codestream: Arc<GpuCodestream>,
        inventory: &CodestreamInventory,
        request: &GpuOutputRequest,
        plan: &FrameExecutionPlan,
    ) -> Result<Self> {
        let (source, output) = if needs_surface(inventory, request, plan) {
            let (source, compositor) =
                composed_source(engine, codestream, inventory, request, plan)?;
            (source, Output::Composed(compositor))
        } else {
            (
                SequenceSource {
                    engine,
                    codestream,
                    inventory: inventory.clone(),
                    request: request.clone(),
                    surface_encodings: None,
                },
                Output::Native,
            )
        };
        // Initial metadata and output negotiation is performed at open, like the still engines.
        // Subsequent physical producers are prepared one at a time while their pending frame runs.
        let first = plan.presentations[0].physical_frames.start;
        let mut carry = Carry {
            source,
            references: std::array::from_fn(|_| None),
            lf: std::array::from_fn(|_| None),
            prepared: None,
        };
        carry.prepared = Some(carry.prepare(first, &plan.nodes[first])?);
        Ok(Self {
            shared: Arc::new(Mutex::new(Shared {
                carry: Some(carry),
                in_flight: None,
                failed: false,
            })),
            output: Arc::new(output),
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
    ) -> Result<DependentPending> {
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
        let physical = presentation.physical_frames.start;
        if carry.prepared.is_none() {
            carry.prepared = Some(carry.prepare(physical, &plan.nodes[physical])?);
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
        self.submissions
            .store(count.load(Ordering::Acquire), Ordering::Release);
        Ok(DependentPending {
            shared: Arc::clone(&self.shared),
            carry: Some(carry),
            output: Arc::clone(&self.output),
            metadata: presentation.metadata.clone(),
            nodes: plan.nodes[presentation.physical_frames.clone()].to_vec(),
            first: presentation.physical_frames.start,
            end: presentation.physical_frames.end,
            physical,
            stage: Some(Stage::Decode {
                pending: Box::new(pending),
                count,
                lf: None,
            }),
            submissions: Arc::clone(&self.submissions),
            completed_submissions: 0,
            finished: false,
        })
    }
}

fn composed_source(
    engine: WgpuDecodeEngine,
    codestream: Arc<GpuCodestream>,
    inventory: &CodestreamInventory,
    request: &GpuOutputRequest,
    plan: &FrameExecutionPlan,
) -> Result<(SequenceSource, Arc<Compositor>)> {
    validate(inventory, plan)?;
    let image = &inventory.image_header;
    if inventory
        .frames
        .iter()
        .any(|frame| frame.encoding == jxl_gpu_bitstream::FrameEncoding::VarDct)
        && matches!(request.format().color_spec, jxl_gpu_formats::ColorSpecification::Defined(color)
                if matches!(color.transfer, jxl_gpu_formats::TransferFunction::Pq | jxl_gpu_formats::TransferFunction::Hlg))
    {
        return Err(crate::VarDctDecodeError::Output(
            crate::color_output::ColorOutputError::HdrLuminanceMappingRequired,
        )
        .into());
    }
    let working = GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgb,
        true,
        crate::vardct_rgb8_format().color_spec,
    ))?
    .for_frame_surface(FrameSurfaceEncoding::Srgb)
    .with_max_frame_slots(request.max_frame_slots());
    let compositor = Arc::new(Compositor::new(
        engine.backend().clone(),
        Extent2d::new(image.width, image.height),
        &image.extra_channels,
        image.grayscale,
        image.bit_depth,
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
        surface_encodings: Some(
            plan.nodes
                .iter()
                .zip(&inventory.frames)
                .map(|(node, frame)| {
                    if image.xyb_encoded
                        && !frame.do_ycbcr
                        && !node.needs_composition
                        && (node.save_reference.is_none() || frame.save_before_color_transform)
                    {
                        FrameSurfaceEncoding::Linear
                    } else {
                        FrameSurfaceEncoding::Srgb
                    }
                })
                .collect(),
        ),
    };
    Ok((source, compositor))
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

#[derive(Debug)]
enum Stage {
    Decode {
        pending: Box<WgpuDecodePendingFrame>,
        count: Arc<AtomicUsize>,
        lf: Option<ProgressiveDcXybPlanes>,
    },
    Blend(GpuWork),
    Pack(GpuWork),
}

#[derive(Debug)]
pub(super) struct DependentPending {
    shared: Arc<Mutex<Shared>>,
    carry: Option<Carry>,
    output: Arc<Output>,
    metadata: FrameMetadata,
    nodes: Vec<crate::FrameExecutionNode>,
    first: usize,
    end: usize,
    physical: usize,
    stage: Option<Stage>,
    submissions: Arc<AtomicUsize>,
    completed_submissions: usize,
    finished: bool,
}

impl DependentPending {
    pub(super) fn unvalidated(&self) -> Result<UnvalidatedGpuImageFrame> {
        match (&*self.output, &self.stage) {
            (Output::Native, Some(Stage::Decode { pending, .. }))
                if self.physical + 1 == self.end =>
            {
                pending.unvalidated_gpu_frame()
            }
            (Output::Composed(compositor), Some(Stage::Pack(work))) => {
                Ok(UnvalidatedGpuImageFrame {
                    token: SubmissionToken(1),
                    outputs: vec![UnvalidatedGpuImageOutput {
                        id: OutputId(0),
                        layout: compositor.layout.clone(),
                        buffer: work.unvalidated()?,
                    }],
                })
            }
            _ => Err(Error::UnvalidatedOutputNotSubmitted),
        }
    }

    fn decoded(
        &mut self,
        frame: SubmittedGpuFrame<GpuImageFrame>,
        count: &AtomicUsize,
        lf: Option<ProgressiveDcXybPlanes>,
    ) -> Result<Option<SubmittedGpuFrame<GpuImageFrame>>> {
        self.completed_submissions =
            update_count(self.completed_submissions, count, &self.submissions)?;
        let carry = self
            .carry
            .as_mut()
            .ok_or(Error::EngineContract("dependent sequence carry was lost"))?;
        let node = &self.nodes[self.physical - self.first];
        let header = &carry.source.inventory.frames[self.physical];
        // Validation has released this consumer's scratch and its LF input ownership. Clear
        // expired slot versions before admitting the next producer, including across presentations.
        for slot in &mut carry.lf {
            if slot
                .as_ref()
                .is_some_and(|lf| lf.last_use <= node.frame_index)
            {
                *slot = None;
            }
        }
        if header.frame_type == FrameType::LowFrequency {
            carry.lf[header.lf_level as usize - 1] = match node.lf_last_use {
                Some(last_use) => Some(LfFrame {
                    frame_index: node.frame_index,
                    last_use,
                    planes: lf.ok_or(Error::EngineContract(
                        "validated LF producer lost its planes",
                    ))?,
                }),
                None => None,
            };
            drop(frame);
            self.advance()?;
            return Ok(None);
        }
        match &*self.output {
            Output::Native => {
                if self.physical + 1 == self.end {
                    return self.finish_output(frame.output).map(Some);
                }
                drop(frame);
                self.advance()?;
            }
            Output::Composed(compositor) => {
                let surface = compositor.import(frame.output.outputs)?;
                if node.needs_composition {
                    self.stage = Some(Stage::Blend(compositor.blend(
                        &surface,
                        &carry.references,
                        header,
                    )?));
                    self.submissions.fetch_add(1, Ordering::AcqRel);
                    self.completed_submissions += 1;
                } else {
                    self.record(surface)?;
                }
            }
        }
        Ok(None)
    }

    fn advance(&mut self) -> Result<()> {
        self.physical += 1;
        if self.physical >= self.end {
            return Err(Error::EngineContract(
                "presentation ended without a color producer",
            ));
        }
        let carry = self
            .carry
            .as_ref()
            .ok_or(Error::EngineContract("dependent sequence carry was lost"))?;
        let mut prepared = carry.prepare(self.physical, &self.nodes[self.physical - self.first])?;
        let pending = prepared
            .submit_next()?
            .ok_or(Error::EngineContract("physical producer returned no frame"))?;
        let count = submission_counter(&pending, prepared.submissions_per_frame());
        update_count(self.completed_submissions, &count, &self.submissions)?;
        self.stage = Some(Stage::Decode {
            pending: Box::new(pending),
            count,
            lf: None,
        });
        Ok(())
    }

    fn record(&mut self, surface: Surface) -> Result<()> {
        let carry = self
            .carry
            .as_mut()
            .ok_or(Error::EngineContract("dependent sequence carry was lost"))?;
        let node = &self.nodes[self.physical - self.first];
        let header = &carry.source.inventory.frames[self.physical];
        if let Some(slot) = node.save_reference {
            // Pre-transform references belong to patches. Never reinterpret RGB as XYB/YCbCr.
            carry.references[slot as usize] =
                (!header.save_before_color_transform).then(|| surface.clone());
        }
        if self.physical + 1 == self.end {
            self.stage = Some(Stage::Pack(self.output.compositor()?.pack(&surface)?));
            self.submissions.fetch_add(1, Ordering::AcqRel);
            self.completed_submissions += 1;
        } else {
            self.advance()?;
        }
        Ok(())
    }

    fn finish_output(&mut self, output: GpuImageFrame) -> Result<SubmittedGpuFrame<GpuImageFrame>> {
        let carry = self
            .carry
            .take()
            .ok_or(Error::EngineContract("dependent sequence finished twice"))?;
        let mut shared = lock(&self.shared);
        shared.in_flight = None;
        shared.carry = (!self.metadata.is_last).then_some(carry);
        self.finished = true;
        Ok(SubmittedGpuFrame::new(self.metadata.clone(), output))
    }

    fn finish(
        &mut self,
        buffer: jxl_wgpu::GpuBufferLease,
    ) -> Result<SubmittedGpuFrame<GpuImageFrame>> {
        let compositor = self.output.compositor()?;
        let extent = compositor.layout.extent;
        self.finish_output(GpuImageFrame {
            token: SubmissionToken(1),
            outputs: vec![GpuImageOutput {
                id: OutputId(0),
                layout: compositor.layout.clone(),
                buffer,
            }],
            changed: ChangedRegions {
                outputs: BTreeMap::from([(
                    OutputId(0),
                    vec![Region::new(0, 0, extent.width, extent.height)],
                )]),
            },
        })
    }

    pub(super) fn poll(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<GpuImageFrame>>> {
        loop {
            let stage = self
                .stage
                .as_mut()
                .ok_or(Error::EngineContract("dependent sequence stage was lost"))?;
            match stage {
                Stage::Decode { pending, lf, count } => {
                    if self.nodes[self.physical - self.first].lf_last_use.is_some() && lf.is_none()
                    {
                        if let WgpuDecodePendingFrame::VarDct(pending) = pending.as_mut() {
                            let result = pending.poll_until_dependency_submitted(context);
                            update_count(self.completed_submissions, count, &self.submissions)?;
                            match result {
                                Poll::Pending => return Poll::Pending,
                                Poll::Ready(result) => result?,
                            }
                        }
                        *lf = Some(lf_planes(pending)?);
                    }
                    let result = Pin::new(pending.as_mut()).poll_complete(context);
                    update_count(self.completed_submissions, count, &self.submissions)?;
                    let frame = match result {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    let Some(Stage::Decode { count, lf, .. }) = self.stage.take() else {
                        unreachable!()
                    };
                    if let Some(frame) = self.decoded(frame, &count, lf)? {
                        return Poll::Ready(Ok(frame));
                    }
                }
                Stage::Blend(work) => {
                    let buffer = match work.poll(context) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    self.stage = None;
                    self.record(self.output.compositor()?.completed_surface(buffer))?;
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
                .ok_or(Error::EngineContract("dependent sequence stage was lost"))?
            {
                Stage::Decode {
                    mut pending,
                    count,
                    mut lf,
                } => {
                    if self.nodes[self.physical - self.first].lf_last_use.is_some() && lf.is_none()
                    {
                        if let WgpuDecodePendingFrame::VarDct(pending) = pending.as_mut() {
                            pending.wait_until_dependency_submitted()?;
                        }
                        lf = Some(lf_planes(&pending)?);
                    }
                    if let Some(frame) = self.decoded(pending.wait()?, &count, lf)? {
                        return Ok(frame);
                    }
                }
                Stage::Blend(work) => {
                    self.record(self.output.compositor()?.completed_surface(work.wait()?))?;
                }
                Stage::Pack(work) => return self.finish(work.wait()?),
            }
        }
    }
}

fn lf_planes(pending: &WgpuDecodePendingFrame) -> Result<ProgressiveDcXybPlanes> {
    match pending {
        WgpuDecodePendingFrame::Modular(pending) => pending.progressive_dc_planes(),
        WgpuDecodePendingFrame::VarDct(pending) => Ok(pending.progressive_dc_planes()?),
        WgpuDecodePendingFrame::Sequence(_) => {
            Err(Error::EngineContract("LF producer is a sequence"))
        }
    }
}

fn update_count(completed: usize, active: &AtomicUsize, total: &AtomicUsize) -> Result<usize> {
    let count = completed
        .checked_add(active.load(Ordering::Acquire))
        .ok_or(Error::EngineContract(
            "sequence submission count overflowed",
        ))?;
    total.store(count, Ordering::Release);
    Ok(count)
}

impl Drop for DependentPending {
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
