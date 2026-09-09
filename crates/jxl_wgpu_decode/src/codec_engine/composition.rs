//! Serial physical-frame execution with resident reference versions and lazy producer admission.
//! Refinements read validated references; only complete physical frames commit new versions.

use std::collections::{BTreeMap, VecDeque};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};

use jxl_gpu_bitstream::{
    CodestreamInventory, ColourEncodingInventory, ColourSpaceInventory, FrameType,
    PrimariesInventory, TransferFunctionInventory, WhitePointInventory,
};
use jxl_gpu_formats::{ImageLayout, PixelFormat, RgbChannelOrder};
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
    Error, FrameExecutionPlan, FrameMetadata, FramePlanError, FrameProgression, GpuCodestream,
    GpuOutputRequest, GpuPendingFrame, GpuSubmissionSession, Result, SubmittedGpuFrame,
    SubmittedGpuUpdate, UnsupportedCodestreamFeature, UnsupportedProfile,
};

mod blend;
mod gpu;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod lf_tests;
mod progression;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod refinement_tests;
mod spot;
mod submission;
use gpu::{Compositor, Surface};
use progression::LfPreview;
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
                    || frame.flags & 1 != 0
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
        progressive: bool,
    ) -> Result<WgpuDecodeSubmissionSession> {
        let mut session = self.source.prepare_physical(index, progressive)?.session;
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
    lf_preview: Option<Arc<LfPreview>>,
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
        let lf_preview = if request.progressive_output()
            && request.mapping() == crate::GpuOutputMapping::Color
            && inventory.image_header.xyb_encoded
            && inventory.image_header.extra_channels.is_empty()
            && inventory
                .frames
                .iter()
                .any(|frame| frame.frame_type == FrameType::LowFrequency)
        {
            let working;
            let request = if matches!(output, Output::Composed(_)) {
                working = GpuOutputRequest::color(FrameSurfaceEncoding::Srgb.format())?
                    .with_orientation_policy(crate::OrientationPolicy::Keep);
                &working
            } else {
                request
            };
            Some(Arc::new(LfPreview::new(
                source.engine.backend().clone(),
                &inventory.image_header,
                request,
            )?))
        } else {
            None
        };
        // Initial metadata and output negotiation is performed at open, like the still engines.
        // Subsequent physical producers are prepared one at a time while their pending frame runs.
        let range = &plan.presentations[0].physical_frames;
        let first = range.start;
        let mut carry = Carry {
            source,
            references: std::array::from_fn(|_| None),
            lf: std::array::from_fn(|_| None),
            prepared: None,
        };
        carry.prepared = Some(carry.prepare(first, &plan.nodes[first], first + 1 == range.end)?);
        Ok(Self {
            shared: Arc::new(Mutex::new(Shared {
                carry: Some(carry),
                in_flight: None,
                failed: false,
            })),
            output: Arc::new(output),
            submissions: Arc::new(AtomicUsize::new(0)),
            lf_preview,
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
        let mut lf_updates = Vec::new();
        let last = presentation.physical_frames.end - 1;
        let mut dependency = plan.nodes[last].lf_source_frame;
        while let Some(frame_index) = dependency {
            lf_updates.push(frame_index);
            dependency = plan
                .nodes
                .iter()
                .find(|node| node.frame_index == frame_index)
                .ok_or(Error::EngineContract(
                    "LF presentation dependency is missing",
                ))?
                .lf_source_frame;
        }
        let carry = shared
            .carry
            .as_mut()
            .ok_or(Error::EngineContract("composition source was lost"))?;
        let lf_preview = self
            .lf_preview
            .as_ref()
            .filter(|_| !lf_updates.is_empty())
            .map(|preview| {
                if let Some(encodings) = &carry.source.surface_encodings {
                    let frame = &carry.source.inventory.frames[last];
                    preview
                        .for_surface(Extent2d::new(frame.width, frame.height), encodings[last])
                        .map(Arc::new)
                } else {
                    Ok(Arc::clone(preview))
                }
            })
            .transpose()?;
        let physical = presentation.physical_frames.start;
        if carry.prepared.is_none() {
            carry.prepared =
                Some(carry.prepare(physical, &plan.nodes[physical], physical == last)?);
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
            stage: Some(Stage::Decode(PhysicalPending {
                pending: Box::new(pending),
                count,
                lf: None,
            })),
            submissions: Arc::clone(&self.submissions),
            completed_submissions: 0,
            finished: false,
            lf_preview,
            lf_updates,
            lf_pending: VecDeque::new(),
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
    .with_progressive_output(
        request.progressive_output() && request.mapping() == crate::GpuOutputMapping::Color,
    )
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
            let position =
                inventory
                    .frame_position(reference.frame_index)
                    .ok_or(Error::EngineContract(
                        "reference is outside the selected image",
                    ))?;
            let producer = &inventory.frames[position];
            if !plan.nodes[position].needs_composition
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
struct PhysicalPending {
    pending: Box<WgpuDecodePendingFrame>,
    count: Arc<AtomicUsize>,
    lf: Option<ProgressiveDcXybPlanes>,
}

#[derive(Debug)]
enum RefinementRender {
    Blend(GpuWork),
    Pack(GpuWork),
}

#[derive(Debug)]
enum Resume {
    Decode(PhysicalPending),
    Advance,
}

impl Resume {
    fn stage(self) -> Stage {
        match self {
            Self::Decode(decode) => Stage::Decode(decode),
            Self::Advance => Stage::Advance,
        }
    }
}

#[derive(Debug)]
struct LfUpdate {
    planes: ProgressiveDcXybPlanes,
    progression: FrameProgression,
}

#[derive(Debug)]
enum Stage {
    Decode(PhysicalPending),
    Blend(GpuWork),
    Pack(GpuWork),
    Refinement {
        resume: Resume,
        render: RefinementRender,
        progression: FrameProgression,
    },
    LfPreview {
        work: GpuWork,
        progression: FrameProgression,
    },
    Advance,
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
    lf_preview: Option<Arc<LfPreview>>,
    lf_updates: Vec<u32>,
    lf_pending: VecDeque<LfUpdate>,
}

impl DependentPending {
    pub(super) fn unvalidated(&self) -> Result<UnvalidatedGpuImageFrame> {
        match (&*self.output, &self.stage) {
            (Output::Native, Some(Stage::Decode(PhysicalPending { pending, .. })))
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
        emit_intermediates: bool,
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
            let preview = emit_intermediates
                && self.lf_updates.contains(&node.frame_index)
                && self.lf_preview.is_some()
                && carry.source.inventory.frames[self.end - 1].frame_type == FrameType::Regular;
            let preview_planes = preview.then(|| lf.clone()).flatten();
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
            if let Some(planes) = preview_planes {
                let level = header.lf_level as u8;
                self.lf_pending.push_back(LfUpdate {
                    planes,
                    progression: FrameProgression::LowFrequency {
                        physical_frame_index: node.frame_index,
                        level,
                    },
                });
            }
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
        if !self.lf_pending.is_empty() && self.lf_references_ready()? {
            let update = self.lf_pending.pop_front().expect("nonempty LF updates");
            let FrameProgression::LowFrequency { level, .. } = update.progression else {
                unreachable!()
            };
            self.stage = Some(Stage::LfPreview {
                work: self
                    .lf_preview
                    .as_ref()
                    .expect("LF preview selected")
                    .submit(&update.planes, level)?,
                progression: update.progression,
            });
            self.submissions.fetch_add(1, Ordering::AcqRel);
            self.completed_submissions += 1;
            return Ok(());
        }
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
        let mut prepared = carry.prepare(
            self.physical,
            &self.nodes[self.physical - self.first],
            self.physical + 1 == self.end,
        )?;
        let pending = prepared
            .submit_next()?
            .ok_or(Error::EngineContract("physical producer returned no frame"))?;
        let count = submission_counter(&pending, prepared.submissions_per_frame());
        update_count(self.completed_submissions, &count, &self.submissions)?;
        self.stage = Some(Stage::Decode(PhysicalPending {
            pending: Box::new(pending),
            count,
            lf: None,
        }));
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
        self.finish_output(packed_frame(compositor.layout.clone(), buffer))
    }

    pub(super) fn poll(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<GpuImageFrame>>> {
        self.poll_update(context, false)
            .map(|result| match result? {
                SubmittedGpuUpdate::Complete(frame) => Ok(frame),
                SubmittedGpuUpdate::Intermediate { .. } => Err(Error::EngineContract(
                    "final-only completion returned an intermediate",
                )),
            })
    }

    pub(super) fn poll_update(
        &mut self,
        context: &mut Context<'_>,
        emit_intermediates: bool,
    ) -> Poll<Result<SubmittedGpuUpdate<GpuImageFrame>>> {
        if !emit_intermediates {
            self.lf_pending.clear();
        }
        loop {
            let stage = self
                .stage
                .as_mut()
                .ok_or(Error::EngineContract("dependent sequence stage was lost"))?;
            match stage {
                Stage::Decode(PhysicalPending { pending, lf, count }) => {
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
                    let result = if emit_intermediates && self.physical + 1 == self.end {
                        Pin::new(pending.as_mut()).poll_next_update(context)
                    } else {
                        Pin::new(pending.as_mut())
                            .poll_complete(context)
                            .map(|r| r.map(SubmittedGpuUpdate::Complete))
                    };
                    update_count(self.completed_submissions, count, &self.submissions)?;
                    let update = match result {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    let frame = match update {
                        SubmittedGpuUpdate::Complete(frame) => frame,
                        SubmittedGpuUpdate::Intermediate {
                            mut frame,
                            progression,
                        } => {
                            if matches!(*self.output, Output::Composed(_)) {
                                self.refine(frame.output, progression)?;
                                continue;
                            }
                            frame.metadata = self.metadata.clone();
                            return Poll::Ready(Ok(SubmittedGpuUpdate::Intermediate {
                                frame,
                                progression,
                            }));
                        }
                    };
                    let Some(Stage::Decode(PhysicalPending { count, lf, .. })) = self.stage.take()
                    else {
                        unreachable!()
                    };
                    if let Some(frame) = self.decoded(frame, &count, lf, emit_intermediates)? {
                        return Poll::Ready(Ok(SubmittedGpuUpdate::Complete(frame)));
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
                    return Poll::Ready(self.finish(buffer).map(SubmittedGpuUpdate::Complete));
                }
                Stage::Refinement {
                    render: RefinementRender::Blend(work),
                    ..
                } => {
                    let buffer = match work.poll(context) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    if !emit_intermediates {
                        let Some(Stage::Refinement { resume, .. }) = self.stage.take() else {
                            unreachable!()
                        };
                        self.stage = Some(resume.stage());
                        continue;
                    }
                    let compositor = self.output.compositor()?;
                    let work = compositor.pack(&compositor.completed_surface(buffer))?;
                    let Some(Stage::Refinement { render, .. }) = self.stage.as_mut() else {
                        unreachable!()
                    };
                    *render = RefinementRender::Pack(work);
                    self.submissions.fetch_add(1, Ordering::AcqRel);
                    self.completed_submissions += 1;
                }
                Stage::Refinement {
                    render: RefinementRender::Pack(work),
                    ..
                } => {
                    let buffer = match work.poll(context) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    let Some(Stage::Refinement {
                        resume,
                        progression,
                        ..
                    }) = self.stage.take()
                    else {
                        unreachable!()
                    };
                    self.stage = Some(resume.stage());
                    if emit_intermediates {
                        return Poll::Ready(Ok(self.update(
                            buffer,
                            self.output.compositor()?.layout.clone(),
                            progression,
                        )));
                    }
                }
                Stage::LfPreview { work, progression } => {
                    let buffer = match work.poll(context) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    let progression = *progression;
                    self.stage = Some(Stage::Advance);
                    if emit_intermediates {
                        let layout = self
                            .lf_preview
                            .as_ref()
                            .ok_or(Error::EngineContract("LF preview renderer was lost"))?
                            .layout
                            .clone();
                        if matches!(*self.output, Output::Composed(_)) {
                            let surface =
                                self.output.compositor()?.import(vec![GpuImageOutput {
                                    id: OutputId(0),
                                    layout,
                                    buffer,
                                }])?;
                            self.render_refinement(
                                surface,
                                self.end - 1,
                                Resume::Advance,
                                progression,
                            )?;
                            continue;
                        }
                        return Poll::Ready(Ok(self.update(buffer, layout, progression)));
                    }
                }
                Stage::Advance => self.advance()?,
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn wait(mut self) -> Result<SubmittedGpuFrame<GpuImageFrame>> {
        self.lf_pending.clear();
        loop {
            match self
                .stage
                .take()
                .ok_or(Error::EngineContract("dependent sequence stage was lost"))?
            {
                Stage::Decode(PhysicalPending {
                    mut pending,
                    count,
                    mut lf,
                }) => {
                    if self.nodes[self.physical - self.first].lf_last_use.is_some() && lf.is_none()
                    {
                        if let WgpuDecodePendingFrame::VarDct(pending) = pending.as_mut() {
                            pending.wait_until_dependency_submitted()?;
                        }
                        lf = Some(lf_planes(&pending)?);
                    }
                    if let Some(frame) = self.decoded(pending.wait()?, &count, lf, false)? {
                        return Ok(frame);
                    }
                }
                Stage::Blend(work) => {
                    self.record(self.output.compositor()?.completed_surface(work.wait()?))?;
                }
                Stage::Pack(work) => return self.finish(work.wait()?),
                Stage::Refinement { resume, render, .. } => {
                    // Final-only completion drains the render, then resumes physical execution.
                    // A blend which has not yet been packed needs no additional presentation work.
                    let (RefinementRender::Blend(work) | RefinementRender::Pack(work)) = render;
                    drop(work.wait()?);
                    self.stage = Some(resume.stage());
                }
                Stage::LfPreview { work, .. } => {
                    drop(work.wait()?);
                    self.advance()?;
                }
                Stage::Advance => self.advance()?,
            }
        }
    }

    /// Render against committed references without publishing a new reference version.
    fn refine(&mut self, frame: GpuImageFrame, progression: FrameProgression) -> Result<()> {
        let compositor = self.output.compositor()?;
        let surface = compositor.import(frame.outputs)?;
        let Some(Stage::Decode(decode)) = self.stage.take() else {
            return Err(Error::EngineContract(
                "refinement has no pending physical producer",
            ));
        };
        self.render_refinement(surface, self.physical, Resume::Decode(decode), progression)
    }

    fn render_refinement(
        &mut self,
        surface: Surface,
        index: usize,
        resume: Resume,
        progression: FrameProgression,
    ) -> Result<()> {
        let compositor = self.output.compositor()?;
        let carry = self
            .carry
            .as_ref()
            .ok_or(Error::EngineContract("dependent sequence carry was lost"))?;
        let render = if self.nodes[index - self.first].needs_composition {
            RefinementRender::Blend(compositor.blend(
                &surface,
                &carry.references,
                &carry.source.inventory.frames[index],
            )?)
        } else {
            RefinementRender::Pack(compositor.pack(&surface)?)
        };
        self.stage = Some(Stage::Refinement {
            resume,
            render,
            progression,
        });
        self.submissions.fetch_add(1, Ordering::AcqRel);
        self.completed_submissions += 1;
        Ok(())
    }

    /// A newly decoded LF dependency may precede hidden layers which write the presentation's
    /// background. Keep its planes until that exact reference version has completed validation.
    fn lf_references_ready(&self) -> Result<bool> {
        if matches!(*self.output, Output::Native) {
            return Ok(true);
        }
        let node = self.nodes.last().ok_or(Error::EngineContract(
            "presentation has no physical producer",
        ))?;
        if !node.needs_composition {
            return Ok(true);
        }
        let carry = self
            .carry
            .as_ref()
            .ok_or(Error::EngineContract("dependent sequence carry was lost"))?;
        let source = carry.source.inventory.frames[self.end - 1]
            .color_blend
            .source as usize;
        Ok(node.references[source].is_none_or(|reference| {
            reference.frame_index <= self.nodes[self.physical - self.first].frame_index
        }))
    }

    fn update(
        &self,
        buffer: jxl_wgpu::GpuBufferLease,
        layout: ImageLayout,
        progression: FrameProgression,
    ) -> SubmittedGpuUpdate<GpuImageFrame> {
        SubmittedGpuUpdate::Intermediate {
            progression,
            frame: SubmittedGpuFrame::new(self.metadata.clone(), packed_frame(layout, buffer)),
        }
    }
}

fn packed_frame(layout: ImageLayout, buffer: jxl_wgpu::GpuBufferLease) -> GpuImageFrame {
    let extent = layout.extent;
    GpuImageFrame {
        token: SubmissionToken(1),
        outputs: vec![GpuImageOutput {
            id: OutputId(0),
            layout,
            buffer,
        }],
        changed: ChangedRegions {
            outputs: BTreeMap::from([(
                OutputId(0),
                vec![Region::new(0, 0, extent.width, extent.height)],
            )]),
        },
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
