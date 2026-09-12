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
mod patches;
mod progression;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod refinement_tests;
mod spot;
mod submission;
mod transform;
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
    // Include reconstruction and the inverse codec transform before selecting an original color
    // component. The resident frame surface keeps those samples unquantized until presentation.
    let numeric_vardct_color = matches!(request.mapping(), crate::GpuOutputMapping::Numeric(_))
        && request.extra_channel().is_none()
        && inventory
            .frames
            .iter()
            .any(|frame| frame.encoding == jxl_gpu_bitstream::FrameEncoding::VarDct);
    modular_rendering
        || inventory.frames.iter().any(|frame| frame.flags & 2 != 0)
        || numeric_vardct_color
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
    fn compositor(&self) -> Result<&Arc<Compositor>> {
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
    prepared: Option<PreparedPhysical>,
}

#[derive(Debug)]
enum PreparedPhysical {
    Codec(WgpuDecodeSubmissionSession),
    Patches(Box<patches::Plan>),
}

impl PreparedPhysical {
    fn submit(&mut self, source: &SequenceSource) -> Result<(Stage, usize)> {
        match self {
            Self::Codec(producer) => {
                let pending = producer
                    .submit_next()?
                    .ok_or(Error::EngineContract("physical producer returned no frame"))?;
                let count = submission_counter(&pending, producer.submissions_per_frame());
                let submissions = count.load(Ordering::Acquire);
                Ok((
                    Stage::Decode(PhysicalPending {
                        pending: Box::new(pending),
                        count,
                        lf: None,
                    }),
                    submissions,
                ))
            }
            Self::Patches(plan) => {
                let pending = (**plan).clone().submit(
                    source.engine.backend().clone(),
                    Arc::clone(&source.codestream),
                )?;
                Ok((Stage::PatchDictionary(Box::new(pending)), 1))
            }
        }
    }
}

impl Carry {
    fn render_patches(
        &self,
        surface: &Surface,
        dictionary: &patches::Dictionary,
    ) -> Result<GpuWork> {
        let image = &self.source.inventory.image_header;
        patches::render(
            self.source.engine.backend(),
            surface,
            &self.references,
            dictionary,
            image.extra_channel_count,
            image.extra_channels.iter().any(|extra| {
                matches!(
                    extra.channel_type,
                    jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha { .. }
                )
            }),
        )
    }

    fn prepare(
        &self,
        index: usize,
        node: &crate::FrameExecutionNode,
        progressive: bool,
    ) -> Result<PreparedPhysical> {
        let frame = &self.source.inventory.frames[index];
        if frame.flags & 2 != 0 {
            let limit = self
                .source
                .engine
                .vardct_engine()
                .stream_window_limit()
                .map_or(
                    self.source
                        .engine
                        .backend()
                        .device()
                        .limits()
                        .max_storage_buffer_binding_size,
                    std::num::NonZeroU64::get,
                );
            let references = self.references.each_ref().map(|slot| {
                slot.as_ref().map_or([0; 4], |surface| {
                    [
                        surface.extent.width,
                        surface.extent.height,
                        1,
                        u32::from(surface.encoding == FrameSurfaceEncoding::Encoded),
                    ]
                })
            });
            return Ok(PreparedPhysical::Patches(Box::new(patches::Plan::new(
                &self.source.codestream,
                frame,
                &self.source.inventory.image_header.extra_channels,
                references,
                limit,
            )?)));
        }
        self.prepare_codec(index, node, progressive, None)
            .map(PreparedPhysical::Codec)
    }

    fn prepare_codec(
        &self,
        index: usize,
        node: &crate::FrameExecutionNode,
        progressive: bool,
        patch_end: Option<u64>,
    ) -> Result<WgpuDecodeSubmissionSession> {
        let mut session = self
            .source
            .prepare_physical_after_features(index, progressive, patch_end)?
            .session;
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
            && inventory.image_header.xyb_encoded
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
        let (stage, count) = producer.submit(&carry.source)?;
        carry.prepared = None;
        // Nothing leaves the queue until the first physical GPU submission is admitted.
        let carry = shared.carry.take().expect("physical producer admitted");
        shared.in_flight = Some(index);
        self.submissions.store(count, Ordering::Release);
        Ok(DependentPending {
            shared: Arc::clone(&self.shared),
            carry: Some(carry),
            output: Arc::clone(&self.output),
            metadata: presentation.metadata.clone(),
            nodes: plan.nodes[presentation.physical_frames.clone()].to_vec(),
            first: presentation.physical_frames.start,
            end: presentation.physical_frames.end,
            physical,
            stage: Some(stage),
            patches: None,
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
    .with_progressive_output(request.progressive_output())
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
                    if frame.flags & 2 != 0
                        || (node.save_reference.is_some() && frame.save_before_color_transform)
                    {
                        FrameSurfaceEncoding::Encoded
                    } else {
                        presentation_encoding(image, node, frame)
                    }
                })
                .collect(),
        ),
    };
    Ok((source, compositor))
}

fn presentation_encoding(
    image: &jxl_gpu_bitstream::ImageHeaderInventory,
    node: &crate::FrameExecutionNode,
    frame: &jxl_gpu_bitstream::FrameInventory,
) -> FrameSurfaceEncoding {
    if image.xyb_encoded
        && !frame.do_ycbcr
        && !node.needs_composition
        && (node.save_reference.is_none() || frame.save_before_color_transform)
    {
        FrameSurfaceEncoding::Linear
    } else {
        FrameSurfaceEncoding::Srgb
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
        if frame.flags & 2 != 0 && (frame.upsampling != 1 || frame.flags & 1 != 0) {
            return Err(UnsupportedProfile::new(
                UnsupportedCodestreamFeature::Patches,
                "patch rendering with frame upsampling or noise is not yet connected",
            )
            .into());
        }
        if (frame.flags & 2 != 0
            || (node.save_reference.is_some() && frame.save_before_color_transform))
            && frame.do_ycbcr
            && frame.jpeg_upsampling != [0; 3]
        {
            return Err(UnsupportedProfile::new(
                UnsupportedCodestreamFeature::Patches,
                "subsampled YCbCr patch components are not yet connected",
            )
            .into());
        }
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
    lf: Option<crate::progressive_dc::ProgressiveDcOutput>,
}

#[derive(Debug)]
enum RefinementRender {
    Patches {
        work: GpuWork,
        source: Surface,
        index: usize,
    },
    Transform {
        work: GpuWork,
        source: Surface,
        index: usize,
    },
    Blend(GpuWork),
    Pack(GpuWork),
}

#[derive(Debug)]
enum Resume {
    Decode(PhysicalPending),
    LfPreviews(PhysicalPending),
    Advance,
}

impl Resume {
    fn stage(self) -> Stage {
        match self {
            Self::Decode(decode) => Stage::Decode(decode),
            Self::LfPreviews(decode) => Stage::LfPreviews(decode),
            Self::Advance => Stage::Advance,
        }
    }
}

#[derive(Debug)]
struct LfUpdate {
    planes: crate::progressive_dc::ProgressiveDcOutput,
    progression: FrameProgression,
}

#[derive(Debug)]
enum Stage {
    PatchDictionary(Box<patches::Pending>),
    LfPatchRender(GpuWork<crate::progressive_dc::ProgressiveDcOutput>),
    PatchRender {
        work: GpuWork,
        source: Surface,
    },
    ColorTransform {
        work: GpuWork,
        source: Surface,
    },
    Decode(PhysicalPending),
    Blend(GpuWork),
    Pack(GpuWork),
    Refinement {
        compositor: Arc<Compositor>,
        resume: Resume,
        render: RefinementRender,
        progression: FrameProgression,
    },
    LfPreview {
        work: GpuWork,
        progression: FrameProgression,
        resume: Resume,
    },
    LfPreviews(PhysicalPending),
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
    patches: Option<patches::Dictionary>,
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
        lf: Option<crate::progressive_dc::ProgressiveDcOutput>,
        emit_intermediates: bool,
    ) -> Result<Option<SubmittedGpuFrame<GpuImageFrame>>> {
        self.completed_submissions =
            update_count(self.completed_submissions, count, &self.submissions)?;
        let preview = self.presents_lf(emit_intermediates);
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
            drop(frame);
            if header.flags & 2 != 0 {
                let dictionary = self.patches.take().ok_or(Error::EngineContract(
                    "LF producer patch dictionary was lost",
                ))?;
                if dictionary.count != 0 {
                    let work = patches::render_lf(
                        carry.source.engine.backend(),
                        lf.as_ref()
                            .ok_or(Error::EngineContract("LF patch planes were lost"))?,
                        &carry.references,
                        &dictionary,
                        &carry.source.inventory.image_header.extra_channels,
                        preview,
                    )?;
                    self.stage = Some(Stage::LfPatchRender(work));
                    self.count_submission();
                    return Ok(None);
                }
            }
            self.record_lf(lf, emit_intermediates)?;
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
                let domain = carry
                    .source
                    .surface_encodings
                    .as_ref()
                    .map(|domains| domains[self.physical]);
                let surface = compositor.import_with_encoding(frame.output.outputs, domain)?;
                if let Some(dictionary) = self
                    .patches
                    .take()
                    .filter(|dictionary| dictionary.count != 0)
                {
                    let work = carry.render_patches(&surface, &dictionary)?;
                    self.stage = Some(Stage::PatchRender {
                        work,
                        source: surface,
                    });
                    self.count_submission();
                } else {
                    self.reconstructed(surface)?;
                }
            }
        }
        Ok(None)
    }

    fn presents_lf(&self, emit_intermediates: bool) -> bool {
        emit_intermediates
            && self
                .lf_updates
                .contains(&self.nodes[self.physical - self.first].frame_index)
            && self.lf_preview.is_some()
            && self.carry.as_ref().is_some_and(|carry| {
                carry.source.inventory.frames[self.end - 1].frame_type == FrameType::Regular
            })
    }

    fn needs_lf_output(&self) -> bool {
        self.nodes[self.physical - self.first].lf_last_use.is_some()
            || self.carry.as_ref().is_some_and(|carry| {
                let frame = &carry.source.inventory.frames[self.physical];
                frame.frame_type == FrameType::LowFrequency && frame.flags & 2 != 0
            })
    }

    /// Publish only the validated, feature-complete LF version. Extra planes belong to queued
    /// presentation, independently of prediction slots and ordinary patch reference slots.
    fn record_lf(
        &mut self,
        lf: Option<crate::progressive_dc::ProgressiveDcOutput>,
        emit_intermediates: bool,
    ) -> Result<()> {
        let preview = self.presents_lf(emit_intermediates);
        let carry = self
            .carry
            .as_mut()
            .ok_or(Error::EngineContract("LF carry was lost"))?;
        let node = &self.nodes[self.physical - self.first];
        let header = &carry.source.inventory.frames[self.physical];
        let preview_planes = preview.then(|| lf.clone()).flatten();
        carry.lf[header.lf_level as usize - 1] = match node.lf_last_use {
            Some(last_use) => Some(LfFrame {
                frame_index: node.frame_index,
                last_use,
                planes: lf
                    .ok_or(Error::EngineContract(
                        "validated LF producer lost its planes",
                    ))?
                    .xyb,
            }),
            None => None,
        };
        if let Some(planes) = preview_planes {
            self.lf_pending.push_back(LfUpdate {
                planes,
                progression: FrameProgression::LowFrequency {
                    physical_frame_index: node.frame_index,
                    level: header.lf_level as u8,
                },
            });
        }
        self.advance()
    }

    fn advance(&mut self) -> Result<()> {
        if !self.lf_pending.is_empty() && self.lf_references_ready()? {
            return self.start_lf_preview(Resume::Advance);
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
        let (stage, count) = prepared.submit(&carry.source)?;
        self.submissions
            .store(self.completed_submissions + count, Ordering::Release);
        self.stage = Some(stage);
        Ok(())
    }

    fn start_lf_preview(&mut self, resume: Resume) -> Result<()> {
        let update = self
            .lf_pending
            .pop_front()
            .ok_or(Error::EngineContract("LF preview queue is empty"))?;
        let FrameProgression::LowFrequency { level, .. } = update.progression else {
            unreachable!()
        };
        self.stage = Some(Stage::LfPreview {
            work: self
                .lf_preview
                .as_ref()
                .ok_or(Error::EngineContract("LF preview renderer was lost"))?
                .submit_with_extras(&update.planes.xyb, update.planes.extras.as_ref(), level)?,
            progression: update.progression,
            resume,
        });
        self.count_submission();
        Ok(())
    }

    fn count_submission(&mut self) {
        self.submissions.fetch_add(1, Ordering::AcqRel);
        self.completed_submissions += 1;
    }

    fn dictionary_decoded(
        &mut self,
        dictionary: patches::Dictionary,
        submissions: usize,
    ) -> Result<()> {
        self.completed_submissions += submissions;
        let carry = self
            .carry
            .as_ref()
            .ok_or(Error::EngineContract("patch carry was lost"))?;
        let mut producer = PreparedPhysical::Codec(carry.prepare_codec(
            self.physical,
            &self.nodes[self.physical - self.first],
            self.physical + 1 == self.end,
            Some(dictionary.end),
        )?);
        let (stage, count) = producer.submit(&carry.source)?;
        self.submissions
            .store(self.completed_submissions + count, Ordering::Release);
        self.patches = Some(dictionary);
        self.stage = Some(
            if self.physical + 1 == self.end && !self.lf_pending.is_empty() {
                let Stage::Decode(decode) = stage else {
                    return Err(Error::EngineContract(
                        "patch body did not submit a codec producer",
                    ));
                };
                // Preserve the admitted body while queued dependencies are presented. Returning an
                // update does not eagerly admit the next preview or consume the retained dictionary.
                Stage::LfPreviews(decode)
            } else {
                stage
            },
        );
        Ok(())
    }

    fn reconstructed(&mut self, surface: Surface) -> Result<()> {
        if surface.encoding != FrameSurfaceEncoding::Encoded {
            return self.transformed(surface);
        }
        let carry = self
            .carry
            .as_mut()
            .ok_or(Error::EngineContract("patch carry was lost"))?;
        let node = &self.nodes[self.physical - self.first];
        let frame = &carry.source.inventory.frames[self.physical];
        if frame.save_before_color_transform
            && let Some(slot) = node.save_reference
        {
            carry.references[slot as usize] = Some(surface.clone());
        }
        if self.physical + 1 != self.end && frame.save_before_color_transform {
            return self.advance();
        }
        let encoding = presentation_encoding(&carry.source.inventory.image_header, node, frame);
        let work = transform::convert(
            carry.source.engine.backend(),
            &surface,
            &carry.source.inventory.image_header,
            frame,
            encoding,
        )?;
        self.stage = Some(Stage::ColorTransform {
            work,
            source: Surface {
                encoding,
                ..surface
            },
        });
        self.count_submission();
        Ok(())
    }

    fn transformed(&mut self, surface: Surface) -> Result<()> {
        let carry = self
            .carry
            .as_ref()
            .ok_or(Error::EngineContract("patch carry was lost"))?;
        if self.nodes[self.physical - self.first].needs_composition {
            self.stage = Some(Stage::Blend(self.output.compositor()?.blend(
                &surface,
                &carry.references,
                &carry.source.inventory.frames[self.physical],
            )?));
            self.count_submission();
            Ok(())
        } else {
            self.record(surface)
        }
    }

    fn record(&mut self, surface: Surface) -> Result<()> {
        let carry = self
            .carry
            .as_mut()
            .ok_or(Error::EngineContract("dependent sequence carry was lost"))?;
        let node = &self.nodes[self.physical - self.first];
        let header = &carry.source.inventory.frames[self.physical];
        if !header.save_before_color_transform
            && let Some(slot) = node.save_reference
        {
            carry.references[slot as usize] = Some(surface.clone());
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
            let needs_lf_output = self.needs_lf_output();
            let stage = self
                .stage
                .as_mut()
                .ok_or(Error::EngineContract("dependent sequence stage was lost"))?;
            match stage {
                Stage::PatchDictionary(pending) => {
                    let result = pending.poll(context);
                    let count = pending.submissions;
                    self.submissions
                        .store(self.completed_submissions + count, Ordering::Release);
                    let dictionary = match result {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    // The body needs only validated commands. Release parser tables/window/history
                    // before its admission, matching the blocking path's resource lifetime.
                    self.stage = None;
                    self.dictionary_decoded(dictionary, count)?;
                }
                Stage::LfPatchRender(work) => {
                    let planes = match work.poll(context) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    self.stage = None;
                    self.record_lf(Some(planes), emit_intermediates)?;
                }
                Stage::PatchRender { work, source } => {
                    let buffer = match work.poll(context) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    let mut surface = source.clone();
                    surface.buffer = buffer;
                    self.stage = None;
                    self.reconstructed(surface)?;
                }
                Stage::ColorTransform { work, source } => {
                    let buffer = match work.poll(context) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    let mut surface = source.clone();
                    surface.buffer = buffer;
                    self.stage = None;
                    self.transformed(surface)?;
                }
                Stage::Decode(PhysicalPending { pending, lf, count }) => {
                    if needs_lf_output && lf.is_none() {
                        if let WgpuDecodePendingFrame::VarDct(pending) = pending.as_mut() {
                            let result = pending.poll_until_dependency_submitted(context);
                            update_count(self.completed_submissions, count, &self.submissions)?;
                            match result {
                                Poll::Pending => return Poll::Pending,
                                Poll::Ready(result) => result?,
                            }
                        }
                        *lf = Some(lf_output(pending)?);
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
                    render:
                        RefinementRender::Patches {
                            work,
                            source,
                            index,
                        }
                        | RefinementRender::Transform {
                            work,
                            source,
                            index,
                        },
                    ..
                } => {
                    let buffer = match work.poll(context) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    let surface = Surface {
                        buffer,
                        ..source.clone()
                    };
                    let index = *index;
                    let Some(Stage::Refinement {
                        compositor,
                        resume,
                        progression,
                        ..
                    }) = self.stage.take()
                    else {
                        unreachable!()
                    };
                    if emit_intermediates {
                        self.render_refinement(compositor, surface, index, resume, progression)?;
                    } else {
                        self.stage = Some(resume.stage());
                    }
                }
                Stage::Refinement {
                    render: RefinementRender::Blend(work),
                    compositor,
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
                        compositor,
                        ..
                    }) = self.stage.take()
                    else {
                        unreachable!()
                    };
                    self.stage = Some(resume.stage());
                    if emit_intermediates {
                        return Poll::Ready(Ok(self.update(
                            buffer,
                            compositor.layout.clone(),
                            progression,
                        )));
                    }
                }
                Stage::LfPreviews(_) => {
                    let Some(Stage::LfPreviews(decode)) = self.stage.take() else {
                        unreachable!()
                    };
                    if self.lf_pending.is_empty() {
                        self.stage = Some(Stage::Decode(decode));
                    } else {
                        if !self.lf_references_ready()? {
                            return Poll::Ready(Err(Error::EngineContract(
                                "patch LF preview references are not ready",
                            )));
                        }
                        self.start_lf_preview(Resume::LfPreviews(decode))?;
                    }
                }
                Stage::LfPreview { work, .. } => {
                    let buffer = match work.poll(context) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(result) => result?,
                    };
                    let Some(Stage::LfPreview {
                        progression,
                        resume,
                        ..
                    }) = self.stage.take()
                    else {
                        unreachable!()
                    };
                    if emit_intermediates {
                        let preview = self
                            .lf_preview
                            .as_ref()
                            .ok_or(Error::EngineContract("LF preview renderer was lost"))?;
                        if let Some(surface_layout) = &preview.surface {
                            let compositor = match &*self.output {
                                Output::Composed(compositor) => Arc::clone(compositor),
                                Output::Native => {
                                    Arc::clone(preview.compositor.as_ref().ok_or(
                                        Error::EngineContract("LF output packer was lost"),
                                    )?)
                                }
                            };
                            let surface = compositor.import_with_encoding(
                                crate::frame_surface::outputs(
                                    &preview.layout,
                                    Some(surface_layout),
                                    &buffer,
                                ),
                                preview.surface_encoding,
                            )?;
                            self.begin_refinement(
                                compositor,
                                surface,
                                self.end - 1,
                                resume,
                                progression,
                            )?;
                        } else {
                            self.stage = Some(resume.stage());
                            return Poll::Ready(Ok(self.update(
                                buffer,
                                preview.layout.clone(),
                                progression,
                            )));
                        }
                    } else {
                        self.stage = Some(resume.stage());
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
            let needs_lf_output = self.needs_lf_output();
            match self
                .stage
                .take()
                .ok_or(Error::EngineContract("dependent sequence stage was lost"))?
            {
                Stage::PatchDictionary(pending) => {
                    let (dictionary, count) = pending.wait()?;
                    self.dictionary_decoded(dictionary, count)?;
                }
                Stage::LfPatchRender(work) => self.record_lf(Some(work.wait()?), false)?,
                Stage::PatchRender { work, mut source } => {
                    source.buffer = work.wait()?;
                    self.reconstructed(source)?;
                }
                Stage::ColorTransform { work, mut source } => {
                    source.buffer = work.wait()?;
                    self.transformed(source)?;
                }
                Stage::Decode(PhysicalPending {
                    mut pending,
                    count,
                    mut lf,
                }) => {
                    if needs_lf_output && lf.is_none() {
                        if let WgpuDecodePendingFrame::VarDct(pending) = pending.as_mut() {
                            pending.wait_until_dependency_submitted()?;
                        }
                        lf = Some(lf_output(&pending)?);
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
                    let (RefinementRender::Patches { work, .. }
                    | RefinementRender::Transform { work, .. }
                    | RefinementRender::Blend(work)
                    | RefinementRender::Pack(work)) = render;
                    drop(work.wait()?);
                    self.stage = Some(resume.stage());
                }
                Stage::LfPreview { work, resume, .. } => {
                    drop(work.wait()?);
                    self.stage = Some(resume.stage());
                }
                Stage::LfPreviews(decode) => self.stage = Some(Stage::Decode(decode)),
                Stage::Advance => self.advance()?,
            }
        }
    }

    /// Render against committed references without publishing a new reference version.
    fn refine(&mut self, frame: GpuImageFrame, progression: FrameProgression) -> Result<()> {
        let compositor = self.output.compositor()?;
        let domain = self
            .carry
            .as_ref()
            .and_then(|carry| carry.source.surface_encodings.as_ref())
            .map(|encodings| encodings[self.physical]);
        let surface = compositor.import_with_encoding(frame.outputs, domain)?;
        let Some(Stage::Decode(decode)) = self.stage.take() else {
            return Err(Error::EngineContract(
                "refinement has no pending physical producer",
            ));
        };
        self.begin_refinement(
            Arc::clone(self.output.compositor()?),
            surface,
            self.physical,
            Resume::Decode(decode),
            progression,
        )
    }

    fn begin_refinement(
        &mut self,
        compositor: Arc<Compositor>,
        surface: Surface,
        index: usize,
        resume: Resume,
        progression: FrameProgression,
    ) -> Result<()> {
        let carry = self
            .carry
            .as_ref()
            .ok_or(Error::EngineContract("patch refinement carry was lost"))?;
        let dictionary = if carry.source.inventory.frames[index].flags & 2 != 0 {
            Some(self.patches.as_ref().ok_or(Error::EngineContract(
                "patch refinement dictionary is not ready",
            ))?)
        } else {
            None
        };
        if let Some(dictionary) = dictionary.filter(|dictionary| dictionary.count != 0) {
            let work = carry.render_patches(&surface, dictionary)?;
            // Each refinement owns a fresh patched surface. Keep the dictionary and committed
            // reference versions available for subsequent passes and the final reconstruction.
            self.stage = Some(Stage::Refinement {
                compositor,
                resume,
                render: RefinementRender::Patches {
                    work,
                    source: surface,
                    index,
                },
                progression,
            });
            self.count_submission();
            return Ok(());
        }
        self.render_refinement(compositor, surface, index, resume, progression)
    }

    fn render_refinement(
        &mut self,
        compositor: Arc<Compositor>,
        surface: Surface,
        index: usize,
        resume: Resume,
        progression: FrameProgression,
    ) -> Result<()> {
        let carry = self
            .carry
            .as_ref()
            .ok_or(Error::EngineContract("dependent sequence carry was lost"))?;
        let node = &self.nodes[index - self.first];
        let frame = &carry.source.inventory.frames[index];
        let render = if surface.encoding == FrameSurfaceEncoding::Encoded {
            let encoding = presentation_encoding(&carry.source.inventory.image_header, node, frame);
            RefinementRender::Transform {
                work: transform::convert(
                    carry.source.engine.backend(),
                    &surface,
                    &carry.source.inventory.image_header,
                    frame,
                    encoding,
                )?,
                source: Surface {
                    encoding,
                    ..surface
                },
                index,
            }
        } else if node.needs_composition {
            RefinementRender::Blend(compositor.blend(
                &surface,
                &carry.references,
                &carry.source.inventory.frames[index],
            )?)
        } else {
            RefinementRender::Pack(compositor.pack(&surface)?)
        };
        self.stage = Some(Stage::Refinement {
            compositor,
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
        let carry = self
            .carry
            .as_ref()
            .ok_or(Error::EngineContract("dependent sequence carry was lost"))?;
        if carry.source.inventory.frames[self.end - 1].flags & 2 != 0
            && (self.physical + 1 != self.end || self.patches.is_none())
        {
            return Ok(false);
        }
        if matches!(*self.output, Output::Native) {
            return Ok(true);
        }
        let node = self.nodes.last().ok_or(Error::EngineContract(
            "presentation has no physical producer",
        ))?;
        if !node.needs_composition {
            return Ok(true);
        }
        let frame = &carry.source.inventory.frames[self.end - 1];
        // Extra channels and their selected alpha may read a different, later hidden producer
        // from the color background. Readiness must cover every reference bound by the blend.
        Ok(std::iter::once(&frame.color_blend)
            .chain(&frame.extra_channel_blends)
            .all(|blend| {
                node.references[blend.source as usize].is_none_or(|reference| {
                    reference.frame_index <= self.nodes[self.physical - self.first].frame_index
                })
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

fn lf_output(
    pending: &WgpuDecodePendingFrame,
) -> Result<crate::progressive_dc::ProgressiveDcOutput> {
    match pending {
        WgpuDecodePendingFrame::Modular(pending) => pending.progressive_dc_output(),
        WgpuDecodePendingFrame::VarDct(pending) => Ok(pending.progressive_dc_output()?),
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
