use super::super::{
    JpegCoefficientPending,
    jpeg::{GpuJpegCoefficients, JpegCoefficientSession},
};
use super::{
    Error, GpuJpegFrame, JpegReconstructionLimits, add, allocate, check, framing, gpu, scan,
};
use crate::{
    FrameMetadata, GpuPendingFrame, GpuSubmissionSession, SubmittedGpuFrame, VarDctSubmissionEngine,
};
use jxl_gpu_bitstream::metadata::{MetadataLimits, MetadataSelection};
use jxl_wgpu::GpuBufferLease;
use std::{
    fmt,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

#[derive(Debug)]
pub(super) struct Plan {
    pub scans: scan::Plan,
    pub framing: framing::Plan,
    pub padding: Vec<u8>,
    pub padding_bits: u32,
    pub has_padding: bool,
    pub limits: JpegReconstructionLimits,
}

/// A bounded original-JPEG reconstruction session. No CPU coefficient or entropy fallback exists.
#[derive(Debug)]
pub struct JpegReconstructionSession {
    inner: JpegCoefficientSession,
    resources: gpu::Resources,
    plan: Option<Arc<Plan>>,
}
impl VarDctSubmissionEngine {
    /// Opens one original JPEG byte output from a complete transport-validated JXL container.
    ///
    /// Metadata is admitted on the host; coefficient restoration, entropy coding, quantizer
    /// validation and byte assembly execute on the GPU. Every intermediate size and the final
    /// output are status-validated before being used as allocation or output authority.
    pub fn open_jpeg_reconstruction(
        &self,
        bytes: &[u8],
        limits: JpegReconstructionLimits,
    ) -> crate::Result<JpegReconstructionSession> {
        let parsed = jxl_gpu_bitstream::parse(bytes, limits.coefficients.parse)?;
        let (inner, inventory) = self.open_jpeg_coefficients_parsed(
            &parsed,
            limits.coefficients,
            Some(limits.max_plan_bytes),
        )?;
        let frame = &inventory.frames[0];
        if inner.layout().extent() != [inventory.image_header.width, inventory.image_header.height]
            || frame.x0 != 0
            || frame.y0 != 0
            || frame.upsampling != 1
            || frame.color_blend.mode != jxl_gpu_bitstream::FrameBlendMode::Replace
        {
            return Err(Error::Unsupported("JPEG canvas binding").into());
        }
        let header = inner.metadata();
        let mut owned = header.logical_owned_bytes();
        let icc = inventory
            .image_header
            .embedded_icc
            .as_ref()
            .map_or(&[][..], |icc| icc.profile.as_ref());
        owned = add(owned, icc.len() as u64)?;
        check(
            "host reconstruction plan bytes",
            owned,
            limits.max_plan_bytes,
        )?;
        let mut plan_limits = limits;
        plan_limits.max_plan_bytes -= owned;
        let scans = scan::build(header, inner.layout(), plan_limits)?;
        owned = add(owned, scans.owned_bytes)?;
        check(
            "host reconstruction plan bytes",
            owned,
            limits.max_plan_bytes,
        )?;
        let metadata_limits = MetadataLimits {
            max_retained_bytes: limits
                .metadata
                .max_retained_bytes
                .min(limits.max_plan_bytes - owned),
            ..limits.metadata
        };
        let metadata = parsed.metadata(
            &MetadataSelection::Types(vec![*b"Exif", *b"xml "]),
            metadata_limits,
        )?;
        for kind in [*b"Exif", *b"xml "] {
            if metadata.boxes_of_type(kind).count() > 1 {
                return Err(Error::Invalid("duplicate JPEG metadata box").into());
            }
        }
        owned = add(owned, metadata.retained_bytes())?;
        let decoded = metadata.decode_all(MetadataLimits {
            max_total_decoded_bytes: limits
                .metadata
                .max_total_decoded_bytes
                .min(limits.max_plan_bytes - owned),
            ..metadata_limits
        })?;
        // Counting all decoded logical bytes is conservative even when an uncompressed box borrows.
        for item in &decoded {
            owned = add(owned, item.payload.len() as u64)?;
        }
        let exif = decoded
            .iter()
            .find(|item| item.box_type == *b"Exif")
            .map_or(&[][..], |item| item.payload.as_ref());
        let xmp = decoded
            .iter()
            .find(|item| item.box_type == *b"xml ")
            .map_or(&[][..], |item| item.payload.as_ref());
        let padding_bits = header.padding_bit_count();
        check(
            "padding bits",
            u64::from(padding_bits),
            u64::from(u32::MAX - 31),
        )?;
        owned = add(owned, header.padding_bytes().len() as u64)?;
        check(
            "host reconstruction plan bytes",
            owned,
            limits.max_plan_bytes,
        )?;
        let mut padding = allocate(header.padding_bytes().len())?;
        padding.extend_from_slice(header.padding_bytes());
        let framing = framing::build(
            header,
            inner.layout(),
            &framing::Extras { icc, exif, xmp },
            limits.max_output_bytes,
            limits.max_plan_bytes - owned,
        )?;
        check(
            "host reconstruction plan bytes",
            add(owned, framing.owned_bytes)?,
            limits.max_plan_bytes,
        )?;
        let plan = Arc::new(Plan {
            scans,
            framing,
            padding,
            padding_bits,
            has_padding: header.has_preserved_padding(),
            limits,
        });
        let resources = gpu::Resources {
            backend: self.backend.clone(),
            memory: self.memory.clone(),
            dispatch_width: self
                .backend
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
            pipelines: Arc::clone(
                self.pipelines
                    .jpeg_output
                    .get_or_init(|| Arc::new(gpu::Pipelines::new(self.backend.device()))),
            ),
        };
        Ok(JpegReconstructionSession {
            inner,
            resources,
            plan: Some(plan),
        })
    }
}
impl GpuSubmissionSession for JpegReconstructionSession {
    type Frame = GpuJpegFrame;
    type Pending = JpegReconstructionPending;
    fn submit_next(&mut self) -> crate::Result<Option<Self::Pending>> {
        let Some(coefficients) = self.inner.submit_next()? else {
            return Ok(None);
        };
        let plan = self
            .plan
            .take()
            .ok_or(Error::Invalid("JPEG session exhausted"))?;
        Ok(Some(JpegReconstructionPending {
            resources: self.resources.clone(),
            plan,
            metadata: None,
            state: Some(State::Coefficients(Box::new(coefficients))),
        }))
    }
}

struct ScanState {
    coefficients: GpuJpegCoefficients,
    padding: GpuBufferLease,
    index: usize,
    outputs: Vec<(GpuBufferLease, u32)>,
    scan: gpu::Scan,
}
enum State {
    Coefficients(Box<JpegCoefficientPending>),
    Scan(Box<ScanState>),
    Assembly(gpu::Assembly),
}
/// Pending GPU reconstruction. Dropping it prevents later stages and retains submitted resources
/// until the current map callback observes GPU completion. It exposes no unvalidated byte output.
pub struct JpegReconstructionPending {
    resources: gpu::Resources,
    plan: Arc<Plan>,
    metadata: Option<FrameMetadata>,
    state: Option<State>,
}
impl fmt::Debug for JpegReconstructionPending {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JpegReconstructionPending")
            .finish_non_exhaustive()
    }
}
impl JpegReconstructionPending {
    fn coefficients_ready(
        &mut self,
        frame: SubmittedGpuFrame<GpuJpegCoefficients>,
    ) -> crate::Result<()> {
        self.metadata = Some(frame.metadata);
        let padding = self.resources.padding()?;
        let outputs = allocate(self.plan.scans.scans.len())?;
        let scan = gpu::Scan::start(&self.resources, &frame.output, &padding, &self.plan, 0)?;
        self.state = Some(State::Scan(Box::new(ScanState {
            coefficients: frame.output,
            padding,
            index: 0,
            outputs,
            scan,
        })));
        Ok(())
    }
    fn operation(&self) -> Option<&gpu::Operation> {
        match self.state.as_ref()? {
            State::Coefficients(_) => None,
            State::Scan(state) => Some(&state.scan.operation),
            State::Assembly(state) => Some(&state.operation),
        }
    }
    fn advance(
        &mut self,
        mapping: std::result::Result<(), String>,
    ) -> crate::Result<Option<SubmittedGpuFrame<GpuJpegFrame>>> {
        match self
            .state
            .take()
            .ok_or(Error::Invalid("completed pending JPEG"))?
        {
            State::Scan(state) => {
                let ScanState {
                    coefficients,
                    padding,
                    mut index,
                    mut outputs,
                    scan,
                } = *state;
                let status = scan.operation.status(mapping, "scan entropy")?;
                if scan.phase == gpu::ScanPhase::Emit {
                    let retained = outputs
                        .iter()
                        .try_fold(self.plan.framing.bytes.len() as u64, |sum, (_, bytes)| {
                            add(sum, u64::from(*bytes))
                        })?;
                    check(
                        "JPEG output bytes",
                        add(retained, u64::from(status[3]))?,
                        self.plan.limits.max_output_bytes,
                    )?;
                }
                if scan.phase != gpu::ScanPhase::Pack {
                    let scan =
                        scan.advance(&self.resources, &coefficients, &padding, &self.plan, status)?;
                    self.state = Some(State::Scan(Box::new(ScanState {
                        coefficients,
                        padding,
                        index,
                        outputs,
                        scan,
                    })));
                } else {
                    if status[3] != scan.byte_len
                        || status[1] == 0
                        || status[1] & 7 != 0
                        || status[2] != status[1] / 8
                    {
                        return Err(Error::GpuStatus {
                            stage: "packed scan",
                            status,
                        }
                        .into());
                    }
                    outputs.push((scan.output.clone(), scan.byte_len));
                    drop(scan);
                    index += 1;
                    if index == self.plan.scans.scans.len() {
                        self.state = Some(State::Assembly(gpu::Assembly::start(
                            &self.resources,
                            &coefficients,
                            &self.plan,
                            &outputs,
                        )?));
                    } else {
                        let scan = gpu::Scan::start(
                            &self.resources,
                            &coefficients,
                            &padding,
                            &self.plan,
                            index,
                        )?;
                        self.state = Some(State::Scan(Box::new(ScanState {
                            coefficients,
                            padding,
                            index,
                            outputs,
                            scan,
                        })));
                    }
                }
                Ok(None)
            }
            State::Assembly(assembly) => {
                let status = assembly.operation.status(mapping, "assembly")?;
                let output = assembly.finish(status)?;
                let metadata = self
                    .metadata
                    .take()
                    .ok_or(Error::Invalid("frame metadata"))?;
                Ok(Some(SubmittedGpuFrame::new(metadata, output)))
            }
            State::Coefficients(_) => Err(Error::Invalid("coefficient stage event").into()),
        }
    }
}
impl GpuPendingFrame for JpegReconstructionPending {
    type Frame = GpuJpegFrame;
    #[cfg(not(target_arch = "wasm32"))]
    fn wait(mut self) -> crate::Result<SubmittedGpuFrame<Self::Frame>> {
        loop {
            if matches!(self.state, Some(State::Coefficients(_))) {
                let Some(State::Coefficients(pending)) = self.state.take() else {
                    unreachable!()
                };
                self.coefficients_ready(pending.wait()?)?;
            } else {
                let mapping = self
                    .operation()
                    .ok_or(Error::Invalid("completed pending JPEG"))?
                    .wait();
                if let Some(frame) = self.advance(mapping)? {
                    return Ok(frame);
                }
            }
        }
    }
    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<crate::Result<SubmittedGpuFrame<Self::Frame>>> {
        self.resources
            .backend
            .device()
            .poll(wgpu::PollType::Poll)
            .map_err(crate::Error::backend)?;
        if let Some(State::Coefficients(pending)) = &mut self.state {
            let frame = match Pin::new(pending.as_mut()).poll_complete(context) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(result) => {
                    self.state = None;
                    result?
                }
            };
            self.coefficients_ready(frame)?;
        } else {
            let Some(mapping) = self
                .operation()
                .ok_or(Error::Invalid("completed pending JPEG"))?
                .poll(context)
            else {
                return Poll::Pending;
            };
            if let Some(frame) = self.advance(mapping)? {
                return Poll::Ready(Ok(frame));
            }
        }
        context.waker().wake_by_ref();
        Poll::Pending
    }
}

#[cfg(test)]
mod tests;
