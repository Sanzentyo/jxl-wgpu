//! Preview metadata, session identity and validated completion share one ownership boundary.
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use jxl_gpu_bitstream::BitWriter;
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{MemoryBudget, MemoryPermit};

use crate::{
    EncodeError, EncodedFrame, FrameBlend, FrameEncodeRequest, FrameIndex, FrameKind,
    FrameSubmission, GpuEncodeJob, GpuFrameArtifacts, PacketError, ReferenceSlot,
};

#[cfg(test)]
mod tests;

/// Encoded preview dimensions, before image orientation; each axis is in 1..=4096.
/// The caller supplies the corresponding GPU samples. No automatic downsampling is implied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreviewSize(Extent2d);

impl PreviewSize {
    pub fn new(width: u32, height: u32) -> Result<Self, EncodeError> {
        if !(1..=4096).contains(&width) || !(1..=4096).contains(&height) {
            return Err(EncodeError::InvalidConfiguration(
                "preview axes must be in 1..=4096",
            ));
        }
        Ok(Self(Extent2d::new(width, height)))
    }

    #[must_use]
    pub const fn extent(self) -> Extent2d {
        self.0
    }

    pub(crate) fn write(self, writer: &mut BitWriter) -> Result<(), EncodeError> {
        // Explicit dimensions cover the complete legal range without an inferred aspect ratio.
        writer.write_bits(0, 1)?;
        Self::write_axis(writer, self.0.height)?;
        writer.write_bits(0, 3)?;
        Self::write_axis(writer, self.0.width)
    }

    fn write_axis(writer: &mut BitWriter, value: u32) -> Result<(), EncodeError> {
        let (selector, offset, bits) = match value {
            1..=64 => (0, 1, 6),
            65..=320 => (1, 65, 8),
            321..=1344 => (2, 321, 10),
            _ => (3, 1345, 12),
        };
        writer.write_bits(selector, 2)?;
        writer.write_bits(u64::from(value - offset), bits)?;
        Ok(())
    }
}

/// GPU-validated encoded preview, bound to its originating sequence. It cannot be fabricated
/// from unvalidated packets. Retained encoded storage stays charged until insertion/assembly/drop.
#[derive(Debug)]
pub struct EncodedPreview {
    identity: Arc<()>,
    frame: EncodedFrame,
    permit: MemoryPermit,
}

impl EncodedPreview {
    #[must_use]
    pub fn encoded_bytes(&self) -> usize {
        self.frame.bytes().len()
    }

    #[must_use]
    pub fn reserved_bytes(&self) -> u64 {
        self.permit.bytes()
    }
}

/// Runtime-neutral preview completion. GPU failures never yield an `EncodedPreview`.
/// Completion also admits the actual packet/assembly storage; pressure there is a typed error.
pub struct PreviewSubmission<J> {
    frame: Option<FrameSubmission<J>>,
    identity: Arc<()>,
    budget: MemoryBudget,
}

impl<J: GpuEncodeJob> PreviewSubmission<J> {
    pub fn wait(mut self) -> Result<EncodedPreview, EncodeError> {
        let artifacts = self
            .frame
            .take()
            .expect("preview completion is single-use")
            .wait()?;
        self.finish(artifacts)
    }

    fn finish(&self, artifacts: GpuFrameArtifacts) -> Result<EncodedPreview, EncodeError> {
        let plan = crate::packet::PreparedFrame::new(artifacts.packets)?;
        let mut permit = self.budget.try_reserve(plan.peak_bytes()?)?;
        let frame = plan.finish()?;
        let retained = frame.storage_bytes() as u64;
        let release =
            permit
                .bytes()
                .checked_sub(retained)
                .ok_or(EncodeError::InvalidConfiguration(
                    "preview assembly exceeded its planned storage",
                ))?;
        drop(permit.split_off(release).map_err(|_| {
            EncodeError::InvalidConfiguration(
                "preview storage reservation lost exclusive ownership",
            )
        })?);
        Ok(EncodedPreview {
            identity: self.identity.clone(),
            frame,
            permit,
        })
    }
}

impl<J: GpuEncodeJob> Future for PreviewSubmission<J> {
    type Output = Result<EncodedPreview, EncodeError>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match Pin::new(
            this.frame
                .as_mut()
                .expect("preview completion is single-use"),
        )
        .poll(cx)
        {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                this.frame.take();
                Poll::Ready(result.and_then(|artifacts| this.finish(artifacts)))
            }
        }
    }
}

pub(crate) struct PreviewState {
    size: PreviewSize,
    identity: Arc<()>,
    submitted: bool,
    output: Option<EncodedPreview>,
}

impl PreviewState {
    pub(crate) fn new(size: PreviewSize) -> Self {
        Self {
            size,
            identity: Arc::new(()),
            submitted: false,
            output: None,
        }
    }

    pub(crate) fn request(
        &self,
        mut request: FrameEncodeRequest,
    ) -> Result<FrameEncodeRequest, EncodeError> {
        if self.submitted {
            return Err(EncodeError::InvalidConfiguration(
                "preview was already submitted",
            ));
        }
        if request.frame_index != FrameIndex::new(0) {
            return Err(EncodeError::InvalidConfiguration(
                "submit the preview before main frames",
            ));
        }
        let options = &request.options;
        if options.kind != FrameKind::Regular
            || options.crop.is_some()
            || options.color_blend != FrameBlend::default()
            || options
                .extra_channel_blends
                .iter()
                .any(|blend| *blend != FrameBlend::default())
            || options.save_as_reference != ReferenceSlot::default()
            || options.save_before_color_transform
        {
            return Err(EncodeError::InvalidConfiguration(
                "previews require a full regular Replace frame without references",
            ));
        }
        request.canvas_width = self.size.0.width;
        request.canvas_height = self.size.0.height;
        request.is_last = true;
        Ok(request)
    }

    pub(crate) fn submitted<J>(
        &mut self,
        frame: FrameSubmission<J>,
        budget: &MemoryBudget,
    ) -> PreviewSubmission<J> {
        self.submitted = true;
        PreviewSubmission {
            frame: Some(frame),
            identity: self.identity.clone(),
            budget: budget.clone(),
        }
    }

    pub(crate) fn insert(&mut self, preview: EncodedPreview) -> Result<(), PacketError> {
        if !self.submitted
            || self.output.is_some()
            || !Arc::ptr_eq(&self.identity, &preview.identity)
        {
            return Err(PacketError::UnexpectedPreview);
        }
        self.output = Some(preview);
        Ok(())
    }

    pub(crate) fn completed(&self) -> Result<&[u8], PacketError> {
        self.output
            .as_ref()
            .map(|output| output.frame.bytes())
            .ok_or(PacketError::MissingPreview)
    }
}
