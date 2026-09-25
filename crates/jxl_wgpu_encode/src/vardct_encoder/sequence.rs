//! Checked stream metadata and ordered assembly for Gray/RGB frame sequences.

use super::{VarDctBackend, VarDctConfig, VarDctJob};
use crate::{
    BufferImageSource, CodestreamAssembler, Determinism, EncodeError, EncodeProfile, EncodeSession,
    FrameIndex, FrameOptions, FrameSubmission, GpuEncoder, GpuFrameArtifacts, GpuFrameSource,
    SessionDescriptor,
};

/// Compatibility name for [`VarDctSequenceDescriptor`].
pub type VarDctAnimationDescriptor = VarDctSequenceDescriptor;

/// Compatibility name for [`VarDctSequenceSession`].
pub type VarDctAnimationSession = VarDctSequenceSession;

/// Compatibility name for the common image sequence descriptor.
pub type VarDctSequenceDescriptor = crate::ImageSequenceDescriptor;

/// Independent GPU frame submissions and deterministic sequence assembly.
///
/// Supports Replace, Add and Multiply, signed crops, hidden zero-duration frames,
/// timecodes and four post-color-transform reference slots. Pre-color-transform storage is
/// rejected until the profile supports its consumers. Alpha-weighted modes require an alpha source and
/// are rejected by this color-only profile. Frame controls are checked before GPU admission;
/// failure leaves the frame index and final-frame state available for retry.
pub struct VarDctSequenceSession {
    descriptor: VarDctSequenceDescriptor,
    session: EncodeSession<VarDctBackend>,
    assembler: CodestreamAssembler,
    metadata_permit: Option<jxl_wgpu::MemoryPermit>,
}

impl VarDctSequenceSession {
    pub(super) fn new(
        encoder: &GpuEncoder<VarDctBackend>,
        config: &VarDctConfig,
        descriptor: VarDctSequenceDescriptor,
    ) -> Result<Self, EncodeError> {
        let session = encoder.begin_session(SessionDescriptor {
            profile: EncodeProfile::VarDct {
                quantization: config.quantization,
            },
            progressive: config.progressive.clone(),
            minimum_determinism: Determinism::SameDevice,
            animation: descriptor.animation(),
            canvas_width: descriptor.canvas_width(),
            canvas_height: descriptor.canvas_height(),
        })?;
        let (header, metadata_permit) = encoder
            .backend()
            .sequence_header(&descriptor)?
            .finish(encoder.memory_budget())?;
        let assembler = CodestreamAssembler::new(header)?;
        Ok(Self {
            descriptor,
            session,
            assembler,
            metadata_permit,
        })
    }

    #[must_use]
    pub const fn descriptor(&self) -> &VarDctSequenceDescriptor {
        &self.descriptor
    }

    #[must_use]
    pub const fn next_frame_index(&self) -> FrameIndex {
        self.session.next_frame_index()
    }

    pub fn submit_frame(
        &mut self,
        source: BufferImageSource,
        options: FrameOptions,
    ) -> Result<FrameSubmission<VarDctJob>, EncodeError> {
        self.session
            .submit_frame(GpuFrameSource::Buffer(source), options)
    }

    pub fn submit_last_frame(
        &mut self,
        source: BufferImageSource,
        options: FrameOptions,
    ) -> Result<FrameSubmission<VarDctJob>, EncodeError> {
        self.session
            .submit_last_frame(GpuFrameSource::Buffer(source), options)
    }

    /// Inserts one validated GPU artifact, in any completion order.
    pub fn insert(&mut self, frame: GpuFrameArtifacts) -> Result<(), EncodeError> {
        self.assembler.insert(frame)?;
        Ok(())
    }

    pub fn finish_raw(self) -> Result<Vec<u8>, EncodeError> {
        let _metadata_permit = self.metadata_permit;
        self.session.ensure_closed()?;
        Ok(self.assembler.finish_raw()?)
    }

    pub fn finish_container(self) -> Result<Vec<u8>, EncodeError> {
        let _metadata_permit = self.metadata_permit;
        self.session.ensure_closed()?;
        self.assembler.finish_container()
    }

    /// Finishes with a header-validated `jxli` index. See
    /// [`CodestreamAssembler::finish_indexed_container`] for limits and restart semantics.
    pub fn finish_indexed_container(
        self,
        inventory_limits: jxl_gpu_bitstream::InventoryLimits,
        index_limits: jxl_gpu_bitstream::FrameIndexLimits,
    ) -> Result<Vec<u8>, EncodeError> {
        let _metadata_permit = self.metadata_permit;
        self.session.ensure_closed()?;
        self.assembler
            .finish_indexed_container(inventory_limits, index_limits)
    }
}
