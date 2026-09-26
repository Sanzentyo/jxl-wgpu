//! Explicit image-domain lowering, preserving physical frame IDs and entropy ranges.

use jxl_gpu_bitstream::{CodestreamInventory, FrameInventory, ImageHeaderInventory};
use std::sync::Arc;

pub use jxl_gpu_bitstream::{ImageSelection, ImageSelectionError};

/// Original metadata and the extent of input validation at engine handoff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageSourceInventory {
    /// Complete codestream inventory following authoritative transport end validation.
    Complete(Arc<CodestreamInventory>),
    /// Image metadata and one completely received preview, before whole-file validation.
    /// No claim is made about the main image, subsequent bytes or container completion.
    PreviewPrefix {
        image_header: Arc<ImageHeaderInventory>,
        frame: Arc<FrameInventory>,
        /// Logical prefix length, ending exactly after the preview's last section.
        codestream_bytes: u64,
    },
}

impl ImageSourceInventory {
    #[must_use]
    pub fn complete_inventory(&self) -> Option<&CodestreamInventory> {
        match self {
            Self::Complete(inventory) => Some(inventory),
            Self::PreviewPrefix { .. } => None,
        }
    }
}

/// A validated image selection supplied to a GPU submission engine.
///
/// The source inventory distinguishes whole-file metadata from a complete preview prefix and
/// retains the original metadata of that source. The reconstruction inventory contains
/// only the selected image domain: preview dimensions replace the main canvas and preview timing
/// becomes one final still presentation. Physical frame IDs, noise seeds, LF dependencies and all
/// source bit/byte ranges remain unchanged. This lowering never reads or rewrites entropy data.
#[derive(Clone, Debug)]
pub struct SelectedImageInventory {
    selection: ImageSelection,
    source: ImageSourceInventory,
    reconstruction: Arc<CodestreamInventory>,
    complete_reconstruction: bool,
}

impl SelectedImageInventory {
    /// Internal reconstruction view for one dependency-complete main-image seek interval.
    /// Original source metadata, physical IDs, noise seeds and byte ranges stay unchanged.
    pub(crate) fn for_seek(
        &self,
        range: std::ops::Range<usize>,
    ) -> Result<Self, ImageSelectionError> {
        if self.selection != ImageSelection::Main || range.is_empty() {
            return Err(ImageSelectionError::InvalidInventory(
                "seek requires a nonempty main interval",
            ));
        }
        let reconstruction = CodestreamInventory {
            codestream_bytes: self.reconstruction.codestream_bytes,
            image_header: self.reconstruction.image_header.clone(),
            frames: self
                .reconstruction
                .frames
                .get(range)
                .ok_or(ImageSelectionError::InvalidInventory(
                    "seek interval exceeds main frames",
                ))?
                .to_vec(),
        };
        Ok(Self {
            selection: self.selection,
            source: self.source.clone(),
            reconstruction: Arc::new(reconstruction),
            complete_reconstruction: false,
        })
    }

    pub fn new(
        source: Arc<CodestreamInventory>,
        selection: ImageSelection,
    ) -> Result<Self, ImageSelectionError> {
        let reconstruction = source.select_image(selection)?;
        Ok(Self {
            selection,
            source: ImageSourceInventory::Complete(source),
            reconstruction,
            complete_reconstruction: true,
        })
    }

    /// Constructs a selected preview only after its FrameEnd event has been received.
    pub(crate) fn from_preview_prefix(
        image_header: Arc<ImageHeaderInventory>,
        frame: Arc<FrameInventory>,
    ) -> Result<Self, ImageSelectionError> {
        let codestream_bytes = frame
            .sections
            .last()
            .and_then(|section| section.bytes.end())
            .ok_or(ImageSelectionError::InvalidInventory(
                "preview has no bounded final section",
            ))?;
        let reconstruction = image_header.select_preview(&frame, codestream_bytes)?;
        Ok(Self {
            selection: ImageSelection::Preview,
            source: ImageSourceInventory::PreviewPrefix {
                image_header,
                frame,
                codestream_bytes,
            },
            reconstruction,
            complete_reconstruction: true,
        })
    }

    #[must_use]
    pub const fn selection(&self) -> ImageSelection {
        self.selection
    }

    #[must_use]
    pub fn source_inventory(&self) -> &ImageSourceInventory {
        &self.source
    }

    #[must_use]
    pub fn reconstruction_inventory(&self) -> &CodestreamInventory {
        &self.reconstruction
    }

    /// Whether reconstruction covers the complete selected image.
    /// A seek interval may stop at a nonfinal presentation without changing its reference
    /// saving or color semantics. Use [`crate::FrameExecutionPlan::negotiate_selected`]
    /// to plan either form.
    #[must_use]
    pub const fn reconstruction_is_complete(&self) -> bool {
        self.complete_reconstruction
    }
}
