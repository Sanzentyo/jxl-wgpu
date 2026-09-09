//! Explicit image-domain lowering, preserving physical frame IDs and entropy ranges.

use jxl_gpu_bitstream::{
    CodestreamInventory, FrameBlendMode, FrameInventory, FrameType, ImageHeaderInventory,
};
use std::sync::Arc;

/// Which independently presented image in a JPEG XL codestream to decode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ImageSelection {
    /// The main still image or complete animation. Preview pixels are not requested.
    #[default]
    Main,
    /// The single embedded preview, presented with its own dimensions and no animation timing.
    Preview,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ImageSelectionError {
    #[error("the codestream has no embedded preview")]
    MissingPreview,
    #[error("the embedded preview was already taken from this stream")]
    PreviewAlreadyTaken,
    #[error("the codestream has no main image frame")]
    MissingMainImage,
    #[error("invalid image frame inventory: {0}")]
    InvalidInventory(&'static str),
}

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
}

impl SelectedImageInventory {
    pub fn new(
        source: Arc<CodestreamInventory>,
        selection: ImageSelection,
    ) -> Result<Self, ImageSelectionError> {
        use ImageSelectionError::{InvalidInventory, MissingMainImage, MissingPreview};
        for (index, frame) in source.frames.iter().enumerate() {
            if usize::try_from(frame.frame_index).ok() != Some(index) {
                return Err(InvalidInventory("noncontiguous physical frame indices"));
            }
        }
        let preview = source.image_header.preview_size;
        let main_start = usize::from(preview.is_some());
        if source.frames.len() <= main_start {
            return Err(MissingMainImage);
        }
        if source
            .frames
            .iter()
            .enumerate()
            .any(|(index, frame)| frame.is_preview != (index == 0 && preview.is_some()))
        {
            return Err(InvalidInventory(
                "preview must occupy exactly the first physical frame",
            ));
        }
        if let Some((width, height)) = preview {
            let frame = &source.frames[0];
            validate_preview(width, height, frame)?;
        }
        let reconstruction = match (selection, preview) {
            (ImageSelection::Main, None) => Arc::clone(&source),
            (ImageSelection::Preview, None) => return Err(MissingPreview),
            (ImageSelection::Main, Some(_)) => {
                let mut image_header = source.image_header.clone();
                image_header.preview_size = None;
                Arc::new(CodestreamInventory {
                    codestream_bytes: source.codestream_bytes,
                    image_header,
                    frames: source.frames[main_start..].to_vec(),
                })
            }
            (ImageSelection::Preview, Some((width, height))) => lower_preview(
                &source.image_header,
                &source.frames[0],
                width,
                height,
                source.codestream_bytes,
            ),
        };
        Ok(Self {
            selection,
            source: ImageSourceInventory::Complete(source),
            reconstruction,
        })
    }

    /// Constructs a selected preview only after its FrameEnd event has been received.
    pub(crate) fn from_preview_prefix(
        image_header: Arc<ImageHeaderInventory>,
        frame: Arc<FrameInventory>,
    ) -> Result<Self, ImageSelectionError> {
        let (width, height) = image_header
            .preview_size
            .ok_or(ImageSelectionError::MissingPreview)?;
        if frame.frame_index != 0 || !frame.is_preview {
            return Err(ImageSelectionError::InvalidInventory(
                "preview must be physical frame zero",
            ));
        }
        validate_preview(width, height, &frame)?;
        let codestream_bytes = frame
            .sections
            .last()
            .and_then(|section| section.bytes.end())
            .ok_or(ImageSelectionError::InvalidInventory(
                "preview has no bounded final section",
            ))?;
        let reconstruction = lower_preview(&image_header, &frame, width, height, codestream_bytes);
        Ok(Self {
            selection: ImageSelection::Preview,
            source: ImageSourceInventory::PreviewPrefix {
                image_header,
                frame,
                codestream_bytes,
            },
            reconstruction,
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
}

fn validate_preview(
    width: u32,
    height: u32,
    frame: &FrameInventory,
) -> Result<(), ImageSelectionError> {
    use ImageSelectionError::InvalidInventory;
    if width == 0 || height == 0 || width > 4096 || height > 4096 {
        return Err(InvalidInventory("preview dimensions must be in 1..=4096"));
    }
    if frame.frame_type != FrameType::Regular
        || frame.lf_level != 0
        || frame.lf_source_frame.is_some()
        || frame.uses_lf_frame()
        || frame.save_as_reference != 0
        || frame.have_crop
        || frame.x0 != 0
        || frame.y0 != 0
        || frame.width != width
        || frame.height != height
        || std::iter::once(&frame.color_blend)
            .chain(&frame.extra_channel_blends)
            .any(|blend| blend.mode != FrameBlendMode::Replace)
    {
        return Err(InvalidInventory(
            "preview must be a self-contained regular frame",
        ));
    }
    Ok(())
}

fn lower_preview(
    source_header: &ImageHeaderInventory,
    source_frame: &FrameInventory,
    width: u32,
    height: u32,
    codestream_bytes: u64,
) -> Arc<CodestreamInventory> {
    let mut image_header = source_header.clone();
    image_header.width = width;
    image_header.height = height;
    image_header.preview_size = None;
    image_header.intrinsic_size = None;
    image_header.animation = None;
    let mut frame = source_frame.clone();
    // Preview is a separate still presentation even if its encoded is_last is false
    // or the image header includes animation timing for the main image.
    frame.is_preview = false;
    frame.is_last = true;
    frame.duration_ticks = 0;
    frame.timecode = None;
    frame.save_as_reference = 0;
    frame.save_before_color_transform = false;
    Arc::new(CodestreamInventory {
        codestream_bytes,
        image_header,
        frames: vec![frame],
    })
}
