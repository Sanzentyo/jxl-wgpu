//! Codec-independent image selection for reconstruction and indexed assembly.
use crate::{CodestreamInventory, FrameBlendMode, FrameInventory, FrameType, ImageHeaderInventory};
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

impl CodestreamInventory {
    /// Selects a complete image domain without changing physical IDs, noise seeds or byte ranges.
    /// This validates metadata only; no entropy or pixel validation is implied.
    pub fn select_image(
        self: &Arc<Self>,
        selection: ImageSelection,
    ) -> Result<Arc<Self>, ImageSelectionError> {
        let source = self;
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
            (ImageSelection::Main, None) => Arc::clone(source),
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
        Ok(reconstruction)
    }
}

impl ImageHeaderInventory {
    /// Lowers one fully inventoried preview frame. The caller remains responsible for transport
    /// completion and entropy validation; this view does not claim that a main image is present.
    pub fn select_preview(
        &self,
        frame: &FrameInventory,
        codestream_bytes: u64,
    ) -> Result<Arc<CodestreamInventory>, ImageSelectionError> {
        let (width, height) = self
            .preview_size
            .ok_or(ImageSelectionError::MissingPreview)?;
        if frame.frame_index != 0 || !frame.is_preview {
            return Err(ImageSelectionError::InvalidInventory(
                "preview must be physical frame zero",
            ));
        }
        validate_preview(width, height, frame)?;
        Ok(lower_preview(self, frame, width, height, codestream_bytes))
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
