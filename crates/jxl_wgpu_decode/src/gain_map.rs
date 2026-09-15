//! GPU reconstruction of the alternate image in a `jhgm` container box.
//!
//! [`GpuDecoder::decode_gain_map`] decodes both codestreams through the stock GPU engine,
//! applies the map to linear RGB, and uses the shared color/orientation/output packer. The
//! rendering profile accepts either headroom direction, one still presentation per codestream,
//! enumerated application primaries and output color, with explicit display-headroom selection.
//! Other profiles are explicit errors; the bitstream crate can still preserve their metadata.

use std::sync::Arc;

use jxl_gpu_bitstream::{
    CodestreamInventory, ColourSpaceInventory, ExtraChannelTypeInventory,
    gain_map::{GainMapBundle, GainMapLimits, JHGM},
    metadata::{MetadataLimits, MetadataSelection},
};
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder};
use jxl_gpu_protocol::{RgbColorEncoding, TransferFunction};
use jxl_wgpu::GpuImageFrame;

use crate::{
    AlphaOutputPolicy, GpuDecoder, GpuFrameLease, GpuOutputMapping, GpuOutputRequest,
    GpuSubmissionEngine, ImageSelection, OrientationPolicy, Result, WgpuDecodeEngine,
};

mod render;
mod rendering;
pub use rendering::{GainMapRendering, GainMapRendition};

/// Independent limits for container metadata and reconstructed color metadata.
/// Primary/auxiliary image dimensions, inventories and GPU storage use the decoder's limits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GainMapDecodeLimits {
    pub metadata: MetadataLimits,
    pub bundle: GainMapLimits,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GainMapDecodeError {
    #[error("alternate reconstruction requires exactly one jhgm box, found {0}")]
    BoxCount(usize),
    #[error("unsupported gain-map rendering profile: {0}")]
    Unsupported(&'static str),
    #[error("invalid gain-map rendering request: {0}")]
    InvalidRequest(&'static str),
    #[error("gain-map GPU contract: {0}")]
    Contract(&'static str),
}

impl GpuDecoder<WgpuDecodeEngine> {
    /// Reconstructs a complete alternate still image, retaining the primary frame's slot and
    /// metadata. All decoded pixels remain GPU resident, including map resampling and packing.
    ///
    /// Gain samples use their original component values, independently of their color transfer.
    /// Resampling is bilinear with aligned image edges and clamped borders. Alpha is preserved
    /// from the baseline; map alpha/extra channels are unsupported. Unit linear RGB represents
    /// the baseline header's intensity target in nits, including for PQ/HLG output.
    ///
    /// Uses the default 203-nit gain reference white. For display-headroom selection or a different
    /// reference white use [`Self::decode_gain_map`]. Animation, previews, progressive delivery,
    /// ICC application spaces and ICC output are currently typed unsupported profiles. Container parsing can
    /// retain these forms independently. Tone mapping needs an alternate-image luminance model
    /// and is rejected; output gamut mapping remains explicit through `request`.
    pub async fn decode_alternate(
        &self,
        encoded: &[u8],
        request: GpuOutputRequest,
        limits: GainMapDecodeLimits,
    ) -> Result<GpuFrameLease<GpuImageFrame>> {
        self.decode_gain_map(encoded, request, GainMapRendering::default(), limits)
            .await
    }

    /// Reconstructs a complete still at the requested display headroom, or the exact alternate.
    /// The gain equation uses `rendering.reference_white`; its result is converted back to the
    /// baseline image's linear units before the normal output color/alpha/orientation conversion.
    /// Zero application returns ordinary baseline output and does not decode the unused gain map.
    /// Equal headrooms select the baseline. A nonzero application requires the supported auxiliary
    /// still/color profile described by [`Self::decode_alternate`].
    pub async fn decode_gain_map(
        &self,
        encoded: &[u8],
        request: GpuOutputRequest,
        rendering: GainMapRendering,
        limits: GainMapDecodeLimits,
    ) -> Result<GpuFrameLease<GpuImageFrame>> {
        validate_request(&request)?;
        let parsed = jxl_gpu_bitstream::parse(encoded, self.parse_limits())?;
        let boxes = parsed.metadata(&MetadataSelection::Types(vec![JHGM]), limits.metadata)?;
        if boxes.boxes().len() != 1 {
            return Err(GainMapDecodeError::BoxCount(boxes.boxes().len()).into());
        }
        let bytes = boxes.boxes()[0].decode(limits.metadata)?;
        let bundle = GainMapBundle::parse(&bytes, limits.bundle)?;
        let metadata = bundle.metadata();
        let weight = rendering.weight(metadata)?;
        let main_selection = crate::SelectedImageInventory::new(
            Arc::new(parsed.codestream_inventory(self.engine().inventory_limits())?),
            ImageSelection::Main,
        )?;
        let main = main_selection.reconstruction_inventory();
        validate_still(main)?;
        let rendering::Weight::Apply(weight) = weight else {
            return self
                .decode_gain_map_still(Arc::from(parsed.codestream()), request)
                .await;
        };
        let map_selection = crate::SelectedImageInventory::new(
            Arc::new(
                jxl_gpu_bitstream::parse(bundle.codestream(), self.parse_limits())?
                    .codestream_inventory(self.engine().inventory_limits())?,
            ),
            ImageSelection::Main,
        )?;
        let map = map_selection.reconstruction_inventory();
        validate_still(map)?;
        if map.image_header.orientation != 1 {
            return Err(
                GainMapDecodeError::Unsupported("nonidentity auxiliary orientation").into(),
            );
        }
        if !map.image_header.extra_channels.is_empty() {
            return Err(GainMapDecodeError::Unsupported("auxiliary alpha/extra channels").into());
        }
        // An ICC profile can describe the baseline when the gain math explicitly selects an
        // enumerated alternate space. The existing GPU ICC connection performs that conversion.
        let working = if metadata.use_base_color_space {
            crate::image_color::original_encoding(&main.image_header).ok_or(
                GainMapDecodeError::Unsupported("ICC or unknown baseline application space"),
            )?
        } else {
            if bundle.alternate_icc().is_some() {
                return Err(
                    GainMapDecodeError::Unsupported("ICC alternate application space").into(),
                );
            }
            let color =
                bundle
                    .alternate_color_encoding()
                    .ok_or(GainMapDecodeError::Unsupported(
                        "alternate application space is unspecified",
                    ))?;
            let gray = matches!(
                color,
                jxl_gpu_bitstream::ColourEncodingInventory::Enumerated {
                    colour_space: ColourSpaceInventory::Grey,
                    ..
                }
            );
            crate::image_color::enumerated_encoding(color, gray).ok_or(
                GainMapDecodeError::Unsupported("unknown alternate application space"),
            )?
        };
        let working = RgbColorEncoding {
            space: working.space,
            transfer: TransferFunction::Linear,
        };
        // Plan the output before either image is submitted.
        let plan = render::Plan::new(
            self.engine().backend(),
            &request,
            &main.image_header,
            working,
            metadata,
            weight,
            rendering.reference_white,
        )?;
        let domain = crate::image_color::original_domain(&map.image_header)?;
        if matches!(
            domain,
            crate::frame_surface::FrameSurfaceEncoding::Cmyk { .. }
        ) {
            return Err(GainMapDecodeError::Unsupported("CMYK gain samples").into());
        }
        let map_request = GpuOutputRequest::frame_surface(domain);
        let color = crate::frame_surface::FrameSurfaceEncoding::Rgb(working)
            .format()
            .color_spec;
        let base_request =
            GpuOutputRequest::color(PixelFormat::rgb_f32(RgbChannelOrder::Rgba, true, color))?
                .with_max_frame_slots(request.max_frame_slots())
                .with_orientation_policy(OrientationPolicy::Keep)
                .with_alpha_output_policy(AlphaOutputPolicy::Unassociated)
                .with_spot_color_policy(request.spot_color_policy())
                .with_white_point_adaptation(request.white_point_adaptation())
                .with_icc_rendering_intent(request.icc_rendering_intent());
        let map_frame = self
            .decode_gain_map_still(Arc::from(bundle.codestream()), map_request)
            .await?;
        let base_frame = self
            .decode_gain_map_still(Arc::from(parsed.codestream()), base_request)
            .await?;
        let mut work = plan.submit(
            self.engine().backend(),
            base_frame.output(),
            map_frame.output(),
        )?;
        let buffer = std::future::poll_fn(|context| work.poll(context)).await?;
        let output = plan.frame(base_frame.output().token, buffer);
        Ok(base_frame.replace_output(output))
    }

    async fn decode_gain_map_still(
        &self,
        code: Arc<[u8]>,
        request: GpuOutputRequest,
    ) -> Result<GpuFrameLease<GpuImageFrame>> {
        let mut session = self.open_shared(code, request)?;
        let frame = session
            .next_frame_async()
            .await?
            .ok_or(GainMapDecodeError::Contract("missing still presentation"))?;
        if !frame.metadata.is_last {
            return Err(GainMapDecodeError::Unsupported("multiple presentations").into());
        }
        Ok(frame)
    }
}

fn validate_request(request: &GpuOutputRequest) -> Result<()> {
    if request.image_selection() != ImageSelection::Main || request.progressive_output() {
        return Err(GainMapDecodeError::Unsupported(
            "alternate reconstruction requires a complete main image",
        )
        .into());
    }
    if request.mapping() != GpuOutputMapping::Color
        || !matches!(request.format().color_spec, ColorSpecification::Defined(_))
    {
        return Err(
            GainMapDecodeError::Unsupported("alternate output requires enumerated color").into(),
        );
    }
    if request.tone_mapping_target().is_some() {
        return Err(GainMapDecodeError::Unsupported(
            "alternate tone mapping requires its own luminance model",
        )
        .into());
    }
    Ok(())
}

fn validate_still(inventory: &CodestreamInventory) -> Result<()> {
    if inventory.image_header.animation.is_some() {
        return Err(GainMapDecodeError::Unsupported("animation gain-map timing").into());
    }
    let plan = crate::FrameExecutionPlan::negotiate(inventory)?;
    if plan.presentations.len() != 1 {
        return Err(GainMapDecodeError::Unsupported(
            "requires one complete presentation per codestream",
        )
        .into());
    }
    Ok(())
}

fn associated_alpha(image: &jxl_gpu_bitstream::ImageHeaderInventory) -> bool {
    image
        .extra_channels
        .iter()
        .find_map(|extra| match extra.channel_type {
            ExtraChannelTypeInventory::Alpha { associated } => Some(associated),
            _ => None,
        })
        .unwrap_or(false)
}
