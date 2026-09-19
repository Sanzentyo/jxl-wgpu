//! Select direct gain packing or the shared resident ICC presentation before decoding either image.

use std::sync::Arc;

use jxl_gpu_bitstream::{
    ExtraChannelInventory, ExtraChannelTypeInventory, ImageHeaderInventory, SampleBitDepth,
    gain_map::GainMapMetadata,
};
use jxl_gpu_formats::{
    Channel, ColorSpecification, ImageLayout, PixelFormat, RgbChannelOrder, SampleKind,
};
use jxl_gpu_protocol::{DisplayIntensity, OutputId, RgbColorEncoding};
use jxl_wgpu::{GpuImageFrame, GpuImageOutput, WgpuBackend};

use crate::frame_surface::{
    FrameSurfaceEncoding, FrameSurfaceLayout,
    compositor::{Surface, icc},
    icc_transform::Transforms,
};
use crate::{AlphaOutputPolicy, GpuFrameLease, GpuOutputRequest, OrientationPolicy, Result};

use super::{GainMapDecodeError, render};

pub(super) struct Plan {
    pub gain: render::Plan,
    icc: Option<IccOutput>,
}

struct IccOutput {
    source: Arc<FrameSurfaceLayout>,
    encoding: FrameSurfaceEncoding,
    presentation: icc::Presentation,
}

impl Plan {
    pub fn new(
        backend: &WgpuBackend,
        request: &GpuOutputRequest,
        image: &ImageHeaderInventory,
        working: RgbColorEncoding,
        metadata: &GainMapMetadata,
        weight: f32,
        reference_white: DisplayIntensity,
    ) -> Result<Self> {
        if !matches!(request.format().color_spec, ColorSpecification::Icc(_)) {
            return Ok(Self {
                gain: render::Plan::new(
                    backend,
                    request,
                    image,
                    working,
                    metadata,
                    weight,
                    reference_white,
                )?,
                icc: None,
            });
        }
        let encoding = FrameSurfaceEncoding::Rgb(working);
        let intermediate = GpuOutputRequest::color(PixelFormat::rgb_f32(
            RgbChannelOrder::Rgba,
            true,
            encoding.format().color_spec,
        ))?
        .with_orientation_policy(OrientationPolicy::Keep)
        .with_alpha_output_policy(AlphaOutputPolicy::Unassociated);
        let gain = render::Plan::new(
            backend,
            &intermediate,
            image,
            working,
            metadata,
            weight,
            reference_white,
        )?;
        let layout = gain.layout();
        let mut alpha_plane = layout.planes[3].clone();
        alpha_plane.plane_index = 0;
        // This intermediate is canonical packed planar RGBA. Views keep its exact offsets;
        // ICC binds the whole allocation, and alpha copies require only word alignment.
        let source = Arc::new(FrameSurfaceLayout {
            color: ImageLayout::from_planes(
                layout.extent,
                encoding.format(),
                layout.planes[..3].to_vec(),
            )?,
            extras: vec![ImageLayout::from_planes(
                layout.extent,
                PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                vec![alpha_plane],
            )?],
            color_plane_bytes: layout.planes[1].offset,
            storage_bytes: layout.logical_size,
        });
        let alpha = ExtraChannelInventory {
            channel_type: ExtraChannelTypeInventory::Alpha { associated: false },
            bit_depth: SampleBitDepth::Float {
                bits_per_sample: 32,
                exponent_bits_per_sample: 8,
            },
            dimension_shift: 0,
            name_bytes: Vec::new(),
        };
        // Gain math receives straight RGBA even when the image declares associated alpha.
        // Resolve Preserve against that original declaration before presenting this new surface.
        let association = if request.alpha_output_policy() == AlphaOutputPolicy::Associated
            || (request.alpha_output_policy() == AlphaOutputPolicy::Preserve
                && super::associated_alpha(image))
        {
            AlphaOutputPolicy::Associated
        } else {
            AlphaOutputPolicy::Unassociated
        };
        let request = request.clone().with_alpha_output_policy(association);
        let presentation = icc::Presentation::new(
            backend,
            &source,
            encoding.clone(),
            icc::Output::Color(&request),
            icc::ImageMetadata {
                orientation: request.orientation_policy().resolve(
                    jxl_gpu_protocol::OutputOrientation::from_exif_value(image.orientation)
                        .ok_or(GainMapDecodeError::Contract("invalid primary orientation"))?,
                ),
                intensity: DisplayIntensity::new(image.tone_mapping.intensity_target.to_f32())
                    .ok_or(GainMapDecodeError::Contract("invalid primary intensity"))?,
                tone_mapping: None,
                extras: &[alpha],
            },
            &mut Transforms::default(),
        )?;
        Ok(Self {
            gain,
            icc: Some(IccOutput {
                source,
                encoding,
                presentation,
            }),
        })
    }

    pub async fn finish(
        &self,
        backend: &WgpuBackend,
        frame: GpuFrameLease<GpuImageFrame>,
    ) -> Result<GpuFrameLease<GpuImageFrame>> {
        let Some(icc) = &self.icc else {
            return Ok(frame);
        };
        let [output] = frame.output().outputs.as_slice() else {
            return Err(GainMapDecodeError::Contract("expected one gain output").into());
        };
        if output.layout != *self.gain.layout() {
            return Err(GainMapDecodeError::Contract("gain output layout changed").into());
        }
        let source = Surface {
            buffer: output.buffer.clone(),
            layout: Arc::clone(&icc.source),
            encoding: icc.encoding.clone(),
        };
        let mut work = icc.presentation.pack(backend, &source)?;
        let buffer = std::future::poll_fn(|context| work.poll(context)).await?;
        let layout = icc.presentation.layout().clone();
        let output = GpuImageFrame {
            token: frame.output().token,
            changed: crate::frame_surface::changed_regions(&layout, None),
            outputs: vec![GpuImageOutput {
                id: OutputId(0),
                layout,
                buffer,
            }],
        };
        Ok(frame.replace_output(output))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
