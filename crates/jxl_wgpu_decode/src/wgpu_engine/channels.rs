//! Select output planes after the complete codestream channel topology has been reconstructed.

use jxl_gpu_bitstream::{ExtraChannelTypeInventory, SampleBitDepth};

use crate::model::native_modular_format;
use crate::modular_sample::{ModularOutputPlane, ModularSampleEncoding};
use crate::modular_transform::GpuModularChannelLayout;
use crate::profile::StandardModularProfile;
use crate::{Error, GpuOutputRequest, ModularChannels, Result};

#[derive(Clone, Debug)]
pub(super) struct OutputChannels {
    pub(super) indices: Vec<usize>,
    pub(super) encodings: Vec<ModularSampleEncoding>,
    pub(super) format_channels: ModularChannels,
    pub(super) encoding: ModularSampleEncoding,
    pub(super) alpha_conversion: jxl_wgpu::AlphaConversion,
}

impl OutputChannels {
    pub(super) fn identity(channels: ModularChannels, encoding: ModularSampleEncoding) -> Self {
        Self {
            indices: (0..channels.count() as usize).collect(),
            encodings: vec![encoding; channels.count() as usize],
            format_channels: channels,
            encoding,
            alpha_conversion: jxl_wgpu::AlphaConversion::Preserve,
        }
    }

    pub(super) fn negotiate(
        profile: &StandardModularProfile,
        request: &GpuOutputRequest,
    ) -> Result<Self> {
        // Prediction needs only XYB. LF patches and presentation also normalize all extras;
        // every source plane is entropy-validated regardless of either selection.
        if profile.progressive_dc.is_some() && !request.retains_lf_extras() {
            return Ok(Self::identity(
                ModularChannels::Rgb,
                profile.sample_encoding,
            ));
        }
        if let Some(index) = request.extra_channel() {
            let extra =
                profile
                    .extra_channels
                    .get(index as usize)
                    .ok_or(Error::ExtraChannelIndex {
                        index,
                        count: profile.channels.extra_count(),
                    })?;
            let encoding = sample_encoding(extra.bit_depth)?;
            return Ok(Self {
                indices: vec![profile.channels.color_count() as usize + index as usize],
                encodings: vec![encoding],
                format_channels: ModularChannels::Gray,
                encoding,
                alpha_conversion: jxl_wgpu::AlphaConversion::Preserve,
            });
        }
        let color_count = profile.channels.color_count() as usize;
        if request.retains_frame_surface() || request.retains_lf_extras() {
            let indices = (0..3)
                .map(|index| if color_count == 1 { 0 } else { index })
                .chain(color_count..color_count + profile.extra_channels.len())
                .collect();
            let mut encodings = vec![profile.sample_encoding; 3];
            encodings.extend(
                profile
                    .extra_channels
                    .iter()
                    .map(|extra| sample_encoding(extra.bit_depth))
                    .collect::<Result<Vec<_>>>()?,
            );
            return Ok(Self {
                indices,
                encodings,
                format_channels: ModularChannels::Rgb,
                encoding: profile.sample_encoding,
                alpha_conversion: jxl_wgpu::AlphaConversion::Preserve,
            });
        }
        if let Some(index) = request.numeric_color_channel(color_count == 1)? {
            return Ok(Self {
                indices: vec![index as usize],
                encodings: vec![profile.sample_encoding],
                format_channels: ModularChannels::Gray,
                encoding: profile.sample_encoding,
                alpha_conversion: jxl_wgpu::AlphaConversion::Preserve,
            });
        }
        if profile
            .extra_channels
            .iter()
            .any(|extra| extra.channel_type == ExtraChannelTypeInventory::NonOptional)
        {
            return Err(crate::UnsupportedProfile::new(
                crate::UnsupportedCodestreamFeature::ExtraChannels,
                "a non-optional unknown extra channel cannot be omitted from color interpretation",
            )
            .into());
        }
        if request.spot_color_policy() == crate::SpotColorPolicy::Render
            && profile.extra_channels.iter().any(|extra| {
                matches!(
                    extra.channel_type,
                    ExtraChannelTypeInventory::SpotColour { .. }
                )
            })
        {
            return Err(crate::UnsupportedProfile::new(crate::UnsupportedCodestreamFeature::ExtraChannels,
                "spot presentation requires the common WgpuDecodeEngine; physical Modular output can preserve the spot planes").into());
        }
        let alpha = profile
            .extra_channels
            .iter()
            .enumerate()
            .find_map(|(index, extra)| {
                matches!(extra.channel_type, ExtraChannelTypeInventory::Alpha { .. })
                    .then_some((color_count + index, extra.bit_depth))
            });
        let native = native_modular_format(request.format());
        let alpha_conversion = request.alpha_conversion(&profile.extra_channels);
        let mut indices: Vec<_> = (0..color_count).collect();
        let mut encodings = vec![profile.sample_encoding; color_count];
        // Gray+alpha is a source topology, never a fabricated two-component RGB pixel format.
        // RGB output can bind the same gray view three times without copying its samples.
        if color_count == 1
            && (alpha.is_some() || native.is_some_and(|n| n.channels != ModularChannels::Gray))
        {
            indices = vec![0; 3];
            encodings = vec![profile.sample_encoding; 3];
        }
        if let Some((index, encoding)) = alpha {
            indices.push(index);
            encodings.push(sample_encoding(encoding)?);
        }
        if let Some(native) = native {
            match native.channels {
                ModularChannels::Gray => {
                    indices.truncate(1);
                    encodings.truncate(1);
                }
                ModularChannels::Rgb => {
                    if alpha_conversion == jxl_wgpu::AlphaConversion::Preserve {
                        indices.truncate(3);
                        encodings.truncate(3);
                    }
                }
                ModularChannels::Rgba => {}
            }
        }
        let format_channels = native.map_or_else(
            || match indices.len() {
                1 => ModularChannels::Gray,
                3 => ModularChannels::Rgb,
                _ => ModularChannels::Rgba,
            },
            |n| n.channels,
        );
        Ok(Self {
            indices,
            encodings,
            format_channels,
            encoding: profile.sample_encoding,
            alpha_conversion,
        })
    }

    pub(super) fn direct(&self, profile: &StandardModularProfile) -> bool {
        self.alpha_conversion == jxl_wgpu::AlphaConversion::Preserve
            && self.indices.len() == profile.channels.count() as usize
            && self.indices.iter().copied().eq(0..self.indices.len())
            && self
                .encodings
                .iter()
                .all(|&encoding| encoding == profile.sample_encoding)
            && self.format_channels.count() == profile.channels.count()
    }

    pub(super) fn select(
        &self,
        planes: &[GpuModularChannelLayout],
    ) -> Result<Vec<ModularOutputPlane>> {
        self.indices
            .iter()
            .zip(&self.encodings)
            .map(|(&index, &encoding)| {
                let plane = *planes
                    .get(index)
                    .ok_or(Error::EngineContract("Modular output plane is missing"))?;
                Ok(ModularOutputPlane::new(plane, encoding))
            })
            .collect()
    }
}

fn sample_encoding(depth: SampleBitDepth) -> Result<ModularSampleEncoding> {
    ModularSampleEncoding::new(depth).ok_or(Error::EngineContract(
        "admitted Modular channel has unsupported precision",
    ))
}
