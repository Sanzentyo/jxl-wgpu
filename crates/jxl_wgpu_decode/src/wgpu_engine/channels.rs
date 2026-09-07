//! Select output planes after the complete codestream channel topology has been reconstructed.

use jxl_gpu_bitstream::{ExtraChannelTypeInventory, SampleBitDepth};

use crate::model::native_modular_format;
use crate::modular_transform::GpuModularChannelLayout;
use crate::profile::StandardModularProfile;
use crate::{Error, GpuOutputRequest, ModularChannels, Result};

#[derive(Clone, Debug)]
pub(super) struct OutputChannels {
    pub(super) indices: Vec<usize>,
    pub(super) depths: Vec<u8>,
    pub(super) format_channels: ModularChannels,
    pub(super) bits: u8,
}

impl OutputChannels {
    pub(super) fn identity(channels: ModularChannels, bits: u8) -> Self {
        Self {
            indices: (0..channels.count() as usize).collect(),
            depths: vec![bits; channels.count() as usize],
            format_channels: channels,
            bits,
        }
    }

    pub(super) fn negotiate(
        profile: &StandardModularProfile,
        request: &GpuOutputRequest,
    ) -> Result<Self> {
        if let Some(index) = request.extra_channel() {
            let extra =
                profile
                    .extra_channels
                    .get(index as usize)
                    .ok_or(Error::ExtraChannelIndex {
                        index,
                        count: profile.channels.extra_count(),
                    })?;
            let bits = integer_bits(extra.bit_depth)?;
            return Ok(Self {
                indices: vec![profile.channels.color_count() as usize + index as usize],
                depths: vec![bits],
                format_channels: ModularChannels::Gray,
                bits,
            });
        }
        let color_count = profile.channels.color_count() as usize;
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
                "spot-color rendering is not yet connected; preserve the spot planes to inspect base color").into());
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
        if native.is_some_and(|native| native.channels == ModularChannels::Gray) && color_count != 1
        {
            return Err(Error::UnsupportedOutputFormat(
                "numeric color output requires a grayscale source".into(),
            ));
        }
        let mut indices: Vec<_> = (0..color_count).collect();
        let mut depths = vec![profile.bits_per_sample; color_count];
        // Gray+alpha is a source topology, never a fabricated two-component RGB pixel format.
        // RGB output can bind the same gray view three times without copying its samples.
        if color_count == 1
            && (alpha.is_some() || native.is_some_and(|n| n.channels != ModularChannels::Gray))
        {
            indices = vec![0; 3];
            depths = vec![profile.bits_per_sample; 3];
        }
        if let Some((index, bits)) = alpha {
            indices.push(index);
            depths.push(integer_bits(bits)?);
        }
        if let Some(native) = native {
            match native.channels {
                ModularChannels::Gray => {
                    indices.truncate(1);
                    depths.truncate(1);
                }
                ModularChannels::Rgb => {
                    indices.truncate(3);
                    depths.truncate(3);
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
            depths,
            format_channels,
            bits: profile.bits_per_sample,
        })
    }

    pub(super) fn direct(&self, profile: &StandardModularProfile) -> bool {
        self.indices.len() == profile.channels.count() as usize
            && self.indices.iter().copied().eq(0..self.indices.len())
            && self
                .depths
                .iter()
                .all(|&bits| bits == profile.bits_per_sample)
            && self.format_channels.count() == profile.channels.count()
    }

    pub(super) fn select(
        &self,
        planes: &[GpuModularChannelLayout],
    ) -> Result<Vec<GpuModularChannelLayout>> {
        self.indices
            .iter()
            .zip(&self.depths)
            .map(|(&index, &bits)| {
                let mut plane = *planes
                    .get(index)
                    .ok_or(Error::EngineContract("Modular output plane is missing"))?;
                // Modular prediction uses the image's working depth. Output normalization uses the
                // independently declared original precision of each extra channel.
                plane.bit_depth = u32::from(bits);
                Ok(plane)
            })
            .collect()
    }
}

fn integer_bits(depth: SampleBitDepth) -> Result<u8> {
    match depth {
        SampleBitDepth::Integer {
            bits_per_sample: bits @ 1..=16,
        } => Ok(bits as u8),
        _ => Err(Error::EngineContract(
            "admitted Modular channel has unsupported precision",
        )),
    }
}
