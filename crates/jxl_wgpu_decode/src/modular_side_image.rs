//! A Modular substream whose ending bit cursor belongs to a surrounding image packet.

use crate::modular_inverse::{ModularInversePlan, plan_modular_inverse};
use crate::modular_transform::{
    GpuModularChannelLayout, ModularChannelTopology, ModularTransformLimits, ModularTransformPlan,
    PackedModularChannelMetadata, parse_modular_transforms,
};
use crate::modular_tree::{BitInput, MaConfigIr, MaTreeLimits, WpHeaderIr, parse_ma_config};
use crate::{Error, Result};

/// Parsed before deciding which transformed channels belong to this substream. In particular,
/// a global header exists even when every image channel belongs to an LF or AC group; such an
/// empty global subimage has no local MA tree or entropy stream following its header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModularSideImageHeader {
    pub(crate) use_global_tree: bool,
    pub(crate) wp_header: WpHeaderIr,
    pub(crate) transforms: ModularTransformPlan,
}

impl ModularSideImageHeader {
    pub(crate) fn parse(
        reader: &mut impl BitInput,
        topology: ModularChannelTopology,
    ) -> Result<Self> {
        Ok(Self {
            use_global_tree: reader.read_bits(1)? != 0,
            wp_header: WpHeaderIr::parse(reader)?,
            transforms: parse_modular_transforms(
                reader,
                topology,
                ModularTransformLimits::default(),
            )?,
        })
    }

    pub(crate) fn finish(
        self,
        reader: &mut impl BitInput,
        bit_depth: u32,
        stream_index: u32,
        global_ma_config: Option<&MaConfigIr>,
    ) -> Result<ModularSideImagePlan> {
        let Self {
            use_global_tree,
            wp_header,
            transforms,
        } = self;
        let ma_config = if use_global_tree {
            global_ma_config
                .ok_or(Error::MissingGlobalMaTree { stream_index })?
                .clone()
        } else {
            parse_ma_config(reader, MaTreeLimits::default())?
        };
        ModularSideImagePlan::from_descriptor(
            reader,
            transforms,
            bit_depth,
            stream_index,
            wp_header,
            ma_config,
        )
    }
}

/// All image samples and inverse transforms execute on the GPU. Only the descriptor is host data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModularSideImagePlan {
    pub bit_depth: u32,
    pub stream_index: u32,
    pub token_bit_offset: u32,
    pub wp_header: WpHeaderIr,
    pub metadata: Vec<u32>,
    pub needs_self_correcting: bool,
    pub channel_metadata: PackedModularChannelMetadata,
    pub meta_channel_count: usize,
    pub inverse_plan: ModularInversePlan,
    pub final_planes: Vec<GpuModularChannelLayout>,
    pub decoded_words: u32,
    pub maximum_width: u32,
    pub lz77_window_words: u32,
}

impl ModularSideImagePlan {
    pub(crate) fn parse(
        reader: &mut impl BitInput,
        topology: ModularChannelTopology,
        bit_depth: u32,
        stream_index: u32,
        global_ma_config: Option<&MaConfigIr>,
    ) -> Result<Self> {
        if topology.channels().is_empty() || !(1..=32).contains(&bit_depth) {
            return Err(Error::EngineContract(
                "invalid Modular side-image source topology",
            ));
        }
        ModularSideImageHeader::parse(reader, topology)?.finish(
            reader,
            bit_depth,
            stream_index,
            global_ma_config,
        )
    }

    fn from_descriptor(
        reader: &impl BitInput,
        transforms: ModularTransformPlan,
        bit_depth: u32,
        stream_index: u32,
        wp_header: WpHeaderIr,
        ma_config: MaConfigIr,
    ) -> Result<Self> {
        let token_bit_offset = u32::try_from(reader.bit_offset())
            .map_err(|_| Error::backend("Modular side-image entropy offset exceeds WGSL u32"))?;
        let channel_metadata = transforms
            .topology
            .gpu_entropy_channels(ma_config.maximum_tree_property())?;
        let decoded_words = channel_metadata
            .channels
            .last()
            .map_or(0, |c| c.decoded_end);
        let maximum_width = channel_metadata
            .channels
            .iter()
            .map(|c| c.width)
            .max()
            .unwrap_or(1);
        let inverse_plan = plan_modular_inverse(&transforms)?;
        let final_planes = inverse_plan.final_gpu_layouts();
        let lz77_window_words = ma_config
            .entropy
            .lz77_window_words(maximum_width, decoded_words)?;
        Ok(Self {
            bit_depth,
            stream_index,
            token_bit_offset,
            wp_header,
            metadata: ma_config.pack_gpu_metadata()?.words,
            needs_self_correcting: ma_config.needs_self_correcting(),
            channel_metadata,
            meta_channel_count: transforms.topology.meta_channel_count(),
            inverse_plan,
            final_planes,
            decoded_words,
            maximum_width,
            lz77_window_words,
        })
    }
}
