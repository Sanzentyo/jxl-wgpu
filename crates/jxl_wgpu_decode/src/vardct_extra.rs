//! Descriptor-only assembly plan for VarDCT's distributed Modular extra channels.

use crate::modular_grouping::{
    ModularPassShiftRange, ModularSubimageKind, ModularSubimageRegion,
    build_modular_pass_shift_ranges, global_subimage_channel_count, grouped_subimage_topology,
};
use crate::modular_inverse::{ModularInversePlan, plan_modular_inverse};
use crate::modular_side_image::{ModularSideImageHeader, ModularSideImagePlan};
use crate::modular_transform::{
    GpuModularChannelLayout, ModularChannelTopology, ModularTransformPlan,
};
use crate::modular_tree::{BitInput, MaConfigIr, WpHeaderIr};
use crate::vardct_frontend::StandardVarDctProfile;
use crate::{Error, Result};

pub(crate) struct VarDctExtraSubimage {
    pub(crate) image: ModularSideImagePlan,
    pub(crate) targets: Vec<GpuModularChannelLayout>,
    pub(crate) packet_end: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VarDctExtraPlan {
    topology: ModularChannelTopology,
    layout: Vec<GpuModularChannelLayout>,
    global_channel_count: usize,
    pass_ranges: Vec<Option<ModularPassShiftRange>>,
    pub(crate) inverse: ModularInversePlan,
    pub(crate) wp_header: WpHeaderIr,
}

impl VarDctExtraPlan {
    pub(crate) fn ac_stream_end(
        &self,
        profile: &StandardVarDctProfile,
        pass: usize,
        group: u32,
    ) -> std::result::Result<
        crate::vardct_pass_group::HfCoefficientStreamEnd,
        crate::vardct_packet::BoundedVarDctPacketError,
    > {
        use crate::vardct_pass_group::HfCoefficientStreamEnd;
        let (topology, _, _) = self.subimage(profile, Some(pass), group).map_err(|error| {
            crate::vardct_packet::BoundedVarDctPacketError::ModularTree(error.to_string())
        })?;
        Ok(if topology.sample_count() == Some(0) {
            HfCoefficientStreamEnd::Packet
        } else {
            HfCoefficientStreamEnd::Continuation
        })
    }
    pub(crate) fn new(
        header: &ModularSideImageHeader,
        profile: &StandardVarDctProfile,
        passes: &jxl_gpu_bitstream::FramePassesInventory,
    ) -> Result<Self> {
        Ok(Self {
            topology: header.transforms.topology.clone(),
            layout: header.transforms.topology.gpu_layout()?,
            global_channel_count: global_subimage_channel_count(
                &header.transforms.topology,
                profile.group_dimension,
            ),
            pass_ranges: build_modular_pass_shift_ranges(
                profile.coefficient_shifts.len() as u32,
                &passes.downsampling,
                &passes.last_pass,
            )?,
            inverse: plan_modular_inverse(&header.transforms)?,
            wp_header: header.wp_header,
        })
    }

    pub(crate) fn parse_global(
        &self,
        reader: &mut impl BitInput,
        mut header: ModularSideImageHeader,
        bit_depth: u32,
        global_ma: Option<&MaConfigIr>,
    ) -> Result<Option<ModularSideImagePlan>> {
        let topology = ModularChannelTopology::new(
            self.topology.channels()[..self.global_channel_count].to_vec(),
            self.topology.meta_channel_count(),
            Default::default(),
        )?;
        if topology.sample_count() == Some(0) {
            return Ok(None);
        }
        // These are already frame-transformed channels. The frame inverse runs after all
        // subimages have been assembled, so the global prefix has only identity reconstruction.
        header.transforms = ModularTransformPlan::from_ir(topology, vec![], Default::default())?;
        header.finish(reader, bit_depth, 0, global_ma).map(Some)
    }

    pub(crate) fn global_targets(&self) -> &[GpuModularChannelLayout] {
        &self.layout[..self.global_channel_count]
    }

    pub(crate) fn has_lf(&self) -> bool {
        self.topology.channels()[self.global_channel_count..]
            .iter()
            .any(|channel| {
                channel.width != 0
                    && channel.height != 0
                    && channel.hshift >= 3
                    && channel.vshift >= 3
            })
    }

    pub(crate) fn subimage(
        &self,
        profile: &StandardVarDctProfile,
        pass: Option<usize>,
        index: u32,
    ) -> Result<(ModularChannelTopology, Vec<GpuModularChannelLayout>, u32)> {
        let (rect, kind, range, stream_index, region_dimension) = if let Some(pass) = pass {
            let rect = profile
                .pass_group_rect(u64::from(index))
                .map_err(crate::vardct_packet::BoundedVarDctPacketError::from)
                .map_err(crate::vardct_engine::VarDctDecodeError::from)?;
            let range = self
                .pass_ranges
                .get(pass)
                .copied()
                .ok_or(Error::EngineContract(
                    "extra channel pass index exceeds frame schedule",
                ))?;
            let stream_index = 18
                + 3 * profile.low_frequency_group_count
                + profile.group_count * pass as u64
                + u64::from(index);
            (
                rect,
                ModularSubimageKind::PassGroup,
                range,
                stream_index,
                profile.group_dimension,
            )
        } else {
            let rect = profile
                .low_frequency_group_rect(u64::from(index))
                .map_err(crate::vardct_packet::BoundedVarDctPacketError::from)
                .map_err(crate::vardct_engine::VarDctDecodeError::from)?;
            (
                rect,
                ModularSubimageKind::LowFrequencyGroup,
                None,
                1 + profile.low_frequency_group_count + u64::from(index),
                profile.group_dimension * 8,
            )
        };
        let (topology, targets) = grouped_subimage_topology(
            &self.topology,
            &self.layout,
            self.global_channel_count,
            ModularSubimageRegion {
                kind,
                column: rect.x / region_dimension,
                row: rect.y / region_dimension,
                group_dimension: profile.group_dimension,
            },
            range,
            Default::default(),
        )?;
        let stream_index = u32::try_from(stream_index)
            .map_err(|_| Error::EngineContract("extra channel stream ID exceeds WGSL u32"))?;
        Ok((topology, targets, stream_index))
    }
}
