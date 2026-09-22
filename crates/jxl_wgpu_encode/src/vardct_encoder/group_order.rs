//! Physical AC-group order from caller metadata, geometry or validated GPU statistics.
use std::sync::Arc;

use super::saliency;
use super::types::{AC_GROUP_DIM_PIXELS, TiledVarDctGrid, VarDctFrameLayout};
use crate::{BackendError, EncodeError, FramePacketSet, GroupPacketKind};

const MAX_AC_GROUPS: usize = (TiledVarDctGrid::MAX_DIMENSION / AC_GROUP_DIM_PIXELS).pow(2) as usize;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
enum Order {
    #[default]
    Raster,
    Center(Option<(u32, u32)>),
    Explicit(Arc<[u32]>),
    Saliency,
}

/// Physical order of 256-pixel AC groups, repeated independently in every pass.
/// LF-global, LF groups and HF-global retain canonical order and precede all AC passes.
/// Automatic saliency uses GPU local contrast; all other choices use caller metadata/geometry.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct VarDctGroupOrder {
    order: Order,
}

impl VarDctGroupOrder {
    /// Prioritizes groups with the largest mean absolute RGB neighbor difference.
    /// The GPU measures left/up image edges using visible pixels only. Exact rational
    /// comparison and raster-ID ties make ordering independent of floating-point arithmetic.
    /// This local-contrast heuristic is not a perceptual quality guarantee.
    #[must_use]
    pub const fn saliency_first() -> Self {
        Self {
            order: Order::Saliency,
        }
    }

    pub(super) const fn requires_saliency(&self) -> bool {
        matches!(self.order, Order::Saliency)
    }

    pub(super) fn validate_scores(
        &self,
        frame: VarDctFrameLayout,
        records: Option<&[saliency::Record]>,
    ) -> Result<(), EncodeError> {
        match (self.requires_saliency(), records) {
            (true, Some(records)) => Ok(saliency::validate(records, frame)?),
            (false, None) => Ok(()),
            _ => Err(BackendError::InvalidArtifact(
                "saliency metadata does not match group-order policy",
            )
            .into()),
        }
    }

    /// Visits concentric group rings around the middle image pixel, nearest first within each ring.
    /// Squared pixel distance followed by raster group ID resolves ties using integer arithmetic.
    #[must_use]
    pub const fn center_first() -> Self {
        Self {
            order: Order::Center(None),
        }
    }

    /// Centers the same ordering on a source pixel, validated against the submitted frame.
    #[must_use]
    pub const fn centered_at(x: u32, y: u32) -> Self {
        Self {
            order: Order::Center(Some((x, y))),
        }
    }

    /// Accepts each raster AC-group ID exactly once in the requested physical order.
    /// The submitted image must have exactly this many AC groups.
    pub fn explicit(groups: Vec<u32>) -> Result<Self, EncodeError> {
        if groups.is_empty() || groups.len() > MAX_AC_GROUPS {
            return Err(EncodeError::InvalidConfiguration(
                "VarDCT group order has invalid length",
            ));
        }
        let mut seen = vec![false; groups.len()];
        for &group in &groups {
            let entry = seen
                .get_mut(group as usize)
                .ok_or(EncodeError::InvalidConfiguration(
                    "VarDCT group order is not an in-range permutation",
                ))?;
            if std::mem::replace(entry, true) {
                return Err(EncodeError::InvalidConfiguration(
                    "VarDCT group order repeats a group",
                ));
            }
        }
        Ok(Self {
            order: Order::Explicit(groups.into()),
        })
    }

    pub(super) fn validate(&self, frame: VarDctFrameLayout) -> Result<(), EncodeError> {
        match &self.order {
            Order::Center(Some((x, y))) if *x >= frame.width || *y >= frame.height => Err(
                EncodeError::InvalidConfiguration("VarDCT group-order center is outside the image"),
            ),
            Order::Explicit(groups) if groups.len() != frame.ac_group_count()? as usize => {
                Err(EncodeError::InvalidConfiguration(
                    "VarDCT group order does not match the image grid",
                ))
            }
            _ => Ok(()),
        }
    }

    pub(super) fn resolve(&self, frame: VarDctFrameLayout) -> Result<Vec<u32>, EncodeError> {
        self.validate(frame)?;
        if self.requires_saliency() {
            return Err(BackendError::InvalidArtifact(
                "saliency order requires validated GPU metadata",
            )
            .into());
        }
        if let Order::Explicit(groups) = &self.order {
            return Ok(groups.to_vec());
        }
        let mut groups: Vec<_> = (0..frame.ac_group_count()?).collect();
        if let Order::Center(center) = self.order {
            let (x, y) = center.unwrap_or(((frame.width - 1) / 2, (frame.height - 1) / 2));
            let (center_x, center_y) = (x / AC_GROUP_DIM_PIXELS, y / AC_GROUP_DIM_PIXELS);
            groups.sort_unstable_by_key(|&id| {
                let (gx, gy) = (id % frame.ac_groups_x, id / frame.ac_groups_x);
                let ring = gx.abs_diff(center_x).max(gy.abs_diff(center_y));
                let (left, top) = (gx * AC_GROUP_DIM_PIXELS, gy * AC_GROUP_DIM_PIXELS);
                // Doubled coordinates represent group-center and pixel-center half units exactly.
                let dx = i64::from(2 * left + (frame.width - left).min(AC_GROUP_DIM_PIXELS))
                    - i64::from(2 * x + 1);
                let dy = i64::from(2 * top + (frame.height - top).min(AC_GROUP_DIM_PIXELS))
                    - i64::from(2 * y + 1);
                (ring, dx * dx + dy * dy, id)
            });
        }
        Ok(groups)
    }

    pub(super) fn apply(
        &self,
        packets: FramePacketSet,
        frame: VarDctFrameLayout,
        records: Option<&[saliency::Record]>,
    ) -> Result<FramePacketSet, EncodeError> {
        self.validate(frame)?;
        self.validate_scores(frame, records)?;
        if matches!(self.order, Order::Raster) || packets.layout.is_fused_single_group() {
            return Ok(packets);
        }
        let groups = if let Some(records) = records {
            let mut groups: Vec<_> = (0..records.len() as u32).collect();
            groups.sort_unstable_by(|&a, &b| {
                let (left, right) = (records[a as usize], records[b as usize]);
                let lhs = u64::from(left.contrast) * u64::from(right.edges.max(1));
                let rhs = u64::from(right.contrast) * u64::from(left.edges.max(1));
                rhs.cmp(&lhs).then(a.cmp(&b))
            });
            groups
        } else {
            self.resolve(frame)?
        };
        let order = std::iter::once(GroupPacketKind::DcGlobal)
            .chain((0..packets.layout.dc_groups()).map(GroupPacketKind::DcGroup))
            .chain(std::iter::once(GroupPacketKind::AcGlobal))
            .chain((0..packets.layout.passes()).flat_map(|pass| {
                groups
                    .iter()
                    .map(move |&group| GroupPacketKind::AcGroup { pass, group })
            }));
        Ok(packets.with_order(order)?)
    }
}
