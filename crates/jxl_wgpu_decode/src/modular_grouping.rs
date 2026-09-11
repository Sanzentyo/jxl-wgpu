//! Shared JPEG XL Modular channel ownership and subimage geometry for both coding modes.
use crate::modular_transform::{
    GpuModularChannelLayout, ModularChannelGeometry, ModularChannelTopology, ModularTransformLimits,
};
use crate::{Error, Result, UnsupportedCodestreamFeature, UnsupportedProfile};

fn unsupported<T>(detail: impl Into<String>) -> Result<T> {
    Err(unsupported_error(detail).into())
}

fn unsupported_error(detail: impl Into<String>) -> UnsupportedProfile {
    UnsupportedProfile::new(
        UnsupportedCodestreamFeature::Other("modular-subimage-topology".into()),
        detail,
    )
}

pub(crate) fn global_subimage_channel_count(
    topology: &ModularChannelTopology,
    group_dimension: u32,
) -> usize {
    topology
        .channels()
        .iter()
        .enumerate()
        .take_while(|(index, channel)| {
            *index < topology.meta_channel_count()
                || (channel.width <= group_dimension && channel.height <= group_dimension)
        })
        .count()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModularSubimageKind {
    LowFrequencyGroup,
    PassGroup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularSubimageRegion {
    pub(crate) kind: ModularSubimageKind,
    pub(crate) column: u32,
    pub(crate) row: u32,
    pub(crate) group_dimension: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularPassShiftRange {
    pub(crate) min_shift: u32,
    pub(crate) max_shift: u32,
}

impl ModularPassShiftRange {
    pub(crate) fn contains(self, shift: u32) -> bool {
        (self.min_shift..self.max_shift).contains(&shift)
    }
}

/// Builds the shift brackets used by `jxl-modular`'s `prepare_groups`.
///
/// A missing entry denotes a pass with no shift bracket. Such a pass still has a physical
/// PassGroup section, but the section must be empty. The final pass is always assigned the
/// remaining `[0, max_shift)` bracket, including when that bracket is empty.
pub(crate) fn build_modular_pass_shift_ranges(
    pass_count: u32,
    downsampling: &[u32],
    last_pass: &[u32],
) -> Result<Vec<Option<ModularPassShiftRange>>> {
    if !(1..=jxl_gpu_bitstream::FramePassesInventory::MAX_PASSES).contains(&pass_count) {
        return unsupported("the Modular frame declares an invalid progressive pass count");
    }
    if downsampling.len() != last_pass.len()
        || downsampling.len() > usize::try_from(pass_count).unwrap_or(usize::MAX)
        || (pass_count == 1 && !downsampling.is_empty())
        || downsampling
            .iter()
            .any(|factor| !matches!(factor, 1 | 2 | 4 | 8))
        || downsampling.windows(2).any(|pair| pair[1] >= pair[0])
        || last_pass.windows(2).any(|pair| pair[1] <= pair[0])
        || last_pass.iter().any(|&pass| pass >= pass_count || pass > 7)
    {
        return unsupported("the Modular frame has inconsistent progressive-pass metadata");
    }

    let pass_count = usize::try_from(pass_count)
        .map_err(|_| unsupported_error("Modular pass count exceeds host address space"))?;
    let mut ranges = vec![None; pass_count];
    let mut max_shift = 3;
    for (&downsample, &pass) in downsampling.iter().zip(last_pass) {
        let pass = usize::try_from(pass).map_err(|_| {
            unsupported_error("Modular progressive-pass index exceeds host address space")
        })?;
        let range = ranges.get_mut(pass).ok_or_else(|| {
            unsupported_error("Modular progressive-pass index exceeds the pass count")
        })?;
        // A boundary declared on the final pass does not reduce the remaining bracket. The
        // final-pass minimum is unconditionally zero, with the maximum from the preceding pass.
        if pass == pass_count - 1 {
            continue;
        }
        if range.is_some() {
            return unsupported("the Modular frame has duplicate progressive-pass boundaries");
        }
        let min_shift = downsample.trailing_zeros();
        if min_shift > max_shift {
            return unsupported("the Modular frame has non-monotonic progressive downsampling");
        }
        *range = Some(ModularPassShiftRange {
            min_shift,
            max_shift,
        });
        max_shift = min_shift;
    }
    ranges[pass_count - 1] = Some(ModularPassShiftRange {
        min_shift: 0,
        max_shift,
    });
    Ok(ranges)
}

pub(crate) fn modular_pass_for_channel(
    hshift: u32,
    vshift: u32,
    pass_ranges: &[Option<ModularPassShiftRange>],
) -> Option<usize> {
    if hshift >= 3 && vshift >= 3 {
        return None;
    }
    let shift = hshift.min(vshift);
    pass_ranges
        .iter()
        .position(|range| range.is_some_and(|range| range.contains(shift)))
}

pub(crate) fn grouped_subimage_topology(
    frame_topology: &ModularChannelTopology,
    frame_layout: &[GpuModularChannelLayout],
    global_channel_count: usize,
    region: ModularSubimageRegion,
    pass_range: Option<ModularPassShiftRange>,
    limits: ModularTransformLimits,
) -> Result<(ModularChannelTopology, Vec<GpuModularChannelLayout>)> {
    if frame_topology.channels().len() != frame_layout.len()
        || global_channel_count > frame_layout.len()
    {
        return Err(Error::EngineContract(
            "frame Modular topology and packed layout disagree",
        ));
    }

    let mut channels = Vec::new();
    let mut targets = Vec::new();
    for (&channel, &layout) in frame_topology.channels()[global_channel_count..]
        .iter()
        .zip(&frame_layout[global_channel_count..])
    {
        let (Ok(hshift), Ok(vshift)) =
            (u32::try_from(channel.hshift), u32::try_from(channel.vshift))
        else {
            return unsupported(
                "a DC-global meta channel was not retained in the global Modular subimage",
            );
        };
        let low_frequency = hshift >= 3 && vshift >= 3;
        if low_frequency != (region.kind == ModularSubimageKind::LowFrequencyGroup) {
            continue;
        }
        if region.kind == ModularSubimageKind::PassGroup
            && !pass_range.is_some_and(|range| range.contains(hshift.min(vshift)))
        {
            continue;
        }
        let (tile_hshift, tile_vshift) = if low_frequency {
            (hshift - 3, vshift - 3)
        } else {
            (hshift, vshift)
        };
        let tile_width = region
            .group_dimension
            .checked_shr(tile_hshift)
            .filter(|value| *value != 0)
            .ok_or_else(|| unsupported_error("Modular horizontal channel shift is too large"))?;
        let tile_height = region
            .group_dimension
            .checked_shr(tile_vshift)
            .filter(|value| *value != 0)
            .ok_or_else(|| unsupported_error("Modular vertical channel shift is too large"))?;
        let origin_x = region
            .column
            .checked_mul(tile_width)
            .ok_or_else(|| unsupported_error("Modular transformed group x origin overflow"))?;
        let origin_y = region
            .row
            .checked_mul(tile_height)
            .ok_or_else(|| unsupported_error("Modular transformed group y origin overflow"))?;
        let width = channel.width.saturating_sub(origin_x).min(tile_width);
        let height = channel.height.saturating_sub(origin_y).min(tile_height);
        if width == 0 || height == 0 {
            // libjxl DecodeGroup omits empty clipped frame channels before parsing local
            // transforms. Empty channels created by those transforms retain their local IDs.
            continue;
        }
        channels.push(ModularChannelGeometry::new(
            width,
            height,
            channel.hshift,
            channel.vshift,
            channel.bit_depth,
        ));
        let word_offset = origin_y
            .checked_mul(layout.row_stride_words)
            .and_then(|offset| offset.checked_add(origin_x))
            .and_then(|offset| offset.checked_add(layout.word_offset))
            .ok_or_else(|| unsupported_error("Modular frame-arena plane offset overflow"))?;
        targets.push(GpuModularChannelLayout {
            word_offset,
            row_stride_words: layout.row_stride_words,
            width,
            height,
            hshift: channel.hshift,
            vshift: channel.vshift,
            bit_depth: channel.bit_depth,
            reserved: 0,
        });
    }
    Ok((ModularChannelTopology::new(channels, 0, limits)?, targets))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_ownership_is_a_prefix_and_clipped_empty_channels_have_no_group_id() {
        let limits = ModularTransformLimits::default();
        let topology = ModularChannelTopology::new(
            vec![
                ModularChannelGeometry::new(3, 1, -1, -1, 8),
                ModularChannelGeometry::new(16, 16, 4, 4, 8),
                ModularChannelGeometry::new(129, 257, 1, 0, 8),
                ModularChannelGeometry::new(128, 257, 1, 0, 8),
                ModularChannelGeometry::new(129, 128, 1, 1, 8),
            ],
            1,
            limits,
        )
        .unwrap();
        let global = global_subimage_channel_count(&topology, 256);
        assert_eq!(global, 2); // The later 129x128 channel belongs to a pass despite being small.
        let layout = topology.gpu_layout().unwrap();
        let (group, targets) = grouped_subimage_topology(
            &topology,
            &layout,
            global,
            ModularSubimageRegion {
                kind: ModularSubimageKind::PassGroup,
                column: 1,
                row: 0,
                group_dimension: 256,
            },
            Some(ModularPassShiftRange {
                min_shift: 0,
                max_shift: 3,
            }),
            limits,
        )
        .unwrap();
        assert_eq!(
            group.channels(),
            &[
                ModularChannelGeometry::new(1, 256, 1, 0, 8),
                ModularChannelGeometry::new(1, 128, 1, 1, 8),
            ]
        );
        assert_eq!(targets[0].word_offset, layout[2].word_offset + 128);
        assert_eq!(targets[1].word_offset, layout[4].word_offset + 128);
        // Entropy/local transforms see consecutive channel IDs after clipping, without a
        // placeholder for the absent 128-wide residual channel.
        let metadata = group.gpu_entropy_channels(Some(0)).unwrap();
        assert_eq!(metadata.channels.len(), 2);
        assert_eq!(metadata.channels[1].decoded_start, 256);
    }

    #[test]
    fn pass_brackets_reject_invalid_factors_and_out_of_order_boundaries() {
        for (passes, factors, boundaries) in [
            (0, vec![], vec![]),
            (12, vec![], vec![]),
            (1, vec![1], vec![0]),
            (3, vec![0], vec![0]),
            (3, vec![3], vec![0]),
            (3, vec![16], vec![0]),
            (3, vec![2, 4], vec![0, 1]),
            (3, vec![4, 4], vec![0, 1]),
            (3, vec![4, 2], vec![1, 0]),
            (3, vec![4, 2], vec![0, 0]),
            (3, vec![4], vec![3]),
            (3, vec![4], vec![]),
            (11, vec![2], vec![8]),
        ] {
            assert!(build_modular_pass_shift_ranges(passes, &factors, &boundaries).is_err());
        }
    }

    #[test]
    fn every_representable_pass_schedule_owns_each_non_lf_shift_exactly_once() {
        for passes in 1..=11 {
            for factor_mask in 0u32..16 {
                let factors = [8, 4, 2, 1]
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, factor)| ((factor_mask >> i) & 1 != 0).then_some(factor))
                    .collect::<Vec<_>>();
                for pass_mask in 0u32..(1 << passes.min(8)) {
                    if pass_mask.count_ones() != factor_mask.count_ones()
                        || (passes == 1 && pass_mask != 0)
                    {
                        continue;
                    }
                    let boundaries = (0..passes.min(8))
                        .filter(|&pass| (pass_mask >> pass) & 1 != 0)
                        .collect::<Vec<_>>();
                    let ranges =
                        build_modular_pass_shift_ranges(passes, &factors, &boundaries).unwrap();
                    assert_eq!(ranges.len(), passes as usize);
                    for horizontal in 0u32..8 {
                        for vertical in 0u32..8 {
                            let shift = horizontal.min(vertical);
                            let owners = ranges
                                .iter()
                                .enumerate()
                                .filter_map(|(pass, range)| {
                                    range
                                        .is_some_and(|range| range.contains(shift))
                                        .then_some(pass)
                                })
                                .collect::<Vec<_>>();
                            if shift >= 3 {
                                assert!(owners.is_empty());
                                assert_eq!(
                                    modular_pass_for_channel(horizontal, vertical, &ranges),
                                    None
                                );
                                continue;
                            }
                            // A transformed plane first appears at the earliest declared detail
                            // that includes it, or at the unconditional full-resolution final pass.
                            let expected = factors
                                .iter()
                                .zip(&boundaries)
                                .find_map(|(&factor, &pass)| (factor <= 1 << shift).then_some(pass))
                                .unwrap_or(passes - 1)
                                as usize;
                            assert_eq!(
                                owners,
                                [expected],
                                "passes={passes} {factors:?}/{boundaries:?}"
                            );
                            assert_eq!(
                                modular_pass_for_channel(horizontal, vertical, &ranges),
                                Some(expected)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn final_pass_boundaries_preserve_all_remaining_channel_shifts() {
        let last_only = build_modular_pass_shift_ranges(3, &[4], &[2]).unwrap();
        assert_eq!(
            last_only,
            vec![
                None,
                None,
                Some(ModularPassShiftRange {
                    min_shift: 0,
                    max_shift: 3
                })
            ]
        );
        for shift in 0..3 {
            assert_eq!(modular_pass_for_channel(shift, shift, &last_only), Some(2));
        }
        let first_and_last = build_modular_pass_shift_ranges(3, &[4, 2], &[0, 2]).unwrap();
        assert_eq!(
            first_and_last,
            vec![
                Some(ModularPassShiftRange {
                    min_shift: 2,
                    max_shift: 3
                }),
                None,
                Some(ModularPassShiftRange {
                    min_shift: 0,
                    max_shift: 2
                }),
            ]
        );
        for (shift, pass) in [(0, 2), (1, 2), (2, 0)] {
            assert_eq!(
                modular_pass_for_channel(shift, shift, &first_and_last),
                Some(pass)
            );
        }
    }
}
