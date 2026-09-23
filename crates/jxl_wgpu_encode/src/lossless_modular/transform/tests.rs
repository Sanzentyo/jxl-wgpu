use super::*;
use crate::{LosslessModularColorTransform, LosslessModularGroupSize};

fn config() -> LosslessModularConfig {
    LosslessModularConfig {
        color_transform: LosslessModularColorTransform::GlobalRct(
            LosslessModularRctType::new(41).unwrap(),
        ),
        palette: Some(
            LosslessModularPalette::new(32)
                .unwrap()
                .with_components(1, 2)
                .unwrap(),
        ),
        squeeze: LosslessModularSqueeze::HorizontalThenVertical,
        ..Default::default()
    }
}

#[test]
fn selected_components_keep_explicit_sources_extents_and_band_order() {
    let grid = LosslessModularGroupGrid::for_extent(5, 3, Default::default()).unwrap();
    let plan =
        ModularTransformPlan::new(grid, LosslessModularFormat::Rgba, 31, 0, config()).unwrap();
    let group = plan.group(grid.group(0).unwrap()).unwrap();
    // Hand-worked odd-sized H then V split: meta, three LL, three HL, three LH, three HH.
    let expected = [
        (5, 0, [15, 2]),
        (0, 0, [3, 2]),
        (4, 0, [3, 2]),
        (3, 0, [3, 2]),
        (0, 1, [2, 2]),
        (4, 1, [2, 2]),
        (3, 1, [2, 2]),
        (0, 2, [3, 1]),
        (4, 2, [3, 1]),
        (3, 2, [3, 1]),
        (0, 3, [2, 1]),
        (4, 3, [2, 1]),
        (3, 3, [2, 1]),
    ];
    let actual: Vec<_> = group
        .channels
        .iter()
        .map(|channel| (channel.source.kernel_value(), channel.band, channel.extent))
        .collect();
    assert_eq!(actual, expected);
    assert_eq!(plan.dispatches, 13);
    assert!(plan.global_operations.is_empty());
    assert!(matches!(group.operations[0], TransformOperation::Rct(rct) if rct.value() == 41));
    let TransformOperation::Squeeze(steps) = &group.operations[2] else {
        panic!("missing planned Squeeze")
    };
    assert_eq!(
        steps
            .iter()
            .map(|step| (step.horizontal, step.range))
            .collect::<Vec<_>>(),
        [
            (true, ChannelRange { begin: 1, count: 3 }),
            (false, ChannelRange { begin: 1, count: 6 }),
        ]
    );
}

#[test]
fn repeated_and_edge_groups_share_plans_with_one_global_rct() {
    let grid = LosslessModularGroupGrid::for_extent(769, 769, LosslessModularGroupSize::Pixels256)
        .unwrap();
    let plan =
        ModularTransformPlan::new(grid, LosslessModularFormat::Rgba, 31, 0, config()).unwrap();
    assert_eq!(plan.shapes.len(), 4);
    assert_eq!(plan.dispatches, 163); // 9*13 + 3*7 + 3*7 + 4
    assert_eq!(plan.max_channels, 13);
    assert!(
        matches!(&plan.global_operations[..], [TransformOperation::Rct(rct)] if rct.value() == 41)
    );
    assert!(std::ptr::eq(
        plan.group(grid.group(0).unwrap()).unwrap(),
        plan.group(grid.group(1).unwrap()).unwrap()
    ));
    for (index, extent, channels, squeeze) in [
        (
            0,
            [256, 256],
            13,
            LosslessModularSqueeze::HorizontalThenVertical,
        ),
        (3, [1, 256], 7, LosslessModularSqueeze::Vertical),
        (12, [256, 1], 7, LosslessModularSqueeze::Horizontal),
        (15, [1, 1], 4, LosslessModularSqueeze::None),
    ] {
        let group = plan.group(grid.group(index).unwrap()).unwrap();
        assert_eq!(group.extent, extent);
        assert_eq!(group.channels.len(), channels);
        assert_eq!(group.squeeze, squeeze);
        assert!(
            group
                .operations
                .iter()
                .all(|operation| !matches!(operation, TransformOperation::Rct(_)))
        );
    }
    let gray = ModularTransformPlan::new(
        LosslessModularGroupGrid::for_extent(1, 1, Default::default()).unwrap(),
        LosslessModularFormat::Gray,
        8,
        0,
        LosslessModularConfig {
            squeeze: LosslessModularSqueeze::Horizontal,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(gray.max_channels, 1);
    assert_eq!(gray.prefix_channels, 2); // preserve the existing entropy policy for an elided axis
    assert!(gray.extended_prediction_domain);
}

#[test]
fn invalid_static_topology_is_rejected_before_dispatch_lowering() {
    let grid = LosslessModularGroupGrid::for_extent(5, 3, Default::default()).unwrap();
    assert!(matches!(
        ModularTransformPlan::new(grid, LosslessModularFormat::GrayAlpha, 8, 0, config()),
        Err(EncodeError::InvalidModularPaletteComponents {
            begin: 1,
            count: 2,
            channels: 2
        })
    ));
    assert!(matches!(
        ModularTransformPlan::new(
            grid,
            LosslessModularFormat::GrayAlpha,
            8,
            0,
            LosslessModularConfig {
                palette: None,
                ..config()
            }
        ),
        Err(EncodeError::ModularRctColorChannels { color_channels: 1 })
    ));
}
