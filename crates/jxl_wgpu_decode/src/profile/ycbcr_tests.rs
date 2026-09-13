use std::path::PathBuf;
use std::sync::Arc;

use jxl_gpu_bitstream::{CodestreamInventory, FrameSectionKind, StreamSlice};
use jxl_test_support::{fixtures::modular_ycbcr, offline::hex};

use super::{parse_dc_global_ir, parse_lf_channel_dequantization, parse_standard_modular_profile};
use crate::GpuCodestream;
use crate::modular_inverse::{ModularInverseJob, plan_modular_inverse};
use crate::modular_transform::{
    ModularChannelGeometry, ModularChannelTopology, ModularTransformIr, ModularTransformLimits,
    ModularTransformPlan, parse_modular_transforms,
};
use crate::modular_tree::{MaTreeLimits, WpHeaderIr, parse_ma_config};

fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data/modular_ycbcr")
}

fn fixture(case: &modular_ycbcr::Case) -> (GpuCodestream, CodestreamInventory) {
    let bytes = hex::unhex(
        &std::fs::read_to_string(directory().join(format!("{}.jxl.hex", case.name))).unwrap(),
    );
    let parsed = jxl_gpu_bitstream::parse(&bytes, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    case.validate(&inventory);
    let source =
        GpuCodestream::from_spans([(0, StreamSlice::from_shared(Arc::from(parsed.codestream())))])
            .unwrap();
    (source, inventory)
}

fn channels(topology: &ModularChannelTopology) -> Vec<[i64; 4]> {
    topology
        .channels()
        .iter()
        .map(|channel| {
            [
                i64::from(channel.width),
                i64::from(channel.height),
                i64::from(channel.hshift),
                i64::from(channel.vshift),
            ]
        })
        .collect()
}

fn require_topology(
    name: &str,
    plan: &ModularTransformPlan,
    native: &modular_ycbcr::NativeTopology,
) {
    assert_eq!(
        channels(&plan.topology),
        native.channels,
        "{name}: channels"
    );
    assert_eq!(
        plan.topology.meta_channel_count(),
        native.meta_channels,
        "{name}: meta channels"
    );
    assert_eq!(
        plan.transforms.len(),
        native.transform_count,
        "{name}: transforms"
    );
    let inverse = plan_modular_inverse(plan).unwrap();
    assert_eq!(
        inverse
            .jobs()
            .iter()
            .filter(|job| matches!(job, ModularInverseJob::Squeeze { .. }))
            .count(),
        native.squeeze_channels,
        "{name}: Squeeze channels"
    );
}

fn require_transforms(
    name: &str,
    actual: &[ModularTransformIr],
    expected: &[modular_ycbcr::Transform],
) {
    assert_eq!(actual.len(), expected.len(), "{name}: transform headers");
    for (actual, expected) in actual.iter().zip(expected) {
        use modular_ycbcr::Transform;
        match (actual, expected) {
            (
                ModularTransformIr::Rct(actual),
                Transform::Rct {
                    begin_channel,
                    rct_type,
                },
            ) => {
                assert_eq!(
                    (actual.begin_channel, actual.rct_type),
                    (*begin_channel, *rct_type),
                    "{name}: RCT header"
                );
            }
            (
                ModularTransformIr::Palette(actual),
                Transform::Palette {
                    begin_channel,
                    channel_count,
                },
            ) => {
                assert_eq!(
                    (actual.begin_channel, actual.channel_count),
                    (*begin_channel, *channel_count),
                    "{name}: Palette header"
                );
            }
            (
                ModularTransformIr::Squeeze {
                    used_default_parameters,
                    parameters,
                },
                Transform::Squeeze(expected),
            ) => {
                assert_eq!(
                    *used_default_parameters,
                    expected.is_empty(),
                    "{name}: defaults"
                );
                if !expected.is_empty() {
                    let actual: Vec<_> = parameters
                        .iter()
                        .map(|parameter| modular_ycbcr::Squeeze {
                            horizontal: parameter.horizontal,
                            in_place: parameter.in_place,
                            begin_channel: parameter.begin_channel,
                            channel_count: parameter.channel_count,
                        })
                        .collect();
                    assert_eq!(actual, *expected, "{name}: Squeeze header");
                }
            }
            _ => panic!("{name}: unexpected transform: {actual:?} versus {expected:?}"),
        }
    }
}

#[test]
fn transformed_ycbcr_topology_matches_native_channels_including_empty_residuals() {
    for case in modular_ycbcr::cases()
        .into_iter()
        .filter(|case| !case.global_transforms.is_empty())
    {
        let (source, inventory) = fixture(&case);
        let native = modular_ycbcr::NativeTopology::parse(
            &std::fs::read_to_string(directory().join(format!("{}.topology", case.name))).unwrap(),
        );
        let frame = &inventory.frames[0];
        let section = frame
            .sections
            .iter()
            .find(|section| {
                matches!(
                    section.kind,
                    FrameSectionKind::Single | FrameSectionKind::LowFrequencyGlobal
                )
            })
            .unwrap();
        let mut reader = source.reader();
        reader.skip_bits(section.bits.offset).unwrap();
        parse_lf_channel_dequantization(&mut reader).unwrap();
        let source_topology =
            crate::modular_geometry::source_topology(&inventory.image_header, frame, 3).unwrap();
        let (_, _, _, _, plan) = parse_dc_global_ir(&mut reader, source_topology).unwrap();
        require_topology(&case.name, &plan, &native);
        require_transforms(&case.name, &plan.transforms, &case.global_transforms);
        parse_standard_modular_profile(&source, &inventory)
            .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
    }
}

#[test]
fn local_ycbcr_transforms_match_native_substream_topologies() {
    let limits = ModularTransformLimits::default();
    for case in modular_ycbcr::cases()
        .into_iter()
        .filter(modular_ycbcr::Case::has_local_transforms)
    {
        let (source, inventory) = fixture(&case);
        let native = modular_ycbcr::NativeSubstream::parse_all(
            &std::fs::read_to_string(directory().join(format!("{}.local", case.name))).unwrap(),
        );
        let profile = parse_standard_modular_profile(&source, &inventory)
            .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
        assert_eq!(
            native.len(),
            profile.entropy_groups.len(),
            "{}: groups",
            case.name
        );
        assert_eq!(native.len(), profile.resident_entropy_plans.len());
        for ((native, group), resident) in native
            .iter()
            .zip(&profile.entropy_groups)
            .zip(&profile.resident_entropy_plans)
        {
            let name = format!("{} stream {}", case.name, native.stream_index);
            assert_eq!(
                native.stream_index, group.stream_index,
                "{name}: execution order"
            );
            let source_topology = ModularChannelTopology::new(
                resident
                    .inverse_plan
                    .final_gpu_layouts()
                    .iter()
                    .map(|plane| {
                        ModularChannelGeometry::new(
                            plane.width,
                            plane.height,
                            plane.hshift,
                            plane.vshift,
                            plane.bit_depth,
                        )
                    })
                    .collect(),
                0,
                limits,
            )
            .unwrap();
            assert_eq!(channels(&source_topology), native.source, "{name}: source");
            let section = inventory.frames[0]
                .sections
                .iter()
                .find(|section| {
                    section.bits.offset < group.token_bit_offset
                        && section.bits.end() == Some(group.token_bit_end)
                })
                .unwrap();
            let expected = match section.kind {
                FrameSectionKind::LowFrequencyGroup { .. } => &case.lf_transforms,
                FrameSectionKind::PassGroup { .. } => &case.pass_transforms,
                _ => panic!("{name}: unexpected local section {:?}", section.kind),
            };
            let mut reader = source.reader();
            reader.skip_bits(section.bits.offset).unwrap();
            assert_eq!(reader.read_bits(1).unwrap(), 0, "{name}: local MA tree");
            let wp_header = WpHeaderIr::parse(&mut reader).unwrap();
            let plan = parse_modular_transforms(&mut reader, source_topology, limits).unwrap();
            require_topology(&name, &plan, &native.transformed);
            require_transforms(&name, &plan.transforms, expected);
            let config = parse_ma_config(&mut reader, MaTreeLimits::default()).unwrap();
            assert_eq!(
                reader.bit_offset(),
                group.token_bit_offset,
                "{name}: token start"
            );
            assert_eq!(wp_header, resident.wp_header, "{name}: weighted predictor");
            assert_eq!(
                &config,
                resident.ma_config.resolve(&profile.ma_config),
                "{name}: MA tree"
            );
            assert_eq!(
                plan.topology
                    .gpu_entropy_channels(config.maximum_tree_property())
                    .unwrap(),
                resident.channel_metadata,
                "{name}: entropy descriptors"
            );
            assert_eq!(
                plan_modular_inverse(&plan).unwrap(),
                resident.inverse_plan,
                "{name}: inverse"
            );
        }
    }
}
